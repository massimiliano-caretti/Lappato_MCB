"""Retrieval-augmented generation evaluation manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import mean, min_value, summary, values

RUN_TAG = "rag_eval"

DEFAULT_THRESHOLDS = {
    "retrieval_recall_warning": 0.80,
    "unsupported_rate_warning": 0.05,
    "citation_mismatch_warning": 0.05,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _retrieval_recall_low(path: Path) -> bool:
    vals = values(path, "recall_at_k", "hit_rate", "answer_recall")
    return bool(vals and min(vals) < THRESHOLDS["retrieval_recall_warning"])


def _retrieval_summary(path: Path) -> dict:
    return summary(path, "min_recall_at_k", min_value(path, "recall_at_k", "hit_rate", "answer_recall"))


def _unsupported_high(path: Path) -> bool:
    return mean(values(path, "unsupported_rate", "faithfulness_error_rate")) > THRESHOLDS["unsupported_rate_warning"]


def _unsupported_summary(path: Path) -> dict:
    return summary(path, "mean_unsupported_rate", mean(values(path, "unsupported_rate", "faithfulness_error_rate")))


def _citation_mismatch_high(path: Path) -> bool:
    return mean(values(path, "citation_mismatch_rate", "bad_citation_rate")) > THRESHOLDS["citation_mismatch_warning"]


def _citation_summary(path: Path) -> dict:
    return summary(path, "mean_citation_mismatch_rate", mean(values(path, "citation_mismatch_rate", "bad_citation_rate")))


MANIFEST = [
    {
        "id": "low_retrieval_recall",
        "title": "Retriever recall is low for answer-bearing documents",
        "evidence": "rag_retrieval.csv",
        "evidence_check": _retrieval_recall_low,
        "evidence_summary": _retrieval_summary,
        "severity": "high",
        "queries": ["retrieval augmented generation recall at k evaluation", "dense retrieval RAG evaluation benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Tune chunking, embeddings and top-k; report recall@k against answer-bearing documents.",
        "why_it_matters": "A generator cannot cite evidence that retrieval never returns.",
        "next_checks": ["Measure recall@k by query type.", "Compare sparse, dense and hybrid retrieval."],
        "success_criteria": ["Recall@k reaches at least 0.80.", "Worst query classes improve."],
        "references": ["RAG evaluation", "retrieval recall", "hybrid search"],
    },
    {
        "id": "unsupported_answers",
        "title": "Generated answers are not fully supported by retrieved context",
        "evidence": "rag_grounding.csv",
        "evidence_check": _unsupported_high,
        "evidence_summary": _unsupported_summary,
        "severity": "high",
        "queries": ["RAG faithfulness grounded answer evaluation", "attribution evaluation retrieval augmented generation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add faithfulness grading and require answer spans to be traceable to retrieved context.",
        "why_it_matters": "RAG can still hallucinate even when retrieval succeeds.",
        "next_checks": ["Inspect unsupported claims.", "Add abstention for weak context."],
        "success_criteria": ["Unsupported-answer rate falls below 5%."],
        "references": ["faithfulness evaluation", "grounded generation", "attribution"],
    },
    {
        "id": "citation_mismatch",
        "title": "Citations do not consistently support the generated claims",
        "evidence": "rag_citations.csv",
        "evidence_check": _citation_mismatch_high,
        "evidence_summary": _citation_summary,
        "severity": "medium",
        "queries": ["citation faithfulness RAG evaluation", "attributed question answering citation support"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Validate claim-citation alignment and penalize citations that do not contain the answer span.",
        "why_it_matters": "A citation-looking answer can be misleading when the cited source is unrelated.",
        "next_checks": ["Grade citation support per claim.", "Track citation mismatch by retriever source."],
        "success_criteria": ["Citation mismatch falls below 5%."],
        "references": ["citation support", "attributed QA", "RAG faithfulness"],
    },
    {
        "id": "reranker_missing",
        "title": "No reranker or second-stage retrieval check is represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["RAG reranking cross encoder retrieval augmented generation", "neural reranking dense retrieval benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Compare first-stage retrieval with a cross-encoder or late-interaction reranker.",
        "why_it_matters": "Top-k recall and ordering often improve with reranking.",
        "next_checks": ["Evaluate reranked recall@k.", "Measure latency added by reranking."],
        "success_criteria": ["Reranking improves answer support without breaking latency budget."],
        "references": ["reranking", "cross encoder", "late interaction retrieval"],
    },
    {
        "id": "chunking_strategy_unchecked",
        "title": "Chunking strategy is not explicitly evaluated",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["chunking strategy retrieval augmented generation evaluation", "document chunking RAG benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compare chunk sizes, overlaps and semantic chunking under identical retrieval metrics.",
        "why_it_matters": "Chunking determines whether answer-bearing evidence is retrievable.",
        "next_checks": ["Sweep chunk size and overlap.", "Inspect failures from split answer spans."],
        "success_criteria": ["Chosen chunking improves recall and support rate."],
        "references": ["RAG chunking", "document retrieval", "semantic chunking"],
    },
]

