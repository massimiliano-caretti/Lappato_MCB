"""Graph-ML manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import max_value, min_value, summary, values

RUN_TAG = "graph_ml"

DEFAULT_THRESHOLDS = {
    "leakage_edge_rate_warning": 0.0,
    "embedding_variance_warning": 0.01,
    "homophily_warning": 0.30,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _split_leakage(path: Path) -> bool:
    return max_value(path, "leakage_edge_rate", "cross_split_edge_rate") > THRESHOLDS["leakage_edge_rate_warning"]


def _split_leakage_summary(path: Path) -> dict:
    return summary(path, "max_leakage_edge_rate", max_value(path, "leakage_edge_rate", "cross_split_edge_rate"))


def _oversmoothing(path: Path) -> bool:
    vals = values(path, "embedding_variance", "pairwise_distance")
    return bool(vals and min(vals) < THRESHOLDS["embedding_variance_warning"])


def _oversmoothing_summary(path: Path) -> dict:
    return summary(path, "min_embedding_variance", min_value(path, "embedding_variance", "pairwise_distance"))


def _low_homophily(path: Path) -> bool:
    vals = values(path, "homophily", "label_homophily")
    return bool(vals and min(vals) < THRESHOLDS["homophily_warning"])


def _homophily_summary(path: Path) -> dict:
    return summary(path, "min_homophily", min_value(path, "homophily", "label_homophily"))


MANIFEST = [
    {
        "id": "graph_split_leakage",
        "title": "Graph split may leak edges or labels across train/test",
        "evidence": "graph_split.csv",
        "evidence_check": _split_leakage,
        "evidence_summary": _split_leakage_summary,
        "severity": "high",
        "queries": ["graph neural network data leakage train test split", "graph machine learning evaluation leakage transductive inductive"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Audit edge, node and label leakage; compare transductive and inductive splits explicitly.",
        "why_it_matters": "Graph splits can leak neighborhood information into evaluation.",
        "next_checks": ["Check cross-split edges.", "Report split protocol."],
        "success_criteria": ["Leakage edge rate is zero or protocol is explicitly transductive."],
        "references": ["graph evaluation", "data leakage", "inductive split"],
    },
    {
        "id": "oversmoothing_signal",
        "title": "Node embeddings show possible over-smoothing",
        "evidence": "graph_embeddings.csv",
        "evidence_check": _oversmoothing,
        "evidence_summary": _oversmoothing_summary,
        "severity": "medium",
        "queries": ["graph neural network over smoothing residual connections", "GNN over-smoothing mitigation normalization"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Reduce depth or add residual/normalization strategies; track embedding variance by layer.",
        "why_it_matters": "Over-smoothed embeddings reduce class separability.",
        "next_checks": ["Plot embedding variance by layer.", "Compare shallow and residual GNNs."],
        "success_criteria": ["Embedding variance stays above threshold and validation improves."],
        "references": ["over-smoothing", "GNN depth", "residual GNN"],
    },
    {
        "id": "homophily_mismatch",
        "title": "Graph homophily may be too low for standard message passing",
        "evidence": "graph_homophily.csv",
        "evidence_check": _low_homophily,
        "evidence_summary": _homophily_summary,
        "severity": "medium",
        "queries": ["heterophily graph neural networks homophily mismatch", "GNN heterophilous graphs benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compare heterophily-aware architectures and non-graph baselines.",
        "why_it_matters": "Standard GNN message passing can fail when neighbors have different labels.",
        "next_checks": ["Report label homophily.", "Compare MLP and heterophily GNN."],
        "success_criteria": ["Chosen model beats MLP under low homophily."],
        "references": ["heterophily", "graph neural networks", "homophily"],
    },
    {
        "id": "inductive_generalization_missing",
        "title": "Inductive generalization to unseen nodes/graphs is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["inductive graph neural networks generalization unseen graphs", "graph neural network out of distribution generalization"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add unseen-node or unseen-graph evaluation if deployment is inductive.",
        "why_it_matters": "Transductive scores can overstate deployment performance.",
        "next_checks": ["Define deployment graph setting.", "Run inductive split."],
        "success_criteria": ["Inductive metric is reported or deployment is declared transductive."],
        "references": ["inductive GNN", "OOD generalization", "graph split"],
    },
    {
        "id": "explainability_missing",
        "title": "Graph explanation or subgraph audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["graph neural network explainability subgraph explanation", "GNNExplainer graph model explanation"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Add explanation checks for critical predictions and compare salient subgraphs with domain expectations.",
        "why_it_matters": "Graph models can exploit spurious structural shortcuts.",
        "next_checks": ["Explain top errors.", "Review salient neighborhoods."],
        "success_criteria": ["Explanations are stable and domain-plausible."],
        "references": ["GNNExplainer", "graph explainability", "subgraph explanations"],
    },
]

