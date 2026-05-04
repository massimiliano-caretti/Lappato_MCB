"""LLM evaluation manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import mean, rows, summary, values

RUN_TAG = "llm_eval"

DEFAULT_THRESHOLDS = {
    "hallucination_rate_warning": 0.05,
    "toxicity_rate_warning": 0.01,
    "prompt_score_spread_warning": 0.15,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _hallucination_high(path: Path) -> bool:
    return mean(values(path, "hallucination_rate", "unsupported_rate")) > THRESHOLDS["hallucination_rate_warning"]


def _hallucination_summary(path: Path) -> dict:
    return summary(path, "mean_hallucination_rate", mean(values(path, "hallucination_rate", "unsupported_rate")))


def _toxicity_high(path: Path) -> bool:
    return mean(values(path, "toxicity_rate", "unsafe_rate")) > THRESHOLDS["toxicity_rate_warning"]


def _toxicity_summary(path: Path) -> dict:
    return summary(path, "mean_toxicity_rate", mean(values(path, "toxicity_rate", "unsafe_rate")))


def _prompt_instability(path: Path) -> bool:
    scores = values(path, "score", "pass_rate", "accuracy")
    return (max(scores) - min(scores)) > THRESHOLDS["prompt_score_spread_warning"] if scores else False


def _prompt_summary(path: Path) -> dict:
    scores = values(path, "score", "pass_rate", "accuracy")
    return {"rows": len(rows(path)), "score_spread": round((max(scores) - min(scores)) if scores else 0.0, 4)}


MANIFEST = [
    {
        "id": "hallucination_rate_high",
        "title": "LLM hallucination or unsupported-answer rate is high",
        "evidence": "llm_hallucination.csv",
        "evidence_check": _hallucination_high,
        "evidence_summary": _hallucination_summary,
        "severity": "high",
        "queries": ["LLM hallucination evaluation factuality benchmark", "large language model factual consistency mitigation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add factuality evaluation, abstention rules and source-grounded answer checks.",
        "why_it_matters": "Unsupported LLM outputs are a direct reliability risk.",
        "next_checks": ["Sample unsupported answers.", "Add citation or evidence-required grading."],
        "success_criteria": ["Unsupported rate falls below 5%.", "Critical factual errors are reviewed manually."],
        "references": ["factuality evaluation", "hallucination mitigation", "LLM evaluation"],
    },
    {
        "id": "safety_failure_rate",
        "title": "Unsafe or toxic output rate is above tolerance",
        "evidence": "llm_safety.csv",
        "evidence_check": _toxicity_high,
        "evidence_summary": _toxicity_summary,
        "severity": "critical",
        "queries": ["LLM safety evaluation toxicity refusal jailbreak benchmark", "red teaming large language models safety"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add safety red-team cases, refusal calibration and policy-specific regression tests.",
        "why_it_matters": "Safety failures can block deployment even when task accuracy is strong.",
        "next_checks": ["Group failures by policy category.", "Add adversarial prompts to CI evals."],
        "success_criteria": ["Unsafe rate falls below 1%.", "No critical policy category regresses."],
        "references": ["LLM safety", "red teaming", "toxicity evaluation"],
    },
    {
        "id": "prompt_instability",
        "title": "Prompt variants produce unstable evaluation scores",
        "evidence": "llm_prompt_variants.csv",
        "evidence_check": _prompt_instability,
        "evidence_summary": _prompt_summary,
        "severity": "medium",
        "queries": ["prompt sensitivity large language model evaluation", "prompt robustness LLM benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Evaluate multiple prompt templates and select by robust mean rather than best single prompt.",
        "why_it_matters": "Prompt-sensitive systems are hard to reproduce and debug.",
        "next_checks": ["Report score distribution across prompts.", "Freeze a template suite."],
        "success_criteria": ["Prompt score spread falls below 0.15.", "Selected prompt remains stable on new tasks."],
        "references": ["prompt robustness", "LLM evaluation", "prompt sensitivity"],
    },
    {
        "id": "cost_latency_unbounded",
        "title": "Cost and latency budget checks are not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["LLM inference latency cost optimization evaluation", "large language model serving cost latency benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Log tokens, latency and cost per request; add budget-aware model or prompt selection.",
        "why_it_matters": "A good LLM eval is incomplete if the winning setup is too slow or expensive.",
        "next_checks": ["Track p50/p95 latency.", "Track input/output tokens per task."],
        "success_criteria": ["p95 latency and cost stay inside the declared budget."],
        "references": ["LLM serving", "latency optimization", "cost-aware inference"],
    },
    {
        "id": "eval_coverage_missing",
        "title": "Evaluation coverage across task slices is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["LLM evaluation coverage task slices benchmark", "large language model test set coverage reliability"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Define task slices and report pass rate by slice, difficulty and input length.",
        "why_it_matters": "Average LLM scores hide brittle slices.",
        "next_checks": ["Add slice labels to eval rows.", "Inspect worst slices."],
        "success_criteria": ["Each critical slice has enough examples and a passing threshold."],
        "references": ["LLM evaluation", "slice-based evaluation", "benchmark coverage"],
    },
]

