"""Causal-ML manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import max_value, min_value, summary, values

RUN_TAG = "causal_ml"

DEFAULT_THRESHOLDS = {
    "standardized_mean_difference_warning": 0.10,
    "min_propensity_warning": 0.05,
    "placebo_effect_abs_warning": 0.05,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _imbalance_high(path: Path) -> bool:
    return max_value(path, "smd", "standardized_mean_difference") > THRESHOLDS["standardized_mean_difference_warning"]


def _imbalance_summary(path: Path) -> dict:
    return summary(path, "max_smd", max_value(path, "smd", "standardized_mean_difference"))


def _positivity_violation(path: Path) -> bool:
    vals = values(path, "propensity", "propensity_score")
    return bool(vals) and min(vals) < THRESHOLDS["min_propensity_warning"]


def _positivity_summary(path: Path) -> dict:
    return summary(path, "min_propensity", min_value(path, "propensity", "propensity_score"))


def _placebo_failure(path: Path) -> bool:
    return max_value(path, "abs_effect", "absolute_effect") > THRESHOLDS["placebo_effect_abs_warning"]


def _placebo_summary(path: Path) -> dict:
    return summary(path, "max_abs_placebo_effect", max_value(path, "abs_effect", "absolute_effect"))


MANIFEST = [
    {
        "id": "covariate_imbalance",
        "title": "Treated/control covariates remain imbalanced",
        "evidence": "causal_balance.csv",
        "evidence_check": _imbalance_high,
        "evidence_summary": _imbalance_summary,
        "severity": "high",
        "queries": ["causal inference covariate balance standardized mean difference propensity score", "causal machine learning matching weighting balance diagnostics"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add balance diagnostics and compare matching, weighting or doubly robust estimators.",
        "why_it_matters": "Effect estimates are weak when covariates remain imbalanced.",
        "next_checks": ["Plot SMD before/after adjustment.", "Inspect high-SMD covariates."],
        "success_criteria": ["All critical SMD values fall below 0.10."],
        "references": ["standardized mean difference", "propensity score", "doubly robust estimation"],
    },
    {
        "id": "positivity_violation",
        "title": "Propensity scores indicate possible positivity violation",
        "evidence": "causal_propensity.csv",
        "evidence_check": _positivity_violation,
        "evidence_summary": _positivity_summary,
        "severity": "high",
        "queries": ["positivity violation causal inference propensity overlap", "causal inference overlap trimming propensity scores"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Inspect overlap, trim unsupported regions or redefine the target population.",
        "why_it_matters": "Causal effects are not identifiable where treatment overlap is absent.",
        "next_checks": ["Plot propensity by treatment.", "Run overlap trimming sensitivity."],
        "success_criteria": ["Overlap support is adequate or target population is narrowed."],
        "references": ["positivity", "overlap", "propensity trimming"],
    },
    {
        "id": "placebo_test_failure",
        "title": "Placebo or negative-control test shows non-zero effect",
        "evidence": "causal_placebo.csv",
        "evidence_check": _placebo_failure,
        "evidence_summary": _placebo_summary,
        "severity": "medium",
        "queries": ["negative control placebo test causal inference machine learning", "causal effect sensitivity analysis negative controls"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add negative-control outcomes/exposures and revisit confounding assumptions.",
        "why_it_matters": "A placebo effect suggests unmeasured confounding or leakage.",
        "next_checks": ["Review placebo design.", "Run sensitivity analysis."],
        "success_criteria": ["Placebo effect is near zero under expected null."],
        "references": ["negative controls", "placebo test", "sensitivity analysis"],
    },
    {
        "id": "heterogeneous_effects_missing",
        "title": "Heterogeneous treatment-effect audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["heterogeneous treatment effects causal machine learning causal forest", "uplift modeling treatment effect heterogeneity"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Estimate and validate treatment effects by prespecified subgroups.",
        "why_it_matters": "Average effects can hide benefit/harm heterogeneity.",
        "next_checks": ["Define subgroups before modeling.", "Compare causal forest or meta-learners."],
        "success_criteria": ["HTE claims pass validation or are removed."],
        "references": ["causal forest", "uplift modeling", "HTE"],
    },
    {
        "id": "sensitivity_analysis_missing",
        "title": "Unmeasured-confounding sensitivity analysis is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["unmeasured confounding sensitivity analysis causal inference E value", "robustness value causal inference sensitivity"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add sensitivity analysis for unmeasured confounding before interpreting effects.",
        "why_it_matters": "Observational causal claims require stress tests for hidden confounding.",
        "next_checks": ["Compute E-values or robustness values.", "Document identifying assumptions."],
        "success_criteria": ["Effect conclusion survives plausible sensitivity ranges."],
        "references": ["sensitivity analysis", "E-value", "unmeasured confounding"],
    },
]

