"""Survival-analysis manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import min_value, summary, values

RUN_TAG = "survival"

DEFAULT_THRESHOLDS = {
    "c_index_warning": 0.65,
    "calibration_slope_gap_warning": 0.20,
    "ph_p_warning": 0.05,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _c_index_low(path: Path) -> bool:
    vals = values(path, "c_index", "concordance")
    return bool(vals and min(vals) < THRESHOLDS["c_index_warning"])


def _c_index_summary(path: Path) -> dict:
    return summary(path, "min_c_index", min_value(path, "c_index", "concordance"))


def _calibration_slope_off(path: Path) -> bool:
    slopes = values(path, "calibration_slope", "slope")
    return any(abs(s - 1.0) > THRESHOLDS["calibration_slope_gap_warning"] for s in slopes)


def _calibration_summary(path: Path) -> dict:
    slopes = values(path, "calibration_slope", "slope")
    gap = max((abs(s - 1.0) for s in slopes), default=0.0)
    return summary(path, "max_slope_gap_from_1", gap)


def _ph_violation(path: Path) -> bool:
    pvals = values(path, "p", "p_value", "schoenfeld_p")
    return bool(pvals and min(pvals) < THRESHOLDS["ph_p_warning"])


def _ph_summary(path: Path) -> dict:
    return summary(path, "min_ph_test_p", min_value(path, "p", "p_value", "schoenfeld_p"))


MANIFEST = [
    {
        "id": "low_discrimination",
        "title": "Survival discrimination is low",
        "evidence": "survival_discrimination.csv",
        "evidence_check": _c_index_low,
        "evidence_summary": _c_index_summary,
        "severity": "medium",
        "queries": ["survival analysis machine learning C index time dependent AUC", "random survival forest gradient boosting survival prediction"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compare Cox, random survival forests and gradient-boosted survival under time-aware validation.",
        "why_it_matters": "Low concordance means the model poorly orders patient risk.",
        "next_checks": ["Report C-index by time horizon.", "Compare time-dependent AUC."],
        "success_criteria": ["C-index exceeds 0.65 or improves over baseline."],
        "references": ["C-index", "time-dependent AUC", "survival prediction"],
    },
    {
        "id": "survival_calibration_slope_off",
        "title": "Survival calibration slope is far from 1",
        "evidence": "survival_calibration.csv",
        "evidence_check": _calibration_slope_off,
        "evidence_summary": _calibration_summary,
        "severity": "high",
        "queries": ["survival model calibration slope prediction", "calibration survival analysis risk prediction"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add horizon-specific calibration plots and recalibrate baseline risk if needed.",
        "why_it_matters": "Survival risk estimates are used for timing decisions, not only ranking.",
        "next_checks": ["Plot calibration by horizon.", "Report integrated Brier score."],
        "success_criteria": ["Calibration slope is within 0.20 of 1."],
        "references": ["survival calibration", "Brier score", "risk prediction"],
    },
    {
        "id": "proportional_hazards_violation",
        "title": "Proportional-hazards assumption may be violated",
        "evidence": "survival_ph_test.csv",
        "evidence_check": _ph_violation,
        "evidence_summary": _ph_summary,
        "severity": "medium",
        "queries": ["proportional hazards violation Schoenfeld residuals time varying effects", "Cox model time varying covariates survival"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use time-varying effects, stratification or non-PH survival models.",
        "why_it_matters": "PH violations can bias hazard estimates and interpretation.",
        "next_checks": ["Inspect Schoenfeld residuals.", "Fit time-varying coefficient models."],
        "success_criteria": ["PH violations are resolved or model choice changes."],
        "references": ["Schoenfeld residuals", "time-varying effects", "Cox model"],
    },
    {
        "id": "censoring_audit_missing",
        "title": "Censoring mechanism audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["informative censoring survival analysis machine learning", "inverse probability censoring weights survival prediction"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report censoring by subgroup and consider IPCW sensitivity analysis.",
        "why_it_matters": "Informative censoring can distort survival estimates.",
        "next_checks": ["Plot censoring by subgroup.", "Run IPCW sensitivity analysis."],
        "success_criteria": ["Censoring assumptions are documented and stress-tested."],
        "references": ["informative censoring", "IPCW", "survival analysis"],
    },
    {
        "id": "competing_risks_missing",
        "title": "Competing-risks analysis is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["competing risks survival machine learning cause specific hazard Fine Gray", "competing risk prediction cumulative incidence"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Check whether competing events exist and compare cause-specific or Fine-Gray models.",
        "why_it_matters": "Ignoring competing risks can overestimate event probabilities.",
        "next_checks": ["Count competing events.", "Report cumulative incidence."],
        "success_criteria": ["Competing-risk handling matches the clinical question."],
        "references": ["competing risks", "Fine Gray", "cumulative incidence"],
    },
]

