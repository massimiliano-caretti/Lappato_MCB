"""Fairness and subgroup-performance manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import spread, summary

RUN_TAG = "fairness"

DEFAULT_THRESHOLDS = {
    "subgroup_metric_gap_warning": 0.10,
    "equal_opportunity_gap_warning": 0.10,
    "calibration_gap_warning": 0.05,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _subgroup_gap(path: Path) -> bool:
    return spread(path, "auc", "accuracy", "f1") > THRESHOLDS["subgroup_metric_gap_warning"]


def _subgroup_summary(path: Path) -> dict:
    return summary(path, "subgroup_metric_gap", spread(path, "auc", "accuracy", "f1"))


def _equal_opportunity_gap(path: Path) -> bool:
    return spread(path, "tpr", "recall", "sensitivity") > THRESHOLDS["equal_opportunity_gap_warning"]


def _equal_opportunity_summary(path: Path) -> dict:
    return summary(path, "equal_opportunity_gap", spread(path, "tpr", "recall", "sensitivity"))


def _calibration_gap(path: Path) -> bool:
    return spread(path, "ece", "brier") > THRESHOLDS["calibration_gap_warning"]


def _calibration_summary(path: Path) -> dict:
    return summary(path, "calibration_gap", spread(path, "ece", "brier"))


MANIFEST = [
    {
        "id": "subgroup_performance_gap",
        "title": "Performance differs materially across subgroups",
        "evidence": "fairness_subgroups.csv",
        "evidence_check": _subgroup_gap,
        "evidence_summary": _subgroup_summary,
        "severity": "high",
        "queries": ["subgroup performance disparity machine learning fairness evaluation", "model fairness subgroup analysis performance gaps"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report subgroup metrics and inspect data support, labels and thresholds for weak groups.",
        "why_it_matters": "Averages can hide harm concentrated in a subgroup.",
        "next_checks": ["Report confidence intervals by subgroup.", "Inspect weak-subgroup errors."],
        "success_criteria": ["Metric gap falls below 0.10 or is explicitly justified."],
        "references": ["subgroup fairness", "performance disparity", "fairness evaluation"],
    },
    {
        "id": "equal_opportunity_gap",
        "title": "Recall/TPR differs across protected groups",
        "evidence": "fairness_equal_opportunity.csv",
        "evidence_check": _equal_opportunity_gap,
        "evidence_summary": _equal_opportunity_summary,
        "severity": "high",
        "queries": ["equal opportunity fairness machine learning true positive rate gap", "fair classification equal opportunity constraints"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Evaluate group-specific thresholds or fairness-constrained training after domain review.",
        "why_it_matters": "TPR gaps mean qualified positives are missed unevenly.",
        "next_checks": ["Plot ROC/PR by group.", "Test threshold policies by group."],
        "success_criteria": ["TPR gap falls below 0.10 under approved policy."],
        "references": ["equal opportunity", "fair classification", "thresholding"],
    },
    {
        "id": "subgroup_calibration_gap",
        "title": "Calibration differs across subgroups",
        "evidence": "fairness_calibration.csv",
        "evidence_check": _calibration_gap,
        "evidence_summary": _calibration_summary,
        "severity": "medium",
        "queries": ["subgroup calibration fairness machine learning", "multi calibration algorithm fairness risk prediction"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report calibration curves by subgroup and compare multi-calibration or subgroup recalibration.",
        "why_it_matters": "Equal scores should mean comparable risk across groups.",
        "next_checks": ["Plot reliability by group.", "Compare subgroup recalibration."],
        "success_criteria": ["Subgroup ECE/Brier gap falls below 0.05."],
        "references": ["calibration fairness", "multi-calibration", "risk prediction"],
    },
    {
        "id": "sensitive_attribute_policy_missing",
        "title": "Sensitive-attribute policy is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["sensitive attributes fairness machine learning governance", "fairness evaluation protected attributes data collection"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Document which subgroup attributes are available, allowed and appropriate for evaluation.",
        "why_it_matters": "Fairness measurement depends on lawful and context-appropriate subgroup data.",
        "next_checks": ["Document attribute provenance.", "Review policy constraints."],
        "success_criteria": ["Fairness audit scope is documented."],
        "references": ["fairness governance", "protected attributes", "model audit"],
    },
    {
        "id": "intersectional_audit_missing",
        "title": "Intersectional subgroup audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["intersectional fairness machine learning subgroup audit", "fairness gerrymandering subgroup machine learning"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add intersectional subgroup slices with minimum support rules.",
        "why_it_matters": "Single-axis fairness can miss harms at intersections.",
        "next_checks": ["Generate supported intersectional slices.", "Flag low-support slices separately."],
        "success_criteria": ["Critical intersections have measured metrics or documented support limits."],
        "references": ["intersectional fairness", "fairness gerrymandering", "subgroup audit"],
    },
]

