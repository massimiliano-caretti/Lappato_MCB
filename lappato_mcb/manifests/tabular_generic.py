"""Generic tabular ML manifest.

Expected CSVs:
  - tabular_calibration.csv: brier or ece
  - tabular_imbalance.csv: class, support or share
  - tabular_drift.csv: feature, psi or ks
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import max_value, rows, summary, values

RUN_TAG = "tabular_generic"

DEFAULT_THRESHOLDS = {
    "brier_warning": 0.10,
    "ece_warning": 0.05,
    "minority_share_warning": 0.10,
    "psi_warning": 0.20,
    "ks_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _calibration_poor(path: Path) -> bool:
    return (
        max_value(path, "brier", "brier_score") > THRESHOLDS["brier_warning"]
        or max_value(path, "ece") > THRESHOLDS["ece_warning"]
    )


def _calibration_summary(path: Path) -> dict:
    return {
        "rows": len(rows(path)),
        "max_brier": round(max_value(path, "brier", "brier_score"), 4),
        "max_ece": round(max_value(path, "ece"), 4),
    }


def _imbalance_present(path: Path) -> bool:
    shares = values(path, "share")
    if shares:
        return min(shares) < THRESHOLDS["minority_share_warning"]
    supports = values(path, "support", "count")
    total = sum(supports)
    return bool(total and min(supports) / total < THRESHOLDS["minority_share_warning"])


def _imbalance_summary(path: Path) -> dict:
    shares = values(path, "share")
    if not shares:
        supports = values(path, "support", "count")
        total = sum(supports)
        shares = [(s / total) for s in supports] if total else []
    return summary(path, "min_class_share", min(shares) if shares else 0.0)


def _drift_detected(path: Path) -> bool:
    return (
        max_value(path, "psi") > THRESHOLDS["psi_warning"]
        or max_value(path, "ks", "ks_stat") > THRESHOLDS["ks_warning"]
    )


def _drift_summary(path: Path) -> dict:
    return {
        "rows": len(rows(path)),
        "max_psi": round(max_value(path, "psi"), 4),
        "max_ks": round(max_value(path, "ks", "ks_stat"), 4),
    }


MANIFEST = [
    {
        "id": "probability_calibration",
        "title": "Tabular probabilities are poorly calibrated",
        "evidence": "tabular_calibration.csv",
        "evidence_check": _calibration_poor,
        "evidence_summary": _calibration_summary,
        "severity": "medium",
        "queries": [
            "tabular machine learning calibration Brier ECE",
            "isotonic Platt calibration gradient boosting tabular",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add held-out calibration and report Brier, ECE and reliability curves.",
        "why_it_matters": "Poor probability calibration makes threshold-based tabular decisions brittle.",
        "next_checks": ["Plot reliability by fold.", "Compare sigmoid, isotonic and beta calibration."],
        "success_criteria": ["ECE falls below 0.05.", "Brier score improves without AUC loss."],
        "references": ["probability calibration", "Brier score", "expected calibration error"],
    },
    {
        "id": "class_imbalance",
        "title": "Minority class share is low",
        "evidence": "tabular_imbalance.csv",
        "evidence_check": _imbalance_present,
        "evidence_summary": _imbalance_summary,
        "severity": "medium",
        "queries": [
            "class imbalance tabular classification focal loss reweighting",
            "imbalanced learning gradient boosting threshold moving",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compare class weighting, threshold moving and PR-AUC model selection.",
        "why_it_matters": "Aggregate scores can hide weak minority-class recall.",
        "next_checks": ["Report per-class precision and recall.", "Inspect PR curves by subgroup."],
        "success_criteria": ["Minority recall improves.", "PR-AUC improves under the same split."],
        "references": ["imbalanced learning", "focal loss", "PR-AUC"],
    },
    {
        "id": "feature_drift",
        "title": "Feature drift is visible between train and evaluation data",
        "evidence": "tabular_drift.csv",
        "evidence_check": _drift_detected,
        "evidence_summary": _drift_summary,
        "severity": "high",
        "queries": [
            "dataset shift tabular machine learning population stability index",
            "covariate shift monitoring tabular model drift",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add drift-aware validation, PSI/KS monitoring and recent-window retraining checks.",
        "why_it_matters": "Tabular models often fail silently when input distributions move.",
        "next_checks": ["Rank drifting features.", "Compare temporal and random splits."],
        "success_criteria": ["High-PSI features are explained or mitigated.", "Temporal validation gap shrinks."],
        "references": ["dataset shift", "covariate shift", "population stability index"],
    },
    {
        "id": "leakage_audit_missing",
        "title": "No explicit target-leakage audit is represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["target leakage detection tabular machine learning", "data leakage prevention predictive modeling"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add leakage checks for post-outcome features, duplicate entities and split contamination.",
        "why_it_matters": "Leakage can create impressive validation scores that disappear in deployment.",
        "next_checks": ["Audit feature timestamps.", "Check duplicate entity overlap across splits."],
        "success_criteria": ["No post-target fields remain.", "Entity overlap across splits is zero."],
        "references": ["target leakage", "data leakage", "temporal validation"],
    },
    {
        "id": "missingness_audit_missing",
        "title": "Missingness mechanism is not explicitly audited",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["missing data imputation tabular machine learning MNAR", "missingness indicators gradient boosting tabular"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report missingness by feature/class and compare imputation plus missingness indicators.",
        "why_it_matters": "Missingness can encode measurement policy rather than the target process.",
        "next_checks": ["Plot missingness by label.", "Compare simple and model-based imputation."],
        "success_criteria": ["Missingness-sensitive features are documented.", "Imputation choice is stable."],
        "references": ["missing data", "imputation", "MNAR"],
    },
]

