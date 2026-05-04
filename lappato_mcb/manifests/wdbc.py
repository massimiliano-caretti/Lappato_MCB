"""WDBC manifest for LAPPATO_MCB.

Five entries (3 evidence-gated + 2 structural) that match the CSVs
produced by ``examples/demo_wdbc.py`` on the Wisconsin Breast Cancer
Diagnostic dataset (loaded via ``sklearn.datasets.load_breast_cancer``).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

RUN_TAG = "wdbc"


# ─── Activation thresholds (auditable & overridable) ───────────────────
# Each threshold is exposed here with its empirical/literature
# justification rather than hidden inside the detector functions. This
# makes activation auditable (reviewers can read the whole policy in
# one place) and tunable (a custom manifest can replace
# ``DEFAULT_THRESHOLDS`` with a domain-specific dict). The
# ``THRESHOLDS`` mapping referenced by the detectors below can be
# rebound at module import time by tooling that wants to override the
# defaults; see :func:`override_thresholds` for the recommended path.
DEFAULT_THRESHOLDS: dict[str, float] = {
    # Class-imbalance gap (mean of |precision-recall| asymmetries
    # between the two classes). 0.05 reflects the conventional 5%
    # absolute difference flagged in clinical-prediction reporting
    # checklists (TRIPOD-AI, Steyerberg 2019, ch. 15).
    "class_gap_warning": 0.05,
    "class_gap_high": 0.15,
    "class_gap_medium": 0.08,
    # Mean Brier score across folds. The 0.10 threshold is the
    # commonly-cited "well-calibrated binary classifier" boundary
    # (Steyerberg 2019; Riley et al. 2019 prognostic-model guide).
    # Critical/high mirror the same source at 0.20 / 0.15.
    "brier_warning": 0.10,
    "brier_high": 0.15,
    "brier_critical": 0.20,
}


THRESHOLDS: dict[str, float] = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    """Mutate the active threshold table without rewriting the manifest.

    Intended for callers that want to adapt the WDBC manifest to a
    different cohort (e.g. higher Brier tolerance for a multi-class
    extension) without forking the file. Unknown keys are accepted so
    that downstream extensions can add new thresholds without touching
    this module.
    """
    THRESHOLDS.update(dict(overrides))


# ─── Evidence checkers ────────────────────────────────────────────────
def _class_imbalance_present(csv_path: Path) -> bool:
    """Active when the mean M-vs-B precision/recall gap exceeds the
    configured warning threshold (default 0.05 — see ``THRESHOLDS``)."""
    return _class_imbalance_summary(csv_path)["mean_class_gap"] > THRESHOLDS["class_gap_warning"]


def _class_imbalance_summary(csv_path: Path) -> dict:
    gaps = []
    if csv_path.exists():
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    pm = float(r.get("precision_M", "nan"))
                    rm = float(r.get("recall_M", "nan"))
                    pb = float(r.get("precision_B", "nan"))
                    rb = float(r.get("recall_B", "nan"))
                    gaps.append(abs((pm + rm) / 2 - (pb + rb) / 2))
                except ValueError:
                    continue
    mean_gap = (sum(gaps) / len(gaps)) if gaps else 0.0
    return {"rows": len(gaps), "mean_class_gap": round(mean_gap, 4)}


def _class_imbalance_severity(csv_path: Path) -> str:
    gap = _class_imbalance_summary(csv_path)["mean_class_gap"]
    if gap >= THRESHOLDS["class_gap_high"]:
        return "high"
    if gap >= THRESHOLDS["class_gap_medium"]:
        return "medium"
    return "low"


def _calibration_poor(csv_path: Path) -> bool:
    """Active when the mean Brier score across folds exceeds the
    configured warning threshold (default 0.10 — see ``THRESHOLDS``)."""
    return _calibration_summary(csv_path)["mean_brier"] > THRESHOLDS["brier_warning"]


def _calibration_summary(csv_path: Path) -> dict:
    bs = []
    if csv_path.exists():
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                v = r.get("brier") or r.get("brier_score")
                if v is None:
                    continue
                try:
                    bs.append(float(v))
                except ValueError:
                    continue
    mean_brier = (sum(bs) / len(bs)) if bs else 0.0
    return {"rows": len(bs), "mean_brier": round(mean_brier, 4)}


def _calibration_severity(csv_path: Path) -> str:
    brier = _calibration_summary(csv_path)["mean_brier"]
    if brier >= THRESHOLDS["brier_critical"]:
        return "critical"
    if brier >= THRESHOLDS["brier_high"]:
        return "high"
    if brier >= THRESHOLDS["brier_warning"]:
        return "medium"
    return "low"


def _features_redundant(csv_path: Path) -> bool:
    """True when top-3 features by gain belong to the same WDBC group."""
    top = _features_summary(csv_path)["top_features"]
    if len(top) < 3:
        return False
    suffixes = {n.rsplit("_", 1)[-1] for n in top if "_" in n}
    if len(suffixes) == 1:
        return True
    bases = {n.rsplit("_", 1)[0].rsplit("_", 1)[0] for n in top if "_" in n}
    return len(bases) == 1


def _features_summary(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"rows": 0, "top_features": []}
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))[:3]
    names = [r.get("feature", "") for r in rows]
    return {"rows": len(rows), "top_features": names}


# ─── Manifest ─────────────────────────────────────────────────────────
MANIFEST: list[dict] = [
    {
        "id": "class_imbalance",
        "title": "Class imbalance leaks into per-class precision/recall asymmetry",
        "evidence": "wdbc_per_fold.csv",
        "evidence_check": _class_imbalance_present,
        "evidence_summary": _class_imbalance_summary,
        "severity": _class_imbalance_severity,
        "queries": [
            "class imbalance binary classification {domain} {model_family} calibration",
            "focal loss reweighting {model_family} imbalanced medical {task}",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Add class-balanced loss (focal or scale_pos_weight in LightGBM) "
            "and report per-class precision/recall, not just balanced accuracy."
        ),
        "why_it_matters": (
            "In clinical binary classification, aggregate accuracy can hide "
            "minority-class harm; precision/recall asymmetry directly affects "
            "false-negative and false-positive review burden."
        ),
        "next_checks": [
            "Report confusion matrix at clinically relevant thresholds.",
            "Compare class weighting, focal loss, threshold moving and PR-AUC.",
            "Inspect whether imbalance is label-level or subgroup-level.",
        ],
        "success_criteria": [
            "Mean class-gap decreases below 0.05 without lowering balanced accuracy.",
            "Minority recall improves at the selected operating threshold.",
        ],
        "references": [
            "class-balanced loss",
            "focal loss",
            "clinical risk prediction model evaluation",
        ],
    },
    {
        "id": "low_calibration",
        "title": "Probabilistic output is poorly calibrated (Brier > 0.10)",
        "evidence": "wdbc_calibration.csv",
        "evidence_check": _calibration_poor,
        "evidence_summary": _calibration_summary,
        "severity": _calibration_severity,
        "queries": [
            "isotonic Platt calibration {model_family} probability medical",
            "Brier decomposition reliability resolution {domain} classifier",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Wrap the LightGBM head in a CalibratedClassifierCV (isotonic) "
            "trained on a held-out fold; report Brier reliability/resolution "
            "decomposition (Murphy 1973)."
        ),
        "why_it_matters": (
            "Clinical model probabilities are often consumed as risk estimates; "
            "poor calibration can make a well-discriminating classifier unsafe "
            "for threshold-based decisions."
        ),
        "next_checks": [
            "Plot reliability diagram by fold.",
            "Report ECE, Brier score and calibration slope/intercept.",
            "Compare sigmoid, isotonic and beta calibration on held-out folds.",
        ],
        "success_criteria": [
            "Mean Brier score falls below 0.10.",
            "Calibration slope approaches 1.0 and intercept approaches 0.0.",
        ],
        "references": [
            "Brier score decomposition",
            "temperature scaling calibration",
            "fit-on-test calibration evaluation",
        ],
    },
    {
        "id": "marker_redundancy",
        "title": "Top-3 features by gain belong to the same WDBC group / base measurement",
        "evidence": "wdbc_feature_importance.csv",
        "evidence_check": _features_redundant,
        "evidence_summary": _features_summary,
        "severity": "medium",
        "queries": [
            "feature redundancy SHAP interaction {domain} biomarker selection",
            "sparse group lasso correlated features clinical classifier",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Run SHAP-interaction analysis over the top-10; consider "
            "sparse-group-lasso to penalise within-group redundancy "
            "across the (mean / se / worst) triplets of WDBC."
        ),
        "why_it_matters": (
            "A model dominated by multiple correlated measurements can look "
            "stable while depending on one biological signal; redundancy limits "
            "interpretability and external robustness."
        ),
        "next_checks": [
            "Cluster features by correlation before importance analysis.",
            "Compute permutation importance grouped by WDBC measurement family.",
            "Compare top features across CV seeds.",
        ],
        "success_criteria": [
            "Top-3 features no longer collapse to one measurement family.",
            "Grouped permutation importance remains stable across folds.",
        ],
        "references": [
            "SHAP interaction values",
            "group lasso correlated features",
            "biomarker redundancy selection",
        ],
    },
    {
        "id": "tabular_baseline_alternatives",
        "title": "LightGBM is the only base learner — TabPFN-2.5 / CatBoost unrepresented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": [
            "TabPFN small dataset {domain} benchmark medical 2025",
            "CatBoost {model_family} stacking heterogeneous tabular clinical 2025",
            "open source software tabular machine learning benchmark clinical",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": (
            "Add a TabPFN-2.5 baseline alongside LightGBM on the 30 WDBC "
            "features; if comparable or better, add to a stacking head "
            "with a logistic meta-learner."
        ),
        "why_it_matters": (
            "Small tabular clinical datasets are exactly where model-family "
            "choice can dominate results; a single GBDT baseline is weak "
            "evidence for methodological adequacy."
        ),
        "next_checks": [
            "Benchmark CatBoost, logistic elastic-net, TabPFN and AutoGluon.",
            "Use repeated stratified CV with identical splits.",
            "Compare discrimination, calibration and net benefit together.",
        ],
        "success_criteria": [
            "At least one non-LightGBM baseline is reported.",
            "Chosen model is justified by performance stability, not only mean score.",
        ],
        "references": [
            "TabPFN foundation model tabular data",
            "CatBoost clinical risk prediction",
            "TabArena tabular benchmark",
        ],
    },
    {
        "id": "decision_curve_missing",
        "title": "No decision-curve analysis — accuracy is not clinical net benefit",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": [
            "decision curve analysis Vickers Elkin clinical classifier 2025",
            "net benefit calibration tutorial medical machine learning 2025",
            "open source software decision curve analysis clinical prediction",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": (
            "Add a decision-curve-analysis report (Vickers & Elkin 2006): "
            "compute net benefit at threshold range [0.05, 0.50] vs "
            "treat-all and treat-none baselines."
        ),
        "why_it_matters": (
            "A clinically useful classifier is not just accurate; it must offer "
            "net benefit at plausible decision thresholds."
        ),
        "next_checks": [
            "Compute net benefit over a clinically meaningful threshold grid.",
            "Compare against treat-all and treat-none policies.",
            "Report calibration and decision curves in the same section.",
        ],
        "success_criteria": [
            "Model net benefit is positive over at least one target threshold band.",
            "Threshold choice is stated explicitly.",
        ],
        "references": [
            "decision curve analysis",
            "clinical net benefit",
            "risk prediction reporting",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG"]
