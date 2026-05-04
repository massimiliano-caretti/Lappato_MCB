"""Anomaly-detection manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import first_last_ratio, mean, min_value, summary, values

RUN_TAG = "anomaly_detection"

DEFAULT_THRESHOLDS = {
    "false_positive_rate_warning": 0.05,
    "recall_warning": 0.70,
    "threshold_ratio_warning": 1.5,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _false_positive_high(path: Path) -> bool:
    return mean(values(path, "false_positive_rate", "fpr")) > THRESHOLDS["false_positive_rate_warning"]


def _false_positive_summary(path: Path) -> dict:
    return summary(path, "mean_fpr", mean(values(path, "false_positive_rate", "fpr")))


def _recall_low(path: Path) -> bool:
    vals = values(path, "recall", "true_positive_rate", "tpr")
    return bool(vals and min(vals) < THRESHOLDS["recall_warning"])


def _recall_summary(path: Path) -> dict:
    return summary(path, "min_recall", min_value(path, "recall", "true_positive_rate", "tpr"))


def _threshold_unstable(path: Path) -> bool:
    return first_last_ratio(path, "threshold", "score_cutoff") > THRESHOLDS["threshold_ratio_warning"]


def _threshold_summary(path: Path) -> dict:
    return summary(path, "last_first_threshold_ratio", first_last_ratio(path, "threshold", "score_cutoff"))


MANIFEST = [
    {
        "id": "false_positive_burden",
        "title": "False-positive burden is high",
        "evidence": "anomaly_rates.csv",
        "evidence_check": _false_positive_high,
        "evidence_summary": _false_positive_summary,
        "severity": "medium",
        "queries": ["anomaly detection false positive rate threshold calibration", "unsupervised anomaly detection precision recall evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Tune thresholds on a labeled validation slice and report precision-recall, not only ROC-AUC.",
        "why_it_matters": "High false-positive rates make anomaly systems unusable operationally.",
        "next_checks": ["Plot alert volume by day.", "Review top false positives."],
        "success_criteria": ["FPR falls below 5% at target recall."],
        "references": ["anomaly detection evaluation", "threshold calibration", "precision recall"],
    },
    {
        "id": "rare_event_recall_low",
        "title": "Rare-event recall is low",
        "evidence": "anomaly_recall.csv",
        "evidence_check": _recall_low,
        "evidence_summary": _recall_summary,
        "severity": "high",
        "queries": ["rare event detection anomaly recall imbalanced data", "semi supervised anomaly detection rare events"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add rare-event focused validation, semi-supervised baselines and recall-at-alert-budget.",
        "why_it_matters": "An anomaly detector that misses rare positives fails its core purpose.",
        "next_checks": ["Measure recall by anomaly type.", "Inspect missed anomalies."],
        "success_criteria": ["Minimum anomaly-type recall exceeds 0.70."],
        "references": ["rare event detection", "semi-supervised anomaly detection", "alert budget"],
    },
    {
        "id": "threshold_instability",
        "title": "Anomaly threshold drifts or changes sharply across windows",
        "evidence": "anomaly_thresholds.csv",
        "evidence_check": _threshold_unstable,
        "evidence_summary": _threshold_summary,
        "severity": "medium",
        "queries": ["anomaly detection threshold stability concept drift", "online anomaly detection drift adaptation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add rolling-window calibration and drift-aware threshold monitoring.",
        "why_it_matters": "Unstable thresholds create unpredictable alert volume.",
        "next_checks": ["Plot threshold by window.", "Compare fixed vs adaptive thresholding."],
        "success_criteria": ["Threshold ratio remains below 1.5 or drift is explained."],
        "references": ["concept drift", "online anomaly detection", "adaptive threshold"],
    },
    {
        "id": "contamination_sensitivity_missing",
        "title": "Contamination sensitivity is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["anomaly detection contamination sensitivity robust outlier detection", "isolation forest contamination parameter evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Sweep contamination assumptions and report sensitivity of alert precision and volume.",
        "why_it_matters": "Many anomaly methods are highly sensitive to assumed contamination.",
        "next_checks": ["Sweep contamination values.", "Track alert overlap across sweeps."],
        "success_criteria": ["Chosen threshold is stable across plausible contamination levels."],
        "references": ["contamination", "outlier detection", "isolation forest"],
    },
    {
        "id": "drift_monitoring_missing",
        "title": "Input drift monitoring is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["concept drift anomaly detection monitoring", "data drift unsupervised anomaly detection"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add feature drift metrics and separate drift alerts from anomaly alerts.",
        "why_it_matters": "Drift can look like anomalies and inflate alert volume.",
        "next_checks": ["Measure PSI/KS by feature.", "Track drift before anomaly spikes."],
        "success_criteria": ["Drift and anomaly alert pathways are distinguishable."],
        "references": ["concept drift", "data drift", "anomaly monitoring"],
    },
]

