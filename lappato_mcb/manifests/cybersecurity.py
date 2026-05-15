"""Cybersecurity ML manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import first_last_ratio, mean, min_value, summary, values

RUN_TAG = "cybersecurity"

DEFAULT_THRESHOLDS = {
    "false_positive_rate_warning": 0.02,
    "attack_recall_warning": 0.80,
    "drift_ratio_warning": 1.5,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _fp_high(path: Path) -> bool:
    return mean(values(path, "false_positive_rate", "fpr")) > THRESHOLDS["false_positive_rate_warning"]


def _fp_summary(path: Path) -> dict:
    return summary(path, "mean_fpr", mean(values(path, "false_positive_rate", "fpr")))


def _attack_recall_low(path: Path) -> bool:
    vals = values(path, "recall", "detection_rate", "tpr")
    return bool(vals and min(vals) < THRESHOLDS["attack_recall_warning"])


def _attack_recall_summary(path: Path) -> dict:
    return summary(path, "min_attack_recall", min_value(path, "recall", "detection_rate", "tpr"))


def _drift_high(path: Path) -> bool:
    return first_last_ratio(path, "alert_rate", "event_rate") > THRESHOLDS["drift_ratio_warning"]


def _drift_summary(path: Path) -> dict:
    return summary(path, "last_first_alert_ratio", first_last_ratio(path, "alert_rate", "event_rate"))


MANIFEST = [
    {
        "id": "alert_false_positive_burden",
        "title": "Security alert false-positive rate is high",
        "evidence": "cyber_alert_rates.csv",
        "evidence_check": _fp_high,
        "evidence_summary": _fp_summary,
        "severity": "high",
        "queries": ["intrusion detection false positive rate machine learning", "cybersecurity alert triage machine learning precision recall"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Tune thresholds against analyst alert budget and report precision-recall by attack family.",
        "why_it_matters": "False positives create alert fatigue and missed incidents.",
        "next_checks": ["Measure alerts per analyst-day.", "Review top false positives."],
        "success_criteria": ["FPR falls below 2% at required recall."],
        "references": ["intrusion detection", "alert triage", "precision recall"],
    },
    {
        "id": "attack_family_recall_low",
        "title": "Detection recall is low for some attack families",
        "evidence": "cyber_attack_families.csv",
        "evidence_check": _attack_recall_low,
        "evidence_summary": _attack_recall_summary,
        "severity": "critical",
        "queries": ["malware intrusion detection attack family recall machine learning", "cyber threat detection class imbalance recall"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report recall by attack family and add targeted examples or family-aware loss weighting.",
        "why_it_matters": "Missing a rare attack family can matter more than average accuracy.",
        "next_checks": ["Inspect missed attack families.", "Compare class weighting."],
        "success_criteria": ["Minimum attack-family recall exceeds 0.80 or risk is accepted."],
        "references": ["attack family", "intrusion detection", "imbalanced learning"],
    },
    {
        "id": "traffic_drift",
        "title": "Network or event traffic drift is visible",
        "evidence": "cyber_drift.csv",
        "evidence_check": _drift_high,
        "evidence_summary": _drift_summary,
        "severity": "medium",
        "queries": ["concept drift intrusion detection cybersecurity machine learning", "network traffic drift anomaly detection"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Separate benign traffic drift from adversarial novelty and add drift monitoring.",
        "why_it_matters": "Security models face both operational drift and adaptive adversaries.",
        "next_checks": ["Plot alert/event rates over time.", "Audit drifted features."],
        "success_criteria": ["Drift response is documented and alert ratio stabilizes."],
        "references": ["concept drift", "intrusion detection", "network traffic"],
    },
    {
        "id": "adversarial_evasion_missing",
        "title": "Adversarial evasion testing is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["adversarial machine learning cybersecurity evasion attacks", "malware detection adversarial evasion robustness"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add evasion tests for feature manipulation and adversarially adapted attacks.",
        "why_it_matters": "Attackers adapt to detectors after deployment.",
        "next_checks": ["Define attacker capabilities.", "Run evasion simulations."],
        "success_criteria": ["Known evasion tests do not break required recall."],
        "references": ["adversarial ML", "evasion attacks", "malware detection"],
    },
    {
        "id": "dataset_staleness_missing",
        "title": "Dataset staleness audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["cybersecurity dataset staleness intrusion detection benchmark drift", "malware dataset temporal validation machine learning"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use temporal validation and document dataset age relative to current threats.",
        "why_it_matters": "Old security benchmarks often fail to represent current adversaries.",
        "next_checks": ["Report sample collection dates.", "Use time-based validation."],
        "success_criteria": ["Evaluation includes recent or temporally held-out threats."],
        "references": ["temporal validation", "dataset staleness", "cybersecurity benchmark"],
    },
]

