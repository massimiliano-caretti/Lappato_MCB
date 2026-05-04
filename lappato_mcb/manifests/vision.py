"""Computer-vision classification/detection manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import max_value, min_value, summary, values

RUN_TAG = "vision"

DEFAULT_THRESHOLDS = {
    "min_class_score_warning": 0.60,
    "ece_warning": 0.05,
    "aug_gap_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _low_class_score(path: Path) -> bool:
    vals = values(path, "f1", "ap", "map", "dice")
    return bool(vals and min(vals) < THRESHOLDS["min_class_score_warning"])


def _class_summary(path: Path) -> dict:
    return summary(path, "min_class_score", min_value(path, "f1", "ap", "map", "dice"))


def _calibration_poor(path: Path) -> bool:
    return max_value(path, "ece", "calibration_error") > THRESHOLDS["ece_warning"]


def _calibration_summary(path: Path) -> dict:
    return summary(path, "max_ece", max_value(path, "ece", "calibration_error"))


def _augmentation_gap(path: Path) -> bool:
    return max_value(path, "aug_gap", "clean_aug_gap") > THRESHOLDS["aug_gap_warning"]


def _augmentation_summary(path: Path) -> dict:
    return summary(path, "max_aug_gap", max_value(path, "aug_gap", "clean_aug_gap"))


MANIFEST = [
    {
        "id": "low_per_class_performance",
        "title": "Some visual classes have low F1/AP",
        "evidence": "vision_per_class.csv",
        "evidence_check": _low_class_score,
        "evidence_summary": _class_summary,
        "severity": "medium",
        "queries": ["long tail image classification class imbalance augmentation", "object detection rare class average precision"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use class-aware sampling, hard-example mining and per-class error review.",
        "why_it_matters": "Mean visual metrics hide classes with poor recall or AP.",
        "next_checks": ["Sort examples by worst class.", "Compare class-balanced sampling."],
        "success_criteria": ["Worst-class score exceeds 0.60.", "Mean score does not regress materially."],
        "references": ["long-tail recognition", "hard example mining", "class-balanced sampling"],
    },
    {
        "id": "vision_calibration",
        "title": "Vision model confidence is poorly calibrated",
        "evidence": "vision_calibration.csv",
        "evidence_check": _calibration_poor,
        "evidence_summary": _calibration_summary,
        "severity": "medium",
        "queries": ["confidence calibration deep neural networks image classification", "temperature scaling computer vision calibration"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add temperature scaling and report ECE by class and confidence bin.",
        "why_it_matters": "Overconfident visual predictions are risky in review workflows.",
        "next_checks": ["Plot reliability diagrams.", "Compare temperature scaling and label smoothing."],
        "success_criteria": ["ECE falls below 0.05."],
        "references": ["temperature scaling", "ECE", "calibration"],
    },
    {
        "id": "augmentation_sensitivity",
        "title": "Evaluation is sensitive to augmentation or corruptions",
        "evidence": "vision_augmentation.csv",
        "evidence_check": _augmentation_gap,
        "evidence_summary": _augmentation_summary,
        "severity": "medium",
        "queries": ["image augmentation robustness corruption benchmark", "computer vision distribution shift augmentation robustness"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run corruption/augmentation robustness checks and tune augmentation policy.",
        "why_it_matters": "Visual models often fail under small acquisition changes.",
        "next_checks": ["Evaluate common corruptions.", "Compare augmentation policies."],
        "success_criteria": ["Augmented-clean gap falls below 0.10."],
        "references": ["robustness", "image corruptions", "augmentation policy"],
    },
    {
        "id": "domain_shift_unchecked",
        "title": "Camera/site/domain shift is not explicitly checked",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["domain adaptation computer vision dataset shift", "test time adaptation image classification domain shift"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add site/camera/domain-split evaluation and report worst-domain performance.",
        "why_it_matters": "Vision data often changes by device, site and acquisition protocol.",
        "next_checks": ["Label validation rows by domain.", "Report worst-domain score."],
        "success_criteria": ["Worst-domain score is within declared tolerance."],
        "references": ["domain adaptation", "dataset shift", "test-time adaptation"],
    },
    {
        "id": "label_noise_audit_missing",
        "title": "Label-noise audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["label noise robust learning image classification", "confident learning label errors computer vision"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Review high-loss examples and likely mislabeled images before model changes.",
        "why_it_matters": "Noisy visual labels can dominate apparent model failures.",
        "next_checks": ["Inspect high-loss examples.", "Estimate label-error rate."],
        "success_criteria": ["Suspect labels are reviewed or excluded in sensitivity analysis."],
        "references": ["label noise", "robust learning", "confident learning"],
    },
]

