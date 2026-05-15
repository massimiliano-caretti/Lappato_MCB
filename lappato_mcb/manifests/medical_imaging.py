"""Medical-imaging manifest for classification and segmentation pipelines."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import min_value, summary, values

RUN_TAG = "medical_imaging"

DEFAULT_THRESHOLDS = {
    "dice_warning": 0.75,
    "small_lesion_recall_warning": 0.70,
    "site_gap_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _segmentation_low(path: Path) -> bool:
    vals = values(path, "dice", "iou")
    return bool(vals and min(vals) < THRESHOLDS["dice_warning"])


def _segmentation_summary(path: Path) -> dict:
    return summary(path, "min_dice_or_iou", min_value(path, "dice", "iou"))


def _small_lesion_low(path: Path) -> bool:
    vals = values(path, "small_lesion_recall", "small_object_recall")
    return bool(vals and min(vals) < THRESHOLDS["small_lesion_recall_warning"])


def _small_lesion_summary(path: Path) -> dict:
    return summary(path, "min_small_lesion_recall", min_value(path, "small_lesion_recall", "small_object_recall"))


def _site_shift_high(path: Path) -> bool:
    vals = values(path, "auc", "dice", "f1")
    return (max(vals) - min(vals)) > THRESHOLDS["site_gap_warning"] if vals else False


def _site_shift_summary(path: Path) -> dict:
    vals = values(path, "auc", "dice", "f1")
    return {"rows": len(vals), "site_score_gap": round((max(vals) - min(vals)) if vals else 0.0, 4)}


MANIFEST = [
    {
        "id": "low_segmentation_overlap",
        "title": "Segmentation overlap metric is low",
        "evidence": "medical_segmentation.csv",
        "evidence_check": _segmentation_low,
        "evidence_summary": _segmentation_summary,
        "severity": "high",
        "queries": ["medical image segmentation Dice loss boundary loss", "medical imaging segmentation uncertainty calibration"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compare Dice, focal/Tversky and boundary-aware losses; inspect boundary failures.",
        "why_it_matters": "Small overlap errors can be clinically meaningful in segmentation.",
        "next_checks": ["Review worst masks.", "Report Dice by anatomy or lesion size."],
        "success_criteria": ["Minimum subgroup Dice exceeds 0.75."],
        "references": ["Dice loss", "Tversky loss", "boundary loss"],
    },
    {
        "id": "small_lesion_recall_low",
        "title": "Small-lesion or small-object recall is low",
        "evidence": "medical_lesion_size.csv",
        "evidence_check": _small_lesion_low,
        "evidence_summary": _small_lesion_summary,
        "severity": "high",
        "queries": ["small lesion detection medical imaging recall focal loss", "class imbalance lesion segmentation medical imaging"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use lesion-size stratified evaluation and compare focal/Tversky loss or patch sampling.",
        "why_it_matters": "Aggregate imaging scores often hide missed small findings.",
        "next_checks": ["Stratify recall by lesion size.", "Inspect false negatives."],
        "success_criteria": ["Small-lesion recall exceeds 0.70."],
        "references": ["small lesion detection", "focal loss", "Tversky loss"],
    },
    {
        "id": "site_shift",
        "title": "Performance varies across scanner/site cohorts",
        "evidence": "medical_site_performance.csv",
        "evidence_check": _site_shift_high,
        "evidence_summary": _site_shift_summary,
        "severity": "high",
        "queries": ["medical imaging domain shift scanner site generalization", "federated domain adaptation medical imaging"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report per-site metrics and evaluate harmonization, augmentation or domain adaptation.",
        "why_it_matters": "Scanner and site shift are common external-validity failures.",
        "next_checks": ["Plot per-site metrics.", "Run leave-one-site-out validation."],
        "success_criteria": ["Worst-site gap falls below 0.10."],
        "references": ["domain shift", "scanner shift", "external validation"],
    },
    {
        "id": "uncertainty_missing",
        "title": "Uncertainty quantification is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["uncertainty quantification medical imaging deep learning calibration", "selective prediction medical image diagnosis uncertainty"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add uncertainty maps or selective prediction and validate calibration by subgroup.",
        "why_it_matters": "Medical imaging models need uncertainty for triage and review.",
        "next_checks": ["Report ECE by site.", "Inspect uncertainty on false negatives."],
        "success_criteria": ["Uncertainty correlates with error and supports abstention."],
        "references": ["uncertainty quantification", "selective prediction", "medical calibration"],
    },
    {
        "id": "external_validation_missing",
        "title": "External validation cohort is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["external validation medical imaging AI reporting checklist", "TRIPOD AI medical imaging external validation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add an external or leave-site-out validation split before claiming generalization.",
        "why_it_matters": "Single-site imaging results frequently overstate deployment performance.",
        "next_checks": ["Identify external cohort.", "Report model selection separate from validation."],
        "success_criteria": ["External validation metrics are reported with confidence intervals."],
        "references": ["external validation", "TRIPOD AI", "medical AI reporting"],
    },
]

