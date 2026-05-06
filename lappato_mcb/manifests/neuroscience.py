"""Neuroscience manifest — fMRI, EEG/MEG, neural-recording pipelines.

For pipelines on functional imaging, electrophysiology, neural decoding,
brain-state classification, and connectomics. Expected CSVs in
``checkpoints/``:

  - ``neuro_motion.csv``: ``subject,framewise_displacement``
  - ``neuro_subject_split.csv``: ``subject,train,test``
  - ``neuro_artifact.csv``: ``trial,line_noise,blink,muscle``
  - ``neuro_smoothing.csv``: ``analysis,fwhm_mm,voxel_mm``
  - ``neuro_multiple_comparison.csv``: ``analysis,n_tests,corrected``
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import f, max_value, rows, summary

RUN_TAG = "neuroscience"

DEFAULT_THRESHOLDS = {
    "framewise_displacement_warning": 0.5,   # mm
    "subject_overlap_warning": 0.0,
    "artifact_share_warning": 0.20,
    "smoothing_to_voxel_ratio_warning": 1.5,
    "uncorrected_share_warning": 0.0,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _motion_high(path: Path) -> bool:
    return max_value(path, "framewise_displacement", "fd", "motion") > THRESHOLDS["framewise_displacement_warning"]


def _motion_summary(path: Path) -> dict:
    return summary(path, "max_framewise_displacement", max_value(path, "framewise_displacement", "fd", "motion"))


def _subject_overlap_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    overlap = 0
    seen = 0
    for row in items:
        in_train = row.get("train") or "0"
        in_test = row.get("test") or "0"
        try:
            if int(float(in_train)) and int(float(in_test)):
                overlap += 1
        except (TypeError, ValueError):
            continue
        seen += 1
    return (overlap / seen) if seen else 0.0


def _subject_split_violation(path: Path) -> bool:
    return _subject_overlap_share(path) > THRESHOLDS["subject_overlap_warning"]


def _subject_split_summary(path: Path) -> dict:
    return summary(path, "subject_overlap_share", _subject_overlap_share(path))


def _artifact_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    flagged = 0
    for row in items:
        if any(
            str(row.get(c, "")).strip().lower() in ("1", "true", "yes")
            for c in ("line_noise", "blink", "muscle", "artifact")
        ):
            flagged += 1
    return flagged / len(items)


def _artifact_high(path: Path) -> bool:
    return _artifact_share(path) > THRESHOLDS["artifact_share_warning"]


def _artifact_summary(path: Path) -> dict:
    return summary(path, "artifact_share", _artifact_share(path))


def _smoothing_to_voxel_ratio(path: Path) -> float:
    ratios: list[float] = []
    for row in rows(path):
        fwhm = f(row, "fwhm_mm", "fwhm")
        voxel = f(row, "voxel_mm", "voxel_size")
        if fwhm is None or voxel is None or voxel == 0:
            continue
        ratios.append(fwhm / voxel)
    return min(ratios) if ratios else 0.0


def _smoothing_below_voxel(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    return _smoothing_to_voxel_ratio(path) < THRESHOLDS["smoothing_to_voxel_ratio_warning"]


def _smoothing_summary(path: Path) -> dict:
    return summary(path, "min_smoothing_to_voxel_ratio", _smoothing_to_voxel_ratio(path))


def _uncorrected_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    seen = 0
    for row in items:
        v = row.get("corrected") or row.get("multiple_comparison") or row.get("mc_correction")
        if v is None or v == "":
            continue
        seen += 1
        if str(v).strip().lower() in ("0", "false", "no", "none", "uncorrected"):
            bad += 1
    return (bad / seen) if seen else 0.0


def _multiple_comparison_uncorrected(path: Path) -> bool:
    return _uncorrected_share(path) > THRESHOLDS["uncorrected_share_warning"]


def _multiple_comparison_summary(path: Path) -> dict:
    return summary(path, "uncorrected_share", _uncorrected_share(path))


MANIFEST: list[dict] = [
    {
        "id": "motion_high",
        "title": "Subject motion exceeds the fMRI/EEG inclusion threshold",
        "evidence": "neuro_motion.csv",
        "evidence_check": _motion_high,
        "evidence_summary": _motion_summary,
        "severity": "high",
        "queries": [
            "framewise displacement motion fMRI exclusion",
            "head motion EEG artifact rejection ICA",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply scrubbing / motion regressors and report inclusion criteria with motion summaries.",
        "why_it_matters": "Motion artefacts mimic neural signal and inflate group-level connectivity and decoding effects.",
        "next_checks": [
            "Compute mean framewise displacement per subject.",
            "Re-run analyses after scrubbing high-motion frames.",
        ],
        "success_criteria": [
            "Mean framewise displacement stays below 0.5 mm across included subjects.",
        ],
        "references": [
            "framewise displacement",
            "fMRI motion",
            "Power et al. scrubbing",
        ],
    },
    {
        "id": "subject_split_violation",
        "title": "Same subject appears in both training and evaluation splits",
        "evidence": "neuro_subject_split.csv",
        "evidence_check": _subject_split_violation,
        "evidence_summary": _subject_split_summary,
        "severity": "high",
        "queries": [
            "subject level cross validation neuroimaging decoding",
            "leave one subject out fMRI EEG generalization",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to leave-one-subject-out CV and confirm zero subject overlap across splits.",
        "why_it_matters": "Within-subject correlations make trial-level splits trivially predictable and break generalisation claims.",
        "next_checks": [
            "Hash subject ids and confirm disjoint folds.",
            "Re-train under LOSO and report the gap to the leaky baseline.",
        ],
        "success_criteria": [
            "All evaluation splits use disjoint subjects.",
        ],
        "references": [
            "leave-one-subject-out",
            "subject-level CV",
            "neuroimaging generalisation",
        ],
    },
    {
        "id": "artifact_share_high",
        "title": "Artefact-flagged trials exceed the rejection budget",
        "evidence": "neuro_artifact.csv",
        "evidence_check": _artifact_high,
        "evidence_summary": _artifact_summary,
        "severity": "high",
        "queries": [
            "EEG artifact rejection ICA blink muscle",
            "fMRI physiological noise correction RETROICOR",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply ICA-based artefact removal or physiological-noise regressors and report retained-trial counts.",
        "why_it_matters": "Pervasive artefacts bias condition-level effect estimates and damage SNR.",
        "next_checks": [
            "Audit artefact rates per condition.",
            "Re-fit the ICA decomposition with stricter component criteria.",
        ],
        "success_criteria": [
            "Artefact-flagged share stays below 20% per condition.",
        ],
        "references": [
            "ICA artifact removal",
            "RETROICOR",
            "EEG cleaning",
        ],
    },
    {
        "id": "smoothing_below_voxel",
        "title": "Smoothing kernel is too small relative to voxel size",
        "evidence": "neuro_smoothing.csv",
        "evidence_check": _smoothing_below_voxel,
        "evidence_summary": _smoothing_summary,
        "severity": "info",
        "queries": [
            "smoothing kernel FWHM voxel size group fMRI",
            "spatial smoothing inference cluster threshold",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Match smoothing FWHM to ≥ 2× voxel size for group inference, or document the rationale for a smaller kernel.",
        "why_it_matters": "Undersized smoothing inflates the false-positive rate for cluster-extent thresholding.",
        "next_checks": [
            "Quote FWHM in mm alongside voxel size.",
            "Verify smoothness estimates with AFNI's 3dFWHMx.",
        ],
        "success_criteria": [
            "Smoothing-to-voxel ratio is ≥ 1.5 for group analyses.",
        ],
        "references": [
            "spatial smoothing",
            "cluster thresholding",
            "smoothness fMRI",
        ],
    },
    {
        "id": "multiple_comparison_uncorrected",
        "title": "Inferential test reports lack multiple-comparison correction",
        "evidence": "neuro_multiple_comparison.csv",
        "evidence_check": _multiple_comparison_uncorrected,
        "evidence_summary": _multiple_comparison_summary,
        "severity": "high",
        "queries": [
            "multiple comparison correction neuroimaging FDR cluster",
            "voxelwise inference whole brain false positive",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply voxel-FDR or threshold-free cluster enhancement and report n_tests and method.",
        "why_it_matters": "Whole-brain analyses without correction generate false positives at well-known inflated rates.",
        "next_checks": [
            "Run FDR or TFCE and re-quote significance.",
            "Audit small-volume corrections.",
        ],
        "success_criteria": [
            "Every reported map has a documented correction method.",
        ],
        "references": [
            "FDR Benjamini Hochberg",
            "TFCE",
            "Eklund cluster failure",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
