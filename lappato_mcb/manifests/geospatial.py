"""Geospatial ML manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import first_last_ratio, max_value, spread, summary

RUN_TAG = "geospatial"

DEFAULT_THRESHOLDS = {
    "spatial_gap_warning": 0.10,
    "temporal_ratio_warning": 1.5,
    "cloud_error_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _spatial_gap(path: Path) -> bool:
    return spread(path, "auc", "f1", "rmse") > THRESHOLDS["spatial_gap_warning"]


def _spatial_summary(path: Path) -> dict:
    return summary(path, "spatial_metric_gap", spread(path, "auc", "f1", "rmse"))


def _temporal_degradation(path: Path) -> bool:
    return first_last_ratio(path, "rmse", "error", "mape") > THRESHOLDS["temporal_ratio_warning"]


def _temporal_summary(path: Path) -> dict:
    return summary(path, "last_first_error_ratio", first_last_ratio(path, "rmse", "error", "mape"))


def _cloud_error_high(path: Path) -> bool:
    return max_value(path, "cloud_error_rate", "masked_error_rate") > THRESHOLDS["cloud_error_warning"]


def _cloud_summary(path: Path) -> dict:
    return summary(path, "max_cloud_error_rate", max_value(path, "cloud_error_rate", "masked_error_rate"))


MANIFEST = [
    {
        "id": "spatial_generalization_gap",
        "title": "Model performance varies across regions",
        "evidence": "geospatial_regions.csv",
        "evidence_check": _spatial_gap,
        "evidence_summary": _spatial_summary,
        "severity": "high",
        "queries": ["geospatial machine learning spatial cross validation domain shift", "remote sensing spatial generalization evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use spatial cross-validation and report worst-region performance.",
        "why_it_matters": "Random splits leak spatial autocorrelation and overstate generalization.",
        "next_checks": ["Run spatial block CV.", "Report metrics by region."],
        "success_criteria": ["Spatial gap falls below 0.10 or is documented."],
        "references": ["spatial cross-validation", "remote sensing", "spatial autocorrelation"],
    },
    {
        "id": "temporal_transfer_degradation",
        "title": "Geospatial error degrades over time",
        "evidence": "geospatial_time.csv",
        "evidence_check": _temporal_degradation,
        "evidence_summary": _temporal_summary,
        "severity": "medium",
        "queries": ["remote sensing temporal domain shift machine learning", "geospatial model temporal transfer evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Evaluate temporal holdouts and retraining cadence under season/year shifts.",
        "why_it_matters": "Land cover, sensors and climate conditions change over time.",
        "next_checks": ["Plot error by month/year.", "Compare temporal holdout."],
        "success_criteria": ["Temporal error ratio falls below 1.5."],
        "references": ["temporal domain shift", "remote sensing", "model drift"],
    },
    {
        "id": "cloud_mask_failures",
        "title": "Cloud/mask-related errors are high",
        "evidence": "geospatial_masks.csv",
        "evidence_check": _cloud_error_high,
        "evidence_summary": _cloud_summary,
        "severity": "medium",
        "queries": ["remote sensing cloud masking machine learning errors", "satellite imagery cloud robust learning"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Audit cloud masks and compare cloud-robust preprocessing or temporal compositing.",
        "why_it_matters": "Mask failures can dominate satellite model errors.",
        "next_checks": ["Inspect cloudy false positives.", "Compare mask algorithms."],
        "success_criteria": ["Cloud-related error falls below 0.10."],
        "references": ["cloud masking", "remote sensing preprocessing", "temporal compositing"],
    },
    {
        "id": "coordinate_leakage_missing",
        "title": "Coordinate leakage audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["coordinate leakage geospatial machine learning spatial autocorrelation", "location leakage spatial prediction model"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Check whether coordinates or nearby proxies leak the target under random splits.",
        "why_it_matters": "Location can memorize labels instead of learning transferable signal.",
        "next_checks": ["Compare with/without coordinates.", "Run spatial-block split."],
        "success_criteria": ["Coordinate usage is justified under spatial validation."],
        "references": ["spatial leakage", "coordinate features", "spatial validation"],
    },
    {
        "id": "resolution_sensitivity_missing",
        "title": "Spatial-resolution sensitivity is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["remote sensing resolution sensitivity machine learning", "multi resolution geospatial deep learning evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Evaluate performance under resampling, sensor resolution and tiling choices.",
        "why_it_matters": "Resolution and tiling choices can change geospatial conclusions.",
        "next_checks": ["Sweep tile size.", "Compare sensor resolutions."],
        "success_criteria": ["Conclusions are stable across plausible resolutions."],
        "references": ["multi-resolution", "tiling", "remote sensing"],
    },
]

