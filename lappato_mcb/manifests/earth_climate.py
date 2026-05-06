"""Earth-and-climate manifest — climate, atmospheric, ocean, hydrology.

For pipelines on numerical weather prediction, climate-model output,
ocean / atmosphere reanalysis, hydrological / cryospheric models, and
remote-sensing-driven Earth-system inference. Expected CSVs in
``checkpoints/``:

  - ``climate_bias.csv``: ``variable,model,observation,bias``
  - ``climate_water_balance.csv``: ``period,P,E,Q,dS`` (precipitation,
    evapotranspiration, runoff, storage change — should sum to ~0)
  - ``climate_extremes.csv``: ``variable,model_99p,obs_99p``
  - ``climate_ensemble.csv``: ``variable,ensemble_spread,ensemble_skill``
  - ``climate_downscaling_train_period.csv``: ``period,start_year,end_year``
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import f, max_value, rows, summary

RUN_TAG = "earth_climate"

DEFAULT_THRESHOLDS = {
    "bias_warning": 1.0,
    "water_balance_residual_warning": 0.05,
    "extremes_relative_gap_warning": 0.20,
    "spread_skill_ratio_warning": 0.30,
    "downscaling_period_overlap_share_warning": 0.0,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _bias_high(path: Path) -> bool:
    return max_value(path, "bias", "absolute_bias") > THRESHOLDS["bias_warning"]


def _bias_summary(path: Path) -> dict:
    return summary(path, "max_bias", max_value(path, "bias", "absolute_bias"))


def _water_balance_residual(path: Path) -> float:
    residuals: list[float] = []
    for row in rows(path):
        P = f(row, "P", "precipitation") or 0.0
        E = f(row, "E", "evapotranspiration") or 0.0
        Q = f(row, "Q", "runoff") or 0.0
        dS = f(row, "dS", "storage_change") or 0.0
        scale = abs(P) if P else 1.0
        residuals.append(abs(P - E - Q - dS) / scale)
    return max(residuals) if residuals else 0.0


def _water_balance_violation(path: Path) -> bool:
    return _water_balance_residual(path) > THRESHOLDS["water_balance_residual_warning"]


def _water_balance_summary(path: Path) -> dict:
    return summary(path, "max_water_balance_residual", _water_balance_residual(path))


def _extremes_relative_gap(path: Path) -> float:
    gaps: list[float] = []
    for row in rows(path):
        m = f(row, "model_99p", "model_extreme")
        o = f(row, "obs_99p", "obs_extreme")
        if m is None or o is None or o == 0:
            continue
        gaps.append(abs(m - o) / abs(o))
    return max(gaps) if gaps else 0.0


def _extremes_underestimated(path: Path) -> bool:
    return _extremes_relative_gap(path) > THRESHOLDS["extremes_relative_gap_warning"]


def _extremes_summary(path: Path) -> dict:
    return summary(path, "max_extremes_gap", _extremes_relative_gap(path))


def _ensemble_spread_skill_ratio(path: Path) -> float:
    ratios: list[float] = []
    for row in rows(path):
        sp = f(row, "ensemble_spread", "spread")
        sk = f(row, "ensemble_skill", "skill", "rmse")
        if sp is None or sk is None or sk == 0:
            continue
        ratios.append(abs(sp / sk))
    return max(abs(1 - r) for r in ratios) if ratios else 0.0


def _ensemble_underdispersive(path: Path) -> bool:
    return _ensemble_spread_skill_ratio(path) > THRESHOLDS["spread_skill_ratio_warning"]


def _ensemble_summary(path: Path) -> dict:
    return summary(path, "spread_skill_deviation", _ensemble_spread_skill_ratio(path))


def _downscaling_period_overlap(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    train: list[tuple[float, float]] = []
    test: list[tuple[float, float]] = []
    for row in items:
        period = (row.get("period") or "").strip().lower()
        a = f(row, "start_year")
        b = f(row, "end_year")
        if a is None or b is None:
            continue
        if "train" in period:
            train.append((a, b))
        elif "test" in period or "eval" in period or "valid" in period:
            test.append((a, b))
    if not train or not test:
        return 0.0
    overlapped = 0
    for a, b in test:
        if any(not (b < ta or a > tb) for ta, tb in train):
            overlapped += 1
    return overlapped / len(test)


def _downscaling_train_period_overlap(path: Path) -> bool:
    return _downscaling_period_overlap(path) > THRESHOLDS["downscaling_period_overlap_share_warning"]


def _downscaling_summary(path: Path) -> dict:
    return summary(path, "test_period_overlap_share", _downscaling_period_overlap(path))


MANIFEST: list[dict] = [
    {
        "id": "model_observation_bias_high",
        "title": "Climate-model variable shows persistent bias against observations",
        "evidence": "climate_bias.csv",
        "evidence_check": _bias_high,
        "evidence_summary": _bias_summary,
        "severity": "high",
        "queries": [
            "climate model bias correction quantile mapping",
            "regional climate model evaluation observation bias",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply quantile-mapping bias correction and report bias separately for the climate mean and seasonal cycle.",
        "why_it_matters": "Persistent biases propagate into impact studies and can flip the sign of projected change.",
        "next_checks": [
            "Decompose bias into mean / seasonal / extreme components.",
            "Verify bias persists across reference observation products.",
        ],
        "success_criteria": [
            "Mean bias on the headline variable falls below 1 unit (chosen scale).",
        ],
        "references": [
            "quantile mapping bias correction",
            "CMIP evaluation",
            "regional climate downscaling",
        ],
    },
    {
        "id": "water_balance_violation",
        "title": "Hydrological water balance does not close",
        "evidence": "climate_water_balance.csv",
        "evidence_check": _water_balance_violation,
        "evidence_summary": _water_balance_summary,
        "severity": "high",
        "queries": [
            "water balance closure remote sensing hydrology",
            "land surface model water budget evaluation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Reconcile P / E / Q / dS sources, audit storage estimates and document residual sources.",
        "why_it_matters": "An unclosed water budget signals systematic errors in one or more flux products.",
        "next_checks": [
            "Compute closure residual at multiple temporal scales.",
            "Cross-check against gravimetric storage (GRACE).",
        ],
        "success_criteria": [
            "Annual closure residual stays below 5% of precipitation.",
        ],
        "references": [
            "hydrological closure",
            "GRACE storage",
            "land surface model evaluation",
        ],
    },
    {
        "id": "extremes_underestimated",
        "title": "Model extremes diverge from observed extremes",
        "evidence": "climate_extremes.csv",
        "evidence_check": _extremes_underestimated,
        "evidence_summary": _extremes_summary,
        "severity": "high",
        "queries": [
            "extreme value analysis climate model evaluation",
            "block maxima generalized extreme value precipitation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Evaluate against block-maxima / GEV fits and report tail metrics separately from bulk metrics.",
        "why_it_matters": "Extremes drive impacts; underestimated tails make risk assessments dangerously optimistic.",
        "next_checks": [
            "Fit GEV / GPD to model and observation samples.",
            "Quantify return-level bias at 10 / 50 / 100-year horizons.",
        ],
        "success_criteria": [
            "Relative gap on the 99th percentile falls below 0.20.",
        ],
        "references": [
            "extreme value statistics",
            "GEV climate evaluation",
            "return level bias",
        ],
    },
    {
        "id": "ensemble_underdispersive",
        "title": "Ensemble spread does not match forecast skill",
        "evidence": "climate_ensemble.csv",
        "evidence_check": _ensemble_underdispersive,
        "evidence_summary": _ensemble_summary,
        "severity": "high",
        "queries": [
            "ensemble spread skill relationship forecast verification",
            "rank histogram reliability ensemble forecast",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run reliability / rank-histogram diagnostics and apply ensemble post-processing (e.g. EMOS, BMA).",
        "why_it_matters": "An under-dispersive ensemble is over-confident; an over-dispersive one wastes skill.",
        "next_checks": [
            "Plot rank histogram across forecast lead times.",
            "Apply EMOS or analog post-processing on the validation period.",
        ],
        "success_criteria": [
            "Spread/skill ratio stays within 30% of unity.",
        ],
        "references": [
            "spread-skill diagnostic",
            "rank histogram",
            "EMOS post-processing",
        ],
    },
    {
        "id": "downscaling_train_period_overlap",
        "title": "Downscaling test period overlaps the training period",
        "evidence": "climate_downscaling_train_period.csv",
        "evidence_check": _downscaling_train_period_overlap,
        "evidence_summary": _downscaling_summary,
        "severity": "high",
        "queries": [
            "statistical downscaling cross validation temporal split",
            "climate model evaluation hold out period",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to leave-one-decade-out or block cross-validation that respects climate non-stationarity.",
        "why_it_matters": "Overlapping calibration and evaluation periods inflate apparent skill and hide non-stationarity.",
        "next_checks": [
            "Re-run with strictly disjoint calibration and evaluation decades.",
            "Test on a future (warming) sub-period if available.",
        ],
        "success_criteria": [
            "Calibration and evaluation periods are disjoint.",
        ],
        "references": [
            "block cross validation",
            "downscaling evaluation",
            "non-stationarity climate",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
