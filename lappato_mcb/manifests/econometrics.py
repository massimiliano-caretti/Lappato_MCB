"""Econometrics manifest — IV, panel data, DiD, RDD pipelines.

For pipelines on instrumental-variable estimation, panel-data fixed
effects, difference-in-differences, regression discontinuity, and
synthetic-control studies. Expected CSVs in ``checkpoints/``:

  - ``econ_iv_strength.csv``: ``instrument,first_stage_F``
  - ``econ_parallel_trends.csv``: ``period,treated,control,gap``
  - ``econ_robust_se.csv``: ``cluster,n_obs,naive_se,clustered_se``
  - ``econ_rdd_bandwidth.csv``: ``bandwidth,estimate,std_error``
  - ``econ_panel_attrition.csv``: ``period,n_units,attrition_share``
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, min_value, rows, summary, values

RUN_TAG = "econometrics"

DEFAULT_THRESHOLDS = {
    "weak_instrument_F_warning": 10.0,
    "parallel_trends_gap_warning": 0.05,
    "se_inflation_ratio_warning": 1.5,
    "rdd_bandwidth_estimate_swing_warning": 0.50,
    "panel_attrition_share_warning": 0.20,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _weak_instrument(path: Path) -> bool:
    vals = values(path, "first_stage_F", "F", "f_stat")
    return bool(vals) and min(vals) < THRESHOLDS["weak_instrument_F_warning"]


def _instrument_summary(path: Path) -> dict:
    return summary(path, "min_first_stage_F", min_value(path, "first_stage_F", "F", "f_stat"))


def _parallel_trends_violation(path: Path) -> bool:
    pre_gaps: list[float] = []
    for row in rows(path):
        period = (row.get("period") or "").strip().lower()
        if period not in ("pre", "pre_treatment", "before"):
            gap = f(row, "gap")
            if gap is None:
                continue
            t = f(row, "treated")
            c = f(row, "control")
            if t is not None and c is not None:
                pre_gaps.append(abs(t - c))
            else:
                pre_gaps.append(abs(gap))
    return bool(pre_gaps) and max(pre_gaps) > THRESHOLDS["parallel_trends_gap_warning"]


def _parallel_trends_summary(path: Path) -> dict:
    return summary(path, "max_pre_period_gap", max_value(path, "gap", "pre_gap"))


def _se_inflation_ratio(path: Path) -> float:
    ratios: list[float] = []
    for row in rows(path):
        n = f(row, "naive_se", "ols_se")
        c = f(row, "clustered_se", "robust_se")
        if n is None or c is None or n == 0:
            continue
        ratios.append(c / n)
    return max(ratios) if ratios else 0.0


def _standard_errors_understated(path: Path) -> bool:
    return _se_inflation_ratio(path) > THRESHOLDS["se_inflation_ratio_warning"]


def _se_summary(path: Path) -> dict:
    return summary(path, "max_clustered_to_naive_se_ratio", _se_inflation_ratio(path))


def _rdd_estimate_swing(path: Path) -> float:
    items = rows(path)
    if len(items) < 2:
        return 0.0
    estimates: list[float] = []
    for row in items:
        v = f(row, "estimate", "tau")
        if v is not None:
            estimates.append(v)
    if len(estimates) < 2:
        return 0.0
    base = abs(estimates[0]) if estimates[0] != 0 else 1.0
    return abs(max(estimates) - min(estimates)) / base


def _rdd_bandwidth_sensitive(path: Path) -> bool:
    return _rdd_estimate_swing(path) > THRESHOLDS["rdd_bandwidth_estimate_swing_warning"]


def _rdd_summary(path: Path) -> dict:
    return summary(path, "rdd_estimate_swing", _rdd_estimate_swing(path))


def _panel_attrition_high(path: Path) -> bool:
    return max_value(path, "attrition_share", "attrition", "drop_share") > THRESHOLDS["panel_attrition_share_warning"]


def _panel_attrition_summary(path: Path) -> dict:
    return summary(path, "max_attrition_share", max_value(path, "attrition_share", "attrition", "drop_share"))


MANIFEST: list[dict] = [
    {
        "id": "weak_instrument",
        "title": "Instrumental variable shows weak first stage",
        "evidence": "econ_iv_strength.csv",
        "evidence_check": _weak_instrument,
        "evidence_summary": _instrument_summary,
        "severity": "high",
        "queries": [
            "weak instrument first stage F statistic",
            "Anderson Rubin weak identification IV",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to weak-IV-robust inference (Anderson-Rubin / CLR) and report identification-robust CIs.",
        "why_it_matters": "Weak instruments produce biased and severely mis-sized 2SLS estimates.",
        "next_checks": [
            "Compute Kleibergen-Paap rk-F.",
            "Report Anderson-Rubin confidence sets.",
        ],
        "success_criteria": [
            "Effective first-stage F exceeds 10 (Stock-Yogo) or weak-IV-robust CIs are reported.",
        ],
        "references": [
            "Stock-Yogo critical values",
            "Anderson-Rubin",
            "Kleibergen-Paap",
        ],
    },
    {
        "id": "parallel_trends_violation",
        "title": "Pre-treatment trends diverge between treated and control",
        "evidence": "econ_parallel_trends.csv",
        "evidence_check": _parallel_trends_violation,
        "evidence_summary": _parallel_trends_summary,
        "severity": "high",
        "queries": [
            "parallel trends test difference in differences event study",
            "Goodman Bacon decomposition staggered treatment",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run an event-study test of pre-trends and apply Callaway-Sant'Anna or honest-DiD bounds where pre-trends diverge.",
        "why_it_matters": "DiD identification rests on parallel pre-trends; pre-period divergence biases the post-period estimate.",
        "next_checks": [
            "Estimate event-study coefficients in the pre-period.",
            "Report honest-DiD sensitivity bounds.",
        ],
        "success_criteria": [
            "Pre-period treated/control gap stays below 0.05 of the outcome scale.",
        ],
        "references": [
            "honest DiD",
            "Callaway-Sant'Anna",
            "Goodman-Bacon",
        ],
    },
    {
        "id": "standard_errors_understated",
        "title": "Naive standard errors much smaller than clustered / HAC",
        "evidence": "econ_robust_se.csv",
        "evidence_check": _standard_errors_understated,
        "evidence_summary": _se_summary,
        "severity": "high",
        "queries": [
            "cluster robust standard errors econometrics",
            "Moulton problem grouped errors inference",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report cluster-robust / HAC standard errors and document the level of clustering.",
        "why_it_matters": "Naive standard errors understate uncertainty when units are correlated within clusters.",
        "next_checks": [
            "Cluster at the policy / sampling level.",
            "Run wild-cluster bootstrap on small G.",
        ],
        "success_criteria": [
            "Inference uses cluster-robust or HAC errors at the appropriate level.",
        ],
        "references": [
            "cluster robust standard errors",
            "wild cluster bootstrap",
            "Moulton effect",
        ],
    },
    {
        "id": "rdd_bandwidth_sensitive",
        "title": "RDD estimate swings strongly with bandwidth choice",
        "evidence": "econ_rdd_bandwidth.csv",
        "evidence_check": _rdd_bandwidth_sensitive,
        "evidence_summary": _rdd_summary,
        "severity": "high",
        "queries": [
            "regression discontinuity bandwidth optimal Imbens Kalyanaraman",
            "RDD robust bias corrected confidence interval",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use IK / CCT optimal bandwidth and report bias-corrected CIs across a sensitivity grid.",
        "why_it_matters": "Bandwidth-sensitive RDD estimates indicate model misspecification near the cutoff.",
        "next_checks": [
            "Plot point estimates across a bandwidth grid.",
            "Apply CCT bias-correction.",
        ],
        "success_criteria": [
            "Estimate swing across plausible bandwidths stays below 50% of the central estimate.",
        ],
        "references": [
            "Calonico Cattaneo Titiunik",
            "Imbens Kalyanaraman",
            "RDD bandwidth",
        ],
    },
    {
        "id": "panel_attrition_high",
        "title": "Panel attrition exceeds the safe threshold",
        "evidence": "econ_panel_attrition.csv",
        "evidence_check": _panel_attrition_high,
        "evidence_summary": _panel_attrition_summary,
        "severity": "high",
        "queries": [
            "panel attrition selection bias longitudinal",
            "inverse probability weighting attrition correction",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply inverse-probability-of-attrition weights and run a Heckman / pattern-mixture sensitivity check.",
        "why_it_matters": "Differential attrition produces biased panel estimates that look stable in the surviving sub-sample.",
        "next_checks": [
            "Tabulate attrition by treatment / covariates.",
            "Re-estimate with attrition-corrected weights.",
        ],
        "success_criteria": [
            "Maximum attrition share stays below 20% per wave or is corrected.",
        ],
        "references": [
            "panel attrition",
            "Heckman selection",
            "inverse probability weighting",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
