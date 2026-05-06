"""Epidemiology / public-health manifest.

For pipelines on disease surveillance, infectious-disease modelling,
intervention evaluation, and population-level outbreak analytics.
Expected CSVs in ``checkpoints/``:

  - ``epi_case_definition.csv``: ``period,case_definition_id``
  - ``epi_reporting_delay.csv``: ``date,delay_days``
  - ``epi_serial_correlation.csv``: ``lag,autocorr``
  - ``epi_underreporting.csv``: ``period,reported,estimated_true``
  - ``epi_intervention_overlap.csv``: ``intervention,start,end``
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import f, max_value, rows, summary, values

RUN_TAG = "epidemiology"

DEFAULT_THRESHOLDS = {
    "case_definition_change_warning": 0.0,
    "reporting_delay_days_warning": 7.0,
    "serial_acf_warning": 0.30,
    "underreporting_ratio_warning": 1.5,
    "intervention_overlap_warning": 0.0,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _case_definition_changes(path: Path) -> int:
    items = rows(path)
    if len(items) < 2:
        return 0
    seen = []
    for row in items:
        d = (row.get("case_definition_id") or row.get("definition") or "").strip()
        if d and (not seen or seen[-1] != d):
            seen.append(d)
    return max(len(seen) - 1, 0)


def _case_definition_changed(path: Path) -> bool:
    return _case_definition_changes(path) > THRESHOLDS["case_definition_change_warning"]


def _case_definition_summary(path: Path) -> dict:
    return summary(path, "n_definition_changes", _case_definition_changes(path))


def _reporting_delay_high(path: Path) -> bool:
    return max_value(path, "delay_days", "reporting_delay") > THRESHOLDS["reporting_delay_days_warning"]


def _reporting_delay_summary(path: Path) -> dict:
    return summary(path, "max_reporting_delay", max_value(path, "delay_days", "reporting_delay"))


def _serial_correlation_high(path: Path) -> bool:
    vals = values(path, "autocorr", "acf", "rho")
    return bool(vals) and max(abs(v) for v in vals) > THRESHOLDS["serial_acf_warning"]


def _serial_correlation_summary(path: Path) -> dict:
    vals = values(path, "autocorr", "acf", "rho")
    return {"rows": len(vals), "max_abs_acf": round(max(abs(v) for v in vals), 4) if vals else 0.0}


def _underreporting_ratio(path: Path) -> float:
    ratios: list[float] = []
    for row in rows(path):
        rep = f(row, "reported")
        true = f(row, "estimated_true", "true")
        if rep is None or true is None or rep <= 0:
            continue
        ratios.append(true / rep)
    return max(ratios) if ratios else 0.0


def _underreporting_high(path: Path) -> bool:
    return _underreporting_ratio(path) > THRESHOLDS["underreporting_ratio_warning"]


def _underreporting_summary(path: Path) -> dict:
    return summary(path, "max_true_to_reported_ratio", _underreporting_ratio(path))


def _intervention_overlap_count(path: Path) -> int:
    items = rows(path)
    if len(items) < 2:
        return 0
    intervals: list[tuple[float, float]] = []
    for row in items:
        a = f(row, "start", "start_day")
        b = f(row, "end", "end_day")
        if a is None or b is None:
            continue
        intervals.append((a, b))
    overlaps = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            a1, b1 = intervals[i]
            a2, b2 = intervals[j]
            if not (b1 < a2 or a1 > b2):
                overlaps += 1
    return overlaps


def _intervention_overlap_present(path: Path) -> bool:
    return _intervention_overlap_count(path) > THRESHOLDS["intervention_overlap_warning"]


def _intervention_overlap_summary(path: Path) -> dict:
    return summary(path, "n_overlapping_pairs", _intervention_overlap_count(path))


MANIFEST: list[dict] = [
    {
        "id": "case_definition_changed",
        "title": "Case definition changes within the surveillance window",
        "evidence": "epi_case_definition.csv",
        "evidence_check": _case_definition_changed,
        "evidence_summary": _case_definition_summary,
        "severity": "high",
        "queries": [
            "case definition change surveillance time series",
            "outbreak ascertainment shift bias epidemiology",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Segment the time series at definition changes and refit, or harmonise definitions before modelling.",
        "why_it_matters": "Definition changes look like outbreaks and bias intervention-effect estimates.",
        "next_checks": [
            "Plot incidence with definition-change markers.",
            "Refit on the post-change sub-window only.",
        ],
        "success_criteria": [
            "Either no definition change occurs within the analysis window, or it is explicitly modelled.",
        ],
        "references": [
            "case definition",
            "ascertainment bias",
            "surveillance data quality",
        ],
    },
    {
        "id": "reporting_delay_high",
        "title": "Reporting delay exceeds the now-casting tolerance",
        "evidence": "epi_reporting_delay.csv",
        "evidence_check": _reporting_delay_high,
        "evidence_summary": _reporting_delay_summary,
        "severity": "high",
        "queries": [
            "reporting delay nowcasting epidemiology",
            "infectious disease backfill correction surveillance",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply nowcasting / backfill correction and model right-truncation explicitly.",
        "why_it_matters": "Right-truncated counts make recent epidemic curves look artificially flat and bias R(t) estimation.",
        "next_checks": [
            "Estimate the delay distribution from past months.",
            "Refit the model with delay-adjusted counts.",
        ],
        "success_criteria": [
            "Maximum reporting delay stays below 7 days or is explicitly modelled.",
        ],
        "references": [
            "nowcasting",
            "right-truncation",
            "reporting delay",
        ],
    },
    {
        "id": "serial_correlation_high",
        "title": "Residual autocorrelation in surveillance time series",
        "evidence": "epi_serial_correlation.csv",
        "evidence_check": _serial_correlation_high,
        "evidence_summary": _serial_correlation_summary,
        "severity": "high",
        "queries": [
            "autocorrelation infectious disease time series GLM",
            "Newey West standard errors epidemiology serial correlation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Fit an autoregressive structure or use HAC standard errors before reporting effect sizes.",
        "why_it_matters": "Ignoring serial correlation underestimates standard errors and produces spurious significance.",
        "next_checks": [
            "Fit AR(1) residuals and inspect ACF/PACF.",
            "Re-quote effect-size CIs with HAC errors.",
        ],
        "success_criteria": [
            "Residual ACF stays below 0.30 at all reported lags.",
        ],
        "references": [
            "Newey-West",
            "AR(1) residuals",
            "time-series epidemiology",
        ],
    },
    {
        "id": "underreporting_high",
        "title": "Underreporting ratio is large and unmodelled",
        "evidence": "epi_underreporting.csv",
        "evidence_check": _underreporting_high,
        "evidence_summary": _underreporting_summary,
        "severity": "high",
        "queries": [
            "underreporting infectious disease capture recapture",
            "ascertainment fraction outbreak epidemiology",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Estimate the ascertainment fraction with capture-recapture / serology and inflate counts before fitting.",
        "why_it_matters": "Underreported counts bias R(t) and severity estimates downward.",
        "next_checks": [
            "Run capture-recapture against an independent data source.",
            "Test the model under a plausible ascertainment range.",
        ],
        "success_criteria": [
            "Underreporting ratio is reported and below 1.5 or explicitly inflated.",
        ],
        "references": [
            "capture recapture",
            "ascertainment fraction",
            "outbreak underreporting",
        ],
    },
    {
        "id": "intervention_overlap_present",
        "title": "Multiple interventions overlap in time",
        "evidence": "epi_intervention_overlap.csv",
        "evidence_check": _intervention_overlap_present,
        "evidence_summary": _intervention_overlap_summary,
        "severity": "high",
        "queries": [
            "intervention overlap identifiability public health policy",
            "co-intervention bias quasi experiment epidemiology",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use synthetic-control / DiD with explicit co-intervention indicators and bound the identifiable effect.",
        "why_it_matters": "Overlapping interventions are not separately identifiable; reported single-policy effects mix them.",
        "next_checks": [
            "Tabulate overlap matrix across interventions.",
            "Audit synthetic-control donor pool exclusivity.",
        ],
        "success_criteria": [
            "No two interventions overlap, or the overlap is explicitly modelled.",
        ],
        "references": [
            "co-intervention bias",
            "synthetic control",
            "policy identifiability",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
