"""Social-science / psychology manifest.

For pipelines on survey-data analysis, experimental psychology, behavioural
research, and replication studies. Expected CSVs in ``checkpoints/``:

  - ``social_effect_inflation.csv``: ``study,effect_size,n_subjects,p_value``
  - ``social_response_rate.csv``: ``survey,n_invited,n_completed``
  - ``social_construct_reliability.csv``: ``construct,cronbach_alpha``
  - ``social_p_curve.csv``: ``study,p_value``
  - ``social_demand_characteristics.csv``: ``arm,prompt_revealed`` (1/0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import f, min_value, rows, summary, values

RUN_TAG = "social_science"

DEFAULT_THRESHOLDS = {
    "small_n_warning": 30.0,
    "large_effect_with_small_n_warning": 0.80,
    "low_response_rate_warning": 0.50,
    "low_alpha_warning": 0.70,
    "p_just_below_0_05_share_warning": 0.30,
    "demand_characteristics_warning": 0.0,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _effect_inflation_count(path: Path) -> int:
    bad = 0
    for row in rows(path):
        n = f(row, "n_subjects", "n", "sample_size")
        d = f(row, "effect_size", "d", "cohen_d")
        if n is None or d is None:
            continue
        if n < THRESHOLDS["small_n_warning"] and abs(d) > THRESHOLDS["large_effect_with_small_n_warning"]:
            bad += 1
    return bad


def _effect_size_inflation(path: Path) -> bool:
    return _effect_inflation_count(path) > 0


def _effect_inflation_summary(path: Path) -> dict:
    return {"rows": len(rows(path)), "n_inflated_studies": _effect_inflation_count(path)}


def _response_rate_low(path: Path) -> bool:
    rates: list[float] = []
    for row in rows(path):
        inv = f(row, "n_invited", "invited")
        comp = f(row, "n_completed", "completed", "n_responded")
        if inv is None or comp is None or inv <= 0:
            continue
        rates.append(comp / inv)
    return bool(rates) and min(rates) < THRESHOLDS["low_response_rate_warning"]


def _response_rate_summary(path: Path) -> dict:
    rates: list[float] = []
    for row in rows(path):
        inv = f(row, "n_invited", "invited")
        comp = f(row, "n_completed", "completed", "n_responded")
        if inv is None or comp is None or inv <= 0:
            continue
        rates.append(comp / inv)
    return {"rows": len(rates), "min_response_rate": round(min(rates), 4) if rates else 0.0}


def _construct_reliability_low(path: Path) -> bool:
    vals = values(path, "cronbach_alpha", "alpha", "omega")
    return bool(vals) and min(vals) < THRESHOLDS["low_alpha_warning"]


def _reliability_summary(path: Path) -> dict:
    return summary(path, "min_alpha", min_value(path, "cronbach_alpha", "alpha", "omega"))


def _p_just_below_share(path: Path) -> float:
    vals = values(path, "p_value", "p")
    if not vals:
        return 0.0
    just_below = sum(1 for v in vals if 0.04 <= v < 0.05)
    return just_below / len(vals)


def _p_hacking_pattern(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    return _p_just_below_share(path) > THRESHOLDS["p_just_below_0_05_share_warning"]


def _p_hacking_summary(path: Path) -> dict:
    return summary(path, "p_just_below_0_05_share", _p_just_below_share(path))


def _demand_characteristics_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    for row in items:
        v = row.get("prompt_revealed") or row.get("hypothesis_revealed")
        if v is None:
            continue
        try:
            if int(float(v)) == 1:
                bad += 1
        except (TypeError, ValueError):
            continue
    return bad / len(items)


def _demand_characteristics_present(path: Path) -> bool:
    return _demand_characteristics_share(path) > THRESHOLDS["demand_characteristics_warning"]


def _demand_summary(path: Path) -> dict:
    return summary(path, "demand_characteristics_share", _demand_characteristics_share(path))


MANIFEST: list[dict] = [
    {
        "id": "effect_size_inflation",
        "title": "Reported effect sizes are large in small samples",
        "evidence": "social_effect_inflation.csv",
        "evidence_check": _effect_size_inflation,
        "evidence_summary": _effect_inflation_summary,
        "severity": "high",
        "queries": [
            "effect size inflation small sample replication psychology",
            "winner curse publication bias psychology",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run a small-telescopes / replication-power analysis and report adjusted effect-size CIs.",
        "why_it_matters": "Large d in small N is a classic signature of effect-size inflation that fails to replicate.",
        "next_checks": [
            "Pre-register a replication with the original power.",
            "Apply small-telescopes and PET-PEESE corrections.",
        ],
        "success_criteria": [
            "Reported effects in small samples come with shrunk / Bayesian estimates.",
        ],
        "references": [
            "small telescopes",
            "PET-PEESE",
            "winner's curse",
        ],
    },
    {
        "id": "response_rate_low",
        "title": "Survey response rate is low",
        "evidence": "social_response_rate.csv",
        "evidence_check": _response_rate_low,
        "evidence_summary": _response_rate_summary,
        "severity": "high",
        "queries": [
            "survey nonresponse bias weighting",
            "low response rate sampling representativeness",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply post-stratification / raking weights and run a non-response sensitivity analysis.",
        "why_it_matters": "Low response rates introduce non-ignorable selection bias that point estimates rarely reveal.",
        "next_checks": [
            "Compare respondent and target-population covariates.",
            "Apply auxiliary-variable raking.",
        ],
        "success_criteria": [
            "Response rate exceeds 50% or non-response is corrected and bounded.",
        ],
        "references": [
            "non-response bias",
            "raking weights",
            "post-stratification",
        ],
    },
    {
        "id": "construct_reliability_low",
        "title": "Construct internal-consistency is below the conventional floor",
        "evidence": "social_construct_reliability.csv",
        "evidence_check": _construct_reliability_low,
        "evidence_summary": _reliability_summary,
        "severity": "high",
        "queries": [
            "Cronbach alpha reliability scale validation",
            "McDonald omega reliability psychometrics",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Drop misfitting items, switch to omega-total and run confirmatory factor analysis on the construct.",
        "why_it_matters": "Low reliability attenuates effect estimates and inflates measurement-error bias.",
        "next_checks": [
            "Compute omega-total alongside alpha.",
            "Run a CFA before reporting structural-model coefficients.",
        ],
        "success_criteria": [
            "Internal consistency reaches at least 0.70 on the headline construct.",
        ],
        "references": [
            "Cronbach alpha",
            "McDonald omega",
            "CFA reliability",
        ],
    },
    {
        "id": "p_hacking_pattern",
        "title": "Distribution of p-values clusters just below 0.05",
        "evidence": "social_p_curve.csv",
        "evidence_check": _p_hacking_pattern,
        "evidence_summary": _p_hacking_summary,
        "severity": "high",
        "queries": [
            "p-curve analysis evidential value Simonsohn",
            "p-hacking detection meta science",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run a p-curve / z-curve analysis and report the share of p-values just below 0.05.",
        "why_it_matters": "Right-skewed p-curves below 0.05 are the canonical signature of selective reporting.",
        "next_checks": [
            "Compute p-curve right-skew and 33% test.",
            "Pre-register the analysis plan for replications.",
        ],
        "success_criteria": [
            "p-just-below share stays below 30% of significant studies.",
        ],
        "references": [
            "p-curve",
            "z-curve",
            "selective reporting",
        ],
    },
    {
        "id": "demand_characteristics_present",
        "title": "Manipulation reveals the hypothesis to participants",
        "evidence": "social_demand_characteristics.csv",
        "evidence_check": _demand_characteristics_present,
        "evidence_summary": _demand_summary,
        "severity": "high",
        "queries": [
            "demand characteristics experimental psychology blinding",
            "manipulation check awareness psychology experiment",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add cover stories, blinded conditions, and run awareness checks before treating effects as causal.",
        "why_it_matters": "Demand characteristics produce participant-driven effects that are not the construct of interest.",
        "next_checks": [
            "Add post-experimental awareness probes.",
            "Re-run blinded variants.",
        ],
        "success_criteria": [
            "No arm reveals the hypothesis to participants.",
        ],
        "references": [
            "demand characteristics",
            "experimenter blinding",
            "manipulation check",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
