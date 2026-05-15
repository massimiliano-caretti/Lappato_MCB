"""Pharmacology manifest — PK/PD, dose-response, clinical-trial endpoints.

For pipelines on pharmacokinetics, pharmacodynamics, dose-response
modelling, and translational drug-development simulation. Expected CSVs
in ``checkpoints/``:

  - ``pharma_dose_response.csv``: ``compound,dose,response``
  - ``pharma_pk_concordance.csv``: ``model,observed_auc,predicted_auc``
  - ``pharma_dde.csv``: ``compound,interaction_score``
  - ``pharma_extrapolation.csv``: ``compound,nearest_train_distance``
  - ``pharma_trial_endpoint.csv``: ``trial,primary_endpoint_specified``
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, rows, summary

RUN_TAG = "pharmacology"

DEFAULT_THRESHOLDS = {
    "dose_response_min_doses_warning": 4.0,
    "pk_auc_relative_error_warning": 0.30,
    "ddi_score_warning": 5.0,
    "extrapolation_distance_warning": 1.5,
    "trial_endpoint_unspecified_share_warning": 0.0,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _doses_per_compound(path: Path) -> int:
    items = rows(path)
    if not items:
        return 0
    counts: dict[str, int] = {}
    for row in items:
        cid = (row.get("compound") or row.get("id") or "").strip()
        if cid:
            counts[cid] = counts.get(cid, 0) + 1
    return min(counts.values()) if counts else 0


def _dose_response_undersampled(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    return _doses_per_compound(path) < THRESHOLDS["dose_response_min_doses_warning"]


def _dose_response_summary(path: Path) -> dict:
    return summary(path, "min_doses_per_compound", _doses_per_compound(path))


def _pk_relative_error(path: Path) -> float:
    errors: list[float] = []
    for row in rows(path):
        obs = f(row, "observed_auc", "obs", "observed")
        pred = f(row, "predicted_auc", "pred", "predicted")
        if obs is None or pred is None or obs == 0:
            continue
        errors.append(abs(pred - obs) / abs(obs))
    return max(errors) if errors else 0.0


def _pk_concordance_low(path: Path) -> bool:
    return _pk_relative_error(path) > THRESHOLDS["pk_auc_relative_error_warning"]


def _pk_summary(path: Path) -> dict:
    return summary(path, "max_relative_pk_error", _pk_relative_error(path))


def _ddi_high(path: Path) -> bool:
    return max_value(path, "interaction_score", "ddi", "ic_score") > THRESHOLDS["ddi_score_warning"]


def _ddi_summary(path: Path) -> dict:
    return summary(path, "max_interaction_score", max_value(path, "interaction_score", "ddi", "ic_score"))


def _extrapolation_high(path: Path) -> bool:
    return max_value(path, "nearest_train_distance", "domain_distance") > THRESHOLDS["extrapolation_distance_warning"]


def _extrapolation_summary(path: Path) -> dict:
    return summary(path, "max_train_distance", max_value(path, "nearest_train_distance", "domain_distance"))


def _trial_endpoint_unspecified_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    seen = 0
    for row in items:
        v = row.get("primary_endpoint_specified") or row.get("primary_specified")
        if v is None or v == "":
            continue
        seen += 1
        if str(v).strip().lower() in ("0", "false", "no", "n", "missing"):
            bad += 1
    return (bad / seen) if seen else 0.0


def _trial_endpoint_unspecified(path: Path) -> bool:
    return _trial_endpoint_unspecified_share(path) > THRESHOLDS["trial_endpoint_unspecified_share_warning"]


def _trial_summary(path: Path) -> dict:
    return summary(path, "endpoint_unspecified_share", _trial_endpoint_unspecified_share(path))


MANIFEST: list[dict] = [
    {
        "id": "dose_response_undersampled",
        "title": "Dose-response curve has too few dose levels per compound",
        "evidence": "pharma_dose_response.csv",
        "evidence_check": _dose_response_undersampled,
        "evidence_summary": _dose_response_summary,
        "severity": "high",
        "queries": [
            "dose response curve fitting Hill equation",
            "EC50 IC50 confidence interval pharmacology",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add ≥4 dose levels covering the inflection region and report Hill-fit CIs.",
        "why_it_matters": "Underspecified dose-response curves produce wide EC50 / IC50 estimates that look precise on point estimates.",
        "next_checks": [
            "Bootstrap Hill-fit confidence intervals.",
            "Add dose levels around the inflection.",
        ],
        "success_criteria": [
            "At least 4 dose levels per compound, with tight EC50 CIs.",
        ],
        "references": [
            "Hill equation",
            "EC50 confidence interval",
            "dose response design",
        ],
    },
    {
        "id": "pk_concordance_low",
        "title": "Predicted PK exposure (AUC) diverges from observation",
        "evidence": "pharma_pk_concordance.csv",
        "evidence_check": _pk_concordance_low,
        "evidence_summary": _pk_summary,
        "severity": "high",
        "queries": [
            "PBPK model evaluation predicted observed AUC",
            "population pharmacokinetics goodness of fit",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Re-fit the structural PK model on observed data and audit covariate effects (renal / hepatic).",
        "why_it_matters": "PK miscalibration biases efficacy / safety simulations and propagates into trial design.",
        "next_checks": [
            "Re-run goodness-of-fit plots stratified by covariate.",
            "Include renal / hepatic covariate effects.",
        ],
        "success_criteria": [
            "Maximum relative AUC error stays below 0.30.",
        ],
        "references": [
            "PBPK",
            "population PK",
            "AUC fold error",
        ],
    },
    {
        "id": "ddi_high",
        "title": "Drug-drug interaction score exceeds the safety threshold",
        "evidence": "pharma_dde.csv",
        "evidence_check": _ddi_high,
        "evidence_summary": _ddi_summary,
        "severity": "high",
        "queries": [
            "drug drug interaction prediction CYP inhibition",
            "transporter mediated drug interaction modelling",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run mechanistic CYP / transporter inhibition simulations and update labelling assumptions.",
        "why_it_matters": "DDIs cause clinically meaningful exposure changes that simple PK models miss.",
        "next_checks": [
            "Profile mechanistic inhibition vs prediction.",
            "Verify against known DDI clinical studies.",
        ],
        "success_criteria": [
            "DDI score is below the safety threshold or the interaction is documented.",
        ],
        "references": [
            "CYP inhibition",
            "P-gp transporter DDI",
            "PBPK DDI",
        ],
    },
    {
        "id": "extrapolation_high",
        "title": "Compound prediction extrapolates from training chemistry",
        "evidence": "pharma_extrapolation.csv",
        "evidence_check": _extrapolation_high,
        "evidence_summary": _extrapolation_summary,
        "severity": "info",
        "queries": [
            "applicability domain QSAR ADMET prediction",
            "out of distribution drug discovery deep learning",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Define an applicability domain on the training fingerprint and flag predictions outside it.",
        "why_it_matters": "Out-of-distribution compounds yield unreliable ADMET / efficacy predictions that look confident.",
        "next_checks": [
            "Compute Tanimoto / k-NN distance to training set.",
            "Rank candidates by uncertainty before triage.",
        ],
        "success_criteria": [
            "Maximum domain distance stays below 1.5 σ on triaged candidates.",
        ],
        "references": [
            "applicability domain ADMET",
            "Tanimoto distance",
            "uncertainty drug discovery",
        ],
    },
    {
        "id": "trial_endpoint_unspecified",
        "title": "Trial primary endpoint is not pre-specified",
        "evidence": "pharma_trial_endpoint.csv",
        "evidence_check": _trial_endpoint_unspecified,
        "evidence_summary": _trial_summary,
        "severity": "high",
        "queries": [
            "primary endpoint pre-specification clinical trial",
            "estimand framework ICH E9 R1",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Pre-specify the primary endpoint and estimand and lock the analysis plan before unblinding.",
        "why_it_matters": "Endpoint switching is a primary source of false-positive trial findings.",
        "next_checks": [
            "Lock the SAP before any interim look.",
            "Document the estimand under ICH E9(R1).",
        ],
        "success_criteria": [
            "Every trial has a documented pre-specified primary endpoint.",
        ],
        "references": [
            "ICH E9(R1)",
            "estimand framework",
            "primary endpoint switching",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
