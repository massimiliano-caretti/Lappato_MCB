"""Astronomy / astrophysics manifest.

For pipelines on photometric / spectroscopic surveys, transient detection,
gravitational-wave inference, and N-body / hydrodynamic cosmological
simulations. Expected CSVs in ``checkpoints/``:

  - ``astro_psf_residuals.csv``: ``object,psf_residual``
  - ``astro_completeness.csv``: ``magnitude,completeness``
  - ``astro_selection_function.csv``: ``zone,selection_probability``
  - ``astro_signal_significance.csv``: ``candidate,sigma``
  - ``astro_simulation_resolution.csv``: ``box,resolution_min,structure_size``
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, min_value, rows, summary, values

RUN_TAG = "astronomy"

DEFAULT_THRESHOLDS = {
    "psf_residual_warning": 0.05,
    "completeness_warning": 0.80,
    "selection_inhomogeneity_warning": 0.20,
    "low_signal_sigma_warning": 5.0,
    "resolution_to_structure_ratio_warning": 0.5,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _psf_residual_high(path: Path) -> bool:
    return max_value(path, "psf_residual", "residual", "chi2_psf") > THRESHOLDS["psf_residual_warning"]


def _psf_summary(path: Path) -> dict:
    return summary(path, "max_psf_residual", max_value(path, "psf_residual", "residual", "chi2_psf"))


def _completeness_low(path: Path) -> bool:
    vals = values(path, "completeness", "detection_efficiency")
    return bool(vals) and min(vals) < THRESHOLDS["completeness_warning"]


def _completeness_summary(path: Path) -> dict:
    return summary(path, "min_completeness", min_value(path, "completeness", "detection_efficiency"))


def _selection_inhomogeneity(path: Path) -> float:
    vals = values(path, "selection_probability", "selection")
    return (max(vals) - min(vals)) if vals else 0.0


def _selection_function_inhomogeneous(path: Path) -> bool:
    return _selection_inhomogeneity(path) > THRESHOLDS["selection_inhomogeneity_warning"]


def _selection_summary(path: Path) -> dict:
    return summary(path, "selection_spread", _selection_inhomogeneity(path))


def _low_signal_significance(path: Path) -> bool:
    vals = values(path, "sigma", "snr", "significance")
    if not vals:
        return False
    return min(vals) < THRESHOLDS["low_signal_sigma_warning"]


def _signal_summary(path: Path) -> dict:
    return summary(path, "min_sigma", min_value(path, "sigma", "snr", "significance"))


def _resolution_inadequate_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    seen = 0
    for row in items:
        rmin = f(row, "resolution_min", "softening", "cell_size")
        struct = f(row, "structure_size", "halo_radius", "feature_size")
        if rmin is None or struct is None or struct <= 0:
            continue
        seen += 1
        if rmin / struct > THRESHOLDS["resolution_to_structure_ratio_warning"]:
            bad += 1
    return (bad / seen) if seen else 0.0


def _simulation_resolution_inadequate(path: Path) -> bool:
    return _resolution_inadequate_share(path) > 0.0


def _resolution_summary(path: Path) -> dict:
    return summary(path, "underresolved_share", _resolution_inadequate_share(path))


MANIFEST: list[dict] = [
    {
        "id": "psf_residual_high",
        "title": "PSF model leaves large per-object residuals",
        "evidence": "astro_psf_residuals.csv",
        "evidence_check": _psf_residual_high,
        "evidence_summary": _psf_summary,
        "severity": "high",
        "queries": [
            "PSF modelling photometry residuals quality",
            "weak lensing PSF systematics correction",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Refit the PSF with spatial / chromatic dependencies and rerun residual maps before science measurements.",
        "why_it_matters": "PSF residuals propagate into photometric, astrometric and shear measurements as systematic errors.",
        "next_checks": [
            "Bin residuals by focal-plane position.",
            "Test alternative PSF models (PixCorr, PSFEx, principal-component).",
        ],
        "success_criteria": [
            "Maximum PSF residual stays below 0.05 of source flux.",
        ],
        "references": [
            "PSF modelling",
            "weak lensing systematics",
            "PSFEx",
        ],
    },
    {
        "id": "completeness_low",
        "title": "Survey completeness drops below the science threshold",
        "evidence": "astro_completeness.csv",
        "evidence_check": _completeness_low,
        "evidence_summary": _completeness_summary,
        "severity": "high",
        "queries": [
            "survey completeness Malmquist bias correction",
            "injection recovery completeness photometric survey",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply injection-recovery completeness corrections and quote magnitude-binned completeness alongside the catalog.",
        "why_it_matters": "Low completeness biases population statistics (luminosity functions, rates) toward the bright end.",
        "next_checks": [
            "Run injection-recovery on representative fields.",
            "Bin science results by magnitude / surface brightness.",
        ],
        "success_criteria": [
            "Completeness exceeds 0.80 at the chosen magnitude limit.",
        ],
        "references": [
            "Malmquist bias",
            "injection recovery",
            "survey selection function",
        ],
    },
    {
        "id": "selection_function_inhomogeneous",
        "title": "Selection function varies sharply across the survey footprint",
        "evidence": "astro_selection_function.csv",
        "evidence_check": _selection_function_inhomogeneous,
        "evidence_summary": _selection_summary,
        "severity": "high",
        "queries": [
            "survey selection function homogeneity",
            "spectroscopic targeting selection bias correction",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Mask or weight inhomogeneous regions and propagate the selection function into clustering / abundance estimators.",
        "why_it_matters": "Spatial inhomogeneities in selection contaminate clustering and large-scale-structure measurements.",
        "next_checks": [
            "Map selection probability across the footprint.",
            "Recompute correlation functions with weighting.",
        ],
        "success_criteria": [
            "Selection-probability spread across the footprint stays below 0.20.",
        ],
        "references": [
            "selection function",
            "angular mask",
            "clustering systematics",
        ],
    },
    {
        "id": "low_signal_significance",
        "title": "Reported detections lie below the standard discovery threshold",
        "evidence": "astro_signal_significance.csv",
        "evidence_check": _low_signal_significance,
        "evidence_summary": _signal_summary,
        "severity": "high",
        "queries": [
            "5 sigma discovery look elsewhere effect significance",
            "transient detection threshold false alarm rate",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compute look-elsewhere-corrected significance and report false-alarm rate alongside detection σ.",
        "why_it_matters": "Sub-5σ detections are easily produced by trial-factor inflation in survey pipelines.",
        "next_checks": [
            "Apply look-elsewhere correction.",
            "Estimate background false-alarm rate via injection.",
        ],
        "success_criteria": [
            "Reported significance exceeds 5σ after trial correction.",
        ],
        "references": [
            "look-elsewhere effect",
            "5 sigma discovery threshold",
            "false alarm rate",
        ],
    },
    {
        "id": "simulation_resolution_inadequate",
        "title": "Simulation resolution does not resolve the structures of interest",
        "evidence": "astro_simulation_resolution.csv",
        "evidence_check": _simulation_resolution_inadequate,
        "evidence_summary": _resolution_summary,
        "severity": "high",
        "queries": [
            "convergence test cosmological simulation resolution",
            "softening length subgrid physics N-body",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run a resolution / softening-length convergence study and restrict science claims to converged scales.",
        "why_it_matters": "Underresolved structures in N-body / hydro simulations bias halo properties, abundances, and feedback impact.",
        "next_checks": [
            "Compare two resolution levels on the same volume.",
            "Quote a minimum resolved mass / scale.",
        ],
        "success_criteria": [
            "All science claims sit above the converged resolution scale.",
        ],
        "references": [
            "cosmological simulation convergence",
            "softening length",
            "subgrid physics",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
