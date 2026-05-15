"""Materials-science manifest — DFT / MD / phase / defect / crystal pipelines.

For pipelines on density-functional theory, ab-initio MD, machine-learning
interatomic potentials (MLIPs), phase diagrams, and crystal-structure
prediction. Expected CSVs in ``checkpoints/``:

  - ``materials_kpoint_convergence.csv``: ``k_density,total_energy``
  - ``materials_basis_convergence.csv``: ``ecut,total_energy``
  - ``materials_force_residuals.csv``: ``structure,force_rmse``
  - ``materials_phase_stability.csv``: ``phase,formation_energy_per_atom``
  - ``materials_extrapolation.csv``: ``structure,nearest_train_distance``
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, rows, summary

RUN_TAG = "materials_science"

DEFAULT_THRESHOLDS = {
    "kpoint_energy_diff_warning": 1.0e-3,    # eV/atom between two k densities
    "basis_energy_diff_warning": 1.0e-3,     # eV/atom between cutoffs
    "force_rmse_warning": 0.05,              # eV/Å
    "phase_stability_residual_warning": 0.05,  # eV/atom above hull
    "extrapolation_distance_warning": 1.5,   # σ in feature space
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _convergence_diff(path: Path, energy_col: str = "total_energy") -> float:
    items = rows(path)
    if len(items) < 2:
        return 0.0
    energies: list[float] = []
    for row in items:
        v = f(row, energy_col, "energy")
        if v is not None:
            energies.append(v)
    if len(energies) < 2:
        return 0.0
    return abs(energies[-1] - energies[-2])


def _kpoint_unconverged(path: Path) -> bool:
    return _convergence_diff(path) > THRESHOLDS["kpoint_energy_diff_warning"]


def _kpoint_summary(path: Path) -> dict:
    return summary(path, "last_two_energy_diff", _convergence_diff(path))


def _basis_unconverged(path: Path) -> bool:
    return _convergence_diff(path) > THRESHOLDS["basis_energy_diff_warning"]


def _basis_summary(path: Path) -> dict:
    return summary(path, "last_two_energy_diff", _convergence_diff(path))


def _force_residuals_high(path: Path) -> bool:
    return max_value(path, "force_rmse", "force_residual", "rmse_force") > THRESHOLDS["force_rmse_warning"]


def _force_summary(path: Path) -> dict:
    return summary(path, "max_force_rmse", max_value(path, "force_rmse", "force_residual", "rmse_force"))


def _phase_above_hull_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    seen = 0
    for row in items:
        v = f(row, "formation_energy_per_atom", "energy_above_hull", "ehull")
        if v is None:
            continue
        seen += 1
        if v > THRESHOLDS["phase_stability_residual_warning"]:
            bad += 1
    return (bad / seen) if seen else 0.0


def _phase_stability_violated(path: Path) -> bool:
    return _phase_above_hull_share(path) > 0.0


def _phase_summary(path: Path) -> dict:
    return summary(path, "above_hull_share", _phase_above_hull_share(path))


def _extrapolation_warning(path: Path) -> bool:
    return max_value(path, "nearest_train_distance", "domain_distance") > THRESHOLDS["extrapolation_distance_warning"]


def _extrapolation_summary(path: Path) -> dict:
    return summary(path, "max_train_distance", max_value(path, "nearest_train_distance", "domain_distance"))


MANIFEST: list[dict] = [
    {
        "id": "kpoint_unconverged",
        "title": "k-point sampling is not converged",
        "evidence": "materials_kpoint_convergence.csv",
        "evidence_check": _kpoint_unconverged,
        "evidence_summary": _kpoint_summary,
        "severity": "high",
        "queries": [
            "k-point convergence DFT total energy Brillouin zone",
            "Monkhorst Pack mesh convergence solid state",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Increase k-point density until total-energy differences fall below 1 meV/atom and report the converged grid.",
        "why_it_matters": "k-point underconvergence biases total energies, forces, and phase ordering.",
        "next_checks": [
            "Run a denser k-grid and compare energy differences.",
            "Repeat for metallic vs insulating systems separately.",
        ],
        "success_criteria": [
            "Energy difference between consecutive k-grids stays below 1 meV/atom.",
        ],
        "references": [
            "Monkhorst-Pack mesh",
            "DFT convergence",
            "Brillouin zone sampling",
        ],
    },
    {
        "id": "basis_unconverged",
        "title": "Plane-wave / basis set cutoff is not converged",
        "evidence": "materials_basis_convergence.csv",
        "evidence_check": _basis_unconverged,
        "evidence_summary": _basis_summary,
        "severity": "high",
        "queries": [
            "plane wave cutoff convergence DFT",
            "basis set superposition error solid state",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Increase the cutoff energy / basis size until total energy plateaus and document the converged choice.",
        "why_it_matters": "An undersized basis biases formation energies and elastic constants.",
        "next_checks": [
            "Plot energy vs cutoff for a representative cell.",
            "Verify forces and stresses also converge.",
        ],
        "success_criteria": [
            "Energy difference between the last two cutoffs stays below 1 meV/atom.",
        ],
        "references": [
            "plane wave cutoff",
            "basis set convergence",
            "DFT convergence study",
        ],
    },
    {
        "id": "force_residuals_high",
        "title": "Machine-learning interatomic potential leaves large force residuals",
        "evidence": "materials_force_residuals.csv",
        "evidence_check": _force_residuals_high,
        "evidence_summary": _force_summary,
        "severity": "high",
        "queries": [
            "machine learning interatomic potential force benchmark",
            "MLIP test set force RMSE generalization",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Augment training data with active learning, retrain, and re-benchmark forces by chemistry / coordination.",
        "why_it_matters": "Force errors above 50 meV/Å destabilise MD trajectories and bias phonon / elasticity predictions.",
        "next_checks": [
            "Stratify force RMSE by element / coordination.",
            "Compare against committee-disagreement uncertainty.",
        ],
        "success_criteria": [
            "Force RMSE stays below 0.05 eV/Å on the held-out test set.",
        ],
        "references": [
            "MLIP benchmarks",
            "active learning interatomic potential",
            "force regression",
        ],
    },
    {
        "id": "phase_stability_violated",
        "title": "Predicted phases sit above the convex hull",
        "evidence": "materials_phase_stability.csv",
        "evidence_check": _phase_stability_violated,
        "evidence_summary": _phase_summary,
        "severity": "high",
        "queries": [
            "convex hull phase stability DFT high throughput",
            "energy above hull stability prediction materials",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Recompute formation energies with consistent reference states and audit known competing phases.",
        "why_it_matters": "Above-hull phases are thermodynamically unstable; reporting them as discoveries inflates novelty claims.",
        "next_checks": [
            "Re-derive the convex hull with all known competing chemistries.",
            "Quantify entropic / temperature corrections.",
        ],
        "success_criteria": [
            "Reported novel phases sit on or below the convex hull within tolerance.",
        ],
        "references": [
            "convex hull",
            "energy above hull",
            "phase diagram DFT",
        ],
    },
    {
        "id": "extrapolation_warning",
        "title": "Predictions extrapolate far from the training distribution",
        "evidence": "materials_extrapolation.csv",
        "evidence_check": _extrapolation_warning,
        "evidence_summary": _extrapolation_summary,
        "severity": "info",
        "queries": [
            "domain applicability machine learning materials",
            "out of distribution composition extrapolation crystals",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Define an applicability-domain check on chemical / structural descriptors and flag extrapolations.",
        "why_it_matters": "Extrapolated predictions on unseen chemistries can fail without warning and inflate apparent screening hit rates.",
        "next_checks": [
            "Compute distance-to-training-manifold for every prediction.",
            "Rank predictions by uncertainty before experimental follow-up.",
        ],
        "success_criteria": [
            "Maximum domain distance stays below 1.5 σ on screened candidates.",
        ],
        "references": [
            "applicability domain",
            "out-of-distribution materials",
            "uncertainty quantification MLIP",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
