"""Physics manifest — computational simulations (DFT, MD, MC, FEM, CFD).

For pipelines that solve physical equations of motion, sample equilibrium
ensembles, or discretise continuum fields. Expected CSVs in
``checkpoints/``:

  - ``physics_conservation.csv``: ``step,energy`` (or ``momentum``,
    ``mass``, ``charge``)
  - ``physics_timestep.csv``:    ``dt,cfl`` (or ``stability``)
  - ``physics_boundary.csv``:    ``cell,interior_value,boundary_value``
  - ``physics_equilibration.csv``: ``step,observable`` (autocorrelated
    series whose first half should match its second half)
  - ``physics_dimensions.csv``:  ``quantity,units_lhs,units_rhs``
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, mean, rows, summary, values

RUN_TAG = "physics"

DEFAULT_THRESHOLDS = {
    "conservation_drift_warning": 1.0e-3,
    "cfl_warning": 1.0,
    "boundary_artifact_ratio_warning": 0.20,
    "equilibration_window_gap_warning": 0.10,
    "dimensional_mismatch_share_warning": 0.0,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


# ─── detector 1 — conservation-law violation ───────────────────────────
def _conservation_drift(path: Path) -> float:
    vals = values(path, "energy", "momentum", "mass", "charge", "total_energy")
    if len(vals) < 2:
        return 0.0
    base = abs(vals[0]) if vals[0] != 0 else 1.0
    return abs(vals[-1] - vals[0]) / base


def _conservation_law_violation(path: Path) -> bool:
    return _conservation_drift(path) > THRESHOLDS["conservation_drift_warning"]


def _conservation_summary(path: Path) -> dict:
    return summary(path, "relative_drift", _conservation_drift(path))


# ─── detector 2 — timestep / CFL violation ─────────────────────────────
def _timestep_or_cfl_violation(path: Path) -> bool:
    cfls = values(path, "cfl", "courant", "stability_number")
    if cfls and max(cfls) > THRESHOLDS["cfl_warning"]:
        return True
    flags = values(path, "stable", "is_stable")
    return bool(flags) and min(flags) <= 0


def _timestep_summary(path: Path) -> dict:
    return summary(path, "max_cfl", max_value(path, "cfl", "courant", "stability_number"))


# ─── detector 3 — boundary-condition artifact ──────────────────────────
def _boundary_ratios(path: Path) -> list[float]:
    out: list[float] = []
    for row in rows(path):
        interior = f(row, "interior_value", "interior", "bulk")
        boundary = f(row, "boundary_value", "boundary", "edge")
        if interior is None or boundary is None or interior == 0:
            continue
        out.append(abs(boundary - interior) / abs(interior))
    return out


def _boundary_condition_artifact(path: Path) -> bool:
    ratios = _boundary_ratios(path)
    return bool(ratios) and max(ratios) > THRESHOLDS["boundary_artifact_ratio_warning"]


def _boundary_summary(path: Path) -> dict:
    ratios = _boundary_ratios(path)
    return {"rows": len(ratios), "max_boundary_anomaly": round(max(ratios), 4) if ratios else 0.0}


# ─── detector 4 — equilibration insufficient ───────────────────────────
def _equilibration_gap(path: Path) -> float:
    vals = values(path, "observable", "energy", "magnetisation", "value")
    if len(vals) < 4:
        return 0.0
    half = len(vals) // 2
    first, second = vals[:half], vals[half:]
    m1, m2 = mean(first), mean(second)
    base = abs(m1) if m1 != 0 else 1.0
    return abs(m2 - m1) / base


def _equilibration_insufficient(path: Path) -> bool:
    return _equilibration_gap(path) > THRESHOLDS["equilibration_window_gap_warning"]


def _equilibration_summary(path: Path) -> dict:
    return summary(path, "two_half_gap", _equilibration_gap(path))


# ─── detector 5 — dimensional inconsistency ────────────────────────────
def _dimensional_mismatch_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    for row in items:
        lhs = (row.get("units_lhs") or row.get("lhs") or "").strip()
        rhs = (row.get("units_rhs") or row.get("rhs") or "").strip()
        if not lhs or not rhs:
            continue
        if lhs != rhs:
            bad += 1
    return bad / len(items)


def _dimensional_inconsistency(path: Path) -> bool:
    return _dimensional_mismatch_share(path) > THRESHOLDS["dimensional_mismatch_share_warning"]


def _dimensional_summary(path: Path) -> dict:
    return summary(path, "mismatch_share", _dimensional_mismatch_share(path))


MANIFEST: list[dict] = [
    {
        "id": "conservation_law_violation",
        "title": "Conserved quantity drifts beyond integrator tolerance",
        "evidence": "physics_conservation.csv",
        "evidence_check": _conservation_law_violation,
        "evidence_summary": _conservation_summary,
        "severity": "high",
        "queries": [
            "energy conservation molecular dynamics symplectic integrator",
            "momentum conservation finite volume scheme",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use a symplectic / conservative integrator and reduce timestep until drift is within tolerance.",
        "why_it_matters": "Drift in conserved quantities indicates the integrator is leaking unphysical energy into the dynamics.",
        "next_checks": [
            "Compare drift across timestep choices.",
            "Verify thermostat / barostat are not masking drift.",
        ],
        "success_criteria": [
            "Relative conserved-quantity drift stays below 1e-3 across the run.",
        ],
        "references": [
            "symplectic integrator",
            "conservation laws numerical",
            "thermostat artifacts",
        ],
    },
    {
        "id": "timestep_or_cfl_violation",
        "title": "Timestep violates CFL / stability condition",
        "evidence": "physics_timestep.csv",
        "evidence_check": _timestep_or_cfl_violation,
        "evidence_summary": _timestep_summary,
        "severity": "high",
        "queries": [
            "CFL stability finite volume hyperbolic",
            "time step constraint explicit scheme stability",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Reduce the timestep or switch to an implicit / adaptive scheme that satisfies the CFL bound.",
        "why_it_matters": "Violating the CFL condition makes explicit schemes unconditionally unstable.",
        "next_checks": [
            "Compute the local CFL on the finest cell.",
            "Compare against the theoretical bound for the chosen scheme.",
        ],
        "success_criteria": [
            "Maximum CFL stays below 1.0 throughout the run.",
        ],
        "references": [
            "Courant-Friedrichs-Lewy condition",
            "explicit scheme stability",
            "adaptive time stepping",
        ],
    },
    {
        "id": "boundary_condition_artifact",
        "title": "Field values near boundaries diverge from the interior",
        "evidence": "physics_boundary.csv",
        "evidence_check": _boundary_condition_artifact,
        "evidence_summary": _boundary_summary,
        "severity": "high",
        "queries": [
            "boundary condition artifact simulation reflection",
            "absorbing boundary perfectly matched layer",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Move the boundary further out, switch to absorbing / periodic conditions, or apply a sponge layer.",
        "why_it_matters": "Boundary artifacts contaminate the interior solution and produce non-physical features.",
        "next_checks": [
            "Compare runs with different domain sizes.",
            "Inspect fields near corners and along inflow / outflow regions.",
        ],
        "success_criteria": [
            "Boundary-to-interior anomaly ratio stays below 0.20.",
        ],
        "references": [
            "absorbing boundary",
            "perfectly matched layer",
            "boundary artifact",
        ],
    },
    {
        "id": "equilibration_insufficient",
        "title": "Sampling has not equilibrated (first half ≠ second half)",
        "evidence": "physics_equilibration.csv",
        "evidence_check": _equilibration_insufficient,
        "evidence_summary": _equilibration_summary,
        "severity": "high",
        "queries": [
            "monte carlo equilibration autocorrelation time",
            "molecular dynamics equilibration block averaging",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Discard burn-in, lengthen the trajectory, and use block-averaging / autocorrelation diagnostics.",
        "why_it_matters": "Pre-equilibration samples bias the ensemble average and uncertainty estimates.",
        "next_checks": [
            "Plot the running average of the observable.",
            "Estimate the integrated autocorrelation time.",
        ],
        "success_criteria": [
            "Two-half mean gap falls below 10% of the observable scale.",
        ],
        "references": [
            "equilibration burn in",
            "block averaging",
            "integrated autocorrelation time",
        ],
    },
    {
        "id": "dimensional_inconsistency",
        "title": "Dimensional analysis fails for one or more derived quantities",
        "evidence": "physics_dimensions.csv",
        "evidence_check": _dimensional_inconsistency,
        "evidence_summary": _dimensional_summary,
        "severity": "high",
        "queries": [
            "dimensional analysis Buckingham pi simulation verification",
            "unit checking scientific software",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add a unit-checking pass (Buckingham-π or symbolic) to every derived expression in the pipeline.",
        "why_it_matters": "Unit mismatches produce results that cannot be physical and often pass numerical sanity checks.",
        "next_checks": [
            "Re-derive each quantity with explicit units.",
            "Add a CI step that verifies dimensional consistency.",
        ],
        "success_criteria": [
            "All audited equations balance dimensionally.",
        ],
        "references": [
            "dimensional analysis",
            "Buckingham pi theorem",
            "unit-aware scientific computing",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
