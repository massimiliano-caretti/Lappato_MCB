"""Quantum-computing manifest — circuits, noise, error mitigation.

For pipelines on quantum-circuit simulation, NISQ benchmarks, variational
quantum algorithms, and QML. Expected CSVs in ``checkpoints/``:

  - ``quantum_fidelity.csv``: ``circuit,fidelity``
  - ``quantum_noise.csv``: ``gate,error_rate``
  - ``quantum_shot_count.csv``: ``observable,n_shots,std_error``
  - ``quantum_barren_plateau.csv``: ``layer,gradient_variance``
  - ``quantum_compilation.csv``: ``circuit,depth,native_gates``
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, min_value, rows, summary, values

RUN_TAG = "quantum_computing"

DEFAULT_THRESHOLDS = {
    "circuit_fidelity_warning": 0.90,
    "gate_error_rate_warning": 1.0e-2,
    "shot_std_error_warning": 0.05,
    "barren_plateau_variance_warning": 1.0e-4,
    "non_native_gate_share_warning": 0.20,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _fidelity_low(path: Path) -> bool:
    vals = values(path, "fidelity", "process_fidelity", "average_fidelity")
    return bool(vals) and min(vals) < THRESHOLDS["circuit_fidelity_warning"]


def _fidelity_summary(path: Path) -> dict:
    return summary(path, "min_fidelity", min_value(path, "fidelity", "process_fidelity", "average_fidelity"))


def _gate_error_high(path: Path) -> bool:
    return max_value(path, "error_rate", "gate_error", "epc") > THRESHOLDS["gate_error_rate_warning"]


def _gate_error_summary(path: Path) -> dict:
    return summary(path, "max_gate_error", max_value(path, "error_rate", "gate_error", "epc"))


def _shot_noise_high(path: Path) -> bool:
    return max_value(path, "std_error", "shot_std", "uncertainty") > THRESHOLDS["shot_std_error_warning"]


def _shot_summary(path: Path) -> dict:
    return summary(path, "max_shot_std_error", max_value(path, "std_error", "shot_std", "uncertainty"))


def _barren_plateau(path: Path) -> bool:
    vals = values(path, "gradient_variance", "var_grad")
    return bool(vals) and min(vals) < THRESHOLDS["barren_plateau_variance_warning"]


def _barren_plateau_summary(path: Path) -> dict:
    return summary(path, "min_gradient_variance", min_value(path, "gradient_variance", "var_grad"))


def _non_native_gate_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    shares: list[float] = []
    for row in items:
        d = f(row, "depth", "circuit_depth")
        n = f(row, "native_gates", "n_native")
        if d is None or n is None or d <= 0:
            continue
        shares.append(max(1 - (n / d), 0.0))
    return max(shares) if shares else 0.0


def _compilation_inefficient(path: Path) -> bool:
    return _non_native_gate_share(path) > THRESHOLDS["non_native_gate_share_warning"]


def _compilation_summary(path: Path) -> dict:
    return summary(path, "max_non_native_share", _non_native_gate_share(path))


MANIFEST: list[dict] = [
    {
        "id": "fidelity_low",
        "title": "Reported circuit fidelity is below the safe-execution floor",
        "evidence": "quantum_fidelity.csv",
        "evidence_check": _fidelity_low,
        "evidence_summary": _fidelity_summary,
        "severity": "high",
        "queries": [
            "quantum process fidelity randomized benchmarking",
            "NISQ device fidelity certification",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Calibrate gates, run randomized benchmarking and apply error mitigation (ZNE, PEC) before reporting.",
        "why_it_matters": "Below-floor fidelity means the circuit output is dominated by noise, not the target unitary.",
        "next_checks": [
            "Run randomized benchmarking per qubit.",
            "Apply zero-noise extrapolation.",
        ],
        "success_criteria": [
            "Minimum reported fidelity exceeds 0.90.",
        ],
        "references": [
            "randomized benchmarking",
            "zero noise extrapolation",
            "probabilistic error cancellation",
        ],
    },
    {
        "id": "gate_error_high",
        "title": "Gate error rate exceeds device specification",
        "evidence": "quantum_noise.csv",
        "evidence_check": _gate_error_high,
        "evidence_summary": _gate_error_summary,
        "severity": "high",
        "queries": [
            "two qubit gate error rate calibration",
            "cross talk readout error suppression",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Recalibrate the affected gate, audit cross-talk and apply dynamical decoupling.",
        "why_it_matters": "Per-gate error rates compound exponentially in deep circuits.",
        "next_checks": [
            "Run interleaved randomized benchmarking on the suspect gate.",
            "Audit cross-talk neighbours.",
        ],
        "success_criteria": [
            "Maximum gate error stays below 1e-2.",
        ],
        "references": [
            "interleaved randomized benchmarking",
            "cross-talk",
            "dynamical decoupling",
        ],
    },
    {
        "id": "shot_noise_high",
        "title": "Observable estimation has too few shots",
        "evidence": "quantum_shot_count.csv",
        "evidence_check": _shot_noise_high,
        "evidence_summary": _shot_summary,
        "severity": "info",
        "queries": [
            "shot noise variance reduction quantum expectation value",
            "classical shadows estimation quantum observables",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Increase shots, group commuting observables, or use classical-shadow tomography.",
        "why_it_matters": "Shot noise above the operating budget makes optimisation gradients unreliable.",
        "next_checks": [
            "Estimate the variance budget per observable.",
            "Group Pauli observables for joint measurement.",
        ],
        "success_criteria": [
            "Shot standard error stays below 5% of the observable scale.",
        ],
        "references": [
            "classical shadows",
            "operator grouping",
            "shot noise variance reduction",
        ],
    },
    {
        "id": "barren_plateau",
        "title": "Variational gradient variance vanishes (barren plateau)",
        "evidence": "quantum_barren_plateau.csv",
        "evidence_check": _barren_plateau,
        "evidence_summary": _barren_plateau_summary,
        "severity": "high",
        "queries": [
            "barren plateau variational quantum eigensolver",
            "parameter initialization quantum neural network",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to local-cost / problem-inspired ansätze and use small-angle / identity-block initialisation.",
        "why_it_matters": "Barren plateaus make variational training scale exponentially badly with system size.",
        "next_checks": [
            "Compare local-cost vs global-cost ansätze.",
            "Run identity-block initialisation.",
        ],
        "success_criteria": [
            "Gradient variance stays above 1e-4 across layers.",
        ],
        "references": [
            "barren plateau",
            "local cost function",
            "identity block initialisation",
        ],
    },
    {
        "id": "compilation_inefficient",
        "title": "Circuit uses many non-native gates after compilation",
        "evidence": "quantum_compilation.csv",
        "evidence_check": _compilation_inefficient,
        "evidence_summary": _compilation_summary,
        "severity": "info",
        "queries": [
            "quantum circuit compilation native gate set",
            "quantum routing optimization swap minimization",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Re-target the transpiler to the device's native gate set with explicit qubit routing constraints.",
        "why_it_matters": "Non-native gates inflate circuit depth and reduce achievable fidelity.",
        "next_checks": [
            "Inspect transpiled depth across optimisation levels.",
            "Compare native vs generic gate-set basis.",
        ],
        "success_criteria": [
            "Non-native gate share stays below 20% of compiled depth.",
        ],
        "references": [
            "transpilation",
            "native gate set",
            "qubit routing",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
