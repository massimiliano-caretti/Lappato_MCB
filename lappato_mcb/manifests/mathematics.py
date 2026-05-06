"""Mathematics manifest — numerical analysis, optimization, symbolic computation.

For pipelines whose computational core is numerical (linear systems,
optimization, ODE/PDE integration, automated theorem proving, symbolic
algebra). Expected CSVs in ``checkpoints/``:

  - ``math_conditioning.csv``: ``matrix,condition_number``
  - ``math_convergence.csv``: ``iter,loss`` (or ``residual``, ``objective``)
  - ``math_discretization.csv``: ``level,error`` (grid level vs true error)
  - ``math_benchmark.csv``: ``method,reference_value,reported_value``
  - ``math_assumption_audit.csv``: ``assumption,verified`` (1/0)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from ._common import f, max_value, rows, summary, values

RUN_TAG = "mathematics"

DEFAULT_THRESHOLDS = {
    "condition_number_warning": 1.0e8,
    "convergence_plateau_ratio": 0.99,
    "convergence_min_iters": 5.0,
    "discretization_error_warning": 0.10,
    "benchmark_log_gap_warning": 1.0,
    "assumption_unverified_share_warning": 0.20,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


# ─── detector 1 — numerical instability (ill-conditioning) ─────────────
def _numerical_instability(path: Path) -> bool:
    return max_value(path, "condition_number", "kappa", "cond") > THRESHOLDS["condition_number_warning"]


def _conditioning_summary(path: Path) -> dict:
    return summary(path, "max_condition_number", max_value(path, "condition_number", "kappa", "cond"))


# ─── detector 2 — convergence failure / plateau ────────────────────────
def _convergence_values(path: Path) -> list[float]:
    return values(path, "loss", "residual", "objective", "error")


def _convergence_failure(path: Path) -> bool:
    vals = _convergence_values(path)
    if len(vals) < THRESHOLDS["convergence_min_iters"]:
        return False
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return True
    if vals[0] == 0:
        return False
    ratio = vals[-1] / vals[0]
    return ratio > THRESHOLDS["convergence_plateau_ratio"]


def _convergence_summary(path: Path) -> dict:
    vals = _convergence_values(path)
    if not vals:
        return {"rows": 0, "convergence_ratio": 0.0}
    has_nan = any(math.isnan(v) or math.isinf(v) for v in vals)
    ratio = (vals[-1] / vals[0]) if vals[0] else 0.0
    return {
        "rows": len(vals),
        "convergence_ratio": round(ratio, 4),
        "nan_or_inf_seen": has_nan,
    }


# ─── detector 3 — discretization error too large ───────────────────────
def _discretization_error(path: Path) -> bool:
    return max_value(path, "error", "global_error", "step_error") > THRESHOLDS["discretization_error_warning"]


def _discretization_summary(path: Path) -> dict:
    return summary(path, "max_error", max_value(path, "error", "global_error", "step_error"))


# ─── detector 4 — benchmark orders-of-magnitude mismatch ───────────────
def _benchmark_log_gaps(path: Path) -> list[float]:
    out: list[float] = []
    for row in rows(path):
        ref = f(row, "reference_value", "reference", "ground_truth")
        rep = f(row, "reported_value", "reported", "computed_value")
        if ref is None or rep is None or ref == 0 or rep == 0:
            continue
        try:
            out.append(abs(math.log10(abs(rep)) - math.log10(abs(ref))))
        except ValueError:
            continue
    return out


def _benchmark_orders_of_magnitude_mismatch(path: Path) -> bool:
    gaps = _benchmark_log_gaps(path)
    return bool(gaps) and max(gaps) > THRESHOLDS["benchmark_log_gap_warning"]


def _benchmark_summary(path: Path) -> dict:
    gaps = _benchmark_log_gaps(path)
    return {"rows": len(gaps), "max_log10_gap": round(max(gaps), 4) if gaps else 0.0}


# ─── detector 5 — proof assumption unverified ──────────────────────────
def _assumption_unverified_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    total = 0
    for row in items:
        v = (
            row.get("verified")
            or row.get("validated")
            or row.get("checked")
        )
        if v is None or v == "":
            continue
        total += 1
        s = str(v).strip().lower()
        if s in ("0", "false", "no", "unverified", "n"):
            bad += 1
    return (bad / total) if total else 0.0


def _proof_assumption_unverified(path: Path) -> bool:
    return _assumption_unverified_share(path) > THRESHOLDS["assumption_unverified_share_warning"]


def _assumption_summary(path: Path) -> dict:
    return summary(path, "unverified_share", _assumption_unverified_share(path))


MANIFEST: list[dict] = [
    {
        "id": "numerical_instability",
        "title": "Linear system or operator is ill-conditioned",
        "evidence": "math_conditioning.csv",
        "evidence_check": _numerical_instability,
        "evidence_summary": _conditioning_summary,
        "severity": "high",
        "queries": [
            "numerical conditioning ill-posed problem regularization",
            "matrix condition number floating point error analysis",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to a numerically stable factorisation, regularise the operator, or rescale variables.",
        "why_it_matters": "Ill-conditioning amplifies floating-point noise and produces results that are insensitive to the underlying problem.",
        "next_checks": [
            "Recompute the condition number of preconditioned operators.",
            "Compare direct vs iterative solvers under the same tolerance.",
        ],
        "success_criteria": [
            "Condition number drops below 1e8 or solution sensitivity is bounded.",
        ],
        "references": [
            "matrix conditioning",
            "ill-posed problem",
            "regularization linear inverse",
        ],
    },
    {
        "id": "convergence_failure",
        "title": "Optimisation does not converge or hits NaN/Inf",
        "evidence": "math_convergence.csv",
        "evidence_check": _convergence_failure,
        "evidence_summary": _convergence_summary,
        "severity": "high",
        "queries": [
            "optimization convergence diagnostics nonlinear",
            "stochastic gradient stability NaN debugging numerical",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Inspect step size / line search, gradient clipping, and use scale-invariant stopping criteria.",
        "why_it_matters": "A non-converged solver returns whatever iterate it stopped at, regardless of optimality.",
        "next_checks": [
            "Plot the residual on a log scale.",
            "Test with stricter tolerance and larger iteration cap.",
        ],
        "success_criteria": [
            "Residual decays by orders of magnitude before stopping.",
        ],
        "references": [
            "convergence analysis",
            "stopping criteria optimization",
            "NaN debugging",
        ],
    },
    {
        "id": "discretization_error",
        "title": "Discretization error exceeds the target tolerance",
        "evidence": "math_discretization.csv",
        "evidence_check": _discretization_error,
        "evidence_summary": _discretization_summary,
        "severity": "high",
        "queries": [
            "discretization error PDE convergence order",
            "mesh refinement Richardson extrapolation finite difference",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Refine the mesh or step size, verify convergence order, and apply Richardson extrapolation where appropriate.",
        "why_it_matters": "Coarse discretization can change qualitative behaviour, not just numeric precision.",
        "next_checks": [
            "Run a refinement study with at least three grid levels.",
            "Estimate the empirical order of convergence.",
        ],
        "success_criteria": [
            "Empirical convergence order matches the scheme's theoretical order.",
        ],
        "references": [
            "convergence order",
            "Richardson extrapolation",
            "mesh refinement",
        ],
    },
    {
        "id": "benchmark_orders_of_magnitude_mismatch",
        "title": "Computed result differs from the reference by orders of magnitude",
        "evidence": "math_benchmark.csv",
        "evidence_check": _benchmark_orders_of_magnitude_mismatch,
        "evidence_summary": _benchmark_summary,
        "severity": "high",
        "queries": [
            "computational mathematics reproducibility benchmark",
            "verification validation scientific computing units scaling",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Audit unit conventions, scaling, and reference implementations before claiming agreement.",
        "why_it_matters": "Cross-scale mismatches usually indicate unit / sign / scaling bugs that invalidate the comparison.",
        "next_checks": [
            "Re-derive the units of inputs and outputs.",
            "Reproduce on a published benchmark with a known answer.",
        ],
        "success_criteria": [
            "Computed and reference values agree within one order of magnitude or with a documented offset.",
        ],
        "references": [
            "verification and validation",
            "reference benchmark",
            "unit consistency",
        ],
    },
    {
        "id": "proof_assumption_unverified",
        "title": "Stated assumptions are not empirically verified",
        "evidence": "math_assumption_audit.csv",
        "evidence_check": _proof_assumption_unverified,
        "evidence_summary": _assumption_summary,
        "severity": "info",
        "queries": [
            "automated theorem proving assumption checking",
            "computer algebra side condition verification",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add automated assumption checks (positivity, invertibility, domain) to each derivation step.",
        "why_it_matters": "Symbolic results are only valid where their preconditions hold; silent assumptions are a common bug source.",
        "next_checks": [
            "List every side condition explicitly.",
            "Add unit tests covering the boundary of each assumption.",
        ],
        "success_criteria": [
            "All declared assumptions are checked at runtime or by proof.",
        ],
        "references": [
            "side conditions symbolic computation",
            "automated theorem proving",
            "computer algebra correctness",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
