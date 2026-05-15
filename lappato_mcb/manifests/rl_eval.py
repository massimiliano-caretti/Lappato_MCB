"""Reinforcement-learning evaluation manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import max_value, mean, summary, values

RUN_TAG = "rl_eval"

DEFAULT_THRESHOLDS = {
    "seed_return_spread_warning": 0.20,
    "constraint_violation_warning": 0.01,
    "off_policy_error_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _seed_instability(path: Path) -> bool:
    vals = values(path, "return", "reward", "score")
    if not vals:
        return False
    denom = abs(mean(vals)) or 1.0
    return (max(vals) - min(vals)) / denom > THRESHOLDS["seed_return_spread_warning"]


def _seed_summary(path: Path) -> dict:
    vals = values(path, "return", "reward", "score")
    denom = abs(mean(vals)) or 1.0
    return summary(path, "relative_return_spread", ((max(vals) - min(vals)) / denom) if vals else 0.0)


def _constraint_violation(path: Path) -> bool:
    return max_value(path, "violation_rate", "unsafe_rate") > THRESHOLDS["constraint_violation_warning"]


def _constraint_summary(path: Path) -> dict:
    return summary(path, "max_violation_rate", max_value(path, "violation_rate", "unsafe_rate"))


def _ope_error_high(path: Path) -> bool:
    return max_value(path, "absolute_error", "ope_error") > THRESHOLDS["off_policy_error_warning"]


def _ope_summary(path: Path) -> dict:
    return summary(path, "max_ope_error", max_value(path, "absolute_error", "ope_error"))


MANIFEST = [
    {
        "id": "seed_instability",
        "title": "RL return is unstable across random seeds",
        "evidence": "rl_seed_returns.csv",
        "evidence_check": _seed_instability,
        "evidence_summary": _seed_summary,
        "severity": "medium",
        "queries": ["deep reinforcement learning reproducibility random seeds evaluation", "reinforcement learning statistical significance evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report multiple seeds with confidence intervals and compare robust aggregate metrics.",
        "why_it_matters": "Single-seed RL results are often misleading.",
        "next_checks": ["Run more seeds.", "Report IQM or confidence intervals."],
        "success_criteria": ["Relative return spread falls below 0.20 or uncertainty is reported."],
        "references": ["RL reproducibility", "random seeds", "statistical evaluation"],
    },
    {
        "id": "safety_constraint_violations",
        "title": "Policy violates safety or environment constraints",
        "evidence": "rl_safety.csv",
        "evidence_check": _constraint_violation,
        "evidence_summary": _constraint_summary,
        "severity": "critical",
        "queries": ["safe reinforcement learning constraint violation evaluation", "constrained reinforcement learning safety benchmark"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add constrained RL baselines and report violation rate separately from reward.",
        "why_it_matters": "Reward gains are not acceptable if constraints fail.",
        "next_checks": ["Log violation types.", "Compare constrained algorithms."],
        "success_criteria": ["Violation rate falls below 1% or domain tolerance."],
        "references": ["safe RL", "constrained RL", "safety evaluation"],
    },
    {
        "id": "off_policy_eval_error",
        "title": "Off-policy evaluation error is high",
        "evidence": "rl_ope.csv",
        "evidence_check": _ope_error_high,
        "evidence_summary": _ope_summary,
        "severity": "high",
        "queries": ["off policy evaluation reinforcement learning doubly robust importance sampling", "offline reinforcement learning evaluation uncertainty"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compare OPE estimators and add uncertainty before selecting offline policies.",
        "why_it_matters": "Bad OPE can select harmful policies without online testing.",
        "next_checks": ["Compare IS, WIS and doubly robust OPE.", "Bootstrap OPE intervals."],
        "success_criteria": ["OPE error falls below 0.10 or uncertainty blocks deployment."],
        "references": ["off-policy evaluation", "doubly robust", "offline RL"],
    },
    {
        "id": "reward_hacking_audit_missing",
        "title": "Reward-hacking audit is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["reward hacking reinforcement learning specification gaming", "RL reward misspecification evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Inspect high-reward trajectories for specification gaming and add auxiliary constraints.",
        "why_it_matters": "Policies can maximize reward while violating intent.",
        "next_checks": ["Review top-reward trajectories.", "Add human or rule-based audits."],
        "success_criteria": ["High-reward behavior matches intended task."],
        "references": ["reward hacking", "specification gaming", "RL safety"],
    },
    {
        "id": "exploration_diagnostics_missing",
        "title": "Exploration diagnostics are not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["reinforcement learning exploration diagnostics state coverage", "exploration collapse reinforcement learning evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Track state/action coverage and compare exploration strategies.",
        "why_it_matters": "Poor exploration can look like algorithm failure or false convergence.",
        "next_checks": ["Measure state coverage.", "Compare entropy or curiosity bonuses."],
        "success_criteria": ["Coverage improves without unsafe violations."],
        "references": ["exploration", "state coverage", "curiosity"],
    },
]

