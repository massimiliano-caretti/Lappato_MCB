"""Robotics manifest — control, locomotion, manipulation, sim2real.

For pipelines on robot learning, model-based / model-free control,
sim-to-real transfer and on-robot deployment. Expected CSVs in
``checkpoints/``:

  - ``robotics_sim2real_gap.csv``: ``task,sim_score,real_score``
  - ``robotics_safety_violations.csv``: ``rollout,collisions,torque_limit``
  - ``robotics_action_saturation.csv``: ``timestep,share_at_limit``
  - ``robotics_jitter.csv``: ``rollout,action_jitter``
  - ``robotics_evaluation_seeds.csv``: ``seed,success_rate``
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import f, max_value, mean, rows, summary, values

RUN_TAG = "robotics"

DEFAULT_THRESHOLDS = {
    "sim2real_gap_warning": 0.20,
    "safety_violation_share_warning": 0.05,
    "action_saturation_share_warning": 0.30,
    "action_jitter_warning": 0.20,
    "min_eval_seeds_warning": 5.0,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _sim2real_gap(path: Path) -> float:
    gaps: list[float] = []
    for row in rows(path):
        s = f(row, "sim_score", "sim", "simulation")
        r = f(row, "real_score", "real", "deployed")
        if s is None or r is None:
            continue
        gaps.append(s - r)
    return max(gaps) if gaps else 0.0


def _sim2real_gap_high(path: Path) -> bool:
    return _sim2real_gap(path) > THRESHOLDS["sim2real_gap_warning"]


def _sim2real_summary(path: Path) -> dict:
    return summary(path, "max_sim2real_gap", _sim2real_gap(path))


def _safety_violation_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    for row in items:
        c = f(row, "collisions", "n_collisions") or 0
        t = f(row, "torque_limit", "n_torque_violation") or 0
        if c > 0 or t > 0:
            bad += 1
    return bad / len(items)


def _safety_violation_high(path: Path) -> bool:
    return _safety_violation_share(path) > THRESHOLDS["safety_violation_share_warning"]


def _safety_summary(path: Path) -> dict:
    return summary(path, "safety_violation_share", _safety_violation_share(path))


def _action_saturation_high(path: Path) -> bool:
    return max_value(path, "share_at_limit", "saturation") > THRESHOLDS["action_saturation_share_warning"]


def _action_saturation_summary(path: Path) -> dict:
    return summary(path, "max_action_saturation", max_value(path, "share_at_limit", "saturation"))


def _action_jitter_high(path: Path) -> bool:
    return max_value(path, "action_jitter", "jitter") > THRESHOLDS["action_jitter_warning"]


def _jitter_summary(path: Path) -> dict:
    return summary(path, "max_action_jitter", max_value(path, "action_jitter", "jitter"))


def _few_eval_seeds(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    seeds = {row.get("seed") for row in items if row.get("seed")}
    return len(seeds) < THRESHOLDS["min_eval_seeds_warning"]


def _seeds_summary(path: Path) -> dict:
    items = rows(path)
    seeds = {row.get("seed") for row in items if row.get("seed")}
    rates = values(path, "success_rate", "success")
    return {
        "rows": len(items),
        "n_unique_seeds": len(seeds),
        "mean_success_rate": round(mean(rates), 4) if rates else 0.0,
    }


MANIFEST: list[dict] = [
    {
        "id": "sim2real_gap_high",
        "title": "Sim-to-real performance gap is large",
        "evidence": "robotics_sim2real_gap.csv",
        "evidence_check": _sim2real_gap_high,
        "evidence_summary": _sim2real_summary,
        "severity": "high",
        "queries": [
            "sim to real transfer reinforcement learning robotics",
            "domain randomization sim2real policy transfer",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply domain randomisation, system-identification fine-tuning, or residual policies before real-robot evaluation.",
        "why_it_matters": "A large sim2real gap renders simulation-based claims meaningless for deployment.",
        "next_checks": [
            "Profile the dynamics gap (mass / friction / latency).",
            "Run a residual real-world fine-tune.",
        ],
        "success_criteria": [
            "Real-robot success rate is within 0.20 of the simulation score.",
        ],
        "references": [
            "domain randomization",
            "system identification",
            "residual policy learning",
        ],
    },
    {
        "id": "safety_violation_high",
        "title": "Rollouts exceed collision or torque limits",
        "evidence": "robotics_safety_violations.csv",
        "evidence_check": _safety_violation_high,
        "evidence_summary": _safety_summary,
        "severity": "high",
        "queries": [
            "safe reinforcement learning constraint violation robotics",
            "control barrier function safe policy deployment",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add a safety filter / control-barrier function and enforce constrained-policy optimisation.",
        "why_it_matters": "Safety violations during evaluation reflect fundamental control-policy unsafety, not just bad luck.",
        "next_checks": [
            "Wrap the policy with a CBF / shielding layer.",
            "Re-run the evaluation under the safety wrapper.",
        ],
        "success_criteria": [
            "Safety-violation share stays below 5% of rollouts.",
        ],
        "references": [
            "safe reinforcement learning",
            "control barrier function",
            "shielded RL",
        ],
    },
    {
        "id": "action_saturation_high",
        "title": "Policy actions saturate at actuator limits",
        "evidence": "robotics_action_saturation.csv",
        "evidence_check": _action_saturation_high,
        "evidence_summary": _action_saturation_summary,
        "severity": "info",
        "queries": [
            "action saturation reinforcement learning continuous control",
            "tanh squashed policy actuator limit",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Tighten action-rate / smoothness penalties and renormalise the action space against actuator limits.",
        "why_it_matters": "Saturated actions waste actuator headroom, raise wear, and prevent fine control.",
        "next_checks": [
            "Quantify share at limit per joint.",
            "Add an action-rate penalty.",
        ],
        "success_criteria": [
            "Action saturation share stays below 30% of timesteps.",
        ],
        "references": [
            "action smoothness reward",
            "actuator limit",
            "policy regularisation",
        ],
    },
    {
        "id": "action_jitter_high",
        "title": "Policy outputs are jittery between consecutive steps",
        "evidence": "robotics_jitter.csv",
        "evidence_check": _action_jitter_high,
        "evidence_summary": _jitter_summary,
        "severity": "info",
        "queries": [
            "action smoothness reinforcement learning robotics",
            "policy jitter low pass filter actuator wear",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add temporal-smoothness penalties or low-pass filtering and verify task success is preserved.",
        "why_it_matters": "Jitter accelerates actuator wear and amplifies noise in real-world deployment.",
        "next_checks": [
            "Plot per-rollout action time-series.",
            "Compare smooth-policy variants.",
        ],
        "success_criteria": [
            "Action jitter falls below 0.20 of the action range.",
        ],
        "references": [
            "smoothness regularisation",
            "low-pass action filter",
            "actuator wear",
        ],
    },
    {
        "id": "few_eval_seeds",
        "title": "Policy evaluation uses too few seeds",
        "evidence": "robotics_evaluation_seeds.csv",
        "evidence_check": _few_eval_seeds,
        "evidence_summary": _seeds_summary,
        "severity": "high",
        "queries": [
            "reinforcement learning evaluation reproducibility seeds",
            "Henderson deep reinforcement learning that matters",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run at least 5 seeds per condition and report bootstrap CIs.",
        "why_it_matters": "Single-seed RL/robotics results are notoriously irreproducible.",
        "next_checks": [
            "Re-run with ≥5 seeds.",
            "Report bootstrap-CI on success rate.",
        ],
        "success_criteria": [
            "Each reported result averages ≥ 5 seeds with dispersion.",
        ],
        "references": [
            "Henderson 2018 RL reproducibility",
            "rliable",
            "RL evaluation",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
