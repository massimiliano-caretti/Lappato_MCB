"""Regression tests for the v1.6 insight detectors.

Covers both the new statistical helpers in :mod:`lappato_mcb._stats`
(stdlib path AND scipy path when available) and the seven new
``pipeline_health`` manifest entries (subgroup_disparity,
calibration_bin_gap, decision_threshold_suboptimal,
failure_clustering, cross_cycle_drift, underpowered_cohort,
syndrome_composition).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lappato_mcb import _stats
from lappato_mcb.manifests import pipeline_health as ph

# ────────────────────────── stat helpers ────────────────────────────

def _force_stdlib(monkeypatch):
    """Force the stdlib code paths for tests that compare against scipy."""
    monkeypatch.setattr(_stats, "_HAS_SCIPY", False)


def test_normal_cdf_inv_roundtrip():
    """Beasley-Springer inverse-CDF must roundtrip math.erf-based CDF."""
    for z in (-2.5, -1.0, 0.0, 0.5, 1.96, 2.5):
        p = _stats._normal_cdf(z)
        z_back = _stats._normal_inv_cdf(p)
        assert abs(z - z_back) < 1e-6


def test_chi2_p_value_basic_smoke():
    """Easy table: clear association → p < 0.05."""
    p = _stats.chi2_p_value([[10, 20, 30], [40, 30, 20]])
    assert 0.0 < p < 0.05


def test_chi2_p_value_no_association():
    """Equal proportions → p close to 1."""
    p = _stats.chi2_p_value([[10, 10], [10, 10]])
    assert p > 0.9


def test_chi2_p_value_stdlib_matches_scipy(monkeypatch):
    """Stdlib chi-square must agree with scipy within 1e-3 on common cases."""
    pytest.importorskip("scipy")
    table = [[10, 20, 30], [40, 30, 20]]
    p_with_scipy = _stats.chi2_p_value(table)
    _force_stdlib(monkeypatch)
    p_stdlib = _stats.chi2_p_value(table)
    assert abs(p_with_scipy - p_stdlib) < 1e-3


def test_fisher_exact_2x2_roundtrip(monkeypatch):
    """Stdlib fisher_exact must agree with scipy on the standard
    Lady-tasting-tea-style 2x2 table."""
    pytest.importorskip("scipy")
    table = [[8, 2], [1, 5]]
    p_scipy = _stats.fisher_exact_2x2(table)
    _force_stdlib(monkeypatch)
    p_stdlib = _stats.fisher_exact_2x2(table)
    assert abs(p_scipy - p_stdlib) < 1e-6


def test_fisher_exact_2x2_extreme_table():
    """A perfectly-separable 2x2 must give a tiny p-value."""
    p = _stats.fisher_exact_2x2([[20, 0], [0, 20]])
    assert p < 1e-6


def test_linear_regression_slope_p_strong_signal():
    xs = list(range(20))
    ys = [2.0 * x + 0.5 for x in xs]
    slope, intercept, p = _stats.linear_regression_slope_p(xs, ys)
    assert abs(slope - 2.0) < 1e-6
    assert abs(intercept - 0.5) < 1e-6
    assert p < 1e-9


def test_linear_regression_slope_p_no_trend():
    xs = list(range(20))
    ys = [5.0] * 20
    _, _, p = _stats.linear_regression_slope_p(xs, ys)
    assert p == 1.0


def test_required_sample_size_observed_below_target():
    """When observation does NOT exceed target, return 0 — no
    finite-N test will reject H₀."""
    n = _stats.required_sample_size_for_proportion(0.85, 0.90)
    assert n == 0


def test_required_sample_size_observed_above_target():
    n = _stats.required_sample_size_for_proportion(0.92, 0.85)
    assert 100 <= n <= 200


# ────────────────────────── detector tests ──────────────────────────

def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# d17 — subgroup_disparity ──────────────────────────────────────────
def test_subgroup_disparity_fires_on_skewed(tmp_path):
    p = tmp_path / "pipeline_subgroup_metrics.csv"
    _write_csv(p,
        ["subgroup_key", "subgroup_value", "n", "n_errors"],
        [
            ["type", "A", 30, 1],
            ["type", "B", 30, 1],
            ["type", "C", 20, 8],   # 40 % errors vs ~10 % global → ratio > 2
        ],
    )
    assert ph._subgroup_disparity_present(p)
    summary = ph._subgroup_disparity_summary(p)
    assert summary["worst_subgroup"] == "type=C"


def test_subgroup_disparity_no_skew(tmp_path):
    p = tmp_path / "pipeline_subgroup_metrics.csv"
    _write_csv(p,
        ["subgroup_key", "subgroup_value", "n", "n_errors"],
        [
            ["type", "A", 30, 3],
            ["type", "B", 30, 3],
            ["type", "C", 30, 4],
        ],
    )
    assert not ph._subgroup_disparity_present(p)


def test_subgroup_disparity_missing_file(tmp_path):
    """Missing file must NOT fire (fail-closed)."""
    p = tmp_path / "does_not_exist.csv"
    assert not ph._subgroup_disparity_present(p)


# d18 — calibration_bin_gap ─────────────────────────────────────────
def test_calibration_bin_gap_fires_on_underconfident(tmp_path):
    p = tmp_path / "pipeline_reliability_diagram.csv"
    _write_csv(p,
        ["bin_lo", "bin_hi", "count", "mean_confidence", "accuracy"],
        [
            [0.5, 0.6, 10, 0.55, 0.55],
            [0.6, 0.7, 24, 0.65, 0.96],   # gap = -0.31
            [0.9, 1.0, 50, 0.95, 0.95],
        ],
    )
    assert ph._calibration_bin_gap_present(p)
    summary = ph._calibration_bin_gap_summary(p)
    assert summary["max_bin_gap"] > 0.10


def test_calibration_bin_gap_well_calibrated(tmp_path):
    p = tmp_path / "pipeline_reliability_diagram.csv"
    _write_csv(p,
        ["bin_lo", "bin_hi", "count", "mean_confidence", "accuracy"],
        [
            [0.5, 0.6, 20, 0.55, 0.54],
            [0.6, 0.7, 24, 0.65, 0.66],
            [0.9, 1.0, 50, 0.95, 0.93],
        ],
    )
    assert not ph._calibration_bin_gap_present(p)


# d19 — decision_threshold_suboptimal ───────────────────────────────
def test_decision_threshold_fires_when_optimum_far_from_05(tmp_path):
    p = tmp_path / "pipeline_predictions_with_probs.csv"
    rows_data: list[list] = []
    # Highly imbalanced negatives clustered near 0.20, positives spread
    # from 0.30 to 0.55 — default 0.5 cut-off classifies most positives
    # as negative; the Youden-J optimum sits around 0.30-0.35, well
    # below 0.5.  Mirrors a real low-prevalence binary screening
    # scenario where the default threshold is sub-optimal.
    for i in range(60):
        rows_data.append([f"neg{i}", 0, 0.10 + 0.005 * (i % 20)])  # 0.10..0.20
    for i in range(40):
        rows_data.append([f"pos{i}", 1, 0.30 + 0.005 * (i % 20)])  # 0.30..0.40
    _write_csv(p, ["item_id", "y_true", "p_positive"], rows_data)
    assert ph._decision_threshold_suboptimal(p)
    s = ph._decision_threshold_summary(p)
    assert abs(s["best_threshold"] - 0.5) >= 0.05


def test_decision_threshold_optimal_near_05(tmp_path):
    p = tmp_path / "pipeline_predictions_with_probs.csv"
    rows_data: list[list] = []
    # Symmetric, well-separated → optimum should be near 0.5.
    for i in range(50):
        rows_data.append([f"neg{i}", 0, 0.10 + 0.005 * i])
    for i in range(50):
        rows_data.append([f"pos{i}", 1, 0.65 + 0.005 * i])
    _write_csv(p, ["item_id", "y_true", "p_positive"], rows_data)
    s = ph._decision_threshold_summary(p)
    # Best threshold should land in [0.40, 0.65] given the gap.
    assert 0.30 <= s["best_threshold"] <= 0.70


# d20 — failure_clustering ──────────────────────────────────────────
def test_failure_clustering_fires_on_concentration(tmp_path):
    p = tmp_path / "pipeline_per_item_predictions.csv"
    rows_data: list[list] = []
    # group A: 0/20 errors; group B: 0/20 errors; group C: 8/10 errors.
    for i in range(20):
        rows_data.append([f"a{i}", "A", 1, 1])
    for i in range(20):
        rows_data.append([f"b{i}", "B", 1, 1])
    for i in range(8):
        rows_data.append([f"c{i}", "C", 1, 0])  # error
    for i in range(2):
        rows_data.append([f"c{i+8}", "C", 1, 1])
    _write_csv(p, ["item_id", "group_id", "y_true", "y_pred"], rows_data)
    assert ph._failure_clustering_present(p)


def test_failure_clustering_uniform(tmp_path):
    p = tmp_path / "pipeline_per_item_predictions.csv"
    rows_data: list[list] = []
    # Every group has 1/10 errors → uniform.
    for g in ("A", "B", "C"):
        for i in range(9):
            rows_data.append([f"{g}{i}", g, 1, 1])
        rows_data.append([f"{g}9", g, 1, 0])
    _write_csv(p, ["item_id", "group_id", "y_true", "y_pred"], rows_data)
    assert not ph._failure_clustering_present(p)


# d21 — cross_cycle_drift ───────────────────────────────────────────
def test_cross_cycle_drift_fires_on_trend(tmp_path):
    p = tmp_path / "pipeline_health_lappato_mcb_meta.csv"
    rows_data: list[list] = []
    # 12 cycles, monotone declining metric → strong slope.
    for c in range(12):
        rows_data.append([c, 100.0 - 4.0 * c])
    _write_csv(p, ["cycle", "primary_metric"], rows_data)
    assert ph._cross_cycle_drift_present(p)


def test_cross_cycle_drift_stable(tmp_path):
    p = tmp_path / "pipeline_health_lappato_mcb_meta.csv"
    rows_data: list[list] = []
    for c in range(12):
        rows_data.append([c, 50.0 + 0.001 * (c % 3)])
    _write_csv(p, ["cycle", "primary_metric"], rows_data)
    assert not ph._cross_cycle_drift_present(p)


# d22 — underpowered_cohort ─────────────────────────────────────────
def test_underpowered_cohort_fires_on_small(tmp_path):
    p = tmp_path / "pipeline_class_counts.csv"
    _write_csv(p, ["class", "count"], [["N", 21], ["T", 42]])
    # Default threshold = 1000 → both classes below.
    assert ph._underpowered_cohort_present(p)
    s = ph._underpowered_cohort_summary(p)
    assert s["minority_count"] == 21
    assert s["additional_needed"] >= 900


# d23 — syndrome_composition ────────────────────────────────────────
def test_syndrome_composition_active_when_triggers_fire():
    """The detector reads ``THRESHOLDS['__active_card_ids__']`` —
    if we inject the canonical Riley-2019 trigger set the syndrome
    fires."""
    saved = ph.THRESHOLDS.get("__active_card_ids__")
    try:
        ph.THRESHOLDS["__active_card_ids__"] = {
            "small_minority_class",
            "high_cross_seed_variance",
            "acc_BAC_gap_high",
        }
        ok = ph._syndrome_composition_present(Path("/dev/null"))
        s  = ph._syndrome_composition_summary(Path("/dev/null"))
        assert ok
        assert "underpowered_imbalanced_clinical_cohort" in s["syndromes"]
    finally:
        if saved is None:
            ph.THRESHOLDS.pop("__active_card_ids__", None)
        else:
            ph.THRESHOLDS["__active_card_ids__"] = saved


def test_syndrome_composition_inactive_when_unrelated_card_fires():
    saved = ph.THRESHOLDS.get("__active_card_ids__")
    try:
        ph.THRESHOLDS["__active_card_ids__"] = {"single_split_used"}
        assert not ph._syndrome_composition_present(Path("/dev/null"))
    finally:
        if saved is None:
            ph.THRESHOLDS.pop("__active_card_ids__", None)
        else:
            ph.THRESHOLDS["__active_card_ids__"] = saved


# ────────────────────────── manifest sanity ─────────────────────────

def test_manifest_has_23_entries_and_unique_ids():
    """v1.6 expanded the pipeline_health manifest from 16 to 23
    entries.  Every id must remain unique to avoid duplicate cards."""
    assert len(ph.MANIFEST) == 23
    ids = [e["id"] for e in ph.MANIFEST]
    assert len(set(ids)) == len(ids)


def test_new_thresholds_present_with_defaults():
    """Each v1.6 detector relies on a documented threshold; their
    defaults must match the literature-cited values in the manifest
    docstring."""
    expected = {
        "subgroup_disparity_ratio_warning": 2.0,
        "subgroup_disparity_p_warning":     0.05,
        "calibration_bin_gap_warning":      0.10,
        "decision_threshold_distance_warning": 0.05,
        "failure_clustering_p_warning":     0.05,
        "drift_p_warning":                  0.05,
    }
    for k, v in expected.items():
        assert ph.DEFAULT_THRESHOLDS[k] == v
