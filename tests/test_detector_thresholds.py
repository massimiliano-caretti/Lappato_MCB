"""Boundary-case tests for the v1.2 detector catalogue.

What this file is — and what it is NOT.

These tests prove that every covered detector **respects its
documented numeric threshold**: a fixture generated just above the
threshold (in the firing direction) must fire, and a fixture
generated just below it must not. The threshold values come from
each manifest's ``THRESHOLDS`` dict (single source of truth — if a
default changes, the boundary fixture automatically follows).

This is a *more rigorous* sentinel system than the simple
fire / fail-closed pairs in ``test_new_manifests.py``. It does
NOT, however, validate the detectors against real-world datasets.
For that you need:
  - a real per-domain corpus with ground-truth labels for the
    weakness each detector targets;
  - domain expertise to confirm whether the detector's threshold
    is biomedically / physically / statistically appropriate.

Until that empirical validation exists, treat this file as
documentation-as-test of the threshold contract: it cannot lie about
where a detector fires, even if the *choice* of threshold is still
debatable.

Naming convention: ``test_<detector_id>_boundary``. One TestCase per
covered manifest. Detectors that are purely structural (categorical /
string-match / row-count) are listed in the docstring and excluded
on purpose.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


def _both_sides(
    test: unittest.TestCase,
    detector: Callable[[Path], bool],
    builder: Callable[[float], str],
    threshold: float,
    fire_direction: str,
    eps: float,
    filename: str,
) -> None:
    """Drive a detector at threshold ± eps and assert the fire / no-fire
    contract. ``fire_direction`` is one of ``"above"`` or ``"below"``.
    """
    above = threshold + eps
    below = max(threshold - eps, 0.0) if fire_direction == "above" else threshold - eps
    if fire_direction == "above":
        firing_value, silent_value = above, below
    elif fire_direction == "below":
        firing_value, silent_value = below, above
    else:
        raise ValueError(f"unknown fire_direction {fire_direction!r}")

    with tempfile.TemporaryDirectory() as td:
        p_fire = _write(Path(td), filename, builder(firing_value))
        test.assertTrue(
            detector(p_fire),
            f"{detector.__name__}: did not fire at value={firing_value} "
            f"(threshold={threshold}, fire_direction={fire_direction})",
        )
    with tempfile.TemporaryDirectory() as td:
        p_silent = _write(Path(td), filename, builder(silent_value))
        test.assertFalse(
            detector(p_silent),
            f"{detector.__name__}: fired at value={silent_value} "
            f"(threshold={threshold}, fire_direction={fire_direction})",
        )


# ─── pipeline_health ──────────────────────────────────────────────────
class PipelineHealthBoundaryTests(unittest.TestCase):
    """Boundary tests for the 14 numeric detectors in pipeline_health.

    Excluded (structural, no numeric threshold to bracket):
      - ``single_split_used``: triggers when the file has exactly 1 row.
      - ``threshold_picked_on_test_set``: matches a categorical pair
        ('threshold_selection' stage on 'test' split + 'final_report'
        stage on 'test' split).

    Both are covered by hand-written cases in test_new_manifests.py.
    """

    def setUp(self):
        from lappato_mcb.manifests import pipeline_health as ph
        self.ph = ph
        self.T = ph.THRESHOLDS

    def test_small_minority_class_boundary(self):
        # Fires when min(class counts) < min_minority_n (default 1000).
        thr = self.T["min_minority_n"]
        _both_sides(
            self, self.ph._small_minority_class,
            lambda v: f"class,count\nA,10000\nB,{int(v)}\n",
            thr, "below", eps=10, filename="pipeline_class_counts.csv",
        )

    def test_high_cross_seed_variance_boundary(self):
        # Fires when stdev(metric across seeds) > seed_std_warning (0.03).
        thr = self.T["seed_std_warning"]
        # Stdev between 0.5-d and 0.5+d on two seeds equals d*sqrt(2).
        # Pick d = thr/sqrt(2) + eps to land above; -eps to land below.
        import math
        def builder(v):
            d = v / math.sqrt(2.0)
            return (
                "seed,balanced_accuracy\n"
                f"0,{0.5 - d:.6f}\n1,{0.5 + d:.6f}\n"
            )
        _both_sides(
            self, self.ph._high_cross_seed_variance, builder,
            thr, "above", eps=0.005, filename="pipeline_cv_metrics.csv",
        )

    def test_acc_bac_gap_high_boundary(self):
        # Fires when max |acc - bac| > acc_bac_gap_warning (0.05).
        thr = self.T["acc_bac_gap_warning"]
        _both_sides(
            self, self.ph._acc_bac_gap_high,
            lambda v: f"seed,accuracy,balanced_accuracy\n0,0.95,{0.95 - v:.6f}\n",
            thr, "above", eps=0.005, filename="pipeline_cv_metrics.csv",
        )

    def test_severe_class_imbalance_boundary(self):
        # Fires when max/min > imbalance_ratio_warning (default 5.0).
        thr = self.T["imbalance_ratio_warning"]
        _both_sides(
            self, self.ph._severe_imbalance,
            lambda v: f"class,count\nA,{int(1000 * v)}\nB,1000\n",
            thr, "above", eps=0.5, filename="pipeline_class_counts.csv",
        )

    def test_feature_importance_concentration_boundary(self):
        # Fires when top-1 share > top_feature_share_warning (0.30).
        thr = self.T["top_feature_share_warning"]
        def builder(v):
            top = v
            rest = (1.0 - top) / 9
            rows = [f"f{i},{rest:.6f}" for i in range(1, 10)]
            rows.insert(0, f"f0,{top:.6f}")
            return "feature,gain\n" + "\n".join(rows) + "\n"
        _both_sides(
            self, self.ph._feature_importance_concentration, builder,
            thr, "above", eps=0.02,
            filename="pipeline_feature_importance.csv",
        )

    def test_small_cohort_size_boundary(self):
        # Fires when sum(counts) < cohort_size_warning (default 30).
        thr = self.T["cohort_size_warning"]
        _both_sides(
            self, self.ph._small_cohort_size,
            lambda v: f"class,count\nA,{int(v // 2)}\nB,{int(v - v // 2)}\n",
            thr, "below", eps=2, filename="pipeline_class_counts.csv",
        )

    def test_external_test_calibration_drop_boundary(self):
        # Fires when (internal - external) > calibration_drop_warning (0.15).
        thr = self.T["calibration_drop_warning"]
        _both_sides(
            self, self.ph._external_test_calibration_drop,
            lambda v: (
                "split,auc\n"
                f"cv,0.95\nexternal,{0.95 - v:.6f}\n"
            ),
            thr, "above", eps=0.01,
            filename="pipeline_external_validation.csv",
        )

    def test_train_test_overlap_detected_boundary(self):
        # Fires when overlap share > 0.0. Boundary at 0 / 1 row out of 5.
        # Above = at least one overlap; below = zero overlaps.
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_split_overlap.csv",
                       "id,train,test\n1,1,0\n2,1,0\n3,0,1\n4,0,1\n5,1,1\n")
            self.assertTrue(self.ph._train_test_overlap_detected(p))
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_split_overlap.csv",
                       "id,train,test\n1,1,0\n2,1,0\n3,0,1\n4,0,1\n5,1,0\n")
            self.assertFalse(self.ph._train_test_overlap_detected(p))

    def test_label_noise_detected_boundary(self):
        # Fires when disagreement > label_noise_disagreement_warning (0.10).
        thr = self.T["label_noise_disagreement_warning"]
        def builder(v):
            disagree = max(int(round(100 * v)), 0)
            agree = 100 - disagree
            return ("item,annotators_agree\n"
                    + "\n".join(f"{i},0" for i in range(disagree))
                    + "\n"
                    + "\n".join(f"{i + disagree},1" for i in range(agree))
                    + "\n")
        _both_sides(
            self, self.ph._label_noise_detected, builder,
            thr, "above", eps=0.02, filename="pipeline_label_noise.csv",
        )

    def test_train_eval_prevalence_shift_boundary(self):
        # Fires when max class prevalence delta > prevalence_shift_warning (0.05).
        thr = self.T["prevalence_shift_warning"]
        def builder(v):
            # Train: A=80%, B=20%. Test: A=(80-v*100)%, B=(20+v*100)%.
            train_a, train_b = 800, 200
            test_total = 1000
            test_a = int(round(test_total * (0.80 - v)))
            test_b = test_total - test_a
            return (
                "split,class,count\n"
                f"train,A,{train_a}\ntrain,B,{train_b}\n"
                f"test,A,{test_a}\ntest,B,{test_b}\n"
            )
        _both_sides(
            self, self.ph._train_eval_prevalence_shift, builder,
            thr, "above", eps=0.02, filename="pipeline_class_counts.csv",
        )

    def test_loss_metric_divergence_boundary(self):
        # Fires when divergence_score > loss_metric_divergence_warning (0.20).
        thr = self.T["loss_metric_divergence_warning"]

        def builder(v):
            # Score formula in the detector:
            # divergence = loss_drop_ratio * (1 - min(metric_gain/0.05, 1))
            # Pick metric_gain = 0 (no progress) so divergence == loss_drop_ratio.
            loss_drop_ratio = v
            return (
                "epoch,train_loss,eval_metric\n"
                "0,1.0,0.7\n"
                f"1,{1.0 - loss_drop_ratio:.6f},0.7\n"
            )
        _both_sides(
            self, self.ph._loss_metric_divergence, builder,
            thr, "above", eps=0.02, filename="pipeline_loss_curve.csv",
        )

    def test_hyperparameter_overfit_to_validation_boundary(self):
        # Fires when (max - mean of trial scores) > hp_top_vs_mean_gap_warning (0.05).
        thr = self.T["hp_top_vs_mean_gap_warning"]
        def builder(v):
            # 5 trials at base=0.50; one trial at 0.50 + delta where
            # delta is chosen so (max - mean) ≈ v. mean = (5*0.50 + (0.50+delta))/6 = 0.5 + delta/6
            # gap = (0.5 + delta) - (0.5 + delta/6) = 5*delta/6 → delta = 6v/5
            delta = 6 * v / 5
            base = 0.5
            rows = [f"t{i},{base:.6f}" for i in range(5)]
            rows.append(f"t5,{base + delta:.6f}")
            return "trial,score\n" + "\n".join(rows) + "\n"
        _both_sides(
            self, self.ph._hyperparameter_overfit_to_validation, builder,
            thr, "above", eps=0.005,
            filename="pipeline_hyperparameter_search.csv",
        )

    def test_constant_or_dead_features_boundary(self):
        # Fires when share of dead features > dead_feature_share_warning (0.10).
        thr = self.T["dead_feature_share_warning"]
        def builder(v):
            n_dead = max(int(round(100 * v)), 0)
            n_live = 100 - n_dead
            return (
                "feature,variance\n"
                + "\n".join(f"d{i},0.0" for i in range(n_dead))
                + ("\n" if n_dead else "")
                + "\n".join(f"l{i},1.0" for i in range(n_live))
                + "\n"
            )
        _both_sides(
            self, self.ph._constant_or_dead_features, builder,
            thr, "above", eps=0.02, filename="pipeline_feature_variance.csv",
        )


# ─── physics ──────────────────────────────────────────────────────────
class PhysicsBoundaryTests(unittest.TestCase):
    """Boundary tests for the 5 detectors in the physics manifest.

    Note: ``dimensional_inconsistency`` is purely structural (string
    inequality of unit columns); it is covered in test_new_manifests.py.
    """

    def setUp(self):
        from lappato_mcb.manifests import physics as ph
        self.ph = ph
        self.T = ph.THRESHOLDS

    def test_conservation_law_violation_boundary(self):
        thr = self.T["conservation_drift_warning"]
        _both_sides(
            self, self.ph._conservation_law_violation,
            lambda v: f"step,energy\n0,1.0\n1,{1.0 + v:.10f}\n",
            thr, "above", eps=1.0e-4,
            filename="physics_conservation.csv",
        )

    def test_timestep_or_cfl_violation_boundary(self):
        thr = self.T["cfl_warning"]
        _both_sides(
            self, self.ph._timestep_or_cfl_violation,
            lambda v: f"dt,cfl\n0.1,{v:.6f}\n",
            thr, "above", eps=0.05, filename="physics_timestep.csv",
        )

    def test_boundary_condition_artifact_boundary(self):
        thr = self.T["boundary_artifact_ratio_warning"]
        _both_sides(
            self, self.ph._boundary_condition_artifact,
            lambda v: f"cell,interior_value,boundary_value\n0,1.0,{1.0 + v:.6f}\n",
            thr, "above", eps=0.02, filename="physics_boundary.csv",
        )

    def test_equilibration_insufficient_boundary(self):
        thr = self.T["equilibration_window_gap_warning"]
        def builder(v):
            # Two halves with means 1.0 and (1 + v) → relative gap = v.
            return (
                "step,observable\n"
                "0,1.0\n1,1.0\n2,1.0\n3,1.0\n"
                f"4,{1.0 + v:.6f}\n5,{1.0 + v:.6f}\n6,{1.0 + v:.6f}\n7,{1.0 + v:.6f}\n"
            )
        _both_sides(
            self, self.ph._equilibration_insufficient, builder,
            thr, "above", eps=0.02, filename="physics_equilibration.csv",
        )


# ─── chemistry ────────────────────────────────────────────────────────
class ChemistryBoundaryTests(unittest.TestCase):
    """Boundary tests for the chemistry manifest (4 numeric detectors;
    ``reaction_balance_violation`` is structural integer-equality)."""

    def setUp(self):
        from lappato_mcb.manifests import chemistry as c
        self.c = c
        self.T = c.THRESHOLDS

    def test_structure_validity_low_boundary(self):
        thr = self.T["structure_validity_warning"]
        def builder(v):
            valid = int(round(1000 * v))
            return f"set,n_total,n_valid\ngen,1000,{valid}\n"
        _both_sides(
            self, self.c._structure_validity_low, builder,
            thr, "below", eps=0.02, filename="chem_validity.csv",
        )

    def test_chemical_space_coverage_low_boundary(self):
        thr = self.T["chemical_space_coverage_warning"]
        _both_sides(
            self, self.c._chemical_space_coverage_low,
            lambda v: f"coverage\n{v:.6f}\n",
            thr, "below", eps=0.02, filename="chem_space_coverage.csv",
        )

    def test_thermodynamic_consistency_violation_boundary(self):
        thr = self.T["thermodynamic_closure_warning"]
        _both_sides(
            self, self.c._thermodynamic_consistency_violation,
            lambda v: f"cycle,closure_error\nC1,{v:.6f}\n",
            thr, "above", eps=0.05, filename="chem_thermodynamics.csv",
        )

    def test_stereochemistry_information_dropped_boundary(self):
        thr = self.T["stereo_loss_share_warning"]
        def builder(v):
            kept = max(int(round(100 * (1 - v))), 0)
            return f"n_input_stereo_centers,n_kept_stereo_centers\n100,{kept}\n"
        _both_sides(
            self, self.c._stereochemistry_information_dropped, builder,
            thr, "above", eps=0.02, filename="chem_stereo.csv",
        )


# ─── materials_science ────────────────────────────────────────────────
class MaterialsScienceBoundaryTests(unittest.TestCase):

    def setUp(self):
        from lappato_mcb.manifests import materials_science as ms
        self.ms = ms
        self.T = ms.THRESHOLDS

    def test_kpoint_unconverged_boundary(self):
        thr = self.T["kpoint_energy_diff_warning"]
        _both_sides(
            self, self.ms._kpoint_unconverged,
            lambda v: f"k_density,total_energy\n4,-100.0000\n8,{-100.0 - v:.6f}\n",
            thr, "above", eps=1e-4,
            filename="materials_kpoint_convergence.csv",
        )

    def test_basis_unconverged_boundary(self):
        thr = self.T["basis_energy_diff_warning"]
        _both_sides(
            self, self.ms._basis_unconverged,
            lambda v: f"ecut,total_energy\n400,-100.0000\n500,{-100.0 - v:.6f}\n",
            thr, "above", eps=1e-4,
            filename="materials_basis_convergence.csv",
        )

    def test_force_residuals_high_boundary(self):
        thr = self.T["force_rmse_warning"]
        _both_sides(
            self, self.ms._force_residuals_high,
            lambda v: f"structure,force_rmse\nS1,{v:.6f}\n",
            thr, "above", eps=0.005,
            filename="materials_force_residuals.csv",
        )

    def test_phase_stability_violated_boundary(self):
        # Fires when ANY phase has energy_above_hull > 0.05 eV/atom.
        thr = self.T["phase_stability_residual_warning"]
        _both_sides(
            self, self.ms._phase_stability_violated,
            lambda v: f"phase,formation_energy_per_atom\nP1,{v:.6f}\n",
            thr, "above", eps=0.005,
            filename="materials_phase_stability.csv",
        )

    def test_extrapolation_warning_boundary(self):
        thr = self.T["extrapolation_distance_warning"]
        _both_sides(
            self, self.ms._extrapolation_warning,
            lambda v: f"structure,nearest_train_distance\nS1,{v:.6f}\n",
            thr, "above", eps=0.05,
            filename="materials_extrapolation.csv",
        )


# ─── neuroscience ─────────────────────────────────────────────────────
class NeuroscienceBoundaryTests(unittest.TestCase):
    """4 numeric detectors. ``subject_split_violation`` and
    ``multiple_comparison_uncorrected`` use share-of-flagged thresholds
    (default 0.0); ``artifact_share_high`` is the same shape."""

    def setUp(self):
        from lappato_mcb.manifests import neuroscience as n
        self.n = n
        self.T = n.THRESHOLDS

    def test_motion_high_boundary(self):
        thr = self.T["framewise_displacement_warning"]
        _both_sides(
            self, self.n._motion_high,
            lambda v: f"subject,framewise_displacement\nS1,{v:.6f}\n",
            thr, "above", eps=0.05, filename="neuro_motion.csv",
        )

    def test_artifact_share_high_boundary(self):
        thr = self.T["artifact_share_warning"]
        def builder(v):
            n_flagged = max(int(round(100 * v)), 0)
            n_clean = 100 - n_flagged
            return (
                "trial,line_noise,blink,muscle\n"
                + "\n".join(f"f{i},1,0,0" for i in range(n_flagged))
                + ("\n" if n_flagged else "")
                + "\n".join(f"c{i},0,0,0" for i in range(n_clean))
                + "\n"
            )
        _both_sides(
            self, self.n._artifact_high, builder,
            thr, "above", eps=0.02, filename="neuro_artifact.csv",
        )

    def test_smoothing_below_voxel_boundary(self):
        thr = self.T["smoothing_to_voxel_ratio_warning"]
        # Fires when min(fwhm/voxel) < threshold. Use voxel=1.0 so ratio=fwhm.
        _both_sides(
            self, self.n._smoothing_below_voxel,
            lambda v: f"analysis,fwhm_mm,voxel_mm\nA,{v:.6f},1.0\n",
            thr, "below", eps=0.05, filename="neuro_smoothing.csv",
        )


# ─── epidemiology ─────────────────────────────────────────────────────
class EpidemiologyBoundaryTests(unittest.TestCase):

    def setUp(self):
        from lappato_mcb.manifests import epidemiology as ep
        self.ep = ep
        self.T = ep.THRESHOLDS

    def test_reporting_delay_high_boundary(self):
        thr = self.T["reporting_delay_days_warning"]
        _both_sides(
            self, self.ep._reporting_delay_high,
            lambda v: f"date,delay_days\n2024-01-01,{v:.6f}\n",
            thr, "above", eps=0.5, filename="epi_reporting_delay.csv",
        )

    def test_serial_correlation_high_boundary(self):
        thr = self.T["serial_acf_warning"]
        _both_sides(
            self, self.ep._serial_correlation_high,
            lambda v: f"lag,autocorr\n1,{v:.6f}\n",
            thr, "above", eps=0.02, filename="epi_serial_correlation.csv",
        )

    def test_underreporting_high_boundary(self):
        thr = self.T["underreporting_ratio_warning"]
        _both_sides(
            self, self.ep._underreporting_high,
            lambda v: f"period,reported,estimated_true\nW1,100,{int(round(100 * v))}\n",
            thr, "above", eps=0.05, filename="epi_underreporting.csv",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
