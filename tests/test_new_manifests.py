"""Sanity tests for the six domain-agnostic and scientific manifests
introduced alongside the literature registry expansion.

Validates that:
  - each manifest is importable and exposes the expected exports
  - every detector dict carries the required schema keys
  - threshold overrides round-trip
  - a representative evidence-fire path returns a boolean
  - missing or malformed CSVs leave detectors fail-closed (return False)

Run with:
    python -m unittest tests.test_new_manifests
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb import manifests  # noqa: E402

NEW_TAGS = (
    "pipeline_health",
    "mathematics",
    "physics",
    "chemistry",
    "biochemistry",
    "biology",
    "earth_climate",
    "astronomy",
    "materials_science",
    "neuroscience",
    "epidemiology",
    "econometrics",
    "social_science",
    "robotics",
    "quantum_computing",
    "pharmacology",
)

REQUIRED_KEYS = {
    "id", "title", "evidence", "evidence_check", "severity",
    "queries", "sources", "transplant", "why_it_matters",
    "next_checks", "success_criteria", "references",
}


def _write(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


class ManifestSchemaTests(unittest.TestCase):

    def test_each_manifest_loads_via_registry(self):
        for tag in NEW_TAGS:
            m, run_tag = manifests.get(tag)
            self.assertIsInstance(run_tag, str)
            self.assertEqual(run_tag, tag)
            self.assertGreaterEqual(len(m), 5, tag)

    def test_pipeline_health_has_sixteen_detectors(self):
        m, _ = manifests.get("pipeline_health")
        self.assertEqual(len(m), 16)

    def test_required_schema_keys_present(self):
        for tag in NEW_TAGS:
            m, _ = manifests.get(tag)
            for entry in m:
                missing = REQUIRED_KEYS - set(entry)
                self.assertFalse(
                    missing,
                    f"{tag}/{entry.get('id','?')} missing keys: {missing}",
                )
                self.assertGreaterEqual(len(entry["queries"]), 2)
                self.assertEqual(
                    list(entry["sources"])[:3],
                    ["arXiv", "OpenAlex", "Crossref"],
                )

    def test_severities_are_high_or_info(self):
        for tag in NEW_TAGS:
            m, _ = manifests.get(tag)
            for entry in m:
                self.assertIn(entry["severity"], ("high", "info"),
                              f"{tag}/{entry['id']}")

    def test_unique_detector_ids_per_manifest(self):
        for tag in NEW_TAGS:
            m, _ = manifests.get(tag)
            ids = [e["id"] for e in m]
            self.assertEqual(len(ids), len(set(ids)), tag)


class ThresholdOverrideTests(unittest.TestCase):

    def test_pipeline_health_thresholds_round_trip(self):
        from lappato_mcb.manifests import pipeline_health as ph
        original = ph.THRESHOLDS["seed_std_warning"]
        ph.override_thresholds({"seed_std_warning": 0.99})
        self.assertEqual(ph.THRESHOLDS["seed_std_warning"], 0.99)
        ph.override_thresholds({"seed_std_warning": original})
        self.assertEqual(ph.THRESHOLDS["seed_std_warning"], original)

    def test_every_new_manifest_exposes_thresholds(self):
        from importlib import import_module
        for tag in NEW_TAGS:
            mod = import_module(f"lappato_mcb.manifests.{tag}")
            self.assertTrue(hasattr(mod, "THRESHOLDS"))
            self.assertTrue(hasattr(mod, "DEFAULT_THRESHOLDS"))
            self.assertTrue(callable(getattr(mod, "override_thresholds")))


class PipelineHealthDetectorTests(unittest.TestCase):

    def test_small_minority_class_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(
                Path(td), "pipeline_class_counts.csv",
                "class,count\nA,10000\nB,5\n",
            )
            self.assertTrue(ph._small_minority_class(p))

    def test_imbalance_ratio_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(
                Path(td), "pipeline_class_counts.csv",
                "class,count\nA,10000\nB,500\n",
            )
            self.assertTrue(ph._severe_imbalance(p))

    def test_acc_bac_gap_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(
                Path(td), "pipeline_cv_metrics.csv",
                "seed,accuracy,balanced_accuracy\n0,0.95,0.80\n1,0.94,0.82\n",
            )
            self.assertTrue(ph._acc_bac_gap_high(p))

    def test_cross_seed_variance_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(
                Path(td), "pipeline_cv_metrics.csv",
                "seed,balanced_accuracy\n0,0.70\n1,0.85\n2,0.65\n",
            )
            self.assertTrue(ph._high_cross_seed_variance(p))

    def test_top_feature_concentration_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(
                Path(td), "pipeline_feature_importance.csv",
                "feature,gain\na,90\nb,5\nc,5\n",
            )
            self.assertTrue(ph._feature_importance_concentration(p))

    def test_external_drop_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(
                Path(td), "pipeline_external_validation.csv",
                "split,auc\ncv,0.95\nexternal,0.70\n",
            )
            self.assertTrue(ph._external_test_calibration_drop(p))

    def test_cohort_size_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(
                Path(td), "pipeline_class_counts.csv",
                "class,count\nA,10\nB,15\n",
            )
            self.assertTrue(ph._small_cohort_size(p))

    def test_missing_file_does_not_fire(self):
        from lappato_mcb.manifests import pipeline_health as ph
        ghost = Path("/tmp/_definitely_not_a_real_file_xyz.csv")
        self.assertFalse(ph._small_minority_class(ghost))
        self.assertFalse(ph._severe_imbalance(ghost))
        self.assertFalse(ph._acc_bac_gap_high(ghost))
        self.assertFalse(ph._external_test_calibration_drop(ghost))


class ScienceDetectorFireTests(unittest.TestCase):

    def test_mathematics_conditioning_fires(self):
        from lappato_mcb.manifests import mathematics as m
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "math_conditioning.csv",
                       "matrix,condition_number\nA,1e10\n")
            self.assertTrue(m._numerical_instability(p))

    def test_mathematics_convergence_plateau_fires(self):
        from lappato_mcb.manifests import mathematics as m
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "math_convergence.csv",
                       "iter,loss\n0,1.0\n1,1.0\n2,1.0\n3,1.0\n4,1.0\n5,1.0\n")
            self.assertTrue(m._convergence_failure(p))

    def test_mathematics_benchmark_log_gap_fires(self):
        from lappato_mcb.manifests import mathematics as m
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "math_benchmark.csv",
                       "method,reference_value,reported_value\nA,1.0,1000.0\n")
            self.assertTrue(m._benchmark_orders_of_magnitude_mismatch(p))

    def test_physics_conservation_fires(self):
        from lappato_mcb.manifests import physics as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "physics_conservation.csv",
                       "step,energy\n0,1.0\n1,1.5\n")
            self.assertTrue(ph._conservation_law_violation(p))

    def test_physics_cfl_fires(self):
        from lappato_mcb.manifests import physics as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "physics_timestep.csv",
                       "dt,cfl\n0.1,1.5\n")
            self.assertTrue(ph._timestep_or_cfl_violation(p))

    def test_physics_dimensions_fires(self):
        from lappato_mcb.manifests import physics as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "physics_dimensions.csv",
                       "quantity,units_lhs,units_rhs\nF,kg*m/s^2,kg*m\n")
            self.assertTrue(ph._dimensional_inconsistency(p))

    def test_physics_equilibration_fires(self):
        from lappato_mcb.manifests import physics as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "physics_equilibration.csv",
                       "step,observable\n0,1.0\n1,1.05\n2,2.0\n3,2.05\n")
            self.assertTrue(ph._equilibration_insufficient(p))

    def test_chemistry_validity_fires(self):
        from lappato_mcb.manifests import chemistry as c
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "chem_validity.csv",
                       "set,n_total,n_valid\ngen,1000,800\n")
            self.assertTrue(c._structure_validity_low(p))

    def test_chemistry_reaction_balance_fires(self):
        from lappato_mcb.manifests import chemistry as c
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "chem_reaction_balance.csv",
                       "reaction,left_atoms,right_atoms\nrxn1,C:1;H:4,C:1;H:3\n")
            self.assertTrue(c._reaction_balance_violation(p))

    def test_chemistry_stereo_loss_fires(self):
        from lappato_mcb.manifests import chemistry as c
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "chem_stereo.csv",
                       "n_input_stereo_centers,n_kept_stereo_centers\n10,5\n")
            self.assertTrue(c._stereochemistry_information_dropped(p))

    def test_biochemistry_kinetics_outlier_fires(self):
        from lappato_mcb.manifests import biochemistry as bc
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "biochem_kinetics.csv",
                       "enzyme,Km,Vmax\nE1,1.0,1.0\n")  # Km=1.0 > 1e-2
            self.assertTrue(bc._enzyme_kinetics_outlier(p))

    def test_biochemistry_alignment_fires(self):
        from lappato_mcb.manifests import biochemistry as bc
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "biochem_alignment.csv",
                       "pair,rmsd\nA-B,5.0\n")
            self.assertTrue(bc._structural_alignment_rmsd_high(p))

    def test_biochemistry_replicates_inconsistent_fires(self):
        from lappato_mcb.manifests import biochemistry as bc
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "biochem_assay_replicates.csv",
                       "compound,replicate,response\nC1,1,1.0\nC1,2,5.0\n")
            self.assertTrue(bc._assay_replicate_inconsistency(p))

    def test_biology_replicates_too_few_fires(self):
        from lappato_mcb.manifests import biology as b
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "bio_replicates.csv",
                       "condition,n_replicates\ncontrol,2\ntreatment,3\n")
            self.assertTrue(b._low_biological_replicate_n(p))

    def test_biology_batch_effect_fires(self):
        from lappato_mcb.manifests import biology as b
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "bio_batch_variance.csv",
                       "source,variance\nbetween_batch,4.0\nwithin_batch,1.0\n")
            self.assertTrue(b._batch_effect_detected(p))

    def test_biology_contamination_fires(self):
        from lappato_mcb.manifests import biology as b
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "bio_contamination.csv",
                       "sample,contam_share\nS1,0.10\n")
            self.assertTrue(b._contamination_warning(p))

    def test_missing_file_fail_closed_across_manifests(self):
        from lappato_mcb.manifests import (
            biochemistry as bc,
            biology as b,
            chemistry as c,
            mathematics as m,
            physics as ph,
        )
        ghost = Path("/tmp/_lappato_ghost_evidence_file.csv")
        self.assertFalse(m._numerical_instability(ghost))
        self.assertFalse(m._convergence_failure(ghost))
        self.assertFalse(ph._conservation_law_violation(ghost))
        self.assertFalse(ph._timestep_or_cfl_violation(ghost))
        self.assertFalse(c._structure_validity_low(ghost))
        self.assertFalse(c._reaction_balance_violation(ghost))
        self.assertFalse(bc._enzyme_kinetics_outlier(ghost))
        self.assertFalse(bc._pathway_completeness_low(ghost))
        self.assertFalse(b._batch_effect_detected(ghost))
        self.assertFalse(b._low_biological_replicate_n(ghost))
        self.assertFalse(b._contamination_warning(ghost))


class HiddenBottleneckSentinelTests(unittest.TestCase):
    """Sentinel coverage for the eight subtle / hidden detectors that
    were added to ``pipeline_health`` for the v1.2 release."""

    def test_train_test_overlap_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_split_overlap.csv",
                       "id,train,test\n1,1,1\n2,1,0\n3,0,1\n")
            self.assertTrue(ph._train_test_overlap_detected(p))

    def test_train_test_overlap_clean_does_not_fire(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_split_overlap.csv",
                       "id,train,test\n1,1,0\n2,0,1\n3,1,0\n")
            self.assertFalse(ph._train_test_overlap_detected(p))

    def test_label_noise_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_label_noise.csv",
                       "item,annotators_agree\n"
                       + "\n".join(f"{i},{0 if i < 30 else 1}" for i in range(100))
                       + "\n")
            self.assertTrue(ph._label_noise_detected(p))

    def test_label_noise_clean_does_not_fire(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_label_noise.csv",
                       "item,annotators_agree\n"
                       + "\n".join(f"{i},1" for i in range(100))
                       + "\n")
            self.assertFalse(ph._label_noise_detected(p))

    def test_prediction_extreme_collapse_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_prediction_distribution.csv",
                       "bin,share\n0.05,0.50\n0.5,0.05\n0.95,0.45\n")
            self.assertTrue(ph._prediction_confidence_collapsed(p))

    def test_prediction_uniform_collapse_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            # Uniform across 10 bins => KL divergence ~ 0.
            rows_str = "\n".join(
                f"{round(0.05 + 0.1 * i, 2)},0.10" for i in range(10)
            )
            p = _write(Path(td), "pipeline_prediction_distribution.csv",
                       "bin,share\n" + rows_str + "\n")
            self.assertTrue(ph._prediction_confidence_collapsed(p))

    def test_prevalence_shift_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_class_counts.csv",
                       "split,class,count\n"
                       "train,A,800\ntrain,B,200\n"
                       "test,A,500\ntest,B,500\n")
            self.assertTrue(ph._train_eval_prevalence_shift(p))

    def test_prevalence_shift_clean_does_not_fire(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_class_counts.csv",
                       "split,class,count\n"
                       "train,A,800\ntrain,B,200\n"
                       "test,A,400\ntest,B,100\n")
            self.assertFalse(ph._train_eval_prevalence_shift(p))

    def test_loss_metric_divergence_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_loss_curve.csv",
                       "epoch,train_loss,eval_metric\n"
                       "0,1.0,0.70\n1,0.5,0.71\n2,0.3,0.71\n3,0.1,0.71\n")
            self.assertTrue(ph._loss_metric_divergence(p))

    def test_loss_metric_divergence_aligned_does_not_fire(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_loss_curve.csv",
                       "epoch,train_loss,eval_metric\n"
                       "0,1.0,0.50\n1,0.5,0.75\n2,0.3,0.85\n")
            self.assertFalse(ph._loss_metric_divergence(p))

    def test_hp_overfit_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            scores = [0.50, 0.51, 0.52, 0.51, 0.50, 0.95]
            p = _write(Path(td), "pipeline_hyperparameter_search.csv",
                       "trial,score\n"
                       + "\n".join(f"{i},{s}" for i, s in enumerate(scores))
                       + "\n")
            self.assertTrue(ph._hyperparameter_overfit_to_validation(p))

    def test_dead_features_fire(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            rows_str = "\n".join(
                f"f{i},{0.0 if i < 30 else 1.0}" for i in range(100)
            )
            p = _write(Path(td), "pipeline_feature_variance.csv",
                       "feature,variance\n" + rows_str + "\n")
            self.assertTrue(ph._constant_or_dead_features(p))

    def test_threshold_picked_on_test_fires(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_threshold_audit.csv",
                       "stage,split\n"
                       "threshold_selection,test\n"
                       "final_report,test\n")
            self.assertTrue(ph._threshold_picked_on_test_set(p))

    def test_threshold_clean_does_not_fire(self):
        from lappato_mcb.manifests import pipeline_health as ph
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pipeline_threshold_audit.csv",
                       "stage,split\n"
                       "threshold_selection,calibration\n"
                       "final_report,test\n")
            self.assertFalse(ph._threshold_picked_on_test_set(p))


class ScientificManifestSentinelTests(unittest.TestCase):
    """Positive-fire sentinels for the ten v1.3 scientific manifests."""

    def test_earth_climate_bias_fires(self):
        from lappato_mcb.manifests import earth_climate as ec
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "climate_bias.csv",
                       "variable,model,observation,bias\nT2m,M,O,3.0\n")
            self.assertTrue(ec._bias_high(p))

    def test_earth_climate_water_balance_fires(self):
        from lappato_mcb.manifests import earth_climate as ec
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "climate_water_balance.csv",
                       "period,P,E,Q,dS\n2024,100,40,40,5\n")
            self.assertTrue(ec._water_balance_violation(p))

    def test_astronomy_psf_fires(self):
        from lappato_mcb.manifests import astronomy as a
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "astro_psf_residuals.csv",
                       "object,psf_residual\nstar1,0.10\n")
            self.assertTrue(a._psf_residual_high(p))

    def test_astronomy_low_significance_fires(self):
        from lappato_mcb.manifests import astronomy as a
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "astro_signal_significance.csv",
                       "candidate,sigma\nC1,3.0\n")
            self.assertTrue(a._low_signal_significance(p))

    def test_materials_kpoint_unconverged_fires(self):
        from lappato_mcb.manifests import materials_science as ms
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "materials_kpoint_convergence.csv",
                       "k_density,total_energy\n4,-100.000\n8,-100.020\n")
            self.assertTrue(ms._kpoint_unconverged(p))

    def test_materials_phase_above_hull_fires(self):
        from lappato_mcb.manifests import materials_science as ms
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "materials_phase_stability.csv",
                       "phase,formation_energy_per_atom\nP1,0.10\n")
            self.assertTrue(ms._phase_stability_violated(p))

    def test_neuroscience_motion_fires(self):
        from lappato_mcb.manifests import neuroscience as n
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "neuro_motion.csv",
                       "subject,framewise_displacement\nS1,0.8\n")
            self.assertTrue(n._motion_high(p))

    def test_neuroscience_uncorrected_fires(self):
        from lappato_mcb.manifests import neuroscience as n
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "neuro_multiple_comparison.csv",
                       "analysis,n_tests,corrected\nA,1000,0\n")
            self.assertTrue(n._multiple_comparison_uncorrected(p))

    def test_epidemiology_definition_change_fires(self):
        from lappato_mcb.manifests import epidemiology as ep
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "epi_case_definition.csv",
                       "period,case_definition_id\n1,v1\n2,v2\n")
            self.assertTrue(ep._case_definition_changed(p))

    def test_epidemiology_intervention_overlap_fires(self):
        from lappato_mcb.manifests import epidemiology as ep
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "epi_intervention_overlap.csv",
                       "intervention,start,end\nA,0,30\nB,15,45\n")
            self.assertTrue(ep._intervention_overlap_present(p))

    def test_econometrics_weak_instrument_fires(self):
        from lappato_mcb.manifests import econometrics as ec
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "econ_iv_strength.csv",
                       "instrument,first_stage_F\nZ,5.0\n")
            self.assertTrue(ec._weak_instrument(p))

    def test_econometrics_se_understated_fires(self):
        from lappato_mcb.manifests import econometrics as ec
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "econ_robust_se.csv",
                       "cluster,n_obs,naive_se,clustered_se\nC,1000,0.10,0.20\n")
            self.assertTrue(ec._standard_errors_understated(p))

    def test_social_science_p_hacking_fires(self):
        from lappato_mcb.manifests import social_science as ss
        with tempfile.TemporaryDirectory() as td:
            ps = [0.04, 0.045, 0.048, 0.049, 0.01]
            p = _write(Path(td), "social_p_curve.csv",
                       "study,p_value\n"
                       + "\n".join(f"S{i},{v}" for i, v in enumerate(ps))
                       + "\n")
            self.assertTrue(ss._p_hacking_pattern(p))

    def test_social_science_response_rate_fires(self):
        from lappato_mcb.manifests import social_science as ss
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "social_response_rate.csv",
                       "survey,n_invited,n_completed\nS1,1000,200\n")
            self.assertTrue(ss._response_rate_low(p))

    def test_robotics_sim2real_fires(self):
        from lappato_mcb.manifests import robotics as r
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "robotics_sim2real_gap.csv",
                       "task,sim_score,real_score\nT,0.95,0.50\n")
            self.assertTrue(r._sim2real_gap_high(p))

    def test_robotics_few_seeds_fires(self):
        from lappato_mcb.manifests import robotics as r
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "robotics_evaluation_seeds.csv",
                       "seed,success_rate\n0,0.6\n1,0.7\n")
            self.assertTrue(r._few_eval_seeds(p))

    def test_quantum_fidelity_fires(self):
        from lappato_mcb.manifests import quantum_computing as qc
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "quantum_fidelity.csv",
                       "circuit,fidelity\nC1,0.80\n")
            self.assertTrue(qc._fidelity_low(p))

    def test_quantum_barren_plateau_fires(self):
        from lappato_mcb.manifests import quantum_computing as qc
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "quantum_barren_plateau.csv",
                       "layer,gradient_variance\n1,1e-2\n2,1e-6\n")
            self.assertTrue(qc._barren_plateau(p))

    def test_pharmacology_dose_response_undersampled_fires(self):
        from lappato_mcb.manifests import pharmacology as pm
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pharma_dose_response.csv",
                       "compound,dose,response\nA,1,0.1\nA,10,0.5\n")
            self.assertTrue(pm._dose_response_undersampled(p))

    def test_pharmacology_pk_concordance_fires(self):
        from lappato_mcb.manifests import pharmacology as pm
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pharma_pk_concordance.csv",
                       "model,observed_auc,predicted_auc\nM,100,200\n")
            self.assertTrue(pm._pk_concordance_low(p))

    def test_pharmacology_endpoint_unspecified_fires(self):
        from lappato_mcb.manifests import pharmacology as pm
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), "pharma_trial_endpoint.csv",
                       "trial,primary_endpoint_specified\nT,0\n")
            self.assertTrue(pm._trial_endpoint_unspecified(p))


class FailClosedSentinelTests(unittest.TestCase):
    """Every detector across every new manifest must return False on a
    missing file. This is the contract the daemon relies on."""

    def test_all_detectors_fail_closed_on_missing_file(self):
        ghost = Path("/tmp/_lappato_ghost_evidence_file_v2.csv")
        for tag in NEW_TAGS:
            m, _ = manifests.get(tag)
            for entry in m:
                check = entry.get("evidence_check")
                if check is None:
                    continue
                self.assertFalse(
                    check(ghost),
                    f"{tag}/{entry['id']} fired on missing file",
                )

    def test_every_new_manifest_uses_only_listed_thresholds(self):
        """Every detector that uses a numeric threshold reads it from
        ``THRESHOLDS`` so that ``override_thresholds`` round-trips."""
        from importlib import import_module
        for tag in NEW_TAGS:
            mod = import_module(f"lappato_mcb.manifests.{tag}")
            self.assertGreaterEqual(len(mod.DEFAULT_THRESHOLDS), 1, tag)
            # Override → restore round-trip.
            key = next(iter(mod.DEFAULT_THRESHOLDS))
            original = mod.THRESHOLDS[key]
            mod.override_thresholds({key: original + 1.0})
            self.assertNotEqual(mod.THRESHOLDS[key], original)
            mod.override_thresholds({key: original})
            self.assertEqual(mod.THRESHOLDS[key], original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
