"""Tests for the stdlib-only trigram reranker and the Phase 0.5
micro-benchmark.

Two layers:

1. **TrigramJaccardReranker contract.** The reranker must satisfy the
   ``Reranker`` Protocol, return scores in ``[0, 1]``, never raise on
   well-formed input, and produce identical scores across runs (the
   determinism contract that makes it usable inside LAPPATO_MCB at
   all).

2. **Phase 0.5 benchmark regression.** Pin the headline numbers from
   ``examples/benchmark_reranker_stdlib.py`` so silent drifts in
   gold-set keywords, fingerprint code, or fixture synthesis surface
   here. The pinned numbers reflect the v1.4 measurement and are the
   evidence on which the Phase 1/2 decision is being made.

The benchmark numbers below come from the deterministic seed=42 run
documented in ``docs/reranker_benchmark_phase05.md``. If the report
is updated, update the assertions here in lockstep.

Run with::

    python -m unittest tests.test_trigram_reranker
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lappato_mcb.core import Reranker  # noqa: E402
from lappato_mcb.rerankers.trigram import TrigramJaccardReranker  # noqa: E402


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "_bench_phase05",
        ROOT / "examples" / "benchmark_reranker_stdlib.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ─── Reranker contract ─────────────────────────────────────────────────
class TrigramRerankerContractTests(unittest.TestCase):

    def setUp(self):
        self.r = TrigramJaccardReranker()

    def test_satisfies_reranker_protocol(self):
        self.assertTrue(isinstance(self.r, Reranker))

    def test_audit_attributes_present(self):
        self.assertEqual(self.r.model_name, "trigram-jaccard")
        self.assertEqual(self.r.model_version, "1.0")
        self.assertTrue(self.r.model_sha)

    def test_empty_inputs_return_zero_vector(self):
        self.assertEqual(self.r.score("", []), [])
        self.assertEqual(self.r.score("", [{"title": "x"}]), [0.0])
        self.assertEqual(self.r.score("anything", []), [])

    def test_scores_lie_in_unit_interval(self):
        hits = [
            {"title": "Isotonic regression for probability calibration"},
            {"title": "Bayesian model averaging unrelated topic"},
            {"title": ""},
        ]
        for s in self.r.score("isotonic calibration", hits):
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_morphological_variant_gets_partial_overlap(self):
        # 'calibration' / 'calibrating' share the trigrams 'cal',
        # 'ali', 'lib', 'ibr', 'bra', 'rat'. The score must therefore
        # be strictly positive (and below 1.0).
        scores = self.r.score(
            "calibration",
            [{"title": "calibrating deep classifiers"}],
        )
        self.assertGreater(scores[0], 0.0)
        self.assertLess(scores[0], 1.0)

    def test_unrelated_text_scores_zero(self):
        scores = self.r.score(
            "isotonic calibration",
            [{"title": "qqq xyz wzy zzzz"}],  # no shared trigrams
        )
        self.assertEqual(scores[0], 0.0)

    def test_determinism_across_runs(self):
        hits = [
            {"title": "Isotonic regression"},
            {"title": "Random text content"},
            {"title": "Calibrating deep classifiers"},
        ]
        a = self.r.score("isotonic calibration", hits)
        b = self.r.score("isotonic calibration", hits)
        c = self.r.score("isotonic calibration", hits)
        self.assertEqual(a, b)
        self.assertEqual(b, c)


# ─── Phase 0.5 benchmark regression ───────────────────────────────────
class BenchmarkRegressionTests(unittest.TestCase):
    """Pins the headline numbers from the v1.4 micro-benchmark.

    The benchmark is deterministic (seed=42), so any drift here means
    something measurable changed in: gold-set keywords, fingerprint
    trigrams, fixture synthesis, or scoring functions. Update the
    pins together with ``docs/reranker_benchmark_phase05.md`` if the
    drift is intentional.
    """

    @classmethod
    def setUpClass(cls):
        cls.bench = _load_benchmark_module()
        cls.report = cls.bench.run_benchmark()

    def _macro(self, scenario: str, mode: str, k: int) -> float:
        n = len(self.report["domains"])
        total = 0.0
        for d, scen in self.report["domains"].items():
            total += scen[scenario]["summary"][f"macro_recall_at_{k}_{mode}"]
        return total / n

    def test_easy_scenario_recall_is_one_for_every_mode(self):
        # All modes must hit 100% on the easy scenario; if not, the
        # reranker / blend is hurting the obvious cases.
        for mode in ("lex", "additive", "rrf", "rrf_2x", "rerank_only"):
            self.assertAlmostEqual(
                self._macro("easy", mode, 5), 1.0, places=4,
                msg=f"easy / {mode} / R@5 dropped below 1.0",
            )
            self.assertAlmostEqual(
                self._macro("easy", mode, 10), 1.0, places=4,
                msg=f"easy / {mode} / R@10 dropped below 1.0",
            )

    def test_hard_lexical_baseline_is_around_half(self):
        # Confirms the hard fixtures are actually hard for token-overlap.
        r5 = self._macro("hard", "lex", 5)
        self.assertGreater(r5, 0.40, f"hard/lex R@5 = {r5}")
        self.assertLess(r5, 0.60, f"hard/lex R@5 = {r5}")

    def test_additive_blend_does_not_help_on_hard_scenario(self):
        # The current core blend (additive + clip-floor) is dominated
        # by the integer lexical score on the hard fixtures and
        # delivers no measurable uplift. This is the negative
        # finding that motivates Phase 0.5: if it ever changes,
        # update the report.
        delta = self._macro("hard", "additive", 5) - self._macro("hard", "lex", 5)
        self.assertAlmostEqual(delta, 0.0, places=4)

    def test_rrf_blend_delivers_meaningful_uplift_on_hard_scenario(self):
        # The headline number from the benchmark: RRF (lex + 2x
        # rerank) lifts hard-scenario R@5 by at least +0.30. Below
        # this, either the gold-set keyword pool drifted or the
        # trigram reranker degraded.
        delta = self._macro("hard", "rrf_2x", 5) - self._macro("hard", "lex", 5)
        self.assertGreaterEqual(
            delta, 0.30,
            f"RRF/2x R@5 uplift dropped to {delta:+.4f} (expected >= +0.30)",
        )

    def test_balanced_rrf_also_improves_hard_scenario(self):
        delta = self._macro("hard", "rrf", 5) - self._macro("hard", "lex", 5)
        self.assertGreaterEqual(
            delta, 0.20,
            f"RRF/balanced R@5 uplift dropped to {delta:+.4f} (expected >= +0.20)",
        )

    def test_rerank_only_is_the_theoretical_ceiling(self):
        # Reranker-only ranking, evaluated as if no lexical signal
        # existed: serves as the upper bound the blend tries to
        # approach. We only require it to outperform the lexical
        # baseline by a sizable margin.
        delta = (
            self._macro("hard", "rerank_only", 5)
            - self._macro("hard", "lex", 5)
        )
        self.assertGreaterEqual(
            delta, 0.30,
            f"reranker-only R@5 uplift dropped to {delta:+.4f}",
        )

    def test_at_high_k_lexical_already_recovers(self):
        # At K=10 the lexical baseline already pulls most relevant
        # papers in, so the reranker uplift compresses. This is a
        # consistency check: the gap between baseline and best mode
        # at K=10 is much smaller than at K=5.
        gap5 = (
            self._macro("hard", "rrf_2x", 5)
            - self._macro("hard", "lex", 5)
        )
        gap10 = (
            self._macro("hard", "rrf_2x", 10)
            - self._macro("hard", "lex", 10)
        )
        self.assertGreater(gap5, gap10)

    def test_benchmark_run_is_deterministic(self):
        # Two independent runs of the harness must produce identical
        # macro numbers (same seed, same fixtures, same scoring).
        report_b = self.bench.run_benchmark()
        for d in self.report["domains"]:
            for scenario in ("easy", "hard"):
                a = self.report["domains"][d][scenario]["summary"]
                b = report_b["domains"][d][scenario]["summary"]
                self.assertEqual(a, b, f"non-deterministic on {d}/{scenario}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
