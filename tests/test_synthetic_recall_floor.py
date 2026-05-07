"""Regression test for the synthetic-corpus recall harness.

Every domain's gold set must round-trip at 100% recall when fed a
corpus synthesised from its own ``must_match_keywords``. Drop below
and the offending topic name is named in the failure message — the
common cause is a typo, an empty keyword list, or a topic whose
keywords accidentally collide with a stopword filter.

This is a self-consistency check on the gold sets, not a measure
of real-world retrieval quality. See
``examples/measure_recall_synthetic.py`` for the full rationale.

Run with::

    python -m unittest tests.test_synthetic_recall_floor
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


def _load_synth():
    spec = importlib.util.spec_from_file_location(
        "_measure_recall_synthetic",
        ROOT / "examples" / "measure_recall_synthetic.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class SyntheticRecallFloorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.synth = _load_synth()
        cls.domains = cls.synth._list_known_domains()

    def test_every_known_domain_is_evaluable(self):
        # Catches missing / mis-renamed gold-set files.
        self.assertGreaterEqual(
            len(self.domains), 35,
            "Expected at least 35 gold-set domains on disk.",
        )

    def test_every_domain_round_trips_at_one_hundred_percent(self):
        # The corpus is generated from each topic's own
        # must_match_keywords, so any missing topic indicates a
        # malformed entry in the gold set.
        for domain in self.domains:
            with self.subTest(domain=domain):
                result = self.synth.evaluate_synthetic(domain)
                missing = [
                    (w["weakness"], m)
                    for w in result["per_weakness"]
                    if w["active"]
                    for m in w["missing"]
                ]
                self.assertGreaterEqual(
                    result["macro_recall"], 1.0 - 1e-9,
                    f"{domain}: synthetic macro recall = "
                    f"{result['macro_recall']:.4f} (expected 1.0). "
                    f"Missing topics: {missing}",
                )

    def test_every_active_weakness_has_at_least_one_topic(self):
        # Active weakness with zero matched topics ⇒ either the
        # synthesizer is broken or every topic has empty keywords.
        for domain in self.domains:
            result = self.synth.evaluate_synthetic(domain)
            for w in result["per_weakness"]:
                if w["active"]:
                    self.assertGreater(
                        w["n_found"], 0,
                        f"{domain}/{w['weakness']}: zero topics matched "
                        f"({w['n_topics']} expected)",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
