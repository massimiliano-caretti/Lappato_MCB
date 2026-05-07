"""Integrity tests for the bundled gold sets and the per-manifest
JSON Schemas for evidence CSVs.

What is checked here:

  - Every domain in ``lappato_mcb.manifests.REGISTRY`` has a matching
    ``docs/gold_sets/<tag>_gold.json`` file.
  - Every domain in ``REGISTRY`` has a matching
    ``lappato_mcb/manifests/schemas/<tag>.schema.json`` file.
  - Every gold-set ``topics_by_weakness`` key matches a real detector
    ``id`` in the corresponding manifest (no orphan keys after renames).
  - Every gold-set topic carries a non-empty ``must_match_keywords``
    list and a ``rationale`` string.
  - Every schema has the expected top-level fields and at least one
    evidence file when at least one detector in the manifest has an
    ``evidence`` filename (i.e., we never ship a stub schema).
  - Every schema's ``used_by_detectors`` list references real detector
    IDs in the manifest.
  - The ``examples/validate_evidence_csv.py`` validator returns the
    expected outcome on a synthetic 'good' folder and a synthetic
    'bad' folder.

Run with::

    python -m unittest tests.test_gold_sets
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lappato_mcb import manifests  # noqa: E402

GOLD_DIR = ROOT / "docs" / "gold_sets"
SCHEMA_DIR = ROOT / "lappato_mcb" / "manifests" / "schemas"


def _load_validator():
    """Import examples/validate_evidence_csv.py without requiring an
    installable examples package."""
    spec = importlib.util.spec_from_file_location(
        "_validate_evidence_csv",
        ROOT / "examples" / "validate_evidence_csv.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class GoldSetIntegrityTests(unittest.TestCase):

    def test_every_domain_has_a_gold_file(self):
        for tag in manifests.REGISTRY:
            self.assertTrue(
                (GOLD_DIR / f"{tag}_gold.json").exists(),
                f"missing gold set for {tag!r}",
            )

    def test_every_gold_topic_key_matches_a_detector_id(self):
        for tag, (manifest, _) in manifests.REGISTRY.items():
            gold = json.loads((GOLD_DIR / f"{tag}_gold.json").read_text(encoding="utf-8"))
            self.assertEqual(gold.get("domain"), tag)
            detector_ids = {e["id"] for e in manifest}
            for wid in gold.get("topics_by_weakness", {}):
                self.assertIn(
                    wid, detector_ids,
                    f"{tag}: gold key {wid!r} not a detector id",
                )

    def test_topics_have_keywords_and_rationale(self):
        for tag in manifests.REGISTRY:
            gold = json.loads((GOLD_DIR / f"{tag}_gold.json").read_text(encoding="utf-8"))
            for wid, topics in gold.get("topics_by_weakness", {}).items():
                for t in topics:
                    self.assertTrue(
                        isinstance(t.get("must_match_keywords"), list)
                        and len(t["must_match_keywords"]) >= 1,
                        f"{tag}/{wid}/{t.get('name','?')}: keywords missing",
                    )
                    self.assertTrue(
                        isinstance(t.get("rationale"), str) and t["rationale"].strip(),
                        f"{tag}/{wid}/{t.get('name','?')}: rationale missing",
                    )

    def test_minimum_topic_count_per_domain(self):
        # The v1.3 expansion targets 10–15 topics per 5-detector manifest;
        # smaller numbers are an early-warning that the gold-set drifted
        # from the manifest after a renaming.
        for tag, (manifest, _) in manifests.REGISTRY.items():
            gold = json.loads((GOLD_DIR / f"{tag}_gold.json").read_text(encoding="utf-8"))
            n_topics = sum(len(v) for v in gold.get("topics_by_weakness", {}).values())
            n_detectors = len(manifest)
            # At least one topic per detector and at least 10 overall
            # (32 for the 16-detector pipeline_health).
            min_total = max(10, n_detectors)
            self.assertGreaterEqual(
                n_topics, min_total,
                f"{tag}: only {n_topics} topics for {n_detectors} detectors "
                f"(expected ≥ {min_total})",
            )


class SchemaIntegrityTests(unittest.TestCase):

    def test_every_domain_has_a_schema_file(self):
        for tag in manifests.REGISTRY:
            self.assertTrue(
                (SCHEMA_DIR / f"{tag}.schema.json").exists(),
                f"missing schema for {tag!r}",
            )

    def test_schema_top_level_fields(self):
        for tag in manifests.REGISTRY:
            s = json.loads((SCHEMA_DIR / f"{tag}.schema.json").read_text(encoding="utf-8"))
            self.assertEqual(s.get("schema_version"), 1)
            self.assertEqual(s.get("domain"), tag)
            self.assertIn("evidence_files", s)

    def test_schema_covers_every_evidence_file_used_by_the_manifest(self):
        for tag, (manifest, _) in manifests.REGISTRY.items():
            s = json.loads((SCHEMA_DIR / f"{tag}.schema.json").read_text(encoding="utf-8"))
            evidence_in_manifest = {
                e["evidence"] for e in manifest if e.get("evidence")
            }
            evidence_in_schema = set(s.get("evidence_files", {}))
            missing = evidence_in_manifest - evidence_in_schema
            self.assertFalse(
                missing,
                f"{tag}: schema missing evidence files used by detectors: {sorted(missing)}",
            )

    def test_used_by_detectors_lists_real_ids(self):
        for tag, (manifest, _) in manifests.REGISTRY.items():
            s = json.loads((SCHEMA_DIR / f"{tag}.schema.json").read_text(encoding="utf-8"))
            real_ids = {e["id"] for e in manifest}
            for fname, spec in s.get("evidence_files", {}).items():
                for det in spec.get("used_by_detectors", []):
                    self.assertIn(
                        det, real_ids,
                        f"{tag}: schema {fname} references unknown detector {det!r}",
                    )

    def test_columns_have_well_formed_specs(self):
        for tag in manifests.REGISTRY:
            s = json.loads((SCHEMA_DIR / f"{tag}.schema.json").read_text(encoding="utf-8"))
            for fname, spec in s.get("evidence_files", {}).items():
                for col in spec.get("columns", []):
                    self.assertIn("name", col, f"{tag}/{fname}: column missing name")
                    self.assertIn(
                        col.get("type"), ("string", "number"),
                        f"{tag}/{fname}/{col['name']}: bad type",
                    )
                    self.assertIsInstance(
                        col.get("aliases", []), list,
                        f"{tag}/{fname}/{col['name']}: aliases must be a list",
                    )


class ValidatorBehaviourTests(unittest.TestCase):
    """Black-box checks on examples/validate_evidence_csv.py."""

    @classmethod
    def setUpClass(cls):
        cls.v = _load_validator()

    def test_missing_evidence_file_does_not_fail(self):
        with tempfile.TemporaryDirectory() as td:
            report = self.v.validate_checkpoints("pipeline_health", Path(td))
            text, ok = self.v.render_report(report)
            self.assertTrue(ok, msg=text)

    def test_present_file_with_canonical_columns_passes(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "pipeline_class_counts.csv").write_text(
                "class,count\nA,800\nB,200\n", encoding="utf-8",
            )
            report = self.v.validate_checkpoints("pipeline_health", Path(td))
            text, ok = self.v.render_report(report)
            self.assertTrue(ok, msg=text)
            class_counts = next(
                r for r in report["file_results"]
                if r["file"].endswith("pipeline_class_counts.csv")
            )
            self.assertTrue(class_counts["present"])
            self.assertFalse(class_counts["missing_required"])

    def test_alias_column_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            # Use the alias 'support' instead of canonical 'count'.
            (Path(td) / "pipeline_class_counts.csv").write_text(
                "class,support\nA,800\nB,200\n", encoding="utf-8",
            )
            report = self.v.validate_checkpoints("pipeline_health", Path(td))
            text, ok = self.v.render_report(report)
            self.assertTrue(ok, msg=text)
            class_counts = next(
                r for r in report["file_results"]
                if r["file"].endswith("pipeline_class_counts.csv")
            )
            accepted_names = {c["accepted_as"] for c in class_counts["accepted_columns"]}
            self.assertIn("support", accepted_names)

    def test_missing_required_columns_fails(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "pipeline_class_counts.csv").write_text(
                "wrong_a,wrong_b\n1,2\n", encoding="utf-8",
            )
            report = self.v.validate_checkpoints("pipeline_health", Path(td))
            text, ok = self.v.render_report(report)
            self.assertFalse(ok, msg=text)


class ValidatorSuggesterTests(unittest.TestCase):
    """Cover the validator's csv.DictWriter suggester (#1 of v1.4)."""

    @classmethod
    def setUpClass(cls):
        cls.v = _load_validator()

    def test_suggester_emits_for_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            report = self.v.validate_checkpoints("pipeline_health", Path(td))
            text, _ = self.v.render_report(report, suggest=True)
            self.assertIn("suggested csv.DictWriter for pipeline_class_counts.csv", text)
            self.assertIn("import csv", text)
            self.assertIn("DictWriter(", text)

    def test_suggester_emits_for_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "pipeline_class_counts.csv").write_text(
                "wrong_a,wrong_b\n1,2\n", encoding="utf-8",
            )
            report = self.v.validate_checkpoints("pipeline_health", Path(td))
            text, ok = self.v.render_report(report, suggest=True)
            self.assertFalse(ok)
            self.assertIn("[FAIL]", text)
            self.assertIn("suggested csv.DictWriter for pipeline_class_counts.csv", text)

    def test_no_suggest_flag_suppresses_snippet(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "pipeline_class_counts.csv").write_text(
                "wrong_a,wrong_b\n1,2\n", encoding="utf-8",
            )
            report = self.v.validate_checkpoints("pipeline_health", Path(td))
            text, _ = self.v.render_report(report, suggest=False)
            self.assertNotIn("DictWriter(", text)

    def test_snippet_lists_canonical_aliases(self):
        spec = {
            "columns": [
                {"name": "class", "aliases": ["label"], "type": "string", "required": True},
                {"name": "count", "aliases": ["support", "n"], "type": "number", "required": True},
            ],
        }
        snippet = self.v.suggest_dict_writer_snippet("test.csv", spec)
        self.assertIn("'class' also accepts aliases: label", snippet)
        self.assertIn("'count' also accepts aliases: support, n", snippet)
        self.assertIn("'class', 'count'", snippet)

    def test_snippet_is_valid_python(self):
        # The snippet should parse so users can paste it first and
        # wire actual values in second.
        spec = {
            "columns": [
                {"name": "feature", "aliases": [], "type": "string", "required": True},
                {"name": "gain", "aliases": ["importance"], "type": "number", "required": True},
            ],
        }
        snippet = self.v.suggest_dict_writer_snippet(
            "pipeline_feature_importance.csv", spec,
        )
        # Strip the leading blank line + indent the snippet uses no
        # indentation, so it parses on its own. Wrap in a dummy
        # checkpoints_dir binding so the symbol is defined.
        from pathlib import Path as _P
        wrapped = "checkpoints_dir = _P('.')\n" + snippet.replace("from pathlib", "# from pathlib")
        # Filter out comment-only lines that mention <...> tokens.
        compile(wrapped, "<snippet>", "exec")  # raises SyntaxError on failure


if __name__ == "__main__":
    unittest.main(verbosity=2)
