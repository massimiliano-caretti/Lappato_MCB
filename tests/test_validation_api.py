"""Tests for the importable evidence validator (``lappato_mcb.validation``).

The behaviour of the checks themselves is covered black-box by
``test_gold_sets.py`` through the ``examples/`` front-end; these tests
pin the library-level API that the front-end and the MCP server share.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lappato_mcb import manifests
from lappato_mcb.validation import list_schema_domains, load_schema, validate_checkpoints


class ValidationApiTests(unittest.TestCase):

    def test_every_registered_domain_ships_a_schema(self):
        self.assertEqual(set(list_schema_domains()), set(manifests.REGISTRY))

    def test_unknown_domain_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_schema("no_such_domain")

    def test_missing_required_column_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "wdbc_calibration.csv").write_text("fold,other\n0,1\n", encoding="utf-8")
            report = validate_checkpoints("wdbc", Path(td))
        by_name = {r["filename"]: r for r in report["file_results"]}
        self.assertTrue(by_name["wdbc_calibration.csv"]["present"])
        self.assertEqual(by_name["wdbc_calibration.csv"]["missing_required"], ["brier"])
        self.assertFalse(by_name["wdbc_per_fold.csv"]["present"])


if __name__ == "__main__":
    unittest.main()
