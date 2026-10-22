"""``validate_evidence``."""
from __future__ import annotations

import unittest

from .support import ServerTestCase, make_wdbc_project, require_mcp_sdk, write_csv

require_mcp_sdk()


class ValidateEvidenceToolTests(ServerTestCase):

    def test_complete_file_passes_and_missing_files_get_a_snippet(self):
        make_wdbc_project(self.root)
        out = self.ok("validate_evidence", domain="wdbc", project_root="project")
        self.assertTrue(out["ok"])
        files = {f["filename"]: f for f in out["files"]}
        self.assertTrue(files["wdbc_per_fold.csv"]["present"])
        self.assertEqual(files["wdbc_per_fold.csv"]["suggested_snippet"], "")
        self.assertFalse(files["wdbc_calibration.csv"]["present"])
        self.assertIn("csv.DictWriter", files["wdbc_calibration.csv"]["suggested_snippet"])

    def test_missing_required_column_fails(self):
        write_csv(self.root / "checkpoints" / "wdbc_calibration.csv", ["fold"], [[0]])
        out = self.ok("validate_evidence", domain="wdbc")
        self.assertFalse(out["ok"])
        files = {f["filename"]: f for f in out["files"]}
        self.assertEqual(files["wdbc_calibration.csv"]["missing_required"], ["brier"])
        self.assertIn("[FAIL]", out["report_text"])

    def test_project_root_outside_the_sandbox_is_rejected(self):
        self.assertIn("outside the allowed root", self.error(
            "validate_evidence", domain="wdbc", project_root="..",
        ))


if __name__ == "__main__":
    unittest.main()
