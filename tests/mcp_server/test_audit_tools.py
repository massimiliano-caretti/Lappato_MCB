"""``run_audit`` / ``get_weakness_cards`` (offline: no network)."""
from __future__ import annotations

import unittest

from lappato_mcb import RunArtefacts

from .support import ServerTestCase, make_wdbc_project, require_mcp_sdk, write_corpus

require_mcp_sdk()


class AuditToolTests(ServerTestCase):

    def setUp(self):
        super().setUp()
        self.project = make_wdbc_project(self.root)
        write_corpus(self.project, [{
            "source": "arXiv", "id": "2401.00001", "year": "2024",
            "title": "Focal loss reweighting for class imbalance in medical classification",
            "abstract": "class imbalance binary classification calibration",
        }])

    def test_run_audit_returns_the_cards_of_this_cycle_only(self):
        first = self.ok("run_audit", domain="wdbc", project_root="project")
        self.assertTrue(first["offline"])
        ids = {c["weakness"] for c in first["cards"]}
        self.assertIn("class_imbalance", ids)
        self.assertEqual(first["n_active_weaknesses"], len(first["cards"]))
        imbalance = next(c for c in first["cards"] if c["weakness"] == "class_imbalance")
        self.assertEqual(imbalance["top_papers"][0]["id"], "2401.00001")

        second = self.ok("run_audit", domain="wdbc", project_root="project")
        self.assertEqual(len(second["cards"]), len(first["cards"]))

    def test_get_weakness_cards_reads_back_the_latest_audit(self):
        empty = self.ok("get_weakness_cards", domain="wdbc", project_root="project")
        self.assertEqual(empty["cards"], [])
        self.assertIsNone(empty["updated_at"])

        self.ok("run_audit", domain="wdbc", project_root="project")
        latest = self.ok("get_weakness_cards", domain="wdbc", project_root="project")
        self.assertIn("class_imbalance", {c["weakness"] for c in latest["cards"]})
        self.assertIsNotNone(latest["updated_at"])

    def test_outputs_land_in_the_project_checkpoints(self):
        self.ok("run_audit", domain="wdbc", project_root="project")
        artefacts = RunArtefacts.for_project(self.project, "wdbc")
        self.assertTrue(artefacts.log.exists())
        self.assertTrue(artefacts.cards_latest.exists())

    def test_bad_arguments_are_tool_errors(self):
        self.error("run_audit", domain="nope", project_root="project")
        self.error("run_audit", domain="wdbc", project_root="../elsewhere")
        self.error("run_audit", domain="wdbc", project_root="missing_dir")


if __name__ == "__main__":
    unittest.main()
