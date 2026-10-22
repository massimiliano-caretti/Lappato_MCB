"""Tests for ``RunArtefacts``, the single source of the output-file naming."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lappato_mcb import LAPPATO_MCB, RunArtefacts


class RunArtefactsTests(unittest.TestCase):

    def test_names_match_the_documented_convention(self):
        a = RunArtefacts.for_project(Path("/p"), "wdbc")
        self.assertEqual(a.checkpoints, Path("/p/checkpoints"))
        self.assertEqual(a.log.name, "wdbc_lappato_mcb_log.md")
        self.assertEqual(a.papers_jsonl.name, "wdbc_lappato_mcb_papers.jsonl")
        self.assertEqual(a.meta_log.name, "wdbc_lappato_mcb_meta.csv")
        self.assertEqual(a.cards_jsonl.name, "wdbc_lappato_mcb_weakness_cards.jsonl")
        self.assertEqual(a.cards_latest.name, "wdbc_lappato_mcb_weakness_cards.latest.json")

    def test_daemon_writes_where_run_artefacts_points(self):
        with tempfile.TemporaryDirectory() as td:
            auditor = LAPPATO_MCB(Path(td), run_tag="wdbc", offline=True)
            self.assertEqual(auditor.artefacts, RunArtefacts.for_project(Path(td), "wdbc"))
            auditor.run_once()
            self.assertTrue(auditor.artefacts.log.exists())
            self.assertTrue(auditor.artefacts.cards_latest.exists())


if __name__ == "__main__":
    unittest.main()
