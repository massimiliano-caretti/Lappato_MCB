"""Tests for the public ad-hoc search API (``lappato_mcb.literature``).

Online mode is exercised against stubbed source adapters (no network);
offline mode against a small on-disk corpus.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lappato_mcb.literature import MAX_RESULTS_PER_SOURCE, search_literature
from lappato_mcb.literature import search as search_module


def _stub(source: str, n: int = 3):
    def fn(query: str, max_results: int) -> list[dict]:
        return [
            {"id": f"{source}-{i}", "title": f"{query} paper {i}", "year": "2025"}
            for i in range(n)
        ]
    return fn


def _boom(query: str, max_results: int) -> list[dict]:
    raise TimeoutError("upstream timed out")


class OnlineSearchTests(unittest.TestCase):

    def test_hits_are_tagged_with_source_and_capped(self):
        stubs = {"arXiv": _stub("arXiv", 9), "OpenAlex": _stub("OpenAlex", 9)}
        with mock.patch.dict(search_module._SOURCE_FN, stubs):
            out = search_literature("calibration", sources=["arXiv", "OpenAlex"], max_results=2)
        self.assertEqual(out["errors"], {})
        self.assertFalse(out["offline"])
        self.assertEqual([h["source"] for h in out["hits"]], ["arXiv"] * 2 + ["OpenAlex"] * 2)

    def test_failing_source_is_reported_without_sinking_others(self):
        stubs = {"arXiv": _boom, "Crossref": _stub("Crossref")}
        with mock.patch.dict(search_module._SOURCE_FN, stubs):
            out = search_literature("drift", sources=["arXiv", "Crossref"])
        self.assertIn("TimeoutError", out["errors"]["arXiv"])
        self.assertEqual({h["source"] for h in out["hits"]}, {"Crossref"})

    def test_duplicate_sources_are_queried_once(self):
        calls: list[str] = []

        def spy(query: str, max_results: int) -> list[dict]:
            calls.append(query)
            return []

        with mock.patch.dict(search_module._SOURCE_FN, {"arXiv": spy}):
            out = search_literature("x", sources=["arXiv", "arXiv"])
        self.assertEqual(out["sources"], ["arXiv"])
        self.assertEqual(len(calls), 1)


class OfflineSearchTests(unittest.TestCase):

    def test_reads_corpus_filtered_by_source(self):
        rows = [
            {"source": "arXiv", "id": "a1", "title": "Temperature scaling calibration"},
            {"source": "OpenAlex", "id": "o1", "title": "Calibration of neural networks"},
            {"source": "arXiv", "id": "a2", "title": "Unrelated topic"},
        ]
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "corpus.jsonl"
            corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            out = search_literature(
                "calibration", sources=["arXiv"], offline=True, corpus_path=corpus,
            )
        self.assertTrue(out["offline"])
        self.assertEqual([h["id"] for h in out["hits"]], ["a1"])

    def test_missing_corpus_yields_no_hits(self):
        with tempfile.TemporaryDirectory() as td:
            out = search_literature(
                "anything", offline=True, corpus_path=Path(td) / "absent.jsonl",
            )
        self.assertEqual(out["hits"], [])


class ArgumentValidationTests(unittest.TestCase):

    def test_rejects_bad_arguments(self):
        bad_calls = [
            dict(query="  "),
            dict(query="q", sources=["Scholar"]),
            dict(query="q", sources=[]),
            dict(query="q", max_results=0),
            dict(query="q", max_results=MAX_RESULTS_PER_SOURCE + 1),
            dict(query="q", offline=True),
        ]
        for kwargs in bad_calls:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                search_literature(**kwargs)


if __name__ == "__main__":
    unittest.main()
