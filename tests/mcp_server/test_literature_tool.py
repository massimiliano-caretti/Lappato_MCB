"""``search_literature`` (online sources are stubbed: no network)."""
from __future__ import annotations

import unittest
from unittest import mock

from lappato_mcb.literature import search as search_module

from .support import ServerTestCase, require_mcp_sdk, write_corpus

require_mcp_sdk()


def _stub(query: str, max_results: int) -> list[dict]:
    return [{"id": "x1", "title": f"About {query}", "year": "2025"}]


class OfflineLiteratureToolTests(ServerTestCase):

    def test_searches_the_project_corpus(self):
        write_corpus(self.root, [
            {"source": "OpenAlex", "id": "W1", "title": "Conformal prediction intervals"},
        ])
        out = self.ok("search_literature", query="conformal prediction")
        self.assertTrue(out["offline"])
        self.assertEqual([h["id"] for h in out["hits"]], ["W1"])

    def test_invalid_arguments_are_tool_errors(self):
        self.assertIn("sources", self.error("search_literature", query="q", sources=["Scholar"]))
        self.assertIn("max_results", self.error("search_literature", query="q", max_results=0))


class OnlineLiteratureToolTests(ServerTestCase):

    default_offline = False

    def test_queries_the_requested_sources(self):
        with mock.patch.dict(search_module._SOURCE_FN, {"JOSS": _stub}):
            out = self.ok("search_literature", query="pipelines", sources=["JOSS"])
        self.assertFalse(out["offline"])
        self.assertEqual(out["hits"], [
            {"id": "x1", "title": "About pipelines", "year": "2025", "source": "JOSS"},
        ])


if __name__ == "__main__":
    unittest.main()
