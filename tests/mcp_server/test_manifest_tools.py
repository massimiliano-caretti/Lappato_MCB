"""``list_manifests`` / ``describe_manifest`` and the tool catalogue."""
from __future__ import annotations

import asyncio
import unittest

from lappato_mcb import manifests

from .support import ServerTestCase, require_mcp_sdk

require_mcp_sdk()

EXPECTED_TOOLS = {
    "list_manifests",
    "describe_manifest",
    "validate_evidence",
    "run_audit",
    "get_weakness_cards",
    "search_literature",
}


class ToolCatalogueTests(ServerTestCase):

    def test_exposes_exactly_the_documented_tools_with_output_schemas(self):
        from mcp import Client

        async def _list():
            async with Client(self.server) as client:
                return (await client.list_tools()).tools

        tools = {t.name: t for t in asyncio.run(_list())}
        self.assertEqual(set(tools), EXPECTED_TOOLS)
        for name, tool in tools.items():
            with self.subTest(tool=name):
                self.assertTrue(tool.description)
                self.assertIsNotNone(tool.output_schema)


class ManifestToolTests(ServerTestCase):

    def test_list_manifests_covers_the_registry(self):
        out = self.ok("list_manifests")
        self.assertEqual({m["domain"] for m in out["manifests"]}, set(manifests.REGISTRY))
        wdbc = next(m for m in out["manifests"] if m["domain"] == "wdbc")
        self.assertEqual(wdbc["n_weaknesses"], len(manifests.REGISTRY["wdbc"][0]))
        self.assertTrue(wdbc["has_evidence_schema"])

    def test_describe_manifest_lists_weaknesses_and_schema(self):
        out = self.ok("describe_manifest", domain="wdbc")
        by_id = {w["id"]: w for w in out["weaknesses"]}
        self.assertEqual(by_id["class_imbalance"]["evidence_file"], "wdbc_per_fold.csv")
        self.assertFalse(by_id["class_imbalance"]["structural"])
        self.assertTrue(any(w["structural"] for w in out["weaknesses"]))
        self.assertEqual(out["evidence_schema"]["domain"], "wdbc")

    def test_unknown_domain_is_a_tool_error(self):
        self.assertIn("list_manifests", self.error("describe_manifest", domain="nope"))


if __name__ == "__main__":
    unittest.main()
