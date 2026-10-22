"""``lappato://`` resources."""
from __future__ import annotations

import asyncio
import json
import unittest

from .support import ServerTestCase, require_mcp_sdk

require_mcp_sdk()


class ResourceTests(ServerTestCase):

    def read(self, uri: str) -> dict:
        """Return the resource's JSON body, or raise the client-side MCPError.

        The error is caught inside the session: escaping the client's
        context manager would wrap it in an ExceptionGroup.
        """
        from mcp import Client
        from mcp.shared.exceptions import MCPError

        async def _read():
            async with Client(self.server) as client:
                try:
                    return await client.read_resource(uri)
                except MCPError as exc:
                    return exc

        result = asyncio.run(_read())
        if isinstance(result, Exception):
            raise result
        return json.loads(result.contents[0].text)

    def test_manifest_and_schema_are_readable(self):
        self.assertEqual(self.read("lappato://manifests/wdbc")["run_tag"], "wdbc")
        self.assertEqual(self.read("lappato://schemas/wdbc")["domain"], "wdbc")

    def test_unknown_domain_raises(self):
        from mcp.shared.exceptions import MCPError

        for uri in ("lappato://manifests/nope", "lappato://schemas/nope"):
            with self.subTest(uri=uri), self.assertRaises(MCPError):
                self.read(uri)


if __name__ == "__main__":
    unittest.main()
