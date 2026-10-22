"""End-to-end: launch ``python -m lappato_mcb.mcp_server`` and talk MCP over stdio.

This is the path real clients (Claude Code, Claude Desktop) take, so
it also guards against anything polluting stdout.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .support import require_mcp_sdk

require_mcp_sdk()

REPO_ROOT = Path(__file__).resolve().parents[2]


class StdioEntryPointTests(unittest.TestCase):

    def test_client_can_list_and_call_tools_over_stdio(self):
        from mcp import Client, StdioServerParameters

        with tempfile.TemporaryDirectory() as td:
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "lappato_mcb.mcp_server", "--root", td],
                cwd=str(REPO_ROOT),
            )

            async def _session():
                async with Client(params) as client:
                    tools = (await client.list_tools()).tools
                    result = await client.call_tool("list_manifests", {})
                    return tools, result

            tools, result = asyncio.run(_session())
        self.assertIn("run_audit", {t.name for t in tools})
        self.assertFalse(result.is_error)

    def test_invalid_root_exits_with_status_2(self):
        result = subprocess.run(
            [sys.executable, "-m", "lappato_mcb.mcp_server", "--root", "/no/such/dir"],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("allowed_root", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
