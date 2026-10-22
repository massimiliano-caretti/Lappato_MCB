"""Shared helpers for the MCP server tests (not a test module)."""
from __future__ import annotations

import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from lappato_mcb.mcp_server import ServerConfig, build_server


def require_mcp_sdk() -> None:
    """Skip the calling test module when the optional ``mcp`` extra is absent."""
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("optional 'mcp' extra not installed") from exc


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def make_wdbc_project(root: Path, name: str = "project") -> Path:
    """Create a project whose evidence activates WDBC's ``class_imbalance``."""
    project = root / name
    write_csv(
        project / "checkpoints" / "wdbc_per_fold.csv",
        ["fold", "precision_M", "recall_M", "precision_B", "recall_B"],
        [[0, 0.70, 0.60, 0.95, 0.97], [1, 0.72, 0.58, 0.96, 0.98]],
    )
    return project


def write_corpus(project: Path, rows: list[dict]) -> Path:
    corpus = project / "checkpoints" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return corpus


class ServerTestCase(unittest.TestCase):
    """Builds a server confined to a fresh temporary root for every test.

    Tools are called through the SDK's in-memory client, so arguments
    and results go through the same validation and serialisation as
    with a real MCP client.
    """

    default_offline = True

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        config = ServerConfig(self.root, default_offline=self.default_offline)
        self.server = build_server(config)

    def tearDown(self):
        self._tmp.cleanup()

    def call(self, tool: str, **arguments: Any):
        from mcp import Client

        async def _call():
            async with Client(self.server) as client:
                return await client.call_tool(tool, arguments)

        return asyncio.run(_call())

    def ok(self, tool: str, **arguments: Any) -> dict:
        """Call ``tool``, assert success, return its structured content."""
        result = self.call(tool, **arguments)
        self.assertFalse(result.is_error, result.content[0].text if result.content else "")
        return result.structured_content

    def error(self, tool: str, **arguments: Any) -> str:
        """Call ``tool``, assert failure, return the error message."""
        result = self.call(tool, **arguments)
        self.assertTrue(result.is_error, f"{tool} unexpectedly succeeded")
        return result.content[0].text
