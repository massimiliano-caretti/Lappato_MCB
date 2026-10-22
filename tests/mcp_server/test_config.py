"""Server configuration and the optional-dependency contract.

Stdlib-only: runs even without the ``mcp`` extra.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lappato_mcb.mcp_server.config import ENV_ONLINE, ENV_ROOT, ServerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


class ServerConfigTests(unittest.TestCase):

    def test_defaults_to_cwd_and_offline(self):
        config = ServerConfig.from_env({})
        self.assertEqual(config.allowed_root, Path.cwd().resolve())
        self.assertTrue(config.default_offline)

    def test_environment_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            config = ServerConfig.from_env({ENV_ROOT: td, ENV_ONLINE: "yes"})
        self.assertEqual(config.allowed_root, Path(td).resolve())
        self.assertFalse(config.default_offline)

    def test_root_must_be_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError):
            ServerConfig(Path(td) / "missing")


class OptionalDependencyTests(unittest.TestCase):

    def test_core_library_never_imports_the_mcp_sdk(self):
        """Plain library use must keep working without the ``mcp`` extra."""
        code = (
            "import sys, lappato_mcb, lappato_mcb.literature, lappato_mcb.validation, "
            "lappato_mcb.mcp_server; "
            "sys.exit(1 if any(m == 'mcp' or m.startswith('mcp.') for m in sys.modules) else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
