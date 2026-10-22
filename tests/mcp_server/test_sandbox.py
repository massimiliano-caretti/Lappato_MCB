"""Path confinement (stdlib-only: runs even without the ``mcp`` extra)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from lappato_mcb.mcp_server.sandbox import PathOutsideRootError, resolve_within


class ResolveWithinTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "proj").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_relative_and_absolute_paths_inside_root_are_accepted(self):
        self.assertEqual(resolve_within(self.root, "."), self.root)
        self.assertEqual(resolve_within(self.root, "proj"), self.root / "proj")
        self.assertEqual(resolve_within(self.root, self.root / "proj"), self.root / "proj")

    def test_escapes_are_rejected(self):
        for candidate in ("..", "proj/../../x", "/etc", str(self.root) + "-sibling"):
            with self.subTest(candidate=candidate), self.assertRaises(PathOutsideRootError):
                resolve_within(self.root, candidate)

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_symlink_pointing_outside_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "link").symlink_to(outside)
            with self.assertRaises(PathOutsideRootError):
                resolve_within(self.root, "link")


if __name__ == "__main__":
    unittest.main()
