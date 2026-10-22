"""Confine client-supplied paths to the configured ``allowed_root``.

MCP tool arguments are chosen by a language model, so every path is
treated as untrusted: it is resolved (following symlinks and ``..``)
and rejected unless it stays inside the root the server was started
with. Stdlib-only, so it is testable without the ``mcp`` SDK.
"""
from __future__ import annotations

from pathlib import Path


class PathOutsideRootError(ValueError):
    """A client-supplied path resolves outside the allowed root."""


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` against ``root`` and return the absolute path.

    Relative paths are interpreted relative to ``root``. Raises
    :class:`PathOutsideRootError` when the resolved path escapes it.
    """
    root = Path(root).resolve()
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise PathOutsideRootError(
            f"{str(candidate)!r} resolves outside the allowed root {root}"
        )
    return path
