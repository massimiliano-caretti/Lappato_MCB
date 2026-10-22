"""Argument resolution shared by the tool modules.

Every helper turns a library-level failure into a
:class:`~mcp.server.mcpserver.exceptions.ToolError`, which the SDK
returns to the client as an ``isError`` result with a readable message
instead of a crash.
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError

from ... import manifests
from ..config import ServerConfig
from ..sandbox import PathOutsideRootError, resolve_within


def resolve_project_root(config: ServerConfig, project_root: str) -> Path:
    """Resolve a client-supplied project root inside ``config.allowed_root``."""
    try:
        root = resolve_within(config.allowed_root, project_root)
    except PathOutsideRootError as exc:
        raise ToolError(str(exc)) from exc
    if not root.is_dir():
        raise ToolError(f"project_root is not a directory: {root}")
    return root


def resolve_manifest(domain: str) -> tuple[list[dict], str]:
    """Return ``(manifest, run_tag)`` for a registered domain id."""
    try:
        return manifests.get(domain)
    except KeyError as exc:
        raise ToolError(
            f"unknown domain {domain!r}; call list_manifests for the valid ids"
        ) from exc


def resolve_offline(config: ServerConfig, offline: bool | None) -> bool:
    """Apply the server-wide default when the client leaves ``offline`` unset."""
    return config.default_offline if offline is None else bool(offline)
