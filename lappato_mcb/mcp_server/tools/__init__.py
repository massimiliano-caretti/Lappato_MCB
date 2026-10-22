"""MCP tool modules.

Each module exposes ``register(server, config)``; tools are thin
adapters that validate arguments, call the library, and shape the
result into a :mod:`..models` TypedDict. To add a tool: write a module
here with a ``register`` function and append it to ``TOOL_MODULES``.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ..config import ServerConfig
from . import audit, evidence, literature, manifests

TOOL_MODULES = (manifests, evidence, audit, literature)


def register_all(server: MCPServer, config: ServerConfig) -> None:
    for module in TOOL_MODULES:
        module.register(server, config)
