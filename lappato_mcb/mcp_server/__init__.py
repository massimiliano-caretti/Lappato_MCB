"""Model Context Protocol server for LAPPATO_MCB (optional).

Install with ``pip install 'lappato_mcb[mcp]'`` (Python >= 3.10) and run
``lappato-mcb-mcp``. The core library never imports this package, so
regular ``import lappato_mcb`` usage needs neither the extra nor the SDK.

Layout:
    config.py     ServerConfig (allowed root, offline default)
    sandbox.py    confinement of client-supplied paths
    models.py     structured output types
    tools/        one module per tool family, each with register()
    resources.py  manifests and schemas as lappato:// resources
    app.py        build_server(): assembles the server
    __main__.py   command-line entry point (stdio transport)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ServerConfig

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

__all__ = ["ServerConfig", "build_server"]


def build_server(config: ServerConfig | None = None) -> MCPServer:
    """Build the server (imports the ``mcp`` SDK lazily; see :mod:`.app`)."""
    from .app import build_server as _build_server

    return _build_server(config)
