"""Read-only MCP resources: manifests and evidence schemas as JSON documents.

Same content as the ``describe_manifest`` tool, addressable by URI so
clients can attach a manifest or schema to a conversation as context.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from ..validation import load_schema
from .models import ManifestDescription
from .tools.manifests import describe_manifest

_JSON = "application/json"


def register(server: MCPServer) -> None:
    @server.resource(
        "lappato://manifests/{domain}",
        mime_type=_JSON,
        description="Weaknesses a manifest detects, with their evidence files and guidance.",
    )
    def manifest_resource(domain: str) -> ManifestDescription:
        try:
            return describe_manifest(domain)
        except ToolError as exc:
            raise ResourceError(str(exc)) from exc

    @server.resource(
        "lappato://schemas/{domain}",
        mime_type=_JSON,
        description="JSON Schema of the evidence CSVs a manifest's detectors read.",
    )
    def schema_resource(domain: str) -> dict:
        try:
            return load_schema(domain)
        except FileNotFoundError as exc:
            raise ResourceError(f"no evidence schema for domain {domain!r}") from exc
