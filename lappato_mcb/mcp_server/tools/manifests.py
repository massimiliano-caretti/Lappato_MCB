"""Discovery tools: which domains exist and what each manifest detects."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from ... import manifests
from ...validation import list_schema_domains, load_schema
from ..config import ServerConfig
from ..models import (
    ManifestCatalog,
    ManifestDescription,
    WeaknessDescription,
)
from ._common import resolve_manifest

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def list_manifests() -> ManifestCatalog:
    """List every registered manifest domain.

    Each entry gives the domain id (pass it to the other tools), the
    run_tag used to name its output files, how many weaknesses it can
    detect, and whether an evidence-CSV schema ships for it.
    """
    with_schema = set(list_schema_domains())
    return {
        "manifests": [
            {
                "domain": domain,
                "run_tag": run_tag,
                "n_weaknesses": len(manifest),
                "has_evidence_schema": domain in with_schema,
            }
            for domain, (manifest, run_tag) in sorted(manifests.REGISTRY.items())
        ]
    }


def describe_manifest(domain: str) -> ManifestDescription:
    """Describe the weaknesses a manifest detects and the evidence each needs.

    For every weakness: the evidence CSV it reads from the project's
    ``checkpoints/`` folder (structural weaknesses need none and are
    always active), the literature sources and query templates it uses,
    and the remediation guidance (transplant, next checks, success
    criteria). Also returns the evidence-CSV schema when one exists.
    """
    manifest, run_tag = resolve_manifest(domain)
    try:
        schema = load_schema(domain)
    except FileNotFoundError:
        schema = None
    return {
        "domain": domain,
        "run_tag": run_tag,
        "weaknesses": [_describe_weakness(w) for w in manifest],
        "evidence_schema": schema,
    }


def _describe_weakness(w: dict) -> WeaknessDescription:
    return {
        "id": w["id"],
        "title": w["title"],
        "evidence_file": w.get("evidence") or None,
        "structural": not w.get("evidence"),
        "sources": list(w.get("sources") or []),
        "query_templates": list(w.get("queries") or []),
        "transplant": w.get("transplant", ""),
        "why_it_matters": w.get("why_it_matters", ""),
        "next_checks": list(w.get("next_checks") or []),
        "success_criteria": list(w.get("success_criteria") or []),
    }


def register(server: MCPServer, config: ServerConfig) -> None:
    server.add_tool(list_manifests, annotations=_READ_ONLY)
    server.add_tool(describe_manifest, annotations=_READ_ONLY)
