"""Evidence tool: check a project's CSVs against a manifest's schema."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from ...core import CHECKPOINTS_DIRNAME
from ...validation import render_report, suggest_dict_writer_snippet, validate_checkpoints
from ..config import ServerConfig
from ..models import EvidenceFileReport, EvidenceReport
from ._common import resolve_manifest, resolve_project_root

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def register(server: MCPServer, config: ServerConfig) -> None:
    def validate_evidence(domain: str, project_root: str = ".") -> EvidenceReport:
        """Check the evidence CSVs in ``<project_root>/checkpoints/`` against a manifest.

        Reports, per expected file: whether it exists, which required
        columns are missing, which aliases were accepted, and which
        detectors stay inactive without it. When a file is missing or
        incomplete, ``suggested_snippet`` holds a ready-to-paste Python
        ``csv.DictWriter`` block that emits it. ``ok`` is false only when
        a present file lacks required columns (missing files are allowed:
        their detectors simply stay inactive). Run this before run_audit.
        """
        resolve_manifest(domain)
        root = resolve_project_root(config, project_root)
        checkpoints = root / CHECKPOINTS_DIRNAME
        try:
            report = validate_checkpoints(domain, checkpoints)
        except FileNotFoundError as exc:
            raise ToolError(f"no evidence schema ships for domain {domain!r}") from exc
        text, ok = render_report(report, suggest=False)
        return {
            "domain": domain,
            "schema_version": report.get("schema_version"),
            "checkpoints": str(checkpoints),
            "ok": ok,
            "files": [_file_report(r) for r in report["file_results"]],
            "report_text": text,
        }

    server.add_tool(validate_evidence, annotations=_READ_ONLY)


def _file_report(r: dict) -> EvidenceFileReport:
    needs_fix = not r["present"] or bool(r["missing_required"])
    snippet = (
        suggest_dict_writer_snippet(r["filename"], r["file_spec"]).strip()
        if needs_fix else ""
    )
    return {
        "filename": r["filename"],
        "present": r["present"],
        "missing_required": r["missing_required"],
        "accepted_columns": r["accepted_columns"],
        "extra_columns": r["extra_columns"],
        "type_warnings": r["type_warnings"],
        "used_by_detectors": r["used_by_detectors"],
        "suggested_snippet": snippet,
    }
