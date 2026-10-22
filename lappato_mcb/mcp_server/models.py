"""Structured output types of the MCP tools.

The SDK derives each tool's JSON ``outputSchema`` from these
TypedDicts, so clients get typed results instead of free text.
``typing_extensions.TypedDict`` is required by pydantic on Python < 3.12.
Weakness cards and literature hits keep an open ``dict`` shape: their
fields are defined by the core library, not by the MCP layer.
"""
from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ManifestSummary(TypedDict):
    domain: str
    run_tag: str
    n_weaknesses: int
    has_evidence_schema: bool


class ManifestCatalog(TypedDict):
    manifests: list[ManifestSummary]


class WeaknessDescription(TypedDict):
    id: str
    title: str
    evidence_file: str | None
    structural: bool
    sources: list[str]
    query_templates: list[str]
    transplant: str
    why_it_matters: str
    next_checks: list[str]
    success_criteria: list[str]


class ManifestDescription(TypedDict):
    domain: str
    run_tag: str
    weaknesses: list[WeaknessDescription]
    evidence_schema: dict[str, Any] | None


class EvidenceFileReport(TypedDict):
    filename: str
    present: bool
    missing_required: list[str]
    accepted_columns: list[dict[str, Any]]
    extra_columns: list[str]
    type_warnings: list[str]
    used_by_detectors: list[str]
    suggested_snippet: str


class EvidenceReport(TypedDict):
    domain: str
    schema_version: int | None
    checkpoints: str
    ok: bool
    files: list[EvidenceFileReport]
    report_text: str


class AuditResult(TypedDict):
    domain: str
    run_tag: str
    offline: bool
    checkpoints: str
    log_path: str
    n_active_weaknesses: int
    cards: list[dict[str, Any]]


class WeaknessCardsSnapshot(TypedDict):
    run_tag: str
    updated_at: str | None
    cards: list[dict[str, Any]]


class LiteratureSearchResult(TypedDict):
    query: str
    sources: list[str]
    offline: bool
    hits: list[dict[str, Any]]
    errors: dict[str, str]
