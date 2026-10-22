"""Contract checks between a host pipeline and a manifest.

Stdlib-only. See :mod:`lappato_mcb.validation.evidence`.
"""
from __future__ import annotations

from .evidence import (
    SCHEMA_DIR,
    list_schema_domains,
    load_schema,
    render_report,
    suggest_dict_writer_snippet,
    validate_checkpoints,
    validate_file,
)

__all__ = [
    "SCHEMA_DIR",
    "list_schema_domains",
    "load_schema",
    "render_report",
    "suggest_dict_writer_snippet",
    "validate_checkpoints",
    "validate_file",
]
