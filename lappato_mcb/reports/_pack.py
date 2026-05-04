"""ReportPack dataclass — separated to break the import cycle.

The pack dataclass is imported by every domain module; the registry
(`__init__.py`) imports the domains. Putting the dataclass in its own
module keeps both directions of the import dag well-defined.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from matplotlib.figure import Figure


@dataclass(frozen=True)
class ReportPack:
    """Domain reporting contract — parsing + summarising + plotting.

    All callables are pure: ``parsers[name](path)`` reads a CSV and
    returns a list of dicts; ``summarise`` aggregates the parsed
    dicts into headline statistics; ``build_figures`` emits the
    domain-specific Figures keyed by a stable id used as the dict key
    in the bundle and as the lookup key in ``captions``.
    """
    run_tag: str
    pipeline_csvs: dict[str, str]                       # name -> filename
    parsers: dict[str, Callable[[Path], list[dict]]]    # name -> parser
    summarise: Callable[[dict[str, list[dict]]], dict[str, Any]]
    build_figures: Callable[[dict[str, list[dict]]], dict[str, Figure]]
    captions: dict[str, str] = field(default_factory=dict)
    cover_block: Callable[[dict[str, Any]], list[tuple[str, str]]] | None = None

    def parse_all(self, checkpoints_dir: Path) -> dict[str, list[dict]]:
        """Run every parser against ``checkpoints_dir``."""
        out: dict[str, list[dict]] = {}
        for name, parser in self.parsers.items():
            csv = checkpoints_dir / self.pipeline_csvs[name]
            out[name] = parser(csv)
        return out


__all__ = ["ReportPack"]
