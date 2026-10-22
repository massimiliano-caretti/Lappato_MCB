"""Audit tools: run one monitoring cycle and read the resulting weakness cards."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from ...core import LAPPATO_MCB, RunArtefacts
from ..config import ServerConfig
from ..models import AuditResult, WeaknessCardsSnapshot
from ._common import resolve_manifest, resolve_offline, resolve_project_root

# A cycle appends to files shared by every run of the same (project,
# run_tag); concurrent calls on that pair are serialised.
_RUN_LOCKS: dict[tuple[Path, str], threading.Lock] = {}
_RUN_LOCKS_GUARD = threading.Lock()


def _run_lock(root: Path, run_tag: str) -> threading.Lock:
    with _RUN_LOCKS_GUARD:
        return _RUN_LOCKS.setdefault((root, run_tag), threading.Lock())


def register(server: MCPServer, config: ServerConfig) -> None:
    def run_audit(
        domain: str,
        project_root: str = ".",
        offline: bool | None = None,
    ) -> AuditResult:
        """Run one LAPPATO_MCB audit cycle on a project and return its weakness cards.

        Reads the evidence CSVs in ``<project_root>/checkpoints/``, activates
        the manifest's weakness detectors whose thresholds are crossed,
        retrieves targeted literature for each active weakness, and returns
        one card per active weakness (severity, evidence summary, top papers,
        remediation guidance). Outputs are also appended to the run's log,
        JSONL and meta-log files under ``checkpoints/``.

        ``offline`` searches the local corpus instead of arXiv/OpenAlex/
        Crossref; leave it unset to use the server default (offline unless
        the server was started with --online). Online cycles take tens of
        seconds per active weakness because of polite-use rate limits.
        """
        manifest, run_tag = resolve_manifest(domain)
        root = resolve_project_root(config, project_root)
        use_offline = resolve_offline(config, offline)
        with _run_lock(root, run_tag):
            auditor = LAPPATO_MCB(root, manifest=manifest, run_tag=run_tag, offline=use_offline)
            artefacts = auditor.artefacts
            offset = _size(artefacts.cards_jsonl)
            auditor.run_once()
            cards = _read_cards_from(artefacts.cards_jsonl, offset)
        return {
            "domain": domain,
            "run_tag": run_tag,
            "offline": use_offline,
            "checkpoints": str(artefacts.checkpoints),
            "log_path": str(artefacts.log),
            "n_active_weaknesses": len(cards),
            "cards": cards,
        }

    def get_weakness_cards(domain: str, project_root: str = ".") -> WeaknessCardsSnapshot:
        """Return the latest weakness card of every weakness seen in past audits.

        Reads ``checkpoints/<run_tag>_lappato_mcb_weakness_cards.latest.json``
        without running anything (no network, no writes). Use it to inspect
        the results of an audit run earlier by run_audit or by the daemon
        inside a training process. Returns an empty list when the project
        has never been audited for this domain.
        """
        _, run_tag = resolve_manifest(domain)
        root = resolve_project_root(config, project_root)
        latest = RunArtefacts.for_project(root, run_tag).cards_latest
        if not latest.exists():
            return {"run_tag": run_tag, "updated_at": None, "cards": []}
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"cannot read {latest.name}: {exc}") from exc
        return {
            "run_tag": run_tag,
            "updated_at": data.get("updated_at"),
            "cards": list(data.get("cards") or []),
        }

    server.add_tool(
        run_audit,
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    server.add_tool(
        get_weakness_cards,
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _read_cards_from(path: Path, offset: int) -> list[dict]:
    """Parse the JSONL card rows appended after byte ``offset``."""
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read().decode("utf-8")
    cards: list[dict] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cards.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return cards
