"""Literature tool: ad-hoc search across the scholarly sources."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from ...core import CHECKPOINTS_DIRNAME, CORPUS_FILENAME, GENERAL_SOURCES
from ...literature import search_literature as _search_literature
from ..config import ServerConfig
from ..models import LiteratureSearchResult
from ._common import resolve_offline, resolve_project_root


def register(server: MCPServer, config: ServerConfig) -> None:
    def search_literature(
        query: str,
        sources: list[str] | None = None,
        max_results: int = 5,
        offline: bool | None = None,
        project_root: str = ".",
    ) -> LiteratureSearchResult:
        """Search scholarly literature for a free-text query, independent of any manifest.

        ``sources`` is any subset of "arXiv", "OpenAlex", "Crossref" and "JOSS"
        (default: the first three). ``max_results`` is per source (1-25).
        Offline mode searches the corpus harvested by earlier audits in
        ``<project_root>/checkpoints/corpus.jsonl`` instead of the network;
        leave ``offline`` unset to use the server default. A source that fails
        is reported in ``errors`` while the others still answer.
        """
        use_offline = resolve_offline(config, offline)
        corpus = None
        if use_offline:
            root = resolve_project_root(config, project_root)
            corpus = root / CHECKPOINTS_DIRNAME / CORPUS_FILENAME
        try:
            return _search_literature(
                query,
                sources=sources or GENERAL_SOURCES,
                max_results=max_results,
                offline=use_offline,
                corpus_path=corpus,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    server.add_tool(
        search_literature,
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
    )
