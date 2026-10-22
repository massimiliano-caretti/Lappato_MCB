"""Server factory: assemble the MCP server from config, tools and resources."""
from __future__ import annotations

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The LAPPATO_MCB MCP server needs the optional 'mcp' extra "
        "(Python >= 3.10): pip install 'lappato_mcb[mcp]'"
    ) from exc

from .. import __version__
from . import resources, tools
from .config import ServerConfig

SERVER_NAME = "lappato-mcb"

INSTRUCTIONS = """\
LAPPATO_MCB audits machine-learning pipelines: it reads diagnostic CSVs
that a training/evaluation pipeline writes to <project>/checkpoints/,
detects known weaknesses (class imbalance, poor calibration, leakage,
subgroup disparity, ...) and retrieves targeted scientific literature
for each one from arXiv, OpenAlex, Crossref and JOSS.

Typical flow:
1. list_manifests -> pick the domain matching the user's pipeline
   (pipeline_health is a good domain-agnostic default).
2. describe_manifest(domain) -> which evidence CSVs each detector reads.
3. validate_evidence(domain, project_root) -> what is missing; each gap
   comes with a ready-to-paste csv.DictWriter snippet.
4. run_audit(domain, project_root) -> one weakness card per active
   weakness, with severity, evidence summary, papers and next checks.
5. get_weakness_cards(domain, project_root) -> re-read past results.

search_literature answers ad-hoc literature questions without a manifest.
All paths are relative to, and confined within, the server's root.
"""


def build_server(config: ServerConfig | None = None) -> MCPServer:
    """Return a ready-to-run server; ``config`` defaults to the environment."""
    config = config or ServerConfig.from_env()
    server = MCPServer(
        name=SERVER_NAME,
        title="LAPPATO_MCB",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    tools.register_all(server, config)
    resources.register(server)
    return server
