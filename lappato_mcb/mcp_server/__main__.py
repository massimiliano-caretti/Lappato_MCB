"""Command-line entry point: ``lappato-mcb-mcp`` / ``python -m lappato_mcb.mcp_server``.

Serves over stdio, the transport MCP clients (Claude Code, Claude
Desktop, IDEs) use for local servers. Nothing but protocol messages
may be written to stdout; diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ENV_ONLINE, ENV_ROOT, ServerConfig


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="lappato-mcb-mcp",
        description="Run the LAPPATO_MCB Model Context Protocol server over stdio.",
    )
    p.add_argument(
        "--root", type=Path, default=None,
        help=f"Directory every project_root must stay inside "
             f"(default: ${ENV_ROOT}, else the current directory).",
    )
    p.add_argument(
        "--online", action="store_true",
        help=f"Query arXiv/OpenAlex/Crossref by default instead of the local "
             f"corpus (default: ${ENV_ONLINE}, else offline).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    env = dict(os.environ)  # command-line flags override the environment
    if args.root is not None:
        env[ENV_ROOT] = str(args.root)
    if args.online:
        env[ENV_ONLINE] = "1"
    try:
        config = ServerConfig.from_env(env)
        from .app import build_server
    except (ImportError, ValueError) as exc:
        print(f"lappato-mcb-mcp: {exc}", file=sys.stderr)
        return 2
    build_server(config).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
