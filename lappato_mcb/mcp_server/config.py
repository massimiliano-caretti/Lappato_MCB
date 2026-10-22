"""Runtime configuration of the MCP server.

Resolved once at start-up (CLI flags win over environment variables)
and passed explicitly to every tool, so tools stay pure functions of
``(config, arguments)`` and are trivial to test.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENV_ROOT = "LAPPATO_MCB_MCP_ROOT"
ENV_ONLINE = "LAPPATO_MCB_MCP_ONLINE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class ServerConfig:
    """Settings shared by every tool.

    allowed_root     : every ``project_root`` a client passes must resolve
                       inside this directory (see :mod:`.sandbox`)
    default_offline  : whether ``run_audit`` / ``search_literature`` use the
                       local corpus when the client does not say otherwise.
                       Offline is the default because an online audit issues
                       several rate-limited HTTP requests per active weakness
                       and can outlive a client's request timeout.
    """
    allowed_root: Path
    default_offline: bool = True

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"allowed_root is not a directory: {root}")
        object.__setattr__(self, "allowed_root", root)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerConfig:
        """Build a config from ``LAPPATO_MCB_MCP_*`` variables.

        ``LAPPATO_MCB_MCP_ROOT`` defaults to the current directory;
        ``LAPPATO_MCB_MCP_ONLINE`` (1/true/yes/on) flips the default to online.
        """
        env = os.environ if environ is None else environ
        online = env.get(ENV_ONLINE, "").strip().lower() in _TRUTHY
        return cls(
            allowed_root=Path(env.get(ENV_ROOT) or Path.cwd()),
            default_offline=not online,
        )
