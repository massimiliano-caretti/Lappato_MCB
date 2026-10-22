# LAPPATO_MCB as an MCP server

LAPPATO_MCB ships an optional [Model Context Protocol](https://modelcontextprotocol.io)
server. It lets an MCP client — Claude Code, Claude Desktop, an IDE
assistant — discover manifests, check a project's evidence CSVs, run an
audit cycle and search the scholarly literature, using the same code
paths as the Python library.

The server is an **addition**, not a replacement: `import lappato_mcb`,
`LAPPATO_MCB(...).start()` and every other library API keep working
exactly as before, without the extra and without the `mcp` SDK.

## Install

Requires Python >= 3.10 (the `mcp` SDK does not support 3.9; the core
library still does).

```bash
python -m pip install "lappato_mcb[mcp]"      # from PyPI / a wheel
python -m pip install -e ".[mcp]"             # from a clone
```

This installs the `lappato-mcb-mcp` command. `python -m lappato_mcb.mcp_server`
is equivalent.

## Connect a client

The server speaks MCP over **stdio**: the client launches it as a
subprocess. Pick the directory the server may read and write with
`--root`; every `project_root` a tool receives must resolve inside it.

**Claude Code**

```bash
claude mcp add lappato-mcb -- lappato-mcb-mcp --root /path/to/your/projects
```

**Claude Desktop** (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "lappato-mcb": {
      "command": "lappato-mcb-mcp",
      "args": ["--root", "/path/to/your/projects"]
    }
  }
}
```

If the command is not on the client's `PATH` (e.g. it lives in a
virtualenv), use the absolute path of `lappato-mcb-mcp`, or
`"command": "/path/to/venv/bin/python", "args": ["-m", "lappato_mcb.mcp_server", ...]`.

## Configuration

| Setting | Flag | Environment variable | Default |
|---|---|---|---|
| Directory all paths are confined to | `--root DIR` | `LAPPATO_MCB_MCP_ROOT` | current directory |
| Query the network by default | `--online` | `LAPPATO_MCB_MCP_ONLINE=1` | offline |

Flags override environment variables.

**Why offline by default.** An online audit sends several rate-limited
requests (arXiv asks for 3 s between calls) per active weakness, which
can outlast a client's request timeout. Offline mode searches the
corpus that earlier online runs saved to `checkpoints/corpus.jsonl`.
A client can still pass `offline=false` on a single call.

## Tools

All project paths are relative to `--root` (`"."` is the root itself).
Outputs are structured JSON with a published output schema.

| Tool | Arguments | What it does | Side effects |
|---|---|---|---|
| `list_manifests` | — | Every domain id, its `run_tag`, number of weaknesses, whether an evidence schema exists | none |
| `describe_manifest` | `domain` | Each weakness: evidence CSV it reads (or *structural*), sources, query templates, transplant, next checks, success criteria; plus the evidence schema | none |
| `validate_evidence` | `domain`, `project_root="."` | Checks `checkpoints/*.csv` against the schema; missing or incomplete files come with a ready-to-paste `csv.DictWriter` snippet. `ok` is false only when a present file lacks required columns | none |
| `run_audit` | `domain`, `project_root="."`, `offline=null` | One synchronous cycle (`LAPPATO_MCB.run_once`); returns the weakness cards produced **by this cycle** | appends to the run's log, JSONL and meta-log under `checkpoints/`; online mode calls arXiv/OpenAlex/Crossref/JOSS |
| `get_weakness_cards` | `domain`, `project_root="."` | Latest card per weakness from `<run_tag>_lappato_mcb_weakness_cards.latest.json` — also works for runs made by the daemon inside a training process | none |
| `search_literature` | `query`, `sources=null`, `max_results=5`, `offline=null`, `project_root="."` | Ad-hoc search without a manifest. `sources` ⊆ arXiv, OpenAlex, Crossref, JOSS (default: the first three); `max_results` is per source (1–25). A failing source is reported in `errors` | online mode calls the selected sources |

Typical flow for an assistant: `list_manifests` → `describe_manifest` →
`validate_evidence` (add the missing CSVs to the pipeline) → `run_audit`
→ `get_weakness_cards` on later turns.

Invalid arguments (unknown domain, path outside the root, bad source…)
return a tool error with a readable message; they never crash the server.

## Resources

| URI | Content |
|---|---|
| `lappato://manifests/{domain}` | Same JSON as `describe_manifest` |
| `lappato://schemas/{domain}` | The evidence-CSV JSON Schema |

## Security model

- Tool arguments are chosen by a language model and treated as
  untrusted. Every path is resolved (symlinks and `..` included) and
  rejected unless it stays inside `--root`.
- The only files written are the run artefacts `run_audit` produces
  under `<project_root>/checkpoints/`.
- Network access happens only in online mode, only towards the public
  scholarly indices the library already uses, without credentials.
- Concurrent `run_audit` calls on the same project and domain are
  serialised, because they append to the same files.

## Architecture

```
lappato_mcb/mcp_server/
  __main__.py     CLI entry point (stdio transport)
  app.py          build_server(config): assembles the MCPServer
  config.py       ServerConfig: allowed root, offline default
  sandbox.py      resolve_within(): path confinement
  models.py       TypedDict output types -> JSON output schemas
  resources.py    lappato:// resources
  tools/
    __init__.py   TOOL_MODULES + register_all()
    _common.py    argument resolution -> ToolError
    manifests.py  list_manifests, describe_manifest
    evidence.py   validate_evidence
    audit.py      run_audit, get_weakness_cards
    literature.py search_literature
```

Tools are thin adapters: they validate arguments, call the library
(`lappato_mcb.manifests`, `.validation`, `.literature`, `LAPPATO_MCB`)
and shape the result. No detection or retrieval logic lives in the
server package.

### Adding a tool

1. Put the logic in the library (stdlib-only), with unit tests.
2. Add its output `TypedDict` to `models.py`.
3. Create `tools/<family>.py` with a `register(server, config)` function
   that calls `server.add_tool(...)` with accurate `ToolAnnotations`, or
   add the tool to an existing family module.
4. Append a new module to `TOOL_MODULES` in `tools/__init__.py`.
5. Add `tests/mcp_server/test_<family>_tool.py` using
   `support.ServerTestCase`, and update `EXPECTED_TOOLS` in
   `test_manifest_tools.py` and the table above.

## Testing

```bash
python -m pip install -e ".[mcp,reports]"
python -m unittest discover -s tests
```

`tests/mcp_server/` drives the server through the SDK's in-memory
client, plus one end-to-end test over a real stdio subprocess. Modules
that need the SDK skip themselves when the extra is not installed; the
path-sandbox and configuration tests always run.
