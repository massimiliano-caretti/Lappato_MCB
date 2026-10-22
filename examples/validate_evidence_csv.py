"""Validate the evidence CSVs in a checkpoints/ folder against the
versioned JSON Schema for a manifest.

The schemas live under ``lappato_mcb/manifests/schemas/<run_tag>.schema.json``
(one per registered domain) and document, for every evidence file the
manifest's detectors read:
  - the canonical column names + accepted aliases
  - the column type (``string`` or ``number``, best-effort)
  - which detectors will fire on the file

This validator is the contract between a host pipeline and a manifest:
if it passes, the manifest's detectors will see the data they need; if
it fails, the report tells you exactly what to add or rename.

Behaviour
---------
- Missing evidence files are reported but **do not** fail the run — the
  corresponding detectors stay inactive (the daemon's design contract is
  fail-closed on missing files).
- Required columns missing on a present file **do** fail the run, so the
  validator's exit code is suitable for CI.
- Aliases are accepted: a CSV emitting ``acc`` is treated the same as
  ``accuracy`` if the schema lists the alias.

Usage::

    # validate the bundled checkpoints/ against a manifest
    python examples/validate_evidence_csv.py --domain pipeline_health \\
        --checkpoints checkpoints/

    # validate every manifest's expected files against the same folder
    python examples/validate_evidence_csv.py --domain all \\
        --checkpoints checkpoints/

    # spot-check a single CSV against a single manifest
    python examples/validate_evidence_csv.py --domain physics \\
        --csv checkpoints/physics_conservation.csv

This script is a thin command-line front-end: the validation logic
lives in :mod:`lappato_mcb.validation.evidence` (stdlib-only), which is
also what the MCP server's ``validate_evidence`` tool calls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lappato_mcb.validation.evidence import (  # noqa: E402
    SCHEMA_DIR,
    list_schema_domains,
    load_schema,
    render_report,
    suggest_dict_writer_snippet,
    validate_checkpoints,
    validate_file,
)

DEFAULT_CHECKPOINTS = ROOT / "checkpoints"

__all__ = [
    "SCHEMA_DIR",
    "render_report",
    "suggest_dict_writer_snippet",
    "validate_checkpoints",
    "validate_file",
]


def _load_schema(domain: str) -> dict:
    try:
        return load_schema(domain)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--domain", required=True,
        help="Manifest tag, or 'all' to iterate over every registered schema.",
    )
    p.add_argument(
        "--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS,
        help="Folder containing the host-pipeline evidence CSVs.",
    )
    p.add_argument(
        "--csv", type=Path, default=None,
        help="Validate a single CSV file (must match a known evidence file name in the manifest).",
    )
    p.add_argument(
        "--no-suggest", action="store_true",
        help="Skip the copy-pasteable csv.DictWriter snippets that are emitted "
             "for missing files / missing required columns.",
    )
    args = p.parse_args()

    if args.domain == "all":
        domains = list_schema_domains()
    else:
        domains = [args.domain]

    overall_ok = True
    reports: list[dict] = []
    for d in domains:
        if args.csv is not None:
            schema = _load_schema(d)
            ev = schema.get("evidence_files", {})
            if args.csv.name not in ev:
                print(
                    f"[skip] {args.csv.name} not in {d} schema (known files: "
                    + ", ".join(ev.keys()) + ")",
                    file=sys.stderr,
                )
                continue
            r = validate_file(args.csv, ev[args.csv.name])
            r["used_by_detectors"] = ev[args.csv.name].get("used_by_detectors", [])
            report = {
                "domain": d,
                "schema_version": schema.get("schema_version"),
                "checkpoints": str(args.csv.parent),
                "file_results": [r],
            }
        else:
            _load_schema(d)  # clean SystemExit on an unknown domain
            report = validate_checkpoints(d, args.checkpoints)
        text, ok = render_report(report, suggest=not args.no_suggest)
        print(text)
        print()
        overall_ok = overall_ok and ok
        reports.append(report)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
