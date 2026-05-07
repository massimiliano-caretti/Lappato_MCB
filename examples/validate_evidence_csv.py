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

The validator is stdlib-only: it uses ``csv`` + ``json`` and lives in
``examples/`` rather than the daemon core.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "lappato_mcb" / "manifests" / "schemas"
DEFAULT_CHECKPOINTS = ROOT / "checkpoints"


def _load_schema(domain: str) -> dict:
    p = SCHEMA_DIR / f"{domain}.schema.json"
    if not p.exists():
        raise SystemExit(f"no schema for domain {domain!r} at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _list_schema_domains() -> list[str]:
    return sorted(p.stem.replace(".schema", "") for p in SCHEMA_DIR.glob("*.schema.json"))


def _read_csv_header_and_sample(path: Path, sample_rows: int = 5) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows: list[dict] = []
        for i, row in enumerate(reader):
            if i >= sample_rows:
                break
            rows.append(row)
    return header, rows


def _column_present(spec: dict, header: list[str]) -> tuple[bool, str | None]:
    """Return (found, accepted_name)."""
    canonical = spec["name"]
    aliases = spec.get("aliases") or []
    candidates = [canonical, *aliases]
    lower_header = [h.strip() for h in header]
    for c in candidates:
        if c in lower_header:
            return True, c
    return False, None


def _looks_numeric(rows: list[dict], col: str) -> bool:
    if not rows:
        return True
    seen = 0
    for r in rows:
        v = r.get(col)
        if v in (None, ""):
            continue
        seen += 1
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return seen == 0 or True


def validate_file(path: Path, file_spec: dict) -> dict:
    """Validate a single CSV against its file spec entry."""
    if not path.exists():
        return {
            "file": str(path),
            "present": False,
            "missing_required": [],
            "accepted_columns": [],
            "extra_columns": [],
            "type_warnings": [],
        }

    header, sample = _read_csv_header_and_sample(path)

    missing_required: list[str] = []
    accepted: list[dict] = []
    type_warnings: list[str] = []
    matched_columns: set[str] = set()

    for col_spec in file_spec.get("columns", []):
        ok, name = _column_present(col_spec, header)
        if ok and name is not None:
            matched_columns.add(name)
            accepted.append({
                "canonical": col_spec["name"],
                "accepted_as": name,
                "via_alias": name != col_spec["name"],
            })
            if col_spec.get("type") == "number" and not _looks_numeric(sample, name):
                type_warnings.append(
                    f"column {name!r} expected numeric values"
                )
        elif col_spec.get("required"):
            missing_required.append(col_spec["name"])

    extra_columns = [h for h in header if h not in matched_columns]

    return {
        "file": str(path),
        "present": True,
        "missing_required": missing_required,
        "accepted_columns": accepted,
        "extra_columns": extra_columns,
        "type_warnings": type_warnings,
    }


def validate_checkpoints(domain: str, checkpoints: Path) -> dict:
    schema = _load_schema(domain)
    file_results: list[dict] = []
    for fname, spec in schema.get("evidence_files", {}).items():
        path = checkpoints / fname
        result = validate_file(path, spec)
        result["used_by_detectors"] = spec.get("used_by_detectors", [])
        result["file_spec"] = spec  # retained so render_report can suggest fixes
        result["filename"] = fname
        file_results.append(result)
    return {
        "domain": domain,
        "schema_version": schema.get("schema_version"),
        "checkpoints": str(checkpoints),
        "file_results": file_results,
    }


def suggest_dict_writer_snippet(
    filename: str,
    file_spec: dict,
    checkpoints_var: str = "checkpoints_dir",
) -> str:
    """Return a copy-pasteable Python snippet that an AI assistant or
    a human can drop into the host pipeline to emit the missing
    evidence file.

    The snippet uses canonical column names from the schema; aliases
    are listed as a comment so the human can rename to match an
    existing log convention if desired. Required columns become
    DictWriter fields; optional columns are emitted as comments to
    flag that they would unlock additional detectors.
    """
    columns = file_spec.get("columns", [])
    if not columns:
        return ""

    required_names = [c["name"] for c in columns if c.get("required")]
    optional_names = [c["name"] for c in columns if not c.get("required")]

    alias_lines: list[str] = []
    for c in columns:
        aliases = c.get("aliases") or []
        if aliases:
            alias_lines.append(
                f"#   '{c['name']}' also accepts aliases: {', '.join(aliases)}"
            )

    field_list = required_names + optional_names
    field_list_repr = ", ".join(repr(n) for n in field_list)

    # Use ``None`` as a placeholder so the snippet parses unchanged
    # (the user / AI assistant replaces None with the live values).
    example_row = "{ " + ", ".join(
        f"{n!r}: None" for n in required_names or field_list[:1]
    ) + " }"

    snippet_lines = [
        "",
        f"# --- suggested csv.DictWriter for {filename} -----------------",
        "# Drop this into your training/eval code at the point where the",
        "# matching numbers are already computed. No new dependencies.",
        "# Replace `None` placeholders with the live values from your",
        "# pipeline. The snippet parses out of the box so you can paste",
        "# it first and wire the values in second.",
    ]
    if alias_lines:
        snippet_lines.append("# Schema-accepted aliases for the columns below:")
        snippet_lines.extend(alias_lines)
    snippet_lines.extend([
        "import csv",
        f"path = {checkpoints_var} / {filename!r}",
        f"with path.open('w', newline='', encoding='utf-8') as fh:",
        f"    writer = csv.DictWriter(fh, fieldnames=[{field_list_repr}])",
        "    writer.writeheader()",
        f"    writer.writerow({example_row})",
        f"# -------------------------------------------------------------",
    ])
    return "\n".join(snippet_lines)


def render_report(report: dict, suggest: bool = True) -> tuple[str, bool]:
    """Pretty-print the validation report. Returns (text, ok).

    When ``suggest`` is True (default) and a present file is missing
    required columns, append a copy-pasteable ``csv.DictWriter``
    snippet that emits the file with the canonical column names. The
    snippet is the same one the README's three-line AI prompt expects
    the assistant to produce, so a user can either paste it directly
    into the pipeline or paste the validator output into a chat
    panel and let the assistant apply it.
    """
    lines = [
        f"=== {report['domain']} (schema_version={report['schema_version']}) ===",
        f"checkpoints: {report['checkpoints']}",
        "",
    ]
    ok = True
    for r in report["file_results"]:
        if not r["present"]:
            lines.append(f"[--] {Path(r['file']).name}: not present")
            if r["used_by_detectors"]:
                lines.append(
                    "       detectors that stay inactive: "
                    + ", ".join(r["used_by_detectors"])
                )
            if suggest and r.get("file_spec"):
                snippet = suggest_dict_writer_snippet(
                    r.get("filename") or Path(r["file"]).name,
                    r["file_spec"],
                )
                if snippet:
                    for ln in snippet.splitlines():
                        lines.append(f"       {ln}")
            continue
        if r["missing_required"]:
            ok = False
            lines.append(
                f"[FAIL] {Path(r['file']).name}: missing required columns: "
                + ", ".join(r["missing_required"])
            )
        else:
            lines.append(f"[ OK ] {Path(r['file']).name}")
        for col in r["accepted_columns"]:
            if col["via_alias"]:
                lines.append(
                    f"        accepted '{col['accepted_as']}' as alias for '{col['canonical']}'"
                )
        for w in r["type_warnings"]:
            lines.append(f"        warn: {w}")
        if r["extra_columns"]:
            lines.append(
                f"        extra columns ignored: " + ", ".join(r["extra_columns"])
            )
        # When the file is present but missing required columns, the
        # snippet helps the human (or AI assistant) close the gap
        # without re-reading the schema. Skipped when only optional
        # columns are missing, since those don't break the contract.
        if suggest and r["missing_required"] and r.get("file_spec"):
            snippet = suggest_dict_writer_snippet(
                r.get("filename") or Path(r["file"]).name,
                r["file_spec"],
            )
            if snippet:
                for ln in snippet.splitlines():
                    lines.append(f"       {ln}")
    return "\n".join(lines), ok


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
        domains = _list_schema_domains()
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
            report = validate_checkpoints(d, args.checkpoints)
        text, ok = render_report(report, suggest=not args.no_suggest)
        print(text)
        print()
        overall_ok = overall_ok and ok
        reports.append(report)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
