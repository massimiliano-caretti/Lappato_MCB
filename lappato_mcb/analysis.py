"""Pure analysis layer — parses pipeline + LAPPATO_MCB artefacts into dicts.

NO matplotlib, NO Tk, NO PDF code at module level. Everything here is a
plain function that takes a path and returns a dict / list of dicts. The
GUI, the plot module and the PDF builder all consume what these
functions return; none of them ever read CSVs themselves.

This module is fully pack-aware: domain-specific pipeline parsing and
summarisation are delegated to a ``ReportPack`` looked up via
``lappato_mcb.reports.get(run_tag)``. The framework owns only the LAPPATO_MCB-side
artefacts (papers JSONL, meta-log CSV, baseline-comparison CSV) — there
is no remaining hard-coded knowledge of any specific ML task here.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import reports
from .reports._common import coerce_float, read_csv_dicts, sample_stats


# ─── LAPPATO_MCB sidecars (run-tag agnostic) ──────────────────────────────
def parse_lappato_mcb_papers(path: Path) -> list[dict]:
    """Read ``<run_tag>_lappato_mcb_papers.jsonl`` (one paper per line)."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_meta_log(path: Path) -> list[dict]:
    """Read ``<run_tag>_lappato_mcb_meta.csv`` (one row per cycle)."""
    rows = []
    for r in read_csv_dicts(path):
        out = {"cycle": coerce_float(r.get("cycle")) or 0,
               "ts_started_iso": r.get("ts_started_iso", "")}
        for k in ("wall_seconds", "n_active_weaknesses", "n_queries_executed",
                  "n_arxiv_hits_raw", "n_openalex_hits_raw",
                  "n_crossref_hits_raw", "n_joss_hits_raw",
                  "n_arxiv_errors", "n_openalex_errors",
                  "n_crossref_errors", "n_joss_errors",
                  "n_id_dedup_blocked", "n_fp_dedup_blocked",
                  "n_papers_kept", "first_paper_seconds"):
            out[k] = coerce_float(r.get(k))
        rows.append(out)
    rows.sort(key=lambda d: d["cycle"])
    return rows


def parse_baseline_comparison(path: Path) -> dict[str, float]:
    """Read ``<run_tag>_baseline_comparison.csv`` into a flat dict."""
    out: dict[str, float] = {}
    for r in read_csv_dicts(path):
        v = coerce_float(r.get("value"))
        out[r.get("metric", "")] = v if v is not None else 0.0
    return out


def parse_weakness_cards(path: Path) -> list[dict]:
    """Read ``<run_tag>_lappato_mcb_weakness_cards.jsonl``."""
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda d: (int(d.get("cycle", 0) or 0), d.get("weakness", "")))
    return rows


# ─── Aggregations ─────────────────────────────────────────────────────
def summarise_lappato_mcb(papers: list[dict]) -> dict[str, Any]:
    """Counts useful for the report header: total, by source, by weakness."""
    by_source = Counter(p.get("source", "?") for p in papers)
    by_weakness = Counter(p.get("weakness", "?") for p in papers)
    by_year = Counter(str(p.get("year", "?")) for p in papers)
    cycles = sorted({int(p["cycle"]) for p in papers if "cycle" in p})
    return {
        "n_total": len(papers),
        "n_arxiv": by_source.get("arXiv", 0),
        "n_openalex": by_source.get("OpenAlex", 0),
        "n_crossref": by_source.get("Crossref", 0),
        "n_joss": by_source.get("JOSS", 0),
        "by_source": dict(by_source),
        "by_weakness": dict(by_weakness),
        "by_year": dict(by_year),
        "n_cycles": len(cycles),
        "first_cycle": cycles[0] if cycles else None,
        "last_cycle": cycles[-1] if cycles else None,
    }


def summarise_meta_log(meta_rows: list[dict]) -> dict[str, Any]:
    """Aggregate meta-log into headline numbers reported in the PDF."""
    if not meta_rows:
        return {
            "n_cycles": 0,
            "total_queries": 0,
            "total_raw_hits": 0,
            "total_http_errors": 0,
            "total_id_dedup": 0,
            "total_fp_dedup": 0,
            "total_kept": 0,
            "dedup_rate_id": 0.0,
            "dedup_rate_fp": 0.0,
            "first_paper_sec_median": None,
            "first_paper_sec_n": 0,
            "wall_seconds_median": 0.0,
        }

    def _sum(k: str) -> float:
        return float(sum((r.get(k) or 0) for r in meta_rows))

    total_queries = _sum("n_queries_executed")
    raw = _sum("n_arxiv_hits_raw") + _sum("n_openalex_hits_raw") \
        + _sum("n_crossref_hits_raw") + _sum("n_joss_hits_raw")
    errors = _sum("n_arxiv_errors") + _sum("n_openalex_errors") \
        + _sum("n_crossref_errors") + _sum("n_joss_errors")
    id_dup = _sum("n_id_dedup_blocked")
    fp_dup = _sum("n_fp_dedup_blocked")
    kept = _sum("n_papers_kept")
    fpsec = [r["first_paper_seconds"] for r in meta_rows
             if r.get("first_paper_seconds") not in (None, "")]
    wall = [r["wall_seconds"] for r in meta_rows
            if r.get("wall_seconds") not in (None, "")]
    fp_mean, fp_sd, fp_n = sample_stats(fpsec) if fpsec else (None, None, 0)

    return {
        "n_cycles": len(meta_rows),
        "total_queries": int(total_queries),
        "total_raw_hits": int(raw),
        "total_http_errors": int(errors),
        "total_id_dedup": int(id_dup),
        "total_fp_dedup": int(fp_dup),
        "total_kept": int(kept),
        "dedup_rate_id": (id_dup / raw) if raw > 0 else 0.0,
        "dedup_rate_fp": (fp_dup / raw) if raw > 0 else 0.0,
        "first_paper_sec_median": (
            sorted(fpsec)[len(fpsec) // 2] if fpsec else None
        ),
        "first_paper_sec_mean": fp_mean,
        "first_paper_sec_sd": fp_sd,
        "first_paper_sec_n": fp_n,
        "wall_seconds_median": (
            sorted(wall)[len(wall) // 2] if wall else 0.0
        ),
    }


def summarise_weakness_cards(cards: list[dict]) -> dict[str, Any]:
    """Compact run-level summary of evidence-gated literature routing."""
    if not cards:
        return {
            "n_cards": 0,
            "n_high_or_critical": 0,
            "by_severity": {},
            "latest_by_weakness": {},
            "total_new_vs_previous_runs": 0,
        }
    order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    latest: dict[str, dict] = {}
    by_severity = Counter()
    total_new = 0
    for c in cards:
        sev = str(c.get("severity") or "medium")
        by_severity[sev] += 1
        total_new += int(c.get("n_new_vs_previous_runs") or 0)
        latest[c.get("weakness", "?")] = c
    high = sum(
        1 for c in latest.values()
        if order.get(str(c.get("severity") or "medium"), 2) >= order["high"]
    )
    return {
        "n_cards": len(cards),
        "n_high_or_critical": high,
        "by_severity": dict(by_severity),
        "latest_by_weakness": latest,
        "total_new_vs_previous_runs": total_new,
    }


def cumulative_papers_by_cycle(papers: list[dict]) -> list[tuple[int, int]]:
    """Return ``[(cycle, cumulative_unique_papers)]`` sorted by cycle."""
    if not papers:
        return []
    by_cycle: defaultdict[int, list[str]] = defaultdict(list)
    for p in papers:
        by_cycle[int(p["cycle"])].append(p.get("id", ""))
    out = []
    seen: set[str] = set()
    for c in sorted(by_cycle):
        for pid in by_cycle[c]:
            if pid:
                seen.add(pid)
        out.append((c, len(seen)))
    return out


def top_papers(papers: list[dict], k: int = 10) -> list[dict]:
    """Return the k most-cited unique papers, ties broken by year desc."""
    seen: dict[str, dict] = {}
    for p in papers:
        pid = p.get("id", "")
        if pid and pid not in seen:
            seen[pid] = p

    def _key(d: dict):
        cited = -int(d.get("cited_by_count") or 0)
        try:
            year = -int(str(d.get("year") or "0"))
        except ValueError:
            year = 0
        return (cited, year)

    return sorted(seen.values(), key=_key)[:k]


# ─── Convenience: full bundle ──────────────────────────────────────────
def load_all(checkpoints_dir: Path, run_tag: str = "wdbc") -> dict[str, Any]:
    """Single entry point: read every artefact for one run, return one dict.

    Domain parsing and summarising are delegated to the report pack
    registered for ``run_tag``. If no pack is registered for the tag,
    a fallback empty pack is used and only the LAPPATO_MCB-side artefacts
    are populated — the framework still produces a valid bundle.
    """
    cp = Path(checkpoints_dir)
    pack = reports.get(run_tag)

    pipeline = pack.parse_all(cp)
    pipeline_summary = pack.summarise(pipeline)

    papers = parse_lappato_mcb_papers(cp / f"{run_tag}_lappato_mcb_papers.jsonl")
    meta_log = parse_meta_log(cp / f"{run_tag}_lappato_mcb_meta.csv")
    weakness_cards = parse_weakness_cards(
        cp / f"{run_tag}_lappato_mcb_weakness_cards.jsonl"
    )
    baseline_cmp = parse_baseline_comparison(
        cp / f"{run_tag}_baseline_comparison.csv"
    )

    return {
        "run_tag": run_tag,
        "pipeline": pipeline,                # dict[name -> list[dict]]
        "pipeline_summary": pipeline_summary,
        "papers": papers,
        "meta_log": meta_log,
        "weakness_cards": weakness_cards,
        "baseline_comparison": baseline_cmp,
        "lappato_mcb_summary": summarise_lappato_mcb(papers),
        "meta_summary": summarise_meta_log(meta_log),
        "weakness_card_summary": summarise_weakness_cards(weakness_cards),
        "cumulative_by_cycle": cumulative_papers_by_cycle(papers),
        "top_papers": top_papers(papers, k=10),
    }
