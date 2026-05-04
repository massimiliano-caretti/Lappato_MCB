"""Pure PDF assembly — no Tk, no analysis logic.

Takes the bundle returned by ``analysis.load_all`` (and optionally a
pre-built figure dict from ``plots.build_all``) and writes a multi-page
PDF report with a cover page, the figures, a meta-log table, and a
top-papers table.

The report layout is fully pack-driven: domain-specific pages and
captions come from the active ReportPack (``lappato_mcb.reports.get(run_tag)``).
The framework owns only the cover page, the LAPPATO_MCB-side figures, the
meta-log table, and the top-papers table.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from . import reports
from .plots import build_all

_PAGE = (8.27, 11.69)  # A4 portrait inches


# Captions for the framework-side figures (the same ones build_all
# emits unconditionally). Domain captions live inside each pack.
_FRAMEWORK_CAPTIONS = {
    "weakness_severity":
        "Evidence-Gated Literature Retrieval cards grouped by latest "
        "severity. Each card links a measured pipeline weakness to "
        "evidence, rendered queries, harvested papers and a transplant sketch.",
    "cumulative_papers":
        "Cumulative unique papers harvested by LAPPATO_MCB vs poll cycle "
        "number. Saturation indicates the manifest's recall ceiling.",
    "papers_per_weakness":
        "Papers per weakness id. Skewed distributions hint at over- or "
        "under-specified manifest entries.",
    "source_split":
        "Per-source contribution. Diversity across arXiv, OpenAlex and "
        "Crossref reduces single-platform bias; JOSS adds a specialised "
        "research-software channel.",
    "year_distribution":
        "Recency of harvested literature. LAPPATO_MCB filters OpenAlex and "
        "Crossref to publication_year > 2023 by default; JOSS software papers "
        "are retrieved via Crossref ISSN filtering.",
    "dedup_breakdown":
        "Decomposition of raw hits into kept vs dropped by id-dedup "
        "vs trigram-dedup. The trigram path catches cross-source "
        "duplicates the id path cannot.",
    "first_paper_latency":
        "Per-cycle latency from cycle start to first kept paper — a "
        "quantitative responsiveness metric for LAPPATO_MCB. Dashed line = "
        "sample mean ± SD across cycles.",
    "baseline_overlap":
        "Targeted manifest vs naive single-query baseline harvest. The "
        "targeted-only fraction is the concrete value-add of the "
        "manifest design over a generic query.",
}


# ─── individual page builders ──────────────────────────────────────────
def _cover_page(bundle: dict[str, Any]) -> Figure:
    fig = plt.figure(figsize=_PAGE)
    tag = bundle.get("run_tag", "")
    fig.text(0.5, 0.94, f"LAPPATO_MCB — {tag} run report",
             ha="center", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.91,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha="center", fontsize=9, color="grey")

    cs = bundle["lappato_mcb_summary"]
    ms = bundle.get("meta_summary", {})
    cmp_ = bundle.get("baseline_comparison") or {}
    pack = reports.get(tag)
    pipeline_summary = bundle.get("pipeline_summary") or {}

    # Cover block: pack-specific pipeline rows first, then LAPPATO_MCB-side
    # rows that are framework-owned.
    block: list[tuple[str, str]] = [("Run tag", tag)]
    if pack.cover_block is not None:
        block += pack.cover_block(pipeline_summary)
    block += [
        ("─", "─"),
        ("LAPPATO_MCB cycles",
         f"{cs['n_cycles']}  (first → last: "
         f"{cs['first_cycle']} → {cs['last_cycle']})"),
        ("Total papers harvested", f"{cs['n_total']}"),
        ("Per-source split",
         ", ".join(f"{k}={v}" for k, v in cs.get("by_source", {}).items())
         or "—"),
        ("Papers per weakness",
         ", ".join(f"{k}={v}" for k, v in cs["by_weakness"].items()) or "—"),
    ]
    card_summary = bundle.get("weakness_card_summary") or {}
    if card_summary:
        block += [
            ("Weakness cards", str(card_summary.get("n_cards", 0))),
            ("High/critical latest cards",
             str(card_summary.get("n_high_or_critical", 0))),
            ("New papers vs previous runs",
             str(card_summary.get("total_new_vs_previous_runs", 0))),
        ]
    if ms:
        block += [
            ("─", "─"),
            ("Meta-log cycles", str(ms.get("n_cycles", 0))),
            ("Total queries executed", str(ms.get("total_queries", 0))),
            ("Raw hits returned", str(ms.get("total_raw_hits", 0))),
            ("ID-dedup blocked", str(ms.get("total_id_dedup", 0))),
            ("Trigram-dedup blocked", str(ms.get("total_fp_dedup", 0))),
        ]
        m = ms.get("first_paper_sec_mean")
        sd = ms.get("first_paper_sec_sd")
        n = ms.get("first_paper_sec_n", 0)
        block.append((
            "Time-to-first-paper",
            f"{m:.2f} ± {sd:.2f} s (sample SD, N={n})"
            if (m is not None and n) else "—",
        ))

    if cmp_:
        j = float(cmp_.get("jaccard", 0.0))
        block += [
            ("─", "─"),
            ("Baseline comparison", "targeted vs naive single-query manifest"),
            ("Targeted-only papers", str(int(cmp_.get("targeted_only", 0)))),
            ("Naive-only papers", str(int(cmp_.get("naive_only", 0)))),
            ("Intersection (∩)", str(int(cmp_.get("n_intersection", 0)))),
            ("Jaccard overlap", f"{j:.3f}"),
        ]

    y0, dy = 0.84, 0.030
    for i, (k, v) in enumerate(block):
        y = y0 - i * dy
        fig.text(0.07, y, f"{k}", fontsize=9, fontweight="bold")
        fig.text(0.34, y, v, fontsize=9)

    fig.text(0.07, 0.06,
             "This report is auto-generated by lappato_mcb.pdf_builder from\n"
             "artefacts in checkpoints/ produced by the host pipeline. The\n"
             "domain-specific pages on the left are owned by the active\n"
             "ReportPack; the LAPPATO_MCB-side pages (papers, dedup, meta, baseline)\n"
             "are framework-owned and apply to every domain.",
             fontsize=8, color="#333333", va="top")
    return fig


def _figure_page(fig: Figure, caption: str) -> Figure:
    """Wrap a result figure into an A4 page with a caption underneath."""
    page = plt.figure(figsize=_PAGE)
    page.canvas.draw_idle()
    ax = page.add_axes((0.07, 0.30, 0.86, 0.60))
    ax.axis("off")
    fig.canvas.draw()
    img = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    import numpy as np
    arr = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 4)
    ax.imshow(arr)
    page.text(0.07, 0.22, caption, fontsize=9, color="#333333", wrap=True)
    return page


def _meta_table_page(bundle: dict[str, Any]) -> Figure:
    """One row per cycle of the meta-log — compact diagnostic table."""
    page = plt.figure(figsize=_PAGE)
    page.text(0.5, 0.94, "LAPPATO_MCB meta-log — per-cycle metrics",
              ha="center", fontsize=14, fontweight="bold")
    rows = bundle.get("meta_log", [])
    if not rows:
        page.text(0.5, 0.5, "(no meta-log available)", ha="center",
                  fontsize=11, color="grey")
        return page
    headers = ["#", "wall(s)", "active", "queries",
               "arXiv", "OpenA.", "Crossref", "JOSS",
               "id-dup", "fp-dup", "kept", "ttf(s)"]
    keys = ["cycle", "wall_seconds", "n_active_weaknesses",
            "n_queries_executed", "n_arxiv_hits_raw",
            "n_openalex_hits_raw", "n_crossref_hits_raw",
            "n_joss_hits_raw",
            "n_id_dedup_blocked", "n_fp_dedup_blocked",
            "n_papers_kept", "first_paper_seconds"]
    y0, dy = 0.88, 0.024
    x_starts = [0.035, 0.115, 0.205, 0.295, 0.385, 0.475,
                0.565, 0.655, 0.735, 0.805, 0.875, 0.935]
    for i, h in enumerate(headers):
        page.text(x_starts[i], y0, h, fontsize=9, fontweight="bold")
    for j, r in enumerate(rows[:30]):
        y = y0 - (j + 1) * dy
        for i, k in enumerate(keys):
            v = r.get(k)
            txt = "—" if v in (None, "") else (
                f"{v:.1f}" if isinstance(v, float) else str(int(v))
            )
            page.text(x_starts[i], y, txt, fontsize=8)

    ms = bundle.get("meta_summary", {})
    page.text(0.06, 0.10,
              f"Time-to-first-paper: "
              f"mean = {ms.get('first_paper_sec_mean', '—')} s "
              f"(N={ms.get('first_paper_sec_n', 0)})  ·   "
              f"id-dedup rate: {ms.get('dedup_rate_id', 0)*100:.1f}%   ·   "
              f"trigram-dedup rate: {ms.get('dedup_rate_fp', 0)*100:.1f}%",
              fontsize=9, color="#333333")
    return page


def _top_papers_page(bundle: dict[str, Any], k: int = 10) -> Figure:
    page = plt.figure(figsize=_PAGE)
    page.text(0.5, 0.94, f"Top-{k} harvested papers (by citation count)",
              ha="center", fontsize=14, fontweight="bold")
    rows = bundle["top_papers"][:k]
    if not rows:
        page.text(0.5, 0.5, "(no papers harvested)", ha="center",
                  fontsize=11, color="grey")
        return page
    y0, dy = 0.88, 0.085
    for i, p in enumerate(rows):
        y = y0 - i * dy
        title = (p.get("title") or "").strip().replace("\n", " ")
        if len(title) > 110:
            title = title[:107] + "…"
        meta = (f"[{p.get('source','?')} · {p.get('year','?')} · "
                f"cited {p.get('cited_by_count', 0)}× · weakness={p.get('weakness','?')}]")
        url = p.get("url") or p.get("id") or ""
        page.text(0.06, y, f"{i+1}.", fontsize=9, fontweight="bold")
        page.text(0.10, y, title, fontsize=9)
        page.text(0.10, y - 0.025, meta, fontsize=8, color="#555555",
                  style="italic")
        if url:
            page.text(0.10, y - 0.045, url, fontsize=7, color="#1f77b4")
    page.text(0.5, 0.04,
              "Inclusion of a paper in this list is not an endorsement; the researcher must "
              "read each abstract and judge relevance before adopting any suggestion.",
              ha="center", fontsize=7, color="grey", style="italic")
    return page


# ─── data-presence predicates (skip empty-placeholder pages) ──────────
def _has_data(key: str, bundle: dict[str, Any], pack) -> bool:
    """Return True iff the named figure has real data behind it."""
    cs = bundle.get("lappato_mcb_summary", {})
    ms = bundle.get("meta_summary", {})
    cmp_ = bundle.get("baseline_comparison") or {}
    if key in pack.captions:
        # Domain figures: present iff the pack's pipeline parsed any
        # row for any of its CSVs.
        return any(bool(rows) for rows in bundle.get("pipeline", {}).values())
    framework_predicates = {
        "weakness_severity": lambda: bool(
            (bundle.get("weakness_card_summary") or {}).get("latest_by_weakness")
        ),
        "cumulative_papers": lambda: bool(bundle.get("cumulative_by_cycle")),
        "papers_per_weakness": lambda: bool(cs.get("by_weakness")),
        "source_split": lambda: cs.get("n_total", 0) > 0,
        "year_distribution": lambda: bool(cs.get("by_year")),
        "dedup_breakdown": lambda: bool(ms.get("total_raw_hits")),
        "first_paper_latency": lambda: any(
            (r.get("first_paper_seconds") not in (None, ""))
            for r in bundle.get("meta_log", [])
        ),
        "baseline_overlap": lambda: bool(cmp_) and cmp_.get("n_union", 0) > 0,
    }
    return framework_predicates.get(key, lambda: False)()


# ─── orchestrator ──────────────────────────────────────────────────────
def build_pdf(bundle: dict[str, Any], out_path: Path,
              figures: dict[str, Figure] | None = None) -> Path:
    """Assemble the full report. Returns the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if figures is None:
        figures = build_all(bundle)

    pack = reports.get(bundle.get("run_tag", ""))
    captions = {**_FRAMEWORK_CAPTIONS, **pack.captions}

    # Page order: pack figures first (in pack-declared order, when
    # available via the pack.build_figures dict ordering — Python 3.7+
    # preserves insertion order), then framework figures.
    domain_keys = list(pack.captions.keys())
    framework_keys = [
        "weakness_severity",
        "cumulative_papers", "papers_per_weakness", "source_split",
        "year_distribution", "dedup_breakdown", "first_paper_latency",
        "baseline_overlap",
    ]
    ordered_keys = domain_keys + framework_keys

    pages: list[Figure] = [_cover_page(bundle)]
    for key in ordered_keys:
        if key in figures and _has_data(key, bundle, pack):
            cap = captions.get(key, "")
            pages.append(_figure_page(figures[key], cap))
    if bundle.get("meta_log"):
        pages.append(_meta_table_page(bundle))
    if bundle.get("top_papers"):
        pages.append(_top_papers_page(bundle))

    with PdfPages(out_path) as pdf:
        for p in pages:
            pdf.savefig(p)
            plt.close(p)
    for fig in figures.values():
        plt.close(fig)
    return out_path
