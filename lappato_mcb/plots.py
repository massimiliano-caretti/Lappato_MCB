"""Pure plotting layer — matplotlib Figures from analysis output.

NO Tk, NO file I/O at module level (the PDF builder writes; here we
only build). Each function takes plain Python dicts/lists (the output
of ``analysis.py``) and returns a ``matplotlib.figure.Figure``.

This module owns only the *framework* figures — those that depend
solely on LAPPATO_MCB-side artefacts (papers, meta-log, baseline). The
domain-specific figures come from the active ReportPack via
``build_all`` (which delegates to ``pack.build_figures(parsed)``).
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; the GUI embeds via FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import reports

_FIGSIZE = (7.5, 4.5)
_TITLE_FS = 12
_LABEL_FS = 10


def _new_fig() -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=_FIGSIZE)
    return fig, fig.add_subplot(111)


def _empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=_LABEL_FS, color="grey")


# ─── Framework figures (LAPPATO_MCB-side, domain-agnostic) ────────────────
def fig_cumulative_papers(curve: list[tuple[int, int]]) -> Figure:
    """Cumulative unique papers harvested vs LAPPATO_MCB cycle number."""
    fig, ax = _new_fig()
    if not curve:
        _empty(ax, "No LAPPATO_MCB log yet")
        return fig
    xs = [c for c, _ in curve]
    ys = [n for _, n in curve]
    ax.plot(xs, ys, marker="o", color="#e76f51", lw=2)
    ax.set_xlabel("LAPPATO_MCB cycle #", fontsize=_LABEL_FS)
    ax.set_ylabel("Unique papers harvested", fontsize=_LABEL_FS)
    ax.set_title("Cumulative literature harvested vs poll cycle", fontsize=_TITLE_FS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_papers_per_weakness(by_weakness: dict[str, int]) -> Figure:
    """Bar chart of unique papers found per weakness id."""
    fig, ax = _new_fig()
    if not by_weakness:
        _empty(ax, "No LAPPATO_MCB data yet")
        return fig
    items = sorted(by_weakness.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    ax.bar(names, counts, color="#264653", alpha=0.9)
    ax.set_ylabel("Papers harvested", fontsize=_LABEL_FS)
    ax.set_title("Papers per weakness (manifest entry)", fontsize=_TITLE_FS)
    ax.tick_params(axis="x", rotation=30)
    for tl in ax.get_xticklabels():
        tl.set_horizontalalignment("right")
        tl.set_fontsize(8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_source_split(by_source: dict[str, int]) -> Figure:
    """N-bar comparison of source contributions (arXiv vs OpenAlex vs Crossref vs ...).

    Generic over the source dict so adding a 4th source requires no
    changes here.
    """
    fig, ax = _new_fig()
    items = [(k, v) for k, v in by_source.items() if v > 0]
    if not items:
        _empty(ax, "No LAPPATO_MCB data yet")
        return fig
    items.sort(key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    palette = ["#e9c46a", "#f4a261", "#2a9d8f", "#264653", "#8d99ae"]
    colors = [palette[i % len(palette)] for i in range(len(items))]
    ax.bar(names, counts, color=colors, alpha=0.9)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.02 + 0.1, str(v),
                ha="center", fontsize=_LABEL_FS, fontweight="bold")
    ax.set_ylabel("Unique papers", fontsize=_LABEL_FS)
    ax.set_title("Harvest split per source", fontsize=_TITLE_FS)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_year_distribution(by_year: dict[str, int]) -> Figure:
    """Distribution of paper publication years."""
    fig, ax = _new_fig()
    if not by_year:
        _empty(ax, "No LAPPATO_MCB data yet")
        return fig
    items = sorted((k, v) for k, v in by_year.items() if k.isdigit())
    if not items:
        _empty(ax, "No parseable years")
        return fig
    ax.bar([k for k, _ in items], [v for _, v in items],
           color="#8d99ae", alpha=0.9)
    ax.set_xlabel("Publication year", fontsize=_LABEL_FS)
    ax.set_ylabel("Papers harvested", fontsize=_LABEL_FS)
    ax.set_title("Recency of harvested literature", fontsize=_TITLE_FS)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_dedup_breakdown(meta_summary: dict) -> Figure:
    """Stacked bar: raw hits = kept + id-dedup + fp-dedup."""
    fig, ax = _new_fig()
    raw = meta_summary.get("total_raw_hits", 0)
    if raw == 0:
        _empty(ax, "No meta-log yet")
        return fig
    kept = meta_summary.get("total_kept", 0)
    id_d = meta_summary.get("total_id_dedup", 0)
    fp_d = meta_summary.get("total_fp_dedup", 0)
    cats = ["raw hits"]
    ax.bar(cats, [kept], color="#2a9d8f", label=f"kept ({kept})")
    ax.bar(cats, [id_d], bottom=[kept],
           color="#e9c46a", label=f"id-dedup ({id_d})")
    ax.bar(cats, [fp_d], bottom=[kept + id_d],
           color="#e76f51", label=f"fp-dedup ({fp_d})")
    ax.set_ylabel("Papers", fontsize=_LABEL_FS)
    ax.set_title("Dedup breakdown — raw hits decomposed",
                 fontsize=_TITLE_FS)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    rates = (
        f"id-dedup rate: {meta_summary.get('dedup_rate_id', 0)*100:.1f}%   "
        f"fp-dedup rate: {meta_summary.get('dedup_rate_fp', 0)*100:.1f}%"
    )
    ax.text(0.02, 0.98, rates, transform=ax.transAxes,
            fontsize=8, color="#333333", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    fig.tight_layout()
    return fig


def fig_first_paper_latency(meta_log: list[dict],
                            meta_summary: dict | None = None) -> Figure:
    """Per-cycle latency-to-first-kept-paper (LAPPATO_MCB responsiveness)."""
    fig, ax = _new_fig()
    if not meta_log:
        _empty(ax, "No meta-log yet")
        return fig
    xs, ys = [], []
    for r in meta_log:
        v = r.get("first_paper_seconds")
        if v is None or v == "":
            continue
        try:
            xs.append(int(r["cycle"]))
            ys.append(float(v))
        except (ValueError, TypeError):
            continue
    if not xs:
        _empty(ax, "No cycles produced a paper")
        return fig
    ax.plot(xs, ys, marker="s", color="#264653", lw=2,
            label=f"per-cycle observation (N={len(ys)})")
    if meta_summary:
        m = meta_summary.get("first_paper_sec_mean")
        sd = meta_summary.get("first_paper_sec_sd")
        if m is not None:
            ax.axhline(m, ls="--", color="black", lw=1,
                       label=f"mean = {m:.2f} ± {sd:.2f} s (sample SD)")
    ax.set_xlabel("LAPPATO_MCB cycle #", fontsize=_LABEL_FS)
    ax.set_ylabel("Seconds to first kept paper (lower is better)",
                  fontsize=_LABEL_FS)
    ax.set_title("Time-to-first-paper per cycle",
                 fontsize=_TITLE_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def fig_baseline_overlap(cmp: dict[str, float]) -> Figure:
    """Three-bar Venn-style summary of targeted-vs-naive harvest."""
    fig, ax = _new_fig()
    if not cmp or cmp.get("n_union", 0) == 0:
        _empty(ax, "No baseline comparison yet — run compare_targeted_vs_naive.py")
        return fig
    cats = ["targeted-only", "∩ both", "naive-only"]
    vals = [int(cmp.get("targeted_only", 0)),
            int(cmp.get("n_intersection", 0)),
            int(cmp.get("naive_only", 0))]
    colors = ["#2a9d8f", "#8d99ae", "#e76f51"]
    ax.bar(cats, vals, color=colors, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02 + 0.1, str(v),
                ha="center", fontsize=_LABEL_FS, fontweight="bold")
    ax.set_ylabel("Unique papers", fontsize=_LABEL_FS)
    j = float(cmp.get("jaccard", 0.0))
    ax.set_title(
        f"Targeted vs naive harvest — Jaccard = {j:.3f}",
        fontsize=_TITLE_FS,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_weakness_severity(card_summary: dict) -> Figure:
    """Latest active weakness cards by severity."""
    fig, ax = _new_fig()
    latest = card_summary.get("latest_by_weakness") or {}
    if not latest:
        _empty(ax, "No weakness cards yet")
        return fig
    order = ["info", "low", "medium", "high", "critical"]
    counts = {k: 0 for k in order}
    for c in latest.values():
        sev = str(c.get("severity") or "medium")
        counts[sev if sev in counts else "medium"] += 1
    colors = {
        "info": "#8d99ae",
        "low": "#2a9d8f",
        "medium": "#e9c46a",
        "high": "#f4a261",
        "critical": "#e76f51",
    }
    vals = [counts[k] for k in order]
    ax.bar(order, vals, color=[colors[k] for k in order], alpha=0.9)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v + max(vals) * 0.02 + 0.1, str(v),
                    ha="center", fontsize=_LABEL_FS, fontweight="bold")
    ax.set_ylabel("Latest active weakness cards", fontsize=_LABEL_FS)
    ax.set_title("Evidence-gated literature routing — severity", fontsize=_TITLE_FS)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ─── Orchestrator ──────────────────────────────────────────────────────
def build_all(bundle: dict[str, Any]) -> dict[str, Figure]:
    """Build every figure from a single ``analysis.load_all`` bundle.

    Returns a dict keyed by stable figure id. Domain-specific figures
    (provided by the active ReportPack) come first; framework figures
    follow in a stable order.
    """
    cs = bundle["lappato_mcb_summary"]
    ms = bundle.get("meta_summary", {})
    cmp_ = bundle.get("baseline_comparison") or {}

    # Domain figures from the active pack.
    pack = reports.get(bundle.get("run_tag", ""))
    domain_figs = pack.build_figures(bundle.get("pipeline", {}))

    framework = {
        "weakness_severity": fig_weakness_severity(
            bundle.get("weakness_card_summary", {}),
        ),
        "cumulative_papers": fig_cumulative_papers(bundle["cumulative_by_cycle"]),
        "papers_per_weakness": fig_papers_per_weakness(cs["by_weakness"]),
        "source_split": fig_source_split(cs.get("by_source") or {}),
        "year_distribution": fig_year_distribution(cs["by_year"]),
        "dedup_breakdown": fig_dedup_breakdown(ms),
        "first_paper_latency": fig_first_paper_latency(
            bundle.get("meta_log", []), ms,
        ),
        "baseline_overlap": fig_baseline_overlap(cmp_),
    }
    # Domain first, then framework — both contribute to the GUI tabs
    # and the PDF figure pages.
    return {**domain_figs, **framework}


__all__ = [
    "fig_cumulative_papers",
    "fig_papers_per_weakness",
    "fig_source_split",
    "fig_year_distribution",
    "fig_dedup_breakdown",
    "fig_first_paper_latency",
    "fig_baseline_overlap",
    "fig_weakness_severity",
    "build_all",
]
