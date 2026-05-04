"""WDBC report pack — LightGBM + 5-fold × 3-seed CV diagnostics.

Owns three diagnostic CSVs emitted by ``examples/demo_wdbc.py``:
  - wdbc_per_fold.csv          BAC + per-class precision/recall per fold
  - wdbc_calibration.csv       Brier per fold
  - wdbc_feature_importance.csv pooled LightGBM gain (top-10)

Headline metric: balanced accuracy. Secondary: Brier. Statistics use
the sample standard deviation (Bessel's correction) and report N
explicitly so consumers of the cover page can recover confidence
intervals downstream.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ._common import read_csv_dicts, sample_stats
from ._pack import ReportPack

_FIGSIZE = (7.5, 4.5)


# ─── Parsers ───────────────────────────────────────────────────────────
def _parse_per_fold(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        try:
            rows.append({
                "seed": int(r["seed"]),
                "fold": int(r["fold"]),
                "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]),
                "bac": float(r["bac"]),
                "precision_M": float(r["precision_M"]),
                "recall_M": float(r["recall_M"]),
                "precision_B": float(r["precision_B"]),
                "recall_B": float(r["recall_B"]),
            })
        except (KeyError, ValueError):
            continue
    return rows


def _parse_calibration(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        try:
            rows.append({
                "seed": int(r["seed"]),
                "fold": int(r["fold"]),
                "brier": float(r["brier"]),
            })
        except (KeyError, ValueError):
            continue
    return rows


def _parse_importance(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        try:
            rows.append({"feature": r["feature"], "gain": float(r["gain"])})
        except (KeyError, ValueError):
            continue
    rows.sort(key=lambda d: -d["gain"])
    return rows


# ─── Summary ───────────────────────────────────────────────────────────
def _summarise(parsed: dict[str, list[dict]]) -> dict[str, Any]:
    per_fold = parsed.get("per_fold", [])
    calib = parsed.get("calibration", [])
    bac_mean, bac_sd, n_bac = sample_stats([r["bac"] for r in per_fold])
    bri_mean, bri_sd, n_bri = sample_stats([r["brier"] for r in calib])
    return {
        "n_experiments": len(per_fold),
        "n_seeds": len({r["seed"] for r in per_fold}),
        "n_folds": len({r["fold"] for r in per_fold}),
        "headline_metric_name": "balanced_accuracy",
        "headline_metric_unit": "fraction",
        "headline_metric_mean": bac_mean,
        "headline_metric_sd": bac_sd,
        "headline_metric_n": n_bac,
        "secondary_metrics": [
            {"name": "brier", "unit": "score (lower is better)",
             "mean": bri_mean, "sd": bri_sd, "n": n_bri},
        ],
    }


def _cover_block(summary: dict[str, Any]) -> list[tuple[str, str]]:
    """Cover-page rows specific to WDBC; framework prepends the run-tag row."""
    out: list[tuple[str, str]] = [
        ("Dataset",
         "Wisconsin Breast Cancer Diagnostic (sklearn; 569 × 30, 212 M / 357 B)"),
        ("Model", "LightGBM (lr=0.05, num_leaves=31, n_est=500, early_stop=50)"),
        ("CV protocol",
         f"Stratified {summary.get('n_folds', 0)}-fold × "
         f"{summary.get('n_seeds', 0)}-seed = "
         f"{summary.get('n_experiments', 0)} experiments"),
    ]
    bac_n = summary.get("headline_metric_n", 0)
    out.append((
        "Mean balanced accuracy",
        (f"{summary['headline_metric_mean']:.4f} ± "
         f"{summary['headline_metric_sd']:.4f}  (sample SD, N={bac_n})")
        if bac_n else "—",
    ))
    for sm in summary.get("secondary_metrics", []):
        out.append((
            f"Mean {sm['name']}",
            f"{sm['mean']:.4f} ± {sm['sd']:.4f}  (sample SD, N={sm['n']})"
            if sm["n"] else "—",
        ))
    return out


# ─── Figures ───────────────────────────────────────────────────────────
def _new_fig() -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=_FIGSIZE)
    return fig, fig.add_subplot(111)


def _empty(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="grey")


def _fig_bac_per_fold(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    per_fold = parsed.get("per_fold", [])
    if not per_fold:
        _empty(ax, "No per-fold data yet")
        return fig
    by_seed: defaultdict[int, list[float]] = defaultdict(list)
    for r in per_fold:
        by_seed[r["seed"]].append(r["bac"])
    seeds = sorted(by_seed)
    for i, s in enumerate(seeds):
        ax.scatter([i] * len(by_seed[s]), by_seed[s],
                   s=40, alpha=0.75, label=f"seed {s} (n={len(by_seed[s])})")
    bac_mean, bac_sd, n = sample_stats([r["bac"] for r in per_fold])
    ax.axhline(bac_mean, ls="--", color="black", lw=1,
               label=f"overall mean = {bac_mean:.3f} ± {bac_sd:.3f} (N={n})")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("Balanced accuracy (fraction)", fontsize=10)
    ax.set_title("Balanced accuracy per fold (LightGBM × WDBC)", fontsize=12)
    all_bac = [r["bac"] for r in per_fold]
    ax.set_ylim(min(0.85, min(all_bac) - 0.02), 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _fig_brier_per_fold(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    calib = parsed.get("calibration", [])
    if not calib:
        _empty(ax, "No calibration data yet")
        return fig
    xs = list(range(len(calib)))
    ys = [r["brier"] for r in calib]
    labels = [f"s{r['seed']}f{r['fold']}" for r in calib]
    ax.bar(xs, ys, color="#3a7ca5", alpha=0.85)
    ax.axhline(0.10, ls="--", color="red", lw=1,
               label="LAPPATO_MCB threshold = 0.10")
    mean, sd, n = sample_stats(ys)
    ax.axhline(mean, ls=":", color="black", lw=1,
               label=f"mean = {mean:.4f} ± {sd:.4f} (N={n})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_ylabel("Brier score (lower is better)", fontsize=10)
    ax.set_title("Brier score per (seed, fold)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_feature_importance(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    importance = parsed.get("importance", [])[:10]
    if not importance:
        _empty(ax, "No feature importance yet")
        return fig
    names = [r["feature"] for r in importance][::-1]
    gains = [r["gain"] for r in importance][::-1]
    ax.barh(names, gains, color="#2a9d8f", alpha=0.85)
    ax.set_xlabel("LightGBM gain (pooled across folds, log-likelihood units)",
                  fontsize=10)
    ax.set_title(f"Top-{len(importance)} features by importance", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _build_figures(parsed: dict[str, list[dict]]) -> dict[str, Figure]:
    return {
        "wdbc_bac_per_fold": _fig_bac_per_fold(parsed),
        "wdbc_brier_per_fold": _fig_brier_per_fold(parsed),
        "wdbc_feature_importance": _fig_feature_importance(parsed),
    }


CAPTIONS = {
    "wdbc_bac_per_fold":
        "Balanced accuracy per (seed, fold). Dashed line = overall sample "
        "mean ± sample SD (Bessel-corrected) with N = number of folds.",
    "wdbc_brier_per_fold":
        "Brier score per (seed, fold). Red dashed line = 0.10 alarm "
        "threshold below which LAPPATO_MCB's low_calibration weakness "
        "deactivates. Dotted line = sample mean ± SD with N folds.",
    "wdbc_feature_importance":
        "Top-10 features by pooled LightGBM gain across all folds. "
        "Used by LAPPATO_MCB's marker_redundancy evidence check (top-3 "
        "grouping test on WDBC's mean / se / worst suffix triplets).",
}


PACK = ReportPack(
    run_tag="wdbc",
    pipeline_csvs={
        "per_fold": "wdbc_per_fold.csv",
        "calibration": "wdbc_calibration.csv",
        "importance": "wdbc_feature_importance.csv",
    },
    parsers={
        "per_fold": _parse_per_fold,
        "calibration": _parse_calibration,
        "importance": _parse_importance,
    },
    summarise=_summarise,
    build_figures=_build_figures,
    captions=CAPTIONS,
    cover_block=_cover_block,
)


__all__ = ["PACK"]
