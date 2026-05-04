"""NLP report pack — text classification diagnostics (TF-IDF + LogReg).

Owns three diagnostic CSVs emitted by ``examples/demo_nlp.py``:
  - nlp_per_class.csv          per-class precision/recall/F1
  - nlp_confusion_top.csv      top-K off-diagonal confusion pairs
  - nlp_token_importance.csv   top tokens by absolute coefficient

Headline metric: macro-F1 (uniform mean of per-class F1). Secondary:
F1 spread (max - min across classes), a non-aggregate diagnostic that
the manifest's ``per_class_f1_spread`` weakness gates on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ._common import coerce_float, read_csv_dicts, sample_stats
from ._pack import ReportPack

_FIGSIZE = (7.5, 4.5)


# ─── Parsers ───────────────────────────────────────────────────────────
def _parse_per_class(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        try:
            rows.append({
                "class": r["class"],
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1": float(r["f1"]),
            })
        except (KeyError, ValueError):
            continue
    return rows


def _parse_confusion_top(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        c = coerce_float(r.get("count"))
        s = coerce_float(r.get("share"))
        if c is None or s is None:
            continue
        rows.append({
            "true": r.get("true", ""),
            "predicted": r.get("predicted", ""),
            "count": int(c),
            "share": s,
        })
    rows.sort(key=lambda d: -d["share"])
    return rows


def _parse_token_importance(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        w = coerce_float(r.get("weight"))
        if w is None:
            continue
        rows.append({"token": r.get("token", ""), "weight": w})
    rows.sort(key=lambda d: -d["weight"])
    return rows


# ─── Summary ───────────────────────────────────────────────────────────
def _summarise(parsed: dict[str, list[dict]]) -> dict[str, Any]:
    per_class = parsed.get("per_class", [])
    conf = parsed.get("confusion_top", [])
    f1s = [r["f1"] for r in per_class]
    macro_f1, macro_f1_sd, n_classes = sample_stats(f1s)
    spread = (max(f1s) - min(f1s)) if f1s else float("nan")
    top_conf_share = conf[0]["share"] if conf else 0.0
    return {
        "n_experiments": n_classes,  # NLP "experiment unit" = a class entry
        "n_classes": n_classes,
        "headline_metric_name": "macro_F1",
        "headline_metric_unit": "fraction",
        "headline_metric_mean": macro_f1,
        "headline_metric_sd": macro_f1_sd,
        "headline_metric_n": n_classes,
        "secondary_metrics": [
            {"name": "F1 spread (max - min)", "unit": "fraction",
             "mean": spread, "sd": float("nan"), "n": n_classes},
            {"name": "top-1 confusion share", "unit": "fraction of test rows",
             "mean": top_conf_share, "sd": float("nan"), "n": len(conf)},
        ],
    }


def _cover_block(summary: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = [
        ("Dataset", "20newsgroups (4-category subset, headers/footers/quotes removed)"),
        ("Model", "TF-IDF (1-2grams, max 20k feats) + Logistic Regression (one-vs-rest)"),
        ("CV protocol",
         f"Stratified 5-fold × 3-seed; {summary.get('n_classes', 0)} classes"),
    ]
    n = summary.get("headline_metric_n", 0)
    out.append((
        "Macro F1",
        (f"{summary['headline_metric_mean']:.4f} ± "
         f"{summary['headline_metric_sd']:.4f}  (sample SD across classes, N={n})")
        if n else "—",
    ))
    for sm in summary.get("secondary_metrics", []):
        v = sm["mean"]
        txt = (f"{v:.4f}  (over {sm['n']} entries)"
               if (isinstance(v, float) and v == v) else "—")
        out.append((sm["name"], txt))
    return out


# ─── Figures ───────────────────────────────────────────────────────────
def _new_fig() -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=_FIGSIZE)
    return fig, fig.add_subplot(111)


def _empty(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="grey")


def _fig_per_class_f1(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    rows = parsed.get("per_class", [])
    if not rows:
        _empty(ax, "No per-class data yet")
        return fig
    rows = sorted(rows, key=lambda d: d["f1"])
    classes = [r["class"] for r in rows]
    f1s = [r["f1"] for r in rows]
    ax.barh(classes, f1s, color="#2a9d8f", alpha=0.85)
    mean, sd, n = sample_stats(f1s)
    ax.axvline(mean, ls="--", color="black", lw=1,
               label=f"macro-F1 = {mean:.3f} ± {sd:.3f} (N={n} classes)")
    spread = max(f1s) - min(f1s)
    ax.set_title(f"Per-class F1 — spread (max−min) = {spread:.3f}", fontsize=12)
    ax.set_xlabel("F1 (fraction)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _fig_confusion_top(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    rows = parsed.get("confusion_top", [])[:8]
    if not rows:
        _empty(ax, "No confusion-pair data yet")
        return fig
    labels = [f"{r['true']} → {r['predicted']}" for r in rows][::-1]
    shares = [r["share"] * 100.0 for r in rows][::-1]
    ax.barh(labels, shares, color="#e76f51", alpha=0.9)
    ax.axvline(8.0, ls="--", color="red", lw=1,
               label="LAPPATO_MCB threshold = 8% of test rows")
    ax.set_xlabel("Share of test rows (%)", fontsize=10)
    ax.set_title("Top-K confusable class pairs (off-diagonal)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _fig_token_importance(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    rows = parsed.get("token_importance", [])[:15]
    if not rows:
        _empty(ax, "No token-importance data yet")
        return fig
    tokens = [r["token"] for r in rows][::-1]
    weights = [r["weight"] for r in rows][::-1]
    ax.barh(tokens, weights, color="#264653", alpha=0.85)
    ax.set_xlabel("Mean |coef| pooled across folds", fontsize=10)
    ax.set_title("Top-15 tokens by absolute coefficient (one-vs-rest)",
                 fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _build_figures(parsed: dict[str, list[dict]]) -> dict[str, Figure]:
    return {
        "nlp_per_class_f1": _fig_per_class_f1(parsed),
        "nlp_confusion_top": _fig_confusion_top(parsed),
        "nlp_token_importance": _fig_token_importance(parsed),
    }


CAPTIONS = {
    "nlp_per_class_f1":
        "Per-class F1 sorted ascending. Dashed line = macro-F1 with sample "
        "SD across classes. Spread (max − min) > 0.15 triggers the "
        "per_class_f1_spread weakness in the manifest.",
    "nlp_confusion_top":
        "Top-K off-diagonal confusion pairs (true → predicted) by share "
        "of test rows. Red dashed line = 8% threshold above which the "
        "confusable_classes weakness fires.",
    "nlp_token_importance":
        "Top-15 tokens by absolute coefficient pooled across folds. "
        "Short tokens dominating the top suggest stop-/function-word "
        "leakage (the stopword_dominance weakness gates on top-3 length < 5).",
}


PACK = ReportPack(
    run_tag="nlp",
    pipeline_csvs={
        "per_class": "nlp_per_class.csv",
        "confusion_top": "nlp_confusion_top.csv",
        "token_importance": "nlp_token_importance.csv",
    },
    parsers={
        "per_class": _parse_per_class,
        "confusion_top": _parse_confusion_top,
        "token_importance": _parse_token_importance,
    },
    summarise=_summarise,
    build_figures=_build_figures,
    captions=CAPTIONS,
    cover_block=_cover_block,
)


__all__ = ["PACK"]
