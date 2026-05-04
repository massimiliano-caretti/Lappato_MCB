"""Time-series report pack — univariate forecasting diagnostics.

Owns three diagnostic CSVs emitted by ``examples/demo_timeseries.py``:
  - timeseries_per_horizon.csv     mean MAPE per forecast horizon h ∈ [1..H]
  - timeseries_residual_acf.csv    sample residual ACF at lags 1..K
  - timeseries_residual_var.csv    in-sample residual variance per rolling-origin fold

Headline metric: MAPE at horizon 1 (immediate next step). Secondary:
horizon-degradation ratio MAPE_H / MAPE_1 (a non-aggregate diagnostic
the manifest's ``horizon_degradation`` weakness gates on).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ._common import coerce_float, read_csv_dicts
from ._pack import ReportPack

_FIGSIZE = (7.5, 4.5)


# ─── Parsers ───────────────────────────────────────────────────────────
def _parse_per_horizon(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        h = coerce_float(r.get("horizon"))
        m = coerce_float(r.get("mape"))
        if h is None or m is None:
            continue
        sd = coerce_float(r.get("mape_sd")) or 0.0
        n = coerce_float(r.get("n_folds")) or 0.0
        rows.append({"horizon": int(h), "mape": m,
                     "mape_sd": sd, "n_folds": int(n)})
    rows.sort(key=lambda d: d["horizon"])
    return rows


def _parse_residual_acf(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        lag = coerce_float(r.get("lag"))
        a = coerce_float(r.get("acf"))
        if lag is None or a is None:
            continue
        rows.append({"lag": int(lag), "acf": a})
    rows.sort(key=lambda d: d["lag"])
    return rows


def _parse_residual_var(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_dicts(path):
        f = coerce_float(r.get("fold"))
        v = coerce_float(r.get("variance"))
        if f is None or v is None:
            continue
        rows.append({"fold": int(f), "variance": v})
    rows.sort(key=lambda d: d["fold"])
    return rows


# ─── Summary ───────────────────────────────────────────────────────────
def _summarise(parsed: dict[str, list[dict]]) -> dict[str, Any]:
    per_h = parsed.get("per_horizon", [])
    var_rows = parsed.get("residual_var", [])
    horizons = [r["horizon"] for r in per_h]
    if per_h:
        mape_h1 = next((r["mape"] for r in per_h if r["horizon"] == 1),
                       per_h[0]["mape"])
        mape_h_max = per_h[-1]["mape"] if per_h else float("nan")
        deg_ratio = (mape_h_max / mape_h1) if mape_h1 else float("nan")
        n_folds = per_h[0].get("n_folds", 0) if per_h else 0
        sd_h1 = next((r["mape_sd"] for r in per_h if r["horizon"] == 1),
                     0.0)
    else:
        mape_h1 = float("nan")
        deg_ratio = float("nan")
        n_folds = 0
        sd_h1 = float("nan")
    var_ratio = (
        var_rows[-1]["variance"] / var_rows[0]["variance"]
        if len(var_rows) >= 2 and var_rows[0]["variance"] > 0
        else float("nan")
    )
    return {
        "n_experiments": n_folds,
        "horizons_max": max(horizons) if horizons else 0,
        "headline_metric_name": "MAPE_h1",
        "headline_metric_unit": "fraction",
        "headline_metric_mean": mape_h1,
        "headline_metric_sd": sd_h1,
        "headline_metric_n": n_folds,
        "secondary_metrics": [
            {"name": "MAPE_hH / MAPE_h1", "unit": "ratio (>1 means degradation)",
             "mean": deg_ratio, "sd": float("nan"), "n": n_folds},
            {"name": "Var(res) last/first fold", "unit": "ratio (>1 means heteroscedastic)",
             "mean": var_ratio, "sd": float("nan"), "n": len(var_rows)},
        ],
    }


def _cover_block(summary: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = [
        ("Dataset", "Synthetic AR(1) + seasonal + heteroscedastic noise (seed=0, n=1200)"),
        ("Model", "AR(p=4) ordinary least squares, recursive multi-step rollout"),
        ("CV protocol",
         f"Rolling-origin CV; {summary.get('n_experiments', 0)} folds; "
         f"horizon ≤ {summary.get('horizons_max', 0)}"),
    ]
    n = summary.get("headline_metric_n", 0)
    out.append((
        "MAPE at horizon 1",
        (f"{summary['headline_metric_mean']:.4f} ± "
         f"{summary['headline_metric_sd']:.4f}  (sample SD across {n} folds)")
        if n else "—",
    ))
    for sm in summary.get("secondary_metrics", []):
        v = sm["mean"]
        if isinstance(v, float) and v == v:
            out.append((sm["name"], f"{v:.3f}  ({sm['unit']})"))
        else:
            out.append((sm["name"], "—"))
    return out


# ─── Figures ───────────────────────────────────────────────────────────
def _new_fig() -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=_FIGSIZE)
    return fig, fig.add_subplot(111)


def _empty(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="grey")


def _fig_mape_per_horizon(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    rows = parsed.get("per_horizon", [])
    if not rows:
        _empty(ax, "No per-horizon MAPE yet")
        return fig
    xs = [r["horizon"] for r in rows]
    ys = [r["mape"] for r in rows]
    sds = [r["mape_sd"] for r in rows]
    ax.errorbar(xs, ys, yerr=sds, marker="o", color="#264653", lw=2,
                capsize=3, label=f"mean ± SD over {rows[0].get('n_folds', 0)} folds")
    if ys[0] > 0:
        ax.axhline(1.5 * ys[0], ls="--", color="red", lw=1,
                   label="LAPPATO_MCB threshold = 1.5 × MAPE_h1")
    ax.set_xlabel("Forecast horizon (steps)", fontsize=10)
    ax.set_ylabel("MAPE (fraction, lower is better)", fontsize=10)
    ax.set_title("MAPE vs forecast horizon (rolling-origin CV)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def _fig_residual_acf(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    rows = parsed.get("residual_acf", [])
    if not rows:
        _empty(ax, "No residual ACF yet")
        return fig
    xs = [r["lag"] for r in rows]
    ys = [r["acf"] for r in rows]
    ax.bar(xs, ys, color="#3a7ca5", alpha=0.85, width=0.6)
    ax.axhline(0.20, ls="--", color="red", lw=1,
               label="LAPPATO_MCB threshold |ACF| = 0.20")
    ax.axhline(-0.20, ls="--", color="red", lw=1)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("Lag k", fontsize=10)
    ax.set_ylabel("Sample ACF of in-sample residuals", fontsize=10)
    ax.set_title("Residual autocorrelation function (Box-Jenkins diagnostic)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def _fig_residual_variance(parsed: dict[str, list[dict]]) -> Figure:
    fig, ax = _new_fig()
    rows = parsed.get("residual_var", [])
    if not rows:
        _empty(ax, "No residual variance per fold yet")
        return fig
    xs = [r["fold"] for r in rows]
    ys = [r["variance"] for r in rows]
    ax.plot(xs, ys, marker="s", color="#e76f51", lw=2)
    if len(ys) >= 2 and ys[0] > 0:
        ax.axhline(1.5 * ys[0], ls="--", color="red", lw=1,
                   label="LAPPATO_MCB threshold = 1.5 × Var(res)_fold0")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Rolling-origin fold #", fontsize=10)
    ax.set_ylabel("Variance of in-sample residuals", fontsize=10)
    ax.set_title("Residual variance per fold (heteroscedasticity check)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _build_figures(parsed: dict[str, list[dict]]) -> dict[str, Figure]:
    return {
        "timeseries_mape_per_horizon": _fig_mape_per_horizon(parsed),
        "timeseries_residual_acf": _fig_residual_acf(parsed),
        "timeseries_residual_variance": _fig_residual_variance(parsed),
    }


CAPTIONS = {
    "timeseries_mape_per_horizon":
        "Mean MAPE per forecast horizon with sample-SD error bars over "
        "rolling-origin folds. Red dashed line = 1.5 × MAPE_h1 threshold; "
        "values above it trigger the horizon_degradation weakness.",
    "timeseries_residual_acf":
        "Sample autocorrelation function of in-sample residuals (last "
        "fitted fold). Red dashed lines = ±0.20 alarm bands; |ACF(k)| > 0.20 "
        "at any lag k ≤ 5 triggers the residual_autocorrelation weakness.",
    "timeseries_residual_variance":
        "In-sample residual variance per rolling-origin fold. Red dashed "
        "line = 1.5 × variance of fold 0; an upward drift past it triggers "
        "the heteroscedastic_residuals weakness.",
}


PACK = ReportPack(
    run_tag="timeseries",
    pipeline_csvs={
        "per_horizon": "timeseries_per_horizon.csv",
        "residual_acf": "timeseries_residual_acf.csv",
        "residual_var": "timeseries_residual_var.csv",
    },
    parsers={
        "per_horizon": _parse_per_horizon,
        "residual_acf": _parse_residual_acf,
        "residual_var": _parse_residual_var,
    },
    summarise=_summarise,
    build_figures=_build_figures,
    captions=CAPTIONS,
    cover_block=_cover_block,
)


__all__ = ["PACK"]
