"""Example time-series host pipeline for LAPPATO_MCB.

Generates a deterministic synthetic series — AR(1) + seasonal +
heteroscedastic-noise — and fits a small AR(p) baseline forecaster
under rolling-origin cross-validation. The synthetic generator has a
fixed seed so the demo is bit-for-bit reproducible without any
external download.

Emits three diagnostic CSVs the time-series manifest gates on:

  - timeseries_per_horizon.csv     mean MAPE per forecast horizon h ∈ [1..H]
  - timeseries_residual_acf.csv    residual ACF at lags 1..K
  - timeseries_residual_var.csv    residual variance per rolling-origin fold

Usage:
    python examples/demo_timeseries.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Make the `lappato_mcb` package importable when this file is run directly
# as a script (no `pip install -e .` required).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb import LAPPATO_MCB  # noqa: E402
from lappato_mcb.manifests import get as get_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


# ─── Data generator ────────────────────────────────────────────────────
def synth_series(
    n: int = 1200,
    phi: float = 0.7,
    season_len: int = 24,
    season_amp: float = 1.5,
    noise_floor: float = 0.4,
    noise_growth: float = 0.0015,
    seed: int = 0,
) -> np.ndarray:
    """AR(1) + seasonal + linearly-growing-variance Gaussian noise.

    The mild noise growth is what makes the heteroscedastic_residuals
    weakness in manifests/timeseries.py fire on later folds.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    season = season_amp * np.sin(2 * np.pi * np.arange(n) / season_len)
    for t in range(1, n):
        sigma = noise_floor + noise_growth * t
        x[t] = phi * x[t - 1] + season[t] + rng.normal(0.0, sigma)
    return x


# ─── Forecaster ────────────────────────────────────────────────────────
class ARpForecaster:
    """Tiny ordinary-least-squares AR(p) forecaster with recursive rollout.

    Deliberately under-specified (no seasonal term, no GARCH) so the
    LAPPATO_MCB's manifest entries fire on the residual diagnostics.
    """
    def __init__(self, p: int = 4):
        self.p = p
        self.coefs_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray) -> "ARpForecaster":
        p = self.p
        if len(x) <= p:
            raise ValueError(f"need len(x) > p={p}, got {len(x)}")
        # Build a Hankel-style design matrix [x_{t-1}, x_{t-2}, ..., x_{t-p}].
        X = np.column_stack([x[p - k - 1: -k - 1] for k in range(p)])
        y = x[p:]
        Xb = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coefs_ = beta[1:]
        return self

    def in_sample_residuals(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        X = np.column_stack([x[p - k - 1: -k - 1] for k in range(p)])
        yhat = self.intercept_ + X @ self.coefs_
        return x[p:] - yhat

    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Recursive multi-step rollout — accumulating-error baseline."""
        p = self.p
        out = []
        buf = list(history[-p:])
        for _ in range(horizon):
            window = np.array(buf[-p:][::-1])  # x_{t-1}..x_{t-p}
            yhat = self.intercept_ + float(window @ self.coefs_)
            out.append(yhat)
            buf.append(yhat)
        return np.asarray(out)


# ─── CV protocol ───────────────────────────────────────────────────────
def rolling_origin_folds(
    series: np.ndarray, *, init: int = 600, step: int = 100, horizon: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train, test) splits with growing train and fixed-h test."""
    folds = []
    end = init
    while end + horizon <= len(series):
        train = series[:end]
        test = series[end : end + horizon]
        folds.append((train, test))
        end += step
    return folds


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main(p: int = 4, horizon: int = 12, init: int = 600, step: int = 100) -> None:
    series = synth_series()
    print(f"TS loaded: n={len(series)}  mean={series.mean():.3f}  "
          f"std={series.std():.3f}")

    manifest, run_tag = get_manifest("timeseries")
    (CHECKPOINTS / f"{run_tag}_lappato_context.json").write_text(
        json.dumps({
            "domain": "univariate time series",
            "task": "multi-horizon forecasting",
            "model_family": "AR(p) OLS",
            "dataset": "synthetic AR seasonal heteroscedastic series",
        }, indent=2),
        encoding="utf-8",
    )
    lappato_mcb = LAPPATO_MCB(
        project_root=ROOT, manifest=manifest, run_tag=run_tag, poll_interval=90.0,
    )
    lappato_mcb.start()
    print(f"LAPPATO_MCB started; log -> {lappato_mcb._log_path}")

    horizon_errors: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}
    residual_var_per_fold: list[float] = []
    last_residuals: np.ndarray | None = None

    try:
        folds = rolling_origin_folds(series, init=init, step=step, horizon=horizon)
        print(f"  {len(folds)} rolling-origin folds, horizon={horizon}")
        for fold_i, (train, test) in enumerate(folds):
            model = ARpForecaster(p=p).fit(train)
            preds = model.forecast(train, horizon=horizon)
            for h in range(horizon):
                # MAPE with a guard against tiny denominators (the
                # synthetic series can pass through zero).
                denom = max(abs(test[h]), 1e-3)
                horizon_errors[h + 1].append(abs(test[h] - preds[h]) / denom)
            res = model.in_sample_residuals(train)
            residual_var_per_fold.append(float(np.var(res)))
            last_residuals = res

            _flush_per_horizon(horizon_errors)
            _flush_residual_var(residual_var_per_fold)
            if last_residuals is not None and fold_i == len(folds) - 1:
                _flush_residual_acf(last_residuals)

            print(f"  fold {fold_i}: |train|={len(train)}  "
                  f"MAPE_h1={horizon_errors[1][-1]:.3f}  "
                  f"MAPE_h{horizon}={horizon_errors[horizon][-1]:.3f}  "
                  f"res_var={residual_var_per_fold[-1]:.3f}")

        # Final ACF flush in case the loop never hit the "last fold" branch.
        if last_residuals is not None:
            _flush_residual_acf(last_residuals)

        print(f"\nMean MAPE_h1   = {np.mean(horizon_errors[1]):.4f}")
        print(f"Mean MAPE_h{horizon}  = {np.mean(horizon_errors[horizon]):.4f}")
        print(f"Var ratio last/first = "
              f"{residual_var_per_fold[-1] / residual_var_per_fold[0]:.3f}")
    finally:
        lappato_mcb.stop(timeout=120.0)
        print("LAPPATO_MCB stopped.")


def _flush_per_horizon(acc: dict[int, list[float]]) -> None:
    rows = []
    for h, errs in sorted(acc.items()):
        if not errs:
            continue
        rows.append([h,
                     f"{np.mean(errs):.4f}",
                     f"{np.std(errs):.4f}",
                     len(errs)])
    write_csv(
        CHECKPOINTS / "timeseries_per_horizon.csv",
        ["horizon", "mape", "mape_sd", "n_folds"],
        rows,
    )


def _flush_residual_var(vars_: list[float]) -> None:
    rows = [[i, f"{v:.6f}"] for i, v in enumerate(vars_)]
    write_csv(
        CHECKPOINTS / "timeseries_residual_var.csv",
        ["fold", "variance"],
        rows,
    )


def _flush_residual_acf(residuals: np.ndarray, k_max: int = 10) -> None:
    """Sample autocorrelation up to lag k_max (Box-Jenkins-style)."""
    r = residuals - residuals.mean()
    denom = float(np.dot(r, r))
    rows = []
    if denom > 0:
        for k in range(1, k_max + 1):
            num = float(np.dot(r[:-k], r[k:]))
            rows.append([k, f"{num / denom:.4f}"])
    write_csv(
        CHECKPOINTS / "timeseries_residual_acf.csv",
        ["lag", "acf"],
        rows,
    )


if __name__ == "__main__":
    main()
