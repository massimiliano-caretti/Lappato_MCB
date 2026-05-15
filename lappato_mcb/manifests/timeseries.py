"""Time-series manifest — single-step forecasting on synthetic AR + seasonal data.

LAPPATO_MCB's evidence-gated entries fire on residual diagnostics
common to forecasting pipelines: residual autocorrelation (an under-
fit signal), heteroscedasticity (variance increases with horizon or
level), and held-out vs in-sample MAPE gap.

Diagnostic CSVs expected in ``checkpoints/`` (emitted by
``examples/demo_timeseries.py``):

  - timeseries_per_horizon.csv         per-horizon MAPE / RMSE on the test split
  - timeseries_residual_acf.csv        residual autocorrelation at lags 1..K
  - timeseries_residual_var.csv        residual variance per fold (rolling-origin)
"""
from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

RUN_TAG = "timeseries"


# ─── Activation thresholds (auditable & overridable) ───────────────────
# Same auditable-policy contract as the WDBC/NLP manifests. Defaults
# are calibrated to the synthetic AR + seasonal demo bundled in
# ``examples/demo_timeseries.py``; rebind via :func:`override_thresholds`
# for series with very different sample sizes.
DEFAULT_THRESHOLDS: dict[str, float] = {
    # Maximum absolute residual ACF over lags 1..5. The 0.20 threshold
    # is the classic Box-Jenkins ~2/sqrt(N) cut-off for N≈100, which
    # matches the synthetic series length of the bundled demo. For
    # other sample sizes, override with ``2 / sqrt(N)``.
    "acf_warning": 0.20,
    "acf_high": 0.40,
    # Heteroscedasticity proxy: ratio of residual variance in the last
    # rolling-origin fold to the first. 1.5 marks a 50% growth, the
    # threshold above which Box-Cox or a GARCH layer typically improves
    # interval coverage on stationary baselines (Hyndman & Athanasopoulos
    # 2021, ch. 8).
    "var_ratio_warning": 1.5,
    "var_ratio_high": 3.0,
    # Horizon-h MAPE / horizon-1 MAPE ratio. 1.5 is the level at which
    # recursive multi-step rollouts start losing to direct or seq2seq
    # alternatives in the M5 retrospective (Makridakis et al. 2022).
    "horizon_mape_ratio_warning": 1.5,
    "horizon_mape_ratio_high": 3.0,
}


THRESHOLDS: dict[str, float] = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    """Mutate the active threshold table — see WDBC manifest for the contract."""
    THRESHOLDS.update(dict(overrides))


# ─── Evidence checkers ────────────────────────────────────────────────
def _residual_acf_significant(csv_path: Path) -> bool:
    """Active when |ACF(lag k)| exceeds the configured warning threshold
    at any lag k ≤ 5 (default 0.20 — see ``THRESHOLDS``).

    A residual ACF that crosses ~2/sqrt(N) is the classic Box-Jenkins
    "your model is missing structure" alarm. For series with very
    different N, override ``acf_warning`` with ``2 / sqrt(N)``.
    """
    return _acf_summary(csv_path)["max_abs_acf_lag_1_5"] > THRESHOLDS["acf_warning"]


def _acf_summary(csv_path: Path) -> dict:
    vals: list[float] = []
    if csv_path.exists():
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    lag = int(r.get("lag", "0"))
                    val = abs(float(r.get("acf", "0")))
                except ValueError:
                    continue
                if 1 <= lag <= 5:
                    vals.append(val)
    return {"rows": len(vals), "max_abs_acf_lag_1_5": round(max(vals), 4) if vals else 0.0}


def _acf_severity(csv_path: Path) -> str:
    val = _acf_summary(csv_path)["max_abs_acf_lag_1_5"]
    if val >= THRESHOLDS["acf_high"]:
        return "high"
    if val >= THRESHOLDS["acf_warning"]:
        return "medium"
    return "low"


def _heteroscedastic_residuals(csv_path: Path) -> bool:
    """Active when the last-fold/first-fold residual-variance ratio
    exceeds the configured warning threshold (default 1.5 — see
    ``THRESHOLDS``).

    Rolling-origin folds on a non-stationary or volatility-clustering
    series will show an increasing residual variance; the ratio is a
    cheap heteroscedasticity proxy.
    """
    return _residual_var_summary(csv_path)["last_first_variance_ratio"] > THRESHOLDS["var_ratio_warning"]


def _residual_var_summary(csv_path: Path) -> dict:
    rows: list[float] = []
    if csv_path.exists():
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append(float(r.get("variance", "nan")))
                except ValueError:
                    continue
    ratio = (rows[-1] / rows[0]) if len(rows) >= 2 and rows[0] else 0.0
    return {"rows": len(rows), "last_first_variance_ratio": round(ratio, 4)}


def _residual_var_severity(csv_path: Path) -> str:
    ratio = _residual_var_summary(csv_path)["last_first_variance_ratio"]
    if ratio >= THRESHOLDS["var_ratio_high"]:
        return "high"
    if ratio >= THRESHOLDS["var_ratio_warning"]:
        return "medium"
    return "low"


def _horizon_degradation(csv_path: Path) -> bool:
    """Active when horizon-h MAPE / horizon-1 MAPE exceeds the
    configured warning threshold (default 1.5 — see ``THRESHOLDS``).

    Indicates the model fails to propagate information well multiple
    steps ahead — an opportunity for direct (vs recursive) multi-step
    forecasting or for sequence models.
    """
    return _horizon_summary(csv_path)["max_horizon1_mape_ratio"] > THRESHOLDS["horizon_mape_ratio_warning"]


def _horizon_summary(csv_path: Path) -> dict:
    mapes: list[tuple[int, float]] = []
    if csv_path.exists():
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    mapes.append((int(r.get("horizon", "0")),
                                  float(r.get("mape", "nan"))))
                except ValueError:
                    continue
    mapes.sort(key=lambda t: t[0])
    ratio = 0.0
    if len(mapes) >= 2 and mapes[0][1]:
        ratio = max(m for _, m in mapes[1:]) / mapes[0][1]
    return {"rows": len(mapes), "max_horizon1_mape_ratio": round(ratio, 4)}


def _horizon_severity(csv_path: Path) -> str:
    ratio = _horizon_summary(csv_path)["max_horizon1_mape_ratio"]
    if ratio >= THRESHOLDS["horizon_mape_ratio_high"]:
        return "high"
    if ratio >= THRESHOLDS["horizon_mape_ratio_warning"]:
        return "medium"
    return "low"


# ─── Manifest ─────────────────────────────────────────────────────────
MANIFEST: list[dict] = [
    {
        "id": "residual_autocorrelation",
        "title": "Residual ACF significant at short lags — model under-fits structure",
        "evidence": "timeseries_residual_acf.csv",
        "evidence_check": _residual_acf_significant,
        "evidence_summary": _acf_summary,
        "severity": _acf_severity,
        "queries": [
            "Ljung Box residual autocorrelation forecasting diagnostics",
            "ARIMA SARIMA model selection residual whitening",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Increase AR/MA orders or add seasonal terms; re-test "
            "residual ACF and the Ljung-Box Q statistic."
        ),
        "why_it_matters": (
            "Residual autocorrelation means the forecaster leaves predictable "
            "structure unused; intervals and multi-step rollouts then inherit "
            "biased errors."
        ),
        "next_checks": [
            "Run Ljung-Box test on residuals.",
            "Add seasonal terms or increase AR/MA order.",
            "Inspect residual ACF/PACF after the change.",
        ],
        "success_criteria": [
            "Max absolute ACF at lags 1-5 drops below 0.20.",
            "Rolling-origin MAPE does not worsen after residual whitening.",
        ],
        "references": [
            "Ljung Box residual diagnostics",
            "ARIMA SARIMA residual whitening",
            "forecasting diagnostics",
        ],
    },
    {
        "id": "heteroscedastic_residuals",
        "title": "Residual variance grows across rolling-origin folds",
        "evidence": "timeseries_residual_var.csv",
        "evidence_check": _heteroscedastic_residuals,
        "evidence_summary": _residual_var_summary,
        "severity": _residual_var_severity,
        "queries": [
            "GARCH heteroscedasticity time series volatility clustering",
            "log transform variance stabilisation forecasting time series",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": (
            "Apply Box-Cox / log variance-stabilising transform; if "
            "volatility clustering remains, layer a GARCH residual "
            "model on top of the mean forecaster."
        ),
        "why_it_matters": (
            "Growing residual variance means point errors and uncertainty are "
            "not exchangeable across time; this harms interval coverage and "
            "risk-aware forecasting."
        ),
        "next_checks": [
            "Plot residual variance by rolling-origin fold.",
            "Compare log/Box-Cox transforms.",
            "Test a GARCH residual model after the mean forecaster.",
        ],
        "success_criteria": [
            "Last/first residual variance ratio falls below 1.5.",
            "Prediction interval empirical coverage approaches the target level.",
        ],
        "references": [
            "GARCH volatility clustering",
            "variance stabilising transform forecasting",
            "heteroscedastic time series residuals",
        ],
    },
    {
        "id": "horizon_degradation",
        "title": "Horizon-h MAPE > 1.5x horizon-1 MAPE — recursive errors compound",
        "evidence": "timeseries_per_horizon.csv",
        "evidence_check": _horizon_degradation,
        "evidence_summary": _horizon_summary,
        "severity": _horizon_severity,
        "queries": [
            "direct vs recursive multi step forecasting strategy",
            "sequence to sequence forecasting transformer multi horizon 2025",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Switch from recursive single-step rollout to direct multi-output "
            "forecasting (one head per horizon) or to a seq2seq model."
        ),
        "why_it_matters": (
            "Recursive forecasts compound errors; horizon-specific degradation "
            "identifies when a one-step model is not fit for multi-step use."
        ),
        "next_checks": [
            "Compare recursive, direct and multi-output strategies.",
            "Plot MAPE/RMSE by horizon.",
            "Bootstrap uncertainty around horizon-specific error curves.",
        ],
        "success_criteria": [
            "Max horizon-1 MAPE ratio falls below 1.5.",
            "Long-horizon error decreases under identical rolling-origin splits.",
        ],
        "references": [
            "direct vs recursive multi-step forecasting",
            "sequence to sequence forecasting",
            "multi-horizon forecasting benchmark",
        ],
    },
    {
        "id": "ts_model_alternatives",
        "title": "ARIMA-only baseline — modern forecasters (NHITS, TFT) unrepresented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": [
            "NHITS NBEATS univariate forecasting benchmark 2025",
            "Temporal Fusion Transformer TFT time series benchmark 2025",
            "open source software time series forecasting benchmark",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": (
            "Add an N-HITS or NBEATS baseline (PyTorch Forecasting); "
            "compare on rolling-origin MAPE with bootstrap CIs."
        ),
        "why_it_matters": (
            "An ARIMA-only baseline cannot establish whether modern global or "
            "neural forecasters are unnecessary for the observed series family."
        ),
        "next_checks": [
            "Compare ARIMA, ETS, N-BEATS/N-HITS and TFT where data scale allows.",
            "Use the same rolling-origin splits for every model.",
            "Report mean error and fold dispersion.",
        ],
        "success_criteria": [
            "At least one modern baseline is reported.",
            "Model choice is justified by rolling-origin performance and stability.",
        ],
        "references": [
            "N-BEATS forecasting",
            "N-HiTS forecasting",
            "Temporal Fusion Transformer benchmark",
        ],
    },
    {
        "id": "probabilistic_forecast_missing",
        "title": "Point forecasts only — no prediction intervals reported",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": [
            "conformal prediction intervals time series 2025",
            "quantile loss probabilistic forecasting calibration",
            "open source software conformal prediction time series forecasting",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": (
            "Wrap the forecaster in a conformalised quantile regressor or "
            "use quantile loss heads; report empirical coverage at 90%."
        ),
        "why_it_matters": (
            "Point forecasts are insufficient for operational decisions; "
            "intervals expose whether the model knows when it is uncertain."
        ),
        "next_checks": [
            "Add conformal or quantile prediction intervals.",
            "Report empirical coverage by horizon.",
            "Check interval width vs coverage trade-off.",
        ],
        "success_criteria": [
            "90% intervals achieve near-90% empirical coverage.",
            "Coverage is reported separately by forecast horizon.",
        ],
        "references": [
            "conformal prediction intervals time series",
            "quantile loss probabilistic forecasting",
            "multistep conformal forecasting",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG"]
