"""Example WDBC host pipeline for LAPPATO_MCB.

Loads the Wisconsin Breast Cancer Diagnostic dataset directly from
``sklearn.datasets.load_breast_cancer`` (the same UCI dataset, mirrored
inside scikit-learn), trains a LightGBM binary classifier under 5-fold
x 3-seed stratified cross-validation, and emits three diagnostic CSVs
into ``checkpoints/`` that LAPPATO_MCB daemon polls in parallel:

  - wdbc_per_fold.csv          per-fold metrics (BAC, precision/recall per class)
  - wdbc_calibration.csv       per-fold Brier scores
  - wdbc_feature_importance.csv pooled gain importance (top-10)

LAPPATO_MCB is launched on the background thread before training
starts and stopped after training finishes — the standard two-line
integration into a host ML pipeline.

Usage:
    python examples/demo_wdbc.py

Requires: scikit-learn, lightgbm. LAPPATO_MCB itself is stdlib-only.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

# Make the `lappato_mcb` package importable when this file is run directly
# as a script (no `pip install -e .` required).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb import LAPPATO_MCB  # noqa: E402
from lappato_mcb.manifests import get as get_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


def _sklearn_to_underscore_name(name: str) -> str:
    """Map a sklearn WDBC feature name to the historical CSV convention.

    sklearn returns names like ``"mean radius"`` / ``"radius error"`` /
    ``"worst radius"``; the manifest's ``marker_redundancy`` evidence
    check expects the underscore form ``"radius_mean"`` / ``"radius_se"``
    / ``"radius_worst"`` (and tests group membership by the trailing
    suffix). Converting once at load time keeps the manifest unchanged.
    """
    parts = name.split()
    if parts and parts[0] == "mean":
        return "_".join(parts[1:]) + "_mean"
    if parts and parts[-1] == "error":
        return "_".join(parts[:-1]) + "_se"
    if parts and parts[0] == "worst":
        return "_".join(parts[1:]) + "_worst"
    return name.replace(" ", "_")


def load_wdbc() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load WDBC via ``sklearn.datasets.load_breast_cancer``.

    Convention: M (malignant) -> 1, B (benign) -> 0. Note that sklearn
    uses the opposite encoding (target=0 is malignant), so we flip.
    """
    bc = load_breast_cancer()
    X = bc.data.astype(np.float64)
    y = (bc.target == 0).astype(np.int32)  # sklearn: 0 == malignant -> M
    feature_cols = [_sklearn_to_underscore_name(n) for n in bc.feature_names]
    return X, y, feature_cols


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main(seeds: tuple[int, ...] = (0, 1, 2), n_folds: int = 5) -> None:
    X, y, feature_names = load_wdbc()
    print(f"WDBC loaded: n={len(y)}  M={int(y.sum())}  B={int((1 - y).sum())}  "
          f"features={X.shape[1]}")

    manifest, run_tag = get_manifest("wdbc")
    (CHECKPOINTS / f"{run_tag}_lappato_context.json").write_text(
        json.dumps({
            "domain": "clinical tabular",
            "task": "binary classification",
            "model_family": "LightGBM",
            "dataset": "Wisconsin Breast Cancer Diagnostic",
        }, indent=2),
        encoding="utf-8",
    )
    lappato_mcb = LAPPATO_MCB(
        project_root=ROOT, manifest=manifest, run_tag=run_tag, poll_interval=90.0,
    )
    lappato_mcb.start()
    print(f"LAPPATO_MCB started; log -> {lappato_mcb._log_path}")

    per_fold_rows: list[list] = []
    calibration_rows: list[list] = []
    importance_acc: defaultdict[str, float] = defaultdict(float)

    try:
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_i, (tr, te) in enumerate(skf.split(X, y)):
                model = lgb.LGBMClassifier(
                    learning_rate=0.05, num_leaves=31, n_estimators=500,
                    random_state=seed, verbose=-1,
                )
                model.fit(
                    X[tr], y[tr],
                    eval_set=[(X[te], y[te])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                proba = model.predict_proba(X[te])[:, 1]
                pred = (proba >= 0.5).astype(int)

                p, r, _, _ = precision_recall_fscore_support(
                    y[te], pred, labels=[0, 1], zero_division=0,
                )
                bac = 0.5 * (r[0] + r[1])
                per_fold_rows.append([
                    seed, fold_i, len(tr), len(te),
                    f"{bac:.4f}",
                    f"{p[1]:.4f}", f"{r[1]:.4f}",  # M class
                    f"{p[0]:.4f}", f"{r[0]:.4f}",  # B class
                ])
                calibration_rows.append([
                    seed, fold_i, f"{brier_score_loss(y[te], proba):.4f}",
                ])
                for name, gain in zip(feature_names, model.booster_.feature_importance(importance_type="gain")):
                    importance_acc[name] += float(gain)

                # Flush diagnostic CSVs progressively so LAPPATO_MCB
                # can pick them up between polls.
                write_csv(
                    CHECKPOINTS / "wdbc_per_fold.csv",
                    ["seed", "fold", "n_train", "n_test", "bac",
                     "precision_M", "recall_M", "precision_B", "recall_B"],
                    per_fold_rows,
                )
                write_csv(
                    CHECKPOINTS / "wdbc_calibration.csv",
                    ["seed", "fold", "brier"],
                    calibration_rows,
                )
                ranked = sorted(importance_acc.items(), key=lambda kv: -kv[1])
                write_csv(
                    CHECKPOINTS / "wdbc_feature_importance.csv",
                    ["feature", "gain"],
                    [[n, f"{g:.2f}"] for n, g in ranked[:10]],
                )
                print(f"  seed={seed} fold={fold_i}: BAC={bac:.4f}  "
                      f"Brier={float(calibration_rows[-1][2]):.4f}")

        bac_all = [float(r[4]) for r in per_fold_rows]
        brier_all = [float(r[2]) for r in calibration_rows]
        print(f"\nMean BAC   = {np.mean(bac_all):.4f} ± {np.std(bac_all):.4f}")
        print(f"Mean Brier = {np.mean(brier_all):.4f} ± {np.std(brier_all):.4f}")
        print(f"Top-3 features by gain: "
              f"{[n for n, _ in sorted(importance_acc.items(), key=lambda kv: -kv[1])[:3]]}")
    finally:
        lappato_mcb.stop(timeout=120.0)
        print("LAPPATO_MCB stopped.")


if __name__ == "__main__":
    main()
