"""Example NLP host pipeline for LAPPATO_MCB (20newsgroups subset).

Trains a TF-IDF + Logistic-Regression text classifier under stratified
k-fold CV on a four-category subset of 20newsgroups (sklearn handles the
download transparently, with a one-off cache in ``~/scikit_learn_data/``).
Emits three diagnostic CSVs the NLP manifest gates on:

  - nlp_per_class.csv         per-class precision/recall/F1
  - nlp_confusion_top.csv     top-K off-diagonal confusion pairs (share of test rows)
  - nlp_token_importance.csv  top tokens by absolute coefficient

LAPPATO_MCB is launched on a daemon thread before training and stopped
after — the standard two-line integration into a host ML pipeline.

Usage:
    python examples/demo_nlp.py

Requires: scikit-learn. LAPPATO_MCB itself remains stdlib-only.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

# Make the `lappato_mcb` package importable when this file is run directly
# as a script (no `pip install -e .` required).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb import LAPPATO_MCB  # noqa: E402
from lappato_mcb.manifests import get as get_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"

# Four categories chosen for class-difficulty heterogeneity:
# comp.graphics is "easy", talk.religion.misc is "hard" — exactly the
# kind of per-class F1 spread the NLP manifest is designed to detect.
_CATEGORIES = (
    "comp.graphics",
    "sci.med",
    "rec.sport.baseball",
    "talk.religion.misc",
)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def load_dataset() -> tuple[list[str], np.ndarray, list[str]]:
    """Load the 4-category subset; remove headers/footers/quotes for fairness."""
    ng = fetch_20newsgroups(
        subset="train",
        categories=list(_CATEGORIES),
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=0,
    )
    return ng.data, np.asarray(ng.target), list(ng.target_names)


def main(seeds: tuple[int, ...] = (0, 1, 2), n_folds: int = 5) -> None:
    texts, y, class_names = load_dataset()
    print(f"NLP loaded: n={len(texts)}  classes={len(class_names)}  "
          f"{dict(Counter(y))}")

    manifest, run_tag = get_manifest("nlp")
    (CHECKPOINTS / f"{run_tag}_lappato_context.json").write_text(
        json.dumps({
            "domain": "supervised NLP",
            "task": "multiclass text classification",
            "model_family": "TF-IDF logistic regression",
            "dataset": "20newsgroups subset",
        }, indent=2),
        encoding="utf-8",
    )
    lappato_mcb = LAPPATO_MCB(
        project_root=ROOT, manifest=manifest, run_tag=run_tag, poll_interval=90.0,
    )
    lappato_mcb.start()
    print(f"LAPPATO_MCB started; log -> {lappato_mcb._log_path}")

    per_class_acc: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    confusion_acc = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    token_weight_acc: defaultdict[str, float] = defaultdict(float)
    token_weight_n = 0
    n_total_test = 0

    try:
        # Vectorise once (the splits act on indices, not on raw text).
        vec = TfidfVectorizer(
            max_features=20_000, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True,
        )
        X = vec.fit_transform(texts)
        feature_names = np.asarray(vec.get_feature_names_out())

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_i, (tr, te) in enumerate(skf.split(X, y)):
                clf = LogisticRegression(
                    max_iter=1000, random_state=seed,
                    C=1.0, solver="lbfgs",
                )
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])

                p, r, f1, _ = precision_recall_fscore_support(
                    y[te], pred, labels=range(len(class_names)),
                    zero_division=0,
                )
                for i, name in enumerate(class_names):
                    per_class_acc[name].append((p[i], r[i], f1[i]))

                cm = confusion_matrix(y[te], pred, labels=range(len(class_names)))
                confusion_acc += cm
                n_total_test += len(te)

                # Pool absolute coefficient mass per token (one-vs-rest).
                W = clf.coef_  # shape (n_classes, n_features)
                token_mass = np.abs(W).sum(axis=0)  # (n_features,)
                top_idx = np.argsort(-token_mass)[:200]
                for idx in top_idx:
                    token_weight_acc[feature_names[idx]] += float(token_mass[idx])
                token_weight_n += 1

                # Flush diagnostic CSVs progressively so LAPPATO_MCB
                # can pick them up between polls.
                _flush_per_class(per_class_acc)
                _flush_confusion_top(confusion_acc, class_names, n_total_test)
                _flush_token_importance(token_weight_acc, token_weight_n)

                fold_f1 = float(np.mean([per_class_acc[c][-1][2]
                                         for c in class_names]))
                print(f"  seed={seed} fold={fold_i}: macro-F1={fold_f1:.3f}")

        spread = _f1_spread(per_class_acc)
        print(f"\nPer-class F1 spread (max-min): {spread:.4f}")
    finally:
        lappato_mcb.stop(timeout=120.0)
        print("LAPPATO_MCB stopped.")


def _flush_per_class(acc: dict[str, list[tuple[float, float, float]]]) -> None:
    rows = []
    for name, vals in acc.items():
        if not vals:
            continue
        prec = float(np.mean([v[0] for v in vals]))
        rec = float(np.mean([v[1] for v in vals]))
        f1 = float(np.mean([v[2] for v in vals]))
        rows.append([name, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])
    write_csv(
        CHECKPOINTS / "nlp_per_class.csv",
        ["class", "precision", "recall", "f1"],
        rows,
    )


def _flush_confusion_top(cm: np.ndarray, class_names: list[str],
                         n_total: int, top_k: int = 10) -> None:
    """Top-K off-diagonal entries by share of total test rows."""
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda t: -t[2])
    pairs = pairs[:top_k]
    rows = [
        [tr, pr, cnt, f"{(cnt / max(n_total, 1)):.4f}"]
        for tr, pr, cnt in pairs
    ]
    write_csv(
        CHECKPOINTS / "nlp_confusion_top.csv",
        ["true", "predicted", "count", "share"],
        rows,
    )


def _flush_token_importance(acc: dict[str, float], n: int,
                            top_k: int = 20) -> None:
    if n == 0:
        return
    ranked = sorted(acc.items(), key=lambda kv: -kv[1])[:top_k]
    rows = [[tok, f"{w / n:.4f}"] for tok, w in ranked]
    write_csv(
        CHECKPOINTS / "nlp_token_importance.csv",
        ["token", "weight"],
        rows,
    )


def _f1_spread(acc: dict[str, list[tuple[float, float, float]]]) -> float:
    f1_means = [float(np.mean([v[2] for v in vs])) for vs in acc.values() if vs]
    return (max(f1_means) - min(f1_means)) if f1_means else 0.0


if __name__ == "__main__":
    main()
