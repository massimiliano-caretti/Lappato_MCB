"""Pipeline-health manifest — domain-agnostic methodological diagnostics.

Detects statistical/methodological weaknesses common to *any* ML pipeline
(NLP, vision, recommender, RL, tabular, …) without leaning on a domain-
specific vocabulary. Sixteen detectors cover both the obvious bottlenecks
(class imbalance, cross-seed variance) and the subtle / hidden ones
(train-test overlap, label noise, prediction collapse, threshold tuned
on test, hyper-parameter overfitting, dead features, loss/metric
divergence, prevalence shift).

Expected CSVs in ``checkpoints/``:

  - ``pipeline_class_counts.csv``: ``class,count`` (or ``support``); add
    a ``split`` column to also enable prevalence-shift detection.
  - ``pipeline_cv_metrics.csv``: ``seed,accuracy,balanced_accuracy,f1,auc``.
  - ``pipeline_feature_importance.csv``: ``feature,gain``.
  - ``pipeline_external_validation.csv``: ``split,auc,accuracy,f1``.
  - ``pipeline_split_overlap.csv``: ``id,train,test`` (1/0 membership).
  - ``pipeline_label_noise.csv``: ``item,annotators_agree`` (1/0) or a
    ``disagreement`` rate column.
  - ``pipeline_prediction_distribution.csv``: ``bin,share`` over the
    predicted-probability histogram.
  - ``pipeline_threshold_audit.csv``: ``stage,split`` — flags whether
    the operating threshold was selected on the same split that reports
    final metrics.
  - ``pipeline_hyperparameter_search.csv``: one row per HP trial with
    ``trial,score``.
  - ``pipeline_feature_variance.csv``: ``feature,variance``.
  - ``pipeline_loss_curve.csv``: ``epoch,train_loss,eval_metric``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import (
    count_rows,
    f,
    mean,
    rows,
    summary,
    values,
)

RUN_TAG = "pipeline_health"

DEFAULT_THRESHOLDS = {
    "min_minority_n": 1000.0,
    "seed_std_warning": 0.03,
    "acc_bac_gap_warning": 0.05,
    "imbalance_ratio_warning": 5.0,
    "top_feature_share_warning": 0.30,
    "cohort_size_warning": 30.0,
    "calibration_drop_warning": 0.15,
    # Hidden-bottleneck thresholds.
    "train_test_overlap_warning": 0.0,           # any overlap is a defect
    "label_noise_disagreement_warning": 0.10,
    "prediction_extreme_share_warning": 0.90,    # >90% of mass at extremes
    "prediction_uniform_kl_warning": 0.02,       # max-entropy collapse
    "prevalence_shift_warning": 0.05,            # absolute prevalence delta
    "loss_metric_divergence_warning": 0.20,
    "hp_top_vs_mean_gap_warning": 0.05,
    "dead_feature_share_warning": 0.10,
    "dead_feature_variance_floor": 1.0e-6,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _stdev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _class_counts(path: Path) -> list[float]:
    counts = values(path, "count", "support", "n", "size")
    if counts:
        return counts
    out: list[float] = []
    for row in rows(path):
        for k, v in row.items():
            if k in ("class", "label", "name"):
                continue
            try:
                out.append(float(v))
                break
            except (TypeError, ValueError):
                continue
    return out


# ─── detector 1 — small minority class ─────────────────────────────────
def _small_minority_class(path: Path) -> bool:
    counts = _class_counts(path)
    return bool(counts) and min(counts) < THRESHOLDS["min_minority_n"]


def _small_minority_class_summary(path: Path) -> dict:
    counts = _class_counts(path)
    return {
        "rows": len(counts),
        "min_class_count": int(min(counts)) if counts else 0,
    }


# ─── detector 2 — high cross-seed variance ─────────────────────────────
def _seed_metric_values(path: Path) -> list[float]:
    return values(path, "balanced_accuracy", "bac", "f1", "auc", "accuracy")


def _high_cross_seed_variance(path: Path) -> bool:
    vals = _seed_metric_values(path)
    return len(vals) >= 2 and _stdev(vals) > THRESHOLDS["seed_std_warning"]


def _seed_variance_summary(path: Path) -> dict:
    vals = _seed_metric_values(path)
    return {"rows": len(vals), "metric_std": round(_stdev(vals), 4)}


# ─── detector 3 — single-split / no multi-seed eval ────────────────────
def _single_split_used(path: Path) -> bool:
    return 0 < count_rows(path) < 2


def _single_split_summary(path: Path) -> dict:
    return {"rows": count_rows(path)}


# ─── detector 4 — accuracy / balanced-accuracy gap ─────────────────────
def _acc_bac_gaps(path: Path) -> list[float]:
    out: list[float] = []
    for row in rows(path):
        acc = f(row, "accuracy", "acc")
        bac = f(row, "balanced_accuracy", "bac")
        if acc is None or bac is None:
            continue
        out.append(abs(acc - bac))
    return out


def _acc_bac_gap_high(path: Path) -> bool:
    gaps = _acc_bac_gaps(path)
    return bool(gaps) and max(gaps) > THRESHOLDS["acc_bac_gap_warning"]


def _acc_bac_gap_summary(path: Path) -> dict:
    gaps = _acc_bac_gaps(path)
    return {"rows": len(gaps), "max_acc_bac_gap": round(max(gaps), 4) if gaps else 0.0}


# ─── detector 5 — severe class imbalance (ratio) ───────────────────────
def _severe_imbalance(path: Path) -> bool:
    counts = _class_counts(path)
    if len(counts) < 2 or min(counts) <= 0:
        return False
    return (max(counts) / min(counts)) > THRESHOLDS["imbalance_ratio_warning"]


def _imbalance_summary(path: Path) -> dict:
    counts = _class_counts(path)
    ratio = (max(counts) / min(counts)) if counts and min(counts) > 0 else 0.0
    return {"rows": len(counts), "imbalance_ratio": round(ratio, 3)}


# ─── detector 6 — feature-importance concentration ─────────────────────
def _top_feature_share(path: Path) -> float:
    gains = values(path, "gain", "importance", "weight")
    total = sum(gains)
    if total <= 0:
        return 0.0
    return max(gains) / total


def _feature_importance_concentration(path: Path) -> bool:
    return _top_feature_share(path) > THRESHOLDS["top_feature_share_warning"]


def _feature_importance_summary(path: Path) -> dict:
    return summary(path, "top_feature_share", _top_feature_share(path))


# ─── detector 7 — small cohort size ────────────────────────────────────
def _small_cohort_size(path: Path) -> bool:
    counts = _class_counts(path)
    if not counts:
        return False
    return sum(counts) < THRESHOLDS["cohort_size_warning"]


def _cohort_size_summary(path: Path) -> dict:
    counts = _class_counts(path)
    return {"rows": len(counts), "total_units": int(sum(counts)) if counts else 0}


# ─── detector 8 — external-test calibration / performance drop ─────────
def _split_metric(path: Path, *labels: str) -> float | None:
    target = {l.lower() for l in labels}
    out: list[float] = []
    for row in rows(path):
        split = (row.get("split") or row.get("set") or "").strip().lower()
        if split in target:
            v = f(row, "auc", "accuracy", "balanced_accuracy", "f1")
            if v is not None:
                out.append(v)
    return mean(out) if out else None


def _external_test_drop(path: Path) -> float:
    internal = _split_metric(path, "cv", "internal", "train", "validation")
    external = _split_metric(path, "external", "test", "held_out", "held-out")
    if internal is None or external is None:
        return 0.0
    return internal - external


def _external_test_calibration_drop(path: Path) -> bool:
    return _external_test_drop(path) > THRESHOLDS["calibration_drop_warning"]


def _external_test_summary(path: Path) -> dict:
    return summary(path, "internal_minus_external", _external_test_drop(path))


# ─── detector 9 — train/test sample overlap ────────────────────────────
def _train_test_overlap_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    overlap = 0
    seen = 0
    for row in items:
        in_train = row.get("train") or row.get("in_train") or "0"
        in_test = row.get("test") or row.get("in_test") or "0"
        try:
            if int(float(in_train)) and int(float(in_test)):
                overlap += 1
        except (TypeError, ValueError):
            continue
        seen += 1
    return (overlap / seen) if seen else 0.0


def _train_test_overlap_detected(path: Path) -> bool:
    return _train_test_overlap_share(path) > THRESHOLDS["train_test_overlap_warning"]


def _train_test_overlap_summary(path: Path) -> dict:
    return summary(path, "overlap_share", _train_test_overlap_share(path))


# ─── detector 10 — label noise / inter-annotator disagreement ──────────
def _label_disagreement_rate(path: Path) -> float:
    direct = values(path, "disagreement", "disagreement_rate", "noise_rate")
    if direct:
        return max(direct)
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    total = 0
    for row in items:
        v = row.get("annotators_agree") or row.get("agree")
        if v is None or v == "":
            continue
        total += 1
        try:
            if int(float(v)) == 0:
                bad += 1
        except (TypeError, ValueError):
            continue
    return (bad / total) if total else 0.0


def _label_noise_detected(path: Path) -> bool:
    return _label_disagreement_rate(path) > THRESHOLDS["label_noise_disagreement_warning"]


def _label_noise_summary(path: Path) -> dict:
    return summary(path, "disagreement_rate", _label_disagreement_rate(path))


# ─── detector 11 — prediction confidence collapsed ─────────────────────
def _prediction_distribution_features(path: Path) -> tuple[float, float]:
    """Return (extreme_share, uniformity_kl).

    extreme_share is the share of probability mass in the two outermost
    bins of a 0..1 confidence histogram. uniformity_kl is the KL
    divergence of the predicted-confidence distribution from the
    uniform distribution; near-zero values flag a model that hedges on
    every example.
    """
    items = rows(path)
    if not items:
        return 0.0, 0.0
    bins: list[tuple[float, float]] = []
    total = 0.0
    for row in items:
        b = f(row, "bin", "bin_center", "p")
        s = f(row, "share", "fraction", "count")
        if b is None or s is None:
            continue
        bins.append((b, s))
        total += s
    if total <= 0 or not bins:
        return 0.0, 0.0
    bins = [(b, s / total) for b, s in bins]
    extremes = sum(s for b, s in bins if b <= 0.1 or b >= 0.9)
    n = len(bins)
    uniform = 1.0 / n
    import math as _m
    kl = 0.0
    for _, s in bins:
        if s > 0:
            kl += s * _m.log(s / uniform)
    return extremes, abs(kl)


def _prediction_confidence_collapsed(path: Path) -> bool:
    extremes, kl = _prediction_distribution_features(path)
    if extremes >= THRESHOLDS["prediction_extreme_share_warning"]:
        return True
    return 0 < kl < THRESHOLDS["prediction_uniform_kl_warning"]


def _prediction_distribution_summary(path: Path) -> dict:
    extremes, kl = _prediction_distribution_features(path)
    return {
        "rows": count_rows(path),
        "extreme_mass": round(extremes, 4),
        "uniformity_kl": round(kl, 4),
    }


# ─── detector 12 — train/eval prevalence shift ─────────────────────────
def _prevalence_by_split(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows(path):
        split = (row.get("split") or row.get("set") or "").strip().lower()
        cls = (row.get("class") or row.get("label") or "").strip()
        cnt = f(row, "count", "support", "n", "size")
        if not split or cls in ("",) or cnt is None:
            continue
        out.setdefault(split, {})[cls] = out.setdefault(split, {}).get(cls, 0.0) + cnt
    return out


def _max_prevalence_delta(path: Path) -> float:
    by_split = _prevalence_by_split(path)
    if len(by_split) < 2:
        return 0.0
    splits = list(by_split.keys())
    classes = set().union(*(d.keys() for d in by_split.values()))
    deltas: list[float] = []
    for cls in classes:
        shares: list[float] = []
        for s in splits:
            d = by_split[s]
            tot = sum(d.values())
            if tot <= 0:
                continue
            shares.append(d.get(cls, 0.0) / tot)
        if len(shares) >= 2:
            deltas.append(max(shares) - min(shares))
    return max(deltas) if deltas else 0.0


def _train_eval_prevalence_shift(path: Path) -> bool:
    return _max_prevalence_delta(path) > THRESHOLDS["prevalence_shift_warning"]


def _prevalence_shift_summary(path: Path) -> dict:
    return summary(path, "max_prevalence_delta", _max_prevalence_delta(path))


# ─── detector 13 — loss / metric divergence ────────────────────────────
def _loss_metric_divergence_score(path: Path) -> float:
    """Return how much train_loss improves while eval_metric stagnates.

    The score is the relative loss drop scaled by (1 − metric progress
    fraction). Returns 0 when the metric also progresses meaningfully.
    """
    losses = values(path, "train_loss", "loss", "training_loss")
    metrics = values(path, "eval_metric", "val_metric", "validation_metric", "auc", "f1", "accuracy")
    if len(losses) < 2 or len(metrics) < 2 or losses[0] <= 0:
        return 0.0
    loss_drop_ratio = max((losses[0] - losses[-1]) / abs(losses[0]), 0.0)
    metric_gain = abs(metrics[-1] - metrics[0])
    # Treat any absolute metric gain ≥ 0.05 as "the metric is following".
    metric_progress_fraction = min(metric_gain / 0.05, 1.0)
    return loss_drop_ratio * (1.0 - metric_progress_fraction)


def _loss_metric_divergence(path: Path) -> bool:
    return _loss_metric_divergence_score(path) > THRESHOLDS["loss_metric_divergence_warning"]


def _loss_metric_divergence_summary(path: Path) -> dict:
    return summary(path, "loss_minus_metric_progress", _loss_metric_divergence_score(path))


# ─── detector 14 — hyperparameter overfit to validation ────────────────
def _hp_top_vs_mean_gap(path: Path) -> float:
    scores = values(path, "score", "validation_score", "metric")
    if len(scores) < 5:
        return 0.0
    return max(scores) - mean(scores)


def _hyperparameter_overfit_to_validation(path: Path) -> bool:
    return _hp_top_vs_mean_gap(path) > THRESHOLDS["hp_top_vs_mean_gap_warning"]


def _hp_summary(path: Path) -> dict:
    return summary(path, "top_minus_mean_score", _hp_top_vs_mean_gap(path))


# ─── detector 15 — constant / dead features ────────────────────────────
def _dead_feature_share(path: Path) -> float:
    vars_ = values(path, "variance", "var", "std")
    if not vars_:
        return 0.0
    floor = THRESHOLDS["dead_feature_variance_floor"]
    return sum(1 for v in vars_ if v <= floor) / len(vars_)


def _constant_or_dead_features(path: Path) -> bool:
    return _dead_feature_share(path) > THRESHOLDS["dead_feature_share_warning"]


def _dead_features_summary(path: Path) -> dict:
    return summary(path, "dead_feature_share", _dead_feature_share(path))


# ─── detector 16 — threshold picked on test set ────────────────────────
def _threshold_picked_on_test_set(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    pick_split = None
    report_split = None
    for row in items:
        stage = (row.get("stage") or "").strip().lower()
        split = (row.get("split") or row.get("set") or "").strip().lower()
        if not stage or not split:
            continue
        if "threshold" in stage or "operating" in stage or "tune" in stage:
            pick_split = split
        elif "report" in stage or "final" in stage or "evaluation" in stage:
            report_split = split
    if pick_split is None or report_split is None:
        return False
    return pick_split == report_split and pick_split in ("test", "external", "held_out", "held-out")


def _threshold_audit_summary(path: Path) -> dict:
    items = rows(path)
    return {"rows": len(items), "stages_reported": len(items)}


MANIFEST: list[dict] = [
    {
        "id": "small_minority_class",
        "title": "Minority-class support is below the safe-evaluation floor",
        "evidence": "pipeline_class_counts.csv",
        "evidence_check": _small_minority_class,
        "evidence_summary": _small_minority_class_summary,
        "severity": "high",
        "queries": [
            "imbalanced classification minimum sample size statistical power",
            "rare class evaluation metric stability machine learning",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Quantify confidence intervals on minority-class metrics; consider stratified resampling or reweighting.",
        "why_it_matters": "Minority-class scores are noisy and unstable when support is too small to estimate them reliably.",
        "next_checks": [
            "Bootstrap minority-class precision/recall confidence intervals.",
            "Compare cost-sensitive learning against the current loss.",
        ],
        "success_criteria": [
            "Minority-class metric CIs are reported and narrow enough to support claims.",
        ],
        "references": [
            "minimum sample size classification",
            "imbalanced learning",
            "metric stability rare class",
        ],
    },
    {
        "id": "high_cross_seed_variance",
        "title": "Cross-seed metric variance is high",
        "evidence": "pipeline_cv_metrics.csv",
        "evidence_check": _high_cross_seed_variance,
        "evidence_summary": _seed_variance_summary,
        "severity": "high",
        "queries": [
            "deep learning random seed variance reproducibility",
            "machine learning model evaluation seed stability statistical test",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report mean and standard deviation across seeds and add paired statistical tests against baselines.",
        "why_it_matters": "A single-seed score can over- or under-state model quality when seed-to-seed variance is large.",
        "next_checks": [
            "Run at least 5 seeds with fixed splits.",
            "Report effect sizes and paired comparisons across seeds.",
        ],
        "success_criteria": [
            "Cross-seed standard deviation falls below 0.03 on the headline metric.",
        ],
        "references": [
            "random seed sensitivity",
            "reproducibility crisis machine learning",
            "paired bootstrap comparison",
        ],
    },
    {
        "id": "single_split_used",
        "title": "Only a single split / seed is reported (no multi-seed evaluation)",
        "evidence": "pipeline_cv_metrics.csv",
        "evidence_check": _single_split_used,
        "evidence_summary": _single_split_summary,
        "severity": "info",
        "queries": [
            "cross validation reporting machine learning multiple seeds",
            "single split evaluation pitfalls model selection",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Replace single-split numbers with repeated stratified CV or multi-seed runs and report dispersion.",
        "why_it_matters": "Single-split results have unknown variance and are easy to over-interpret.",
        "next_checks": [
            "Switch to repeated stratified k-fold or nested CV.",
            "Report the seed count and full distribution of fold scores.",
        ],
        "success_criteria": [
            "At least 5 seeds or folds are reported with dispersion statistics.",
        ],
        "references": [
            "repeated cross validation",
            "nested cross validation",
            "evaluation protocol",
        ],
    },
    {
        "id": "acc_BAC_gap_high",
        "title": "Gap between accuracy and balanced accuracy is high",
        "evidence": "pipeline_cv_metrics.csv",
        "evidence_check": _acc_bac_gap_high,
        "evidence_summary": _acc_bac_gap_summary,
        "severity": "high",
        "queries": [
            "balanced accuracy class imbalance evaluation metric",
            "macro F1 micro F1 imbalanced classification",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report balanced accuracy / macro-F1 alongside accuracy and select models on the imbalance-aware metric.",
        "why_it_matters": "Plain accuracy on imbalanced data can hide systematic minority-class failures.",
        "next_checks": [
            "Inspect per-class precision and recall.",
            "Compare model selection on accuracy vs balanced accuracy.",
        ],
        "success_criteria": [
            "Accuracy / BAC gap shrinks below 0.05.",
        ],
        "references": [
            "balanced accuracy",
            "macro F1",
            "imbalanced classification metrics",
        ],
    },
    {
        "id": "severe_class_imbalance",
        "title": "Majority/minority class ratio exceeds the safety threshold",
        "evidence": "pipeline_class_counts.csv",
        "evidence_check": _severe_imbalance,
        "evidence_summary": _imbalance_summary,
        "severity": "high",
        "queries": [
            "extreme class imbalance learning strategies survey",
            "cost sensitive learning class weighting threshold moving",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add class-balanced loss, threshold tuning, or resampling and report PR-AUC alongside ROC-AUC.",
        "why_it_matters": "Trivial majority-class baselines can dominate when the imbalance ratio is large.",
        "next_checks": [
            "Compute PR-AUC and operating-point precision/recall.",
            "Compare class weighting, focal loss, and threshold moving.",
        ],
        "success_criteria": [
            "Imbalance-aware metrics improve without degrading aggregate metrics.",
        ],
        "references": [
            "class imbalance survey",
            "PR-AUC vs ROC-AUC",
            "cost-sensitive learning",
        ],
    },
    {
        "id": "feature_importance_concentration",
        "title": "A single feature dominates the model's importance budget",
        "evidence": "pipeline_feature_importance.csv",
        "evidence_check": _feature_importance_concentration,
        "evidence_summary": _feature_importance_summary,
        "severity": "info",
        "queries": [
            "feature importance concentration leakage detection",
            "permutation importance stability machine learning",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Audit the dominant feature for leakage, leave-one-feature-out re-training, and SHAP / permutation cross-checks.",
        "why_it_matters": "When one feature drives most of the gain, the model is fragile to its drift and may be exploiting leakage.",
        "next_checks": [
            "Remove the top feature and re-evaluate.",
            "Cross-check importance via SHAP and permutation.",
        ],
        "success_criteria": [
            "Top-feature share is below 0.30 or its dominance is justified by domain knowledge.",
        ],
        "references": [
            "feature importance leakage",
            "permutation importance",
            "SHAP stability",
        ],
    },
    {
        "id": "small_cohort_size",
        "title": "Total cohort / sample size is below the inference floor",
        "evidence": "pipeline_class_counts.csv",
        "evidence_check": _small_cohort_size,
        "evidence_summary": _cohort_size_summary,
        "severity": "high",
        "queries": [
            "small sample size machine learning generalization risk",
            "learning curve sample size requirement classification",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Plot a learning curve, prefer simple regularised models, and report wide confidence intervals.",
        "why_it_matters": "With small cohorts, complex models overfit and reported metrics carry large uncertainty.",
        "next_checks": [
            "Generate a learning curve over training-set fraction.",
            "Report CIs on every headline metric.",
        ],
        "success_criteria": [
            "Sample size or uncertainty is reported and matches the model complexity.",
        ],
        "references": [
            "learning curve",
            "small sample inference",
            "model complexity sample size",
        ],
    },
    {
        "id": "external_test_calibration_drop",
        "title": "Performance drops sharply between internal CV and external test",
        "evidence": "pipeline_external_validation.csv",
        "evidence_check": _external_test_calibration_drop,
        "evidence_summary": _external_test_summary,
        "severity": "high",
        "queries": [
            "internal external validation generalization gap machine learning",
            "distribution shift held out evaluation transportability",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Investigate covariate / label / prevalence shift between sources and recalibrate on the external split.",
        "why_it_matters": "A large CV-to-external drop signals overfitting to the internal cohort and weak transportability.",
        "next_checks": [
            "Quantify shift via PSI, KS, or label-prevalence delta.",
            "Recalibrate probabilities on a held-out external slice.",
        ],
        "success_criteria": [
            "Internal-to-external drop is below 0.15 or explained by a documented shift.",
        ],
        "references": [
            "transportability",
            "external validation",
            "distribution shift",
        ],
    },
    {
        "id": "train_test_overlap_detected",
        "title": "Same units appear in both training and evaluation splits",
        "evidence": "pipeline_split_overlap.csv",
        "evidence_check": _train_test_overlap_detected,
        "evidence_summary": _train_test_overlap_summary,
        "severity": "high",
        "queries": [
            "data leakage train test split duplicate detection",
            "patient level entity grouped cross validation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to grouped / entity-level splits and audit the pipeline for duplicate identifiers across folds.",
        "why_it_matters": "Train/test overlap silently inflates every reported metric and is the single most common cause of irreproducible ML results.",
        "next_checks": [
            "Hash entities and verify zero set intersection across splits.",
            "Re-train on grouped CV and compare against the leaky baseline.",
        ],
        "success_criteria": [
            "Reported overlap share is zero on all evaluation splits.",
        ],
        "references": [
            "data leakage",
            "grouped cross validation",
            "entity-level splits",
        ],
    },
    {
        "id": "label_noise_detected",
        "title": "Inter-annotator disagreement exceeds the safe-noise floor",
        "evidence": "pipeline_label_noise.csv",
        "evidence_check": _label_noise_detected,
        "evidence_summary": _label_noise_summary,
        "severity": "high",
        "queries": [
            "label noise robust training deep learning",
            "inter annotator agreement Cohen kappa machine learning labels",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Estimate label-noise rate, compare adjudication strategies, and apply noise-robust losses where appropriate.",
        "why_it_matters": "Noisy labels impose a Bayes-error ceiling that no model can cross; ignoring it makes the pipeline chase phantom gains.",
        "next_checks": [
            "Compute κ / α agreement on a held-out adjudication set.",
            "Run noise-aware training and report the noise-corrected metric.",
        ],
        "success_criteria": [
            "Disagreement rate is reported and below the chosen tolerance.",
        ],
        "references": [
            "label noise",
            "Cohen's kappa",
            "noise-robust training",
        ],
    },
    {
        "id": "prediction_confidence_collapsed",
        "title": "Predicted-confidence histogram has collapsed to extremes or to a flat distribution",
        "evidence": "pipeline_prediction_distribution.csv",
        "evidence_check": _prediction_confidence_collapsed,
        "evidence_summary": _prediction_distribution_summary,
        "severity": "high",
        "queries": [
            "neural network overconfidence calibration miscalibration",
            "uniform prediction degenerate classifier collapse",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Inspect logits histogram, add temperature scaling / label smoothing and verify ranking calibration.",
        "why_it_matters": "Models stuck at 0/1 or at uniform confidence have no useful operating point and produce misleading calibration metrics.",
        "next_checks": [
            "Plot the predicted-probability histogram per class.",
            "Compare temperature scaling against the raw classifier.",
        ],
        "success_criteria": [
            "Predicted confidence covers a non-degenerate distribution between extremes.",
        ],
        "references": [
            "temperature scaling",
            "label smoothing",
            "miscalibration deep learning",
        ],
    },
    {
        "id": "train_eval_prevalence_shift",
        "title": "Class prevalence differs between training and evaluation splits",
        "evidence": "pipeline_class_counts.csv",
        "evidence_check": _train_eval_prevalence_shift,
        "evidence_summary": _prevalence_shift_summary,
        "severity": "high",
        "queries": [
            "label shift prior shift estimation machine learning",
            "prevalence correction classification calibration",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply BBSE / EM-based label-shift correction or recalibrate priors before scoring.",
        "why_it_matters": "Prevalence shift silently biases threshold-dependent metrics even when the conditional p(y|x) is unchanged.",
        "next_checks": [
            "Estimate the test-time prior with BBSE.",
            "Recompute precision / recall under the corrected prior.",
        ],
        "success_criteria": [
            "Maximum prevalence delta across splits is documented and below 0.05.",
        ],
        "references": [
            "label shift",
            "prior shift",
            "BBSE",
        ],
    },
    {
        "id": "loss_metric_divergence",
        "title": "Training loss decreases without matching gain in the evaluation metric",
        "evidence": "pipeline_loss_curve.csv",
        "evidence_check": _loss_metric_divergence,
        "evidence_summary": _loss_metric_divergence_summary,
        "severity": "high",
        "queries": [
            "loss metric mismatch surrogate optimization machine learning",
            "training objective ranking metric mismatch deep learning",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Audit the surrogate loss alignment with the evaluation metric and consider direct-metric optimisation or differentiable approximations.",
        "why_it_matters": "Loss-metric divergence indicates the model is optimising the wrong objective and further training will not help.",
        "next_checks": [
            "Plot loss and eval metric together on the same x-axis.",
            "Try a differentiable surrogate of the eval metric.",
        ],
        "success_criteria": [
            "Loss reductions translate into proportional metric gains.",
        ],
        "references": [
            "surrogate loss",
            "ranking metric optimization",
            "metric-loss alignment",
        ],
    },
    {
        "id": "hyperparameter_overfit_to_validation",
        "title": "Best hyper-parameter trial differs sharply from the trial population",
        "evidence": "pipeline_hyperparameter_search.csv",
        "evidence_check": _hyperparameter_overfit_to_validation,
        "evidence_summary": _hp_summary,
        "severity": "high",
        "queries": [
            "hyperparameter overfitting validation winner curse",
            "nested cross validation hyperparameter selection bias",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to nested CV, hold out a fresh test split, and report the dispersion of hyper-parameter scores.",
        "why_it_matters": "When the top trial is much higher than the trial mean, the validation set is being optimised against and the reported score is upward-biased.",
        "next_checks": [
            "Run nested CV on the search procedure.",
            "Report HP-search trial distribution alongside the chosen point.",
        ],
        "success_criteria": [
            "Top-vs-mean HP gap stays below 0.05 or is justified by a held-out test.",
        ],
        "references": [
            "winner's curse hyperparameter search",
            "nested cross validation",
            "model selection bias",
        ],
    },
    {
        "id": "constant_or_dead_features",
        "title": "Many input features are constant or near-zero variance",
        "evidence": "pipeline_feature_variance.csv",
        "evidence_check": _constant_or_dead_features,
        "evidence_summary": _dead_features_summary,
        "severity": "info",
        "queries": [
            "feature selection low variance pruning machine learning",
            "dead neuron feature collapse representation learning",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Drop or merge constant / near-constant features and re-evaluate; investigate whether they reflect a preprocessing bug.",
        "why_it_matters": "Dead features dilute regularisation, slow training, and often signal an upstream preprocessing or extraction failure.",
        "next_checks": [
            "Audit the feature pipeline for stages that emit constants.",
            "Re-train without dead features and compare metrics.",
        ],
        "success_criteria": [
            "Dead-feature share stays below 0.10 or is explained by domain choice.",
        ],
        "references": [
            "low variance filter",
            "feature selection",
            "preprocessing pipeline audit",
        ],
    },
    {
        "id": "threshold_picked_on_test_set",
        "title": "Operating threshold was selected on the same split that reports final metrics",
        "evidence": "pipeline_threshold_audit.csv",
        "evidence_check": _threshold_picked_on_test_set,
        "evidence_summary": _threshold_audit_summary,
        "severity": "high",
        "queries": [
            "operating point selection test set leakage classifier",
            "threshold tuning held out validation classification",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Tune the operating threshold on a dedicated calibration split that is never used for final reporting.",
        "why_it_matters": "Threshold tuning on the test split silently inflates F1, precision, and recall by a few percentage points.",
        "next_checks": [
            "Add a calibration split and rerun threshold selection.",
            "Report metrics at the calibration-set threshold on the held-out test.",
        ],
        "success_criteria": [
            "Threshold-selection split is disjoint from the final-evaluation split.",
        ],
        "references": [
            "operating point selection",
            "threshold tuning leakage",
            "calibration split",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
