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

from collections.abc import Mapping
from pathlib import Path

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
    # ── v1.6 insight-detector thresholds (introduced 2026-05-10) ──────
    # subgroup-disparity: minimum (error_rate(subgroup)/error_rate(global))
    # ratio at which a subgroup is flagged.  2.0 = "twice as many errors
    # as average" — Fisher exact p-value < α confirms it.
    "subgroup_disparity_ratio_warning": 2.0,
    "subgroup_disparity_p_warning": 0.05,
    "subgroup_min_size": 5,                      # ignore tiny subgroups
    # calibration-bin gap: max |conf - acc| inside any populated bin.
    # 0.10 follows Guo et al. 2017 ("On Calibration of Modern Neural
    # Networks") for "reliably miscalibrated" bins.
    "calibration_bin_gap_warning": 0.10,
    "calibration_bin_min_count": 10,
    # decision-threshold ROC: flag if Youden's-J optimum is more than
    # this distance from 0.5 (i.e. the binary decision rule is
    # operating sub-optimally for the held-out set).
    "decision_threshold_distance_warning": 0.05,
    # failure-clustering: chi-square p-value below which we declare
    # the misclassifications are NOT randomly distributed across groups.
    "failure_clustering_p_warning": 0.05,
    # cross-cycle drift: |slope| × N_cycles bigger than this fraction
    # of metric range counts as drift.  0.05 = "5 % of the metric range
    # over the observed cycle window" — generous floor.
    "drift_total_change_warning": 0.05,
    "drift_p_warning": 0.05,
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
    target = {label.lower() for label in labels}
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


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  v1.6 — Insight detectors (2026-05-10)                            ║
# ║                                                                   ║
# ║  Seven new detectors that move LAPPATO_MCB from "pattern-match    ║
# ║  compliance auditor" toward "insight engine" while preserving     ║
# ║  the stdlib-only-by-default core (statistical helpers in          ║
# ║  ``lappato_mcb._stats`` use SciPy when present, otherwise fall   ║
# ║  back to the pure-Python implementation).                         ║
# ║                                                                   ║
# ║  Each detector follows the same fail-closed convention as the     ║
# ║  v1.5 detectors above: missing or malformed CSV → no card.        ║
# ╚═══════════════════════════════════════════════════════════════════╝

# ── d17 — Subgroup-stratified failure disparity ─────────────────────────
# Reads ``pipeline_subgroup_metrics.csv``: rows of
# (subgroup_key, subgroup_value, n, n_errors).
# Fires when ANY subgroup's error rate is ≥ X× the global rate AND a
# Fisher's-exact (or chi-square fallback) p-value crosses α.
# Use case: would have surfaced our "R-suffix biopsies fail 4× more"
# pattern automatically.

def _subgroup_disparity_records(path: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in rows(path):
        n  = f(row, "n", "support", "size")
        ne = f(row, "n_errors", "errors", "fail_count")
        if n is None or ne is None or n <= 0:
            continue
        out.append({
            "key":        str(row.get("subgroup_key",   "")).strip() or "?",
            "value":      str(row.get("subgroup_value", "")).strip() or "?",
            "n":          float(n),
            "n_errors":   float(ne),
            "error_rate": float(ne) / float(n) if n > 0 else 0.0,
        })
    return out


def _subgroup_disparity_present(path: Path) -> bool:
    recs = _subgroup_disparity_records(path)
    if not recs:
        return False
    n_total       = sum(r["n"] for r in recs)
    err_total     = sum(r["n_errors"] for r in recs)
    if n_total <= 0 or err_total <= 0:
        return False
    global_rate   = err_total / n_total
    ratio_thresh  = THRESHOLDS["subgroup_disparity_ratio_warning"]
    p_thresh      = THRESHOLDS["subgroup_disparity_p_warning"]
    min_size      = int(THRESHOLDS["subgroup_min_size"])
    from .._stats import fisher_exact_2x2
    for r in recs:
        if r["n"] < min_size:
            continue
        if global_rate <= 0:
            continue
        if (r["error_rate"] / global_rate) < ratio_thresh:
            continue
        # 2x2: rows = [subgroup, rest]; cols = [errors, correct]
        sub_n, sub_e = r["n"], r["n_errors"]
        rest_n, rest_e = n_total - sub_n, err_total - sub_e
        table = [
            [sub_e,            sub_n  - sub_e],
            [rest_e,           rest_n - rest_e],
        ]
        try:
            p = fisher_exact_2x2(table)
        except Exception:
            continue
        if p < p_thresh:
            return True
    return False


def _subgroup_disparity_summary(path: Path) -> dict:
    recs = _subgroup_disparity_records(path)
    if not recs:
        return {"rows": 0}
    total_n = sum(r["n"] for r in recs)
    total_e = sum(r["n_errors"] for r in recs)
    if total_n <= 0:
        return {"rows": len(recs)}
    global_rate = total_e / total_n
    worst = max(recs, key=lambda r: r["error_rate"])
    return {
        "rows":               len(recs),
        "global_error_rate":  round(global_rate, 4),
        "worst_subgroup":     f"{worst['key']}={worst['value']}",
        "worst_error_rate":   round(worst["error_rate"], 4),
        "worst_ratio":        round(
            (worst["error_rate"] / global_rate) if global_rate > 0 else 0.0,
            2,
        ),
    }


# ── d18 — Calibration-bin gap ───────────────────────────────────────────
# Reads ``pipeline_reliability_diagram.csv``: rows of
# (bin_lo, bin_hi, count, mean_confidence, accuracy).
# Fires when ANY populated bin has |conf − acc| ≥ Δ (default 0.10).
# Use case: surfaces the "underconfident-midrange / overconfident-high"
# pattern that uniform temperature scaling cannot fix.

def _calibration_bin_gap_present(path: Path) -> bool:
    min_cnt = int(THRESHOLDS["calibration_bin_min_count"])
    delta   = float(THRESHOLDS["calibration_bin_gap_warning"])
    for row in rows(path):
        cnt  = f(row, "count", "n", "size")
        conf = f(row, "mean_confidence", "mean_conf", "confidence", "conf")
        acc  = f(row, "accuracy", "acc", "fraction_correct")
        if cnt is None or conf is None or acc is None:
            continue
        if cnt < min_cnt:
            continue
        if abs(conf - acc) >= delta:
            return True
    return False


def _calibration_bin_gap_summary(path: Path) -> dict:
    out_rows = rows(path)
    worst = {"bin": "", "gap": 0.0, "count": 0}
    for row in out_rows:
        cnt  = f(row, "count", "n", "size")
        conf = f(row, "mean_confidence", "mean_conf", "confidence", "conf")
        acc  = f(row, "accuracy", "acc", "fraction_correct")
        if cnt is None or conf is None or acc is None or cnt <= 0:
            continue
        gap = abs(conf - acc)
        if gap > worst["gap"]:
            lo = f(row, "bin_lo", "lo", "low") or 0.0
            hi = f(row, "bin_hi", "hi", "high") or 1.0
            worst = {"bin": f"[{lo:.2f},{hi:.2f}]", "gap": float(gap),
                     "count": int(cnt), "conf": float(conf), "acc": float(acc)}
    return {
        "rows":            len(out_rows),
        "max_bin_gap":     round(worst["gap"], 4),
        "worst_bin":       worst["bin"],
        "worst_bin_count": worst.get("count", 0),
    }


# ── d19 — Decision-threshold sub-optimal ────────────────────────────────
# Reads ``pipeline_predictions_with_probs.csv``: rows of
# (item_id, y_true, p_positive).
# Sweeps the binary decision threshold over [0.05, 0.95] and reports the
# one maximising Youden's J = sens(+) + sens(-) − 1.  Fires when the
# optimum is ≥ ``decision_threshold_distance_warning`` away from 0.5.
# Use case: would have flagged our 0.36 threshold tuning opportunity.

def _decision_threshold_optimum(path: Path) -> tuple[float, float, int, int]:
    """Sweep thresholds → (best_threshold, best_J, n_pos, n_neg)."""
    p_pos:  list[float] = []
    y_true: list[int]   = []
    for row in rows(path):
        p = f(row, "p_positive", "p_pos", "p_tumor", "score", "prob")
        y = f(row, "y_true", "true", "label")
        if p is None or y is None:
            continue
        p_pos.append(float(p))
        y_true.append(int(y))
    if len(p_pos) < 10:
        return 0.5, 0.0, 0, 0
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0, n_pos, n_neg
    best_t, best_j = 0.5, 0.0
    # 0.05-step sweep — coarse enough for stdlib speed yet adequate.
    t = 0.05
    while t <= 0.95:
        tp = sum(1 for p, y in zip(p_pos, y_true) if y == 1 and p >= t)
        fp = sum(1 for p, y in zip(p_pos, y_true) if y == 0 and p >= t)
        tn = n_neg - fp
        sens_p = tp / n_pos if n_pos > 0 else 0.0
        sens_n = tn / n_neg if n_neg > 0 else 0.0
        j = sens_p + sens_n - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
        t += 0.05
    return best_t, best_j, n_pos, n_neg


def _decision_threshold_suboptimal(path: Path) -> bool:
    best_t, _, n_pos, n_neg = _decision_threshold_optimum(path)
    if n_pos == 0 or n_neg == 0:
        return False
    delta = float(THRESHOLDS["decision_threshold_distance_warning"])
    return abs(best_t - 0.5) >= delta


def _decision_threshold_summary(path: Path) -> dict:
    best_t, best_j, n_pos, n_neg = _decision_threshold_optimum(path)
    return {
        "rows":            n_pos + n_neg,
        "n_positive":      n_pos,
        "n_negative":      n_neg,
        "best_threshold":  round(best_t, 3),
        "best_youden_j":   round(best_j, 4),
        "distance_from_05": round(abs(best_t - 0.5), 3),
    }


# ── d20 — Failure clustering ────────────────────────────────────────────
# Reads ``pipeline_per_item_predictions.csv``: rows of
# (item_id, group_id, y_true, y_pred, p_top).
# Chi-square test of independence on the (group × correct/incorrect)
# 2-column table.  Fires when p < α and at least one group has ≥ 5
# items.  Use case: surfaces "errors concentrate in patient X /
# scanner Y / annotator Z" patterns automatically.

def _failure_clustering_table(path: Path) -> tuple[dict[str, list[int]], int]:
    """Return ``({group: [n_errors, n_correct]}, total_groups)``.

    Uses explicit None-checks instead of ``or``-fallbacks because a
    valid prediction value of ``0`` is falsy and would otherwise be
    silently replaced by the default sentinel — a classic Python bug
    that would erase every "predicted negative" row from the table.
    """
    table: dict[str, list[int]] = {}
    for row in rows(path):
        gid = str(row.get("group_id", "")).strip()
        if not gid:
            continue
        yt_val = f(row, "y_true", "true", "label")
        yp_val = f(row, "y_pred", "pred")
        if yt_val is None or yp_val is None:
            continue
        try:
            yt = int(yt_val)
            yp = int(yp_val)
        except (TypeError, ValueError):
            continue
        cell = table.setdefault(gid, [0, 0])
        if yt != yp:
            cell[0] += 1
        else:
            cell[1] += 1
    return table, len(table)


def _failure_clustering_present(path: Path) -> bool:
    table, n_groups = _failure_clustering_table(path)
    if n_groups < 2:
        return False
    # Drop tiny groups (n < 5) to stabilise the test.
    big_groups = {g: cnt for g, cnt in table.items() if sum(cnt) >= 5}
    if len(big_groups) < 2:
        return False
    contingency = [cnt for cnt in big_groups.values()]
    from .._stats import chi2_p_value
    p = chi2_p_value(contingency)
    return p < float(THRESHOLDS["failure_clustering_p_warning"])


def _failure_clustering_summary(path: Path) -> dict:
    table, n_groups = _failure_clustering_table(path)
    if n_groups == 0:
        return {"rows": 0}
    n_total_err = sum(c[0] for c in table.values())
    n_total     = sum(sum(c) for c in table.values())
    worst_g, worst_rate = "?", 0.0
    for g, c in table.items():
        if sum(c) >= 5:
            rate = c[0] / sum(c)
            if rate > worst_rate:
                worst_rate, worst_g = rate, g
    return {
        "rows":              n_total,
        "n_groups":          n_groups,
        "global_error_rate": round((n_total_err / n_total) if n_total > 0 else 0.0, 4),
        "worst_group":       worst_g,
        "worst_group_rate":  round(worst_rate, 4),
    }


# ── d21 — Cross-cycle drift ─────────────────────────────────────────────
# Reads the run-tag-prefixed meta-log
# ``<run_tag>_lappato_mcb_meta.csv`` — already produced by the daemon —
# specifically the ``cycle`` and ``n_papers_kept`` columns (or, when
# present, a user-supplied ``primary_metric`` column).  Linear regression
# on the metric vs cycle index; fires when |slope| × N is large AND
# the t-test p-value is below α.  Use case: surfaces silent metric drift
# across long monitoring windows.

def _cross_cycle_drift_series(path: Path) -> tuple[list[float], list[float]]:
    out_x: list[float] = []
    out_y: list[float] = []
    for row in rows(path):
        cyc = f(row, "cycle", "cycle_n", "cycle_id")
        metric = f(row, "primary_metric", "metric_value", "score",
                    "n_papers_kept", "n_active_weaknesses")
        if cyc is None or metric is None:
            continue
        out_x.append(float(cyc))
        out_y.append(float(metric))
    return out_x, out_y


def _cross_cycle_drift_present(path: Path) -> bool:
    xs, ys = _cross_cycle_drift_series(path)
    if len(xs) < 5:
        return False
    metric_range = max(ys) - min(ys) if ys else 0.0
    if metric_range <= 0.0:
        return False
    from .._stats import linear_regression_slope_p
    slope, _, p = linear_regression_slope_p(xs, ys)
    n_cycles = max(xs) - min(xs)
    total_change = abs(slope) * n_cycles
    rel_change = total_change / metric_range
    return (
        p < float(THRESHOLDS["drift_p_warning"])
        and rel_change >= float(THRESHOLDS["drift_total_change_warning"])
    )


def _cross_cycle_drift_summary(path: Path) -> dict:
    xs, ys = _cross_cycle_drift_series(path)
    if len(xs) < 2:
        return {"rows": len(xs)}
    from .._stats import linear_regression_slope_p
    slope, intercept, p = linear_regression_slope_p(xs, ys)
    return {
        "rows":          len(xs),
        "n_cycles":      int(max(xs) - min(xs)) if xs else 0,
        "slope":         round(slope, 6),
        "p_value":       round(p, 6),
        "metric_range":  round((max(ys) - min(ys)) if ys else 0.0, 4),
    }


# ── d22 — Underpowered cohort (sample-size) ─────────────────────────────
# Re-uses ``pipeline_class_counts.csv`` (already a v1.5 evidence file).
# Quantifies how far the minority class is from the
# ``min_minority_n`` literature target.  Reports the ABSOLUTE number of
# additional samples needed (vs the abstract threshold flag of d1).
# Always-info severity — meant to be a constructive "you need N more
# events for 80 % power" companion to d1's pure threshold check.

def _underpowered_cohort_present(path: Path) -> bool:
    counts = _class_counts(path)
    if not counts:
        return False
    min_target = float(THRESHOLDS["min_minority_n"])
    return min(counts) < min_target


def _underpowered_cohort_summary(path: Path) -> dict:
    counts = _class_counts(path)
    if not counts:
        return {"rows": 0}
    min_target = float(THRESHOLDS["min_minority_n"])
    minority   = float(min(counts))
    needed     = max(0, int(min_target - minority))
    return {
        "rows":             len(counts),
        "minority_count":   int(minority),
        "literature_target": int(min_target),
        "additional_needed": needed,
    }


# ── d23 — Compositional syndrome detector ──────────────────────────────
# Post-processor that fires when ≥ 80 % of a *named syndrome's* trigger
# detectors are active.  Doesn't read its own CSV — it inspects the
# active-card list of the current cycle.  ``LAPPATO_MCB._run_cycle``
# evaluates every detector, then this one looks at the names of the
# fired ones to decide whether a composite syndrome card is also
# emitted.  Encoded as a special "manifest sentinel": its
# ``evidence_check`` is True iff the composition condition holds, given
# a snapshot of all currently-fired detector ids.

SYNDROME_DEFINITIONS: dict[str, dict] = {
    "underpowered_imbalanced_clinical_cohort": {
        "trigger_ids": [
            "small_minority_class",
            "high_cross_seed_variance",
            "acc_BAC_gap_high",
        ],
        "min_active_fraction": 0.66,             # ≥ 2 of 3
        "title": ("Syndrome: underpowered imbalanced clinical cohort "
                  "(Riley 2019 + Steyerberg 2019)"),
        "interpretation": (
            "The combination of small minority class, high cross-seed "
            "variance, and a measurable accuracy / balanced-accuracy "
            "gap is the canonical signature of a clinical-prediction "
            "cohort below the sample size required for stable "
            "evaluation (Riley et al., BMJ 2019; Steyerberg, Clinical "
            "Prediction Models 2019).  Chasing single-metric "
            "improvements without more data tends to amplify "
            "instability rather than reduce it."
        ),
    },
    "miscalibrated_modern_NN": {
        "trigger_ids": [
            "calibration_bin_gap",
            "prediction_confidence_collapsed",
        ],
        "min_active_fraction": 0.5,
        "title": ("Syndrome: miscalibrated modern neural network "
                  "(Guo et al. 2017)"),
        "interpretation": (
            "Modern neural networks tend to be over-confident; the "
            "combination of a populated calibration bin gap with a "
            "collapsed prediction histogram matches the Guo et al. "
            "2017 ICML diagnosis.  Temperature scaling fixes the "
            "uniform component; isotonic regression is required when "
            "the gap pattern is non-monotone across bins."
        ),
    },
}


def _syndrome_composition_active() -> tuple[bool, dict]:
    """Inspect ``THRESHOLDS["__active_card_ids__"]`` (populated by the
    daemon at cycle time) and decide whether a syndrome fires.  Returns
    ``(active, summary_dict)``.

    The daemon is responsible for populating
    ``THRESHOLDS["__active_card_ids__"]`` with the set of fired card
    ids BEFORE invoking this detector; if it doesn't, the detector
    fails closed.
    """
    active_ids = THRESHOLDS.get("__active_card_ids__", set())
    if not isinstance(active_ids, (set, list, tuple, frozenset)):
        return False, {"rows": 0}
    active_ids = set(active_ids)
    fired: list[str] = []
    for syndrome_name, sp in SYNDROME_DEFINITIONS.items():
        triggers   = set(sp["trigger_ids"])
        min_frac   = float(sp.get("min_active_fraction", 0.66))
        n_required = int(round(len(triggers) * min_frac))
        if len(triggers & active_ids) >= max(1, n_required):
            fired.append(syndrome_name)
    return bool(fired), {
        "rows":           len(active_ids),
        "active_cards":   sorted(active_ids),
        "syndromes":      fired,
    }


def _syndrome_composition_present(path: Path) -> bool:
    """``path`` is unused — kept for the manifest-uniformity contract.
    The detector reads the runtime active-card set instead."""
    active, _ = _syndrome_composition_active()
    return active


def _syndrome_composition_summary(path: Path) -> dict:
    _, summary_dict = _syndrome_composition_active()
    return summary_dict


# Append the seven v1.6 entries to MANIFEST (extending in place keeps
# the daemon discovery loop unchanged — it iterates ``MANIFEST`` only).
MANIFEST.extend([
    {
        "id": "subgroup_disparity",
        "title": "Failure rate is significantly higher in a subgroup of items",
        "evidence": "pipeline_subgroup_metrics.csv",
        "evidence_check": _subgroup_disparity_present,
        "evidence_summary": _subgroup_disparity_summary,
        "severity": "high",
        "queries": [
            "subgroup performance gap fairness machine learning",
            "stratified evaluation classifier rare subgroup",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Stratify training and evaluation by the offending subgroup; "
                       "consider subgroup-aware loss reweighting or per-subgroup "
                       "calibration before aggregating metrics.",
        "why_it_matters": "Aggregate metrics hide pockets of poor performance; "
                            "subgroup disparities often signal a confounder, a "
                            "labelling shortcut, or an unmodelled covariate.",
        "next_checks": [
            "Inspect feature distribution and class balance within the affected subgroup.",
            "Compare per-subgroup confidence calibration to the global one.",
        ],
        "success_criteria": [
            "Subgroup error-rate ratio falls below 2× and Fisher's exact p ≥ 0.05.",
        ],
        "references": [
            "subgroup robustness",
            "fairness through awareness",
            "stratified evaluation",
        ],
    },
    {
        "id": "calibration_bin_gap",
        "title": "Reliability-diagram bin shows a confidence-vs-accuracy gap",
        "evidence": "pipeline_reliability_diagram.csv",
        "evidence_check": _calibration_bin_gap_present,
        "evidence_summary": _calibration_bin_gap_summary,
        "severity": "high",
        "queries": [
            "reliability diagram calibration neural network",
            "isotonic regression Platt scaling probability calibration",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply isotonic regression or Platt scaling on a held-out "
                       "calibration split.  Single-parameter temperature scaling "
                       "is INSUFFICIENT when the bin-level gap is non-monotone.",
        "why_it_matters": "A monotone temperature is the simplest fix but cannot "
                            "correct bin-specific gaps.  Identifying the worst-bin "
                            "tells you whether temperature scaling will work or "
                            "whether you need a non-parametric calibrator.",
        "next_checks": [
            "Plot the reliability diagram and inspect bin populations.",
            "Compare temperature-scaled ECE against isotonic-regression ECE.",
        ],
        "success_criteria": [
            "All populated bins have |conf − acc| ≤ 0.10.",
        ],
        "references": [
            "Guo et al. 2017 calibration",
            "isotonic regression probability",
            "expected calibration error",
        ],
    },
    {
        "id": "decision_threshold_suboptimal",
        "title": "Binary decision threshold is sub-optimal for the held-out set",
        "evidence": "pipeline_predictions_with_probs.csv",
        "evidence_check": _decision_threshold_suboptimal,
        "evidence_summary": _decision_threshold_summary,
        "severity": "high",
        "queries": [
            "Youden index optimal threshold ROC clinical",
            "decision threshold tuning class imbalance binary",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Replace the default 0.5 cut-off with the Youden's-J optimal "
                       "threshold tuned on a calibration split, never on the test set.",
        "why_it_matters": "The default 0.5 cut-off is rarely optimal under class "
                            "imbalance or asymmetric error costs; tuning the "
                            "threshold can convert false negatives into false "
                            "positives without retraining.",
        "next_checks": [
            "Sweep the threshold on the calibration split and report the chosen value.",
            "Compute the cost of FN vs FP for the deployment context.",
        ],
        "success_criteria": [
            "The deployment threshold is documented and reported alongside metrics.",
        ],
        "references": [
            "Youden J statistic",
            "operating point selection",
            "cost-sensitive classification",
        ],
    },
    {
        "id": "failure_clustering",
        "title": "Misclassifications cluster non-randomly across groups",
        "evidence": "pipeline_per_item_predictions.csv",
        "evidence_check": _failure_clustering_present,
        "evidence_summary": _failure_clustering_summary,
        "severity": "high",
        "queries": [
            "error clustering machine learning batch effect",
            "concentrated failures group level confound",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Investigate group-level confounders (acquisition site, "
                       "annotator, time window).  Consider group-aware "
                       "cross-validation to expose generalisation gaps.",
        "why_it_matters": "Clustered failures usually reveal an unmodelled "
                            "covariate.  Random-fold metrics will over-state "
                            "performance until the confound is accounted for.",
        "next_checks": [
            "Compute per-group precision and recall; inspect the worst group.",
            "Switch to group-aware k-fold and recompute the headline metric.",
        ],
        "success_criteria": [
            "Group-level chi-square p-value ≥ 0.05 on the failure × group table.",
        ],
        "references": [
            "leave-one-group-out CV",
            "batch effect machine learning",
            "subpopulation generalization",
        ],
    },
    {
        "id": "cross_cycle_drift",
        "title": "Headline metric drifts significantly across monitoring cycles",
        "evidence": "pipeline_health_lappato_mcb_meta.csv",   # default RUN_TAG
        "evidence_check": _cross_cycle_drift_present,
        "evidence_summary": _cross_cycle_drift_summary,
        "severity": "info",
        "queries": [
            "machine learning model drift monitoring",
            "performance degradation deployment time",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Investigate environment / data drift before further model "
                       "tuning.  Compare current vs baseline distributions at the "
                       "feature, label, and prediction levels.",
        "why_it_matters": "A linear trend in the headline metric across cycles "
                            "usually points to upstream data drift (acquisition, "
                            "labelling, preprocessing) rather than a model defect.",
        "next_checks": [
            "Run a population-stability index (PSI) on input features.",
            "Re-fit the model on the most recent window and compare metrics.",
        ],
        "success_criteria": [
            "Cross-cycle slope is statistically indistinguishable from zero.",
        ],
        "references": [
            "concept drift",
            "data drift detection",
            "ML monitoring pipelines",
        ],
    },
    {
        "id": "underpowered_cohort",
        "title": "Cohort size is below the literature-cited target for the minority class",
        "evidence": "pipeline_class_counts.csv",
        "evidence_check": _underpowered_cohort_present,
        "evidence_summary": _underpowered_cohort_summary,
        "severity": "info",
        "queries": [
            "minimum sample size machine learning clinical prediction",
            "events per variable Riley sample size calculator",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Frame the deficit as a study limitation; quantify the "
                       "additional events required for the standard 80 %-power, "
                       "α=0.05 test against the literature target.",
        "why_it_matters": "Without enough events per predictor, all downstream "
                            "metrics carry irreducible variance.  Acknowledging "
                            "the sample-size gap is more credible than chasing "
                            "model improvements.",
        "next_checks": [
            "Document the additional events required to reach the target.",
            "Plan an external-cohort validation as a substitute for size growth.",
        ],
        "success_criteria": [
            "Minority-class events meet the published target for the chosen domain.",
        ],
        "references": [
            "Riley 2019 sample size",
            "events per variable",
            "clinical prediction models",
        ],
    },
    {
        "id": "syndrome_composition",
        "title": "Two or more weakness cards combine into a named syndrome",
        # No evidence file — driven by the daemon's active-card list.
        # The path argument is ignored by the detector; we point it
        # at the same meta-log file so the loader passes through.
        "evidence": "pipeline_health_lappato_mcb_meta.csv",
        "evidence_check": _syndrome_composition_present,
        "evidence_summary": _syndrome_composition_summary,
        "severity": "info",
        "queries": [
            "machine learning weakness combination diagnosis",
            "clinical prediction model validation checklist TRIPOD-AI",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Treat the combination as a single named syndrome rather "
                       "than three independent alerts; the recommended action is "
                       "the syndrome's collective transplant, not the sum of the "
                       "individual ones.",
        "why_it_matters": "Several weakness cards firing together usually point "
                            "to a single underlying cause; addressing them "
                            "individually wastes effort.",
        "next_checks": [
            "Read the syndrome's interpretation block (lappato_mcb.manifests."
            "pipeline_health.SYNDROME_DEFINITIONS).",
            "Apply the syndrome's collective transplant before the individual ones.",
        ],
        "success_criteria": [
            "After the syndrome's transplant is applied, < 50 % of its trigger "
            "cards remain active in the next cycle.",
        ],
        "references": [
            "diagnostic syndromes",
            "clinical model validation",
            "TRIPOD-AI",
        ],
    },
])


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS",
           "override_thresholds", "SYNDROME_DEFINITIONS"]
