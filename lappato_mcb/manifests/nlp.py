"""NLP manifest — 20newsgroups text classification with sklearn.

Mirrors the WDBC manifest's 5-entry shape (3 evidence-gated +
2 structural) for direct cross-domain comparability.

Diagnostic CSVs expected in ``checkpoints/`` (emitted by
``examples/demo_nlp.py``):

  - nlp_per_class.csv          per-class precision/recall/F1 across folds
  - nlp_confusion_top.csv      top off-diagonal confusion pairs
  - nlp_token_importance.csv   top tokens by linear-model coefficient
"""
from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

RUN_TAG = "nlp"


# ─── Activation thresholds (auditable & overridable) ───────────────────
# As in ``manifests/wdbc.py``: thresholds live here with their
# rationale rather than inside the detectors. The values match the
# 20newsgroups baseline behaviour validated on the bundled demo and
# can be rebound by callers via :func:`override_thresholds`.
DEFAULT_THRESHOLDS: dict[str, float] = {
    # Per-class macro-F1 spread (max - min). 0.15 is the boundary used
    # in the long-tail text-classification literature (e.g. Yang et al.
    # 2020 on imbalanced multiclass) above which class-level intervention
    # is typically warranted; 0.30 marks "severe spread" in the same
    # studies and triggers high severity here.
    "f1_spread_warning": 0.15,
    "f1_spread_high": 0.30,
    # Share of test rows accounted for by the single largest off-diagonal
    # confusion pair. 0.08 corresponds to ~1 in 12 examples being
    # systematically confused — a level at which a binary specialist or
    # a hierarchical relabel becomes cost-effective on 20newsgroups.
    "confusion_top_warning": 0.08,
    "confusion_top_high": 0.16,
}


THRESHOLDS: dict[str, float] = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    """Mutate the active threshold table — see WDBC manifest for the contract."""
    THRESHOLDS.update(dict(overrides))


# ─── Evidence checkers ────────────────────────────────────────────────
def _per_class_f1_spread_high(csv_path: Path) -> bool:
    """Active when min/max per-class F1 spread exceeds the configured
    warning threshold (default 0.15 — see ``THRESHOLDS``).

    A high spread means some classes are systematically harder than
    others — a classic 20newsgroups symptom (e.g. talk.religion.misc
    vs comp.* categories).
    """
    return _f1_spread_summary(csv_path)["f1_spread"] > THRESHOLDS["f1_spread_warning"]


def _f1_spread_summary(csv_path: Path) -> dict:
    f1s: list[float] = []
    if csv_path.exists():
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    f1s.append(float(r.get("f1", "nan")))
                except ValueError:
                    continue
    spread = (max(f1s) - min(f1s)) if f1s else 0.0
    return {"rows": len(f1s), "f1_spread": round(spread, 4)}


def _f1_spread_severity(csv_path: Path) -> str:
    spread = _f1_spread_summary(csv_path)["f1_spread"]
    if spread >= THRESHOLDS["f1_spread_high"]:
        return "high"
    if spread >= THRESHOLDS["f1_spread_warning"]:
        return "medium"
    return "low"


def _confusion_top_high(csv_path: Path) -> bool:
    """Active when the top-1 off-diagonal confusion exceeds the
    configured warning threshold (default 8% of test rows — see
    ``THRESHOLDS``).

    Suggests two classes are highly confusable — informative for the
    LAPPATO_MCB's ``confusable_classes`` weakness.
    """
    return _confusion_summary(csv_path)["top_share"] > THRESHOLDS["confusion_top_warning"]


def _confusion_summary(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"rows": 0, "top_share": 0.0}
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    top_share = 0.0
    if rows:
        try:
            top_share = float(rows[0].get("share", "0"))
        except ValueError:
            top_share = 0.0
    return {"rows": len(rows), "top_share": round(top_share, 4)}


def _confusion_severity(csv_path: Path) -> str:
    share = _confusion_summary(csv_path)["top_share"]
    if share >= THRESHOLDS["confusion_top_high"]:
        return "high"
    if share >= THRESHOLDS["confusion_top_warning"]:
        return "medium"
    return "low"


def _stopword_dominance(csv_path: Path) -> bool:
    """True when the top-3 features by absolute weight are short tokens (<5 chars).

    Symptom of insufficient stopword filtering or of a model that
    over-relies on function words rather than topical content.
    """
    tokens = _token_summary(csv_path)["top_tokens"]
    return len(tokens) >= 3 and all(len((t or "").strip()) < 5 for t in tokens)


def _token_summary(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"rows": 0, "top_tokens": []}
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))[:3]
    return {"rows": len(rows), "top_tokens": [r.get("token", "") for r in rows]}


# ─── Manifest ─────────────────────────────────────────────────────────
MANIFEST: list[dict] = [
    {
        "id": "per_class_f1_spread",
        "title": "Per-class F1 spread > 0.15 — some classes systematically harder",
        "evidence": "nlp_per_class.csv",
        "evidence_check": _per_class_f1_spread_high,
        "evidence_summary": _f1_spread_summary,
        "severity": _f1_spread_severity,
        "queries": [
            "long tail {task} text classification per class macro F1",
            "imbalanced multiclass classification {model_family} fine tuning",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Inspect the worst-F1 classes; consider class-weighted loss, "
            "data augmentation (back-translation) for the tail, or "
            "macro-F1 early stopping rather than accuracy."
        ),
        "why_it_matters": (
            "Macro performance hides whether a text classifier is useful across "
            "all labels; per-class F1 spread exposes brittle categories."
        ),
        "next_checks": [
            "Sort classes by F1 and inspect representative false negatives.",
            "Compare class weights, macro-F1 model selection and augmentation.",
            "Check whether train/test class supports differ.",
        ],
        "success_criteria": [
            "F1 spread drops below 0.15.",
            "Worst-class recall improves without large macro-F1 loss.",
        ],
        "references": [
            "long-tail text classification",
            "macro F1 early stopping",
            "class weighted multiclass learning",
        ],
    },
    {
        "id": "confusable_classes",
        "title": "A pair of classes accounts for > 8% of test rows in confusion",
        "evidence": "nlp_confusion_top.csv",
        "evidence_check": _confusion_top_high,
        "evidence_summary": _confusion_summary,
        "severity": _confusion_severity,
        "queries": [
            "confusable classes hierarchical text classification taxonomy",
            "contrastive learning nearest class boundary text",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Add a binary specialist head for the top confusable pair, "
            "or merge them into a coarse class and decompose downstream."
        ),
        "why_it_matters": (
            "A concentrated confusion pair usually indicates a missing taxonomy, "
            "feature representation weakness or label-policy ambiguity."
        ),
        "next_checks": [
            "Review examples in the top confusion pair.",
            "Try hierarchical labels or pairwise specialist classifiers.",
            "Inspect nearest-neighbor documents across the confused classes.",
        ],
        "success_criteria": [
            "Top confusion pair falls below 8% of test rows.",
            "The pair-specific F1 improves under the same split protocol.",
        ],
        "references": [
            "hierarchical text classification",
            "contrastive text classification",
            "confusable classes taxonomy",
        ],
    },
    {
        "id": "stopword_dominance",
        "title": "Top-3 model features are short tokens (likely stop-/function words)",
        "evidence": "nlp_token_importance.csv",
        "evidence_check": _stopword_dominance,
        "evidence_summary": _token_summary,
        "severity": "medium",
        "queries": [
            "stopword removal text classification feature engineering",
            "TF IDF n gram preprocessing text classification pipeline",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": (
            "Tighten preprocessing: apply a domain stopword list, raise "
            "min_df, or switch to character n-grams to dilute function-word "
            "dominance."
        ),
        "why_it_matters": (
            "Top coefficients dominated by very short tokens are a warning that "
            "the model may exploit formatting or function-word artefacts rather "
            "than topical signal."
        ),
        "next_checks": [
            "Audit top positive/negative tokens per class.",
            "Compare word n-grams with character n-grams.",
            "Run an ablation with stricter stopword and min_df settings.",
        ],
        "success_criteria": [
            "Top absolute coefficients include semantically meaningful tokens.",
            "Macro-F1 remains stable after preprocessing tightening.",
        ],
        "references": [
            "TF IDF preprocessing text classification",
            "stopword leakage",
            "character n-gram text classification",
        ],
    },
    {
        "id": "embedding_alternatives",
        "title": "TF-IDF is the only representation — sentence-transformer embeddings unrepresented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": [
            "sentence transformers MiniLM {task} benchmark 2025",
            "fastText vs TF IDF text classification small dataset 2025",
            "open source software text classification embeddings benchmark",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": (
            "Add a Sentence-Transformer embedding (paraphrase-MiniLM-L6) + "
            "logistic head as an alternative representation; compare on "
            "macro-F1 not accuracy."
        ),
        "why_it_matters": (
            "TF-IDF is strong but lexical; embeddings test whether semantic "
            "similarity reduces class confusions that share vocabulary."
        ),
        "next_checks": [
            "Compare TF-IDF, fastText and sentence-transformer embeddings.",
            "Evaluate macro-F1 and confusion-pair changes.",
            "Keep a linear head for fair representation comparison.",
        ],
        "success_criteria": [
            "Embedding baseline is reported under identical splits.",
            "Representation choice is justified by macro-F1 or confusion reduction.",
        ],
        "references": [
            "sentence transformers text classification",
            "fastText vs TF IDF",
            "small dataset text classification benchmark",
        ],
    },
    {
        "id": "calibration_missing_text",
        "title": "No probability calibration reported — softmax confidences are biased",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": [
            "softmax calibration text classifier temperature scaling 2025",
            "expected calibration error multiclass language model 2025",
            "open source software classifier calibration expected calibration error",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": (
            "Wrap the classifier output in temperature scaling on a "
            "held-out split; report Expected Calibration Error and a "
            "reliability diagram."
        ),
        "why_it_matters": (
            "Confidence scores from text classifiers are often used for routing "
            "or human review; uncalibrated probabilities make triage policies fragile."
        ),
        "next_checks": [
            "Report ECE, Brier score and reliability diagram.",
            "Compare temperature scaling and isotonic calibration.",
            "Calibrate on a held-out fold, not the test fold.",
        ],
        "success_criteria": [
            "ECE decreases after calibration.",
            "Macro-F1 is unchanged or changes only within sampling noise.",
        ],
        "references": [
            "temperature scaling text classifier",
            "expected calibration error",
            "classifier calibration evaluation",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG"]
