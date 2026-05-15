"""Recommender-system manifest."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import max_value, min_value, summary, values

RUN_TAG = "recommender"

DEFAULT_THRESHOLDS = {
    "cold_start_recall_warning": 0.05,
    "popularity_share_warning": 0.50,
    "coverage_warning": 0.20,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _cold_start_low(path: Path) -> bool:
    vals = values(path, "cold_start_recall", "new_item_recall", "new_user_recall")
    return bool(vals and min(vals) < THRESHOLDS["cold_start_recall_warning"])


def _cold_start_summary(path: Path) -> dict:
    return summary(path, "min_cold_start_recall", min_value(path, "cold_start_recall", "new_item_recall", "new_user_recall"))


def _popularity_bias_high(path: Path) -> bool:
    return max_value(path, "top_decile_recommendation_share", "popularity_share") > THRESHOLDS["popularity_share_warning"]


def _popularity_summary(path: Path) -> dict:
    return summary(path, "max_popularity_share", max_value(path, "top_decile_recommendation_share", "popularity_share"))


def _coverage_low(path: Path) -> bool:
    vals = values(path, "catalog_coverage", "item_coverage")
    return bool(vals and min(vals) < THRESHOLDS["coverage_warning"])


def _coverage_summary(path: Path) -> dict:
    return summary(path, "min_catalog_coverage", min_value(path, "catalog_coverage", "item_coverage"))


MANIFEST = [
    {
        "id": "cold_start_recall_low",
        "title": "Cold-start user or item recall is low",
        "evidence": "recommender_cold_start.csv",
        "evidence_check": _cold_start_low,
        "evidence_summary": _cold_start_summary,
        "severity": "high",
        "queries": ["cold start recommender systems hybrid content collaborative filtering", "new item recommendation cold start evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add content/user metadata baselines and report cold-start metrics separately.",
        "why_it_matters": "Overall ranking metrics hide failure on new users and new items.",
        "next_checks": ["Split metrics by user/item age.", "Compare hybrid recommenders."],
        "success_criteria": ["Cold-start recall exceeds 0.05 or improves against baseline."],
        "references": ["cold start", "hybrid recommender", "content-based recommendation"],
    },
    {
        "id": "popularity_bias",
        "title": "Recommendations are dominated by popular items",
        "evidence": "recommender_popularity.csv",
        "evidence_check": _popularity_bias_high,
        "evidence_summary": _popularity_summary,
        "severity": "medium",
        "queries": ["popularity bias recommender systems long tail diversification", "debiasing recommender systems popularity bias"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Report long-tail exposure and compare popularity-regularized ranking.",
        "why_it_matters": "Popularity bias can reduce discovery and unfairly suppress niche items.",
        "next_checks": ["Plot exposure by item popularity decile.", "Measure long-tail recall."],
        "success_criteria": ["Popularity share falls below 0.50 while relevance remains stable."],
        "references": ["popularity bias", "long-tail recommendation", "diversification"],
    },
    {
        "id": "catalog_coverage_low",
        "title": "Catalog coverage is low",
        "evidence": "recommender_coverage.csv",
        "evidence_check": _coverage_low,
        "evidence_summary": _coverage_summary,
        "severity": "medium",
        "queries": ["catalog coverage recommender systems diversity evaluation", "recommender system coverage novelty diversity metrics"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add coverage, novelty and diversity metrics next to ranking metrics.",
        "why_it_matters": "High ranking accuracy can still recommend from a tiny subset of the catalog.",
        "next_checks": ["Track unique recommended items.", "Compare diversity-aware reranking."],
        "success_criteria": ["Catalog coverage exceeds 0.20 without large relevance loss."],
        "references": ["coverage", "diversity", "novelty"],
    },
    {
        "id": "temporal_leakage_unchecked",
        "title": "Temporal leakage in train/test split is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["temporal leakage recommender systems offline evaluation", "time based split recommender system evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use time-based splits and ensure future interactions do not inform training.",
        "why_it_matters": "Random splits often overstate recommender performance.",
        "next_checks": ["Audit split timestamps.", "Compare random and temporal splits."],
        "success_criteria": ["Reported metrics use a temporal holdout."],
        "references": ["offline evaluation", "temporal split", "data leakage"],
    },
    {
        "id": "online_offline_gap_missing",
        "title": "Offline-to-online metric gap is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["offline online evaluation gap recommender systems A B testing", "counterfactual evaluation recommender systems implicit feedback"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Plan counterfactual or A/B evaluation before treating offline ranking gains as product gains.",
        "why_it_matters": "Offline improvements may not translate to user behavior.",
        "next_checks": ["Define online KPI.", "Estimate exposure bias."],
        "success_criteria": ["Offline metric is linked to an online validation plan."],
        "references": ["counterfactual evaluation", "A/B testing", "exposure bias"],
    },
]

