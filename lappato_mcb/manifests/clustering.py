"""Clustering and unsupervised-learning manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import max_value, min_value, summary, values

RUN_TAG = "clustering"

DEFAULT_THRESHOLDS = {
    "silhouette_warning": 0.20,
    "stability_warning": 0.70,
    "noise_share_warning": 0.30,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


def _silhouette_low(path: Path) -> bool:
    vals = values(path, "silhouette", "silhouette_score")
    return bool(vals and min(vals) < THRESHOLDS["silhouette_warning"])


def _silhouette_summary(path: Path) -> dict:
    return summary(path, "min_silhouette", min_value(path, "silhouette", "silhouette_score"))


def _stability_low(path: Path) -> bool:
    vals = values(path, "ari", "nmi", "stability")
    return bool(vals and min(vals) < THRESHOLDS["stability_warning"])


def _stability_summary(path: Path) -> dict:
    return summary(path, "min_stability", min_value(path, "ari", "nmi", "stability"))


def _noise_high(path: Path) -> bool:
    return max_value(path, "noise_share", "outlier_share") > THRESHOLDS["noise_share_warning"]


def _noise_summary(path: Path) -> dict:
    return summary(path, "max_noise_share", max_value(path, "noise_share", "outlier_share"))


MANIFEST = [
    {
        "id": "weak_cluster_separation",
        "title": "Cluster separation is weak",
        "evidence": "clustering_quality.csv",
        "evidence_check": _silhouette_low,
        "evidence_summary": _silhouette_summary,
        "severity": "medium",
        "queries": ["clustering validation silhouette score unsupervised learning", "cluster validity indices machine learning"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compare cluster validity indices and inspect embeddings before treating clusters as real groups.",
        "why_it_matters": "Weak separation means clusters may be artifacts of the algorithm.",
        "next_checks": ["Plot clusters in embedding space.", "Compare k and algorithms."],
        "success_criteria": ["Silhouette exceeds 0.20 or downstream utility justifies clusters."],
        "references": ["silhouette score", "cluster validity", "unsupervised evaluation"],
    },
    {
        "id": "cluster_instability",
        "title": "Cluster assignments are unstable across seeds or bootstraps",
        "evidence": "clustering_stability.csv",
        "evidence_check": _stability_low,
        "evidence_summary": _stability_summary,
        "severity": "medium",
        "queries": ["clustering stability bootstrap adjusted rand index", "consensus clustering stability evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run bootstrap/seed stability and prefer clusters that are stable under perturbation.",
        "why_it_matters": "Unstable clusters are hard to interpret or operationalize.",
        "next_checks": ["Report ARI/NMI across seeds.", "Use consensus clustering."],
        "success_criteria": ["Stability metric exceeds 0.70."],
        "references": ["cluster stability", "adjusted Rand index", "consensus clustering"],
    },
    {
        "id": "noise_share_high",
        "title": "Large share of points is marked as noise/outlier",
        "evidence": "clustering_noise.csv",
        "evidence_check": _noise_high,
        "evidence_summary": _noise_summary,
        "severity": "medium",
        "queries": ["DBSCAN HDBSCAN noise points clustering parameter selection", "density based clustering noise evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref", "JOSS"],
        "transplant": "Tune density parameters and compare density-based methods with k-means/GMM baselines.",
        "why_it_matters": "High noise share can mean parameters are wrong or data lacks cluster structure.",
        "next_checks": ["Sweep epsilon/min_samples.", "Inspect noise points."],
        "success_criteria": ["Noise share is justified or falls below 0.30."],
        "references": ["DBSCAN", "HDBSCAN", "density clustering"],
    },
    {
        "id": "external_validation_missing",
        "title": "External or downstream validation of clusters is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["external validation clustering downstream task evaluation", "cluster interpretation validation unsupervised learning"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Validate clusters with external labels, expert review or downstream task utility.",
        "why_it_matters": "Internal cluster metrics alone do not prove usefulness.",
        "next_checks": ["Map clusters to known labels.", "Review cluster exemplars."],
        "success_criteria": ["Clusters have documented external or downstream value."],
        "references": ["external validation", "cluster interpretation", "downstream evaluation"],
    },
    {
        "id": "embedding_sensitivity_missing",
        "title": "Embedding or scaling sensitivity is not represented",
        "evidence": None,
        "evidence_check": None,
        "severity": "info",
        "queries": ["clustering sensitivity feature scaling embedding dimensionality reduction", "UMAP t-SNE clustering stability evaluation"],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Repeat clustering under scaling and embedding choices before interpreting groups.",
        "why_it_matters": "Unsupervised results can be dominated by preprocessing choices.",
        "next_checks": ["Compare scaled/unscaled features.", "Compare PCA/UMAP embeddings."],
        "success_criteria": ["Cluster conclusions are stable across reasonable preprocessing choices."],
        "references": ["feature scaling", "dimensionality reduction", "UMAP"],
    },
]

