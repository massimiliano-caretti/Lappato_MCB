"""Biology manifest — genomics, ecology, evolution, comparative biology.

For pipelines on high-throughput sequencing, comparative-species data,
ecological surveys, and population genetics. Expected CSVs in
``checkpoints/``:

  - ``bio_batch_variance.csv``: ``source,variance`` with ``source`` in
    ``between_batch`` / ``within_batch``
  - ``bio_replicates.csv``: ``condition,n_replicates``
  - ``bio_geneset_overlap.csv``: ``set_a,set_b,jaccard``
  - ``bio_phylogenetic.csv``: ``analysis,phylogenetic_correction`` (1/0)
  - ``bio_contamination.csv``: ``sample,contam_share``
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ._common import f, max_value, rows, summary, values

RUN_TAG = "biology"

DEFAULT_THRESHOLDS = {
    "batch_variance_ratio_warning": 1.0,
    "min_biological_replicates_warning": 3.0,
    "geneset_overlap_warning": 0.50,
    "phylogenetic_uncorrected_share_warning": 0.50,
    "contamination_share_warning": 0.05,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


# ─── detector 1 — batch effect detected ────────────────────────────────
def _batch_variance_ratio(path: Path) -> float:
    between = 0.0
    within = 0.0
    for row in rows(path):
        src = (row.get("source") or row.get("level") or "").strip().lower().replace("-", "_")
        v = f(row, "variance", "var")
        if v is None:
            continue
        if src.startswith("between"):
            between += v
        elif src.startswith("within") or src.startswith("intra"):
            within += v
    if within <= 0:
        return 0.0
    return between / within


def _batch_effect_detected(path: Path) -> bool:
    return _batch_variance_ratio(path) > THRESHOLDS["batch_variance_ratio_warning"]


def _batch_summary(path: Path) -> dict:
    return summary(path, "between_within_ratio", _batch_variance_ratio(path))


# ─── detector 2 — low biological replicate count ──────────────────────
def _min_replicates(path: Path) -> float:
    counts = values(path, "n_replicates", "replicates", "n")
    return min(counts) if counts else 0.0


def _low_biological_replicate_n(path: Path) -> bool:
    counts = values(path, "n_replicates", "replicates", "n")
    if not counts:
        return False
    return min(counts) < THRESHOLDS["min_biological_replicates_warning"]


def _replicate_summary(path: Path) -> dict:
    return summary(path, "min_replicates", _min_replicates(path))


# ─── detector 3 — gene set overlap inflation ───────────────────────────
def _gene_set_overlap_high(path: Path) -> bool:
    return max_value(path, "jaccard", "overlap", "share") > THRESHOLDS["geneset_overlap_warning"]


def _geneset_summary(path: Path) -> dict:
    return summary(path, "max_geneset_overlap", max_value(path, "jaccard", "overlap", "share"))


# ─── detector 4 — phylogenetic correction missing ──────────────────────
def _uncorrected_share(path: Path) -> float:
    items = rows(path)
    if not items:
        return 0.0
    bad = 0
    total = 0
    for row in items:
        v = (
            row.get("phylogenetic_correction")
            or row.get("pgls")
            or row.get("corrected")
        )
        if v is None or v == "":
            continue
        total += 1
        s = str(v).strip().lower()
        if s in ("0", "false", "no", "n", "missing"):
            bad += 1
    return (bad / total) if total else 0.0


def _phylogenetic_signal_unmeasured(path: Path) -> bool:
    return _uncorrected_share(path) > THRESHOLDS["phylogenetic_uncorrected_share_warning"]


def _phylogenetic_summary(path: Path) -> dict:
    return summary(path, "uncorrected_share", _uncorrected_share(path))


# ─── detector 5 — contamination warning ────────────────────────────────
def _contamination_warning(path: Path) -> bool:
    return max_value(path, "contam_share", "contamination", "alien_share") > THRESHOLDS["contamination_share_warning"]


def _contamination_summary(path: Path) -> dict:
    return summary(path, "max_contamination_share", max_value(path, "contam_share", "contamination", "alien_share"))


MANIFEST: list[dict] = [
    {
        "id": "batch_effect_detected",
        "title": "Between-batch variance dominates within-batch variance",
        "evidence": "bio_batch_variance.csv",
        "evidence_check": _batch_effect_detected,
        "evidence_summary": _batch_summary,
        "severity": "high",
        "queries": [
            "batch effect correction high-throughput sequencing",
            "ComBat surrogate variable analysis batch correction",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Apply ComBat / SVA / RUV correction and re-evaluate variance partitioning before biological inference.",
        "why_it_matters": "Uncorrected batch effects masquerade as biological signal and produce non-reproducible results.",
        "next_checks": [
            "Visualise PCA coloured by batch.",
            "Re-fit downstream models on corrected counts.",
        ],
        "success_criteria": [
            "Between-batch variance falls below within-batch variance.",
        ],
        "references": [
            "ComBat batch correction",
            "surrogate variable analysis",
            "high-throughput batch effects",
        ],
    },
    {
        "id": "low_biological_replicate_n",
        "title": "Biological replicates are below the inference floor (n < 3)",
        "evidence": "bio_replicates.csv",
        "evidence_check": _low_biological_replicate_n,
        "evidence_summary": _replicate_summary,
        "severity": "high",
        "queries": [
            "biological replicate sample size differential expression",
            "RNA-seq replicate power analysis",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Increase biological replicates and run a power calculation for the targeted effect size.",
        "why_it_matters": "Differential analyses with too few replicates have unreliable variance estimates and high false-discovery rates.",
        "next_checks": [
            "Run a power analysis for the planned contrast.",
            "Distinguish biological from technical replicates.",
        ],
        "success_criteria": [
            "Each condition has at least 3 biological replicates.",
        ],
        "references": [
            "biological vs technical replicates",
            "RNA-seq power analysis",
            "differential expression sample size",
        ],
    },
    {
        "id": "gene_set_overlap_high",
        "title": "Gene-set comparisons share too many members",
        "evidence": "bio_geneset_overlap.csv",
        "evidence_check": _gene_set_overlap_high,
        "evidence_summary": _geneset_summary,
        "severity": "high",
        "queries": [
            "gene set overlap multiple testing correction",
            "pathway enrichment redundancy curation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Cluster overlapping gene sets and report independent representatives only, with explicit FDR control.",
        "why_it_matters": "Highly overlapping gene sets inflate the apparent number of independent discoveries.",
        "next_checks": [
            "Run set-similarity clustering before enrichment.",
            "Use overlap-aware multiple-testing corrections.",
        ],
        "success_criteria": [
            "Reported sets have pairwise Jaccard below 0.50.",
        ],
        "references": [
            "gene set redundancy",
            "pathway enrichment",
            "overlap aware multiple testing",
        ],
    },
    {
        "id": "phylogenetic_signal_unmeasured",
        "title": "Comparative analyses lack phylogenetic correction",
        "evidence": "bio_phylogenetic.csv",
        "evidence_check": _phylogenetic_signal_unmeasured,
        "evidence_summary": _phylogenetic_summary,
        "severity": "high",
        "queries": [
            "phylogenetic comparative method correction PGLS",
            "phylogenetic signal Pagel lambda Blomberg K",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use phylogenetic generalised least squares or independent contrasts when comparing across species.",
        "why_it_matters": "Cross-species observations are not independent samples; ignoring shared ancestry inflates effect sizes.",
        "next_checks": [
            "Fit PGLS and report Pagel's λ.",
            "Compare ordinary regressions to phylogenetically informed models.",
        ],
        "success_criteria": [
            "All cross-species claims are reported with a phylogenetic correction.",
        ],
        "references": [
            "PGLS phylogenetic generalised least squares",
            "Felsenstein independent contrasts",
            "phylogenetic signal",
        ],
    },
    {
        "id": "contamination_warning",
        "title": "Contamination signal exceeds the safe-floor in one or more samples",
        "evidence": "bio_contamination.csv",
        "evidence_check": _contamination_warning,
        "evidence_summary": _contamination_summary,
        "severity": "high",
        "queries": [
            "contamination detection sequencing decontam",
            "host DNA cross-sample contamination microbiome",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run decontam-style filtering against negative controls and re-do downstream analyses on cleaned reads.",
        "why_it_matters": "Even small contamination fractions can dominate low-biomass results and produce spurious findings.",
        "next_checks": [
            "Inspect alien-read fractions per sample.",
            "Re-run with stricter decontamination thresholds.",
        ],
        "success_criteria": [
            "Maximum contamination share stays below 5%.",
        ],
        "references": [
            "decontam",
            "kraken2 contamination",
            "low-biomass microbiome QC",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
