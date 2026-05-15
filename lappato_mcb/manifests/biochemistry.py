"""Biochemistry manifest — proteins, enzymes, metabolic networks, drug discovery.

For pipelines on enzyme kinetics, protein structure, metabolic networks,
and assay-based drug discovery. Expected CSVs in ``checkpoints/``:

  - ``biochem_kinetics.csv``: ``enzyme,Km,Vmax``
  - ``biochem_pathway.csv``:  ``pathway,n_expected,n_observed`` (intermediates)
  - ``biochem_alignment.csv``: ``pair,rmsd``
  - ``biochem_assay_replicates.csv``: ``compound,replicate,response``
  - ``biochem_assay_conditions.csv``: ``parameter,assay_value,target_value``
    (e.g. pH, temperature)
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, mean, rows, summary

RUN_TAG = "biochemistry"

DEFAULT_THRESHOLDS = {
    "km_max_warning": 1.0e-2,
    "km_min_warning": 1.0e-9,
    "vmax_max_warning": 1.0e6,
    "pathway_completeness_warning": 0.80,
    "rmsd_warning": 3.0,
    "replicate_cv_warning": 0.30,
    "assay_condition_relative_gap_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


# ─── detector 1 — enzyme kinetics outlier ──────────────────────────────
def _kinetic_outlier_count(path: Path) -> int:
    bad = 0
    for row in rows(path):
        km = f(row, "Km", "km")
        vm = f(row, "Vmax", "vmax")
        if km is not None and (km <= 0 or km > THRESHOLDS["km_max_warning"] or km < THRESHOLDS["km_min_warning"]):
            bad += 1
            continue
        if vm is not None and (vm <= 0 or vm > THRESHOLDS["vmax_max_warning"]):
            bad += 1
    return bad


def _enzyme_kinetics_outlier(path: Path) -> bool:
    return _kinetic_outlier_count(path) > 0


def _kinetics_summary(path: Path) -> dict:
    return {"rows": len(rows(path)), "n_outliers": _kinetic_outlier_count(path)}


# ─── detector 2 — pathway completeness ─────────────────────────────────
def _pathway_completeness(path: Path) -> float:
    items = rows(path)
    expected = 0.0
    observed = 0.0
    for row in items:
        e = f(row, "n_expected", "expected")
        o = f(row, "n_observed", "observed")
        if e is None or o is None or e <= 0:
            continue
        expected += e
        observed += o
    return (observed / expected) if expected > 0 else 1.0


def _pathway_completeness_low(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    return _pathway_completeness(path) < THRESHOLDS["pathway_completeness_warning"]


def _pathway_summary(path: Path) -> dict:
    return summary(path, "pathway_completeness", _pathway_completeness(path))


# ─── detector 3 — structural alignment RMSD ────────────────────────────
def _structural_alignment_rmsd_high(path: Path) -> bool:
    return max_value(path, "rmsd", "RMSD") > THRESHOLDS["rmsd_warning"]


def _alignment_summary(path: Path) -> dict:
    return summary(path, "max_rmsd", max_value(path, "rmsd", "RMSD"))


# ─── detector 4 — assay replicate inconsistency ────────────────────────
def _replicate_cv_max(path: Path) -> float:
    groups: dict[str, list[float]] = {}
    for row in rows(path):
        cid = (row.get("compound") or row.get("id") or "").strip()
        v = f(row, "response", "value", "signal")
        if not cid or v is None:
            continue
        groups.setdefault(cid, []).append(v)
    cvs: list[float] = []
    for vals in groups.values():
        if len(vals) < 2:
            continue
        mu = mean(vals)
        if mu == 0:
            continue
        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        sd = var ** 0.5
        cvs.append(sd / abs(mu))
    return max(cvs) if cvs else 0.0


def _assay_replicate_inconsistency(path: Path) -> bool:
    return _replicate_cv_max(path) > THRESHOLDS["replicate_cv_warning"]


def _replicate_summary(path: Path) -> dict:
    return summary(path, "max_replicate_cv", _replicate_cv_max(path))


# ─── detector 5 — assay-condition mismatch ─────────────────────────────
def _assay_condition_relative_gap(path: Path) -> float:
    gaps: list[float] = []
    for row in rows(path):
        a = f(row, "assay_value", "assay")
        t = f(row, "target_value", "target")
        if a is None or t is None or t == 0:
            continue
        gaps.append(abs(a - t) / abs(t))
    return max(gaps) if gaps else 0.0


def _assay_conditions_mismatch(path: Path) -> bool:
    return _assay_condition_relative_gap(path) > THRESHOLDS["assay_condition_relative_gap_warning"]


def _assay_condition_summary(path: Path) -> dict:
    return summary(path, "max_relative_gap", _assay_condition_relative_gap(path))


MANIFEST: list[dict] = [
    {
        "id": "enzyme_kinetics_outlier",
        "title": "Reported Km / Vmax falls outside biologically plausible ranges",
        "evidence": "biochem_kinetics.csv",
        "evidence_check": _enzyme_kinetics_outlier,
        "evidence_summary": _kinetics_summary,
        "severity": "high",
        "queries": [
            "enzyme kinetics quality control Michaelis Menten",
            "Km Vmax curation database biochemistry",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Filter against curated kinetic ranges and refit on initial-rate data with model adequacy diagnostics.",
        "why_it_matters": "Out-of-range kinetic constants typically reflect fitting or unit errors, not real catalysis.",
        "next_checks": [
            "Inspect substrate concentration coverage relative to Km.",
            "Compare against curated databases (BRENDA / SABIO-RK ranges).",
        ],
        "success_criteria": [
            "All Km / Vmax values fall within documented physiological windows.",
        ],
        "references": [
            "Michaelis-Menten kinetics",
            "BRENDA enzyme database",
            "kinetic parameter curation",
        ],
    },
    {
        "id": "pathway_completeness_low",
        "title": "Metabolic pathway misses expected intermediates",
        "evidence": "biochem_pathway.csv",
        "evidence_check": _pathway_completeness_low,
        "evidence_summary": _pathway_summary,
        "severity": "high",
        "queries": [
            "metabolic pathway completeness gap filling",
            "genome scale metabolic model curation reconstruction",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run gap-filling against a reference reconstruction and inspect orphan reactions.",
        "why_it_matters": "Missing intermediates break flux balance and bias downstream stoichiometric inference.",
        "next_checks": [
            "Cross-reference observed metabolites with KEGG / MetaCyc.",
            "Audit reactions with no associated genes (orphans).",
        ],
        "success_criteria": [
            "Pathway completeness exceeds 0.80 on the expected intermediate set.",
        ],
        "references": [
            "metabolic network gap filling",
            "genome-scale model curation",
            "pathway completeness",
        ],
    },
    {
        "id": "structural_alignment_rmsd_high",
        "title": "Protein structural alignment RMSD exceeds tolerance",
        "evidence": "biochem_alignment.csv",
        "evidence_check": _structural_alignment_rmsd_high,
        "evidence_summary": _alignment_summary,
        "severity": "high",
        "queries": [
            "protein structure alignment RMSD quality",
            "TM-score structural superposition benchmark",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Switch to length-normalised metrics (TM-score / GDT) and inspect alignment lengths.",
        "why_it_matters": "Raw RMSD penalises flexible regions and can hide otherwise good topological matches.",
        "next_checks": [
            "Compute TM-score and global distance test.",
            "Inspect domain motion before computing alignment.",
        ],
        "success_criteria": [
            "Alignment quality is reported with a length-normalised score in addition to RMSD.",
        ],
        "references": [
            "TM-score",
            "structural alignment quality",
            "protein superposition",
        ],
    },
    {
        "id": "assay_replicate_inconsistency",
        "title": "Replicate measurements disagree beyond assay tolerance",
        "evidence": "biochem_assay_replicates.csv",
        "evidence_check": _assay_replicate_inconsistency,
        "evidence_summary": _replicate_summary,
        "severity": "high",
        "queries": [
            "biological replicate variance assay quality",
            "high throughput screening replicate reproducibility",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Compute per-compound CV / Z'-factor and exclude wells / plates above the dispersion ceiling.",
        "why_it_matters": "High replicate variance inflates apparent activity differences and rejects true hits.",
        "next_checks": [
            "Compute Z'-factor across plates.",
            "Inspect plate edge effects and DMSO controls.",
        ],
        "success_criteria": [
            "Maximum replicate CV stays below 0.30.",
        ],
        "references": [
            "Z' factor",
            "high-throughput screening QC",
            "biological replicate variance",
        ],
    },
    {
        "id": "assay_conditions_mismatch",
        "title": "Assay conditions diverge from the target biological context",
        "evidence": "biochem_assay_conditions.csv",
        "evidence_check": _assay_conditions_mismatch,
        "evidence_summary": _assay_condition_summary,
        "severity": "info",
        "queries": [
            "assay condition pH temperature relevance physiological",
            "in vitro to in vivo extrapolation assay context",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Document the assay envelope and flag predictions that extrapolate outside it.",
        "why_it_matters": "Conditions far from the physiological target can produce activity that does not transfer in vivo.",
        "next_checks": [
            "Tabulate assay vs target conditions.",
            "Run sensitivity analyses around pH / temperature.",
        ],
        "success_criteria": [
            "Relative condition gap stays within 10% of target values.",
        ],
        "references": [
            "in vitro in vivo extrapolation",
            "assay relevance",
            "physiological conditions screening",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
