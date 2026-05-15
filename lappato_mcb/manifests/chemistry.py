"""Chemistry manifest — cheminformatics, computational chemistry, sampling.

For pipelines that operate on molecular structures, reactions, or chemical
spaces (generative models, QSAR, force fields, retrosynthesis). Expected
CSVs in ``checkpoints/``:

  - ``chem_validity.csv``: ``set,n_total,n_valid``
  - ``chem_space_coverage.csv``: ``train_descriptor_min,train_descriptor_max,
    test_descriptor_min,test_descriptor_max`` (or ``coverage`` directly)
  - ``chem_thermodynamics.csv``: ``cycle,closure_error``
  - ``chem_reaction_balance.csv``: ``reaction,left_atoms,right_atoms``
    (atom counts per reaction, e.g. ``C:6;H:12;O:6``)
  - ``chem_stereo.csv``: ``n_input_stereo_centers,n_kept_stereo_centers``
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ._common import f, max_value, mean, rows, summary, values

RUN_TAG = "chemistry"

DEFAULT_THRESHOLDS = {
    "structure_validity_warning": 0.95,
    "chemical_space_coverage_warning": 0.80,
    "thermodynamic_closure_warning": 1.0,
    "stereo_loss_share_warning": 0.10,
}
THRESHOLDS = dict(DEFAULT_THRESHOLDS)


def override_thresholds(overrides: Mapping[str, float]) -> None:
    THRESHOLDS.update(dict(overrides))


# ─── detector 1 — structure validity ───────────────────────────────────
def _validity_rate(path: Path) -> float:
    items = rows(path)
    total_n = 0.0
    valid_n = 0.0
    for row in items:
        n = f(row, "n_total", "total")
        v = f(row, "n_valid", "valid")
        if n is None or v is None or n <= 0:
            continue
        total_n += n
        valid_n += v
    if total_n > 0:
        return valid_n / total_n
    rates = values(path, "validity", "valid_share", "fraction_valid")
    return mean(rates) if rates else 1.0


def _structure_validity_low(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    rate = _validity_rate(path)
    return rate < THRESHOLDS["structure_validity_warning"]


def _validity_summary(path: Path) -> dict:
    return summary(path, "valid_share", _validity_rate(path))


# ─── detector 2 — chemical space coverage ──────────────────────────────
def _space_coverage(path: Path) -> float:
    direct = values(path, "coverage", "covered_share")
    if direct:
        return min(direct)
    overlaps: list[float] = []
    for row in rows(path):
        tr_lo = f(row, "train_descriptor_min", "train_min")
        tr_hi = f(row, "train_descriptor_max", "train_max")
        te_lo = f(row, "test_descriptor_min", "test_min")
        te_hi = f(row, "test_descriptor_max", "test_max")
        if None in (tr_lo, tr_hi, te_lo, te_hi):
            continue
        if te_hi <= te_lo:
            continue
        lo = max(tr_lo, te_lo)
        hi = min(tr_hi, te_hi)
        overlap = max(hi - lo, 0.0) / (te_hi - te_lo)
        overlaps.append(overlap)
    return min(overlaps) if overlaps else 1.0


def _chemical_space_coverage_low(path: Path) -> bool:
    items = rows(path)
    if not items:
        return False
    return _space_coverage(path) < THRESHOLDS["chemical_space_coverage_warning"]


def _coverage_summary(path: Path) -> dict:
    return summary(path, "min_descriptor_coverage", _space_coverage(path))


# ─── detector 3 — thermodynamic consistency ────────────────────────────
def _thermodynamic_consistency_violation(path: Path) -> bool:
    return max_value(path, "closure_error", "cycle_error", "delta") > THRESHOLDS["thermodynamic_closure_warning"]


def _thermodynamics_summary(path: Path) -> dict:
    return summary(path, "max_closure_error", max_value(path, "closure_error", "cycle_error", "delta"))


# ─── detector 4 — reaction-balance violation ───────────────────────────
def _parse_atom_string(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not s:
        return out
    for part in s.replace(",", ";").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def _reaction_unbalanced_count(path: Path) -> int:
    bad = 0
    for row in rows(path):
        left = _parse_atom_string(row.get("left_atoms", "") or "")
        right = _parse_atom_string(row.get("right_atoms", "") or "")
        if not left or not right:
            continue
        keys = set(left) | set(right)
        if any(left.get(k, 0) != right.get(k, 0) for k in keys):
            bad += 1
    return bad


def _reaction_balance_violation(path: Path) -> bool:
    return _reaction_unbalanced_count(path) > 0


def _reaction_summary(path: Path) -> dict:
    return {"rows": len(rows(path)), "n_unbalanced": _reaction_unbalanced_count(path)}


# ─── detector 5 — stereochemistry information dropped ──────────────────
def _stereo_loss_share(path: Path) -> float:
    items = rows(path)
    total = 0.0
    kept = 0.0
    for row in items:
        n_in = f(row, "n_input_stereo_centers", "input_stereo")
        n_keep = f(row, "n_kept_stereo_centers", "kept_stereo")
        if n_in is None or n_keep is None or n_in <= 0:
            continue
        total += n_in
        kept += n_keep
    if total <= 0:
        return 0.0
    return 1.0 - (kept / total)


def _stereochemistry_information_dropped(path: Path) -> bool:
    return _stereo_loss_share(path) > THRESHOLDS["stereo_loss_share_warning"]


def _stereo_summary(path: Path) -> dict:
    return summary(path, "stereo_loss_share", _stereo_loss_share(path))


MANIFEST: list[dict] = [
    {
        "id": "structure_validity_low",
        "title": "Generated or curated structures fail validity checks",
        "evidence": "chem_validity.csv",
        "evidence_check": _structure_validity_low,
        "evidence_summary": _validity_summary,
        "severity": "high",
        "queries": [
            "molecular validity SMILES sanitization generative model",
            "valence check chemistry generative validity benchmark",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Add valence and sanitisation filters and report validity / uniqueness / novelty together.",
        "why_it_matters": "Invalid structures cannot be evaluated downstream and inflate apparent novelty.",
        "next_checks": [
            "Stratify validity by atom and bond pattern.",
            "Compare sanitisation and tokenisation strategies.",
        ],
        "success_criteria": [
            "Validity exceeds 0.95 on the held-out generation set.",
        ],
        "references": [
            "molecular validity",
            "SMILES sanitization",
            "generative chemistry benchmark",
        ],
    },
    {
        "id": "chemical_space_coverage_low",
        "title": "Test descriptors fall outside the training applicability domain",
        "evidence": "chem_space_coverage.csv",
        "evidence_check": _chemical_space_coverage_low,
        "evidence_summary": _coverage_summary,
        "severity": "high",
        "queries": [
            "chemical space domain applicability QSAR",
            "applicability domain regression cheminformatics",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Define an applicability domain (descriptor box / leverage / k-NN) and report extrapolation flags.",
        "why_it_matters": "Predictions outside the training applicability domain are extrapolations with weak guarantees.",
        "next_checks": [
            "Compute leverage / Mahalanobis distance for test compounds.",
            "Report metrics inside vs outside the applicability domain.",
        ],
        "success_criteria": [
            "At least 80% of test compounds fall inside the applicability domain.",
        ],
        "references": [
            "applicability domain",
            "QSAR extrapolation",
            "chemical space sampling",
        ],
    },
    {
        "id": "thermodynamic_consistency_violation",
        "title": "Thermodynamic cycle closure or Hess law check fails",
        "evidence": "chem_thermodynamics.csv",
        "evidence_check": _thermodynamic_consistency_violation,
        "evidence_summary": _thermodynamics_summary,
        "severity": "high",
        "queries": [
            "thermodynamic consistency benchmark Hess law computational chemistry",
            "free energy cycle closure validation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Audit the thermodynamic cycle, refit reference data, and report cycle-closure errors per pathway.",
        "why_it_matters": "Cycle-closure failures usually indicate inconsistent reference states or sampling protocols.",
        "next_checks": [
            "Re-run end-state pairs with the same protocol.",
            "Check for hysteresis between forward and reverse paths.",
        ],
        "success_criteria": [
            "Cycle-closure error stays below 1 kcal/mol (or chosen tolerance).",
        ],
        "references": [
            "Hess's law",
            "free energy cycle",
            "thermodynamic consistency",
        ],
    },
    {
        "id": "reaction_balance_violation",
        "title": "Reaction is not atom-balanced",
        "evidence": "chem_reaction_balance.csv",
        "evidence_check": _reaction_balance_violation,
        "evidence_summary": _reaction_summary,
        "severity": "high",
        "queries": [
            "reaction balance check cheminformatics dataset",
            "atom mapping reaction template validation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Run an atom-mapping pre-check and exclude unbalanced reactions before training or scoring.",
        "why_it_matters": "Unbalanced reactions corrupt training signal and break stoichiometric reasoning.",
        "next_checks": [
            "Atom-map every reaction and reject mismatches.",
            "Audit dataset provenance for stripped counter-ions or solvents.",
        ],
        "success_criteria": [
            "Zero unbalanced reactions remain in the modelling set.",
        ],
        "references": [
            "atom mapping",
            "reaction balance",
            "reaction template curation",
        ],
    },
    {
        "id": "stereochemistry_information_dropped",
        "title": "Stereo information is lost during preprocessing",
        "evidence": "chem_stereo.csv",
        "evidence_check": _stereochemistry_information_dropped,
        "evidence_summary": _stereo_summary,
        "severity": "info",
        "queries": [
            "stereochemistry SMILES preprocessing canonicalization",
            "chiral center cheminformatics representation",
        ],
        "sources": ["arXiv", "OpenAlex", "Crossref"],
        "transplant": "Use stereo-aware canonicalisation and represent chirality explicitly in tokens / fingerprints.",
        "why_it_matters": "Dropping stereo collapses enantiomers / diastereomers and corrupts activity prediction.",
        "next_checks": [
            "Audit the canonicalisation pipeline for stereo loss.",
            "Compare predictions on stereoisomer pairs.",
        ],
        "success_criteria": [
            "Stereo-loss share stays below 0.10 across the dataset.",
        ],
        "references": [
            "stereochemistry representation",
            "chirality SMILES",
            "canonical molecular representation",
        ],
    },
]


__all__ = ["MANIFEST", "RUN_TAG", "THRESHOLDS", "DEFAULT_THRESHOLDS", "override_thresholds"]
