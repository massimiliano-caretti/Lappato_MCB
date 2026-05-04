"""Manifest registry — one entry per supported domain.

A *manifest* is a list of weakness dicts (see ``lappato_mcb.core`` for the
expected schema: id, title, evidence, evidence_check, queries,
transplant). Domain-specific manifests live in sibling modules and are
exposed here under stable names so callers can pick by string id.

Adding a new domain is a three-step process:
  1. Drop a ``lappato_mcb/manifests/<domain>.py`` exporting MANIFEST + RUN_TAG.
  2. Register it in ``REGISTRY`` below.
  3. Provide a host pipeline that emits the CSVs the manifest gates on.
"""
from __future__ import annotations

from .nlp import MANIFEST as NLP_MANIFEST
from .nlp import RUN_TAG as NLP_TAG
from .timeseries import MANIFEST as TS_MANIFEST
from .timeseries import RUN_TAG as TS_TAG
from .wdbc import MANIFEST as WDBC_MANIFEST
from .wdbc import RUN_TAG as WDBC_TAG

# Map domain id -> (manifest, run_tag). The run_tag is a short string
# used by LAPPATO_MCB to namespace its log + JSONL + meta-log files
# inside the shared ``checkpoints/`` directory, so multiple domains can
# coexist without collision.
REGISTRY: dict[str, tuple[list[dict], str]] = {
    "wdbc":       (WDBC_MANIFEST, WDBC_TAG),
    "nlp":        (NLP_MANIFEST,  NLP_TAG),
    "timeseries": (TS_MANIFEST,   TS_TAG),
}


def get(domain: str) -> tuple[list[dict], str]:
    """Look up a manifest by domain id; raise KeyError on unknown."""
    if domain not in REGISTRY:
        raise KeyError(
            f"Unknown manifest domain {domain!r}. "
            f"Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[domain]


def naive_baseline_manifest(run_tag: str) -> list[dict]:
    """A single-entry manifest used for the baseline-vs-targeted experiment.

    Issues one untargeted query — what a researcher would type into a
    paper-search box without first inspecting any diagnostic CSV. Used
    by ``eval_baseline.py`` as the control arm.
    """
    return [{
        "id": "naive_baseline",
        "title": f"Naive baseline query for {run_tag}",
        "evidence": None,
        "evidence_check": None,
        "queries": [f"{run_tag} machine learning classification"],
        "transplant": "(baseline arm — no targeted recommendation)",
    }]


__all__ = ["REGISTRY", "get", "naive_baseline_manifest"]
