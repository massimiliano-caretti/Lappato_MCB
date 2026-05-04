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

from .anomaly_detection import MANIFEST as ANOMALY_DETECTION_MANIFEST
from .anomaly_detection import RUN_TAG as ANOMALY_DETECTION_TAG
from .causal_ml import MANIFEST as CAUSAL_ML_MANIFEST
from .causal_ml import RUN_TAG as CAUSAL_ML_TAG
from .clustering import MANIFEST as CLUSTERING_MANIFEST
from .clustering import RUN_TAG as CLUSTERING_TAG
from .cybersecurity import MANIFEST as CYBERSECURITY_MANIFEST
from .cybersecurity import RUN_TAG as CYBERSECURITY_TAG
from .fairness import MANIFEST as FAIRNESS_MANIFEST
from .fairness import RUN_TAG as FAIRNESS_TAG
from .geospatial import MANIFEST as GEOSPATIAL_MANIFEST
from .geospatial import RUN_TAG as GEOSPATIAL_TAG
from .graph_ml import MANIFEST as GRAPH_ML_MANIFEST
from .graph_ml import RUN_TAG as GRAPH_ML_TAG
from .llm_eval import MANIFEST as LLM_EVAL_MANIFEST
from .llm_eval import RUN_TAG as LLM_EVAL_TAG
from .medical_imaging import MANIFEST as MEDICAL_IMAGING_MANIFEST
from .medical_imaging import RUN_TAG as MEDICAL_IMAGING_TAG
from .nlp import MANIFEST as NLP_MANIFEST
from .nlp import RUN_TAG as NLP_TAG
from .rag_eval import MANIFEST as RAG_EVAL_MANIFEST
from .rag_eval import RUN_TAG as RAG_EVAL_TAG
from .recommender import MANIFEST as RECOMMENDER_MANIFEST
from .recommender import RUN_TAG as RECOMMENDER_TAG
from .rl_eval import MANIFEST as RL_EVAL_MANIFEST
from .rl_eval import RUN_TAG as RL_EVAL_TAG
from .speech_audio import MANIFEST as SPEECH_AUDIO_MANIFEST
from .speech_audio import RUN_TAG as SPEECH_AUDIO_TAG
from .survival import MANIFEST as SURVIVAL_MANIFEST
from .survival import RUN_TAG as SURVIVAL_TAG
from .tabular_generic import MANIFEST as TABULAR_GENERIC_MANIFEST
from .tabular_generic import RUN_TAG as TABULAR_GENERIC_TAG
from .timeseries import MANIFEST as TS_MANIFEST
from .timeseries import RUN_TAG as TS_TAG
from .vision import MANIFEST as VISION_MANIFEST
from .vision import RUN_TAG as VISION_TAG
from .wdbc import MANIFEST as WDBC_MANIFEST
from .wdbc import RUN_TAG as WDBC_TAG

# Map domain id -> (manifest, run_tag). The run_tag is a short string
# used by LAPPATO_MCB to namespace its log + JSONL + meta-log files
# inside the shared ``checkpoints/`` directory, so multiple domains can
# coexist without collision.
REGISTRY: dict[str, tuple[list[dict], str]] = {
    "anomaly_detection": (ANOMALY_DETECTION_MANIFEST, ANOMALY_DETECTION_TAG),
    "causal_ml":         (CAUSAL_ML_MANIFEST,         CAUSAL_ML_TAG),
    "clustering":        (CLUSTERING_MANIFEST,        CLUSTERING_TAG),
    "cybersecurity":     (CYBERSECURITY_MANIFEST,     CYBERSECURITY_TAG),
    "fairness":          (FAIRNESS_MANIFEST,          FAIRNESS_TAG),
    "geospatial":        (GEOSPATIAL_MANIFEST,        GEOSPATIAL_TAG),
    "graph_ml":          (GRAPH_ML_MANIFEST,          GRAPH_ML_TAG),
    "llm_eval":          (LLM_EVAL_MANIFEST,          LLM_EVAL_TAG),
    "medical_imaging":   (MEDICAL_IMAGING_MANIFEST,   MEDICAL_IMAGING_TAG),
    "nlp":               (NLP_MANIFEST,               NLP_TAG),
    "rag_eval":          (RAG_EVAL_MANIFEST,          RAG_EVAL_TAG),
    "recommender":       (RECOMMENDER_MANIFEST,       RECOMMENDER_TAG),
    "rl_eval":           (RL_EVAL_MANIFEST,           RL_EVAL_TAG),
    "speech_audio":      (SPEECH_AUDIO_MANIFEST,      SPEECH_AUDIO_TAG),
    "survival":          (SURVIVAL_MANIFEST,          SURVIVAL_TAG),
    "tabular_generic":   (TABULAR_GENERIC_MANIFEST,   TABULAR_GENERIC_TAG),
    "timeseries":        (TS_MANIFEST,                TS_TAG),
    "vision":            (VISION_MANIFEST,            VISION_TAG),
    "wdbc":              (WDBC_MANIFEST,              WDBC_TAG),
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
