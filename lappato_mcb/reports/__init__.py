"""Report-pack registry — domain-specific parsing and plotting bundles.

A *report pack* tells the framework (analysis + plots + pdf_builder)
how to interpret the diagnostic CSVs emitted by a host pipeline for a
given run_tag, without baking that knowledge into the framework code.
This makes LAPPATO_MCB's reporting layer pluggable in the same way the
weakness-manifest registry made its querying layer pluggable.

A pack supplies:
  - the CSV filenames it owns (relative to checkpoints/);
  - parsers (path -> list[dict]) for each owned CSV;
  - a summarise(parsed: dict) -> dict mapping with headline statistics
    suitable for the PDF cover page (must include the keys the cover
    page renders; see lappato_mcb.reports.common.empty_pipeline_summary
    for the agreed schema);
  - a build_figures(parsed: dict) -> dict[fig_id, Figure] callable;
  - human-readable captions for the PDF (one per fig_id).

Adding a new domain is a three-step process:
  1. Drop ``lappato_mcb/reports/<domain>.py`` exporting a ``PACK`` instance.
  2. Drop ``lappato_mcb/manifests/<domain>.py`` exporting MANIFEST + RUN_TAG.
  3. Register the pack here in ``REGISTRY``.

The framework never inspects the contents of the pack beyond the
ReportPack interface, so the same code path serves WDBC, NLP, time
series, and any future domain (computer vision, RL, generative).
"""
from __future__ import annotations

from typing import Any

from ._pack import ReportPack
from .nlp import PACK as NLP_PACK
from .timeseries import PACK as TS_PACK
from .wdbc import PACK as WDBC_PACK

# Maps a run_tag -> ReportPack. Run tags are short string ids reused
# across the manifest registry and this report registry (the pair
# (manifest, pack) constitutes a domain).
REGISTRY: dict[str, ReportPack] = {
    WDBC_PACK.run_tag: WDBC_PACK,
    NLP_PACK.run_tag: NLP_PACK,
    TS_PACK.run_tag: TS_PACK,
}


def get(run_tag: str) -> ReportPack:
    """Look up a pack by run_tag; raise KeyError if unregistered.

    Returns a *generic* fallback pack (no pipeline CSVs, empty
    summary, empty figures) when ``run_tag`` is unknown — this keeps
    the reporting code path well-defined for the LAPPATO_MCB-only artefacts
    (papers, meta-log, baseline-comparison) when the user supplies
    their own custom manifest without a matching pack.
    """
    if run_tag in REGISTRY:
        return REGISTRY[run_tag]
    return _generic_fallback_pack(run_tag)


def _generic_fallback_pack(run_tag: str) -> ReportPack:
    """Empty pack used when a run_tag has no registered ReportPack.

    The framework still reports LAPPATO_MCB-side artefacts (papers,
    meta-log, baseline) but no domain-specific pipeline figures.
    """
    return ReportPack(
        run_tag=run_tag,
        pipeline_csvs={},
        parsers={},
        summarise=lambda parsed: _empty_pipeline_summary(),
        build_figures=lambda parsed: {},
        captions={},
        cover_block=None,
    )


def _empty_pipeline_summary() -> dict[str, Any]:
    """Stable schema for the cover page when no pipeline data is present."""
    return {
        "n_experiments": 0,
        "headline_metric_name": None,
        "headline_metric_mean": float("nan"),
        "headline_metric_sd": float("nan"),
        "headline_metric_n": 0,
        "secondary_metrics": [],
    }


__all__ = ["ReportPack", "REGISTRY", "get"]
