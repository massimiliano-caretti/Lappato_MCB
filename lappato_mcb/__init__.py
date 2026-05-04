"""LAPPATO_MCB public API.

LAPPATO_MCB stands for Literature-Aware Pipeline Partner for Adaptive
Transplant Optimization. It is a lightweight framework for
machine-learning pipelines: it watches diagnostic checkpoint artefacts,
activates manifest-defined weaknesses, and harvests targeted scientific
literature from public sources.
"""
from __future__ import annotations

from .core import (
    DEFAULT_SCORE_WEIGHTS,
    GENERAL_SOURCES,
    LAPPATO_MCB,
    SOURCES,
    ScoreWeights,
)

__version__ = "1.0.0"

__all__ = [
    "LAPPATO_MCB",
    "ScoreWeights",
    "DEFAULT_SCORE_WEIGHTS",
    "SOURCES",
    "GENERAL_SOURCES",
    "__version__",
]
