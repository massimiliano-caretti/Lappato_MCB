"""Ad-hoc literature search, independent of any manifest.

See :mod:`lappato_mcb.literature.search`.
"""
from __future__ import annotations

from .search import MAX_RESULTS_PER_SOURCE, search_literature

__all__ = ["MAX_RESULTS_PER_SOURCE", "search_literature"]
