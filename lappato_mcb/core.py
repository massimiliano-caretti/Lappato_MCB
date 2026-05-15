# **********************************************************************
# *  lappato_mcb/core.py  --  LAPPATO_MCB core (multi-domain).   *
# *                                                                     *
# *  Daemon thread launched alongside a host ML pipeline. While the     *
# *  pipeline trains, polls a checkpoints/ folder for new diagnostic    *
# *  CSVs and harvests recent scholarly literature targeted at          *
# *  the SPECIFIC weaknesses observable in the current run.             *
# *                                                                     *
# *  Pure stdlib (urllib + xml + json + threading + csv). No new        *
# *  dependencies. No paid API keys. Two-line integration into any      *
# *  Python pipeline: start() at training launch, stop() at completion. *
# *                                                                     *
# *  LAPPATO_MCB features:                                                  *
# *    - cross-source dedup via title trigram + Jaccard (fingerprint.py)*
# *    - per-cycle quantitative meta-log (meta_log.py)                  *
# *    - offline replay-from-corpus mode (cache.py)                     *
# *    - pluggable manifest by run_tag — enables WDBC, NLP, TS demos    *
# **********************************************************************
"""LAPPATO_MCB — multi-domain core.

Background literature-harvesting daemon that takes a *manifest* (list
of weakness dicts) and a *run_tag* (short string used to namespace
output files in ``checkpoints/``). Defaults reproduce the bundled
WDBC demo.
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
import traceback
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .cache import CorpusCache
from .fingerprint import DEFAULT_JACCARD_TAU, TitleDeduper
from .meta_log import CycleMetrics, MetaLogWriter


# ─── Reranker plug-in interface (Phase 0 — stdlib-only contract) ──────
# Optional embedding-aware reranker. The core never imports an
# implementation; users opt in by passing an object that satisfies the
# Reranker Protocol to the LAPPATO_MCB constructor. Default behaviour
# (``reranker=None``) is bit-identical to v1.4 — no rerank, no extra
# field in weakness cards. The default implementation lives in the
# extras package ``lappato_mcb[embed]`` (proposed for Phase 2); third
# parties can plug their own retriever as long as it satisfies this
# Protocol.
#
# Determinism contract (Reproducibility expert's veto over breaches):
#   - same inputs ⇒ same output ordering, on every run, every machine;
#   - the reranker MUST NOT mutate ``hits`` in-place;
#   - the reranker MUST return one float per hit, in the same order;
#   - returning NaN / +inf is treated as "no signal" (skipped from blend);
#   - raising any exception ⇒ graceful degrade to lexical-only ranking.
@runtime_checkable
class Reranker(Protocol):
    """Minimum contract for an optional embedding-aware reranker.

    Implementations live OUTSIDE ``lappato_mcb/core.py``. The core
    only depends on the call shape, never on a concrete class.
    """

    def score(self, query: str, hits: list[dict]) -> list[float]:
        """Return one numeric score per hit, in the same order.

        Higher = more relevant. Score scale is implementation-defined
        but should typically lie in ``[0, 1]`` for cosine-similarity
        rerankers. Implementations can include identifying metadata
        as attributes (``model_sha``, ``model_version``,
        ``model_name``) for the weakness-card audit trail; they
        MUST NOT block on network I/O.
        """
        ...


# Below this threshold, the embedding signal is treated as "the
# model isn't sure" and the lexical ranking is left untouched. Set
# conservatively to favour graceful degrade.
_DEFAULT_RERANKER_FLOOR = 0.30
_DEFAULT_RERANKER_WEIGHT = 0.5

# Blend modes supported by ``_apply_reranker``.
#
#   "additive"  — Phase 0 default (back-compat). Computes
#                 ``lappato_score = lex + weight * max(rerank − floor, 0)``.
#                 The reranker can only ADD score (never demote).
#                 Floor is honoured. Conservative.
#
#   "rrf"       — Phase 0.6 addition (Cormack, Clarke & Buettcher,
#                 SIGIR 2009). Computes
#                 ``lappato_score = 1/(60 + rank_lex)
#                                 + weight * 1/(60 + rank_rerank)``.
#                 Scale-invariant — combines two heterogeneous
#                 rankings via ranks rather than raw scores. Floor
#                 is IGNORED (would be semantically out of place
#                 against ranks). On the Phase 0.5 benchmark this
#                 mode delivered +0.34 R@5 on the hard scenario
#                 vs. the additive default; documented in
#                 ``docs/reranker_benchmark_phase05.md``.
_VALID_BLEND_MODES = ("additive", "rrf")
_DEFAULT_BLEND_MODE = "additive"
_RRF_K_CONST = 60  # Cormack et al. 2009 default; not user-tunable.

# ─── Polite-use defaults ───────────────────────────────────────────────
# arXiv recommends >=3 s between queries; OpenAlex grants polite-pool
# access when a User-Agent with a mailto is provided. Crossref grants
# polite-pool access via the same mailto convention (in the User-Agent
# header or in a ?mailto= query parameter).
_ARXIV_DELAY_SEC = 3.0
_OPENALEX_DELAY_SEC = 0.25
_CROSSREF_DELAY_SEC = 0.25
_LAPPATO_MCB_MAILTO = os.environ.get("LAPPATO_MCB_MAILTO", "").strip()
_USER_AGENT = (
    f"lappato_mcb/1.0 (mailto:{_LAPPATO_MCB_MAILTO})"
    if _LAPPATO_MCB_MAILTO else "lappato_mcb/1.0"
)
_HTTP_TIMEOUT = 20

# HTTP retry policy. Two retries with exponential backoff bound the
# extra latency at ~3 s for transient failures (timeouts, 5xx, parse
# errors) while remaining polite. Errors that survive all retries are
# counted in the meta-log (n_<source>_errors) so a noisy upstream is
# auditable rather than silent.
_HTTP_MAX_RETRIES = 2
_HTTP_BACKOFF_BASE_SEC = 1.0

# Sources known to LAPPATO_MCB. JOSS is a specialised software-paper
# channel backed by Crossref ISSN filtering, not a fourth general index.
SOURCES: tuple[str, ...] = ("arXiv", "OpenAlex", "Crossref", "JOSS")
GENERAL_SOURCES: tuple[str, ...] = ("arXiv", "OpenAlex", "Crossref")


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for the deterministic local relevance score.

    The score is a sum of four contributions: query-token overlap with
    the title, query-token overlap with the abstract, a small recency
    bonus, and a logarithmic citation bonus. Defaults reflect the
    weights validated by ``examples/validate_score.py`` against the
    bundled hand-curated relevance set; callers can override any
    weight without touching the framework code.

    All weights are non-negative; ``recency_max_years`` clips the
    recency bonus so it does not dominate older but seminal work.
    """
    title_overlap: float = 4.0
    abstract_overlap: float = 1.5
    recency_per_year: float = 0.25
    recency_max_years: int = 4
    recency_base_year: int = 2022
    citation_log_weight: float = 0.15


DEFAULT_SCORE_WEIGHTS = ScoreWeights()

# Table-driven dispatch: source name -> search fn / polite-use delay.
# Defined after the search functions are declared (see end of file).

# Default cadence for short demo runs (~2-4 min): 90 s yields ~2 cycles.
_DEFAULT_POLL_SEC = 90.0


# ─── HTTP helpers (stdlib-only) ────────────────────────────────────────
def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return resp.read()


# Stop-words and year-tokens dropped before sending a query to arXiv:
# they bloat the recall set with off-topic hits because arXiv's relevance
# scorer still rewards any match on these low-signal tokens.
_QUERY_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "on", "in", "to", "with", "and", "or",
    "by", "is", "are", "be", "this", "that", "from", "as", "at", "it",
    "we", "our", "via", "using", "based",
    "2023", "2024", "2025", "2026", "2027",
})

# Adaptive AND-of-keywords cap. Querying with too many AND-ed terms
# collapses recall to zero on arXiv; querying with too few floods the
# result set with off-topic hits. The strategy is to start at the
# initial cap (informative) and, if the response is empty, fall back
# stepwise to a wider query down to a minimum cap (last-chance recall).
# At each retry we respect the arXiv 3 s polite-use spacing.
_ARXIV_INITIAL_KEYWORDS = 5
_ARXIV_MIN_KEYWORDS = 2

# Kept for backwards compatibility with code/tests that imported the
# old constant. New code should reference the INITIAL/MIN pair above.
_ARXIV_MAX_KEYWORDS = _ARXIV_INITIAL_KEYWORDS


def _arxiv_keywords(human_query: str) -> list[str]:
    """Return the ordered list of distinctive keywords for an arXiv query.

    Keeps insertion order and drops short tokens, stop-words and pure
    year tokens. The caller decides how many to AND together.
    """
    tokens: list[str] = []
    raw = human_query.replace("/", " ").replace("-", " ")
    for tok in raw.split():
        clean = "".join(c for c in tok if c.isalnum())
        if not clean or len(clean) < 3:
            continue
        if clean.lower() in _QUERY_STOPWORDS:
            continue
        tokens.append(clean)
    return tokens


def _arxiv_query_for(human_query: str, max_keywords: int = _ARXIV_INITIAL_KEYWORDS) -> str:
    """Rewrite a human query into an AND-of-abs-fields arXiv query.

    ``max_keywords`` caps the number of AND-ed terms. The default keeps
    the legacy single-shot behaviour for direct callers; the adaptive
    retry path in :func:`_arxiv_search` sweeps this argument from the
    initial cap down to ``_ARXIV_MIN_KEYWORDS`` until at least one hit
    is returned.
    """
    tokens = _arxiv_keywords(human_query)
    if not tokens:
        return f"all:{human_query}"
    tokens = tokens[: max(1, max_keywords)]
    return " AND ".join(f"abs:{t}" for t in tokens)


def _arxiv_fetch_one(query_string: str, max_results: int) -> list[dict]:
    """Single arXiv HTTP call — used both by the search and tests."""
    qs = urllib.parse.urlencode({
        "search_query": query_string,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    raw = _http_get(f"http://export.arxiv.org/api/query?{qs}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    out = []
    for entry in root.findall("a:entry", ns):
        eid = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("a:published", default="", namespaces=ns) or ""
        year = published[:4] if published else "?"
        out.append({
            "id": eid,
            "title": title,
            "abstract": summary,
            "year": year,
            "url": eid,
        })
    return out


def _arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """Hit the public arXiv API with adaptive keyword fallback.

    Strategy: try the most-restrictive query first (``_ARXIV_INITIAL_KEYWORDS``
    AND-ed terms). If the response is empty, retry once with a wider
    query (one keyword fewer) and so on until ``_ARXIV_MIN_KEYWORDS`` is
    reached. The arXiv 3 s polite-use spacing is honoured between
    retries so the worst-case cost of a single ``_arxiv_search`` call
    is bounded by ``(initial - min) * 3 s`` of extra latency on top of
    the network round-trips.
    """
    keywords = _arxiv_keywords(query)
    if not keywords:
        return _arxiv_fetch_one(f"all:{query}", max_results)
    upper = min(_ARXIV_INITIAL_KEYWORDS, len(keywords))
    lower = min(_ARXIV_MIN_KEYWORDS, upper)
    for n in range(upper, lower - 1, -1):
        q = " AND ".join(f"abs:{t}" for t in keywords[:n])
        results = _arxiv_fetch_one(q, max_results)
        if results:
            return results
        if n > lower:
            time.sleep(_ARXIV_DELAY_SEC)
    return []


def _openalex_recover_abstract(inv_idx: dict | None) -> str:
    """Reconstruct an abstract from OpenAlex's inverted-index encoding."""
    if not inv_idx:
        return ""
    pos2word: dict[int, str] = {}
    for word, positions in inv_idx.items():
        for p in positions:
            pos2word[p] = word
    if not pos2word:
        return ""
    return " ".join(pos2word[i] for i in sorted(pos2word))


def _openalex_search(query: str, max_results: int = 5) -> list[dict]:
    """Hit the OpenAlex works endpoint and return relevance-ranked recent works."""
    params = {
        "search": query,
        "filter": "publication_year:>2023",
        "per-page": max_results,
    }
    if _LAPPATO_MCB_MAILTO:
        params["mailto"] = _LAPPATO_MCB_MAILTO
    qs = urllib.parse.urlencode(params)
    raw = _http_get(f"https://api.openalex.org/works?{qs}")
    data = json.loads(raw.decode("utf-8", errors="replace"))
    out = []
    for w in data.get("results", []):
        out.append({
            "id": w.get("id", ""),
            "title": (w.get("title") or "").strip(),
            "abstract": _openalex_recover_abstract(w.get("abstract_inverted_index")),
            "year": str(w.get("publication_year", "?")),
            "url": (w.get("doi") or w.get("id") or ""),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
            "cited_by_count": w.get("cited_by_count", 0),
        })
    return out


def _crossref_year(message: dict) -> str:
    """Extract a 4-digit year from a Crossref ``issued.date-parts`` block."""
    issued = (message.get("issued") or {}).get("date-parts") or []
    if issued and isinstance(issued[0], list) and issued[0]:
        return str(issued[0][0])
    return "?"


def _strip_jats_tags(abs_raw: str) -> str:
    """Cheap tag-stripper for Crossref's <jats:p>-wrapped abstracts.

    Crossref records sometimes embed a JATS-XML fragment in the
    ``abstract`` field. A dedicated XML parser is over-engineering for
    what is in practice tag soup. The helper walks character-by-character
    and emits only the content at depth-0, which is robust enough for
    every record observed in production.
    """
    if not abs_raw.startswith("<"):
        return abs_raw
    depth = 0
    buf: list[str] = []
    for ch in abs_raw:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif depth == 0:
            buf.append(ch)
    return "".join(buf).strip()


def _crossref_query_works(
    query: str,
    *,
    max_results: int,
    filter_clause: str,
    default_venue: str = "",
    extra_fields: dict | None = None,
) -> list[dict]:
    """Internal Crossref ``/works`` query, shared by both
    :func:`_crossref_search` and :func:`_joss_search`.

    The two public callers differ only in:

      - ``filter_clause``: ``"from-pub-date:2024"`` for the general
        Crossref channel; ``"issn:2475-9066,from-pub-date:2020"`` for
        the JOSS-only channel.
      - ``default_venue``: the human-readable fall-back when Crossref
        omits ``container-title``. Empty string for general Crossref;
        ``"Journal of Open Source Software"`` for JOSS.
      - ``extra_fields``: additional key/value pairs to merge into
        every output record (e.g. ``{"software_channel": "JOSS"}``).

    All other behaviour — URL, polite-pool ``mailto``, abstract
    sanitation, DOI fallback for ``url`` — is identical and would
    drift between the two functions if duplicated, so it lives here.
    """
    params = {
        "query": query,
        "rows": max_results,
        "filter": filter_clause,
        "select": ("DOI,title,abstract,issued,container-title,"
                   "is-referenced-by-count,URL"),
    }
    if _LAPPATO_MCB_MAILTO:
        params["mailto"] = _LAPPATO_MCB_MAILTO
    qs = urllib.parse.urlencode(params)
    raw = _http_get(f"https://api.crossref.org/works?{qs}")
    data = json.loads(raw.decode("utf-8", errors="replace"))
    items = (data.get("message") or {}).get("items") or []
    out: list[dict] = []
    for w in items:
        title_list = w.get("title") or []
        venue_list = w.get("container-title") or []
        doi = w.get("DOI", "")
        record = {
            "id": doi,
            "title": (title_list[0] if title_list else "").strip(),
            "abstract": _strip_jats_tags((w.get("abstract") or "").strip()),
            "year": _crossref_year(w),
            "url": w.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
            "venue": (venue_list[0] if venue_list else default_venue),
            "cited_by_count": int(w.get("is-referenced-by-count", 0) or 0),
        }
        if extra_fields:
            record.update(extra_fields)
        out.append(record)
    return out


def _crossref_search(query: str, max_results: int = 5) -> list[dict]:
    """Hit the Crossref /works endpoint and return relevance-ranked recent works.

    Crossref is the third public scholarly source LAPPATO_MCB consults
    (alongside arXiv and OpenAlex). Like OpenAlex it accepts a
    ``mailto`` query parameter to opt into the polite pool — no API
    key required. We restrict to ``from-pub-date:2024`` to mirror the
    OpenAlex recency filter and keep the three sources comparable.
    """
    return _crossref_query_works(
        query,
        max_results=max_results,
        filter_clause="from-pub-date:2024",
    )


def _joss_search(query: str, max_results: int = 5) -> list[dict]:
    """Search JOSS papers via Crossref metadata.

    JOSS is not treated as a fourth general scholarly index. It is a
    specialised research-software channel: JOSS articles are short,
    peer-reviewed software papers with Crossref DOIs. Querying them via
    Crossref keeps LAPPATO_MCB zero-key and avoids adding another API
    dependency while surfacing citable software implementations that can
    make a weakness card more actionable.
    """
    return _crossref_query_works(
        query,
        max_results=max_results,
        filter_clause="issn:2475-9066,from-pub-date:2020",
        default_venue="Journal of Open Source Software",
        extra_fields={"software_channel": "JOSS"},
    )


# ─── LAPPATO_MCB ─────────────────────────────────────────────────────────
class LAPPATO_MCB:
    """Daemon-thread literature harvester for a host ML pipeline.

    Constructor parameters:
        project_root   : repo root; ``checkpoints/`` is created under it
        manifest       : list of weakness dicts (see manifests/*.py)
        run_tag        : short id used to namespace output files
        poll_interval  : seconds between cycles
        offline        : if True, do not hit the network — search the
                         local corpus cache instead (cache.py)
        corpus_path    : where to read/write the persistent corpus
                         (defaults to checkpoints/corpus.jsonl)
        fp_tau         : Jaccard threshold for cross-source dedup
                         (see fingerprint.py)
        score_weights  : optional :class:`ScoreWeights` override. When
                         omitted, the validated ``DEFAULT_SCORE_WEIGHTS``
                         are used.
        http_max_retries : number of retries on transient HTTP failures
                         per source (default 2 for a worst-case 3 attempts).
                         Errors that survive all retries are logged in the
                         meta-log under ``n_<source>_errors``.

    Public surface intentionally tiny: ``start()``, ``stop()``,
    ``is_running()``.
    """

    def __init__(
        self,
        project_root: Path,
        manifest: list[dict] | None = None,
        run_tag: str = "wdbc",
        poll_interval: float = _DEFAULT_POLL_SEC,
        *,
        offline: bool = False,
        corpus_path: Path | None = None,
        fp_tau: float = DEFAULT_JACCARD_TAU,
        score_weights: ScoreWeights | None = None,
        http_max_retries: int = _HTTP_MAX_RETRIES,
        reranker: Reranker | None = None,
        reranker_weight: float = _DEFAULT_RERANKER_WEIGHT,
        reranker_floor: float = _DEFAULT_RERANKER_FLOOR,
        blend_mode: str = _DEFAULT_BLEND_MODE,
    ):
        if blend_mode not in _VALID_BLEND_MODES:
            raise ValueError(
                f"blend_mode must be one of {_VALID_BLEND_MODES}, got "
                f"{blend_mode!r}. See docs/reranker_benchmark_phase05.md "
                f"for the empirical comparison."
            )
        self._root = Path(project_root)
        self._checkpoints = self._root / "checkpoints"
        self._run_tag = run_tag
        # Manifest defaults to WDBC for backward compat with the
        # original WDBC example call sites.
        if manifest is None:
            from .manifests.wdbc import MANIFEST as _wdbc
            manifest = _wdbc
        self._manifest = list(manifest)

        # Run-tagged artefact paths — enables multi-domain runs to
        # coexist in a single ``checkpoints/`` directory.
        self._log_path = self._checkpoints / f"{run_tag}_lappato_mcb_log.md"
        self._papers_jsonl_path = self._checkpoints / f"{run_tag}_lappato_mcb_papers.jsonl"
        self._meta_log_path = self._checkpoints / f"{run_tag}_lappato_mcb_meta.csv"
        self._cards_jsonl_path = self._checkpoints / f"{run_tag}_lappato_mcb_weakness_cards.jsonl"
        self._cards_latest_path = self._checkpoints / f"{run_tag}_lappato_mcb_weakness_cards.latest.json"

        self._poll = float(poll_interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Source-id dedup (one set per source) + cross-source title dedup.
        self._seen_ids: dict[str, set[str]] = {s: set() for s in SOURCES}
        self._seen_global_ids: set[str] = set()
        self._titles = TitleDeduper(tau=fp_tau)

        self._cycle_n = 0
        self._lock = threading.Lock()

        # Persistent corpus + offline mode.
        self._cache = CorpusCache(
            corpus_path or self._checkpoints / "corpus.jsonl"
        )
        self._offline = bool(offline)

        # Tunable knobs (relevance score weights + HTTP retry policy).
        self._score_weights = score_weights or DEFAULT_SCORE_WEIGHTS
        self._http_max_retries = max(0, int(http_max_retries))

        # Optional embedding-aware reranker. None ⇒ behaviour identical
        # to v1.4 (lexical-only, bit-for-bit). The reranker contract
        # is the Reranker Protocol declared at module top.
        self._reranker = reranker
        self._reranker_weight = float(reranker_weight)
        self._reranker_floor = float(reranker_floor)
        # Phase 0.6: which blending strategy ``_apply_reranker`` will
        # use. See ``_VALID_BLEND_MODES`` above for the trade-offs.
        self._blend_mode = blend_mode
        # Audit trail metadata captured once at init so weakness cards
        # can record which reranker (if any) shaped the ranking.
        self._reranker_audit: dict[str, str] = {}
        if reranker is not None:
            for attr in ("model_name", "model_version", "model_sha"):
                v = getattr(reranker, attr, None)
                if v is not None:
                    self._reranker_audit[attr] = str(v)

        # Quantitative self-instrumentation.
        self._meta = MetaLogWriter(self._meta_log_path)
        self._prior_paper_keys_by_weakness = self._seed_dedup_from_existing()
        self._last_card_by_weakness = self._load_last_cards()

    # ── lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"lappato_mcb-{self._run_tag}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 10.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> None:
        """Run one synchronous harvest cycle against current checkpoints.

        This is useful after a fast pipeline has already written its
        diagnostic artefacts, or in CI/examples where a deterministic
        one-cycle literature audit is preferable to a background daemon.
        It uses the same code path as the daemon loop and writes the same
        log, JSONL, meta-log and weakness-card sidecars.
        """
        self._cycle_n += 1
        if self._cycle_n == 1:
            self._write_start_banner()
        self._run_cycle()

    # ── main loop ──────────────────────────────────────────────────
    def _run_loop(self) -> None:
        try:
            self._write_start_banner()
            while not self._stop.is_set():
                self._cycle_n += 1
                try:
                    self._run_cycle()
                except Exception:
                    self._append("⚠️  Cycle crashed",
                                 f"```\n{traceback.format_exc()}\n```")
                self._stop.wait(self._poll)
            self._write_stop_banner()
        except Exception:
            try:
                self._append("💥 Fatal LAPPATO_MCB error",
                             f"```\n{traceback.format_exc()}\n```")
            except Exception:
                pass

    # ── one cycle ──────────────────────────────────────────────────
    def _run_cycle(self) -> None:
        ts = datetime.now()
        context = self._load_context()
        metrics = CycleMetrics(
            cycle=self._cycle_n,
            ts_started_iso=ts.isoformat(timespec="seconds"),
        )
        lines: list[str] = [f"_Cycle started {ts.strftime('%Y-%m-%d %H:%M:%S')}_\n"]

        evidence_files = sorted({
            w["evidence"] for w in self._manifest if w.get("evidence")
        })
        lines.append("**Evidence visible in `checkpoints/`:**\n")
        for fname in evidence_files:
            present = (self._checkpoints / fname).exists()
            lines.append(f"- {'✓' if present else '–'}  `{fname}`")
        lines.append("")

        active_ids: list[str] = []
        for w in self._manifest:
            if not self._is_active(w):
                continue
            active_ids.append(w["id"])
            rendered, card = self._harvest_for(w, metrics, context)
            lines.extend(rendered)
            self._record_weakness_card(card)

        metrics.n_active_weaknesses = len(active_ids)
        if not active_ids:
            lines.append("_No active weaknesses this cycle._")

        title = (
            f"## Cycle {self._cycle_n}  ·  "
            f"active: {', '.join(active_ids) if active_ids else '∅'}"
        )
        self._append(title, "\n".join(lines))
        self._meta.append(metrics.finalise())

    def _is_active(self, w: dict) -> bool:
        check = w.get("evidence_check")
        if check is None:
            return True
        evidence = w.get("evidence")
        if not evidence:
            return True
        try:
            return bool(check(self._checkpoints / evidence))
        except Exception:
            return False

    def _harvest_for(
        self,
        w: dict,
        metrics: CycleMetrics,
        context: dict,
    ) -> tuple[list[str], dict]:
        out: list[str] = []
        out.append(f"\n### {w['title']}")
        out.append(
            f"_Evidence: `{w.get('evidence') or '(structural — always active)'}`_\n"
        )
        severity = self._weakness_severity(w)
        evidence_summary = self._evidence_summary(w)

        new_papers: list[tuple[str, dict]] = []
        raw_hits_by_source = {s: 0 for s in SOURCES}
        rendered_queries = [
            self._render_query_template(q, context, w, severity)
            for q in w["queries"]
        ]
        source_pipeline = _source_pipeline_for(w)
        for q in rendered_queries:
            metrics.n_queries_executed += 1
            for source, delay in source_pipeline:
                hits = self._fetch(source, q, metrics)
                raw_hits_by_source[source] += len(hits)
                metrics.add_source_hits(source, len(hits))
                for hit in hits:
                    kept = self._consider_hit(w["id"], source, q, hit, metrics)
                    if kept is not None:
                        new_papers.append((source, kept))
                if not self._offline:
                    time.sleep(delay)
                if self._stop.is_set():
                    break
            if self._stop.is_set():
                break

        # Apply the optional embedding-aware reranker before sorting
        # so that the visible top-N reflects the blended ranking.
        # Default (``reranker=None``) is a no-op and the lexical
        # ordering wins bit-for-bit, matching v1.4 behaviour.
        rerank_contributed = self._apply_reranker(new_papers)

        if not new_papers:
            out.append("_No new literature this cycle._\n")
        else:
            new_papers.sort(
                key=lambda sp: float(sp[1].get("lappato_score", 0.0)),
                reverse=True,
            )
            for src, p in new_papers[:8]:
                out.extend(_format_paper_md(src, p))
            out.append("")

        if evidence_summary:
            summary_txt = ", ".join(
                f"{k}={v}" for k, v in evidence_summary.items()
                if v not in (None, "")
            )
            if summary_txt:
                out.append(f"**Evidence summary:** {summary_txt}")
        out.append(f"**Severity:** {severity}")
        out.append(f"**Transplant sketch:** {w['transplant']}\n")

        previous_keys = self._prior_paper_keys_by_weakness.get(w["id"], set())
        new_keys = {_paper_key(p) for _, p in new_papers}
        previous_card = self._last_card_by_weakness.get(w["id"], {})
        previous_severity = previous_card.get("severity")
        card = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cycle": self._cycle_n,
            "run_tag": self._run_tag,
            "weakness": w["id"],
            "title": w["title"],
            "evidence": w.get("evidence"),
            "evidence_present": (
                bool((self._checkpoints / w["evidence"]).exists())
                if w.get("evidence") else True
            ),
            "severity": severity,
            "previous_severity": previous_severity or "",
            "severity_changed": bool(previous_severity and previous_severity != severity),
            "evidence_summary": evidence_summary,
            "queries": rendered_queries,
            "raw_hits_by_source": raw_hits_by_source,
            "n_raw_hits": int(sum(raw_hits_by_source.values())),
            "n_papers_kept": len(new_papers),
            "n_new_vs_previous_runs": len(new_keys - previous_keys),
            "n_seen_before": len(new_keys & previous_keys),
            "top_papers": [
                {
                    "source": src,
                    "id": p.get("id", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "url": p.get("url", ""),
                    "score": p.get("lappato_score", 0.0),
                }
                for src, p in new_papers[:10]
            ],
            "why_it_matters": w.get("why_it_matters", ""),
            "next_checks": list(w.get("next_checks", []) or []),
            "success_criteria": list(w.get("success_criteria", []) or []),
            "references": list(w.get("references", []) or []),
            "transplant": w.get("transplant", ""),
        }
        # Reranker audit trail (Phase 0 + 0.6). Only emitted when a
        # reranker is configured; absent in the default
        # v1.4-compatible path so existing card consumers stay
        # unaffected. ``blend_mode`` lets a downstream consumer
        # interpret the scale of ``lappato_score`` in ``top_papers``:
        #   - additive: lexical baseline + clipped boost (≈ 0..10)
        #   - rrf:      sum of reciprocal ranks       (≈ 0..0.04)
        if self._reranker is not None:
            card["reranker"] = {
                "configured": True,
                "contributed": bool(rerank_contributed),
                "blend_mode": self._blend_mode,
                "weight": self._reranker_weight,
                "floor": (
                    self._reranker_floor
                    if self._blend_mode == "additive"
                    else None
                ),
                **self._reranker_audit,
            }
        return out, card

    # ── fetch + filter pipeline ────────────────────────────────────
    def _fetch(self, source: str, query: str,
               metrics: CycleMetrics | None = None) -> list[dict]:
        """Return hits for one (source, query) — online or offline.

        Online failures are retried up to ``self._http_max_retries``
        times with exponential backoff. If all attempts fail, the
        per-source error counter on ``metrics`` is incremented and an
        empty list is returned, keeping the harvest loop robust against
        transient upstream issues without silently masking them.
        """
        if self._offline:
            cached = self._cache.search(query, k=5)
            return [c for c in cached if c.get("source") == source]
        fn = _SOURCE_FN.get(source)
        if fn is None:
            return []
        attempts = self._http_max_retries + 1
        for attempt in range(attempts):
            try:
                return fn(query) or []
            except Exception:
                if attempt == attempts - 1:
                    if metrics is not None:
                        metrics.add_source_error(source)
                    return []
                time.sleep(_HTTP_BACKOFF_BASE_SEC * (2 ** attempt))
        return []

    def _consider_hit(
        self,
        weakness_id: str,
        source: str,
        query: str,
        hit: dict,
        metrics: CycleMetrics,
    ) -> dict | None:
        """Apply the two dedup paths; on success record + return the hit.

        Returns None when the hit is dropped (already seen by id, or
        near-duplicate of something already kept by trigram match).
        """
        eid = hit.get("id") or ""
        gid = _canonical_paper_id(hit)
        if gid and gid in self._seen_global_ids:
            metrics.n_id_dedup_blocked += 1
            return None
        # 1) source-native id dedup (cheap, identical-paper case)
        seen_set = self._seen_ids.setdefault(source, set())
        if not eid or eid in seen_set:
            metrics.n_id_dedup_blocked += 1
            return None
        # 2) cross-source title-fingerprint dedup
        is_dup, _, _ = self._titles.seen_or_register(hit.get("title", ""))
        if is_dup:
            metrics.n_fp_dedup_blocked += 1
            return None
        # passed both gates — register the source id and persist
        seen_set.add(eid)
        if gid:
            self._seen_global_ids.add(gid)
        hit["lappato_score"] = _score_hit(query, hit, self._score_weights)
        # Attach the matching query so an optional reranker (Phase 0
        # plug-in point) can score (query, title) pairs without
        # changing the persisted sidecar schema. The field is ignored
        # by every consumer in the v1.4 pipeline.
        hit["lappato_matched_query"] = query
        metrics.n_papers_kept += 1
        metrics.mark_first_paper()
        self._record_paper(weakness_id, source, query, hit)
        return hit

    # ── reranker plug-in (Phase 0 + 0.6) ───────────────────────────
    @staticmethod
    def _rrf_ranks(
        scores: list[float],
        original_indices: list[int] | None = None,
    ) -> list[int]:
        """Return 1-based ranks for ``scores`` (higher score = better rank).

        Tie-breaking is **stable** by original index so the rank
        produced for the same input is deterministic across runs and
        Python implementations.
        """
        n = len(scores)
        if original_indices is None:
            original_indices = list(range(n))
        # Sort indices by (-score, original_index) — ascending rank.
        order = sorted(range(n), key=lambda i: (-scores[i], original_indices[i]))
        ranks = [0] * n
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank
        return ranks

    def _apply_reranker(
        self,
        new_papers: list[tuple[str, dict]],
    ) -> bool:
        """Blend an optional embedding-aware reranker into ``lappato_score``.

        Mutates each paper's ``lappato_score`` in place when a
        reranker is configured. Returns ``True`` if the reranker
        contributed any signal, ``False`` otherwise (no reranker,
        no papers, reranker raised, no usable scores).

        Determinism contract — the reranker:
          - is called once per matched query, with all papers from
            that query in their existing order;
          - must return one float per paper, same order;
          - any exception is swallowed and the lexical ranking is
            kept (graceful degrade);
          - NaN / +/-Inf scores are skipped from the blend.

        Two blend modes (selected at construction time via
        ``blend_mode``):

          ``additive`` (default, back-compatible with v1.4)::

              lappato_score = lex + weight * max(rerank - floor, 0)

          The reranker only *adds* signal when it is confident
          (above ``reranker_floor``). It cannot demote a paper
          below its lexical baseline. Conservative; preferred when
          the reranker is unproven.

          ``rrf`` (Cormack et al. SIGIR 2009 Reciprocal Rank Fusion)::

              lappato_score = 1/(60 + rank_lex)
                            + weight * 1/(60 + rank_rerank)

          Scale-invariant fusion of the lexical and reranker
          rankings. Floor is ignored (RRF works on ranks). On the
          Phase 0.5 benchmark this mode delivered +0.34 R@5 on the
          hard scenario vs. additive default; see
          ``docs/reranker_benchmark_phase05.md``.
        """
        if self._reranker is None or not new_papers:
            return False

        # Group papers by their matching query so each call to
        # ``score`` is over a homogeneous (query, hits) pair.
        groups: dict[str, list[tuple[str, dict]]] = {}
        for src, p in new_papers:
            q = p.get("lappato_matched_query") or ""
            groups.setdefault(q, []).append((src, p))

        contributed = False
        for q, group in groups.items():
            hits_only = [p for _, p in group]
            try:
                scores = self._reranker.score(q, hits_only)
            except Exception:
                # Honour the graceful-degrade contract: the lexical
                # ranking continues exactly as without a reranker.
                continue
            if not isinstance(scores, list) or len(scores) != len(hits_only):
                continue

            # Sanitize the reranker output: anything non-finite is
            # treated as "no signal" (0.0). This keeps both blend
            # modes well-defined in edge cases.
            sanitized: list[float] = []
            any_finite = False
            for s in scores:
                try:
                    fs = float(s)
                except (TypeError, ValueError):
                    fs = 0.0
                if fs != fs or fs in (float("inf"), float("-inf")):
                    fs = 0.0
                else:
                    any_finite = True
                sanitized.append(fs)
            if not any_finite:
                continue

            if self._blend_mode == "additive":
                contributed |= self._apply_additive_blend(group, sanitized)
            else:  # "rrf" — validated at construction time
                contributed |= self._apply_rrf_blend(group, sanitized)
        return contributed

    def _apply_additive_blend(
        self,
        group: list[tuple[str, dict]],
        rerank: list[float],
    ) -> bool:
        """Phase 0 additive blend; one paper at a time, floor-clipped.

        Returns True iff at least one paper received a contribution
        > 0 (i.e. the reranker score was above ``reranker_floor``).
        """
        contributed = False
        for (_, paper), fs in zip(group, rerank):
            contribution = max(fs - self._reranker_floor, 0.0)
            if contribution <= 0.0:
                paper["lappato_rerank_score"] = fs
                continue
            paper["lappato_score"] = (
                float(paper.get("lappato_score", 0.0))
                + self._reranker_weight * contribution
            )
            paper["lappato_rerank_score"] = fs
            contributed = True
        return contributed

    def _apply_rrf_blend(
        self,
        group: list[tuple[str, dict]],
        rerank: list[float],
    ) -> bool:
        """Phase 0.6 Reciprocal Rank Fusion (Cormack et al. 2009).

        Combines the lexical ranking and the reranker ranking via
        sums of reciprocal ranks. Scale-invariant; ``reranker_floor``
        is ignored (would be semantically out of place against ranks).

        Mutates ``lappato_score`` to the fused score. Always returns
        True when at least one paper is in the group, since RRF
        always produces a (potentially identical) ranking.
        """
        if not group:
            return False
        lex = [float(p.get("lappato_score", 0.0)) for _, p in group]
        # Stable tie-breaking by original position keeps RRF
        # deterministic when several papers share a score.
        n = len(group)
        original_idx = list(range(n))
        lex_ranks = self._rrf_ranks(lex, original_idx)
        rerank_ranks = self._rrf_ranks(rerank, original_idx)
        for (_, paper), lr, rr, rs in zip(group, lex_ranks, rerank_ranks, rerank):
            fused = (
                1.0 / (_RRF_K_CONST + lr)
                + self._reranker_weight * 1.0 / (_RRF_K_CONST + rr)
            )
            paper["lappato_score"] = fused
            paper["lappato_rerank_score"] = rs
        return True

    # ── evidence-gated literature retrieval helpers ─────────────────
    def _load_context(self) -> dict:
        """Load optional query-template context from JSON sidecars.

        Users can write ``checkpoints/lappato_context.json`` or
        ``checkpoints/<run_tag>_lappato_context.json`` with small fields
        such as ``model_family``, ``task`` and ``domain``. Query strings in
        the manifest may then use ``{model_family}`` placeholders. Missing
        files or malformed JSON simply yield an empty context.
        """
        out: dict = {}
        for path in (
            self._checkpoints / "lappato_context.json",
            self._checkpoints / f"{self._run_tag}_lappato_context.json",
        ):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                # Log once per malformed context so the user can find
                # why ``{model_family}`` etc. are not being
                # substituted. Without this note the failure is
                # invisible because the daemon keeps running with an
                # empty context (fail-closed by design).
                self._append(
                    f"⚠️  Context file {path.name} failed to parse",
                    f"```\nJSONDecodeError: {exc}\n```\n"
                    "Manifest query placeholders will fall back to "
                    "literal text until the file is fixed."
                )
                continue
            if isinstance(data, dict):
                out.update({str(k): v for k, v in data.items()})
        return out

    def _render_query_template(
        self,
        query: str,
        context: dict,
        weakness: dict,
        severity: str,
    ) -> str:
        values = {
            **context,
            "run_tag": self._run_tag,
            "weakness_id": weakness.get("id", ""),
            "weakness_title": weakness.get("title", ""),
            "severity": severity,
        }
        try:
            return str(query).format_map(_SafeFormatDict(values))
        except Exception:
            return str(query)

    def _weakness_severity(self, w: dict) -> str:
        sev = w.get("severity") or w.get("severity_check")
        path = self._checkpoints / w["evidence"] if w.get("evidence") else None
        try:
            if callable(sev):
                return _normalise_severity(str(sev(path) if path else sev(None)))
            if isinstance(sev, str) and sev:
                return _normalise_severity(sev)
        except Exception:
            pass
        return "info" if not w.get("evidence") else "medium"

    def _evidence_summary(self, w: dict) -> dict:
        evidence = w.get("evidence")
        if not evidence:
            return {"type": "structural", "active": True}
        path = self._checkpoints / evidence
        summary_fn = w.get("evidence_summary")
        if callable(summary_fn):
            try:
                out = summary_fn(path)
                if isinstance(out, dict):
                    return out
            except Exception:
                return {"file": evidence, "present": path.exists(), "summary_error": True}
        return _generic_evidence_summary(path)

    def _seed_dedup_from_existing(self) -> dict[str, set[str]]:
        """Seed per-run dedup state from an existing paper sidecar.

        The daemon may be restarted, or ``run_once()`` may be called after
        previous cycles already wrote papers. Without this seed step, the
        same DOI/arXiv id could be appended again in a later process. This
        keeps the JSONL sidecar append-only in normal use while preventing
        duplicate records across restarts.
        """
        by_weakness: dict[str, set[str]] = {}
        if not self._papers_jsonl_path.exists():
            return by_weakness
        try:
            with self._papers_jsonl_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    wid = row.get("weakness", "")
                    if not wid:
                        continue
                    src = row.get("source", "")
                    eid = row.get("id", "")
                    if src and eid:
                        self._seen_ids.setdefault(src, set()).add(eid)
                    gid = _canonical_paper_id(row)
                    if gid:
                        self._seen_global_ids.add(gid)
                    title = row.get("title", "")
                    if title:
                        self._titles.seen_or_register(title)
                    by_weakness.setdefault(wid, set()).add(_paper_key(row))
        except Exception:
            return {}
        return by_weakness

    def _load_last_cards(self) -> dict[str, dict]:
        cards: dict[str, dict] = {}
        if not self._cards_jsonl_path.exists():
            return cards
        try:
            with self._cards_jsonl_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    wid = row.get("weakness", "")
                    if wid:
                        cards[wid] = row
        except Exception:
            return {}
        return cards

    def _record_weakness_card(self, card: dict) -> None:
        with self._lock:
            self._cards_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cards_jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(card, ensure_ascii=False) + "\n")
                fh.flush()
            self._last_card_by_weakness[card["weakness"]] = card
            latest = {
                "run_tag": self._run_tag,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "cards": list(self._last_card_by_weakness.values()),
            }
            self._cards_latest_path.write_text(
                json.dumps(latest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ── log I/O ────────────────────────────────────────────────────
    def _append(self, heading: str, body: str) -> None:
        chunk = f"\n{heading}\n\n{body}\n"
        with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(chunk)
                fh.flush()

    def _write_start_banner(self) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = ("OFFLINE (replay-from-corpus)" if self._offline
                else f"ONLINE ({' + '.join(SOURCES)})")
        banner = (
            f"# LAPPATO_MCB — {self._run_tag} log\n\n"
            f"_Run started {ts}_  ·  poll interval: {self._poll:.0f} s  ·  "
            f"weaknesses tracked: {len(self._manifest)}  ·  mode: {mode}\n\n"
            "This log is appended to by LAPPATO_MCB daemon "
            "while the host pipeline trains. Each cycle scans "
            "`checkpoints/` for new diagnostic CSVs and harvests "
            "literature targeted at the weaknesses currently observable. "
            "The presence of a paper in this log is NOT an endorsement: "
            "the researcher must read each abstract and judge relevance "
            "before adopting any suggestion.\n"
            "\n---\n"
        )
        with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(banner)
                fh.flush()

    def _write_stop_banner(self) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        s = self._titles.stats
        per_source = "  ·  ".join(
            f"unique {src} ids: {len(self._seen_ids.get(src, set()))}"
            for src in SOURCES
        )
        banner = (
            f"\n---\n_LAPPATO_MCB stopped {ts}_  ·  "
            f"cycles run: {self._cycle_n}  ·  {per_source}  ·  "
            f"trigram dedup: {s['duplicates_blocked']} blocked / "
            f"{s['unique_kept']} kept "
            f"(rate {s['dedup_rate']*100:.1f}%, τ={s['tau']:.2f})\n"
        )
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(banner)
                fh.flush()

    def _record_paper(self, weakness_id: str, source: str,
                      query: str, hit: dict) -> None:
        """Append one structured row to JSONL sidecar AND the corpus."""
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cycle": self._cycle_n,
            "run_tag": self._run_tag,
            "weakness": weakness_id,
            "source": source,
            "query": query,
            "id": hit.get("id", ""),
            "title": hit.get("title", ""),
            "abstract": hit.get("abstract", ""),
            "year": hit.get("year", ""),
            "url": hit.get("url", ""),
            "venue": hit.get("venue", ""),
            "cited_by_count": hit.get("cited_by_count", 0),
        }
        with self._lock:
            self._papers_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._papers_jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
        # Persistent corpus: only record when *online* (in offline mode
        # the row already came out of the corpus and re-recording it
        # would inflate counts).
        if not self._offline:
            self._cache.record(row)


# ─── module-private helpers ────────────────────────────────────────────
def _format_paper_md(src: str, p: dict) -> list[str]:
    """Render one harvested paper as the existing 4-line markdown block."""
    title = " ".join((p.get("title") or "").split())
    yr = p.get("year", "?")
    url = p.get("url") or ""
    venue = p.get("venue", "")
    cited = p.get("cited_by_count")
    meta = f"{src}, {yr}"
    if venue:
        meta += f", {venue}"
    if cited:
        meta += f", cited {cited}×"
    abstract = (p.get("abstract") or "").strip()
    if len(abstract) > 420:
        abstract = abstract[:420].rstrip() + "…"
    out = [f"- **{title}**  _({meta})_"]
    if url:
        out.append(f"  <{url}>")
    if abstract:
        out.append(f"  > {abstract}")
    score = p.get("lappato_score")
    if score not in (None, ""):
        try:
            out.append(f"  _LAPPATO score: {float(score):.2f}_")
        except (TypeError, ValueError):
            pass
    return out


class _SafeFormatDict(dict):
    """``str.format_map`` helper that drops unknown optional placeholders."""

    def __missing__(self, key):
        return ""


def _normalise_severity(raw: str) -> str:
    s = (raw or "").strip().lower()
    allowed = {"info", "low", "medium", "high", "critical"}
    return s if s in allowed else "medium"


def _generic_evidence_summary(path: Path) -> dict:
    """Small CSV summary used when a manifest has no custom summariser."""
    out = {"file": path.name, "present": path.exists()}
    if not path.exists():
        return out
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        out["rows"] = len(rows)
        out["columns"] = list(reader.fieldnames or [])
        return out
    except Exception:
        out["parse_error"] = True
        return out


def _tokens_for_score(text: str) -> set[str]:
    clean = "".join(c.lower() if c.isalnum() else " " for c in (text or ""))
    return {t for t in clean.split() if len(t) >= 3 and t not in _QUERY_STOPWORDS}


def _score_hit(
    query: str,
    hit: dict,
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> float:
    """Deterministic local relevance score, independent of external APIs.

    It preserves the zero-dependency design while making cross-source
    results comparable enough for logs/cards: title matches matter most,
    abstract matches help, recent work receives a small bonus, and
    citation count contributes logarithmically.

    The default weights are the shipped, validated values
    (``DEFAULT_SCORE_WEIGHTS``). Callers can pass a customised
    :class:`ScoreWeights` to retune the ranking — see
    ``examples/validate_score.py`` for a Spearman-rank validation
    harness against a small bundled relevance set.
    """
    import math

    q = _tokens_for_score(query)
    title = _tokens_for_score(hit.get("title", ""))
    abstract = _tokens_for_score(hit.get("abstract", ""))
    title_overlap = len(q & title)
    abstract_overlap = len(q & abstract)
    try:
        year = int(str(hit.get("year") or "0"))
    except ValueError:
        year = 0
    if year:
        recency_steps = max(0, min(weights.recency_max_years,
                                   year - weights.recency_base_year))
        recency = recency_steps * weights.recency_per_year
    else:
        recency = 0.0
    citations = float(hit.get("cited_by_count") or 0)
    citation_bonus = math.log1p(max(0.0, citations)) * weights.citation_log_weight
    return round(
        weights.title_overlap * title_overlap
        + weights.abstract_overlap * abstract_overlap
        + recency + citation_bonus,
        3,
    )


def _paper_key(row: dict) -> str:
    eid = _canonical_paper_id(row)
    if eid:
        return eid
    from .fingerprint import normalise_title
    return normalise_title(row.get("title", ""))


def _canonical_paper_id(row: dict) -> str:
    """Canonical cross-source identifier for exact-paper deduplication."""
    raw = (row.get("id") or row.get("url") or "").strip().lower()
    if not raw:
        return ""
    raw = raw.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    raw = raw.removeprefix("doi:")
    if "doi.org/" in raw:
        raw = raw.split("doi.org/", 1)[1]
    raw = raw.rstrip("/")
    return raw


def _source_pipeline_for(weakness: dict) -> tuple[tuple[str, float], ...]:
    """Return the source pipeline requested by a manifest entry.

    Manifest entries default to the three general literature indices.
    Entries that can benefit from citable research software may opt into
    JOSS explicitly with ``"sources": ["arXiv", "OpenAlex", "Crossref",
    "JOSS"]``. This keeps JOSS scientifically positioned as a software
    channel rather than as a general-purpose paper index.
    """
    requested = weakness.get("sources") or GENERAL_SOURCES
    allowed = set(SOURCES)
    requested_set = {s for s in requested if s in allowed}
    return tuple((s, d) for s, d in _SOURCE_PIPELINE if s in requested_set)


# ─── Source dispatch tables (declared after the search fns above) ─────
# Adding a fourth source means: write _<src>_search above, register the
# triple here. The harvest loop and the meta-log are otherwise generic.
_SOURCE_FN: dict[str, Callable[[str, int], list[dict]]] = {
    "arXiv": _arxiv_search,
    "OpenAlex": _openalex_search,
    "Crossref": _crossref_search,
    "JOSS": _joss_search,
}

_SOURCE_PIPELINE: tuple[tuple[str, float], ...] = (
    ("arXiv", _ARXIV_DELAY_SEC),
    ("OpenAlex", _OPENALEX_DELAY_SEC),
    ("Crossref", _CROSSREF_DELAY_SEC),
    ("JOSS", _CROSSREF_DELAY_SEC),
)


__all__ = [
    "LAPPATO_MCB",
    "SOURCES",
    "GENERAL_SOURCES",
    "ScoreWeights",
    "DEFAULT_SCORE_WEIGHTS",
]
