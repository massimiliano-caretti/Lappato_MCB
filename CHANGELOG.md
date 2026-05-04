# Changelog

All notable changes to LAPPATO_MCB are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-05-04

Domain expansion release. The manifest registry grows from 3 to 19
domains, covering most mainstream supervised, unsupervised, evaluation
and domain-specific ML tasks. The framework code, public API and the
existing 3 manifests (WDBC, NLP, time series) are unchanged.

### Added

- **Sixteen new manifest domains**, each with five evidence-gated or
  structural weakness entries (5 entries × 16 = 80 new
  weakness checks):
  - `anomaly_detection` — outlier / novelty detection diagnostics
  - `causal_ml` — causal-inference / treatment-effect estimation
  - `clustering` — unsupervised cluster-quality diagnostics
  - `cybersecurity` — intrusion / malware detection
  - `fairness` — group-fairness, demographic parity, EO/EOpp
  - `geospatial` — spatial-autocorrelation, CRS, leakage diagnostics
  - `graph_ml` — node/edge classification + link prediction
  - `llm_eval` — LLM evaluation harness diagnostics
  - `medical_imaging` — radiology / CT / MRI classification
  - `rag_eval` — retrieval-augmented generation evaluation
  - `recommender` — top-K recommendation, coverage, diversity
  - `rl_eval` — RL policy evaluation diagnostics
  - `speech_audio` — speech recognition / audio classification
  - `survival` — survival analysis, C-index, calibration
  - `tabular_generic` — domain-agnostic tabular fallback
  - `vision` — image classification diagnostics
- A shared `lappato_mcb/manifests/_common.py` module factoring out
  helper functions reused across the new manifests.

### Changed

- The test `test_three_domains_registered` was renamed to
  `test_manifest_domains_registered` and its expected set lifted
  from 3 to 19 domain ids (parametrised via the
  `EXPECTED_MANIFEST_DOMAINS` constant for easy future extensions).
- The report-pack registry still ships only the original 3 packs
  (WDBC, NLP, time series); domains without a registered pack fall
  back to the framework-side artefact rendering through the existing
  `_generic_fallback_pack` mechanism, so the new manifests are fully
  usable for retrieval and weakness-card emission with the standard
  reporting still working through the fallback.

### Backwards compatibility

- The constructor signature, manifest schema, weakness-card schema
  and meta-log columns are unchanged.
- Existing user code calling
  `lappato_mcb.manifests.get("wdbc"|"nlp"|"timeseries")` keeps
  working without modification.

## [1.0.0] — 2026-05-04

First production-ready release. The public API is stable
(`LAPPATO_MCB.start() / stop() / run_once()`), the core remains pure
stdlib, and every behavioural change ships with a regression test.

### Added

- **Adaptive arXiv keyword fallback** (W1). `_arxiv_search` now sweeps
  from the initial keyword cap (5) down to a minimum cap (2) and stops
  at the first non-empty result, so informative queries are no longer
  silently truncated to zero recall when the most-restrictive
  combination matches nothing. Polite-use 3 s spacing is honoured
  between retries.
- **Configurable relevance score** (W2). The local `_score_hit` weights
  are now a public `ScoreWeights` dataclass (`title_overlap`,
  `abstract_overlap`, `recency_per_year`, `recency_max_years`,
  `recency_base_year`, `citation_log_weight`). The `LAPPATO_MCB`
  constructor accepts `score_weights=ScoreWeights(...)` for callers
  that want to retune ranking. A new `examples/validate_score.py`
  harness reports Spearman rank correlation against a bundled
  hand-curated relevance set (`docs/score_validation/relevance_set.json`,
  ~17 entries). Baseline correlation is ~0.89.
- **Documented + overridable thresholds** (W3). Each manifest now
  exposes a `THRESHOLDS` dict with literature/empirical citations and
  an `override_thresholds()` helper for cohorts whose distribution
  warrants different cut-offs.
- **HTTP retry with exponential backoff + per-source error logging**
  (W4). `_fetch` retries up to `http_max_retries` times (default 2)
  with 1 s / 2 s backoff. Failures that survive every attempt
  increment a per-source counter recorded in the meta-log under the
  new `n_arxiv_errors`, `n_openalex_errors`, `n_crossref_errors`,
  `n_joss_errors` columns, so noisy upstreams are auditable rather
  than silent.
- **Topic-recall harness with bundled gold sets** (W5). New
  `examples/measure_recall.py` and `docs/gold_sets/{wdbc,nlp,timeseries}_gold.json`
  ship a small, maintainer-curated topic list per weakness and
  compute per-weakness, per-domain and macro `Recall@K`. Macro recall
  is computed only over weaknesses that fired in the source run.
  Honest scope: this is a sanity-check, not an absolute benchmark.
- **CI:** Python matrix expanded to 3.9 → 3.13. New `validate-score`
  job catches regressions to the score weights. New `lint` job runs
  `ruff` on `lappato_mcb`, `examples`, `tests`.
- **Project files:** `CONTRIBUTING.md`, GitHub issue templates
  (bug report, feature request, new manifest).

### Changed

- `_score_hit` accepts an optional `weights: ScoreWeights` argument
  while preserving the legacy positional signature.
- `User-Agent` advertises `lappato_mcb/1.0`.
- `pyproject.toml` classifier upgraded from `Development Status ::
  3 - Alpha` to `5 - Production/Stable`.
- `analysis.py` and meta-log analysis surface the new
  `total_http_errors` aggregate.

### Backwards compatibility

- The `LAPPATO_MCB` constructor signature remains source-compatible:
  the new `score_weights` and `http_max_retries` keyword arguments
  default to the prior behaviour.
- The meta-log CSV gains four columns; older log files parse cleanly
  because `analysis.py` defaults missing columns to `None`.
- `_arxiv_query_for(human_query)` retains its single-argument form;
  the new `max_keywords` argument has the legacy default.
- `_score_hit(query, hit)` still works with two positional arguments.

## [0.2.0]

Initial public alpha. Core daemon, three-source polite-pool retrieval
(arXiv + OpenAlex + Crossref) plus JOSS software channel, cross-source
title deduplication, weakness cards, offline corpus replay, three
bundled manifests (WDBC, NLP, time series).
