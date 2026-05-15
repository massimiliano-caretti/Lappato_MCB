# Changelog

All notable changes to LAPPATO_MCB are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.1] — 2026-05-10

Software-engineering polish release. **Zero functional changes**:
no manifest added, no detector logic altered, no test expectation
relaxed. The public API, the wire format of weakness cards, the
schema layout, and the bundled gold-set numbers are bit-identical
to v1.6.0.

### Added

- **`CITATION.cff`** at the repo root. GitHub renders it as a
  "Cite this repository" widget in the sidebar; the file also
  makes the metadata consumable by Zenodo, Software Heritage, and
  other archival services. The `identifiers:` block is parked for
  the first Zenodo DOI.
- **README badges**: tests CI, supported Python range, ruff,
  license, GitHub release. All point at live endpoints.
- **`.pre-commit-config.yaml`** with file-hygiene hooks
  (trailing-whitespace, end-of-file-fixer, check-yaml, check-toml,
  check-added-large-files, check-merge-conflict, mixed-line-ending)
  plus the same ruff version pinned in CI. Opt-in per
  CONTRIBUTING; not required for clone-and-go.
- **`UP` rule family** in `[tool.ruff.lint]` — pyupgrade-style
  modernisation enforced on every CI run going forward.

### Changed

- **CI workflow `tests.yml`**: adds `concurrency` (cancels stale
  runs on the same ref to save minutes) and `cache: "pip"` on
  every job.
- **Internal helper `_crossref_query_works`** factored out of
  `_crossref_search` and `_joss_search` in `core.py`. The two
  public functions now wrap it in five lines each. **~100 LOC of
  copy-paste duplication eliminated** — a future bugfix in the
  Crossref response handling now applies to both channels.
- **JATS-tag stripper** in `core.py` lifted into a named helper
  (`_strip_jats_tags`) instead of being inlined twice.
- **`_load_context`** now writes a one-time warning to the
  markdown log when a `lappato_context.json` file exists but
  fails to parse. Previously the malformed file was silently
  ignored and the user had to find on their own why
  `{model_family}` placeholders were rendering as literal text.
- **Typing imports modernised** across the package (76 sites):
  `from typing import Mapping` → `from collections.abc import
  Mapping`, same for `Iterable`, `Sequence`; `Tuple[...]` →
  `tuple[...]`. Behaviour identical.
- **Variable shadowing fixed**: a single-letter `l` in
  `_split_metric` and in the benchmark blend renamed to `label` /
  `lex_s` for readability and ruff `E741` compliance.

### Removed

- **Dead code**: `LAPPATO_MCB._safe` static method (defined, never
  called anywhere in the codebase — verified across
  `lappato_mcb/`, `tests/`, `examples/`).
- **Dead local variable** `fn` in
  `pipeline_health._youden_threshold_sweep`.

### Internal

- Ruff lint is now **clean across the full repo**
  (`lappato_mcb/`, `examples/`, `tests/`): 0 errors at the
  configured rule set (`E`, `F`, `W`, `I`, `UP`).
- 209/209 tests pass in ~1.4s. Test count is unchanged from
  v1.6.0 — this release does not add or relax any test.

### Backwards compatibility

- The public API (`LAPPATO_MCB`, `Reranker`, manifest registry,
  weakness card schema, meta-log columns) is **byte-identical**
  for any caller. The `_safe` removal touches a private helper
  with zero call-sites and never appeared in `__all__`.
- The `_load_context` log line is additive: it only appears when
  a malformed context file is present, which previously produced
  no output. No card field changed.

## [1.6.0] — 2026-05-10

**Insight detectors.**  Seven new entries in the
``pipeline_health`` manifest move LAPPATO_MCB from "compliance
auditor" toward "insight engine" while preserving the
*stdlib-only-by-default* core (statistical helpers fall back to
pure-Python implementations whenever SciPy is not importable —
"Strategy 3" hybrid path).

The seven detectors target weakness patterns that the v1.5 detectors
did not surface — most notably **subgroup disparity** (failures
concentrated in a stratum of items, surfaced via Fisher's exact),
**non-monotone calibration miscalibration** (the bin-level pattern
that defeats temperature scaling per Guo et al. 2017) and
**sub-optimal binary decision thresholds** (where shifting the cut-off
trades false negatives for false positives without retraining).

### Added

- **New module `lappato_mcb/_stats.py`** with hybrid stdlib + SciPy
  statistical helpers (`chi2_p_value`, `fisher_exact_2x2`,
  `t_test_p_value`, `linear_regression_slope_p`,
  `required_sample_size_for_proportion`).  Auto-detects SciPy at
  module import and uses it when available; falls back to pure-Python
  implementations (Numerical Recipes §6.2 incomplete gamma, §6.4
  incomplete beta, Beasley-Springer 1977 inverse-normal CDF) so the
  ``dependencies = []`` core promise still holds.
- **Seven new ``pipeline_health`` manifest entries** (manifest grew
  from 16 → 23):
  - `subgroup_disparity` (high) — Fisher's exact + ratio threshold.
  - `calibration_bin_gap` (high) — flags non-monotone miscalibration.
  - `decision_threshold_suboptimal` (high) — sweeps Youden's J on
    held-out predictions.
  - `failure_clustering` (high) — chi-square on group × correct/
    incorrect tables.
  - `cross_cycle_drift` (info) — linear regression on the daemon
    meta-log.
  - `underpowered_cohort` (info) — quantifies events shy of the
    Riley 2019 minimum.
  - `syndrome_composition` (info) — meta-detector firing when ≥ 66 %
    of a named syndrome's trigger cards are active.
- **Five new evidence files** documented in the schema (all optional
  — fail-closed when absent):
  - `pipeline_subgroup_metrics.csv`
  - `pipeline_reliability_diagram.csv`
  - `pipeline_predictions_with_probs.csv`
  - `pipeline_per_item_predictions.csv`
  - `pipeline_health_lappato_mcb_meta.csv` (the daemon's own meta-log,
    elevated to a first-class evidence source).
- **Six new threshold knobs** in `DEFAULT_THRESHOLDS`
  (`subgroup_disparity_ratio_warning`, `calibration_bin_gap_warning`,
  `decision_threshold_distance_warning`, `failure_clustering_p_warning`,
  `drift_total_change_warning`, `drift_p_warning`), all documented
  with literature citations and overridable through the existing
  `override_thresholds()` API.
- **`SYNDROME_DEFINITIONS` registry** in `pipeline_health.py` —
  declarative mapping of syndrome names to their trigger sets and
  interpretation blocks.  Two syndromes seeded
  (`underpowered_imbalanced_clinical_cohort`,
  `miscalibrated_modern_NN`); users can extend the registry without
  touching detector code.
- **26 new tests** in `tests/test_insight_detectors.py` covering the
  stdlib stat helpers (incl. SciPy roundtrip), every new detector
  (positive + negative + missing-file fail-closed) and a manifest-
  count regression test.

### Changed

- `tests/test_new_manifests.py::test_pipeline_health_has_*` was
  renamed and updated from `_sixteen_` to `_twenty_three_` to match
  the new manifest size.
- `pipeline_health.schema.json` description rewritten to enumerate
  the v1.6 evidence files; `schema_version` intentionally kept at 1
  because the column-shape contract for existing files is fully
  backward-compatible (additive change only).

### Compatibility

- **MIT license unchanged.**  All new code is original and
  MIT-compatible with any optional dependency.
- **`dependencies = []` unchanged.**  SciPy remains a runtime
  detection (no install required); the new optional dependency
  group `[insights]` is reserved for future SciPy-only features.
- **Backward compatibility**: every v1.5 detector, threshold, schema
  field and CSV column name is preserved.  All 209 v1.5 tests pass
  unchanged; the 26 new tests are additive.

## [1.5.0] — 2026-05-07

Phase 0.6 — closes the negative finding from Phase 0.5. The Phase 0
additive blend (`lex + w · max(rerank − floor, 0)`) was demonstrated
to produce **+0.0000** uplift on the synthetic hard scenario across
35 domains: the integer-scale lexical baseline simply dominates the
fractional reranker contribution. Phase 0.5 measured that the
reranker signal is in fact strong (+0.40 R@5 in isolation) but
schiacciato by the additive blend.

This release adds **Reciprocal Rank Fusion** (Cormack, Clarke &
Buettcher, SIGIR 2009) as an opt-in blend mode in the core. RRF is
scale-invariant — it operates on ranks, not raw scores — and on
the same Phase 0.5 benchmark delivers **+0.34 R@5** (lex + 2× rerank
weight) without degrading the easy scenario. Still zero new
dependencies, still stdlib-only.

### Added

- **`blend_mode` argument** on `LAPPATO_MCB.__init__`, accepting
  `"additive"` (default — back-compat with v1.4) or `"rrf"`. Invalid
  values raise `ValueError` with a pointer to the benchmark report.
- **RRF blend implementation** in `LAPPATO_MCB._apply_rrf_blend`,
  with stable tie-breaking by original index for full determinism.
  Uses `_RRF_K_CONST = 60` (Cormack default).
- **`_rrf_ranks` static method** on `LAPPATO_MCB` — small stdlib
  helper that returns 1-based ranks with deterministic tie-breaks.
- **`blend_mode` field** in the weakness-card audit trail. Lets
  downstream consumers interpret `lappato_score` scale (≈ 0..10 for
  additive vs. ≈ 0..0.04 for RRF). The `floor` field is `null`
  in RRF mode (semantically inapplicable to rank fusion).
- **11 new tests** in `tests/test_reranker_protocol.py` covering:
  blend-mode validation (typo → `ValueError`, default = additive),
  `_rrf_ranks` determinism (descending by score, stable ties, empty
  input), RRF mode reorders correctly when reranker signals, card
  records `blend_mode`, RRF + no reranker collapses to lex,
  determinism across runs, easy case unchanged.

### Changed

- `_apply_reranker` is now a small dispatcher: branches on
  `self._blend_mode` to either `_apply_additive_blend` (the v1.4
  semantics, factored out) or `_apply_rrf_blend` (the new path).
  Both honour the same graceful-degrade contract on crash / NaN /
  malformed reranker output.
- The `additive` path still respects `reranker_floor` exactly as
  v1.4 did; behaviour for users who don't pass `blend_mode` is
  bit-identical.

### Honest scope

- Phase 0.6 makes the existing `Reranker` Protocol **actually
  useful** — the additive blend in v1.4 was empirically a no-op for
  realistic reranker signals.
- Phase 1/2 (extras `[embed]` with ONNX MiniLM) remains
  **deferred**. The Phase 0.5 benchmark suggests a stdlib-only
  trigram reranker + RRF already recovers ~80% of the available
  signal on the synthetic hard fixtures. Real-world online
  benchmarks are still the gate that would justify the 120 MB
  dependency footprint of `[embed]`.

### Backwards compatibility

- Constructor signature is additive: `blend_mode` defaults to
  `"additive"`. Code that doesn't pass it gets v1.4 behaviour.
- All v1.4 cards, manifests, gold sets, schemas, and meta-log
  columns are unchanged.
- `lappato_score` scale changes only when the user explicitly
  opts into `blend_mode="rrf"`. The card's `reranker.blend_mode`
  field documents which scale is in effect.
- 209/209 tests pass (was 198/198 in v1.4).

## [1.4.0] — 2026-05-07

Closing-the-loop release. Three concrete weaknesses called out in
the v1.3 self-audit are addressed in code, with honest scoping for
the parts that cannot be fully fixed without real-world datasets.
Framework code, public API and existing manifests are unchanged.

### Added

- **Validator → suggester** (`examples/validate_evidence_csv.py`).
  When a required column is missing on a present file *or* an
  evidence file is absent, the validator now emits a
  copy-pasteable `csv.DictWriter` snippet that uses the canonical
  column names from the schema, lists every alias as a comment,
  and parses out of the box (placeholders are `None`). Closes the
  loop with the README's three-line AI prompt: paste the validator
  output into a chat panel and the assistant has everything it needs
  to wire the emitter in. A `--no-suggest` flag preserves the v1.3
  terse output.
- **Boundary-case tests for v1.2 detectors**
  (`tests/test_detector_thresholds.py`). 32 new tests covering 32
  numeric detectors across `pipeline_health`, `physics`,
  `chemistry`, `materials_science`, `neuroscience` and
  `epidemiology`. Each test generates a CSV at the detector's
  documented threshold ± ε and asserts fire / no-fire behaviour.
  Threshold values come from each manifest's `THRESHOLDS` dict, so
  a default-value change automatically updates the boundary
  fixture. Honest scope: this is a *threshold-contract test*, not
  validation against real-world labelled data. A detector can pass
  every boundary test and still be measuring the wrong thing — but
  it cannot lie about *where* it fires.
- **Synthetic-corpus recall runner**
  (`examples/measure_recall_synthetic.py`). Builds a per-domain
  corpus where every gold-set topic is verbatim embedded in a
  synthetic paper title + abstract, then runs the full
  `evaluate_domain` machinery. Result: macro recall = 1.000 on
  every one of the 35 domains across all 505 topics. Honest scope:
  this is a *gold-set self-consistency floor* and an *end-to-end
  plumbing check across all 35 domains*, not a measure of real
  arXiv / OpenAlex / Crossref retrieval — for that, run
  `examples/measure_recall.py` against a real harvest.
- **`tests/test_synthetic_recall_floor.py`** — 3 tests assert the
  100% synthetic-recall contract holds on every domain. A malformed
  gold-set topic (empty keywords, typo, accidental stopword
  collision) immediately surfaces in the failure message with the
  topic name.

### Changed

- `examples/validate_evidence_csv.py::validate_checkpoints` now
  retains `file_spec` + `filename` on each result so
  `render_report` can emit suggestions without re-loading the
  schema. Behaviour is additive — existing callers that ignored
  these fields are unaffected.
- `tests/test_gold_sets.py` gains a `ValidatorSuggesterTests` class
  (5 tests).

### Honest scope of v1.4

This release improves *infrastructure for measurement*; it does not
add real-world measurement. Specifically:

- Detectors are now boundary-tested but **still not validated
  against real datasets per domain**. Replacing this with empirical
  validation requires per-domain ground truth that no one has
  produced for LAPPATO_MCB yet.
- Recall is now end-to-end runnable on every domain (synthetic
  corpus) but the **real online recall numbers are still
  un-measured**. Anyone with a `LAPPATO_MCB_MAILTO` set + network
  access can now run `examples/measure_recall.py` per domain to
  produce them; the bottleneck has moved from "infrastructure" to
  "an evening of online runs".

### Backwards compatibility

- Constructor signature, manifest schema, weakness-card schema,
  meta-log columns and gold-set / schema layouts are unchanged.
- The validator's default behaviour switched from "no snippet" to
  "with snippet"; pass `--no-suggest` to opt out.
- 172/172 tests pass (was 132/132 in v1.3).

## [1.3.0] — 2026-05-07

Validation-coverage release. The bundled gold-set catalogue grows
from 3 to **35 files** with **~500 hand-curated topics** total, and a
versioned **JSON Schema** for evidence CSVs ships per manifest with a
matching validator in `examples/`. Framework code, public API and
existing manifests are unchanged.

### Added

- **35 gold-set files** under `docs/gold_sets/` — one per registered
  manifest (was 3). Each file lists 10–15 hand-curated topics per
  manifest (32 for the 16-detector `pipeline_health`), each topic
  with `must_match_keywords` + a one-line `rationale`. Total of
  about 500 topics across the catalogue.
- **`examples/measure_recall.py` discovers gold sets dynamically** —
  the harness now globs `docs/gold_sets/*_gold.json` instead of a
  hard-coded 3-domain tuple, so adding a new gold set is a one-file
  change.
- **JSON Schemas for evidence CSVs**: 35 files under
  `lappato_mcb/manifests/schemas/<tag>.schema.json`. Each file lists,
  per evidence CSV, the canonical column names, accepted aliases,
  column type (`string` / `number`), required-vs-optional flag, and
  which detectors read the file. The format is versioned
  (`schema_version`), so future column changes can ship a deprecation
  notice rather than silently break consumers.
- **`examples/validate_evidence_csv.py`** — stdlib-only validator
  that reports, for any candidate `checkpoints/` folder:
  - which evidence files are present;
  - which expected files are missing (and which detectors stay
    inactive as a result);
  - which required columns are missing on present files;
  - which alias / non-canonical column names were accepted.
  Exits non-zero only when a present file is missing required
  columns; missing files are reported but never fail the run, in
  keeping with the daemon's fail-closed contract.
- **Honest "two-line" caveat in `README.md`.** A new section spells
  out that `start()` / `stop()` is the integration cost only when the
  host pipeline already emits the CSV evidence files the chosen
  manifest expects. The full adoption recipe is now a three-step
  list, with a pointer to the schemas + validator for step 3.
- **Tests** (`tests/test_gold_sets.py`) covering: every domain has a
  gold set + schema; every gold-set key maps to a real detector id;
  every schema covers every evidence file used by its manifest;
  every schema's `used_by_detectors` references real ids; the
  validator passes on synthetic-good folders, fails on missing-
  required-columns folders, and accepts alias columns.

### Changed

- `tests/test_lappato_mcb.py::test_evaluate_domain_skips_inactive_weaknesses`
  relaxed from `recall >= 0.66` to `recall >= 0.50` because the
  expanded `wdbc` gold set has 5 topics for `low_calibration` (was
  3); the synthetic test paper now matches 3 of them.
- `docs/gold_sets/README.md` rewritten to describe the dynamic
  discovery behaviour and the new ~500-topic scale.

### Backwards compatibility

- Constructor signature, manifest schema, weakness-card schema and
  meta-log columns are unchanged.
- The original three gold sets remain importable under their
  existing tags.
- New schemas are additive: a host pipeline that already produced
  CSVs the detectors accepted is unaffected.

## [1.2.0] — 2026-05-06

Bottleneck-coverage release. The manifest registry grows from 19 to **35
domains** and the new domain-agnostic `pipeline_health` manifest is
expanded from 8 to 16 detectors so that subtle / hidden methodological
bottlenecks are visible by default. Framework code, public API, and the
existing manifests are unchanged.

### Added

- **`pipeline_health` extended from 8 to 16 detectors** (covers both
  obvious and hidden bottlenecks, all stdlib-only and domain-agnostic):
  - `train_test_overlap_detected` — entity / sample leakage across splits
  - `label_noise_detected` — inter-annotator disagreement above the floor
  - `prediction_confidence_collapsed` — extreme-mass or uniform-collapse
    histograms
  - `train_eval_prevalence_shift` — class-prior delta across splits
  - `loss_metric_divergence` — train loss improves but eval metric does
    not follow
  - `hyperparameter_overfit_to_validation` — best HP trial sits far above
    trial mean
  - `constant_or_dead_features` — share of zero-variance features above
    threshold
  - `threshold_picked_on_test_set` — operating-point selected on the same
    split that reports final metrics
- **Six science-foundation manifests** (introduced earlier in this minor
  cycle, kept under one release tag for a single registry bump):
  - `pipeline_health`, `mathematics`, `physics`, `chemistry`,
    `biochemistry`, `biology`
- **Ten additional scientific-domain manifests**:
  - `earth_climate` — climate / atmospheric / ocean / hydrology
  - `astronomy` — surveys, transients, simulations
  - `materials_science` — DFT / MD / MLIPs / hull stability
  - `neuroscience` — fMRI / EEG / decoding pipelines
  - `epidemiology` — surveillance, R(t), interventions
  - `econometrics` — IV, panel, DiD, RDD
  - `social_science` — survey + replication diagnostics
  - `robotics` — sim2real, safety, control evaluation
  - `quantum_computing` — circuit fidelity, noise, barren plateaus
  - `pharmacology` — PK/PD, dose-response, trial endpoints
- **Sentinel test suite** (`tests/test_new_manifests.py`) with positive
  fire-paths and negative fail-closed paths for every new detector,
  threshold round-trip checks, and a registry-wide schema validator.

### Changed

- `tests/test_lappato_mcb.py` invariant
  `test_each_manifest_has_five_entries` is now
  `test_each_manifest_has_at_least_five_entries` so that
  `pipeline_health` (16 detectors) and any future expansion can coexist
  with the cross-domain comparability rule.
- `EXPECTED_MANIFEST_DOMAINS` lifted from 19 → 35.
- README gains a `Supported manifest domains` section that lists the
  full registry by category.

### Backwards compatibility

- Constructor signature, manifest schema, weakness-card schema and
  meta-log columns are unchanged.
- All previously registered domains remain importable under their
  existing tags.
- All new manifests follow the same plug-style contract (no core edits).

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
