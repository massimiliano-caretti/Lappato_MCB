# Contributing to LAPPATO_MCB

LAPPATO_MCB is a small, opinionated framework. Contributions are
welcome but please read this short guide first — it captures the
constraints that make the project useful.

## Design constraints (non-negotiable)

These are the properties that distinguish LAPPATO_MCB from generic
MLOps trackers. PRs that break any of them will be asked to rework.

1. **Stdlib-only daemon core.** The activation/retrieval path
   (`lappato_mcb/core.py`, `cache.py`, `fingerprint.py`,
   `meta_log.py`) must not import third-party packages. Optional
   reporting and example modules may use NumPy / matplotlib /
   scikit-learn behind the `[reports]` and `[examples]` extras.
2. **Determinism.** Same inputs + same external-source state ⇒ same
   outputs. Avoid randomised tie-breaks, hashed-iteration ordering or
   timestamps in dedup keys. Reproducibility is a feature.
3. **No paid services.** No API key, no LLM, no embedding service in
   the core path.
4. **Two-line integration.** `start()` before training, `stop()` after.
   Do not add wrappers, decorators or callbacks.
5. **Auditable activation.** Every weakness must state in code (not in
   prose) when it fires (`evidence_check`) and what it suggests
   (`transplant`). Thresholds belong in a manifest-level
   `THRESHOLDS` dict with a literature/empirical citation, not buried
   in detector functions.

## What we accept

- **Bug fixes** with a regression test in `tests/test_lappato_mcb.py`.
- **New manifests** for new domains. A complete contribution is
  three files: `lappato_mcb/manifests/<domain>.py` (the manifest),
  `lappato_mcb/manifests/schemas/<domain>.schema.json` (the
  evidence-CSV contract), and `docs/gold_sets/<domain>_gold.json`
  (the recall sanity check), plus a registry update in
  `lappato_mcb/manifests/__init__.py`. The 35 bundled manifests are
  all reference templates — `wdbc`, `nlp`, `timeseries` are the
  smallest; `pipeline_health` is the largest (16 detectors). A
  matching `lappato_mcb/reports/<domain>.py` report pack is optional;
  the framework falls back to a generic pack when absent.
- **Threshold reviews.** If you have empirical evidence that a default
  cut-off is wrong for a target cohort, open a PR that updates
  `DEFAULT_THRESHOLDS` *and* the citation in the comment above it.
- **Score-weight retunes.** Update `DEFAULT_SCORE_WEIGHTS` *and*
  re-run `examples/validate_score.py` and document the new Spearman
  baseline in the PR description.
- **Gold-set extensions.** Adding a topic to
  `docs/gold_sets/<domain>_gold.json` is welcome. Each entry needs a
  short `rationale`.
- **Documentation, packaging, CI** improvements.

## What we don't accept (out of scope)

- LLM summarisers, autonomous research agents, embedding-based
  rerankers — these break the stdlib-only contract.
- Cloud or hosted backends.
- Wrappers around `start()`/`stop()` that turn LAPPATO_MCB into a
  decorator/callback API.
- Removing the offline replay path.

## How to develop

```bash
git clone <your fork>
cd lappato_mcb
python -m pip install -e ".[reports,dev]"

# unit tests (no network access required)
python -m unittest discover -s tests -v

# score-weight regression check
python examples/validate_score.py --min-spearman 0.70

# topic-recall against the bundled gold sets (needs a previous demo run)
python examples/demo_wdbc.py
python examples/measure_recall.py --domain wdbc
```

CI runs the same three steps on Python 3.9 → 3.13 plus `ruff` lint.
A PR turns green when all jobs pass.

### Optional: pre-commit hooks

A `.pre-commit-config.yaml` ships with the repo. It runs the same
`ruff` lint pinned in CI plus a handful of file-hygiene hooks
(trailing whitespace, end-of-file newline, YAML/TOML syntax,
large-file guard, merge-conflict marker check, line-ending
normalisation). The hooks never run the full test suite — that
stays in CI — so the local commit feedback loop is sub-second.

```bash
python -m pip install pre-commit
pre-commit install                  # one-time, per clone
# pre-commit then runs automatically on every `git commit`.

pre-commit run --all-files          # to run all hooks on demand
```

Skip a hook for a single commit when needed:

```bash
SKIP=ruff git commit -m "..."
```

## Pull-request checklist

- [ ] New behaviour has a unit test in `tests/`.
- [ ] Public API changes are mirrored in the README and CHANGELOG.
- [ ] Threshold or score-weight changes include a citation/measurement.
- [ ] No new third-party imports in the daemon core path.
- [ ] `python -m unittest discover -s tests -v` passes locally.
- [ ] `ruff check lappato_mcb examples tests` passes locally.

## Reporting issues

Please use the GitHub issue templates (`Bug report`, `Feature
request`, `New manifest`). Include the LAPPATO_MCB version
(`python -c "import lappato_mcb; print(lappato_mcb.__version__)"`),
Python version, and minimal reproducer.
