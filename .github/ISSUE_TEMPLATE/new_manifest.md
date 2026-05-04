---
name: New manifest / domain
about: Propose a new domain manifest (e.g. computer vision, RL, generative)
title: "[manifest] "
labels: enhancement, manifest
assignees: ''
---

## Domain

<!-- One-line domain id (e.g. "computer-vision-classification",
"rl-policy-eval", "generative-text-eval") and what kind of pipeline it
targets. -->

## Diagnostic CSVs the host pipeline writes

<!-- List each `checkpoints/<name>.csv` the manifest will gate on,
with the columns it expects and how the host pipeline produces it. -->

| File | Columns | Producer |
| --- | --- | --- |
| `cv_per_class.csv` | class, precision, recall, f1 | `examples/demo_cv.py` |

## Weakness entries

For each entry, provide:

- `id`: short, snake_case identifier.
- `evidence` + `evidence_check`: the file and the boolean function
  that activates the entry (or `None` for a structural entry).
- `severity`: static string or callable returning `info|low|medium|high|critical`.
- `queries`: 1–3 templated query strings.
- `transplant`: actionable, evidence-grounded recommendation.
- `next_checks` and `success_criteria`: each 2–4 bullets.
- `references`: short list of canonical works.

## Threshold rationale

For each numeric threshold proposed in `DEFAULT_THRESHOLDS`, cite the
source (paper, reporting checklist, empirical study). Soft heuristics
without a citation should be flagged in the PR description.

## Validation

- [ ] Bundled `examples/demo_<domain>.py` produces the expected CSVs.
- [ ] `examples/measure_recall.py --domain <domain>` passes (gold set
  added under `docs/gold_sets/<domain>_gold.json`).
- [ ] Manifest registry updated (`lappato_mcb/manifests/__init__.py`).
- [ ] Report pack registered (`lappato_mcb/reports/__init__.py`).
- [ ] Unit tests added for evidence detectors.
