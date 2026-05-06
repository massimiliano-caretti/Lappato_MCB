# LAPPATO_MCB

**LAPPATO_MCB** — *Literature-Aware Pipeline Partner for Adaptive Transplant
Optimization*.

LAPPATO_MCB is a lightweight Python library for literature-aware monitoring
of machine-learning pipelines. It runs alongside a host training process as
a daemon thread, watches diagnostic artefacts written to a `checkpoints/`
folder, activates manifest-defined weakness detectors when their evidence
thresholds are crossed, and retrieves targeted scientific literature from
three public scholarly indices: **arXiv**, **OpenAlex**, and **Crossref**.
It also includes a specialised **JOSS** channel, implemented via
Crossref ISSN filtering, to surface peer-reviewed research-software papers
that may make a weakness card immediately actionable.

The library core uses only the Python standard library
(`urllib`, `csv`, `json`, `threading`, `xml.etree`). Optional bundled
examples and the PDF reporting utilities require scientific-Python packages
(NumPy, scikit-learn, LightGBM, matplotlib).

For polite API use, set a contact email before online runs:

```bash
export LAPPATO_MCB_MAILTO="you@example.org"
```

When set, this value is sent to OpenAlex/Crossref/JOSS through their
documented `mailto` mechanisms and included in the User-Agent.

---

## Install

```bash
python -m pip install -e .
```

For the bundled examples and the PDF report builder:

```bash
python -m pip install -e ".[examples,reports]"
```

## Minimal Usage

```python
from pathlib import Path

from lappato_mcb import LAPPATO_MCB
from lappato_mcb.manifests import get as get_manifest

manifest, run_tag = get_manifest("timeseries")

lappato_mcb = LAPPATO_MCB(
    project_root=Path("."),
    manifest=manifest,
    run_tag=run_tag,
    poll_interval=90.0,
)

lappato_mcb.start()
# Run your ML pipeline here. It should write diagnostic CSVs into
# project_root / "checkpoints".
lappato_mcb.stop()
```

## Repository Layout

- `lappato_mcb/`: installable library package.
- `lappato_mcb/core.py`: daemon-thread harvesting core.
- `lappato_mcb/manifests/`: weakness manifests for 35 supported domains —
  see [`Supported manifest domains`](#supported-manifest-domains) below.
- `lappato_mcb/cache.py`: local corpus cache and offline replay support.
- `lappato_mcb/fingerprint.py`: cross-source title deduplication
  (character-trigram + Jaccard).
- `lappato_mcb/meta_log.py`: per-cycle self-instrumentation.
- `lappato_mcb/analysis.py`, `plots.py`, `pdf_builder.py`: optional
  analysis and reporting utilities.
- `examples/`: runnable experimental pipelines and baseline evaluation.
- `tests/`: stdlib unit tests for the library core.

## Supported manifest domains

LAPPATO_MCB ships **35 plug-style manifests** out of the box. Each
manifest is a stdlib-only Python module under `lappato_mcb/manifests/`
that exposes evidence-gated weakness detectors with tunable thresholds
(`override_thresholds`). Pick one with `manifests.get("<tag>")`.

**Methodology / cross-cutting**

| Tag | Focus | Detectors |
| --- | --- | ---: |
| `pipeline_health` | Domain-agnostic statistical pipeline diagnostics — both classical bottlenecks (class imbalance, cross-seed variance) and *hidden* ones (train-test overlap, label noise, prediction collapse, prevalence shift, loss/metric divergence, HP overfitting, dead features, threshold-on-test). | 16 |
| `tabular_generic` | Generic tabular fallback (calibration, drift, leakage / missingness audits). | 5 |

**Mainstream supervised / unsupervised / evaluation**

| Tag | Focus |
| --- | --- |
| `nlp` | text classification + token attribution |
| `vision` | image-classification diagnostics |
| `timeseries` | forecasting residuals, ACF, horizon |
| `recommender` | top-K, coverage, popularity bias |
| `clustering` | unsupervised cluster-quality |
| `anomaly_detection` | outlier / novelty detection |
| `survival` | survival, C-index, calibration |
| `causal_ml` | causal-inference / treatment-effect |
| `fairness` | group fairness, demographic parity, EOpp |
| `graph_ml` | node / edge classification, link prediction |
| `rl_eval` | RL policy-evaluation diagnostics |
| `llm_eval` | LLM evaluation harness |
| `rag_eval` | RAG retrieval + answer faithfulness |
| `speech_audio` | ASR / audio classification |
| `cybersecurity` | intrusion / malware detection |
| `geospatial` | spatial autocorrelation, CRS, leakage |
| `medical_imaging` | radiology classification + segmentation |
| `wdbc` | reference manifest on the Wisconsin Breast-Cancer cohort |

**Scientific domains**

| Tag | Focus |
| --- | --- |
| `mathematics` | conditioning, convergence, discretization |
| `physics` | conservation laws, CFL, equilibration, dimensions |
| `chemistry` | structure validity, applicability domain, reaction balance |
| `biochemistry` | enzyme kinetics, pathways, RMSD, assay QC |
| `biology` | batch effects, replicates, phylogenetic correction |
| `earth_climate` | model bias, water balance, extremes, ensembles |
| `astronomy` | PSF residuals, completeness, selection function |
| `materials_science` | k-point / basis convergence, MLIPs, hull stability |
| `neuroscience` | motion, subject splits, smoothing, MC correction |
| `epidemiology` | case-definition shifts, reporting delay, underreporting |
| `econometrics` | weak IV, parallel trends, clustered SEs, RDD |
| `social_science` | effect-size inflation, response rate, p-curve |
| `robotics` | sim2real gap, safety violations, action smoothness |
| `quantum_computing` | fidelity, gate error, barren plateaus |
| `pharmacology` | dose-response, PK concordance, DDI, endpoints |

The full registry is exposed via `lappato_mcb.manifests.REGISTRY`. Adding
a new domain is a two-step change (drop a `<tag>.py` file, register it).
See `.github/ISSUE_TEMPLATE/new_manifest.md` for the contract.

## Examples

```bash
python examples/demo_wdbc.py
python examples/demo_timeseries.py
python examples/compare_targeted_vs_naive.py --domain wdbc --use-corpus
```

The examples write generated artefacts under `checkpoints/`. These outputs
are excluded from version control by `.gitignore`.

### Sanity checks (no network required)

Two harnesses ship in `examples/` and are wired into CI:

```bash
# 1. Spearman correlation between LAPPATO_MCB's local relevance score
#    and a small hand-curated relevance set
#    (docs/score_validation/relevance_set.json). Shipped baseline ~0.89.
python examples/validate_score.py --min-spearman 0.70

# 2. Topic Recall@K against per-domain gold sets
#    (docs/gold_sets/{wdbc,nlp,timeseries}_gold.json). Macro recall is
#    computed only over weaknesses that fired in the source run.
python examples/measure_recall.py --domain all --min-macro-recall 0.70
```

Both are sanity checks, not absolute benchmarks — see
[`docs/gold_sets/README.md`](docs/gold_sets/README.md) for the contract.

---

## Definition

LAPPATO_MCB is best described as a **literature-aware framework for
machine-learning pipelines**, not as a single algorithm. Its behaviour
combines pipeline monitoring, evidence-gated weakness activation, targeted
literature retrieval, cross-source deduplication, structured logging, and
offline corpus replay.

## Working With AI Coding Assistants

LAPPATO_MCB can be used on its own: it writes Markdown logs, JSONL paper
sidecars, weakness cards, meta-logs and PDF reports that a researcher can
read directly.

It becomes especially useful inside editors such as Cursor, Visual Studio
Code, JetBrains IDEs or any environment with an AI coding assistant
(GitHub Copilot, Claude, Gemini, ChatGPT or similar). The assistant is good
at writing and changing code, but it usually does not know which
scientific weakness your latest ML run actually exposed. LAPPATO_MCB fills
that gap.

The workflow is:

```text
run the ML pipeline
-> LAPPATO_MCB detects measured weaknesses
-> LAPPATO_MCB writes weakness cards
-> paste or attach a card to the editor chat
-> ask the AI coding assistant to implement the next check or intervention
```

In practice, LAPPATO_MCB gives the assistant a compact, evidence-grounded
brief:

- what failed or is missing in the pipeline;
- which metric or diagnostic file proves it;
- how severe the weakness is;
- which papers and software are relevant;
- which next checks should be implemented;
- what success would look like after the code change.

Example prompt for an editor chat:

```text
Use this LAPPATO_MCB weakness card as context.
Implement the next_check in my current pipeline, add the required CSV output
under checkpoints/, and keep the change consistent with the existing code.
Also add a small test or sanity check if practical.

[paste weakness card JSON here]
```

This makes the AI assistant more targeted. It is no longer guessing from a
general request such as "improve my model"; it is acting on a measured
pipeline weakness and on literature-specific guidance.

The distinction is simple:

```text
Copilot / Claude / ChatGPT help write code.
LAPPATO_MCB tells them which real pipeline problem to solve,
why it matters, and which literature/software should guide the fix.
```

## Evidence-Gated Literature Retrieval

The central design feature is **Evidence-Gated Literature Retrieval**:
LAPPATO_MCB maps a measured pipeline weakness to targeted scientific
queries while the training run is still in progress.

Each active manifest entry now produces a structured **weakness card**:

- the diagnostic evidence file that activated the weakness;
- a compact evidence summary, such as mean Brier score, F1 spread,
  residual ACF, or top confused class pair;
- severity (`info`, `low`, `medium`, `high`, `critical`);
- the rendered queries actually sent to arXiv, OpenAlex, Crossref and
  the specialised JOSS software channel;
- per-source raw hit counts;
- top harvested papers ranked by a deterministic local score;
- the number of papers new relative to previous runs;
- the associated transplant sketch.

Cards are written next to the existing log artefacts:

```text
checkpoints/<run_tag>_lappato_mcb_weakness_cards.jsonl
checkpoints/<run_tag>_lappato_mcb_weakness_cards.latest.json
```

The JSONL file is append-only and preserves the per-cycle history. The
`latest.json` file is optimized for dashboards, CI summaries and future
MLOps integrations.

### Concrete Run Evidence

The bundled demos have been run end-to-end and generated concrete
weakness cards plus PDF reports under `checkpoints/`. Since `checkpoints/`
is intentionally git-ignored, compact sample outputs are versioned under
`docs/sample_outputs/`:

| Run | Papers harvested | Latest cards | Sample file |
| --- | ---: | ---: | --- |
| WDBC clinical tabular | 36 | 2 | `docs/sample_outputs/wdbc_sample.json` |
| NLP 20newsgroups | 53 | 2 | `docs/sample_outputs/nlp_sample.json` |
| Synthetic time series | 130 | 4 | `docs/sample_outputs/timeseries_sample.json` |

The generated paper sidecars were deduplicated by canonical DOI/arXiv id and
normalized title. The sample outputs were checked to contain no duplicate ids
or duplicate titles.

Example card excerpts from the generated artefacts:

```json
{
  "run_tag": "wdbc",
  "weakness": "tabular_baseline_alternatives",
  "severity": "info",
  "evidence_summary": {"type": "structural", "active": true},
  "query": "TabPFN small dataset clinical tabular benchmark medical 2025",
  "papers_kept": 20,
  "top_paper": "Predicting dementia in Parkinson's disease on a small tabular dataset using hybrid LightGBM-TabPFN and SHAP",
  "next_check": "Benchmark CatBoost, logistic elastic-net, TabPFN and AutoGluon."
}
```

```json
{
  "run_tag": "nlp",
  "weakness": "calibration_missing_text",
  "severity": "info",
  "query": "softmax calibration text classifier temperature scaling 2025",
  "papers_kept": 28,
  "top_paper": "Reliable Cybersecurity Threat Detection through Probability Calibration in Multiclass Classification",
  "next_check": "Report ECE, Brier score and reliability diagram."
}
```

```json
{
  "run_tag": "timeseries",
  "weakness": "heteroscedastic_residuals",
  "severity": "medium",
  "evidence_summary": {"rows": 6, "last_first_variance_ratio": 1.8484},
  "query": "GARCH heteroscedasticity time series volatility clustering",
  "papers_kept": 31,
  "top_paper": "FTLinear: MLP based on Fourier Transform for Multivariate Time-series Forecasting",
  "next_check": "Plot residual variance by rolling-origin fold."
}
```

JOSS can make the card more actionable by surfacing citable software:

```json
{
  "run_tag": "timeseries",
  "weakness": "heteroscedastic_residuals",
  "source": "JOSS",
  "title": "bmgarch: An R-Package for Bayesian Multivariate GARCH models",
  "url": "https://doi.org/10.21105/joss.03452",
  "query": "GARCH heteroscedasticity time series volatility clustering"
}
```

```json
{
  "run_tag": "timeseries",
  "weakness": "probabilistic_forecast_missing",
  "source": "JOSS",
  "title": "quantile-forest: A Python Package for Quantile Regression Forests",
  "url": "https://doi.org/10.21105/joss.05976",
  "query": "quantile loss probabilistic forecasting calibration"
}
```

### Optional Run Context

Pipelines can make queries more precise by writing a tiny JSON context
before starting LAPPATO_MCB:

```json
{
  "domain": "clinical tabular",
  "task": "binary classification",
  "model_family": "LightGBM",
  "dataset": "Wisconsin Breast Cancer Diagnostic"
}
```

Accepted paths:

```text
checkpoints/lappato_context.json
checkpoints/<run_tag>_lappato_context.json
```

Manifest queries may then use placeholders:

```python
"Brier calibration {model_family} probability {domain}"
```

Missing context values are dropped, so this remains fully optional and
does not change the two-line integration pattern.

### Synchronous One-Cycle Audit

Fast pipelines can finish before a long online harvest cycle completes.
For checkpoint-first workflows, run a single synchronous cycle:

```python
from pathlib import Path
from lappato_mcb import LAPPATO_MCB
from lappato_mcb.manifests import get as get_manifest

manifest, run_tag = get_manifest("timeseries")
lappato_mcb = LAPPATO_MCB(Path("."), manifest=manifest, run_tag=run_tag)
lappato_mcb.run_once()
```

---

## Positioning relative to existing tools

LAPPATO_MCB sits at the intersection of three established categories of
tooling. None of its individual technical components is novel in isolation;
the contribution is the specific **combination of properties** integrated
into a single two-line drop-in for any host training pipeline. The
comparison below is a *design-level audit* — what each tool is built to do
and what it requires — not a performance benchmark.

### 1. Tool families surveyed

| Family | Representative implementations |
| --- | --- |
| Academic literature search and discovery | Elicit, Connected Papers, Semantic Scholar, Research Rabbit, LiteRev, LitLLM |
| ML experiment tracking | MLflow, Weights & Biases, TensorBoard, Comet, Neptune, ZenML |
| Autonomous LLM research agents | Agent Laboratory (Schmidgall et al., 2025, arXiv:2501.04227), AI-Researcher (HKUDS, NeurIPS 2025), AgentRxiv, AutoResearchClaw, Multi-Agent AutoResearch |
| Feedback-driven training optimisation | Interactive Training (2025) |

### 2. Comparison axes

The axes are intentionally orthogonal: each captures a single design
property that can be inspected from a tool's source code or documentation.

| Axis | Operational definition |
| --- | --- |
| `co-resident` | Runs concurrently with the host training process (not in batch, not as a separate review session). |
| `evidence-gated` | Activation is conditional on a measurable diagnostic threshold computed from training artefacts. |
| `deterministic` | Output is reproducible across runs given identical inputs and identical external-source state. |
| `stdlib-core` | The activation/retrieval core depends on no third-party Python packages. |
| `multi-source` | Queries ≥ 3 independent scholarly indices in a single run. |
| `cross-source-dedup` | Detects duplicates across sources, not only within a single source's native identifier space. |
| `offline-mode` | Can run without network access against a previously-collected local corpus. |
| `no-paid-services` | Operates without any paid API key (LLM, embeddings, hosted database). |
| `auditable-manifest` | Activation logic is declared in version-controlled code, not in learned weights or implicit LLM prompts. |

### 3. Comparative audit

A `✓` indicates the property is present by design; `–` absent or not
applicable; `~` partial or implementation-dependent.

| Tool / family | co-resident | evidence-gated | deterministic | stdlib-core | multi-source | cross-source-dedup | offline-mode | no-paid-services | auditable-manifest |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Literature search (Elicit, CP, SS, RR) | – | – | ~ | – | ~ | – | – | ~ | – |
| Experiment trackers (MLflow, W&B, TB) | ✓ | – | ✓ | – | – | – | ~ | ~ | – |
| LLM research agents (Agent Lab, AI-Researcher) | – | – | – | – | ~ | – | – | – | – |
| Feedback-driven training (Interactive Training) | ✓ | ~ | – | – | – | – | – | – | ~ |
| **LAPPATO_MCB** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Notes on the `~` cells. Search-tool determinism depends on whether the
backend is a stable index (Connected Papers, Semantic Scholar) or an LLM
(Elicit, LitLLM). Tracker offline-mode is supported by self-hosting (e.g.
local MLflow) but the default deployment target is a hosted service.
Some research agents query multiple sources via tool-use, but the source
list is decided at run time by the LLM and is not contractually fixed.

### 4. Determinism — what it does and does not mean

LAPPATO_MCB's algorithmic surface — manifest evaluation, query construction,
trigram dedup, corpus search — is fully deterministic given identical
inputs. The *retrieved set* across two runs at different points in time is
not guaranteed to be identical, because the upstream indices (arXiv,
OpenAlex, Crossref) are mutable: papers are added daily and re-ranked.
This is a property of the data sources, not of LAPPATO_MCB. The offline
replay-from-corpus mode (see `lappato_mcb.cache.CorpusCache`) freezes a
snapshot of the corpus and re-runs against that snapshot are bit-for-bit
deterministic.

### 5. What LAPPATO_MCB does *not* attempt

- It is not a paper-summariser, paper-writer, or autonomous research agent.
  It returns paper records (id, title, abstract, year, source, weakness
  matched), not generated prose.
- It does not interpret the relevance of retrieved papers. Each entry
  carries the originating query and the matched weakness id; the human
  researcher decides whether to act on it.
- It does not modify the host model. Each manifest entry exposes a
  *transplant sketch* — a short textual recommendation — but applying the
  recommendation is the user's responsibility.
- It does not perform semantic ranking. Online retrieval relies on each
  source's relevance scoring; the offline corpus uses token-overlap. There
  is no embedding model and no learned ranker.
- It is not a benchmark of "literature recall". Establishing recall would
  require a gold-standard relevance set per weakness; that is out of scope
  for the current release.

### 6. Distinctive technical properties (cross-checked against source)

The following properties are verifiable in the codebase and are the basis
of the comparative claims above.

1. **Two-line integration.** `start()` before training, `stop()` after — no
   wrapping of the model, no callback API, no decorator. See
   [`lappato_mcb/core.py`](lappato_mcb/core.py).
2. **Stdlib-only daemon core.** No third-party imports in the activation
   and retrieval path. Optional reporting modules are isolated behind an
   extras flag (`pip install -e ".[reports]"`).
3. **Three-source polite-pool retrieval + JOSS software channel.** arXiv
   (≥3 s spacing per their guidelines), OpenAlex (`mailto=` polite-pool
   parameter), Crossref (`mailto=` polite-pool parameter), plus JOSS as a
   Crossref-filtered research-software channel (`ISSN 2475-9066`). No paid
   API keys. See
   [`lappato_mcb/core.py`](lappato_mcb/core.py).
4. **Cross-source title deduplication.** Character-trigram fingerprints
   with a Jaccard threshold (default τ=0.80, tuned on a small set of
   arXiv/OpenAlex cross-listings). Algorithm O(N) per query at LAPPATO_MCB
   scale (10¹–10² papers per run). See
   [`lappato_mcb/fingerprint.py`](lappato_mcb/fingerprint.py).
5. **Append-only corpus + offline replay.** Each online run records the
   full harvest into a JSONL corpus; subsequent runs in `offline=True`
   mode search the corpus by token overlap with no network access. See
   [`lappato_mcb/cache.py`](lappato_mcb/cache.py).
6. **Per-cycle self-instrumentation.** Each cycle emits one CSV row with
   `n_queries_executed`, raw hits per source, dedup blocks (id-path vs
   trigram-path), kept count, time-to-first-paper. The framework
   measures itself. See [`lappato_mcb/meta_log.py`](lappato_mcb/meta_log.py).
7. **Weakness cards.** Each active weakness emits a machine-readable card
   linking evidence, severity, rendered queries, ranked papers, delta vs
   previous runs and transplant sketch. This is the library's core
   evidence-to-literature contract.
8. **Pluggable manifests and report packs.** Adding a new domain is a
   two-file change (one weakness manifest, one report pack); the
   framework code is domain-agnostic. See
   [`lappato_mcb/manifests/__init__.py`](lappato_mcb/manifests/__init__.py)
   and [`lappato_mcb/reports/__init__.py`](lappato_mcb/reports/__init__.py).
9. **Targeted-vs-naive evaluation harness.** A bundled experiment
   (`examples/compare_targeted_vs_naive.py`) runs the same diagnostic CSVs through
   the domain manifest and through a single-query baseline manifest, and
   reports the Jaccard overlap of the two harvest sets — converting a
   qualitative claim into a measurable artefact.

### 7. Limitations and threats to validity

- **No semantic ranking.** Token-overlap and source-side relevance scoring
  may miss papers whose titles use disjoint vocabulary from the manifest
  queries. Adding an embedding-based retriever is a documented future
  extension and would change the dependency profile.
- **Manifest authoring is a human task.** The quality of the harvest is a
  function of how well the manifest entries (`evidence_check`, `queries`,
  `transplant`) match the host pipeline's failure modes. The framework
  enforces structural consistency, not content correctness.
- **Recall is not measured.** The `targeted-vs-naive` evaluation reports
  set-overlap, not absolute recall against a curated relevance set. The
  latter requires per-domain ground truth and is left to downstream users.
- **Source-API stability.** The three general indices and the JOSS
  Crossref-filtered software channel can change endpoints, rate limits,
  and ranking from time to time. The HTTP layer in
  `lappato_mcb/core.py` swallows network errors per cycle and continues;
  it does not attempt to recover semantically.

### 8. References

- Schmidgall, S. et al. *Agent Laboratory: Using LLM Agents as Research Assistants*. arXiv:2501.04227 (2025). <https://arxiv.org/abs/2501.04227>
- HKUDS. *AI-Researcher: Autonomous Scientific Innovation*. NeurIPS 2025. <https://github.com/HKUDS/AI-Researcher>
- AgentRxiv project page. <https://agentrxiv.github.io/>
- Priem, J., Piwowar, H., Orr, R. *OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts.* arXiv:2205.01833 (2022). <https://arxiv.org/abs/2205.01833>
- *Monitoring Machine Learning Systems: A Multivocal Literature Review.* arXiv:2509.14294 (2025). <https://arxiv.org/abs/2509.14294>
- *Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents.* arXiv:2503.24047 (2025). <https://arxiv.org/abs/2503.24047>
- MLflow documentation. <https://mlflow.org/docs/>
- Connected Papers. <https://www.connectedpapers.com/>
- Semantic Scholar API. <https://www.semanticscholar.org/product/api>
- Elicit. <https://elicit.com/>
- arXiv API user guide. <https://info.arxiv.org/help/api/index.html>
- Crossref REST API. <https://api.crossref.org/swagger-ui/index.html>

---

## License

MIT — see [LICENSE](LICENSE).
