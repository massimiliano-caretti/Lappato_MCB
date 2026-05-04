# Sample Outputs

This directory contains compact, versionable summaries produced from real
demo runs. Full `checkpoints/` artefacts are intentionally ignored by Git
because they include generated JSONL corpora, Markdown logs and PDFs.

Files:

- `summary.json`: run-level counts for the bundled demos.
- `wdbc_sample.json`: clinical tabular WDBC sample cards and papers.
- `nlp_sample.json`: 20newsgroups NLP sample cards and papers.
- `timeseries_sample.json`: synthetic time-series sample cards and papers,
  including JOSS research-software hits.

The sample paper lists were deduplicated by canonical DOI/arXiv id and by
normalized title before these files were written.
