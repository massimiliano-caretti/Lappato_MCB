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

Abstracts are kept only for arXiv records, whose metadata (abstracts
included) arXiv releases under CC0 1.0. Abstracts obtained through
OpenAlex or Crossref were removed on 2026-10-22: they may be subject to
the copyright of their publishers or authors. Titles, identifiers and
URLs are bibliographic facts and are kept.
