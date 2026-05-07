# Evidence-CSV JSON Schemas

One file per manifest in `lappato_mcb.manifests.REGISTRY`. Each
schema lists the **evidence CSV files** the manifest expects under
`<project_root>/checkpoints/`, the **canonical column names** for
each file, the **aliases** the detectors will accept, the column
**type**, and which **detectors** read the file.

These schemas are the source of truth for the host-pipeline contract
that activates a manifest's detectors. They are intentionally
versioned (`schema_version`) so that future column changes can ship
a deprecation notice rather than silently break consumers.

## Validation

Validate any candidate CSV (or the contents of a `checkpoints/`
folder) against a manifest's schema with:

```bash
python examples/validate_evidence_csv.py --domain pipeline_health \
    --checkpoints checkpoints/
```

The validator reports:
- which evidence files are present;
- which expected files are missing (and which detectors that disables);
- which required columns are missing on each present file;
- which alias / non-canonical column names were accepted.

It exits non-zero when a required column is missing on a file that is
present (i.e., the file is there but unusable). Missing files are
reported but do not fail the run — the corresponding detector simply
stays inactive (fail-closed by design).

## Schema format

```json
{
  "schema_version": 1,
  "domain": "pipeline_health",
  "description": "Domain-agnostic methodological diagnostics.",
  "evidence_files": {
    "pipeline_class_counts.csv": {
      "description": "Per-class sample counts.",
      "columns": [
        {"name": "class", "aliases": ["label", "name"],
         "type": "string", "required": true},
        {"name": "count", "aliases": ["support", "n", "size"],
         "type": "number", "required": true}
      ],
      "used_by_detectors": [
        "small_minority_class",
        "severe_class_imbalance"
      ]
    }
  }
}
```

The `aliases` list documents the alternative column names the
detector helpers (`f`, `values`) recognise — the host pipeline can
emit any of them. The `type` is one of `"string"` or `"number"` and
is checked best-effort by the validator.
