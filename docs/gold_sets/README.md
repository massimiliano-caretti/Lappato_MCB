# LAPPATO_MCB gold sets

Hand-curated **topic** lists used by [`examples/measure_recall.py`](../../examples/measure_recall.py)
to estimate how often LAPPATO_MCB surfaces well-known relevant work for
each weakness in the bundled manifests.

## Honest scope

These are **not** authoritative gold standards. They are small,
maintainer-curated sanity checks: a topic is considered "found" when at
least one harvested paper title or abstract contains *every* keyword in
the topic's `must_match_keywords` list (case-insensitive). The harness
reports per-weakness, per-domain and macro `Recall@K`.

A healthy recall (≥ 70% on the bundled corpus) means LAPPATO_MCB is
reaching the topics a domain expert would expect to see surface for
that weakness. A drop is a regression signal — not proof of universal
coverage.

## File layout

One file per registered manifest domain — the v1.3 release expanded
the gold-set catalogue from 3 to **35 files**, one for every manifest
in `lappato_mcb.manifests.REGISTRY`. Across those files there are
**~500 hand-curated topics** (10–15 per 5-detector manifest, 32 for
the 16-detector `pipeline_health`).

The harness in [`examples/measure_recall.py`](../../examples/measure_recall.py)
discovers gold sets dynamically by globbing
`docs/gold_sets/*_gold.json`. Adding a new gold set is a one-file
change — no harness edit needed.

Coverage at a glance (auto-derived):

```bash
python -c "
import json, glob, os
for p in sorted(glob.glob('docs/gold_sets/*_gold.json')):
    g = json.load(open(p))
    n = sum(len(v) for v in g['topics_by_weakness'].values())
    print(f'{os.path.basename(p):36s} topics={n}')
"
```

For a sanity check that every gold-set key still maps to a real
detector ID after a manifest edit, run
`python -m unittest tests.test_gold_sets`.

## Schema

```json
{
  "schema_version": 1,
  "domain": "wdbc",
  "topics_by_weakness": {
    "low_calibration": [
      {
        "name": "isotonic regression",
        "must_match_keywords": ["isotonic", "calibration"],
        "rationale": "Standard nonparametric calibration (Zadrozny & Elkan 2001)."
      }
    ]
  }
}
```

`must_match_keywords` are matched case-insensitively against the
concatenation of paper title and abstract. Adding a new topic is a
one-line change; please include a short `rationale` so future readers
understand why the topic was chosen.
