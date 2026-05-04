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

One file per registered manifest domain:

| File | Domain | Weaknesses covered |
| --- | --- | --- |
| `wdbc_gold.json` | `wdbc` | calibration, class imbalance, marker redundancy, tabular alternatives, decision-curve analysis |
| `nlp_gold.json` | `nlp` | per-class F1 spread, confusable classes, stopword dominance, embedding alternatives, calibration (text) |
| `timeseries_gold.json` | `timeseries` | residual ACF, heteroscedasticity, horizon degradation, model alternatives, probabilistic forecasting |

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
