"""Validate LAPPATO_MCB's local relevance score against a small curated set.

LAPPATO_MCB ranks harvested papers per-query with the deterministic
``_score_hit`` function (title overlap × 4 + abstract overlap × 1.5 +
recency bonus + log-citations × 0.15 by default — see
:class:`lappato_mcb.core.ScoreWeights`). The weights were chosen by
inspection; this harness checks that the ranking they produce
correlates with human judgement on a small bundled relevance set.

What it does:
  1. Loads ``docs/score_validation/relevance_set.json`` (≈15 entries
     spanning calibration, class imbalance, long-tail text, GARCH,
     conformal forecasting and decision-curve analysis).
  2. Computes the deterministic LAPPATO score for each (query, paper)
     pair under the default weights and any user-supplied override.
  3. Reports the Spearman rank correlation between the human relevance
     ratings and the LAPPATO scores, plus the per-query ordering.

Honest scope: this is a *sanity-check*, not a benchmark. The relevance
set is small and curated by the maintainers, not externally validated;
a healthy correlation does not establish absolute recall or precision.
The script's value is regression-detection: if a future weight tweak
sends the correlation below the documented baseline, that is a signal
to re-justify the change.

Run:
    python examples/validate_score.py
    python examples/validate_score.py --weights title_overlap=3.0,abstract_overlap=2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the lappato_mcb package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb.core import (  # noqa: E402
    DEFAULT_SCORE_WEIGHTS,
    ScoreWeights,
    _score_hit,
)

ROOT = Path(__file__).resolve().parents[1]
RELEVANCE_PATH = ROOT / "docs" / "score_validation" / "relevance_set.json"


def _ranks(values: list[float]) -> list[float]:
    """Average-rank assignment that handles ties (Spearman convention)."""
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation of two equal-length sequences."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need ≥2 paired observations")
    rx = _ranks(xs)
    ry = _ranks(ys)
    n = len(xs)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = sum((a - mean_x) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_y) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def parse_weight_overrides(raw: str | None) -> ScoreWeights:
    """``"title_overlap=3.0,recency_per_year=0.5"`` -> ScoreWeights.

    Unknown keys raise ``ValueError`` so typos do not silently degrade
    to default behaviour. Empty / None input returns the defaults.
    """
    if not raw:
        return DEFAULT_SCORE_WEIGHTS
    overrides: dict[str, float] = {}
    valid = set(DEFAULT_SCORE_WEIGHTS.__dataclass_fields__)
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k = k.strip()
        if k not in valid:
            raise ValueError(f"unknown weight {k!r} (valid: {sorted(valid)})")
        overrides[k] = float(v.strip())
    if not overrides:
        return DEFAULT_SCORE_WEIGHTS
    base = {k: getattr(DEFAULT_SCORE_WEIGHTS, k) for k in valid}
    base.update(overrides)
    # recency_max_years is an int — coerce to keep the dataclass typed.
    if "recency_max_years" in overrides:
        base["recency_max_years"] = int(base["recency_max_years"])
    return ScoreWeights(**base)


def load_relevance_set(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    if not entries:
        raise SystemExit(f"empty relevance set at {path}")
    return entries


def evaluate(entries: list[dict], weights: ScoreWeights) -> dict:
    """Score every (query, paper) and report Spearman + per-query overlap."""
    scores = [
        _score_hit(e["query"], e["paper"], weights) for e in entries
    ]
    ratings = [float(e["human_relevance"]) for e in entries]
    rho = spearman(ratings, scores)

    by_query: dict[str, list[tuple[float, float, str]]] = {}
    for e, s in zip(entries, scores):
        by_query.setdefault(e["query"], []).append(
            (float(e["human_relevance"]), s, e["paper"]["title"])
        )

    per_query_rho = []
    for q, items in by_query.items():
        if len(items) < 2:
            continue
        rs = [it[0] for it in items]
        ss = [it[1] for it in items]
        try:
            per_query_rho.append((q, spearman(rs, ss), len(items)))
        except ValueError:
            continue

    return {
        "n_entries": len(entries),
        "spearman_overall": rho,
        "per_query": per_query_rho,
        "by_query": by_query,
        "weights": weights,
    }


def render_report(result: dict) -> str:
    lines: list[str] = []
    w = result["weights"]
    lines.append("LAPPATO_MCB score validation")
    lines.append("=" * 60)
    lines.append(
        f"weights: title={w.title_overlap}, abstract={w.abstract_overlap}, "
        f"recency_step={w.recency_per_year} (max {w.recency_max_years} yrs "
        f"from {w.recency_base_year}), citation_log={w.citation_log_weight}"
    )
    lines.append(f"n_entries        : {result['n_entries']}")
    lines.append(f"Spearman overall : {result['spearman_overall']:+.3f}")
    lines.append("")
    lines.append("Per-query Spearman (min 2 entries):")
    for q, rho, n in result["per_query"]:
        lines.append(f"  ρ={rho:+.3f}  n={n}  query={q!r}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights",
        help=("Comma-separated key=value overrides, e.g. "
              "'title_overlap=3.0,citation_log_weight=0.05'"),
    )
    p.add_argument(
        "--relevance-set", type=Path, default=RELEVANCE_PATH,
        help="Path to the curated relevance set JSON (default bundled).",
    )
    p.add_argument(
        "--min-spearman", type=float, default=0.50,
        help=("Exit non-zero when the overall Spearman drops below this "
              "value. CI uses the default to catch regressions."),
    )
    args = p.parse_args()

    weights = parse_weight_overrides(args.weights)
    entries = load_relevance_set(args.relevance_set)
    result = evaluate(entries, weights)
    print(render_report(result))

    if result["spearman_overall"] < args.min_spearman:
        print(
            f"\n[fail] Spearman {result['spearman_overall']:+.3f} below "
            f"min-spearman={args.min_spearman:+.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
