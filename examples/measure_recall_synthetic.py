"""Measure recall against a synthetic, gold-set-derived corpus.

What this is — and what it is NOT.

This harness builds a *deliberately easy* corpus: for every gold-set
topic across every domain, it synthesises a paper whose title +
abstract verbatim contain the topic's ``must_match_keywords``. It
then runs the same recall calculation that ``measure_recall.py``
uses against the bundled gold sets.

The point is not to claim "real" retrieval works — by construction
this corpus would also be found by a regex grep. The harness is
useful for two **regression** purposes:

  1. **End-to-end coverage check across all 35 domains.** Real online
     retrieval against arXiv / OpenAlex / Crossref takes minutes per
     domain and consumes API quota. A synthetic corpus runs offline,
     in milliseconds, and exercises every domain's gold set + recall
     plumbing. Adding a domain or renaming a detector immediately
     surfaces in the report.
  2. **Self-consistency floor for the gold set.** A ``must_match_keywords``
     list that is empty, contradictory, or refers to terminology that
     no realistic paper would use will produce <100% recall on this
     synthetic corpus. The gold-set regression test in
     ``tests/test_synthetic_recall_floor.py`` enforces a 100%
     synthetic floor on every domain — drop below and the offending
     topic is reported by name.

What this harness DOES NOT measure:
  - whether the manifest's ``queries`` would actually surface the
    relevant papers when issued against arXiv / OpenAlex / Crossref;
  - whether the gold-set topics are the *right* ones for each domain;
  - whether the local relevance score discriminates correctly when
    the corpus contains both relevant and noisy papers.

For those measurements, run ``examples/measure_recall.py`` against a
real online (or offline-cached) harvest. This file is the
infrastructure that makes future per-domain measurement studies
straightforward to wire in.

Usage::

    # synthetic recall on every domain
    python examples/measure_recall_synthetic.py

    # one domain, with details on missing topics
    python examples/measure_recall_synthetic.py --domain pipeline_health --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from measure_recall import GOLD_DIR, evaluate_domain  # noqa: E402


def _list_known_domains() -> list[str]:
    return sorted(p.stem.replace("_gold", "") for p in GOLD_DIR.glob("*_gold.json"))


def synthesize_corpus(domain: str) -> list[dict]:
    """Return a synthetic LAPPATO_MCB-style paper sidecar for ``domain``.

    For every (weakness, topic) in the domain's gold set, emit one
    'paper' whose ``weakness`` matches the gold-set key and whose
    title verbatim contains the topic's keywords. By construction
    every topic is reachable.
    """
    gold_path = GOLD_DIR / f"{domain}_gold.json"
    if not gold_path.exists():
        raise FileNotFoundError(f"no gold set for domain {domain!r}")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    papers: list[dict] = []
    counter = 0
    for weakness, topics in gold.get("topics_by_weakness", {}).items():
        for topic in topics:
            kws = [k for k in (topic.get("must_match_keywords") or []) if k]
            if not kws:
                continue
            counter += 1
            title = (
                f"Synthetic paper #{counter}: "
                + " ".join(kws)
                + " (gold-set topic: " + (topic.get("name") or "") + ")"
            )
            papers.append({
                "weakness": weakness,
                "title": title,
                "abstract": (
                    "This synthetic abstract mentions "
                    + ", ".join(kws)
                    + " for round-trip recall regression testing."
                ),
                "year": 2026,
                "source": "synthetic",
                "id": f"synthetic:{domain}:{counter}",
                "topic_name": topic.get("name"),
            })
    return papers


def evaluate_synthetic(domain: str) -> dict:
    """Build a synthetic corpus for ``domain`` and run ``evaluate_domain``."""
    papers = synthesize_corpus(domain)
    return evaluate_domain(domain, papers)


def render_report(results: list[dict], verbose: bool = False) -> str:
    lines = [
        "Synthetic recall (gold-set round-trip; expected = 1.000 per domain)",
        "=" * 72,
    ]
    grand: list[float] = []
    for r in results:
        macro = r["macro_recall"]
        grand.append(macro)
        flag = "OK" if macro >= 1.0 - 1e-9 else "  "
        lines.append(
            f" [{flag}] {r['domain']:>20s}  topics={r['n_topics_total']:3d}  "
            f"found={r['n_topics_found']:3d}  macro_recall={macro:+.3f}"
        )
        if verbose or macro < 1.0 - 1e-9:
            for w in r["per_weakness"]:
                if w["missing"]:
                    lines.append(
                        f"        miss in {w['weakness']:>40s}: "
                        + ", ".join(w["missing"])
                    )
    if grand:
        lines.append("")
        lines.append(
            f" GRAND macro recall (avg over {len(grand)} domains): "
            f"{sum(grand) / len(grand):+.4f}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--domain", default="all",
        help="Single domain to evaluate, or 'all' (default).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="List the gold-set topic names that did not round-trip.",
    )
    p.add_argument(
        "--min-macro-recall", type=float, default=1.0,
        help=("Exit non-zero when any domain macro-recall drops below this "
              "value. Default 1.0 because the corpus is synthetic by "
              "construction; <1.0 indicates a malformed gold-set topic."),
    )
    args = p.parse_args()

    domains = _list_known_domains() if args.domain == "all" else [args.domain]
    results = [evaluate_synthetic(d) for d in domains]
    print(render_report(results, verbose=args.verbose))
    failures = [r for r in results if r["macro_recall"] < args.min_macro_recall - 1e-9]
    if failures:
        print(
            "\n[fail] synthetic macro recall below "
            f"min={args.min_macro_recall:+.4f} for: "
            + ", ".join(f["domain"] for f in failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
