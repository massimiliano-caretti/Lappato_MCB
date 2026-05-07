"""Measure LAPPATO_MCB topic recall against the bundled gold sets.

LAPPATO_MCB has no semantic ranker and no learned retriever; the
``compare_targeted_vs_naive.py`` example reports set overlap between
the targeted and naive arms but does not say whether either arm
actually surfaces well-known relevant work for the manifest's
weaknesses. This harness closes that gap with a small, honest
sanity-check.

What it does:
  1. Loads ``docs/gold_sets/<domain>_gold.json`` — a hand-curated list
     of expected topics per weakness, each defined as a conjunction of
     case-insensitive keyword matches.
  2. Reads the JSONL paper sidecar from a previous LAPPATO_MCB run
     (``checkpoints/<run_tag>_lappato_mcb_papers.jsonl``).
  3. For each expected topic, marks it as *found* when at least one
     harvested paper's title or abstract contains all of the topic's
     keywords (case-insensitive).
  4. Reports per-weakness, per-domain and macro Recall@K.

Honest scope: the gold sets are small and curated by the maintainers,
not externally validated. A high recall does *not* establish absolute
coverage — it shows that LAPPATO_MCB reaches the topics a domain
expert would expect to see for the bundled manifests. The harness's
operational role is regression detection: if the macro recall on the
bundled corpus drops below the documented baseline (default 0.70),
investigate.

Run:
    # measure recall on a previously-completed run
    python examples/measure_recall.py --domain wdbc

    # combined report across every domain that has a gold set on disk
    python examples/measure_recall.py --domain all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "docs" / "gold_sets"
CHECKPOINTS = ROOT / "checkpoints"


def _discover_domains() -> tuple[str, ...]:
    """Return every domain that has a ``<tag>_gold.json`` next to this file.

    The harness used to hard-code the three bundled domains; with the
    expansion to 35 manifests + 35 gold sets we now enumerate them
    dynamically. The original three remain importable under the same
    names — the only behavioural change is that ``--domain all``
    iterates over every gold set found on disk.
    """
    if not GOLD_DIR.exists():
        return ()
    tags: list[str] = []
    for path in sorted(GOLD_DIR.glob("*_gold.json")):
        tag = path.name[: -len("_gold.json")]
        if tag:
            tags.append(tag)
    return tuple(tags)


KNOWN_DOMAINS = _discover_domains() or ("wdbc", "nlp", "timeseries")


def _load_gold(domain: str) -> dict:
    path = GOLD_DIR / f"{domain}_gold.json"
    if not path.exists():
        raise SystemExit(f"missing gold set for domain {domain!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_papers(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _topic_found(topic: dict, paper_blobs: list[str]) -> bool:
    """True when at least one blob contains *every* topic keyword."""
    keywords = [k.lower() for k in (topic.get("must_match_keywords") or []) if k]
    if not keywords:
        return False
    for blob in paper_blobs:
        if all(k in blob for k in keywords):
            return True
    return False


def evaluate_domain(domain: str, papers: list[dict]) -> dict:
    """Compute per-weakness recall for a single domain."""
    gold = _load_gold(domain)
    topics_by_weakness = gold.get("topics_by_weakness") or {}

    # Group harvested papers by weakness so per-weakness recall reflects
    # what the matching manifest entry actually retrieved.
    by_weakness: dict[str, list[str]] = {}
    for p in papers:
        wid = p.get("weakness", "")
        if not wid:
            continue
        blob = (
            (p.get("title") or "") + " " + (p.get("abstract") or "")
        ).lower()
        by_weakness.setdefault(wid, []).append(blob)

    per_weakness: list[dict] = []
    total_topics_active = 0
    total_found_active = 0
    for wid, topics in topics_by_weakness.items():
        blobs = by_weakness.get(wid, [])
        found = [t for t in topics if _topic_found(t, blobs)]
        active = bool(blobs)
        per_weakness.append({
            "weakness": wid,
            "n_topics": len(topics),
            "n_found": len(found),
            "recall": (len(found) / len(topics)) if topics else 0.0,
            "missing": [
                t["name"] for t in topics if not _topic_found(t, blobs)
            ],
            "n_papers_in_weakness": len(blobs),
            "active": active,
        })
        if active:
            total_topics_active += len(topics)
            total_found_active += len(found)

    # Macro/micro recall is computed over *active* weaknesses only —
    # weaknesses that did not fire in the source run cannot retrieve
    # papers and would otherwise mechanically depress the score. The
    # report still shows inactive weaknesses for transparency.
    active_weaknesses = [w for w in per_weakness if w["active"]]
    macro_recall = (
        sum(w["recall"] for w in active_weaknesses) / len(active_weaknesses)
    ) if active_weaknesses else 0.0
    micro_recall = (
        total_found_active / total_topics_active
    ) if total_topics_active else 0.0

    return {
        "domain": domain,
        "n_papers_total": len(papers),
        "per_weakness": per_weakness,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "n_topics_total": total_topics_active,
        "n_topics_found": total_found_active,
        "n_active_weaknesses": len(active_weaknesses),
    }


def render_report(results: list[dict]) -> str:
    lines: list[str] = ["LAPPATO_MCB topic recall vs. bundled gold sets",
                        "=" * 60]
    macro_grand: list[float] = []
    for r in results:
        lines.append("")
        lines.append(
            f"domain={r['domain']:>10s}  papers={r['n_papers_total']:5d}  "
            f"topics={r['n_topics_total']:3d}  found={r['n_topics_found']:3d}  "
            f"macro_recall={r['macro_recall']:+.3f}  "
            f"micro_recall={r['micro_recall']:+.3f}"
        )
        for w in r["per_weakness"]:
            if not w["active"]:
                tag = "--"
            elif w["recall"] >= 0.5:
                tag = "OK"
            else:
                tag = "  "
            lines.append(
                f"  [{tag}] {w['weakness']:>32s}  "
                f"{w['n_found']:>2d}/{w['n_topics']:<2d}  "
                f"recall={w['recall']:+.3f}  "
                f"papers={w['n_papers_in_weakness']:3d}"
                + ("  (inactive)" if not w["active"] else "")
            )
            if w["active"]:
                for m in w["missing"]:
                    lines.append(f"        miss: {m}")
        macro_grand.append(r["macro_recall"])
    if macro_grand:
        grand = sum(macro_grand) / len(macro_grand)
        lines.append("")
        lines.append(f"GRAND macro recall (avg over domains): {grand:+.3f}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--domain", default="all",
        choices=("all",) + KNOWN_DOMAINS,
        help="Which domain to evaluate. 'all' runs every bundled domain.",
    )
    p.add_argument(
        "--checkpoints", type=Path, default=CHECKPOINTS,
        help="Override the checkpoints directory (where *_papers.jsonl live).",
    )
    p.add_argument(
        "--min-macro-recall", type=float, default=0.70,
        help=("Exit non-zero when any domain macro-recall drops below "
              "this value. CI uses the default to catch regressions."),
    )
    args = p.parse_args()

    domains = KNOWN_DOMAINS if args.domain == "all" else (args.domain,)
    results: list[dict] = []
    missing_runs: list[str] = []
    for d in domains:
        jsonl = args.checkpoints / f"{d}_lappato_mcb_papers.jsonl"
        papers = _load_papers(jsonl)
        if not papers:
            missing_runs.append(d)
            continue
        results.append(evaluate_domain(d, papers))

    if not results:
        print(
            "No paper sidecars found in "
            f"{args.checkpoints}. Run a demo first, e.g.:\n"
            "    python examples/demo_wdbc.py",
            file=sys.stderr,
        )
        return 2

    print(render_report(results))
    if missing_runs:
        print(
            "\n[skip] no paper sidecar for: "
            + ", ".join(missing_runs),
            file=sys.stderr,
        )

    failures = [
        r for r in results if r["macro_recall"] < args.min_macro_recall
    ]
    if failures:
        print(
            "\n[fail] macro recall below "
            f"min-macro-recall={args.min_macro_recall:+.3f} for: "
            + ", ".join(f["domain"] for f in failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
