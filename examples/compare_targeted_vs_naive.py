"""Targeted-vs-naive baseline example for LAPPATO_MCB.

Runs LAPPATO_MCB twice on the SAME diagnostic CSVs, once with the
domain manifest (e.g. WDBC's 5 weakness-targeted entries) and once
with a single-entry naive manifest (a generic "<domain> machine
learning" query). Then computes set-precision / set-recall / overlap
between the two harvested sets, treating the targeted run as the
"reference" set.

This converts the qualitative claim ("targeted queries surface
different literature than naive queries") into a measurable artefact:
the Jaccard overlap of the two harvest sets, broken down by source.

The script is deliberately offline-friendly: pass --use-corpus to make
both arms search the local cache (cache.py) instead of hitting the
network. This is the path used in CI / reproducibility runs where
network access is undesirable.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Make the `lappato_mcb` package importable when this file is run directly
# as a script (no `pip install -e .` required).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb import LAPPATO_MCB  # noqa: E402
from lappato_mcb.cache import CorpusCache  # noqa: E402
from lappato_mcb.fingerprint import TitleDeduper  # noqa: E402
from lappato_mcb.manifests import get as get_manifest  # noqa: E402
from lappato_mcb.manifests import naive_baseline_manifest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _set_keys(rows: list[dict]) -> set[str]:
    """Stable per-paper key — id when available, else normalised title."""
    from lappato_mcb.fingerprint import normalise_title
    keys: set[str] = set()
    for r in rows:
        eid = (r.get("id") or "").strip()
        if eid:
            keys.add(eid)
        else:
            keys.add(normalise_title(r.get("title", "")))
    return keys


def _run_one_arm(arm_tag: str, manifest: list[dict], *, offline: bool,
                 max_cycles: int) -> Path:
    """Drive a single LAPPATO_MCB run for `max_cycles` polls and stop it.

    Returns the path to the JSONL sidecar written by the arm.
    """
    print(f"  arm={arm_tag}  manifest_entries={len(manifest)}  offline={offline}")
    # Polite-use cycles can be slow because of the 3 s arXiv sleep
    # (~6-12 s per active weakness when online); offline mode is fast.
    poll = 1.0 if offline else 5.0
    lappato_mcb = LAPPATO_MCB(
        project_root=ROOT,
        manifest=manifest,
        run_tag=arm_tag,
        poll_interval=poll,
        offline=offline,
    )
    lappato_mcb.start()
    try:
        # Loop on cycle counter rather than wall clock so that long
        # online cycles still yield exactly `max_cycles` cycles.
        target = max_cycles
        deadline = time.monotonic() + (60.0 if offline else 600.0)
        while lappato_mcb._cycle_n < target and time.monotonic() < deadline:
            time.sleep(0.5)
    finally:
        lappato_mcb.stop()
    return CHECKPOINTS / f"{arm_tag}_lappato_mcb_papers.jsonl"


def _comparison_report(targeted: list[dict], naive: list[dict]) -> dict:
    t_keys = _set_keys(targeted)
    n_keys = _set_keys(naive)
    inter = t_keys & n_keys
    union = t_keys | n_keys
    return {
        "n_targeted": len(t_keys),
        "n_naive": len(n_keys),
        "n_intersection": len(inter),
        "n_union": len(union),
        "jaccard": (len(inter) / len(union)) if union else 0.0,
        "targeted_only": len(t_keys - n_keys),
        "naive_only": len(n_keys - t_keys),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="wdbc",
                   choices=["wdbc", "nlp", "timeseries"])
    p.add_argument("--use-corpus", action="store_true",
                   help="Run both arms in offline mode against checkpoints/corpus.jsonl.")
    p.add_argument("--cycles", type=int, default=2,
                   help="Number of cycles per arm (default 2).")
    args = p.parse_args()

    manifest, run_tag = get_manifest(args.domain)
    if args.use_corpus:
        n_corpus = len(CorpusCache(CHECKPOINTS / "corpus.jsonl"))
        if n_corpus == 0:
            print("⚠️  --use-corpus set but corpus.jsonl is empty. "
                  "Run a domain demo at least once online first.")
            return
        print(f"Using local corpus: {n_corpus} rows")

    # Targeted arm.
    targeted_path = _run_one_arm(
        f"{run_tag}_targeted", manifest,
        offline=args.use_corpus, max_cycles=args.cycles,
    )
    # Naive arm.
    naive_path = _run_one_arm(
        f"{run_tag}_naive", naive_baseline_manifest(run_tag),
        offline=args.use_corpus, max_cycles=args.cycles,
    )

    targeted = _read_jsonl(targeted_path)
    naive = _read_jsonl(naive_path)
    report = _comparison_report(targeted, naive)

    out_csv = CHECKPOINTS / f"{run_tag}_baseline_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in report.items():
            w.writerow([k, v])

    # Trigram dedup overlap report — how many naive hits the
    # cross-source title dedup would have suppressed if both arms had
    # been run inside one LAPPATO_MCB. This re-uses the production
    # fingerprint module to keep semantics consistent.
    dedup = TitleDeduper()
    for r in targeted:
        dedup.seen_or_register(r.get("title", ""))
    naive_dups = sum(
        1 for r in naive
        if dedup.is_near_duplicate(r.get("title", ""))[0]
    )
    print()
    print(f"  targeted     : {report['n_targeted']:5d}")
    print(f"  naive        : {report['n_naive']:5d}")
    print(f"  ∩            : {report['n_intersection']:5d}")
    print(f"  Jaccard      : {report['jaccard']:.3f}")
    print(f"  targeted-only: {report['targeted_only']}")
    print(f"  naive-only   : {report['naive_only']}")
    print(f"  trigram-dedup naive hits ⊂ targeted: {naive_dups}")
    print(f"\n  written -> {out_csv}")


if __name__ == "__main__":
    main()
