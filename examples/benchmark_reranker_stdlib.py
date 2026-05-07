"""Phase 0.5 micro-benchmark: does a stdlib-only reranker move recall?

What this is
------------
A controlled offline experiment that answers a single question:

    On synthetic corpora derived from each of the 35 bundled gold
    sets, does adding a reranker to the lexical baseline improve
    Recall@K?

The reranker is the stdlib-only
:class:`lappato_mcb.rerankers.trigram.TrigramJaccardReranker`. We
deliberately pick a *weak* reranker (char-trigram Jaccard) so the
result is a **lower bound** on what an embedding-based reranker
(Phase 2) could deliver. If even this weak reranker produces
meaningful uplift, a real embedding model is plausibly worth its
dependency footprint. If it doesn't, embedding-based reranking
needs stronger evidence before we commit to shipping it.

What this is NOT
----------------
- Not a measurement on real arXiv / OpenAlex / Crossref harvests.
- Not a study with human-labelled relevance judgements.
- Not a validation that the *manifests' detectors* work; that lives
  in ``tests/test_detector_thresholds.py``.

The synthetic corpora are constructed so that the lexical baseline
will fail or struggle on the **hard** scenario by design (titles
use morphological variants of the keywords instead of the verbatim
keywords). The trigram reranker is *expected* to help on those.

Two scenarios per domain
------------------------
**Easy (control)**

  For every (weakness, topic) tuple in the gold set:
    - 1 "verbatim relevant" paper whose title contains every
      keyword verbatim → both rankers should put it in top-K.
    - 19 "noise" papers whose titles are random unrelated text
      drawn from a fixed seed.

  Expectation: both rankers hit Recall@K ≈ 1.0. A drop here would
  signal a regression.

**Hard (the interesting case)**

  For every (weakness, topic) tuple:
    - 1 "morphologically shifted relevant" paper whose title
      contains *transformed* keywords (suffix added, prefix added,
      compound word). The original keyword string is NOT present.
    - 19 "near-noise" papers whose titles randomly contain ONE
      keyword from the topic (so token-overlap may rank them above
      the relevant paper).

  Expectation: lexical baseline degrades; reranker rescues some
  topics by exploiting the trigram overlap between the keyword and
  its morphological variant.

Output
------
For each domain and scenario, prints (and optionally saves to JSON):
  - Recall@5 lexical only
  - Recall@5 with reranker
  - Recall@10 lexical only
  - Recall@10 with reranker
  - Δ at K=5 and K=10
  - Per-topic detail in --verbose

A summary at the bottom averages Δ across domains and decides
whether the **decision gate** is satisfied (Δ_macro_recall ≥ 0.05
on the hard scenario).

Usage
-----
::

    # full benchmark, summary only
    python examples/benchmark_reranker_stdlib.py

    # one domain, with per-topic detail
    python examples/benchmark_reranker_stdlib.py --domain neuroscience --verbose

    # JSON dump for further analysis / regression test
    python examples/benchmark_reranker_stdlib.py --json out/bench.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lappato_mcb.core import _DEFAULT_RERANKER_FLOOR, _DEFAULT_RERANKER_WEIGHT  # noqa: E402
from lappato_mcb.rerankers.trigram import TrigramJaccardReranker  # noqa: E402

GOLD_DIR = ROOT / "docs" / "gold_sets"

# ─── Knobs (deterministic, reproducible) ─────────────────────────────
N_NOISE_PAPERS_PER_TOPIC = 19   # keeps corpus large enough for K=5/10
RANDOM_SEED = 42
KS = (5, 10)


def _list_known_domains() -> list[str]:
    return sorted(p.stem.replace("_gold", "") for p in GOLD_DIR.glob("*_gold.json"))


def _load_gold(domain: str) -> dict:
    return json.loads((GOLD_DIR / f"{domain}_gold.json").read_text(encoding="utf-8"))


# ─── Fixture synthesis ────────────────────────────────────────────────
# Morphological transformations: each keyword is rewritten so that
# the resulting string shares trigrams with the original (which the
# trigram reranker can exploit) but does NOT contain the original
# keyword as a substring (so token-overlap fails).
_SUFFIXES = ("ation", "ative", "ising", "ized", "ical", "istic", "ly", "ist")
_PREFIXES = ("post", "pre", "sub", "meta", "anti", "non")


def _morph_variant(kw: str, rng: random.Random) -> str:
    """Return a morphological variant of ``kw`` that:
      - starts with ``kw[:3]`` (so trigram overlap is non-zero),
      - does not contain ``kw`` as a substring,
      - ends with one of ``_SUFFIXES`` *or* starts with one of
        ``_PREFIXES`` (the rng decides).
    """
    kw = kw.strip()
    if not kw:
        return kw
    if len(kw) < 3:
        # Acronyms: append a suffix that preserves trigrams ('rdd' →
        # 'rddistic'). The reranker still sees overlap, the lexical
        # baseline still misses.
        return kw + rng.choice(_SUFFIXES)
    base = kw[:-1]  # drop last char so the original keyword is no
    # longer a substring
    suffix = rng.choice(_SUFFIXES)
    prefix = rng.choice(_PREFIXES)
    if rng.random() < 0.5:
        return base + suffix
    return prefix + "-" + kw[:-1] + suffix


def _noise_title(seed_words: list[str], rng: random.Random) -> str:
    n = rng.randint(4, 9)
    chosen = rng.sample(seed_words, k=min(n, len(seed_words)))
    return " ".join(chosen).capitalize()


_NOISE_VOCAB = (
    "the of and in for to on a method analysis study evaluation "
    "approach framework system results comparison overview survey "
    "novel proposed empirical theoretical unrelated arbitrary"
).split()


def synthesize_easy(domain: str, rng: random.Random) -> list[dict]:
    """Easy scenario: relevant papers contain keywords verbatim."""
    gold = _load_gold(domain)
    papers: list[dict] = []
    counter = 0
    for weakness, topics in gold.get("topics_by_weakness", {}).items():
        for topic in topics:
            kws = [k for k in (topic.get("must_match_keywords") or []) if k]
            if not kws:
                continue
            counter += 1
            papers.append({
                "weakness": weakness,
                "topic_name": topic.get("name", ""),
                "title": (
                    f"Easy paper #{counter}: " + " ".join(kws)
                ),
                "abstract": "Synthetic abstract mentioning " + ", ".join(kws),
                "_relevant": True,
                "_keywords": kws,
            })
            for j in range(N_NOISE_PAPERS_PER_TOPIC):
                papers.append({
                    "weakness": weakness,
                    "topic_name": topic.get("name", ""),
                    "title": _noise_title(_NOISE_VOCAB, rng),
                    "abstract": "Filler text without specific terminology.",
                    "_relevant": False,
                    "_keywords": kws,
                })
    return papers


def synthesize_hard(domain: str, rng: random.Random) -> list[dict]:
    """Hard scenario: relevant papers use morphological variants;
    noise papers may share ONE keyword to confuse lexical scoring.
    """
    gold = _load_gold(domain)
    papers: list[dict] = []
    counter = 0
    for weakness, topics in gold.get("topics_by_weakness", {}).items():
        for topic in topics:
            kws = [k for k in (topic.get("must_match_keywords") or []) if k]
            if not kws:
                continue
            counter += 1
            variants = [_morph_variant(k, rng) for k in kws]
            papers.append({
                "weakness": weakness,
                "topic_name": topic.get("name", ""),
                "title": (
                    f"Hard paper #{counter}: " + " ".join(variants)
                ),
                # Abstract avoids the verbatim keywords too.
                "abstract": "Methodological note on " + " and ".join(variants),
                "_relevant": True,
                "_keywords": kws,
                "_variants": variants,
            })
            for j in range(N_NOISE_PAPERS_PER_TOPIC):
                # ~30% of noise papers contain ONE verbatim keyword
                # so the lexical baseline ranks them above the
                # variant-based relevant paper.
                if rng.random() < 0.3 and kws:
                    decoy = rng.choice(kws)
                    title = (
                        _noise_title(_NOISE_VOCAB, rng)
                        + " " + decoy
                    )
                else:
                    title = _noise_title(_NOISE_VOCAB, rng)
                papers.append({
                    "weakness": weakness,
                    "topic_name": topic.get("name", ""),
                    "title": title,
                    "abstract": "Filler text.",
                    "_relevant": False,
                    "_keywords": kws,
                })
    return papers


# ─── Scoring ──────────────────────────────────────────────────────────
def _lexical_score(kws: list[str], paper: dict) -> float:
    """Cheap, deterministic token-overlap score. Favours the relevant
    paper proportionally to how many query keywords appear in the
    title+abstract blob (lowercased, substring match — same shape
    used by the gold-set harness).
    """
    blob = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    return float(sum(1 for k in kws if k.lower() in blob))


def _blend_scores(lex: list[float], rerank: list[float],
                  weight: float, floor: float) -> list[float]:
    """Additive blend currently used in ``LAPPATO_MCB._apply_reranker``."""
    out: list[float] = []
    for l, r in zip(lex, rerank):
        contrib = max(r - floor, 0.0)
        out.append(l + weight * contrib)
    return out


def _rrf_score(scores: list[float], k_const: int = 60) -> list[float]:
    """Reciprocal Rank Fusion contribution for a single ranking.

    Cormack, Clarke & Buettcher (SIGIR 2009): for each item with rank
    ``rank`` (1-based), contribute ``1 / (k_const + rank)``. The
    ``k_const`` default of 60 is the value used in the original paper
    and is robust across most retrieval domains.

    RRF is **scale-invariant**: it depends only on the *ranking*
    produced by each scorer, not on the absolute scores. This is
    what makes it suitable for blending heterogeneous rankers
    (lexical vs semantic, raw counts vs cosine, etc.) without
    normalisation.
    """
    n = len(scores)
    if n == 0:
        return []
    paired = sorted(range(n), key=lambda i: -scores[i])
    out = [0.0] * n
    for rank, idx in enumerate(paired):
        out[idx] = 1.0 / (k_const + rank + 1)
    return out


def _rrf_blend(lex: list[float], rerank: list[float],
               rerank_weight: float = 1.0) -> list[float]:
    """RRF blend of two rankings, with optional reranker weight.

    ``rerank_weight=1.0`` is the symmetric / balanced RRF
    (Cormack default). ``rerank_weight > 1.0`` puts the semantic /
    reranker channel ahead — useful when the lexical baseline is
    known to be brittle for the target use case.
    """
    rrf_l = _rrf_score(lex)
    rrf_r = _rrf_score(rerank)
    return [a + rerank_weight * b for a, b in zip(rrf_l, rrf_r)]


def _topic_query(kws: list[str]) -> str:
    """The query the reranker / lexical scorer sees: the keywords
    joined by spaces. This is what a manifest's queries reduce to in
    practice once placeholders are rendered."""
    return " ".join(kws)


def _topic_groups(papers: list[dict]) -> dict:
    """Group synthesised papers by their (weakness, topic) so we can
    measure recall on a per-topic basis. Each group has 1 relevant
    paper and N_NOISE_PAPERS_PER_TOPIC noise papers."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for p in papers:
        key = (p["weakness"], p["topic_name"])
        groups.setdefault(key, []).append(p)
    return groups


def _recall_at_k(group: list[dict], scores: list[float], k: int) -> float:
    paired = list(zip(group, scores))
    # Stable secondary key: original order, so ties are deterministic.
    paired.sort(key=lambda gs: (-gs[1], group.index(gs[0])))
    top = paired[:k]
    relevant_in_top = sum(1 for p, _ in top if p.get("_relevant"))
    return float(relevant_in_top)


_MODES = ("lex", "additive", "rrf", "rrf_2x", "rerank_only")


def _scores_for_mode(mode: str, lex: list[float], rerank: list[float],
                     weight: float, floor: float) -> list[float]:
    if mode == "lex":
        return list(lex)
    if mode == "additive":
        return _blend_scores(lex, rerank, weight, floor)
    if mode == "rrf":
        return _rrf_blend(lex, rerank, rerank_weight=1.0)
    if mode == "rrf_2x":
        return _rrf_blend(lex, rerank, rerank_weight=2.0)
    if mode == "rerank_only":
        return list(rerank)
    raise ValueError(f"unknown mode: {mode!r}")


def evaluate_scenario(domain: str, papers: list[dict],
                      reranker, *, weight: float, floor: float) -> dict:
    groups = _topic_groups(papers)
    per_topic: list[dict] = []
    sums = {(m, k): 0.0 for m in _MODES for k in KS}
    for (weakness, topic_name), group in groups.items():
        kws = group[0].get("_keywords") or []
        query = _topic_query(kws)
        lex = [_lexical_score(kws, p) for p in group]
        rerank = reranker.score(query, group) if reranker else [0.0] * len(group)
        topic_row = {
            "weakness": weakness,
            "topic": topic_name,
            "n_relevant": sum(1 for p in group if p.get("_relevant")),
            "n_papers": len(group),
        }
        for mode in _MODES:
            scores = _scores_for_mode(mode, lex, rerank, weight, floor)
            for k in KS:
                r = _recall_at_k(group, scores, k)
                topic_row[f"recall_at_{k}_{mode}"] = r
                sums[(mode, k)] += r
        per_topic.append(topic_row)
    n = len(per_topic)
    summary = {
        "domain": domain,
        "n_topics": n,
        "n_papers_per_topic": N_NOISE_PAPERS_PER_TOPIC + 1,
    }
    for mode in _MODES:
        for k in KS:
            summary[f"macro_recall_at_{k}_{mode}"] = (
                (sums[(mode, k)] / n) if n else 0.0
            )
    # Backwards-compatible aliases used by the older render path.
    for k in KS:
        summary[f"macro_recall_at_{k}_lex"] = summary[f"macro_recall_at_{k}_lex"]
        summary[f"macro_recall_at_{k}_blend"] = summary[f"macro_recall_at_{k}_additive"]
        summary[f"delta_at_{k}"] = (
            summary[f"macro_recall_at_{k}_additive"]
            - summary[f"macro_recall_at_{k}_lex"]
        )
    return {"summary": summary, "per_topic": per_topic}


def render_text(report: dict, verbose: bool = False) -> str:
    lines: list[str] = []
    lines.append("Phase 0.5 micro-benchmark — TrigramJaccardReranker")
    lines.append("=" * 76)
    lines.append("")
    lines.append(
        f"Reranker:        {report['reranker']['name']} "
        f"(weight={report['reranker']['weight']}, "
        f"floor={report['reranker']['floor']})"
    )
    lines.append(
        f"Domains:         {len(report['domains'])}  "
        f"papers per topic: {N_NOISE_PAPERS_PER_TOPIC + 1}  "
        f"seed: {RANDOM_SEED}"
    )
    lines.append("")
    n_dom = len(report["domains"])
    for scenario in ("easy", "hard"):
        lines.append(f"--- {scenario.upper()} scenario ---")
        lines.append("")
        lines.append("  Per-mode macro recall, averaged across domains:")
        lines.append("")
        lines.append(
            f"  {'mode':>32s}    {'R@5':>6s}     {'R@10':>6s}     "
            f"{'Δ@5 vs lex':>10s}     {'Δ@10 vs lex':>11s}"
        )
        lines.append("  " + "-" * 78)
        sums = {(m, k): 0.0 for m in _MODES for k in KS}
        for d, scen in report["domains"].items():
            s = scen[scenario]["summary"]
            for m in _MODES:
                for k in KS:
                    sums[(m, k)] += s[f"macro_recall_at_{k}_{m}"]
        baselines = {k: sums[("lex", k)] / n_dom for k in KS}
        labels = {
            "lex": "lexical only (current core baseline)",
            "additive": "additive blend (Phase 0 default)",
            "rrf": "RRF balanced (lex + rerank)",
            "rrf_2x": "RRF (lex + 2x rerank)",
            "rerank_only": "reranker only (theoretical max)",
        }
        for m in _MODES:
            r5 = sums[(m, 5)] / n_dom
            r10 = sums[(m, 10)] / n_dom
            d5 = r5 - baselines[5]
            d10 = r10 - baselines[10]
            tag = "" if m == "lex" else (
                "  ←" if d5 > 0.05 else (" ✗" if d5 < -0.005 else "")
            )
            lines.append(
                f"  {labels[m]:>32s}    {r5:.4f}     {r10:.4f}     "
                f"{d5:+.4f}      {d10:+.4f}{tag}"
            )
        lines.append("")
        if verbose:
            lines.append(f"  Per-domain detail ({scenario}):")
            for d, scen in report["domains"].items():
                s = scen[scenario]["summary"]
                lines.append(
                    f"    {d:>22s}  topics={s['n_topics']:>2d}  "
                    f"lex5={s['macro_recall_at_5_lex']:.3f}  "
                    f"add5={s['macro_recall_at_5_additive']:.3f}  "
                    f"rrf5={s['macro_recall_at_5_rrf']:.3f}  "
                    f"rrf2x5={s['macro_recall_at_5_rrf_2x']:.3f}  "
                    f"rro5={s['macro_recall_at_5_rerank_only']:.3f}"
                )
            lines.append("")
    return "\n".join(lines)


def run_benchmark(*, domains: list[str] | None = None,
                  weight: float = _DEFAULT_RERANKER_WEIGHT,
                  floor: float = _DEFAULT_RERANKER_FLOOR) -> dict:
    if domains is None:
        domains = _list_known_domains()
    rng_seed = RANDOM_SEED
    reranker = TrigramJaccardReranker()
    out: dict = {
        "reranker": {
            "name": reranker.model_name,
            "version": reranker.model_version,
            "weight": weight,
            "floor": floor,
        },
        "n_noise_per_topic": N_NOISE_PAPERS_PER_TOPIC,
        "seed": rng_seed,
        "domains": {},
    }
    for d in domains:
        rng_easy = random.Random(rng_seed)
        rng_hard = random.Random(rng_seed + 1)
        easy_papers = synthesize_easy(d, rng_easy)
        hard_papers = synthesize_hard(d, rng_hard)
        out["domains"][d] = {
            "easy": evaluate_scenario(d, easy_papers, reranker,
                                      weight=weight, floor=floor),
            "hard": evaluate_scenario(d, hard_papers, reranker,
                                      weight=weight, floor=floor),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="all",
                   help="Single domain or 'all' (default).")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-topic recall changes.")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to dump the full report as JSON.")
    p.add_argument("--weight", type=float, default=_DEFAULT_RERANKER_WEIGHT)
    p.add_argument("--floor", type=float, default=_DEFAULT_RERANKER_FLOOR)
    args = p.parse_args()

    domains = _list_known_domains() if args.domain == "all" else [args.domain]
    report = run_benchmark(domains=domains, weight=args.weight,
                           floor=args.floor)
    print(render_text(report, verbose=args.verbose))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
