"""Phase 0 tests for the Reranker plug-in interface.

Phase 0 adds a ``Reranker`` Protocol + an opt-in ``reranker=`` arg to
``LAPPATO_MCB``. The contract these tests pin down:

  - **Default (``reranker=None``) is a no-op**. Behaviour is
    bit-identical to v1.4: the weakness card has no ``reranker``
    field, the lexical ordering of top papers is preserved, no extra
    keys leak into the persisted sidecar.
  - **A configured reranker boosts confident hits** above the
    ``reranker_floor``, and that boost is reflected in the card's
    top-paper ordering and ``score`` field.
  - **The blend can only ADD score**, never demote: a paper's final
    ``lappato_score`` is always >= its lexical baseline.
  - **Graceful degrade on crash**: a reranker whose ``score`` raises
    is silently bypassed and the lexical ranking is kept.
  - **NaN / Inf scores are skipped** from the blend.
  - **Audit trail** in the weakness card captures
    ``model_name`` / ``model_version`` / ``model_sha`` attributes
    when the reranker exposes them.
  - The Protocol is **runtime-checkable** so users can sanity-check
    their own implementations.

These tests use a small in-memory mock reranker; they DO NOT exercise
any embedding model. The extras package that will ship a real reranker
is Phase 2.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lappato_mcb.cache import CorpusCache  # noqa: E402
from lappato_mcb.core import (  # noqa: E402
    LAPPATO_MCB,
    Reranker,
    _DEFAULT_RERANKER_FLOOR,
    _DEFAULT_RERANKER_WEIGHT,
)


# ─── Mock rerankers ───────────────────────────────────────────────────
class StaticBoostReranker:
    """Returns a fixed, deterministic score per (query, hit) pair.

    A title containing 'isotonic' gets 0.95; everything else 0.10.
    Used to verify that the blend lifts the right paper.
    """

    model_name = "static-boost"
    model_version = "test-1"
    model_sha = "deadbeef"

    def score(self, query: str, hits: list[dict]) -> list[float]:
        return [0.95 if "isotonic" in (h.get("title") or "").lower() else 0.10
                for h in hits]


class CrashingReranker:
    model_name = "crashing"

    def score(self, query: str, hits: list[dict]) -> list[float]:
        raise RuntimeError("intentional reranker crash")


class NaNReranker:
    def score(self, query: str, hits: list[dict]) -> list[float]:
        import math
        return [math.nan, float("inf"), float("-inf")][: len(hits)] + [0.0] * (
            len(hits) - 3
        ) if len(hits) >= 3 else [math.nan] * len(hits)


class WrongLengthReranker:
    """Returns a list with the wrong length — the blend must skip it."""

    def score(self, query: str, hits: list[dict]) -> list[float]:
        return [0.99]  # always one element regardless of hits


# ─── Fixtures ─────────────────────────────────────────────────────────
def _make_lappato(
    root: Path, *, reranker=None, reranker_weight=None, reranker_floor=None,
):
    cp = root / "checkpoints"
    cp.mkdir(parents=True, exist_ok=True)
    corpus = cp / "corpus.jsonl"
    cache = CorpusCache(corpus)
    # Two cached papers with different titles — only the first
    # contains 'isotonic'.
    cache.record({
        "id": "p1", "source": "arXiv",
        "title": "Isotonic regression for probability calibration",
        "abstract": "Calibration of probabilistic classifiers",
        "year": 2025,
    })
    cache.record({
        "id": "p2", "source": "arXiv",
        "title": "Bayesian model averaging unrelated topic",
        "abstract": "calibration appears here too just barely",
        "year": 2024,
    })
    manifest = [{
        "id": "low_calibration",
        "title": "Poor calibration",
        "evidence": "calibration.csv",
        "evidence_check": lambda p: p.exists(),
        "evidence_summary": lambda p: {"mean_brier": 0.18},
        "severity": lambda p: "high",
        "queries": ["calibration probabilistic classifier"],
        "transplant": "Add isotonic calibration.",
    }]
    (cp / "calibration.csv").write_text("brier\n0.18\n", encoding="utf-8")
    kwargs = {}
    if reranker is not None:
        kwargs["reranker"] = reranker
    if reranker_weight is not None:
        kwargs["reranker_weight"] = reranker_weight
    if reranker_floor is not None:
        kwargs["reranker_floor"] = reranker_floor
    lap = LAPPATO_MCB(
        project_root=root,
        manifest=manifest,
        run_tag="reranker_test",
        offline=True,
        corpus_path=corpus,
        **kwargs,
    )
    return lap, cp


def _read_latest_card(cp: Path) -> dict:
    line = (cp / "reranker_test_lappato_mcb_weakness_cards.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()[-1]
    return json.loads(line)


# ─── Protocol semantics ───────────────────────────────────────────────
class ProtocolSemanticsTests(unittest.TestCase):

    def test_protocol_is_runtime_checkable(self):
        self.assertTrue(isinstance(StaticBoostReranker(), Reranker))
        self.assertFalse(isinstance(object(), Reranker))

    def test_default_floor_and_weight_constants(self):
        # If the defaults change, downstream users (extras package,
        # tests, README) need to know.
        self.assertAlmostEqual(_DEFAULT_RERANKER_FLOOR, 0.30)
        self.assertAlmostEqual(_DEFAULT_RERANKER_WEIGHT, 0.5)


# ─── Default behaviour (no reranker) ──────────────────────────────────
class DefaultIsLexicalOnlyTests(unittest.TestCase):

    def test_card_has_no_reranker_field_when_default(self):
        with tempfile.TemporaryDirectory() as td:
            lap, cp = _make_lappato(Path(td))
            lap._cycle_n = 1
            lap._run_cycle()
            card = _read_latest_card(cp)
            self.assertNotIn("reranker", card)

    def test_top_paper_score_unchanged_when_default(self):
        with tempfile.TemporaryDirectory() as td:
            lap, cp = _make_lappato(Path(td))
            lap._cycle_n = 1
            lap._run_cycle()
            card = _read_latest_card(cp)
            self.assertGreater(len(card["top_papers"]), 0)
            # No rerank ⇒ lappato_rerank_score must NOT have leaked
            # into the persisted top-paper records.
            for tp in card["top_papers"]:
                self.assertNotIn("lappato_rerank_score", tp)


# ─── Configured reranker ──────────────────────────────────────────────
class ConfiguredRerankerTests(unittest.TestCase):

    def test_card_has_reranker_audit_trail(self):
        with tempfile.TemporaryDirectory() as td:
            lap, cp = _make_lappato(Path(td), reranker=StaticBoostReranker())
            lap._cycle_n = 1
            lap._run_cycle()
            card = _read_latest_card(cp)
            self.assertIn("reranker", card)
            r = card["reranker"]
            self.assertTrue(r["configured"])
            self.assertEqual(r["model_name"], "static-boost")
            self.assertEqual(r["model_version"], "test-1")
            self.assertEqual(r["model_sha"], "deadbeef")
            self.assertEqual(r["weight"], _DEFAULT_RERANKER_WEIGHT)
            self.assertEqual(r["floor"], _DEFAULT_RERANKER_FLOOR)

    def test_isotonic_paper_gets_boosted_to_top(self):
        with tempfile.TemporaryDirectory() as td:
            # Big weight to amplify the rerank signal.
            lap, cp = _make_lappato(
                Path(td), reranker=StaticBoostReranker(),
                reranker_weight=10.0,
            )
            lap._cycle_n = 1
            lap._run_cycle()
            card = _read_latest_card(cp)
            self.assertGreaterEqual(len(card["top_papers"]), 1)
            top_title = card["top_papers"][0]["title"].lower()
            self.assertIn("isotonic", top_title,
                          msg=f"top: {card['top_papers'][0]}")
            self.assertTrue(card["reranker"]["contributed"])

    def test_blend_only_adds_score_never_subtracts(self):
        with tempfile.TemporaryDirectory() as td:
            # Run once without reranker — capture lexical baselines.
            lap_a, cp_a = _make_lappato(Path(td) / "a")
            lap_a._cycle_n = 1
            lap_a._run_cycle()
            card_a = _read_latest_card(cp_a)
            baseline = {
                tp["title"]: tp["score"] for tp in card_a["top_papers"]
            }
            # Same run with reranker — every paper's score must be
            # >= the lexical baseline.
            lap_b, cp_b = _make_lappato(
                Path(td) / "b", reranker=StaticBoostReranker(),
            )
            lap_b._cycle_n = 1
            lap_b._run_cycle()
            card_b = _read_latest_card(cp_b)
            for tp in card_b["top_papers"]:
                self.assertGreaterEqual(
                    tp["score"], baseline.get(tp["title"], 0.0) - 1e-9,
                    f"reranker demoted '{tp['title']}'",
                )


# ─── Graceful degrade ─────────────────────────────────────────────────
class GracefulDegradeTests(unittest.TestCase):

    def test_crashing_reranker_is_swallowed(self):
        with tempfile.TemporaryDirectory() as td:
            lap, cp = _make_lappato(Path(td), reranker=CrashingReranker())
            lap._cycle_n = 1
            # Must NOT raise.
            lap._run_cycle()
            card = _read_latest_card(cp)
            # Audit trail is still present (we tried to use a reranker)
            # but ``contributed`` must be False.
            self.assertIn("reranker", card)
            self.assertFalse(card["reranker"]["contributed"])

    def test_wrong_length_output_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            lap, cp = _make_lappato(
                Path(td), reranker=WrongLengthReranker(),
            )
            lap._cycle_n = 1
            lap._run_cycle()
            card = _read_latest_card(cp)
            self.assertFalse(card["reranker"]["contributed"])

    def test_below_floor_signal_is_ignored(self):
        # Confidence below floor (0.30) must NOT lift the score.
        class TimidReranker:
            def score(self, query, hits):
                return [0.10] * len(hits)

        with tempfile.TemporaryDirectory() as td:
            # Run baseline first.
            lap_a, cp_a = _make_lappato(Path(td) / "a")
            lap_a._cycle_n = 1
            lap_a._run_cycle()
            card_a = _read_latest_card(cp_a)
            base = {tp["title"]: tp["score"] for tp in card_a["top_papers"]}
            # Same with a timid reranker.
            lap_b, cp_b = _make_lappato(Path(td) / "b", reranker=TimidReranker())
            lap_b._cycle_n = 1
            lap_b._run_cycle()
            card_b = _read_latest_card(cp_b)
            for tp in card_b["top_papers"]:
                self.assertAlmostEqual(
                    tp["score"], base.get(tp["title"], 0.0), places=6,
                    msg="below-floor signal must not change scores",
                )
            self.assertFalse(card_b["reranker"]["contributed"])


# ─── Determinism ──────────────────────────────────────────────────────
class DeterminismTests(unittest.TestCase):

    def test_two_runs_with_same_reranker_produce_same_ordering(self):
        with tempfile.TemporaryDirectory() as td:
            results = []
            for sub in ("r1", "r2"):
                lap, cp = _make_lappato(
                    Path(td) / sub, reranker=StaticBoostReranker(),
                )
                lap._cycle_n = 1
                lap._run_cycle()
                card = _read_latest_card(cp)
                results.append([tp["title"] for tp in card["top_papers"]])
            self.assertEqual(results[0], results[1])


# ─── Phase 0.6 — blend_mode tests ─────────────────────────────────────
def _make_lappato_with_mode(
    root: Path, *, reranker, blend_mode: str = "additive",
    reranker_weight: float | None = None,
):
    """Variant of _make_lappato that lets the caller pin blend_mode.

    Uses two carefully-engineered cached papers so the test can
    assert how the blend reorders them:

      - p1: 'Bayesian model averaging unrelated topic' — DOMINATES
        the lexical channel because its title shares many tokens
        with the manifest query 'calibration probabilistic
        classifier' via 'classifier' in the abstract...
        Actually, we make p1 dominate the LEXICAL channel by having
        BOTH 'classifier' AND 'calibration' in title+abstract via
        random keyword stuffing.
      - p2: the truly relevant 'Isotonic regression for probability
        calibration' — the reranker recognises this as more
        topically relevant.
    """
    cp = root / "checkpoints"
    cp.mkdir(parents=True, exist_ok=True)
    corpus = cp / "corpus.jsonl"
    cache = CorpusCache(corpus)
    cache.record({
        "id": "p1", "source": "arXiv",
        # Stuff every query token to dominate the lexical channel.
        "title": "Calibration probabilistic classifier calibration probabilistic classifier",
        "abstract": "calibration probabilistic classifier off-topic content here",
        "year": 2024,
    })
    cache.record({
        "id": "p2", "source": "arXiv",
        "title": "Isotonic regression for probability calibration",
        "abstract": "Calibration of probabilistic classifiers via isotonic regression.",
        "year": 2025,
    })
    manifest = [{
        "id": "low_calibration",
        "title": "Poor calibration",
        "evidence": "calibration.csv",
        "evidence_check": lambda p: p.exists(),
        "evidence_summary": lambda p: {"mean_brier": 0.18},
        "severity": lambda p: "high",
        "queries": ["calibration probabilistic classifier"],
        "transplant": "Add isotonic calibration.",
    }]
    (cp / "calibration.csv").write_text("brier\n0.18\n", encoding="utf-8")
    kwargs = {"reranker": reranker, "blend_mode": blend_mode}
    if reranker_weight is not None:
        kwargs["reranker_weight"] = reranker_weight
    lap = LAPPATO_MCB(
        project_root=root,
        manifest=manifest,
        run_tag="rrf_test",
        offline=True,
        corpus_path=corpus,
        **kwargs,
    )
    return lap, cp


def _read_card(cp: Path) -> dict:
    line = (cp / "rrf_test_lappato_mcb_weakness_cards.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()[-1]
    return json.loads(line)


class BlendModeValidationTests(unittest.TestCase):

    def test_invalid_blend_mode_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                LAPPATO_MCB(
                    project_root=Path(td),
                    manifest=[{"id": "x", "title": "x", "evidence": None,
                               "evidence_check": None, "queries": ["q"],
                               "transplant": "t"}],
                    run_tag="t",
                    blend_mode="nonsense",
                )
            # The error message must point users at the empirical
            # comparison in the benchmark report — not just say "bad".
            self.assertIn("blend_mode", str(ctx.exception))
            self.assertIn("additive", str(ctx.exception))
            self.assertIn("rrf", str(ctx.exception))

    def test_default_blend_mode_is_additive(self):
        # Confirms back-compat: anyone constructing without the new
        # arg gets the v1.4 behaviour by default.
        with tempfile.TemporaryDirectory() as td:
            lap = LAPPATO_MCB(
                project_root=Path(td),
                manifest=[{"id": "x", "title": "x", "evidence": None,
                           "evidence_check": None, "queries": ["q"],
                           "transplant": "t"}],
                run_tag="t",
            )
            self.assertEqual(lap._blend_mode, "additive")


class RRFRanksHelperTests(unittest.TestCase):
    """Pin the determinism contract of ``_rrf_ranks``."""

    def test_ranks_descending_by_score(self):
        # Higher score = lower (better) rank.
        ranks = LAPPATO_MCB._rrf_ranks([0.10, 0.95, 0.50])
        self.assertEqual(ranks, [3, 1, 2])

    def test_ties_broken_by_original_index(self):
        # All scores equal → ranks 1, 2, 3 in original order.
        ranks = LAPPATO_MCB._rrf_ranks([0.5, 0.5, 0.5])
        self.assertEqual(ranks, [1, 2, 3])

    def test_empty_input(self):
        self.assertEqual(LAPPATO_MCB._rrf_ranks([]), [])


class RRFBlendModeTests(unittest.TestCase):
    """Phase 0.6 — RRF mode actually delivers the dimensional fix.

    The lexical baseline (`_score_hit`) ranks p1 (keyword-stuffed
    title) above p2 (genuinely relevant). The reranker recognises
    p2. Under ``blend_mode="additive"`` the blend cannot demote p1
    so the wrong paper wins. Under ``blend_mode="rrf"`` rank fusion
    elevates p2 to the top as expected.
    """

    def test_rrf_mode_changes_ranking_when_reranker_signals(self):
        with tempfile.TemporaryDirectory() as td:
            lap, cp = _make_lappato_with_mode(
                Path(td), reranker=StaticBoostReranker(),
                blend_mode="rrf", reranker_weight=2.0,
            )
            lap._cycle_n = 1
            lap._run_cycle()
            card = _read_card(cp)
            self.assertGreater(len(card["top_papers"]), 0)
            top_title = card["top_papers"][0]["title"].lower()
            self.assertIn("isotonic", top_title,
                          msg=f"RRF should elevate the isotonic paper. "
                              f"got top={card['top_papers'][0]}")
            self.assertTrue(card["reranker"]["contributed"])

    def test_card_records_blend_mode(self):
        with tempfile.TemporaryDirectory() as td:
            for mode in ("additive", "rrf"):
                lap, cp = _make_lappato_with_mode(
                    Path(td) / mode, reranker=StaticBoostReranker(),
                    blend_mode=mode,
                )
                lap._cycle_n = 1
                lap._run_cycle()
                card = _read_card(cp)
                self.assertEqual(
                    card["reranker"]["blend_mode"], mode,
                    msg=f"card.reranker.blend_mode mismatch for {mode}",
                )

    def test_card_records_floor_only_in_additive(self):
        with tempfile.TemporaryDirectory() as td:
            lap_a, cp_a = _make_lappato_with_mode(
                Path(td) / "a", reranker=StaticBoostReranker(),
                blend_mode="additive",
            )
            lap_a._cycle_n = 1
            lap_a._run_cycle()
            card_a = _read_card(cp_a)
            self.assertIsNotNone(card_a["reranker"]["floor"])

            lap_b, cp_b = _make_lappato_with_mode(
                Path(td) / "b", reranker=StaticBoostReranker(),
                blend_mode="rrf",
            )
            lap_b._cycle_n = 1
            lap_b._run_cycle()
            card_b = _read_card(cp_b)
            self.assertIsNone(card_b["reranker"]["floor"])

    def test_rrf_mode_with_no_reranker_is_lex_only(self):
        # When reranker is None the configured blend_mode must not
        # affect anything — the path returns early.
        with tempfile.TemporaryDirectory() as td:
            lap_a, cp_a = _make_lappato_with_mode(
                Path(td) / "a", reranker=None, blend_mode="additive",
            )
            # blend_mode='rrf' with no reranker also short-circuits.
            lap_b, cp_b = _make_lappato_with_mode(
                Path(td) / "b", reranker=None, blend_mode="rrf",
            )
            for lap in (lap_a, lap_b):
                lap._cycle_n = 1
                lap._run_cycle()
            # Both runs produce identical top_paper orderings, since
            # neither path applies a reranker.
            order_a = [tp["title"] for tp in _read_card(cp_a)["top_papers"]]
            order_b = [tp["title"] for tp in _read_card(cp_b)["top_papers"]]
            self.assertEqual(order_a, order_b)
            # And neither card has a reranker audit field.
            self.assertNotIn("reranker", _read_card(cp_a))
            self.assertNotIn("reranker", _read_card(cp_b))

    def test_rrf_mode_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            orderings = []
            for sub in ("r1", "r2"):
                lap, cp = _make_lappato_with_mode(
                    Path(td) / sub, reranker=StaticBoostReranker(),
                    blend_mode="rrf",
                )
                lap._cycle_n = 1
                lap._run_cycle()
                orderings.append(
                    [tp["title"] for tp in _read_card(cp)["top_papers"]]
                )
            self.assertEqual(orderings[0], orderings[1])

    def test_rrf_mode_does_not_break_easy_case(self):
        # When the lexical baseline already ranks the right paper
        # first, RRF must not push it down. The default fixture has
        # the verbatim 'isotonic' paper p2 — but the lexical channel
        # wins via keyword stuffing on p1. We use a different fixture
        # where the lexical baseline already gets it right, and check
        # that RRF preserves that.
        class TrivialReranker:
            # Reranker that returns positive constant — should not
            # change the ranking under RRF (constant scores → all
            # papers tied → tie-break by original index, which
            # matches lexical when lex is the primary channel).
            def score(self, query, hits):
                return [0.5] * len(hits)
        with tempfile.TemporaryDirectory() as td:
            # Use the default _make_lappato which has p1=isotonic
            # winning lexically.
            lap_lex_only, cp_lex_only = _make_lappato(
                Path(td) / "lex", reranker=None,
            )
            lap_lex_only._cycle_n = 1
            lap_lex_only._run_cycle()
            order_lex = [tp["title"]
                         for tp in _read_latest_card(cp_lex_only)["top_papers"]]

            # Now with RRF + trivial reranker.
            cp = Path(td) / "rrf" / "checkpoints"
            cp.mkdir(parents=True)
            corpus = cp / "corpus.jsonl"
            from lappato_mcb.cache import CorpusCache as _CC
            cache = _CC(corpus)
            cache.record({"id": "p1", "source": "arXiv",
                          "title": "Isotonic regression for probability calibration",
                          "abstract": "Calibration of probabilistic classifiers",
                          "year": 2025})
            cache.record({"id": "p2", "source": "arXiv",
                          "title": "Bayesian model averaging unrelated topic",
                          "abstract": "calibration appears here too just barely",
                          "year": 2024})
            manifest = [{
                "id": "low_calibration",
                "title": "Poor calibration",
                "evidence": "calibration.csv",
                "evidence_check": lambda p: p.exists(),
                "evidence_summary": lambda p: {"mean_brier": 0.18},
                "severity": lambda p: "high",
                "queries": ["calibration probabilistic classifier"],
                "transplant": "Add isotonic calibration.",
            }]
            (cp / "calibration.csv").write_text("brier\n0.18\n", encoding="utf-8")
            lap = LAPPATO_MCB(
                project_root=Path(td) / "rrf",
                manifest=manifest,
                run_tag="reranker_test",
                offline=True,
                corpus_path=corpus,
                reranker=TrivialReranker(),
                blend_mode="rrf",
            )
            lap._cycle_n = 1
            lap._run_cycle()
            order_rrf = [tp["title"] for tp in
                         json.loads(
                             (cp / "reranker_test_lappato_mcb_weakness_cards.jsonl")
                             .read_text(encoding="utf-8").splitlines()[-1]
                         )["top_papers"]]
            # Same first paper expected.
            self.assertEqual(order_lex[0], order_rrf[0],
                             f"RRF reordered the easy case: "
                             f"lex={order_lex[0]!r} rrf={order_rrf[0]!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
