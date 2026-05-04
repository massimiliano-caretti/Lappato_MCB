"""Stdlib-unittest test suite for LAPPATO_MCB core.

Scope on purpose: only the pure modules that have non-trivial
behaviour worth defending against regression — fingerprint, meta_log,
cache, manifest registry. Network paths (arXiv/OpenAlex HTTP) and
matplotlib rendering are NOT exercised here; they're best validated
end-to-end via examples/demo_*.py.

Run with:
    python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Make the package importable when running from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb import core as core_mod  # noqa: E402
from lappato_mcb import (
    manifests,  # noqa: E402
    reports,  # noqa: E402
)
from lappato_mcb.cache import CorpusCache, merge_jsonl_into_corpus  # noqa: E402
from lappato_mcb.core import (  # noqa: E402
    DEFAULT_SCORE_WEIGHTS,
    GENERAL_SOURCES,
    LAPPATO_MCB,
    SOURCES,
    ScoreWeights,
    _arxiv_keywords,
    _arxiv_query_for,
    _canonical_paper_id,
    _crossref_year,
    _score_hit,
    _source_pipeline_for,
)
from lappato_mcb.fingerprint import (  # noqa: E402
    DEFAULT_JACCARD_TAU,
    TitleDeduper,
    jaccard,
    normalise_title,
    trigrams,
)
from lappato_mcb.meta_log import (  # noqa: E402
    CycleMetrics,  # noqa: E402,F811
    MetaLogWriter,
)


# ─── fingerprint.py ────────────────────────────────────────────────────
class FingerprintTests(unittest.TestCase):

    def test_normalise_strips_punct_and_accents(self):
        self.assertEqual(normalise_title("Café — Wisconsin!"),
                         "cafe wisconsin")

    def test_normalise_empty_input(self):
        self.assertEqual(normalise_title(""), "")
        self.assertEqual(normalise_title(None), "")  # type: ignore[arg-type]

    def test_trigrams_short_input(self):
        self.assertEqual(trigrams("ab"), {"ab "})
        self.assertEqual(trigrams(""), set())

    def test_trigrams_count_matches_length(self):
        # Length-N normalised text -> N-2 trigrams.
        s = "wisconsin"
        self.assertEqual(len(trigrams(s)), len(s) - 2)

    def test_jaccard_identical_is_one(self):
        a = trigrams("specter document level representation")
        self.assertEqual(jaccard(a, a), 1.0)

    def test_jaccard_disjoint_is_zero(self):
        self.assertEqual(jaccard(trigrams("alpha"), trigrams("zzzzz")), 0.0)

    def test_jaccard_handles_empty(self):
        self.assertEqual(jaccard(set(), {"abc"}), 0.0)

    def test_dedup_blocks_near_duplicate_titles(self):
        d = TitleDeduper()
        # Same paper, two punctuation variants — typical arXiv vs OpenAlex.
        d.seen_or_register("SPECTER: Document-level Representation Learning")
        is_dup, sim, match = d.seen_or_register(
            "Specter — Document Level Representation Learning"
        )
        self.assertTrue(is_dup, f"sim={sim} match={match!r}")
        self.assertGreaterEqual(sim, DEFAULT_JACCARD_TAU)

    def test_dedup_keeps_unrelated_titles(self):
        d = TitleDeduper()
        d.seen_or_register("LightGBM a highly efficient gradient boosting tree")
        is_dup, _, _ = d.seen_or_register(
            "Decision curve analysis novel method evaluating prediction models"
        )
        self.assertFalse(is_dup)

    def test_dedup_stats_track_counts(self):
        d = TitleDeduper()
        d.seen_or_register("alpha beta gamma delta")
        d.seen_or_register("alpha beta gamma delta")  # duplicate
        d.seen_or_register("epsilon zeta eta theta")
        s = d.stats
        self.assertEqual(s["unique_kept"], 2)
        self.assertEqual(s["duplicates_blocked"], 1)
        self.assertAlmostEqual(s["dedup_rate"], 1 / 3)


# ─── meta_log.py ───────────────────────────────────────────────────────
class MetaLogTests(unittest.TestCase):

    def test_first_paper_recorded_once(self):
        m = CycleMetrics(cycle=1, ts_started_iso="2026-05-03T20:00:00")
        m.mark_first_paper()
        first = m.first_paper_seconds
        m.mark_first_paper()  # idempotent
        self.assertEqual(m.first_paper_seconds, first)

    def test_finalise_writes_all_columns(self):
        m = CycleMetrics(cycle=2, ts_started_iso="2026-05-03T20:00:00")
        m.n_active_weaknesses = 3
        m.n_queries_executed = 6
        m.n_arxiv_hits_raw = 12
        m.n_papers_kept = 9
        row = m.finalise()
        for k in ("cycle", "wall_seconds", "n_active_weaknesses",
                  "n_queries_executed", "n_arxiv_hits_raw", "n_papers_kept"):
            self.assertIn(k, row)
        self.assertGreaterEqual(row["wall_seconds"], 0.0)

    def test_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "meta.csv"
            w = MetaLogWriter(path)
            w.append({"cycle": 1, "wall_seconds": 0.1,
                      "n_papers_kept": 5})
            w.append({"cycle": 2, "wall_seconds": 0.2,
                      "n_papers_kept": 8})
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertIn("cycle,ts_started_iso,wall_seconds", content)
            self.assertEqual(content.count("\n"), 3)  # header + 2 rows


# ─── cache.py ──────────────────────────────────────────────────────────
class CacheTests(unittest.TestCase):

    def test_record_and_search_token_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "corpus.jsonl"
            c = CorpusCache(path)
            c.record({"id": "a", "title": "deep learning calibration",
                      "abstract": "isotonic regression Brier", "year": 2024,
                      "source": "arXiv"})
            c.record({"id": "b", "title": "transformer recipe",
                      "abstract": "language model attention", "year": 2025,
                      "source": "OpenAlex"})
            hits = c.search("calibration isotonic Brier", k=2)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["id"], "a")

    def test_search_empty_corpus_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            c = CorpusCache(Path(td) / "corpus.jsonl")
            self.assertEqual(c.search("anything", k=3), [])

    def test_merge_jsonl_helper(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.jsonl"
            corpus = Path(td) / "corpus.jsonl"
            src.write_text(
                json.dumps({"id": "x", "title": "t", "abstract": "a"})
                + "\n"
            )
            n = merge_jsonl_into_corpus([src], corpus)
            self.assertEqual(n, 1)
            self.assertEqual(len(CorpusCache(corpus)), 1)


# ─── manifests/__init__.py ─────────────────────────────────────────────
class ManifestRegistryTests(unittest.TestCase):

    def test_three_domains_registered(self):
        self.assertEqual(set(manifests.REGISTRY), {"wdbc", "nlp", "timeseries"})

    def test_each_manifest_has_five_entries(self):
        # Cross-domain comparability is part of the design.
        for name, (m, _) in manifests.REGISTRY.items():
            self.assertEqual(len(m), 5,
                             f"manifest {name!r} has {len(m)} entries, expected 5")

    def test_all_entries_have_required_keys(self):
        required = {"id", "title", "queries", "transplant"}
        for name, (m, _) in manifests.REGISTRY.items():
            for entry in m:
                missing = required - set(entry)
                self.assertFalse(missing,
                                 f"{name}/{entry.get('id','?')} missing {missing}")
                self.assertGreater(len(entry["queries"]), 0)

    def test_unknown_domain_raises(self):
        with self.assertRaises(KeyError):
            manifests.get("does-not-exist")

    def test_naive_baseline_has_one_entry(self):
        m = manifests.naive_baseline_manifest("wdbc")
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0]["id"], "naive_baseline")

    def test_wdbc_evidence_detectors_return_booleans(self):
        from lappato_mcb.manifests import wdbc
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "wdbc_per_fold.csv"
            p.write_text(
                "precision_M,recall_M,precision_B,recall_B\n"
                "0.70,0.70,0.95,0.95\n",
                encoding="utf-8",
            )
            self.assertIs(wdbc._class_imbalance_present(p), True)
            c = Path(td) / "wdbc_calibration.csv"
            c.write_text("brier\n0.18\n", encoding="utf-8")
            self.assertIs(wdbc._calibration_poor(c), True)
            f = Path(td) / "wdbc_feature_importance.csv"
            f.write_text(
                "feature,gain\nradius_worst,10\narea_worst,8\nperimeter_worst,7\n",
                encoding="utf-8",
            )
            self.assertIs(wdbc._features_redundant(f), True)

    def test_nlp_evidence_detectors_return_booleans(self):
        from lappato_mcb.manifests import nlp
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nlp_per_class.csv"
            p.write_text("class,f1\na,0.95\nb,0.70\n", encoding="utf-8")
            self.assertIs(nlp._per_class_f1_spread_high(p), True)
            c = Path(td) / "nlp_confusion_top.csv"
            c.write_text("true,predicted,count,share\na,b,12,0.12\n", encoding="utf-8")
            self.assertIs(nlp._confusion_top_high(c), True)
            t = Path(td) / "nlp_token_importance.csv"
            t.write_text("token,weight\na,1\nhe,1\nof,1\n", encoding="utf-8")
            self.assertIs(nlp._stopword_dominance(t), True)

    def test_timeseries_evidence_detectors_return_booleans(self):
        from lappato_mcb.manifests import timeseries
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "acf.csv"
            a.write_text("lag,acf\n1,0.31\n", encoding="utf-8")
            self.assertIs(timeseries._residual_acf_significant(a), True)
            v = Path(td) / "var.csv"
            v.write_text("fold,variance\n0,1.0\n1,2.0\n", encoding="utf-8")
            self.assertIs(timeseries._heteroscedastic_residuals(v), True)
            h = Path(td) / "horizon.csv"
            h.write_text("horizon,mape\n1,1.0\n2,2.0\n", encoding="utf-8")
            self.assertIs(timeseries._horizon_degradation(h), True)


# ─── reports/__init__.py ───────────────────────────────────────────────
class ReportPackTests(unittest.TestCase):
    """Sanity checks on the pluggable reporting layer."""

    def test_three_packs_registered(self):
        self.assertEqual(set(reports.REGISTRY), {"wdbc", "nlp", "timeseries"})

    def test_each_pack_has_pipeline_csvs_and_parsers(self):
        for name, pack in reports.REGISTRY.items():
            self.assertEqual(set(pack.pipeline_csvs), set(pack.parsers),
                             f"pack {name!r}: csv vs parser keys diverge")
            self.assertGreater(len(pack.pipeline_csvs), 0)

    def test_summarise_emits_expected_keys(self):
        for name, pack in reports.REGISTRY.items():
            empty_parsed = {k: [] for k in pack.parsers}
            summary = pack.summarise(empty_parsed)
            for k in ("n_experiments", "headline_metric_name",
                      "headline_metric_mean", "headline_metric_sd",
                      "headline_metric_n", "secondary_metrics"):
                self.assertIn(k, summary, f"{name} pack: summary missing {k!r}")

    def test_build_figures_returns_dict_of_figures(self):
        from matplotlib.figure import Figure
        for name, pack in reports.REGISTRY.items():
            empty_parsed = {k: [] for k in pack.parsers}
            figs = pack.build_figures(empty_parsed)
            self.assertIsInstance(figs, dict)
            for fid, fig in figs.items():
                self.assertIsInstance(fig, Figure,
                                      f"{name}/{fid} is not a Figure")
                # caption coverage
                self.assertIn(fid, pack.captions,
                              f"{name}/{fid} missing in captions")

    def test_unknown_run_tag_returns_fallback_pack(self):
        pack = reports.get("totally-new-domain")
        self.assertEqual(pack.run_tag, "totally-new-domain")
        self.assertEqual(pack.pipeline_csvs, {})
        # Fallback pack: empty parse + empty figures, but stable schema.
        summary = pack.summarise({})
        self.assertEqual(summary["n_experiments"], 0)


# ─── core.py — Crossref + multi-source dispatch ────────────────────────
class SourceDispatchTests(unittest.TestCase):
    def test_three_sources_registered(self):
        self.assertEqual(SOURCES, ("arXiv", "OpenAlex", "Crossref", "JOSS"))
        self.assertEqual(GENERAL_SOURCES, ("arXiv", "OpenAlex", "Crossref"))

    def test_joss_is_opt_in_per_manifest_entry(self):
        self.assertEqual(
            [s for s, _ in _source_pipeline_for({})],
            ["arXiv", "OpenAlex", "Crossref"],
        )
        self.assertEqual(
            [s for s, _ in _source_pipeline_for({"sources": ["JOSS"]})],
            ["JOSS"],
        )

    def test_crossref_year_extraction(self):
        self.assertEqual(
            _crossref_year({"issued": {"date-parts": [[2025, 4, 1]]}}),
            "2025",
        )
        self.assertEqual(_crossref_year({}), "?")
        self.assertEqual(_crossref_year({"issued": {"date-parts": []}}), "?")

    def test_metrics_add_source_hits_targets_right_attr(self):
        m = CycleMetrics(cycle=1, ts_started_iso="2026-05-03T20:00:00")
        m.add_source_hits("arXiv", 3)
        m.add_source_hits("OpenAlex", 5)
        m.add_source_hits("Crossref", 7)
        m.add_source_hits("JOSS", 11)
        m.add_source_hits("UnknownSource", 99)  # silently ignored
        self.assertEqual(m.n_arxiv_hits_raw, 3)
        self.assertEqual(m.n_openalex_hits_raw, 5)
        self.assertEqual(m.n_crossref_hits_raw, 7)
        self.assertEqual(m.n_joss_hits_raw, 11)

    def test_local_score_prefers_title_and_citations(self):
        low = _score_hit("calibration brier classifier", {
            "title": "unrelated method",
            "abstract": "classifier calibration",
            "year": 2024,
            "cited_by_count": 0,
        })
        high = _score_hit("calibration brier classifier", {
            "title": "Brier calibration for classifiers",
            "abstract": "probability calibration",
            "year": 2026,
            "cited_by_count": 50,
        })
        self.assertGreater(high, low)

    def test_canonical_paper_id_normalises_doi_urls(self):
        self.assertEqual(
            _canonical_paper_id({"id": "https://doi.org/10.21105/joss.03733"}),
            "10.21105/joss.03733",
        )
        self.assertEqual(
            _canonical_paper_id({"id": "doi:10.21105/JOSS.03733/"}),
            "10.21105/joss.03733",
        )

    def test_existing_sidecar_seeds_dedup_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp = root / "checkpoints"
            cp.mkdir()
            sidecar = cp / "test_lappato_mcb_papers.jsonl"
            sidecar.write_text(
                json.dumps({
                    "weakness": "software",
                    "source": "Crossref",
                    "id": "10.21105/joss.03733",
                    "title": "CategoricalTimeSeries.jl",
                }) + "\n",
                encoding="utf-8",
            )
            lappato = LAPPATO_MCB(
                project_root=root,
                manifest=[{
                    "id": "software",
                    "title": "Software",
                    "evidence": None,
                    "evidence_check": None,
                    "queries": ["time series software"],
                    "transplant": "try software",
                }],
                run_tag="test",
            )
            m = CycleMetrics(cycle=1, ts_started_iso="2026-05-04T09:00:00")
            dup = lappato._consider_hit(
                "software",
                "JOSS",
                "time series software",
                {
                    "id": "https://doi.org/10.21105/joss.03733",
                    "title": "CategoricalTimeSeries.jl: A toolbox",
                },
                m,
            )
            self.assertIsNone(dup)
            self.assertEqual(m.n_id_dedup_blocked, 1)


class WeaknessCardTests(unittest.TestCase):
    def test_offline_cycle_writes_structured_weakness_card(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp = root / "checkpoints"
            cp.mkdir()
            (cp / "lappato_context.json").write_text(
                json.dumps({"domain": "clinical", "model_family": "LightGBM"})
            )
            corpus = cp / "corpus.jsonl"
            CorpusCache(corpus).record({
                "id": "paper-1",
                "title": "Brier calibration for LightGBM clinical classifiers",
                "abstract": "isotonic calibration improves clinical probability",
                "year": 2025,
                "source": "arXiv",
                "url": "https://example.test/paper-1",
            })
            evidence = cp / "calibration.csv"
            evidence.write_text("brier\n0.18\n", encoding="utf-8")
            manifest = [{
                "id": "low_calibration",
                "title": "Poor calibration",
                "evidence": "calibration.csv",
                "evidence_check": lambda p: p.exists(),
                "evidence_summary": lambda p: {"mean_brier": 0.18},
                "severity": lambda p: "high",
                "queries": ["Brier calibration {model_family} {domain}"],
                "transplant": "Add isotonic calibration.",
            }]
            lappato = LAPPATO_MCB(
                project_root=root,
                manifest=manifest,
                run_tag="test",
                offline=True,
                corpus_path=corpus,
            )
            lappato._cycle_n = 1
            lappato._run_cycle()
            cards = [
                json.loads(line)
                for line in (cp / "test_lappato_mcb_weakness_cards.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(cards), 1)
            card = cards[0]
            self.assertEqual(card["severity"], "high")
            self.assertEqual(card["evidence_summary"]["mean_brier"], 0.18)
            self.assertEqual(card["queries"], ["Brier calibration LightGBM clinical"])
            self.assertEqual(card["n_papers_kept"], 1)
            self.assertGreater(card["top_papers"][0]["score"], 0)


# ─── core.py — adaptive arXiv keyword retry (W1) ───────────────────────
class AdaptiveArxivKeywordTests(unittest.TestCase):

    def test_keyword_extraction_drops_stopwords_and_short_tokens(self):
        kws = _arxiv_keywords("the GARCH heteroscedasticity time series 2025")
        self.assertNotIn("the", kws)
        self.assertNotIn("2025", kws)
        self.assertEqual(kws[0], "GARCH")
        self.assertIn("heteroscedasticity", kws)

    def test_query_for_caps_at_max_keywords(self):
        # 6 distinctive tokens; cap to 3 — order preserved.
        q = _arxiv_query_for("alpha bravo charlie delta echo foxtrot",
                             max_keywords=3)
        self.assertEqual(q, "abs:alpha AND abs:bravo AND abs:charlie")

    def test_arxiv_search_falls_back_to_wider_query_when_empty(self):
        """When the most-restrictive query returns 0, retry with one
        keyword fewer until the minimum cap is reached."""
        original = core_mod._arxiv_fetch_one
        sleep_calls: list[float] = []
        captured: list[str] = []

        def fake_fetch(query_string, max_results):
            captured.append(query_string)
            # Return hits only when the query has ≤ 3 AND-ed terms.
            if query_string.count("AND") <= 1:  # 2 keywords -> 1 AND
                return [{"id": "x", "title": "ok",
                         "abstract": "", "year": "2025", "url": "u"}]
            return []

        try:
            core_mod._arxiv_fetch_one = fake_fetch
            core_mod.time.sleep = lambda s: sleep_calls.append(s)
            results = core_mod._arxiv_search("alpha bravo charlie delta echo")
        finally:
            core_mod._arxiv_fetch_one = original
            core_mod.time.sleep = time.sleep
        self.assertEqual(len(results), 1)
        # Tries 5 → 4 → 3 → 2 keywords; 4 attempts, 3 sleeps between them
        # but only the first three fall back. The exact attempt count is
        # the number of failed widenings.
        self.assertGreaterEqual(len(captured), 2)
        # Polite-use spacing was honoured between retries.
        self.assertTrue(all(s == core_mod._ARXIV_DELAY_SEC for s in sleep_calls))


# ─── core.py — HTTP retry + per-source error logging (W4) ──────────────
class FetchRetryTests(unittest.TestCase):

    def _make_lappato(self, root: Path, **kwargs):
        manifest = [{
            "id": "w",
            "title": "w",
            "evidence": None,
            "evidence_check": None,
            "queries": ["q"],
            "transplant": "t",
        }]
        return LAPPATO_MCB(
            project_root=root, manifest=manifest, run_tag="t", **kwargs
        )

    def test_fetch_retries_then_records_error(self):
        with tempfile.TemporaryDirectory() as td:
            lap = self._make_lappato(Path(td), http_max_retries=2)
            calls = {"n": 0}
            sleep_calls: list[float] = []

            def always_fails(_q):
                calls["n"] += 1
                raise RuntimeError("upstream blew up")

            core_mod._SOURCE_FN["arXiv"] = always_fails
            core_mod.time.sleep = lambda s: sleep_calls.append(s)
            try:
                m = CycleMetrics(cycle=1, ts_started_iso="t")
                hits = lap._fetch("arXiv", "q", m)
            finally:
                # Restore real arXiv fn for other tests.
                from lappato_mcb.core import _arxiv_search
                core_mod._SOURCE_FN["arXiv"] = _arxiv_search
                core_mod.time.sleep = time.sleep
            self.assertEqual(hits, [])
            self.assertEqual(calls["n"], 3)             # 1 + 2 retries
            self.assertEqual(m.n_arxiv_errors, 1)
            # Backoff is exponential 1s, 2s — final attempt does not sleep.
            self.assertEqual(sleep_calls, [1.0, 2.0])

    def test_fetch_succeeds_after_transient_failure(self):
        with tempfile.TemporaryDirectory() as td:
            lap = self._make_lappato(Path(td), http_max_retries=2)
            calls = {"n": 0}

            def flaky(_q):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise TimeoutError("transient")
                return [{"id": "x", "title": "ok"}]

            core_mod._SOURCE_FN["arXiv"] = flaky
            core_mod.time.sleep = lambda _s: None
            try:
                m = CycleMetrics(cycle=1, ts_started_iso="t")
                hits = lap._fetch("arXiv", "q", m)
            finally:
                from lappato_mcb.core import _arxiv_search
                core_mod._SOURCE_FN["arXiv"] = _arxiv_search
                core_mod.time.sleep = time.sleep
            self.assertEqual(len(hits), 1)
            self.assertEqual(m.n_arxiv_errors, 0)


# ─── core.py — configurable score weights (W2) ─────────────────────────
class ScoreWeightsTests(unittest.TestCase):

    def test_default_weights_match_legacy_constants(self):
        self.assertEqual(DEFAULT_SCORE_WEIGHTS.title_overlap, 4.0)
        self.assertEqual(DEFAULT_SCORE_WEIGHTS.abstract_overlap, 1.5)
        self.assertEqual(DEFAULT_SCORE_WEIGHTS.recency_per_year, 0.25)
        self.assertEqual(DEFAULT_SCORE_WEIGHTS.recency_max_years, 4)
        self.assertEqual(DEFAULT_SCORE_WEIGHTS.citation_log_weight, 0.15)

    def test_custom_weights_change_score(self):
        hit = {
            "title": "calibration of clinical classifiers",
            "abstract": "isotonic", "year": 2025, "cited_by_count": 0,
        }
        baseline = _score_hit("calibration clinical", hit)
        boosted = _score_hit(
            "calibration clinical", hit,
            ScoreWeights(title_overlap=10.0),
        )
        self.assertGreater(boosted, baseline)

    def test_lappato_uses_constructor_weights_for_kept_papers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp = root / "checkpoints"
            cp.mkdir()
            corpus = cp / "corpus.jsonl"
            from lappato_mcb.cache import CorpusCache as _CC
            _CC(corpus).record({
                "id": "p1", "source": "arXiv",
                "title": "calibration calibration",
                "abstract": "", "year": 2025,
            })
            manifest = [{
                "id": "w", "title": "w",
                "evidence": None, "evidence_check": None,
                "queries": ["calibration"],
                "transplant": "t",
            }]
            lap = LAPPATO_MCB(
                project_root=root, manifest=manifest, run_tag="t",
                offline=True, corpus_path=corpus,
                score_weights=ScoreWeights(title_overlap=100.0),
            )
            lap._cycle_n = 1
            lap._run_cycle()
            # Read the kept paper sidecar — score must reflect the
            # custom weight.
            line = (cp / "t_lappato_mcb_papers.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            row = json.loads(line)
            # Score not persisted in the sidecar; assert by reading
            # the latest card instead.
            card = json.loads(
                (cp / "t_lappato_mcb_weakness_cards.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )
            self.assertGreater(card["top_papers"][0]["score"], 90.0)
            self.assertEqual(row["id"], "p1")


# ─── manifests/* — threshold overrides (W3) ────────────────────────────
class ThresholdOverrideTests(unittest.TestCase):

    def test_wdbc_thresholds_are_overridable(self):
        from lappato_mcb.manifests import wdbc
        original = wdbc.THRESHOLDS["brier_warning"]
        try:
            wdbc.override_thresholds({"brier_warning": 0.50})
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "wdbc_calibration.csv"
                p.write_text("brier\n0.18\n", encoding="utf-8")
                # 0.18 < 0.50 -> calibration is no longer "poor" under
                # the relaxed threshold.
                self.assertFalse(wdbc._calibration_poor(p))
        finally:
            wdbc.override_thresholds({"brier_warning": original})

    def test_nlp_thresholds_round_trip(self):
        from lappato_mcb.manifests import nlp
        original = nlp.THRESHOLDS["f1_spread_warning"]
        nlp.override_thresholds({"f1_spread_warning": 0.99})
        self.assertEqual(nlp.THRESHOLDS["f1_spread_warning"], 0.99)
        nlp.override_thresholds({"f1_spread_warning": original})
        self.assertEqual(nlp.THRESHOLDS["f1_spread_warning"], original)

    def test_timeseries_thresholds_documented_and_present(self):
        from lappato_mcb.manifests import timeseries
        for k in ("acf_warning", "acf_high",
                  "var_ratio_warning", "var_ratio_high",
                  "horizon_mape_ratio_warning", "horizon_mape_ratio_high"):
            self.assertIn(k, timeseries.DEFAULT_THRESHOLDS, k)


# ─── examples/measure_recall — topic match semantics (W5) ──────────────
class RecallHarnessTests(unittest.TestCase):

    def setUp(self):
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "examples"))
        # Import lazily so the test file runs even when examples/ is
        # absent (e.g. minimal install).
        import importlib
        self.harness = importlib.import_module("measure_recall")

    def test_topic_found_requires_all_keywords(self):
        topic = {"must_match_keywords": ["isotonic", "calibration"]}
        self.assertTrue(
            self.harness._topic_found(
                topic, ["isotonic regression for probability calibration"]
            )
        )
        self.assertFalse(
            self.harness._topic_found(
                topic, ["temperature scaling for calibration"]
            )
        )

    def test_evaluate_domain_skips_inactive_weaknesses(self):
        # Synthetic papers covering only one weakness.
        papers = [{
            "weakness": "low_calibration",
            "title": "Isotonic regression for probability calibration",
            "abstract": "Brier reliability decomposition",
        }]
        result = self.harness.evaluate_domain("wdbc", papers)
        # Active weakness recall is 1.0 (all topics matched).
        active = [w for w in result["per_weakness"] if w["active"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["weakness"], "low_calibration")
        self.assertGreaterEqual(active[0]["recall"], 0.66)
        # Macro recall is computed only over active weaknesses.
        self.assertGreaterEqual(result["macro_recall"], 0.66)


if __name__ == "__main__":
    unittest.main(verbosity=2)
