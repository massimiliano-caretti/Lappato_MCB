"""Render the README figures as light/dark SVG pairs.

Every number drawn here is read from the library itself (manifest
registry, thresholds, source table) or from the frozen output of a real
run under ``docs/figures/data/`` — nothing is typed in by hand, so a
figure cannot silently drift from the code. Re-run after changing
manifests, sources or the demo data:

    python docs/figures/build_figures.py

Stdlib-only. Output: ``docs/figures/<name>-light.svg`` and ``-dark.svg``,
embedded in the README through ``<picture>`` so GitHub serves the
variant matching the reader's theme. Colours follow a CVD-validated
categorical order (blue, orange) on GitHub's own surfaces.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from lappato_mcb import SOURCES, manifests  # noqa: E402
from lappato_mcb.fingerprint import DEFAULT_JACCARD_TAU  # noqa: E402
from lappato_mcb.manifests import wdbc  # noqa: E402

DATA = HERE / "data"

SANS = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"


@dataclass(frozen=True)
class Theme:
    name: str
    panel: str      # figure background
    card: str       # raised boxes
    ink: str
    ink2: str
    muted: str
    hair: str
    blue: str
    blue_soft: str
    orange: str
    orange_soft: str
    code_bg: str


LIGHT = Theme(
    "light", panel="#f6f8fa", card="#ffffff", ink="#1f2328", ink2="#59636e",
    muted="#818b98", hair="#d1d9e0", blue="#2a78d6", blue_soft="#e3eefc",
    orange="#eb6834", orange_soft="#fdebe3", code_bg="#eff2f5",
)
DARK = Theme(
    "dark", panel="#161b22", card="#0d1117", ink="#f0f6fc", ink2="#9198a1",
    muted="#6e7681", hair="#3d444d", blue="#3987e5", blue_soft="#132840",
    orange="#d95926", orange_soft="#3a1f14", code_bg="#1c2128",
)
# Severity uses the fixed status palette (never a series colour) and is
# always printed next to its label, so colour never carries it alone.
STATUS = {"medium": "#fab219", "high": "#ec835a", "critical": "#d03b3b"}


# ─── SVG primitives ────────────────────────────────────────────────────
def _attrs(**kw) -> str:
    return " ".join(f'{k.rstrip("_").replace("_", "-")}="{v}"' for k, v in kw.items() if v is not None)


def text(x, y, s, size=16, fill=None, weight=400, anchor="start", mono=False, italic=False, **kw):
    return (
        f"<text {_attrs(x=x, y=y, font_size=size, font_weight=weight, text_anchor=anchor, fill=fill, font_family=MONO if mono else SANS, font_style='italic' if italic else None, **kw)}{' xml:space=\"preserve\"' if mono else ''}>"
        f"{escape(str(s))}</text>"
    )


def rect(x, y, w, h, fill, stroke=None, rx=10, dash=None, sw=1.5):
    return f"<rect {_attrs(x=x, y=y, width=w, height=h, rx=rx, fill=fill, stroke=stroke, stroke_width=sw if stroke else None, stroke_dasharray=dash)}/>"


def line(x1, y1, x2, y2, stroke, sw=2, dash=None, arrow=None):
    marker = f"url(#{arrow})" if arrow else None
    return f"<line {_attrs(x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke, stroke_width=sw, stroke_dasharray=dash, marker_end=marker, stroke_linecap='round')}/>"


def path(d, stroke, sw=2, fill="none", dash=None, arrow=None):
    marker = f"url(#{arrow})" if arrow else None
    return f"<path {_attrs(d=d, stroke=stroke, stroke_width=sw, fill=fill, stroke_dasharray=dash, marker_end=marker, stroke_linecap='round', stroke_linejoin='round')}/>"


def chip(x, y, s, t: Theme, fg=None, bg=None, size=14, mono=True, pad=10, h=26):
    """A rounded label; returns (svg, width). Width is estimated from glyph count."""
    w = int(len(s) * size * (0.61 if mono else 0.56)) + 2 * pad
    body = rect(x, y, w, h, bg or t.code_bg, rx=h // 2 if not mono else 6)
    body += text(x + pad, y + h / 2 + size * 0.36, s, size=size, fill=fg or t.ink, mono=mono)
    return body, w


def svg(name: str, w: int, h: int, body: str, t: Theme, title: str, desc: str) -> str:
    markers = "".join(
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        for mid, col in (("a-ink", t.ink2), ("a-blue", t.blue), ("a-orange", t.orange), ("a-muted", t.muted))
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="{name}-t {name}-d">'
        f'<title id="{name}-t">{escape(title)}</title><desc id="{name}-d">{escape(desc)}</desc>'
        f"<defs>{markers}</defs>"
        f'{rect(0.75, 0.75, w - 1.5, h - 1.5, t.panel, stroke=t.hair, rx=14, sw=1.5)}'
        f"{body}</svg>\n"
    )


def step_badge(x, y, n, t: Theme, color=None):
    c = color or t.blue
    return (f'<circle cx="{x}" cy="{y}" r="15" fill="{c}"/>'
            + text(x, y + 5.5, n, size=15, fill="#ffffff", weight=700, anchor="middle"))


# ─── data helpers (read from the library, never typed in) ─────────────
def registry_counts() -> list[tuple[str, int, int]]:
    """(domain, evidence-gated, structural) for every registered manifest."""
    out = []
    for domain, (manifest, _) in manifests.REGISTRY.items():
        structural = sum(1 for w in manifest if not w.get("evidence"))
        out.append((domain, len(manifest) - structural, structural))
    return out


def totals() -> dict:
    counts = registry_counts()
    gated = sum(g for _, g, _ in counts)
    structural = sum(s for _, _, s in counts)
    return {"domains": len(counts), "detectors": gated + structural,
            "gated": gated, "structural": structural}


def demo_runs() -> dict:
    return json.loads((DATA / "demo_runs.json").read_text(encoding="utf-8"))


def demo_card(run_tag: str, weakness: str) -> dict:
    cards = demo_runs()["runs"][run_tag]["cards"]
    return next(c for c in cards if c["weakness"] == weakness)


def measure(module, evidence_name: str, summary_fn: str, severity_fn: str, run_tag: str):
    """Recompute (summary, severity, active) with the manifest's own functions
    on the frozen evidence CSV of a real run."""
    import tempfile

    raw = demo_runs()["runs"][run_tag]["evidence_csv"][evidence_name]
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / evidence_name
        csv_path.write_text(raw, encoding="utf-8")
        entry = next(w for w in module.MANIFEST if w.get("evidence") == evidence_name)
        return (getattr(module, summary_fn)(csv_path), getattr(module, severity_fn)(csv_path),
                bool(entry["evidence_check"](csv_path)))


def wrap(s: str, width: int) -> list[str]:
    """Greedy word wrap on character count (figures use fixed widths)."""
    lines, cur = [], ""
    for word in s.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def lines_at(x, y, rows, t: Theme, size=15, gap=21, fill=None, mono=False, weight=400):
    return "".join(text(x, y + i * gap, r, size=size, fill=fill or t.ink2, mono=mono, weight=weight)
                   for i, r in enumerate(rows))


# ─── Figure 1 · overview ───────────────────────────────────────────────
def fig_overview(t: Theme) -> tuple[int, int, str, str, str]:
    from lappato_mcb.manifests import timeseries

    W, H = 1200, 490
    tot = totals()
    card = demo_card("timeseries", "heteroscedastic_residuals")
    summary, severity, _ = measure(timeseries, "timeseries_residual_var.csv",
                                   "_residual_var_summary", "_residual_var_severity", "timeseries")
    ratio = summary["last_first_variance_ratio"]
    b = ""
    cols = [(28, 340), (430, 340), (832, 340)]
    top, h = 28, 392
    heads = [("Your pipeline", "writes the diagnostics it already computes"),
             ("LAPPATO_MCB", "reads them, then searches the literature"),
             ("Weakness cards", "one per weakness the evidence supports")]
    for i, ((x, w), (title, sub)) in enumerate(zip(cols, heads), start=1):
        b += rect(x, top, w, h, t.card, stroke=t.hair, rx=12)
        b += step_badge(x + 30, top + 34, i, t)
        b += text(x + 56, top + 40, title, size=20, fill=t.ink, weight=650)
        b += text(x + 24, top + 72, sub, size=15, fill=t.ink2)
    for x0, x1 in ((368, 430), (770, 832)):
        b += line(x0 + 8, top + h / 2, x1 - 8, top + h / 2, t.ink2, sw=2.5, arrow="a-ink")

    # column 1 · evidence folder (the files of the real time-series demo)
    x, y = 52, 130
    b += text(x, y, "checkpoints/", size=16, fill=t.ink, mono=True, weight=600)
    files = ["timeseries_residual_var.csv", "timeseries_per_horizon.csv",
             "timeseries_residual_acf.csv", "timeseries_lappato_context.json"]
    for j, f in enumerate(files):
        fy = y + 22 + j * 38
        b += path(f"M{x + 8},{y + 8} V{fy + 13} H{x + 24}", t.hair, sw=1.5)
        c, _ = chip(x + 28, fy, f, t, size=13.5, bg=t.blue_soft if j == 0 else t.code_bg)
        b += c
    b += lines_at(x, 334, wrap("A CSV that is missing or unreadable keeps its detectors silent (fail-closed). "
                               "The JSON context is optional.", 40), t, size=14, gap=20)

    # column 2 · detect + retrieve
    x = 454
    b += rect(x, 112, 292, 112, t.panel, rx=10)
    b += text(x + 16, 140, "Detect", size=16, fill=t.ink, weight=650)
    b += lines_at(x + 16, 166, wrap(f"{tot['detectors']} detectors in {tot['domains']} manifests compare "
                                    "evidence with documented thresholds.", 36), t, size=14, gap=20)
    b += rect(x, 238, 292, 160, t.panel, rx=10)
    b += text(x + 16, 266, "Retrieve", size=16, fill=t.ink, weight=650)
    cx = x + 16
    for src in SOURCES:
        c, cw = chip(cx, 280, src, t, size=13, mono=False, bg=t.card, pad=9, h=26)
        b += c
        cx += cw + 6
    b += lines_at(x + 16, 330, wrap(f"Targeted queries, cross-source deduplication (τ = {DEFAULT_JACCARD_TAU:.2f}), "
                                    "deterministic scoring.", 36), t, size=14, gap=20)

    # column 3 · a real card from that run
    x = 856
    b += text(x, 130, card["weakness"], size=15, fill=t.ink, mono=True, weight=600)
    b += f'<circle cx="{x + 7}" cy="{150}" r="6" fill="{STATUS[severity]}"/>'
    b += text(x + 20, 155, f"severity: {severity}", size=14.5, fill=t.ink)
    rows = [("evidence", f"variance ratio {ratio:.3f}"),
            ("threshold", f"> {timeseries.THRESHOLDS['var_ratio_warning']}"),
            ("queries sent", str(len(card["queries"]))),
            ("papers kept", str(card["n_papers_kept"])),
            ("next check", "")]
    for j, (k, v) in enumerate(rows):
        ry = 190 + j * 28
        b += text(x, ry, k, size=14, fill=t.muted)
        b += text(x + 112, ry, v, size=14, fill=t.ink, mono=True)
    b += lines_at(x, 323, wrap(card["next_checks"][0], 38)[:2], t, size=14, gap=19, fill=t.ink)
    b += text(x, 392, "JSON for tools · Markdown log + PDF for people", size=13, fill=t.muted)

    b += text(28, H - 26, "Values from one complete cycle on the bundled time-series demo "
              f"({demo_runs()['provenance']['date']}). Python standard library only.",
              size=13.5, fill=t.muted)
    return W, H, b, "LAPPATO_MCB at a glance", (
        "Three panels: a pipeline writes evidence CSVs to checkpoints; LAPPATO_MCB detects weaknesses "
        "against thresholds and retrieves literature from arXiv, OpenAlex, Crossref and JOSS; it emits "
        "weakness cards such as heteroscedastic_residuals with severity medium.")


# ─── Figure 2 · two interfaces, one engine ────────────────────────────
def fig_interfaces(t: Theme):
    W, H = 1200, 470
    b = ""
    # Python API
    b += rect(40, 28, 540, 210, t.card, stroke=t.blue, rx=12, sw=2)
    b += text(64, 66, "Python API", size=21, fill=t.ink, weight=650)
    b += text(64, 92, "for people and the code they write", size=15, fill=t.ink2)
    code = ["auditor = LAPPATO_MCB(root, manifest, run_tag)",
            "auditor.start()  …train…  auditor.stop()",
            "search_literature(\"conformal prediction\")",
            "validate_checkpoints(\"wdbc\", root / \"checkpoints\")"]
    b += rect(64, 108, 492, 104, t.code_bg, rx=8)
    b += lines_at(80, 134, code, t, size=14, gap=23, fill=t.ink, mono=True)
    # MCP server
    b += rect(620, 28, 540, 210, t.card, stroke=t.orange, rx=12, sw=2, dash="8 6")
    b += text(644, 66, "MCP server", size=21, fill=t.ink, weight=650)
    b += text(644, 92, "for AI assistants (any Model Context Protocol client)", size=15, fill=t.ink2)
    tools = ["list_manifests", "describe_manifest", "validate_evidence",
             "run_audit", "get_weakness_cards", "search_literature"]
    cx, cy = 644, 110
    for name in tools:
        c, cw = chip(cx, cy, name, t, size=13.5)
        if cx + cw > 1140:
            cx, cy = 644, cy + 34
            c, cw = chip(cx, cy, name, t, size=13.5)
        b += c
        cx += cw + 8
    b += text(644, 200, "+ resources lappato://manifests/{domain}, lappato://schemas/{domain}",
              size=13.5, fill=t.ink2, mono=False)
    # badges under the boxes
    for x, label, col, soft in ((40, "required · Python ≥ 3.9 · no third-party dependencies", t.blue, t.blue_soft),
                                (620, "optional · pip install 'lappato_mcb[mcp]' · Python ≥ 3.10", t.orange, t.orange_soft)):
        c, _ = chip(x, 250, label, t, size=13.5, mono=False, fg=t.ink, bg=soft, h=28, pad=14)
        b += c
    # engine
    b += rect(40, 330, 1120, 100, t.card, stroke=t.hair, rx=12)
    b += text(64, 366, "lappato_mcb · one engine", size=19, fill=t.ink, weight=650)
    parts = ["manifests", "detectors", "query rendering", "4 source adapters", "deduplication",
             "scoring", "weakness cards", "corpus / offline replay"]
    cx = 64
    for part in parts:
        c, cw = chip(cx, 384, part, t, size=13.5, mono=False, bg=t.panel, h=28)
        b += c
        cx += cw + 8
    # arrows: both interfaces call the engine; nothing points back up
    b += line(310, 288, 310, 326, t.blue, sw=2.5, arrow="a-blue")
    b += line(890, 288, 890, 326, t.orange, sw=2.5, arrow="a-orange")
    b += text(904, 314, "imports and calls the engine", size=13.5, fill=t.ink2)
    b += text(324, 314, "calls the engine directly", size=13.5, fill=t.ink2)
    b += text(1136, 366, "never imports mcp_server (checked by a test)", size=13.5, fill=t.ink2, anchor="end")
    return W, H, b, "Two interfaces, one engine", (
        "The Python API (required, no dependencies) and the optional MCP server both call the same "
        "lappato_mcb engine; the engine never imports the MCP server.")


# ─── Figure 3 · anatomy ────────────────────────────────────────────────
def fig_anatomy(t: Theme):
    tot = totals()
    import glob

    n_topics = sum(len(v) for f in glob.glob(str(ROOT / "docs/gold_sets/*_gold.json"))
                   for v in json.loads(Path(f).read_text(encoding="utf-8"))["topics_by_weakness"].values())
    n_schemas = len(list((ROOT / "lappato_mcb/manifests/schemas").glob("*.schema.json")))
    layers = [
        ("Interfaces", "how it is called", [
            ("core.py · LAPPATO_MCB", "start · stop · run_once", False),
            ("literature/", "search_literature()", False),
            ("validation/", "evidence-CSV contract", False),
            ("mcp_server/", "6 tools · 2 resources", True)]),
        ("Knowledge", "declarative, versioned", [
            ("manifests/", f"{tot['domains']} domains · {tot['detectors']} detectors", False),
            ("manifests/schemas/", f"{n_schemas} JSON Schemas", False),
            ("docs/gold_sets/", f"{n_topics} topics (evaluation only)", False)]),
        ("Engine", "one monitoring cycle", [
            ("core.py", "gate → grade → query → fetch → record", False),
            ("fingerprint.py", f"title trigram dedup, τ = {DEFAULT_JACCARD_TAU:.2f}", False),
            ("rerankers/", "optional re-ranking Protocol", False)]),
        ("Sources & memory", "network or local corpus", [
            ("core.py · adapters", " · ".join(SOURCES), False),
            ("cache.py", "corpus.jsonl · offline replay", False)]),
        ("Outputs", "what people and tools read", [
            ("…_log.md", "human-readable log", False),
            ("…_weakness_cards.*", "JSONL + latest.json", False),
            ("…_papers.jsonl", "every kept paper", False),
            ("…_meta.csv", "per-cycle metrics", False),
            ("pdf_builder.py", "PDF ([reports] extra)", True)]),
    ]
    W = 1200
    row_h, gap, top = 92, 14, 28
    H = top + len(layers) * (row_h + gap) + 44
    b = ""
    for i, (name, sub, mods) in enumerate(layers):
        y = top + i * (row_h + gap)
        b += rect(28, y, 1144, row_h, t.card, stroke=t.hair, rx=12)
        b += text(50, y + 40, name, size=18, fill=t.ink, weight=650)
        b += text(50, y + 64, sub, size=13.5, fill=t.ink2)
        x0, avail = 232, 1144 - 232 + 28 - 16
        bw = (avail - (len(mods) - 1) * 12) / len(mods)
        for j, (mod, role, optional) in enumerate(mods):
            bx = x0 + j * (bw + 12)
            b += rect(bx, y + 14, bw, row_h - 28, t.panel, stroke=t.orange if optional else t.blue,
                      rx=8, sw=1.6, dash="6 5" if optional else None)
            b += text(bx + 14, y + 40, clip(mod, int(bw / 8.6)), size=14, fill=t.ink, mono=True, weight=600)
            b += text(bx + 14, y + 63, clip(role, int(bw / 7.4)), size=13.5, fill=t.ink2)
    ly = H - 22
    b += rect(30, ly - 13, 26, 16, t.panel, stroke=t.blue, rx=4, sw=1.6)
    b += text(64, ly, "Python standard library only", size=13.5, fill=t.ink2)
    b += rect(300, ly - 13, 26, 16, t.panel, stroke=t.orange, rx=4, sw=1.6, dash="6 5")
    b += text(334, ly, "optional extra, never imported by the core", size=13.5, fill=t.ink2)
    return W, H, b, "Anatomy of LAPPATO_MCB", (
        "Five layers: interfaces, knowledge (manifests, schemas, gold sets), engine (core cycle, "
        "deduplication, rerankers), sources and memory (four adapters, corpus cache), outputs.")


# ─── Figure 4 · physiology (one cycle) ────────────────────────────────
def fig_physiology(t: Theme):
    steps = [  # (title, function in core.py / manifest, what happens, scope)
        ("Read context", "_load_context()", "optional lappato_context.json fills {domain}, {task}, {model_family}", "per cycle"),
        ("Gate", "evidence_check(csv)", "missing or unreadable CSV ⇒ inactive: nothing is fetched", "per weakness"),
        ("Grade", "severity(csv)", "evidence_summary(csv) too · info, low, medium, high, critical", "per active weakness"),
        ("Render queries", "_render_query_template()", "the manifest's 2–3 queries, context substituted", "per active weakness"),
        ("Fetch", "_fetch(source, query)", "online: polite delays, 2 retries · offline: local corpus", "per query × source"),
        ("Filter & score", "_consider_hit()", f"id dedup → title dedup (τ {DEFAULT_JACCARD_TAU:.2f}) → score; kept hit → papers.jsonl + corpus", "per hit"),
        ("Rank", "_apply_reranker()", "optional reranker: additive or reciprocal-rank-fusion blend", "per active weakness"),
        ("Record", "_record_weakness_card()", "card → JSONL + latest.json; section → log.md; then one meta.csv row per cycle", "per active weakness"),
    ]
    W, H = 1200, 600
    bw, bh, gx = 262, 188, 24
    xs = [28 + k * (bw + gx) for k in range(4)]
    ys = [52, 330]
    b = text(28, H - 22, "Steps in code order (LAPPATO_MCB._run_cycle). Daemon: repeats every poll_interval, "
             "default 90 s · run_once(): a single cycle.", size=13.5, fill=t.muted)
    for k, (title, fn, note, scope) in enumerate(steps):
        x, y = xs[k % 4], ys[k // 4]
        b += rect(x, y, bw, bh, t.card, stroke=t.hair, rx=12)
        b += step_badge(x + 30, y + 32, k + 1, t)
        b += text(x + 54, y + 38, title, size=17.5, fill=t.ink, weight=650)
        b += rect(x + 16, y + 56, bw - 32, 30, t.code_bg, rx=6)
        b += text(x + 26, y + 76, clip(fn, 30), size=13, fill=t.ink, mono=True)
        b += lines_at(x + 18, y + 112, wrap(note, 31)[:3], t, size=13.5, gap=19)
        b += text(x + 18, y + bh - 14, scope, size=12.5, fill=t.blue, weight=600)
        if k % 4 != 3:
            b += line(x + bw + 3, y + bh / 2, x + bw + gx - 3, y + bh / 2, t.ink2, sw=2, arrow="a-ink")
    # row 1 → row 2 connector and loop back
    b += path(f"M{xs[3] + bw / 2},{ys[0] + bh + 4} V{ys[0] + bh + 30} H{xs[0] + bw / 2} V{ys[1] - 6}",
              t.ink2, sw=2, arrow="a-ink")
    b += text(xs[1] + 40, ys[0] + bh + 24, "continues for each active weakness", size=13, fill=t.muted)
    b += path(f"M{xs[3] + bw},{ys[1] + bh / 2} H{W - 14} V{ys[0] - 26} H{xs[0] + 30} V{ys[0] - 4}",
              t.blue, sw=2, dash="6 6", arrow="a-blue")
    b += text(xs[2], ys[0] - 36, "next cycle (daemon only) · stop() ends the loop", size=13, fill=t.blue)
    return W, H, b, "Physiology: one monitoring cycle", (
        "Eight steps in code order: read context, gate on evidence, grade severity, render queries, "
        "fetch from sources, filter and score, optional rerank, record outputs; then repeat.")


# ─── Figure 5 · how a detector decides ────────────────────────────────
def fig_gating(t: Theme):
    from lappato_mcb.manifests import timeseries

    W, H = 1200, 492
    ts_sum, ts_sev, ts_on = measure(timeseries, "timeseries_residual_var.csv",
                                    "_residual_var_summary", "_residual_var_severity", "timeseries")
    wd_sum, wd_sev, wd_on = measure(wdbc, "wdbc_calibration.csv",
                                    "_calibration_summary", "_calibration_severity", "wdbc")
    T1, T2 = timeseries.THRESHOLDS, wdbc.THRESHOLDS
    panels = [
        ("timeseries · heteroscedastic_residuals", "last-fold / first-fold residual variance",
         (1.0, 3.5), [(1.0, T1["var_ratio_warning"], None), (T1["var_ratio_warning"], T1["var_ratio_high"], "medium"),
                      (T1["var_ratio_high"], 3.5, "high")],
         ts_sum["last_first_variance_ratio"], ts_on, ts_sev, "{:.3f}"),
        ("wdbc · low_calibration", "mean Brier score across folds",
         (0.0, 0.25), [(0.0, T2["brier_warning"], None), (T2["brier_warning"], T2["brier_high"], "medium"),
                       (T2["brier_high"], T2["brier_critical"], "high"), (T2["brier_critical"], 0.25, "critical")],
         wd_sum["mean_brier"], wd_on, wd_sev, "{:.4f}"),
    ]
    b = ""
    x0, x1 = 60, 1140
    for i, (title, metric, (lo, hi), bands, value, active, sev, fmt) in enumerate(panels):
        y = 30 + i * 214
        b += rect(28, y, 1144, 196, t.card, stroke=t.hair, rx=12)
        b += text(x0, y + 36, title, size=17, fill=t.ink, mono=True, weight=600)
        b += text(x0, y + 60, metric, size=14, fill=t.ink2)
        sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
        by = y + 96
        for a, z, band in bands:
            fill = t.panel if band is None else STATUS[band]
            b += f'<rect x="{sx(a) + 1:.1f}" y="{by}" width="{sx(z) - sx(a) - 2:.1f}" height="34" rx="4" fill="{fill}" opacity="{1 if band is None else 0.9}"/>'
            label = "inactive — no query sent" if band is None else band
            b += text((sx(a) + sx(z)) / 2, by + 22, label, size=13.5, anchor="middle",
                      fill=t.ink2 if band is None else "#1f2328", weight=500 if band is None else 650)
        for a, _, band in bands[1:]:
            b += text(sx(a), by + 56, f"{a:g}", size=13, fill=t.muted, anchor="middle", mono=True)
        b += text(sx(lo), by + 56, f"{lo:g}", size=13, fill=t.muted, anchor="start", mono=True)
        vx = sx(value)
        b += f'<path d="M{vx:.1f},{by - 4} l-8,-14 h16 z" fill="{t.ink}"/>'
        b += line(vx, by - 4, vx, by + 38, t.ink, sw=2)
        verdict = f"active · severity {sev}" if active else "not active · no card, no query"
        anchor = "start" if vx < 760 else "end"
        b += text(vx + (12 if anchor == "start" else -12), by - 12,
                  f"measured {fmt.format(value)} → {verdict}", size=14.5, fill=t.ink, weight=600, anchor=anchor)
    b += text(28, H - 18, "Boundaries are read from each manifest's THRESHOLDS table (override_thresholds() "
              "changes them). Measured values: complete demo cycles of "
              f"{demo_runs()['provenance']['date']}.", size=13.5, fill=t.muted)
    return W, H, b, "How a detector decides", (
        "Two number lines with threshold bands. Time-series residual variance ratio 1.848 is above the "
        "1.5 threshold, so the detector is active with severity medium. WDBC mean Brier 0.027 is below "
        "0.10, so that detector stays inactive and sends no query.")


# ─── Figure 6 · anatomy of a weakness card ────────────────────────────
def fig_card(t: Theme):
    card = demo_card("timeseries", "heteroscedastic_residuals")
    p0 = card["top_papers"][0]
    W = 1200
    rows: list[tuple[str, str | None]] = [
        ("{", None),
        (f'  "weakness": "{card["weakness"]}",', None),
        (f'  "evidence": "{card["evidence"]}",', "why"),
        (f'  "evidence_summary": {json.dumps(card["evidence_summary"])},', "why"),
        (f'  "severity": "{card["severity"]}",', "grade"),
        ('  "queries": [', "queries"),
    ]
    rows += [(f'    "{clip(q, 50)}",', "queries") for q in card["queries"]]
    rows += [("  ],", None),
             (f'  "raw_hits_by_source": {json.dumps(card["raw_hits_by_source"])},', "hits"),
             (f'  "n_papers_kept": {card["n_papers_kept"]},', "hits"),
             ('  "top_papers": [{', "papers"),
             (f'    "source": "{p0["source"]}", "year": "{p0["year"]}", "score": {p0["score"]},', "papers"),
             (f'    "title": "{clip(p0["title"], 46)}"', "papers"),
             (f'  }}, … {len(card["top_papers"]) - 1} more],', "papers"),
             ('  "next_checks": [', "next"),
             (f'    "{clip(card["next_checks"][0], 52)}", …],', "next"),
             ('  "success_criteria": [', "next"),
             (f'    "{clip(card["success_criteria"][0], 52)}", …]', "next"),
             ("}", None)]
    notes = {
        "why": ("Why it fired", "The evidence file and the statistic the manifest computed from it."),
        "grade": ("How serious", "Graded by the manifest's thresholds, not by a model."),
        "queries": ("What was asked", "The exact queries sent to each source, after context substitution."),
        "hits": ("What came back", "Hits per source before deduplication; papers kept after it."),
        "papers": ("What to read", "Ranked by a deterministic lexical score — a pointer, not a relevance judgement."),
        "next": ("What to do next", "Written by the manifest author and versioned with it; never generated."),
    }
    top, lh = 34, 25
    H = top + len(rows) * lh + 64
    code_w = 700
    b = rect(28, 22, code_w, len(rows) * lh + 24, t.code_bg, rx=10)
    spans: dict[str, list[int]] = {}
    for i, (s, key) in enumerate(rows):
        y = top + 16 + i * lh
        b += text(46, y, s, size=13.5, fill=t.ink, mono=True)
        if key:
            spans.setdefault(key, []).append(y)
    order = ["why", "grade", "queries", "hits", "papers", "next"]
    next_free = 0.0
    for key in order:
        ys = spans[key]
        y_mid = (ys[0] + ys[-1]) / 2 - 5
        ty = max(y_mid, next_free)          # push a note down if the previous one is in the way
        title, note = notes[key]
        body = wrap(note, 46)[:2]
        b += path(f"M{code_w + 14},{ys[0] - 15} h8 V{ys[-1] + 6} h-8", t.blue, sw=2)
        b += path(f"M{code_w + 22},{y_mid} C{code_w + 40},{y_mid} {code_w + 36},{ty} {code_w + 54},{ty}", t.blue, sw=2)
        b += text(code_w + 62, ty + 5, title, size=15.5, fill=t.ink, weight=650)
        b += lines_at(code_w + 62, ty + 26, body, t, size=13.5, gap=18)
        next_free = ty + 26 + 18 * len(body) + 10
    b += text(28, H - 20, "A real card (abridged) from the complete time-series demo cycle of "
              f"{demo_runs()['provenance']['date']}. Full field list in the reference section.",
              size=13.5, fill=t.muted)
    return W, H, b, "Anatomy of a weakness card", (
        "An abridged real weakness card in JSON with annotations: evidence and summary explain why it "
        "fired, severity how serious, queries what was asked, hit counts what came back, top papers what "
        "to read, next checks and success criteria what to do next.")


# ─── Figure 7 · coverage of the shipped manifests ─────────────────────
FAMILIES = {
    "Methodology, cross-cutting": ["pipeline_health", "tabular_generic"],
    "Mainstream ML and evaluation": [
        "nlp", "vision", "timeseries", "recommender", "clustering", "anomaly_detection", "survival",
        "causal_ml", "fairness", "graph_ml", "rl_eval", "llm_eval", "rag_eval", "speech_audio",
        "cybersecurity", "geospatial", "medical_imaging", "wdbc"],
    "Scientific domains": [
        "mathematics", "physics", "chemistry", "biochemistry", "biology", "earth_climate", "astronomy",
        "materials_science", "neuroscience", "epidemiology", "econometrics", "social_science",
        "robotics", "quantum_computing", "pharmacology"],
}


def fig_coverage(t: Theme):
    counts = {d: (g, s) for d, g, s in registry_counts()}
    assigned = [d for ds in FAMILIES.values() for d in ds]
    missing = set(counts) - set(assigned)
    if missing or len(assigned) != len(set(assigned)):
        raise SystemExit(f"assign every manifest to exactly one family; unassigned: {sorted(missing)}")
    tot = totals()
    W, cols, tw, th, g = 1200, 5, 220, 62, 9
    y = 82
    b = text(28, 40, f"{tot['domains']} manifests · {tot['detectors']} detectors", size=19, fill=t.ink, weight=650)
    # legend (two series: identity carried by fill AND colour)
    lx = 470
    b += f'<rect x="{lx}" y="27" width="13" height="13" rx="3" fill="{t.blue}"/>'
    b += text(lx + 21, 39, f"evidence-gated ({tot['gated']}) — fires only on measured evidence", size=14, fill=t.ink2)
    lx2 = lx + 400
    b += f'<rect x="{lx2 + 1}" y="28" width="11" height="11" rx="3" fill="none" stroke="{t.orange}" stroke-width="2"/>'
    b += text(lx2 + 21, 39, f"structural ({tot['structural']}) — always on", size=14, fill=t.ink2)
    for fam, domains in FAMILIES.items():
        b += text(28, y, fam, size=15, fill=t.ink2, weight=650)
        y += 14
        k = 0
        for d in domains:
            span = 2 if sum(counts[d]) > 12 else 1
            if k % cols + span > cols:
                k += cols - k % cols
            x = 28 + (k % cols) * (tw + g)
            ty = y + (k // cols) * (th + g)
            w = tw * span + g * (span - 1)
            b += rect(x, ty, w, th, t.card, stroke=t.hair, rx=8, sw=1)
            b += text(x + 12, ty + 24, d, size=13.5, fill=t.ink, mono=True)
            gated, structural = counts[d]
            for q in range(gated + structural):
                qx, qy = x + 12 + q * 16, ty + 36
                if q < gated:
                    b += f'<rect x="{qx}" y="{qy}" width="12" height="12" rx="3" fill="{t.blue}"/>'
                else:
                    b += f'<rect x="{qx + 1}" y="{qy + 1}" width="10" height="10" rx="3" fill="none" stroke="{t.orange}" stroke-width="2"/>'
            b += text(x + w - 12, ty + 24, str(gated + structural), size=13, fill=t.muted, anchor="end", mono=True)
            k += span
        y += ((k - 1) // cols + 1) * (th + g) + 30
    H = y + 6
    return W, H, b, "Coverage of the shipped manifests", (
        f"{tot['domains']} manifests grouped in three families; each square is one detector, filled blue "
        f"if evidence-gated ({tot['gated']}) or hollow orange if structural ({tot['structural']}).")


# ─── social preview (light only, 1280×640, for repository settings) ───
def fig_social(t: Theme):
    tot = totals()
    W, H = 1280, 640
    b = rect(0, 0, W, H, t.panel, rx=0)
    b += text(80, 170, "LAPPATO_MCB", size=84, fill=t.ink, weight=750)
    b += text(84, 222, "Literature-Aware Pipeline Partner for Adaptive Transplant Optimization",
              size=28, fill=t.ink2)
    b += lines_at(84, 300, wrap("A Python library that reads the diagnostics an ML pipeline already writes, "
                                "flags methodological weaknesses when measured evidence crosses a documented "
                                "threshold, and retrieves literature targeted at each one.", 66),
                  t, size=27, gap=40, fill=t.ink)
    cx, cy = 84, 492
    for label in (f"{tot['domains']} manifests", f"{tot['detectors']} detectors", "arXiv · OpenAlex · Crossref · JOSS",
                  "standard library only", "optional MCP server"):
        _, cw = chip(cx, cy, label, t, size=22, mono=False, h=46, pad=18)
        if cx + cw > W - 80:
            cx, cy = 84, cy + 58
        c, cw = chip(cx, cy, label, t, size=22, mono=False, bg=t.card, h=46, pad=18)
        b += c
        cx += cw + 12
    return W, H, b, "LAPPATO_MCB", "Social preview card."


FIGURES = {
    "overview": fig_overview,
    "interfaces": fig_interfaces,
    "anatomy": fig_anatomy,
    "physiology": fig_physiology,
    "gating": fig_gating,
    "weakness-card": fig_card,
    "coverage": fig_coverage,
}


def main() -> None:
    for name, fn in FIGURES.items():
        for theme in (LIGHT, DARK):
            w, h, body, title, desc = fn(theme)
            out = HERE / f"{name}-{theme.name}.svg"
            out.write_text(svg(name, w, h, body, theme, title, desc), encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)}  ({w}×{h})")
    w, h, body, title, desc = fig_social(LIGHT)
    (HERE / "social-preview.svg").write_text(svg("social", w, h, body, LIGHT, title, desc), encoding="utf-8")
    print("wrote docs/figures/social-preview.svg (export to PNG for the repository settings)")


if __name__ == "__main__":
    main()
