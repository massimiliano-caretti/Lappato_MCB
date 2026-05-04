"""Tkinter GUI — orchestration and visualization only.

This module imports the pure layers (analysis, plots, pdf_builder, the
example drivers) and wires them to buttons. It NEVER reads
CSVs, computes statistics, builds plots, or assembles PDFs by itself —
everything is delegated. If you ran this file with --headless it should
still be possible to drive the underlying pipeline via the same modules
without instantiating Tk at all (each module is independently usable).
"""
from __future__ import annotations

import importlib
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Make the `lappato_mcb` package importable when this file is run directly
# as a script (no `pip install -e .` required).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lappato_mcb import analysis, pdf_builder, plots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


# Domain id -> (demo module name, run_tag, pdf filename, label).
DOMAINS: dict[str, tuple[str, str, str, str]] = {
    "wdbc": ("demo_wdbc", "wdbc",
             "wdbc_lappato_mcb_report.pdf",
             "WDBC (LightGBM, tabular clinical)"),
    "nlp": ("demo_nlp", "nlp",
            "nlp_lappato_mcb_report.pdf",
            "NLP (TF-IDF + LogReg, 20newsgroups)"),
    "timeseries": ("demo_timeseries", "timeseries",
                   "timeseries_lappato_mcb_report.pdf",
                   "Time series (AR(p), synthetic)"),
}


# ─── thin orchestrator (still import-able from CLI) ────────────────────
def run_pipeline_blocking(domain: str, log_cb) -> None:
    """Run the chosen demo + LAPPATO_MCB in this process, streaming logs.

    log_cb is a callable that accepts a string line — the GUI feeds it
    into a Text widget; a CLI caller could feed it into print().
    """
    mod_name, _, _, _ = DOMAINS[domain]
    mod = importlib.import_module(f"examples.{mod_name}")
    import builtins
    real_print = builtins.print

    def _patched(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        log_cb(msg)
        real_print(*args, **kwargs)

    builtins.print = _patched
    try:
        mod.main()
    finally:
        builtins.print = real_print


# ─── GUI ───────────────────────────────────────────────────────────────
class LappatoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LAPPATO_MCB — multi-domain demo")
        root.geometry("1180x780")

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._figures: dict = {}
        self._domain = tk.StringVar(value="wdbc")

        self._build_layout()
        self._poll_log()
        self._refresh_data()  # in case checkpoints already exist

    # ── layout ─────────────────────────────────────────────────────
    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="Domain:").pack(side="left", padx=(4, 2))
        self.cmb_domain = ttk.Combobox(
            bar, textvariable=self._domain,
            values=[label for _, _, _, label in DOMAINS.values()],
            state="readonly", width=42,
        )
        self.cmb_domain.set(DOMAINS["wdbc"][3])
        self.cmb_domain.pack(side="left", padx=4)
        self.cmb_domain.bind("<<ComboboxSelected>>", self._on_domain_change)

        self.btn_run = ttk.Button(bar, text="▶  Run pipeline + LAPPATO_MCB",
                                  command=self._on_run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_refresh = ttk.Button(bar, text="↻  Refresh charts",
                                      command=self._refresh_data)
        self.btn_refresh.pack(side="left", padx=4)
        self.btn_pdf = ttk.Button(bar, text="📄  Build PDF report",
                                  command=self._on_build_pdf)
        self.btn_pdf.pack(side="left", padx=4)
        self.btn_open_pdf = ttk.Button(bar, text="↗  Open PDF",
                                       command=self._on_open_pdf,
                                       state="disabled")
        self.btn_open_pdf.pack(side="left", padx=4)
        self.lbl_status = ttk.Label(bar, text="idle", foreground="grey")
        self.lbl_status.pack(side="right", padx=8)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Console tab
        console_frame = ttk.Frame(self.nb)
        self.nb.add(console_frame, text="Console")
        self.console = scrolledtext.ScrolledText(
            console_frame, wrap="word", font=("monospace", 9), height=10,
        )
        self.console.pack(fill="both", expand=True)

        # Figure tabs are populated dynamically on each refresh from
        # the bundle's actual figures (pack-supplied + framework). Tab
        # set adapts to the active domain — no hard-coding here.
        self.fig_tabs: dict[str, tk.Frame] = {}

        sum_frame = ttk.Frame(self.nb)
        self.nb.add(sum_frame, text="Summary")
        self.summary = scrolledtext.ScrolledText(
            sum_frame, wrap="word", font=("monospace", 10),
        )
        self.summary.pack(fill="both", expand=True)

    # ── log pump ───────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        self._log_queue.put(msg)

    def _poll_log(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.console.insert("end", msg + "\n")
                self.console.see("end")
        except queue.Empty:
            pass
        self.root.after(150, self._poll_log)

    def _set_status(self, text: str, color: str = "grey") -> None:
        self.lbl_status.config(text=text, foreground=color)

    # ── actions ────────────────────────────────────────────────────
    def _selected_domain(self) -> str:
        sel = self.cmb_domain.get()
        for did, (_, _, _, label) in DOMAINS.items():
            if label == sel:
                return did
        return "wdbc"

    def _on_domain_change(self, _evt=None) -> None:
        self._refresh_data()

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Already running",
                                "Pipeline is already in progress.")
            return
        domain = self._selected_domain()
        self.btn_run.config(state="disabled")
        self._set_status(f"running {domain}…", "#c1440e")
        self._log(f"─── pipeline launch ({domain}) ─────────────────────")

        def _target() -> None:
            try:
                run_pipeline_blocking(domain, self._log)
                self._log("─── pipeline finished ───────────────────────────")
                self._set_status("done", "#2a9d8f")
            except Exception:
                self._log(f"!!! pipeline crashed:\n{traceback.format_exc()}")
                self._set_status("crashed", "red")
            finally:
                self.root.after(0, lambda: self.btn_run.config(state="normal"))
                self.root.after(0, self._refresh_data)

        self._worker = threading.Thread(target=_target, daemon=True,
                                        name="pipeline-worker")
        self._worker.start()

    def _refresh_data(self) -> None:
        domain = self._selected_domain()
        _, run_tag, _, _ = DOMAINS[domain]
        bundle = analysis.load_all(CHECKPOINTS, run_tag=run_tag)
        self._render_figures(bundle)
        self._render_summary(bundle)
        ps, cs = bundle["pipeline_summary"], bundle["lappato_mcb_summary"]
        n_exp = ps.get("n_experiments", 0)
        if n_exp > 0 or cs["n_total"] > 0:
            self._set_status(
                f"refreshed [{run_tag}] · "
                f"{n_exp} pipeline rows · "
                f"{cs['n_total']} papers", "#264653",
            )
        else:
            self._set_status(f"no data for [{run_tag}] yet", "grey")

    def _render_figures(self, bundle) -> None:
        figs = plots.build_all(bundle)
        # Drop any previously-mounted figure tabs (Console + Summary
        # remain — the rest get rebuilt to match the active domain).
        for key, frame in self.fig_tabs.items():
            for child in frame.winfo_children():
                child.destroy()
            self.nb.forget(frame)
        self.fig_tabs = {}
        # Rebuild — Summary is the LAST tab; insert figure tabs before it.
        summary_idx = self.nb.index("end") - 1
        for key, fig in figs.items():
            frame = ttk.Frame(self.nb)
            self.nb.insert(summary_idx, frame, text=_friendly_label(key))
            summary_idx += 1
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.fig_tabs[key] = frame
        self._figures = figs

    def _render_summary(self, bundle) -> None:
        ps = bundle["pipeline_summary"]
        cs = bundle["lappato_mcb_summary"]
        ms = bundle.get("meta_summary", {})
        cmp = bundle.get("baseline_comparison", {})
        wc = bundle.get("weakness_card_summary", {})
        # PIPELINE block: pack-driven, no WDBC names hard-coded.
        lines = [
            f"RUN TAG : {bundle.get('run_tag', '?')}",
            "─" * 60,
            "PIPELINE",
            f"experiments         : {ps.get('n_experiments', 0)}",
        ]
        head_name = ps.get("headline_metric_name")
        if head_name:
            n = ps.get("headline_metric_n", 0)
            mean = ps.get("headline_metric_mean")
            sd = ps.get("headline_metric_sd")
            if isinstance(mean, float) and mean == mean:
                lines.append(
                    f"{head_name:20s}: {mean:.4f} ± {sd:.4f}  (sample SD, N={n})"
                )
        for sm in ps.get("secondary_metrics", []):
            v = sm.get("mean")
            if isinstance(v, float) and v == v:
                lines.append(f"{sm['name']:20s}: {v:.4f}  ({sm.get('unit','')})")
        sources_str = ", ".join(
            f"{k}={v}" for k, v in cs.get("by_source", {}).items()
        ) or "—"
        lines += [
            "",
            "LAPPATO_MCB",
            "─" * 60,
            f"cycles              : {cs['n_cycles']} "
            f"(first → last: {cs['first_cycle']} → {cs['last_cycle']})",
            f"unique papers       : {cs['n_total']}  ({sources_str})",
            f"weakness cards      : {wc.get('n_cards', 0)}",
            f"high/critical cards : {wc.get('n_high_or_critical', 0)}",
            f"new vs prior runs   : {wc.get('total_new_vs_previous_runs', 0)}",
            "",
            "META-LOG",
            "─" * 60,
            f"queries executed    : {ms.get('total_queries', 0)}",
            f"raw hits            : {ms.get('total_raw_hits', 0)}",
            f"id-dedup blocked    : {ms.get('total_id_dedup', 0)}  "
            f"({ms.get('dedup_rate_id', 0)*100:.1f}%)",
            f"trigram-dedup blocked: {ms.get('total_fp_dedup', 0)}  "
            f"({ms.get('dedup_rate_fp', 0)*100:.1f}%)",
            f"TTF (s)             : "
            f"mean={ms.get('first_paper_sec_mean')}  "
            f"sd={ms.get('first_paper_sec_sd')}  "
            f"N={ms.get('first_paper_sec_n', 0)}",
        ]
        if cmp:
            lines += [
                "",
                "BASELINE vs TARGETED",
                "─" * 60,
                f"targeted-only       : {int(cmp.get('targeted_only', 0))}",
                f"naive-only          : {int(cmp.get('naive_only', 0))}",
                f"intersection        : {int(cmp.get('n_intersection', 0))}",
                f"Jaccard             : {float(cmp.get('jaccard', 0.0)):.3f}",
            ]
        lines += ["", "papers per weakness :"]
        for k, v in sorted(cs["by_weakness"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:32s}  {v}")
        latest_cards = wc.get("latest_by_weakness", {})
        if latest_cards:
            lines += ["", "latest weakness cards :"]
            for wid, card in sorted(latest_cards.items()):
                lines.append(
                    f"  {wid:32s}  sev={card.get('severity', '?'):8s}  "
                    f"kept={card.get('n_papers_kept', 0)}  "
                    f"new={card.get('n_new_vs_previous_runs', 0)}"
                )
        lines += ["", "papers per year :"]
        for k, v in sorted(cs["by_year"].items()):
            lines.append(f"  {k:8s}  {v}")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(lines))

    def _on_build_pdf(self) -> None:
        try:
            domain = self._selected_domain()
            _, run_tag, pdf_name, _ = DOMAINS[domain]
            bundle = analysis.load_all(CHECKPOINTS, run_tag=run_tag)
            if (bundle["pipeline_summary"].get("n_experiments", 0) == 0
                    and bundle["lappato_mcb_summary"]["n_total"] == 0):
                messagebox.showwarning(
                    "No data",
                    f"Run the {run_tag} pipeline first (no CSVs in checkpoints/).",
                )
                return
            self._set_status(f"building PDF [{run_tag}]…", "#c1440e")
            out_path = CHECKPOINTS / pdf_name
            out = pdf_builder.build_pdf(bundle, out_path)
            self._last_pdf = out
            self._log(f"PDF written → {out}")
            self.btn_open_pdf.config(state="normal")
            self._set_status(f"PDF ready: {out.name}", "#2a9d8f")
        except Exception:
            self._log(f"!!! PDF build crashed:\n{traceback.format_exc()}")
            self._set_status("PDF crashed", "red")

    def _on_open_pdf(self) -> None:
        pdf = getattr(self, "_last_pdf", None)
        if pdf is None or not pdf.exists():
            messagebox.showinfo("Not yet", "Build the PDF first.")
            return
        if sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(pdf)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(pdf)])
        else:
            import os
            os.startfile(str(pdf))  # type: ignore[attr-defined]


def _friendly_label(figure_id: str) -> str:
    """Render ``wdbc_bac_per_fold`` as ``Wdbc Bac Per Fold`` (used as tab text)."""
    return figure_id.replace("_", " ").title()


def main() -> None:
    root = tk.Tk()
    LappatoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
