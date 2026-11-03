#!/usr/bin/env python3
"""Figures 6.2 and 6.3 of the paper: ConeMALA against the three baseline
Langevin samplers (Fisher-adaptive, Euclidean, generic Riemannian MALA) as the
graph size m and the SPD block dimension d grow.

Both figures use the SAME two quantities, read from the run summaries:

  (a) relative-W production ESS/s   column  rel_W_error_ESSsec_sampling
      (classical ESS divided by the frozen production-kernel runtime)
  (b) maximum classical split-Rhat  max of  rel_W_error_Rhat, test_nll_Rhat,
                                            logdet_Q_Rhat, Phi_Rhat
      i.e. the paper's convergence criterion (threshold 1.05).

Usage
-----
  python3 plot_mala_scaling.py \
      --graph-summary results/graph_size/combined_summary_d5_m_sweep.csv \
      --dsweep-summary results/block_dimension/combined_summary_m50_d_sweep.csv \
      --outdir paper_outputs

Only rows with the common budget (--retained, default 12000 retained
transitions per chain) are plotted.  The plotted values are also written to
fig_mala_scaling_data.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["ConeMALA", "FisherAdaptiveMALA", "EuclideanMALA", "GenericRMALA"]
LABEL = {"ConeMALA": "ConeMALA", "FisherAdaptiveMALA": "Fisher-adaptive MALA",
         "EuclideanMALA": "Euclidean MALA", "GenericRMALA": "Generic RMALA"}
COLOR = {"ConeMALA": "#2a78d6", "FisherAdaptiveMALA": "#d4881c",
         "EuclideanMALA": "#7a7a7a", "GenericRMALA": "#9b59b6"}
MARKER = {"ConeMALA": "o", "FisherAdaptiveMALA": "s", "EuclideanMALA": "^",
          "GenericRMALA": "D"}
# method names used by older scripts / tables -> canonical names above
ALIASES = {"Fisher-adaptive MALA": "FisherAdaptiveMALA", "Euclidean MALA": "EuclideanMALA",
           "Generic RMALA": "GenericRMALA", "cone_geom_mala": "ConeMALA"}

ESS_COL = "rel_W_error_ESSsec_sampling"
RHAT_COLS = ["rel_W_error_Rhat", "test_nll_Rhat", "logdet_Q_Rhat", "Phi_Rhat"]
THRESHOLD = 1.05


def load(paths, sweep_col, retained):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["Method"] = df["Method"].replace(ALIASES)
    missing = [c for c in [sweep_col, "Method", ESS_COL, *RHAT_COLS] if c not in df]
    if missing:
        raise SystemExit(f"{paths}: missing columns {missing}")
    found = sorted(int(v) for v in df[sweep_col].unique())
    df = df[df["Method"].isin(METHODS)].copy()
    show = [c for c in (sweep_col, "Method", "retained", "status") if c in df]
    if "status" in df:
        bad = df["status"].fillna("OK") != "OK"
        if bad.any():
            print(f"[{sweep_col}-sweep] dropped rows with status != OK:\n"
                  f"{df.loc[bad, show].to_string(index=False)}")
        df = df[~bad]
    if "retained" in df and retained is not None:
        other = df["retained"] != retained
        if other.any():
            print(f"[{sweep_col}-sweep] dropped rows with retained != {retained}:\n"
                  f"{df.loc[other, show].to_string(index=False)}")
        df = df[~other]
        if df.empty:
            raise SystemExit(f"{paths}: no rows with retained == {retained}")
    print(f"[{sweep_col}-sweep] {sweep_col} values in the input: {found}; "
          f"plotted: {sorted(int(v) for v in df[sweep_col].unique())}")
    df["max_classical_Rhat"] = df[RHAT_COLS].max(axis=1)
    dup = df.duplicated(["Method", sweep_col], keep=False)
    if dup.any():
        raise SystemExit(f"{paths}: several rows for the same method and {sweep_col}:\n"
                         f"{df.loc[dup, ['Method', sweep_col]]}")
    return df.sort_values([sweep_col, "Method"])


def figure(df, sweep_col, xlabel, stem):
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    xs = sorted(df[sweep_col].unique())
    for m in METHODS:
        g = df[df["Method"] == m].sort_values(sweep_col)
        if g.empty:
            continue
        kw = dict(color=COLOR[m], marker=MARKER[m], ms=4.5, lw=1.2)
        axes[0].plot(g[sweep_col], g[ESS_COL], label=LABEL[m], **kw)
        axes[1].plot(g[sweep_col], g["max_classical_Rhat"], **kw)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"relative-$W$ production ESS/s")
    axes[0].set_title("(a) efficiency", fontsize=8.5, loc="left")
    axes[1].axhline(THRESHOLD, color="0.35", lw=0.8, ls="--")
    axes[1].text(xs[-1], THRESHOLD, f"{THRESHOLD:g}", ha="right", va="bottom",
                 fontsize=6.5, color="0.35")
    axes[1].set_ylabel(r"maximum split-$\widehat R$")
    axes[1].set_title("(b) convergence", fontsize=8.5, loc="left")
    top = max(2.35, float(df["max_classical_Rhat"].max()) + 0.05)
    axes[1].set_ylim(0.95, top)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-summary", nargs="+", required=True)
    ap.add_argument("--dsweep-summary", nargs="+", required=True)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--retained", type=int, default=12000,
                    help="common retained budget per chain to plot (default 12000)")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    graph = load(a.graph_summary, "m", a.retained)
    dsweep = load(a.dsweep_summary, "d", a.retained)
    figure(graph, "m", r"graph size $m$ ($d=5$)", out / "fig_mala_graph_size")
    figure(dsweep, "d", r"SPD block dimension $d$ ($m=50$)", out / "fig_mala_block_dim")

    cols = ["Method", ESS_COL, *RHAT_COLS, "max_classical_Rhat"]
    data = pd.concat([graph.assign(sweep="graph_size", value=graph["m"])[["sweep", "value", *cols]],
                      dsweep.assign(sweep="block_dim", value=dsweep["d"])[["sweep", "value", *cols]]])
    data.to_csv(out / "fig_mala_scaling_data.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(data.round(3).to_string(index=False))
    print(f"\nWrote {out / 'fig_mala_graph_size.pdf'}, {out / 'fig_mala_block_dim.pdf'}, "
          f"{out / 'fig_mala_scaling_data.csv'}")


if __name__ == "__main__":
    main()
