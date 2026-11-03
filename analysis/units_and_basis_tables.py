#!/usr/bin/env python3
"""Section 6.1 (units and basis): Table 6.1, Table 6.2 and Figure 6.1.

  python3 units_and_basis_tables.py results/units_and_basis/scaling_summary.csv [outdir]

Reads the summary written by experiments/run_scaling_experiment.py and writes
table_basis_invariance.tex, table_nuts.tex and fig_basis_invariance.pdf/.png
to outdir (default: paper_outputs); the numbers quoted in the text are printed.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1]
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "paper_outputs")
OUT.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(src)
ESS = "inv_airm_dist_We_ESSsec_bulk_median_total"
ESSD = "inv_airm_dist_We_ESS_bulk_median"
RH = "strict_Rhat_max"
METHODS = ["ConeMALA", "LogCholeskyMALA", "LogCholeskyMALA_DiagMetric"]
NAME = {"ConeMALA": "ConeMALA", "LogCholeskyMALA": "LC-MALA", "LogCholeskyMALA_DiagMetric": "LC-MALA + diag"}
FLAG = 1.5

def gm(x): return float(np.exp(np.mean(np.log(x))))

rows = [("identity", "identity", None),
        ("dec1", r"diagonal $D$, $10^{1}$", ["V3_scale_dec1"]),
        ("dec2", r"diagonal $D$, $10^{2}$", ["V3_scale_dec2"]),
        ("dec3", r"diagonal $D$, $10^{3}$", ["V3_scale_dec3"]),
        ("rot", r"$DQ$, $10^{2}$", ["V3_scale_dec2_rotated"]),
        ("qd0", r"$QD$, $10^{2}$ (rotation 1)", ["V3_scale_dec2_QD"]),
        ("qd1", r"$QD$, $10^{2}$ (rotation 2)", ["V3_scale_dec2_QD_r1"]),
        ("qd2", r"$QD$, $10^{2}$ (rotation 3)", ["V3_scale_dec2_QD_r2"])]
ident = t[t.family == "identity"]
h0 = {m: gm(ident[ident.Method == m]["h"]) for m in METHODS[:2]}
cell = {}
for key, _, vs in rows:
    for m in METHODS:
        g = ident[ident.Method == m] if vs is None else t[(t.Method == m) & t.variant.isin(vs)]
        assert len(g) >= 1, (key, m)
        cell[key, m] = dict(
            ess=gm(g[ESS]), essd=gm(g[ESSD]), rhat=float(g[RH].max()),
            h=(gm(g["h"]) / h0[m]) if m in h0 else np.nan,
            phi=float(g["Phi_mean"].mean()), n=len(g))

# ---------- LaTeX table ----------
def fmt_ess(v): return f"{v:.2f}" if v < 10 else f"{v:.1f}"
def fmt_h(v):
    if np.isnan(v): return "--"
    return f"{v:.2f}" if v >= 0.1 else (f"{v:.3f}" if v >= 0.01 else f"{v:.4f}")
lines = []
for key, label, _ in rows:
    parts = [label]
    for m in METHODS:
        c = cell[key, m]
        dag = r"$^{\dagger}$" if c["rhat"] > FLAG else ""
        parts += [fmt_ess(c["ess"]) + dag, f"{c['rhat']:.2f}"]
        if m in h0: parts.append(fmt_h(c["h"]))
    lines.append(" & ".join(parts) + r" \\")
    if key in ("identity", "dec3", "rot"): lines.append(r"\addlinespace")
tex = r"""\begin{table}[htbp]
\centering
\caption{Sensitivity to feature units and basis, $d=5$, $m=20$ cycle graph,
one dataset transformed by $A\in\{I,D,DQ,QD\}$ (\cref{sec:basis-invariance}).
$D$ is diagonal with feature scales spread over the stated range (maximum over
minimum), $Q$ is a Haar-random orthogonal matrix, and the three $QD$ rows use
independent $Q$. ESS/s is the median-over-edges bulk ESS of the AIRM distance
to the data-generating $W^\star$ divided by end-to-end wall-clock (identity row:
geometric mean of three chain replicates, with $\widehat R$ the maximum over
them); $\widehat R$ is the maximum
rank-normalized split-$\widehat R$ over 43 congruence-invariant functionals;
$h/h_0$ is the tuned step relative to the identity basis.
\ConeMALA{} keeps $48{,}000$ draws per chain, the LC samplers $12{,}000$.
$^{\dagger}$: $\widehat R>1.5$, chains have not mixed, so the ESS/s value is
not a stationary efficiency estimate.}
\label{tab:basis-invariance}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{ConeMALA} & \multicolumn{3}{c}{LC-MALA}
& \multicolumn{2}{c}{LC-MALA + diag} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
Basis $A$ & ESS/s & $\widehat R$ & $h/h_0$ & ESS/s & $\widehat R$ & $h/h_0$ & ESS/s & $\widehat R$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
open(OUT / "table_basis_invariance.tex", "w").write(tex)

# ---------- numbers quoted in the text ----------
print("== cell summary ==")
for key, label, _ in rows:
    print(key.ljust(9), "  ".join(f"{NAME[m][:9]:9s} ess={cell[key,m]['ess']:.3f} R={cell[key,m]['rhat']:.2f} h={cell[key,m]['h']:.4f} phi={cell[key,m]['phi']:.0f}" for m in METHODS))
print("per-draw identity geomean:", {m: round(cell['identity', m]['essd'], 1) for m in METHODS})
for key in ("qd0", "qd1", "qd2"):
    c, l, d = (cell[key, m] for m in METHODS)
    print(key, "Cone/diag ESS/s %.2f  per draw %.1f  Cone/LC ESS/s %.2f" % (c['ess']/d['ess'], c['essd']/d['essd'], c['ess']/l['ess']))
print("identity: LC/Cone ESS/s %.2f diag/Cone %.2f ; per-draw LC/Cone %.2f diag/Cone %.2f" % (
    cell['identity','LogCholeskyMALA']['ess']/cell['identity','ConeMALA']['ess'],
    cell['identity','LogCholeskyMALA_DiagMetric']['ess']/cell['identity','ConeMALA']['ess'],
    cell['identity','LogCholeskyMALA']['essd']/cell['identity','ConeMALA']['essd'],
    cell['identity','LogCholeskyMALA_DiagMetric']['essd']/cell['identity','ConeMALA']['essd']))
sc = [k for k, _, _ in rows if k != "identity"]
cm = [cell[k, "ConeMALA"]["ess"] / cell["identity", "ConeMALA"]["ess"] for k in sc]
print("Cone ESS/s ratio to identity over 7 transformed:", np.round(cm, 3), "range", min(cm), max(cm))
print("Cone h/h0:", [round(cell[k,'ConeMALA']['h'],3) for k in sc])
print("LC h0 / h for qd:", [round(1/cell[k,'LogCholeskyMALA']['h'],1) for k in ("qd0","qd1","qd2")])
print("LC h0/h dec1..3:", [round(1/cell[k,'LogCholeskyMALA']['h'],1) for k in ("dec1","dec2","dec3")])


# ---------- LC-NUTS table (identity, DQ, QD rotation 1) ----------
LEAP = "n_leapfrog_mean"
def grads(g):  # production gradient evaluations per retained draw
    return g[LEAP].astype(float).fillna(1.0) if LEAP in g else pd.Series(1.0, index=g.index)
NUTS = ["LogCholeskyNUTS", "LogCholeskyNUTS_IdentityMetric"]
NAME.update({"LogCholeskyNUTS": "LC-NUTS (diag)", "LogCholeskyNUTS_IdentityMetric": "LC-NUTS (identity)"})
ALL = METHODS + NUTS
nrows = [("identity", "identity", None), ("rot", r"$DQ$, $10^{2}$", ["V3_scale_dec2_rotated"]),
         ("qd0", r"$QD$, $10^{2}$", ["V3_scale_dec2_QD"]),
         ("qd3", r"$QD$, $10^{3}$", ["V3_scale_dec3_QD"])]
for key, _, vs in rows + nrows:
    for m in ALL:
        if (key, m) in cell and "kg" in cell[key, m]:
            continue
        g = ident[ident.Method == m] if vs is None else t[(t.Method == m) & t.variant.isin(vs)]
        if g.empty:
            continue
        nd = (g["retained"] * g["n_chains"]).astype(float)
        c = cell.setdefault((key, m), dict(ess=gm(g[ESS]), essd=gm(g[ESSD]), rhat=float(g[RH].max()),
                                            h=np.nan, phi=float(g["Phi_mean"].mean()), n=len(g)))
        c["gpd"] = gm(grads(g))
        c["kg"] = gm(1000 * g[ESSD] / (nd * grads(g)))
        c["sat"] = float(g["treedepth_saturated_rate"].max()) if m in NUTS else np.nan
        c["mspg"] = gm(1e3 * g["runtime_sampling_sec"] / (nd * grads(g)))
NM = ["ConeMALA"] + NUTS
nl = []
for key, label, _ in nrows:
    parts = [label]
    for m in NM:
        c = cell.get((key, m))
        parts += ([fmt_ess(c["ess"]), f"{c['gpd']:.0f}", fmt_ess(c["kg"])] if c and "kg" in c else ["--"] * 3)
    nl.append(" & ".join(parts) + r" \\")
nl.append(r"\midrule")
def f3(v): return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"
for key, lab in (("qd0", r"$QD\,10^{2}$ / identity"), ("qd3", r"$QD\,10^{3}$ / identity")):
    r = []
    for m in NM:
        a, b = cell["identity", m], cell.get((key, m))
        r += ([f3(b['ess']/a['ess']), f"{b['gpd']/a['gpd']:.1f}", f3(b['kg']/a['kg'])] if b and "kg" in b else ["--"] * 3)
    nl.append(lab + " & " + " & ".join(r) + r" \\")
ntex = r"""\begin{table}[htbp]
\centering
\caption{\ConeMALA{} against log-Cholesky NUTS on the identity, $DQ$ ($10^{2}$), and $QD$
($10^{2}$ and $10^{3}$, first rotation) copies of the posterior of \cref{tab:basis-invariance}.
grad/draw: production gradient evaluations per retained draw (leapfrog steps for
NUTS, one for \ConeMALA{}); ESS/kgrad: median-over-edges bulk ESS of the AIRM
distance per $1000$ production gradient evaluations. \ConeMALA{}'s identity row is
the geometric mean of three chain replicates, the NUTS identity rows one replicate.
The $QD$ copies at $10^{2}$ and $10^{3}$ use the same rotation $Q$; at $10^{3}$
LC-NUTS keeps $3{,}000$ draws per chain and reaches the maximum tree depth
($1023$ leapfrog steps) on every draw, and LC-NUTS with identity metric was not run.
Every run here has maximum rank-normalized $\widehat R\le1.03$ (NUTS:
$\le1.004$) and no divergent transitions. \ConeMALA{} keeps $48{,}000$ draws per
chain, LC-NUTS $12{,}000$. One \ConeMALA{} step costs
about $4\times$ one log-Cholesky gradient in our implementation, which is why it
leads in ESS/kgrad but not in ESS/s on the $QD$ copy at $10^{2}$.}
\label{tab:nuts}
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{ConeMALA} & \multicolumn{3}{c}{LC-NUTS, diag.\ metric}
& \multicolumn{3}{c}{LC-NUTS, identity metric} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Basis $A$ & ESS/s & \begin{tabular}[b]{@{}c@{}}grad/\\draw\end{tabular} & \begin{tabular}[b]{@{}c@{}}ESS/\\kgrad\end{tabular} & ESS/s & \begin{tabular}[b]{@{}c@{}}grad/\\draw\end{tabular} & \begin{tabular}[b]{@{}c@{}}ESS/\\kgrad\end{tabular} & ESS/s & \begin{tabular}[b]{@{}c@{}}grad/\\draw\end{tabular} & \begin{tabular}[b]{@{}c@{}}ESS/\\kgrad\end{tabular} \\
\midrule
""" + "\n".join(nl) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(OUT / "table_nuts.tex", "w").write(ntex)
print("== NUTS cells ==")
for key, _, _ in nrows:
    for m in NM:
        if (key, m) not in cell: continue
        c = cell[key, m]
        print(key.ljust(9), m.ljust(32), "ess/s=%.3f gpd=%.1f kg=%.3f sat=%.3f R=%.4f ms/grad=%.3f phi=%.1f essd=%.0f" % (
            c['ess'], c['gpd'], c['kg'], c['sat'], c['rhat'], c['mspg'], c['phi'], c['essd']))
print("identity ESS/s NUTS/Cone:", {m: round(cell['identity', m]['ess']/cell['identity','ConeMALA']['ess'], 1) for m in NUTS})
print("QD ESS/s NUTS/Cone:", {m: round(cell['qd0', m]['ess']/cell['qd0','ConeMALA']['ess'], 2) for m in NUTS})
print("QD3:", {m: (round(cell['qd3', m]['ess'],3), round(cell['qd3', m]['kg'],3), cell['qd3',m]['sat'], round(cell['qd3',m]['rhat'],4)) for m in NM if ('qd3', m) in cell})
print("QD ESS/kgrad Cone/NUTS:", {m: round(cell['qd0','ConeMALA']['kg']/cell['qd0', m]['kg'], 2) for m in NUTS})
print("ms per grad:", {m: round(cell['identity', m]['mspg'], 3) for m in ALL})
# ---------- figure ----------
COL = {"ConeMALA": "#2a78d6", "LogCholeskyMALA": "#eb6834", "LogCholeskyMALA_DiagMetric": "#1baf7a",
       "LogCholeskyNUTS": "#7d4fc4", "LogCholeskyNUTS_IdentityMetric": "#c2407e"}
MK = {"ConeMALA": "o", "LogCholeskyMALA": "^", "LogCholeskyMALA_DiagMetric": "s",
      "LogCholeskyNUTS": "D", "LogCholeskyNUTS_IdentityMetric": "P"}
JIT = dict(zip(ALL, (-0.2, -0.1, 0.0, 0.1, 0.2)))
xl = ["identity", r"$D$" "\n" r"$10^1$", r"$D$" "\n" r"$10^2$", r"$D$" "\n" r"$10^3$", r"$DQ$" "\n" r"$10^2$", r"$QD$" "\n" r"$10^2$", r"$QD$" "\n" r"$10^3$"]
XKEY = [("identity", 0), ("dec1", 1), ("dec2", 2), ("dec3", 3), ("rot", 4), ("qd0", 5), ("qd1", 5), ("qd2", 5), ("qd3", 6)]
plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
panels = [("ess", "ESS/s (AIRM distance)", "(a) throughput"),
          ("kg", "ESS per 1000 gradient evaluations", "(b) per gradient"),
          ("rhat", r"max rank-normalized $\widehat R$", "(c) convergence")]
for ax, (field, ylab, ttl) in zip(axes, panels):
    for m in ALL:
        P = [(x, cell[k, m]) for k, x in XKEY if (k, m) in cell and field in cell[k, m]]
        line = [(x, c[field]) for x, c in P if x <= 3]
        if len(line) > 1:
            ax.plot([x for x, _ in line], [v for _, v in line], color=COL[m], lw=1.0, alpha=0.8, zorder=2)
        for x, c in P:
            bad = c["rhat"] > FLAG
            ax.scatter(x + JIT[m], c[field], marker=MK[m], s=22, zorder=3,
                       facecolors="none" if bad else COL[m], edgecolors=COL[m], linewidths=0.9)
    ax.set_yscale("log"); ax.set_xticks(range(7)); ax.set_xticklabels(xl, fontsize=6.8)
    ax.set_ylabel(ylab, fontsize=7.5)
    if field == "rhat":
        ax.set_yticks([1, 1.5, 2, 3]); ax.set_yticklabels(["1", "1.5", "2", "3"]); ax.minorticks_off()
        ax.axhline(1.05, color="0.4", lw=0.7, ls=":"); ax.axhline(FLAG, color="0.4", lw=0.7, ls="--")
        ax.text(6.55, FLAG, "1.5", va="bottom", ha="right", fontsize=6, color="0.35")
    ax.grid(axis="y", lw=0.4, alpha=0.4); ax.set_xlim(-0.5, 6.6)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.set_title(ttl, fontsize=8, loc="left")
h = [plt.Line2D([], [], color=COL[m], marker=MK[m], ls="-" if "NUTS" not in m else "", lw=1.0, ms=4.5, label=NAME[m]) for m in ALL]
h.append(plt.Line2D([], [], color="0.3", marker="o", mfc="none", ls="", ms=4.5, label=r"hollow: $\widehat R>1.5$"))
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01),
           columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
fig.savefig(OUT / "fig_basis_invariance.pdf"); fig.savefig(OUT / "fig_basis_invariance.png", dpi=220)
print("wrote tables + figure")
