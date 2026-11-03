#!/usr/bin/env python3
"""Block dimension (Section 6.2): Tables SM3.1 and SM3.2 (LaTeX) and the
numbers quoted in the text, from the summary written by
experiments/run_m50_d_sweep_wishart_with_fisher.py.

  python3 block_dimension_tables.py results/block_dimension/combined_summary_m50_d_sweep.csv
"""
import sys

import pandas as pd

ORDER = ["ConeMALA", "FisherAdaptiveMALA", "EuclideanMALA", "GenericRMALA"]
LABEL = {"ConeMALA": r"\textsc{ConeMALA}", "FisherAdaptiveMALA": "Fisher-adaptive MALA",
         "EuclideanMALA": "Euclidean MALA", "GenericRMALA": "Generic RMALA"}
RHAT = ["rel_W_error_Rhat", "test_nll_Rhat", "logdet_Q_Rhat", "Phi_Rhat"]
CHAINS = 4

df = pd.read_csv(sys.argv[1]).copy()
df = df[df["Method"].isin(ORDER) & (df["status"].fillna("OK") == "OK")].copy()
df["max_rhat"] = df[RHAT].max(axis=1)
df["ms"] = 1e3 * df["runtime_sampling_all_chains"] / (CHAINS * df["retained"])

rh, main = [], []
for d in sorted(df["d"].unique()):
    g = df[df["d"] == d].set_index("Method")
    for i, k in enumerate(ORDER):
        r = g.loc[k]
        lead = f"{int(d)}" if i == 0 else ""
        rh.append(f"{lead} & {LABEL[k]} & " + " & ".join(f"{r[c]:.3f}" for c in RHAT) + r" \\")
        main.append(f"{lead} & {LABEL[k]} & {r.accept_mean:.3f} & {r.invalid_rate_mean:.3f} & "
                    f"{r.ms:.2f} & {r.rel_W_error_ESSsec_sampling:.2f} & "
                    f"{r.test_nll_ESSsec_sampling:.2f} & {r.max_rhat:.3f}" + r" \\")
    rh.append(r"\addlinespace"); main.append(r"\addlinespace")
rh.pop(); main.pop()

print(r"""% ===== Table SM3.1 -- replaces the table labelled tab:d_sweep_rhat =====
\begin{table}[htbp]
\centering
\caption{Classical split-\(\widehat R\) diagnostics for the \(m=50\) block-dimension sweep.}
\label{tab:d_sweep_rhat}
\scriptsize
\begin{tabular}{llrrrr}
\toprule
\(d\) & Method & \(\widehat R_W\) & \(\widehat R_{\rm NLL}\) & \(\widehat R_{\log\det Q}\) & \(\widehat R_\Phi\) \\
\midrule""")
print("\n".join(rh))
print(r"""\bottomrule
\end{tabular}
\end{table}

% ===== Table SM3.2 -- replace the rows between \midrule and \bottomrule of
% tab:d_sweep_main; caption and footnote unchanged =====""")
print("\n".join(main))

print("\n% ===== numbers for Section 6.2 (block-dimension paragraph) =====")
c = df[df["Method"] == "ConeMALA"].set_index("d")
b = df[df["Method"] != "ConeMALA"]
rw = [c.loc[d, "rel_W_error_ESSsec_sampling"] / r.rel_W_error_ESSsec_sampling for d, r in zip(b.d, b.itertuples())]
nl = [c.loc[d, "test_nll_ESSsec_sampling"] / r.test_nll_ESSsec_sampling for d, r in zip(b.d, b.itertuples())]
print(f"% ConeMALA / baseline relative-W production ESS/s: {min(rw):.1f}-{max(rw):.1f}x")
print(f"% ConeMALA / baseline held-out-NLL production ESS/s: {min(nl):.1f}-{max(nl):.1f}x")
print(f"% ConeMALA acceptance {c.accept_mean.min():.3f}-{c.accept_mean.max():.3f}; "
      f"ms/trans {c.ms.min():.1f}-{c.ms.max():.1f}; relW ESS/s "
      f"{c.rel_W_error_ESSsec_sampling.max():.1f} -> {c.rel_W_error_ESSsec_sampling.min():.1f}")
iv = b.loc[b["invalid_rate_mean"].idxmax()]
print(f"% largest baseline invalid rate: {iv.invalid_rate_mean:.2f} ({iv.Method}, d={int(iv.d)})")
for k in ORDER:
    s = df[df["Method"] == k].set_index("d")["max_rhat"].round(3).sort_index()
    print(f"% max split-Rhat {k}: " + ", ".join(f"d={int(d)}: {v:.3f}" for d, v in s.items()))
