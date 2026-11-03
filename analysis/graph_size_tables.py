#!/usr/bin/env python3
"""Graph size (Section 6.2): Tables SM6.1 and SM6.3 (LaTeX rows) and the
efficiency ratios quoted in the text, from the summary written by
experiments/run_d5_m_sweep_wishart_with_fisher.py.

  python3 graph_size_tables.py results/graph_size/combined_summary_d5_m_sweep.csv
"""
import sys
import numpy as np
import pandas as pd

ORDER = ["ConeMALA", "FisherAdaptiveMALA", "EuclideanMALA", "GenericRMALA"]
LABEL = {"ConeMALA": "ConeMALA", "FisherAdaptiveMALA": "Fisher-adaptive MALA",
         "EuclideanMALA": "Euclidean MALA", "GenericRMALA": "Generic RMALA"}
CHAINS = 4


def sci(x):
    e = int(np.floor(np.log10(abs(x))))
    return rf"${x / 10**e:.2f}\times 10^{{{e}}}$"


df = pd.read_csv(sys.argv[1])
df = df[(df["status"].fillna("OK") == "OK")]
eff, rh, ratios = [], [], []
for m in sorted(df["m"].unique()):
    g = df[df["m"] == m].set_index("Method")
    N, B = int(g["retained"].iloc[0]), int(g["preburn"].iloc[0])
    eff.append(rf"\multicolumn{{11}}{{l}}{{\emph{{$m={m}$; {B:,} discarded and {N:,} retained transitions per chain}}}} \\")
    rh.append(rf"\multicolumn{{5}}{{l}}{{\emph{{$m={m}$}}}} \\")
    for k in ORDER:
        r = g.loc[k]
        ms = 1e3 * r.runtime_sampling_all_chains / (CHAINS * N)
        e1, e2 = (1e3 * r[c] / (CHAINS * N) for c in ("rel_W_error_ESS", "test_nll_ESS"))
        eff.append(" & ".join([LABEL[k], sci(r.h_final_mean), f"{r.accept_mean:.3f}",
                    f"{r.invalid_rate_mean:.3f}", f"{ms:.2f}", f"{e1:.2f}", f"{e2:.2f}",
                    f"{r.rel_W_error_ESSsec_sampling:.2f}", f"{r.test_nll_ESSsec_sampling:.2f}",
                    f"{r.rel_W_error_ESSsec_total:.2f}", f"{r.test_nll_ESSsec_total:.2f}"]) + r" \\")
        rh.append(" & ".join([LABEL[k]] + [f"{r[c]:.3f}" for c in
                  ("rel_W_error_Rhat", "test_nll_Rhat", "logdet_Q_Rhat", "Phi_Rhat")]) + r" \\")
        if k != "ConeMALA":
            c = g.loc["ConeMALA"]
            ratios.append((m, k, c.rel_W_error_ESSsec_sampling / r.rel_W_error_ESSsec_sampling,
                           c.test_nll_ESSsec_sampling / r.test_nll_ESSsec_sampling,
                           c.rel_W_error_ESSsec_total / r.rel_W_error_ESSsec_total,
                           c.test_nll_ESSsec_total / r.test_nll_ESSsec_total))
    eff.append(r"\addlinespace"); rh.append(r"\addlinespace")

print("%%% efficiency rows\n" + "\n".join(eff))
print("\n%%% Rhat rows\n" + "\n".join(rh))
R = pd.DataFrame(ratios, columns=["m", "baseline", "relW_prod", "nll_prod", "relW_e2e", "nll_e2e"])
print("\n%%% ConeMALA / baseline ESS/s ratios\n" + R.round(2).to_string(index=False))
print("ranges:", {c: (round(R[c].min(), 1), round(R[c].max(), 1)) for c in R.columns[2:]})
df = df.copy()
mx = df.assign(mx=df[["rel_W_error_Rhat", "test_nll_Rhat", "logdet_Q_Rhat", "Phi_Rhat"]].max(axis=1))
print(mx.pivot(index="m", columns="Method", values="mx").round(3))
base = df[df.Method.isin(["FisherAdaptiveMALA", "EuclideanMALA"])]
print("invalid range Fisher/Euclidean:", base.invalid_rate_mean.min(), base.invalid_rate_mean.max())
