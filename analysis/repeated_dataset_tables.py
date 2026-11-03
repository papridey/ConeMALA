#!/usr/bin/env python3
"""Repeated datasets (Section 6.2): Tables SM5.1 and SM5.2 (LaTeX rows) and
the numbers quoted in the text, from the by-seed summary written by
experiments/run_d5_m50_seed_robustness_wishart.py.

  python3 repeated_dataset_tables.py results/repeated_datasets/robustness_d5_m50_by_seed.csv

Pooled production ESS/s = total ESS over all datasets / total production
runtime over all datasets.
"""
import sys

import pandas as pd

ORDER = ["ConeMALA", "FisherAdaptiveMALA", "EuclideanMALA", "GenericRMALA"]
RHAT = ["rel_W_error_Rhat", "test_nll_Rhat", "logdet_Q_Rhat", "Phi_Rhat"]

df = pd.read_csv(sys.argv[1])
df = df[df["Method"].isin(ORDER)].copy()
df["max_rhat"] = df[RHAT].max(axis=1)

print(f"{'method':20s} {'acceptance':>15s} {'invalid':>15s} {'converged':>9s} "
      f"{'median/max Rhat':>16s} {'pooled relW ESS/s':>18s} {'pooled NLL ESS/s':>17s}")
pooled = {}
for k in ORDER:
    g = df[(df["Method"] == k) & (df["status"].fillna("OK") == "OK")]
    t = g["runtime_sampling_all_chains"].sum()
    pooled[k] = (g["rel_W_error_ESS"].sum() / t, g["test_nll_ESS"].sum() / t)
    conv = int((g["max_rhat"] <= 1.05).sum())
    print(f"{k:20s} {g.accept_mean.mean():.3f} ± {g.accept_mean.std(ddof=1):.3f}  "
          f"{g.invalid_rate_mean.mean():.3f} ± {g.invalid_rate_mean.std(ddof=1):.3f}  "
          f"{conv:>4d}/{len(g)}   {g.max_rhat.median():.3f}/{g.max_rhat.max():.3f}  "
          f"{pooled[k][0]:>16.3f}  {pooled[k][1]:>16.3f}")

print("\nConeMALA / baseline, pooled production ESS/s:")
for k in ORDER[1:]:
    print(f"  {k:20s} relative W {pooled['ConeMALA'][0] / pooled[k][0]:6.1f}x   "
          f"held-out NLL {pooled['ConeMALA'][1] / pooled[k][1]:6.1f}x")

# ---------------------------------------------------------------- LaTeX rows
N_CHAINS = 4
LABEL = {"ConeMALA": r"\textbf{ConeMALA}", "FisherAdaptiveMALA": "Fisher-adaptive MALA",
         "EuclideanMALA": "Euclidean MALA", "GenericRMALA": "Generic RMALA"}


def pm(values, digits):
    return rf"${values.mean():.{digits}f}\pm{values.std(ddof=1):.{digits}f}$"


df["ms"] = 1e3 * df["runtime_sampling_all_chains"] / (N_CHAINS * df["retained"])
for c in ("rel_W_error", "test_nll"):
    df[f"{c}_ess_per_trans"] = 1e3 * df[f"{c}_ESS"] / (N_CHAINS * df["retained"])

print("\n% ===== tab:d5-m50-seed-efficiency-corrected: rows between \\midrule and \\bottomrule =====")
for k in ORDER:
    g = df[(df["Method"] == k) & (df["status"].fillna("OK") == "OK")]
    big = k == "ConeMALA"
    cells = [pm(g.accept_mean, 3), pm(g.invalid_rate_mean, 3), pm(g.ms, 2),
             pm(g.rel_W_error_ess_per_trans, 2 if big else 3),
             pm(g.test_nll_ess_per_trans, 2 if big else 3),
             pm(g.rel_W_error_ESSsec_sampling, 3), pm(g.test_nll_ESSsec_sampling, 3),
             pm(g.rel_W_error_ESSsec_total, 3), pm(g.test_nll_ESSsec_total, 3)]
    print(LABEL[k] + "\n" + "\n".join("& " + c for c in cells) + "\n\\\\")

print("\n% ===== tab:d5-m50-seed-recovery-corrected: rows between \\midrule and \\bottomrule =====")
for k in ORDER:
    g = df[(df["Method"] == k) & (df["status"].fillna("OK") == "OK")]
    cov = [100 * g[c] if g[c].max() <= 1.0 else g[c]
           for c in ("entry_coverage", "trace_coverage", "eigenvalue_coverage")]
    cells = [pm(g.posterior_mean_rel_W_error, 3)] + [pm(c, 1) for c in cov] + \
            [f"${g.max_rhat.median():.3f}$", f"${int((g.max_rhat <= 1.05).sum())}/{len(g)}$"]
    print(LABEL[k] + "\n" + "\n".join("& " + c for c in cells) + "\n\\\\")

ms = {k: df[df["Method"] == k]["ms"] for k in ORDER}
print("\n% SM5 opening paragraph: production ms/trans " + ", ".join(
    f"{k} ${ms[k].mean():.2f}\\pm{ms[k].std(ddof=1):.2f}$" for k in ORDER))
