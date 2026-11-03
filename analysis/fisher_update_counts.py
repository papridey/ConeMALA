#!/usr/bin/env python3
"""Number of rank-one Fisher preconditioner updates actually made during
discarded adaptation, per chain, relative to the parameter dimension q.

An update needs the gradient at the proposal, so proposals outside the SPD
cone contribute none.  Reads every fisher_adaptation.csv below the given
output folders.

  python3 fisher_update_counts.py <outdir of an experiment> [more outdirs]
"""
import re
import sys
from pathlib import Path

import pandas as pd

rows = []
for root in sys.argv[1:]:
    for f in sorted(Path(root).glob("graph_posterior_*/results/fisher_adaptation.csv")):
        tag = f.parent.parent.name                      # graph_posterior_d5_m50 / _m50_d8
        m = int(re.search(r"m(\d+)", tag).group(1))
        d = int(re.search(r"d(\d+)", tag).group(1))
        q = m * d * (d + 1) // 2                         # cycle graph: |E| = m
        a = pd.read_csv(f)
        rows.append(dict(folder=root, m=m, d=d, q=q,
                         updates_mean=a["n_updates"].mean(),
                         updates_min=a["n_updates"].min(),
                         updates_per_q=a["n_updates"].mean() / q,
                         adapt_invalid_rate=a["invalid_rate"].mean()))
if not rows:
    raise SystemExit("no fisher_adaptation.csv found")
print(pd.DataFrame(rows).round(3).to_string(index=False))
