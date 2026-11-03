# ConeMALA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19103698.svg)](https://doi.org/10.5281/zenodo.19103698)

Code and results for

> P. Dey, *ConeMALA: Exact and Unit-Invariant Langevin Sampling for Uncertainty
> Quantification in Matrix-Weighted Graphs*.

ConeMALA is a Metropolis-adjusted Langevin sampler for graph models whose edge
weights are symmetric positive-definite (SPD) matrices. It proposes each edge
along the affine-invariant geometry of the SPD cone, drifts along the full
posterior score of the shared Gaussian precision matrix, and targets the
posterior exactly.

## Contents

```
experiments/   scripts that run the samplers (one per study)
  boundary/    boundary-stress and drift-ablation studies (Supplement SM7-SM9)
analysis/      scripts that build every table and figure from results/
results/       summary files of the runs reported in the paper
configs/       configuration of the boundary-stress studies
make_paper_outputs.sh   rebuilds all tables and figures from results/
```

All scripts are self-contained Python (NumPy, SciPy, pandas, Matplotlib).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild the tables and figures (seconds)

```bash
./make_paper_outputs.sh
```

This writes every table (LaTeX), figure (PDF) and number quoted in the text
to `paper_outputs/`, using the summary files in `results/`. No sampler is run.

## Paper item → code

| Paper | Experiment | Summary in `results/` | Tables / figures |
|---|---|---|---|
| §6.1 Tables 6.1–6.2, Fig. 6.1 (units and basis) | `run_scaling_experiment.py` | `units_and_basis/scaling_summary.csv` | `analysis/units_and_basis_tables.py` |
| §6.2 Fig. 6.2, Tables SM6.1–SM6.3 (graph size) | `run_d5_m_sweep_wishart_with_fisher.py` | `graph_size/combined_summary_d5_m_sweep.csv` | `analysis/plot_mala_scaling.py`, `analysis/graph_size_tables.py` |
| §6.2 Fig. 6.3, Tables SM3.1–SM3.2 (block dimension) | `run_m50_d_sweep_wishart_with_fisher.py` | `block_dimension/combined_summary_m50_d_sweep.csv` | `analysis/plot_mala_scaling.py`, `analysis/block_dimension_tables.py` |
| §6.2 doubled budget at d = 5, 8 | `run_m50_d_sweep_conemala_longrun.py` | `block_dimension/longrun/` | written by the experiment |
| §6.2–6.3, Tables SM5.1–SM5.2 (five datasets) | `run_d5_m50_seed_robustness_wishart.py` | `repeated_datasets/robustness_d5_m50_by_seed.csv` | `analysis/repeated_dataset_tables.py` |
| SM7, Table SM7.1, Fig. SM7.1 (drift ablation) | `boundary/run_boundary_drift_ablation.py` | `boundary/drift_ablation/` | written by the experiment |
| SM8–SM9, Table SM8.1, Fig. SM9.1 (boundary stress) | `boundary/run_boundary_*_typical_set.py` | `boundary/` | `boundary/build_boundary_outputs.py` |

## Run the experiments

Every experiment writes to the folder given by `--outdir` / `--output-dir`
and can be restarted with the same command (`--resume`). Runtimes are for one
laptop core; the production ESS/s in the paper is wall-clock time, so run on an
otherwise idle machine. On macOS, prefix long runs with `caffeinate -i`.
The four samplers compared in §6.2 are passed as

```bash
M=ConeMALA,FisherAdaptiveMALA,EuclideanMALA,GenericRMALA
```

**Graph size** (§6.2; several hours):
```bash
python3 -u experiments/run_d5_m_sweep_wishart_with_fisher.py --m_list 20,50,100 \
    --methods $M --skip_edge_diagnostics --outdir outputs/graph_size --resume
```

**Block dimension** (§6.2; several hours), then the doubled ConeMALA budget at d = 5, 8:
```bash
python3 -u experiments/run_m50_d_sweep_wishart_with_fisher.py --d_list 2,5,8,10 \
    --methods $M --skip_edge_diagnostics --outdir outputs/block_dimension --resume
python3 -u experiments/run_m50_d_sweep_conemala_longrun.py \
    --input-root outputs/block_dimension --output-root outputs/block_dimension_longrun \
    --d-list 5,8 --retained 24000
```

**Five datasets** (§6.2–6.3; several hours):
```bash
python3 -u experiments/run_d5_m50_seed_robustness_wishart.py --seeds 11,22,33,44,55 \
    --outdir outputs/repeated_datasets --resume
```

**Units and basis** (§6.1; the NUTS runs take up to nine hours each):
```bash
U="python3 -u experiments/run_scaling_experiment.py --n_replicates 3 --qd_reps 2 --outdir outputs/units_and_basis --resume"
$U --methods LogCholeskyMALA,LogCholeskyMALA_DiagMetric
$U --methods ConeMALA --retained 48000
$U --methods LogCholeskyNUTS,LogCholeskyNUTS_IdentityMetric \
   --variants V0_identity_rep1,V3_scale_dec2_rotated,V3_scale_dec2_QD
$U --special_decade 3 --methods ConeMALA --retained 48000 --variants V3_scale_dec3_QD
$U --special_decade 3 --methods LogCholeskyNUTS --retained 3000 --variants V3_scale_dec3_QD
$U --summarize_only
```
`python3 experiments/run_scaling_experiment.py --selftest` checks in seconds that
the transformed posteriors are exact transports of the original one.

**Boundary stress and drift ablation** (Supplement SM7–SM9; run from `experiments/boundary/`):
```bash
cd experiments/boundary
B="--config ../../configs/boundary_base_config.json --profile paper --epsilons 1e-2 \
   --prior-mode boundary-friendly --boundary-nu-offset 1.0 --preburn-steps 8000 \
   --settle-steps 1000 --tune-warmup 500 --tune-steps 1000"
L="--production-warmup 10000 --retained 20000"
python3 -u run_boundary_conemala_typical_set.py   $B $L --output-dir ../../outputs/boundary/conemala
python3 -u run_boundary_fisher_typical_set.py     $B    --output-dir ../../outputs/boundary/fisher
python3 -u run_boundary_euclidean_typical_set.py  $B    --output-dir ../../outputs/boundary/euclidean
python3 -u run_boundary_retraction_typical_set.py $B $L --output-dir ../../outputs/boundary/retraction
python3 -u run_boundary_drift_ablation.py         $B $L --output-dir ../../outputs/boundary/drift_ablation --make-plots
python3 build_boundary_outputs.py --cone-dir ../../outputs/boundary/conemala \
    --fisher-dir ../../outputs/boundary/fisher --euclidean-dir ../../outputs/boundary/euclidean \
    --retraction-dir ../../outputs/boundary/retraction --output-dir ../../outputs/boundary/combined
```

To rebuild the paper's tables from a fresh run, point the analysis scripts at
the new summaries, e.g.
`python3 analysis/graph_size_tables.py outputs/graph_size/combined_summary_d5_m_sweep_with_fisher.csv`.

## Settings used throughout

* Four chains per run; for §6.2, 4,000 discarded and 12,000 retained
  transitions per chain.
* Step sizes are chosen in discarded pilot runs and then frozen.
  Fisher-adaptive MALA learns, pools and freezes a dense preconditioner during
  discarded adaptation, updating it at every adaptive iteration as in
  Algorithm 1 of Titsias (2023).
* Production ESS/s divides the ESS by the runtime of the retained transitions
  only; end-to-end ESS/s also counts burn-in, adaptation and tuning.
* Convergence criterion: maximum classical split-R̂ over relative-W error,
  held-out NLL, log det Q(W) and Φ(W) at most 1.05 (§6.2); rank-normalized
  split-R̂ over 43 invariant functionals (§6.1).

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
