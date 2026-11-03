#!/usr/bin/env python3
"""Feature-basis invariance experiment (variants V0-V2).

Question
--------
Does the performance of a sampler depend on an arbitrary choice that has no
scientific meaning: the ordering or the orthogonal basis of the d features at
every graph node?

Construction
------------
One base problem (cycle graph, m nodes, d features) is generated exactly as in
``run_m50_d_sweep_wishart_with_fisher.py``.  For an orthogonal d x d matrix A
the data are re-expressed in a new feature basis,

    y'  = (I_m kron A) y,     W*'_e = A W*_e A^T,     y_test' likewise.

With R = r I (so R' = R) and the trace penalty lambda*tr(W_e) (invariant under
orthogonal congruence) the transformed posterior is *exactly* the pushforward
of the original one under  W_e -> A W_e A^T.  Hence any difference between
variants is caused by the sampler, not by the statistical problem.

Variants
--------
  V0_identity_rep<k>   untransformed problem; replicates use other chain seeds
                       and give the Monte-Carlo noise floor of every metric
  V1_perm_<k>          feature permutations (V1_perm_rev is the reversal)
  V2_orth_<k>          Haar-random orthogonal changes of basis

Methods (4 chains each, all started from the transported initial states)
  ConeMALA                    AIRM, chart-free, scalar h (pilot-tuned)
  LogCholeskyMALA             log-Cholesky chart, scalar h (pilot-tuned)
  LogCholeskyMALA_DiagMetric  log-Cholesky chart, windowed diagonal metric
  LogCholeskyNUTS             log-Cholesky chart, Stan-style NUTS, diag metric
  LogCholeskyNUTS_IdentityMetric  same with the identity metric

Metrics
-------
Primary metrics are *basis invariant* functionals, so that an exactly
equivariant sampler has the same ESS in every variant: Phi, log det Q, test
NLL, ||W - W*||_F, and per edge log det W_e, lambda_min(W_e), lambda_max(W_e)
and the AIRM distance d(W_e, W*_e).  Worst-entry ESS over the raw matrix
entries is also reported, but it is NOT invariant (the entries themselves
change with the basis), so it is a secondary column (prefix ``edge_``).

ESS/s uses the end-to-end clock (burn-in/warm-up included, discarded step-size
pilots excluded), summed over chains, as in the d-sweep.

This file imports the samplers and diagnostics from
``run_m50_d_sweep_wishart_with_fisher.py`` (keep both in one folder); that
file is not modified.

Examples
--------
  python3 run_invariance_experiment.py --selftest
  python3 -u run_invariance_experiment.py --m 20 --d 5 --outdir out_inv --resume
  # stage it: one method at a time
  python3 -u run_invariance_experiment.py --methods ConeMALA,LogCholeskyMALA --outdir out_inv --resume
"""

from __future__ import annotations

INVARIANCE_VERSION = "1.0"

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m50_d_sweep_wishart_with_fisher as base  # noqa: E402

Array = np.ndarray

CONE = "ConeMALA"
CONE_EDGE = base.CONE_EDGE_METHOD  # "ConeMALA_EdgeStep": per-edge, congruence-invariant steps
INV_METHODS = [
    CONE,
    CONE_EDGE,
    base.LCHOL_METHOD,
    base.LCHOL_DIAG_METHOD,
    base.NUTS_METHOD,
    base.NUTS_IDENTITY_METHOD,
]
SEED_OFFSET = {
    CONE: 1_000_000,
    CONE_EDGE: 1_000_000,  # same random streams as ConeMALA: paired comparison
    base.LCHOL_METHOD: 7_000_000,
    base.LCHOL_DIAG_METHOD: 7_000_000,
    base.NUTS_METHOD: 9_000_000,
    base.NUTS_IDENTITY_METHOD: 9_000_000,
}
EDGE_FAMILIES = ("logdet_We", "lam_min_We", "lam_max_We", "airm_dist_We")


# =============================================================================
# Variants and problem transport
# =============================================================================

def haar_orthogonal(d: int, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))[None, :]


def permutation_matrix(perm: Sequence[int]) -> Array:
    d = len(perm)
    a = np.zeros((d, d))
    a[np.arange(d), np.asarray(perm)] = 1.0
    return a


def make_variants(
    d: int, n_rep: int, n_perm: int, n_orth: int, seed: int
) -> List[Dict]:
    """List of {name, family, rep, A}.  A is orthogonal; replicates use A=I."""
    rng = np.random.default_rng(seed + 99)
    variants: List[Dict] = []
    for k in range(max(n_rep, 1)):
        variants.append(
            {"name": f"V0_identity_rep{k + 1}", "family": "identity",
             "rep": k, "A": np.eye(d)}
        )
    seen = {tuple(range(d))}
    perms: List[Tuple[int, ...]] = []
    rev = tuple(range(d - 1, -1, -1))
    if rev not in seen:
        perms.append(rev)
        seen.add(rev)
    max_perms = math.factorial(d) - 1
    tries = 0
    while len(perms) < min(n_perm, max_perms) and tries < 1000:
        tries += 1
        p = tuple(int(i) for i in rng.permutation(d))
        if p not in seen:
            perms.append(p)
            seen.add(p)
    for k, p in enumerate(perms[:n_perm]):
        name = "V1_perm_rev" if p == rev else f"V1_perm_{k}"
        variants.append(
            {"name": name, "family": "permutation", "rep": 0,
             "A": permutation_matrix(p)}
        )
    for k in range(n_orth):
        variants.append(
            {"name": f"V2_orth_{k + 1}", "family": "orthogonal", "rep": 0,
             "A": haar_orthogonal(d, rng)}
        )
    return variants


def build_base_problem(args: argparse.Namespace) -> Dict:
    m, d = int(args.m), int(args.d)
    rng = np.random.default_rng(args.seed + 17 * m)
    edges = base.make_cycle_graph(m)
    w_star = base.sample_ground_truth_w(len(edges), d, rng, scale=args.truth_scale)
    r_matrix = args.r_scale * np.eye(m * d)
    q_star = base.block_laplacian(w_star, edges, m, d) + r_matrix
    y_train = base.sample_graph_signals(q_star, args.n_train, rng)
    y_test = base.sample_graph_signals(q_star, args.n_test, rng)
    return {
        "m": m, "d": d, "edges": edges, "w_star": w_star,
        "r_matrix": r_matrix, "y_train": y_train, "y_test": y_test,
    }


def congruence(a: Array, w: Array) -> Array:
    """W_e -> A W_e A^T for every edge (w has shape (E, d, d))."""
    out = a @ w @ a.T
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def transport_problem(problem: Dict, a: Array, args: argparse.Namespace) -> Dict:
    m, d = problem["m"], problem["d"]
    if not np.allclose(a @ a.T, np.eye(d), atol=1e-10):
        raise ValueError(
            "Only orthogonal A is supported here (R=rI and the trace penalty "
            "are invariant only under orthogonal congruence)."
        )
    big_a = np.kron(np.eye(m), a)
    r_new = big_a @ problem["r_matrix"] @ big_a.T
    if not np.allclose(r_new, problem["r_matrix"], atol=1e-10):
        raise ValueError("R is not invariant under this A; model not equivariant.")
    post = base.PSDGraphPosterior(
        edges=problem["edges"], m=m, d=d,
        y_train=big_a @ problem["y_train"],
        r_matrix=r_new,
        lambda_tr=args.lambda_tr,
        alpha_logdet=(0.5 * (d + 1.0) if args.alpha_logdet is None else args.alpha_logdet),
    )
    return {
        "posterior": post,
        "w_star": congruence(a, problem["w_star"]),
        "y_test": big_a @ problem["y_test"],
    }


# =============================================================================
# Basis-invariant functionals
# =============================================================================

def _inv_sqrt_blocks(w_star: Array) -> Array:
    out = []
    for block in w_star:
        vals, vecs = np.linalg.eigh(0.5 * (block + block.T))
        out.append((vecs / np.sqrt(vals)[None, :]) @ vecs.T)
    return np.asarray(out)


def edge_functionals(
    samples: Array, w_star: Array, chunk: int = 1500
) -> Dict[str, Array]:
    """Per-edge invariant functionals, each of shape (n, E)."""
    sinv = _inv_sqrt_blocks(w_star)[None]
    parts: Dict[str, List[Array]] = {k: [] for k in EDGE_FAMILIES}
    for start in range(0, samples.shape[0], chunk):
        block = samples[start:start + chunk]
        eig = np.maximum(np.linalg.eigvalsh(block), 1e-300)
        parts["logdet_We"].append(np.log(eig).sum(-1))
        parts["lam_min_We"].append(eig[..., 0])
        parts["lam_max_We"].append(eig[..., -1])
        mid = sinv @ block @ sinv
        mid = 0.5 * (mid + np.swapaxes(mid, -1, -2))
        ev = np.maximum(np.linalg.eigvalsh(mid), 1e-300)
        parts["airm_dist_We"].append(np.sqrt((np.log(ev) ** 2).sum(-1)))
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def invariant_columns(
    results: Sequence["base.ChainResult"],
    w_star: Array,
    scalar_cols: Dict[str, float],
    runtime_total: float,
    runtime_sampling: float,
) -> Dict[str, float]:
    cols: Dict[str, float] = {}
    func = [edge_functionals(r.samples, w_star) for r in results]
    bulk_all = [scalar_cols[f"{n}_ESS_bulk"] for n in base.SCALAR_DIAGNOSTIC_OBSERVABLES]
    tail_all = [scalar_cols[f"{n}_ESS_tail"] for n in base.SCALAR_DIAGNOSTIC_OBSERVABLES]
    rhat_all = [scalar_cols[f"{n}_Rhat_rank"] for n in base.SCALAR_DIAGNOSTIC_OBSERVABLES]
    for fam in EDGE_FAMILIES:
        diag = base.mcmc_rank_diagnostics([f[fam] for f in func])
        cols[f"inv_{fam}_ESS_bulk_min"] = float(np.nanmin(diag["ess_bulk"]))
        cols[f"inv_{fam}_ESS_tail_min"] = float(np.nanmin(diag["ess_tail"]))
        cols[f"inv_{fam}_ESS_bulk_median"] = float(np.nanmedian(diag["ess_bulk"]))
        cols[f"inv_{fam}_Rhat_max"] = float(np.nanmax(diag["rhat_rank"]))
        cols[f"inv_{fam}_ESSsec_bulk_median_total"] = (
            cols[f"inv_{fam}_ESS_bulk_median"] / max(runtime_total, 1e-12)
        )
        bulk_all.append(cols[f"inv_{fam}_ESS_bulk_min"])
        tail_all.append(cols[f"inv_{fam}_ESS_tail_min"])
        rhat_all.append(cols[f"inv_{fam}_Rhat_max"])
    cols["inv_ESS_bulk_min"] = float(np.nanmin(bulk_all))
    cols["inv_ESS_tail_min"] = float(np.nanmin(tail_all))
    cols["inv_Rhat_max"] = float(np.nanmax(rhat_all))
    cols["inv_ESSsec_bulk_min_total"] = cols["inv_ESS_bulk_min"] / max(runtime_total, 1e-12)
    cols["inv_ESSsec_bulk_min_sampling"] = cols["inv_ESS_bulk_min"] / max(runtime_sampling, 1e-12)
    return cols


# =============================================================================
# One (variant, method) cell
# =============================================================================

def run_cell(
    method: str,
    variant: Dict,
    prob: Dict,
    initial_states: Sequence[Array],
    h_lchol: Optional[float],
    args: argparse.Namespace,
    cell_dir: Path,
) -> Tuple[pd.DataFrame, Optional[float]]:
    post = prob["posterior"]
    m, d = post.m, post.d
    rep_shift = 1000 * int(variant["rep"])
    seeds = [
        10_000 + 100 * m + c + SEED_OFFSET[method] + rep_shift
        for c in range(args.n_chains)
    ]
    preburn, retained = int(args.preburn), int(args.retained)
    h_used = float("nan")
    t_pilot = 0.0

    if method in (CONE, CONE_EDGE):
        t0 = time.perf_counter()
        edge_scale = None
        if method == CONE_EDGE:
            # multipliers from the Riemannian-GD mode; counted in the pilot time
            w_ref = base.cone_reference_point(post, initial_states[0])
            edge_scale, kappa = base.cone_edge_scales(post, w_ref)
            pd.DataFrame({"edge": range(len(kappa)), "kappa": kappa,
                          "step_multiplier": edge_scale}).to_csv(
                cell_dir / "cone_edge_step_multipliers.csv", index=False)
        h_used, rows = base.tune_fixed_method_h(
            method=method, posterior=post, anchors=initial_states,
            h_center=float(args.h_cone_center),
            target_accept=args.fixed_target_accept,
            acceptance_tolerance=args.fixed_acceptance_tolerance,
            tune_warmup=args.fixed_tune_warmup, tune_steps=args.fixed_tune_steps,
            tune_chains=args.fixed_tune_chains,
            factors=tuple(float(v) for v in args.fixed_grid_factors.split(",")),
            max_bracket_steps=args.fixed_max_bracket_steps,
            bisection_steps=args.fixed_bisection_steps,
            seed_base=70_000_000 + 100 * m + rep_shift,
            edge_scale=edge_scale,
        )
        t_pilot = time.perf_counter() - t0
        pd.DataFrame(rows).to_csv(cell_dir / f"tuning_{method}.csv", index=False)
        results = [
            base.cone_mala_chain(post, initial_states[c], preburn, retained, h_used,
                                 seeds[c], edge_scale)
            for c in range(args.n_chains)
        ]
    elif method in (base.LCHOL_METHOD, base.LCHOL_DIAG_METHOD):
        if h_lchol is None:
            t0 = time.perf_counter()
            h_lchol, rows = base.tune_lchol_step(
                post, initial_states, args,
                seed_base=80_000_000 + 100 * m + rep_shift,
            )
            t_pilot = time.perf_counter() - t0
            pd.DataFrame(rows).to_csv(cell_dir / "tuning_LogCholeskyMALA.csv", index=False)
            (cell_dir / "h_lchol.json").write_text(json.dumps({"h_lchol": h_lchol}))
        h_used = float(h_lchol)
        results = base.run_release_method_chains(
            method, post, initial_states, preburn, retained, h_used, seeds, args
        )
    else:
        results = base.run_release_method_chains(
            method, post, initial_states, preburn, retained, float("nan"), seeds, args
        )
        h_used = float(np.mean([r.h_final for r in results]))

    rt_total = float(sum(r.runtime_total_sec for r in results))
    rt_samp = float(sum(r.runtime_sampling_sec for r in results))
    obs = [
        base.observables(r.samples, post, prob["w_star"], prob["y_test"])
        for r in results
    ]
    edge_chains = (
        None if args.skip_edge_diagnostics
        else [base.edge_entry_matrix(r.samples) for r in results]
    )
    cols = base.extended_diagnostic_columns(obs, edge_chains, rt_total, rt_samp)
    cols.update(invariant_columns(results, prob["w_star"], cols, rt_total, rt_samp))

    means = {
        f"{name}_mean": float(np.mean([np.mean(o[name]) for o in obs]))
        for name in base.SCALAR_DIAGNOSTIC_OBSERVABLES
    }
    sds = {
        f"{name}_sd": float(np.std(np.concatenate([o[name] for o in obs])))
        for name in base.SCALAR_DIAGNOSTIC_OBSERVABLES
    }
    row = {
        "variant": variant["name"], "family": variant["family"],
        "rep": variant["rep"], "Method": method,
        "tier": base.METHOD_TIER.get(method, ""),
        "m": m, "d": d, "n_chains": args.n_chains,
        "preburn": preburn, "retained": retained,
        "h": h_used,
        "h_final_mean": float(np.mean([r.h_final for r in results])),
        "acceptance": float(np.mean([r.acceptance for r in results])),
        "runtime_total_sec": rt_total, "runtime_sampling_sec": rt_samp,
        "pilot_runtime_sec_excluded": t_pilot,
        "numerical_failure_rate": float(np.mean([r.numerical_failure_rate for r in results])),
        "preconditioner_condition": float(np.mean([r.preconditioner_condition for r in results])),
        **base.nuts_extra_columns(results),
        **means, **sds, **cols,
    }
    if args.save_samples:
        np.savez_compressed(
            cell_dir / f"samples_{method}.npz",
            **{f"chain{c}": r.samples.astype(np.float32) for c, r in enumerate(results)},
        )
    return pd.DataFrame([row]), h_lchol


# =============================================================================
# Summaries
# =============================================================================

SPREAD_METRICS = [
    ("inv_ESS_bulk_min", "min invariant bulk ESS"),
    ("inv_ESSsec_bulk_min_total", "min invariant bulk ESS/s (total clock)"),
    ("inv_airm_dist_We_ESS_bulk_median", "median-over-edges AIRM-distance bulk ESS"),
    ("inv_airm_dist_We_ESSsec_bulk_median_total", "median-over-edges AIRM-distance bulk ESS/s"),
    ("inv_logdet_We_ESS_bulk_median", "median-over-edges log det W_e bulk ESS"),
    ("inv_logdet_We_ESSsec_bulk_median_total", "median-over-edges log det W_e bulk ESS/s"),
    ("Phi_ESS_bulk", "Phi bulk ESS"),
    ("Phi_ESSsec_bulk_total", "Phi bulk ESS/s (total clock)"),
    ("edge_ESS_bulk_min", "raw-entry min bulk ESS (NOT invariant)"),
    ("edge_ESSsec_bulk_min_total", "raw-entry min bulk ESS/s (NOT invariant)"),
    ("inv_Rhat_max", "max invariant rank R-hat"),
]


def _ratio(x: Sequence[float]) -> float:
    x = np.asarray([v for v in x if np.isfinite(v) and v > 0], dtype=float)
    return float(x.max() / x.min()) if x.size >= 2 else float("nan")


def summarize(outdir: Path) -> None:
    rows = [pd.read_csv(p) for p in sorted(outdir.glob("*/row_*.csv"))]
    if not rows:
        print("No row files found.")
        return
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(outdir / "invariance_summary.csv", index=False)

    spread = []
    for method, g in table.groupby("Method"):
        ident = g[g["family"] == "identity"]
        for col, label in SPREAD_METRICS:
            if col not in g:
                continue
            vals = g[col].astype(float)
            base_vals = ident[col].astype(float)
            ref = float(np.exp(np.nanmean(np.log(base_vals)))) if len(base_vals) else float("nan")
            entry = {
                "Method": method, "metric": col, "label": label,
                "n_variants": int(vals.notna().sum()),
                "min": float(vals.min()), "max": float(vals.max()),
                "max_over_min_all": _ratio(vals),
                "max_over_min_identity_replicates_noise": _ratio(base_vals),
                "identity_geomean": ref,
            }
            for fam in ("permutation", "orthogonal"):
                fv = g[g["family"] == fam][col].astype(float)
                entry[f"{fam}_geomean_over_identity"] = (
                    float(np.exp(np.nanmean(np.log(fv))) / ref)
                    if len(fv) and ref > 0 else float("nan")
                )
                entry[f"{fam}_min_over_identity"] = (
                    float(fv.min() / ref) if len(fv) and ref > 0 else float("nan")
                )
            spread.append(entry)
    pd.DataFrame(spread).to_csv(outdir / "invariance_spread.csv", index=False)

    show = pd.DataFrame(spread)
    for metric, title in (
        ("inv_airm_dist_We_ESSsec_bulk_median_total",
         "median-over-edges AIRM-distance bulk ESS/s (more stable than a minimum)"),
        ("inv_ESSsec_bulk_min_total", "min invariant bulk ESS/s (extreme-value metric, noisier)"),
    ):
        _print_spread(show, metric, title)
    print(f"\nWrote {outdir / 'invariance_summary.csv'} and invariance_spread.csv")


def _print_spread(show: pd.DataFrame, metric: str, title: str) -> None:
    show = show[show["metric"] == metric]
    if len(show):
        print(f"\n{title}; total clock; across variants")
        print(
            show[
                ["Method", "n_variants", "min", "max", "max_over_min_all",
                 "max_over_min_identity_replicates_noise",
                 "permutation_geomean_over_identity",
                 "orthogonal_geomean_over_identity"]
            ].to_string(index=False, float_format=lambda v: f"{v:.3g}")
        )


# =============================================================================
# Self-test: exact equivariance of the target, one-step acceptance at a fixed state
# =============================================================================

def selftest(args: argparse.Namespace) -> None:
    problem = build_base_problem(args)
    d, m = problem["d"], problem["m"]
    post0 = base.PSDGraphPosterior(
        edges=problem["edges"], m=m, d=d, y_train=problem["y_train"],
        r_matrix=problem["r_matrix"], lambda_tr=args.lambda_tr,
        alpha_logdet=0.5 * (d + 1.0) if args.alpha_logdet is None else args.alpha_logdet,
    )
    variants = [
        v for v in make_variants(d, 1, 1, 1, args.seed) if v["family"] != "identity"
    ]
    rng = np.random.default_rng(5)
    edge_count = len(problem["edges"])
    w_rand = np.asarray([
        base.project_spd(0.3 * (lambda a: a @ a.T)(rng.normal(size=(d, d))) + 0.2 * np.eye(d))
        for _ in range(edge_count)
    ])
    w0 = [base.make_initial_w(edge_count, d, 0)]
    h_cone, h_lc = args.h_cone_center, args.lchol_h_center

    # a state near the typical set: short ConeMALA burn-in on the base problem
    burn = base.cone_mala_chain(post0, w0[0], 0, 400, h_cone, 1).samples[-1]

    print("1) exact equivariance of the target (should be ~1e-12)")
    ok = True
    for v in variants:
        a = v["A"]
        t = transport_problem(problem, a, args)
        post1 = t["posterior"]
        phi0, g0 = post0.phi_and_grad(w_rand)
        w1 = congruence(a, w_rand)
        phi1, g1 = post1.phi_and_grad(w1)
        e_phi = abs(phi0 - phi1)
        e_g = float(np.max(np.abs(g1 - np.asarray([a @ x @ a.T for x in g0]))))
        big_a = np.kron(np.eye(m), a)
        e_q = float(np.max(np.abs(post1.q(w1) - big_a @ post0.q(w_rand) @ big_a.T)))
        good = max(e_phi / max(abs(phi0), 1.0), e_g, e_q) < 1e-8
        ok &= good
        print(f"   {v['name']:<14s} |dPhi|={e_phi:.2e} max|grad err|={e_g:.2e} max|Q err|={e_q:.2e}  {'OK' if good else 'FAIL'}")

    print("\n2) one-step acceptance at a fixed near-typical state (same h; 600 proposals)")
    print("   ConeMALA must agree across bases (equivariant in distribution);")
    print("   LC-MALA is allowed to differ.")
    n_prop = 600

    def accept_cone(post, w) -> float:
        phi, grad = post.phi_and_grad(w)
        r = np.random.default_rng(11)
        return float(np.mean([
            base._cone_transition(post, w, phi, grad, h_cone, r)[3] for _ in range(n_prop)
        ]))

    def accept_lc(post, w) -> float:
        tgt = base.LogCholeskyTarget(post)
        theta = tgt.from_weights(w)
        pot, grad = tgt.potential_and_grad(theta)
        r = np.random.default_rng(11)
        return float(np.mean([
            base._lchol_transition(tgt, theta, pot, grad, h_lc, r)[3] for _ in range(n_prop)
        ]))

    ref_c, ref_l = accept_cone(post0, burn), accept_lc(post0, burn)
    se = math.sqrt(max(ref_c * (1 - ref_c), 1e-4) / n_prop)
    print(f"   {'identity':<14s} ConeMALA acc={ref_c:.3f}   LC-MALA acc={ref_l:.3f}")
    for v in variants:
        a = v["A"]
        post1 = transport_problem(problem, a, args)["posterior"]
        w1 = congruence(a, burn)
        ac, al = accept_cone(post1, w1), accept_lc(post1, w1)
        z = (ac - ref_c) / (math.sqrt(2) * se)
        print(f"   {v['name']:<14s} ConeMALA acc={ac:.3f} (z={z:+.2f})   LC-MALA acc={al:.3f}")
    print("\nself-test", "passed" if ok else "FAILED (target equivariance)")
    if not ok:
        sys.exit(1)


# =============================================================================
# Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=20, help="Cycle-graph nodes (|E|=m).")
    p.add_argument("--d", type=int, default=5, help="Feature dimension per node.")
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--n_test", type=int, default=80)
    p.add_argument("--n_chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="out_invariance")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_samples", action="store_true")
    p.add_argument("--selftest", action="store_true",
                   help="Check target equivariance and one-step acceptance, then exit.")
    p.add_argument("--summarize_only", action="store_true")
    p.add_argument("--n_replicates", type=int, default=2,
                   help="Identity-basis runs with different chain seeds (noise floor).")
    p.add_argument("--n_perm", type=int, default=3)
    p.add_argument("--n_orth", type=int, default=3)
    p.add_argument("--variants", type=str, default="all",
                   help="Comma-separated variant names to run (default all).")
    p.add_argument("--truth_scale", type=float, default=0.25)
    p.add_argument("--r_scale", type=float, default=0.75)
    p.add_argument("--lambda_tr", type=float, default=0.05)
    p.add_argument("--alpha_logdet", type=float, default=None)
    p.add_argument("--preburn", type=int, default=int(base.D5_REFERENCE_SETTINGS["preburn"]))
    p.add_argument("--retained", type=int, default=int(base.D5_REFERENCE_SETTINGS["retained"]))
    p.add_argument("--h_cone_center", type=float, default=base.D5_REFERENCE_SETTINGS["h_cone"])
    p.add_argument("--fixed_target_accept", type=float, default=0.574)
    p.add_argument("--fixed_acceptance_tolerance", type=float, default=0.08)
    p.add_argument("--fixed_tune_warmup", type=int, default=100)
    p.add_argument("--fixed_tune_steps", type=int, default=400)
    p.add_argument("--fixed_tune_chains", type=int, default=2)
    p.add_argument("--fixed_grid_factors", type=str, default="0.125,0.25,0.5,1,2,4")
    p.add_argument("--fixed_max_bracket_steps", type=int, default=20)
    p.add_argument("--fixed_bisection_steps", type=int, default=6)
    base.add_release_arguments(p)
    for action in p._actions:  # reuse --methods, re-document it for this script
        if "--methods" in action.option_strings:
            action.help = (
                "Comma-separated subset of: " + ", ".join(INV_METHODS)
                + " (default: all five)."
            )
    return p


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.selftest:
        selftest(args)
        return
    if args.summarize_only:
        summarize(outdir)
        return

    methods = base.parse_method_list(args.methods, INV_METHODS, default=INV_METHODS)
    variants = make_variants(args.d, args.n_replicates, args.n_perm, args.n_orth, args.seed)
    if args.variants.strip().lower() != "all":
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [w for w in wanted if w not in {v["name"] for v in variants}]
        if unknown:
            raise SystemExit(f"Unknown variants {unknown}; available: {[v['name'] for v in variants]}")
        variants = [v for v in variants if v["name"] in wanted]

    problem = build_base_problem(args)
    edge_count = len(problem["edges"])
    (outdir / "config.json").write_text(
        json.dumps({**vars(args), "version": INVARIANCE_VERSION,
                    "methods": methods, "variants": [v["name"] for v in variants]},
                   indent=2) + "\n")
    print(f"m={args.m}, d={args.d}, |E|={edge_count}, D={edge_count * args.d * (args.d + 1) // 2}, "
          f"chains={args.n_chains}, preburn={args.preburn}, retained={args.retained}")
    print("methods :", methods)
    print("variants:", [v["name"] for v in variants], flush=True)

    base_w0 = [base.make_initial_w(edge_count, args.d, c) for c in range(args.n_chains)]
    for variant in variants:
        cell_dir = outdir / variant["name"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        np.save(cell_dir / "A.npy", variant["A"])
        prob = transport_problem(problem, variant["A"], args)
        initial = [congruence(variant["A"], w) for w in base_w0]
        h_file = cell_dir / "h_lchol.json"
        h_lchol = json.loads(h_file.read_text())["h_lchol"] if (args.resume and h_file.exists()) else None
        print("\n" + "=" * 90 + f"\nvariant {variant['name']} ({variant['family']})\n" + "=" * 90, flush=True)
        # LC-MALA first so that its tuned step is reused by the diag-metric variant
        for method in sorted(methods, key=lambda s: INV_METHODS.index(s)):
            row_path = cell_dir / f"row_{method}.csv"
            if args.resume and row_path.exists():
                print(f"  {method}: found {row_path.name}, skipping", flush=True)
                continue
            t0 = time.perf_counter()
            print(f"  {method} ...", flush=True)
            df, h_lchol = run_cell(method, variant, prob, initial, h_lchol, args, cell_dir)
            df.to_csv(row_path, index=False)
            r = df.iloc[0]
            print(
                f"  {method}: acc={r['acceptance']:.3f} h={r['h']:.3e} "
                f"minESS_inv(bulk)={r['inv_ESS_bulk_min']:.0f} "
                f"ESS/s_inv={r['inv_ESSsec_bulk_min_total']:.3g} "
                f"Rhat_inv={r['inv_Rhat_max']:.3f}  [{time.perf_counter() - t0:.0f}s]",
                flush=True,
            )
        summarize(outdir)


if __name__ == "__main__":
    main()
