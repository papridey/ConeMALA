#!/usr/bin/env python3
"""Feature-scaling experiment (variant V3): does the chart/metric survive changes of units?

Why this experiment
-------------------
V0-V2 (run_invariance_experiment.py) permuted / rotated a problem whose random
generator is itself rotation-invariant, so no basis was special.  Real data
are different: features come in different units.  Re-expressing the features
as  y' = (I_m kron A) y  with an invertible A (here a diagonal scaling
D = diag(10^u), optionally combined with a rotation) maps the posterior
*exactly* onto the posterior of a transported model

    W'_e = A^{-T} W_e A^{-1},
    R'   = (I kron A^{-T}) R (I kron A^{-1})        (R = r I  ->  r (I kron (A A^T)^{-1})),
    trace penalty   lambda tr(W_e)  ->  lambda tr(P' W'_e),  P' = A A^T,
    y_test' = (I kron A) y_test,                    W*'_e = A^{-T} W*_e A^{-1}.

The determinant-neutral prior det(W_e)^{-(d+1)/2} dW_e is invariant under
congruence, so Phi' = Phi + const and the posteriors correspond exactly.
Everything below the posterior (chains, tuning, diagnostics) is the code of
run_invariance_experiment.py / run_m50_d_sweep_wishart_with_fisher.py.

What it can show
----------------
* ConeMALA is equivariant under every congruence W -> C W C^T, so its
  performance must not change with the units.
* In log-Cholesky coordinates a rescaling changes the posterior width of the
  off-diagonal coordinates by the feature scale; one scalar step size is then
  limited by the narrowest coordinate (prediction: LC-MALA degrades).
* A learned diagonal metric (LC-MALA + diag, NUTS) can undo a pure *diagonal*
  rescaling but not a rescaling combined with a rotation (V3_..._rotated).
  This is the fair test of what invariance buys beyond adaptation.

Variants
--------
  V0_identity_rep<k>      identity (re-used from out_inv with --reuse_identity_from)
  V3_scale_dec<x>         D = diag(10^u), u linearly spaced over x decades, centred
  V3_scale_dec<x>_shuffled   same scales, assigned to the features in another order
  V3_scale_dec<x>_rotated    A = D Q (rotate, then scale: still a diagonal rescaling of a
                             generic problem, so a diagonal metric can undo it)
  V3_scale_dec<x>_QD         A = Q D (scale, then rotate: scale heterogeneity is spread over
                             rotated axes; a diagonal metric cannot undo it)
  V3_scale_dec<x>_QD_r<k>    --qd_reps K: K further independent random rotations Q

Metrics
-------
Strictly congruence-invariant functionals only (Phi, log det Q, test NLL,
per-edge log det W_e, AIRM distance to W*); ||W-W*||_F, eigenvalues of W_e
and the lambda_min/max families are NOT invariant under scaling and are
excluded from the 'strict_*' columns.  ESS/s uses the end-to-end clock.

Examples
--------
  python3 run_scaling_experiment.py --selftest
  python3 -u run_scaling_experiment.py --reuse_identity_from out_inv --outdir out_scale --resume
  python3 -u run_scaling_experiment.py --methods LogCholeskyNUTS \\
      --variants V0_identity_rep1,V3_scale_dec2,V3_scale_dec2_rotated --outdir out_scale --resume
"""

from __future__ import annotations

SCALING_VERSION = "1.0"

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_invariance_experiment as inv  # noqa: E402
import run_m50_d_sweep_wishart_with_fisher as base  # noqa: E402

Array = np.ndarray

STRICT_ESS_COLS = [
    "Phi_ESS_bulk", "logdet_Q_ESS_bulk", "test_nll_ESS_bulk",
    "inv_logdet_We_ESS_bulk_min", "inv_airm_dist_We_ESS_bulk_min",
]
STRICT_RHAT_COLS = [
    "Phi_Rhat_rank", "logdet_Q_Rhat_rank", "test_nll_Rhat_rank",
    "inv_logdet_We_Rhat_max", "inv_airm_dist_We_Rhat_max",
]
IDENTITY_CONFIG_KEYS = [
    "m", "d", "seed", "n_train", "n_test", "n_chains", "preburn", "retained",
    "truth_scale", "r_scale", "lambda_tr",
]


# =============================================================================
# Posterior with a general trace-penalty matrix P: lambda * tr(P W_e)
# =============================================================================

class ScaledPosterior(base.PSDGraphPosterior):
    """PSDGraphPosterior whose trace penalty is lambda*tr(P W_e) (P=I -> base)."""

    def __init__(self, *args, psi_inv: Array, **kwargs):
        super().__init__(*args, **kwargs)
        self.psi_inv = base.sym(np.asarray(psi_inv, dtype=float))

    def phi_and_grad(self, w: Array) -> Tuple[float, Array]:
        w = np.asarray([base.sym(w[e]) for e in range(len(w))])
        q_matrix = self.q(w)
        logdet_q = base.logdet_spd(q_matrix)
        if not np.isfinite(logdet_q):
            return np.inf, np.zeros_like(w)
        q_inverse = base.sym(base.inv(q_matrix))
        phi_likelihood = (
            -0.5 * self.n_train * logdet_q + 0.5 * np.trace(q_matrix @ self.syy)
        )
        a_matrix = -0.5 * self.n_train * q_inverse + 0.5 * self.syy
        grad_w = base.edge_gradient_from_a(a_matrix, self.edges, self.d)
        phi_prior = 0.0
        for e in range(self.edge_count):
            we = base.sym(w[e])
            logdet_we = base.logdet_spd(we)
            if not np.isfinite(logdet_we):
                return np.inf, np.zeros_like(w)
            phi_prior += self.lambda_tr * float(np.trace(self.psi_inv @ we))
            phi_prior -= self.alpha_logdet * logdet_we
            grad_w[e] += self.lambda_tr * self.psi_inv
            grad_w[e] -= self.alpha_logdet * base.inv(we)
        return (
            float(phi_likelihood + phi_prior),
            np.asarray([base.sym(g) for g in grad_w]),
        )


# =============================================================================
# Variants and transport
# =============================================================================

def make_scale_variants(
    d: int, decades: Sequence[float], n_rep: int, special_decade: float, seed: int,
    qd_reps: int = 0,
) -> List[Dict]:
    rng = np.random.default_rng(seed + 321)
    out: List[Dict] = []
    for k in range(max(n_rep, 1)):
        out.append({"name": f"V0_identity_rep{k + 1}", "family": "identity",
                    "rep": k, "decades": 0.0, "A": np.eye(d)})

    def centred(u: Array) -> Array:
        return 10.0 ** (u - u.mean())

    for dec in decades:
        u = np.linspace(0.0, float(dec), d)
        out.append({"name": f"V3_scale_dec{dec:g}", "family": "scale", "rep": 0,
                    "decades": float(dec), "A": np.diag(centred(u))})
    dec = float(special_decade)
    if dec > 0:
        u = np.linspace(0.0, dec, d)
        perm = np.arange(d)
        while np.array_equal(perm, np.arange(d)) and d > 1:
            perm = rng.permutation(d)
        out.append({"name": f"V3_scale_dec{dec:g}_shuffled", "family": "scale_shuffled",
                    "rep": 0, "decades": dec, "A": np.diag(centred(u)[perm])})
        q = inv.haar_orthogonal(d, rng)
        out.append({"name": f"V3_scale_dec{dec:g}_rotated", "family": "scale_rotated",
                    "rep": 0, "decades": dec, "A": np.diag(centred(u)) @ q})
        # A = Q D: scale first, then rotate.  W' = Q D^-1 W D^-1 Q^T, so the scale
        # heterogeneity is spread over rotated axes and a diagonal metric cannot undo it.
        # (A = D Q above rotates first: W' = D^-1 (Q W Q^T) D^-1 is just a diagonally
        # scaled generic problem, which a diagonal metric CAN undo.)
        out.append({"name": f"V3_scale_dec{dec:g}_QD", "family": "scale_QD",
                    "rep": 0, "decades": dec, "A": q @ np.diag(centred(u))})
        # Extra independent rotations (own RNG stream, so the variants above are unchanged):
        # shows the Q.D result is not an accident of one particular Q.
        for k in range(1, int(qd_reps) + 1):
            qk = inv.haar_orthogonal(d, np.random.default_rng(seed + 9001 + k))
            out.append({"name": f"V3_scale_dec{dec:g}_QD_r{k}", "family": "scale_QD",
                        "rep": k, "decades": dec, "A": qk @ np.diag(centred(u))})
    return out


def transport_general(problem: Dict, a: Array, args: argparse.Namespace) -> Dict:
    m, d = problem["m"], problem["d"]
    big_a = np.kron(np.eye(m), a)
    big_ai = np.linalg.inv(big_a)
    a_inv = np.linalg.inv(a)
    r_new = big_ai.T @ problem["r_matrix"] @ big_ai
    post = ScaledPosterior(
        edges=problem["edges"], m=m, d=d,
        y_train=big_a @ problem["y_train"],
        r_matrix=r_new,
        lambda_tr=args.lambda_tr,
        alpha_logdet=(0.5 * (d + 1.0) if args.alpha_logdet is None else args.alpha_logdet),
        psi_inv=a @ a.T,
    )
    return {
        "posterior": post,
        "w_star": np.asarray([a_inv.T @ w @ a_inv for w in problem["w_star"]]),
        "y_test": big_a @ problem["y_test"],
    }


def to_new(a: Array, w: Array) -> Array:
    """W (E,d,d) -> A^{-T} W A^{-1}."""
    ai = np.linalg.inv(a)
    out = ai.T @ w @ ai
    return 0.5 * (out + np.swapaxes(out, -1, -2))


# =============================================================================
# Reuse of identity-basis rows from out_inv
# =============================================================================

def reuse_identity(src: Path, dst: Path, args: argparse.Namespace) -> int:
    cfg_file = src / "config.json"
    if not cfg_file.exists():
        raise SystemExit(f"{cfg_file} not found; cannot verify the identity runs match.")
    cfg = json.loads(cfg_file.read_text())
    mismatch = [
        (k, cfg.get(k), getattr(args, k)) for k in IDENTITY_CONFIG_KEYS
        if cfg.get(k) != getattr(args, k)
    ]
    if mismatch:
        raise SystemExit(
            "Identity runs in the source folder used different settings "
            f"(key, source, now): {mismatch}"
        )
    copied = 0
    for variant_dir in sorted(src.glob("V0_identity_rep*")):
        target = dst / variant_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for f in list(variant_dir.glob("row_*.csv")) + list(variant_dir.glob("h_lchol.json")):
            if not (target / f.name).exists():
                shutil.copy2(f, target / f.name)
                copied += 1
    return copied


# =============================================================================
# Summaries
# =============================================================================

SCALE_METRICS = [
    ("inv_airm_dist_We_ESS_bulk_median", "median AIRM-distance bulk ESS (per draw)"),
    ("inv_airm_dist_We_ESSsec_bulk_median_total", "median AIRM-distance bulk ESS/s"),
    ("inv_logdet_We_ESSsec_bulk_median_total", "median log det W_e bulk ESS/s"),
    ("strict_ESSsec_bulk_min_total", "min strictly-invariant bulk ESS/s"),
    ("Phi_ESSsec_bulk_total", "Phi bulk ESS/s"),
    ("h", "tuned step size h"),
    ("acceptance", "production acceptance"),
]


def load_rows(outdir: Path) -> pd.DataFrame:
    rows = [pd.read_csv(p) for p in sorted(outdir.glob("*/row_*.csv"))]
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    table["strict_ESS_bulk_min"] = table[[c for c in STRICT_ESS_COLS if c in table]].min(axis=1)
    table["strict_Rhat_max"] = table[[c for c in STRICT_RHAT_COLS if c in table]].max(axis=1)
    table["strict_ESSsec_bulk_min_total"] = (
        table["strict_ESS_bulk_min"] / table["runtime_total_sec"]
    )
    vfile = outdir / "variants.csv"
    if vfile.exists():
        meta = pd.read_csv(vfile)
        table = table.drop(columns=[c for c in ("decades",) if c in table]).merge(
            meta[["variant", "decades"]], on="variant", how="left"
        )
    return table


def summarize(outdir: Path) -> None:
    table = load_rows(outdir)
    if table.empty:
        print("No row files found.")
        return
    table.to_csv(outdir / "scaling_summary.csv", index=False)
    if not (table["family"] == "identity").any():
        print(f"\nNo identity-basis rows in {outdir}: wrote scaling_summary.csv only "
              "(absolute numbers; ratios need --reuse_identity_from).")
        return
    rel_rows = []
    for method, g in table.groupby("Method"):
        ident = g[g["family"] == "identity"]
        for col, label in SCALE_METRICS:
            if col not in g:
                continue
            base_vals = ident[col].astype(float)
            ref = float(np.exp(np.nanmean(np.log(base_vals)))) if len(base_vals) else float("nan")
            for _, r in g.iterrows():
                if r["family"] == "identity":
                    continue
                rel_rows.append({
                    "Method": method, "metric": col, "label": label,
                    "variant": r["variant"], "family": r["family"],
                    "decades": r.get("decades", float("nan")),
                    "value": float(r[col]), "identity_geomean": ref,
                    "ratio_to_identity": float(r[col]) / ref if ref > 0 else float("nan"),
                    "identity_min_ratio": float(base_vals.min() / ref) if ref > 0 else float("nan"),
                    "identity_max_ratio": float(base_vals.max() / ref) if ref > 0 else float("nan"),
                })
    rel = pd.DataFrame(rel_rows)
    if rel.empty:
        print("\nOnly identity-basis rows so far; ratios appear once a scaled variant finishes.")
        return
    rel.to_csv(outdir / "scaling_ratio_to_identity.csv", index=False)

    for metric, title in (
        ("inv_airm_dist_We_ESSsec_bulk_median_total",
         "median AIRM-distance bulk ESS/s relative to identity (identity noise band in brackets)"),
        ("h", "tuned step size h relative to identity"),
    ):
        sub = rel[rel["metric"] == metric]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="variant", columns="Method", values="ratio_to_identity")
        band = sub.groupby("Method")[["identity_min_ratio", "identity_max_ratio"]].first()
        print(f"\n{title}")
        print(piv.to_string(float_format=lambda v: f"{v:.3g}"))
        print("identity band: " + ", ".join(
            f"{m}=[{b.identity_min_ratio:.2f},{b.identity_max_ratio:.2f}]" for m, b in band.iterrows()
        ))
    print(f"\nWrote {outdir / 'scaling_summary.csv'} and scaling_ratio_to_identity.csv")


# =============================================================================
# Self-test
# =============================================================================

def selftest(args: argparse.Namespace) -> None:
    problem = inv.build_base_problem(args)
    d, m = problem["d"], problem["m"]
    alpha = 0.5 * (d + 1.0) if args.alpha_logdet is None else args.alpha_logdet
    post0 = base.PSDGraphPosterior(
        edges=problem["edges"], m=m, d=d, y_train=problem["y_train"],
        r_matrix=problem["r_matrix"], lambda_tr=args.lambda_tr, alpha_logdet=alpha,
    )
    variants = [v for v in make_scale_variants(d, [1, 3], 1, 2, args.seed) if v["family"] != "identity"]
    rng = np.random.default_rng(5)
    edge_count = len(problem["edges"])

    def random_state() -> Array:
        return np.asarray([
            base.project_spd(0.3 * (lambda a: a @ a.T)(rng.normal(size=(d, d))) + 0.2 * np.eye(d))
            for _ in range(edge_count)
        ])

    print("1) exact correspondence of the transported posterior")
    print("   Phi' - Phi must equal the constant (n_train*m + 2*alpha*|E|)*log|det A|;")
    print("   grad' must equal A grad A^T.")
    ok = True
    for v in variants:
        a = v["A"]
        post1 = transport_general(problem, a, args)["posterior"]
        const = (args.n_train * m + 2.0 * alpha * edge_count) * math.log(abs(np.linalg.det(a)))
        errs = []
        for _ in range(2):
            w = random_state()
            phi0, g0 = post0.phi_and_grad(w)
            phi1, g1 = post1.phi_and_grad(to_new(a, w))
            e_phi = abs((phi1 - phi0) - const) / max(abs(phi0), 1.0)
            e_g = float(np.max(np.abs(g1 - np.asarray([a @ x @ a.T for x in g0])))
                        / max(np.max(np.abs(g1)), 1.0))
            errs.append((e_phi, e_g))
        good = max(max(e) for e in errs) < 1e-8
        ok &= good
        print(f"   {v['name']:<26s} rel|Phi shift err|={max(e[0] for e in errs):.1e} "
              f"rel|grad err|={max(e[1] for e in errs):.1e}  {'OK' if good else 'FAIL'}")

    print("\n2) one-step acceptance at a transported near-typical state (same h; 600 proposals)")
    print("   ConeMALA must agree across all rescalings (equivariant in distribution);")
    print("   LC-MALA with the same scalar h is expected to collapse as the scale ratio grows.")
    h_cone, h_lc = args.h_cone_center, args.lchol_h_center
    burn = base.cone_mala_chain(post0, base.make_initial_w(edge_count, d, 0), 0, 400, h_cone, 1).samples[-1]
    n_prop = 600

    def acc_cone(post, w):
        phi, g = post.phi_and_grad(w)
        r = np.random.default_rng(11)
        return float(np.mean([base._cone_transition(post, w, phi, g, h_cone, r)[3] for _ in range(n_prop)]))

    def acc_lc(post, w):
        t = base.LogCholeskyTarget(post)
        th = t.from_weights(w)
        pot, g = t.potential_and_grad(th)
        r = np.random.default_rng(11)
        return float(np.mean([base._lchol_transition(t, th, pot, g, h_lc, r)[3] for _ in range(n_prop)]))

    ref_c, ref_l = acc_cone(post0, burn), acc_lc(post0, burn)
    se = math.sqrt(max(ref_c * (1 - ref_c), 1e-4) / n_prop)
    print(f"   {'identity':<26s} ConeMALA acc={ref_c:.3f}   LC-MALA acc={ref_l:.3f}")
    for v in variants:
        a = v["A"]
        post1 = transport_general(problem, a, args)["posterior"]
        w1 = to_new(a, burn)
        ac, al = acc_cone(post1, w1), acc_lc(post1, w1)
        z = (ac - ref_c) / (math.sqrt(2) * se)
        print(f"   {v['name']:<26s} ConeMALA acc={ac:.3f} (z={z:+.2f})   LC-MALA acc={al:.3f}")
    print("\nself-test", "passed" if ok else "FAILED (transport)")
    if not ok:
        sys.exit(1)


# =============================================================================
# Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = inv.build_parser()
    p.description = __doc__
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    p.set_defaults(
        outdir="out_scale",
        methods="ConeMALA,LogCholeskyMALA,LogCholeskyMALA_DiagMetric",
    )
    p.add_argument("--decades", type=str, default="1,2,3",
                   help="Comma-separated scale ranges in decades (max/min feature scale = 10^x).")
    p.add_argument("--special_decade", type=float, default=2.0,
                   help="Decade used for the shuffled and rotated variants (0 = skip them).")
    p.add_argument("--qd_reps", type=int, default=0,
                   help="Number of EXTRA independent random rotations Q for the Q.D variant "
                        "(named V3_scale_dec<x>_QD_r1, _r2, ...).  0 = only the original V3_..._QD.")
    p.add_argument("--reuse_identity_from", type=str, default=None,
                   help="Folder of a finished run_invariance_experiment.py (e.g. out_inv): its "
                        "V0_identity_rep* rows are copied after a settings check.")
    for action in p._actions:
        if any(o in action.option_strings for o in ("--n_perm", "--n_orth")):
            action.help = "(ignored by this script)"
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

    methods = base.parse_method_list(args.methods, inv.INV_METHODS, default=inv.INV_METHODS)
    decades = [float(x) for x in args.decades.split(",") if x.strip()]
    variants = make_scale_variants(args.d, decades, args.n_replicates, args.special_decade, args.seed,
                               qd_reps=args.qd_reps)
    if args.variants.strip().lower() != "all":
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [w for w in wanted if w not in {v["name"] for v in variants}]
        if unknown:
            raise SystemExit(f"Unknown variants {unknown}; available: {[v['name'] for v in variants]}")
        variants = [v for v in variants if v["name"] in wanted]

    if args.reuse_identity_from:
        n = reuse_identity(Path(args.reuse_identity_from), outdir, args)
        print(f"Copied {n} identity-basis files from {args.reuse_identity_from}")

    problem = inv.build_base_problem(args)
    edge_count = len(problem["edges"])
    pd.DataFrame([
        {"variant": v["name"], "family": v["family"], "decades": v["decades"],
         "scale_ratio": float(np.max(np.diag(v["A"])) / np.min(np.diag(v["A"])))
         if v["family"] in ("scale", "scale_shuffled") else float("nan"),
         "cond_A": float(np.linalg.cond(v["A"]))}
        for v in make_scale_variants(args.d, decades, args.n_replicates, args.special_decade, args.seed,
                               qd_reps=args.qd_reps)
    ]).to_csv(outdir / "variants.csv", index=False)
    (outdir / "config.json").write_text(json.dumps(
        {**vars(args), "version": SCALING_VERSION, "methods": methods,
         "variants": [v["name"] for v in variants]}, indent=2) + "\n")
    print(f"m={args.m}, d={args.d}, |E|={edge_count}, chains={args.n_chains}, "
          f"preburn={args.preburn}, retained={args.retained}")
    print("methods :", methods)
    print("variants:", [v["name"] for v in variants], flush=True)

    base_w0 = [base.make_initial_w(edge_count, args.d, c) for c in range(args.n_chains)]
    for variant in variants:
        cell_dir = outdir / variant["name"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        np.save(cell_dir / "A.npy", variant["A"])
        prob = transport_general(problem, variant["A"], args)
        initial = [to_new(variant["A"], w) for w in base_w0]
        h_file = cell_dir / "h_lchol.json"
        h_lchol = json.loads(h_file.read_text())["h_lchol"] if (args.resume and h_file.exists()) else None
        print("\n" + "=" * 90 + f"\nvariant {variant['name']} ({variant['family']}, "
              f"cond(A)={np.linalg.cond(variant['A']):.3g})\n" + "=" * 90, flush=True)
        for method in sorted(methods, key=lambda s: inv.INV_METHODS.index(s)):
            row_path = cell_dir / f"row_{method}.csv"
            if args.resume and row_path.exists():
                print(f"  {method}: found {row_path.name}, skipping", flush=True)
                continue
            t0 = time.perf_counter()
            print(f"  {method} ...", flush=True)
            try:
                df, h_lchol = inv.run_cell(method, variant, prob, initial, h_lchol, args, cell_dir)
            except RuntimeError as exc:  # e.g. no feasible pilot step size
                print(f"  {method}: FAILED ({exc})", flush=True)
                pd.DataFrame([{"variant": variant["name"], "family": variant["family"],
                               "Method": method, "failed": str(exc)}]
                             ).to_csv(cell_dir / f"failed_{method}.csv", index=False)
                continue
            df.to_csv(row_path, index=False)
            r = df.iloc[0]
            print(
                f"  {method}: acc={r['acceptance']:.3f} h={r['h']:.3e} "
                f"median AIRM ESS={r['inv_airm_dist_We_ESS_bulk_median']:.0f} "
                f"(ESS/s={r['inv_airm_dist_We_ESSsec_bulk_median_total']:.3g}) "
                f"[{time.perf_counter() - t0:.0f}s]", flush=True)
        summarize(outdir)


if __name__ == "__main__":
    main()
