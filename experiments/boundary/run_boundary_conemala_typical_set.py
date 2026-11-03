#!/usr/bin/env python3
r"""
Typical-set-aware boundary experiment for ConeMALA.

Purpose
-------
For a fully fair boundary comparison, ConeMALA should use the same broad
experimental stages as the Fisher and Euclidean baselines:

1. discarded adaptive pre-burn-in from dispersed initial states;
2. extraction of terminal states near the posterior typical set;
3. deterministic shared step-size tuning from clones of those terminal states;
4. fixed-kernel production warmup;
5. retained fixed-kernel sampling.

ConeMALA does not adapt a dense metric.  Its discarded pre-burn-in adapts only
the scalar step size h by diminishing Robbins--Monro updates.  The production
kernel uses a fixed h and the exact product-AIRM exponential proposal with the
full Metropolis--Hastings correction.

All pre-burn-in, settling, and tuning draws are discarded.

Required companion files
------------------------
Keep this file beside

    conemala_fisher_adaptive_comparison.py
    boundary_stress_comparison.py

Recommended determinant-neutral pilot
-------------------------------------
python3 run_boundary_conemala_typical_set.py \
    --config results_fisher_paper/config.json \
    --profile pilot \
    --epsilons 1e-1,1e-2,1e-3 \
    --prior-mode boundary-friendly \
    --boundary-nu-offset 1.0 \
    --output-dir results_boundary_conemala_typical \
    --make-plots

Paper run
---------
python3 run_boundary_conemala_typical_set.py \
    --config results_fisher_paper/config.json \
    --profile paper \
    --epsilons 1e-1,1e-2,1e-3 \
    --prior-mode boundary-friendly \
    --boundary-nu-offset 1.0 \
    --preburn-steps 4000 \
    --settle-steps 500 \
    --production-warmup 2000 \
    --tune-warmup 200 \
    --tune-steps 400 \
    --retained 4000 \
    --output-dir results_boundary_conemala_typical_paper \
    --make-plots
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import conemala_fisher_adaptive_comparison as base
import boundary_stress_comparison as boundary


Array = np.ndarray
METHOD = "ConeMALA"


@dataclass
class ConeBurninResult:
    terminal_w: Array
    h_final: float
    acceptance: float
    invalid_spd_rate: float
    numerical_failure_rate: float
    runtime_sec: float
    adaptive_steps: int
    settle_steps: int


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(
        float(item.strip()) for item in text.split(",") if item.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError("At least one epsilon is required.")
    if any(not (0.0 < value <= 1.0) for value in values):
        raise argparse.ArgumentTypeError(
            "Every epsilon must lie in (0,1]."
        )
    return values


def diminishing_rate(
    iteration: int,
    initial_rate: float,
    decay: float,
    exponent: float,
) -> float:
    return float(
        initial_rate
        / (1.0 + iteration / max(decay, 1e-12)) ** exponent
    )


def propose_conemala_step(
    post: base.GraphPosterior,
    w: Array,
    evaluation,
    h: float,
    rng: np.random.Generator,
) -> Tuple[Array, object, float, bool, bool]:
    """
    Perform one exact ConeMALA proposal.

    Returns
    -------
    w_next:
        Accepted state or the unchanged current state.
    evaluation_next:
        Matching posterior evaluation.
    alpha_prob:
        Metropolis--Hastings acceptance probability.
    accepted:
        Whether the proposal was accepted.
    numerical_failure:
        True only if matrix algebra or posterior evaluation failed.

    Notes
    -----
    The exponential proposal is SPD by construction.  A Cholesky check is
    retained solely as a numerical safeguard.
    """
    w_prop = np.empty_like(w)
    tangent_forward: List[Array] = []
    tangent_mean_forward: List[Array] = []
    numerical_failure = False

    try:
        for edge_index in range(post.e_count):
            root, _ = base.spd_sqrt_and_inv_sqrt(w[edge_index])
            mean_tangent = -h * base.sym(
                root @ evaluation.grad_phi[edge_index] @ root
            )
            noise = base.sample_standard_symmetric(post.d, rng)
            tangent = base.sym(
                mean_tangent + math.sqrt(2.0 * h) * noise
            )
            w_prop[edge_index] = base.sym(
                root @ base.sym_expm(tangent) @ root
            )

            # The exponential map should preserve SPD exactly in exact
            # arithmetic.  This check detects only numerical failures.
            np.linalg.cholesky(w_prop[edge_index])
            tangent_forward.append(tangent)
            tangent_mean_forward.append(mean_tangent)
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return w, evaluation, 0.0, False, True

    try:
        eval_prop = post.evaluate(w_prop)
        logq_forward = 0.0
        logq_reverse = 0.0

        for edge_index in range(post.e_count):
            forward_tangent = tangent_forward[edge_index]
            forward_mean = tangent_mean_forward[edge_index]
            logq_forward += (
                base.log_gaussian_isotropic(
                    base.svec(forward_tangent - forward_mean),
                    variance=2.0 * h,
                )
                - base.log_spd_exp_jacobian(forward_tangent)
            )

            reverse_tangent = base.reverse_airm_tangent(
                w_prop[edge_index],
                w[edge_index],
            )
            root_prop, _ = base.spd_sqrt_and_inv_sqrt(
                w_prop[edge_index]
            )
            reverse_mean = -h * base.sym(
                root_prop
                @ eval_prop.grad_phi[edge_index]
                @ root_prop
            )
            logq_reverse += (
                base.log_gaussian_isotropic(
                    base.svec(reverse_tangent - reverse_mean),
                    variance=2.0 * h,
                )
                - base.log_spd_exp_jacobian(reverse_tangent)
            )

        log_alpha = (
            -eval_prop.phi
            + logq_reverse
            - (-evaluation.phi + logq_forward)
        )
        alpha_prob = float(math.exp(min(0.0, log_alpha)))
        accepted = bool(rng.uniform() < alpha_prob)
        if accepted:
            return w_prop, eval_prop, alpha_prob, True, False
        return w, evaluation, alpha_prob, False, False

    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return w, evaluation, 0.0, False, True


def adaptive_conemala_burnin(
    post: base.GraphPosterior,
    w_init: Array,
    adaptive_steps: int,
    settle_steps: int,
    h0: float,
    target_accept: float,
    adapt_rate: float,
    adapt_decay: float,
    adapt_exponent: float,
    h_min: float,
    h_max: float,
    seed: int,
) -> ConeBurninResult:
    """
    Discarded ConeMALA pre-burn-in with diminishing adaptation of log h.

    The first phase adapts h.  The second phase freezes h and lets the chain
    settle before its terminal state is used as a typical-set tuning anchor.
    """
    if adaptive_steps < 1:
        raise ValueError("adaptive_steps must be positive.")
    if settle_steps < 0:
        raise ValueError("settle_steps must be nonnegative.")
    if not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must lie in (0,1).")
    if not 0.5 < adapt_exponent <= 1.0:
        raise ValueError("adapt_exponent should lie in (0.5,1].")
    if not 0.0 < h_min < h_max:
        raise ValueError("Require 0 < h_min < h_max.")

    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    log_h = float(
        np.clip(math.log(h0), math.log(h_min), math.log(h_max))
    )

    accepted = 0
    numerical_failures = 0
    start = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: adapt only the scalar h.  All states are discarded.
    # ------------------------------------------------------------------
    for iteration in range(adaptive_steps):
        h_now = float(math.exp(log_h))
        (
            w,
            evaluation,
            alpha_prob,
            accepted_now,
            numerical_failure,
        ) = propose_conemala_step(
            post=post,
            w=w,
            evaluation=evaluation,
            h=h_now,
            rng=rng,
        )

        accepted += int(accepted_now)
        numerical_failures += int(numerical_failure)

        eta = diminishing_rate(
            iteration=iteration,
            initial_rate=adapt_rate,
            decay=adapt_decay,
            exponent=adapt_exponent,
        )
        log_h += eta * (alpha_prob - target_accept)
        log_h = float(
            np.clip(log_h, math.log(h_min), math.log(h_max))
        )

    h_final = float(math.exp(log_h))

    # ------------------------------------------------------------------
    # Phase 2: fixed-h settling.  These states are also discarded.
    # ------------------------------------------------------------------
    if settle_steps > 0:
        settle = boundary.cone_mala_boundary_chain(
            post=post,
            w_init=w,
            warmup=0,
            retained=settle_steps,
            thin=1,
            h=h_final,
            seed=seed + 5_000_000,
        )
        if settle.samples.shape[0] == 0:
            raise RuntimeError("ConeMALA settling produced no samples.")
        w = np.asarray(settle.samples[-1]).copy()
        accepted += int(
            round(settle.acceptance * settle.final_proposals)
        )
        numerical_failures += settle.final_numerical_failures

    total_steps = adaptive_steps + settle_steps
    return ConeBurninResult(
        terminal_w=w.copy(),
        h_final=h_final,
        acceptance=accepted / max(total_steps, 1),
        invalid_spd_rate=0.0,
        numerical_failure_rate=(
            numerical_failures / max(total_steps, 1)
        ),
        runtime_sec=float(time.perf_counter() - start),
        adaptive_steps=adaptive_steps,
        settle_steps=settle_steps,
    )


def product_airm_esjd_per_iteration(samples: Array) -> float:
    """
    Mean squared product-AIRM jump distance per retained transition.
    """
    if samples.shape[0] < 2:
        return 0.0

    values: List[float] = []
    for draw_index in range(samples.shape[0] - 1):
        squared_distance = 0.0
        for edge_index in range(samples.shape[1]):
            tangent = base.reverse_airm_tangent(
                samples[draw_index, edge_index],
                samples[draw_index + 1, edge_index],
            )
            squared_distance += float(np.sum(tangent * tangent))
        values.append(squared_distance)

    return float(np.mean(values))


def tune_h_from_typical_states(
    post: base.GraphPosterior,
    anchor_states: Sequence[Array],
    h_center: float,
    target_accept: float,
    acceptance_tolerance: float,
    tune_warmup: int,
    tune_steps: int,
    factors: Sequence[float],
    max_bracket_steps: int,
    bisection_steps: int,
    seed_base: int,
) -> Tuple[float, float, float, List[Dict[str, float]]]:
    """
    Tune one shared ConeMALA h from clones of typical-set anchor states.

    Candidate selection uses runtime-independent product-AIRM ESJD per
    retained transition.  Runtime is recorded but never used for selection.
    """
    if not anchor_states:
        raise ValueError("At least one anchor state is required.")
    if tune_warmup < 0:
        raise ValueError("tune_warmup must be nonnegative.")
    if tune_steps < 10:
        raise ValueError("tune_steps must be at least 10.")
    if acceptance_tolerance < 0.0:
        raise ValueError("acceptance_tolerance must be nonnegative.")

    tested: Dict[str, Dict[str, float]] = {}

    def evaluate(h: float) -> Dict[str, float]:
        h = float(max(h, 1e-14))
        key = f"{h:.17e}"
        if key in tested:
            return tested[key]

        acceptances: List[float] = []
        esjd_values: List[float] = []
        runtimes: List[float] = []
        numerical_failure_rates: List[float] = []

        for chain_index, anchor in enumerate(anchor_states):
            result = boundary.cone_mala_boundary_chain(
                post=post,
                w_init=np.asarray(anchor, dtype=float).copy(),
                warmup=tune_warmup,
                retained=tune_steps,
                thin=1,
                h=h,
                seed=seed_base + chain_index,
            )
            acceptances.append(result.acceptance)
            esjd_values.append(
                product_airm_esjd_per_iteration(result.samples)
            )
            runtimes.append(result.runtime_sec)
            numerical_failure_rates.append(
                result.final_numerical_failures
                / max(result.final_proposals, 1)
            )

        row = {
            "method": METHOD,
            "stage": "typical_set_fixed_h_tuning",
            "h": h,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1)
                if len(acceptances) > 1
                else 0.0
            ),
            "airm_esjd_per_iteration": float(np.mean(esjd_values)),
            "pilot_invalid_spd_rate": 0.0,
            "pilot_numerical_failure_rate": float(
                np.mean(numerical_failure_rates)
            ),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[key] = row
        return row

    for factor in sorted(
        set(float(value) for value in factors if value > 0.0)
    ):
        evaluate(h_center * factor)

    lower = target_accept - acceptance_tolerance
    upper = target_accept + acceptance_tolerance

    # Expand upward when all tested candidates accept too frequently.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] <= upper for row in rows):
            break
        evaluate(rows[-1]["h"] * 2.0)

    # Expand downward when all tested candidates accept too rarely.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] >= lower for row in rows):
            break
        evaluate(rows[0]["h"] / 2.0)

    # Geometric bisection around the target acceptance.
    for _ in range(bisection_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        high_accept = [
            row
            for row in rows
            if row["pilot_acceptance"] >= target_accept
        ]
        low_accept = [
            row
            for row in rows
            if row["pilot_acceptance"] <= target_accept
        ]
        if not high_accept or not low_accept:
            break

        left = max(high_accept, key=lambda row: row["h"])
        right = min(low_accept, key=lambda row: row["h"])
        if left["h"] >= right["h"]:
            break

        h_mid = math.sqrt(left["h"] * right["h"])
        if f"{h_mid:.17e}" in tested:
            break
        evaluate(h_mid)

    rows = sorted(tested.values(), key=lambda row: row["h"])
    admissible = [
        row
        for row in rows
        if lower <= row["pilot_acceptance"] <= upper
    ]

    if admissible:
        best = max(
            admissible,
            key=lambda row: (
                row["airm_esjd_per_iteration"],
                -abs(row["pilot_acceptance"] - target_accept),
                -row["h"],
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                abs(row["pilot_acceptance"] - target_accept),
                -row["airm_esjd_per_iteration"],
                row["h"],
            ),
        )

    for row in rows:
        row["selected"] = bool(
            math.isclose(
                row["h"],
                best["h"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

    total_runtime = float(
        sum(row["pilot_runtime_sec"] for row in rows)
    )
    return (
        float(best["h"]),
        float(best["pilot_acceptance"]),
        total_runtime,
        rows,
    )


def default_preburn_steps(
    profile: str,
    config: base.ExperimentConfig,
) -> int:
    if profile == "smoke":
        return 20
    if profile == "pilot":
        return max(2000, 2 * config.warmup)
    return max(4000, 2 * config.warmup)


def default_settle_steps(profile: str) -> int:
    return {
        "smoke": 5,
        "pilot": 250,
        "paper": 500,
    }[profile]


def default_tune_warmup(profile: str) -> int:
    return {
        "smoke": 0,
        "pilot": 100,
        "paper": 200,
    }[profile]


def checkpoint(
    output_dir: Path,
    burnin_rows: List[Dict[str, float]],
    tuning_rows: List[Dict[str, float]],
    seed_rows: List[Dict[str, float]],
    diagnostic_rows: List[Dict[str, float]],
    truth_rows: List[Dict[str, float]],
) -> None:
    pd.DataFrame(burnin_rows).to_csv(
        output_dir / "conemala_typical_burnin.csv",
        index=False,
    )
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "conemala_typical_tuning.csv",
        index=False,
    )
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "conemala_typical_by_seed.csv",
        index=False,
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "conemala_typical_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(truth_rows).to_csv(
        output_dir / "boundary_truth.csv",
        index=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Typical-set-aware ConeMALA boundary experiment."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "paper"),
        default="pilot",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--epsilons",
        type=parse_float_tuple,
        default=(1e-1, 1e-2, 1e-3),
    )
    parser.add_argument(
        "--prior-mode",
        choices=("original", "boundary-friendly"),
        default="boundary-friendly",
    )
    parser.add_argument(
        "--boundary-nu-offset",
        type=float,
        default=1.0,
    )

    parser.add_argument("--preburn-steps", type=int)
    parser.add_argument("--settle-steps", type=int)
    parser.add_argument("--production-warmup", type=int)
    parser.add_argument("--retained", type=int)
    parser.add_argument("--thin", type=int)

    parser.add_argument("--h0", type=float)
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--adapt-decay", type=float, default=100.0)
    parser.add_argument("--adapt-exponent", type=float, default=0.6)
    parser.add_argument("--h-min", type=float, default=1e-10)
    parser.add_argument("--h-max", type=float, default=1.0)

    parser.add_argument("--tune-warmup", type=int)
    parser.add_argument("--tune-steps", type=int)
    parser.add_argument(
        "--acceptance-tolerance",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--max-bracket-steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--bisection-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--boundary-stride",
        type=int,
        default=10,
    )
    parser.add_argument("--save-anchor-states", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    source_config = boundary.load_base_config(args.config)
    config = boundary.resolve_profile(source_config, args.profile)
    config = replace(config, methods=(METHOD,))
    base.validate_config(config)

    preburn_steps = (
        default_preburn_steps(args.profile, config)
        if args.preburn_steps is None
        else args.preburn_steps
    )
    settle_steps = (
        default_settle_steps(args.profile)
        if args.settle_steps is None
        else args.settle_steps
    )
    production_warmup = (
        config.warmup
        if args.production_warmup is None
        else args.production_warmup
    )
    retained = (
        config.retained
        if args.retained is None
        else args.retained
    )
    thin = config.thin if args.thin is None else args.thin
    tune_warmup = (
        default_tune_warmup(args.profile)
        if args.tune_warmup is None
        else args.tune_warmup
    )
    tune_steps = (
        config.pilot_steps
        if args.tune_steps is None
        else args.tune_steps
    )
    h0 = config.cone_h0 if args.h0 is None else args.h0

    if preburn_steps < 1:
        raise ValueError("preburn_steps must be positive.")
    if settle_steps < 0:
        raise ValueError("settle_steps must be nonnegative.")
    if production_warmup < 0:
        raise ValueError("production_warmup must be nonnegative.")
    if retained < 4:
        raise ValueError("retained must be at least 4.")
    if thin < 1:
        raise ValueError("thin must be positive.")
    if args.max_bracket_steps < 0 or args.bisection_steps < 0:
        raise ValueError(
            "Bracket and bisection step counts must be nonnegative."
        )

    run_config = {
        "source_config": str(args.config),
        "profile": args.profile,
        "output_dir": str(args.output_dir),
        "epsilons": list(args.epsilons),
        "prior_mode": args.prior_mode,
        "boundary_nu_offset": args.boundary_nu_offset,
        "preburn_steps": preburn_steps,
        "settle_steps": settle_steps,
        "production_warmup": production_warmup,
        "retained": retained,
        "thin": thin,
        "h0": h0,
        "adapt_rate": args.adapt_rate,
        "adapt_decay": args.adapt_decay,
        "adapt_exponent": args.adapt_exponent,
        "h_min": args.h_min,
        "h_max": args.h_max,
        "tune_warmup": tune_warmup,
        "tune_steps": tune_steps,
        "acceptance_tolerance": args.acceptance_tolerance,
        "max_bracket_steps": args.max_bracket_steps,
        "bisection_steps": args.bisection_steps,
        "boundary_stride": args.boundary_stride,
        "resolved_experiment_config": asdict(config),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print("Typical-set-aware ConeMALA boundary experiment")
    print("=" * 88)
    print(json.dumps(run_config, indent=2))

    burnin_rows: List[Dict[str, float]] = []
    tuning_rows: List[Dict[str, float]] = []
    seed_rows: List[Dict[str, float]] = []
    diagnostic_rows: List[Dict[str, float]] = []
    truth_rows: List[Dict[str, float]] = []

    for epsilon_index, epsilon in enumerate(args.epsilons):
        print("\n" + "#" * 88)
        print(
            f"epsilon={epsilon:.1e}, "
            f"target edge condition={1.0 / epsilon:.1e}"
        )
        print("#" * 88)

        for experiment_seed in config.experiment_seeds:
            print("\n" + "=" * 88)
            print(f"Independent data seed: {experiment_seed}")
            print("=" * 88)

            post = boundary.make_boundary_problem(
                config=config,
                experiment_seed=experiment_seed,
                epsilon=epsilon,
                prior_mode=args.prior_mode,
                boundary_nu_offset=args.boundary_nu_offset,
            )
            truth_stats = boundary.matrix_family_stats(post.w_true)
            truth_rows.append(
                {
                    "epsilon": epsilon,
                    "truth_condition_target": 1.0 / epsilon,
                    "experiment_seed": experiment_seed,
                    "prior_mode": args.prior_mode,
                    "nu": post.nu,
                    **truth_stats,
                }
            )

            w_center = base.initial_weights(post)
            anchors: List[Array] = []
            adapted_h: List[float] = []
            burnin_runtime = 0.0

            print("Preliminary adaptive ConeMALA burn-in")
            for chain_index in range(config.n_chains):
                w_init = boundary.fixed_initial_state(
                    w_center=w_center,
                    experiment_seed=experiment_seed,
                    epsilon_index=epsilon_index,
                    method=METHOD,
                    chain_index=chain_index,
                    n_chains=config.n_chains,
                )
                result = adaptive_conemala_burnin(
                    post=post,
                    w_init=w_init,
                    adaptive_steps=preburn_steps,
                    settle_steps=settle_steps,
                    h0=h0,
                    target_accept=config.target_accept,
                    adapt_rate=args.adapt_rate,
                    adapt_decay=args.adapt_decay,
                    adapt_exponent=args.adapt_exponent,
                    h_min=args.h_min,
                    h_max=args.h_max,
                    seed=(
                        310_000_000
                        + 1_000_000 * epsilon_index
                        + 10_000 * experiment_seed
                        + chain_index
                    ),
                )
                anchors.append(result.terminal_w)
                adapted_h.append(result.h_final)
                burnin_runtime += result.runtime_sec

                burnin_rows.append(
                    {
                        "epsilon": epsilon,
                        "truth_condition_target": 1.0 / epsilon,
                        "experiment_seed": experiment_seed,
                        "method": METHOD,
                        "chain": chain_index + 1,
                        "stage": "adaptive_conemala_preburn",
                        "h_final": result.h_final,
                        "acceptance": result.acceptance,
                        "invalid_spd_rate": result.invalid_spd_rate,
                        "numerical_failure_rate": (
                            result.numerical_failure_rate
                        ),
                        "runtime_sec": result.runtime_sec,
                        "adaptive_steps": result.adaptive_steps,
                        "settle_steps": result.settle_steps,
                    }
                )

                print(
                    f"  chain {chain_index + 1}: "
                    f"h_final={result.h_final:.3e}, "
                    f"acc={result.acceptance:.3f}, "
                    f"invalid={result.invalid_spd_rate:.3f}, "
                    f"failures={result.numerical_failure_rate:.3f}, "
                    f"time={result.runtime_sec:.1f}s"
                )

            h_center = float(np.median(adapted_h))
            (
                h_selected,
                pilot_acceptance,
                tuning_runtime,
                rows,
            ) = tune_h_from_typical_states(
                post=post,
                anchor_states=anchors,
                h_center=h_center,
                target_accept=config.target_accept,
                acceptance_tolerance=args.acceptance_tolerance,
                tune_warmup=tune_warmup,
                tune_steps=tune_steps,
                factors=(0.25, 0.5, 1.0, 2.0, 4.0),
                max_bracket_steps=args.max_bracket_steps,
                bisection_steps=args.bisection_steps,
                seed_base=(
                    320_000_000
                    + 1_000_000 * epsilon_index
                    + 10_000 * experiment_seed
                ),
            )

            for row in rows:
                row.update(
                    {
                        "epsilon": epsilon,
                        "truth_condition_target": 1.0 / epsilon,
                        "experiment_seed": experiment_seed,
                    }
                )
                tuning_rows.append(row)

            print(
                f"Typical-set tuning: center={h_center:.3e}, "
                f"selected h={h_selected:.3e}, "
                f"pilot acceptance={pilot_acceptance:.3f}, "
                f"time={tuning_runtime:.1f}s"
            )

            if args.save_anchor_states:
                np.savez_compressed(
                    args.output_dir
                    / (
                        f"conemala_typical_plan_eps{epsilon:.0e}"
                        f"_seed{experiment_seed}.npz"
                    ),
                    anchors=np.stack(anchors, axis=0),
                    adapted_h=np.asarray(adapted_h),
                    h_center=np.asarray(h_center),
                    h_selected=np.asarray(h_selected),
                )

            chain_results: List[boundary.BoundaryChainResult] = []
            print("Fixed-h production chains")
            for chain_index, anchor in enumerate(anchors):
                result = boundary.cone_mala_boundary_chain(
                    post=post,
                    w_init=np.asarray(anchor).copy(),
                    warmup=production_warmup,
                    retained=retained,
                    thin=thin,
                    h=h_selected,
                    seed=(
                        330_000_000
                        + 1_000_000 * epsilon_index
                        + 10_000 * experiment_seed
                        + chain_index
                    ),
                )
                chain_results.append(result)

                print(
                    f"  chain {chain_index + 1}: "
                    f"acc={result.acceptance:.3f}, "
                    f"warmup={result.warmup_acceptance:.3f}, "
                    f"invalid="
                    f"{result.final_invalid_spd / max(result.final_proposals,1):.3f}, "
                    f"failures="
                    f"{result.final_numerical_failures / max(result.final_proposals,1):.3f}, "
                    f"time={result.runtime_sec:.1f}s"
                )

            row, diagnostics = boundary.summarize_boundary_method_seed(
                method=METHOD,
                experiment_seed=experiment_seed,
                epsilon=epsilon,
                post=post,
                chain_results=chain_results,
                pilot_runtime_sec=burnin_runtime + tuning_runtime,
                parallel_runtime=config.parallel_runtime,
                boundary_stride=args.boundary_stride,
            )
            row["prior_mode"] = args.prior_mode
            row["nu"] = post.nu
            row["preburn_runtime_sec"] = burnin_runtime
            row["typical_tuning_runtime_sec"] = tuning_runtime
            row["preburn_h_median"] = h_center
            row["selected_h"] = h_selected
            seed_rows.append(row)
            diagnostic_rows.extend(diagnostics)

            print(
                "Summary: "
                f"acc={row['acceptance']:.3f}, "
                f"invalid={row['final_invalid_spd_rate']:.3f}, "
                f"min ESS/iter={row['min_ess_per_iteration']:.4f}, "
                f"min ESS/s samp="
                f"{row['min_ess_per_sec_sampling']:.3f}, "
                f"Rhat={row['max_rhat']:.3f}"
            )

            checkpoint(
                output_dir=args.output_dir,
                burnin_rows=burnin_rows,
                tuning_rows=tuning_rows,
                seed_rows=seed_rows,
                diagnostic_rows=diagnostic_rows,
                truth_rows=truth_rows,
            )

    summary_df = boundary.aggregate_boundary_results(
        pd.DataFrame(seed_rows)
    )
    summary_df["prior_mode"] = args.prior_mode
    summary_df.to_csv(
        args.output_dir / "conemala_typical_summary.csv",
        index=False,
    )
    boundary.write_boundary_latex_table(
        summary_df,
        args.output_dir / "table_conemala_typical.tex",
    )

    if args.make_plots:
        boundary.make_plots(summary_df, args.output_dir)

    print("\nAcross-seed typical-set-aware ConeMALA summary")
    print("=" * 88)
    columns = [
        "epsilon",
        "method",
        "acceptance_mean",
        "min_ess_per_iteration_mean",
        "min_ess_per_sec_sampling_mean",
        "min_ess_per_sec_total_mean",
        "max_rhat_mean",
        "final_invalid_spd_rate_mean",
    ]
    print(summary_df[columns].to_string(index=False))

    print("\nSaved outputs:")
    for name in (
        "config.json",
        "boundary_truth.csv",
        "conemala_typical_burnin.csv",
        "conemala_typical_tuning.csv",
        "conemala_typical_by_seed.csv",
        "conemala_typical_diagnostics.csv",
        "conemala_typical_summary.csv",
        "table_conemala_typical.tex",
    ):
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
