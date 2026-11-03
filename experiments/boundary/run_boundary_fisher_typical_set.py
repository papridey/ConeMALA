#!/usr/bin/env python3
r"""
Typical-set-aware boundary experiment for Fisher-adaptive MALA.

Purpose
-------
The original boundary pilot estimated a Fisher preconditioner and tuned h from
relatively interior initial states.  Production chains later moved toward a
boundary-concentrated typical set, where Euclidean-coordinate proposals became
mostly infeasible.

This script uses a fairer five-stage protocol for every (epsilon, data seed):

1. Run one discarded adaptive Fisher burn-in per chain.
2. Retain only each terminal state and adapted inverse-Fisher square root.
3. Pool and condition-cap the chain-specific preconditioners.
4. Tune one shared scalar h from clones of the terminal boundary-near states
   with the pooled preconditioner frozen, using P^{-1}-standardized ESJD per
   retained transition.
5. Run production chains from the terminal states with both P and h fixed.

All adaptation and tuning draws are discarded.  The retained kernel is exact
fixed-preconditioner MALA.

Required companion files
------------------------
Keep this script beside

    conemala_fisher_adaptive_comparison.py
    boundary_stress_comparison.py
    fisher_adaptive_retune_deterministic.py

Recommended pilot
-----------------
python3 run_boundary_fisher_typical_set.py \
    --config results_fisher_paper/config.json \
    --profile pilot \
    --epsilons 1e-1,1e-2,1e-3 \
    --prior-mode boundary-friendly \
    --boundary-nu-offset 1.0 \
    --output-dir results_boundary_fisher_typical

Paper run
---------
python3 run_boundary_fisher_typical_set.py \
    --config results_fisher_paper/config.json \
    --profile paper \
    --epsilons 1e-1,1e-2,1e-3 \
    --prior-mode boundary-friendly \
    --boundary-nu-offset 1.0 \
    --preburn-steps 4000 \
    --settle-steps 500 \
    --tune-warmup 200 \
    --tune-steps 400 \
    --condition-cap 100 \
    --output-dir results_boundary_fisher_typical_paper
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import conemala_fisher_adaptive_comparison as base
import boundary_stress_comparison as boundary
import fisher_adaptive_retune_deterministic as fisher


Array = np.ndarray
METHOD = "FisherAdaptiveMALA"


@dataclass
class FisherBurninResult:
    terminal_w: Array
    root: Array
    h_final: float
    acceptance: float
    invalid_spd_rate: float
    numerical_failure_rate: float
    runtime_sec: float
    n_fisher_updates: int
    adaptive_steps: int
    settle_steps: int


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one epsilon is required.")
    if any(not (0.0 < value <= 1.0) for value in values):
        raise argparse.ArgumentTypeError("Every epsilon must lie in (0,1].")
    return values


def diminishing_rate(
    iteration: int,
    initial_rate: float,
    decay: float,
    exponent: float,
) -> float:
    return float(
        initial_rate / (1.0 + iteration / max(decay, 1e-12)) ** exponent
    )


def adaptive_fisher_burnin(
    post: base.GraphPosterior,
    w_init: Array,
    adaptive_steps: int,
    settle_steps: int,
    h0: float,
    target_accept: float,
    damping: float,
    fisher_init_steps: int,
    update_interval: int,
    condition_cap: float,
    adapt_rate: float,
    adapt_decay: float,
    adapt_exponent: float,
    h_min: float,
    h_max: float,
    seed: int,
) -> FisherBurninResult:
    """
    Discarded Fisher-adaptive burn-in returning both its terminal state and its
    normalized inverse-Fisher square-root preconditioner.

    The first phase adapts P and log h.  The second phase freezes both and
    allows the chain to settle before its terminal state becomes a tuning
    anchor.
    """
    if adaptive_steps < 1:
        raise ValueError("adaptive_steps must be positive.")
    if settle_steps < 0:
        raise ValueError("settle_steps must be nonnegative.")
    if update_interval < 1:
        raise ValueError("update_interval must be positive.")
    if damping <= 0.0:
        raise ValueError("damping must be positive.")
    if not 0.5 < adapt_exponent <= 1.0:
        raise ValueError("adapt_exponent should lie in (0.5,1].")
    if not 0.0 < h_min < h_max:
        raise ValueError("Require 0 < h_min < h_max.")

    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    x = base.flatten_weights(w)
    qdim = x.size

    root = np.eye(qdim)
    fisher_initialized = False
    log_h = float(np.clip(math.log(h0), math.log(h_min), math.log(h_max)))

    accepted = 0
    invalid_spd = 0
    numerical_failures = 0
    n_updates = 0
    start = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: adapt P and h.  All states are discarded.
    # ------------------------------------------------------------------
    for iteration in range(adaptive_steps):
        h_now = float(math.exp(log_h))
        score_x = base.flatten_weights(evaluation.score_dx)
        p_score_x, average_eigenvalue = base._normalized_fisher_action(
            root, score_x
        )
        mean_x = x + h_now * p_score_x
        y = mean_x + (
            math.sqrt(2.0 * h_now / average_eigenvalue)
            * (root @ rng.standard_normal(qdim))
        )
        w_prop = base.unflatten_weights(y, post.e_count, post.d)

        alpha_prob = 0.0
        accepted_now = False
        score_y: Optional[Array] = None
        proposal_invalid = not all(
            base.is_spd(w_prop[edge]) for edge in range(post.e_count)
        )

        if proposal_invalid:
            invalid_spd += 1
        else:
            try:
                eval_prop = post.evaluate(w_prop)
                score_y = base.flatten_weights(eval_prop.score_dx)
                log_q_ratio = base._fisher_mala_log_proposal_ratio(
                    x=x,
                    y=y,
                    score_x=score_x,
                    score_y=score_y,
                    root=root,
                    average_eigenvalue=average_eigenvalue,
                    h=h_now,
                )
                log_alpha = (
                    eval_prop.log_target_dx
                    - evaluation.log_target_dx
                    + log_q_ratio
                )
                alpha_prob = float(math.exp(min(0.0, log_alpha)))
                if rng.uniform() < alpha_prob:
                    x = y
                    w = w_prop
                    evaluation = eval_prop
                    accepted_now = True
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                numerical_failures += 1
                alpha_prob = 0.0
                score_y = None

        accepted += int(accepted_now)

        use_signal = (
            iteration >= fisher_init_steps
            and (iteration - fisher_init_steps) % update_interval == 0
            and score_y is not None
            and alpha_prob > 0.0
        )
        if use_signal:
            signal = math.sqrt(alpha_prob) * (score_y - score_x)
            if np.all(np.isfinite(signal)) and np.linalg.norm(signal) > 0.0:
                if not fisher_initialized:
                    root = base._fisher_sqrt_initialize(signal, damping)
                    fisher_initialized = True
                else:
                    root = base._fisher_sqrt_update(root, signal)
                n_updates += 1

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

    root, condition_number = fisher.normalized_root(root, condition_cap)
    h_final = float(math.exp(log_h))

    # ------------------------------------------------------------------
    # Phase 2: fixed-P, fixed-h settling.  These states are also discarded.
    # ------------------------------------------------------------------
    if settle_steps > 0:
        settle = boundary.fixed_fisher_boundary_chain(
            post=post,
            w_init=w,
            warmup=0,
            retained=settle_steps,
            thin=1,
            h=h_final,
            root=root,
            condition_number=condition_number,
            seed=seed + 5_000_000,
        )
        w = np.asarray(settle.samples[-1]).copy()
        accepted += int(round(settle.acceptance * settle.final_proposals))
        invalid_spd += settle.final_invalid_spd
        numerical_failures += settle.final_numerical_failures

    total_steps = adaptive_steps + settle_steps
    return FisherBurninResult(
        terminal_w=w.copy(),
        root=root,
        h_final=h_final,
        acceptance=accepted / max(total_steps, 1),
        invalid_spd_rate=invalid_spd / max(total_steps, 1),
        numerical_failure_rate=numerical_failures / max(total_steps, 1),
        runtime_sec=float(time.perf_counter() - start),
        n_fisher_updates=n_updates,
        adaptive_steps=adaptive_steps,
        settle_steps=settle_steps,
    )


def pool_fisher_burnins(
    results: Sequence[FisherBurninResult],
    condition_cap: float,
) -> Tuple[Array, float, float]:
    if not results:
        raise ValueError("At least one Fisher burn-in result is required.")

    adaptation_results = [
        fisher.AdaptationResult(
            root=result.root,
            h_final=result.h_final,
            acceptance=result.acceptance,
            runtime_sec=result.runtime_sec,
            n_fisher_updates=result.n_fisher_updates,
        )
        for result in results
    ]
    return fisher.pool_preconditioners(
        adaptation_results,
        max_condition=condition_cap,
    )


def tune_fisher_h_from_typical_states(
    post: base.GraphPosterior,
    anchor_states: Sequence[Array],
    root: Array,
    condition_number: float,
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
    Tune one shared h from boundary-near anchors with shared P frozen.

    Candidate selection uses P^{-1}-standardized ESJD per retained transition.
    """
    if not anchor_states:
        raise ValueError("At least one anchor state is required.")
    if tune_steps < 10:
        raise ValueError("tune_steps must be at least 10.")

    tested: Dict[str, Dict[str, float]] = {}

    def evaluate(h: float) -> Dict[str, float]:
        h = float(max(h, 1e-14))
        key = f"{h:.17e}"
        if key in tested:
            return tested[key]

        acceptances: List[float] = []
        esjd_values: List[float] = []
        movement_rates: List[float] = []
        invalid_rates: List[float] = []
        runtimes: List[float] = []

        for chain_index, anchor in enumerate(anchor_states):
            result = boundary.fixed_fisher_boundary_chain(
                post=post,
                w_init=np.asarray(anchor, dtype=float).copy(),
                warmup=tune_warmup,
                retained=tune_steps,
                thin=1,
                h=h,
                root=root,
                condition_number=condition_number,
                seed=seed_base + chain_index,
            )
            esjd_iter, movement_sec = fisher.standardized_esjd_metrics(
                result.samples,
                root,
                result.runtime_sec,
            )
            acceptances.append(result.acceptance)
            esjd_values.append(esjd_iter)
            movement_rates.append(movement_sec)
            invalid_rates.append(
                result.final_invalid_spd / max(result.final_proposals, 1)
            )
            runtimes.append(result.runtime_sec)

        row = {
            "method": METHOD,
            "stage": "typical_set_fixed_h_tuning",
            "h": h,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1)
                if len(acceptances) > 1 else 0.0
            ),
            "standardized_esjd_per_iteration": float(
                np.mean(esjd_values)
            ),
            "standardized_movement_per_sec": float(
                np.mean(movement_rates)
            ),
            "pilot_invalid_spd_rate": float(np.mean(invalid_rates)),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[key] = row
        return row

    for factor in sorted(set(float(value) for value in factors if value > 0.0)):
        evaluate(h_center * factor)

    lower = target_accept - acceptance_tolerance
    upper = target_accept + acceptance_tolerance

    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] <= upper for row in rows):
            break
        evaluate(rows[-1]["h"] * 2.0)

    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] >= lower for row in rows):
            break
        evaluate(rows[0]["h"] / 2.0)

    for _ in range(bisection_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        high = [
            row for row in rows
            if row["pilot_acceptance"] >= target_accept
        ]
        low = [
            row for row in rows
            if row["pilot_acceptance"] <= target_accept
        ]
        if not high or not low:
            break
        left = max(high, key=lambda row: row["h"])
        right = min(low, key=lambda row: row["h"])
        if left["h"] >= right["h"]:
            break
        h_mid = math.sqrt(left["h"] * right["h"])
        if f"{h_mid:.17e}" in tested:
            break
        evaluate(h_mid)

    rows = sorted(tested.values(), key=lambda row: row["h"])
    admissible = [
        row for row in rows
        if lower <= row["pilot_acceptance"] <= upper
    ]

    if admissible:
        best = max(
            admissible,
            key=lambda row: (
                row["standardized_esjd_per_iteration"],
                -abs(row["pilot_acceptance"] - target_accept),
                -row["h"],
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                abs(row["pilot_acceptance"] - target_accept),
                -row["standardized_esjd_per_iteration"],
                row["h"],
            ),
        )

    for row in rows:
        row["selected"] = bool(
            math.isclose(
                row["h"], best["h"], rel_tol=1e-12, abs_tol=1e-15
            )
        )

    total_runtime = float(sum(row["pilot_runtime_sec"] for row in rows))
    return (
        float(best["h"]),
        float(best["pilot_acceptance"]),
        total_runtime,
        rows,
    )


def default_preburn_steps(profile: str, config: base.ExperimentConfig) -> int:
    if profile == "smoke":
        return 20
    if profile == "pilot":
        return max(2000, 2 * config.warmup)
    return max(4000, 2 * config.warmup)


def default_settle_steps(profile: str) -> int:
    return {"smoke": 5, "pilot": 250, "paper": 500}[profile]


def default_tune_warmup(profile: str) -> int:
    return {"smoke": 0, "pilot": 100, "paper": 200}[profile]


def checkpoint(
    output_dir: Path,
    burnin_rows: List[Dict[str, float]],
    tuning_rows: List[Dict[str, float]],
    seed_rows: List[Dict[str, float]],
    diagnostic_rows: List[Dict[str, float]],
    truth_rows: List[Dict[str, float]],
) -> None:
    pd.DataFrame(burnin_rows).to_csv(
        output_dir / "fisher_typical_burnin.csv", index=False
    )
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "fisher_typical_tuning.csv", index=False
    )
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "fisher_typical_by_seed.csv", index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "fisher_typical_diagnostics.csv", index=False
    )
    pd.DataFrame(truth_rows).to_csv(
        output_dir / "boundary_truth.csv", index=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Typical-set-aware Fisher-MALA boundary experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "paper"), default="pilot"
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
    parser.add_argument("--boundary-nu-offset", type=float, default=1.0)

    parser.add_argument("--preburn-steps", type=int)
    parser.add_argument("--settle-steps", type=int)
    parser.add_argument("--production-warmup", type=int)
    parser.add_argument("--retained", type=int)
    parser.add_argument("--thin", type=int)

    parser.add_argument("--h0", type=float)
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--adapt-decay", type=float, default=100.0)
    parser.add_argument("--adapt-exponent", type=float, default=0.6)
    parser.add_argument("--h-min", type=float, default=1e-12)
    parser.add_argument("--h-max", type=float, default=1.0)

    parser.add_argument("--damping", type=float)
    parser.add_argument("--fisher-init-steps", type=int)
    parser.add_argument("--update-interval", type=int)
    parser.add_argument("--condition-cap", type=float, default=100.0)

    parser.add_argument("--tune-warmup", type=int)
    parser.add_argument("--tune-steps", type=int)
    parser.add_argument("--acceptance-tolerance", type=float, default=0.08)
    parser.add_argument("--max-bracket-steps", type=int, default=20)
    parser.add_argument("--bisection-steps", type=int, default=8)
    parser.add_argument("--boundary-stride", type=int, default=10)
    parser.add_argument("--save-preconditioners", action="store_true")
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
        if args.preburn_steps is None else args.preburn_steps
    )
    settle_steps = (
        default_settle_steps(args.profile)
        if args.settle_steps is None else args.settle_steps
    )
    production_warmup = (
        config.warmup
        if args.production_warmup is None else args.production_warmup
    )
    retained = config.retained if args.retained is None else args.retained
    thin = config.thin if args.thin is None else args.thin
    tune_warmup = (
        default_tune_warmup(args.profile)
        if args.tune_warmup is None else args.tune_warmup
    )
    tune_steps = (
        config.pilot_steps if args.tune_steps is None else args.tune_steps
    )

    h0 = config.fisher_h0 if args.h0 is None else args.h0
    damping = (
        config.fisher_damping if args.damping is None else args.damping
    )
    fisher_init_steps = (
        min(config.fisher_init_steps, max(preburn_steps - 1, 0))
        if args.fisher_init_steps is None else args.fisher_init_steps
    )
    update_interval = (
        config.fisher_update_interval
        if args.update_interval is None else args.update_interval
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
        "damping": damping,
        "fisher_init_steps": fisher_init_steps,
        "update_interval": update_interval,
        "condition_cap": args.condition_cap,
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
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print("Typical-set-aware Fisher-adaptive MALA boundary experiment")
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
            burnins: List[FisherBurninResult] = []
            burnin_runtime = 0.0

            print("Preliminary adaptive Fisher burn-in")
            for chain_index in range(config.n_chains):
                w_init = boundary.fixed_initial_state(
                    w_center=w_center,
                    experiment_seed=experiment_seed,
                    epsilon_index=epsilon_index,
                    method=METHOD,
                    chain_index=chain_index,
                    n_chains=config.n_chains,
                )
                result = adaptive_fisher_burnin(
                    post=post,
                    w_init=w_init,
                    adaptive_steps=preburn_steps,
                    settle_steps=settle_steps,
                    h0=h0,
                    target_accept=config.target_accept,
                    damping=damping,
                    fisher_init_steps=fisher_init_steps,
                    update_interval=update_interval,
                    condition_cap=args.condition_cap,
                    adapt_rate=args.adapt_rate,
                    adapt_decay=args.adapt_decay,
                    adapt_exponent=args.adapt_exponent,
                    h_min=args.h_min,
                    h_max=args.h_max,
                    seed=(
                        210_000_000
                        + 1_000_000 * epsilon_index
                        + 10_000 * experiment_seed
                        + chain_index
                    ),
                )
                burnins.append(result)
                burnin_runtime += result.runtime_sec
                burnin_rows.append(
                    {
                        "epsilon": epsilon,
                        "truth_condition_target": 1.0 / epsilon,
                        "experiment_seed": experiment_seed,
                        "method": METHOD,
                        "chain": chain_index + 1,
                        "stage": "adaptive_fisher_preburn",
                        "h_final": result.h_final,
                        "acceptance": result.acceptance,
                        "invalid_spd_rate": result.invalid_spd_rate,
                        "numerical_failure_rate": (
                            result.numerical_failure_rate
                        ),
                        "runtime_sec": result.runtime_sec,
                        "n_fisher_updates": result.n_fisher_updates,
                        "adaptive_steps": result.adaptive_steps,
                        "settle_steps": result.settle_steps,
                    }
                )
                print(
                    f"  chain {chain_index + 1}: "
                    f"h_final={result.h_final:.3e}, "
                    f"acc={result.acceptance:.3f}, "
                    f"invalid={result.invalid_spd_rate:.3f}, "
                    f"updates={result.n_fisher_updates}, "
                    f"time={result.runtime_sec:.1f}s"
                )

            root, condition_number, h_center = pool_fisher_burnins(
                burnins,
                condition_cap=args.condition_cap,
            )
            anchors = [result.terminal_w for result in burnins]

            h_selected, pilot_acceptance, tuning_runtime, rows = (
                tune_fisher_h_from_typical_states(
                    post=post,
                    anchor_states=anchors,
                    root=root,
                    condition_number=condition_number,
                    h_center=h_center,
                    target_accept=config.target_accept,
                    acceptance_tolerance=args.acceptance_tolerance,
                    tune_warmup=tune_warmup,
                    tune_steps=tune_steps,
                    factors=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
                    max_bracket_steps=args.max_bracket_steps,
                    bisection_steps=args.bisection_steps,
                    seed_base=(
                        220_000_000
                        + 1_000_000 * epsilon_index
                        + 10_000 * experiment_seed
                    ),
                )
            )
            for row in rows:
                row.update(
                    {
                        "epsilon": epsilon,
                        "truth_condition_target": 1.0 / epsilon,
                        "experiment_seed": experiment_seed,
                        "shared_preconditioner_condition": condition_number,
                    }
                )
                tuning_rows.append(row)

            print(
                f"Typical-set tuning: condition={condition_number:.3e}, "
                f"center={h_center:.3e}, selected h={h_selected:.3e}, "
                f"pilot acceptance={pilot_acceptance:.3f}, "
                f"time={tuning_runtime:.1f}s"
            )

            if args.save_preconditioners:
                np.savez_compressed(
                    args.output_dir
                    / f"fisher_typical_plan_eps{epsilon:.0e}_seed{experiment_seed}.npz",
                    root=root,
                    condition_number=np.asarray(condition_number),
                    h_center=np.asarray(h_center),
                    h_selected=np.asarray(h_selected),
                )

            chain_results: List[boundary.BoundaryChainResult] = []
            print("Fixed-P, fixed-h production chains")
            for chain_index, anchor in enumerate(anchors):
                result = boundary.fixed_fisher_boundary_chain(
                    post=post,
                    w_init=np.asarray(anchor).copy(),
                    warmup=production_warmup,
                    retained=retained,
                    thin=thin,
                    h=h_selected,
                    root=root,
                    condition_number=condition_number,
                    seed=(
                        230_000_000
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
                    f"invalid={result.final_invalid_spd / max(result.final_proposals,1):.3f}, "
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
            row["shared_preconditioner_condition"] = condition_number
            seed_rows.append(row)
            diagnostic_rows.extend(diagnostics)

            print(
                "Summary: "
                f"acc={row['acceptance']:.3f}, "
                f"invalid={row['final_invalid_spd_rate']:.3f}, "
                f"min ESS/iter={row['min_ess_per_iteration']:.4f}, "
                f"min ESS/s samp={row['min_ess_per_sec_sampling']:.3f}, "
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
        args.output_dir / "fisher_typical_summary.csv", index=False
    )
    boundary.write_boundary_latex_table(
        summary_df,
        args.output_dir / "table_fisher_typical.tex",
    )

    if args.make_plots:
        boundary.make_plots(summary_df, args.output_dir)

    print("\nAcross-seed typical-set-aware Fisher summary")
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
        "fisher_typical_burnin.csv",
        "fisher_typical_tuning.csv",
        "fisher_typical_by_seed.csv",
        "fisher_typical_diagnostics.csv",
        "fisher_typical_summary.csv",
        "table_fisher_typical.tex",
    ):
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
