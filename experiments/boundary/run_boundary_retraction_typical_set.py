#!/usr/bin/env python3
r"""
Post-burn-in typical-set tuning for the Retraction RMALA boundary experiment.

This runner matches the long convergence-rescue setup used for ConeMALA,
Fisher-adaptive MALA, and Euclidean MALA:

1. method-specific adaptive pre-burn-in from four dispersed chains;
2. a fixed-h settling period;
3. typical-set tuning from the terminal burn-in states;
4. fixed-kernel production warmup and retained sampling.

All adaptation and tuning samples are discarded.

Important interpretation
------------------------
The retraction proposal used here is

    W' = W + W^{1/2} S W^{1/2},

with an exact Metropolis--Hastings correction for that proposal density.
Unlike the affine-invariant exponential map, this first-order retraction does
not guarantee that W' is SPD. Invalid-SPD proposals are therefore counted.
This experiment isolates the benefit of the exact exponential-map geometry
relative to a cheaper first-order affine-invariant retraction; it is not a
pure positivity-preserving-coordinate ablation.

Required companion files
------------------------
Keep this script beside

    conemala_fisher_adaptive_comparison.py
    boundary_stress_comparison.py

Long rescue run
---------------
python3 run_boundary_retraction_typical_set.py \
  --config results_fisher_paper/config.json \
  --profile paper \
  --epsilons 1e-2 \
  --prior-mode boundary-friendly \
  --boundary-nu-offset 1.0 \
  --preburn-steps 8000 \
  --settle-steps 1000 \
  --production-warmup 10000 \
  --tune-warmup 500 \
  --tune-steps 1000 \
  --retained 20000 \
  --output-dir results_boundary_retraction_convergence_rescue_sequential
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
METHOD = "RetractionRMALA"


@dataclass
class RetractionBurninResult:
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


def prior_metadata(
    config: base.ExperimentConfig,
    prior_mode: str,
    boundary_nu_offset: float,
) -> Dict[str, object]:
    effective_nu, effective_psi = boundary.resolve_prior(
        config=config,
        prior_mode=prior_mode,
        boundary_nu_offset=boundary_nu_offset,
    )
    diagonal = np.diag(effective_psi)
    effective_scale = float(np.mean(diagonal))
    if not np.allclose(
        effective_psi,
        effective_scale * np.eye(config.d),
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("Expected an isotropic effective prior scale.")

    source_mean_scale = float(config.nu * config.prior_scale)
    effective_mean_scale = float(effective_nu * effective_scale)

    return {
        "prior_mode": prior_mode,
        "boundary_nu_offset": float(boundary_nu_offset),
        "dimension_d": int(config.d),
        "source_nu": float(config.nu),
        "source_prior_scale": float(config.prior_scale),
        "source_prior_mean_scale": source_mean_scale,
        "effective_nu": float(effective_nu),
        "effective_prior_scale": effective_scale,
        "effective_prior_mean_scale": effective_mean_scale,
        "determinant_exponent": float(
            0.5 * (effective_nu - config.d - 1.0)
        ),
        "prior_mean_preserved": bool(
            math.isclose(
                source_mean_scale,
                effective_mean_scale,
                rel_tol=1e-12,
                abs_tol=1e-14,
            )
        ),
        "wishart_parameterization": "E[W_e] = nu * Psi",
        "metadata_origin": "explicit",
    }


def effective_experiment_config(
    config: base.ExperimentConfig,
    prior_info: Dict[str, object],
) -> Dict[str, object]:
    result = asdict(config)
    result["nu"] = prior_info["effective_nu"]
    result["prior_scale"] = prior_info["effective_prior_scale"]
    result["prior_mode"] = prior_info["prior_mode"]
    result["boundary_nu_offset"] = prior_info[
        "boundary_nu_offset"
    ]
    result["determinant_exponent"] = prior_info[
        "determinant_exponent"
    ]
    return result


def adaptive_retraction_burnin(
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
) -> RetractionBurninResult:
    """
    Discarded Retraction-RMALA burn-in with diminishing Robbins--Monro
    adaptation of log h, followed by a fixed-h settling phase.

    Invalid-SPD proposals receive acceptance probability zero and therefore
    push h downward during adaptation.
    """
    if adaptive_steps < 1:
        raise ValueError("adaptive_steps must be positive.")
    if settle_steps < 0:
        raise ValueError("settle_steps must be nonnegative.")
    if not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must lie in (0,1).")
    if not 0.5 < adapt_exponent <= 1.0:
        raise ValueError("adapt_exponent must lie in (0.5,1].")
    if not 0.0 < h_min < h_max:
        raise ValueError("Require 0 < h_min < h_max.")

    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    volume_c = 0.5 * (post.d + 1.0)

    log_h = float(
        np.clip(math.log(h0), math.log(h_min), math.log(h_max))
    )
    accepted = 0
    invalid_spd = 0
    numerical_failures = 0
    total_steps = adaptive_steps + settle_steps
    start = time.perf_counter()

    for iteration in range(total_steps):
        h_now = float(math.exp(log_h))
        w_prop = np.empty_like(w)
        tangent_forward: List[Array] = []
        tangent_mean_forward: List[Array] = []

        proposal_invalid = False
        numerical_failure = False
        alpha_prob = 0.0
        accepted_now = False

        try:
            for edge_index in range(post.e_count):
                root, _ = base.spd_sqrt_and_inv_sqrt(w[edge_index])
                mean_tangent = h_now * base.sym(
                    root @ evaluation.score_dx[edge_index] @ root
                )
                tangent = base.sym(
                    mean_tangent
                    + math.sqrt(2.0 * h_now)
                    * base.sample_standard_symmetric(post.d, rng)
                )
                w_prop[edge_index] = base.sym(
                    w[edge_index] + root @ tangent @ root
                )
                if not base.is_spd(w_prop[edge_index]):
                    proposal_invalid = True
                    break
                tangent_forward.append(tangent)
                tangent_mean_forward.append(mean_tangent)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            numerical_failure = True

        if not proposal_invalid and not numerical_failure:
            try:
                eval_prop = post.evaluate(w_prop)
                logq_forward = 0.0
                logq_reverse = 0.0

                for edge_index in range(post.e_count):
                    sf = tangent_forward[edge_index]
                    mf = tangent_mean_forward[edge_index]
                    logq_forward += (
                        base.log_gaussian_isotropic(
                            base.svec(sf - mf), 2.0 * h_now
                        )
                        - volume_c
                        * base.spd_logdet(w[edge_index])
                    )

                    root_prop, inv_root_prop = (
                        base.spd_sqrt_and_inv_sqrt(
                            w_prop[edge_index]
                        )
                    )
                    reverse_tangent = base.sym(
                        inv_root_prop
                        @ (w[edge_index] - w_prop[edge_index])
                        @ inv_root_prop
                    )
                    reverse_mean = h_now * base.sym(
                        root_prop
                        @ eval_prop.score_dx[edge_index]
                        @ root_prop
                    )
                    logq_reverse += (
                        base.log_gaussian_isotropic(
                            base.svec(
                                reverse_tangent - reverse_mean
                            ),
                            2.0 * h_now,
                        )
                        - volume_c
                        * base.spd_logdet(w_prop[edge_index])
                    )

                log_alpha = (
                    eval_prop.log_target_dx
                    + logq_reverse
                    - evaluation.log_target_dx
                    - logq_forward
                )
                alpha_prob = float(math.exp(min(0.0, log_alpha)))

                if rng.uniform() < alpha_prob:
                    w = w_prop
                    evaluation = eval_prop
                    accepted_now = True
            except (
                np.linalg.LinAlgError,
                FloatingPointError,
                ValueError,
            ):
                numerical_failure = True
                alpha_prob = 0.0

        if proposal_invalid:
            invalid_spd += 1
        if numerical_failure:
            numerical_failures += 1
        accepted += int(accepted_now)

        if iteration < adaptive_steps:
            eta = diminishing_rate(
                iteration=iteration,
                initial_rate=adapt_rate,
                decay=adapt_decay,
                exponent=adapt_exponent,
            )
            log_h += eta * (alpha_prob - target_accept)
            log_h = float(
                np.clip(
                    log_h,
                    math.log(h_min),
                    math.log(h_max),
                )
            )

    return RetractionBurninResult(
        terminal_w=w.copy(),
        h_final=float(math.exp(log_h)),
        acceptance=accepted / max(total_steps, 1),
        invalid_spd_rate=invalid_spd / max(total_steps, 1),
        numerical_failure_rate=(
            numerical_failures / max(total_steps, 1)
        ),
        runtime_sec=float(time.perf_counter() - start),
        adaptive_steps=adaptive_steps,
        settle_steps=settle_steps,
    )


def product_airm_esjd_per_iteration(samples: Array) -> float:
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
    Tune one shared h from clones of post-burn-in anchor states.

    Candidate selection uses product-AIRM ESJD per retained transition and
    does not use wall-clock time.
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
        invalid_rates: List[float] = []
        failure_rates: List[float] = []
        runtimes: List[float] = []

        for chain_index, anchor in enumerate(anchor_states):
            result = boundary.retraction_rmala_boundary_chain(
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
            invalid_rates.append(
                result.final_invalid_spd
                / max(result.final_proposals, 1)
            )
            failure_rates.append(
                result.final_numerical_failures
                / max(result.final_proposals, 1)
            )
            runtimes.append(result.runtime_sec)

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
            "airm_esjd_per_iteration": float(
                np.mean(esjd_values)
            ),
            "pilot_invalid_spd_rate": float(
                np.mean(invalid_rates)
            ),
            "pilot_numerical_failure_rate": float(
                np.mean(failure_rates)
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

    # Expand toward larger h if all tested acceptances are too high.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] <= upper for row in rows):
            break
        evaluate(rows[-1]["h"] * 2.0)

    # Expand toward smaller h if all tested acceptances are too low.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] >= lower for row in rows):
            break
        evaluate(rows[0]["h"] / 2.0)

    for _ in range(bisection_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        high = [
            row
            for row in rows
            if row["pilot_acceptance"] >= target_accept
        ]
        low = [
            row
            for row in rows
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
        row
        for row in rows
        if lower <= row["pilot_acceptance"] <= upper
    ]

    if admissible:
        best = max(
            admissible,
            key=lambda row: (
                row["airm_esjd_per_iteration"],
                -abs(
                    row["pilot_acceptance"] - target_accept
                ),
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
    return {"smoke": 5, "pilot": 250, "paper": 500}[profile]


def default_tune_warmup(profile: str) -> int:
    return {"smoke": 0, "pilot": 100, "paper": 200}[profile]


def checkpoint(
    output_dir: Path,
    burnin_rows: List[Dict[str, object]],
    tuning_rows: List[Dict[str, object]],
    seed_rows: List[Dict[str, object]],
    diagnostic_rows: List[Dict[str, object]],
    truth_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(burnin_rows).to_csv(
        output_dir / "retraction_typical_burnin.csv",
        index=False,
    )
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "retraction_typical_tuning.csv",
        index=False,
    )
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "retraction_typical_by_seed.csv",
        index=False,
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "retraction_typical_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(truth_rows).to_csv(
        output_dir / "boundary_truth.csv",
        index=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-burn-in typical-set Retraction-RMALA "
            "boundary experiment."
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
    parser.add_argument("--h-min", type=float, default=1e-12)
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
    h0 = (
        config.retraction_h0
        if args.h0 is None
        else args.h0
    )

    prior_info = prior_metadata(
        config=config,
        prior_mode=args.prior_mode,
        boundary_nu_offset=args.boundary_nu_offset,
    )
    source_profile_config = asdict(config)
    effective_config = effective_experiment_config(
        config=config,
        prior_info=prior_info,
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
        "acceptance_tolerance": (
            args.acceptance_tolerance
        ),
        "max_bracket_steps": args.max_bracket_steps,
        "bisection_steps": args.bisection_steps,
        "boundary_stride": args.boundary_stride,
        "source_profile_config": source_profile_config,
        "resolved_experiment_config": effective_config,
        "effective_prior": prior_info,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Typical-set-aware Retraction RMALA boundary experiment")
    print("=" * 88)
    print(json.dumps(run_config, indent=2))

    burnin_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    seed_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    truth_rows: List[Dict[str, object]] = []

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
            truth_stats = boundary.matrix_family_stats(
                post.w_true
            )
            truth_rows.append(
                {
                    "epsilon": epsilon,
                    "truth_condition_target": 1.0 / epsilon,
                    "experiment_seed": experiment_seed,
                    "prior_mode": args.prior_mode,
                    "nu": post.nu,
                    "source_nu": prior_info["source_nu"],
                    "effective_nu": prior_info["effective_nu"],
                    "source_prior_scale": prior_info[
                        "source_prior_scale"
                    ],
                    "effective_prior_scale": prior_info[
                        "effective_prior_scale"
                    ],
                    "source_prior_mean_scale": prior_info[
                        "source_prior_mean_scale"
                    ],
                    "effective_prior_mean_scale": prior_info[
                        "effective_prior_mean_scale"
                    ],
                    "determinant_exponent": prior_info[
                        "determinant_exponent"
                    ],
                    "prior_mean_preserved": prior_info[
                        "prior_mean_preserved"
                    ],
                    **truth_stats,
                }
            )

            w_center = base.initial_weights(post)
            anchors: List[Array] = []
            adapted_h: List[float] = []
            burnin_runtime = 0.0

            print("Preliminary adaptive Retraction-RMALA burn-in")
            for chain_index in range(config.n_chains):
                w_init = boundary.fixed_initial_state(
                    w_center=w_center,
                    experiment_seed=experiment_seed,
                    epsilon_index=epsilon_index,
                    method=METHOD,
                    chain_index=chain_index,
                    n_chains=config.n_chains,
                )
                result = adaptive_retraction_burnin(
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
                        140_000_000
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
                        "stage": "adaptive_preburn",
                        "h_final": result.h_final,
                        "acceptance": result.acceptance,
                        "invalid_spd_rate": (
                            result.invalid_spd_rate
                        ),
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
                acceptance_tolerance=(
                    args.acceptance_tolerance
                ),
                tune_warmup=tune_warmup,
                tune_steps=tune_steps,
                factors=(0.25, 0.5, 1.0, 2.0, 4.0),
                max_bracket_steps=(
                    args.max_bracket_steps
                ),
                bisection_steps=args.bisection_steps,
                seed_base=(
                    150_000_000
                    + 1_000_000 * epsilon_index
                    + 10_000 * experiment_seed
                ),
            )

            for row in rows:
                row.update(
                    {
                        "epsilon": epsilon,
                        "truth_condition_target": (
                            1.0 / epsilon
                        ),
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

            chain_results: List[
                boundary.BoundaryChainResult
            ] = []
            print("Fixed-h production chains")
            for chain_index, anchor in enumerate(anchors):
                result = (
                    boundary.retraction_rmala_boundary_chain(
                        post=post,
                        w_init=np.asarray(anchor).copy(),
                        warmup=production_warmup,
                        retained=retained,
                        thin=thin,
                        h=h_selected,
                        seed=(
                            160_000_000
                            + 1_000_000 * epsilon_index
                            + 10_000 * experiment_seed
                            + chain_index
                        ),
                    )
                )
                chain_results.append(result)
                print(
                    f"  chain {chain_index + 1}: "
                    f"acc={result.acceptance:.3f}, "
                    f"warmup={result.warmup_acceptance:.3f}, "
                    f"invalid="
                    f"{result.final_invalid_spd / max(result.final_proposals, 1):.3f}, "
                    f"failures="
                    f"{result.final_numerical_failures / max(result.final_proposals, 1):.3f}, "
                    f"time={result.runtime_sec:.1f}s"
                )

            row, diagnostics = (
                boundary.summarize_boundary_method_seed(
                    method=METHOD,
                    experiment_seed=experiment_seed,
                    epsilon=epsilon,
                    post=post,
                    chain_results=chain_results,
                    pilot_runtime_sec=(
                        burnin_runtime + tuning_runtime
                    ),
                    parallel_runtime=config.parallel_runtime,
                    boundary_stride=args.boundary_stride,
                )
            )
            row["prior_mode"] = args.prior_mode
            row["nu"] = post.nu
            row["source_nu"] = prior_info["source_nu"]
            row["effective_nu"] = prior_info["effective_nu"]
            row["source_prior_scale"] = prior_info[
                "source_prior_scale"
            ]
            row["effective_prior_scale"] = prior_info[
                "effective_prior_scale"
            ]
            row["determinant_exponent"] = prior_info[
                "determinant_exponent"
            ]
            row["preburn_runtime_sec"] = burnin_runtime
            row["typical_tuning_runtime_sec"] = (
                tuning_runtime
            )
            row["preburn_h_median"] = h_center

            seed_rows.append(row)
            diagnostic_rows.extend(diagnostics)

            print(
                "Summary: "
                f"acc={row['acceptance']:.3f}, "
                f"invalid={row['final_invalid_spd_rate']:.3f}, "
                f"min ESS/iter="
                f"{row['min_ess_per_iteration']:.4f}, "
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
        args.output_dir / "retraction_typical_summary.csv",
        index=False,
    )
    boundary.write_boundary_latex_table(
        summary_df,
        args.output_dir / "table_retraction_typical.tex",
    )

    if args.make_plots:
        boundary.make_plots(summary_df, args.output_dir)

    print("\nAcross-seed typical-set-aware Retraction summary")
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
        "retraction_typical_burnin.csv",
        "retraction_typical_tuning.csv",
        "retraction_typical_by_seed.csv",
        "retraction_typical_diagnostics.csv",
        "retraction_typical_summary.csv",
        "table_retraction_typical.tex",
    ):
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
