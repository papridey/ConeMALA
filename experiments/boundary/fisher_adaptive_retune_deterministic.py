#!/usr/bin/env python3
r"""
Two-stage retuning for the Fisher-adaptive MALA baseline used in
conemala_fisher_adaptive_comparison.py.

The procedure is deliberately separated into three phases:

1. Adapt one empirical inverse-Fisher preconditioner in each pilot chain.
2. Pool the pilot preconditioners into one shared matrix P and freeze it.
3. Tune the scalar step size h on a logarithmic grid with P fixed, selecting
   among acceptance-calibrated candidates by Fisher-standardized ESJD per
   iteration, then run independent exact preconditioned-MALA chains with both
   P and h fixed.

The retained kernel is

    y | x ~ N(x + h P score(x), 2 h P),

where score(x) = grad log pi_Leb(x). Because P and h are fixed during final
sampling, the ordinary Metropolis--Hastings correction is exact.

This file is intended to be placed in the same directory as

    conemala_fisher_adaptive_comparison.py

and run separately. It can optionally replace the old Fisher rows in the
comparison CSV files and regenerate the full LaTeX table.

Examples
--------
Pilot retuning:

    python fisher_adaptive_retune.py \
        --config results_fisher_pilot/config.json \
        --output-dir results_fisher_pilot_retuned

Paper retuning:

    python fisher_adaptive_retune.py \
        --config results_fisher_paper/config.json \
        --output-dir results_fisher_paper_retuned

After both the original comparison and the retuned Fisher run have finished,
replace the original Fisher rows and regenerate the complete table:

    python fisher_adaptive_retune.py \
        --config results_fisher_paper/config.json \
        --output-dir results_fisher_paper_retuned \
        --update-comparison-dir results_fisher_paper
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import conemala_fisher_adaptive_comparison as base


Array = np.ndarray


@dataclass
class AdaptationResult:
    root: Array
    h_final: float
    acceptance: float
    runtime_sec: float
    n_fisher_updates: int


@dataclass
class FisherPlan:
    root: Array
    h_selected: float
    condition_number: float
    adaptation_runtime_sec: float
    tuning_runtime_sec: float
    adaptation_rows: List[Dict[str, float]]
    tuning_rows: List[Dict[str, float]]


def normalized_root(root: Array, max_condition: float) -> Tuple[Array, float]:
    """
    Normalize P = root root^T so that tr(P)/q = 1 and clip its condition number.
    """
    p = base.sym(root @ root.T)
    eig, vec = np.linalg.eigh(p)
    eig = np.maximum(eig, np.finfo(float).tiny)
    largest = float(np.max(eig))
    eig = np.maximum(eig, largest / max_condition)
    eig /= float(np.mean(eig))
    p = base.sym((vec * eig) @ vec.T)
    root_out = np.linalg.cholesky(p)
    return root_out, float(np.max(eig) / np.min(eig))


def adapt_single_fisher_chain(
    post: base.GraphPosterior,
    w_init: Array,
    n_steps: int,
    h0: float,
    target_accept: float,
    damping: float,
    init_steps: int,
    update_interval: int,
    max_condition: float,
    step_adapt_rate: float,
    seed: int,
) -> AdaptationResult:
    """
    Pilot-only Fisher adaptation. No draw from this phase is retained.

    This follows the same score-difference update used in the original script,
    but returns the adapted square-root preconditioner so several pilot chains
    can be pooled before final sampling.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    if update_interval < 1:
        raise ValueError("update_interval must be positive.")
    if damping <= 0.0:
        raise ValueError("damping must be positive.")

    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    x = base.flatten_weights(w)
    qdim = x.size

    root = np.eye(qdim)
    fisher_initialized = False
    log_h = math.log(h0)
    accepted = 0
    n_updates = 0
    t0 = time.perf_counter()

    for iteration in range(n_steps):
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
        eval_prop: Optional[base.PosteriorEvaluation] = None

        if all(base.is_spd(w_prop[e]) for e in range(post.e_count)):
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
                alpha_prob = 0.0

        accepted += int(accepted_now)

        use_signal = (
            iteration >= init_steps
            and (iteration - init_steps) % update_interval == 0
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

        # Robbins--Monro-style multiplicative adaptation of h.
        factor = 1.0 + step_adapt_rate * (alpha_prob - target_accept)
        factor = max(factor, 1e-6)
        log_h += math.log(factor)
        log_h = float(np.clip(log_h, math.log(1e-12), math.log(10.0)))

    root, _ = normalized_root(root, max_condition)
    return AdaptationResult(
        root=root,
        h_final=float(math.exp(log_h)),
        acceptance=accepted / n_steps,
        runtime_sec=float(time.perf_counter() - t0),
        n_fisher_updates=n_updates,
    )


def pool_preconditioners(
    results: Sequence[AdaptationResult],
    max_condition: float,
) -> Tuple[Array, float, float]:
    """
    Average the normalized inverse-Fisher preconditioners from pilot chains.

    Returns
    -------
    root_shared:
        Cholesky factor of the shared P, normalized by tr(P)/q = 1.
    condition_number:
        Condition number of the frozen shared P.
    h_center:
        Median adapted scalar step size, used as the center of the tuning grid.
    """
    if not results:
        raise ValueError("At least one adaptation result is required.")

    matrices = [base.sym(result.root @ result.root.T) for result in results]
    p_shared = base.sym(np.mean(np.stack(matrices, axis=0), axis=0))
    eig, vec = np.linalg.eigh(p_shared)
    eig = np.maximum(eig, np.finfo(float).tiny)
    largest = float(np.max(eig))
    eig = np.maximum(eig, largest / max_condition)
    eig /= float(np.mean(eig))
    p_shared = base.sym((vec * eig) @ vec.T)
    root_shared = np.linalg.cholesky(p_shared)
    condition_number = float(np.max(eig) / np.min(eig))
    h_center = float(np.median([result.h_final for result in results]))
    return root_shared, condition_number, h_center


def fixed_fisher_mala_chain(
    post: base.GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    root: Array,
    condition_number: float,
    seed: int,
) -> base.ChainResult:
    """
    Exact fixed-preconditioner MALA.

    The preconditioner P = root root^T and the scalar h are fixed throughout
    warmup and retained sampling. Warmup here is ordinary burn-in, not
    adaptation.
    """
    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    x = base.flatten_weights(w)
    qdim = x.size

    samples: List[Array] = []
    traces: Dict[str, List[float]] = {
        "phi": [],
        "logdet_q": [],
        "heldout_nll": [],
        "rel_w_draw": [],
    }

    n_total = warmup + retained * thin
    accepted_warm = 0
    accepted_keep = 0
    proposed_keep = 0
    t0 = time.perf_counter()

    for iteration in range(n_total):
        score_x = base.flatten_weights(evaluation.score_dx)
        p_score_x = root @ (root.T @ score_x)
        mean_x = x + h * p_score_x
        y = mean_x + math.sqrt(2.0 * h) * (
            root @ rng.standard_normal(qdim)
        )
        w_prop = base.unflatten_weights(y, post.e_count, post.d)
        accepted = False

        if all(base.is_spd(w_prop[e]) for e in range(post.e_count)):
            try:
                eval_prop = post.evaluate(w_prop)
                score_y = base.flatten_weights(eval_prop.score_dx)
                log_q_ratio = base._fisher_mala_log_proposal_ratio(
                    x=x,
                    y=y,
                    score_x=score_x,
                    score_y=score_y,
                    root=root,
                    average_eigenvalue=1.0,
                    h=h,
                )
                log_alpha = (
                    eval_prop.log_target_dx
                    - evaluation.log_target_dx
                    + log_q_ratio
                )
                if math.log(rng.uniform()) < min(0.0, log_alpha):
                    x = y
                    w = w_prop
                    evaluation = eval_prop
                    accepted = True
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                accepted = False

        if iteration < warmup:
            accepted_warm += int(accepted)
        else:
            proposed_keep += 1
            accepted_keep += int(accepted)
            if (iteration - warmup) % thin == 0:
                base.append_retained(samples, traces, post, w, evaluation)

    return base.ChainResult(
        samples=np.asarray(samples),
        traces={key: np.asarray(value) for key, value in traces.items()},
        acceptance=accepted_keep / max(proposed_keep, 1),
        warmup_acceptance=accepted_warm / max(warmup, 1),
        runtime_sec=float(time.perf_counter() - t0),
        h_final=float(h),
        preconditioner_condition=float(condition_number),
    )


def standardized_esjd_metrics(
    samples: Array,
    root: Array,
    runtime_sec: float,
) -> Tuple[float, float]:
    """
    Compute Fisher-standardized jump diagnostics.

    Let P = root @ root.T and delta_k = x_{k+1} - x_k.  The squared
    preconditioner-standardized jump is

        delta_k.T @ P^{-1} @ delta_k
        = ||root^{-1} delta_k||_2^2.

    Returns
    -------
    esjd_per_iteration:
        Mean standardized squared jump per retained transition.  This quantity
        is independent of wall-clock time and is used to select the tuning
        step size reproducibly.
    movement_per_second:
        Sum of standardized squared jumps divided by runtime.  This is saved
        only as a timing diagnostic and is never used to select the step size.

    Samples have shape (n_draws, n_edges, d, d).
    """
    if samples.shape[0] < 2:
        return 0.0, 0.0

    flat = np.stack([base.flatten_weights(w) for w in samples], axis=0)
    deltas = np.diff(flat, axis=0)

    squared_jumps = np.empty(deltas.shape[0], dtype=float)
    for index, delta in enumerate(deltas):
        whitened = np.linalg.solve(root, delta)
        squared_jumps[index] = float(np.dot(whitened, whitened))

    esjd_per_iteration = float(np.mean(squared_jumps))
    movement_per_second = float(
        np.sum(squared_jumps) / max(float(runtime_sec), 1e-15)
    )
    return esjd_per_iteration, movement_per_second


def tune_fixed_fisher_h(
    post: base.GraphPosterior,
    w_center: Array,
    root: Array,
    condition_number: float,
    h_center: float,
    target_accept: float,
    tune_steps: int,
    tune_chains: int,
    n_chains_total: int,
    factors: Sequence[float],
    seed_base: int,
    acceptance_tolerance: float = 0.08,
    max_bracket_steps: int = 20,
    bisection_steps: int = 8,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Tune h with the preconditioner frozen using log-scale bracketing.

    The routine first evaluates the requested relative grid. If all candidates
    are too conservative, it repeatedly doubles the largest h until the target
    acceptance is bracketed. If all candidates are too aggressive, it halves
    the smallest h. Once a high-acceptance and a low-acceptance candidate are
    available, it performs logarithmic bisection.
    """
    if tune_steps < 10:
        raise ValueError("tune_steps should be at least 10.")
    if tune_chains < 1:
        raise ValueError("tune_chains must be positive.")
    if max_bracket_steps < 1:
        raise ValueError("max_bracket_steps must be positive.")
    if bisection_steps < 0:
        raise ValueError("bisection_steps must be nonnegative.")
    if acceptance_tolerance <= 0.0:
        raise ValueError("acceptance_tolerance must be positive.")
    if not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must lie in (0, 1).")

    tested: Dict[float, Dict[str, float]] = {}
    total_runtime = 0.0

    def evaluate_candidate(h: float) -> Dict[str, float]:
        nonlocal total_runtime
        h = float(h)
        if h <= 0.0 or not np.isfinite(h):
            raise ValueError(f"Invalid step size h={h}.")
        if h in tested:
            return tested[h]

        acceptances: List[float] = []
        esjd_per_iteration_values: List[float] = []
        movement_per_second_values: List[float] = []
        runtimes: List[float] = []

        # Common chain seeds are deliberately reused across h values so the
        # acceptance curve is less noisy when candidates are compared.
        for chain_index in range(tune_chains):
            init_rng = base.set_seed(seed_base + 100 * chain_index)
            w_init = base.perturb_initial_weights(
                w_center,
                chain_index=chain_index,
                n_chains=max(n_chains_total, tune_chains),
                rng=init_rng,
            )
            result = fixed_fisher_mala_chain(
                post=post,
                w_init=w_init,
                warmup=0,
                retained=tune_steps,
                thin=1,
                h=h,
                root=root,
                condition_number=condition_number,
                seed=seed_base + 1_000_000 + chain_index,
            )
            acceptances.append(result.acceptance)
            runtimes.append(result.runtime_sec)
            esjd_per_iteration, movement_per_second = (
                standardized_esjd_metrics(
                    result.samples,
                    root,
                    result.runtime_sec,
                )
            )
            esjd_per_iteration_values.append(esjd_per_iteration)
            movement_per_second_values.append(movement_per_second)

        row = {
            "method": "FisherAdaptiveMALA",
            "stage": "fixed_h_tuning",
            "h": h,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1)
                if len(acceptances) > 1 else 0.0
            ),
            "standardized_esjd_per_iteration": float(
                np.mean(esjd_per_iteration_values)
            ),
            "standardized_movement_per_sec": float(
                np.mean(movement_per_second_values)
            ),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[h] = row
        total_runtime += row["pilot_runtime_sec"]
        return row

    factor_list = sorted(set(float(f) for f in factors if f > 0.0))
    if not factor_list:
        raise ValueError("At least one positive Fisher grid factor is required.")
    h_center = max(float(h_center), 1e-12)
    for factor in factor_list:
        evaluate_candidate(h_center * factor)

    upper_target = target_accept + acceptance_tolerance
    lower_target = target_accept - acceptance_tolerance

    # Expand upward until at least one candidate is not overly conservative.
    for _ in range(max_bracket_steps):
        rows = [tested[h] for h in sorted(tested)]
        if any(row["pilot_acceptance"] <= upper_target for row in rows):
            break
        evaluate_candidate(max(tested) * 2.0)

    # Expand downward only when all candidates are too aggressive.
    for _ in range(max_bracket_steps):
        rows = [tested[h] for h in sorted(tested)]
        if any(row["pilot_acceptance"] >= lower_target for row in rows):
            break
        evaluate_candidate(min(tested) / 2.0)

    # Find the nearest acceptance bracket around the target and refine it.
    for _ in range(bisection_steps):
        rows = [tested[h] for h in sorted(tested)]
        high_accept = [
            row for row in rows if row["pilot_acceptance"] >= target_accept
        ]
        low_accept = [
            row for row in rows if row["pilot_acceptance"] <= target_accept
        ]
        if not high_accept or not low_accept:
            break

        left = max(high_accept, key=lambda row: row["h"])
        right = min(low_accept, key=lambda row: row["h"])
        if left["h"] >= right["h"]:
            break
        h_mid = math.sqrt(float(left["h"]) * float(right["h"]))
        if h_mid in tested:
            break
        evaluate_candidate(h_mid)

    rows = [tested[h] for h in sorted(tested)]

    # Use the requested acceptance interval.  Within that interval, select the
    # candidate with the largest Fisher-standardized ESJD per iteration.
    # Runtime is deliberately excluded from the selection criterion so that
    # repeating the run on the same seeds does not change h merely because of
    # CPU scheduling or background load.
    admissible = [
        row
        for row in rows
        if (
            lower_target
            <= float(row["pilot_acceptance"])
            <= upper_target
        )
    ]

    if admissible:
        best = max(
            admissible,
            key=lambda row: (
                float(row["standardized_esjd_per_iteration"]),
                -abs(float(row["pilot_acceptance"]) - target_accept),
                -float(row["h"]),
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                abs(float(row["pilot_acceptance"]) - target_accept),
                -float(row["standardized_esjd_per_iteration"]),
                float(row["h"]),
            ),
        )

    for row in rows:
        row["selected"] = bool(
            math.isclose(
                float(row["h"]),
                float(best["h"]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

    return float(best["h"]), float(total_runtime), rows

def prepare_fisher_plan(
    post: base.GraphPosterior,
    w_center: Array,
    n_chains: int,
    adapt_chains: int,
    adapt_steps: int,
    tune_chains: int,
    tune_steps: int,
    h0: float,
    target_accept: float,
    damping: float,
    init_steps: int,
    update_interval: int,
    max_condition: float,
    step_adapt_rate: float,
    grid_factors: Sequence[float],
    seed_base: int,
    acceptance_tolerance: float,
    max_bracket_steps: int,
    bisection_steps: int,
) -> FisherPlan:
    """
    Estimate one shared preconditioner and tune one shared h for a dataset.
    """
    adaptation_results: List[AdaptationResult] = []
    adaptation_rows: List[Dict[str, float]] = []
    adaptation_runtime = 0.0

    for chain_index in range(adapt_chains):
        init_rng = base.set_seed(seed_base + 100 * chain_index)
        w_init = base.perturb_initial_weights(
            w_center,
            chain_index=chain_index,
            n_chains=max(n_chains, adapt_chains),
            rng=init_rng,
        )
        result = adapt_single_fisher_chain(
            post=post,
            w_init=w_init,
            n_steps=adapt_steps,
            h0=h0,
            target_accept=target_accept,
            damping=damping,
            init_steps=min(init_steps, max(adapt_steps - 1, 0)),
            update_interval=update_interval,
            max_condition=max_condition,
            step_adapt_rate=step_adapt_rate,
            seed=seed_base + 10_000 + chain_index,
        )
        adaptation_results.append(result)
        adaptation_runtime += result.runtime_sec
        adaptation_rows.append(
            {
                "method": "FisherAdaptiveMALA",
                "stage": "preconditioner_adaptation",
                "adaptation_chain": chain_index + 1,
                "h": result.h_final,
                "pilot_acceptance": result.acceptance,
                "pilot_runtime_sec": result.runtime_sec,
                "n_fisher_updates": result.n_fisher_updates,
            }
        )

    root, condition_number, h_center = pool_preconditioners(
        adaptation_results, max_condition
    )

    h_selected, tuning_runtime, tuning_rows = tune_fixed_fisher_h(
        post=post,
        w_center=w_center,
        root=root,
        condition_number=condition_number,
        h_center=h_center,
        target_accept=target_accept,
        tune_steps=tune_steps,
        tune_chains=tune_chains,
        n_chains_total=n_chains,
        factors=grid_factors,
        seed_base=seed_base + 2_000_000,
        acceptance_tolerance=acceptance_tolerance,
        max_bracket_steps=max_bracket_steps,
        bisection_steps=bisection_steps,
    )

    return FisherPlan(
        root=root,
        h_selected=h_selected,
        condition_number=condition_number,
        adaptation_runtime_sec=float(adaptation_runtime),
        tuning_runtime_sec=float(tuning_runtime),
        adaptation_rows=adaptation_rows,
        tuning_rows=tuning_rows,
    )


def load_config(path: Path) -> base.ExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["experiment_seeds"] = tuple(int(x) for x in data["experiment_seeds"])
    data["methods"] = tuple(str(x) for x in data["methods"])
    return base.ExperimentConfig(**data)


def aggregate_and_write(
    seed_rows: List[Dict[str, float]],
    diagnostic_rows: List[Dict[str, float]],
    tuning_rows: List[Dict[str, float]],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_df = pd.DataFrame(seed_rows)
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    tuning_df = pd.DataFrame(tuning_rows)
    summary_df = base.aggregate_across_seeds(seed_df)

    seed_df.to_csv(output_dir / "fisher_retuned_by_seed.csv", index=False)
    diagnostic_df.to_csv(
        output_dir / "fisher_retuned_diagnostics.csv", index=False
    )
    tuning_df.to_csv(output_dir / "fisher_retuned_tuning.csv", index=False)
    summary_df.to_csv(output_dir / "fisher_retuned_summary.csv", index=False)
    base.write_latex_table(
        summary_df, output_dir / "table_fisher_retuned_only.tex"
    )
    return seed_df, diagnostic_df, tuning_df, summary_df


def update_existing_comparison(
    comparison_dir: Path,
    retuned_seed_df: pd.DataFrame,
    retuned_diagnostic_df: pd.DataFrame,
    retuned_tuning_df: pd.DataFrame,
) -> None:
    """
    Replace old Fisher rows in the original comparison outputs and regenerate
    the complete repeated-seed table.
    """
    seed_path = comparison_dir / "sampler_comparison_by_seed.csv"
    diagnostic_path = comparison_dir / "sampler_diagnostics_by_seed.csv"
    tuning_path = comparison_dir / "pilot_tuning.csv"

    if not seed_path.exists() or not diagnostic_path.exists():
        raise FileNotFoundError(
            "The original comparison must finish before --update-comparison-dir "
            "can be used."
        )

    old_seed = pd.read_csv(seed_path)
    old_diag = pd.read_csv(diagnostic_path)
    old_tuning = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()

    merged_seed = pd.concat(
        [
            old_seed[old_seed["method"] != "FisherAdaptiveMALA"],
            retuned_seed_df,
        ],
        ignore_index=True,
    )
    merged_diag = pd.concat(
        [
            old_diag[old_diag["method"] != "FisherAdaptiveMALA"],
            retuned_diagnostic_df,
        ],
        ignore_index=True,
    )

    if old_tuning.empty:
        merged_tuning = retuned_tuning_df.copy()
    else:
        merged_tuning = pd.concat(
            [
                old_tuning[old_tuning["method"] != "FisherAdaptiveMALA"],
                retuned_tuning_df,
            ],
            ignore_index=True,
        )

    merged_summary = base.aggregate_across_seeds(merged_seed)

    merged_seed.to_csv(
        comparison_dir / "sampler_comparison_by_seed_retuned.csv", index=False
    )
    merged_diag.to_csv(
        comparison_dir / "sampler_diagnostics_by_seed_retuned.csv", index=False
    )
    merged_tuning.to_csv(
        comparison_dir / "pilot_tuning_retuned.csv", index=False
    )
    merged_summary.to_csv(
        comparison_dir / "sampler_comparison_across_seeds_retuned.csv",
        index=False,
    )
    base.write_latex_table(
        merged_summary,
        comparison_dir / "table_fisher_repeated_seed_comparison_retuned.tex",
    )


def parse_factors(text: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "Fisher grid factors must be positive comma-separated numbers."
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage pooled Fisher preconditioner and fixed-h retuning."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--update-comparison-dir", type=Path)

    parser.add_argument("--adapt-chains", type=int)
    parser.add_argument("--adapt-steps", type=int)
    parser.add_argument("--tune-chains", type=int)
    parser.add_argument("--tune-steps", type=int)
    parser.add_argument("--final-warmup", type=int)
    parser.add_argument("--retained", type=int)
    parser.add_argument("--thin", type=int)

    parser.add_argument(
        "--grid-factors",
        type=parse_factors,
        default=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        help="Comma-separated factors around the median adapted h.",
    )
    parser.add_argument(
        "--condition-cap", type=float, default=100.0,
        help="Maximum condition number of the frozen pooled preconditioner.",
    )
    parser.add_argument(
        "--acceptance-tolerance", type=float, default=0.08,
        help="Tolerance around the target acceptance during bracketing.",
    )
    parser.add_argument("--max-bracket-steps", type=int, default=20)
    parser.add_argument("--bisection-steps", type=int, default=8)
    parser.add_argument("--save-preconditioners", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    base.validate_config(config)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    adapt_chains = (
        config.n_chains if args.adapt_chains is None else args.adapt_chains
    )
    adapt_steps = config.warmup if args.adapt_steps is None else args.adapt_steps
    tune_chains = (
        min(config.n_chains, 4)
        if args.tune_chains is None else args.tune_chains
    )
    tune_steps = (
        config.pilot_steps if args.tune_steps is None else args.tune_steps
    )
    final_warmup = (
        config.warmup if args.final_warmup is None else args.final_warmup
    )
    retained = config.retained if args.retained is None else args.retained
    thin = config.thin if args.thin is None else args.thin

    run_config = {
        "source_config": str(args.config),
        "adapt_chains": adapt_chains,
        "adapt_steps": adapt_steps,
        "tune_chains": tune_chains,
        "tune_steps": tune_steps,
        "final_warmup": final_warmup,
        "retained": retained,
        "thin": thin,
        "grid_factors": list(args.grid_factors),
        "condition_cap": args.condition_cap,
        "acceptance_tolerance": args.acceptance_tolerance,
        "max_bracket_steps": args.max_bracket_steps,
        "bisection_steps": args.bisection_steps,
        "fisher_step_selection_metric": (
            "standardized_esjd_per_iteration"
        ),
    }
    (output_dir / "retuning_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    all_seed_rows: List[Dict[str, float]] = []
    all_diagnostic_rows: List[Dict[str, float]] = []
    all_tuning_rows: List[Dict[str, float]] = []

    print("Two-stage Fisher-adaptive MALA retuning")
    print("=" * 88)
    print(json.dumps(run_config, indent=2))

    for experiment_seed in config.experiment_seeds:
        print("\n" + "=" * 88)
        print(f"Independent experiment seed: {experiment_seed}")
        print("=" * 88)

        post = base.make_synthetic_problem(config, experiment_seed)
        w_center = base.initial_weights(post)

        plan = prepare_fisher_plan(
            post=post,
            w_center=w_center,
            n_chains=config.n_chains,
            adapt_chains=adapt_chains,
            adapt_steps=adapt_steps,
            tune_chains=tune_chains,
            tune_steps=tune_steps,
            h0=config.fisher_h0,
            target_accept=config.target_accept,
            damping=config.fisher_damping,
            init_steps=config.fisher_init_steps,
            update_interval=config.fisher_update_interval,
            max_condition=min(config.fisher_max_condition, args.condition_cap),
            step_adapt_rate=config.step_adapt_rate,
            grid_factors=args.grid_factors,
            seed_base=7_000_000 + 10_000 * experiment_seed,
            acceptance_tolerance=args.acceptance_tolerance,
            max_bracket_steps=args.max_bracket_steps,
            bisection_steps=args.bisection_steps,
        )

        for row in plan.adaptation_rows + plan.tuning_rows:
            row["experiment_seed"] = int(experiment_seed)
            row["shared_preconditioner_condition"] = plan.condition_number
            all_tuning_rows.append(row)

        print(
            f"shared P condition={plan.condition_number:.3e}, "
            f"selected h={plan.h_selected:.3e}, "
            f"adapt time={plan.adaptation_runtime_sec:.1f}s, "
            f"tune time={plan.tuning_runtime_sec:.1f}s"
        )

        if args.save_preconditioners:
            np.savez_compressed(
                output_dir / f"fisher_plan_seed{experiment_seed}.npz",
                root=plan.root,
                h_selected=np.asarray(plan.h_selected),
                condition_number=np.asarray(plan.condition_number),
            )

        chain_results: List[base.ChainResult] = []
        for chain_index in range(config.n_chains):
            init_rng = base.set_seed(
                8_000_000 + 10_000 * experiment_seed + chain_index
            )
            w_init = base.perturb_initial_weights(
                w_center,
                chain_index=chain_index,
                n_chains=config.n_chains,
                rng=init_rng,
            )
            result = fixed_fisher_mala_chain(
                post=post,
                w_init=w_init,
                warmup=final_warmup,
                retained=retained,
                thin=thin,
                h=plan.h_selected,
                root=plan.root,
                condition_number=plan.condition_number,
                seed=9_000_000 + 10_000 * experiment_seed + chain_index,
            )
            chain_results.append(result)
            print(
                f"chain {chain_index + 1}: "
                f"acc={result.acceptance:.3f}, "
                f"burnin_acc={result.warmup_acceptance:.3f}, "
                f"h={result.h_final:.3e}, "
                f"time={result.runtime_sec:.1f}s"
            )

        pilot_runtime = (
            plan.adaptation_runtime_sec + plan.tuning_runtime_sec
        )
        row, diagnostics = base.summarize_method_seed(
            method="FisherAdaptiveMALA",
            experiment_seed=experiment_seed,
            post=post,
            chain_results=chain_results,
            pilot_runtime_sec=pilot_runtime,
            parallel_runtime=config.parallel_runtime,
        )
        all_seed_rows.append(row)
        all_diagnostic_rows.extend(diagnostics)

        print(
            "summary: "
            f"acc={row['acceptance']:.3f}, "
            f"min ESS/s={row['min_ess_per_sec']:.3f}, "
            f"MCSE(NLL)={row['mcse_heldout_nll']:.4g}, "
            f"max Rhat={row['max_rhat']:.3f}, "
            f"NLL={row['heldout_nll']:.4f}, "
            f"relWerr={row['relative_w_error']:.4f}"
        )

    seed_df, diagnostic_df, tuning_df, summary_df = aggregate_and_write(
        all_seed_rows,
        all_diagnostic_rows,
        all_tuning_rows,
        output_dir,
    )

    print("\nAcross-seed retuned Fisher summary")
    print("=" * 88)
    columns = [
        "method",
        "acceptance_mean",
        "min_ess_per_sec_mean",
        "mcse_heldout_nll_mean",
        "max_rhat_mean",
        "heldout_nll_mean",
        "relative_w_error_mean",
    ]
    print(summary_df[columns].to_string(index=False))

    if args.update_comparison_dir is not None:
        update_existing_comparison(
            args.update_comparison_dir,
            seed_df,
            diagnostic_df,
            tuning_df,
        )
        print(
            "\nUpdated full comparison outputs written to:\n"
            f"  {args.update_comparison_dir / 'sampler_comparison_by_seed_retuned.csv'}\n"
            f"  {args.update_comparison_dir / 'sampler_comparison_across_seeds_retuned.csv'}\n"
            f"  {args.update_comparison_dir / 'table_fisher_repeated_seed_comparison_retuned.tex'}"
        )

    print("\nSaved retuning outputs:")
    for name in (
        "retuning_config.json",
        "fisher_retuned_by_seed.csv",
        "fisher_retuned_diagnostics.csv",
        "fisher_retuned_tuning.csv",
        "fisher_retuned_summary.csv",
        "table_fisher_retuned_only.tex",
    ):
        print(f"  {output_dir / name}")


if __name__ == "__main__":
    main()
