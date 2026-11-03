"""
Drift ablation for ConeMALA on the PSD-weighted graph boundary experiment.

Scientific question
-------------------
Holding the product affine-invariant geometry, exponential-map proposal,
tangent Gaussian noise, full posterior target, and exact Metropolis--Hastings
correction fixed, how much does the proposal drift contribute?

The four proposal variants are

    zero   : AIRM random walk,                  M_e(W) = 0,
    prior  : intrinsic prior-only drift,        M_e(W) = M_e^prior(W),
    local  : edge-local likelihood approximation,
    full   : exact globally coupled graph drift.

For every mode, the proposal is

    S_e = h M_e(W) + sqrt(2h) Z_e,
    W'_e = W_e^{1/2} exp(S_e) W_e^{1/2},

with the exact forward/reverse proposal densities and the same full posterior
in the MH ratio.  Consequently all four fixed production kernels target the
same posterior; only proposal efficiency changes.

Intrinsic drift convention
--------------------------
The target density used by the AIRM kernel is relative to product AIRM volume,

    Phi(W) = -log pi_Leb(W | Y) - (d+1)/2 sum_e log det W_e.

Hence the full whitened drift is

    M_e^full(W)
      = -W_e^{1/2} [grad_e Phi(W)] W_e^{1/2}
      = W_e^{1/2} [score_like,e + score_prior,e] W_e^{1/2}
        + (d+1)/2 I.

The prior-only and local variants retain the same AIRM-volume correction.  The
zero-drift variant sets the entire proposal mean to zero.

Edge-local approximation
------------------------
Let B_e be the p-by-d edge-incidence block operator, Q(W) the graph precision,
and G_e = B_e^T B_e (equal to 2 I_d for an ordinary graph edge).  The exact
likelihood score contains

    C_e^full = B_e^T Q(W)^{-1} B_e.

The local ablation replaces it by the projected-precision approximation

    C_e^local = G_e [B_e^T Q(W) B_e]^{-1} G_e.

This uses only the precision seen in the edge-difference subspace.  It is exact
when Q=rho I and avoids the globally coupled solve Q^{-1}[B_1 ... B_E].

Fair tuning
-----------
Each mode receives:

1. the same dispersed initial states;
2. discarded adaptive pre-burn-in;
3. discarded fixed-h settling;
4. typical-set step-size tuning using product-AIRM ESJD per transition;
5. fixed-h production warmup and retained sampling.

The random-walk mode has its own acceptance target (default 0.30), while the
three drifted modes use the MALA target inherited from the source config
(default 0.574).  This avoids deliberately handicapping the random walk.  To
force an identical target, pass --random-walk-target-accept 0.574.

Required companion files
------------------------
Keep this script beside

    conemala_fisher_adaptive_comparison.py
    boundary_stress_comparison.py
    fisher_adaptive_retune_deterministic.py

Smoke test
----------
python3 -u run_boundary_drift_ablation.py \
  --config results_fisher_paper/config.json \
  --profile smoke \
  --epsilons 1e-2 \
  --prior-mode boundary-friendly \
  --boundary-nu-offset 1.0 \
  --preburn-steps 20 \
  --settle-steps 5 \
  --production-warmup 10 \
  --tune-warmup 0 \
  --tune-steps 20 \
  --retained 20 \
  --output-dir results_boundary_drift_ablation_smoke

Long boundary-rescue ablation
-----------------------------
python3 -u run_boundary_drift_ablation.py \
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
  --output-dir results_boundary_drift_ablation_sequential \
  --make-plots
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import conemala_fisher_adaptive_comparison as base
import boundary_stress_comparison as boundary


Array = np.ndarray

MODE_ORDER = ("zero", "prior", "local", "full")
METHOD_NAME = {
    "zero": "AIRMRandomWalk",
    "prior": "AIRMPriorDrift",
    "local": "AIRMLocalDrift",
    "full": "AIRMFullGraphDrift",
}
METHOD_LABEL = {
    "AIRMRandomWalk": "AIRM random walk",
    "AIRMPriorDrift": "AIRM prior-only drift",
    "AIRMLocalDrift": "AIRM edge-local drift",
    "AIRMFullGraphDrift": "AIRM full graph drift",
}
SHARED_INITIALIZATION_TAG = 710_000_000


@dataclass
class DriftEvaluation:
    """Target evaluation and the selected whitened tangent drift."""

    log_target_dx: float
    phi: float
    q_matrix: Array
    logdet_q: float
    heldout_nll: float
    roots: Array
    drift_tangent: Array


@dataclass
class DriftBurninResult:
    terminal_w: Array
    h_final: float
    acceptance: float
    invalid_spd_rate: float
    numerical_failure_rate: float
    runtime_sec: float
    adaptive_steps: int
    settle_steps: int


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one epsilon is required.")
    if any(not (0.0 < x <= 1.0) for x in values):
        raise argparse.ArgumentTypeError("Every epsilon must lie in (0,1].")
    return values


def parse_modes(text: str) -> Tuple[str, ...]:
    values = tuple(x.strip().lower() for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one drift mode is required.")
    unknown = sorted(set(values) - set(MODE_ORDER))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown drift mode(s): {unknown}. Choose from {MODE_ORDER}."
        )
    # Preserve canonical order and remove duplicates.
    return tuple(mode for mode in MODE_ORDER if mode in set(values))


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
    result["boundary_nu_offset"] = prior_info["boundary_nu_offset"]
    result["determinant_exponent"] = prior_info["determinant_exponent"]
    return result


def shared_initial_state(
    w_center: Array,
    experiment_seed: int,
    epsilon_index: int,
    chain_index: int,
    n_chains: int,
) -> Array:
    """Return the same dispersed initial state for every drift mode."""
    rng = base.set_seed(
        SHARED_INITIALIZATION_TAG
        + 1_000_000 * epsilon_index
        + 10_000 * experiment_seed
        + chain_index
    )
    return base.perturb_initial_weights(
        w_center,
        chain_index=chain_index,
        n_chains=n_chains,
        rng=rng,
    )


class DriftEvaluator:
    """
    Mode-specific target and drift evaluator.

    The zero/prior/local modes do not compute the globally coupled
    Q^{-1}[B_1 ... B_E] solve.  The full mode does.  This makes ESS/sec a
    meaningful measure of the cost of enriching the drift, while every mode
    still evaluates the exact full posterior target and exact MH ratio.
    """

    def __init__(self, post: base.GraphPosterior, mode: str) -> None:
        if mode not in MODE_ORDER:
            raise ValueError(f"Unknown drift mode: {mode}")
        self.post = post
        self.mode = mode
        self.eye_d = np.eye(post.d)
        self.edge_grams = tuple(
            base.sym(b.T @ b) for b in post.edge_ops
        )

    def evaluate(self, w: Array) -> DriftEvaluation:
        post = self.post
        w = np.asarray(w, dtype=float)
        expected_shape = (post.e_count, post.d, post.d)
        if w.shape != expected_shape:
            raise ValueError(
                f"Expected W shape {expected_shape}, got {w.shape}."
            )

        edge_inverses = np.empty_like(w)
        edge_logdets = np.empty(post.e_count)
        roots = np.empty_like(w)

        for edge_index in range(post.e_count):
            edge = base.sym(w[edge_index])
            chol = np.linalg.cholesky(edge)
            edge_logdets[edge_index] = float(
                2.0 * np.sum(np.log(np.diag(chol)))
            )
            y = np.linalg.solve(chol, self.eye_d)
            edge_inverses[edge_index] = base.sym(
                np.linalg.solve(chol.T, y)
            )
            roots[edge_index], _ = base.spd_sqrt_and_inv_sqrt(edge)

        q = post.assemble_q(w)
        chol_q = np.linalg.cholesky(q)
        logdet_q = float(2.0 * np.sum(np.log(np.diag(chol_q))))
        trace_train = float(np.sum(post.s_train * q))
        log_likelihood = 0.5 * post.n_train * (logdet_q - trace_train)

        prior_scores = np.empty_like(w)
        log_prior = 0.0
        determinant_coefficient = 0.5 * (post.nu - post.d - 1.0)
        for edge_index in range(post.e_count):
            prior_scores[edge_index] = base.sym(
                determinant_coefficient * edge_inverses[edge_index]
                - 0.5 * post.psi_inv
            )
            log_prior += float(
                determinant_coefficient * edge_logdets[edge_index]
                - 0.5 * np.trace(post.psi_inv @ w[edge_index])
            )

        drift = np.zeros_like(w)

        if self.mode == "zero":
            pass

        elif self.mode == "prior":
            for edge_index in range(post.e_count):
                root = roots[edge_index]
                drift[edge_index] = base.sym(
                    root @ prior_scores[edge_index] @ root
                    + post.volume_c * self.eye_d
                )

        elif self.mode == "local":
            for edge_index, b in enumerate(post.edge_ops):
                projected_precision = base.sym(b.T @ q @ b)
                projected_inverse = base.spd_inverse(projected_precision)
                gram = self.edge_grams[edge_index]
                local_covariance = base.sym(
                    gram @ projected_inverse @ gram
                )
                score_like_local = 0.5 * post.n_train * base.sym(
                    local_covariance - post.bt_s_b[edge_index]
                )
                root = roots[edge_index]
                drift[edge_index] = base.sym(
                    root
                    @ (prior_scores[edge_index] + score_like_local)
                    @ root
                    + post.volume_c * self.eye_d
                )

        elif self.mode == "full":
            # One shared multiple-right-hand-side solve supplies all exact
            # matrices B_e^T Q^{-1} B_e.
            y = np.linalg.solve(chol_q, post.bcat)
            z = np.linalg.solve(chol_q.T, y)
            for edge_index, b in enumerate(post.edge_ops):
                z_e = z[
                    :,
                    edge_index * post.d : (edge_index + 1) * post.d,
                ]
                exact_covariance = base.sym(b.T @ z_e)
                score_like_full = 0.5 * post.n_train * base.sym(
                    exact_covariance - post.bt_s_b[edge_index]
                )
                root = roots[edge_index]
                drift[edge_index] = base.sym(
                    root
                    @ (prior_scores[edge_index] + score_like_full)
                    @ root
                    + post.volume_c * self.eye_d
                )

        log_target_dx = float(log_likelihood + log_prior)
        phi = float(
            -log_target_dx - post.volume_c * np.sum(edge_logdets)
        )
        heldout_nll = 0.5 * (
            post.p * base.LOG_2PI
            - logdet_q
            + float(np.sum(post.s_test * q))
        )

        return DriftEvaluation(
            log_target_dx=log_target_dx,
            phi=phi,
            q_matrix=q,
            logdet_q=logdet_q,
            heldout_nll=float(heldout_nll),
            roots=roots,
            drift_tangent=drift,
        )


def propose_drift_step(
    post: base.GraphPosterior,
    evaluator: DriftEvaluator,
    w: Array,
    evaluation: DriftEvaluation,
    h: float,
    rng: np.random.Generator,
) -> Tuple[Array, DriftEvaluation, float, bool, bool]:
    """Perform one exact exponential-map MH proposal for one drift mode."""
    w_prop = np.empty_like(w)
    forward_tangents: List[Array] = []
    forward_means: List[Array] = []

    try:
        for edge_index in range(post.e_count):
            mean = h * evaluation.drift_tangent[edge_index]
            noise = base.sample_standard_symmetric(post.d, rng)
            tangent = base.sym(
                mean + math.sqrt(2.0 * h) * noise
            )
            root = evaluation.roots[edge_index]
            w_prop[edge_index] = base.sym(
                root @ base.sym_expm(tangent) @ root
            )
            # The exponential map is SPD in exact arithmetic.  This check
            # catches only finite-precision failures.
            np.linalg.cholesky(w_prop[edge_index])
            forward_tangents.append(tangent)
            forward_means.append(mean)
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return w, evaluation, 0.0, False, True

    try:
        eval_prop = evaluator.evaluate(w_prop)
        logq_forward = 0.0
        logq_reverse = 0.0

        for edge_index in range(post.e_count):
            forward_tangent = forward_tangents[edge_index]
            forward_mean = forward_means[edge_index]
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
            reverse_mean = h * eval_prop.drift_tangent[edge_index]
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


def drift_boundary_chain(
    post: base.GraphPosterior,
    mode: str,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
) -> boundary.BoundaryChainResult:
    """Run a fixed-h production or tuning chain for one drift mode."""
    if warmup < 0 or retained < 1 or thin < 1:
        raise ValueError("Require warmup>=0, retained>=1, and thin>=1.")
    if h <= 0.0:
        raise ValueError("h must be positive.")

    rng = base.set_seed(seed)
    evaluator = DriftEvaluator(post, mode)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = evaluator.evaluate(w)

    samples: List[Array] = []
    traces: Dict[str, List[float]] = {
        "phi": [],
        "logdet_q": [],
        "heldout_nll": [],
        "rel_w_draw": [],
    }

    accepted_warm = accepted_keep = 0
    warmup_proposals = retained_proposals = 0
    warmup_failures = final_failures = 0
    # Exact exponential-map proposals have no genuine invalid-SPD events.
    warmup_invalid = final_invalid = 0

    total_iterations = warmup + retained * thin
    start = time.perf_counter()

    for iteration in range(total_iterations):
        in_warmup = iteration < warmup
        if in_warmup:
            warmup_proposals += 1
        else:
            retained_proposals += 1

        (
            w,
            evaluation,
            _alpha_prob,
            accepted,
            numerical_failure,
        ) = propose_drift_step(
            post=post,
            evaluator=evaluator,
            w=w,
            evaluation=evaluation,
            h=h,
            rng=rng,
        )

        if numerical_failure:
            if in_warmup:
                warmup_failures += 1
            else:
                final_failures += 1

        if in_warmup:
            accepted_warm += int(accepted)
        else:
            accepted_keep += int(accepted)
            if (iteration - warmup) % thin == 0:
                base.append_retained(
                    samples,
                    traces,
                    post,
                    w,
                    evaluation,
                )

    return boundary._finish_chain(
        samples=samples,
        traces=traces,
        accepted_keep=accepted_keep,
        accepted_warm=accepted_warm,
        retained_proposals=retained_proposals,
        warmup_proposals=warmup_proposals,
        runtime_sec=time.perf_counter() - start,
        h=h,
        preconditioner_condition=1.0,
        final_invalid_spd=final_invalid,
        warmup_invalid_spd=warmup_invalid,
        final_numerical_failures=final_failures,
        warmup_numerical_failures=warmup_failures,
    )


def adaptive_drift_burnin(
    post: base.GraphPosterior,
    mode: str,
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
) -> DriftBurninResult:
    """Discarded adaptive pre-burn-in followed by fixed-h settling."""
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
    evaluator = DriftEvaluator(post, mode)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = evaluator.evaluate(w)
    log_h = float(
        np.clip(math.log(h0), math.log(h_min), math.log(h_max))
    )

    accepted = 0
    failures = 0
    start = time.perf_counter()

    for iteration in range(adaptive_steps):
        h_now = float(math.exp(log_h))
        (
            w,
            evaluation,
            alpha_prob,
            accepted_now,
            numerical_failure,
        ) = propose_drift_step(
            post=post,
            evaluator=evaluator,
            w=w,
            evaluation=evaluation,
            h=h_now,
            rng=rng,
        )
        accepted += int(accepted_now)
        failures += int(numerical_failure)

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

    if settle_steps > 0:
        settle = drift_boundary_chain(
            post=post,
            mode=mode,
            w_init=w,
            warmup=0,
            retained=settle_steps,
            thin=1,
            h=h_final,
            seed=seed + 5_000_000,
        )
        if settle.samples.shape[0] == 0:
            raise RuntimeError("Settling produced no samples.")
        w = np.asarray(settle.samples[-1]).copy()
        accepted += int(round(settle.acceptance * settle.final_proposals))
        failures += settle.final_numerical_failures

    total_steps = adaptive_steps + settle_steps
    return DriftBurninResult(
        terminal_w=w,
        h_final=h_final,
        acceptance=accepted / max(total_steps, 1),
        invalid_spd_rate=0.0,
        numerical_failure_rate=failures / max(total_steps, 1),
        runtime_sec=float(time.perf_counter() - start),
        adaptive_steps=adaptive_steps,
        settle_steps=settle_steps,
    )


def product_airm_esjd_per_iteration(samples: Array) -> float:
    if samples.shape[0] < 2:
        return 0.0
    jumps: List[float] = []
    for draw_index in range(samples.shape[0] - 1):
        squared_distance = 0.0
        for edge_index in range(samples.shape[1]):
            tangent = base.reverse_airm_tangent(
                samples[draw_index, edge_index],
                samples[draw_index + 1, edge_index],
            )
            squared_distance += float(np.sum(tangent * tangent))
        jumps.append(squared_distance)
    return float(np.mean(jumps))


def tune_h_from_typical_states(
    post: base.GraphPosterior,
    mode: str,
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
) -> Tuple[float, float, float, List[Dict[str, object]]]:
    """Tune h by product-AIRM ESJD from cloned typical-set anchors."""
    if not anchor_states:
        raise ValueError("At least one anchor state is required.")
    if tune_warmup < 0 or tune_steps < 10:
        raise ValueError("Require tune_warmup>=0 and tune_steps>=10.")
    if acceptance_tolerance < 0.0:
        raise ValueError("acceptance_tolerance must be nonnegative.")

    tested: Dict[str, Dict[str, object]] = {}

    def evaluate_candidate(h: float) -> Dict[str, object]:
        h = float(max(h, 1e-14))
        key = f"{h:.17e}"
        if key in tested:
            return tested[key]

        acceptances: List[float] = []
        esjd_values: List[float] = []
        runtimes: List[float] = []
        failure_rates: List[float] = []

        for chain_index, anchor in enumerate(anchor_states):
            result = drift_boundary_chain(
                post=post,
                mode=mode,
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
            failure_rates.append(
                result.final_numerical_failures
                / max(result.final_proposals, 1)
            )

        row: Dict[str, object] = {
            "mode": mode,
            "method": METHOD_NAME[mode],
            "stage": "typical_set_fixed_h_tuning",
            "h": h,
            "target_acceptance": target_accept,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1)
                if len(acceptances) > 1
                else 0.0
            ),
            "airm_esjd_per_iteration": float(np.mean(esjd_values)),
            "pilot_invalid_spd_rate": 0.0,
            "pilot_numerical_failure_rate": float(
                np.mean(failure_rates)
            ),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[key] = row
        return row

    for factor in sorted(set(float(x) for x in factors if x > 0.0)):
        evaluate_candidate(h_center * factor)

    lower = max(0.0, target_accept - acceptance_tolerance)
    upper = min(1.0, target_accept + acceptance_tolerance)

    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: float(row["h"]))
        if any(float(row["pilot_acceptance"]) <= upper for row in rows):
            break
        evaluate_candidate(float(rows[-1]["h"]) * 2.0)

    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: float(row["h"]))
        if any(float(row["pilot_acceptance"]) >= lower for row in rows):
            break
        evaluate_candidate(float(rows[0]["h"]) / 2.0)

    for _ in range(bisection_steps):
        rows = sorted(tested.values(), key=lambda row: float(row["h"]))
        high_accept = [
            row
            for row in rows
            if float(row["pilot_acceptance"]) >= target_accept
        ]
        low_accept = [
            row
            for row in rows
            if float(row["pilot_acceptance"]) <= target_accept
        ]
        if not high_accept or not low_accept:
            break
        left = max(high_accept, key=lambda row: float(row["h"]))
        right = min(low_accept, key=lambda row: float(row["h"]))
        if float(left["h"]) >= float(right["h"]):
            break
        h_mid = math.sqrt(float(left["h"]) * float(right["h"]))
        if f"{h_mid:.17e}" in tested:
            break
        evaluate_candidate(h_mid)

    rows = sorted(tested.values(), key=lambda row: float(row["h"]))
    admissible = [
        row
        for row in rows
        if lower <= float(row["pilot_acceptance"]) <= upper
    ]

    if admissible:
        best = max(
            admissible,
            key=lambda row: (
                float(row["airm_esjd_per_iteration"]),
                -abs(float(row["pilot_acceptance"]) - target_accept),
                -float(row["h"]),
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                abs(float(row["pilot_acceptance"]) - target_accept),
                -float(row["airm_esjd_per_iteration"]),
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

    total_runtime = float(sum(float(row["pilot_runtime_sec"]) for row in rows))
    return (
        float(best["h"]),
        float(best["pilot_acceptance"]),
        total_runtime,
        rows,
    )


def drift_geometry_diagnostics(
    post: base.GraphPosterior,
    w: Array,
) -> Dict[str, float]:
    """Compare prior/local/full tangent drifts at one representative state."""
    prior = DriftEvaluator(post, "prior").evaluate(w).drift_tangent
    local = DriftEvaluator(post, "local").evaluate(w).drift_tangent
    full = DriftEvaluator(post, "full").evaluate(w).drift_tangent

    prior_vec = prior.ravel()
    local_vec = local.ravel()
    full_vec = full.ravel()

    prior_norm = float(np.linalg.norm(prior_vec))
    local_norm = float(np.linalg.norm(local_vec))
    full_norm = float(np.linalg.norm(full_vec))
    denom = max(full_norm, 1e-15)
    cosine_denom = max(local_norm * full_norm, 1e-15)

    return {
        "prior_drift_norm": prior_norm,
        "local_drift_norm": local_norm,
        "full_drift_norm": full_norm,
        "prior_to_full_norm_ratio": prior_norm / denom,
        "local_to_full_norm_ratio": local_norm / denom,
        "local_relative_error": float(
            np.linalg.norm(local_vec - full_vec) / denom
        ),
        "prior_relative_error": float(
            np.linalg.norm(prior_vec - full_vec) / denom
        ),
        "local_full_cosine": float(
            np.dot(local_vec, full_vec) / cosine_denom
        ),
        "global_increment_norm_ratio": float(
            np.linalg.norm(full_vec - local_vec) / denom
        ),
    }


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
    drift_rows: List[Dict[str, object]],
    truth_rows: List[Dict[str, object]],
) -> None:
    pd.DataFrame(burnin_rows).to_csv(
        output_dir / "drift_ablation_burnin.csv", index=False
    )
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "drift_ablation_tuning.csv", index=False
    )
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "drift_ablation_by_seed.csv", index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "drift_ablation_diagnostics.csv", index=False
    )
    pd.DataFrame(drift_rows).to_csv(
        output_dir / "drift_geometry_diagnostics.csv", index=False
    )
    pd.DataFrame(truth_rows).to_csv(
        output_dir / "boundary_truth.csv", index=False
    )


def aggregate_results(
    seed_df: pd.DataFrame,
    rhat_threshold: float,
) -> pd.DataFrame:
    summary = boundary.aggregate_boundary_results(seed_df)

    extras: List[Dict[str, object]] = []
    for (epsilon, method), group in seed_df.groupby(
        ["epsilon", "method"], sort=True
    ):
        extras.append(
            {
                "epsilon": float(epsilon),
                "method": method,
                "mode": str(group["mode"].iloc[0]),
                "target_acceptance": float(
                    group["target_acceptance"].iloc[0]
                ),
                "selected_h_mean": float(group["selected_h"].mean()),
                "selected_h_sd": float(group["selected_h"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "airm_esjd_per_iteration_mean": float(
                    group["airm_esjd_per_iteration"].mean()
                ),
                "airm_esjd_per_iteration_sd": float(
                    group["airm_esjd_per_iteration"].std(ddof=1)
                )
                if len(group) > 1
                else 0.0,
                "n_converged_seed_runs": int(
                    np.sum(group["max_rhat"] <= rhat_threshold)
                ),
                "n_seed_runs": int(len(group)),
                "converged_seed_fraction": float(
                    np.mean(group["max_rhat"] <= rhat_threshold)
                ),
            }
        )

    extras_df = pd.DataFrame(extras)
    return summary.merge(extras_df, on=["epsilon", "method"], how="left")


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    table = summary.copy()
    order = {METHOD_NAME[m]: i for i, m in enumerate(MODE_ORDER)}
    table["_order"] = table["method"].map(order)
    table = table.sort_values(["epsilon", "_order"], ascending=[False, True])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Drift ablation with product affine-invariant geometry, "
            r"the exact exponential map, and the same full posterior MH "
            r"correction.  ESJD is the product-AIRM squared jump distance per "
            r"retained transition.  Conv. is the number of independent data "
            r"seeds satisfying the declared maximum split-$\widehat R$ threshold.}"
        ),
        r"\label{tab:drift_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        (
            r"$\varepsilon$ & Drift & $h$ & Acceptance & AIRM ESJD/iter. & "
            r"Min. ESS/iter. & Min. ESS/sec & Max. $\widehat R$ & Conv. & "
            r"Numerical fail. \\"
        ),
        r"\midrule",
    ]

    previous_epsilon = None
    for _, row in table.iterrows():
        epsilon = float(row["epsilon"])
        if previous_epsilon is not None and epsilon != previous_epsilon:
            lines.append(r"\addlinespace")
        previous_epsilon = epsilon
        label = METHOD_LABEL.get(str(row["method"]), str(row["method"]))
        lines.append(
            " & ".join(
                [
                    f"${epsilon:.0e}$",
                    label,
                    f"${row['selected_h_mean']:.2e}$",
                    f"${row['acceptance_mean']:.3f}\\pm{row['acceptance_sd']:.3f}$",
                    f"${row['airm_esjd_per_iteration_mean']:.2e}$",
                    f"${row['min_ess_per_iteration_mean']:.2e}$",
                    f"${row['min_ess_per_sec_sampling_mean']:.3f}$",
                    f"${row['max_rhat_mean']:.3f}$",
                    (
                        f"${int(row['n_converged_seed_runs'])}/"
                        f"{int(row['n_seed_runs'])}$"
                    ),
                    f"${row['final_numerical_failure_rate_mean']:.2e}$",
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_method_barplots(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    order = [METHOD_NAME[m] for m in MODE_ORDER]
    present = [m for m in order if m in set(summary["method"])]
    labels = [METHOD_LABEL[m] for m in present]
    table = summary.set_index("method").loc[present]
    x = np.arange(len(present))

    specifications = [
        (
            "min_ess_per_iteration_mean",
            "min_ess_per_iteration_sd",
            "Minimum ESS per retained iteration",
            "fig_drift_min_ess_per_iteration.pdf",
            True,
            None,
        ),
        (
            "max_rhat_mean",
            "max_rhat_sd",
            r"Mean maximum split-$\widehat R$",
            "fig_drift_max_rhat.pdf",
            False,
            1.05,
        ),
        (
            "airm_esjd_per_iteration_mean",
            "airm_esjd_per_iteration_sd",
            "Product-AIRM ESJD per iteration",
            "fig_drift_airm_esjd.pdf",
            True,
            None,
        ),
        (
            "converged_seed_fraction",
            None,
            "Fraction of converged data-seed runs",
            "fig_drift_convergence_fraction.pdf",
            False,
            None,
        ),
        (
            "min_ess_per_sec_sampling_mean",
            "min_ess_per_sec_sampling_sd",
            "Minimum ESS per second (production)",
            "fig_drift_ess_per_second.pdf",
            False,
            None,
        ),
    ]

    for value_col, error_col, ylabel, filename, log_y, threshold in specifications:
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        errors = None if error_col is None else table[error_col].to_numpy()
        ax.bar(x, table[value_col].to_numpy(), yerr=errors, capsize=4)
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        if threshold is not None:
            ax.axhline(threshold, linestyle="--", linewidth=1.2)
        ax.set_title(r"Drift ablation at $\varepsilon=10^{-2}$")
        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Product-AIRM ConeMALA drift ablation."
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
        default=(1e-2,),
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=MODE_ORDER,
        help="Comma-separated subset of zero,prior,local,full.",
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

    parser.add_argument(
        "--h0",
        type=float,
        default=None,
        help="Common initial h; defaults to cone_h0 from the source config.",
    )
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--adapt-decay", type=float, default=100.0)
    parser.add_argument("--adapt-exponent", type=float, default=0.6)
    parser.add_argument("--h-min", type=float, default=1e-10)
    parser.add_argument("--h-max", type=float, default=1.0)

    parser.add_argument("--tune-warmup", type=int)
    parser.add_argument("--tune-steps", type=int)
    parser.add_argument("--acceptance-tolerance", type=float, default=0.10)
    parser.add_argument("--max-bracket-steps", type=int, default=20)
    parser.add_argument("--bisection-steps", type=int, default=8)
    parser.add_argument(
        "--random-walk-target-accept",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--drift-target-accept",
        type=float,
        default=None,
        help="Defaults to target_accept in the source config.",
    )
    parser.add_argument("--rhat-threshold", type=float, default=1.05)
    parser.add_argument("--boundary-stride", type=int, default=10)
    parser.add_argument("--save-anchor-states", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    source_config = boundary.load_base_config(args.config)
    config = boundary.resolve_profile(source_config, args.profile)
    config = replace(
        config,
        methods=tuple(METHOD_NAME[m] for m in args.modes),
    )
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
    retained = config.retained if args.retained is None else args.retained
    thin = config.thin if args.thin is None else args.thin
    tune_warmup = (
        default_tune_warmup(args.profile)
        if args.tune_warmup is None
        else args.tune_warmup
    )
    tune_steps = (
        config.pilot_steps if args.tune_steps is None else args.tune_steps
    )
    h0 = config.cone_h0 if args.h0 is None else args.h0
    drift_target_accept = (
        config.target_accept
        if args.drift_target_accept is None
        else args.drift_target_accept
    )

    if preburn_steps < 1:
        raise ValueError("preburn_steps must be positive.")
    if settle_steps < 0 or production_warmup < 0:
        raise ValueError("settle_steps and production_warmup must be nonnegative.")
    if retained < 8:
        raise ValueError("retained must be at least 8 for diagnostics.")
    if thin < 1:
        raise ValueError("thin must be positive.")
    if not 0.0 < args.rhat_threshold < 2.0:
        raise ValueError("rhat_threshold must lie in (0,2).")
    for target in (args.random_walk_target_accept, drift_target_accept):
        if not 0.0 < target < 1.0:
            raise ValueError("Acceptance targets must lie in (0,1).")

    prior_info = prior_metadata(
        config=config,
        prior_mode=args.prior_mode,
        boundary_nu_offset=args.boundary_nu_offset,
    )
    run_config = {
        "source_config": str(args.config),
        "profile": args.profile,
        "output_dir": str(args.output_dir),
        "epsilons": list(args.epsilons),
        "modes": list(args.modes),
        "method_names": {m: METHOD_NAME[m] for m in args.modes},
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
        "random_walk_target_accept": args.random_walk_target_accept,
        "drift_target_accept": drift_target_accept,
        "rhat_threshold": args.rhat_threshold,
        "boundary_stride": args.boundary_stride,
        "shared_initial_states_across_modes": True,
        "local_score_approximation": (
            "G_e @ inv(B_e.T @ Q(W) @ B_e) @ G_e, "
            "where G_e=B_e.T@B_e"
        ),
        "source_profile_config": asdict(config),
        "resolved_experiment_config": effective_experiment_config(
            config, prior_info
        ),
        "effective_prior": prior_info,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("ConeMALA drift ablation with fixed product-AIRM geometry")
    print("=" * 88)
    print(json.dumps(run_config, indent=2))

    burnin_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    seed_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    drift_rows: List[Dict[str, object]] = []
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
            truth_stats = boundary.matrix_family_stats(post.w_true)
            truth_rows.append(
                {
                    "epsilon": epsilon,
                    "truth_condition_target": 1.0 / epsilon,
                    "experiment_seed": experiment_seed,
                    "prior_mode": args.prior_mode,
                    "nu": post.nu,
                    "rho": post.rho,
                    "source_nu": prior_info["source_nu"],
                    "effective_nu": prior_info["effective_nu"],
                    "source_prior_scale": prior_info["source_prior_scale"],
                    "effective_prior_scale": prior_info[
                        "effective_prior_scale"
                    ],
                    "determinant_exponent": prior_info[
                        "determinant_exponent"
                    ],
                    **truth_stats,
                }
            )

            w_center = base.initial_weights(post)
            shared_initials = [
                shared_initial_state(
                    w_center=w_center,
                    experiment_seed=experiment_seed,
                    epsilon_index=epsilon_index,
                    chain_index=chain_index,
                    n_chains=config.n_chains,
                )
                for chain_index in range(config.n_chains)
            ]

            for mode_index, mode in enumerate(args.modes):
                method = METHOD_NAME[mode]
                target_accept = (
                    args.random_walk_target_accept
                    if mode == "zero"
                    else drift_target_accept
                )

                print("\n" + "-" * 88)
                print(
                    f"Mode={mode} ({METHOD_LABEL[method]}), "
                    f"target acceptance={target_accept:.3f}"
                )
                print("-" * 88)

                anchors: List[Array] = []
                adapted_h: List[float] = []
                burnin_runtime = 0.0

                print("Adaptive pre-burn-in")
                for chain_index, w_init in enumerate(shared_initials):
                    result = adaptive_drift_burnin(
                        post=post,
                        mode=mode,
                        w_init=np.asarray(w_init).copy(),
                        adaptive_steps=preburn_steps,
                        settle_steps=settle_steps,
                        h0=h0,
                        target_accept=target_accept,
                        adapt_rate=args.adapt_rate,
                        adapt_decay=args.adapt_decay,
                        adapt_exponent=args.adapt_exponent,
                        h_min=args.h_min,
                        h_max=args.h_max,
                        seed=(
                            720_000_000
                            + 10_000_000 * mode_index
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
                            "mode": mode,
                            "method": method,
                            "chain": chain_index + 1,
                            "stage": "adaptive_preburn",
                            "target_acceptance": target_accept,
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
                        f"failures={result.numerical_failure_rate:.3f}, "
                        f"time={result.runtime_sec:.1f}s"
                    )

                h_center = float(np.median(adapted_h))
                (
                    h_selected,
                    pilot_acceptance,
                    tuning_runtime,
                    candidate_rows,
                ) = tune_h_from_typical_states(
                    post=post,
                    mode=mode,
                    anchor_states=anchors,
                    h_center=h_center,
                    target_accept=target_accept,
                    acceptance_tolerance=args.acceptance_tolerance,
                    tune_warmup=tune_warmup,
                    tune_steps=tune_steps,
                    factors=(0.25, 0.5, 1.0, 2.0, 4.0),
                    max_bracket_steps=args.max_bracket_steps,
                    bisection_steps=args.bisection_steps,
                    seed_base=(
                        730_000_000
                        + 10_000_000 * mode_index
                        + 1_000_000 * epsilon_index
                        + 10_000 * experiment_seed
                    ),
                )

                for candidate in candidate_rows:
                    candidate.update(
                        {
                            "epsilon": epsilon,
                            "truth_condition_target": 1.0 / epsilon,
                            "experiment_seed": experiment_seed,
                        }
                    )
                    tuning_rows.append(candidate)

                print(
                    f"Typical-set tuning: center={h_center:.3e}, "
                    f"selected h={h_selected:.3e}, "
                    f"pilot acceptance={pilot_acceptance:.3f}, "
                    f"time={tuning_runtime:.1f}s"
                )

                for chain_index, anchor in enumerate(anchors):
                    drift_row = drift_geometry_diagnostics(post, anchor)
                    drift_row.update(
                        {
                            "epsilon": epsilon,
                            "truth_condition_target": 1.0 / epsilon,
                            "experiment_seed": experiment_seed,
                            "anchor_source_mode": mode,
                            "anchor_source_method": method,
                            "chain": chain_index + 1,
                        }
                    )
                    drift_rows.append(drift_row)

                if args.save_anchor_states:
                    np.savez_compressed(
                        args.output_dir
                        / (
                            f"drift_plan_{mode}_eps{epsilon:.0e}"
                            f"_seed{experiment_seed}.npz"
                        ),
                        anchors=np.stack(anchors, axis=0),
                        adapted_h=np.asarray(adapted_h),
                        h_center=np.asarray(h_center),
                        h_selected=np.asarray(h_selected),
                    )

                print("Fixed-h production chains")
                chain_results: List[boundary.BoundaryChainResult] = []
                for chain_index, anchor in enumerate(anchors):
                    result = drift_boundary_chain(
                        post=post,
                        mode=mode,
                        w_init=np.asarray(anchor).copy(),
                        warmup=production_warmup,
                        retained=retained,
                        thin=thin,
                        h=h_selected,
                        seed=(
                            740_000_000
                            + 10_000_000 * mode_index
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
                        f"failures="
                        f"{result.final_numerical_failures / max(result.final_proposals,1):.3f}, "
                        f"time={result.runtime_sec:.1f}s"
                    )

                row, diagnostics = boundary.summarize_boundary_method_seed(
                    method=method,
                    experiment_seed=experiment_seed,
                    epsilon=epsilon,
                    post=post,
                    chain_results=chain_results,
                    pilot_runtime_sec=burnin_runtime + tuning_runtime,
                    parallel_runtime=config.parallel_runtime,
                    boundary_stride=args.boundary_stride,
                )
                production_esjd = [
                    product_airm_esjd_per_iteration(result.samples)
                    for result in chain_results
                ]
                row.update(
                    {
                        "mode": mode,
                        "method_label": METHOD_LABEL[method],
                        "target_acceptance": target_accept,
                        "prior_mode": args.prior_mode,
                        "nu": post.nu,
                        "rho": post.rho,
                        "source_nu": prior_info["source_nu"],
                        "effective_nu": prior_info["effective_nu"],
                        "source_prior_scale": prior_info[
                            "source_prior_scale"
                        ],
                        "effective_prior_scale": prior_info[
                            "effective_prior_scale"
                        ],
                        "determinant_exponent": prior_info[
                            "determinant_exponent"
                        ],
                        "preburn_runtime_sec": burnin_runtime,
                        "typical_tuning_runtime_sec": tuning_runtime,
                        "preburn_h_median": h_center,
                        "selected_h": h_selected,
                        "airm_esjd_per_iteration": float(
                            np.mean(production_esjd)
                        ),
                        "airm_esjd_per_iteration_sd_across_chains": float(
                            np.std(production_esjd, ddof=1)
                            if len(production_esjd) > 1
                            else 0.0
                        ),
                    }
                )
                seed_rows.append(row)
                for diagnostic in diagnostics:
                    diagnostic["mode"] = mode
                    diagnostic["method_label"] = METHOD_LABEL[method]
                    diagnostic_rows.append(diagnostic)

                print(
                    "Summary: "
                    f"acc={row['acceptance']:.3f}, "
                    f"AIRM ESJD/iter={row['airm_esjd_per_iteration']:.3e}, "
                    f"min ESS/iter={row['min_ess_per_iteration']:.4f}, "
                    f"min ESS/s={row['min_ess_per_sec_sampling']:.3f}, "
                    f"Rhat={row['max_rhat']:.3f}"
                )

                checkpoint(
                    output_dir=args.output_dir,
                    burnin_rows=burnin_rows,
                    tuning_rows=tuning_rows,
                    seed_rows=seed_rows,
                    diagnostic_rows=diagnostic_rows,
                    drift_rows=drift_rows,
                    truth_rows=truth_rows,
                )

    seed_df = pd.DataFrame(seed_rows)
    summary_df = aggregate_results(seed_df, args.rhat_threshold)
    summary_df.to_csv(
        args.output_dir / "drift_ablation_summary.csv", index=False
    )
    write_latex_table(
        summary_df,
        args.output_dir / "table_drift_ablation.tex",
    )

    drift_df = pd.DataFrame(drift_rows)
    if not drift_df.empty:
        drift_summary = (
            drift_df.groupby(
                ["epsilon", "anchor_source_mode", "anchor_source_method"],
                sort=True,
            )
            .agg(
                prior_to_full_norm_ratio_mean=(
                    "prior_to_full_norm_ratio",
                    "mean",
                ),
                prior_to_full_norm_ratio_sd=(
                    "prior_to_full_norm_ratio",
                    "std",
                ),
                local_relative_error_mean=(
                    "local_relative_error",
                    "mean",
                ),
                local_relative_error_sd=(
                    "local_relative_error",
                    "std",
                ),
                local_full_cosine_mean=("local_full_cosine", "mean"),
                local_full_cosine_sd=("local_full_cosine", "std"),
            )
            .reset_index()
        )
        drift_summary.to_csv(
            args.output_dir / "drift_geometry_summary.csv", index=False
        )

    if args.make_plots:
        make_method_barplots(summary_df, args.output_dir)

    print("\nAcross-seed drift-ablation summary")
    print("=" * 88)
    display_columns = [
        "epsilon",
        "method",
        "selected_h_mean",
        "acceptance_mean",
        "airm_esjd_per_iteration_mean",
        "min_ess_per_iteration_mean",
        "min_ess_per_sec_sampling_mean",
        "max_rhat_mean",
        "n_converged_seed_runs",
        "n_seed_runs",
        "final_numerical_failure_rate_mean",
    ]
    print(summary_df[display_columns].to_string(index=False))

    print("\nSaved outputs:")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
