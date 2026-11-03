#!/usr/bin/env python3
r"""
Boundary-stress comparison for PSD-weighted graph samplers.

This experiment compares

    ConeMALA,
    Fisher-adaptive MALA,
    Euclidean MALA,
    retraction RMALA,

on a paired family of synthetic graph posteriors whose true edge matrices
approach the SPD boundary.

For every edge e, the truth is

    W_e^* = U_e diag(lambda_e) U_e^T,

with a fixed-mean spectrum

    lambda_e = c_e * d/(d-1+epsilon) * (epsilon, 1, ..., 1).

Consequently,

    kappa(W_e^*) = 1/epsilon,

while the average eigenvalue remains c_e.  The rotations U_e, edge scales c_e,
and underlying Gaussian random numbers are held fixed across epsilon within
each independent data seed.  This gives a paired boundary-stress family rather
than a collection of unrelated datasets.

The script uses runtime-independent pilot criteria:

* fixed samplers: product-AIRM ESJD per retained transition;
* Fisher baseline: P^{-1}-standardized ESJD per retained transition.

Wall-clock runtime is never used to select a step size.  It is recorded only
for final efficiency reporting.

Required companion files
------------------------
Place this file in the same directory as

    conemala_fisher_adaptive_comparison.py
    fisher_adaptive_retune_deterministic.py

The latter supplies the deterministic two-stage Fisher preconditioner
adaptation and scalar step-size tuning.

Recommended screening run
-------------------------
python3 boundary_stress_comparison.py \
    --config results_fisher_paper/config.json \
    --profile pilot \
    --epsilons 1e-1,1e-2,1e-3,1e-4 \
    --prior-mode original \
    --output-dir results_boundary_pilot

A stronger posterior-boundary sensitivity run keeps the prior mean fixed but
reduces its boundary repulsion:

python3 boundary_stress_comparison.py \
    --config results_fisher_paper/config.json \
    --profile pilot \
    --epsilons 1e-1,1e-2,1e-3,1e-4 \
    --prior-mode boundary-friendly \
    --boundary-nu-offset 0.5 \
    --output-dir results_boundary_pilot_boundary_prior

Full paper run
--------------
python3 boundary_stress_comparison.py \
    --config results_fisher_paper/config.json \
    --profile paper \
    --epsilons 1e-1,1e-2,1e-3,1e-4 \
    --prior-mode original \
    --output-dir results_boundary_paper

Important interpretation
------------------------
A near-singular truth does not automatically imply a near-boundary posterior.
The original Wishart prior may repel the boundary.  Therefore the script
supports both:

1. prior-mode=original:
   changes only W^* and preserves the original posterior specification;

2. prior-mode=boundary-friendly:
   keeps the prior mean fixed but uses nu=d+offset, which permits appreciably
   more mass near the SPD boundary.

Both experiments are scientifically useful and should be reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import conemala_fisher_adaptive_comparison as base
import fisher_adaptive_retune_deterministic as fisher


Array = np.ndarray
METHODS = (
    "ConeMALA",
    "FisherAdaptiveMALA",
    "EuclideanMALA",
    "RetractionRMALA",
)
FIXED_METHODS = (
    "ConeMALA",
    "EuclideanMALA",
    "RetractionRMALA",
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BoundaryRunConfig:
    source_config: str
    profile: str
    output_dir: str
    epsilons: Tuple[float, ...]
    methods: Tuple[str, ...]
    prior_mode: str
    boundary_nu_offset: float
    pilot_chains: int
    pilot_steps: int
    acceptance_tolerance: float
    max_bracket_steps: int
    bisection_steps: int
    fisher_adapt_chains: int
    fisher_adapt_steps: int
    fisher_tune_chains: int
    fisher_tune_steps: int
    fisher_condition_cap: float
    boundary_stride: int
    make_plots: bool


@dataclass
class BoundaryChainResult:
    """
    Chain result with the fields expected by the base diagnostic functions,
    plus boundary-specific numerical diagnostics.
    """

    samples: Array
    traces: Dict[str, Array]
    acceptance: float
    warmup_acceptance: float
    runtime_sec: float
    h_final: float
    preconditioner_condition: float

    final_proposals: int
    warmup_proposals: int
    final_invalid_spd: int
    warmup_invalid_spd: int
    final_numerical_failures: int
    warmup_numerical_failures: int


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def parse_str_tuple(text: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one method is required.")
    unknown = sorted(set(values) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown methods: {unknown}")
    return values


def load_base_config(path: Path) -> base.ExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["experiment_seeds"] = tuple(int(x) for x in data["experiment_seeds"])
    data["methods"] = tuple(str(x) for x in data["methods"])
    return base.ExperimentConfig(**data)


def resolve_profile(
    config: base.ExperimentConfig,
    profile: str,
) -> base.ExperimentConfig:
    """
    Apply only computational-size overrides.  Statistical hyperparameters
    rho, prior mean, and true-weight scale remain inherited from the source
    configuration.
    """
    if profile == "paper":
        return replace(config, profile="boundary-paper")

    if profile == "pilot":
        return replace(
            config,
            profile="boundary-pilot",
            m=10,
            d=3,
            n_train=200,
            n_test=100,
            n_chains=3,
            warmup=500,
            retained=1000,
            thin=1,
            pilot_steps=150,
            experiment_seeds=(11, 22, 33),
            cone_h0=5e-3,
            euclidean_h0=1e-5,
            retraction_h0=5e-5,
            fisher_h0=1e-5,
            fisher_init_steps=min(150, 500),
        )

    if profile == "smoke":
        return replace(
            config,
            profile="boundary-smoke",
            m=4,
            d=2,
            n_train=40,
            n_test=20,
            n_chains=2,
            warmup=10,
            retained=20,
            thin=1,
            pilot_steps=10,
            experiment_seeds=(11,),
            cone_h0=2e-3,
            euclidean_h0=1e-5,
            retraction_h0=5e-5,
            fisher_h0=1e-5,
            fisher_init_steps=3,
        )

    raise ValueError(f"Unknown profile: {profile}")


# =============================================================================
# Paired boundary-stress problem construction
# =============================================================================


def random_orthogonal(d: int, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def generate_boundary_true_weights(
    e_count: int,
    d: int,
    scale: float,
    epsilon: float,
    rng: np.random.Generator,
) -> Array:
    """
    Generate a fixed-mean boundary family.

    The spectrum on edge e is

        c_e * d/(d-1+epsilon) * (epsilon, 1, ..., 1).

    Hence the mean eigenvalue is c_e and the condition number is 1/epsilon.
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must lie in (0,1].")

    base_spectrum = np.ones(d, dtype=float)
    base_spectrum[0] = epsilon
    base_spectrum *= d / float(np.sum(base_spectrum))

    weights = np.empty((e_count, d, d), dtype=float)
    for edge_index in range(e_count):
        rotation = random_orthogonal(d, rng)
        edge_scale = scale * (0.75 + 0.5 * rng.random())
        spectrum = edge_scale * base_spectrum
        weights[edge_index] = base.sym(
            (rotation * spectrum) @ rotation.T
        )
    return weights


def build_edge_operators(
    edges: Sequence[Tuple[int, int]],
    m: int,
    d: int,
) -> List[Array]:
    p = m * d
    eye = np.eye(d)
    operators: List[Array] = []
    for i, j in edges:
        b = np.zeros((p, d), dtype=float)
        b[i * d : (i + 1) * d] = eye
        b[j * d : (j + 1) * d] = -eye
        operators.append(b)
    return operators


def resolve_prior(
    config: base.ExperimentConfig,
    prior_mode: str,
    boundary_nu_offset: float,
) -> Tuple[float, Array]:
    """
    Return (nu, Psi), preserving the original prior mean nu*Psi.

    In boundary-friendly mode, nu=d+offset.  Since the Wishart density has
    determinant exponent (nu-d-1)/2, smaller positive offsets reduce boundary
    repulsion while preserving propriety and the original prior mean.
    """
    original_mean_scale = config.nu * config.prior_scale

    if prior_mode == "original":
        nu = float(config.nu)
        psi_scale = float(config.prior_scale)
    elif prior_mode == "boundary-friendly":
        nu = float(config.d + boundary_nu_offset)
        if nu <= config.d - 1:
            raise ValueError(
                "boundary-friendly nu must satisfy nu>d-1."
            )
        psi_scale = original_mean_scale / nu
    else:
        raise ValueError(f"Unknown prior mode: {prior_mode}")

    return nu, psi_scale * np.eye(config.d)


def make_boundary_problem(
    config: base.ExperimentConfig,
    experiment_seed: int,
    epsilon: float,
    prior_mode: str,
    boundary_nu_offset: float,
) -> base.GraphPosterior:
    """
    Build paired datasets across epsilon.

    The rotations, edge scales, and standard Gaussian innovations are fixed
    within each experiment seed.  Only epsilon changes.
    """
    edges = base.build_edges(config.m, config.graph)
    operators = build_edge_operators(edges, config.m, config.d)

    truth_rng = base.set_seed(10_000_000 + experiment_seed)
    w_true = generate_boundary_true_weights(
        e_count=len(edges),
        d=config.d,
        scale=config.true_weight_scale,
        epsilon=epsilon,
        rng=truth_rng,
    )
    q_true = base.assemble_q_from_ops(w_true, operators, config.rho)

    # Common random numbers across epsilon for paired stress comparisons.
    data_rng = base.set_seed(20_000_000 + experiment_seed)
    y_train = base.sample_from_precision(q_true, config.n_train, data_rng)
    y_test = base.sample_from_precision(q_true, config.n_test, data_rng)
    s_train = base.sym(y_train.T @ y_train / config.n_train)
    s_test = base.sym(y_test.T @ y_test / config.n_test)

    nu, psi = resolve_prior(config, prior_mode, boundary_nu_offset)
    return base.GraphPosterior(
        edges=edges,
        s_train=s_train,
        s_test=s_test,
        n_train=config.n_train,
        m=config.m,
        d=config.d,
        rho=config.rho,
        nu=nu,
        psi=psi,
        w_true=w_true,
    )


def matrix_family_stats(weights: Array) -> Dict[str, float]:
    minimum_eigenvalues: List[float] = []
    condition_numbers: List[float] = []
    for matrix in weights:
        eig = np.linalg.eigvalsh(base.sym(matrix))
        minimum_eigenvalues.append(float(np.min(eig)))
        condition_numbers.append(float(np.max(eig) / np.min(eig)))
    return {
        "min_eigenvalue_min": float(np.min(minimum_eigenvalues)),
        "min_eigenvalue_mean": float(np.mean(minimum_eigenvalues)),
        "condition_number_max": float(np.max(condition_numbers)),
        "condition_number_mean": float(np.mean(condition_numbers)),
    }


# =============================================================================
# Boundary-aware final-chain kernels
# =============================================================================


def _empty_traces() -> Dict[str, List[float]]:
    return {
        "phi": [],
        "logdet_q": [],
        "heldout_nll": [],
        "rel_w_draw": [],
    }


def _finish_chain(
    samples: List[Array],
    traces: Dict[str, List[float]],
    accepted_keep: int,
    accepted_warm: int,
    retained_proposals: int,
    warmup_proposals: int,
    runtime_sec: float,
    h: float,
    preconditioner_condition: float,
    final_invalid_spd: int,
    warmup_invalid_spd: int,
    final_numerical_failures: int,
    warmup_numerical_failures: int,
) -> BoundaryChainResult:
    return BoundaryChainResult(
        samples=np.asarray(samples),
        traces={key: np.asarray(value) for key, value in traces.items()},
        acceptance=accepted_keep / max(retained_proposals, 1),
        warmup_acceptance=accepted_warm / max(warmup_proposals, 1),
        runtime_sec=float(runtime_sec),
        h_final=float(h),
        preconditioner_condition=float(preconditioner_condition),
        final_proposals=int(retained_proposals),
        warmup_proposals=int(warmup_proposals),
        final_invalid_spd=int(final_invalid_spd),
        warmup_invalid_spd=int(warmup_invalid_spd),
        final_numerical_failures=int(final_numerical_failures),
        warmup_numerical_failures=int(warmup_numerical_failures),
    )


def cone_mala_boundary_chain(
    post: base.GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
) -> BoundaryChainResult:
    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)

    samples: List[Array] = []
    traces = _empty_traces()
    accepted_warm = accepted_keep = 0
    warmup_proposals = retained_proposals = 0
    warmup_invalid = final_invalid = 0
    warmup_failures = final_failures = 0

    n_total = warmup + retained * thin
    start = time.perf_counter()

    for iteration in range(n_total):
        in_warmup = iteration < warmup
        if in_warmup:
            warmup_proposals += 1
        else:
            retained_proposals += 1

        w_prop = np.empty_like(w)
        tangent_forward: List[Array] = []
        tangent_mean_forward: List[Array] = []
        proposal_spd_invalid = False
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
                try:
                    np.linalg.cholesky(w_prop[edge_index])
                except np.linalg.LinAlgError:
                    proposal_spd_invalid = True
                    break
                tangent_forward.append(tangent)
                tangent_mean_forward.append(mean_tangent)
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            numerical_failure = True

        accepted = False
        if not proposal_spd_invalid and not numerical_failure:
            try:
                eval_prop = post.evaluate(w_prop)
                logq_forward = 0.0
                logq_reverse = 0.0

                for edge_index in range(post.e_count):
                    sf = tangent_forward[edge_index]
                    mf = tangent_mean_forward[edge_index]
                    logq_forward += (
                        base.log_gaussian_isotropic(
                            base.svec(sf - mf), variance=2.0 * h
                        )
                        - base.log_spd_exp_jacobian(sf)
                    )

                    reverse_tangent = base.reverse_airm_tangent(
                        w_prop[edge_index], w[edge_index]
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
                    -eval_prop.phi + logq_reverse
                    - (-evaluation.phi + logq_forward)
                )
                if math.log(rng.uniform()) < min(0.0, log_alpha):
                    w = w_prop
                    evaluation = eval_prop
                    accepted = True
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                numerical_failure = True

        if proposal_spd_invalid:
            if in_warmup:
                warmup_invalid += 1
            else:
                final_invalid += 1
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
                base.append_retained(samples, traces, post, w, evaluation)

    return _finish_chain(
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


def euclidean_mala_boundary_chain(
    post: base.GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
) -> BoundaryChainResult:
    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    x = base.flatten_weights(w)

    samples: List[Array] = []
    traces = _empty_traces()
    accepted_warm = accepted_keep = 0
    warmup_proposals = retained_proposals = 0
    warmup_invalid = final_invalid = 0
    warmup_failures = final_failures = 0

    n_total = warmup + retained * thin
    start = time.perf_counter()

    for iteration in range(n_total):
        in_warmup = iteration < warmup
        if in_warmup:
            warmup_proposals += 1
        else:
            retained_proposals += 1

        score_x = base.flatten_weights(evaluation.score_dx)
        mean_x = x + h * score_x
        y = mean_x + math.sqrt(2.0 * h) * rng.standard_normal(x.size)
        w_prop = base.unflatten_weights(y, post.e_count, post.d)

        accepted = False
        proposal_spd_invalid = not all(
            base.is_spd(w_prop[e]) for e in range(post.e_count)
        )
        numerical_failure = False

        if not proposal_spd_invalid:
            try:
                eval_prop = post.evaluate(w_prop)
                score_y = base.flatten_weights(eval_prop.score_dx)
                mean_y = y + h * score_y
                logq_forward = base.log_gaussian_isotropic(
                    y - mean_x, 2.0 * h
                )
                logq_reverse = base.log_gaussian_isotropic(
                    x - mean_y, 2.0 * h
                )
                log_alpha = (
                    eval_prop.log_target_dx + logq_reverse
                    - evaluation.log_target_dx - logq_forward
                )
                if math.log(rng.uniform()) < min(0.0, log_alpha):
                    x = y
                    w = w_prop
                    evaluation = eval_prop
                    accepted = True
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                numerical_failure = True

        if proposal_spd_invalid:
            if in_warmup:
                warmup_invalid += 1
            else:
                final_invalid += 1
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
                base.append_retained(samples, traces, post, w, evaluation)

    return _finish_chain(
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


def retraction_rmala_boundary_chain(
    post: base.GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
) -> BoundaryChainResult:
    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    volume_c = 0.5 * (post.d + 1.0)

    samples: List[Array] = []
    traces = _empty_traces()
    accepted_warm = accepted_keep = 0
    warmup_proposals = retained_proposals = 0
    warmup_invalid = final_invalid = 0
    warmup_failures = final_failures = 0

    n_total = warmup + retained * thin
    start = time.perf_counter()

    for iteration in range(n_total):
        in_warmup = iteration < warmup
        if in_warmup:
            warmup_proposals += 1
        else:
            retained_proposals += 1

        w_prop = np.empty_like(w)
        tangent_forward: List[Array] = []
        tangent_mean_forward: List[Array] = []
        proposal_spd_invalid = False
        numerical_failure = False

        try:
            for edge_index in range(post.e_count):
                root, _ = base.spd_sqrt_and_inv_sqrt(w[edge_index])
                mean_tangent = h * base.sym(
                    root @ evaluation.score_dx[edge_index] @ root
                )
                tangent = base.sym(
                    mean_tangent
                    + math.sqrt(2.0 * h)
                    * base.sample_standard_symmetric(post.d, rng)
                )
                w_prop[edge_index] = base.sym(
                    w[edge_index] + root @ tangent @ root
                )
                if not base.is_spd(w_prop[edge_index]):
                    proposal_spd_invalid = True
                    break
                tangent_forward.append(tangent)
                tangent_mean_forward.append(mean_tangent)
        except (np.linalg.LinAlgError, FloatingPointError):
            numerical_failure = True

        accepted = False
        if not proposal_spd_invalid and not numerical_failure:
            try:
                eval_prop = post.evaluate(w_prop)
                logq_forward = 0.0
                logq_reverse = 0.0

                for edge_index in range(post.e_count):
                    sf = tangent_forward[edge_index]
                    mf = tangent_mean_forward[edge_index]
                    logq_forward += (
                        base.log_gaussian_isotropic(
                            base.svec(sf - mf), 2.0 * h
                        )
                        - volume_c * base.spd_logdet(w[edge_index])
                    )

                    root_prop, inv_root_prop = (
                        base.spd_sqrt_and_inv_sqrt(w_prop[edge_index])
                    )
                    reverse_tangent = base.sym(
                        inv_root_prop
                        @ (w[edge_index] - w_prop[edge_index])
                        @ inv_root_prop
                    )
                    reverse_mean = h * base.sym(
                        root_prop
                        @ eval_prop.score_dx[edge_index]
                        @ root_prop
                    )
                    logq_reverse += (
                        base.log_gaussian_isotropic(
                            base.svec(reverse_tangent - reverse_mean),
                            2.0 * h,
                        )
                        - volume_c * base.spd_logdet(w_prop[edge_index])
                    )

                log_alpha = (
                    eval_prop.log_target_dx + logq_reverse
                    - evaluation.log_target_dx - logq_forward
                )
                if math.log(rng.uniform()) < min(0.0, log_alpha):
                    w = w_prop
                    evaluation = eval_prop
                    accepted = True
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                numerical_failure = True

        if proposal_spd_invalid:
            if in_warmup:
                warmup_invalid += 1
            else:
                final_invalid += 1
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
                base.append_retained(samples, traces, post, w, evaluation)

    return _finish_chain(
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


def fixed_fisher_boundary_chain(
    post: base.GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    root: Array,
    condition_number: float,
    seed: int,
) -> BoundaryChainResult:
    rng = base.set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    x = base.flatten_weights(w)
    qdim = x.size

    samples: List[Array] = []
    traces = _empty_traces()
    accepted_warm = accepted_keep = 0
    warmup_proposals = retained_proposals = 0
    warmup_invalid = final_invalid = 0
    warmup_failures = final_failures = 0

    n_total = warmup + retained * thin
    start = time.perf_counter()

    for iteration in range(n_total):
        in_warmup = iteration < warmup
        if in_warmup:
            warmup_proposals += 1
        else:
            retained_proposals += 1

        score_x = base.flatten_weights(evaluation.score_dx)
        p_score_x = root @ (root.T @ score_x)
        mean_x = x + h * p_score_x
        y = mean_x + math.sqrt(2.0 * h) * (
            root @ rng.standard_normal(qdim)
        )
        w_prop = base.unflatten_weights(y, post.e_count, post.d)

        accepted = False
        proposal_spd_invalid = not all(
            base.is_spd(w_prop[e]) for e in range(post.e_count)
        )
        numerical_failure = False

        if not proposal_spd_invalid:
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
                numerical_failure = True

        if proposal_spd_invalid:
            if in_warmup:
                warmup_invalid += 1
            else:
                final_invalid += 1
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
                base.append_retained(samples, traces, post, w, evaluation)

    return _finish_chain(
        samples=samples,
        traces=traces,
        accepted_keep=accepted_keep,
        accepted_warm=accepted_warm,
        retained_proposals=retained_proposals,
        warmup_proposals=warmup_proposals,
        runtime_sec=time.perf_counter() - start,
        h=h,
        preconditioner_condition=condition_number,
        final_invalid_spd=final_invalid,
        warmup_invalid_spd=warmup_invalid,
        final_numerical_failures=final_failures,
        warmup_numerical_failures=warmup_failures,
    )


def fixed_boundary_runner(
    method: str,
) -> Callable[..., BoundaryChainResult]:
    return {
        "ConeMALA": cone_mala_boundary_chain,
        "EuclideanMALA": euclidean_mala_boundary_chain,
        "RetractionRMALA": retraction_rmala_boundary_chain,
    }[method]


# =============================================================================
# Deterministic fixed-method tuning
# =============================================================================


def product_airm_esjd_per_iteration(samples: Array) -> float:
    if samples.shape[0] < 2:
        return 0.0

    values: List[float] = []
    for draw_index in range(samples.shape[0] - 1):
        total = 0.0
        for edge_index in range(samples.shape[1]):
            tangent = base.reverse_airm_tangent(
                samples[draw_index, edge_index],
                samples[draw_index + 1, edge_index],
            )
            total += float(np.sum(tangent * tangent))
        values.append(total)
    return float(np.mean(values))


def deterministic_tune_fixed_h(
    method: str,
    post: base.GraphPosterior,
    w_center: Array,
    experiment_seed: int,
    epsilon_index: int,
    h0: float,
    pilot_steps: int,
    pilot_chains: int,
    target_accept: float,
    acceptance_tolerance: float,
    grid_factors: Sequence[float],
    max_bracket_steps: int,
    bisection_steps: int,
) -> Tuple[float, float, float, List[Dict[str, float]]]:
    runner = fixed_boundary_runner(method)
    method_index = METHODS.index(method)
    tested: Dict[str, Dict[str, float]] = {}

    def evaluate(h: float) -> Dict[str, float]:
        h = float(max(h, 1e-14))
        key = f"{h:.17e}"
        if key in tested:
            return tested[key]

        acceptances: List[float] = []
        esjd_values: List[float] = []
        runtimes: List[float] = []
        invalid_rates: List[float] = []

        for chain_index in range(pilot_chains):
            init_rng = base.set_seed(
                30_000_000
                + 1_000_000 * epsilon_index
                + 10_000 * experiment_seed
                + 100 * method_index
                + chain_index
            )
            w_init = base.perturb_initial_weights(
                w_center,
                chain_index=chain_index,
                n_chains=pilot_chains,
                rng=init_rng,
            )
            result = runner(
                post=post,
                w_init=w_init,
                warmup=0,
                retained=pilot_steps,
                thin=1,
                h=h,
                seed=(
                    40_000_000
                    + 1_000_000 * epsilon_index
                    + 10_000 * experiment_seed
                    + 100 * method_index
                    + chain_index
                ),
            )
            acceptances.append(result.acceptance)
            esjd_values.append(
                product_airm_esjd_per_iteration(result.samples)
            )
            runtimes.append(result.runtime_sec)
            invalid_rates.append(
                result.final_invalid_spd
                / max(result.final_proposals, 1)
            )

        row = {
            "method": method,
            "h": h,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1)
                if len(acceptances) > 1 else 0.0
            ),
            "airm_esjd_per_iteration": float(np.mean(esjd_values)),
            "pilot_invalid_spd_rate": float(np.mean(invalid_rates)),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[key] = row
        return row

    for factor in sorted(set(float(x) for x in grid_factors if x > 0.0)):
        evaluate(h0 * factor)

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
        mid_key = f"{h_mid:.17e}"
        if mid_key in tested:
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


# =============================================================================
# Summaries and boundary diagnostics
# =============================================================================


def retained_boundary_stats(
    samples: Array,
    stride: int,
) -> Dict[str, float]:
    """
    Evaluate eigenvalue diagnostics on every `stride`th retained draw.
    """
    if samples.ndim != 4 or samples.shape[0] == 0:
        return {
            "sampled_min_eigenvalue_min": float("nan"),
            "sampled_min_eigenvalue_q01": float("nan"),
            "sampled_min_eigenvalue_median": float("nan"),
            "sampled_condition_number_q99": float("nan"),
            "sampled_condition_number_max": float("nan"),
        }

    draw_minima: List[float] = []
    draw_max_conditions: List[float] = []

    for draw in samples[::max(stride, 1)]:
        edge_minima: List[float] = []
        edge_conditions: List[float] = []
        for matrix in draw:
            eig = np.linalg.eigvalsh(base.sym(matrix))
            edge_minima.append(float(np.min(eig)))
            edge_conditions.append(float(np.max(eig) / np.min(eig)))
        draw_minima.append(float(np.min(edge_minima)))
        draw_max_conditions.append(float(np.max(edge_conditions)))

    minimums = np.asarray(draw_minima)
    conditions = np.asarray(draw_max_conditions)
    return {
        "sampled_min_eigenvalue_min": float(np.min(minimums)),
        "sampled_min_eigenvalue_q01": float(np.quantile(minimums, 0.01)),
        "sampled_min_eigenvalue_median": float(np.median(minimums)),
        "sampled_condition_number_q99": float(
            np.quantile(conditions, 0.99)
        ),
        "sampled_condition_number_max": float(np.max(conditions)),
    }


def summarize_boundary_method_seed(
    method: str,
    experiment_seed: int,
    epsilon: float,
    post: base.GraphPosterior,
    chain_results: Sequence[BoundaryChainResult],
    pilot_runtime_sec: float,
    parallel_runtime: bool,
    boundary_stride: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    trace_names = ("phi", "logdet_q", "heldout_nll", "rel_w_draw")
    traces = {
        name: np.stack(
            [result.traces[name] for result in chain_results], axis=0
        )
        for name in trace_names
    }

    rhats = {
        name: base.rank_normalized_split_rhat(values)
        for name, values in traces.items()
    }
    esses = {
        name: base.effective_sample_size(values)
        for name, values in traces.items()
    }
    mcses = {
        name: base.mcse_mean(traces[name], esses[name])
        for name in trace_names
    }

    chain_times = np.asarray(
        [result.runtime_sec for result in chain_results], dtype=float
    )
    sampling_runtime = float(
        np.max(chain_times) if parallel_runtime else np.sum(chain_times)
    )
    total_runtime = float(pilot_runtime_sec + sampling_runtime)

    ess_per_sec_sampling = {
        name: esses[name] / max(sampling_runtime, 1e-15)
        for name in trace_names
    }
    ess_per_sec_total = {
        name: esses[name] / max(total_runtime, 1e-15)
        for name in trace_names
    }
    ess_per_iteration = {
        name: esses[name] / max(traces[name].size, 1)
        for name in trace_names
    }

    all_samples = np.concatenate(
        [result.samples for result in chain_results], axis=0
    )
    posterior_mean_w = np.mean(all_samples, axis=0)
    posterior_mean_stats = matrix_family_stats(posterior_mean_w)
    truth_stats = matrix_family_stats(post.w_true)
    sampled_stats = retained_boundary_stats(
        all_samples, stride=boundary_stride
    )

    final_proposals = sum(
        result.final_proposals for result in chain_results
    )
    warmup_proposals = sum(
        result.warmup_proposals for result in chain_results
    )
    final_invalid = sum(
        result.final_invalid_spd for result in chain_results
    )
    warmup_invalid = sum(
        result.warmup_invalid_spd for result in chain_results
    )
    final_failures = sum(
        result.final_numerical_failures for result in chain_results
    )
    warmup_failures = sum(
        result.warmup_numerical_failures for result in chain_results
    )

    row: Dict[str, float] = {
        "epsilon": float(epsilon),
        "truth_condition_target": float(1.0 / epsilon),
        "experiment_seed": int(experiment_seed),
        "method": method,
        "acceptance": float(
            np.mean([result.acceptance for result in chain_results])
        ),
        "warmup_acceptance": float(
            np.mean(
                [result.warmup_acceptance for result in chain_results]
            )
        ),
        "h_final_mean": float(
            np.mean([result.h_final for result in chain_results])
        ),
        "preconditioner_condition_mean": float(
            np.mean(
                [
                    result.preconditioner_condition
                    for result in chain_results
                ]
            )
        ),
        "pilot_runtime_sec": float(pilot_runtime_sec),
        "sampling_runtime_sec": sampling_runtime,
        "total_runtime_sec": total_runtime,
        "min_ess_per_iteration": float(
            min(ess_per_iteration.values())
        ),
        "min_ess_per_sec_sampling": float(
            min(ess_per_sec_sampling.values())
        ),
        "min_ess_per_sec_total": float(
            min(ess_per_sec_total.values())
        ),
        "mcse_heldout_nll": float(mcses["heldout_nll"]),
        "max_rhat": float(max(rhats.values())),
        "heldout_nll": float(np.mean(traces["heldout_nll"])),
        "relative_w_error": float(post.relative_w_error(posterior_mean_w)),
        "final_invalid_spd_rate": float(
            final_invalid / max(final_proposals, 1)
        ),
        "warmup_invalid_spd_rate": float(
            warmup_invalid / max(warmup_proposals, 1)
        ),
        "final_numerical_failure_rate": float(
            final_failures / max(final_proposals, 1)
        ),
        "warmup_numerical_failure_rate": float(
            warmup_failures / max(warmup_proposals, 1)
        ),
        "truth_min_eigenvalue_min": truth_stats["min_eigenvalue_min"],
        "truth_min_eigenvalue_mean": truth_stats["min_eigenvalue_mean"],
        "truth_condition_number_max": truth_stats["condition_number_max"],
        "posterior_mean_min_eigenvalue_min": (
            posterior_mean_stats["min_eigenvalue_min"]
        ),
        "posterior_mean_condition_number_max": (
            posterior_mean_stats["condition_number_max"]
        ),
        **sampled_stats,
    }

    diagnostics: List[Dict[str, float]] = []
    for observable in trace_names:
        diagnostics.append(
            {
                "epsilon": float(epsilon),
                "truth_condition_target": float(1.0 / epsilon),
                "experiment_seed": int(experiment_seed),
                "method": method,
                "observable": observable,
                "rhat": rhats[observable],
                "ess": esses[observable],
                "ess_per_iteration": ess_per_iteration[observable],
                "ess_per_sec_sampling": ess_per_sec_sampling[observable],
                "ess_per_sec_total": ess_per_sec_total[observable],
                "mcse": mcses[observable],
            }
        )

    return row, diagnostics


SUMMARY_METRICS = (
    "acceptance",
    "h_final_mean",
    "preconditioner_condition_mean",
    "min_ess_per_iteration",
    "min_ess_per_sec_sampling",
    "min_ess_per_sec_total",
    "mcse_heldout_nll",
    "max_rhat",
    "heldout_nll",
    "relative_w_error",
    "final_invalid_spd_rate",
    "final_numerical_failure_rate",
    "posterior_mean_min_eigenvalue_min",
    "posterior_mean_condition_number_max",
    "sampled_min_eigenvalue_q01",
    "sampled_condition_number_q99",
)


def aggregate_boundary_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    grouped = seed_df.groupby(["epsilon", "method"], sort=True)
    for (epsilon, method), group in grouped:
        row: Dict[str, float] = {
            "epsilon": float(epsilon),
            "truth_condition_target": float(1.0 / epsilon),
            "method": method,
            "n_experiment_seeds": int(
                group["experiment_seed"].nunique()
            ),
            "truth_min_eigenvalue_mean": float(
                group["truth_min_eigenvalue_min"].mean()
            ),
        }
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = (
                float(np.std(values, ddof=1))
                if values.size > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def latex_mean_sd(mean: float, sd: float, digits: int) -> str:
    return f"${mean:.{digits}f}\\pm {sd:.{digits}f}$"


def write_boundary_latex_table(
    summary_df: pd.DataFrame,
    path: Path,
) -> None:
    method_order = {name: index for index, name in enumerate(METHODS)}
    table = summary_df.copy()
    table["_method_order"] = table["method"].map(method_order)
    table = table.sort_values(["epsilon", "_method_order"], ascending=[False, True])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Boundary-stress comparison across independent "
            r"data-generation seeds.  The true edge condition number is "
            r"$\kappa(W_e^\star)=1/\varepsilon$.  ESS/sec$_{\rm samp}$ "
            r"uses only final-chain runtime, whereas ESS/sec$_{\rm total}$ "
            r"also includes adaptation and step-size tuning.  Invalid denotes "
            r"the fraction of final proposals leaving the SPD cone before the "
            r"Metropolis--Hastings test.}"
        ),
        r"\label{tab:boundary_stress}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cclccccccc}",
        r"\toprule",
        (
            r"$\varepsilon$ & $\kappa^\star$ & Method & Acceptance & "
            r"Min. ESS/iter. & Min. ESS/sec$_{\rm samp}$ & "
            r"Min. ESS/sec$_{\rm total}$ & Max. $\widehat R$ & "
            r"Invalid & Post. mean $\lambda_{\min}$ \\"
        ),
        r"\midrule",
    ]

    previous_epsilon: Optional[float] = None
    for _, row in table.iterrows():
        epsilon = float(row["epsilon"])
        if previous_epsilon is not None and epsilon != previous_epsilon:
            lines.append(r"\addlinespace")
        previous_epsilon = epsilon

        lines.append(
            " & ".join(
                [
                    f"${epsilon:.0e}$",
                    f"${1.0 / epsilon:.0f}$",
                    base.method_label(str(row["method"])),
                    latex_mean_sd(
                        row["acceptance_mean"], row["acceptance_sd"], 3
                    ),
                    latex_mean_sd(
                        row["min_ess_per_iteration_mean"],
                        row["min_ess_per_iteration_sd"],
                        3,
                    ),
                    latex_mean_sd(
                        row["min_ess_per_sec_sampling_mean"],
                        row["min_ess_per_sec_sampling_sd"],
                        2,
                    ),
                    latex_mean_sd(
                        row["min_ess_per_sec_total_mean"],
                        row["min_ess_per_sec_total_sd"],
                        2,
                    ),
                    latex_mean_sd(
                        row["max_rhat_mean"], row["max_rhat_sd"], 3
                    ),
                    latex_mean_sd(
                        row["final_invalid_spd_rate_mean"],
                        row["final_invalid_spd_rate_sd"],
                        3,
                    ),
                    latex_mean_sd(
                        row["posterior_mean_min_eigenvalue_min_mean"],
                        row["posterior_mean_min_eigenvalue_min_sd"],
                        4,
                    ),
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


def make_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    method_order = [method for method in METHODS if method in set(summary_df["method"])]

    for metric, ylabel, filename, log_y in (
        (
            "min_ess_per_sec_sampling_mean",
            "Minimum ESS/sec (sampling only)",
            "boundary_ess_per_sec_sampling.pdf",
            True,
        ),
        (
            "max_rhat_mean",
            "Maximum split-Rhat",
            "boundary_rhat.pdf",
            False,
        ),
        (
            "final_invalid_spd_rate_mean",
            "Invalid-SPD proposal rate",
            "boundary_invalid_spd_rate.pdf",
            True,
        ),
        (
            "posterior_mean_min_eigenvalue_min_mean",
            "Posterior-mean minimum eigenvalue",
            "boundary_posterior_min_eigenvalue.pdf",
            True,
        ),
    ):
        fig, ax = plt.subplots()
        for method in method_order:
            subset = summary_df[summary_df["method"] == method].sort_values(
                "epsilon", ascending=False
            )
            ax.plot(
                subset["truth_condition_target"],
                subset[metric],
                marker="o",
                label=base.method_label(method),
            )
        ax.set_xscale("log")
        if log_y:
            positive = summary_df[metric].to_numpy(float)
            if np.any(positive > 0.0):
                ax.set_yscale("log")
        ax.set_xlabel(r"True edge condition number $1/\varepsilon$")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Main experiment
# =============================================================================


def fixed_initial_state(
    w_center: Array,
    experiment_seed: int,
    epsilon_index: int,
    method: str,
    chain_index: int,
    n_chains: int,
) -> Array:
    method_index = METHODS.index(method)
    rng = base.set_seed(
        50_000_000
        + 1_000_000 * epsilon_index
        + 10_000 * experiment_seed
        + 100 * method_index
        + chain_index
    )
    return base.perturb_initial_weights(
        w_center,
        chain_index=chain_index,
        n_chains=n_chains,
        rng=rng,
    )


def run_method(
    method: str,
    post: base.GraphPosterior,
    config: base.ExperimentConfig,
    run_config: BoundaryRunConfig,
    experiment_seed: int,
    epsilon: float,
    epsilon_index: int,
) -> Tuple[
    Dict[str, float],
    List[Dict[str, float]],
    List[Dict[str, float]],
]:
    w_center = base.initial_weights(post)
    tuning_rows: List[Dict[str, float]] = []

    if method in FIXED_METHODS:
        h0_map = {
            "ConeMALA": config.cone_h0,
            "EuclideanMALA": config.euclidean_h0,
            "RetractionRMALA": config.retraction_h0,
        }
        h_selected, pilot_acceptance, pilot_runtime, rows = (
            deterministic_tune_fixed_h(
                method=method,
                post=post,
                w_center=w_center,
                experiment_seed=experiment_seed,
                epsilon_index=epsilon_index,
                h0=h0_map[method],
                pilot_steps=run_config.pilot_steps,
                pilot_chains=run_config.pilot_chains,
                target_accept=config.target_accept,
                acceptance_tolerance=run_config.acceptance_tolerance,
                grid_factors=(0.25, 0.5, 1.0, 2.0, 4.0),
                max_bracket_steps=run_config.max_bracket_steps,
                bisection_steps=run_config.bisection_steps,
            )
        )
        for row in rows:
            row.update(
                {
                    "epsilon": epsilon,
                    "truth_condition_target": 1.0 / epsilon,
                    "experiment_seed": experiment_seed,
                    "stage": "fixed_h_tuning",
                }
            )
            tuning_rows.append(row)

        print(
            f"{method}: selected h={h_selected:.6e}, "
            f"pilot acceptance={pilot_acceptance:.3f}, "
            f"pilot time={pilot_runtime:.1f}s"
        )

        runner = fixed_boundary_runner(method)
        chain_results: List[BoundaryChainResult] = []
        for chain_index in range(config.n_chains):
            w_init = fixed_initial_state(
                w_center=w_center,
                experiment_seed=experiment_seed,
                epsilon_index=epsilon_index,
                method=method,
                chain_index=chain_index,
                n_chains=config.n_chains,
            )
            result = runner(
                post=post,
                w_init=w_init,
                warmup=config.warmup,
                retained=config.retained,
                thin=config.thin,
                h=h_selected,
                seed=(
                    60_000_000
                    + 1_000_000 * epsilon_index
                    + 10_000 * experiment_seed
                    + 100 * METHODS.index(method)
                    + chain_index
                ),
            )
            chain_results.append(result)
            print(
                f"  chain {chain_index + 1}: acc={result.acceptance:.3f}, "
                f"warmup={result.warmup_acceptance:.3f}, "
                f"invalid={result.final_invalid_spd / max(result.final_proposals,1):.3f}, "
                f"time={result.runtime_sec:.1f}s"
            )

    elif method == "FisherAdaptiveMALA":
        plan = fisher.prepare_fisher_plan(
            post=post,
            w_center=w_center,
            n_chains=config.n_chains,
            adapt_chains=run_config.fisher_adapt_chains,
            adapt_steps=run_config.fisher_adapt_steps,
            tune_chains=run_config.fisher_tune_chains,
            tune_steps=run_config.fisher_tune_steps,
            h0=config.fisher_h0,
            target_accept=config.target_accept,
            damping=config.fisher_damping,
            init_steps=min(
                config.fisher_init_steps,
                max(run_config.fisher_adapt_steps - 1, 0),
            ),
            update_interval=config.fisher_update_interval,
            max_condition=min(
                config.fisher_max_condition,
                run_config.fisher_condition_cap,
            ),
            step_adapt_rate=config.step_adapt_rate,
            grid_factors=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
            seed_base=(
                70_000_000
                + 1_000_000 * epsilon_index
                + 10_000 * experiment_seed
            ),
            acceptance_tolerance=run_config.acceptance_tolerance,
            max_bracket_steps=run_config.max_bracket_steps,
            bisection_steps=run_config.bisection_steps,
        )
        pilot_runtime = (
            plan.adaptation_runtime_sec + plan.tuning_runtime_sec
        )
        for row in plan.adaptation_rows + plan.tuning_rows:
            row.update(
                {
                    "epsilon": epsilon,
                    "truth_condition_target": 1.0 / epsilon,
                    "experiment_seed": experiment_seed,
                    "shared_preconditioner_condition": (
                        plan.condition_number
                    ),
                }
            )
            tuning_rows.append(row)

        print(
            f"FisherAdaptiveMALA: condition={plan.condition_number:.3e}, "
            f"h={plan.h_selected:.6e}, "
            f"adapt+tune time={pilot_runtime:.1f}s"
        )

        chain_results = []
        for chain_index in range(config.n_chains):
            w_init = fixed_initial_state(
                w_center=w_center,
                experiment_seed=experiment_seed,
                epsilon_index=epsilon_index,
                method=method,
                chain_index=chain_index,
                n_chains=config.n_chains,
            )
            result = fixed_fisher_boundary_chain(
                post=post,
                w_init=w_init,
                warmup=config.warmup,
                retained=config.retained,
                thin=config.thin,
                h=plan.h_selected,
                root=plan.root,
                condition_number=plan.condition_number,
                seed=(
                    80_000_000
                    + 1_000_000 * epsilon_index
                    + 10_000 * experiment_seed
                    + chain_index
                ),
            )
            chain_results.append(result)
            print(
                f"  chain {chain_index + 1}: acc={result.acceptance:.3f}, "
                f"warmup={result.warmup_acceptance:.3f}, "
                f"invalid={result.final_invalid_spd / max(result.final_proposals,1):.3f}, "
                f"time={result.runtime_sec:.1f}s"
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    summary, diagnostics = summarize_boundary_method_seed(
        method=method,
        experiment_seed=experiment_seed,
        epsilon=epsilon,
        post=post,
        chain_results=chain_results,
        pilot_runtime_sec=pilot_runtime,
        parallel_runtime=config.parallel_runtime,
        boundary_stride=run_config.boundary_stride,
    )

    print(
        f"  summary: acc={summary['acceptance']:.3f}, "
        f"min ESS/s samp={summary['min_ess_per_sec_sampling']:.3f}, "
        f"min ESS/s total={summary['min_ess_per_sec_total']:.3f}, "
        f"Rhat={summary['max_rhat']:.3f}, "
        f"invalid={summary['final_invalid_spd_rate']:.3f}, "
        f"post mean min eig={summary['posterior_mean_min_eigenvalue_min']:.3e}"
    )

    return summary, diagnostics, tuning_rows


def checkpoint(
    output_dir: Path,
    seed_rows: List[Dict[str, float]],
    diagnostic_rows: List[Dict[str, float]],
    tuning_rows: List[Dict[str, float]],
    truth_rows: List[Dict[str, float]],
) -> None:
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "boundary_comparison_by_seed.csv", index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "boundary_diagnostics.csv", index=False
    )
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "boundary_tuning.csv", index=False
    )
    pd.DataFrame(truth_rows).to_csv(
        output_dir / "boundary_truth.csv", index=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare four samplers on a paired SPD-boundary family."
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
        default=(1e-1, 1e-2, 1e-3, 1e-4),
    )
    parser.add_argument(
        "--methods",
        type=parse_str_tuple,
        default=METHODS,
        help="Comma-separated method names.",
    )
    parser.add_argument(
        "--prior-mode",
        choices=("original", "boundary-friendly"),
        default="original",
    )
    parser.add_argument(
        "--boundary-nu-offset",
        type=float,
        default=0.5,
        help="Uses nu=d+offset in boundary-friendly prior mode.",
    )
    parser.add_argument("--pilot-chains", type=int)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--acceptance-tolerance", type=float, default=0.08)
    parser.add_argument("--max-bracket-steps", type=int, default=20)
    parser.add_argument("--bisection-steps", type=int, default=8)
    parser.add_argument("--fisher-adapt-chains", type=int)
    parser.add_argument("--fisher-adapt-steps", type=int)
    parser.add_argument("--fisher-tune-chains", type=int)
    parser.add_argument("--fisher-tune-steps", type=int)
    parser.add_argument("--fisher-condition-cap", type=float, default=100.0)
    parser.add_argument("--boundary-stride", type=int, default=10)
    parser.add_argument("--make-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_config = load_base_config(args.config)
    config = resolve_profile(source_config, args.profile)
    config = replace(config, methods=tuple(args.methods))
    base.validate_config(config)

    epsilons = tuple(float(value) for value in args.epsilons)
    if any(not (0.0 < value <= 1.0) for value in epsilons):
        raise ValueError("Every epsilon must lie in (0,1].")

    pilot_chains = (
        min(config.n_chains, 4)
        if args.pilot_chains is None else args.pilot_chains
    )
    pilot_steps = (
        config.pilot_steps
        if args.pilot_steps is None else args.pilot_steps
    )
    fisher_adapt_chains = (
        config.n_chains
        if args.fisher_adapt_chains is None
        else args.fisher_adapt_chains
    )
    fisher_adapt_steps = (
        config.warmup
        if args.fisher_adapt_steps is None
        else args.fisher_adapt_steps
    )
    fisher_tune_chains = (
        min(config.n_chains, 4)
        if args.fisher_tune_chains is None
        else args.fisher_tune_chains
    )
    fisher_tune_steps = (
        config.pilot_steps
        if args.fisher_tune_steps is None
        else args.fisher_tune_steps
    )

    run_config = BoundaryRunConfig(
        source_config=str(args.config),
        profile=args.profile,
        output_dir=str(args.output_dir),
        epsilons=epsilons,
        methods=tuple(args.methods),
        prior_mode=args.prior_mode,
        boundary_nu_offset=args.boundary_nu_offset,
        pilot_chains=pilot_chains,
        pilot_steps=pilot_steps,
        acceptance_tolerance=args.acceptance_tolerance,
        max_bracket_steps=args.max_bracket_steps,
        bisection_steps=args.bisection_steps,
        fisher_adapt_chains=fisher_adapt_chains,
        fisher_adapt_steps=fisher_adapt_steps,
        fisher_tune_chains=fisher_tune_chains,
        fisher_tune_steps=fisher_tune_steps,
        fisher_condition_cap=args.fisher_condition_cap,
        boundary_stride=args.boundary_stride,
        make_plots=args.make_plots,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "boundary_run_config": asdict(run_config),
        "resolved_experiment_config": asdict(config),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    print("PSD-weighted graph boundary-stress comparison")
    print("=" * 88)
    print(json.dumps(resolved, indent=2))

    seed_rows: List[Dict[str, float]] = []
    diagnostic_rows: List[Dict[str, float]] = []
    tuning_rows: List[Dict[str, float]] = []
    truth_rows: List[Dict[str, float]] = []

    for epsilon_index, epsilon in enumerate(epsilons):
        print("\n" + "#" * 88)
        print(
            f"epsilon={epsilon:.1e}, "
            f"target edge condition number={1.0 / epsilon:.1e}"
        )
        print("#" * 88)

        for experiment_seed in config.experiment_seeds:
            print("\n" + "=" * 88)
            print(f"Independent data seed: {experiment_seed}")
            print("=" * 88)

            post = make_boundary_problem(
                config=config,
                experiment_seed=experiment_seed,
                epsilon=epsilon,
                prior_mode=run_config.prior_mode,
                boundary_nu_offset=run_config.boundary_nu_offset,
            )
            truth_stats = matrix_family_stats(post.w_true)
            truth_rows.append(
                {
                    "epsilon": epsilon,
                    "truth_condition_target": 1.0 / epsilon,
                    "experiment_seed": experiment_seed,
                    "prior_mode": run_config.prior_mode,
                    "nu": post.nu,
                    **truth_stats,
                }
            )
            print(
                f"truth min eig={truth_stats['min_eigenvalue_min']:.3e}, "
                f"truth max condition={truth_stats['condition_number_max']:.3e}, "
                f"nu={post.nu:.3f}"
            )

            for method in run_config.methods:
                print("\n" + "-" * 88)
                print(base.method_label(method))
                print("-" * 88)
                row, diagnostics, tuning = run_method(
                    method=method,
                    post=post,
                    config=config,
                    run_config=run_config,
                    experiment_seed=experiment_seed,
                    epsilon=epsilon,
                    epsilon_index=epsilon_index,
                )
                row["prior_mode"] = run_config.prior_mode
                row["nu"] = post.nu
                seed_rows.append(row)
                diagnostic_rows.extend(diagnostics)
                tuning_rows.extend(tuning)

                checkpoint(
                    output_dir=args.output_dir,
                    seed_rows=seed_rows,
                    diagnostic_rows=diagnostic_rows,
                    tuning_rows=tuning_rows,
                    truth_rows=truth_rows,
                )

    seed_df = pd.DataFrame(seed_rows)
    summary_df = aggregate_boundary_results(seed_df)
    summary_df["prior_mode"] = run_config.prior_mode
    summary_df.to_csv(
        args.output_dir / "boundary_summary.csv", index=False
    )
    write_boundary_latex_table(
        summary_df,
        args.output_dir / "table_boundary_stress.tex",
    )

    if run_config.make_plots:
        make_plots(summary_df, args.output_dir)

    print("\nAcross-seed boundary summary")
    print("=" * 88)
    display_columns = [
        "epsilon",
        "method",
        "acceptance_mean",
        "min_ess_per_iteration_mean",
        "min_ess_per_sec_sampling_mean",
        "min_ess_per_sec_total_mean",
        "max_rhat_mean",
        "final_invalid_spd_rate_mean",
        "posterior_mean_min_eigenvalue_min_mean",
    ]
    print(summary_df[display_columns].to_string(index=False))

    print("\nSaved outputs:")
    for name in (
        "config.json",
        "boundary_truth.csv",
        "boundary_comparison_by_seed.csv",
        "boundary_diagnostics.csv",
        "boundary_tuning.csv",
        "boundary_summary.csv",
        "table_boundary_stress.tex",
    ):
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
