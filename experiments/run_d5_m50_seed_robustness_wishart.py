#!/usr/bin/env python3
"""
Five-seed robustness study for the determinant-neutral d=5, m=50
cycle-graph posterior.

This is a minimally modified version of the graph-size sweep script.  The
posterior, four samplers, strict SPD checks, Fisher adaptation/tuning
workflow, observables, ESS calculations, and m=50 budgets are unchanged.
The synthetic data-generating process is repeated independently for dataset
seeds 11, 22, 33, 44, and 55.  In addition to the original efficiency and
convergence diagnostics, this version computes posterior-mean relative
edge-weight recovery and equal-tailed marginal credible-interval coverage for
unique edge-matrix entries, edge traces, and ordered edge eigenvalues.

For every dataset seed:
* d=5, m=50, n_train=n_test=80, and four chains;
* alpha_logdet=(d+1)/2=3 (determinant neutral);
* 4000 discarded and 12000 retained transitions per chain;
* fixed h values: ConeMALA 6e-3, Euclidean MALA 2e-5,
  generic RMALA 1e-4;
* Fisher-adaptive MALA performs discarded chain-specific adaptation,
  pools a shared dense preconditioner, tunes a shared scalar h, and freezes
  both before retained production;
* invalid additive proposals are rejected and are never projected.
* Fisher-adaptive MALA updates its preconditioner at every discarded
  adaptive iteration (Titsias 2023, Algorithm 1).
* --methods selects the samplers (default: all four); the five-dataset
  tables are written when all four samplers are present for every seed.

The script writes per-seed results plus:
* robustness_d5_m50_by_seed.csv;
* robustness_d5_m50_summary.csv;
* table_d5_m50_seed_robustness.tex;
* table_d5_m50_seed_recovery_coverage.tex;
* per-seed, per-method credible-interval detail CSV files;
* fig_d5_m50_seed_robustness.pdf and .png.

Example
-------
    python3 -u run_d5_m50_seed_robustness_wishart.py \
        --seeds 11,22,33,44,55 \
        --outdir d5_m50_seed_robustness \
        --resume

"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh, inv, slogdet
from scipy.linalg import cholesky


Array = np.ndarray

WISHART_ROBUSTNESS_VERSION = "1.0"


# =============================================================================
# Numerical helpers
# =============================================================================

class DrawBuffer:
    """Preallocated store for retained draws.

    Behaves like the list it replaces (``append`` then ``np.asarray``), but
    writes each draw into one preallocated array, so a chain needs the memory
    of its draws once instead of twice (list of copies + final stacked array).
    The stored values are bit-for-bit identical.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buf: Optional[Array] = None
        self.count = 0

    def append(self, draw: Array) -> None:
        if self.buf is None:
            self.buf = np.empty((self.capacity,) + np.shape(draw),
                                dtype=np.asarray(draw).dtype)
        self.buf[self.count] = draw
        self.count += 1

    def __len__(self) -> int:
        return self.count

    def __array__(self, dtype=None, copy=None):
        if self.buf is None:
            return np.empty((0,), dtype=dtype or float)
        out = self.buf[: self.count]
        return out if dtype is None else out.astype(dtype)


def sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


def is_spd(a: Array) -> bool:
    """Strict SPD test: no diagonal jitter is added."""
    try:
        np.linalg.cholesky(sym(a))
        return True
    except np.linalg.LinAlgError:
        return False


def project_spd(a: Array, eps: float = 1e-8) -> Array:
    a = sym(a)
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, eps)
    return sym((vecs * vals) @ vecs.T)


def mat_sqrt_inv_sqrt(x: Array) -> Tuple[Array, Array]:
    x = sym(x)
    vals, vecs = np.linalg.eigh(x)
    vals = np.maximum(vals, 1e-12)
    xsqrt = (vecs * np.sqrt(vals)) @ vecs.T
    xisqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    return sym(xsqrt), sym(xisqrt)


def logdet_spd(x: Array) -> float:
    sign, value = slogdet(sym(x))
    return float(value) if sign > 0 else -np.inf


def expm_sym(s: Array) -> Array:
    vals, vecs = np.linalg.eigh(sym(s))
    return sym((vecs * np.exp(vals)) @ vecs.T)


def logm_spd(a: Array) -> Array:
    vals, vecs = np.linalg.eigh(project_spd(a))
    vals = np.maximum(vals, 1e-15)
    return sym((vecs * np.log(vals)) @ vecs.T)


def spd_congruence_log(x: Array, y: Array) -> Array:
    _, xisqrt = mat_sqrt_inv_sqrt(x)
    return logm_spd(sym(xisqrt @ y @ xisqrt))


def sym_to_orthovec(a: Array) -> Array:
    a = sym(a)
    d = a.shape[0]
    values: List[float] = []
    for i in range(d):
        values.append(float(a[i, i]))
    sqrt2 = math.sqrt(2.0)
    for i in range(d):
        for j in range(i + 1, d):
            values.append(float(sqrt2 * a[i, j]))
    return np.asarray(values, dtype=float)


def orthovec_to_sym(v: Array, d: int) -> Array:
    v = np.asarray(v, dtype=float).reshape(-1)
    expected = d * (d + 1) // 2
    if v.size != expected:
        raise ValueError(f"Expected {expected} symmetric coordinates, got {v.size}.")
    a = np.zeros((d, d), dtype=float)
    index = 0
    for i in range(d):
        a[i, i] = v[index]
        index += 1
    sqrt2 = math.sqrt(2.0)
    for i in range(d):
        for j in range(i + 1, d):
            a[i, j] = v[index] / sqrt2
            a[j, i] = a[i, j]
            index += 1
    return sym(a)


def random_sym_normal(d: int, rng: np.random.Generator) -> Array:
    p = d * (d + 1) // 2
    return orthovec_to_sym(rng.normal(size=p), d)


def flatten_weights(w: Array) -> Array:
    return np.concatenate([sym_to_orthovec(w[e]) for e in range(w.shape[0])])


def unflatten_weights(x: Array, edge_count: int, d: int) -> Array:
    block = d * (d + 1) // 2
    x = np.asarray(x, dtype=float)
    if x.size != edge_count * block:
        raise ValueError(
            f"Expected {edge_count * block} coordinates, got {x.size}."
        )
    return np.asarray(
        [
            orthovec_to_sym(x[e * block:(e + 1) * block], d)
            for e in range(edge_count)
        ]
    )


# =============================================================================
# Graph posterior: unchanged from the original d=5 sweep
# =============================================================================

def make_cycle_graph(m: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % m) for i in range(m)]


def block_laplacian(
    w: Array,
    edges: Sequence[Tuple[int, int]],
    m: int,
    d: int,
) -> Array:
    laplacian = np.zeros((m * d, m * d), dtype=float)
    for edge_index, (i, j) in enumerate(edges):
        we = sym(w[edge_index])
        si = slice(i * d, (i + 1) * d)
        sj = slice(j * d, (j + 1) * d)
        laplacian[si, si] += we
        laplacian[sj, sj] += we
        laplacian[si, sj] -= we
        laplacian[sj, si] -= we
    return sym(laplacian)


def edge_gradient_from_a(
    a: Array,
    edges: Sequence[Tuple[int, int]],
    d: int,
) -> Array:
    gradients: List[Array] = []
    for i, j in edges:
        si = slice(i * d, (i + 1) * d)
        sj = slice(j * d, (j + 1) * d)
        gradient = a[si, si] + a[sj, sj] - a[si, sj] - a[sj, si]
        gradients.append(sym(gradient))
    return np.asarray(gradients)


def sample_ground_truth_w(
    edge_count: int,
    d: int,
    rng: np.random.Generator,
    scale: float = 0.25,
) -> Array:
    result = np.zeros((edge_count, d, d), dtype=float)
    for edge_index in range(edge_count):
        a = rng.normal(size=(d, d))
        result[edge_index] = (
            scale * (a @ a.T) + 0.15 * np.eye(d)
        )
    return np.asarray([project_spd(result[e]) for e in range(edge_count)])


def sample_graph_signals(
    precision: Array,
    n_samples: int,
    rng: np.random.Generator,
) -> Array:
    covariance = sym(inv(precision))
    chol = cholesky(covariance, lower=True)
    return chol @ rng.normal(size=(precision.shape[0], n_samples))


class PSDGraphPosterior:
    def __init__(
        self,
        edges: Sequence[Tuple[int, int]],
        m: int,
        d: int,
        y_train: Array,
        r_matrix: Array,
        lambda_tr: float = 0.05,
        eta_frob: float = 0.0,
        alpha_logdet: float = 3.0,
    ):
        self.edges = list(edges)
        self.m = int(m)
        self.d = int(d)
        self.edge_count = len(edges)
        self.y_train = np.asarray(y_train, dtype=float)
        self.n_train = self.y_train.shape[1]
        self.syy = self.y_train @ self.y_train.T
        self.r_matrix = sym(r_matrix)
        self.lambda_tr = float(lambda_tr)
        self.eta_frob = float(eta_frob)
        self.alpha_logdet = float(alpha_logdet)
        self.volume_coefficient = 0.5 * (self.d + 1.0)

    def q(self, w: Array) -> Array:
        return (
            block_laplacian(w, self.edges, self.m, self.d)
            + self.r_matrix
        )

    def phi_and_grad(self, w: Array) -> Tuple[float, Array]:
        w = np.asarray([sym(w[e]) for e in range(len(w))])
        q_matrix = self.q(w)
        logdet_q = logdet_spd(q_matrix)
        if not np.isfinite(logdet_q):
            return np.inf, np.zeros_like(w)

        q_inverse = sym(inv(q_matrix))
        phi_likelihood = (
            -0.5 * self.n_train * logdet_q
            + 0.5 * np.trace(q_matrix @ self.syy)
        )
        a_matrix = (
            -0.5 * self.n_train * q_inverse
            + 0.5 * self.syy
        )
        grad_w = edge_gradient_from_a(a_matrix, self.edges, self.d)

        phi_prior = 0.0
        for edge_index in range(self.edge_count):
            we = sym(w[edge_index])
            logdet_we = logdet_spd(we)
            if not np.isfinite(logdet_we):
                return np.inf, np.zeros_like(w)

            we_inverse = sym(inv(we))
            phi_prior += self.lambda_tr * np.trace(we)
            phi_prior += 0.5 * self.eta_frob * np.sum(we * we)
            phi_prior -= self.alpha_logdet * logdet_we

            grad_w[edge_index] += self.lambda_tr * np.eye(self.d)
            grad_w[edge_index] += self.eta_frob * we
            grad_w[edge_index] -= self.alpha_logdet * we_inverse

        return (
            float(phi_likelihood + phi_prior),
            np.asarray([sym(gradient) for gradient in grad_w]),
        )

    def log_volume_density_dx(self, w: Array) -> float:
        total = 0.0
        for edge_index in range(self.edge_count):
            value = logdet_spd(w[edge_index])
            if not np.isfinite(value):
                return -np.inf
            total -= self.volume_coefficient * value
        return float(total)

    def log_target_dx(self, w: Array) -> float:
        phi, _ = self.phi_and_grad(w)
        return float(-phi + self.log_volume_density_dx(w))

    def euclidean_score_dx(self, w: Array, grad_phi: Array) -> Array:
        """Score of the common target with respect to Euclidean svec coordinates."""
        score_blocks: List[Array] = []
        for edge_index in range(self.edge_count):
            negative_log_gradient = (
                grad_phi[edge_index]
                + self.volume_coefficient * inv(w[edge_index])
            )
            score_blocks.append(sym_to_orthovec(-negative_log_gradient))
        return np.concatenate(score_blocks)


# =============================================================================
# Proposal-density helpers
# =============================================================================

def log_gaussian_orthosym(s: Array, mean: Array, variance: float) -> float:
    sv = sym_to_orthovec(s)
    mv = sym_to_orthovec(mean)
    residual = sv - mv
    p = len(sv)
    return float(
        -0.5 * p * np.log(2.0 * np.pi * variance)
        -0.5 * np.dot(residual, residual) / variance
    )


def log_spd_exp_jacobian(s: Array) -> float:
    values = eigvalsh(sym(s))
    total = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a = 0.5 * (values[i] - values[j])
            if abs(a) < 1e-8:
                total += (a * a) / 6.0
            else:
                total += np.log(abs(np.sinh(a) / a))
    return float(total)


# =============================================================================
# Chain result
# =============================================================================

@dataclass
class ChainResult:
    samples: Array
    acceptance: float
    preburn_acceptance: float
    runtime_total_sec: float
    runtime_sampling_sec: float
    h_final: float
    invalid_rate: float
    preburn_invalid_rate: float
    numerical_failure_rate: float
    preconditioner_condition: float = 1.0


# =============================================================================
# Fixed-step ConeMALA
# =============================================================================

def _cone_transition(
    posterior: PSDGraphPosterior,
    w: Array,
    phi: float,
    grad: Array,
    h: float,
    rng: np.random.Generator,
) -> Tuple[Array, float, Array, bool, bool]:
    edge_count, d, _ = w.shape
    proposal = np.zeros_like(w)
    tangent_list: List[Array] = []
    mean_list: List[Array] = []

    try:
        for edge_index in range(edge_count):
            root, _ = mat_sqrt_inv_sqrt(w[edge_index])
            mean = -h * sym(root @ grad[edge_index] @ root)
            tangent = sym(
                mean
                + math.sqrt(2.0 * h) * random_sym_normal(d, rng)
            )
            proposal[edge_index] = sym(
                root @ expm_sym(tangent) @ root
            )
            tangent_list.append(tangent)
            mean_list.append(mean)

        phi_prop, grad_prop = posterior.phi_and_grad(proposal)
        if not np.isfinite(phi_prop):
            return w, phi, grad, False, True

        logq_forward = 0.0
        logq_reverse = 0.0
        for edge_index in range(edge_count):
            logq_forward += (
                log_gaussian_orthosym(
                    tangent_list[edge_index],
                    mean_list[edge_index],
                    2.0 * h,
                )
                - log_spd_exp_jacobian(tangent_list[edge_index])
            )

            reverse_tangent = spd_congruence_log(
                proposal[edge_index], w[edge_index]
            )
            root_prop, _ = mat_sqrt_inv_sqrt(proposal[edge_index])
            reverse_mean = -h * sym(
                root_prop @ grad_prop[edge_index] @ root_prop
            )
            logq_reverse += (
                log_gaussian_orthosym(
                    reverse_tangent,
                    reverse_mean,
                    2.0 * h,
                )
                - log_spd_exp_jacobian(reverse_tangent)
            )

        log_alpha = (
            -phi_prop + logq_reverse
            - (-phi + logq_forward)
        )
        if math.log(rng.uniform()) < min(0.0, log_alpha):
            return proposal, phi_prop, grad_prop, True, False
        return w, phi, grad, False, False
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return w, phi, grad, False, True


def cone_mala_chain(
    posterior: PSDGraphPosterior,
    w0: Array,
    preburn: int,
    retained: int,
    h: float,
    seed: int,
) -> ChainResult:
    rng = np.random.default_rng(seed)
    w = np.asarray([project_spd(block) for block in w0])
    phi, grad = posterior.phi_and_grad(w)

    accepted_preburn = 0
    failures_preburn = 0
    total_start = time.perf_counter()

    for _ in range(preburn):
        w, phi, grad, accepted, failed = _cone_transition(
            posterior, w, phi, grad, h, rng
        )
        accepted_preburn += int(accepted)
        failures_preburn += int(failed)

    samples = DrawBuffer(retained)
    accepted_sampling = 0
    failures_sampling = 0
    sampling_start = time.perf_counter()

    for _ in range(retained):
        w, phi, grad, accepted, failed = _cone_transition(
            posterior, w, phi, grad, h, rng
        )
        accepted_sampling += int(accepted)
        failures_sampling += int(failed)
        samples.append(w.copy())

    runtime_sampling = time.perf_counter() - sampling_start
    runtime_total = time.perf_counter() - total_start
    return ChainResult(
        samples=np.asarray(samples),
        acceptance=accepted_sampling / max(retained, 1),
        preburn_acceptance=accepted_preburn / max(preburn, 1),
        runtime_total_sec=float(runtime_total),
        runtime_sampling_sec=float(runtime_sampling),
        h_final=float(h),
        invalid_rate=0.0,
        preburn_invalid_rate=0.0,
        numerical_failure_rate=(
            failures_sampling / max(retained, 1)
        ),
    )


# =============================================================================
# Fixed-step Euclidean MALA
# =============================================================================

def _euclidean_transition(
    posterior: PSDGraphPosterior,
    w: Array,
    phi: float,
    grad: Array,
    logpi: float,
    h: float,
    rng: np.random.Generator,
) -> Tuple[Array, float, Array, float, bool, bool, bool]:
    edge_count, d, _ = w.shape
    proposal = np.zeros_like(w)

    try:
        for edge_index in range(edge_count):
            x = sym_to_orthovec(w[edge_index])
            negative_log_gradient = (
                grad[edge_index]
                + posterior.volume_coefficient * inv(w[edge_index])
            )
            mean = x - h * sym_to_orthovec(negative_log_gradient)
            y = mean + math.sqrt(2.0 * h) * rng.normal(size=len(x))
            proposal[edge_index] = orthovec_to_sym(y, d)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return w, phi, grad, logpi, False, False, True

    if not all(is_spd(proposal[e]) for e in range(edge_count)):
        return w, phi, grad, logpi, False, True, False

    try:
        phi_prop, grad_prop = posterior.phi_and_grad(proposal)
        logpi_prop = posterior.log_target_dx(proposal)
        if not np.isfinite(logpi_prop):
            return w, phi, grad, logpi, False, False, True

        logq_forward = 0.0
        logq_reverse = 0.0
        for edge_index in range(edge_count):
            x = sym_to_orthovec(w[edge_index])
            y = sym_to_orthovec(proposal[edge_index])
            gx = sym_to_orthovec(
                grad[edge_index]
                + posterior.volume_coefficient * inv(w[edge_index])
            )
            gy = sym_to_orthovec(
                grad_prop[edge_index]
                + posterior.volume_coefficient * inv(proposal[edge_index])
            )
            mean_x = x - h * gx
            mean_y = y - h * gy
            p = len(x)
            logq_forward += (
                -0.5 * p * np.log(4.0 * np.pi * h)
                - np.sum((y - mean_x) ** 2) / (4.0 * h)
            )
            logq_reverse += (
                -0.5 * p * np.log(4.0 * np.pi * h)
                - np.sum((x - mean_y) ** 2) / (4.0 * h)
            )

        log_alpha = (
            logpi_prop + logq_reverse
            - logpi - logq_forward
        )
        if math.log(rng.uniform()) < min(0.0, log_alpha):
            return (
                proposal,
                phi_prop,
                grad_prop,
                logpi_prop,
                True,
                False,
                False,
            )
        return w, phi, grad, logpi, False, False, False
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return w, phi, grad, logpi, False, False, True


def euclidean_mala_chain(
    posterior: PSDGraphPosterior,
    w0: Array,
    preburn: int,
    retained: int,
    h: float,
    seed: int,
) -> ChainResult:
    rng = np.random.default_rng(seed)
    w = np.asarray([project_spd(block) for block in w0])
    phi, grad = posterior.phi_and_grad(w)
    logpi = posterior.log_target_dx(w)

    accepted_preburn = invalid_preburn = failures_preburn = 0
    total_start = time.perf_counter()

    for _ in range(preburn):
        (
            w, phi, grad, logpi,
            accepted, invalid, failed,
        ) = _euclidean_transition(
            posterior, w, phi, grad, logpi, h, rng
        )
        accepted_preburn += int(accepted)
        invalid_preburn += int(invalid)
        failures_preburn += int(failed)

    samples = DrawBuffer(retained)
    accepted_sampling = invalid_sampling = failures_sampling = 0
    sampling_start = time.perf_counter()

    for _ in range(retained):
        (
            w, phi, grad, logpi,
            accepted, invalid, failed,
        ) = _euclidean_transition(
            posterior, w, phi, grad, logpi, h, rng
        )
        accepted_sampling += int(accepted)
        invalid_sampling += int(invalid)
        failures_sampling += int(failed)
        samples.append(w.copy())

    runtime_sampling = time.perf_counter() - sampling_start
    runtime_total = time.perf_counter() - total_start
    return ChainResult(
        samples=np.asarray(samples),
        acceptance=accepted_sampling / max(retained, 1),
        preburn_acceptance=accepted_preburn / max(preburn, 1),
        runtime_total_sec=float(runtime_total),
        runtime_sampling_sec=float(runtime_sampling),
        h_final=float(h),
        invalid_rate=invalid_sampling / max(retained, 1),
        preburn_invalid_rate=invalid_preburn / max(preburn, 1),
        numerical_failure_rate=failures_sampling / max(retained, 1),
    )


# =============================================================================
# Fixed-step generic RMALA / first-order retraction
# =============================================================================

def _rmala_transition(
    posterior: PSDGraphPosterior,
    w: Array,
    phi: float,
    grad: Array,
    logpi: float,
    h: float,
    rng: np.random.Generator,
) -> Tuple[Array, float, Array, float, bool, bool, bool]:
    edge_count, d, _ = w.shape
    proposal = np.zeros_like(w)
    jacobian_coefficient = 0.5 * (d + 1.0)

    try:
        for edge_index in range(edge_count):
            we = sym(w[edge_index])
            root, _ = mat_sqrt_inv_sqrt(we)
            negative_log_gradient = (
                grad[edge_index]
                + posterior.volume_coefficient * inv(we)
            )
            mean = -h * sym(root @ negative_log_gradient @ root)
            tangent = sym(
                mean
                + math.sqrt(2.0 * h) * random_sym_normal(d, rng)
            )
            proposal[edge_index] = sym(
                we + root @ tangent @ root
            )
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return w, phi, grad, logpi, False, False, True

    if not all(is_spd(proposal[e]) for e in range(edge_count)):
        return w, phi, grad, logpi, False, True, False

    try:
        phi_prop, grad_prop = posterior.phi_and_grad(proposal)
        logpi_prop = posterior.log_target_dx(proposal)
        if not np.isfinite(logpi_prop):
            return w, phi, grad, logpi, False, False, True

        logq_forward = 0.0
        logq_reverse = 0.0
        for edge_index in range(edge_count):
            we = sym(w[edge_index])
            ye = sym(proposal[edge_index])
            root_w, inv_root_w = mat_sqrt_inv_sqrt(we)
            root_y, inv_root_y = mat_sqrt_inv_sqrt(ye)

            forward_tangent = sym(
                inv_root_w @ (ye - we) @ inv_root_w
            )
            reverse_tangent = sym(
                inv_root_y @ (we - ye) @ inv_root_y
            )

            gx = (
                grad[edge_index]
                + posterior.volume_coefficient * inv(we)
            )
            gy = (
                grad_prop[edge_index]
                + posterior.volume_coefficient * inv(ye)
            )
            mean_x = -h * sym(root_w @ gx @ root_w)
            mean_y = -h * sym(root_y @ gy @ root_y)

            logq_forward += (
                log_gaussian_orthosym(
                    forward_tangent, mean_x, 2.0 * h
                )
                - jacobian_coefficient * logdet_spd(we)
            )
            logq_reverse += (
                log_gaussian_orthosym(
                    reverse_tangent, mean_y, 2.0 * h
                )
                - jacobian_coefficient * logdet_spd(ye)
            )

        log_alpha = (
            logpi_prop + logq_reverse
            - logpi - logq_forward
        )
        if math.log(rng.uniform()) < min(0.0, log_alpha):
            return (
                proposal,
                phi_prop,
                grad_prop,
                logpi_prop,
                True,
                False,
                False,
            )
        return w, phi, grad, logpi, False, False, False
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return w, phi, grad, logpi, False, False, True


def generic_rmala_chain(
    posterior: PSDGraphPosterior,
    w0: Array,
    preburn: int,
    retained: int,
    h: float,
    seed: int,
) -> ChainResult:
    rng = np.random.default_rng(seed)
    w = np.asarray([project_spd(block) for block in w0])
    phi, grad = posterior.phi_and_grad(w)
    logpi = posterior.log_target_dx(w)

    accepted_preburn = invalid_preburn = failures_preburn = 0
    total_start = time.perf_counter()

    for _ in range(preburn):
        (
            w, phi, grad, logpi,
            accepted, invalid, failed,
        ) = _rmala_transition(
            posterior, w, phi, grad, logpi, h, rng
        )
        accepted_preburn += int(accepted)
        invalid_preburn += int(invalid)
        failures_preburn += int(failed)

    samples = DrawBuffer(retained)
    accepted_sampling = invalid_sampling = failures_sampling = 0
    sampling_start = time.perf_counter()

    for _ in range(retained):
        (
            w, phi, grad, logpi,
            accepted, invalid, failed,
        ) = _rmala_transition(
            posterior, w, phi, grad, logpi, h, rng
        )
        accepted_sampling += int(accepted)
        invalid_sampling += int(invalid)
        failures_sampling += int(failed)
        samples.append(w.copy())

    runtime_sampling = time.perf_counter() - sampling_start
    runtime_total = time.perf_counter() - total_start
    return ChainResult(
        samples=np.asarray(samples),
        acceptance=accepted_sampling / max(retained, 1),
        preburn_acceptance=accepted_preburn / max(preburn, 1),
        runtime_total_sec=float(runtime_total),
        runtime_sampling_sec=float(runtime_sampling),
        h_final=float(h),
        invalid_rate=invalid_sampling / max(retained, 1),
        preburn_invalid_rate=invalid_preburn / max(preburn, 1),
        numerical_failure_rate=failures_sampling / max(retained, 1),
    )


# =============================================================================
# Fisher-adaptive MALA: discarded adaptation -> pooled metric -> fixed-h tuning
# =============================================================================

@dataclass
class FisherAdaptationResult:
    terminal_w: Array
    root: Array
    h_final: float
    acceptance: float
    invalid_rate: float
    numerical_failure_rate: float
    runtime_sec: float
    n_updates: int


@dataclass
class FisherPlan:
    root: Array
    condition_number: float
    h_selected: float
    pilot_acceptance: float
    pilot_invalid_rate: float
    adaptation_runtime_sec: float
    tuning_runtime_sec: float
    anchors: List[Array]
    adaptation_rows: List[Dict[str, float]]
    tuning_rows: List[Dict[str, float]]
    feasible: bool
    failure_reason: str


def fisher_sqrt_initialize(signal: Array, damping: float) -> Array:
    signal = np.asarray(signal, dtype=float)
    norm_squared = float(np.dot(signal, signal))
    coefficient = 1.0 / (
        1.0 + math.sqrt(damping / (damping + norm_squared))
    )
    return (
        np.eye(signal.size)
        - coefficient * np.outer(signal, signal)
        / (damping + norm_squared)
    ) / math.sqrt(damping)


def fisher_sqrt_update(root: Array, signal: Array) -> Array:
    phi = root.T @ signal
    denominator = 1.0 + float(np.dot(phi, phi))
    coefficient = 1.0 / (
        1.0 + math.sqrt(1.0 / denominator)
    )
    return root - coefficient * np.outer(root @ phi, phi) / denominator


def normalized_fisher_action(
    root: Array,
    vector: Array,
) -> Tuple[Array, float]:
    qdim = root.shape[0]
    average_eigenvalue = float(np.sum(root * root) / qdim)
    average_eigenvalue = max(
        average_eigenvalue, np.finfo(float).tiny
    )
    action = root @ (root.T @ vector) / average_eigenvalue
    return action, average_eigenvalue


def normalize_fisher_root(
    root: Array,
    max_condition: float,
) -> Tuple[Array, float]:
    """Normalize P=root root^T to tr(P)/q=1 and cap cond(P)."""
    p_matrix = sym(root @ root.T)
    eigenvalues, eigenvectors = np.linalg.eigh(p_matrix)
    eigenvalues = np.maximum(eigenvalues, np.finfo(float).tiny)
    largest = float(np.max(eigenvalues))
    eigenvalues = np.maximum(
        eigenvalues, largest / max_condition
    )
    eigenvalues /= float(np.mean(eigenvalues))
    p_matrix = sym((eigenvectors * eigenvalues) @ eigenvectors.T)
    normalized_root = np.linalg.cholesky(p_matrix)
    condition_number = float(
        np.max(eigenvalues) / np.min(eigenvalues)
    )
    return normalized_root, condition_number


def fisher_log_proposal_ratio(
    x: Array,
    y: Array,
    score_x: Array,
    score_y: Array,
    root: Array,
    average_eigenvalue: float,
    h: float,
) -> float:
    """Return log q(x|y)-log q(y|x) for fixed Gaussian P."""
    p_score_x = root @ (root.T @ score_x) / average_eigenvalue
    p_score_y = root @ (root.T @ score_y) / average_eigenvalue
    h_xy = 0.5 * np.dot(
        x - y - 0.5 * h * p_score_y,
        score_y,
    )
    h_yx = 0.5 * np.dot(
        y - x - 0.5 * h * p_score_x,
        score_x,
    )
    return float(h_xy - h_yx)


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


def _fixed_fisher_transition(
    posterior: PSDGraphPosterior,
    w: Array,
    phi: float,
    grad: Array,
    logpi: float,
    x: Array,
    root: Array,
    h: float,
    rng: np.random.Generator,
) -> Tuple[Array, float, Array, float, Array, bool, bool, bool]:
    """One exact fixed-preconditioner Fisher-MALA transition."""
    score_x = posterior.euclidean_score_dx(w, grad)
    p_score_x = root @ (root.T @ score_x)
    mean_x = x + h * p_score_x
    y = mean_x + math.sqrt(2.0 * h) * (
        root @ rng.standard_normal(x.size)
    )
    proposal = unflatten_weights(
        y, posterior.edge_count, posterior.d
    )

    if not all(
        is_spd(proposal[e])
        for e in range(posterior.edge_count)
    ):
        return w, phi, grad, logpi, x, False, True, False

    try:
        phi_prop, grad_prop = posterior.phi_and_grad(proposal)
        logpi_prop = posterior.log_target_dx(proposal)
        if not np.isfinite(logpi_prop):
            return w, phi, grad, logpi, x, False, False, True
        score_y = posterior.euclidean_score_dx(proposal, grad_prop)
        log_q_ratio = fisher_log_proposal_ratio(
            x=x,
            y=y,
            score_x=score_x,
            score_y=score_y,
            root=root,
            average_eigenvalue=1.0,
            h=h,
        )
        log_alpha = logpi_prop - logpi + log_q_ratio
        if math.log(rng.uniform()) < min(0.0, log_alpha):
            return (
                proposal,
                phi_prop,
                grad_prop,
                logpi_prop,
                y,
                True,
                False,
                False,
            )
        return w, phi, grad, logpi, x, False, False, False
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return w, phi, grad, logpi, x, False, False, True


def fixed_fisher_mala_chain(
    posterior: PSDGraphPosterior,
    w0: Array,
    warmup: int,
    retained: int,
    h: float,
    root: Array,
    condition_number: float,
    seed: int,
) -> ChainResult:
    """Fixed P and fixed h. Warmup is ordinary discarded settling."""
    rng = np.random.default_rng(seed)
    # IMPORTANT: w0 may be a post-adaptation anchor.  Do not project it
    # here: projection is not an MH transition and can place an eigenvalue
    # exactly on the artificial floor, which makes every additive proposal
    # look infeasible.  Validate and copy the anchor unchanged instead.
    w = np.asarray([sym(block).copy() for block in w0])
    if not all(is_spd(block) for block in w):
        raise ValueError(
            "Fixed Fisher chain received a non-SPD anchor. "
            "The adaptive pilot must return a strictly SPD state."
        )
    phi, grad = posterior.phi_and_grad(w)
    logpi = posterior.log_target_dx(w)
    x = flatten_weights(w)

    accepted_warm = invalid_warm = failures_warm = 0
    total_start = time.perf_counter()
    for _ in range(warmup):
        (
            w, phi, grad, logpi, x,
            accepted, invalid, failed,
        ) = _fixed_fisher_transition(
            posterior, w, phi, grad, logpi, x, root, h, rng
        )
        accepted_warm += int(accepted)
        invalid_warm += int(invalid)
        failures_warm += int(failed)

    samples = DrawBuffer(retained)
    accepted_keep = invalid_keep = failures_keep = 0
    sampling_start = time.perf_counter()
    for _ in range(retained):
        (
            w, phi, grad, logpi, x,
            accepted, invalid, failed,
        ) = _fixed_fisher_transition(
            posterior, w, phi, grad, logpi, x, root, h, rng
        )
        accepted_keep += int(accepted)
        invalid_keep += int(invalid)
        failures_keep += int(failed)
        samples.append(w.copy())

    runtime_sampling = time.perf_counter() - sampling_start
    runtime_total = time.perf_counter() - total_start
    return ChainResult(
        samples=np.asarray(samples),
        acceptance=accepted_keep / max(retained, 1),
        preburn_acceptance=accepted_warm / max(warmup, 1),
        runtime_total_sec=float(runtime_total),
        runtime_sampling_sec=float(runtime_sampling),
        h_final=float(h),
        invalid_rate=invalid_keep / max(retained, 1),
        preburn_invalid_rate=invalid_warm / max(warmup, 1),
        numerical_failure_rate=failures_keep / max(retained, 1),
        preconditioner_condition=float(condition_number),
    )


def adapt_single_fisher_chain(
    posterior: PSDGraphPosterior,
    w0: Array,
    adaptive_steps: int,
    individual_settle_steps: int,
    h0: float,
    seed: int,
    target_accept: float,
    damping: float,
    init_steps: int,
    update_interval: int,
    max_condition: float,
    adapt_rate: float,
    adapt_decay: float,
    adapt_exponent: float,
    h_min: float,
    h_max: float,
) -> FisherAdaptationResult:
    """Discarded chain-specific adaptation followed by fixed settling."""
    if adaptive_steps < 1:
        raise ValueError("adaptive_steps must be positive.")
    if update_interval < 1:
        raise ValueError("update_interval must be positive.")
    if not 0.5 < adapt_exponent <= 1.0:
        raise ValueError("adapt_exponent must lie in (0.5, 1].")

    rng = np.random.default_rng(seed)
    w = np.asarray([project_spd(block) for block in w0])
    phi, grad = posterior.phi_and_grad(w)
    logpi = posterior.log_target_dx(w)
    x = flatten_weights(w)
    qdim = x.size

    root = np.eye(qdim)
    initialized = False
    log_h = float(np.clip(math.log(h0), math.log(h_min), math.log(h_max)))
    accepted = invalid = failures = updates = 0
    start = time.perf_counter()

    for iteration in range(adaptive_steps):
        h_now = float(math.exp(log_h))
        score_x = posterior.euclidean_score_dx(w, grad)
        p_score_x, average_eigenvalue = normalized_fisher_action(
            root, score_x
        )
        mean_x = x + h_now * p_score_x
        y = mean_x + (
            math.sqrt(2.0 * h_now / average_eigenvalue)
            * (root @ rng.standard_normal(qdim))
        )
        proposal = unflatten_weights(
            y, posterior.edge_count, posterior.d
        )

        alpha_probability = 0.0
        accepted_now = False
        score_y: Optional[Array] = None

        if not all(
            is_spd(proposal[e])
            for e in range(posterior.edge_count)
        ):
            invalid += 1
        else:
            try:
                phi_prop, grad_prop = posterior.phi_and_grad(proposal)
                logpi_prop = posterior.log_target_dx(proposal)
                score_y = posterior.euclidean_score_dx(
                    proposal, grad_prop
                )
                log_q_ratio = fisher_log_proposal_ratio(
                    x=x,
                    y=y,
                    score_x=score_x,
                    score_y=score_y,
                    root=root,
                    average_eigenvalue=average_eigenvalue,
                    h=h_now,
                )
                log_alpha = logpi_prop - logpi + log_q_ratio
                alpha_probability = float(
                    math.exp(min(0.0, log_alpha))
                )
                if rng.uniform() < alpha_probability:
                    x = y
                    w = proposal
                    phi = phi_prop
                    grad = grad_prop
                    logpi = logpi_prop
                    accepted_now = True
            except (
                np.linalg.LinAlgError,
                FloatingPointError,
                OverflowError,
                ValueError,
            ):
                failures += 1
                score_y = None
                alpha_probability = 0.0

        accepted += int(accepted_now)

        use_signal = (
            iteration >= init_steps
            and (iteration - init_steps) % update_interval == 0
            and score_y is not None
            and alpha_probability > 0.0
        )
        if use_signal:
            signal = math.sqrt(alpha_probability) * (score_y - score_x)
            if np.all(np.isfinite(signal)) and np.linalg.norm(signal) > 0.0:
                if not initialized:
                    root = fisher_sqrt_initialize(signal, damping)
                    initialized = True
                else:
                    root = fisher_sqrt_update(root, signal)
                updates += 1

        eta = diminishing_rate(
            iteration,
            initial_rate=adapt_rate,
            decay=adapt_decay,
            exponent=adapt_exponent,
        )
        log_h += eta * (alpha_probability - target_accept)
        log_h = float(
            np.clip(log_h, math.log(h_min), math.log(h_max))
        )

    root, condition_number = normalize_fisher_root(
        root, max_condition
    )
    h_final = float(math.exp(log_h))

    # Individual fixed settling keeps the terminal anchor in the state region
    # reached by its adapted pilot while all adaptation is already frozen.
    if individual_settle_steps > 0:
        settled = fixed_fisher_mala_chain(
            posterior=posterior,
            w0=w,
            warmup=0,
            retained=individual_settle_steps,
            h=h_final,
            root=root,
            condition_number=condition_number,
            seed=seed + 500_000,
        )
        w = settled.samples[-1].copy()

    return FisherAdaptationResult(
        terminal_w=w,
        root=root,
        h_final=h_final,
        acceptance=accepted / max(adaptive_steps, 1),
        invalid_rate=invalid / max(adaptive_steps, 1),
        numerical_failure_rate=failures / max(adaptive_steps, 1),
        runtime_sec=float(time.perf_counter() - start),
        n_updates=updates,
    )


def pool_fisher_preconditioners(
    results: Sequence[FisherAdaptationResult],
    max_condition: float,
) -> Tuple[Array, float, float]:
    if not results:
        raise ValueError("At least one Fisher adaptation result is required.")
    matrices = [sym(result.root @ result.root.T) for result in results]
    p_shared = sym(np.mean(np.stack(matrices), axis=0))
    eigenvalues, eigenvectors = np.linalg.eigh(p_shared)
    eigenvalues = np.maximum(eigenvalues, np.finfo(float).tiny)
    largest = float(np.max(eigenvalues))
    eigenvalues = np.maximum(
        eigenvalues, largest / max_condition
    )
    eigenvalues /= float(np.mean(eigenvalues))
    p_shared = sym((eigenvectors * eigenvalues) @ eigenvectors.T)
    root_shared = np.linalg.cholesky(p_shared)
    condition_number = float(
        np.max(eigenvalues) / np.min(eigenvalues)
    )
    h_center = float(np.median([result.h_final for result in results]))
    return root_shared, condition_number, h_center


def fisher_standardized_esjd(samples: Array, root: Array) -> float:
    if samples.shape[0] < 2:
        return 0.0
    flat = np.stack([flatten_weights(w) for w in samples])
    deltas = np.diff(flat, axis=0)
    values: List[float] = []
    for delta in deltas:
        whitened = np.linalg.solve(root, delta)
        values.append(float(np.dot(whitened, whitened)))
    return float(np.mean(values))


def tune_fixed_fisher_h(
    posterior: PSDGraphPosterior,
    anchors: Sequence[Array],
    root: Array,
    condition_number: float,
    h_center: float,
    target_accept: float,
    acceptance_tolerance: float,
    tune_warmup: int,
    tune_steps: int,
    tune_chains: int,
    factors: Sequence[float],
    max_bracket_steps: int,
    bisection_steps: int,
    seed_base: int,
) -> Tuple[float, float, float, List[Dict[str, float]]]:
    """Freeze P, bracket h on the typical set, maximize standardized ESJD."""
    if tune_steps < 10:
        raise ValueError("tune_steps must be at least 10.")
    use_anchors = list(anchors[: min(tune_chains, len(anchors))])
    if not use_anchors:
        raise ValueError("At least one Fisher tuning anchor is required.")

    tested: Dict[str, Dict[str, float]] = {}

    def evaluate_candidate(h: float) -> Dict[str, float]:
        h = float(max(h, 1e-14))
        key = f"{h:.17e}"
        if key in tested:
            return tested[key]
        acceptances: List[float] = []
        invalid_rates: List[float] = []
        esjd_values: List[float] = []
        runtimes: List[float] = []
        for chain_index, anchor in enumerate(use_anchors):
            result = fixed_fisher_mala_chain(
                posterior=posterior,
                w0=anchor,
                warmup=tune_warmup,
                retained=tune_steps,
                h=h,
                root=root,
                condition_number=condition_number,
                seed=seed_base + chain_index,
            )
            acceptances.append(result.acceptance)
            invalid_rates.append(result.invalid_rate)
            esjd_values.append(
                fisher_standardized_esjd(result.samples, root)
            )
            runtimes.append(result.runtime_total_sec)
        row = {
            "h": h,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1)
                if len(acceptances) > 1 else 0.0
            ),
            "pilot_invalid_rate": float(np.mean(invalid_rates)),
            "standardized_esjd_per_iteration": float(
                np.mean(esjd_values)
            ),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[key] = row
        print(
            f"    Fisher tuning h={h:.3e}: "
            f"acc={row['pilot_acceptance']:.3f}, "
            f"invalid={row['pilot_invalid_rate']:.3f}, "
            f"std-ESJD={row['standardized_esjd_per_iteration']:.3e}",
            flush=True,
        )
        return row

    for factor in sorted(set(float(f) for f in factors if f > 0.0)):
        evaluate_candidate(max(h_center, 1e-12) * factor)

    lower = target_accept - acceptance_tolerance
    upper = target_accept + acceptance_tolerance

    # If every candidate is too aggressive, keep halving until feasible.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] >= lower for row in rows):
            break
        evaluate_candidate(rows[0]["h"] / 2.0)

    # If every candidate is too conservative, keep doubling.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] <= upper for row in rows):
            break
        evaluate_candidate(rows[-1]["h"] * 2.0)

    for _ in range(bisection_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        high_accept = [
            row for row in rows
            if row["pilot_acceptance"] >= target_accept
        ]
        low_accept = [
            row for row in rows
            if row["pilot_acceptance"] <= target_accept
        ]
        if not high_accept or not low_accept:
            break
        left = max(high_accept, key=lambda row: row["h"])
        right = min(low_accept, key=lambda row: row["h"])
        if left["h"] >= right["h"]:
            break
        midpoint = math.sqrt(left["h"] * right["h"])
        midpoint_key = f"{midpoint:.17e}"
        if midpoint_key in tested:
            break
        evaluate_candidate(midpoint)

    rows = sorted(tested.values(), key=lambda row: row["h"])
    admissible = [
        row for row in rows
        if lower <= row["pilot_acceptance"] <= upper
        and row["pilot_invalid_rate"] < 0.95
    ]
    if admissible:
        best = max(
            admissible,
            key=lambda row: (
                row["standardized_esjd_per_iteration"],
                -abs(row["pilot_acceptance"] - target_accept),
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                abs(row["pilot_acceptance"] - target_accept),
                row["pilot_invalid_rate"],
                -row["standardized_esjd_per_iteration"],
            ),
        )

    feasible = not (
        best["pilot_acceptance"] <= 0.01
        or best["pilot_invalid_rate"] >= 0.99
    )

    for row in rows:
        row["selected"] = bool(
            feasible
            and math.isclose(
                row["h"], best["h"], rel_tol=1e-12, abs_tol=1e-15
            )
        )
        row["tuning_feasible"] = bool(feasible)

    if not feasible:
        # Preserve all tuning diagnostics and let the experiment continue.
        # Returning NaN prevents accidental Fisher production sampling.
        return (
            float("nan"),
            float(best["pilot_acceptance"]),
            float(best["pilot_invalid_rate"]),
            rows,
        )

    total_runtime = float(sum(row["pilot_runtime_sec"] for row in rows))
    return (
        float(best["h"]),
        float(best["pilot_acceptance"]),
        float(best["pilot_invalid_rate"]),
        rows,
    )


def prepare_fisher_plan(
    posterior: PSDGraphPosterior,
    initial_states: Sequence[Array],
    adaptive_steps: int,
    individual_settle_steps: int,
    h0: float,
    target_accept: float,
    damping: float,
    init_steps: int,
    update_interval: int,
    max_condition: float,
    adapt_rate: float,
    adapt_decay: float,
    adapt_exponent: float,
    h_min: float,
    h_max: float,
    tune_warmup: int,
    tune_steps: int,
    tune_chains: int,
    acceptance_tolerance: float,
    factors: Sequence[float],
    max_bracket_steps: int,
    bisection_steps: int,
    seed_base: int,
) -> FisherPlan:
    results: List[FisherAdaptationResult] = []
    adaptation_rows: List[Dict[str, float]] = []
    adaptation_runtime = 0.0

    for chain_index, w0 in enumerate(initial_states):
        result = adapt_single_fisher_chain(
            posterior=posterior,
            w0=w0,
            adaptive_steps=adaptive_steps,
            individual_settle_steps=individual_settle_steps,
            h0=h0,
            seed=seed_base + chain_index,
            target_accept=target_accept,
            damping=damping,
            init_steps=min(init_steps, max(adaptive_steps - 1, 0)),
            update_interval=update_interval,
            max_condition=max_condition,
            adapt_rate=adapt_rate,
            adapt_decay=adapt_decay,
            adapt_exponent=adapt_exponent,
            h_min=h_min,
            h_max=h_max,
        )
        results.append(result)
        adaptation_runtime += result.runtime_sec
        terminal_min_eigenvalue = float(
            min(
                np.min(np.linalg.eigvalsh(sym(block)))
                for block in result.terminal_w
            )
        )
        terminal_max_condition = float(
            max(
                np.linalg.cond(sym(block))
                for block in result.terminal_w
            )
        )
        adaptation_rows.append({
            "chain": chain_index + 1,
            "h_final": result.h_final,
            "acceptance": result.acceptance,
            "invalid_rate": result.invalid_rate,
            "numerical_failure_rate": result.numerical_failure_rate,
            "runtime_sec": result.runtime_sec,
            "n_updates": result.n_updates,
            "terminal_min_edge_eigenvalue": terminal_min_eigenvalue,
            "terminal_max_edge_condition": terminal_max_condition,
        })
        print(
            f"  Fisher pilot chain {chain_index + 1}: "
            f"h={result.h_final:.3e}, "
            f"acc={result.acceptance:.3f}, "
            f"invalid={result.invalid_rate:.3f}, "
            f"min eig={terminal_min_eigenvalue:.3e}, "
            f"updates={result.n_updates}",
            flush=True,
        )

    root, condition_number, h_center = pool_fisher_preconditioners(
        results, max_condition
    )
    anchors = [result.terminal_w for result in results]
    (
        h_selected,
        pilot_acceptance,
        pilot_invalid_rate,
        tuning_rows,
    ) = tune_fixed_fisher_h(
        posterior=posterior,
        anchors=anchors,
        root=root,
        condition_number=condition_number,
        h_center=h_center,
        target_accept=target_accept,
        acceptance_tolerance=acceptance_tolerance,
        tune_warmup=tune_warmup,
        tune_steps=tune_steps,
        tune_chains=tune_chains,
        factors=factors,
        max_bracket_steps=max_bracket_steps,
        bisection_steps=bisection_steps,
        seed_base=seed_base + 1_000_000,
    )
    tuning_runtime = float(
        sum(row["pilot_runtime_sec"] for row in tuning_rows)
    )
    feasible = bool(np.isfinite(h_selected))
    failure_reason = (
        ""
        if feasible
        else (
            "No fixed pooled-Fisher kernel achieved both nonzero pilot "
            "acceptance and an invalid-SPD rate below 0.99."
        )
    )
    return FisherPlan(
        root=root,
        condition_number=condition_number,
        h_selected=h_selected,
        pilot_acceptance=pilot_acceptance,
        pilot_invalid_rate=pilot_invalid_rate,
        adaptation_runtime_sec=adaptation_runtime,
        tuning_runtime_sec=tuning_runtime,
        anchors=anchors,
        adaptation_rows=adaptation_rows,
        tuning_rows=tuning_rows,
        feasible=feasible,
        failure_reason=failure_reason,
    )


# =============================================================================
# Diagnostics
# =============================================================================

def ess_1d(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return float(n)
    centered = x - np.mean(x)
    variance = np.dot(centered, centered) / n
    if variance <= 1e-14:
        # A constant chain has no demonstrated Monte Carlo movement.
        return 0.0

    max_lag = min(n // 2, 1000)
    rho_sum = 0.0
    for lag in range(1, max_lag):
        autocovariance = (
            np.dot(centered[:-lag], centered[lag:])
            / (n - lag)
        )
        rho = autocovariance / variance
        if rho <= 0.0:
            break
        rho_sum += rho
    return float(n / (1.0 + 2.0 * rho_sum))


def split_rhat(chains_1d: Sequence[Array]) -> float:
    chains = [np.asarray(chain, dtype=float) for chain in chains_1d]
    minimum_length = min(len(chain) for chain in chains)
    minimum_length = 2 * (minimum_length // 2)
    if minimum_length < 20 or len(chains) < 2:
        return np.nan

    array = np.asarray(
        [chain[:minimum_length] for chain in chains]
    )
    split: List[Array] = []
    for chain in array:
        half = minimum_length // 2
        split.extend([chain[:half], chain[half:]])
    split_array = np.asarray(split)

    n = split_array.shape[1]
    chain_means = split_array.mean(axis=1)
    chain_variances = split_array.var(axis=1, ddof=1)
    within = chain_variances.mean()
    between = n * chain_means.var(ddof=1)
    variance_hat = (
        ((n - 1.0) / n) * within + between / n
    )
    if within <= 1e-15:
        # Distinct constant chains imply complete nonconvergence; identical
        # constant chains do not provide enough information to assess Rhat.
        return float("inf") if between > 1e-15 else np.nan
    return float(np.sqrt(variance_hat / within))


def observables(
    samples: Array,
    posterior: PSDGraphPosterior,
    w_star: Array,
    y_test: Array,
) -> Dict[str, Array]:
    result: Dict[str, List[float]] = {
        "logdet_Q": [],
        "lambda_min_Q": [],
        "Phi": [],
        "rel_W_error": [],
        "test_nll": [],
    }
    n_test = y_test.shape[1]
    s_test = y_test @ y_test.T
    denominator = math.sqrt(float(np.sum(w_star * w_star)))

    for w in samples:
        q_matrix = posterior.q(w)
        phi, _ = posterior.phi_and_grad(w)
        result["logdet_Q"].append(logdet_spd(q_matrix))
        result["lambda_min_Q"].append(
            float(np.min(eigvalsh(q_matrix)))
        )
        result["Phi"].append(float(phi))
        result["rel_W_error"].append(
            math.sqrt(float(np.sum((w - w_star) ** 2)))
            / denominator
        )
        nll = (
            -0.5 * n_test * logdet_spd(q_matrix)
            + 0.5 * np.trace(q_matrix @ s_test)
        )
        result["test_nll"].append(float(nll / n_test))

    return {
        key: np.asarray(values, dtype=float)
        for key, values in result.items()
    }


def posterior_recovery_and_coverage(
    sample_chains: Sequence[Array],
    w_star: Array,
    credible_level: float,
    method: str,
    dataset_seed: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute graph recovery and marginal equal-tailed interval coverage.

    Entry coverage uses only the unique symmetric coordinates ``i <= j``.
    Trace coverage uses one scalar trace per edge.  Eigenvalue coverage uses
    eigenvalues ordered increasingly within every retained draw and within the
    truth.  All retained draws from all chains are pooled with equal weight.

    Coverage from a nonconverged sampler is still computed as a descriptive
    quantity; its inferential interpretation must be qualified by split-Rhat.
    """
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must lie strictly between 0 and 1.")
    if not sample_chains:
        raise ValueError("At least one retained sample chain is required.")

    n_total = int(sum(chain.shape[0] for chain in sample_chains))
    if n_total < 2:
        raise ValueError("At least two pooled retained draws are required.")

    edge_count, d, _ = w_star.shape
    for chain in sample_chains:
        if chain.ndim != 4 or chain.shape[1:] != w_star.shape:
            raise ValueError(
                "Each sample chain must have shape (draw, edge, d, d) "
                f"with trailing shape {w_star.shape}; got {chain.shape}."
            )

    posterior_sum = np.zeros_like(w_star, dtype=float)
    for chain in sample_chains:
        posterior_sum += np.sum(chain, axis=0)
    posterior_mean = posterior_sum / float(n_total)
    truth_norm = max(float(np.linalg.norm(w_star)), np.finfo(float).tiny)
    posterior_mean_rel_error = float(
        np.linalg.norm(posterior_mean - w_star) / truth_norm
    )

    tail = 0.5 * (1.0 - credible_level)
    quantiles = [tail, 1.0 - tail]
    tri_i, tri_j = np.triu_indices(d)
    detail_rows: List[Dict[str, object]] = []

    entry_covered = 0
    entry_total = 0
    entry_widths: List[float] = []
    trace_covered = 0
    trace_total = 0
    trace_widths: List[float] = []
    eigen_covered = 0
    eigen_total = 0
    eigen_widths: List[float] = []

    for edge_index in range(edge_count):
        # This edge-wise pooling avoids forming a second full 4-chain tensor.
        edge_samples = np.concatenate(
            [chain[:, edge_index, :, :] for chain in sample_chains],
            axis=0,
        )

        entry_values = edge_samples[:, tri_i, tri_j]
        entry_bounds = np.quantile(entry_values, quantiles, axis=0)
        entry_truth = w_star[edge_index, tri_i, tri_j]
        for coordinate_index, (i, j) in enumerate(zip(tri_i, tri_j)):
            lower = float(entry_bounds[0, coordinate_index])
            upper = float(entry_bounds[1, coordinate_index])
            truth = float(entry_truth[coordinate_index])
            covered = bool(lower <= truth <= upper)
            width = upper - lower
            entry_covered += int(covered)
            entry_total += 1
            entry_widths.append(width)
            detail_rows.append({
                "dataset_seed": int(dataset_seed),
                "Method": method,
                "credible_level": float(credible_level),
                "quantity": "entry",
                "edge": int(edge_index),
                "row": int(i),
                "column": int(j),
                "eigen_index": np.nan,
                "truth": truth,
                "lower": lower,
                "upper": upper,
                "width": width,
                "covered": covered,
            })

        trace_values = np.trace(edge_samples, axis1=1, axis2=2)
        trace_lower, trace_upper = np.quantile(trace_values, quantiles)
        trace_truth = float(np.trace(w_star[edge_index]))
        trace_is_covered = bool(trace_lower <= trace_truth <= trace_upper)
        trace_width = float(trace_upper - trace_lower)
        trace_covered += int(trace_is_covered)
        trace_total += 1
        trace_widths.append(trace_width)
        detail_rows.append({
            "dataset_seed": int(dataset_seed),
            "Method": method,
            "credible_level": float(credible_level),
            "quantity": "trace",
            "edge": int(edge_index),
            "row": np.nan,
            "column": np.nan,
            "eigen_index": np.nan,
            "truth": trace_truth,
            "lower": float(trace_lower),
            "upper": float(trace_upper),
            "width": trace_width,
            "covered": trace_is_covered,
        })

        eigen_values = np.linalg.eigvalsh(edge_samples)
        eigen_bounds = np.quantile(eigen_values, quantiles, axis=0)
        eigen_truth = np.linalg.eigvalsh(sym(w_star[edge_index]))
        for eigen_index in range(d):
            lower = float(eigen_bounds[0, eigen_index])
            upper = float(eigen_bounds[1, eigen_index])
            truth = float(eigen_truth[eigen_index])
            covered = bool(lower <= truth <= upper)
            width = upper - lower
            eigen_covered += int(covered)
            eigen_total += 1
            eigen_widths.append(width)
            detail_rows.append({
                "dataset_seed": int(dataset_seed),
                "Method": method,
                "credible_level": float(credible_level),
                "quantity": "eigenvalue",
                "edge": int(edge_index),
                "row": np.nan,
                "column": np.nan,
                # One-based index in increasing eigenvalue order.
                "eigen_index": int(eigen_index + 1),
                "truth": truth,
                "lower": lower,
                "upper": upper,
                "width": width,
                "covered": covered,
            })

    summary = {
        "posterior_mean_rel_W_error": posterior_mean_rel_error,
        "credible_level": float(credible_level),
        "entry_coverage": entry_covered / max(entry_total, 1),
        "entry_covered_count": int(entry_covered),
        "entry_interval_count": int(entry_total),
        "entry_interval_width_mean": float(np.mean(entry_widths)),
        "entry_interval_width_median": float(np.median(entry_widths)),
        "trace_coverage": trace_covered / max(trace_total, 1),
        "trace_covered_count": int(trace_covered),
        "trace_interval_count": int(trace_total),
        "trace_interval_width_mean": float(np.mean(trace_widths)),
        "trace_interval_width_median": float(np.median(trace_widths)),
        "eigenvalue_coverage": eigen_covered / max(eigen_total, 1),
        "eigenvalue_covered_count": int(eigen_covered),
        "eigenvalue_interval_count": int(eigen_total),
        "eigenvalue_interval_width_mean": float(np.mean(eigen_widths)),
        "eigenvalue_interval_width_median": float(np.median(eigen_widths)),
    }
    return summary, pd.DataFrame(detail_rows)


# =============================================================================
# Experiment settings, tables, and plotting
# =============================================================================

PAPER_SETTINGS: Dict[int, Dict[str, float]] = {
    20: {
        "preburn": 4000,
        "retained": 12000,
        "h_cone": 8.0e-3,
        "h_euclid": 4.0e-5,
        "h_rmala": 3.0e-4,
        "h_fisher": 1.0e-6,
    },
    50: {
        "preburn": 4000,
        "retained": 12000,
        "h_cone": 6.0e-3,
        "h_euclid": 2.0e-5,
        "h_rmala": 1.0e-4,
        "h_fisher": 5.0e-7,
    },
    100: {
        "preburn": 2000,
        "retained": 6000,
        "h_cone": 4.0e-3,
        "h_euclid": 2.0e-5,
        "h_rmala": 5.0e-5,
        "h_fisher": 2.5e-7,
    },
}


METHOD_ORDER = [
    "ConeMALA",
    "FisherAdaptiveMALA",
    "EuclideanMALA",
    "GenericRMALA",
]


METHOD_LABELS = {
    "ConeMALA": "ConeMALA",
    "FisherAdaptiveMALA": "Fisher-adaptive MALA",
    "EuclideanMALA": "Euclidean MALA",
    "GenericRMALA": "Generic RMALA",
}


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def make_initial_w(
    edge_count: int,
    d: int,
    chain_index: int,
) -> Array:
    rng = np.random.default_rng(1000 + chain_index)
    return np.asarray(
        [
            project_spd(
                0.35 * np.eye(d)
                + 0.03 * sym(rng.normal(size=(d, d)))
            )
            for _ in range(edge_count)
        ]
    )


def resolve_settings(args: argparse.Namespace, m: int) -> Dict[str, float]:
    if m not in PAPER_SETTINGS:
        raise ValueError(
            f"No paper default is registered for m={m}. "
            "Use m in {20,50,100} or extend PAPER_SETTINGS."
        )
    settings = dict(PAPER_SETTINGS[m])
    for name in (
        "preburn",
        "retained",
        "h_cone",
        "h_euclid",
        "h_rmala",
        "h_fisher",
    ):
        override = getattr(args, name)
        if override is not None:
            settings[name] = override
    return settings


def latex_method(method: str) -> str:
    return {
        "ConeMALA": r"\texttt{cone\_geom\_mala}",
        "FisherAdaptiveMALA": r"\texttt{FisherAdaptiveMALA}",
        "EuclideanMALA": r"\texttt{Euclidean\_MALA}",
        "GenericRMALA": r"\texttt{generic\_RMALA}",
    }[method]


def latex_value(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return r"\texttt{TF}"
    return f"{float(value):.{digits}f}"


def write_latex_table(
    summary: pd.DataFrame,
    path: Path,
    m: int,
    d: int,
    preburn: int,
    retained: int,
    fisher_update_interval: int,
    fisher_adaptive_steps: int,
    fisher_individual_settle: int,
    fisher_final_settle: int,
) -> None:
    indexed = summary.set_index("Method").loc[METHOD_ORDER]
    rows: List[str] = []
    for method in METHOD_ORDER:
        row = indexed.loc[method]
        rows.append(
            f"{latex_method(method)}"
            f" & {latex_value(row['accept_mean'], 3)}"
            f" & {latex_value(row['rel_W_error_ESSsec_total'], 2)}"
            f" & {latex_value(row['test_nll_ESSsec_total'], 2)}"
            f" & {latex_value(row['rel_W_error_Rhat'], 3)}"
            f" & {latex_value(row['test_nll_Rhat'], 3)}"
            f" & {latex_value(row['logdet_Q_Rhat'], 3)}"
            f" & {latex_value(row['Phi_Rhat'], 3)}"
            f" & {latex_value(row['invalid_rate_mean'], 3)}"
            r" \\"
        )

    content = rf"""
\begin{{table*}}[t]
\centering
\caption{{Sampler performance for the $d={d}$, $m={m}$ graph posterior.
ConeMALA, Euclidean MALA, and generic RMALA use {preburn} discarded fixed-kernel burn-in transitions and {retained} retained transitions per chain. Fisher-adaptive MALA uses {fisher_adaptive_steps} discarded adaptive transitions, {fisher_individual_settle} discarded chain-specific fixed-kernel settling transitions, pooled-preconditioner fixed-$h$ tuning, {fisher_final_settle} discarded shared-kernel settling transitions, and {retained} retained transitions. Its preconditioner update interval is {fisher_update_interval}. ESS/sec includes all adaptation, tuning, settling, and production costs.
\texttt{{TF}} denotes Fisher fixed-kernel tuning failure; no production
inference is reported for that method.}}
\label{{tab:d5-m{m}-with-fisher}}
\footnotesize
\begin{{tabular}}{{lcccccccc}}
\toprule
Method
& Mean Acc.
& Rel.\ $W$ ESS/sec
& NLL ESS/sec
& $\widehat R_W$
& $\widehat R_{{\rm NLL}}$
& $\widehat R_{{\log\det}}$
& $\widehat R_\Phi$
& Invalid \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip() + "\n"
    path.write_text(content, encoding="utf-8")


def ecdf_values(x: Array) -> Tuple[Array, Array]:
    sorted_x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(sorted_x) + 1) / len(sorted_x)
    return sorted_x, y


def set_large_plot_fonts() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_two_panel_ecdf(
    plot_data: Dict[str, List[pd.DataFrame]],
    outdir: Path,
    x_left: str,
    x_right: str,
    filename: str,
    title_left: str,
    title_right: str,
) -> None:
    import matplotlib.pyplot as plt

    set_large_plot_fonts()
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.0), constrained_layout=True
    )
    for method in METHOD_ORDER:
        if method not in plot_data or not plot_data[method]:
            continue
        frames = plot_data[method]
        left_values = np.concatenate(
            [frame[x_left].dropna().to_numpy() for frame in frames]
        )
        right_values = np.concatenate(
            [frame[x_right].dropna().to_numpy() for frame in frames]
        )
        xl, yl = ecdf_values(left_values)
        xr, yr = ecdf_values(right_values)
        axes[0].plot(
            xl, yl, linewidth=2.2, label=METHOD_LABELS[method]
        )
        axes[1].plot(
            xr, yr, linewidth=2.2, label=METHOD_LABELS[method]
        )

    axes[0].set_title(title_left)
    axes[1].set_title(title_right)
    axes[0].set_xlabel(x_left.replace("_", " "))
    axes[1].set_xlabel(x_right.replace("_", " "))
    axes[0].set_ylabel("ECDF")
    axes[1].set_ylabel("ECDF")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.savefig(
        plot_dir / f"{filename}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        plot_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_acceptance_plot(
    summary: pd.DataFrame,
    outdir: Path,
    m: int,
) -> None:
    import matplotlib.pyplot as plt

    set_large_plot_fonts()
    ordered = summary.set_index("Method").loc[METHOD_ORDER].reset_index()
    ordered = ordered[np.isfinite(ordered["accept_mean"])].copy()
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    x = np.arange(len(ordered))
    bars = axis.bar(
        x,
        ordered["accept_mean"],
        yerr=ordered["accept_sd"],
        capsize=5,
    )
    axis.set_xticks(
        x,
        [METHOD_LABELS[value] for value in ordered["Method"]],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("Mean production acceptance")
    axis.set_title(rf"Acceptance rates, $d=5$, $m={m}$")
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, ordered["accept_mean"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    figure.tight_layout()
    figure.savefig(
        outdir / "plots" / f"acceptance_d5_m{m}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        outdir / "plots" / f"acceptance_d5_m{m}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_scaling_plots(
    combined: pd.DataFrame,
    outdir: Path,
) -> None:
    import matplotlib.pyplot as plt

    set_large_plot_fonts()
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2), constrained_layout=True
    )
    for method in METHOD_ORDER:
        subset = (
            combined[combined["Method"] == method]
            .sort_values("m")
        )
        subset = subset[
            np.isfinite(subset["rel_W_error_ESSsec_total"])
            & np.isfinite(subset["Phi_Rhat"])
        ]
        if subset.empty:
            continue
        axes[0].plot(
            subset["m"],
            subset["rel_W_error_ESSsec_total"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            subset["m"],
            subset["Phi_Rhat"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )

    axes[0].set_title("(a) Sampling efficiency")
    axes[0].set_xlabel(r"Graph size $m$")
    axes[0].set_ylabel(r"Relative-$W$ ESS/sec")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title(r"(b) Energy split-$\widehat R$")
    axes[1].set_xlabel(r"Graph size $m$")
    axes[1].set_ylabel(r"Split-$\widehat R_\Phi$")
    axes[1].axhline(
        1.05, linestyle="--", linewidth=1.3,
        label=r"$\widehat R=1.05$",
    )
    axes[1].axhline(
        1.10, linestyle=":", linewidth=1.3,
        label=r"$\widehat R=1.10$",
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    available_m = sorted(combined["m"].unique())
    for axis in axes:
        axis.set_xticks(available_m)

    figure.savefig(
        plot_dir / "scaling_d5_with_fisher.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        plot_dir / "scaling_d5_with_fisher.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    # Log-scale version prevents Fisher from visually hiding the other methods.
    figure, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2), constrained_layout=True
    )
    for method in METHOD_ORDER:
        subset = (
            combined[combined["Method"] == method]
            .sort_values("m")
        )
        subset = subset[
            np.isfinite(subset["rel_W_error_ESSsec_total"])
            & np.isfinite(subset["Phi_Rhat"])
        ]
        if subset.empty:
            continue
        axes[0].plot(
            subset["m"],
            subset["rel_W_error_ESSsec_total"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            subset["m"],
            subset["Phi_Rhat"],
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )

    axes[0].set_yscale("log")
    axes[0].set_title("(a) Sampling efficiency")
    axes[0].set_xlabel(r"Graph size $m$")
    axes[0].set_ylabel(r"Relative-$W$ ESS/sec (log scale)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()

    axes[1].set_title(r"(b) Energy split-$\widehat R$")
    axes[1].set_xlabel(r"Graph size $m$")
    axes[1].set_ylabel(r"Split-$\widehat R_\Phi$")
    axes[1].axhline(1.05, linestyle="--", linewidth=1.3)
    axes[1].axhline(1.10, linestyle=":", linewidth=1.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for axis in axes:
        axis.set_xticks(available_m)

    figure.savefig(
        plot_dir / "scaling_d5_with_fisher_logess.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        plot_dir / "scaling_d5_with_fisher_logess.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# Run one m
# =============================================================================

def selected_methods(args: argparse.Namespace) -> List[str]:
    if args.methods.strip().lower() == "all":
        return list(METHOD_ORDER)
    chosen = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = [name for name in chosen if name not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {METHOD_ORDER}.")
    return [name for name in METHOD_ORDER if name in chosen]


def run_one_seed(
    args: argparse.Namespace,
    dataset_seed: int,
) -> pd.DataFrame:
    d = 5
    m = 50
    settings = resolve_settings(args, m)
    preburn = int(settings["preburn"])
    retained = int(settings["retained"])
    fisher_individual_settle = min(args.fisher_individual_settle, max(preburn - 1, 0))
    fisher_adaptive_steps = (
        args.fisher_adaptive_steps
        if args.fisher_adaptive_steps is not None
        else max(preburn - fisher_individual_settle, 1)
    )

    experiment_dir = Path(args.outdir) / f"graph_posterior_d5_m50_seed{dataset_seed}"
    results_dir = experiment_dir / "results"
    plots_dir = experiment_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.outdir) / f"summary_d5_m50_seed{dataset_seed}.csv"
    if args.resume and summary_path.exists():
        existing = pd.read_csv(summary_path)
        required_coverage_columns = {
            "posterior_mean_rel_W_error",
            "entry_coverage",
            "trace_coverage",
            "eigenvalue_coverage",
        }
        if required_coverage_columns.issubset(existing.columns) and set(
            selected_methods(args)
        ) <= set(existing["Method"]):
            print(f"Skipping completed dataset seed={dataset_seed}: {summary_path}")
            return existing
        print(
            f"Existing summary for seed={dataset_seed} predates the coverage "
            "diagnostics; rerunning this seed.",
            flush=True,
        )

    rng = np.random.default_rng(dataset_seed)
    edges = make_cycle_graph(m)
    edge_count = len(edges)
    w_star = sample_ground_truth_w(
        edge_count, d, rng, scale=args.truth_scale
    )
    r_matrix = args.r_scale * np.eye(m * d)
    q_star = block_laplacian(w_star, edges, m, d) + r_matrix
    y_train = sample_graph_signals(q_star, args.n_train, rng)
    y_test = sample_graph_signals(q_star, args.n_test, rng)

    posterior = PSDGraphPosterior(
        edges=edges,
        m=m,
        d=d,
        y_train=y_train,
        r_matrix=r_matrix,
        lambda_tr=args.lambda_tr,
        eta_frob=args.eta_frob,
        alpha_logdet=args.alpha_logdet,
    )

    np.save(results_dir / "W_star.npy", w_star)
    np.save(results_dir / "Q_star.npy", q_star)
    np.save(results_dir / "Y_train.npy", y_train)
    np.save(results_dir / "Y_test.npy", y_test)

    fisher_factors = tuple(
        float(value.strip())
        for value in args.fisher_grid_factors.split(",")
        if value.strip()
    )
    run_config = {
        **vars(args),
        "active_m": m,
        "active_d": d,
        "dataset_seed": int(dataset_seed),
        "preburn_fixed_methods": preburn,
        "retained": retained,
        "settings": settings,
        "fisher_adaptive_steps_resolved": fisher_adaptive_steps,
        "fisher_individual_settle_resolved": fisher_individual_settle,
        "strict_spd_test": True,
        "credible_interval_level": float(args.credible_level),
        "credible_interval_type": "equal-tailed marginal",
        "entry_coverage_coordinates": "unique symmetric entries i<=j",
        "eigenvalue_coverage_order": "ascending ordered eigenvalues",
        "fisher_workflow": (
            "discarded adaptation -> chain-specific settle -> pooled metric -> "
            "fixed-h typical-set tuning -> shared-kernel settle -> production"
        ),
    }
    (results_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    chain_seeds = [10_000_000 + 1000 * dataset_seed + chain for chain in range(args.n_chains)]
    initial_states = [
        make_initial_w(edge_count, d, chain)
        for chain in range(args.n_chains)
    ]

    print("\n" + "=" * 100)
    print(
        f"dataset seed={dataset_seed}, d=5, m={m}, edges={edge_count}, "
        f"fixed burn={preburn}, retained={retained}"
    )
    print("=" * 100)

    rows: List[Dict[str, float]] = []
    plot_data: Dict[str, List[pd.DataFrame]] = {}

    methods_now = selected_methods(args)
    fisher_plan = None
    if "FisherAdaptiveMALA" in methods_now:
        # Prepare the shared Fisher kernel before any retained Fisher sampling.
        print("\nPreparing Fisher-adaptive shared production kernel")
        fisher_plan = prepare_fisher_plan(
            posterior=posterior,
            initial_states=initial_states,
            adaptive_steps=fisher_adaptive_steps,
            individual_settle_steps=fisher_individual_settle,
            h0=float(settings["h_fisher"]),
            target_accept=args.target_accept,
            damping=args.fisher_damping,
            init_steps=args.fisher_init_steps,
            update_interval=args.fisher_update_interval,
            max_condition=args.fisher_max_condition,
            adapt_rate=args.fisher_adapt_rate,
            adapt_decay=args.fisher_adapt_decay,
            adapt_exponent=args.fisher_adapt_exponent,
            h_min=args.fisher_h_min,
            h_max=args.fisher_h_max,
            tune_warmup=args.fisher_tune_warmup,
            tune_steps=args.fisher_tune_steps,
            tune_chains=args.fisher_tune_chains,
            acceptance_tolerance=args.fisher_acceptance_tolerance,
            factors=fisher_factors,
            max_bracket_steps=args.fisher_max_bracket_steps,
            bisection_steps=args.fisher_bisection_steps,
            seed_base=50_000_000 + 1000 * dataset_seed,
        )
        pd.DataFrame(fisher_plan.adaptation_rows).to_csv(
            results_dir / "fisher_adaptation.csv", index=False
        )
        pd.DataFrame(fisher_plan.tuning_rows).to_csv(
            results_dir / "fisher_tuning.csv", index=False
        )
        if fisher_plan.feasible:
            print(
                f"  shared h={fisher_plan.h_selected:.3e}, "
                f"pilot acc={fisher_plan.pilot_acceptance:.3f}, "
                f"pilot invalid={fisher_plan.pilot_invalid_rate:.3f}, "
                f"cond(P)={fisher_plan.condition_number:.3e}",
                flush=True,
            )
        else:
            failure_row = {
                "method": "FisherAdaptiveMALA",
                "stage": "fixed_metric_tuning",
                "reason": fisher_plan.failure_reason,
                "best_pilot_acceptance": fisher_plan.pilot_acceptance,
                "best_pilot_invalid_rate": fisher_plan.pilot_invalid_rate,
                "condition_number": fisher_plan.condition_number,
            }
            pd.DataFrame([failure_row]).to_csv(
                results_dir / "fisher_failure.csv", index=False
            )
            print(
                "  Fisher tuning failed cleanly. "
                "The run will continue with the other methods and mark Fisher "
                "as TF in the table. See fisher_adaptation.csv, "
                "fisher_tuning.csv, and fisher_failure.csv.",
                flush=True,
            )

    for method in methods_now:
        print(f"\nMethod: {method}")

        if method == "FisherAdaptiveMALA" and not fisher_plan.feasible:
            failed_row: Dict[str, float] = {
                "dataset_seed": int(dataset_seed),
                "d": d,
                "m": m,
                "E": edge_count,
                "preburn": preburn,
                "retained": retained,
                "Method": method,
                "status": "TUNING_FAILED",
                "runtime_overhead_sec": (
                    fisher_plan.adaptation_runtime_sec
                    + fisher_plan.tuning_runtime_sec
                ),
                "runtime_total_all_chains": (
                    fisher_plan.adaptation_runtime_sec
                    + fisher_plan.tuning_runtime_sec
                ),
                "runtime_sampling_all_chains": 0.0,
                "runtime_total_mean": (
                    fisher_plan.adaptation_runtime_sec
                    + fisher_plan.tuning_runtime_sec
                ) / args.n_chains,
                "runtime_sampling_mean": 0.0,
                "accept_mean": np.nan,
                "accept_sd": np.nan,
                "preburn_accept_mean": float(
                    np.mean(
                        [
                            row["acceptance"]
                            for row in fisher_plan.adaptation_rows
                        ]
                    )
                ),
                "h_final_mean": np.nan,
                "h_final_sd": np.nan,
                "invalid_rate_mean": np.nan,
                "preburn_invalid_rate_mean": float(
                    np.mean(
                        [
                            row["invalid_rate"]
                            for row in fisher_plan.adaptation_rows
                        ]
                    )
                ),
                "numerical_failure_rate_mean": np.nan,
                "preconditioner_condition_mean": (
                    fisher_plan.condition_number
                ),
                "fisher_pilot_acceptance": (
                    fisher_plan.pilot_acceptance
                ),
                "fisher_pilot_invalid_rate": (
                    fisher_plan.pilot_invalid_rate
                ),
            }
            for observable_name in (
                "rel_W_error",
                "test_nll",
                "logdet_Q",
                "lambda_min_Q",
                "Phi",
            ):
                for suffix in (
                    "mean",
                    "sd",
                    "Rhat",
                    "ESS",
                    "ESSsec_total",
                    "ESSsec_sampling",
                ):
                    failed_row[f"{observable_name}_{suffix}"] = np.nan
            failed_row.update({
                "posterior_mean_rel_W_error": np.nan,
                "credible_level": float(args.credible_level),
                "entry_coverage": np.nan,
                "entry_covered_count": 0,
                "entry_interval_count": 0,
                "entry_interval_width_mean": np.nan,
                "entry_interval_width_median": np.nan,
                "trace_coverage": np.nan,
                "trace_covered_count": 0,
                "trace_interval_count": 0,
                "trace_interval_width_mean": np.nan,
                "trace_interval_width_median": np.nan,
                "eigenvalue_coverage": np.nan,
                "eigenvalue_covered_count": 0,
                "eigenvalue_interval_count": 0,
                "eigenvalue_interval_width_mean": np.nan,
                "eigenvalue_interval_width_median": np.nan,
            })
            rows.append(failed_row)
            print(
                "  Skipped production: no feasible fixed Fisher kernel.",
                flush=True,
            )
            continue

        chain_results: List[ChainResult] = []
        observable_chains: List[Dict[str, Array]] = []

        for chain_index in range(args.n_chains):
            if method == "ConeMALA":
                result = cone_mala_chain(
                    posterior, initial_states[chain_index], preburn,
                    retained, float(settings["h_cone"]), chain_seeds[chain_index]
                )
            elif method == "EuclideanMALA":
                result = euclidean_mala_chain(
                    posterior, initial_states[chain_index], preburn,
                    retained, float(settings["h_euclid"]), chain_seeds[chain_index]
                )
            elif method == "GenericRMALA":
                result = generic_rmala_chain(
                    posterior, initial_states[chain_index], preburn,
                    retained, float(settings["h_rmala"]), chain_seeds[chain_index]
                )
            elif method == "FisherAdaptiveMALA":
                result = fixed_fisher_mala_chain(
                    posterior=posterior,
                    w0=fisher_plan.anchors[chain_index],
                    warmup=args.fisher_final_settle,
                    retained=retained,
                    h=fisher_plan.h_selected,
                    root=fisher_plan.root,
                    condition_number=fisher_plan.condition_number,
                    seed=chain_seeds[chain_index] + 5_000_000,
                )
            else:
                raise KeyError(method)

            chain_results.append(result)
            observable = observables(result.samples, posterior, w_star, y_test)
            observable_chains.append(observable)
            pd.DataFrame(observable).to_csv(
                results_dir / f"{method}_chain{chain_index}_observables.csv",
                index=False,
            )
            if args.save_samples:
                np.save(
                    results_dir / f"{method}_chain{chain_index}_samples.npy",
                    result.samples,
                )
            print(
                f"  chain {chain_index + 1}: h={result.h_final:.3e}, "
                f"settle/burn acc={result.preburn_acceptance:.3f}, "
                f"production acc={result.acceptance:.3f}, "
                f"invalid={result.invalid_rate:.3f}, "
                f"total={result.runtime_total_sec:.1f}s"
            )

        coverage_summary, coverage_details = posterior_recovery_and_coverage(
            sample_chains=[result.samples for result in chain_results],
            w_star=w_star,
            credible_level=args.credible_level,
            method=method,
            dataset_seed=dataset_seed,
        )
        coverage_details.to_csv(
            results_dir / f"{method}_credible_intervals.csv",
            index=False,
        )

        if method == "FisherAdaptiveMALA":
            runtime_overhead = (
                fisher_plan.adaptation_runtime_sec
                + fisher_plan.tuning_runtime_sec
            )
            if all(result.acceptance <= 0.0 for result in chain_results):
                raise RuntimeError(
                    "Fisher passed pilot tuning but had zero production "
                    "acceptance in every chain. This indicates a separate "
                    "production-kernel failure; inspect fisher_tuning.csv."
                )
        else:
            runtime_overhead = 0.0

        runtime_sampling = float(
            sum(result.runtime_sampling_sec for result in chain_results)
        )
        runtime_total = float(
            runtime_overhead
            + sum(result.runtime_total_sec for result in chain_results)
        )

        row: Dict[str, float] = {
            "dataset_seed": int(dataset_seed),
            "d": d,
            "m": m,
            "E": edge_count,
            "preburn": preburn,
            "retained": retained,
            "Method": method,
            "status": "OK",
            "runtime_overhead_sec": runtime_overhead,
            "runtime_total_all_chains": runtime_total,
            "runtime_sampling_all_chains": runtime_sampling,
            "runtime_total_mean": runtime_total / args.n_chains,
            "runtime_sampling_mean": runtime_sampling / args.n_chains,
            "accept_mean": float(np.mean([r.acceptance for r in chain_results])),
            "accept_sd": float(np.std([r.acceptance for r in chain_results], ddof=1)),
            "preburn_accept_mean": float(np.mean([r.preburn_acceptance for r in chain_results])),
            "h_final_mean": float(np.mean([r.h_final for r in chain_results])),
            "h_final_sd": float(np.std([r.h_final for r in chain_results], ddof=1)),
            "invalid_rate_mean": float(np.mean([r.invalid_rate for r in chain_results])),
            "preburn_invalid_rate_mean": float(np.mean([r.preburn_invalid_rate for r in chain_results])),
            "numerical_failure_rate_mean": float(np.mean([r.numerical_failure_rate for r in chain_results])),
            "preconditioner_condition_mean": float(np.mean([r.preconditioner_condition for r in chain_results])),
            "fisher_pilot_acceptance": (
                fisher_plan.pilot_acceptance if method == "FisherAdaptiveMALA" else np.nan
            ),
            "fisher_pilot_invalid_rate": (
                fisher_plan.pilot_invalid_rate if method == "FisherAdaptiveMALA" else np.nan
            ),
            **coverage_summary,
        }

        for observable_name in (
            "rel_W_error", "test_nll", "logdet_Q", "lambda_min_Q", "Phi"
        ):
            chains = [obs[observable_name] for obs in observable_chains]
            pooled = np.concatenate(chains)
            total_ess = float(sum(ess_1d(chain) for chain in chains))
            row[f"{observable_name}_mean"] = float(np.mean(pooled))
            row[f"{observable_name}_sd"] = float(np.std(pooled, ddof=1))
            row[f"{observable_name}_Rhat"] = split_rhat(chains)
            row[f"{observable_name}_ESS"] = total_ess
            row[f"{observable_name}_ESSsec_total"] = total_ess / max(runtime_total, 1e-12)
            row[f"{observable_name}_ESSsec_sampling"] = total_ess / max(runtime_sampling, 1e-12)

        rows.append(row)
        plot_data[method] = [pd.DataFrame(obs) for obs in observable_chains]

    summary = pd.DataFrame(rows).set_index("Method").loc[methods_now].reset_index()
    summary.to_csv(results_dir / "summary_multichain.csv", index=False)
    summary.to_csv(results_dir / f"summary_d5_m50_seed{dataset_seed}.csv", index=False)
    summary.to_csv(summary_path, index=False)

    if methods_now == list(METHOD_ORDER):
        write_latex_table(
            summary=summary,
            path=Path(args.outdir) / f"table_d5_m50_seed{dataset_seed}_with_fisher.tex",
            m=m,
            d=d,
            preburn=preburn,
            retained=retained,
            fisher_update_interval=args.fisher_update_interval,
            fisher_adaptive_steps=fisher_adaptive_steps,
            fisher_individual_settle=fisher_individual_settle,
            fisher_final_settle=args.fisher_final_settle,
        )

    save_two_panel_ecdf(
        plot_data, experiment_dir, "logdet_Q", "Phi",
        f"ecdf_logdet_phi_d5_m{m}",
        "(a) Log-determinant observable", "(b) Posterior energy",
    )
    save_two_panel_ecdf(
        plot_data, experiment_dir, "rel_W_error", "test_nll",
        f"ecdf_error_nll_d5_m{m}",
        "(a) Relative edge-weight error", "(b) Test negative log-likelihood",
    )
    if methods_now == list(METHOD_ORDER):
        save_acceptance_plot(summary, experiment_dir, m)

    compact = [
        "Method", "accept_mean", "h_final_mean",
        "rel_W_error_ESSsec_total", "test_nll_ESSsec_total",
        "rel_W_error_Rhat", "test_nll_Rhat", "Phi_Rhat",
        "invalid_rate_mean",
    ]
    print("\nCompact summary:")
    print(summary[compact].to_string(index=False))
    print(f"Saved: {summary_path}")
    return summary


def collect_existing_seed_summaries(outdir: Path) -> pd.DataFrame:
    """Collect all completed per-dataset-seed summaries."""
    frames: List[pd.DataFrame] = []
    pattern = re.compile(r"summary_d5_m50_seed(\d+)\.csv$")
    for path in sorted(outdir.glob("summary_d5_m50_seed*.csv")):
        match = pattern.search(path.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        if "dataset_seed" not in frame.columns:
            frame["dataset_seed"] = int(match.group(1))
        frames.append(frame)
    if not frames:
        raise RuntimeError("No completed m=50 seed-summary files were found.")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["dataset_seed", "Method"]).reset_index(drop=True)


def add_seed_convergence_columns(
    by_seed: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Add the worst displayed Rhat and an all-diagnostics convergence flag."""
    result = by_seed.copy()
    rhat_columns = [
        "rel_W_error_Rhat",
        "test_nll_Rhat",
        "logdet_Q_Rhat",
        "Phi_Rhat",
    ]
    result["max_displayed_Rhat"] = result[rhat_columns].max(axis=1)
    result["converged_all_displayed"] = (
        result["status"].eq("OK")
        & result[rhat_columns].notna().all(axis=1)
        & result["max_displayed_Rhat"].le(threshold)
    )
    return result


def summarize_across_dataset_seeds(
    by_seed: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Aggregate feasibility, efficiency, and convergence across datasets."""
    metrics = [
        "accept_mean",
        "invalid_rate_mean",
        "rel_W_error_ESSsec_sampling",
        "test_nll_ESSsec_sampling",
        "rel_W_error_ESSsec_total",
        "test_nll_ESSsec_total",
        "rel_W_error_Rhat",
        "test_nll_Rhat",
        "logdet_Q_Rhat",
        "Phi_Rhat",
        "max_displayed_Rhat",
        "posterior_mean_rel_W_error",
        "entry_coverage",
        "trace_coverage",
        "eigenvalue_coverage",
        "entry_interval_width_mean",
        "trace_interval_width_mean",
        "eigenvalue_interval_width_mean",
    ]
    rows: List[Dict[str, float]] = []
    for method in METHOD_ORDER:
        group = by_seed.loc[by_seed["Method"] == method].copy()
        successful = group.loc[group["status"] == "OK"]
        row: Dict[str, float] = {
            "Method": method,
            "n_seed_runs": int(len(group)),
            "n_successful_seed_runs": int(len(successful)),
            "n_converged_seed_runs": int(
                group["converged_all_displayed"].fillna(False).sum()
            ),
            "convergence_threshold": float(threshold),
        }
        for metric in metrics:
            values = pd.to_numeric(successful[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method").loc[METHOD_ORDER].reset_index()


def mean_sd_text(mean: float, sd: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return r"\texttt{TF}"
    return rf"${mean:.{digits}f}\pm {sd:.{digits}f}$"


def write_seed_robustness_latex(
    summary: pd.DataFrame,
    path: Path,
    seeds: Sequence[int],
    threshold: float,
) -> None:
    indexed = summary.set_index("Method").loc[METHOD_ORDER]
    rows: List[str] = []
    for method in METHOD_ORDER:
        row = indexed.loc[method]
        rows.append(
            f"{METHOD_LABELS[method]}"
            f" & {mean_sd_text(row['accept_mean_mean'], row['accept_mean_sd'])}"
            f" & {mean_sd_text(row['invalid_rate_mean_mean'], row['invalid_rate_mean_sd'])}"
            f" & {mean_sd_text(row['rel_W_error_ESSsec_sampling_mean'], row['rel_W_error_ESSsec_sampling_sd'], 2)}"
            f" & {mean_sd_text(row['test_nll_ESSsec_sampling_mean'], row['test_nll_ESSsec_sampling_sd'], 2)}"
            f" & {latex_value(row['max_displayed_Rhat_median'], 3)}"
            f" & {int(row['n_converged_seed_runs'])}/{int(row['n_seed_runs'])}"
            r" \\"
        )

    seed_text = ",".join(str(int(seed)) for seed in seeds)
    content = rf"""
\begin{{table*}}[t]
\centering
\caption{{Dataset-seed robustness for the determinant-neutral $d=5$, $m=50$
cycle-graph posterior over seeds $\{{{seed_text}\}}$. Entries are
mean $\pm$ standard deviation across independently generated datasets.
Production ESS/s uses retained frozen-kernel runtime only. The penultimate
column is the median, across datasets, of the largest split-$\widehat R$
among relative-$W$, held-out NLL, $\log\det Q$, and posterior energy.}}
\label{{tab:d5-m50-seed-robustness}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
Method & Acc. & Invalid & Rel.-$W$ ESS/s & NLL ESS/s
& Median max $\widehat R$ & Converged seeds \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.96\textwidth}}
\footnotesize
All runs use four chains, $4{{,}}000$ discarded transitions, $12{{,}}000$
retained transitions per chain, and $\alpha_{{\log\det}}=3$. Invalid
proposals are rejected without projection. A seed is counted as converged
only when all four displayed split-$\widehat R$ values are at most
${threshold:.2f}$.
\end{{minipage}}
\end{{table*}}
""".strip() + "\n"
    path.write_text(content, encoding="utf-8")


def write_seed_recovery_coverage_latex(
    summary: pd.DataFrame,
    path: Path,
    seeds: Sequence[int],
    credible_level: float,
    threshold: float,
) -> None:
    """Write a compact table focused on recovery and uncertainty coverage."""
    indexed = summary.set_index("Method").loc[METHOD_ORDER]
    rows: List[str] = []
    for method in METHOD_ORDER:
        row = indexed.loc[method]
        rows.append(
            f"{METHOD_LABELS[method]}"
            f" & {mean_sd_text(row['posterior_mean_rel_W_error_mean'], row['posterior_mean_rel_W_error_sd'], 3)}"
            f" & {mean_sd_text(row['entry_coverage_mean'], row['entry_coverage_sd'], 3)}"
            f" & {mean_sd_text(row['trace_coverage_mean'], row['trace_coverage_sd'], 3)}"
            f" & {mean_sd_text(row['eigenvalue_coverage_mean'], row['eigenvalue_coverage_sd'], 3)}"
            f" & {latex_value(row['max_displayed_Rhat_median'], 3)}"
            f" & {int(row['n_converged_seed_runs'])}/{int(row['n_seed_runs'])}"
            r" \\"
        )

    seed_text = ",".join(str(int(seed)) for seed in seeds)
    nominal = 100.0 * credible_level
    content = rf"""
\begin{{table*}}[t]
\centering
\caption{{Graph recovery and uncertainty calibration for the determinant-neutral
$d=5$, $m=50$ posterior over independently generated datasets with seeds
$\{{{seed_text}\}}$. Entries are mean $\pm$ standard deviation across seeds.
Rel.-$W$ is the relative Frobenius error of the pooled posterior mean. Entry,
trace, and eigenvalue columns report empirical coverage of nominal
${nominal:.0f}\%$ equal-tailed marginal credible intervals.}}
\label{{tab:d5-m50-seed-recovery-coverage}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
Method & Rel.-$W$ error & Entry cov. & Trace cov. & Eigenvalue cov.
& Median max $\widehat R$ & Converged seeds \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.96\textwidth}}
\footnotesize
Entry coverage uses the $d(d+1)/2$ unique symmetric coordinates of every
edge matrix; eigenvalue intervals correspond to increasingly ordered
eigenvalues. All retained draws from the four chains are pooled. A seed is
counted as converged only when the four displayed split-$\widehat R$
diagnostics are at most {threshold:.2f}; coverage from nonconverged chains is
descriptive and should not be interpreted as calibrated posterior uncertainty.
\end{{minipage}}
\end{{table*}}
""".strip() + "\n"
    path.write_text(content, encoding="utf-8")


def save_seed_robustness_plot(by_seed: pd.DataFrame, outdir: Path) -> None:
    """Two-panel seed plot: production efficiency and worst displayed Rhat."""
    import matplotlib.pyplot as plt

    set_large_plot_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for method in METHOD_ORDER:
        group = by_seed.loc[
            (by_seed["Method"] == method) & by_seed["status"].eq("OK")
        ].sort_values("dataset_seed")
        efficiency = pd.to_numeric(
            group["rel_W_error_ESSsec_sampling"], errors="coerce"
        )
        valid_efficiency = np.isfinite(efficiency) & (efficiency > 0.0)
        axes[0].plot(
            group.loc[valid_efficiency, "dataset_seed"],
            efficiency.loc[valid_efficiency],
            marker="o",
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )
        rhat = pd.to_numeric(group["max_displayed_Rhat"], errors="coerce")
        valid_rhat = np.isfinite(rhat)
        axes[1].plot(
            group.loc[valid_rhat, "dataset_seed"],
            rhat.loc[valid_rhat],
            marker="o",
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )

    axes[0].set_yscale("log")
    axes[0].set_title("(a) Production relative-$W$ ESS/s")
    axes[1].set_title("(b) Worst displayed split-$\\widehat R$")
    axes[0].set_xlabel("Dataset seed")
    axes[1].set_xlabel("Dataset seed")
    axes[0].set_ylabel("ESS/s (log scale)")
    axes[1].set_ylabel("Maximum split-$\\widehat R$")
    axes[1].axhline(1.05, linestyle="--", linewidth=1.4)
    axes[1].axhline(1.01, linestyle=":", linewidth=1.2)
    for axis in axes:
        axis.set_xticks(sorted(by_seed["dataset_seed"].unique()))
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.savefig(outdir / "fig_d5_m50_seed_robustness.pdf", bbox_inches="tight")
    figure.savefig(
        outdir / "fig_d5_m50_seed_robustness.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Five-seed robustness study for the d=5, m=50 pure-Wishart graph posterior."
        )
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=[11, 22, 33, 44, 55],
        help="Comma-separated dataset seeds.",
    )
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=80)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument(
        "--outdir",
        type=str,
        default="d5_m50_seed_robustness_wishart",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_samples", action="store_true")

    parser.add_argument("--truth_scale", type=float, default=0.25)
    parser.add_argument("--r_scale", type=float, default=0.75)
    parser.add_argument("--lambda_tr", type=float, default=0.05)
    parser.add_argument("--eta_frob", type=float, default=0.0)
    parser.add_argument("--alpha_logdet", type=float, default=3.0)

    # Optional overrides of the paper-matched m-specific settings.
    parser.add_argument("--preburn", type=int)
    parser.add_argument("--retained", type=int)
    parser.add_argument("--h_cone", type=float)
    parser.add_argument("--h_euclid", type=float)
    parser.add_argument("--h_rmala", type=float)
    parser.add_argument("--h_fisher", type=float)

    parser.add_argument("--target_accept", type=float, default=0.574)
    parser.add_argument("--fisher_damping", type=float, default=10.0)
    parser.add_argument("--fisher_init_steps", type=int, default=500)
    parser.add_argument("--fisher_adaptive_steps", type=int)
    parser.add_argument("--fisher_individual_settle", type=int, default=1000)
    parser.add_argument("--fisher_final_settle", type=int, default=500)
    parser.add_argument("--fisher_tune_warmup", type=int, default=100)
    parser.add_argument("--fisher_tune_steps", type=int, default=400)
    parser.add_argument("--fisher_tune_chains", type=int, default=2)
    parser.add_argument("--fisher_grid_factors", type=str, default="0.125,0.25,0.5,1,2,4")
    parser.add_argument("--fisher_acceptance_tolerance", type=float, default=0.08)
    parser.add_argument("--fisher_max_bracket_steps", type=int, default=24)
    parser.add_argument("--fisher_bisection_steps", type=int, default=8)
    parser.add_argument(
        "--fisher_update_interval", type=int, default=1,
        help="Preconditioner update interval during discarded adaptation "
             "(default 1: every iteration, as in Titsias 2023).",
    )
    parser.add_argument("--fisher_max_condition", type=float, default=100.0)
    parser.add_argument("--fisher_adapt_rate", type=float, default=0.05)
    parser.add_argument("--fisher_adapt_decay", type=float, default=100.0)
    parser.add_argument("--fisher_adapt_exponent", type=float, default=0.6)
    parser.add_argument("--fisher_h_min", type=float, default=1e-12)
    parser.add_argument("--fisher_h_max", type=float, default=1.0)
    parser.add_argument("--rhat_threshold", type=float, default=1.05)
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated subset of " + ",".join(METHOD_ORDER) + " (default: all).",
    )
    parser.add_argument(
        "--credible_level",
        type=float,
        default=0.95,
        help=(
            "Nominal level for equal-tailed marginal credible intervals "
            "for entries, traces, and ordered eigenvalues."
        ),
    )
    return parser


def main() -> None:
    print(f"Script version: {WISHART_ROBUSTNESS_VERSION}")
    args = build_parser().parse_args()
    if args.n_chains < 2:
        raise ValueError("At least two chains are needed for split-Rhat.")
    if args.fisher_update_interval < 1:
        raise ValueError("fisher_update_interval must be positive.")
    if not args.seeds:
        raise ValueError("At least one dataset seed is required.")
    if not 0.0 < args.credible_level < 1.0:
        raise ValueError("credible_level must lie strictly between 0 and 1.")
    if abs(float(args.eta_frob)) > 0.0:
        raise ValueError("Final paper configuration requires --eta_frob 0.0 (pure Wishart target).")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for dataset_seed in args.seeds:
        run_one_seed(args, int(dataset_seed))

    by_seed = collect_existing_seed_summaries(outdir)
    by_seed = by_seed.loc[by_seed["dataset_seed"].isin(args.seeds)].copy()
    missing = [
        (seed, method)
        for seed in args.seeds
        for method in METHOD_ORDER
        if not ((by_seed["dataset_seed"] == seed) & (by_seed["Method"] == method)).any()
    ]
    if missing:
        print(
            f"Per-seed results saved.  Five-dataset tables need all four "
            f"samplers; missing (dataset seed, method) = {missing}."
        )
        return
    by_seed = add_seed_convergence_columns(by_seed, args.rhat_threshold)
    by_seed.to_csv(outdir / "robustness_d5_m50_by_seed.csv", index=False)

    aggregate = summarize_across_dataset_seeds(
        by_seed=by_seed,
        threshold=args.rhat_threshold,
    )
    aggregate.to_csv(outdir / "robustness_d5_m50_summary.csv", index=False)
    write_seed_robustness_latex(
        aggregate,
        outdir / "table_d5_m50_seed_robustness.tex",
        seeds=args.seeds,
        threshold=args.rhat_threshold,
    )
    write_seed_recovery_coverage_latex(
        aggregate,
        outdir / "table_d5_m50_seed_recovery_coverage.tex",
        seeds=args.seeds,
        credible_level=args.credible_level,
        threshold=args.rhat_threshold,
    )
    save_seed_robustness_plot(by_seed, outdir)

    seed_columns = [
        "dataset_seed",
        "Method",
        "accept_mean",
        "invalid_rate_mean",
        "rel_W_error_ESSsec_sampling",
        "test_nll_ESSsec_sampling",
        "posterior_mean_rel_W_error",
        "entry_coverage",
        "trace_coverage",
        "eigenvalue_coverage",
        "max_displayed_Rhat",
        "converged_all_displayed",
    ]
    print("\nSeed-level robustness summary:")
    print(by_seed[seed_columns].to_string(index=False))

    aggregate_columns = [
        "Method",
        "accept_mean_mean",
        "accept_mean_sd",
        "invalid_rate_mean_mean",
        "invalid_rate_mean_sd",
        "rel_W_error_ESSsec_sampling_mean",
        "rel_W_error_ESSsec_sampling_sd",
        "test_nll_ESSsec_sampling_mean",
        "test_nll_ESSsec_sampling_sd",
        "posterior_mean_rel_W_error_mean",
        "posterior_mean_rel_W_error_sd",
        "entry_coverage_mean",
        "entry_coverage_sd",
        "trace_coverage_mean",
        "trace_coverage_sd",
        "eigenvalue_coverage_mean",
        "eigenvalue_coverage_sd",
        "max_displayed_Rhat_median",
        "n_converged_seed_runs",
        "n_seed_runs",
    ]
    print("\nAcross-seed summary:")
    print(aggregate[aggregate_columns].to_string(index=False))
    print(f"\nSaved robustness outputs under: {outdir.resolve()}")


if __name__ == "__main__":
    main()
