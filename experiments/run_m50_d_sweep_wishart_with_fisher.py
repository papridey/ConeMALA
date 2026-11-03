#!/usr/bin/env python3
"""
m=50 block-dimension sweep for the Wishart graph posterior.

Samplers: ConeMALA, Fisher-adaptive MALA, Euclidean MALA, generic RMALA
(the four compared in the paper), plus log-Cholesky MALA and NUTS, which are
used by the units-and-basis experiment (run_scaling_experiment.py).  Choose
samplers with --methods.  Diagnostics include classical split-Rhat and ESS
(columns *_Rhat, *_ESS, *_ESSsec_*) and the rank-normalized versions of
Vehtari et al. (2021) (columns *_Rhat_rank, *_ESS_bulk, *_ESS_tail).

This script extends the existing d=5 graph-size sweep in the orthogonal
direction: graph size is fixed at m=50 and the SPD block dimension is swept
over d in {2,5,8,10} by default.

Design inherited from the graph-size sweep
------------------------------------------
* Cycle graph with m=50 and |E|=50.
* Same synthetic truth generator, Gaussian data model, initialization,
  posterior, four samplers, diagnostics, and aggregation logic.
* n_train=n_test=80, four chains, 4,000 discarded fixed-kernel burn-in
  transitions, and 12,000 retained transitions per chain by default.
* Determinant-neutral Wishart prior at every d:
      nu=d+1, Psi^{-1}=0.1 I_d,
  equivalently alpha_logdet=(d+1)/2 and lambda_tr=0.05 in the intrinsic
  potential.
* The d=5 point reuses the original m=50 paper step sizes exactly.
* For d != 5, ConeMALA, Euclidean MALA, and generic RMALA select scalar step
  sizes in separate discarded pilot runs, freeze them, and then use the same
  burn-in/production flow as the original graph-size experiment.
* Fisher-adaptive MALA preserves the original workflow:
  discarded chain-specific adaptation -> pooled preconditioner -> fixed-h
  typical-set tuning -> shared-kernel settling -> retained production.
  The preconditioner is updated at every adaptive iteration at every d
  (Titsias 2023, Algorithm 1), the same rule as the graph-size sweep.
* The output records both the total free edge-parameter dimension
      D = |E| d(d+1)/2
  and the lifted precision dimension p=md.

The discarded fixed-method pilots are intentionally kept separate from the
reported production and end-to-end ESS/s, matching the original paper's
protocol in which fixed step sizes were selected in discarded pilot runs.

Example
-------
python3 -u run_m50_d_sweep_wishart_with_fisher.py \
    --d_list 2,5,8,10 \
    --outdir m50_d_sweep_wishart_with_fisher \
    --resume
"""

from __future__ import annotations

D_SWEEP_VERSION = "1.0"

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


# =============================================================================
# Numerical helpers
# =============================================================================

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


class PSDGraphPosterior:
    def __init__(
        self,
        edges: Sequence[Tuple[int, int]],
        m: int,
        d: int,
        y_train: Array,
        r_matrix: Array,
        lambda_tr: float = 0.05,
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
            phi_prior -= self.alpha_logdet * logdet_we

            grad_w[edge_index] += self.lambda_tr * np.eye(self.d)
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
    n_leapfrog_mean: float = float("nan")
    treedepth_mean: float = float("nan")
    treedepth_saturated_rate: float = float("nan")
    divergence_rate: float = float("nan")


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
    edge_scale: Optional[Array] = None,
) -> Tuple[Array, float, Array, bool, bool]:
    """One exact ConeMALA transition; with ``edge_scale`` edge e uses
    h_e = h * edge_scale[e] in both the forward and reverse proposal densities,
    so the MH correction stays exact."""
    edge_count, d, _ = w.shape
    hs = (np.full(edge_count, float(h)) if edge_scale is None
          else float(h) * np.asarray(edge_scale, dtype=float))
    proposal = np.zeros_like(w)
    tangent_list: List[Array] = []
    mean_list: List[Array] = []

    try:
        for edge_index in range(edge_count):
            he = hs[edge_index]
            root, _ = mat_sqrt_inv_sqrt(w[edge_index])
            mean = -he * sym(root @ grad[edge_index] @ root)
            tangent = sym(
                mean
                + math.sqrt(2.0 * he) * random_sym_normal(d, rng)
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
                    2.0 * hs[edge_index],
                )
                - log_spd_exp_jacobian(tangent_list[edge_index])
            )

            reverse_tangent = spd_congruence_log(
                proposal[edge_index], w[edge_index]
            )
            root_prop, _ = mat_sqrt_inv_sqrt(proposal[edge_index])
            reverse_mean = -hs[edge_index] * sym(
                root_prop @ grad_prop[edge_index] @ root_prop
            )
            logq_reverse += (
                log_gaussian_orthosym(
                    reverse_tangent,
                    reverse_mean,
                    2.0 * hs[edge_index],
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
    edge_scale: Optional[Array] = None,
) -> ChainResult:
    rng = np.random.default_rng(seed)
    w = np.asarray([project_spd(block) for block in w0])
    phi, grad = posterior.phi_and_grad(w)

    accepted_preburn = 0
    failures_preburn = 0
    total_start = time.perf_counter()

    for _ in range(preburn):
        w, phi, grad, accepted, failed = _cone_transition(
            posterior, w, phi, grad, h, rng, edge_scale
        )
        accepted_preburn += int(accepted)
        failures_preburn += int(failed)

    samples = DrawBuffer(retained)
    accepted_sampling = 0
    failures_sampling = 0
    sampling_start = time.perf_counter()

    for _ in range(retained):
        w, phi, grad, accepted, failed = _cone_transition(
            posterior, w, phi, grad, h, rng, edge_scale
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


CONE_EDGE_METHOD = "ConeMALA_EdgeStep"


def cone_edge_scales(
    posterior: PSDGraphPosterior, w_ref: Array
) -> Tuple[Array, Array]:
    """Congruence-invariant per-edge step multipliers for ConeMALA.

    At the reference point (the MAP) the block of the Riemannian Hessian of Phi
    for edge e, in AIRM normal coordinates W_e^{1/2} exp(V) W_e^{1/2}, is
    V -> (n/2) M_e V M_e with M_e = W_e^{1/2} B_e^T Q^{-1} B_e W_e^{1/2}
    (the gradient terms vanish at a critical point).  Its largest eigenvalue is
    (n/2) mu_max(M_e)^2 with 0 <= mu < 1.  We add alpha, the curvature of the
    Wishart prior at its own mode, so that edges the likelihood barely informs
    are not given unbounded steps, and set

        kappa_e = (n/2) mu_max(M_e)^2 + alpha,   c_e = gm(kappa) / kappa_e,

    so the multipliers have geometric mean 1 and the global h keeps its usual
    meaning.  Under W_e -> A^{-T} W_e A^{-1} (with data, R and the prior
    transformed as in the equivariance proposition) M_e -> U_e M_e U_e^T with
    U_e orthogonal, so kappa_e and c_e are unchanged: the kernel stays exactly
    equivariant.
    """
    w_ref = np.asarray([sym(block) for block in w_ref])
    q_inverse = sym(inv(posterior.q(w_ref)))
    k_blocks = edge_gradient_from_a(q_inverse, posterior.edges, posterior.d)
    kappa = np.empty(posterior.edge_count)
    for edge_index in range(posterior.edge_count):
        root, _ = mat_sqrt_inv_sqrt(w_ref[edge_index])
        mu = eigvalsh(sym(root @ k_blocks[edge_index] @ root))
        kappa[edge_index] = (
            0.5 * posterior.n_train * float(np.max(mu)) ** 2
            + posterior.alpha_logdet
        )
    scales = float(np.exp(np.mean(np.log(kappa)))) / kappa
    return scales, kappa

def cone_reference_point(
    posterior, w0: Array, max_iter: int = 400, tol: float = 1e-5
) -> Array:
    """Approximate mode of the intrinsic potential Phi by Riemannian gradient
    descent on the product AIRM manifold (Armijo backtracking).  Every step is
    congruence-equivariant, so the iterates of a transformed problem are the
    transformed iterates; used only to place the edge-step multipliers."""
    w = np.asarray([project_spd(block) for block in w0])
    phi, grad = posterior.phi_and_grad(w)
    step = 1e-2
    for _ in range(max_iter):
        roots = [mat_sqrt_inv_sqrt(block)[0] for block in w]
        rgrad = [sym(r @ g @ r) for r, g in zip(roots, grad)]
        gnorm2 = float(sum(np.sum(g * g) for g in rgrad))
        if gnorm2 <= tol * tol:
            break
        while True:
            cand = np.asarray([
                sym(r @ expm_sym(-step * g) @ r) for r, g in zip(roots, rgrad)
            ])
            phi_c, grad_c = posterior.phi_and_grad(cand)
            if np.isfinite(phi_c) and phi_c <= phi - 0.5 * step * gnorm2:
                break
            step *= 0.5
            if step < 1e-14:
                return w
        w, phi, grad = cand, phi_c, grad_c
        step *= 1.5
    return w


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


# =============================================================================
# >>> BEGIN RELEASE EXTRAS (byte-identical in all scripts of this release) <<<
#
# 1. Two unconstrained-reparametrization baselines on the *same* posterior:
#      * LogCholeskyMALA  : fixed-step MALA in log-Cholesky coordinates
#      * LogCholeskyNUTS  : Stan-style multinomial NUTS (diagonal metric,
#                           dual-averaging step size, windowed adaptation)
#    Both target exactly the Lebesgue-density posterior used by Euclidean MALA
#    (including the Cholesky Jacobian), so all six samplers target one law.
# 2. Vehtari et al. (2021) diagnostics: rank-normalised split-Rhat,
#    bulk-ESS and tail-ESS, for scalar observables and for every free
#    edge-weight entry (minimum / median over entries).
#
# Nothing in the original ConeMALA / Fisher / Euclidean / RMALA code path is
# modified.
# =============================================================================

EXTRAS_VERSION = "1.0"

LCHOL_METHOD = "LogCholeskyMALA"
NUTS_METHOD = "LogCholeskyNUTS"

# Pilot-tuning centres.  They only seed the bracketing search; the frozen step
# size is whatever the discarded pilot selects.
LCHOL_H_CENTER = 3.0e-3


# -----------------------------------------------------------------------------
# Log-Cholesky target
# -----------------------------------------------------------------------------

class LogCholeskyTarget:
    """Unconstrained log-Cholesky coordinates for the edge weights.

    For every edge  W_e = L_e L_e^T  with L_e lower triangular and
    L_e[i, i] = exp(ell_i).  theta stacks, edge by edge,
        [ell_0, ..., ell_{d-1},  strictly-lower entries of L_e (row-major)].

    The sampled density is the *same* Lebesgue density used by Euclidean MALA,
        pi_dx(W) = exp(-Phi(W)) * prod_e det(W_e)^(-(d+1)/2),
    pushed to theta, which adds the Cholesky Jacobian
        dW_e = 2^d * prod_i L_ii^(d-i+1) dL_e,   dL_ii = L_ii d ell_i,
    i.e. log|J| = sum_i (d - i + 1) ell_i + const  for 0-indexed i.
    """

    def __init__(self, posterior: "PSDGraphPosterior") -> None:
        self.posterior = posterior
        self.d = int(posterior.d)
        self.edge_count = int(posterior.edge_count)
        self.n_off = self.d * (self.d - 1) // 2
        self.block = self.d * (self.d + 1) // 2
        self.dim = self.edge_count * self.block
        self.tril_i, self.tril_j = np.tril_indices(self.d, k=-1)
        self.diag_index = np.arange(self.d)
        self.jacobian_coefficient = (self.d - np.arange(self.d) + 1.0)

    def to_cholesky(self, theta: Array) -> Array:
        th = np.asarray(theta, dtype=float).reshape(self.edge_count, self.block)
        chol = np.zeros((self.edge_count, self.d, self.d), dtype=float)
        chol[:, self.diag_index, self.diag_index] = np.exp(th[:, : self.d])
        chol[:, self.tril_i, self.tril_j] = th[:, self.d:]
        return chol

    def to_weights(self, theta: Array) -> Array:
        chol = self.to_cholesky(theta)
        w = chol @ np.swapaxes(chol, 1, 2)
        return 0.5 * (w + np.swapaxes(w, 1, 2))

    def from_weights(self, w: Array) -> Array:
        chol = np.linalg.cholesky(0.5 * (w + np.swapaxes(w, 1, 2)))
        th = np.empty((self.edge_count, self.block), dtype=float)
        th[:, : self.d] = np.log(chol[:, self.diag_index, self.diag_index])
        th[:, self.d:] = chol[:, self.tril_i, self.tril_j]
        return th.reshape(-1)

    def potential_and_grad(self, theta: Array) -> Tuple[float, Array]:
        """U(theta) = -log pi_theta(theta) (up to a constant) and grad U."""
        chol = self.to_cholesky(theta)
        w = chol @ np.swapaxes(chol, 1, 2)
        w = 0.5 * (w + np.swapaxes(w, 1, 2))
        phi, grad_phi = self.posterior.phi_and_grad(w)
        if not np.isfinite(phi):
            return np.inf, np.zeros(self.dim)

        th = np.asarray(theta, dtype=float).reshape(self.edge_count, self.block)
        ell = th[:, : self.d]
        vol = self.posterior.volume_coefficient
        potential = (
            phi
            + vol * 2.0 * float(np.sum(ell))
            - float(np.sum(self.jacobian_coefficient * ell))
        )
        negative_log_gradient = grad_phi + vol * np.linalg.inv(w)
        grad_chol = 2.0 * (negative_log_gradient @ chol)

        grad = np.empty_like(th)
        grad[:, : self.d] = (
            grad_chol[:, self.diag_index, self.diag_index]
            * chol[:, self.diag_index, self.diag_index]
            - self.jacobian_coefficient
        )
        grad[:, self.d:] = grad_chol[:, self.tril_i, self.tril_j]
        return float(potential), grad.reshape(-1)


# -----------------------------------------------------------------------------
# Fixed-step MALA in log-Cholesky coordinates
# -----------------------------------------------------------------------------

def _lchol_transition(
    target: LogCholeskyTarget,
    theta: Array,
    potential: float,
    grad: Array,
    h: float,
    rng: np.random.Generator,
) -> Tuple[Array, float, Array, bool, bool]:
    mean_forward = theta - h * grad
    proposal = mean_forward + math.sqrt(2.0 * h) * rng.normal(size=theta.size)
    try:
        potential_prop, grad_prop = target.potential_and_grad(proposal)
        if not np.isfinite(potential_prop):
            return theta, potential, grad, False, True
        mean_reverse = proposal - h * grad_prop
        log_q_forward = -np.sum((proposal - mean_forward) ** 2) / (4.0 * h)
        log_q_reverse = -np.sum((theta - mean_reverse) ** 2) / (4.0 * h)
        log_alpha = (
            -potential_prop + log_q_reverse
            - (-potential + log_q_forward)
        )
        if math.log(rng.uniform()) < min(0.0, log_alpha):
            return proposal, potential_prop, grad_prop, True, False
        return theta, potential, grad, False, False
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return theta, potential, grad, False, True


def lchol_mala_chain(
    posterior: "PSDGraphPosterior",
    w0: Array,
    preburn: int,
    retained: int,
    h: float,
    seed: int,
) -> "ChainResult":
    target = LogCholeskyTarget(posterior)
    rng = np.random.default_rng(seed)
    w_start = np.asarray([project_spd(block) for block in w0])
    theta = target.from_weights(w_start)
    potential, grad = target.potential_and_grad(theta)
    w_current = target.to_weights(theta)

    accepted_preburn = failures_preburn = 0
    total_start = time.perf_counter()
    for _ in range(preburn):
        theta, potential, grad, accepted, failed = _lchol_transition(
            target, theta, potential, grad, h, rng
        )
        accepted_preburn += int(accepted)
        failures_preburn += int(failed)

    samples = DrawBuffer(retained)
    accepted_sampling = failures_sampling = 0
    sampling_start = time.perf_counter()
    for _ in range(retained):
        theta, potential, grad, accepted, failed = _lchol_transition(
            target, theta, potential, grad, h, rng
        )
        accepted_sampling += int(accepted)
        failures_sampling += int(failed)
        if accepted:
            w_current = target.to_weights(theta)
        samples.append(w_current.copy())

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
        numerical_failure_rate=failures_sampling / max(retained, 1),
    )


# -----------------------------------------------------------------------------
# Generic discarded-pilot step-size tuner (same protocol as the fixed-step
# methods of the paper: tune outside production, then freeze h).
# -----------------------------------------------------------------------------

def tune_scalar_step(
    method: str,
    run_chain: Callable[[Array, int, int, float, int], "ChainResult"],
    anchors: Sequence[Array],
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
) -> Tuple[float, List[Dict[str, float]]]:
    """Bracket + geometric bisection of a scalar h around the target acceptance.

    Logic mirrors ``tune_fixed_method_h`` of the original sweep script.
    """
    if tune_steps < 10:
        raise ValueError("fixed_tune_steps must be at least 10.")
    use_anchors = list(anchors[: min(tune_chains, len(anchors))])
    if not use_anchors:
        raise ValueError("At least one tuning anchor is required.")

    tested: Dict[str, Dict[str, float]] = {}

    def run_candidate(h: float) -> Dict[str, float]:
        h = float(max(h, 1e-14))
        key = f"{h:.17e}"
        if key in tested:
            return tested[key]
        acceptances: List[float] = []
        invalid_rates: List[float] = []
        runtimes: List[float] = []
        for chain_index, anchor in enumerate(use_anchors):
            result = run_chain(
                anchor, tune_warmup, tune_steps, h, seed_base + chain_index
            )
            acceptances.append(result.acceptance)
            invalid_rates.append(result.invalid_rate)
            runtimes.append(result.runtime_total_sec)
        row = {
            "method": method,
            "h": h,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1) if len(acceptances) > 1 else 0.0
            ),
            "pilot_invalid_rate": float(np.mean(invalid_rates)),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[key] = row
        print(
            f"    {method} pilot h={h:.3e}: "
            f"acc={row['pilot_acceptance']:.3f}, "
            f"invalid={row['pilot_invalid_rate']:.3f}",
            flush=True,
        )
        return row

    for factor in sorted(set(float(f) for f in factors if float(f) > 0.0)):
        run_candidate(max(h_center, 1e-14) * factor)

    lower = target_accept - acceptance_tolerance
    upper = target_accept + acceptance_tolerance

    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] >= lower for row in rows):
            break
        run_candidate(rows[0]["h"] / 2.0)
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] <= upper for row in rows):
            break
        run_candidate(rows[-1]["h"] * 2.0)
    for _ in range(bisection_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        high_accept = [r for r in rows if r["pilot_acceptance"] >= target_accept]
        low_accept = [r for r in rows if r["pilot_acceptance"] <= target_accept]
        if not high_accept or not low_accept:
            break
        left = max(high_accept, key=lambda row: row["h"])
        right = min(low_accept, key=lambda row: row["h"])
        if left["h"] >= right["h"]:
            break
        midpoint = math.sqrt(left["h"] * right["h"])
        if f"{midpoint:.17e}" in tested:
            break
        run_candidate(midpoint)

    rows = sorted(tested.values(), key=lambda row: row["h"])
    admissible = [
        r for r in rows
        if lower <= r["pilot_acceptance"] <= upper
        and r["pilot_invalid_rate"] < 0.95
    ]
    pool = admissible if admissible else rows
    best = min(
        pool,
        key=lambda r: (
            abs(r["pilot_acceptance"] - target_accept),
            r["pilot_invalid_rate"],
        ),
    )
    feasible = not (
        best["pilot_acceptance"] <= 0.01 or best["pilot_invalid_rate"] >= 0.99
    )
    for row in rows:
        row["selected"] = bool(
            feasible
            and math.isclose(row["h"], best["h"], rel_tol=1e-12, abs_tol=1e-15)
        )
        row["tuning_feasible"] = bool(feasible)
    if not feasible:
        raise RuntimeError(
            f"No feasible pilot step size found for {method}. "
            "Inspect the saved tuning grid or expand the factor/bracket range."
        )
    return float(best["h"]), rows


# -----------------------------------------------------------------------------
# NUTS (Stan-style multinomial NUTS, diagonal metric)
# -----------------------------------------------------------------------------

class _NutsTree:
    __slots__ = (
        "left", "right", "proposal", "log_weight", "rho",
        "turning", "diverging", "sum_accept", "n_leaf",
    )

    def __init__(
        self, left, right, proposal, log_weight, rho,
        turning, diverging, sum_accept, n_leaf,
    ) -> None:
        self.left = left
        self.right = right
        self.proposal = proposal
        self.log_weight = log_weight
        self.rho = rho
        self.turning = turning
        self.diverging = diverging
        self.sum_accept = sum_accept
        self.n_leaf = n_leaf


def _kinetic(p: Array, minv: Array) -> float:
    return 0.5 * float(np.dot(minv * p, p))


def _leapfrog(
    target: LogCholeskyTarget,
    state: Tuple[Array, Array, float, Array],
    step: float,
    minv: Array,
) -> Optional[Tuple[Array, Array, float, Array]]:
    q, p, _, g = state
    p_half = p - 0.5 * step * g
    q_new = q + step * minv * p_half
    try:
        u_new, g_new = target.potential_and_grad(q_new)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError):
        return None
    if not np.isfinite(u_new) or not np.all(np.isfinite(g_new)):
        return None
    p_new = p_half - 0.5 * step * g_new
    return q_new, p_new, u_new, g_new


def _nuts_criterion(
    p_sharp_minus: Array, p_sharp_plus: Array, rho: Array
) -> bool:
    return bool(np.dot(p_sharp_plus, rho) > 0.0 and np.dot(p_sharp_minus, rho) > 0.0)


def _nuts_merge_ok(a: _NutsTree, b: _NutsTree, minv: Array) -> bool:
    """Generalised no-U-turn test for adjacent subtrees a (left) and b (right).

    Includes the two extra between-subtree checks used by Stan.
    """
    rho = a.rho + b.rho
    ok = _nuts_criterion(minv * a.left[1], minv * b.right[1], rho)
    rho_ext = a.rho + b.left[1]
    ok = ok and _nuts_criterion(minv * a.left[1], minv * b.left[1], rho_ext)
    rho_ext = b.rho + a.right[1]
    ok = ok and _nuts_criterion(minv * a.right[1], minv * b.right[1], rho_ext)
    return ok


def _nuts_build(
    target: LogCholeskyTarget,
    state: Tuple[Array, Array, float, Array],
    direction: int,
    depth: int,
    step: float,
    minv: Array,
    h0: float,
    rng: np.random.Generator,
    max_delta_h: float,
) -> _NutsTree:
    if depth == 0:
        new_state = _leapfrog(target, state, direction * step, minv)
        if new_state is None:
            return _NutsTree(state, state, state, -np.inf, state[1],
                             False, True, 0.0, 1)
        h_new = new_state[2] + _kinetic(new_state[1], minv)
        if not np.isfinite(h_new):
            return _NutsTree(state, state, state, -np.inf, state[1],
                             False, True, 0.0, 1)
        log_weight = h0 - h_new
        diverging = (h_new - h0) > max_delta_h
        accept = 1.0 if log_weight >= 0.0 else math.exp(log_weight)
        return _NutsTree(
            new_state, new_state, new_state, log_weight, new_state[1].copy(),
            False, bool(diverging), accept, 1,
        )

    first = _nuts_build(
        target, state, direction, depth - 1, step, minv, h0, rng, max_delta_h
    )
    if first.turning or first.diverging:
        return first
    start = first.right if direction > 0 else first.left
    second = _nuts_build(
        target, start, direction, depth - 1, step, minv, h0, rng, max_delta_h
    )
    n_leaf = first.n_leaf + second.n_leaf
    sum_accept = first.sum_accept + second.sum_accept
    if second.turning or second.diverging:
        return _NutsTree(
            first.left, first.right, first.proposal, first.log_weight,
            first.rho, second.turning, second.diverging, sum_accept, n_leaf,
        )

    log_weight = float(np.logaddexp(first.log_weight, second.log_weight))
    if math.log(rng.uniform()) < second.log_weight - log_weight:
        proposal = second.proposal
    else:
        proposal = first.proposal
    if direction > 0:
        left_tree, right_tree = first, second
    else:
        left_tree, right_tree = second, first
    merged = _NutsTree(
        left_tree.left, right_tree.right, proposal, log_weight,
        first.rho + second.rho, False, False, sum_accept, n_leaf,
    )
    merged.turning = not _nuts_merge_ok(left_tree, right_tree, minv)
    return merged


def _nuts_transition(
    target: LogCholeskyTarget,
    q: Array,
    potential: float,
    grad: Array,
    step: float,
    minv: Array,
    max_depth: int,
    rng: np.random.Generator,
    max_delta_h: float = 1000.0,
) -> Dict[str, object]:
    p0 = rng.normal(size=q.size) / np.sqrt(minv)
    h0 = potential + _kinetic(p0, minv)
    z0 = (q, p0, potential, grad)

    cur = _NutsTree(z0, z0, z0, 0.0, p0.copy(), False, False, 0.0, 0)
    proposal = z0
    log_sum_weight = 0.0
    depth = 0
    n_leaf = 0
    sum_accept = 0.0
    diverged = False

    while depth < max_depth:
        direction = 1 if rng.uniform() < 0.5 else -1
        start = cur.right if direction > 0 else cur.left
        sub = _nuts_build(
            target, start, direction, depth, step, minv, h0, rng, max_delta_h
        )
        n_leaf += sub.n_leaf
        sum_accept += sub.sum_accept
        if sub.diverging:
            diverged = True
            break
        if sub.turning:
            break
        # Biased progressive sampling at the trajectory level.
        if sub.log_weight > log_sum_weight or (
            math.log(rng.uniform()) < sub.log_weight - log_sum_weight
        ):
            proposal = sub.proposal
        log_sum_weight = float(np.logaddexp(log_sum_weight, sub.log_weight))
        if direction > 0:
            a, b = cur, sub
        else:
            a, b = sub, cur
        cur = _NutsTree(
            a.left, b.right, proposal, log_sum_weight, a.rho + b.rho,
            False, False, 0.0, 0,
        )
        depth += 1
        if not _nuts_merge_ok(a, b, minv):
            break

    return {
        "q": proposal[0],
        "potential": proposal[2],
        "grad": proposal[3],
        "accept_stat": sum_accept / max(n_leaf, 1),
        "n_leapfrog": n_leaf,
        "depth": depth,
        "diverged": diverged,
        "saturated": depth >= max_depth,
    }


def _nuts_initial_step(
    target: LogCholeskyTarget,
    q: Array,
    potential: float,
    grad: Array,
    step: float,
    minv: Array,
    rng: np.random.Generator,
) -> float:
    """Stan's init_stepsize heuristic (double/halve until accept crosses 0.8)."""
    log_threshold = math.log(0.8)

    def delta_h(eps: float) -> float:
        p = rng.normal(size=q.size) / np.sqrt(minv)
        h_old = potential + _kinetic(p, minv)
        new = _leapfrog(target, (q, p, potential, grad), eps, minv)
        if new is None:
            return -np.inf
        h_new = new[2] + _kinetic(new[1], minv)
        return h_old - h_new if np.isfinite(h_new) else -np.inf

    direction = 1 if delta_h(step) > log_threshold else -1
    for _ in range(100):
        value = delta_h(step)
        if direction == 1 and not (value > log_threshold):
            break
        if direction == -1 and not (value < log_threshold):
            break
        step = step * 2.0 if direction == 1 else step / 2.0
        if step > 1e7 or step < 1e-12:
            break
    return float(step)


class _DualAveraging:
    def __init__(
        self, step: float, target_accept: float,
        gamma: float = 0.05, t0: float = 10.0, kappa: float = 0.75,
    ) -> None:
        self.target_accept = target_accept
        self.gamma, self.t0, self.kappa = gamma, t0, kappa
        self.restart(step)

    def restart(self, step: float) -> None:
        self.mu = math.log(10.0 * step)
        self.counter = 0
        self.s_bar = 0.0
        self.x_bar = 0.0
        self.step = step

    def update(self, accept_stat: float) -> float:
        self.counter += 1
        accept_stat = min(1.0, accept_stat)
        eta = 1.0 / (self.counter + self.t0)
        self.s_bar = (1.0 - eta) * self.s_bar + eta * (self.target_accept - accept_stat)
        x = self.mu - self.s_bar * math.sqrt(self.counter) / self.gamma
        x_eta = self.counter ** (-self.kappa)
        self.x_bar = (1.0 - x_eta) * self.x_bar + x_eta * x
        self.step = math.exp(x)
        return self.step

    def final_step(self) -> float:
        return math.exp(self.x_bar)


class _StanWindows:
    """Stan's windowed adaptation schedule for the diagonal metric."""

    def __init__(
        self, num_warmup: int, init_buffer: int = 75,
        term_buffer: int = 50, base_window: int = 25,
    ) -> None:
        self.num_warmup = num_warmup
        self.enabled = num_warmup >= 20
        if init_buffer + base_window + term_buffer > num_warmup:
            init_buffer = int(0.15 * num_warmup)
            term_buffer = int(0.1 * num_warmup)
            base_window = num_warmup - (init_buffer + term_buffer)
        self.init_buffer = init_buffer
        self.term_buffer = term_buffer
        self.base_window = base_window
        self.window_size = base_window
        self.next_window = init_buffer + base_window - 1
        self.counter = 0

    def in_window(self) -> bool:
        return (
            self.enabled
            and self.counter >= self.init_buffer
            and self.counter < self.num_warmup - self.term_buffer
            and self.counter != self.num_warmup
        )

    def end_window(self) -> bool:
        return (
            self.enabled
            and self.counter == self.next_window
            and self.counter != self.num_warmup
        )

    def compute_next_window(self) -> None:
        if self.next_window == self.num_warmup - self.term_buffer - 1:
            return
        self.window_size *= 2
        self.next_window = self.counter + self.window_size
        if self.next_window == self.num_warmup - self.term_buffer - 1:
            return
        boundary = self.next_window + 2 * self.window_size
        if boundary >= self.num_warmup - self.term_buffer:
            self.next_window = self.num_warmup - self.term_buffer - 1


class _Welford:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.restart()

    def restart(self) -> None:
        self.n = 0
        self.mean = np.zeros(self.dim)
        self.m2 = np.zeros(self.dim)

    def add(self, x: Array) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def variance(self) -> Array:
        return self.m2 / max(self.n - 1, 1)


def lchol_nuts_chain(
    posterior: "PSDGraphPosterior",
    w0: Array,
    warmup: int,
    retained: int,
    seed: int,
    target_accept: float = 0.8,
    max_depth: int = 10,
    init_step: float = 0.1,
    adapt_metric: bool = True,
) -> "ChainResult":
    """NUTS in log-Cholesky coordinates; warm-up cost counts as overhead.

    ``adapt_metric=False`` keeps the identity metric (only the step size is
    adapted); this is the metric ablation, not the default baseline.

    ``runtime_total_sec`` = warm-up (step size + diagonal metric adaptation)
    + retained transitions; ``runtime_sampling_sec`` = retained only.
    """
    target = LogCholeskyTarget(posterior)
    rng = np.random.default_rng(seed)
    w_start = np.asarray([project_spd(block) for block in w0])
    q = target.from_weights(w_start)
    potential, grad = target.potential_and_grad(q)
    minv = np.ones(q.size)

    total_start = time.perf_counter()
    step = _nuts_initial_step(target, q, potential, grad, init_step, minv, rng)
    dual = _DualAveraging(step, target_accept)
    windows = _StanWindows(warmup)
    welford = _Welford(q.size)

    warm_accept: List[float] = []
    for _ in range(warmup):
        out = _nuts_transition(
            target, q, potential, grad, step, minv, max_depth, rng
        )
        q, potential, grad = out["q"], out["potential"], out["grad"]
        warm_accept.append(float(out["accept_stat"]))
        step = dual.update(float(out["accept_stat"]))
        if adapt_metric and windows.in_window():
            welford.add(q)
        if adapt_metric and windows.end_window():
            windows.compute_next_window()
            n = welford.n
            var = welford.variance()
            var = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            minv = var
            welford.restart()
            windows.counter += 1
            step = _nuts_initial_step(target, q, potential, grad, step, minv, rng)
            dual.restart(step)
        else:
            windows.counter += 1
    if warmup > 0:
        step = dual.final_step()

    samples = DrawBuffer(retained)
    accept_stats: List[float] = []
    n_leapfrogs: List[int] = []
    depths: List[int] = []
    divergences = saturated = 0
    w_current = target.to_weights(q)
    sampling_start = time.perf_counter()
    for _ in range(retained):
        out = _nuts_transition(
            target, q, potential, grad, step, minv, max_depth, rng
        )
        moved = out["q"] is not q
        q, potential, grad = out["q"], out["potential"], out["grad"]
        if moved:
            w_current = target.to_weights(q)
        samples.append(w_current.copy())
        accept_stats.append(float(out["accept_stat"]))
        n_leapfrogs.append(int(out["n_leapfrog"]))
        depths.append(int(out["depth"]))
        divergences += int(out["diverged"])
        saturated += int(out["saturated"])
    runtime_sampling = time.perf_counter() - sampling_start
    runtime_total = time.perf_counter() - total_start

    return ChainResult(
        samples=np.asarray(samples),
        acceptance=float(np.mean(accept_stats)) if accept_stats else np.nan,
        preburn_acceptance=float(np.mean(warm_accept)) if warm_accept else np.nan,
        runtime_total_sec=float(runtime_total),
        runtime_sampling_sec=float(runtime_sampling),
        h_final=float(step),
        invalid_rate=0.0,
        preburn_invalid_rate=0.0,
        numerical_failure_rate=divergences / max(retained, 1),
        n_leapfrog_mean=float(np.mean(n_leapfrogs)) if n_leapfrogs else np.nan,
        treedepth_mean=float(np.mean(depths)) if depths else np.nan,
        treedepth_saturated_rate=saturated / max(retained, 1),
        divergence_rate=divergences / max(retained, 1),
    )


# -----------------------------------------------------------------------------
# Vehtari et al. (2021) diagnostics, vectorised over coordinates
# -----------------------------------------------------------------------------
#
# Arrays are laid out as (B coordinates, C chains, n draws).  Every function
# works on a batch of coordinates at once; the FFT autocovariance and the
# Geyer initial-monotone-sequence truncation are fully vectorised.

def _split_chains(x: Array) -> Array:
    half = x.shape[2] // 2
    return np.concatenate([x[:, :, :half], x[:, :, half: 2 * half]], axis=1)


def _rank_normalize(x: Array) -> Array:
    from scipy.special import ndtri
    from scipy.stats import rankdata

    b, m, n = x.shape
    flat = x.reshape(b, m * n)
    ranks = rankdata(flat, axis=1)
    z = ndtri((ranks - 0.375) / (m * n + 0.25))
    return z.reshape(b, m, n)


def _rhat_presplit(x: Array) -> Array:
    """Classical Rhat of already-split chains, shape (B, M, N)."""
    b, m, n = x.shape
    if m < 2 or n < 2:
        return np.full(b, np.nan)
    means = x.mean(axis=2)
    within = x.var(axis=2, ddof=1).mean(axis=1)
    between = n * means.var(axis=1, ddof=1)
    var_plus = ((n - 1.0) / n) * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_plus / within)
    degenerate = within <= 1e-15
    rhat = np.where(
        degenerate, np.where(between > 1e-15, np.inf, np.nan), rhat
    )
    return rhat


def _ess_presplit(x: Array) -> Array:
    """Multi-chain ESS (Stan / Vehtari 2021) of already-split chains."""
    from scipy.fft import irfft, next_fast_len, rfft

    b, m, n = x.shape
    if n < 4:
        return np.full(b, np.nan)
    centred = x - x.mean(axis=2, keepdims=True)
    nfft = next_fast_len(2 * n)
    spectrum = rfft(centred, n=nfft, axis=2)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).sum(axis=1)
    acov = irfft(power, n=nfft, axis=1)[:, :n] / (n * m)
    chain_mean = x.mean(axis=2)
    mean_var = acov[:, 0] * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n
    if m > 1:
        var_plus = var_plus + chain_mean.var(axis=1, ddof=1)
    scale = np.maximum(np.abs(x).max(axis=(1, 2)), 1.0)
    ok = var_plus > 1e-24 * scale ** 2
    safe = np.where(ok, var_plus, 1.0)
    rho = 1.0 - (mean_var[:, None] - acov) / safe[:, None]

    npairs = n // 2
    pairs = rho[:, 0: 2 * npairs: 2] + rho[:, 1: 2 * npairs: 2]
    negative = pairs < 0.0
    negative[:, 0] = False
    has_negative = negative.any(axis=1)
    k_stop = np.where(has_negative, negative.argmax(axis=1), npairs)
    monotone = np.minimum.accumulate(pairs, axis=1)
    keep = np.arange(npairs)[None, :] < k_stop[:, None]
    pair_sum = (monotone * keep).sum(axis=1)
    even_next = rho[np.arange(b), np.minimum(2 * k_stop, n - 1)]
    extra = np.where(k_stop < npairs, np.maximum(even_next, 0.0), 0.0)
    tau = -1.0 + 2.0 * pair_sum + extra
    tau = np.maximum(tau, 1.0 / np.log10(m * n))
    ess = (m * n) / tau
    return np.where(ok, ess, 0.0)


def rank_diagnostics_batch(x: Array) -> Dict[str, Array]:
    """Rank-normalised split-Rhat, bulk-ESS and tail-ESS for x of shape (B,C,n)."""
    xs = _split_chains(x)
    b = xs.shape[0]
    z_bulk = _rank_normalize(xs)
    rhat_bulk = _rhat_presplit(z_bulk)
    median = np.median(xs.reshape(b, -1), axis=1)
    z_fold = _rank_normalize(np.abs(xs - median[:, None, None]))
    rhat_fold = _rhat_presplit(z_fold)
    rhat = np.fmax(rhat_bulk, rhat_fold)
    ess_bulk = _ess_presplit(z_bulk)
    q05, q95 = np.quantile(xs.reshape(b, -1), [0.05, 0.95], axis=1)
    ess_q05 = _ess_presplit((xs <= q05[:, None, None]).astype(float))
    ess_q95 = _ess_presplit((xs <= q95[:, None, None]).astype(float))
    return {
        "rhat_rank": rhat,
        "rhat_bulk": rhat_bulk,
        "rhat_fold": rhat_fold,
        "ess_bulk": ess_bulk,
        "ess_tail": np.minimum(ess_q05, ess_q95),
    }


def mcmc_rank_diagnostics(
    chains: Sequence[Array], batch_size: int = 48
) -> Dict[str, Array]:
    """Diagnostics for coordinates stored as per-chain arrays (n_draws, B)."""
    arrays = [np.asarray(c) for c in chains]
    arrays = [a.reshape(a.shape[0], -1) for a in arrays]
    n = min(a.shape[0] for a in arrays)
    n = 2 * (n // 2)
    total = arrays[0].shape[1]
    keys = ("rhat_rank", "rhat_bulk", "rhat_fold", "ess_bulk", "ess_tail")
    out = {key: np.empty(total) for key in keys}
    if n < 20 or len(arrays) < 2:
        for key in keys:
            out[key][:] = np.nan
        return out
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        block = np.stack(
            [a[:n, start:stop].astype(np.float64) for a in arrays], axis=0
        )
        block = np.transpose(block, (2, 0, 1))
        result = rank_diagnostics_batch(block)
        for key in keys:
            out[key][start:stop] = result[key]
    return out


def edge_entry_matrix(samples: Array) -> Array:
    """(n, E, d, d) draws -> (n, E*d(d+1)/2) upper-triangular entries (float32)."""
    n, e, d, _ = samples.shape
    iu = np.triu_indices(d)
    return samples[:, :, iu[0], iu[1]].reshape(n, -1).astype(np.float32)


def edge_entry_labels(edge_count: int, d: int) -> List[Tuple[int, int, int]]:
    iu = np.triu_indices(d)
    return [
        (e, int(i), int(j))
        for e in range(edge_count)
        for i, j in zip(iu[0], iu[1])
    ]


SCALAR_DIAGNOSTIC_OBSERVABLES = ("rel_W_error", "test_nll", "logdet_Q", "Phi")


def extended_diagnostic_columns(
    observable_chains: Sequence[Dict[str, Array]],
    edge_chains: Optional[Sequence[Array]],
    runtime_total: float,
    runtime_sampling: float,
    edge_csv_path: Optional[Path] = None,
    edge_labels: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict[str, float]:
    """Summary-row columns for rank-Rhat, bulk/tail ESS and edge-entry minima.

    ``lambda_min_Q`` is skipped: it equals the constant ridge value by
    construction (the block Laplacian annihilates node-constant vectors).
    """
    columns: Dict[str, float] = {}
    r_total = max(runtime_total, 1e-12)
    r_samp = max(runtime_sampling, 1e-12)
    worst_rhat = [0.0]
    worst_ess = [np.inf]

    for name in SCALAR_DIAGNOSTIC_OBSERVABLES:
        series = [obs[name].reshape(-1, 1) for obs in observable_chains]
        diag = mcmc_rank_diagnostics(series)
        rhat, ess_b, ess_t = (
            float(diag["rhat_rank"][0]),
            float(diag["ess_bulk"][0]),
            float(diag["ess_tail"][0]),
        )
        columns[f"{name}_Rhat_rank"] = rhat
        columns[f"{name}_ESS_bulk"] = ess_b
        columns[f"{name}_ESS_tail"] = ess_t
        columns[f"{name}_ESSsec_bulk_sampling"] = ess_b / r_samp
        columns[f"{name}_ESSsec_bulk_total"] = ess_b / r_total
        columns[f"{name}_ESSsec_tail_sampling"] = ess_t / r_samp
        columns[f"{name}_ESSsec_tail_total"] = ess_t / r_total
        if np.isfinite(rhat) or np.isinf(rhat):
            worst_rhat.append(rhat)
        if np.isfinite(ess_b):
            worst_ess.append(min(ess_b, ess_t) if np.isfinite(ess_t) else ess_b)

    if edge_chains is not None:
        diag = mcmc_rank_diagnostics(edge_chains)
        rhat = diag["rhat_rank"]
        ess_b = diag["ess_bulk"]
        ess_t = diag["ess_tail"]
        finite_rhat = rhat[np.isfinite(rhat) | np.isinf(rhat)]
        columns["edge_n_entries"] = int(rhat.size)
        columns["edge_Rhat_rank_max"] = (
            float(np.max(finite_rhat)) if finite_rhat.size else np.nan
        )
        columns["edge_Rhat_rank_median"] = float(np.nanmedian(rhat))
        columns["edge_frac_Rhat_rank_gt_1p01"] = float(np.mean(rhat > 1.01))
        columns["edge_frac_Rhat_rank_gt_1p05"] = float(np.mean(rhat > 1.05))
        columns["edge_ESS_bulk_min"] = float(np.nanmin(ess_b))
        columns["edge_ESS_bulk_median"] = float(np.nanmedian(ess_b))
        columns["edge_ESS_tail_min"] = float(np.nanmin(ess_t))
        columns["edge_ESS_tail_median"] = float(np.nanmedian(ess_t))
        columns["edge_ESSsec_bulk_min_sampling"] = columns["edge_ESS_bulk_min"] / r_samp
        columns["edge_ESSsec_bulk_min_total"] = columns["edge_ESS_bulk_min"] / r_total
        columns["edge_ESSsec_tail_min_sampling"] = columns["edge_ESS_tail_min"] / r_samp
        columns["edge_ESSsec_tail_min_total"] = columns["edge_ESS_tail_min"] / r_total
        columns["edge_ESSsec_bulk_median_sampling"] = columns["edge_ESS_bulk_median"] / r_samp
        worst_rhat.append(columns["edge_Rhat_rank_max"])
        worst_ess.append(min(columns["edge_ESS_bulk_min"], columns["edge_ESS_tail_min"]))
        if edge_csv_path is not None and edge_labels is not None:
            pd.DataFrame(
                {
                    "edge": [t[0] for t in edge_labels],
                    "row": [t[1] for t in edge_labels],
                    "col": [t[2] for t in edge_labels],
                    "Rhat_rank": rhat,
                    "ESS_bulk": ess_b,
                    "ESS_tail": ess_t,
                }
            ).to_csv(edge_csv_path, index=False)

    columns["all_Rhat_rank_max"] = float(np.max(worst_rhat))
    columns["all_ESS_min"] = float(np.min(worst_ess)) if np.isfinite(np.min(worst_ess)) else np.nan
    return columns


def nuts_extra_columns(chain_results: Sequence["ChainResult"]) -> Dict[str, float]:
    def avg(name: str) -> float:
        values = [getattr(r, name) for r in chain_results]
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(values)) if values else np.nan

    return {
        "n_leapfrog_mean": avg("n_leapfrog_mean"),
        "treedepth_mean": avg("treedepth_mean"),
        "treedepth_saturated_rate": avg("treedepth_saturated_rate"),
        "divergence_rate": avg("divergence_rate"),
    }



# -----------------------------------------------------------------------------
# Plot styles, CLI, optional parallel chains, tables and figures
# -----------------------------------------------------------------------------

METHOD_STYLE: Dict[str, Dict[str, str]] = {
    "ConeMALA": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "FisherAdaptiveMALA": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "EuclideanMALA": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "GenericRMALA": {"color": "#d62728", "marker": "D", "linestyle": ":"},
    "LogCholeskyMALA": {"color": "#9467bd", "marker": "v", "linestyle": "-"},
    "LogCholeskyNUTS": {"color": "#8c564b", "marker": "P", "linestyle": "--"},
}


def parse_method_list(
    text: Optional[str],
    allowed: Sequence[str],
    default: Optional[Sequence[str]] = None,
    groups: Optional[Dict[str, Sequence[str]]] = None,
) -> List[str]:
    """Parse --methods.  ``all`` (or empty) -> ``default`` (else ``allowed``);
    group keywords such as ``ablations`` expand to their members."""
    if text is None or not str(text).strip() or str(text).strip().lower() == "all":
        return list(default if default is not None else allowed)
    chosen: List[str] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "all":
            chosen.extend(default if default is not None else allowed)
        elif groups and item.lower() in groups:
            chosen.extend(groups[item.lower()])
        else:
            chosen.append(item)
    chosen = list(dict.fromkeys(chosen))
    unknown = [item for item in chosen if item not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown method(s) {unknown}; choose from {list(allowed)}."
        )
    return chosen


def add_release_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "release extras: baselines, diagnostics, run control"
    )
    group.add_argument(
        "--methods", type=str, default="all",
        help=(
            "Comma-separated subset of methods to run (default: the six main "
            "methods). Extra keyword 'ablations' adds LogCholeskyMALA_DiagMetric "
            "and LogCholeskyNUTS_IdentityMetric (e.g. --methods all,ablations). "
            "Finished methods are stored as row_<method>.csv and re-used by "
            "--resume, so long runs (e.g. NUTS) can be staged."
        ),
    )
    group.add_argument(
        "--lchol_h", type=float, default=None,
        help="Fixed log-Cholesky MALA step; default is discarded-pilot tuning.",
    )
    group.add_argument("--lchol_h_center", type=float, default=LCHOL_H_CENTER)
    group.add_argument("--lchol_target_accept", type=float, default=0.574)
    group.add_argument("--lchol_acceptance_tolerance", type=float, default=0.08)
    group.add_argument("--lchol_tune_warmup", type=int, default=100)
    group.add_argument("--lchol_tune_steps", type=int, default=400)
    group.add_argument("--lchol_tune_chains", type=int, default=2)
    group.add_argument("--lchol_grid_factors", type=str, default="0.125,0.25,0.5,1,2,4")
    group.add_argument("--lchol_max_bracket_steps", type=int, default=20)
    group.add_argument("--lchol_bisection_steps", type=int, default=6)
    group.add_argument(
        "--nuts_warmup", type=int, default=1000,
        help="Discarded NUTS warm-up transitions (step size + diagonal metric).",
    )
    group.add_argument("--nuts_target_accept", type=float, default=0.8)
    group.add_argument("--nuts_max_depth", type=int, default=10)
    group.add_argument(
        "--skip_edge_diagnostics", action="store_true",
        help="Skip the per-entry rank-Rhat / bulk-ESS / tail-ESS pass.",
    )
    group.add_argument(
        "--chain_workers", type=int, default=1,
        help=(
            "Run the chains of the two new baselines in parallel processes. "
            "Per-chain wall-clock times then include CPU contention; use 1 "
            "for the final timing tables."
        ),
    )


def _run_release_chain(spec: Tuple) -> "ChainResult":
    (method, posterior, w0, warm, retained, h, seed, nuts_kwargs, limit_blas) = spec
    nuts_kwargs = dict(nuts_kwargs)
    mala_target_accept = float(nuts_kwargs.pop("mala_target_accept", 0.574))
    context = None
    if limit_blas:
        try:
            from threadpoolctl import threadpool_limits

            context = threadpool_limits(limits=1)
        except Exception:
            context = None
    try:
        if method == LCHOL_METHOD:
            return lchol_mala_chain(posterior, w0, warm, retained, h, seed)
        if method == LCHOL_DIAG_METHOD:
            return lchol_mala_diag_chain(
                posterior, w0, warm, retained, h, seed,
                target_accept=mala_target_accept,
            )
        if method == NUTS_IDENTITY_METHOD:
            return lchol_nuts_chain(
                posterior, w0, warm, retained, seed,
                adapt_metric=False, **nuts_kwargs
            )
        return lchol_nuts_chain(
            posterior, w0, warm, retained, seed, **nuts_kwargs
        )
    finally:
        if context is not None:
            context.unregister()


def run_release_method_chains(
    method: str,
    posterior: "PSDGraphPosterior",
    initial_states: Sequence[Array],
    preburn: int,
    retained: int,
    h: float,
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> List["ChainResult"]:
    nuts_kwargs = {
        "target_accept": float(args.nuts_target_accept),
        "max_depth": int(args.nuts_max_depth),
        "mala_target_accept": float(args.lchol_target_accept),
    }
    warm = (
        int(args.nuts_warmup)
        if method in (NUTS_METHOD, NUTS_IDENTITY_METHOD)
        else int(preburn)
    )
    workers = max(int(args.chain_workers), 1)
    specs = [
        (method, posterior, initial_states[i], warm, retained, h,
         int(seeds[i]), nuts_kwargs, workers > 1)
        for i in range(len(initial_states))
    ]
    if workers == 1:
        results = []
        for spec in specs:
            results.append(_run_release_chain(spec))
        return results
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        return list(pool.map(_run_release_chain, specs))


def tune_lchol_step(
    posterior: "PSDGraphPosterior",
    anchors: Sequence[Array],
    args: argparse.Namespace,
    seed_base: int,
) -> Tuple[float, List[Dict[str, float]]]:
    factors = tuple(
        float(v.strip()) for v in args.lchol_grid_factors.split(",") if v.strip()
    )
    return tune_scalar_step(
        LCHOL_METHOD,
        lambda anchor, warm, steps, h, seed: lchol_mala_chain(
            posterior, anchor, warm, steps, h, seed
        ),
        anchors,
        float(args.lchol_h_center),
        float(args.lchol_target_accept),
        float(args.lchol_acceptance_tolerance),
        int(args.lchol_tune_warmup),
        int(args.lchol_tune_steps),
        int(args.lchol_tune_chains),
        factors,
        int(args.lchol_max_bracket_steps),
        int(args.lchol_bisection_steps),
        seed_base,
    )


def write_combined_diagnostics_table(
    combined: pd.DataFrame,
    path: Path,
    group_col: str,
    group_symbol: str,
    caption_tail: str = "",
) -> None:
    """Rank-normalised Rhat and bulk/tail ESS (min over edge entries)."""
    needed = {"edge_ESS_bulk_min", "edge_Rhat_rank_max", "all_Rhat_rank_max"}
    if not needed.issubset(combined.columns):
        return
    blocks: List[str] = []
    for value in sorted(combined[group_col].unique()):
        subset = (
            combined[combined[group_col] == value]
            .set_index("Method")
            .reindex(METHOD_ORDER)
        )
        blocks.append(
            rf"\multicolumn{{7}}{{l}}{{\emph{{${group_symbol}={int(value)}$}}}} \\"
        )
        for method in METHOD_ORDER:
            row = subset.loc[method]
            blocks.append(
                f"{METHOD_LABELS[method]}"
                f" & {latex_value(row.get('all_Rhat_rank_max', np.nan), 3)}"
                f" & {latex_value(row.get('edge_frac_Rhat_rank_gt_1p01', np.nan), 3)}"
                f" & {latex_value(row.get('edge_ESS_bulk_min', np.nan), 1)}"
                f" & {latex_value(row.get('edge_ESS_tail_min', np.nan), 1)}"
                f" & {latex_value(row.get('edge_ESSsec_bulk_min_total', np.nan), 3)}"
                f" & {latex_value(row.get('edge_ESSsec_tail_min_total', np.nan), 3)}"
                r" \\"
            )
        blocks.append(r"\addlinespace")
    content = rf"""
\begin{{table*}}[t]
\centering
\caption{{Rank-normalised diagnostics (Vehtari et al., 2021). $\max\widehat R$
is the largest rank-normalised split-$\widehat R$ over the scalar observables
and all free edge-weight entries; ``frac.'' is the fraction of entries with
$\widehat R>1.01$. ESS columns report the minimum over entries of bulk- and
tail-ESS pooled over chains; ESS/s uses end-to-end runtime.{caption_tail}}}
\label{{tab:rank_diagnostics}}
\scriptsize
\begin{{tabular}}{{lcccccc}}
\toprule
Method & $\max\widehat R$ & frac.\ $>1.01$ & min bulk-ESS & min tail-ESS
& min bulk-ESS/s & min tail-ESS/s \\
\midrule
{chr(10).join(blocks)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip() + "\n"
    path.write_text(content, encoding="utf-8")


def save_diagnostic_scaling_plots(
    combined: pd.DataFrame,
    outdir: Path,
    xcol: str,
    xlabel: str,
    stem: str,
) -> None:
    """Three-panel figure of the new diagnostics versus d or m."""
    import matplotlib.pyplot as plt

    if "edge_ESSsec_bulk_min_sampling" not in combined.columns:
        return
    set_large_plot_fonts()
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("edge_ESSsec_bulk_min_sampling", "(a) Min bulk-ESS/s over edge entries", True),
        ("edge_ESSsec_tail_min_sampling", "(b) Min tail-ESS/s over edge entries", True),
        ("all_Rhat_rank_max", r"(c) Max rank-normalised $\widehat R$", False),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(19.0, 5.2), constrained_layout=True)
    for axis, (column, title, logy) in zip(axes, panels):
        for method in METHOD_ORDER:
            subset = combined[combined["Method"] == method].sort_values(xcol)
            subset = subset[np.isfinite(subset[column])]
            subset = subset[subset[column] > 0] if logy else subset
            if subset.empty:
                continue
            axis.plot(
                subset[xcol], subset[column], linewidth=2.2, markersize=7,
                label=METHOD_LABELS[method], **METHOD_STYLE[method],
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_xticks(sorted(combined[xcol].unique()))
        axis.grid(True, alpha=0.3, which="both")
        if logy:
            axis.set_yscale("log")
        else:
            axis.axhline(1.01, linestyle=":", linewidth=1.3, color="0.3")
            axis.axhline(1.05, linestyle="--", linewidth=1.3, color="0.3")
            if np.nanmax(combined[column].replace(np.inf, np.nan)) > 5:
                axis.set_yscale("log")
    axes[0].legend(fontsize=10)
    for ext in ("pdf", "png"):
        figure.savefig(plot_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# >>> END RELEASE EXTRAS <<<
# =============================================================================


# =============================================================================
# Experiment settings, tables, and plotting
# =============================================================================

M_FIXED = 50

# d=5 reuses the original m=50 paper settings exactly.  For other d values,
# these are only pilot centers; discarded pilot tuning brackets them before
# the production kernel is frozen.
D5_REFERENCE_SETTINGS: Dict[str, float] = {
    "preburn": 4000,
    "retained": 12000,
    "h_cone": 6.0e-3,
    "h_euclid": 2.0e-5,
    "h_rmala": 1.0e-4,
    "h_fisher": 5.0e-7,
}

# Fisher-adaptive MALA updates its preconditioner square root at every
# discarded adaptive iteration, as in Algorithm 1 of Titsias (NeurIPS 2023).
# The O(q^2) rank-one update has the same order of cost as the preconditioned
# proposal itself, and adaptation is discarded.
FISHER_UPDATE_INTERVAL_DEFAULT = 1


METHOD_ORDER = [
    "ConeMALA",
    "FisherAdaptiveMALA",
    "EuclideanMALA",
    "GenericRMALA",
    LCHOL_METHOD,
    NUTS_METHOD,
]


METHOD_LABELS = {
    "ConeMALA": "ConeMALA",
    "FisherAdaptiveMALA": "Fisher-adaptive MALA",
    "EuclideanMALA": "Euclidean MALA",
    "GenericRMALA": "Generic RMALA",
    "LogCholeskyMALA": "Log-Cholesky MALA",
    "LogCholeskyNUTS": "Log-Cholesky NUTS",
}


# =============================================================================
# >>> BEGIN RELEASE ABLATIONS (d-sweep script only) <<<
#
# Tiers used when reporting (column ``tier``; ``role`` marks ablations):
#   T1_scalar_h         fixed kernel, ONE scalar step size frozen by a discarded
#                       pilot: ConeMALA, EuclideanMALA, GenericRMALA, LogCholeskyMALA
#   T2_adaptive_warmup  the warm-up learns a metric/preconditioner and the step
#                       size, then everything is frozen: FisherAdaptiveMALA,
#                       LogCholeskyNUTS (standard Stan adaptation, the practitioner
#                       default), and the two ablations below.
# Ablations separate the effect of the metric from the effect of the dynamics:
#   * LogCholeskyMALA_DiagMetric    : MALA + the same windowed diagonal metric
#                                     and dual-averaged step size as NUTS
#   * LogCholeskyNUTS_IdentityMetric: NUTS with identity metric (step size only)
# Warm-up cost (metric + step-size adaptation) is INCLUDED in end-to-end ESS/s,
# exactly as for NUTS; only the scalar-h pilots of tier T1 are excluded.
# Every method targets the same Lebesgue-density posterior.
# =============================================================================

LCHOL_DIAG_METHOD = "LogCholeskyMALA_DiagMetric"
NUTS_IDENTITY_METHOD = "LogCholeskyNUTS_IdentityMetric"
ABLATION_METHODS = [LCHOL_DIAG_METHOD, NUTS_IDENTITY_METHOD]
ALL_METHODS = list(METHOD_ORDER) + ABLATION_METHODS

METHOD_TIER: Dict[str, str] = {
    "ConeMALA": "T1_scalar_h",
    "EuclideanMALA": "T1_scalar_h",
    "GenericRMALA": "T1_scalar_h",
    LCHOL_METHOD: "T1_scalar_h",
    "FisherAdaptiveMALA": "T2_adaptive_warmup",
    NUTS_METHOD: "T2_adaptive_warmup",
    LCHOL_DIAG_METHOD: "T2_adaptive_warmup",
    NUTS_IDENTITY_METHOD: "T2_adaptive_warmup",
}
METHOD_LABELS.update({
    LCHOL_DIAG_METHOD: "Log-Cholesky MALA + diag metric",
    NUTS_IDENTITY_METHOD: "Log-Cholesky NUTS, identity metric",
})
METHOD_STYLE.update({
    LCHOL_DIAG_METHOD: {"color": "#e377c2", "marker": "X", "linestyle": "-"},
    NUTS_IDENTITY_METHOD: {"color": "#7f7f7f", "marker": "*", "linestyle": "--"},
})


def _lchol_diag_transition(
    target: LogCholeskyTarget,
    theta: Array,
    potential: float,
    grad: Array,
    h: float,
    minv: Array,
    rng: np.random.Generator,
) -> Tuple[Array, float, Array, bool, bool, float]:
    """Diagonally preconditioned MALA: theta' = theta - h P grad U + sqrt(2h P) xi.

    P = diag(minv) is frozen before production, so the kernel is an ordinary
    Metropolis-Hastings kernel (exactly pi-invariant).  Returns
    (theta, potential, grad, accepted, failed, accept_probability).
    """
    mean_forward = theta - h * minv * grad
    proposal = mean_forward + np.sqrt(2.0 * h * minv) * rng.normal(size=theta.size)
    try:
        potential_prop, grad_prop = target.potential_and_grad(proposal)
        if not np.isfinite(potential_prop) or not np.all(np.isfinite(grad_prop)):
            return theta, potential, grad, False, True, 0.0
        mean_reverse = proposal - h * minv * grad_prop
        log_q_forward = -np.sum((proposal - mean_forward) ** 2 / minv) / (4.0 * h)
        log_q_reverse = -np.sum((theta - mean_reverse) ** 2 / minv) / (4.0 * h)
        log_alpha = -potential_prop + log_q_reverse - (-potential + log_q_forward)
        prob = float(math.exp(min(0.0, log_alpha))) if np.isfinite(log_alpha) else 0.0
        if math.log(rng.uniform()) < min(0.0, log_alpha):
            return proposal, potential_prop, grad_prop, True, False, prob
        return theta, potential, grad, False, False, prob
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError):
        return theta, potential, grad, False, True, 0.0


def _mala_initial_step(
    target: LogCholeskyTarget,
    theta: Array,
    potential: float,
    grad: Array,
    h: float,
    minv: Array,
    rng: np.random.Generator,
    target_accept: float,
    probes: int = 5,
) -> float:
    """Double/halve h until the mean one-step acceptance crosses the target."""

    def mean_accept(step: float) -> float:
        values = [
            _lchol_diag_transition(target, theta, potential, grad, step, minv, rng)[5]
            for _ in range(probes)
        ]
        return float(np.mean(values))

    direction = 1 if mean_accept(h) > target_accept else -1
    for _ in range(60):
        value = mean_accept(h)
        if direction == 1 and not value > target_accept:
            break
        if direction == -1 and not value < target_accept:
            break
        h = h * 2.0 if direction == 1 else h / 2.0
        if h > 1e4 or h < 1e-10:
            break
    return float(h)


def lchol_mala_diag_chain(
    posterior: "PSDGraphPosterior",
    w0: Array,
    warmup: int,
    retained: int,
    h_init: float,
    seed: int,
    target_accept: float = 0.574,
) -> "ChainResult":
    """MALA in log-Cholesky coordinates with a learned diagonal metric.

    Warm-up (``warmup`` transitions, cost included in runtime_total) uses the
    same Stan-style windowed schedule as NUTS: the diagonal metric is the
    regularised posterior variance of each coordinate over expanding windows
    and the scalar step size is dual-averaged to ``target_accept`` (0.574,
    the MALA optimum).  The terminal step-size buffer is 15% of the warm-up.
    Metric and step are frozen before the retained transitions.
    """
    target = LogCholeskyTarget(posterior)
    rng = np.random.default_rng(seed)
    w_start = np.asarray([project_spd(block) for block in w0])
    theta = target.from_weights(w_start)
    potential, grad = target.potential_and_grad(theta)
    minv = np.ones(theta.size)

    total_start = time.perf_counter()
    step = _mala_initial_step(
        target, theta, potential, grad, h_init, minv, rng, target_accept
    )
    dual = _DualAveraging(step, target_accept)
    windows = _StanWindows(warmup, term_buffer=max(50, int(0.15 * warmup)))
    welford = _Welford(theta.size)

    warm_prob: List[float] = []
    warm_accept = warm_failures = 0
    for _ in range(warmup):
        theta, potential, grad, accepted, failed, prob = _lchol_diag_transition(
            target, theta, potential, grad, step, minv, rng
        )
        warm_accept += int(accepted)
        warm_failures += int(failed)
        warm_prob.append(prob)
        step = dual.update(prob)
        if windows.in_window():
            welford.add(theta)
        if windows.end_window():
            windows.compute_next_window()
            n = welford.n
            var = welford.variance()
            minv = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            welford.restart()
            windows.counter += 1
            step = _mala_initial_step(
                target, theta, potential, grad, step, minv, rng, target_accept
            )
            dual.restart(step)
        else:
            windows.counter += 1
    if warmup > 0:
        step = dual.final_step()

    samples = DrawBuffer(retained)
    accepted_sampling = failures_sampling = 0
    w_current = target.to_weights(theta)
    sampling_start = time.perf_counter()
    for _ in range(retained):
        theta, potential, grad, accepted, failed, _ = _lchol_diag_transition(
            target, theta, potential, grad, step, minv, rng
        )
        accepted_sampling += int(accepted)
        failures_sampling += int(failed)
        if accepted:
            w_current = target.to_weights(theta)
        samples.append(w_current.copy())
    runtime_sampling = time.perf_counter() - sampling_start
    runtime_total = time.perf_counter() - total_start
    return ChainResult(
        samples=np.asarray(samples),
        acceptance=accepted_sampling / max(retained, 1),
        preburn_acceptance=warm_accept / max(warmup, 1),
        runtime_total_sec=float(runtime_total),
        runtime_sampling_sec=float(runtime_sampling),
        h_final=float(step),
        invalid_rate=0.0,
        preburn_invalid_rate=0.0,
        numerical_failure_rate=failures_sampling / max(retained, 1),
        preconditioner_condition=float(np.max(minv) / np.min(minv)),
    )


# >>> END RELEASE ABLATIONS <<<


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def tune_fixed_method_h(
    method: str,
    posterior: PSDGraphPosterior,
    anchors: Sequence[Array],
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
    edge_scale: Optional[Array] = None,
) -> Tuple[float, List[Dict[str, float]]]:
    """Tune one scalar h in discarded pilots, then return a frozen value.

    This follows the same paper protocol used for the original fixed-step
    methods: tune outside retained production, freeze h, then run the matched
    burn-in/production budget.  The pilot is intentionally separate from the
    reported end-to-end production budget, matching the original graph sweep.
    """
    if method not in {"ConeMALA", CONE_EDGE_METHOD, "EuclideanMALA", "GenericRMALA"}:
        raise ValueError(f"Unsupported fixed method: {method}")
    if tune_steps < 10:
        raise ValueError("fixed_tune_steps must be at least 10.")

    use_anchors = list(anchors[: min(tune_chains, len(anchors))])
    if not use_anchors:
        raise ValueError("At least one fixed-method tuning anchor is required.")

    tested: Dict[str, Dict[str, float]] = {}

    def run_candidate(h: float) -> Dict[str, float]:
        h = float(max(h, 1e-14))
        key = f"{h:.17e}"
        if key in tested:
            return tested[key]

        acceptances: List[float] = []
        invalid_rates: List[float] = []
        runtimes: List[float] = []
        for chain_index, anchor in enumerate(use_anchors):
            seed = seed_base + chain_index
            if method in ("ConeMALA", CONE_EDGE_METHOD):
                result = cone_mala_chain(
                    posterior, anchor, tune_warmup, tune_steps, h, seed,
                    edge_scale if method == CONE_EDGE_METHOD else None,
                )
            elif method == "EuclideanMALA":
                result = euclidean_mala_chain(
                    posterior, anchor, tune_warmup, tune_steps, h, seed
                )
            else:
                result = generic_rmala_chain(
                    posterior, anchor, tune_warmup, tune_steps, h, seed
                )
            acceptances.append(result.acceptance)
            invalid_rates.append(result.invalid_rate)
            runtimes.append(result.runtime_total_sec)

        row = {
            "method": method,
            "h": h,
            "pilot_acceptance": float(np.mean(acceptances)),
            "pilot_acceptance_sd": float(
                np.std(acceptances, ddof=1)
                if len(acceptances) > 1 else 0.0
            ),
            "pilot_invalid_rate": float(np.mean(invalid_rates)),
            "pilot_runtime_sec": float(np.sum(runtimes)),
        }
        tested[key] = row
        print(
            f"    {method} pilot h={h:.3e}: "
            f"acc={row['pilot_acceptance']:.3f}, "
            f"invalid={row['pilot_invalid_rate']:.3f}",
            flush=True,
        )
        return row

    positive_factors = sorted(set(float(f) for f in factors if float(f) > 0.0))
    for factor in positive_factors:
        run_candidate(max(h_center, 1e-14) * factor)

    lower = target_accept - acceptance_tolerance
    upper = target_accept + acceptance_tolerance

    # Too aggressive everywhere -> halve until the acceptance enters range.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] >= lower for row in rows):
            break
        run_candidate(rows[0]["h"] / 2.0)

    # Too conservative everywhere -> double until the acceptance enters range.
    for _ in range(max_bracket_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
        if any(row["pilot_acceptance"] <= upper for row in rows):
            break
        run_candidate(rows[-1]["h"] * 2.0)

    # Geometric bisection around the target acceptance.
    for _ in range(bisection_steps):
        rows = sorted(tested.values(), key=lambda row: row["h"])
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
        midpoint = math.sqrt(left["h"] * right["h"])
        if f"{midpoint:.17e}" in tested:
            break
        run_candidate(midpoint)

    rows = sorted(tested.values(), key=lambda row: row["h"])
    admissible = [
        row for row in rows
        if lower <= row["pilot_acceptance"] <= upper
        and row["pilot_invalid_rate"] < 0.95
    ]
    if admissible:
        best = min(
            admissible,
            key=lambda row: (
                abs(row["pilot_acceptance"] - target_accept),
                row["pilot_invalid_rate"],
            ),
        )
    else:
        best = min(
            rows,
            key=lambda row: (
                abs(row["pilot_acceptance"] - target_accept),
                row["pilot_invalid_rate"],
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
        raise RuntimeError(
            f"No feasible pilot step size found for {method}. "
            "Inspect the saved tuning grid or expand the factor/bracket range."
        )
    return float(best["h"]), rows


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


def resolve_settings(args: argparse.Namespace, d: int) -> Dict[str, float]:
    settings = dict(D5_REFERENCE_SETTINGS)
    for name in ("preburn", "retained", "h_cone", "h_euclid", "h_rmala", "h_fisher"):
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
        "LogCholeskyMALA": r"\texttt{LogChol\_MALA}",
        "LogCholeskyNUTS": r"\texttt{LogChol\_NUTS}",
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
    indexed = summary.set_index("Method")
    rows: List[str] = []
    for method in [name for name in METHOD_ORDER if name in indexed.index]:
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
ConeMALA, Euclidean MALA, and generic RMALA use {preburn} discarded fixed-kernel burn-in transitions and {retained} retained transitions per chain. Fisher-adaptive MALA uses {fisher_adaptive_steps} discarded adaptive transitions, {fisher_individual_settle} discarded chain-specific fixed-kernel settling transitions, pooled-preconditioner fixed-$h$ tuning, {fisher_final_settle} discarded shared-kernel settling transitions, and {retained} retained transitions. Its preconditioner update interval is {fisher_update_interval}. ESS/sec includes all adaptation, tuning, settling, and production costs. \texttt{{LogChol\_MALA}} is fixed-step MALA in log-Cholesky coordinates (pilot-tuned $h$, same burn-in as the other fixed-step methods); \texttt{{LogChol\_NUTS}} is NUTS in the same coordinates with a diagonal metric and dual-averaging step size, adapted in a discarded warm-up whose cost is included.
\texttt{{TF}} denotes Fisher fixed-kernel tuning failure; no production
inference is reported for that method.}}
\label{{tab:m50-d{d}-with-fisher}}
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
    d: int,
) -> None:
    import matplotlib.pyplot as plt

    set_large_plot_fonts()
    ordered = (
        summary.set_index("Method")
        .reindex([name for name in METHOD_ORDER if name in set(summary["Method"])])
        .reset_index()
    )
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
    axis.set_title(rf"Acceptance rates, $m=50$, $d={d}$")
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
        outdir / "plots" / f"acceptance_m50_d{d}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        outdir / "plots" / f"acceptance_m50_d{d}.png",
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
        subset = combined[combined["Method"] == method].sort_values("d")
        subset = subset[
            np.isfinite(subset["rel_W_error_ESSsec_sampling"])
            & np.isfinite(subset["Phi_Rhat"])
        ]
        if subset.empty:
            continue
        axes[0].plot(
            subset["d"], subset["rel_W_error_ESSsec_sampling"],
            **METHOD_STYLE[method], linewidth=2.2, markersize=7,
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            subset["d"], subset["Phi_Rhat"],
            **METHOD_STYLE[method], linewidth=2.2, markersize=7,
            label=METHOD_LABELS[method],
        )

    axes[0].set_yscale("log")
    axes[0].set_title("(a) Sampling efficiency")
    axes[0].set_xlabel(r"SPD block dimension $d$")
    axes[0].set_ylabel(r"Relative-$W$ production ESS/sec")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()

    axes[1].set_title(r"(b) Energy split-$\widehat R$")
    axes[1].set_xlabel(r"SPD block dimension $d$")
    axes[1].set_ylabel(r"Split-$\widehat R_\Phi$")
    axes[1].axhline(1.05, linestyle="--", linewidth=1.3,
                    label=r"$\widehat R=1.05$")
    axes[1].axhline(1.01, linestyle=":", linewidth=1.3,
                    label=r"$\widehat R=1.01$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    available_d = sorted(int(v) for v in combined["d"].unique())
    for axis in axes:
        axis.set_xticks(available_d)

    figure.savefig(
        plot_dir / "scaling_m50_over_d_with_fisher.pdf",
        dpi=300, bbox_inches="tight",
    )
    figure.savefig(
        plot_dir / "scaling_m50_over_d_with_fisher.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# Run one d
# =============================================================================

def run_one_experiment(
    args: argparse.Namespace,
    d: int,
) -> pd.DataFrame:
    m = M_FIXED
    settings = resolve_settings(args, d)
    preburn = int(settings["preburn"])
    retained = int(settings["retained"])
    fisher_individual_settle = min(args.fisher_individual_settle, max(preburn - 1, 0))
    fisher_adaptive_steps = (
        args.fisher_adaptive_steps
        if args.fisher_adaptive_steps is not None
        else max(preburn - fisher_individual_settle, 1)
    )

    experiment_dir = Path(args.outdir) / f"graph_posterior_m50_d{d}"
    results_dir = experiment_dir / "results"
    plots_dir = experiment_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    selected_methods = parse_method_list(
        args.methods, ALL_METHODS, default=METHOD_ORDER,
        groups={"ablations": ABLATION_METHODS},
    )
    summary_path = Path(args.outdir) / f"summary_m50_d{d}.csv"
    if args.resume and summary_path.exists():
        existing = pd.read_csv(summary_path)
        if set(selected_methods) <= set(existing["Method"]):
            print(f"Skipping completed d={d}: {summary_path}")
            return existing

    rng = np.random.default_rng(args.seed + 17 * m)
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
        alpha_logdet=(0.5 * (d + 1.0) if args.alpha_logdet is None else args.alpha_logdet),
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
    chain_seeds = [10_000 + 100 * m + chain for chain in range(args.n_chains)]
    initial_states = [
        make_initial_w(edge_count, d, chain)
        for chain in range(args.n_chains)
    ]

    fisher_update_interval = (
        args.fisher_update_interval
        if args.fisher_update_interval is not None
        else FISHER_UPDATE_INTERVAL_DEFAULT
    )

    # Reuse the exact published d=5 fixed h values.  For new dimensions,
    # tune the three fixed-step methods in discarded pilots and freeze h.
    fixed_tuning_rows: List[Dict[str, float]] = []
    if d != 5 and not args.skip_fixed_tuning:
        fixed_factors = tuple(
            float(value.strip())
            for value in args.fixed_grid_factors.split(",")
            if value.strip()
        )
        for method, setting_key in (
            ("ConeMALA", "h_cone"),
            ("EuclideanMALA", "h_euclid"),
            ("GenericRMALA", "h_rmala"),
        ):
            if method not in selected_methods:
                # e.g. a Fisher-only rerun: its pilot h is never used.
                continue
            tuned_h, tuning_rows = tune_fixed_method_h(
                method=method,
                posterior=posterior,
                anchors=initial_states,
                h_center=float(settings[setting_key]),
                target_accept=args.fixed_target_accept,
                acceptance_tolerance=args.fixed_acceptance_tolerance,
                tune_warmup=args.fixed_tune_warmup,
                tune_steps=args.fixed_tune_steps,
                tune_chains=args.fixed_tune_chains,
                factors=fixed_factors,
                max_bracket_steps=args.fixed_max_bracket_steps,
                bisection_steps=args.fixed_bisection_steps,
                seed_base=70_000_000 + 10_000 * d
                    + {"ConeMALA": 100, "EuclideanMALA": 200, "GenericRMALA": 300}[method],
            )
            settings[setting_key] = tuned_h
            for row in tuning_rows:
                row["d"] = d
                row["m"] = m
            fixed_tuning_rows.extend(tuning_rows)

        if fixed_tuning_rows:
            pd.DataFrame(fixed_tuning_rows).to_csv(
                results_dir / "fixed_method_tuning.csv", index=False
            )


    run_config = {
        **vars(args),
        "active_m": m,
        "active_d": d,
        "free_edge_parameter_dimension": int(edge_count * d * (d + 1) // 2),
        "precision_dimension": int(m * d),
        "preburn_fixed_methods": preburn,
        "retained": retained,
        "settings": settings,
        "fixed_method_tuning": "reused d=5 paper h" if d == 5 else ("skipped" if args.skip_fixed_tuning else "discarded pilot tuning"),
        "fisher_update_interval_resolved": fisher_update_interval,
        "fisher_adaptive_steps_resolved": fisher_adaptive_steps,
        "fisher_individual_settle_resolved": fisher_individual_settle,
        "strict_spd_test": True,
        "prior_family": "Wishart",
        "nu": float(d + 1.0 if args.alpha_logdet is None else 2.0 * args.alpha_logdet),
        "psi_inverse_scale": float(2.0 * args.lambda_tr),
        "wishart_parameterization": (
            "intrinsic edge potential = lambda_tr*tr(W_e) "
            "- alpha_logdet*logdet(W_e); "
            "nu=2*alpha_logdet and Psi^{-1}=2*lambda_tr*I"
        ),
        "fisher_workflow": (
            "discarded adaptation -> chain-specific settle -> pooled metric -> "
            "fixed-h typical-set tuning -> shared-kernel settle -> production"
        ),
    }
    (results_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(
        f"m={m}, d={d}, edges={edge_count}, D={edge_count*d*(d+1)//2}, fixed burn={preburn}, "
        f"retained={retained}"
    )
    print("=" * 100)

    # ---- release extras: discarded-pilot tuning of the log-Cholesky MALA step ----
    lchol_saved = args.resume and (results_dir / f"row_{LCHOL_METHOD}.csv").exists()
    if args.lchol_h is not None:
        settings["h_lchol"] = float(args.lchol_h)
    elif (
        LCHOL_METHOD in selected_methods
        or (LCHOL_DIAG_METHOD in selected_methods and not lchol_saved)
    ) and not lchol_saved:
        print("\nTuning log-Cholesky MALA step size (discarded pilots)")
        h_lchol, lchol_rows = tune_lchol_step(
            posterior, initial_states, args,
            seed_base=80_000_000 + 10_000 * d + m,
        )
        settings["h_lchol"] = h_lchol
        pd.DataFrame(lchol_rows).to_csv(
            results_dir / "lchol_tuning.csv", index=False
        )
    else:
        settings["h_lchol"] = float(args.lchol_h_center)
        previous_config = results_dir / "release_config.json"
        if args.resume and previous_config.exists():
            try:  # keep the step tuned in an earlier run (do not reset to the centre)
                settings["h_lchol"] = float(
                    json.loads(previous_config.read_text())["h_lchol"]
                )
            except (ValueError, KeyError, OSError):
                pass
    (results_dir / "release_config.json").write_text(
        json.dumps(
            {
                "extras_version": EXTRAS_VERSION,
                "methods": selected_methods,
                "h_lchol": settings["h_lchol"],
                "nuts_warmup": args.nuts_warmup,
                "nuts_target_accept": args.nuts_target_accept,
                "nuts_max_depth": args.nuts_max_depth,
                "retained": retained,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    rows: List[Dict[str, float]] = []
    plot_data: Dict[str, List[pd.DataFrame]] = {}

    fisher_row_saved = args.resume and (
        results_dir / "row_FisherAdaptiveMALA.csv"
    ).exists()
    fisher_plan = None
    if "FisherAdaptiveMALA" in selected_methods and not fisher_row_saved:
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
            update_interval=fisher_update_interval,
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
            seed_base=50_000_000 + 1000 * m,
        )
        pd.DataFrame(fisher_plan.adaptation_rows).to_csv(
            results_dir / "fisher_adaptation.csv", index=False
        )
        pd.DataFrame(fisher_plan.tuning_rows).to_csv(
            results_dir / "fisher_tuning.csv", index=False
        )
        if fisher_plan.feasible:
            np.save(results_dir / "fisher_shared_root.npy", fisher_plan.root)
            np.save(
                results_dir / "fisher_shared_preconditioner.npy",
                fisher_plan.root @ fisher_plan.root.T,
            )
            fisher_kernel_metadata = {
                "selected_step_size": float(fisher_plan.h_selected),
                "condition_number": float(fisher_plan.condition_number),
                "pilot_acceptance": float(fisher_plan.pilot_acceptance),
                "pilot_invalid_rate": float(fisher_plan.pilot_invalid_rate),
                "adaptation_runtime_sec": float(
                    fisher_plan.adaptation_runtime_sec
                ),
                "tuning_runtime_sec": float(fisher_plan.tuning_runtime_sec),
            }
            (results_dir / "fisher_shared_kernel.json").write_text(
                json.dumps(fisher_kernel_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
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

    for method in ALL_METHODS:
        row_path = results_dir / f"row_{method}.csv"
        if args.resume and row_path.exists():
            rows.append(pd.read_csv(row_path).iloc[0].to_dict())
            try:
                plot_data[method] = [
                    pd.read_csv(results_dir / f"{method}_chain{c}_observables.csv")
                    for c in range(args.n_chains)
                ]
            except FileNotFoundError:
                pass
            print(f"\nMethod: {method} (re-used {row_path.name})")
            continue
        if method not in selected_methods:
            continue
        print(f"\nMethod: {method}")

        if method == "FisherAdaptiveMALA" and not fisher_plan.feasible:
            failed_row: Dict[str, float] = {
                "d": d,
                "m": m,
                "E": edge_count,
                "D_free": int(edge_count * d * (d + 1) // 2),
                "p_precision": int(m * d),
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
            rows.append(failed_row)
            pd.DataFrame([failed_row]).to_csv(row_path, index=False)
            print(
                "  Skipped production: no feasible fixed Fisher kernel.",
                flush=True,
            )
            continue

        chain_results: List[ChainResult] = []
        observable_chains: List[Dict[str, Array]] = []
        edge_chains: List[Array] = []
        prebuilt = None
        if method in (LCHOL_METHOD, NUTS_METHOD, LCHOL_DIAG_METHOD, NUTS_IDENTITY_METHOD):
            # ablations reuse their parent's seeds (paired comparison)
            prebuilt = run_release_method_chains(
                method, posterior, initial_states, preburn, retained,
                float(settings["h_lchol"]),
                [
                    chain_seeds[c]
                    + (7_000_000 if method in (LCHOL_METHOD, LCHOL_DIAG_METHOD) else 9_000_000)
                    for c in range(args.n_chains)
                ],
                args,
            )

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
            elif prebuilt is not None:
                result = prebuilt[chain_index]
            else:
                raise KeyError(method)

            chain_results.append(result)
            observable = observables(result.samples, posterior, w_star, y_test)
            observable_chains.append(observable)
            if not args.skip_edge_diagnostics:
                edge_chains.append(edge_entry_matrix(result.samples))
            pd.DataFrame(observable).to_csv(
                results_dir / f"{method}_chain{chain_index}_observables.csv",
                index=False,
            )
            if args.save_samples:
                np.save(
                    results_dir / f"{method}_chain{chain_index}_samples.npy",
                    result.samples,
                )
            result.samples = None  # draws are no longer needed; free memory
            print(
                f"  chain {chain_index + 1}: h={result.h_final:.3e}, "
                f"settle/burn acc={result.preburn_acceptance:.3f}, "
                f"production acc={result.acceptance:.3f}, "
                f"invalid={result.invalid_rate:.3f}, "
                f"total={result.runtime_total_sec:.1f}s"
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
            "d": d,
            "m": m,
            "E": edge_count,
            "D_free": int(edge_count * d * (d + 1) // 2),
            "p_precision": int(m * d),
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

        row.update(
            extended_diagnostic_columns(
                observable_chains,
                None if args.skip_edge_diagnostics else edge_chains,
                runtime_total,
                runtime_sampling,
                edge_csv_path=results_dir / f"{method}_edge_entry_diagnostics.csv",
                edge_labels=edge_entry_labels(edge_count, d),
            )
        )
        if args.prefix_retained:
            prefix_rows: List[Dict[str, float]] = []
            for k in sorted({int(v) for v in str(args.prefix_retained).split(",") if v.strip()}):
                if not (20 <= k <= retained):
                    continue
                frac = k / retained
                prefix_row = {
                    "d": d, "Method": method, "retained": k,
                    "runtime_sampling_all_chains_est": runtime_sampling * frac,
                    "runtime_total_all_chains_est": runtime_total - runtime_sampling * (1.0 - frac),
                }
                prefix_row.update(
                    extended_diagnostic_columns(
                        [{n: np.asarray(v)[:k] for n, v in obs.items()} for obs in observable_chains],
                        None if args.skip_edge_diagnostics else [e[:k] for e in edge_chains],
                        prefix_row["runtime_total_all_chains_est"],
                        prefix_row["runtime_sampling_all_chains_est"],
                    )
                )
                prefix_row["max_Rhat"] = prefix_row.get("all_Rhat_rank_max", np.nan)
                prefix_rows.append(prefix_row)
            if prefix_rows:
                pd.DataFrame(prefix_rows).to_csv(
                    results_dir / f"longrun_prefix_{method}.csv", index=False
                )
        row.update(nuts_extra_columns(chain_results))
        row["tier"] = METHOD_TIER[method]
        row["role"] = "ablation" if method in ABLATION_METHODS else "main"
        row["warmup_transitions"] = (
            args.nuts_warmup if method in (NUTS_METHOD, NUTS_IDENTITY_METHOD)
            else args.fisher_final_settle if method == "FisherAdaptiveMALA"
            else preburn
        )
        rows.append(row)
        pd.DataFrame([row]).to_csv(row_path, index=False)
        plot_data[method] = [pd.DataFrame(obs) for obs in observable_chains]

    summary = pd.DataFrame(rows)
    summary["_order"] = summary["Method"].map(
        {name: i for i, name in enumerate(ALL_METHODS)}
    )
    summary = (
        summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    )
    summary.to_csv(results_dir / "summary_multichain.csv", index=False)
    summary.to_csv(results_dir / f"summary_m50_d{d}.csv", index=False)
    summary.to_csv(summary_path, index=False)

    write_latex_table(
        summary=summary,
        path=Path(args.outdir) / f"table_m50_d{d}_with_fisher.tex",
        m=m,
        d=d,
        preburn=preburn,
        retained=retained,
        fisher_update_interval=fisher_update_interval,
        fisher_adaptive_steps=fisher_adaptive_steps,
        fisher_individual_settle=fisher_individual_settle,
        fisher_final_settle=args.fisher_final_settle,
    )

    save_two_panel_ecdf(
        plot_data, experiment_dir, "logdet_Q", "Phi",
        f"ecdf_logdet_phi_m50_d{d}",
        "(a) Log-determinant observable", "(b) Posterior energy",
    )
    save_two_panel_ecdf(
        plot_data, experiment_dir, "rel_W_error", "test_nll",
        f"ecdf_error_nll_m50_d{d}",
        "(a) Relative edge-weight error", "(b) Test negative log-likelihood",
    )
    save_acceptance_plot(summary, experiment_dir, d)

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


def collect_existing_summaries(outdir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    pattern = re.compile(r"summary_m50_d(\d+)\.csv$")
    for path in sorted(outdir.glob("summary_m50_d*.csv")):
        match = pattern.search(path.name)
        if not match:
            continue
        frame = pd.read_csv(path)
        frame["d"] = int(match.group(1))
        frame["m"] = M_FIXED
        frames.append(frame)
    if not frames:
        raise RuntimeError("No d-specific summary files were found.")
    return pd.concat(frames, ignore_index=True)

# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "m=50 cycle-graph block-dimension sweep with Fisher-adaptive MALA."
        )
    )
    parser.add_argument(
        "--d_list",
        type=parse_int_list,
        default=[2, 5, 8, 10],
        help="Comma-separated SPD block dimensions.",
    )
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=80)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--outdir",
        type=str,
        default="m50_d_sweep_wishart_with_fisher",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument(
        "--prefix_retained", type=str, default=None,
        help=(
            "Comma-separated retained-draw prefixes (e.g. 12000,24000,48000). "
            "For every method, rank-R-hat/ESS diagnostics are recomputed on the "
            "first k retained draws of each chain and written to "
            "longrun_prefix_m50_d<d>.csv. Production runtime is scaled "
            "proportionally (an estimate; burn-in/overhead are kept fixed)."
        ),
    )

    parser.add_argument("--truth_scale", type=float, default=0.25)
    parser.add_argument("--r_scale", type=float, default=0.75)
    parser.add_argument("--lambda_tr", type=float, default=0.05)
    parser.add_argument(
        "--alpha_logdet",
        type=float,
        default=None,
        help=(
            "Intrinsic logdet coefficient. Default is (d+1)/2 separately "
            "for each d, equivalent to determinant-neutral nu=d+1."
        ),
    )

    parser.add_argument("--preburn", type=int)
    parser.add_argument("--retained", type=int)
    parser.add_argument("--h_cone", type=float)
    parser.add_argument("--h_euclid", type=float)
    parser.add_argument("--h_rmala", type=float)
    parser.add_argument("--h_fisher", type=float)

    # Fixed-method discarded pilot tuning for d != 5.
    parser.add_argument("--skip_fixed_tuning", action="store_true")
    parser.add_argument("--fixed_target_accept", type=float, default=0.574)
    parser.add_argument("--fixed_acceptance_tolerance", type=float, default=0.08)
    parser.add_argument("--fixed_tune_warmup", type=int, default=100)
    parser.add_argument("--fixed_tune_steps", type=int, default=400)
    parser.add_argument("--fixed_tune_chains", type=int, default=2)
    parser.add_argument(
        "--fixed_grid_factors",
        type=str,
        default="0.125,0.25,0.5,1,2,4",
    )
    parser.add_argument("--fixed_max_bracket_steps", type=int, default=20)
    parser.add_argument("--fixed_bisection_steps", type=int, default=6)

    # Fisher workflow, inherited from the graph-size sweep.
    parser.add_argument("--target_accept", type=float, default=0.574)
    parser.add_argument("--fisher_damping", type=float, default=10.0)
    parser.add_argument("--fisher_init_steps", type=int, default=500)
    parser.add_argument("--fisher_adaptive_steps", type=int)
    parser.add_argument("--fisher_individual_settle", type=int, default=1000)
    parser.add_argument("--fisher_final_settle", type=int, default=500)
    parser.add_argument("--fisher_tune_warmup", type=int, default=100)
    parser.add_argument("--fisher_tune_steps", type=int, default=400)
    parser.add_argument("--fisher_tune_chains", type=int, default=2)
    parser.add_argument(
        "--fisher_grid_factors", type=str,
        default="0.125,0.25,0.5,1,2,4",
    )
    parser.add_argument("--fisher_acceptance_tolerance", type=float, default=0.08)
    parser.add_argument("--fisher_max_bracket_steps", type=int, default=24)
    parser.add_argument("--fisher_bisection_steps", type=int, default=8)
    parser.add_argument(
        "--fisher_update_interval",
        type=int,
        default=None,
        help="Preconditioner update interval during discarded adaptation "
             "(default 1: every iteration, as in Titsias 2023).",
    )
    parser.add_argument("--fisher_max_condition", type=float, default=100.0)
    parser.add_argument("--fisher_adapt_rate", type=float, default=0.05)
    parser.add_argument("--fisher_adapt_decay", type=float, default=100.0)
    parser.add_argument("--fisher_adapt_exponent", type=float, default=0.6)
    parser.add_argument("--fisher_h_min", type=float, default=1e-12)
    parser.add_argument("--fisher_h_max", type=float, default=1.0)
    add_release_arguments(parser)
    return parser



def _latex_sci(value: float, digits: int = 2) -> str:
    """Format a positive scalar in compact LaTeX scientific notation."""
    if value is None or not np.isfinite(float(value)):
        return r"\texttt{TF}"
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10.0 ** exponent)
    if -2 <= exponent <= 2:
        return f"{value:.{digits}f}"
    return rf"{coefficient:.{digits}f}\times 10^{{{exponent}}}"


def write_combined_table1(
    combined: pd.DataFrame,
    path: Path,
) -> None:
    """Write the paper-style m=50 block-dimension comparison table."""
    blocks: List[str] = []
    for d in sorted(int(v) for v in combined["d"].unique()):
        subset = (
            combined[combined["d"] == d]
            .set_index("Method")
            .reindex(METHOD_ORDER)
        )
        retained = int(subset["retained"].dropna().iloc[0])
        preburn = int(subset["preburn"].dropna().iloc[0])
        D_free = int(subset["D_free"].dropna().iloc[0])
        p_precision = int(subset["p_precision"].dropna().iloc[0])
        blocks.append(
            rf"\multicolumn{{8}}{{l}}{{\emph{{"
            rf"$d={d}$; $D={D_free}$ free edge parameters; "
            rf"$p={p_precision}$; {preburn:,} discarded and "
            rf"{retained:,} retained transitions per chain}}}} \\"
        )
        for method in METHOD_ORDER:
            row = subset.loc[method]
            blocks.append(
                f"{METHOD_LABELS[method]}"
                f" & ${_latex_sci(row['h_final_mean'])}$"
                f" & {latex_value(row['accept_mean'], 3)}"
                f" & {latex_value(row['invalid_rate_mean'], 3)}"
                f" & {latex_value(row['rel_W_error_ESSsec_sampling'], 2)}"
                f" & {latex_value(row['test_nll_ESSsec_sampling'], 2)}"
                f" & {latex_value(row['rel_W_error_ESSsec_total'], 2)}"
                f" & {latex_value(row['test_nll_ESSsec_total'], 2)}"
                r" \\"
            )
        blocks.append(r"\addlinespace")

    content = rf"""
\begin{{table*}}[t]
\centering
\caption{{Block-dimension sweep for $m=50$ cycle-graph posteriors.
At every $d$, the determinant-neutral Wishart specification uses
$\nu=d+1$ and $\Psi^{{-1}}=0.1I_d$.}}
\label{{tab:block_dimension_sweep_wishart}}
\scriptsize
\begin{{tabular}}{{lccccccc}}
\toprule
Method & $h$ & Acc. & Invalid
& \multicolumn{{2}}{{c}}{{Production ESS/s}}
& \multicolumn{{2}}{{c}}{{End-to-end ESS/s}} \\
\cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}
& & & & Rel.-$W$ & NLL & Rel.-$W$ & NLL \\
\midrule
{chr(10).join(blocks)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""".strip() + "\n"
    path.write_text(content, encoding="utf-8")


def write_combined_rhat_table(
    combined: pd.DataFrame,
    path: Path,
) -> None:
    blocks: List[str] = []
    for d in sorted(int(v) for v in combined["d"].unique()):
        subset = (
            combined[combined["d"] == d]
            .set_index("Method")
            .reindex(METHOD_ORDER)
        )
        blocks.append(rf"\multicolumn{{5}}{{l}}{{\emph{{$d={d}$}}}} \\")
        for method in METHOD_ORDER:
            row = subset.loc[method]
            blocks.append(
                f"{METHOD_LABELS[method]}"
                f" & {latex_value(row['rel_W_error_Rhat'], 3)}"
                f" & {latex_value(row['test_nll_Rhat'], 3)}"
                f" & {latex_value(row['logdet_Q_Rhat'], 3)}"
                f" & {latex_value(row['Phi_Rhat'], 3)}"
                r" \\"
            )
        blocks.append(r"\addlinespace")

    content = rf"""
\begin{{table}}[t]
\centering
\caption{{Split-$\widehat R$ diagnostics for the $m=50$
block-dimension sweep.}}
\label{{tab:block_dimension_sweep_wishart_rhat}}
\scriptsize
\begin{{tabular}}{{lcccc}}
\toprule
Method
& $\widehat R_W$
& $\widehat R_{{\rm NLL}}$
& $\widehat R_{{\log\det Q}}$
& $\widehat R_\Phi$ \\
\midrule
{chr(10).join(blocks)}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip() + "\n"
    path.write_text(content, encoding="utf-8")

def main() -> None:
    print(f"Script version: {D_SWEEP_VERSION}")
    print(f"Release extras: {EXTRAS_VERSION}")
    args = build_parser().parse_args()
    if args.n_chains < 2:
        raise ValueError("At least two chains are needed for split-Rhat.")
    if args.fisher_update_interval is not None and args.fisher_update_interval < 1:
        raise ValueError("fisher_update_interval must be positive.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for d in args.d_list:
        if d < 1:
            raise ValueError("Every d must be positive.")
        run_one_experiment(args, d)

    combined = collect_existing_summaries(outdir)
    combined.to_csv(
        outdir / "combined_summary_m50_d_sweep_with_fisher.csv",
        index=False,
    )
    save_scaling_plots(combined, outdir)
    write_combined_table1(
        combined,
        outdir / "table_block_dimension_sweep_wishart.tex",
    )
    write_combined_rhat_table(
        combined,
        outdir / "table_block_dimension_split_rhat.tex",
    )
    write_combined_diagnostics_table(
        combined, outdir / "table_block_dimension_rank_diagnostics.tex", "d", "d",
    )
    save_diagnostic_scaling_plots(
        combined, outdir, "d", r"SPD block dimension $d$",
        "diagnostics_m50_over_d",
    )

    columns = [
        "d", "D_free", "p_precision", "Method", "accept_mean",
        "rel_W_error_ESSsec_total", "test_nll_ESSsec_total",
        "rel_W_error_Rhat", "test_nll_Rhat", "Phi_Rhat",
        "invalid_rate_mean",
    ]
    print("\nCombined block-dimension summary:")
    print(combined[columns].to_string(index=False))
    print(
        f"\nSaved combined summaries, tables, and plots under: {outdir.resolve()}"
    )


if __name__ == "__main__":
    main()

