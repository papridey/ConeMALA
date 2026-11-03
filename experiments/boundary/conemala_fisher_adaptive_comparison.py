r"""
Repeated-seed sampler comparison for a synthetic PSD-weighted graph posterior.

This self-contained script aligns its notation with the revised manuscript:

    W = (W_e)_{e in E},       W_e in S_{++}^d,
    Q(W) = R + sum_e B_e W_e B_e^T,       R = rho I_p,
    Phi(W) = intrinsic negative log posterior relative to product AIRM volume.

It compares four exact Metropolis--Hastings kernels targeting the same
product-Lebesgue posterior:

    1. ConeMALA: product-AIRM exponential proposal and exact Jacobian.
    2. Fisher-adaptive MALA: Euclidean MALA with a regularized empirical
       Fisher preconditioner adapted only during warm-up and then frozen.
    3. Euclidean MALA: isotropic MALA in orthonormal symmetric coordinates.
    4. Retraction RMALA: first-order product-AIRM retraction baseline.

For every independent synthetic-data seed, each method is run with multiple
chains. The script reports

    acceptance,
    minimum ESS/sec across core posterior observables,
    MCSE of held-out NLL,
    maximum rank-normalized split-Rhat across core observables,
    held-out NLL,
    relative edge-recovery error.

The paper table uses mean +/- standard deviation across independent experiment
seeds. The code also saves all seed-level and diagnostic-level values.

The Fisher adaptation uses the Rao--Blackwellized score-difference signal

    delta_n = sqrt(alpha_n) [score(y_n) - score(x_n)],

forms the damped empirical Fisher matrix

    A_n \propto (\sum_i delta_i delta_i^T + lambda I)^{-1}

during warm-up, and freezes the preconditioner before retained sampling.
Therefore the retained chain uses a fixed preconditioned MALA kernel with an
ordinary exact MH correction.

Dependencies
------------
    numpy
    scipy
    pandas

Examples
--------
Fast smoke test:

    python conemala_fisher_adaptive_comparison.py \
        --profile smoke --output-dir results_fisher_smoke

Moderate pilot:

    python conemala_fisher_adaptive_comparison.py \
        --profile pilot --output-dir results_fisher_pilot

Paper configuration (computationally expensive):

    python conemala_fisher_adaptive_comparison.py \
        --profile paper --output-dir results_fisher_paper

The main LaTeX table is written to

    <output-dir>/table_fisher_repeated_seed_comparison.tex

and can replace the placeholder table in Section 9 of the revised manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


Array = np.ndarray
Edge = Tuple[int, int]
SQRT2 = math.sqrt(2.0)
LOG_2PI = math.log(2.0 * math.pi)


# =============================================================================
# Configuration and reproducibility
# =============================================================================


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    output_dir: str
    graph: str
    m: int
    d: int
    n_train: int
    n_test: int
    rho: float
    nu: float
    prior_scale: float
    true_weight_scale: float
    n_chains: int
    warmup: int
    retained: int
    thin: int
    pilot_steps: int
    experiment_seeds: Tuple[int, ...]
    methods: Tuple[str, ...]
    cone_h0: float
    euclidean_h0: float
    retraction_h0: float
    fisher_h0: float
    tune_fixed_methods: bool
    target_accept: float
    fisher_damping: float
    fisher_init_steps: int
    fisher_update_interval: int
    fisher_max_condition: float
    step_adapt_rate: float
    parallel_runtime: bool
    save_samples: bool


def set_seed(seed: int) -> np.random.Generator:
    """Return a NumPy generator with a fixed seed."""
    return np.random.default_rng(int(seed))


# =============================================================================
# Symmetric/SPD linear algebra
# =============================================================================


def sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


def is_spd(a: Array) -> bool:
    try:
        np.linalg.cholesky(sym(a))
        return True
    except np.linalg.LinAlgError:
        return False


def project_spd(a: Array, floor: float = 1e-8) -> Array:
    """Projection used only for truth/initialization, never for MH proposals."""
    vals, vecs = np.linalg.eigh(sym(a))
    vals = np.maximum(vals, floor)
    return sym((vecs * vals) @ vecs.T)


def spd_logdet(a: Array) -> float:
    chol = np.linalg.cholesky(sym(a))
    return float(2.0 * np.sum(np.log(np.diag(chol))))


def spd_inverse(a: Array) -> Array:
    chol = np.linalg.cholesky(sym(a))
    eye = np.eye(a.shape[0])
    y = np.linalg.solve(chol, eye)
    return sym(np.linalg.solve(chol.T, y))


def spd_sqrt_and_inv_sqrt(a: Array) -> Tuple[Array, Array]:
    vals, vecs = np.linalg.eigh(sym(a))
    if np.min(vals) <= 0.0:
        raise np.linalg.LinAlgError("Matrix is not SPD.")
    root = sym((vecs * np.sqrt(vals)) @ vecs.T)
    inv_root = sym((vecs * (1.0 / np.sqrt(vals))) @ vecs.T)
    return root, inv_root


def sym_expm(a: Array) -> Array:
    vals, vecs = np.linalg.eigh(sym(a))
    # Avoid floating overflow from a pathological rejected proposal.
    if np.max(vals) > 700.0:
        raise FloatingPointError("Symmetric exponential overflow.")
    return sym((vecs * np.exp(vals)) @ vecs.T)


def sym_logm_spd(a: Array) -> Array:
    vals, vecs = np.linalg.eigh(sym(a))
    if np.min(vals) <= 0.0:
        raise np.linalg.LinAlgError("Matrix logarithm requires SPD input.")
    return sym((vecs * np.log(vals)) @ vecs.T)


def svec(a: Array) -> Array:
    """Orthonormal vectorization of Sym(d) under the Frobenius inner product."""
    d = a.shape[0]
    out: List[float] = []
    for i in range(d):
        for j in range(i, d):
            out.append(float(a[i, j]) if i == j else float(SQRT2 * a[i, j]))
    return np.asarray(out, dtype=float)


def smat(v: Array, d: int) -> Array:
    """Inverse of svec."""
    a = np.zeros((d, d), dtype=float)
    k = 0
    for i in range(d):
        for j in range(i, d):
            if i == j:
                a[i, i] = v[k]
            else:
                a[i, j] = a[j, i] = v[k] / SQRT2
            k += 1
    return a


def flatten_weights(w: Array) -> Array:
    return np.concatenate([svec(we) for we in w], axis=0)


def unflatten_weights(x: Array, e_count: int, d: int) -> Array:
    block_dim = d * (d + 1) // 2
    return np.stack(
        [smat(x[e * block_dim : (e + 1) * block_dim], d) for e in range(e_count)],
        axis=0,
    )


def sample_standard_symmetric(d: int, rng: np.random.Generator) -> Array:
    return smat(rng.standard_normal(d * (d + 1) // 2), d)


def log_gaussian_isotropic(delta: Array, variance: float) -> float:
    q = delta.size
    return float(
        -0.5 * q * math.log(2.0 * math.pi * variance)
        - 0.5 * np.dot(delta, delta) / variance
    )


def _log_sinhc(x: float) -> float:
    """Stable log(sinh(x)/x), continuously extended at x=0."""
    ax = abs(float(x))
    if ax < 1e-4:
        x2 = ax * ax
        return x2 / 6.0 - x2 * x2 / 180.0 + x2 * x2 * x2 / 2835.0
    if ax > 20.0:
        return ax - math.log(2.0 * ax) + math.log1p(-math.exp(-2.0 * ax))
    return math.log(math.sinh(ax) / ax)


def log_spd_exp_jacobian(s: Array) -> float:
    """Log Jacobian of the SPD exponential map relative to AIRM volume."""
    eig = np.linalg.eigvalsh(sym(s))
    total = 0.0
    for i in range(eig.size):
        for j in range(i + 1, eig.size):
            total += _log_sinhc(0.5 * (eig[i] - eig[j]))
    return float(total)


def reverse_airm_tangent(w_from: Array, w_to: Array) -> Array:
    """T satisfying w_to = w_from^{1/2} exp(T) w_from^{1/2}."""
    _, inv_root = spd_sqrt_and_inv_sqrt(w_from)
    return sym_logm_spd(sym(inv_root @ w_to @ inv_root))


def log_mvn_fixed_cov(
    x: Array,
    mean: Array,
    chol_p: Array,
    logdet_p: float,
    h: float,
) -> float:
    """Log N(mean, 2 h P), where chol_p @ chol_p.T = P."""
    delta = x - mean
    whitened = np.linalg.solve(chol_p, delta)
    q = x.size
    return float(
        -0.5 * q * math.log(4.0 * math.pi * h)
        - 0.5 * logdet_p
        - np.dot(whitened, whitened) / (4.0 * h)
    )


# =============================================================================
# Graph model and posterior evaluation (Algorithm 1 notation)
# =============================================================================


@dataclass
class PosteriorEvaluation:
    log_target_dx: float
    phi: float
    score_dx: Array
    grad_phi: Array
    q_matrix: Array
    logdet_q: float
    heldout_nll: float


class GraphPosterior:
    """
    Gaussian PSD-weighted graph posterior.

    The target is specified relative to product Lebesgue measure:

        pi_Leb(W | Y) proportional to
        exp{ n/2 [log det Q(W) - tr(S_train Q(W))] }
        prod_e Wishart(W_e | nu, Psi).

    ConeMALA uses the equivalent intrinsic potential

        Phi(W) = -log pi_Leb(W|Y) - (d+1)/2 sum_e log det W_e,

    relative to product AIRM volume.
    """

    def __init__(
        self,
        edges: Sequence[Edge],
        s_train: Array,
        s_test: Array,
        n_train: int,
        m: int,
        d: int,
        rho: float,
        nu: float,
        psi: Array,
        w_true: Array,
    ) -> None:
        if nu <= d - 1:
            raise ValueError("Wishart degrees of freedom must satisfy nu > d - 1.")
        self.edges = tuple(edges)
        self.s_train = sym(np.asarray(s_train, dtype=float))
        self.s_test = sym(np.asarray(s_test, dtype=float))
        self.n_train = int(n_train)
        self.m = int(m)
        self.d = int(d)
        self.p = self.m * self.d
        self.e_count = len(self.edges)
        self.rho = float(rho)
        self.nu = float(nu)
        self.psi = sym(np.asarray(psi, dtype=float))
        self.psi_inv = spd_inverse(self.psi)
        self.w_true = np.asarray(w_true, dtype=float)
        self.volume_c = 0.5 * (self.d + 1.0)

        self.edge_ops = self._build_edge_operators()
        self.bcat = np.concatenate(self.edge_ops, axis=1)
        self.bt_s_b = np.stack(
            [sym(b.T @ self.s_train @ b) for b in self.edge_ops], axis=0
        )

    def _build_edge_operators(self) -> Tuple[Array, ...]:
        ops: List[Array] = []
        eye_d = np.eye(self.d)
        for i, j in self.edges:
            b = np.zeros((self.p, self.d), dtype=float)
            b[i * self.d : (i + 1) * self.d, :] = eye_d
            b[j * self.d : (j + 1) * self.d, :] = -eye_d
            ops.append(b)
        return tuple(ops)

    def assemble_q(self, w: Array) -> Array:
        q = self.rho * np.eye(self.p)
        for e, b in enumerate(self.edge_ops):
            q += b @ w[e] @ b.T
        return sym(q)

    def evaluate(self, w: Array) -> PosteriorEvaluation:
        """
        Evaluate target, all-edge score, and intrinsic gradient.

        One Cholesky factorization of Q(W) and one multiple-right-hand-side
        solve provide all matrices B_e^T Q(W)^{-1} B_e.
        """
        w = np.asarray(w, dtype=float)
        if w.shape != (self.e_count, self.d, self.d):
            raise ValueError(
                f"Expected W shape {(self.e_count, self.d, self.d)}, got {w.shape}."
            )

        edge_inverses = np.empty_like(w)
        edge_logdets = np.empty(self.e_count)
        for e in range(self.e_count):
            chol_e = np.linalg.cholesky(sym(w[e]))
            edge_logdets[e] = 2.0 * np.sum(np.log(np.diag(chol_e)))
            eye_d = np.eye(self.d)
            y = np.linalg.solve(chol_e, eye_d)
            edge_inverses[e] = sym(np.linalg.solve(chol_e.T, y))

        q = self.assemble_q(w)
        chol_q = np.linalg.cholesky(q)
        logdet_q = float(2.0 * np.sum(np.log(np.diag(chol_q))))

        # Shared solves: Q Z = [B_1 ... B_E].
        y = np.linalg.solve(chol_q, self.bcat)
        z = np.linalg.solve(chol_q.T, y)

        trace_train = float(np.sum(self.s_train * q))
        log_likelihood = 0.5 * self.n_train * (logdet_q - trace_train)

        log_prior = 0.0
        score_dx = np.empty_like(w)
        grad_phi = np.empty_like(w)

        for e, b in enumerate(self.edge_ops):
            z_e = z[:, e * self.d : (e + 1) * self.d]
            c_e = sym(b.T @ z_e)

            score_like = 0.5 * self.n_train * (c_e - self.bt_s_b[e])
            score_prior = (
                0.5 * (self.nu - self.d - 1.0) * edge_inverses[e]
                - 0.5 * self.psi_inv
            )
            score_dx[e] = sym(score_like + score_prior)

            # Phi = -log pi_Leb - volume_c * sum logdet W_e.
            grad_phi[e] = sym(
                -score_dx[e] - self.volume_c * edge_inverses[e]
            )

            log_prior += (
                0.5 * (self.nu - self.d - 1.0) * edge_logdets[e]
                - 0.5 * np.trace(self.psi_inv @ w[e])
            )

        log_target_dx = float(log_likelihood + log_prior)
        phi = float(-log_target_dx - self.volume_c * np.sum(edge_logdets))

        heldout_nll = 0.5 * (
            self.p * LOG_2PI - logdet_q + float(np.sum(self.s_test * q))
        )

        return PosteriorEvaluation(
            log_target_dx=log_target_dx,
            phi=phi,
            score_dx=score_dx,
            grad_phi=grad_phi,
            q_matrix=q,
            logdet_q=logdet_q,
            heldout_nll=float(heldout_nll),
        )

    def relative_w_error(self, w: Array) -> float:
        num = np.linalg.norm(np.asarray(w) - self.w_true)
        den = np.linalg.norm(self.w_true)
        return float(num / max(den, 1e-15))


# =============================================================================
# Synthetic graph and data generation
# =============================================================================


def build_edges(m: int, graph: str) -> List[Edge]:
    if m < 2:
        raise ValueError("m must be at least 2.")
    if graph == "chain":
        return [(i, i + 1) for i in range(m - 1)]
    if graph == "cycle":
        return [(i, (i + 1) % m) for i in range(m)]
    raise ValueError(f"Unknown graph type: {graph}")


def generate_true_weights(
    e_count: int,
    d: int,
    scale: float,
    rng: np.random.Generator,
) -> Array:
    w_true = np.empty((e_count, d, d), dtype=float)
    for e in range(e_count):
        a = rng.normal(size=(d, d))
        c = sym(a @ a.T / d)
        c *= d / np.trace(c)
        # Edge-specific scale and anisotropy with a uniform eigenvalue floor.
        edge_scale = scale * (0.75 + 0.5 * rng.random())
        w_true[e] = project_spd(edge_scale * (0.35 * np.eye(d) + 0.65 * c))
    return w_true


def assemble_q_from_ops(w: Array, edge_ops: Sequence[Array], rho: float) -> Array:
    p = edge_ops[0].shape[0]
    q = rho * np.eye(p)
    for e, b in enumerate(edge_ops):
        q += b @ w[e] @ b.T
    return sym(q)


def sample_from_precision(
    q: Array,
    n: int,
    rng: np.random.Generator,
) -> Array:
    chol = np.linalg.cholesky(q)
    z = rng.standard_normal((n, q.shape[0]))
    # If Q = L L^T, then L^T y = z gives Cov(y) = Q^{-1}.
    return np.linalg.solve(chol.T, z.T).T


def make_synthetic_problem(config: ExperimentConfig, seed: int) -> GraphPosterior:
    rng = set_seed(seed)
    edges = build_edges(config.m, config.graph)

    # Temporary operators for generating Q(W*).
    p = config.m * config.d
    eye_d = np.eye(config.d)
    edge_ops: List[Array] = []
    for i, j in edges:
        b = np.zeros((p, config.d))
        b[i * config.d : (i + 1) * config.d] = eye_d
        b[j * config.d : (j + 1) * config.d] = -eye_d
        edge_ops.append(b)

    w_true = generate_true_weights(
        e_count=len(edges),
        d=config.d,
        scale=config.true_weight_scale,
        rng=rng,
    )
    q_true = assemble_q_from_ops(w_true, edge_ops, config.rho)

    y_train = sample_from_precision(q_true, config.n_train, rng)
    y_test = sample_from_precision(q_true, config.n_test, rng)
    s_train = sym(y_train.T @ y_train / config.n_train)
    s_test = sym(y_test.T @ y_test / config.n_test)

    psi = config.prior_scale * np.eye(config.d)
    return GraphPosterior(
        edges=edges,
        s_train=s_train,
        s_test=s_test,
        n_train=config.n_train,
        m=config.m,
        d=config.d,
        rho=config.rho,
        nu=config.nu,
        psi=psi,
        w_true=w_true,
    )


def initial_weights(post: GraphPosterior) -> Array:
    # Wishart prior mean, used only as a common deterministic initial center.
    return np.stack([post.nu * post.psi for _ in range(post.e_count)], axis=0)


def perturb_initial_weights(
    w0: Array,
    chain_index: int,
    n_chains: int,
    rng: np.random.Generator,
    noise_scale: float = 0.08,
) -> Array:
    shift = chain_index - 0.5 * (n_chains - 1)
    out = np.empty_like(w0)
    for e in range(w0.shape[0]):
        root, _ = spd_sqrt_and_inv_sqrt(w0[e])
        tangent = (
            0.04 * shift * np.eye(w0.shape[1])
            + noise_scale * sample_standard_symmetric(w0.shape[1], rng)
        )
        out[e] = sym(root @ sym_expm(tangent) @ root)
    return out


# =============================================================================
# Chain result and retained diagnostic traces
# =============================================================================


@dataclass
class ChainResult:
    samples: Array
    traces: Dict[str, Array]
    acceptance: float
    warmup_acceptance: float
    runtime_sec: float
    h_final: float
    preconditioner_condition: float


def state_diagnostics(
    post: GraphPosterior,
    w: Array,
    evaluation: PosteriorEvaluation,
) -> Dict[str, float]:
    return {
        "phi": float(evaluation.phi),
        "logdet_q": float(evaluation.logdet_q),
        "heldout_nll": float(evaluation.heldout_nll),
        "rel_w_draw": float(post.relative_w_error(w)),
    }


def append_retained(
    samples: List[Array],
    traces: Dict[str, List[float]],
    post: GraphPosterior,
    w: Array,
    evaluation: PosteriorEvaluation,
) -> None:
    samples.append(w.copy())
    values = state_diagnostics(post, w, evaluation)
    for key, value in values.items():
        traces[key].append(value)


# =============================================================================
# ConeMALA (Algorithms 2--3 notation)
# =============================================================================


def cone_mala_chain(
    post: GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
) -> ChainResult:
    rng = set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
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
        w_prop = np.empty_like(w)
        s_forward: List[Array] = []
        m_forward: List[Array] = []
        valid = True

        try:
            for e in range(post.e_count):
                root, _ = spd_sqrt_and_inv_sqrt(w[e])
                mean_tangent = -h * sym(root @ evaluation.grad_phi[e] @ root)
                noise = sample_standard_symmetric(post.d, rng)
                tangent = sym(mean_tangent + math.sqrt(2.0 * h) * noise)
                w_prop[e] = sym(root @ sym_expm(tangent) @ root)
                # The exponential proposal should be SPD; this is only a guard.
                np.linalg.cholesky(w_prop[e])
                s_forward.append(tangent)
                m_forward.append(mean_tangent)
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            valid = False

        accepted = False
        if valid:
            try:
                eval_prop = post.evaluate(w_prop)
                logq_forward = 0.0
                logq_reverse = 0.0
                block_dim = post.d * (post.d + 1) // 2

                for e in range(post.e_count):
                    sf = s_forward[e]
                    mf = m_forward[e]
                    logq_forward += (
                        log_gaussian_isotropic(
                            svec(sf - mf), variance=2.0 * h
                        )
                        - log_spd_exp_jacobian(sf)
                    )

                    tr = reverse_airm_tangent(w_prop[e], w[e])
                    root_prop, _ = spd_sqrt_and_inv_sqrt(w_prop[e])
                    mr = -h * sym(
                        root_prop @ eval_prop.grad_phi[e] @ root_prop
                    )
                    logq_reverse += (
                        log_gaussian_isotropic(
                            svec(tr - mr), variance=2.0 * h
                        )
                        - log_spd_exp_jacobian(tr)
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
                accepted = False

        if iteration < warmup:
            accepted_warm += int(accepted)
        else:
            proposed_keep += 1
            accepted_keep += int(accepted)
            if (iteration - warmup) % thin == 0:
                append_retained(samples, traces, post, w, evaluation)

    runtime = time.perf_counter() - t0
    return ChainResult(
        samples=np.asarray(samples),
        traces={k: np.asarray(v) for k, v in traces.items()},
        acceptance=accepted_keep / max(proposed_keep, 1),
        warmup_acceptance=accepted_warm / max(warmup, 1),
        runtime_sec=float(runtime),
        h_final=float(h),
        preconditioner_condition=1.0,
    )


# =============================================================================
# Euclidean MALA in orthonormal product-symmetric coordinates
# =============================================================================


def euclidean_mala_chain(
    post: GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
) -> ChainResult:
    rng = set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    x = flatten_weights(w)

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
        score_x = flatten_weights(evaluation.score_dx)
        mean_x = x + h * score_x
        y = mean_x + math.sqrt(2.0 * h) * rng.standard_normal(x.size)
        w_prop = unflatten_weights(y, post.e_count, post.d)

        accepted = False
        if all(is_spd(w_prop[e]) for e in range(post.e_count)):
            try:
                eval_prop = post.evaluate(w_prop)
                score_y = flatten_weights(eval_prop.score_dx)
                mean_y = y + h * score_y
                logq_forward = log_gaussian_isotropic(y - mean_x, 2.0 * h)
                logq_reverse = log_gaussian_isotropic(x - mean_y, 2.0 * h)
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
                accepted = False

        if iteration < warmup:
            accepted_warm += int(accepted)
        else:
            proposed_keep += 1
            accepted_keep += int(accepted)
            if (iteration - warmup) % thin == 0:
                append_retained(samples, traces, post, w, evaluation)

    runtime = time.perf_counter() - t0
    return ChainResult(
        samples=np.asarray(samples),
        traces={k: np.asarray(v) for k, v in traces.items()},
        acceptance=accepted_keep / max(proposed_keep, 1),
        warmup_acceptance=accepted_warm / max(warmup, 1),
        runtime_sec=float(runtime),
        h_final=float(h),
        preconditioner_condition=1.0,
    )


# =============================================================================
# Retraction-based RMALA baseline
# =============================================================================


def retraction_rmala_chain(
    post: GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
) -> ChainResult:
    rng = set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    volume_c = 0.5 * (post.d + 1.0)

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
        w_prop = np.empty_like(w)
        tangent_forward: List[Array] = []
        tangent_mean_forward: List[Array] = []
        roots: List[Array] = []
        valid = True

        try:
            for e in range(post.e_count):
                root, _ = spd_sqrt_and_inv_sqrt(w[e])
                mean_tangent = h * sym(root @ evaluation.score_dx[e] @ root)
                tangent = sym(
                    mean_tangent
                    + math.sqrt(2.0 * h) * sample_standard_symmetric(post.d, rng)
                )
                w_prop[e] = sym(w[e] + root @ tangent @ root)
                if not is_spd(w_prop[e]):
                    valid = False
                    break
                roots.append(root)
                tangent_forward.append(tangent)
                tangent_mean_forward.append(mean_tangent)
        except (np.linalg.LinAlgError, FloatingPointError):
            valid = False

        accepted = False
        if valid:
            try:
                eval_prop = post.evaluate(w_prop)
                logq_forward = 0.0
                logq_reverse = 0.0

                for e in range(post.e_count):
                    sf = tangent_forward[e]
                    mf = tangent_mean_forward[e]
                    logq_forward += (
                        log_gaussian_isotropic(svec(sf - mf), 2.0 * h)
                        - volume_c * spd_logdet(w[e])
                    )

                    root_prop, inv_root_prop = spd_sqrt_and_inv_sqrt(w_prop[e])
                    sr = sym(inv_root_prop @ (w[e] - w_prop[e]) @ inv_root_prop)
                    mr = h * sym(
                        root_prop @ eval_prop.score_dx[e] @ root_prop
                    )
                    logq_reverse += (
                        log_gaussian_isotropic(svec(sr - mr), 2.0 * h)
                        - volume_c * spd_logdet(w_prop[e])
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
                accepted = False

        if iteration < warmup:
            accepted_warm += int(accepted)
        else:
            proposed_keep += 1
            accepted_keep += int(accepted)
            if (iteration - warmup) % thin == 0:
                append_retained(samples, traces, post, w, evaluation)

    runtime = time.perf_counter() - t0
    return ChainResult(
        samples=np.asarray(samples),
        traces={k: np.asarray(v) for k, v in traces.items()},
        acceptance=accepted_keep / max(proposed_keep, 1),
        warmup_acceptance=accepted_warm / max(warmup, 1),
        runtime_sec=float(runtime),
        h_final=float(h),
        preconditioner_condition=1.0,
    )


# =============================================================================
# Fisher-adaptive MALA baseline
# =============================================================================


def _fisher_sqrt_initialize(signal: Array, damping: float) -> Array:
    """
    Initialize R so that

        R R^T = (signal signal^T + damping I)^{-1}.

    This is the square-root inverse-Fisher initialization used in the
    FisherMALA recursion.
    """
    signal = np.asarray(signal, dtype=float)
    norm2 = float(np.dot(signal, signal))
    coefficient = 1.0 / (1.0 + math.sqrt(damping / (damping + norm2)))
    return (
        np.eye(signal.size)
        - coefficient * np.outer(signal, signal) / (damping + norm2)
    ) / math.sqrt(damping)


def _fisher_sqrt_update(root: Array, signal: Array) -> Array:
    """
    Rank-one O(q^2) square-root update.

    If root root^T = (sum_{i<n} s_i s_i^T + lambda I)^{-1}, the returned
    matrix satisfies

        root_new root_new^T
        = (sum_{i<=n} s_i s_i^T + lambda I)^{-1}.
    """
    phi = root.T @ signal
    denominator = 1.0 + float(np.dot(phi, phi))
    coefficient = 1.0 / (1.0 + math.sqrt(1.0 / denominator))
    return root - coefficient * np.outer(root @ phi, phi) / denominator


def _normalized_fisher_action(
    root: Array,
    vector: Array,
) -> Tuple[Array, float]:
    """Return [A/tr(A)/q] vector and average eigenvalue of A=RR^T."""
    q = root.shape[0]
    average_eigenvalue = float(np.sum(root * root) / q)
    average_eigenvalue = max(average_eigenvalue, np.finfo(float).tiny)
    action = root @ (root.T @ vector) / average_eigenvalue
    return action, average_eigenvalue


def _fisher_mala_log_proposal_ratio(
    x: Array,
    y: Array,
    score_x: Array,
    score_y: Array,
    root: Array,
    average_eigenvalue: float,
    h: float,
) -> float:
    """
    Compute log q(x|y)-log q(y|x) without forming A^{-1}.

    The proposal is N(x + h P score_x, 2h P), where
    P = RR^T / [tr(RR^T)/q].
    """
    p_score_x = root @ (root.T @ score_x) / average_eigenvalue
    p_score_y = root @ (root.T @ score_y) / average_eigenvalue
    h_xy = 0.5 * np.dot(x - y - 0.5 * h * p_score_y, score_y)
    h_yx = 0.5 * np.dot(y - x - 0.5 * h * p_score_x, score_x)
    return float(h_xy - h_yx)

def fisher_adaptive_mala_chain(
    post: GraphPosterior,
    w_init: Array,
    warmup: int,
    retained: int,
    thin: int,
    h: float,
    seed: int,
    target_accept: float,
    damping: float,
    init_steps: int,
    update_interval: int,
    max_condition: float,
    step_adapt_rate: float,
) -> ChainResult:
    """
    Fisher-adaptive Euclidean MALA with warm-up-only adaptation.

    During the initialization phase the method uses isotropic MALA.  It then
    adapts a square root R of the damped inverse empirical Fisher matrix from
    Rao--Blackwellized score increments.  The proposal is

        y ~ N(x + h P score(x), 2h P),
        P = RR^T / {tr(RR^T)/q}.

    Both h and R are frozen before retained sampling, and the fixed collection
    kernel uses an exact Metropolis--Hastings correction.
    """
    if warmup < 1:
        raise ValueError("Fisher-adaptive MALA requires positive warmup.")
    if update_interval < 1:
        raise ValueError("update_interval must be at least one.")

    rng = set_seed(seed)
    w = np.asarray(w_init, dtype=float).copy()
    evaluation = post.evaluate(w)
    x = flatten_weights(w)
    qdim = x.size

    root = np.eye(qdim)
    fisher_initialized = False
    log_h = math.log(h)
    accepted_warm = 0
    t0 = time.perf_counter()

    # --------------------------- adaptive warm-up ---------------------------
    for iteration in range(warmup):
        h_now = float(math.exp(log_h))
        score_x = flatten_weights(evaluation.score_dx)
        p_score_x, average_eigenvalue = _normalized_fisher_action(root, score_x)
        mean_x = x + h_now * p_score_x
        noise = (
            math.sqrt(2.0 * h_now / average_eigenvalue)
            * (root @ rng.standard_normal(qdim))
        )
        y = mean_x + noise
        w_prop = unflatten_weights(y, post.e_count, post.d)

        alpha_prob = 0.0
        accepted = False
        score_y: Optional[Array] = None
        eval_prop: Optional[PosteriorEvaluation] = None

        if all(is_spd(w_prop[e]) for e in range(post.e_count)):
            try:
                eval_prop = post.evaluate(w_prop)
                score_y = flatten_weights(eval_prop.score_dx)
                log_q_ratio = _fisher_mala_log_proposal_ratio(
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
                    accepted = True
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                alpha_prob = 0.0

        accepted_warm += int(accepted)

        # First init_steps iterations use isotropic MALA. Thereafter, adapt R
        # from the Rao--Blackwellized score increment. By default every signal
        # is used; update_interval>1 is available only as a cost-saving ablation.
        use_signal = (
            iteration >= init_steps
            and (iteration - init_steps) % update_interval == 0
            and score_y is not None
            and alpha_prob > 0.0
        )
        if use_signal:
            signal = math.sqrt(alpha_prob) * (score_y - score_x)
            if np.all(np.isfinite(signal)):
                if not fisher_initialized:
                    root = _fisher_sqrt_initialize(signal, damping)
                    fisher_initialized = True
                else:
                    root = _fisher_sqrt_update(root, signal)

        # Multiplicative adaptation of sigma^2=2h towards alpha*=0.574.
        factor = 1.0 + step_adapt_rate * (alpha_prob - target_accept)
        factor = max(factor, 1e-6)
        log_h += math.log(factor)
        log_h = float(np.clip(log_h, math.log(1e-12), math.log(1.0)))

    h_frozen = float(math.exp(log_h))
    # The condition number is computed once, after adaptation.
    _, singular_values, _ = np.linalg.svd(root, full_matrices=False)
    singular_values = np.maximum(singular_values, np.finfo(float).tiny)
    condition_p = float((np.max(singular_values) / np.min(singular_values)) ** 2)
    if condition_p > max_condition:
        # Numerical guard: clip the frozen preconditioner once. This should be
        # rare when damping is adequate. It is recorded in the output condition.
        a = sym(root @ root.T)
        eig, vec = np.linalg.eigh(a)
        largest = float(np.max(eig))
        eig = np.maximum(eig, largest / max_condition)
        root = np.linalg.cholesky(sym((vec * eig) @ vec.T))
        condition_p = float(np.max(eig) / np.min(eig))

    # ------------------------- fixed sampling phase -------------------------
    samples: List[Array] = []
    traces: Dict[str, List[float]] = {
        "phi": [],
        "logdet_q": [],
        "heldout_nll": [],
        "rel_w_draw": [],
    }
    accepted_keep = 0
    proposed_keep = 0

    for iteration in range(retained * thin):
        score_x = flatten_weights(evaluation.score_dx)
        p_score_x, average_eigenvalue = _normalized_fisher_action(root, score_x)
        mean_x = x + h_frozen * p_score_x
        y = mean_x + (
            math.sqrt(2.0 * h_frozen / average_eigenvalue)
            * (root @ rng.standard_normal(qdim))
        )
        w_prop = unflatten_weights(y, post.e_count, post.d)
        accepted = False

        if all(is_spd(w_prop[e]) for e in range(post.e_count)):
            try:
                eval_prop = post.evaluate(w_prop)
                score_y = flatten_weights(eval_prop.score_dx)
                log_q_ratio = _fisher_mala_log_proposal_ratio(
                    x=x,
                    y=y,
                    score_x=score_x,
                    score_y=score_y,
                    root=root,
                    average_eigenvalue=average_eigenvalue,
                    h=h_frozen,
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

        proposed_keep += 1
        accepted_keep += int(accepted)
        if iteration % thin == 0:
            append_retained(samples, traces, post, w, evaluation)

    runtime = time.perf_counter() - t0
    return ChainResult(
        samples=np.asarray(samples),
        traces={k: np.asarray(v) for k, v in traces.items()},
        acceptance=accepted_keep / max(proposed_keep, 1),
        warmup_acceptance=accepted_warm / max(warmup, 1),
        runtime_sec=float(runtime),
        h_final=h_frozen,
        preconditioner_condition=condition_p,
    )


# =============================================================================
# Diagnostics: rank-normalized split-Rhat, ESS, MCSE
# =============================================================================


def _split_chains(chains: Array) -> Array:
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must have shape (n_chains, n_draws).")
    _, n = chains.shape
    half = n // 2
    if half < 2:
        raise ValueError("At least four retained draws per chain are required.")
    return np.concatenate([chains[:, :half], chains[:, n - half :]], axis=0)


def _classic_rhat(chains: Array, eps: float = 1e-15) -> float:
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    chain_means = np.mean(chains, axis=1)
    between = n * np.var(chain_means, ddof=1)
    within = np.mean(np.var(chains, axis=1, ddof=1))
    if within < eps:
        return 1.0 if between < eps else float("inf")
    var_hat = (n - 1.0) / n * within + between / n
    return float(math.sqrt(var_hat / within))


def _rank_normalize(values: Array) -> Array:
    flat = np.asarray(values, dtype=float).reshape(-1)
    ranks = rankdata(flat, method="average")
    z = norm.ppf((ranks - 3.0 / 8.0) / (flat.size + 1.0 / 4.0))
    return z.reshape(np.asarray(values).shape)


def rank_normalized_split_rhat(chains: Array) -> float:
    split = _split_chains(chains)
    bulk = _classic_rhat(_rank_normalize(split))
    median = float(np.median(split))
    folded = np.abs(split - median)
    tail = _classic_rhat(_rank_normalize(folded))
    return float(max(bulk, tail))


def _autocorrelation_fft(x: Array, eps: float = 1e-15) -> Array:
    x = np.asarray(x, dtype=float)
    n = x.size
    centered = x - np.mean(x)
    variance_mass = float(np.dot(centered, centered))
    if variance_mass < eps:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    fft = np.fft.rfft(np.concatenate([centered, np.zeros(n)]))
    acov = np.fft.irfft(fft * np.conjugate(fft), n=2 * n)[:n]
    return np.asarray(acov / acov[0], dtype=float)


def effective_sample_size(chains: Array) -> float:
    """
    Conservative multi-chain ESS using rank-normalized chains and Geyer's
    initial positive paired sequence.
    """
    chains = np.asarray(chains, dtype=float)
    z = _rank_normalize(chains)
    m, n = z.shape
    rho = np.mean(np.stack([_autocorrelation_fft(z[c]) for c in range(m)]), axis=0)
    tau = 1.0
    for lag in range(1, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if not np.isfinite(pair) or pair < 0.0:
            break
        tau += 2.0 * pair
    tau = max(tau, 1.0)
    return float(min(m * n, max(1.0, m * n / tau)))


def mcse_mean(chains: Array, ess: Optional[float] = None) -> float:
    flat = np.asarray(chains, dtype=float).reshape(-1)
    if ess is None:
        ess = effective_sample_size(chains)
    return float(np.std(flat, ddof=1) / math.sqrt(max(ess, 1.0)))


# =============================================================================
# Pilot tuning and method dispatch
# =============================================================================


FIXED_METHODS = ("ConeMALA", "EuclideanMALA", "RetractionRMALA")
ALL_METHODS = (
    "ConeMALA",
    "FisherAdaptiveMALA",
    "EuclideanMALA",
    "RetractionRMALA",
)


def method_label(name: str) -> str:
    return {
        "ConeMALA": "ConeMALA",
        "FisherAdaptiveMALA": "Fisher-adaptive MALA",
        "EuclideanMALA": "Euclidean MALA",
        "RetractionRMALA": "Retraction RMALA",
    }[name]


def fixed_runner(name: str) -> Callable[..., ChainResult]:
    return {
        "ConeMALA": cone_mala_chain,
        "EuclideanMALA": euclidean_mala_chain,
        "RetractionRMALA": retraction_rmala_chain,
    }[name]


def pilot_tune_h(
    method: str,
    post: GraphPosterior,
    w_init: Array,
    h0: float,
    pilot_steps: int,
    target_accept: float,
    seed: int,
) -> Tuple[float, float, float, List[Dict[str, float]]]:
    """Use the same five-point multiplicative pilot grid for fixed methods."""
    candidates = [h0 * factor for factor in (0.25, 0.5, 1.0, 2.0, 4.0)]
    runner = fixed_runner(method)
    rows: List[Dict[str, float]] = []
    total_runtime = 0.0

    for index, h in enumerate(candidates):
        result = runner(
            post=post,
            w_init=w_init,
            warmup=0,
            retained=pilot_steps,
            thin=1,
            h=h,
            seed=seed + 17 * index,
        )
        total_runtime += result.runtime_sec
        rows.append(
            {
                "method": method,
                "h": float(h),
                "pilot_acceptance": float(result.acceptance),
                "pilot_runtime_sec": float(result.runtime_sec),
            }
        )

    def pilot_loss(row: Mapping[str, float]) -> float:
        acc = row["pilot_acceptance"]
        # Penalize degenerate acceptance more strongly than a mild target miss.
        boundary_penalty = 2.0 if (acc < 0.05 or acc > 0.98) else 0.0
        return abs(acc - target_accept) + boundary_penalty

    best = min(rows, key=pilot_loss)
    return (
        float(best["h"]),
        float(best["pilot_acceptance"]),
        float(total_runtime),
        rows,
    )


# =============================================================================
# One independent experiment seed
# =============================================================================


def summarize_method_seed(
    method: str,
    experiment_seed: int,
    post: GraphPosterior,
    chain_results: Sequence[ChainResult],
    pilot_runtime_sec: float,
    parallel_runtime: bool,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    trace_names = ("phi", "logdet_q", "heldout_nll", "rel_w_draw")
    traces = {
        key: np.stack([result.traces[key] for result in chain_results], axis=0)
        for key in trace_names
    }

    rhats = {key: rank_normalized_split_rhat(value) for key, value in traces.items()}
    esses = {key: effective_sample_size(value) for key, value in traces.items()}
    mcse = {key: mcse_mean(traces[key], esses[key]) for key in trace_names}

    chain_times = np.asarray([r.runtime_sec for r in chain_results])
    sampling_runtime = float(np.max(chain_times) if parallel_runtime else np.sum(chain_times))
    total_runtime = float(pilot_runtime_sec + sampling_runtime)
    ess_per_sec = {key: esses[key] / max(total_runtime, 1e-15) for key in trace_names}

    all_samples = np.concatenate([r.samples for r in chain_results], axis=0)
    posterior_mean_w = np.mean(all_samples, axis=0)
    recovery_error = post.relative_w_error(posterior_mean_w)

    row: Dict[str, float] = {
        "experiment_seed": int(experiment_seed),
        "method": method,
        "acceptance": float(np.mean([r.acceptance for r in chain_results])),
        "warmup_acceptance": float(
            np.mean([r.warmup_acceptance for r in chain_results])
        ),
        "h_final_mean": float(np.mean([r.h_final for r in chain_results])),
        "preconditioner_condition_mean": float(
            np.mean([r.preconditioner_condition for r in chain_results])
        ),
        "pilot_runtime_sec": float(pilot_runtime_sec),
        "sampling_runtime_sec": sampling_runtime,
        "total_runtime_sec": total_runtime,
        "min_ess_per_sec": float(min(ess_per_sec.values())),
        "mcse_heldout_nll": float(mcse["heldout_nll"]),
        "max_rhat": float(max(rhats.values())),
        "heldout_nll": float(np.mean(traces["heldout_nll"])),
        "relative_w_error": float(recovery_error),
    }

    diagnostic_rows: List[Dict[str, float]] = []
    for key in trace_names:
        row[f"rhat_{key}"] = rhats[key]
        row[f"ess_{key}"] = esses[key]
        row[f"ess_per_sec_{key}"] = ess_per_sec[key]
        row[f"mcse_{key}"] = mcse[key]
        diagnostic_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "method": method,
                "observable": key,
                "rhat": rhats[key],
                "ess": esses[key],
                "ess_per_sec": ess_per_sec[key],
                "mcse": mcse[key],
            }
        )

    return row, diagnostic_rows


def run_one_experiment_seed(
    config: ExperimentConfig,
    experiment_seed: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]], Dict[str, Array]]:
    print("\n" + "=" * 88)
    print(f"Independent experiment seed: {experiment_seed}")
    print("=" * 88)

    post = make_synthetic_problem(config, experiment_seed)
    w_center = initial_weights(post)
    seed_rows: List[Dict[str, float]] = []
    diagnostic_rows: List[Dict[str, float]] = []
    tuning_rows: List[Dict[str, float]] = []
    saved_samples: Dict[str, Array] = {}

    h0_map = {
        "ConeMALA": config.cone_h0,
        "EuclideanMALA": config.euclidean_h0,
        "RetractionRMALA": config.retraction_h0,
    }

    for method_index, method in enumerate(config.methods):
        print(f"\n{method_label(method)}")
        print("-" * 88)

        pilot_runtime = 0.0
        h_used = config.fisher_h0 if method == "FisherAdaptiveMALA" else h0_map[method]

        if method in FIXED_METHODS and config.tune_fixed_methods:
            h_used, pilot_acc, pilot_runtime, rows = pilot_tune_h(
                method=method,
                post=post,
                w_init=w_center,
                h0=h_used,
                pilot_steps=config.pilot_steps,
                target_accept=config.target_accept,
                seed=10_000 + experiment_seed + 100 * method_index,
            )
            for row in rows:
                row["experiment_seed"] = int(experiment_seed)
                tuning_rows.append(row)
            print(
                f"pilot selected h={h_used:.3e}, "
                f"acceptance={pilot_acc:.3f}, pilot time={pilot_runtime:.1f}s"
            )

        chain_results: List[ChainResult] = []
        for chain_index in range(config.n_chains):
            init_rng = set_seed(
                1_000_000 + 10_000 * experiment_seed + 100 * method_index + chain_index
            )
            w_init = perturb_initial_weights(
                w_center,
                chain_index=chain_index,
                n_chains=config.n_chains,
                rng=init_rng,
            )
            chain_seed = (
                2_000_000
                + 10_000 * experiment_seed
                + 100 * method_index
                + chain_index
            )

            if method == "ConeMALA":
                result = cone_mala_chain(
                    post, w_init, config.warmup, config.retained,
                    config.thin, h_used, chain_seed
                )
            elif method == "EuclideanMALA":
                result = euclidean_mala_chain(
                    post, w_init, config.warmup, config.retained,
                    config.thin, h_used, chain_seed
                )
            elif method == "RetractionRMALA":
                result = retraction_rmala_chain(
                    post, w_init, config.warmup, config.retained,
                    config.thin, h_used, chain_seed
                )
            elif method == "FisherAdaptiveMALA":
                result = fisher_adaptive_mala_chain(
                    post=post,
                    w_init=w_init,
                    warmup=config.warmup,
                    retained=config.retained,
                    thin=config.thin,
                    h=h_used,
                    seed=chain_seed,
                    target_accept=config.target_accept,
                    damping=config.fisher_damping,
                    init_steps=min(config.fisher_init_steps, config.warmup),
                    update_interval=config.fisher_update_interval,
                    max_condition=config.fisher_max_condition,
                    step_adapt_rate=config.step_adapt_rate,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            chain_results.append(result)
            print(
                f"chain {chain_index + 1}: "
                f"acc={result.acceptance:.3f}, warmup_acc={result.warmup_acceptance:.3f}, "
                f"h={result.h_final:.3e}, time={result.runtime_sec:.1f}s"
            )

        row, diag = summarize_method_seed(
            method=method,
            experiment_seed=experiment_seed,
            post=post,
            chain_results=chain_results,
            pilot_runtime_sec=pilot_runtime,
            parallel_runtime=config.parallel_runtime,
        )
        seed_rows.append(row)
        diagnostic_rows.extend(diag)

        print(
            "summary: "
            f"acc={row['acceptance']:.3f}, "
            f"min ESS/s={row['min_ess_per_sec']:.3f}, "
            f"MCSE(NLL)={row['mcse_heldout_nll']:.4g}, "
            f"max Rhat={row['max_rhat']:.3f}, "
            f"NLL={row['heldout_nll']:.4f}, "
            f"relWerr={row['relative_w_error']:.4f}"
        )

        if config.save_samples:
            saved_samples[f"{method}_seed{experiment_seed}"] = np.stack(
                [r.samples for r in chain_results], axis=0
            )

    return seed_rows, diagnostic_rows, tuning_rows, saved_samples


# =============================================================================
# Across-seed aggregation and LaTeX output
# =============================================================================


PRIMARY_METRICS = (
    "acceptance",
    "min_ess_per_sec",
    "mcse_heldout_nll",
    "max_rhat",
    "heldout_nll",
    "relative_w_error",
)


def aggregate_across_seeds(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for method, group in seed_df.groupby("method", sort=False):
        row: Dict[str, float] = {
            "method": method,
            "n_experiment_seeds": int(group["experiment_seed"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def latex_mean_sd(mean: float, sd: float, digits: int) -> str:
    if not np.isfinite(mean) or not np.isfinite(sd):
        return "--"
    return f"${mean:.{digits}f}\\pm {sd:.{digits}f}$"


def write_latex_table(summary_df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Repeated-seed sampler comparison on the synthetic "
            r"PSD-weighted graph posterior. Entries are mean $\pm$ standard "
            r"deviation across independent data-generation seeds. The ESS/sec "
            r"column is the minimum over $\Phi(W)$, $\log\det Q(W)$, held-out "
            r"NLL, and the per-draw relative edge error. MCSE is reported for "
            r"the posterior mean held-out NLL, and $\widehat R_{\max}$ is the "
            r"maximum rank-normalized split-$\widehat R$ over the same observables.}"
        ),
        r"\label{tab:fisher_repeated_seed_comparison}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Method & Acceptance & Min. ESS/sec & MCSE(NLL) & "
            r"$\widehat R_{\max}$ & Held-out NLL & Rel. $W$ error \\"
        ),
        r"\midrule",
    ]

    preferred_order = list(ALL_METHODS)
    summary_df = summary_df.copy()
    summary_df["_order"] = summary_df["method"].map(
        {name: i for i, name in enumerate(preferred_order)}
    )
    summary_df = summary_df.sort_values("_order")

    for _, row in summary_df.iterrows():
        lines.append(
            " & ".join(
                [
                    method_label(str(row["method"])),
                    latex_mean_sd(row["acceptance_mean"], row["acceptance_sd"], 3),
                    latex_mean_sd(
                        row["min_ess_per_sec_mean"], row["min_ess_per_sec_sd"], 2
                    ),
                    latex_mean_sd(
                        row["mcse_heldout_nll_mean"],
                        row["mcse_heldout_nll_sd"],
                        4,
                    ),
                    latex_mean_sd(row["max_rhat_mean"], row["max_rhat_sd"], 3),
                    latex_mean_sd(row["heldout_nll_mean"], row["heldout_nll_sd"], 3),
                    latex_mean_sd(
                        row["relative_w_error_mean"],
                        row["relative_w_error_sd"],
                        3,
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
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notation_snippet(path: Path) -> None:
    text = r"""
% Insert immediately before Table~\ref{tab:fisher_repeated_seed_comparison}.
For each method and independent data-generation seed, we run multiple chains
against the same product-Lebesgue posterior.  Let
\(f\in\{\Phi,\log\det Q,\NLL_{\rm test},e_W\}\), where
\[
e_W(W)=
\frac{\bigl(\sum_{e\in\mathcal E}\|W_e-W_e^\star\|_F^2\bigr)^{1/2}}
     {\bigl(\sum_{e\in\mathcal E}\|W_e^\star\|_F^2\bigr)^{1/2}}.
\]
We report the conservative efficiency
\[
\min_f \frac{\ESS(f)}{t_{\rm total}},
\]
where \(t_{\rm total}\) includes pilot adaptation and sequential chain time.
The reported Monte Carlo standard error is
\[
\MCSE(\overline{\NLL}_{\rm test})
=
\sqrt{\frac{\widehat{\Var}(\NLL_{\rm test})}
           {\ESS(\NLL_{\rm test})}},
\]
and \(\widehat R_{\max}=\max_f\widehat R(f)\).  Held-out NLL is the
posterior average of the per-observation Gaussian negative log-likelihood,
while the final recovery error is computed from the posterior mean
\(\overline W\).
""".lstrip()
    path.write_text(text, encoding="utf-8")


# =============================================================================
# CLI and main
# =============================================================================


def profile_defaults(profile: str) -> Dict[str, object]:
    if profile == "smoke":
        return dict(
            m=5,
            d=2,
            n_train=80,
            n_test=50,
            n_chains=2,
            warmup=40,
            retained=60,
            thin=1,
            pilot_steps=20,
            experiment_seeds=(11, 22),
            cone_h0=5e-3,
            euclidean_h0=5e-5,
            retraction_h0=2e-4,
            fisher_h0=5e-5,
            fisher_update_interval=1,
            fisher_init_steps=10,
        )
    if profile == "pilot":
        return dict(
            m=10,
            d=3,
            n_train=200,
            n_test=100,
            n_chains=4,
            warmup=500,
            retained=1000,
            thin=1,
            pilot_steps=150,
            experiment_seeds=(11, 22, 33),
            cone_h0=5e-3,
            euclidean_h0=2e-5,
            retraction_h0=1e-4,
            fisher_h0=2e-5,
            fisher_update_interval=1,
            fisher_init_steps=100,
        )
    if profile == "paper":
        return dict(
            m=20,
            d=5,
            n_train=400,
            n_test=200,
            n_chains=4,
            warmup=2000,
            retained=4000,
            thin=1,
            pilot_steps=400,
            experiment_seeds=(11, 22, 33, 44, 55),
            cone_h0=5e-3,
            euclidean_h0=1e-5,
            retraction_h0=5e-5,
            fisher_h0=1e-5,
            fisher_update_interval=1,
            fisher_init_steps=500,
        )
    raise ValueError(f"Unknown profile: {profile}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeated-seed ConeMALA/Fisher-adaptive MALA comparison."
    )
    parser.add_argument("--profile", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--output-dir", type=str, default="results_fisher_comparison")
    parser.add_argument("--graph", choices=("chain", "cycle"), default="cycle")
    parser.add_argument("--m", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--nu", type=float)
    parser.add_argument("--prior-scale", type=float, default=0.05)
    parser.add_argument("--true-weight-scale", type=float, default=0.4)
    parser.add_argument("--n-chains", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--retained", type=int)
    parser.add_argument("--thin", type=int)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--experiment-seeds", nargs="+", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=list(ALL_METHODS),
    )
    parser.add_argument("--cone-h0", type=float)
    parser.add_argument("--euclidean-h0", type=float)
    parser.add_argument("--retraction-h0", type=float)
    parser.add_argument("--fisher-h0", type=float)
    parser.add_argument("--no-tune-fixed-methods", action="store_true")
    parser.add_argument("--target-accept", type=float, default=0.574)
    parser.add_argument("--fisher-damping", type=float, default=10.0)
    parser.add_argument("--fisher-init-steps", type=int)
    parser.add_argument("--fisher-update-interval", type=int)
    parser.add_argument("--fisher-max-condition", type=float, default=1e6)
    parser.add_argument("--step-adapt-rate", type=float, default=0.015)
    parser.add_argument(
        "--parallel-runtime",
        action="store_true",
        help="Use max(chain times) rather than sum(chain times) in ESS/sec.",
    )
    parser.add_argument("--save-samples", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    defaults = profile_defaults(args.profile)

    def choose(name: str):
        value = getattr(args, name)
        return defaults[name] if value is None else value

    d = int(choose("d"))
    nu = float(args.nu if args.nu is not None else d + 3.0)
    return ExperimentConfig(
        profile=args.profile,
        output_dir=args.output_dir,
        graph=args.graph,
        m=int(choose("m")),
        d=d,
        n_train=int(choose("n_train")),
        n_test=int(choose("n_test")),
        rho=float(args.rho),
        nu=nu,
        prior_scale=float(args.prior_scale),
        true_weight_scale=float(args.true_weight_scale),
        n_chains=int(choose("n_chains")),
        warmup=int(choose("warmup")),
        retained=int(choose("retained")),
        thin=int(choose("thin")),
        pilot_steps=int(choose("pilot_steps")),
        experiment_seeds=tuple(
            defaults["experiment_seeds"]
            if args.experiment_seeds is None
            else args.experiment_seeds
        ),
        methods=tuple(args.methods),
        cone_h0=float(choose("cone_h0")),
        euclidean_h0=float(choose("euclidean_h0")),
        retraction_h0=float(choose("retraction_h0")),
        fisher_h0=float(choose("fisher_h0")),
        tune_fixed_methods=not args.no_tune_fixed_methods,
        target_accept=float(args.target_accept),
        fisher_damping=float(args.fisher_damping),
        fisher_init_steps=int(choose("fisher_init_steps")),
        fisher_update_interval=int(choose("fisher_update_interval")),
        fisher_max_condition=float(args.fisher_max_condition),
        step_adapt_rate=float(args.step_adapt_rate),
        parallel_runtime=bool(args.parallel_runtime),
        save_samples=bool(args.save_samples),
    )


def validate_config(config: ExperimentConfig) -> None:
    if config.nu <= config.d - 1:
        raise ValueError("nu must satisfy nu > d - 1.")
    if config.rho <= 0.0:
        raise ValueError("rho must be positive.")
    if config.n_chains < 2:
        raise ValueError("At least two chains are required for Rhat.")
    if config.retained < 8:
        raise ValueError("At least eight retained draws per chain are required.")
    if config.warmup < 1 and "FisherAdaptiveMALA" in config.methods:
        raise ValueError("FisherAdaptiveMALA requires positive warmup.")
    if config.fisher_damping <= 0.0:
        raise ValueError("fisher_damping must be positive.")
    if config.fisher_init_steps < 0:
        raise ValueError("fisher_init_steps must be nonnegative.")
    if config.thin < 1:
        raise ValueError("thin must be at least 1.")


def main() -> None:
    args = build_parser().parse_args()
    config = resolve_config(args)
    validate_config(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    print("Repeated-seed PSD-weighted graph sampler comparison")
    print("=" * 88)
    print(json.dumps(asdict(config), indent=2))

    all_seed_rows: List[Dict[str, float]] = []
    all_diagnostic_rows: List[Dict[str, float]] = []
    all_tuning_rows: List[Dict[str, float]] = []
    all_saved_samples: Dict[str, Array] = {}

    for experiment_seed in config.experiment_seeds:
        seed_rows, diagnostic_rows, tuning_rows, saved_samples = run_one_experiment_seed(
            config, experiment_seed
        )
        all_seed_rows.extend(seed_rows)
        all_diagnostic_rows.extend(diagnostic_rows)
        all_tuning_rows.extend(tuning_rows)
        all_saved_samples.update(saved_samples)

        # Checkpoint after every independent seed.
        pd.DataFrame(all_seed_rows).to_csv(
            output_dir / "sampler_comparison_by_seed.csv", index=False
        )
        pd.DataFrame(all_diagnostic_rows).to_csv(
            output_dir / "sampler_diagnostics_by_seed.csv", index=False
        )
        pd.DataFrame(all_tuning_rows).to_csv(
            output_dir / "pilot_tuning.csv", index=False
        )

    seed_df = pd.DataFrame(all_seed_rows)
    diagnostic_df = pd.DataFrame(all_diagnostic_rows)
    tuning_df = pd.DataFrame(all_tuning_rows)
    summary_df = aggregate_across_seeds(seed_df)

    seed_df.to_csv(output_dir / "sampler_comparison_by_seed.csv", index=False)
    diagnostic_df.to_csv(output_dir / "sampler_diagnostics_by_seed.csv", index=False)
    tuning_df.to_csv(output_dir / "pilot_tuning.csv", index=False)
    summary_df.to_csv(output_dir / "sampler_comparison_across_seeds.csv", index=False)

    write_latex_table(
        summary_df,
        output_dir / "table_fisher_repeated_seed_comparison.tex",
    )
    write_notation_snippet(
        output_dir / "fisher_comparison_notation.tex"
    )

    if config.save_samples and all_saved_samples:
        np.savez_compressed(output_dir / "retained_samples.npz", **all_saved_samples)

    print("\n" + "=" * 88)
    print("Across-seed summary")
    print("=" * 88)
    display_cols = [
        "method",
        "acceptance_mean",
        "min_ess_per_sec_mean",
        "mcse_heldout_nll_mean",
        "max_rhat_mean",
        "heldout_nll_mean",
        "relative_w_error_mean",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("\nSaved outputs:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
