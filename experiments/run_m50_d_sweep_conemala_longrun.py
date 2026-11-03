#!/usr/bin/env python3
"""Long-run ConeMALA convergence diagnostic for the m=50 block-dimension sweep.

This script is deliberately *not* a new experiment. It reuses, for each d,
exactly the dataset, target, fixed ConeMALA step size, initial states, chain
seeds, and fixed-kernel burn-in from the completed d-sweep, and increases only
the number of retained production transitions.

Typical use
-----------
From the repository root, with this script placed next to
``run_m50_d_sweep_wishart_with_fisher.py``::

    caffeinate -i python3 -u experiments/d_sweep/run_m50_d_sweep_conemala_longrun.py \
        --input-root outputs/m50_d_sweep_wishart_with_fisher \
        --output-root outputs/m50_d_sweep_conemala_longrun_24k \
        --d-list 5,8 \
        --retained 24000

The script writes per-chain observable traces, prefix split-Rhat diagnostics,
a final summary, and a validation file comparing the first 12,000 retained
observations against the original d-sweep output when available.

Important
---------
Increasing n_train would change the posterior and therefore is *not* used to
resolve an MCMC convergence diagnostic. Only the retained MCMC budget is
increased here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


OBSERVABLES = ("rel_W_error", "test_nll", "logdet_Q", "Phi")
ALL_OBSERVABLES = ("rel_W_error", "test_nll", "logdet_Q", "lambda_min_Q", "Phi")
RHAT_THRESHOLD = 1.05


def _load_base_module(script_path: Path):
    """Load the completed d-sweep implementation from an explicit path."""
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find base d-sweep script: {script_path}\n"
            "Pass --base-script with the correct path."
        )
    spec = importlib.util.spec_from_file_location("conemala_d_sweep_base", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def _default_prefix_budgets(retained: int) -> List[int]:
    """Quarter-budget prefixes, always including the final budget."""
    raw = [retained // 4, retained // 2, 3 * retained // 4, retained]
    values = sorted({max(20, int(x)) for x in raw if x > 0})
    if values[-1] != retained:
        values.append(retained)
    return values


def _compute_observable_row(
    base,
    posterior,
    w: np.ndarray,
    phi: float,
    w_star: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    denominator: float,
) -> Dict[str, float]:
    q_matrix = posterior.q(w)
    logdet_q = base.logdet_spd(q_matrix)
    lambda_min_q = float(np.min(np.linalg.eigvalsh(q_matrix)))
    rel_w = math.sqrt(float(np.sum((w - w_star) ** 2))) / denominator
    n_test = y_test.shape[1]
    nll = (
        -0.5 * n_test * logdet_q
        + 0.5 * np.trace(q_matrix @ s_test)
    ) / n_test
    return {
        "rel_W_error": float(rel_w),
        "test_nll": float(nll),
        "logdet_Q": float(logdet_q),
        "lambda_min_Q": float(lambda_min_q),
        "Phi": float(phi),
    }


def run_streaming_conemala_chain(
    base,
    posterior,
    w0: np.ndarray,
    preburn: int,
    retained: int,
    h: float,
    seed: int,
    w_star: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run the exact same ConeMALA transition while storing only observables.

    This avoids storing every SPD matrix state, which becomes memory-intensive
    for d=8 or d=10 at long retained budgets. Observable calculations do not
    consume random numbers, so the Markov transition stream is unchanged.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray([base.project_spd(block) for block in w0])
    phi, grad = posterior.phi_and_grad(w)

    accepted_preburn = 0
    failures_preburn = 0
    t_total_start = time.perf_counter()

    for _ in range(preburn):
        w, phi, grad, accepted, failed = base._cone_transition(
            posterior, w, phi, grad, h, rng
        )
        accepted_preburn += int(accepted)
        failures_preburn += int(failed)

    records = {name: np.empty(retained, dtype=float) for name in ALL_OBSERVABLES}
    accepted_sampling = 0
    failures_sampling = 0
    transition_runtime = 0.0

    s_test = y_test @ y_test.T
    denominator = math.sqrt(float(np.sum(w_star * w_star)))

    for t in range(retained):
        t0 = time.perf_counter()
        w, phi, grad, accepted, failed = base._cone_transition(
            posterior, w, phi, grad, h, rng
        )
        transition_runtime += time.perf_counter() - t0
        accepted_sampling += int(accepted)
        failures_sampling += int(failed)

        row = _compute_observable_row(
            base=base,
            posterior=posterior,
            w=w,
            phi=phi,
            w_star=w_star,
            y_test=y_test,
            s_test=s_test,
            denominator=denominator,
        )
        for name in ALL_OBSERVABLES:
            records[name][t] = row[name]

    total_runtime = time.perf_counter() - t_total_start
    frame = pd.DataFrame(records)
    metadata = {
        "acceptance": accepted_sampling / max(retained, 1),
        "preburn_acceptance": accepted_preburn / max(preburn, 1),
        "invalid_rate": 0.0,
        "numerical_failure_rate": failures_sampling / max(retained, 1),
        "preburn_numerical_failure_rate": failures_preburn / max(preburn, 1),
        "runtime_transition_sec": float(transition_runtime),
        "runtime_total_sec": float(total_runtime),
    }
    return frame, metadata


def _prefix_rhat_rows(
    base,
    d: int,
    chain_frames: Sequence[pd.DataFrame],
    budgets: Sequence[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for budget in budgets:
        if budget > min(len(x) for x in chain_frames):
            continue
        row: Dict[str, float] = {"d": d, "retained": int(budget)}
        values = []
        for name in OBSERVABLES:
            rhat = base.split_rhat(
                [frame[name].to_numpy()[:budget] for frame in chain_frames]
            )
            row[f"{name}_Rhat"] = float(rhat)
            values.append(float(rhat))
        row["max_Rhat"] = float(np.nanmax(values))
        row["passes_1p05"] = bool(row["max_Rhat"] <= RHAT_THRESHOLD)
        rows.append(row)
    return rows


def _validate_original_prefix(
    d: int,
    source_results: Path,
    chain_frames: Sequence[pd.DataFrame],
    original_retained: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for chain_index, new_frame in enumerate(chain_frames):
        old_path = source_results / f"ConeMALA_chain{chain_index}_observables.csv"
        if not old_path.exists():
            continue
        old_frame = pd.read_csv(old_path)
        n = min(original_retained, len(old_frame), len(new_frame))
        for name in ALL_OBSERVABLES:
            if name not in old_frame.columns:
                continue
            old = old_frame[name].to_numpy(dtype=float)[:n]
            new = new_frame[name].to_numpy(dtype=float)[:n]
            diff = np.abs(old - new)
            scale = np.maximum(1.0, np.abs(old))
            rows.append(
                {
                    "d": d,
                    "chain": chain_index,
                    "observable": name,
                    "n_compared": n,
                    "max_abs_diff": float(np.max(diff)) if n else np.nan,
                    "max_rel_diff": float(np.max(diff / scale)) if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_one_dimension(args, base, d: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_experiment = args.input_root / f"graph_posterior_m50_d{d}"
    source_results = source_experiment / "results"
    config_path = source_results / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing source config for d={d}: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    m = int(config["active_m"])
    if m != 50:
        raise ValueError(f"Expected m=50, found m={m} for d={d}")

    settings = config["settings"]
    h_cone = float(settings["h_cone"])
    original_retained = int(settings["retained"])
    preburn = int(settings["preburn"] if args.preburn is None else args.preburn)

    w_star = np.load(source_results / "W_star.npy")
    y_train = np.load(source_results / "Y_train.npy")
    y_test = np.load(source_results / "Y_test.npy")

    edges = base.make_cycle_graph(m)
    edge_count = len(edges)
    r_scale = float(config["r_scale"])
    r_matrix = r_scale * np.eye(m * d)
    alpha_config = config.get("alpha_logdet", None)
    alpha_logdet = 0.5 * (d + 1.0) if alpha_config is None else float(alpha_config)

    posterior = base.PSDGraphPosterior(
        edges=edges,
        m=m,
        d=d,
        y_train=y_train,
        r_matrix=r_matrix,
        lambda_tr=float(config["lambda_tr"]),
        alpha_logdet=alpha_logdet,
    )

    n_chains = int(config["n_chains"])
    chain_seeds = [10_000 + 100 * m + chain for chain in range(n_chains)]
    initial_states = [base.make_initial_w(edge_count, d, chain) for chain in range(n_chains)]

    outdir = args.output_root / f"graph_posterior_m50_d{d}" / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"ConeMALA long run: m={m}, d={d}, h={h_cone:.12g}, burn={preburn}, retained={args.retained}")
    print(f"Reusing dataset and target from: {source_results}")
    print("=" * 78, flush=True)

    chain_frames: List[pd.DataFrame] = []
    chain_metadata: List[Dict[str, float]] = []

    for chain_index in range(n_chains):
        print(f"  chain {chain_index + 1}/{n_chains} ...", flush=True)
        frame, metadata = run_streaming_conemala_chain(
            base=base,
            posterior=posterior,
            w0=initial_states[chain_index],
            preburn=preburn,
            retained=args.retained,
            h=h_cone,
            seed=chain_seeds[chain_index],
            w_star=w_star,
            y_test=y_test,
        )
        frame.to_csv(outdir / f"ConeMALA_chain{chain_index}_observables.csv", index=False)
        chain_frames.append(frame)
        chain_metadata.append({"chain": chain_index, **metadata})
        print(
            f"    acc={metadata['acceptance']:.3f}, "
            f"burn acc={metadata['preburn_acceptance']:.3f}, "
            f"transition runtime={metadata['runtime_transition_sec']:.1f}s",
            flush=True,
        )

    metadata_df = pd.DataFrame(chain_metadata)
    metadata_df.insert(0, "d", d)
    metadata_df.to_csv(outdir / "ConeMALA_longrun_chain_metadata.csv", index=False)

    budgets = args.prefix_budgets
    rhat_rows = _prefix_rhat_rows(base, d, chain_frames, budgets)
    rhat_df = pd.DataFrame(rhat_rows)
    rhat_df.to_csv(outdir / "ConeMALA_longrun_prefix_rhat.csv", index=False)

    validation_df = _validate_original_prefix(
        d=d,
        source_results=source_results,
        chain_frames=chain_frames,
        original_retained=original_retained,
    )
    validation_df.to_csv(outdir / "ConeMALA_longrun_prefix_validation.csv", index=False)

    final = rhat_df.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "d": d,
                "m": m,
                "D_free": int(edge_count * d * (d + 1) // 2),
                "p_precision": int(m * d),
                "preburn": preburn,
                "retained": int(args.retained),
                "h_cone": h_cone,
                "accept_mean": float(metadata_df["acceptance"].mean()),
                "accept_sd": float(metadata_df["acceptance"].std(ddof=1)),
                "runtime_transition_all_chains_sec": float(metadata_df["runtime_transition_sec"].sum()),
                "runtime_total_all_chains_sec": float(metadata_df["runtime_total_sec"].sum()),
                **{f"{name}_Rhat": float(final[f"{name}_Rhat"]) for name in OBSERVABLES},
                "max_Rhat": float(final["max_Rhat"]),
                "passes_1p05": bool(final["passes_1p05"]),
            }
        ]
    )
    summary.to_csv(outdir / "ConeMALA_longrun_summary.csv", index=False)

    run_info = {
        "purpose": "long-run convergence diagnostic only",
        "source_experiment": str(source_experiment),
        "m": m,
        "d": d,
        "same_dataset_as_source": True,
        "same_target_as_source": True,
        "same_h_cone_as_source": h_cone,
        "same_initial_states_as_source": True,
        "same_chain_seeds_as_source": True,
        "preburn": preburn,
        "original_retained": original_retained,
        "longrun_retained": int(args.retained),
        "prefix_budgets": [int(x) for x in budgets],
        "rhat_threshold": RHAT_THRESHOLD,
    }
    (outdir / "longrun_config.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )

    print("\nPrefix split-Rhat:")
    print(rhat_df.to_string(index=False))
    print(
        f"\nFinal d={d}: max split-Rhat={float(final['max_Rhat']):.4f} "
        f"({'PASS' if bool(final['passes_1p05']) else 'NOT YET <= 1.05'})",
        flush=True,
    )

    if not validation_df.empty:
        max_abs = float(validation_df["max_abs_diff"].max())
        max_rel = float(validation_df["max_rel_diff"].max())
        print(
            f"Original-prefix validation: max abs diff={max_abs:.3e}, "
            f"max relative diff={max_rel:.3e}",
            flush=True,
        )

    return summary, rhat_df, validation_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-script",
        type=Path,
        default=Path(__file__).with_name("run_m50_d_sweep_wishart_with_fisher.py"),
        help="Path to the completed d-sweep implementation.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/m50_d_sweep_wishart_with_fisher"),
        help="Root directory of the completed d-sweep outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/m50_d_sweep_conemala_longrun_24k"),
        help="Separate output directory for long-run diagnostics.",
    )
    parser.add_argument(
        "--d-list",
        type=str,
        default="5,8",
        help="Dimensions to rerun. Default: 5,8. d=10 already passes 1.05 in the 12k run.",
    )
    parser.add_argument(
        "--retained",
        type=int,
        default=24000,
        help="Long-run retained production transitions per chain. Default: 24000.",
    )
    parser.add_argument(
        "--preburn",
        type=int,
        default=None,
        help="Optional override. By default reuse the source fixed-kernel burn-in (4000).",
    )
    parser.add_argument(
        "--prefix-budgets",
        type=str,
        default=None,
        help=(
            "Comma-separated retained prefixes for Rhat. If omitted, uses "
            "quarter/half/three-quarter/full budgets; for 24k this is "
            "6000,12000,18000,24000."
        ),
    )
    args = parser.parse_args()

    if args.retained < 20:
        raise ValueError("--retained must be at least 20.")

    args.d_list = _parse_int_list(args.d_list)
    if args.prefix_budgets is None:
        args.prefix_budgets = _default_prefix_budgets(args.retained)
    else:
        args.prefix_budgets = sorted(set(_parse_int_list(args.prefix_budgets)))
        if args.prefix_budgets[-1] > args.retained:
            raise ValueError("A prefix budget exceeds --retained.")
        if args.retained not in args.prefix_budgets:
            args.prefix_budgets.append(args.retained)
            args.prefix_budgets = sorted(set(args.prefix_budgets))

    base = _load_base_module(args.base_script)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_rhat = []
    all_validation = []

    for d in args.d_list:
        summary, rhat_df, validation_df = run_one_dimension(args, base, d)
        summaries.append(summary)
        all_rhat.append(rhat_df)
        if not validation_df.empty:
            all_validation.append(validation_df)

    summary_all = pd.concat(summaries, ignore_index=True)
    rhat_all = pd.concat(all_rhat, ignore_index=True)
    summary_all.to_csv(args.output_root / "ConeMALA_longrun_summary_all_d.csv", index=False)
    rhat_all.to_csv(args.output_root / "ConeMALA_longrun_prefix_rhat_all_d.csv", index=False)
    if all_validation:
        pd.concat(all_validation, ignore_index=True).to_csv(
            args.output_root / "ConeMALA_longrun_prefix_validation_all_d.csv",
            index=False,
        )

    print("\n" + "=" * 78)
    print("Long-run summary")
    print("=" * 78)
    print(summary_all.to_string(index=False))
    print(f"\nSaved under: {args.output_root}")


if __name__ == "__main__":
    main()
