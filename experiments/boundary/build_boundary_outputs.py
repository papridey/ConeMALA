#!/usr/bin/env python3
r"""
Build combined paper tables and figures after the separate typical-set-aware
ConeMALA, Fisher MALA, and Euclidean MALA paper runs have finished.

The script does not rerun any sampler. It reads the per-seed CSV outputs,
checks that the experiments are comparable, pools the methods, and creates
paper-ready CSV, LaTeX, and PDF outputs.

Required result directories
---------------------------
Each result directory must contain its corrected config.json and the
corresponding per-seed and diagnostics CSV files:

ConeMALA:
    conemala_typical_by_seed.csv
    conemala_typical_diagnostics.csv
    conemala_typical_tuning.csv

Fisher:
    fisher_typical_by_seed.csv
    fisher_typical_diagnostics.csv
    fisher_typical_tuning.csv

Euclidean:
    euclidean_typical_by_seed.csv
    euclidean_typical_diagnostics.csv
    euclidean_typical_tuning.csv

Example
-------
python3 build_boundary_outputs.py \
  --cone-dir results_boundary_conemala_typical_paper_final \
  --fisher-dir results_boundary_fisher_typical_paper_final \
  --euclidean-dir results_boundary_euclidean_typical_paper_final \
  --output-dir results_boundary_paper_combined \
  --rhat-threshold 1.05
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import boundary_stress_comparison as boundary
import conemala_fisher_adaptive_comparison as base


METHOD_ORDER = (
    "ConeMALA",
    "FisherAdaptiveMALA",
    "EuclideanMALA",
    "RetractionRMALA",
)

RUN_SPECS = {
    "ConeMALA": {
        "by_seed": "conemala_typical_by_seed.csv",
        "diagnostics": "conemala_typical_diagnostics.csv",
        "tuning": "conemala_typical_tuning.csv",
    },
    "FisherAdaptiveMALA": {
        "by_seed": "fisher_typical_by_seed.csv",
        "diagnostics": "fisher_typical_diagnostics.csv",
        "tuning": "fisher_typical_tuning.csv",
    },
    "EuclideanMALA": {
        "by_seed": "euclidean_typical_by_seed.csv",
        "diagnostics": "euclidean_typical_diagnostics.csv",
        "tuning": "euclidean_typical_tuning.csv",
    },
    "RetractionRMALA": {
        "by_seed": "retraction_typical_by_seed.csv",
        "diagnostics": "retraction_typical_diagnostics.csv",
        "tuning": "retraction_typical_tuning.csv",
    },
}


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_section(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Return the user-facing run configuration.

    Typical-set runners store run fields at the top level. The older general
    boundary runner stored them under ``boundary_run_config``.
    """
    nested = payload.get("boundary_run_config")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(payload)


def source_profile_config(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Return the original profile configuration loaded from the source file.

    New runs store this under ``source_profile_config``. Legacy runs stored
    the same source values under ``resolved_experiment_config``.
    """
    source = payload.get("source_profile_config")
    if isinstance(source, dict):
        return dict(source)

    legacy = payload.get("resolved_experiment_config")
    if isinstance(legacy, dict):
        return dict(legacy)

    raise KeyError(
        "config.json contains neither source_profile_config nor "
        "resolved_experiment_config."
    )


def infer_effective_prior(
    payload: Dict[str, object],
) -> Dict[str, object]:
    """
    Infer the prior actually used by legacy boundary runs.

    The legacy samplers already called ``resolve_prior`` correctly; they only
    failed to print the effective values. For boundary-friendly mode the
    runner preserves the isotropic prior mean:

        effective_nu = d + boundary_nu_offset,
        effective_scale = source_nu * source_scale / effective_nu.
    """
    source = source_profile_config(payload)
    run = run_section(payload)

    prior_mode = str(run.get("prior_mode", "original"))
    boundary_nu_offset = float(
        run.get("boundary_nu_offset", 0.0)
    )

    d = int(source["d"])
    source_nu = float(source["nu"])
    source_prior_scale = float(source["prior_scale"])
    source_prior_mean_scale = source_nu * source_prior_scale

    if prior_mode == "original":
        effective_nu = source_nu
        effective_prior_scale = source_prior_scale
    elif prior_mode == "boundary-friendly":
        effective_nu = d + boundary_nu_offset
        if effective_nu <= d - 1:
            raise ValueError(
                "The inferred effective Wishart degrees of freedom must "
                "satisfy nu > d - 1."
            )
        effective_prior_scale = (
            source_prior_mean_scale / effective_nu
        )
    else:
        raise ValueError(
            f"Unsupported prior_mode in config.json: {prior_mode!r}"
        )

    effective_prior_mean_scale = (
        effective_nu * effective_prior_scale
    )

    return {
        "prior_mode": prior_mode,
        "boundary_nu_offset": boundary_nu_offset,
        "dimension_d": d,
        "source_nu": source_nu,
        "source_prior_scale": source_prior_scale,
        "source_prior_mean_scale": source_prior_mean_scale,
        "effective_nu": effective_nu,
        "effective_prior_scale": effective_prior_scale,
        "effective_prior_mean_scale": effective_prior_mean_scale,
        "determinant_exponent": float(
            0.5 * (effective_nu - d - 1.0)
        ),
        "prior_mean_preserved": bool(
            math.isclose(
                source_prior_mean_scale,
                effective_prior_mean_scale,
                rel_tol=1e-12,
                abs_tol=1e-14,
            )
        ),
        "wishart_parameterization": "E[W_e] = nu * Psi",
        "metadata_origin": "inferred_from_legacy_config",
    }


def effective_prior(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Return explicit effective-prior metadata when present, otherwise infer it
    from a legacy config.json.
    """
    prior = payload.get("effective_prior")
    if isinstance(prior, dict):
        result = dict(prior)
        result.setdefault("metadata_origin", "explicit")
        return result
    return infer_effective_prior(payload)


def effective_config(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Return the configuration actually used by the posterior.

    This is backward-compatible with legacy result directories whose
    ``resolved_experiment_config`` still displayed the source prior.
    """
    result = source_profile_config(payload)
    prior = effective_prior(payload)

    result["nu"] = prior["effective_nu"]
    result["prior_scale"] = prior["effective_prior_scale"]
    result["prior_mode"] = prior["prior_mode"]
    result["boundary_nu_offset"] = prior[
        "boundary_nu_offset"
    ]
    result["determinant_exponent"] = prior[
        "determinant_exponent"
    ]
    return result

def canonical_scalar(value: object) -> object:
    if isinstance(value, float):
        return round(value, 14)
    if isinstance(value, list):
        return tuple(canonical_scalar(item) for item in value)
    return value


def validate_configs(
    configs: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Check all scientific settings that must match across methods.
    Method-specific tuning settings are intentionally excluded.
    """
    keys = (
        "profile",
        "graph",
        "m",
        "d",
        "n_train",
        "n_test",
        "rho",
        "nu",
        "prior_scale",
        "true_weight_scale",
        "n_chains",
        "warmup",
        "retained",
        "thin",
        "experiment_seeds",
        "target_accept",
        "prior_mode",
        "boundary_nu_offset",
        "determinant_exponent",
    )

    methods = list(configs)
    reference_method = methods[0]
    reference = effective_config(configs[reference_method])

    errors: List[str] = []
    for method in methods[1:]:
        current = effective_config(configs[method])
        for key in keys:
            left = canonical_scalar(reference.get(key))
            right = canonical_scalar(current.get(key))
            if left != right:
                errors.append(
                    f"{key}: {reference_method}={left!r}, "
                    f"{method}={right!r}"
                )

    # Epsilon grids live at the run-config level, not the resolved profile.
    reference_eps = canonical_scalar(run_section(configs[reference_method]).get("epsilons"))
    for method in methods[1:]:
        current_eps = canonical_scalar(run_section(configs[method]).get("epsilons"))
        if reference_eps != current_eps:
            errors.append(
                f"epsilons: {reference_method}={reference_eps!r}, "
                f"{method}={current_eps!r}"
            )

    # Verify explicit effective-prior records.
    reference_prior = effective_prior(configs[reference_method])
    prior_keys = (
        "effective_nu",
        "effective_prior_scale",
        "determinant_exponent",
        "prior_mean_preserved",
    )
    for method in methods[1:]:
        current_prior = effective_prior(configs[method])
        for key in prior_keys:
            left = canonical_scalar(reference_prior.get(key))
            right = canonical_scalar(current_prior.get(key))
            if left != right:
                errors.append(
                    f"effective_prior.{key}: "
                    f"{reference_method}={left!r}, {method}={right!r}"
                )

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            "The result directories are not scientifically comparable:\n"
            f"  - {joined}"
        )

    return {
        "reference_method": reference_method,
        "effective_config": reference,
        "effective_prior": reference_prior,
        "epsilons": list(run_section(configs[reference_method])["epsilons"]),
    }


def load_method_result(
    method: str,
    directory: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if method not in RUN_SPECS:
        raise KeyError(method)
    spec = RUN_SPECS[method]

    config_path = directory / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = load_json(config_path)

    frames = []
    for key in ("by_seed", "diagnostics", "tuning"):
        path = directory / spec[key]
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "method" not in frame.columns:
            frame["method"] = method
        elif not set(frame["method"].dropna().unique()).issubset({method}):
            raise ValueError(
                f"{path} contains an unexpected method label: "
                f"{frame['method'].unique().tolist()}"
            )
        frames.append(frame)

    by_seed, diagnostics, tuning = frames
    return by_seed, diagnostics, tuning, config


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def method_rank(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def aggregate_with_convergence(
    by_seed: pd.DataFrame,
    rhat_threshold: float,
) -> pd.DataFrame:
    summary = boundary.aggregate_boundary_results(by_seed)

    extra_rows: List[Dict[str, float]] = []
    for (epsilon, method), group in by_seed.groupby(
        ["epsilon", "method"], sort=True
    ):
        converged = group["max_rhat"] <= rhat_threshold
        extra_rows.append(
            {
                "epsilon": float(epsilon),
                "method": method,
                "n_seed_runs": int(len(group)),
                "n_converged_seed_runs": int(converged.sum()),
                "converged_seed_fraction": float(converged.mean()),
                "all_seed_runs_converged": bool(converged.all()),
                "median_max_rhat": float(group["max_rhat"].median()),
                "max_max_rhat": float(group["max_rhat"].max()),
                "conditional_feasible_acceptance_mean": float(
                    np.mean(
                        np.divide(
                            group["acceptance"].to_numpy(float),
                            np.maximum(
                                1.0
                                - group[
                                    "final_invalid_spd_rate"
                                ].to_numpy(float),
                                1e-15,
                            ),
                        )
                    )
                ),
            }
        )

    extra = pd.DataFrame(extra_rows)
    return summary.merge(extra, on=["epsilon", "method"], how="left")


def mean_sd_text(
    mean: float,
    sd: float,
    digits: int,
    scientific: bool = False,
) -> str:
    if scientific:
        return rf"${mean:.{digits}e}\pm {sd:.{digits}e}$"
    return rf"${mean:.{digits}f}\pm {sd:.{digits}f}$"


def write_diagnostics_table(
    summary: pd.DataFrame,
    path: Path,
    rhat_threshold: float,
) -> None:
    table = summary.copy()
    table["_method_order"] = table["method"].map(method_rank)
    table = table.sort_values(
        ["epsilon", "_method_order"],
        ascending=[False, True],
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Typical-set-aware boundary-stress diagnostics across "
            r"independent data-generation seeds. The true edge condition "
            r"number is $\kappa^\star=1/\varepsilon$. Values are mean "
            r"$\pm$ standard deviation across seeds. Conv. reports the "
            rf"number of seed-level runs satisfying "
            rf"$\max\widehat R\leq {rhat_threshold:.2f}$."
        ),
        r"\label{tab:boundary_typical_diagnostics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cclccccccc}",
        r"\toprule",
        (
            r"$\varepsilon$ & $\kappa^\star$ & Method & Acceptance & "
            r"Invalid & Min. ESS/iter. & Max. $\widehat R$ & "
            r"ESS/sec$_{\rm samp}$ & Conv. & "
            r"$\Pr(\mathrm{accept}\mid\mathrm{SPD})$ \\"
        ),
        r"\midrule",
    ]

    previous_epsilon: Optional[float] = None
    for _, row in table.iterrows():
        epsilon = float(row["epsilon"])
        if previous_epsilon is not None and epsilon != previous_epsilon:
            lines.append(r"\addlinespace")
        previous_epsilon = epsilon

        convergence = (
            f"{int(row['n_converged_seed_runs'])}/"
            f"{int(row['n_seed_runs'])}"
        )
        sampling_efficiency = mean_sd_text(
            row["min_ess_per_sec_sampling_mean"],
            row["min_ess_per_sec_sampling_sd"],
            2,
        )
        if not bool(row["all_seed_runs_converged"]):
            sampling_efficiency += r"$^{\dagger}$"

        lines.append(
            " & ".join(
                [
                    f"${epsilon:.0e}$",
                    f"${1.0 / epsilon:.0f}$",
                    base.method_label(str(row["method"])),
                    mean_sd_text(
                        row["acceptance_mean"],
                        row["acceptance_sd"],
                        3,
                    ),
                    mean_sd_text(
                        row["final_invalid_spd_rate_mean"],
                        row["final_invalid_spd_rate_sd"],
                        3,
                    ),
                    mean_sd_text(
                        row["min_ess_per_iteration_mean"],
                        row["min_ess_per_iteration_sd"],
                        4,
                    ),
                    mean_sd_text(
                        row["max_rhat_mean"],
                        row["max_rhat_sd"],
                        3,
                    ),
                    sampling_efficiency,
                    convergence,
                    f"${row['conditional_feasible_acceptance_mean']:.3f}$",
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\parbox{\textwidth}{\scriptsize "
                r"$^{\dagger}$ESS/sec is shown for completeness but should "
                r"not be used for method ranking unless all seed-level runs "
                r"satisfy the stated convergence threshold.}"
            ),
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_accuracy_table(
    summary: pd.DataFrame,
    path: Path,
    rhat_threshold: float,
) -> None:
    table = summary.copy()
    table["_method_order"] = table["method"].map(method_rank)
    table = table.sort_values(
        ["epsilon", "_method_order"],
        ascending=[False, True],
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{Posterior accuracy and boundary-location summaries. "
            r"Values are mean $\pm$ standard deviation across independent "
            r"data-generation seeds. Entries are marked NC when at least one "
            rf"seed-level run has $\max\widehat R>{rhat_threshold:.2f}$."
        ),
        r"\label{tab:boundary_typical_accuracy}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cclccccc}",
        r"\toprule",
        (
            r"$\varepsilon$ & $\kappa^\star$ & Method & Held-out NLL & "
            r"Relative edge error & MCSE(NLL) & "
            r"Post. mean $\lambda_{\min}$ & "
            r"Sampled $\lambda_{\min,0.01}$ \\"
        ),
        r"\midrule",
    ]

    previous_epsilon: Optional[float] = None
    for _, row in table.iterrows():
        epsilon = float(row["epsilon"])
        if previous_epsilon is not None and epsilon != previous_epsilon:
            lines.append(r"\addlinespace")
        previous_epsilon = epsilon

        if bool(row["all_seed_runs_converged"]):
            values = [
                mean_sd_text(
                    row["heldout_nll_mean"],
                    row["heldout_nll_sd"],
                    4,
                ),
                mean_sd_text(
                    row["relative_w_error_mean"],
                    row["relative_w_error_sd"],
                    4,
                ),
                mean_sd_text(
                    row["mcse_heldout_nll_mean"],
                    row["mcse_heldout_nll_sd"],
                    4,
                ),
                mean_sd_text(
                    row["posterior_mean_min_eigenvalue_min_mean"],
                    row["posterior_mean_min_eigenvalue_min_sd"],
                    4,
                ),
                mean_sd_text(
                    row["sampled_min_eigenvalue_q01_mean"],
                    row["sampled_min_eigenvalue_q01_sd"],
                    4,
                ),
            ]
        else:
            values = [r"\multicolumn{1}{c}{NC}"] * 5

        lines.append(
            " & ".join(
                [
                    f"${epsilon:.0e}$",
                    f"${1.0 / epsilon:.0f}$",
                    base.method_label(str(row["method"])),
                    *values,
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


def write_prior_table(
    validation: Dict[str, object],
    path: Path,
) -> None:
    config = validation["effective_config"]
    prior = validation["effective_prior"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Configuration of the determinant-neutral boundary experiment.}",
        r"\label{tab:boundary_configuration}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        rf"Graph & {config['graph']} \\",
        rf"Number of vertices $m$ & {int(config['m'])} \\",
        rf"Block dimension $d$ & {int(config['d'])} \\",
        rf"Training/test sizes & {int(config['n_train'])}/{int(config['n_test'])} \\",
        rf"Chains per data seed & {int(config['n_chains'])} \\",
        rf"Production warmup & {int(config['warmup'])} \\",
        rf"Retained draws per chain & {int(config['retained'])} \\",
        rf"Effective $\nu$ & {float(prior['effective_nu']):.6g} \\",
        (
            rf"Effective isotropic prior scale & "
            rf"{float(prior['effective_prior_scale']):.6g} \\"
        ),
        (
            rf"Determinant exponent $(\nu-d-1)/2$ & "
            rf"{float(prior['determinant_exponent']):.6g} \\"
        ),
        (
            rf"Prior mean preserved & "
            rf"{str(bool(prior['prior_mean_preserved']))} \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    sd_metric: Optional[str],
    ylabel: str,
    filename: Path,
    log_y: bool = False,
    horizontal_line: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    methods = sorted(
        summary["method"].unique(),
        key=method_rank,
    )

    for method in methods:
        subset = summary[summary["method"] == method].sort_values(
            "truth_condition_target"
        )
        x = subset["truth_condition_target"].to_numpy(float)
        y = subset[metric].to_numpy(float)

        if sd_metric is not None and sd_metric in subset.columns:
            yerr = subset[sd_metric].to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=base.method_label(method),
            )
        else:
            ax.plot(
                x,
                y,
                marker="o",
                label=base.method_label(method),
            )

    ax.set_xscale("log")
    if log_y:
        positive = summary[metric].to_numpy(float)
        if np.any(positive > 0.0):
            ax.set_yscale("log")
    if horizontal_line is not None:
        ax.axhline(horizontal_line, linestyle="--")
    ax.set_xlabel(r"True edge condition number $1/\varepsilon$")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_fraction(
    summary: pd.DataFrame,
    filename: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    methods = sorted(
        summary["method"].unique(),
        key=method_rank,
    )

    for method in methods:
        subset = summary[summary["method"] == method].sort_values(
            "truth_condition_target"
        )
        ax.plot(
            subset["truth_condition_target"],
            subset["converged_seed_fraction"],
            marker="o",
            label=base.method_label(method),
        )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"True edge condition number $1/\varepsilon$")
    ax.set_ylabel("Fraction of converged data-seed runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_vs_truth_min_eigenvalue(
    summary: pd.DataFrame,
    filename: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    methods = sorted(
        summary["method"].unique(),
        key=method_rank,
    )

    for method in methods:
        subset = summary[summary["method"] == method].sort_values(
            "truth_condition_target"
        )
        ax.errorbar(
            subset["truth_condition_target"],
            subset["posterior_mean_min_eigenvalue_min_mean"],
            yerr=subset["posterior_mean_min_eigenvalue_min_sd"],
            marker="o",
            capsize=3,
            label=base.method_label(method),
        )

    truth = (
        summary.groupby("truth_condition_target", as_index=False)[
            "truth_min_eigenvalue_mean"
        ]
        .mean()
        .sort_values("truth_condition_target")
    )
    ax.plot(
        truth["truth_condition_target"],
        truth["truth_min_eigenvalue_mean"],
        marker="x",
        linestyle="--",
        label="Truth",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"True edge condition number $1/\varepsilon$")
    ax.set_ylabel("Minimum eigenvalue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create combined boundary paper tables and plots."
    )
    parser.add_argument("--cone-dir", type=Path, required=True)
    parser.add_argument("--fisher-dir", type=Path, required=True)
    parser.add_argument("--euclidean-dir", type=Path, required=True)
    parser.add_argument("--retraction-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rhat-threshold", type=float, default=1.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rhat_threshold <= 1.0:
        raise ValueError("rhat-threshold must exceed 1.")

    directories = {
        "ConeMALA": args.cone_dir,
        "FisherAdaptiveMALA": args.fisher_dir,
        "EuclideanMALA": args.euclidean_dir,
    }
    if args.retraction_dir is not None:
        directories["RetractionRMALA"] = args.retraction_dir

    by_seed_frames: List[pd.DataFrame] = []
    diagnostic_frames: List[pd.DataFrame] = []
    tuning_frames: List[pd.DataFrame] = []
    configs: Dict[str, Dict[str, object]] = {}

    for method, directory in directories.items():
        by_seed, diagnostics, tuning, config = load_method_result(
            method,
            directory,
        )
        by_seed_frames.append(by_seed)
        diagnostic_frames.append(diagnostics)
        tuning_frames.append(tuning)
        configs[method] = config

    validation = validate_configs(configs)

    inferred_methods = [
        method
        for method, payload in configs.items()
        if effective_prior(payload).get("metadata_origin")
        == "inferred_from_legacy_config"
    ]
    if inferred_methods:
        print(
            "Note: effective prior metadata was inferred from legacy "
            "config.json files for: "
            + ", ".join(inferred_methods)
        )
        print(
            "The numerical sampler outputs are unchanged; only the "
            "paper manifest uses the reconstructed effective prior."
        )

    by_seed = pd.concat(by_seed_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)

    require_columns(
        by_seed,
        (
            "epsilon",
            "truth_condition_target",
            "experiment_seed",
            "method",
            "acceptance",
            "h_final_mean",
            "min_ess_per_iteration",
            "min_ess_per_sec_sampling",
            "min_ess_per_sec_total",
            "mcse_heldout_nll",
            "max_rhat",
            "heldout_nll",
            "relative_w_error",
            "final_invalid_spd_rate",
            "posterior_mean_min_eigenvalue_min",
            "sampled_min_eigenvalue_q01",
            "truth_min_eigenvalue_min",
        ),
        "combined by-seed data",
    )

    # Verify one row per method/epsilon/data seed.
    duplicates = by_seed.duplicated(
        ["method", "epsilon", "experiment_seed"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            "Duplicate method/epsilon/seed rows:\n"
            + by_seed.loc[
                duplicates,
                ["method", "epsilon", "experiment_seed"],
            ].to_string(index=False)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = aggregate_with_convergence(
        by_seed=by_seed,
        rhat_threshold=args.rhat_threshold,
    )
    summary["_method_order"] = summary["method"].map(method_rank)
    summary = summary.sort_values(
        ["epsilon", "_method_order"],
        ascending=[False, True],
    ).drop(columns="_method_order")

    # Machine-readable pooled outputs.
    by_seed.to_csv(
        args.output_dir / "boundary_paper_by_seed.csv",
        index=False,
    )
    diagnostics.to_csv(
        args.output_dir / "boundary_paper_diagnostics.csv",
        index=False,
    )
    tuning.to_csv(
        args.output_dir / "boundary_paper_tuning.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "boundary_paper_summary.csv",
        index=False,
    )

    manifest = {
        "rhat_threshold": args.rhat_threshold,
        "input_directories": {
            method: str(directory)
            for method, directory in directories.items()
        },
        "validated_effective_config": validation["effective_config"],
        "validated_effective_prior": validation["effective_prior"],
        "epsilons": validation["epsilons"],
    }
    (args.output_dir / "paper_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    # LaTeX tables.
    write_diagnostics_table(
        summary,
        args.output_dir / "table_boundary_diagnostics.tex",
        rhat_threshold=args.rhat_threshold,
    )
    write_accuracy_table(
        summary,
        args.output_dir / "table_boundary_accuracy.tex",
        rhat_threshold=args.rhat_threshold,
    )
    write_prior_table(
        validation,
        args.output_dir / "table_boundary_configuration.tex",
    )

    # Also retain the original broad table format from the experiment core.
    boundary.write_boundary_latex_table(
        summary,
        args.output_dir / "table_boundary_full.tex",
    )

    # Comparative figures. Each PDF is a separate publication-ready figure.
    plot_metric(
        summary,
        metric="final_invalid_spd_rate_mean",
        sd_metric="final_invalid_spd_rate_sd",
        ylabel="Invalid-SPD proposal rate",
        filename=args.output_dir / "fig_boundary_invalid_rate.pdf",
    )
    plot_metric(
        summary,
        metric="max_rhat_mean",
        sd_metric="max_rhat_sd",
        ylabel=r"Maximum split-$\widehat R$",
        filename=args.output_dir / "fig_boundary_rhat.pdf",
        horizontal_line=args.rhat_threshold,
    )
    plot_metric(
        summary,
        metric="min_ess_per_iteration_mean",
        sd_metric="min_ess_per_iteration_sd",
        ylabel="Minimum ESS per retained iteration",
        filename=args.output_dir / "fig_boundary_ess_per_iteration.pdf",
        log_y=True,
    )
    plot_metric(
        summary,
        metric="acceptance_mean",
        sd_metric="acceptance_sd",
        ylabel="Acceptance rate",
        filename=args.output_dir / "fig_boundary_acceptance.pdf",
    )
    plot_metric(
        summary,
        metric="h_final_mean_mean",
        sd_metric="h_final_mean_sd",
        ylabel="Selected step size",
        filename=args.output_dir / "fig_boundary_step_size.pdf",
        log_y=True,
    )
    plot_metric(
        summary,
        metric="relative_w_error_mean",
        sd_metric="relative_w_error_sd",
        ylabel="Relative posterior-mean edge error",
        filename=args.output_dir / "fig_boundary_relative_edge_error.pdf",
        log_y=True,
    )
    plot_metric(
        summary,
        metric="heldout_nll_mean",
        sd_metric="heldout_nll_sd",
        ylabel="Held-out negative log likelihood",
        filename=args.output_dir / "fig_boundary_heldout_nll.pdf",
    )
    plot_convergence_fraction(
        summary,
        args.output_dir / "fig_boundary_convergence_fraction.pdf",
    )
    plot_posterior_vs_truth_min_eigenvalue(
        summary,
        args.output_dir / "fig_boundary_min_eigenvalue.pdf",
    )

    print("Combined boundary paper outputs")
    print("=" * 88)
    print(
        summary[
            [
                "epsilon",
                "method",
                "acceptance_mean",
                "final_invalid_spd_rate_mean",
                "min_ess_per_iteration_mean",
                "max_rhat_mean",
                "n_converged_seed_runs",
                "n_seed_runs",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
