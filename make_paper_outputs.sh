#!/usr/bin/env bash
# Rebuild every table and figure of Section 6 and the supplement that comes
# from the summary files in results/.  Takes seconds; runs no sampler.
# Output: paper_outputs/ (LaTeX tables, PDF figures, and the numbers quoted
# in the text, in paper_outputs/*.txt).
set -euo pipefail
cd "$(dirname "$0")"
OUT=paper_outputs
R=results
mkdir -p "$OUT"

run() {  # run <needed file> <output .txt> <command...>
    local need=$1 txt=$2; shift 2
    if [[ -f "$need" ]]; then
        "$@" > "$OUT/$txt"
        echo "  $txt"
    else
        echo "  skipped $txt (missing $need)"
    fi
}

UB=$R/units_and_basis/scaling_summary.csv
GS=$R/graph_size/combined_summary_d5_m_sweep.csv
BD=$R/block_dimension/combined_summary_m50_d_sweep.csv
RD=$R/repeated_datasets/robustness_d5_m50_by_seed.csv

run "$UB" units_and_basis_numbers.txt python3 analysis/units_and_basis_tables.py "$UB" "$OUT"
run "$GS" graph_size_tables.txt       python3 analysis/graph_size_tables.py "$GS"
run "$BD" block_dimension_tables.txt  python3 analysis/block_dimension_tables.py "$BD"
run "$RD" repeated_dataset_tables.txt python3 analysis/repeated_dataset_tables.py "$RD"
if [[ -f "$GS" && -f "$BD" ]]; then
    python3 analysis/plot_mala_scaling.py --graph-summary "$GS" --dsweep-summary "$BD" \
        --outdir "$OUT" > "$OUT/mala_scaling_figure_data.txt"
    echo "  fig_mala_graph_size.pdf, fig_mala_block_dim.pdf"
fi
echo "Wrote $OUT/"
