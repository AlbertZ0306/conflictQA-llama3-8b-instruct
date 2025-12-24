#!/usr/bin/env bash
set -euo pipefail

# Run plot_c0_metrics.py and plot_c0_metrics_curve.py for multiple result JSONs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
RESULTS_DIR="$REPO_ROOT/results"
PLOTS_DIR="$REPO_ROOT/plots"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "Results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  mapfile -t MODELS < <(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "No model result directories provided or found under $RESULTS_DIR" >&2
  exit 1
fi

C_FILES=("c0.json" "c3.json" "c4.json" "c5_cp.json" "c5_pc.json")

for model in "${MODELS[@]}"; do
  for cfile in "${C_FILES[@]}"; do
    input="$RESULTS_DIR/$model/$cfile"
    if [ ! -f "$input" ]; then
      echo "Skip missing $input" >&2
      continue
    fi
    variant="${cfile%%.*}"
    outdir="$PLOTS_DIR/$model/$variant"
    mkdir -p "$outdir"
    echo "Plotting $input -> $outdir"
    python3 "$SCRIPT_DIR/plot_c0_metrics.py" --input "$input" --output-dir "$outdir"
    python3 "$SCRIPT_DIR/plot_c0_metrics_curve.py" --input "$input" --output-dir "$outdir"
  done
done
