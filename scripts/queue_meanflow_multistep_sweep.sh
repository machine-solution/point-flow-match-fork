#!/usr/bin/env bash
# Wait for overlap analysis to finish, then run MeanFlow multistep K-sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/results/efficiency"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/queue_meanflow_multistep_sweep.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Queue started: $(date -Is) ==="
echo "Waiting for compare_method_overlaps.py to finish..."

while pgrep -f "scripts/compare_method_overlaps.py" >/dev/null 2>&1; do
  sleep 60
done

echo "Overlap finished (or not running). Starting MeanFlow multistep sweep: $(date -Is)"

/home/machine-solution/miniconda3/bin/conda run -n pfp_env python scripts/sweep_num_k_infer.py \
  --ckpt-name meanflow_open_fridge_1365 \
  --ckpt-episode latest \
  --num-episodes 100 \
  --max-episode-length 120 \
  --ks 1,2,4,6,8,10 \
  --meanflow-multistep \
  --output-csv results/efficiency/k_sweep_meanflow_open_fridge_1365_n100_k1_2_4_6_8_10_multistep.csv \
  --output-json results/efficiency/k_sweep_meanflow_open_fridge_1365_n100_k1_2_4_6_8_10_multistep.json \
  --resume-existing

echo "=== Queue finished: $(date -Is) ==="
