#!/bin/bash
# Validate one run at Composer milestone epochs (300, 600, 900, 1200, 1500).
# Usage (repo root):
#   bash bash/run_validate_milestone_sweep.sh <ckpt_run_name> [num_episodes]
#
# Example:
#   bash bash/run_validate_milestone_sweep.sh 1778935982-pistachio-axolotl 100 \
#     phase_conditioning=enabled phase_prediction=enabled

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT_NAME="${1:?ckpt run name required}"
NUM_EP="${2:-100}"
SEED="${SEED:-228}"
shift 2 || true
EXTRA_HYDRA=("$@")

CONDA_ENV="${CONDA_ENV:-pfp_env}"
MILESTONES=(300 600 900 1200 1500)

if ! command -v conda &>/dev/null; then
    for _c in "$HOME/miniconda3" "/opt/miniconda3"; do
        [ -f "${_c}/etc/profile.d/conda.sh" ] && source "${_c}/etc/profile.d/conda.sh" && break
    done
fi

CS41="$REPO_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
if [ -d "$CS41" ]; then
    export COPPELIASIM_ROOT="$CS41"
    export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
    export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
fi

DIFFUSION_ABS="$(cd "$REPO_ROOT/../diffusion_policy" 2>/dev/null && pwd)"
[ -n "$DIFFUSION_ABS" ] && export PYTHONPATH="${DIFFUSION_ABS}:${PYTHONPATH:-}"

RESULTS_DIR="$REPO_ROOT/results/milestone_sweep_${CKPT_NAME}"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"
echo "milestone sweep: $CKPT_NAME  episodes=$NUM_EP  seed=$SEED" | tee "$SUMMARY"
echo "extra hydra: ${EXTRA_HYDRA[*]:-<none>}" | tee -a "$SUMMARY"
echo "---" | tee -a "$SUMMARY"

for EP in "${MILESTONES[@]}"; do
    OUT="$RESULTS_DIR/ep${EP}.txt"
    echo "=== ep${EP} ===" | tee -a "$SUMMARY"
    JSON_OUT="$RESULTS_DIR/ep${EP}.json"
    conda run --no-capture-output -n "$CONDA_ENV" python scripts/validate_accuracy.py \
        policy.ckpt_name="$CKPT_NAME" \
        policy.ckpt_episode="ep${EP}" \
        env_runner.num_episodes="$NUM_EP" \
        seed="$SEED" \
        results_json="$JSON_OUT" \
        "${EXTRA_HYDRA[@]}" 2>&1 | tee "$OUT"
    ACC=$(grep -E '^Accuracy:' "$OUT" | tail -1 || true)
    echo "ep${EP}: ${ACC:-FAILED}" | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"
done

echo "Summary: $SUMMARY"
