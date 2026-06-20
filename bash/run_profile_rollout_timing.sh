#!/usr/bin/env bash
# Short rollout timing diagnostic (one fresh CoppeliaSim per method).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CS41="$REPO_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
if [ -d "$CS41" ]; then
  export COPPELIASIM_ROOT="$CS41"
  export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
  export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
fi
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export PYTHONUNBUFFERED=1

CONDA_ENV="${CONDA_ENV:-pfp_env}"
PY="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(conda run -n "$CONDA_ENV" which python)"
fi

METHODS="${1:-baseline,meanflow_multistep,shortcut}"
NUM_EPISODES="${2:-5}"
MAX_LEN="${3:-120}"
K="${4:-10}"
OUT_DIR="$REPO_ROOT/results/profiling_runtime"
mkdir -p "$OUT_DIR"

echo "[profile] COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-unset}"
echo "[profile] methods=$METHODS episodes=$NUM_EPISODES max_len=$MAX_LEN K=$K"

xvfb-run -a "$PY" -u scripts/profile_rollout_timing.py \
  --methods "$METHODS" \
  --ks "$K" \
  --num-episodes "$NUM_EPISODES" \
  --max-episode-length "$MAX_LEN" \
  --seed 5678 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/profile_run.log"
