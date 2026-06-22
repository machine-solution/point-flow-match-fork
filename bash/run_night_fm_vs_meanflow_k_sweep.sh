#!/bin/bash
#
# Queue FM baseline vs MeanFlow multistep K-sweep AFTER schedule sweep finishes.
#
# Usage (from repo root):
#   bash bash/run_night_fm_vs_meanflow_k_sweep.sh
#   nohup bash bash/run_night_fm_vs_meanflow_k_sweep.sh > results/efficiency/fm_vs_meanflow_k_sweep_queue.log 2>&1 &
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG="${REPO_ROOT}/results/efficiency/fm_vs_meanflow_k_sweep_queue.log"
mkdir -p "${REPO_ROOT}/results/efficiency"

PYTHON="${PYTHON:-${HOME}/miniconda3/envs/pfp_env/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi

CS41="${REPO_ROOT}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
if [ -d "$CS41" ]; then
  export COPPELIASIM_ROOT="$CS41"
  export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
  export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
fi
export PYTHONPATH="${REPO_ROOT}/../diffusion_policy:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

wait_for_schedule_sweep() {
  while pgrep -f "scripts/sweep_meanflow_schedules.py" >/dev/null 2>&1; do
    log "[night-k-sweep] Waiting for schedule sweep (sweep_meanflow_schedules.py)..."
    sleep 300
  done
  while pgrep -f "validate_accuracy.py.*meanflow_schedule" >/dev/null 2>&1; do
    log "[night-k-sweep] Waiting for schedule-sweep validate_accuracy child..."
    sleep 60
  done
  log "[night-k-sweep] Schedule sweep finished."
}

run_sweep() {
  local mode="$1"
  shift
  log "[night-k-sweep] Starting ${mode} FM vs MeanFlow K sweep..."
  xvfb-run -a "$PYTHON" -u scripts/sweep_fm_vs_meanflow_k.py "$@"
}

log "[night-k-sweep] Queue started. PID=$$"

wait_for_schedule_sweep

log "[night-k-sweep] Running smoke check (num_episodes=2, K=1,10)..."
if ! run_sweep smoke \
  --smoke \
  --methods baseline,meanflow_multistep \
  --baseline-ckpt-name 1779122560-baseline-many-ckpts \
  --baseline-ckpt-episode ep1500 \
  --meanflow-ckpt-name meanflow_open_fridge_1365 \
  --meanflow-ckpt-episode latest \
  --seed 5678 \
  --output-dir results/efficiency/smoke_fm_vs_meanflow_k; then
  log "[night-k-sweep] SMOKE FAILED — full sweep NOT started. See log above."
  exit 1
fi

log "[night-k-sweep] Smoke passed. Starting full K sweep..."
run_sweep full \
  --methods baseline,meanflow_multistep \
  --ks 1,2,5,8,10,15 \
  --num-episodes 100 \
  --max-episode-length 120 \
  --seed 5678 \
  --baseline-ckpt-name 1779122560-baseline-many-ckpts \
  --baseline-ckpt-episode ep1500 \
  --meanflow-ckpt-name meanflow_open_fridge_1365 \
  --meanflow-ckpt-episode latest \
  --output-dir results/efficiency \
  --resume-existing

log "[night-k-sweep] Full K sweep completed."
