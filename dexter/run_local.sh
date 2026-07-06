#!/usr/bin/env bash
# Run a dexter/*.sbatch training script directly on the server (no Slurm).
#
# Usage (from repo root):
#   bash dexter/run_local.sh dexter/run_pointflowmatch_open_fridge.sbatch
#   TASK_NAME=open_box bash dexter/run_local.sh dexter/run_pointflowmatch_open_fridge_meanflow.sbatch
#
# Server limits (override if admin policy changes):
#   PFP_MAX_GPUS=2 PFP_MAX_CPU_THREADS=16 PFP_MAX_RAM_GB=64
#
# Slurm path unchanged: TASK_NAME=open_fridge sbatch dexter/run_pointflowmatch_open_fridge.sbatch

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash dexter/run_local.sh dexter/run_<name>.sbatch" >&2
  exit 2
fi

SCRIPT="$1"
shift

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: script not found: ${SCRIPT}" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${REPO_ROOT}/logs/local_$(basename "${SCRIPT}" .sbatch)_${STAMP}.log"

export PFP_RUN_MODE=local

echo "Logging to ${LOG}"
echo "Tip: tmux new -s train  OR  nohup bash dexter/run_local.sh ${SCRIPT} &"

exec > >(tee -a "${LOG}") 2>&1

bash "${SCRIPT}" "$@"
