#!/usr/bin/env bash
set -euo pipefail

# Dexter submission wrapper for MeanFlow training.
# Usage:
#   bash bash/dexter_train_meanflow.sh
#   bash bash/dexter_train_meanflow.sh open_fridge
#   TASK_NAME=open_box RUN_NAME=my_meanflow_run bash bash/dexter_train_meanflow.sh
#   RUN_LOCAL=1 TASK_NAME=open_fridge bash bash/dexter_train_meanflow.sh

TASK_NAME="${1:-${TASK_NAME:-open_fridge}}"
if [[ $# -gt 0 ]]; then shift; fi
EXPERIMENT="${EXPERIMENT:-pointflowmatch_meanflow}"
RUN_NAME="${RUN_NAME:-meanflow_${TASK_NAME}}"
NUM_GPUS="${NUM_GPUS:-1}"
PARTITION="${PARTITION:-gpu}"
TIME="${TIME:-24:00:00}"
MEMORY="${MEMORY:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
CONDA_ENV="${CONDA_ENV:-pfp_env}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DP_ROOT="$(cd "${REPO_ROOT}/../diffusion_policy" 2>/dev/null && pwd || true)"
SLURM_DIR="${REPO_ROOT}/.slurm"
mkdir -p "${SLURM_DIR}"

SBATCH_FILE="${SLURM_DIR}/dexter_train_meanflow_${TASK_NAME}_$(date +%Y%m%d_%H%M%S).sbatch"

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=mf_train_${TASK_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --output=${REPO_ROOT}/logs/%x_%j.out
#SBATCH --error=${REPO_ROOT}/logs/%x_%j.err
set -euo pipefail
mkdir -p "${REPO_ROOT}/logs"
source ~/.bashrc
conda activate "${CONDA_ENV}"
if [ -n "${DP_ROOT}" ]; then
  export PYTHONPATH="${DP_ROOT}:${REPO_ROOT}:\${PYTHONPATH:-}"
fi
cd "${REPO_ROOT}"
CMD=(python scripts/train.py task_name="${TASK_NAME}" +experiment="${EXPERIMENT}" launch_eval_after_train=false)
if [ -n "${RUN_NAME}" ]; then
  CMD+=(run_name="${RUN_NAME}")
fi
if [ -n "${EXTRA_OVERRIDES}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(\${EXTRA_OVERRIDES})
  CMD+=("\${EXTRA_ARR[@]}")
fi
echo "Running: \${CMD[*]}"
"\${CMD[@]}"
EOF

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1; generated ${SBATCH_FILE}"
  exit 0
fi

if [[ "${RUN_LOCAL:-0}" == "1" ]]; then
  export TASK_NAME
  exec bash "${REPO_ROOT}/dexter/run_local.sh" "${REPO_ROOT}/dexter/run_pointflowmatch_open_fridge_meanflow.sbatch"
fi

sbatch "${SBATCH_FILE}"
echo "Submitted ${SBATCH_FILE}"
