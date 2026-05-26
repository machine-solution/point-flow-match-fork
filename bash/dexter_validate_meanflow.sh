#!/usr/bin/env bash
set -euo pipefail

# Dexter submission wrapper for checkpoint validation.
# Usage:
#   CKPT_NAME=<run_name> bash bash/dexter_validate_meanflow.sh

CKPT_NAME="${CKPT_NAME:-}"
NUM_EPISODES="${NUM_EPISODES:-100}"
PARTITION="${PARTITION:-gpu}"
NUM_GPUS="${NUM_GPUS:-1}"
TIME="${TIME:-12:00:00}"
MEMORY="${MEMORY:-48G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-6}"
CONDA_ENV="${CONDA_ENV:-pfp_env}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "${CKPT_NAME}" ]; then
  echo "ERROR: set CKPT_NAME=<checkpoint_run_name>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_DIR="${REPO_ROOT}/.slurm"
mkdir -p "${SLURM_DIR}" "${REPO_ROOT}/logs"

SBATCH_FILE="${SLURM_DIR}/dexter_validate_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S).sbatch"

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=mf_val_${CKPT_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --output=${REPO_ROOT}/logs/%x_%j.out
#SBATCH --error=${REPO_ROOT}/logs/%x_%j.err
set -euo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
echo "Running validation for ${CKPT_NAME}, episodes=${NUM_EPISODES}"
bash bash/run_validate_accuracy.sh "${CKPT_NAME}" "${NUM_EPISODES}"
EOF

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1; generated ${SBATCH_FILE}"
  exit 0
fi

sbatch "${SBATCH_FILE}"
echo "Submitted ${SBATCH_FILE}"
