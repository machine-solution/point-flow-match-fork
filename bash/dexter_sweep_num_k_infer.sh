#!/usr/bin/env bash
set -euo pipefail

# Dexter submission wrapper for baseline/meanflow num_k_infer sweep.
# Usage:
#   CKPT_NAME=<run_name> bash bash/dexter_sweep_num_k_infer.sh

CKPT_NAME="${CKPT_NAME:-}"
CKPT_EPISODE="${CKPT_EPISODE:-ep1500}"
NUM_EPISODES="${NUM_EPISODES:-100}"
SEED="${SEED:-5678}"
MAX_EPISODE_LENGTH="${MAX_EPISODE_LENGTH:-120}"
KS="${KS:-1,2,4,6,8,10}"
PHASE_CONDITIONING="${PHASE_CONDITIONING:-disabled}"
PHASE_PREDICTION="${PHASE_PREDICTION:-disabled}"
PARTITION="${PARTITION:-gpu}"
NUM_GPUS="${NUM_GPUS:-1}"
TIME="${TIME:-24:00:00}"
MEMORY="${MEMORY:-64G}"
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
mkdir -p "${SLURM_DIR}" "${REPO_ROOT}/logs" "${REPO_ROOT}/results/efficiency"

OUT_CSV="results/efficiency/k_sweep_${CKPT_NAME}.csv"
OUT_JSON="results/efficiency/k_sweep_${CKPT_NAME}.json"
OUT_LOG="results/efficiency/k_sweep_${CKPT_NAME}.log"
SBATCH_FILE="${SLURM_DIR}/dexter_k_sweep_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S).sbatch"

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ksweep_${CKPT_NAME}
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
echo "Running K-sweep for ${CKPT_NAME} ks=${KS}"
xvfb-run -a python scripts/sweep_num_k_infer.py \
  --checkpoint "${CKPT_NAME}" \
  --ckpt-episode "${CKPT_EPISODE}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-episode-length "${MAX_EPISODE_LENGTH}" \
  --seed "${SEED}" \
  --ks "${KS}" \
  --phase-conditioning "${PHASE_CONDITIONING}" \
  --phase-prediction "${PHASE_PREDICTION}" \
  --output-csv "${OUT_CSV}" \
  --output-json "${OUT_JSON}" | tee "${OUT_LOG}"
EOF

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1; generated ${SBATCH_FILE}"
  exit 0
fi

sbatch "${SBATCH_FILE}"
echo "Submitted ${SBATCH_FILE}"
