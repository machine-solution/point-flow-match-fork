#!/usr/bin/env bash
set -euo pipefail

# Dexter submission wrapper for lightweight MeanFlow benchmarks.
# Usage:
#   CKPT_NAME=<run_name> bash bash/dexter_benchmark_meanflow.sh

CKPT_NAME="${CKPT_NAME:-}"
CKPT_EPISODE="${CKPT_EPISODE:-latest}"
NUM_K_INFER="${NUM_K_INFER:-1}"
PARTITION="${PARTITION:-gpu}"
NUM_GPUS="${NUM_GPUS:-1}"
TIME="${TIME:-04:00:00}"
MEMORY="${MEMORY:-24G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
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

SBATCH_FILE="${SLURM_DIR}/dexter_bench_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S).sbatch"

cat > "${SBATCH_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=mf_bench_${CKPT_NAME}
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
python scripts/count_policy_params.py \
  --ckpt-name "${CKPT_NAME}" \
  --ckpt-episode "${CKPT_EPISODE}" \
  --num-k-infer "${NUM_K_INFER}" \
  --output-csv results/efficiency/params.csv
python scripts/benchmark_inference_latency.py \
  --ckpt-name "${CKPT_NAME}" \
  --ckpt-episode "${CKPT_EPISODE}" \
  --num-k-infer "${NUM_K_INFER}" \
  --output-csv results/efficiency/latency.csv
EOF

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1; generated ${SBATCH_FILE}"
  exit 0
fi

sbatch "${SBATCH_FILE}"
echo "Submitted ${SBATCH_FILE}"
