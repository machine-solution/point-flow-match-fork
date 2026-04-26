#!/bin/bash
#
# Slurm: обучение post-grasp (после первого захвата) для open_fridge.
#
# Из корня репозитория:
#   sbatch dexter/run_open_fridge_post_grasp.sbatch
#
# Запуск после pre (вручную или с зависимостью):
#   sbatch --dependency=afterok:<PRE_JOB_ID> dexter/run_open_fridge_post_grasp.sbatch
#
# Датасет (stable3): demos/sim/open_fridge/train_post_grasp_stable3. Если нет data/ — качает
# dexter/download_open_fridge_two_phase.sh --stable3 (оба архива).

#SBATCH --job-name=pfm_of_post
#SBATCH --output=logs/pfm_open_fridge_post_%j.out
#SBATCH --error=logs/pfm_open_fridge_post_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -euo pipefail

CKPT_BACKUP_DIR="${CKPT_BACKUP_DIR:-${HOME}/checkpoints/pointflowmatch_open_fridge_two_phase}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "=== Job info (post-grasp) ==="
echo "Working dir : $(pwd)"
echo "Job ID      : ${SLURM_JOB_ID}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ./pfp-train-env
echo "Python: $(which python) ($(python -V 2>&1))"

export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:${PYTHONPATH:-}"

echo "=== nvidia-smi ==="
nvidia-smi || true

PFP_TRAIN_POST="${PFP_TRAIN_POST:-${SLURM_SUBMIT_DIR}/demos/sim/open_fridge/train_post_grasp_stable3}"
POST_DATA="${PFP_TRAIN_POST}/data"

if [[ ! -d "${POST_DATA}" ]]; then
  echo "=== Downloading stable3 pre/post datasets (dexter/download_open_fridge_two_phase.sh --stable3) ==="
  bash dexter/download_open_fridge_two_phase.sh --stable3
else
  echo "=== train_post_grasp_stable3 already present, skipping download ==="
fi

if [[ ! -d "${POST_DATA}" ]]; then
  echo "ERROR: missing zarr at ${PFP_TRAIN_POST} (expected ${POST_DATA})" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
RUN_NAME="open_fridge_post_${SLURM_JOB_ID}"

echo "=== Disk space ==="
df -h . /tmp 2>/dev/null || df -h .

echo "=== Training post-grasp run_name=${RUN_NAME} ==="
python scripts/train.py \
  --config-name=train_open_fridge_post_grasp \
  dataset_path_train="${PFP_TRAIN_POST}" \
  run_name="${RUN_NAME}" \
  dataloader.num_workers=8 \
  dataloader.batch_size=128 \
  log_wandb=False

echo "=== Post-grasp training finished ==="

if [[ -d "ckpt" ]] && [[ "$(ls -A ckpt 2>/dev/null)" ]]; then
  echo "=== Copying ckpt/ to ${CKPT_BACKUP_DIR} ==="
  mkdir -p "${CKPT_BACKUP_DIR}"
  cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
fi
