#!/bin/bash
#
# Slurm: обучение pre-grasp (до первого захвата) для open_fridge.
#
# Из корня репозитория:
#   sbatch dexter/run_open_fridge_pre_grasp.sbatch
#
# Датасет (stable3): demos/sim/open_fridge/train_pre_grasp_stable3 (zarr: data/ + meta/).
# Если нет — скрипт вызовет dexter/download_open_fridge_two_phase.sh --stable3 (качает pre и post).
#
# Затем post-модель: dexter/run_open_fridge_post_grasp.sbatch
#   или цепочка: sbatch dexter/run_open_fridge_two_phase_chain.sbatch
#   или: bash dexter/submit_open_fridge_post_after_pre.sh <JOBID_PRE>

#SBATCH --job-name=pfm_of_pre
#SBATCH --output=logs/pfm_open_fridge_pre_%j.out
#SBATCH --error=logs/pfm_open_fridge_pre_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -euo pipefail

CKPT_BACKUP_DIR="${CKPT_BACKUP_DIR:-${HOME}/checkpoints/pointflowmatch_open_fridge_two_phase}"

# shellcheck source=dexter/_run_env.sh
source dexter/_run_env.sh

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ./pfp-train-env
echo "Python: $(which python) ($(python -V 2>&1))"

export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="${REPO_ROOT}/../diffusion_policy:${PYTHONPATH:-}"

echo "=== nvidia-smi ==="
nvidia-smi || true

PFP_TRAIN_PRE="${PFP_TRAIN_PRE:-${REPO_ROOT}/demos/sim/open_fridge/train_pre_grasp_stable3}"
PRE_DATA="${PFP_TRAIN_PRE}/data"

if [[ ! -d "${PRE_DATA}" ]]; then
  echo "=== Downloading stable3 pre/post datasets (dexter/download_open_fridge_two_phase.sh --stable3) ==="
  bash dexter/download_open_fridge_two_phase.sh --stable3
else
  echo "=== train_pre_grasp_stable3 already present, skipping download ==="
fi

if [[ ! -d "${PRE_DATA}" ]]; then
  echo "ERROR: missing zarr at ${PFP_TRAIN_PRE} (expected ${PRE_DATA})" >&2
  exit 1
fi

RUN_NAME="open_fridge_pre_${PFP_JOB_ID}"

echo "=== Disk space ==="
df -h . /tmp 2>/dev/null || df -h .

echo "=== Training pre-grasp run_name=${RUN_NAME} ==="
python scripts/train.py \
  --config-name=train_open_fridge_pre_grasp \
  dataset_path_train="${PFP_TRAIN_PRE}" \
  run_name="${RUN_NAME}" \
  dataloader.num_workers=8 \
  dataloader.batch_size=128 \
  log_wandb=False

echo "=== Pre-grasp training finished ==="

if [[ -d "ckpt" ]] && [[ "$(ls -A ckpt 2>/dev/null)" ]]; then
  echo "=== Copying ckpt/ to ${CKPT_BACKUP_DIR} ==="
  mkdir -p "${CKPT_BACKUP_DIR}"
  cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
fi
