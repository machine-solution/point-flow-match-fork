#!/bin/bash
#
# Один Slurm-job: подряд pre-grasp, затем post-grasp (два полных обучения).
# Walltime суммарный — если кластер режет лимит, используйте вместо этого
# два отдельных sbatch + submit_open_fridge_post_after_pre.sh
#
#   sbatch dexter/run_open_fridge_two_phase_chain.sbatch
#
#SBATCH --job-name=pfm_of_chain
#SBATCH --output=logs/pfm_open_fridge_chain_%j.out
#SBATCH --error=logs/pfm_open_fridge_chain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=336:00:00

set -euo pipefail

CKPT_BACKUP_DIR="${CKPT_BACKUP_DIR:-${HOME}/checkpoints/pointflowmatch_open_fridge_two_phase}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ./pfp-train-env
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== Chain job ${SLURM_JOB_ID} on $(hostname) ==="
nvidia-smi || true

PFP_PRE="${PFP_TRAIN_PRE:-${SLURM_SUBMIT_DIR}/demos/sim/open_fridge/train_pre_grasp_stable3}"
PFP_POST="${PFP_TRAIN_POST:-${SLURM_SUBMIT_DIR}/demos/sim/open_fridge/train_post_grasp_stable3}"

if [[ ! -d "${PFP_PRE}/data" ]] || [[ ! -d "${PFP_POST}/data" ]]; then
  echo "=== Downloading datasets ==="
  bash dexter/download_open_fridge_two_phase.sh --stable3
fi
if [[ ! -d "${PFP_PRE}/data" ]] || [[ ! -d "${PFP_POST}/data" ]]; then
  echo "ERROR: missing stable3 pre/post zarr under demos/sim/open_fridge/" >&2
  exit 1
fi

df -h . || true

PRE_JOB="${SLURM_JOB_ID}"
RUN_PRE="open_fridge_pre_${PRE_JOB}_chain"
echo "=== [1/2] Pre-grasp run_name=${RUN_PRE} ==="
python scripts/train.py \
  --config-name=train_open_fridge_pre_grasp \
  dataset_path_train="${PFP_PRE}" \
  run_name="${RUN_PRE}" \
  dataloader.num_workers=8 \
  dataloader.batch_size=128 \
  log_wandb=False

if [[ -d "ckpt" ]] && [[ "$(ls -A ckpt 2>/dev/null)" ]]; then
  mkdir -p "${CKPT_BACKUP_DIR}"
  cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
fi

RUN_POST="open_fridge_post_${PRE_JOB}_chain"
echo "=== [2/2] Post-grasp run_name=${RUN_POST} ==="
python scripts/train.py \
  --config-name=train_open_fridge_post_grasp \
  dataset_path_train="${PFP_POST}" \
  run_name="${RUN_POST}" \
  dataloader.num_workers=8 \
  dataloader.batch_size=128 \
  log_wandb=False

echo "=== Chain finished ==="
if [[ -d "ckpt" ]] && [[ "$(ls -A ckpt 2>/dev/null)" ]]; then
  mkdir -p "${CKPT_BACKUP_DIR}"
  cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
fi
