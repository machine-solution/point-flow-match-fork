#!/bin/bash
#
# Slurm: gripper-weighted FMPolicy + learned phase prediction.
#
# Usage (from repo root):
#   sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted_phase_prediction.sbatch
#
#SBATCH --job-name=pfm_open_fridge_gw_ph_pred
#SBATCH --output=logs/pfm_open_fridge_gw_ph_pred_%j.out
#SBATCH --error=logs/pfm_open_fridge_gw_ph_pred_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -e

CKPT_BACKUP_DIR="${HOME}/checkpoints/pointflowmatch_gripw_phase_prediction"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ./pfp-train-env
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:${PYTHONPATH:-}"

TRAIN_DIR="demos/sim/open_fridge/train"
VALID_DIR="demos/sim/open_fridge/valid"
if [ ! -d "${TRAIN_DIR}" ] || [ ! -d "${VALID_DIR}" ]; then
    bash dexter/download_dataset.sh
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python dexter/verify_training_setup.py \
    --overrides \
    task_name=open_fridge \
    +experiment=pointflowmatch_gripper_weighted \
    phase_conditioning=enabled \
    phase_prediction=enabled

python scripts/train.py \
    task_name=open_fridge \
    +experiment=pointflowmatch_gripper_weighted \
    checkpoint_schedule=milestones_1500 \
    phase_conditioning=enabled \
    phase_prediction=enabled \
    dataloader.num_workers=8 \
    dataloader.batch_size=128 \
    launch_eval_after_train=false \
    log_wandb=False

if [ -d "ckpt" ] && [ "$(ls -A ckpt 2>/dev/null)" ]; then
    mkdir -p "${CKPT_BACKUP_DIR}"
    cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
fi
