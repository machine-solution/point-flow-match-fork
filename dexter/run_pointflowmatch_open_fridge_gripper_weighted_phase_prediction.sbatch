#!/bin/bash
#
# Slurm: gripper-weighted FMPolicy + learned phase prediction.
#
# Usage (from repo root):
#   TASK_NAME=open_fridge sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted_phase_prediction.sbatch
#   TASK_NAME=open_box sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted_phase_prediction.sbatch
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

# shellcheck source=dexter/_run_env.sh
source dexter/_run_env.sh
TASK_NAME="${TASK_NAME:-open_fridge}"
# shellcheck source=dexter/_sbatch_task_preamble.sh
source dexter/_sbatch_task_preamble.sh


python dexter/verify_training_setup.py \
    --overrides \
    task_name="${TASK_NAME}" \
    +experiment=pointflowmatch_gripper_weighted \
    phase_conditioning=enabled \
    phase_prediction=enabled

python scripts/train.py \
    task_name="${TASK_NAME}" \
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
