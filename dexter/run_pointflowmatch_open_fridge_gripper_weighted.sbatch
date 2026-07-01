#!/bin/bash
#
# Slurm job: train PointFlowMatch (gripper-weighted loss) on the open_fridge task.
#
# Usage (from repo root):
#   TASK_NAME=open_fridge sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted.sbatch
#   TASK_NAME=open_box sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted.sbatch
#

#SBATCH --job-name=pfm_open_fridge_gripw
#SBATCH --output=logs/pfm_open_fridge_gripw_%j.out
#SBATCH --error=logs/pfm_open_fridge_gripw_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -e

CKPT_BACKUP_DIR="${HOME}/checkpoints/pointflowmatch_gripw"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "=== Job info ==="
echo "Working dir : $(pwd)"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPUs        : ${SLURM_GPUS:-1}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ./pfp-train-env
echo "Python: $(which python) ($(python -V 2>&1))"

export CUDA_LAUNCH_BLOCKING=1

echo "=== nvidia-smi (before training) ==="
nvidia-smi

TASK_NAME="${TASK_NAME:-open_fridge}"
# shellcheck source=dexter/_sbatch_task_preamble.sh
source dexter/_sbatch_task_preamble.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== Disk space before training ==="
df -h . /tmp 2>/dev/null || df -h .
echo "=== Starting training (gripper-weighted loss) ==="

# Phase-conditioned single model (optional):
#   export PFP_PHASE_CONDITIONING=enabled
# Default is disabled.
PFP_PHASE_CONDITIONING="${PFP_PHASE_CONDITIONING:-disabled}"

python scripts/train.py \
    task_name="${TASK_NAME}" \
    +experiment=pointflowmatch_gripper_weighted \
    checkpoint_schedule=milestones_1500 \
    phase_conditioning="${PFP_PHASE_CONDITIONING}" \
    phase_prediction=disabled \
    dataloader.num_workers=8 \
    dataloader.batch_size=128 \
    launch_eval_after_train=false \
    log_wandb=False

echo "=== Training finished ==="

if [ -d "ckpt" ] && [ "$(ls -A ckpt 2>/dev/null)" ]; then
    echo "=== Copying checkpoints to ${CKPT_BACKUP_DIR} ==="
    mkdir -p "${CKPT_BACKUP_DIR}"
    cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
    echo "Checkpoints backed up to ${CKPT_BACKUP_DIR}"
else
    echo "WARNING: ckpt/ directory is empty or missing, nothing to back up."
fi

