#!/bin/bash
#
# Slurm: train Momentum / Self-Correcting MeanFlow on open_fridge.
# Policy: MomentumMeanFlowPolicy (+experiment=pointflowmatch_momentum_meanflow)
#   - first-step + corrective second-step loss (prev_u conditioning)
#   - multi-step inference, default K=10, schedule=uniform
#
# Usage (from repo root):
#   TASK_NAME=open_fridge sbatch dexter/run_pointflowmatch_open_fridge_momentum_meanflow.sbatch
#   TASK_NAME=open_box sbatch dexter/run_pointflowmatch_open_fridge_momentum_meanflow.sbatch
#
# Logs:    logs/pfm_open_fridge_momentum_mf_<JOB>.out / .err
# Checkpoints: ckpt/momentum_meanflow_open_fridge_<JOB>/
# Backup:  ~/checkpoints/pointflowmatch_momentum_meanflow/
#
#SBATCH --job-name=pfm_open_fridge_momentum_mf
#SBATCH --output=logs/pfm_open_fridge_momentum_mf_%j.out
#SBATCH --error=logs/pfm_open_fridge_momentum_mf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -e

CKPT_BACKUP_DIR="${HOME}/checkpoints/pointflowmatch_momentum_meanflow"

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

echo "=== Starting Momentum MeanFlow training (milestone checkpoints) ==="
python scripts/train.py \
    task_name="${TASK_NAME}" \
    +experiment=pointflowmatch_momentum_meanflow \
    checkpoint_schedule=milestones_1500 \
    phase_conditioning=disabled \
    phase_prediction=disabled \
    dataloader.num_workers=8 \
    dataloader.batch_size=128 \
    launch_eval_after_train=false \
    log_wandb=False \
    run_name="momentum_meanflow_${TASK_NAME}_${SLURM_JOB_ID}"

echo "=== Training finished ==="

if [ -d "ckpt" ] && [ "$(ls -A ckpt 2>/dev/null)" ]; then
    echo "=== Copying checkpoints to ${CKPT_BACKUP_DIR} ==="
    mkdir -p "${CKPT_BACKUP_DIR}"
    cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
    echo "Checkpoints backed up to ${CKPT_BACKUP_DIR}"
else
    echo "WARNING: ckpt/ directory is empty or missing, nothing to back up."
fi
