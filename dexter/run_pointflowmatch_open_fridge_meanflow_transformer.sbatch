#!/bin/bash
#
# Slurm: train MeanFlow + Temporal Transformer on open_fridge.
#
# Usage (from repo root):
#   sbatch dexter/run_pointflowmatch_open_fridge_meanflow_transformer.sbatch
#
#SBATCH --job-name=pfm_open_fridge_mftr
#SBATCH --output=logs/pfm_open_fridge_mftr_%j.out
#SBATCH --error=logs/pfm_open_fridge_mftr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -e

CKPT_BACKUP_DIR="${HOME}/checkpoints/pointflowmatch_meanflow_transformer"

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

export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:${PYTHONPATH:-}"

TRAIN_DIR="demos/sim/open_fridge/train"
VALID_DIR="demos/sim/open_fridge/valid"
if [ ! -d "${TRAIN_DIR}" ] || [ ! -d "${VALID_DIR}" ]; then
    echo "=== Downloading dataset from Yandex Disk ==="
    bash dexter/download_dataset.sh
else
    echo "=== Dataset already present, skipping download ==="
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "=== Disk space before training ==="
df -h . /tmp 2>/dev/null || df -h .

echo "=== Starting MeanFlow Transformer training (milestone checkpoints) ==="
python scripts/train.py \
    task_name=open_fridge \
    +experiment=pointflowmatch_meanflow_transformer \
    checkpoint_schedule=milestones_1500 \
    phase_conditioning=disabled \
    phase_prediction=disabled \
    dataloader.num_workers=8 \
    dataloader.batch_size=128 \
    launch_eval_after_train=false \
    log_wandb=False \
    run_name="meanflow_transformer_open_fridge_${SLURM_JOB_ID}"

echo "=== Training finished ==="

if [ -d "ckpt" ] && [ "$(ls -A ckpt 2>/dev/null)" ]; then
    echo "=== Copying checkpoints to ${CKPT_BACKUP_DIR} ==="
    mkdir -p "${CKPT_BACKUP_DIR}"
    cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
    echo "Checkpoints backed up to ${CKPT_BACKUP_DIR}"
else
    echo "WARNING: ckpt/ directory is empty or missing, nothing to back up."
fi
