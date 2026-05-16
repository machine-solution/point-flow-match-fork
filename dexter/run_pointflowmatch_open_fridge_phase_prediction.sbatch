#!/bin/bash
#
# Slurm: train FMPolicy with learned phase prediction (single shared model).
#
# Model: pfp.policy.fm_policy.FMPolicy
#   phase_conditioning=enabled  -> phase embedding in UNet input
#   phase_prediction=enabled    -> phase_head + CE aux loss; flow uses current GT phase repeated (train)
#
# Usage (from repo root):
#   sbatch dexter/run_pointflowmatch_open_fridge_phase_prediction.sbatch
#
#SBATCH --job-name=pfm_open_fridge_ph_pred
#SBATCH --output=logs/pfm_open_fridge_ph_pred_%j.out
#SBATCH --error=logs/pfm_open_fridge_ph_pred_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -e

CKPT_BACKUP_DIR="${HOME}/checkpoints/pointflowmatch_phase_prediction"

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

echo "=== Verify Hydra config resolves to FMPolicy + phase_prediction ==="
python dexter/verify_training_setup.py \
    --overrides \
    task_name=open_fridge \
    +experiment=pointflowmatch \
    phase_conditioning=enabled \
    phase_prediction=enabled

echo "=== Starting training (learned phase prediction) ==="
python scripts/train.py \
    task_name=open_fridge \
    +experiment=pointflowmatch \
    phase_conditioning=enabled \
    phase_prediction=enabled \
    dataloader.num_workers=8 \
    dataloader.batch_size=128 \
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
