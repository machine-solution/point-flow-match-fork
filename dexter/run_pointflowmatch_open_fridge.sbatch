#!/bin/bash
#
# Slurm job: train PointFlowMatch on the open_fridge task.
#
# Usage (from repo root):
#   sbatch dexter/run_pointflowmatch_open_fridge.sbatch
#
# Prerequisites (run once on Dexter from the repo root):
#   1. Clone the repo (and sibling diffusion_policy repo next to it).
#   2. Create env:
#        conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env
#        conda activate ./pfp-train-env
#        pip install -e ../diffusion_policy
#   3. Either place demos under demos/sim/open_fridge/{train,valid}
#      OR let this script download them automatically (dexter/download_dataset.sh).
#
# Checkpoints are saved to ckpt/<run_name>/ inside the repo root.
# After training, they are also copied to CKPT_BACKUP_DIR (see below).

#SBATCH --job-name=pfm_open_fridge
#SBATCH --output=logs/pfm_open_fridge_%j.out
#SBATCH --error=logs/pfm_open_fridge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00

set -e

# ── Where to back up checkpoints after training ───────────────────────────────
# Change this path to wherever you want checkpoints stored on Dexter
# (e.g. a shared storage directory or your home folder).
CKPT_BACKUP_DIR="${HOME}/checkpoints/pointflowmatch"

# ─────────────────────────────────────────────────────────────────────────────

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "=== Job info ==="
echo "Working dir : $(pwd)"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPUs        : ${SLURM_GPUS:-1}"

# Conda на Dexter: системный путь (как в инструкции кластера)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ./pfp-train-env
echo "Python: $(which python) ($(python -V 2>&1))"

# Чтобы OOM показывал точную строку, а не асинхронный след
export CUDA_LAUNCH_BLOCKING=1

# Что видит процесс по GPU (при MIG/доле — лимит может быть меньше полной карты)
echo "=== nvidia-smi (before training) ==="
nvidia-smi

# Ensure sibling repo is importable (editable install can fail to add it on some systems)
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:${PYTHONPATH:-}"

# ── Dataset ──────────────────────────────────────────────────────────────────
TRAIN_DIR="demos/sim/open_fridge/train"
VALID_DIR="demos/sim/open_fridge/valid"

if [ ! -d "${TRAIN_DIR}" ] || [ ! -d "${VALID_DIR}" ]; then
    echo "=== Downloading dataset from Yandex Disk ==="
    bash dexter/download_dataset.sh
else
    echo "=== Dataset already present, skipping download ==="
fi

# ── Training ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== Starting training ==="
# batch_size=128 (при OOM понизь до 64 в override)
python scripts/train.py \
    task_name=open_fridge \
    +experiment=pointflowmatch \
    dataloader.num_workers=8 \
    dataloader.batch_size=128 \
    log_wandb=False \
    run_name=open_fridge_2302_500

echo "=== Training finished ==="

# ── Backup checkpoints ────────────────────────────────────────────────────────
if [ -d "ckpt" ] && [ "$(ls -A ckpt 2>/dev/null)" ]; then
    echo "=== Copying checkpoints to ${CKPT_BACKUP_DIR} ==="
    mkdir -p "${CKPT_BACKUP_DIR}"
    cp -r ckpt/. "${CKPT_BACKUP_DIR}/"
    echo "Checkpoints backed up to ${CKPT_BACKUP_DIR}"
else
    echo "WARNING: ckpt/ directory is empty or missing, nothing to back up."
fi
