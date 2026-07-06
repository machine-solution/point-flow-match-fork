#!/bin/bash
#
# Slurm job: train PointFlowMatch on the open_fridge task.
#
# Usage (from repo root):
#   TASK_NAME=open_fridge sbatch dexter/run_pointflowmatch_open_fridge.sbatch
#   TASK_NAME=open_box bash dexter/run_local.sh dexter/run_pointflowmatch_open_fridge.sbatch
#
# Prerequisites (run once on Dexter from the repo root):
#   1. Clone the repo (and sibling diffusion_policy repo next to it).
#   2. Create env:
#        conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env
#        conda activate ./pfp-train-env
#        pip install -e ../diffusion_policy
#   3. Place demos under demos/sim/<task_name>/{train,valid}
#      OR copy demos_<task_name>_sim.tar.gz to repo root (auto-extract in sbatch).
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

# shellcheck source=dexter/_run_env.sh
source dexter/_run_env.sh

TASK_NAME="${TASK_NAME:-open_fridge}"
# shellcheck source=dexter/_sbatch_task_preamble.sh
source dexter/_sbatch_task_preamble.sh

# ── Training ─────────────────────────────────────────────────────────────────
echo "=== Starting training (baseline, milestone checkpoints) ==="
# Checkpoints at epochs 300,600,900,1200,1500 only (see conf/checkpoint_schedule/milestones_1500.yaml).
PFP_PHASE_CONDITIONING="${PFP_PHASE_CONDITIONING:-disabled}"
python scripts/train.py \
    task_name="${TASK_NAME}" \
    +experiment=pointflowmatch \
    checkpoint_schedule=milestones_1500 \
    phase_conditioning="${PFP_PHASE_CONDITIONING}" \
    phase_prediction=disabled \
    dataloader.num_workers=8 \
    dataloader.batch_size=128 \
    launch_eval_after_train=false \
    log_wandb=False
    # resume_from_ckpt_name=open_fridge_2302_500 \
    # resume_from_ckpt_episode=ep500 \
    # run_name=open_fridge_2302_500_resume

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
