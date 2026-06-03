#!/bin/bash
#
# Slurm: validate MeanFlow Transformer checkpoint.
#
# Usage (from repo root):
#   sbatch --export=ALL,CKPT_NAME=<run_name>,NUM_EPISODES=100 dexter/run_validate_meanflow_transformer.sbatch
#
# Required:
#   CKPT_NAME - checkpoint run directory under ckpt/
#
# Optional:
#   NUM_EPISODES - defaults to 100
#
#SBATCH --job-name=pfm_val_mftr
#SBATCH --output=logs/pfm_val_mftr_%j.out
#SBATCH --error=logs/pfm_val_mftr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e

if [ -z "${CKPT_NAME:-}" ]; then
    echo "ERROR: CKPT_NAME is required."
    echo "Usage: sbatch --export=ALL,CKPT_NAME=<run_name>,NUM_EPISODES=100 dexter/run_validate_meanflow_transformer.sbatch"
    exit 1
fi

NUM_EPISODES="${NUM_EPISODES:-100}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "=== Validation job info ==="
echo "Working dir : $(pwd)"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "CKPT_NAME   : ${CKPT_NAME}"
echo "Episodes    : ${NUM_EPISODES}"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ./pfp-train-env
echo "Python: $(which python) ($(python -V 2>&1))"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:${PYTHONPATH:-}"

bash bash/run_validate_accuracy.sh "${CKPT_NAME}" "${NUM_EPISODES}"
