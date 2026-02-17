#!/bin/bash
#
# Slurm script to train the PointFlowMatch baseline on the `open_fridge` task.
# Assumes:
#   - This file is submitted from the repository root (where `scripts/train.py` lives)
#   - A conda environment has been created, e.g.:
#       conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env
#   - Demonstrations are available under:
#       demos/sim/open_fridge/train
#       demos/sim/open_fridge/valid
#
# Usage:
#   sbatch dexter/run_pointflowmatch_open_fridge.sbatch

#SBATCH --job-name=pfm_open_fridge
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=pfm_open_fridge_%j.out
#SBATCH --error=pfm_open_fridge_%j.err

set -e

# Go to the directory from which the job was submitted (should be repo root)
cd "${SLURM_SUBMIT_DIR}"

echo "Working directory: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "GPUs requested: ${SLURM_GPUS:-1}"

# Activate conda environment (adjust path to conda.sh if needed on Dexter)
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh. Please adjust the path in dexter/run_pointflowmatch_open_fridge.sbatch."
    exit 1
fi

conda activate ./pfp-train-env

echo "Python executable: $(which python)"
python -V

# Match CPU threads to Slurm allocation
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Starting training: open_fridge + pointflowmatch"

python scripts/train.py \
    task_name=open_fridge \
    +experiment=pointflowmatch \
    log_wandb=False

echo "Training finished."

