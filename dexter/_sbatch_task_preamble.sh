# Shared Slurm preamble: TASK_NAME (default open_fridge) + dataset + PYTHONPATH.
# Usage in .sbatch after `cd "${SLURM_SUBMIT_DIR}"`:
#   TASK_NAME="${TASK_NAME:-open_fridge}"
#   source dexter/_sbatch_task_preamble.sh

TASK_NAME="${TASK_NAME:-open_fridge}"
echo "=== task_name (TASK_NAME): ${TASK_NAME} ==="
bash dexter/ensure_task_dataset.sh "${TASK_NAME}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../diffusion_policy:${PYTHONPATH:-}"
