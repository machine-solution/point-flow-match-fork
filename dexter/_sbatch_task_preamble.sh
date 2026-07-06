# Shared preamble: TASK_NAME (default open_fridge) + dataset.
# Usage in .sbatch after `source dexter/_run_env.sh`:
#   TASK_NAME="${TASK_NAME:-open_fridge}"
#   source dexter/_sbatch_task_preamble.sh

TASK_NAME="${TASK_NAME:-open_fridge}"
echo "=== task_name (TASK_NAME): ${TASK_NAME} ==="
bash dexter/ensure_task_dataset.sh "${TASK_NAME}"
