#!/bin/bash
#
# TMP sbatch: дообучение post-grasp из конкретного чекпоинта Composer.
#
# Запуск (из корня репозитория PointFlowMatch):
#   export CKPT_SRC=/abs/path/to/ep1100-ba72600-rank0.pt
#   export PFP_TRAIN_POST=/abs/path/to/demos/sim/open_fridge/train_post_grasp_stable3
#   sbatch dexter/tmp_finetune_open_fridge_post_from_ep1100.sbatch
#
# По умолчанию пытается использовать conda-окружение ./pfp-train-env (как на Dexter).
# Если его нет, попробует .venv/bin/python.
#
# Результат:
# - веса будут взяты из ckpt/open_fridge_post_953/ep1100*.pt
# - новый run_name по умолчанию: open_fridge_post_953_finetune_<JOBID>
# - чекпоинты пишутся в ckpt/<run_name>/
#
#SBATCH --job-name=pfm_of_post_ft
#SBATCH --output=logs/pfm_open_fridge_post_ft_%j.out
#SBATCH --error=logs/pfm_open_fridge_post_ft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#
set -euo pipefail

# shellcheck source=dexter/_run_env.sh
source dexter/_run_env.sh

# Pick python
PYTHON_BIN=""
if [[ -f "./pfp-train-env/bin/python" ]]; then
  # Dexter-style local conda env
  source /opt/miniconda3/etc/profile.d/conda.sh
  conda activate ./pfp-train-env
  PYTHON_BIN="$(which python)"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="$(pwd)/.venv/bin/python"
else
  PYTHON_BIN="$(which python3 || which python)"
fi
echo "Python: ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"

export CUDA_LAUNCH_BLOCKING=1

# Ensure sibling diffusion_policy is importable (common on clusters)
export PYTHONPATH="${REPO_ROOT}/../diffusion_policy:${PYTHONPATH:-}"

# Inputs
CKPT_SRC="${CKPT_SRC:?Set CKPT_SRC=/abs/path/to/ep1100-....pt}"
PFP_TRAIN_POST="${PFP_TRAIN_POST:-${REPO_ROOT}/demos/sim/open_fridge/train_post_grasp_stable3}"

if [[ ! -f "${CKPT_SRC}" ]]; then
  echo "ERROR: CKPT_SRC not found: ${CKPT_SRC}" >&2
  exit 1
fi
if [[ ! -d "${PFP_TRAIN_POST}/data" ]]; then
  echo "ERROR: missing zarr at ${PFP_TRAIN_POST} (expected ${PFP_TRAIN_POST}/data)" >&2
  exit 1
fi

# Put ckpt into expected location for resume_from_ckpt_name/episode globbing
RESUME_NAME="open_fridge_post_953"
RESUME_EP="ep1100"
mkdir -p "ckpt/${RESUME_NAME}"
cp -f "${CKPT_SRC}" "ckpt/${RESUME_NAME}/"
echo "Copied ckpt -> ckpt/${RESUME_NAME}/$(basename "${CKPT_SRC}")"

RUN_NAME="${RUN_NAME:-open_fridge_post_953_finetune_${PFP_JOB_ID}}"

echo "=== Disk space ==="
df -h . /tmp 2>/dev/null || df -h .

echo "=== Starting finetune: run_name=${RUN_NAME} ==="
exec "${PYTHON_BIN}" scripts/train.py \
  --config-name=train_open_fridge_post_grasp \
  dataset_path_train="${PFP_TRAIN_POST}" \
  run_name="${RUN_NAME}" \
  resume_from_ckpt_name="${RESUME_NAME}" \
  resume_from_ckpt_episode="${RESUME_EP}" \
  dataloader.num_workers=8 \
  dataloader.batch_size=128 \
  log_wandb=False

