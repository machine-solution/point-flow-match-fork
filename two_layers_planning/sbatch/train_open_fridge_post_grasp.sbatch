#!/bin/bash
#SBATCH --job-name=pfp-post
#SBATCH --output=slurm-%j-pfp-open-fridge-post.out
#SBATCH --error=slurm-%j-pfp-open-fridge-post.err
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Пост-захват (stable3). Датасет (train): https://disk.yandex.ru/d/U4bBHq7LZuBg9A
# Распакуйте zarr так, чтобы был каталог data/meta.
#
# Запуск из корня репозитория:
#   sbatch two_layers_planning/sbatch/train_open_fridge_post_grasp.sbatch
#
# Переопределить путь к train (валидация отключена в conf: use_validation=false):
#   export PFP_TRAIN_POST=/path/to/train
#   sbatch two_layers_planning/sbatch/train_open_fridge_post_grasp.sbatch
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DP_ROOT="${DIFFUSION_POLICY_ROOT:-$REPO_ROOT/../diffusion_policy}"
export PYTHONPATH="${REPO_ROOT}:${DP_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

PFP_TRAIN_POST="${PFP_TRAIN_POST:-$REPO_ROOT/demos/sim/open_fridge/train_post_grasp_stable3}"

echo "[sbatch] REPO_ROOT=$REPO_ROOT"
echo "[sbatch] PFP_TRAIN_POST=$PFP_TRAIN_POST"

if [[ ! -d "$PFP_TRAIN_POST/data" ]]; then
  echo "ERROR: нет $PFP_TRAIN_POST/data (ожидается zarr). Задайте PFP_TRAIN_POST." >&2
  exit 1
fi

RUN_NAME="open_fridge_post_${SLURM_JOB_ID:-manual}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

exec .venv/bin/python scripts/train.py \
  --config-name=train_open_fridge_post_grasp \
  dataset_path_train="$PFP_TRAIN_POST" \
  run_name="$RUN_NAME" \
  dataloader.num_workers=8
