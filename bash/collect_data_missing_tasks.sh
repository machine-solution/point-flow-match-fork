#!/usr/bin/env bash
# Collect train+valid demos for paper tasks missing under demos/sim/ (skip existing).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COPPELIA_410="${REPO_ROOT}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
LOG="${REPO_ROOT}/recordings/collect_missing_tasks.log"

TASKS=(
  unplug_charger
  close_door
  open_box
  take_frame_off_hanger
  open_oven
  put_books_on_bookshelf
  take_shoes_out_of_box
)

if [[ ! -d "$COPPELIA_410" ]]; then
  echo "CoppeliaSim 4.1.0 not found at: $COPPELIA_410" | tee -a "$LOG"
  exit 1
fi

export COPPELIASIM_ROOT="$COPPELIA_410"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
[[ -z "${QT_QPA_PLATFORM:-}" ]] && export QT_QPA_PLATFORM=xcb

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== collect_data_missing_tasks started $(date -Is) ==="

source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate pfp_env

cd "$REPO_ROOT"
PY="${HOME}/miniconda3/envs/pfp_env/bin/python"

collect_split() {
  local task="$1"
  local split_cfg="$2"
  local out_dir="${REPO_ROOT}/demos/sim/${task}/$(
    [[ "$split_cfg" == collect_demos_train ]] && echo train || echo valid
  )"

  if [[ -d "$out_dir" ]]; then
    echo "[skip] $task $(basename "$out_dir") already exists: $out_dir"
    return 0
  fi

  echo "[run] $task $split_cfg -> $out_dir"
  xvfb-run -a "$PY" scripts/collect_demos.py --config-name="$split_cfg" \
    save_data=True env_config.vis=False "env_config.task_name=${task}" env_config.headless=True
}

for task in "${TASKS[@]}"; do
  echo "--- task: $task $(date -Is) ---"
  collect_split "$task" collect_demos_train
  collect_split "$task" collect_demos_valid
done

echo "=== collect_data_missing_tasks finished $(date -Is) ==="
