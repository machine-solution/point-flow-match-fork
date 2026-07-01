#!/usr/bin/env bash
# Create demos_<task>_sim.tar.gz at repo root (train+valid under demos/sim/<task>/).
# Removes per-split train.tar.gz / valid.tar.gz after a successful bundle.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${REPO_ROOT}/recordings/create_demos_sim_bundles.log"

TASKS=(
  unplug_charger
  close_door
  open_box
  open_fridge
  take_frame_off_hanger
  open_oven
  put_books_on_bookshelf
  take_shoes_out_of_box
)

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== create_demos_sim_bundles started $(date -Is) force=${FORCE} ==="

bundle_task() {
  local task="$1"
  local task_dir="${REPO_ROOT}/demos/sim/${task}"
  local archive="${REPO_ROOT}/demos_${task}_sim.tar.gz"

  if [[ ! -d "${task_dir}/train/data" || ! -d "${task_dir}/train/meta" || ! -d "${task_dir}/valid/data" || ! -d "${task_dir}/valid/meta" ]]; then
    echo "[error] missing zarr train/valid in ${task_dir} (need data/ + meta/)" >&2
    return 1
  fi

  if [[ $FORCE -eq 0 && -f "$archive" ]]; then
    newest_src=$(find "${task_dir}/train" "${task_dir}/valid" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -n "$newest_src" && "$archive" -nt "$newest_src" ]]; then
      echo "[skip] up to date $archive"
      rm -f "${task_dir}/train.tar.gz" "${task_dir}/valid.tar.gz"
      return 0
    fi
  fi

  echo "[tar] $archive"
  rm -f "$archive"
  tar -czf "$archive" -C "$REPO_ROOT" "demos/sim/${task}/train" "demos/sim/${task}/valid"
  ls -lh "$archive"

  echo "[verify] $archive"
  if ! tar -tf "$archive" "demos/sim/${task}/train/data" &>/dev/null; then
    echo "[error] bundle missing demos/sim/${task}/train/data" >&2
    return 1
  fi
  if ! tar -tf "$archive" "demos/sim/${task}/valid/data" &>/dev/null; then
    echo "[error] bundle missing demos/sim/${task}/valid/data" >&2
    return 1
  fi

  echo "[cleanup] remove per-split archives in ${task_dir}"
  rm -f "${task_dir}/train.tar.gz" "${task_dir}/valid.tar.gz"
}

cd "$REPO_ROOT"
FAILED=0
for task in "${TASKS[@]}"; do
  echo "--- $task $(date -Is) ---"
  if ! bundle_task "$task"; then
    echo "[fatal] bundle failed for ${task}" >&2
    FAILED=1
    break
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "=== create_demos_sim_bundles aborted $(date -Is) ===" >&2
  exit 1
fi

echo "=== create_demos_sim_bundles finished $(date -Is) ==="
ls -lh "${REPO_ROOT}"/demos_*_sim.tar.gz
