#!/usr/bin/env bash
# Ensure demos/sim/<task_name>/{train,valid} exist (extract local bundle or Yandex for open_fridge).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=dexter/paper_tasks.sh
source "${REPO_ROOT}/dexter/paper_tasks.sh"

TASK_NAME="${1:-${TASK_NAME:-${PFP_DEFAULT_TASK_NAME}}}"
pfp_validate_task_name "${TASK_NAME}"

TRAIN_DIR="${REPO_ROOT}/demos/sim/${TASK_NAME}/train"
VALID_DIR="${REPO_ROOT}/demos/sim/${TASK_NAME}/valid"

if [[ -d "${TRAIN_DIR}/data" && -d "${TRAIN_DIR}/meta" && -d "${VALID_DIR}/data" && -d "${VALID_DIR}/meta" ]]; then
  echo "=== Dataset present: demos/sim/${TASK_NAME}/{train,valid} ==="
  exit 0
fi

BUNDLE="${REPO_ROOT}/demos_${TASK_NAME}_sim.tar.gz"
if [[ -f "${BUNDLE}" ]]; then
  echo "=== Extracting ${BUNDLE} ==="
  tar -xzf "${BUNDLE}"
  if [[ ! -d "${TRAIN_DIR}/data" || ! -d "${VALID_DIR}/data" ]]; then
    echo "ERROR: ${BUNDLE} did not produce demos/sim/${TASK_NAME}/{train,valid}" >&2
    exit 1
  fi
  echo "=== Dataset ready after extract ==="
  exit 0
fi

if [[ "${TASK_NAME}" == "open_fridge" ]]; then
  echo "=== Downloading open_fridge from Yandex Disk ==="
  bash dexter/download_dataset.sh
  exit 0
fi

echo "ERROR: no dataset for task_name=${TASK_NAME}" >&2
echo "  Place zarr under demos/sim/${TASK_NAME}/{train,valid}" >&2
echo "  or copy demos_${TASK_NAME}_sim.tar.gz to the repo root and re-run." >&2
exit 1
