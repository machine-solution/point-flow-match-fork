#!/usr/bin/env bash
# Download and unpack paper-task demo bundles: demos/sim/<task>/{train,valid}.
#
# Usage (from repo root):
#   bash dexter/download_dataset.sh                  # all 8 paper tasks
#   bash dexter/download_dataset.sh open_fridge      # one task
#   bash dexter/download_dataset.sh close_door open_box
#   bash dexter/download_dataset.sh --force open_oven
#   bash dexter/download_dataset.sh --list
#
# Two-phase open_fridge pre/post: dexter/download_open_fridge_two_phase.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=dexter/paper_tasks.sh
source "${REPO_ROOT}/dexter/paper_tasks.sh"
# shellcheck source=dexter/yandex_dataset_urls.sh
source "${REPO_ROOT}/dexter/yandex_dataset_urls.sh"

FORCE=0
TASKS=()

usage() {
  cat <<EOF
Usage: bash dexter/download_dataset.sh [--force] [--list] [task_name ...]

  (no args)     download all paper tasks
  task_name     download one or more tasks (aliases: unplug, put_books, take_shoes, take_frame)
  --force       re-download even if data already exists
  --list        print task names and Yandex Disk URLs

Paper tasks: ${PFP_PAPER_TASKS[*]}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --list)
      for t in "${PFP_PAPER_TASKS[@]}"; do
        url="$(pfp_yandex_url_for_task "$t")"
        echo "${t}  ${url}"
      done
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TASKS+=("$(pfp_normalize_task_name "$1")")
      shift
      ;;
  esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  TASKS=("${PFP_PAPER_TASKS[@]}")
fi

resolve_yandex_download_url() {
  local public_key="$1"
  python3 - "$public_key" <<'EOF'
import json
import sys
import urllib.parse
import urllib.request

public_key = sys.argv[1]
key = urllib.parse.quote(public_key, safe="")
api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={key}"
try:
    with urllib.request.urlopen(api_url, timeout=60) as r:
        data = json.loads(r.read())
        print(data["href"])
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

download_file() {
  local url="$1"
  local dest="$2"
  python3 - "$url" "$dest" <<'EOF'
import sys
import urllib.request

url, dest = sys.argv[1], sys.argv[2]

def _progress(count, block_size, total):
    if total <= 0:
        return
    pct = min(count * block_size / total * 100, 100)
    downloaded_mb = count * block_size / 1024 / 1024
    total_mb = total / 1024 / 1024
    print(f"\r  {pct:.1f}%  {downloaded_mb:.0f}/{total_mb:.0f} MB", end="", flush=True)

urllib.request.urlretrieve(url, dest, _progress)
print()
EOF
}

task_dataset_ready() {
  local task="$1"
  local train="${REPO_ROOT}/demos/sim/${task}/train"
  local valid="${REPO_ROOT}/demos/sim/${task}/valid"
  [[ -d "${train}/data" && -d "${train}/meta" && -d "${valid}/data" && -d "${valid}/meta" ]]
}

download_one_task() {
  local task="$1"
  local dest_dir="${REPO_ROOT}/demos/sim/${task}"
  local archive="${REPO_ROOT}/demos_${task}_sim.tar.gz"
  local public_key

  if ! pfp_is_paper_task "$task"; then
    echo "ERROR: unknown task '${task}' (expected one of: ${PFP_PAPER_TASKS[*]})" >&2
    return 1
  fi

  if [[ $FORCE -eq 0 ]] && task_dataset_ready "$task"; then
    echo "[skip] ${task}: dataset already at demos/sim/${task}/{train,valid}"
    return 0
  fi

  if [[ $FORCE -eq 0 && -f "$archive" ]]; then
    echo "[extract] ${task}: using existing ${archive}"
    tar -xzf "$archive"
    if task_dataset_ready "$task"; then
      return 0
    fi
    echo "WARNING: ${archive} did not produce a valid zarr tree; re-downloading." >&2
  fi

  public_key="$(pfp_yandex_url_for_task "$task")" || {
    echo "ERROR: no Yandex URL configured for ${task}" >&2
    return 1
  }

  echo "=== ${task}: resolving Yandex Disk URL ==="
  local download_url
  download_url="$(resolve_yandex_download_url "$public_key")"
  if [[ -z "$download_url" ]]; then
    echo "ERROR: could not resolve download URL for ${task}" >&2
    return 1
  fi

  echo "=== ${task}: downloading ${archive} ==="
  rm -f "$archive"
  download_file "$download_url" "$archive"

  echo "=== ${task}: extracting ${archive} ==="
  tar -xzf "$archive"

  if ! task_dataset_ready "$task"; then
    echo "ERROR: expected zarr at demos/sim/${task}/{train,valid} after extract" >&2
    return 1
  fi

  echo "=== ${task}: cleaning up ${archive} ==="
  rm -f "$archive"
  echo "=== ${task}: done (demos/sim/${task}/) ==="
}

FAILED=0
for task in "${TASKS[@]}"; do
  if ! download_one_task "$task"; then
    echo "[fatal] download failed for ${task}" >&2
    FAILED=1
    break
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi

echo "All requested datasets ready under demos/sim/"
