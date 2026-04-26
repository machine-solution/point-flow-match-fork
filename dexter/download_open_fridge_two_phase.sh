#!/usr/bin/env bash
# Скачивает train_pre_grasp и train_post_grasp с Яндекс.Диска (два .tar.gz) и распаковывает
# в demos/sim/open_fridge/train_{pre,post}_grasp (zarr: data/ + meta/).
#
# Тот же API, что и в download_dataset.sh: публичная ссылка → прямой href → urllib.
#
# Запуск из корня репозитория:
#   bash dexter/download_open_fridge_two_phase.sh
#   bash dexter/download_open_fridge_two_phase.sh --force
#   bash dexter/download_open_fridge_two_phase.sh --stable3
#   bash dexter/download_open_fridge_two_phase.sh --stable3 --force
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
EXTRACT="${REPO_ROOT}/dexter/extract_zarr_tarball.py"

FORCE=0
STABLE3=0
for arg in "$@"; do
  case "${arg}" in
    --force) FORCE=1 ;;
    --stable3) STABLE3=1 ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: bash dexter/download_open_fridge_two_phase.sh [--stable3] [--force]" >&2
      exit 2
      ;;
  esac
done

download_one() {
  local PUBLIC_KEY="$1"
  local ARCHIVE_NAME="$2"
  local DEST_REL="$3"

  local DEST="${REPO_ROOT}/${DEST_REL}"
  if [[ ${FORCE} -eq 0 ]] && [[ -d "${DEST}/data" ]] && [[ -d "${DEST}/meta" ]]; then
    echo "[skip] ${DEST_REL} уже есть (data/meta). --force чтобы перекачать."
    return 0
  fi

  echo "[yandex] ${ARCHIVE_NAME} — получаю ссылку..."
  local DOWNLOAD_URL
  export YANDEX_PUBLIC_KEY="${PUBLIC_KEY}"
  DOWNLOAD_URL=$("${PYTHON}" -c '
import json, os, urllib.parse, urllib.request
key = urllib.parse.quote(os.environ["YANDEX_PUBLIC_KEY"])
api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={key}"
with urllib.request.urlopen(api_url, timeout=120) as r:
    d = json.loads(r.read())
print(d["href"])
')
  unset YANDEX_PUBLIC_KEY

  echo "[download] ${ARCHIVE_NAME}"
  "${PYTHON}" - "${DOWNLOAD_URL}" "${REPO_ROOT}/${ARCHIVE_NAME}" <<'PY'
import sys, urllib.request
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
PY

  echo "[extract] -> ${DEST_REL}"
  "${PYTHON}" "${EXTRACT}" "${REPO_ROOT}/${ARCHIVE_NAME}" "${DEST}"
  rm -f "${REPO_ROOT}/${ARCHIVE_NAME}"
  echo "[ok] ${DEST}"
}

download_one "https://disk.yandex.ru/d/E81_4UbQiwAYpw" "train_pre_grasp.tar.gz" \
  "demos/sim/open_fridge/train_pre_grasp"
download_one "https://disk.yandex.ru/d/OmzMhzSy0lGTMw" "train_post_grasp.tar.gz" \
  "demos/sim/open_fridge/train_post_grasp"

if [[ ${STABLE3} -eq 1 ]]; then
  download_one "https://disk.yandex.ru/d/koQoDdaJ-t4b8A" "train_pre_grasp_stable3.tar.gz" \
    "demos/sim/open_fridge/train_pre_grasp_stable3"
  download_one "https://disk.yandex.ru/d/U4bBHq7LZuBg9A" "train_post_grasp_stable3.tar.gz" \
    "demos/sim/open_fridge/train_post_grasp_stable3"
  echo "Готово: stable3 train zarr в demos/sim/open_fridge/train_pre_grasp_stable3 и train_post_grasp_stable3"
else
  echo "Готово: двухфазные train zarr в demos/sim/open_fridge/train_pre_grasp и train_post_grasp"
fi
