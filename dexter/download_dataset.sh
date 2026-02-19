#!/usr/bin/env bash
# Download and unpack the open_fridge demo dataset from Yandex Disk.
# The public share link resolves to a direct URL via the Yandex Disk API.
#
# Usage:
#   bash dexter/download_dataset.sh
#   bash dexter/download_dataset.sh --force   # re-download even if data exists

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PUBLIC_KEY="https://disk.yandex.ru/d/Ssr_BffZItISOg"
ARCHIVE="demos_open_fridge_sim.tar.gz"
DEST_DIR="demos/sim/open_fridge"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; fi

# Check if data already exists
if [[ $FORCE -eq 0 ]] && [[ -d "${DEST_DIR}/train" ]] && [[ -d "${DEST_DIR}/valid" ]]; then
    echo "Dataset already exists at ${DEST_DIR}. Use --force to re-download."
    exit 0
fi

# Resolve direct download link via Yandex Disk public API
echo "Resolving download URL..."
DOWNLOAD_URL=$(python3 - <<'EOF'
import urllib.request, json, urllib.parse, sys
key = urllib.parse.quote("https://disk.yandex.ru/d/Ssr_BffZItISOg")
api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={key}"
try:
    with urllib.request.urlopen(api_url, timeout=30) as r:
        d = json.loads(r.read())
        print(d["href"])
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [[ -z "${DOWNLOAD_URL}" ]]; then
    echo "ERROR: Could not resolve Yandex Disk download URL." >&2
    exit 1
fi

echo "Downloading ${ARCHIVE} (~4.3 GB)..."
python3 - "${DOWNLOAD_URL}" "${ARCHIVE}" <<'EOF'
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
EOF

echo "Extracting ${ARCHIVE}..."
# Archive contains the full path: demos/sim/open_fridge/{train,valid}/
# so we extract relative to the repo root.
tar -xzf "${ARCHIVE}"

# Verify
if [[ ! -d "${DEST_DIR}/train" ]] || [[ ! -d "${DEST_DIR}/valid" ]]; then
    echo "ERROR: Expected ${DEST_DIR}/train and ${DEST_DIR}/valid after extraction." >&2
    exit 1
fi

echo "Cleaning up archive..."
rm -f "${ARCHIVE}"

echo "Done. Dataset ready at ${DEST_DIR}/"
