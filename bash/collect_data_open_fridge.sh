#!/usr/bin/env bash
# Collect train + valid demos for open_fridge using CoppeliaSim 4.1.0 from the repo.
# Requires: conda env pfp_env, and for headless run: sudo apt install xvfb && xvfb-run -a bash bash/collect_data_open_fridge.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COPPELIA_410="${REPO_ROOT}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"

if [[ ! -d "$COPPELIA_410" ]]; then
  echo "CoppeliaSim 4.1.0 not found at: $COPPELIA_410"
  echo "Extract CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz in the repo root."
  exit 1
fi

export COPPELIASIM_ROOT="$COPPELIA_410"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
# On Wayland, CoppeliaSim 4.1 often needs X11 (xcb) for a working OpenGL context
[[ -z "$QT_QPA_PLATFORM" ]] && export QT_QPA_PLATFORM=xcb

cd "$REPO_ROOT"
source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate pfp_env

echo "Collecting train demos (open_fridge)..."
python scripts/collect_demos.py --config-name=collect_demos_train \
  save_data=True env_config.vis=False env_config.task_name=open_fridge env_config.headless=True

echo "Collecting valid demos (open_fridge)..."
python scripts/collect_demos.py --config-name=collect_demos_valid \
  save_data=True env_config.vis=False env_config.task_name=open_fridge env_config.headless=True

echo "Done. Data in demos/sim/open_fridge/train and demos/sim/open_fridge/valid"
