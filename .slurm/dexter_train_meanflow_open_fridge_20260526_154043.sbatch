#!/usr/bin/env bash
#SBATCH --job-name=mf_train_open_fridge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.out
#SBATCH --error=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.err
set -euo pipefail
mkdir -p "/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs"
source ~/.bashrc
conda activate "pfp_env"
if [ -n "/home/machine-solution/Documents/Manipulators/diffusion_policy" ]; then
  export PYTHONPATH="/home/machine-solution/Documents/Manipulators/diffusion_policy:/home/machine-solution/Documents/Manipulators/PointFlowMatch:${PYTHONPATH:-}"
fi
cd "/home/machine-solution/Documents/Manipulators/PointFlowMatch"
CMD=(python scripts/train.py task_name="open_fridge" +experiment="pointflowmatch_meanflow" launch_eval_after_train=false)
if [ -n "" ]; then
  CMD+=(run_name="")
fi
if [ -n "" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_OVERRIDES})
  CMD+=("${EXTRA_ARR[@]}")
fi
echo "Running: ${CMD[*]}"
"${CMD[@]}"
