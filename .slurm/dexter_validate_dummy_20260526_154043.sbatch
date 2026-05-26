#!/usr/bin/env bash
#SBATCH --job-name=mf_val_dummy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.out
#SBATCH --error=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.err
set -euo pipefail
source ~/.bashrc
conda activate "pfp_env"
cd "/home/machine-solution/Documents/Manipulators/PointFlowMatch"
echo "Running validation for dummy, episodes=100"
bash bash/run_validate_accuracy.sh "dummy" "100"
