#!/usr/bin/env bash
#SBATCH --job-name=ksweep_dummy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.out
#SBATCH --error=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.err
set -euo pipefail
source ~/.bashrc
conda activate "pfp_env"
cd "/home/machine-solution/Documents/Manipulators/PointFlowMatch"
echo "Running K-sweep for dummy ks=1,2,4,6,8,10"
xvfb-run -a python scripts/sweep_num_k_infer.py   --checkpoint "dummy"   --ckpt-episode "ep1500"   --num-episodes "100"   --max-episode-length "120"   --seed "5678"   --ks "1,2,4,6,8,10"   --phase-conditioning "disabled"   --phase-prediction "disabled"   --output-csv "results/efficiency/k_sweep_dummy.csv"   --output-json "results/efficiency/k_sweep_dummy.json" | tee "results/efficiency/k_sweep_dummy.log"
