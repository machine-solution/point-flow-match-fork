#!/usr/bin/env bash
#SBATCH --job-name=mf_bench_dummy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.out
#SBATCH --error=/home/machine-solution/Documents/Manipulators/PointFlowMatch/logs/%x_%j.err
set -euo pipefail
source ~/.bashrc
conda activate "pfp_env"
cd "/home/machine-solution/Documents/Manipulators/PointFlowMatch"
python scripts/count_policy_params.py   --ckpt-name "dummy"   --ckpt-episode "latest"   --num-k-infer "1"   --output-csv results/efficiency/params.csv
python scripts/benchmark_inference_latency.py   --ckpt-name "dummy"   --ckpt-episode "latest"   --num-k-infer "1"   --output-csv results/efficiency/latency.csv
