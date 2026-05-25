# Efficiency Experiments: PointFlowMatch

This document defines a gated workflow for architecture changes focused on efficiency.

## Why K-sweep Comes First

Before implementing a new architecture, check whether baseline quality already holds at low `num_k_infer`.

- If baseline at `K=1` is close to `K=10`, reducing K is already enough and MeanFlow is not a meaningful contribution.
- If baseline at `K=1` degrades strongly vs `K=10`, there is a quality-efficiency gap worth closing with one-step MeanFlow.

## Decision Rule

Run baseline K-sweep first (`K in {1,2,4,6,8,10}`) on a fixed checkpoint and seed.

- **Case A (MeanFlow path):** `K=1` is much worse than `K=10` (example: 50% vs <=30-35%).
- **Case B (LatentPFM path):** `K=1` is close to `K=10` (example: 50% vs >=45%).

Do not implement both paths at once.

## Scripts

- `scripts/sweep_num_k_infer.py`: success + latency + NFE vs K, writes `results/efficiency/k_sweep_*.csv`.
- `scripts/benchmark_inference_latency.py`: micro-benchmark inference time and NFE.
- `scripts/count_policy_params.py`: total/trainable params and module breakdown.
- `scripts/benchmark_train_step.py`: one train step wall-time and peak GPU memory.

## Repro Commands

### 1) Baseline K-sweep (mandatory first step)

```bash
python scripts/sweep_num_k_infer.py \
  --ckpt-name 1779122560-baseline-many-ckpts \
  --ckpt-episode ep1500 \
  --num-episodes 50 \
  --max-episode-length 120 \
  --seed 5678 \
  --ks 1,2,4,6,8,10 \
  --phase-conditioning disabled \
  --phase-prediction disabled \
  --output-csv results/efficiency/k_sweep_baseline_ep1500.csv
```

### 2) Parameter count

```bash
python scripts/count_policy_params.py \
  --ckpt-name 1779122560-baseline-many-ckpts \
  --ckpt-episode ep1500 \
  --num-k-infer 10 \
  --output-csv results/efficiency/params.csv
```

### 3) Inference latency benchmark

```bash
python scripts/benchmark_inference_latency.py \
  --ckpt-name 1779122560-baseline-many-ckpts \
  --ckpt-episode ep1500 \
  --num-k-infer 10 \
  --batch-size 1 \
  --warmup-iters 20 \
  --timed-iters 100 \
  --output-csv results/efficiency/latency.csv
```

### 4) Train-step benchmark

```bash
python scripts/benchmark_train_step.py \
  --train-config conf/train.yaml \
  --batch-size 64 \
  --output-csv results/efficiency/train_step.csv
```

## Required Result Tables

Store all benchmark outputs under `results/efficiency/`.

- `k_sweep*.csv`
- `latency.csv`
- `params.csv`
- `train_step.csv`
- `eval_success.csv` (from evaluation runs)

## Current Decision (Baseline K-sweep)

From `results/efficiency/k_sweep_baseline_ep1500_seed5678_n100_k1_k10.csv`:

- `K=1`: success_rate `0.10` (10/100), mean_inference_ms `718.29`
- `K=10`: success_rate `0.28` (28/100), mean_inference_ms `3179.22`

This is **Case A** (`K=1` degrades strongly vs `K=10`), so the next architecture branch is:

- **MeanFlow / one-step PointFlowMatch**
