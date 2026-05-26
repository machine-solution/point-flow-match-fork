# Efficiency Experiments (Dexter-Only)

This document is Dexter-oriented. Heavy training, RLBench validation, and benchmarks should be submitted on Dexter using the wrapper scripts in `bash/`.

## Repository Mapping (Expected -> Actual)

All expected paths match the repository exactly.

- `conf/train.yaml` -> `conf/train.yaml`
- `conf/model/flow.yaml` -> `conf/model/flow.yaml`
- `conf/model/meanflow.yaml` -> `conf/model/meanflow.yaml`
- `conf/experiment/pointflowmatch.yaml` -> `conf/experiment/pointflowmatch.yaml`
- `conf/experiment/pointflowmatch_meanflow.yaml` -> `conf/experiment/pointflowmatch_meanflow.yaml`
- `pfp/policy/fm_policy.py` -> `pfp/policy/fm_policy.py`
- `pfp/policy/meanflow_policy.py` -> `pfp/policy/meanflow_policy.py`
- `pfp/policy/base_policy.py` -> `pfp/policy/base_policy.py`
- `pfp/envs/rlbench_runner.py` -> `pfp/envs/rlbench_runner.py`
- `scripts/train.py` -> `scripts/train.py`
- `scripts/debug_meanflow_config.py` -> `scripts/debug_meanflow_config.py`
- `scripts/sweep_num_k_infer.py` -> `scripts/sweep_num_k_infer.py`
- `scripts/benchmark_inference_latency.py` -> `scripts/benchmark_inference_latency.py`
- `scripts/benchmark_train_step.py` -> `scripts/benchmark_train_step.py`
- `scripts/count_policy_params.py` -> `scripts/count_policy_params.py`
- `bash/run_validate_accuracy.sh` -> `bash/run_validate_accuracy.sh`

## MeanFlow Summary

MeanFlow predicts interval-averaged velocity and runs one-step inference by design.

- Architecture branch: `MeanFlowPolicy` (`pfp/policy/meanflow_policy.py`)
- Config: `conf/model/meanflow.yaml`
- Inference expectation:
  - `nfe_per_action`: `10 -> 1` (relative to baseline `K=10`)
  - lower inference latency

## Decision Logic (from K-sweep)

Run baseline K-sweep first and compare `K=1` vs `K=10`.

- If `K=1` is close to `K=10` -> MeanFlow is not needed.
- If `K=1` is much worse than `K=10` -> proceed with MeanFlow.

Current status: baseline showed strong degradation at `K=1`; MeanFlow branch selected.

## Dexter Scripts (Copy-Paste)

### 1) MeanFlow Training

```bash
TASK_NAME=open_fridge \
EXPERIMENT=pointflowmatch_meanflow \
RUN_NAME=meanflow_open_fridge_ep1500 \
bash bash/dexter_train_meanflow.sh
```

Main train command inside job:

```bash
python scripts/train.py task_name=open_fridge +experiment=pointflowmatch_meanflow
```

### 2) MeanFlow Validation

```bash
CKPT_NAME=<MEANFLOW_CKPT_NAME> \
NUM_EPISODES=100 \
bash bash/dexter_validate_meanflow.sh
```

Validation entrypoint inside job:

```bash
bash bash/run_validate_accuracy.sh <CKPT_NAME> <NUM_EPISODES>
```

### 3) Baseline K-sweep Validation

```bash
CKPT_NAME=<BASELINE_CKPT_NAME> \
CKPT_EPISODE=ep1500 \
NUM_EPISODES=100 \
KS=1,2,4,6,8,10 \
bash bash/dexter_sweep_num_k_infer.sh
```

Output:

- `results/efficiency/k_sweep_<CKPT_NAME>.csv`
- `results/efficiency/k_sweep_<CKPT_NAME>.json`

### 4) MeanFlow Benchmark (Latency + Params)

```bash
CKPT_NAME=<MEANFLOW_CKPT_NAME> \
CKPT_EPISODE=latest \
NUM_K_INFER=1 \
bash bash/dexter_benchmark_meanflow.sh
```

Outputs:

- `results/efficiency/latency.csv`
- `results/efficiency/params.csv`

## Comparison Table to Produce

Use Dexter runs to compare:

- baseline `K=10`
- baseline `K=1`
- MeanFlow `K=1`

Recommended metrics:

- success rate
- `nfe_per_action`
- mean inference latency
- mean episode time
- parameter count
