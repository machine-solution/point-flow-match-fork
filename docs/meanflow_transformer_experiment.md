# MeanFlow + Small Temporal Transformer Experiment

## Motivation

Current MeanFlow with Conv1D U-Net is very fast (`~25-30 ms/action`, `NFE=1`) but has low success rate (`~6-8%`) on `open_fridge`.
Baseline references are higher in quality (`~10%` at `K=1`, `~28%` at `K=10`).

Hypothesis: one-step MeanFlow needs stronger global temporal modeling across the predicted trajectory (e.g. approach -> grasp -> pull), which a small Transformer may capture better.

## Architecture

Experiment target:
- MeanFlow objective (interval-averaged velocity target)
- one-step inference (`num_k_infer=1`)
- temporal backbone replaced with a small Transformer encoder over trajectory timesteps

Implemented backbone:
- `pfp/models/temporal_transformer.py`
- class `TemporalTransformerBackbone`

Core structure:
- input projection (`Linear(input_dim -> d_model)`)
- learnable temporal positional embedding
- global conditioning projection added to every timestep
- Transformer encoder (`num_layers=2`, `num_heads=4`, GELU MLP block)
- optional final LayerNorm
- output projection (`Linear(d_model -> output_dim)`)

## What Changes

- `MeanFlowPolicy` diffusion backbone changes from Conv1D U-Net to temporal Transformer in:
  - `conf/model/meanflow_transformer.yaml`
- New experiment entrypoint:
  - `+experiment=pointflowmatch_meanflow_transformer`

## What Does NOT Change

- dataset and sampling setup
- observation encoder (`PointNetBackbone` etc. from `backbone` config group)
- normalization pipeline
- MeanFlow objective and one-step inference logic
- train/eval scripts and checkpoint format conventions

## Expected Metrics

- Success rate should improve vs MeanFlow Conv variant.
- `NFE/action` should remain `1`.
- Latency may increase moderately vs Conv MeanFlow, but should remain far below baseline `K=10`.
- Parameter count should stay controlled (small Transformer design).

## Dexter Commands (Only)

Train:

```bash
TASK_NAME=open_fridge \
RUN_NAME=meanflow_transformer_open_fridge \
bash bash/dexter_train_meanflow_transformer.sh
```

Validate:

```bash
CKPT_NAME=<meanflow_transformer_run_name> \
NUM_EPISODES=100 \
bash bash/dexter_validate_meanflow_transformer.sh
```
