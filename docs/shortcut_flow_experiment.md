# Shortcut PointFlowMatch Experiment

## Important Separation from MeanFlow

This experiment is separate from MeanFlow.

- MeanFlow files/configs are not modified.
- Shortcut Flow is implemented as a separate policy (`ShortcutFMPolicy`) and separate configs.

## Idea

Shortcut Flow conditions the model on explicit step size `d`:

- model: `s_theta(x_t, t, d, obs)`
- update: `x_{t+d} = x_t + d * s_theta(x_t, t, d, obs)`

The model receives `d` only via conditioning (not injected into trajectory state).

## Training Objective

Total loss:

- `L = base_loss_weight * L_base + consistency_loss_weight * L_sc`

Where:

- `L_base`: standard FM-style velocity supervision at small step (`d_min = 1 / num_base_steps`)
- `L_sc`: self-consistency loss enforcing:
  - one big step (`2d`) is close to
  - two small steps (`d` then `d`)
  - step levels include mandatory `d=0.5`, so consistency directly supervises big step `2d=1.0`

Implemented consistency relation:

- `x_big = x_t + 2d * s_theta(x_t, t, 2d, obs)`
- `x_two = x_t + d * s_theta(x_t, t, d, obs)` then one more step from `x_mid`
- `L_sc = MSE(x_big, stopgrad(x_two))` (configurable with `stopgrad_target`)
- optional direct one-step branch:
  - sample anchor time `t_anchor ~ Uniform(0, 0.5)`
  - build anchor state on interpolation path:
    - `x_anchor = (1 - t_anchor) * z0 + t_anchor * z1`
  - predict one full step from anchor:
    - `x_full` from `d=1.0` at `t=t_anchor`
  - target from two half steps from same anchor:
    - `d=0.5` at `t=t_anchor` and `t=t_anchor+0.5`
  - weighted by `shortcut.one_step_loss_weight` when `shortcut.include_one_step_target=true`

## Inference Behavior

Supports both:

- `K=1`: one-step inference (`NFE=1`)
- `K>1`: iterative shortcut updates (`NFE=K`)

`last_infer_nfe` is tracked accordingly. This experiment explicitly targets the deployed one-step regime (`num_k_infer=1`).

## What Stays the Same

- dataset
- observation encoder
- normalization
- train/eval pipeline
- 5p/gripper loss structure for base loss

## Main Comparison Grid

Compare:

- baseline `K=10`
- baseline `K=1`
- MeanFlow `K=1`
- Shortcut `K=1`
- Shortcut `K=2/5/10`

## Dexter Commands (Only)

Train:

```bash
TASK_NAME=open_fridge \
RUN_NAME=shortcut_open_fridge \
bash bash/dexter_train_shortcut_flow.sh
```

Validate:

```bash
CKPT_NAME=<shortcut_run_name> \
NUM_EPISODES=100 \
bash bash/dexter_validate_shortcut_flow.sh
```
