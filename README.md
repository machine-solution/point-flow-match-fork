# PointFlowMatch: Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching

Repository providing the source code for the paper "Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching", see the [project website](http://pointflowmatch.cs.uni-freiburg.de/). Please cite the paper as follows:

	@article{chisari2024learning,
	  title={Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching},
      shorttile={PointFlowMatch},
	  author={Chisari, Eugenio and Heppert, Nick and Argus, Max and Welschehold, Tim and Brox, Thomas and Valada, Abhinav},
	  journal={Conference on Robot Learning (CoRL)},
	  year={2024}
	}

## Installation

- Add env variables to your `.bashrc`

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

- Install dependencies

```bash
conda create --name pfp_env python=3.10
conda activate pfp_env
bash bash/install_deps.sh
bash bash/install_rlbench.sh

# Get diffusion_policy from my branch
cd ..
git clone git@github.com:chisarie/diffusion_policy.git && cd diffusion_policy && git checkout develop/eugenio 
pip install -e ../diffusion_policy

# 3dp install
cd ..
git clone git@github.com:YanjieZe/3D-Diffusion-Policy.git && cd 3D-Diffusion-Policy
cd 3D-Diffusion-Policy && pip install -e . && cd ..

# If locally (doesnt work on Ubuntu18):
pip install rerun-sdk==0.15.1
```

### CoppeliaSim dependencies

For stable CoppeliaSim work (without random crashes and video‑codec errors) additional system libraries are required.  
See `COPPELIASIM_DEPS.md` for details or run:

```bash
bash bash/install_coppeliasim_deps.sh
```

On a clean Ubuntu installation you may also need a virtual X server for headless runs:

```bash
sudo apt-get install xvfb
```

## Pretrained Weights Download

Here you can find the pretrained checkpoints of our PointFlowMatch policies for different RLBench environments. Download and unzip them in the `ckpt` folder.

| unplug charger | close door | open box | open fridge | frame hanger | open oven | books on shelf | shoes out of box |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [1717446544-didactic-woodpecker](http://pointflowmatch.cs.uni-freiburg.de/download/1717446544-didactic-woodpecker.zip) | [1717446607-uppish-grebe](http://pointflowmatch.cs.uni-freiburg.de/download/1717446607-uppish-grebe.zip) | [1717446558-qualified-finch](http://pointflowmatch.cs.uni-freiburg.de/download/1717446558-qualified-finch.zip) | [1717446565-astute-stingray](http://pointflowmatch.cs.uni-freiburg.de/download/1717446565-astute-stingray.zip) | [1717446708-analytic-cuckoo](http://pointflowmatch.cs.uni-freiburg.de/download/1717446708-analytic-cuckoo.zip) | [1717446706-natural-scallop](http://pointflowmatch.cs.uni-freiburg.de/download/1717446706-natural-scallop.zip) | [1717446594-astute-panda](http://pointflowmatch.cs.uni-freiburg.de/download/1717446594-astute-panda.zip) | [1717447341-indigo-quokka](http://pointflowmatch.cs.uni-freiburg.de/download/1717447341-indigo-quokka.zip) |

## Evaluation

To reproduce the results from the paper, run:

```bash
python scripts/evaluate.py log_wandb=True env_runner.env_config.vis=False policy.ckpt_name=<ckpt_name>
```

Where `<ckpt_name>` is the folder name of the selected checkpoint. Each checkpoint will be automatically evaluated on the correct environment.

### Recording and playback of actions (open_fridge example)

You can record actions of a pretrained `open_fridge` policy and later play them back in CoppeliaSim.

- **Record a batch of episodes (headless, fast):**

```bash
bash bash/record_open_fridge_batch.sh
```

This will:

- use checkpoint `1717446565-astute-stingray` (downloaded into `ckpt/`);
- run RLBench/CoppeliaSim in headless mode via `xvfb-run`;
- save individual episodes into `recordings/open_fridge_seed<SEED>.json`.

- **Record a single multi‑episode file:**

```bash
bash bash/record_open_fridge.sh
```

This script writes `recorded_actions_<CKPT>.json` (by default to `recordings/`) that contains multiple episodes in one file.

- **Play back recorded actions with GUI:**

```bash
bash bash/playback_open_fridge.sh recordings/open_fridge_seed5678.json
```

or for a multi‑episode file:

```bash
bash bash/playback_open_fridge.sh recordings/recorded_actions_1717446565-astute-stingray.json
```

During playback CoppeliaSim runs with a visible window (`headless=False`, `vis=True`), so you can watch the robot trajectory.

## Training

To train your own policies instead of using the pretrained checkpoints, you first need to collect demonstrations.

### CoppeliaSim version for data collection (RLBench)

**For `collect_demos.py` you must use CoppeliaSim 4.1.0.** Newer versions (e.g. 4.10) cause `RuntimeError: Handle Panda does not exist` because the RLBench scene `task_design.ttt` was made for 4.1.

- Download: [CoppeliaSim Edu V4.1.0 Ubuntu 20.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz) (works on Ubuntu 22.04).
- Extract to a folder. If you extract the archive **inside this repo**, the script `bash/collect_data_open_fridge.sh` will use it automatically (it expects `CoppeliaSim_Edu_V4_1_0_Ubuntu20_04` in the repo root). Otherwise set `COPPELIASIM_ROOT` (and `LD_LIBRARY_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH`) to the extracted folder.


For headless runs, install xvfb; the planner does not always find a path—RLBench retries automatically.

```bash
sudo apt-get install xvfb
```

Then collect demos for open_fridge:

```bash
# Using the helper script (uses CoppeliaSim 4.1.0 from repo if present)
xvfb-run -a bash bash/collect_data_open_fridge.sh
```

Or manually (set `COPPELIASIM_ROOT` first if CoppeliaSim is not in the repo):

```bash
xvfb-run -a python scripts/collect_demos.py --config-name=collect_demos_train save_data=True env_config.vis=False env_config.task_name=open_fridge env_config.headless=True
xvfb-run -a python scripts/collect_demos.py --config-name=collect_demos_valid save_data=True env_config.vis=False env_config.task_name=open_fridge env_config.headless=True
```

Output is written to `demos/sim/open_fridge/train` and `demos/sim/open_fridge/valid`. To collect all tasks as in the paper, use:

```bash
bash bash/collect_data.sh
```

Then, you can train your own policies (default task: `open_fridge`; change `task_name` for other paper tasks):

```bash
python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=open_fridge +experiment=pointflowmatch
```

**Paper task names** (`task_name=...`):

| | | | |
|---|---|---|---|
| `unplug_charger` | `close_door` | `open_box` | `open_fridge` |
| `take_frame_off_hanger` | `open_oven` | `put_books_on_bookshelf` | `take_shoes_out_of_box` |

Demos: `demos/sim/<task_name>/{train,valid}`, extract `demos_<task_name>_sim.tar.gz`, or `bash dexter/download_dataset.sh [task_name ...]` (no args = all 8 tasks).

Valid experiment names are the following, and they represent the different baselines we tested: `adaflow`, `diffusion_policy`, `dp3`, `pointflowmatch`, `pointflowmatch_images`, `pointflowmatch_ddim`, `pointflowmatch_so3`.

### Two-phase training for `open_fridge` (pre / post first grasp)

To train **two** policies—before and after the first gripper close—split demonstrations with `two_layers_planning/split_dataset_at_first_grasp.py`, then train with Hydra configs `train_open_fridge_pre_grasp` and `train_open_fridge_post_grasp`, or submit the Slurm scripts under `two_layers_planning/sbatch/`. Those configs use **`use_validation: false`**, so you only need the **train** zarr (no separate valid split). Step-by-step instructions (Russian), Yandex Disk links for pre-built archives, and environment notes: **`two_layers_planning/README.md`**.

### Phase-Conditioned Single Model (no hard switch)

Instead of training **two separate** policies (pre/post) and hard-switching between them (which can introduce distribution shift), you can train **one** shared PointFlowMatch model with an additional **phase token**:

\[
v_\\theta(x_t, t, obs) \\;\;\\to\\;\; v_\\theta(x_t, t, obs, phase)
\]

Phases are discrete labels:

- `0`: approach / pre-grasp
- `1`: contact window around the first gripper close
- `2`: manipulation / post-grasp

The phase labels are generated from demonstrations using a simple heuristic over the gripper channel `robot_state[:, 9]`:

- detect the first timestep where `gripper_open < gripper_close_threshold`
- set phase `0` before it
- set phase `1` in a `contact_window` around it
- set phase `2` after the window

Implementation note: phase conditioning is done as **true conditioning input** to the velocity network,
not by adding a phase-dependent bias to the trajectory state \(x_t\). Concretely, we concatenate a
per-timestep phase embedding to the sample input of the diffusion network and slice the output so the
network still predicts velocity only for the original action dimensions.

Enable it via the Hydra config group `phase_conditioning`:

```bash
# Baseline (no phase conditioning)
python scripts/train.py task_name=open_fridge +experiment=pointflowmatch

# Phase-conditioned single model
python scripts/train.py task_name=open_fridge +experiment=pointflowmatch phase_conditioning=enabled
```

Config knobs:

- `phase_conditioning.enabled` (bool)
- `phase_conditioning.num_phases` (default 3)
- `phase_conditioning.contact_window` (int)
- `phase_conditioning.gripper_close_threshold` (float)
- `phase_conditioning.phase_embed_dim` (int)

Debugging phase labels on a zarr dataset:

```bash
PYTHONPATH=../diffusion_policy python scripts/debug_phase_labels.py \
  --zarr demos/sim/open_fridge/train \
  --episodes 0,1,2 \
  --thr 0.5 \
  --contact-window 2 \
  --out outputs/debug_phase_labels.png
```

Debugging a forward pass (tensor shapes + one `infer_y`) for phase-enabled/disabled:

```bash
PYTHONPATH=../diffusion_policy python scripts/debug_phase_forward.py --phase enabled
PYTHONPATH=../diffusion_policy python scripts/debug_phase_forward.py --phase disabled
```

### Paper task datasets: download commands

From the **repository root** (needs network, `python3`, `tar`):

```bash
# All 8 paper tasks from Yandex Disk (sequential download)
bash dexter/download_dataset.sh

# Single task (default example: open_fridge)
bash dexter/download_dataset.sh open_fridge
bash dexter/download_dataset.sh close_door open_box

# List public share URLs
bash dexter/download_dataset.sh --list
```

Re-download: append `--force`. On Slurm clusters, run on a login node or interactive session.

**Two-phase `open_fridge` only** (pre/post grasp splits):

```bash
bash dexter/download_open_fridge_two_phase.sh
```

| Goal | Result on disk | `valid`? |
|------|----------------|----------|
| Baseline (`run_pointflowmatch_open_fridge*.sbatch`, any `TASK_NAME`) | `demos/sim/<task>/train`, `valid` | yes |
| Two-phase (`train_open_fridge_*_grasp`) | `train_pre_grasp`, `train_post_grasp` | no |

Alternatively: collect demos with `bash/collect_data.sh`, or split `train` with `two_layers_planning/split_dataset_at_first_grasp.py`. Details: **`two_layers_planning/README.md`**, **`dexter/README_pointflowmatch_dexter.md`** §2.

## Running training on Dexter (DGX A100)

In the `dexter/` folder you can find helper files for running PointFlowMatch training on a Slurm‑managed DGX A100 cluster:

- `dexter/instruction.md` – short Russian introduction to Slurm on Dexter (queues, `sbatch`, how to read `.out/.err` logs).
- `dexter/pfp_train_env.yml` – Conda environment for offline training (no CoppeliaSim / RLBench required).
- `dexter/run_pointflowmatch_open_fridge*.sbatch` – Slurm scripts for baseline / MeanFlow / phase / etc.; set **`TASK_NAME=open_fridge`** (default) or any other paper task.
- `dexter/download_dataset.sh` – download demo bundles from Yandex Disk (`bash dexter/download_dataset.sh` = all 8 tasks; `bash dexter/download_dataset.sh open_fridge` = one task).
- `dexter/download_open_fridge_two_phase.sh` – two-phase pre/post zarr for `open_fridge` only.
- `two_layers_planning/README.md` – two-phase workflow, split script, and **local** `.venv` sbatch under `two_layers_planning/sbatch/`.
- `dexter/verify_training_setup.py` – quick check that Hydra instantiates `FMPolicy` with the intended phase settings (no GPU/dataset).
- `dexter/run_open_fridge_pre_grasp.sbatch`, `dexter/run_open_fridge_post_grasp.sbatch`, `dexter/run_open_fridge_two_phase_chain.sbatch` – two-phase `open_fridge` only.

Typical workflow on Dexter:

```bash
git clone https://github.com/<your_username>/PointFlowMatch.git
cd PointFlowMatch

# Create training environment locally in the repo
conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env

# Submit training (default task: open_fridge; change TASK_NAME for other paper tasks)
TASK_NAME=open_fridge sbatch dexter/run_pointflowmatch_open_fridge.sbatch
TASK_NAME=open_fridge sbatch dexter/run_pointflowmatch_open_fridge_momentum_meanflow.sbatch
```

All `dexter/run_pointflowmatch_open_fridge*.sbatch` scripts accept **`TASK_NAME`** (Hydra `task_name=...`). Data is fetched automatically via `dexter/ensure_task_dataset.sh` (Yandex download for all 8 paper tasks). For **two-phase** `open_fridge`, run `bash dexter/download_open_fridge_two_phase.sh`, then `dexter/run_open_fridge_pre_grasp.sbatch` / `post_grasp.sbatch`. Full guide: **`dexter/README_pointflowmatch_dexter.md`**.

Large training data and checkpoints are **not** committed to this repository (see `.gitignore`); they should be stored locally on the cluster or downloaded separately.