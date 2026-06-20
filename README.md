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

Then, you can train your own policies:

```bash
python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=<task_name> +experiment=<experiment_name>
```

Valid task names are all those supported by RLBench. In this work, we used the following tasks: `unplug_charger`, `close_door`, `open_box`, `open_fridge`, `take_frame_off_hanger`, `open_oven`, `put_books_on_bookshelf`, `take_shoes_out_of_box`.

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

### `open_fridge` datasets: commands (cluster / VM)

From the **repository root** (needs network, `python3`, `tar`):

```bash
# Baseline PointFlowMatch: train + valid (~4.3 GB one archive)
bash dexter/download_dataset.sh

# Two-phase policies: train_pre_grasp + train_post_grasp (two archives; large total size)
bash dexter/download_open_fridge_two_phase.sh
```

Re-download: append `--force` to either script. On Slurm clusters, run these on a login node or an interactive allocation, not as an unattended one-line `sbatch` unless you know the job will finish (downloads can take a long time).

| Goal | Result on disk | `valid`? |
|------|----------------|----------|
| Baseline (`run_pointflowmatch_open_fridge.sbatch`, etc.) | `demos/sim/open_fridge/train`, `valid` | yes |
| Two-phase (`train_open_fridge_*_grasp` configs, `two_layers_planning/sbatch/…`) | `train_pre_grasp`, `train_post_grasp` | no |

Alternatively: collect demos with `collect_demos.py`, or split `train` locally with `two_layers_planning/split_dataset_at_first_grasp.py` instead of `download_open_fridge_two_phase.sh`. More detail: **`two_layers_planning/README.md`**, **`dexter/README_pointflowmatch_dexter.md`** §2.

## Running training on Dexter (DGX A100)

In the `dexter/` folder you can find helper files for running PointFlowMatch training on a Slurm‑managed DGX A100 cluster:

- `dexter/instruction.md` – short Russian introduction to Slurm on Dexter (queues, `sbatch`, how to read `.out/.err` logs).
- `dexter/pfp_train_env.yml` – Conda environment for offline training (no CoppeliaSim / RLBench required).
- `dexter/run_pointflowmatch_open_fridge.sbatch` – example Slurm script for training the PointFlowMatch baseline on the `open_fridge` task using existing demos.
- `dexter/run_pointflowmatch_open_fridge_phase_prediction.sbatch` – Slurm training for **FMPolicy + learned phase head** (`phase_conditioning=enabled`, `phase_prediction=enabled`).
- `dexter/run_pointflowmatch_open_fridge_momentum_meanflow.sbatch` – Slurm training for **Momentum / Self-Correcting MeanFlow** (`+experiment=pointflowmatch_momentum_meanflow`).
- `dexter/verify_training_setup.py` – quick check that Hydra instantiates `FMPolicy` with the intended phase settings (no GPU/dataset).
- `dexter/download_dataset.sh` / `dexter/download_open_fridge_two_phase.sh` – download baseline or two-phase zarr from Yandex Disk (run from repo root).
- `dexter/run_open_fridge_pre_grasp.sbatch`, `dexter/run_open_fridge_post_grasp.sbatch`, `dexter/run_open_fridge_two_phase_chain.sbatch` – Slurm training for two-phase `open_fridge` on Dexter (conda `pfp-train-env`).
- `two_layers_planning/README.md` – two-phase workflow, split script, and **local** `.venv` sbatch under `two_layers_planning/sbatch/`.

Typical workflow on Dexter:

```bash
git clone https://github.com/<your_username>/PointFlowMatch.git
cd PointFlowMatch

# Create training environment locally in the repo
conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env

# Submit training job (from repo root)
sbatch dexter/run_pointflowmatch_open_fridge.sbatch

# Momentum / Self-Correcting MeanFlow (open_fridge, 1500 epochs, milestone checkpoints)
sbatch dexter/run_pointflowmatch_open_fridge_momentum_meanflow.sbatch
```

For **`dexter/run_pointflowmatch_open_fridge.sbatch`**, put data under `demos/sim/open_fridge/train` and `valid` (the sbatch may call `bash dexter/download_dataset.sh` if missing). For **two-phase** training on Dexter, run `bash dexter/download_open_fridge_two_phase.sh`, then e.g. `PRE=$(sbatch --parsable dexter/run_open_fridge_pre_grasp.sbatch)` and `sbatch --dependency=afterok:"$PRE" dexter/run_open_fridge_post_grasp.sbatch`, or use `dexter/run_open_fridge_two_phase_chain.sbatch` for one long job. See **`dexter/README_pointflowmatch_dexter.md`** §3.

Large training data and checkpoints are **not** committed to this repository (see `.gitignore`); they should be stored locally on the cluster or downloaded separately.