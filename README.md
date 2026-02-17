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

To train your own policies instead of using the pretrained checkpoints, you first need to collect demonstrations:

```bash
bash bash/collect_data.sh
```

Then, you can train your own policies:

```bash
python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=<task_name> +experiment=<experiment_name>
```

Valid task names are all those supported by RLBench. In this work, we used the following tasks: `unplug_charger`, `close_door`, `open_box`, `open_fridge`, `take_frame_off_hanger`, `open_oven`, `put_books_on_bookshelf`, `take_shoes_out_of_box`.

Valid experiment names are the following, and they represent the different baselines we tested: `adaflow`, `diffusion_policy`, `dp3`, `pointflowmatch`, `pointflowmatch_images`, `pointflowmatch_ddim`, `pointflowmatch_so3`.

## Running training on Dexter (DGX A100)

In the `dexter/` folder you can find helper files for running PointFlowMatch training on a Slurm‑managed DGX A100 cluster:

- `dexter/instruction.md` – short Russian introduction to Slurm on Dexter (queues, `sbatch`, how to read `.out/.err` logs).
- `dexter/pfp_train_env.yml` – Conda environment for offline training (no CoppeliaSim / RLBench required).
- `dexter/run_pointflowmatch_open_fridge.sbatch` – example Slurm script for training the PointFlowMatch baseline on the `open_fridge` task using existing demos.

Typical workflow on Dexter:

```bash
git clone https://github.com/<your_username>/PointFlowMatch.git
cd PointFlowMatch

# Create training environment locally in the repo
conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env

# Submit training job (from repo root)
sbatch dexter/run_pointflowmatch_open_fridge.sbatch
```

The training script expects demonstrations to be present under:

- `demos/sim/open_fridge/train`
- `demos/sim/open_fridge/valid`

Large training data and checkpoints are **not** committed to this repository (see `.gitignore`); they should be stored locally on the cluster or downloaded separately.