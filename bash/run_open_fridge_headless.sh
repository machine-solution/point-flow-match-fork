#!/bin/bash

# Script to run open_fridge task in headless mode (no visualization)
# Checkpoint: 1717446565-astute-stingray

# Activate conda environment
conda activate pfp_env

# Check if checkpoint exists
CKPT_NAME="1717446565-astute-stingray"
CKPT_PATH="ckpt/${CKPT_NAME}"

if [ ! -d "$CKPT_PATH" ]; then
    echo "Warning: Checkpoint not found at $CKPT_PATH"
    echo "Please download it from: http://pointflowmatch.cs.uni-freiburg.de/download/1717446565-astute-stingray.zip"
    echo "And extract it to the ckpt folder."
    exit 1
fi

echo "Running open_fridge task with checkpoint: $CKPT_NAME"
echo "Running in headless mode (vis=False, headless=True)"

# Run evaluation in headless mode (no visualization)
# Use xvfb-run for virtual display (required for CoppeliaSim headless mode)
# -a: auto display number, -s: screen settings (24 bit color, 1024x768 resolution)
xvfb-run -a -s "-screen 0 1024x768x24" python scripts/evaluate.py \
    log_wandb=False \
    env_runner.env_config.vis=False \
    env_runner.env_config.headless=True \
    env_runner.num_episodes=5 \
    policy.ckpt_name=$CKPT_NAME \
    policy.ckpt_episode=ep1500
