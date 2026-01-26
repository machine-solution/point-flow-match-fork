#!/bin/bash

# Script to run open_fridge task with visualization
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

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "Error: DISPLAY environment variable is not set."
    echo "For local machine: DISPLAY should be set automatically"
    echo "For remote server with X11 forwarding: ssh -X user@server"
    echo "Or set manually: export DISPLAY=:0"
    exit 1
fi

echo "Running open_fridge task with checkpoint: $CKPT_NAME"
echo "Running with visualization (vis=True, headless=False)"
echo "CoppeliaSim window will open - you can watch the robot in action!"

# Run evaluation with visualization enabled
# Note: Requires a display (either physical or X11 forwarding)
python scripts/evaluate.py \
    log_wandb=False \
    env_runner.env_config.vis=True \
    env_runner.env_config.headless=False \
    env_runner.num_episodes=3 \
    policy.ckpt_name=$CKPT_NAME \
    policy.ckpt_episode=ep1500
