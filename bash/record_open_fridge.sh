#!/bin/bash

# Script to record actions for open_fridge task (headless, fast)
# Checkpoint: 1717446565-astute-stingray

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
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

OUTPUT_FILE="recorded_actions_${CKPT_NAME}.json"

echo "Recording actions for open_fridge task with checkpoint: $CKPT_NAME"
echo "Output file: $OUTPUT_FILE"
echo "Using headless mode (fast, no GUI)"

# Record actions in headless mode (fast)
# Use xvfb-run for virtual display (required for CoppeliaSim headless mode)
xvfb-run -a -s "-screen 0 1024x768x24" python scripts/record_actions.py \
    log_wandb=False \
    env_runner.env_config.vis=False \
    env_runner.env_config.headless=True \
    env_runner.num_episodes=10 \
    policy.ckpt_name=$CKPT_NAME \
    policy.ckpt_episode=ep1500 \
    +output_file=$OUTPUT_FILE

echo ""
echo "Recording complete! To play back with visualization, run:"
echo "bash bash/playback_open_fridge.sh $OUTPUT_FILE"
