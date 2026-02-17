#!/bin/bash

# Script to play back recorded actions with visualization
# Usage: bash bash/playback_open_fridge.sh <recorded_actions_file.json>

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pfp_env

# Check if input file is provided
if [ -z "$1" ]; then
    echo "Error: Please provide the recorded actions file"
    echo "Usage: bash bash/playback_open_fridge.sh <recorded_actions_file.json>"
    echo "Example: bash bash/playback_open_fridge.sh recorded_actions_1717446565-astute-stingray.json"
    exit 1
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
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

echo "Playing back actions from: $INPUT_FILE"
echo "Running with visualization (vis=True, headless=False)"
echo "CoppeliaSim window will open - you can watch the robot in action!"

# Play back recorded actions with visualization
python scripts/playback_actions.py +input_file=$INPUT_FILE
