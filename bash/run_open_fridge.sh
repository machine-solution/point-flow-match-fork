#!/bin/bash

# Script to run open_fridge task with visualization
# This is a convenience script that calls run_open_fridge_vis.sh
# For headless mode, use: bash bash/run_open_fridge_headless.sh

bash "$(dirname "$0")/run_open_fridge_vis.sh"
