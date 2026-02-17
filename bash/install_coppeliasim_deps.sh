#!/bin/bash

# Script to install dependencies for CoppeliaSim
# These libraries help CoppeliaSim work properly and avoid crashes

echo "Installing CoppeliaSim dependencies..."
echo "This requires sudo privileges"

sudo apt-get update
sudo apt-get install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    libtheora-dev \
    libxvidcore-dev

echo ""
echo "Dependencies installed successfully!"
echo "You may need to restart CoppeliaSim or your terminal for changes to take effect."
