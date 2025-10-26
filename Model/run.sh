##
# @file run.sh
# @brief Shell script to build and run the Docker container for the sleep analysis project.
#
# This script must be run from a shell that has 'docker' and 'xhost'.
# On NixOS, run this with:
#   nix-shell -p xorg.xhost --run ./run_docker.sh
#
# Steps:
# 1. Stop and remove any old container.
# 2. Grant X server access for GUI forwarding.
# 3. Build the Docker image.
# 4. Run the Docker container with GPU and device access.
# 5. Clean up X server permissions after exit.

#!/usr/bin/env bash
set -e

# --- Configuration ---
IMAGE_NAME="pyt-rocm-project"
CONTAINER_NAME="pyt-rocm-container"

##
# @step 1
# @brief Stop and remove any old container with the same name.
echo "Cleaning up any old container named $CONTAINER_NAME..."
docker rm -f $CONTAINER_NAME || true

##
# @step 2
# @brief Grant Docker container access to your X server for GUI forwarding.
echo "Granting Docker container access to your X server (for XWayland)..."
xhost +local:docker

##
# @step 3
# @brief Build the Docker image from the Model directory.
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME ./Model

##
# @step 4
# @brief Run the Docker container with GPU, device, and X11 access.
docker run --rm -it \
  --name $CONTAINER_NAME \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/ttyACM0:/dev/ttyACM0 \
  --group-add=video \
  --group-add=render \
  -v "$(pwd)":/app \
  -v "$(pwd)/DataCollection":/app/DataCollection \
  -v "$(pwd)/KernelExperiment":/app/KernelExperiment \
  $IMAGE_NAME

##
# @step 5
# @brief Clean up X server permissions after the container exits.
echo "Cleaning up X server permissions..."
xhost -local:docker
