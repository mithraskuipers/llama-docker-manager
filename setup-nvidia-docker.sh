#!/usr/bin/env bash
# Install NVIDIA Container Toolkit for Docker GPU support
# Run once on the host machine (not inside a container).
set -euo pipefail

echo "Installing NVIDIA Container Toolkit..."

# Add GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add APT repo (works for Ubuntu 22.04 / Debian-based distros)
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update -qq
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "Done. Verify with: docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi"
