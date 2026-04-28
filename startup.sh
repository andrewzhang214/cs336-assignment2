#!/bin/bash


# Install uv and setup

mkdir -p /workspace/bin

curl -Ls https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/workspace sh

grep -qxF 'export PATH="/workspace/bin:$PATH"' /root/.bashrc || \
  echo 'export PATH="/workspace/bin:$PATH"' >> /root/.bashrc

export PATH="/workspace/bin:$PATH"
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_LINK_MODE=copy

cd /workspace/cs336-assignment2
uv sync


# Git configurations
git config --global user.name "Andrew Zhang"
git config --global user.email "andrewzhang214@gmail.com"


# Github credentials
mkdir -p /root/.ssh
cp -r /workspace/.ssh/ /root/
chmod 700 /root/.ssh
chmod 600 /root/.ssh/id_ed25519

source /root/.bashrc
