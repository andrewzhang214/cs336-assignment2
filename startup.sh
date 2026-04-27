#!/bin/bash

curl -Ls https://astral.sh/uv/install.sh | sh

source ~/.bashrc

uv sync

git config --global user.name "Andrew Zhang"
git config --global user.email "andrewzhang214@gmail.com"

mkdir -p /root/.ssh
cp -r /workspace/.ssh/ /root/.ssh/
chmod 700 /root/.ssh
chmod 600 /root/.ssh/id_ed25519