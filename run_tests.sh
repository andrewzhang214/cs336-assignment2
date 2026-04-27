#!/bin/bash

git config --global user.name "Andrew Zhang"
git config --global user.email "andrewzhang214@gmail.com"

mkdir -p /root/.ssh
cp -r /workspace/.ssh/ /root/.ssh/
chmod 700 /root/.ssh
chmod 600 /root/.ssh/id_ed25519

clear
clear


# Section 1

# Benchmarking

# 1a
uv run python3 cs336_systems/benchmark.py \
    --out_jsonl="runs/bench.jsonl" \
    --out_md="runs/bench.md" \
    --sweep \