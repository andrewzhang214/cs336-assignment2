#!/bin/bash

clear
clear


# Section 1

# Benchmarking

# 1a
uv run python3 cs336_systems/benchmark.py \
    --out_jsonl="runs/bench.jsonl" \
    --out_md="runs/bench.md" \
    --sweep \