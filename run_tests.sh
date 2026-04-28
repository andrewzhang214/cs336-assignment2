#!/bin/bash

clear
clear


# Section 1

# Benchmarking

# 1b
# uv run python3 cs336_systems/benchmark.py \
#     --out_jsonl="runs/bench.jsonl" \
#     --out_md="runs/bench.md" \
#     --sweep \

# 1c
# uv run python3 cs336_systems/benchmark.py \
#     --out_jsonl="runs/bench.jsonl" \
#     --out_md="runs/bench.md" \
#     --sweep \
#     --num_warmup_steps=0 \

#1d
uv run python3 cs336_systems/benchmark.py \
    --out_jsonl="runs/bench.jsonl" \
    --out_md="runs/bench.md" \
    --sweep \
    --num_warmup_steps=1 \