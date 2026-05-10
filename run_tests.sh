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

# #1d
# uv run python3 cs336_systems/benchmark.py \
#     --out_jsonl="runs/bench.jsonl" \
#     --out_md="runs/bench.md" \
#     --sweep \
#     --num_warmup_steps=1 \


# Problem 2 nsys_profile

# 2a

# bash scripts/profile_nsys_systems.sh


# 2d 

# Forward only

# uv run nsys profile \
#     -o "runs/nsys_d_infer_large_s128" \
#     --force-overwrite=true \
#     --trace=cuda,nvtx \
#     --sample=none \
#     --cpuctxsw=none \
#     python cs336_systems/benchmark.py \
#         --model_size="large" --context_length=128 \
#         --profile --profile_mode="inference"

# Full train step
# uv run nsys profile \
#     -o "runs/nsys_d_train_large_s128" \
#     --force-overwrite=true \
#     --trace=cuda,nvtx \
#     --sample=none \
#     --cpuctxsw=none \
#     python cs336_systems/benchmark.py \
#         --model_size="large" --context_length=128 \
#         --profile --profile_mode="train"

# # 2e Profile attention
uv run nsys profile \
    -o "runs/nsys_e_attn_large_s128" \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    python cs336_systems/benchmark.py \
        --model_size="large" --context_length=128 \
        --profile --profile_mode="inference" --nvtx_attention
