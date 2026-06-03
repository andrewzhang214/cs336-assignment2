#!/bin/bash

clear
clear


# uv run pytest -k test_flash_forward_pass_pytorch

# uv run pytest -k test_flash_forward_pass_triton

# uv run pytest -k test_flash_backward


# Benchmarking

uv run python cs336_systems/bench_pytorch_vs_flash.py \
    --out-jsonl="runs/torch_vs_triton_bench.jsonl" \
    --out-md="runs/torch_vs_triton_bench.md" \
    --dtype="float32"

# uv run python cs336_systems/bench_pytorch_vs_flash.py \
#     --out_jsonl="runs/torch_vs_triton_bench.jsonl" \
#     --out_md="runs/torch_vs_triton_bench.md" \
#     --dtype="bfloat16"