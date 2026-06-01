#!/bin/bash

clear
clear


uv run pytest -k test_flash_forward_pass_pytorch

# uv run pytest -k test_flash_forward_pass_triton