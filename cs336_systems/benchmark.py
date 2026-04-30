"""
Write a script to perform basic end-to-end benchmarking of the forward and backward passes in
your model. Specifically, your script should support the following:
• Given hyperparameters (e.g., number of layers), initialize a model.
• Generate a random batch of data.
• Run w warm-up steps (before you start measuring time), then time the execution of n steps
(either only forward, or both forward and backward passes, depending on an argument). For
timing, you can use the Python timeit module (e.g., either using the timeit function, or
using timeit.default_timer(), which gives you the system’s highest resolution clock, thus
a better default for benchmarking than time.time()).
• Call torch.cuda.synchronize() after each step
"""

import argparse
from contextlib import contextmanager
import time

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.utils import BenchmarkReporter, BenchmarkRow


MODEL_SPECS = {
    "small":    dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium":   dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":    dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":       dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

def create_model(args, device: torch.device):
    # Create model
    model_params = MODEL_SPECS[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=model_params["d_model"],
        num_layers=model_params["num_layers"],
        num_heads=model_params["num_heads"],
        d_ff=model_params["d_ff"],
        rope_theta=args.rope_theta
    ).to(device)
    model.train()
    return model


def create_optimizer(model: BasicsTransformerLM):
    optim = AdamW(params=model.parameters())
    return optim


def create_dummy_data(args, device: torch.device):
    x = torch.randint(
        low=0, high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device
    )
    y = torch.randint(
        low=0, high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device
    )
    return x, y


@contextmanager
def nvtx_range(name: str):
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def measure_forward_backward(model: BasicsTransformerLM, optim: AdamW, x: torch.Tensor, y: torch.Tensor):

    t0 = time.perf_counter()

    with nvtx_range("forward"):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            input=logits.reshape(-1, logits.size(-1)),
            target=y.reshape(-1)
        )
    t1 = time.perf_counter()

    with nvtx_range("backward"):
        optim.zero_grad(set_to_none=True)
        loss.backward()
    t2 = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return t2-t1, t1-t0


def run_benchmark_split(args, device: torch.device):

    model = create_model(args, device)
    optim = create_optimizer(model)
    x, y = create_dummy_data(args, device)
    

    with nvtx_range("benchmark"):
        # Warmup Steps
        with nvtx_range("warmup"):
            for _ in range(args.num_warmup_steps):
                with nvtx_range("step"):
                    measure_forward_backward(model, optim, x, y)
            
        # Run performance
        f_times, b_times = [], []
        with nvtx_range("measure"):
            for _ in range(args.num_measure_steps):
                with nvtx_range("step"):
                    f_time, b_time = measure_forward_backward(model, optim, x, y)
            f_times.append(f_time)
            b_times.append(b_time)
    return f_times, b_times


def emit_row(args, reporter: BenchmarkReporter, mode: str, device: torch.device, avg_time, std_time, f=".3f"):
    row = BenchmarkRow(
        model_size=args.model_size,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        mode=mode,
        num_warmup_steps=args.num_warmup_steps,
        num_measure_steps=args.num_measure_steps,
        mean_ms=format(avg_time, f),
        std_ms=format(std_time, f),
        device=str(device)
    )

    reporter.append(row)


def run_one_setting(args, reporter: BenchmarkReporter, device: torch.device):

    try:
        # Run measurements and collect times
        f_times, b_times = run_benchmark_split(args, device)
        

        # Perform statistical analysis
        f_times = np.array(f_times)
        b_times = np.array(b_times)
        f_avg, f_std = np.mean(f_times), np.std(f_times)
        b_avg, b_std = np.mean(b_times), np.std(b_times)

        # Log the benchmark results with reporter
        emit_row(args, reporter, "forward", device, f_avg, f_std)
        emit_row(args, reporter, "backward", device, b_avg, b_std)

    except torch.OutOfMemoryError:
        print(f"OOM Error for model size: {args.model_size}")
        emit_row(args, reporter, "forward", device, np.nan, np.nan)
        emit_row(args, reporter, "backward", device, np.nan, np.nan)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_sweep(args, reporter: BenchmarkReporter, device: torch.device):
    for model_size in args.sweep_models.split(','):
        for context_length in args.sweep_contexts.split(','):
            args.model_size = model_size
            args.context_length = context_length
            run_one_setting(args, reporter, device)


def run_profile_workload(args, device: torch.device):
    """
    Run profiling on a full training step
    """

    # Forward Step


    # Backward Step


    # Optimizer Step



    pass


def main():

    # Parse arguments from cml
    parser = argparse.ArgumentParser(
                    prog='Benchmark',
                    description='Basic time tracker')
    
    # Model parameters
    parser.add_argument('--model_size', choices=MODEL_SPECS.keys(), default='small')
    parser.add_argument('--vocab_size', type=int, default=10_000)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--rope_theta', type=float, default=10_000)

    # Data parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Measure parameters
    parser.add_argument('--num_warmup_steps', type=int, default=5)
    parser.add_argument('--num_measure_steps', type=int, default=10)
    parser.add_argument('--mode', choices=["inference", "train"], default='train')

    # Sweeper
    parser.add_argument('--sweep', action="store_true")
    parser.add_argument('--sweep_models', type=str, default="small,medium,large,xl,2.7B")

    # Reporter
    parser.add_argument('--out_jsonl', type=str, default=None)
    parser.add_argument('--out_md', type=str, default=None)

    # NVTX
    parser.add_argument('--profile', action="store_true")

    args = parser.parse_args()
 

    torch.manual_seed(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "mps")


    # Reporter
    reporter = BenchmarkReporter(args.out_jsonl, args.out_md)

    if args.profile:
        run_profile_workload(args, device)

    if args.sweep:
        run_sweep(args, reporter, device)

    else:
        run_one_setting(args, reporter, device)

    
    # Render result to md file
    reporter.render_markdown()




 



if __name__ == "__main__":
    main()