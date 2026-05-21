
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable


import numpy as np
import torch



from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.utils import AttentionBenchmarkReporter, AttentionRow


def emit_row(args, reporter, f_avg_time, f_std_time, b_avg_time, b_std_time, mem_before_bwd_mb, f_soln, b_soln, mem_soln, status, f=".3f"):
    row = AttentionRow(
        batch_size=args.batch,
        d_model=args.d_model,
        seq_length=args.seq_length,
        f_avg_ms=format(f_avg_time, f),
        f_std_ms=format(f_std_time, f),
        b_avg_ms=format(b_avg_time, f),
        b_std_ms=format(b_std_time, f),
        f_soln=format(f_soln, f),
        b_soln=format(b_soln, f),
        mem_soln=format(mem_soln, f),
        mem_before_bwd_mb=format(mem_before_bwd_mb, f),
        status=status
    )
    reporter.append(row)


def time_forward(
        fn,
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor
    ) -> float:

    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Run attn
    torch.cuda.synchronize()
    start.record()
    _ = fn(q, k, v, mask=mask)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)



def time_backward(
        fn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor
    ) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Run forward
    out = fn(q, k, v, mask=mask)
    loss = out.sum()

    # Begin recording
    torch.cuda.synchronize()
    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)

def cuda_sync():
    torch.cuda.synchronize()

def time_forward_soln(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor, k: torch. Tensor, v: torch.Tensor,
    iters: int,
    mask
) -> float:
    # Use CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    cuda_sync()
    start.record()
    for _ in range(iters):
        cuda_sync()
        _ = fn(q, k, v, mask)
        cuda_sync()
    end.record()
    cuda_sync()
    return start.elapsed_time(end) / iters


def time_backward_soln(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor, k: torch. Tensor, v: torch.Tensor,
    iters: int,
    mask
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(iters):
        cuda_sync()
        out = fn(q, k, v, mask)
        loss = out.sum()
        cuda_sync()

        start.record()
        loss.backward()
        end.record()
        cuda_sync()
        total_ms += start.elapsed_time(end)

        # clear grads for next iter
        q.grad = None
        k.grad = None
        v.grad = None

    return total_ms / iters






def run_benchmark_split(args, dtype: torch.dtype):

    # Returns list of times
        
    device = torch.device("cuda")
    q = k = v = mask = None

    def attn(q, k, v, mask):
        return scaled_dot_product_attention(q, k, v, mask=mask)

    fn = attn

    if args.compile:
        fn = torch.compile(attn)
    
    try:
        # Initialize K,Q,V of size: [batch, seq, d_model]
        q = torch.randn(args.batch, args.seq_length, args.d_model, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(args.batch, args.seq_length, args.d_model, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(args.batch, args.seq_length, args.d_model, device=device, dtype=dtype, requires_grad=True)
        mask = torch.tril(torch.ones((args.seq_length, args.seq_length), device=device, dtype=torch.bool))

        # Run warmup
        for _ in range(args.num_warmup_steps):
            # Run forward
            time_forward(fn, q, k, v, mask)

            # Run backward
            time_backward(fn, q, k, v, mask)

            q.grad = None
            k.grad = None
            v.grad = None

        # Run measure
        f_times, b_times = [], []
        for _ in range(args.num_measure_steps):
            # Run forward
            f_times.append(time_forward(fn, q, k, v, mask))


        # Collect memory
        mem_before_bwd_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        for _ in range(args.num_measure_steps):
            # Run backward
            b_times.append(time_backward(fn, q, k, v, mask))

            q.grad = None
            k.grad = None
            v.grad = None

        ################################################################
        # Run for solution too
        ################################################################


        # warmup
        for _ in range(args.num_warmup_steps):
            cuda_sync()
            out = fn(q, k, v, mask)
            cuda_sync()
            out.sum().backward()
            cuda_sync()
            q.grad = None
            k.grad = None
            v.grad = None
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_sync()

        iters = args.num_measure_steps

        fwd_soln = time_forward_soln(fn, q, k, v, iters, mask)

        # memory snapshot
        mem_before_bwd_mb_soln = torch.cuda.memory_allocated() / (1024 ** 2)

        bwd_soln = time_backward_soln(fn, q, k, v, iters, mask)
        
        return f_times, b_times, mem_before_bwd_mb, fwd_soln, bwd_soln, mem_before_bwd_mb_soln
    
    except torch.OutOfMemoryError:
        del q, k, v, mask, fn
        torch.cuda.empty_cache()
        raise


def run_one_setting(args, dtype: torch.dtype, reporter):

    # Collect time and memory statistics and record

    try:
        local_vars = {}
        # Run measurements and collect times
        f_times, b_times, mem_before_bwd_mb, fwd_soln, bwd_soln, mem_soln = run_benchmark_split(args, dtype)


        # Perform statistical analysis
        f_times = np.array(f_times)
        b_times = np.array(b_times)
        f_avg, f_std = np.mean(f_times), np.std(f_times)
        b_avg, b_std = np.mean(b_times), np.std(b_times)

        # Log the benchmark results with reporter
        emit_row(args, reporter, f_avg, f_std, b_avg, b_std, mem_before_bwd_mb, fwd_soln, bwd_soln, mem_soln, "ok")

    except torch.OutOfMemoryError:
        print(f"OOM Error for model size: {args.model_size}")
        emit_row(args, reporter, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "oom")


def run_sweep(args, dtype: torch.dtype, reporter: AttentionBenchmarkReporter):
    for d in [int(a) for a in args.sweep_d_model.split(',')]:
        for s in [int(a) for a in args.sweep_seq_length.split(',')]:
            torch.cuda.empty_cache()
            
            # Update args
            args.d_model = d
            args.seq_length = s

            # Run one setting 
            run_one_setting(args, dtype, reporter)
            print(f">>> Finish run for model with size {d} and sequence length {s}")
    
    # Write to md
    reporter.render_markdown()


def main():
    parser = ArgumentParser(
        prog="Pytorch Attention Benchmarker",
        description="Benchmarks basic pytorch attention implementation"
    )

    # K,Q,V dimensions
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=128)


    # Data parameters
    parser.add_argument('--seed', type=int, default=42) # For producing random K,Q,V
    parser.add_argument("--dtype", type=str, default="float32")

    # Measure parameters
    parser.add_argument('--num-warmup-steps', type=int, default=5)
    parser.add_argument('--num-measure-steps', type=int, default=100)
    parser.add_argument('--mode', choices=["inference", "train"], default='train')

    # Sweeper
    parser.add_argument('--sweep', action="store_true")
    parser.add_argument('--sweep-d-model', type=str, default="16,32,64,128")
    parser.add_argument('--sweep-seq-length', type=str, default="256,1024,4096")

    # Reporter
    parser.add_argument('--out-dir', type=str, default="runs/pytorch_attention")

    # Compilation
    parser.add_argument("--compile", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Reporter info
    out_path = Path(args.out_dir)
    reporter = AttentionBenchmarkReporter(
        jsonl_path = out_path / "metrics.jsonl",
        md_path= out_path / "metrics.md"
    )

    if args.sweep:
        run_sweep(args, dtype, reporter)




if __name__ == "__main__":
    main()