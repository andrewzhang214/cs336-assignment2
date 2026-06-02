
import argparse


import numpy as np
import torch
import triton


from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_triton import FlashAttention2Triton

from cs336_systems.utils import FlashBenchmarkReporter, FlashRow


BATCH_SIZE = 1

def pow_2_list(start: int, end: int) -> list[int]:
    out = [start]
    while out[-1] != end:
        out.append(out[-1] * 2)
    return out


@torch.no_grad()
def create_inputs(
        seq_len: int,
        d: int, 
        dtype: torch.dtype,
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = torch.randn(
        3, BATCH_SIZE, seq_len, d, device=device, dtype=dtype
    )
    return q, k, v


def emit_row(args, reporter, f_ms, b_ms, e2e_ms, impl, status, f=".3f"):
    row = FlashRow(
        seq_length=args.seq,
        d_model=args.d,
        f_ms=format(f_ms, f),
        b_ms=format(b_ms, f),
        e2e_ms=format(e2e_ms, f),
        impl=impl,
        status=status
    )
    reporter.append(row)



def run_benchmark_split(args, q, k, v):
        

    ### Time forward
    def forward_fn():
        if args.impl == "pytorch":
            return scaled_dot_product_attention(q, k, v)
        else:
            return FlashAttention2Triton.apply(q, k, v)
    
    
    f_ms = float(triton.testing.do_bench(forward_fn))

    ### Time backward
    # Need to temp allow gradients
    q_ = q.detach().requires_grad_(True)
    k_ = k.detach().requires_grad_(True)
    v_ = v.detach().requires_grad_(True)

    if args.impl == "pytorch":
        o = scaled_dot_product_attention(q_, k_, v_)
    else:
        o = FlashAttention2Triton.apply(q_, k_, v_)

    do = torch.rand_like(o)

    def bwd_only():
        torch.autograd.grad(o, (q_, k_, v_), grad_outputs=do, retain_graph=True)
    
    b_ms = float(triton.test.do_bench(bwd_only))

    ### Time e2e
    # Need to temp allow gradients
    qx = q.detach().requires_grad_(True)
    kx = k.detach().requires_grad_(True)
    vx = v.detach().requires_grad_(True)

    def e2e():
        if args.impl == "pytorch":
            oy = scaled_dot_product_attention(qx, kx, vx)
        else:
            oy = FlashAttention2Triton.apply(qx, kx, vx)
        
        doy = torch.rand_like(oy)
        torch.autograd.grad(oy, (qx, kx, vx), grad_outputs=doy, retain_graph=False)
    
    e2e_ms = float(triton.test.do_bench(e2e))

    
    return f_ms, b_ms, e2e_ms





def run_one_setting(args, q, k, v, reporter):

    try:
        local_vars = {}
        # Run measurements and collect times
        f_ms, b_ms, e2e_ms = run_benchmark_split(args, q, k, v)

        # Log the benchmark results with reporter
        emit_row(args, reporter, f_ms, b_ms, e2e_ms, args.impl, "ok")

    except torch.OutOfMemoryError:
        print(f"OOM Error for model size: {args.d}")
        emit_row(args, reporter, np.nan, np.nan, np.nan, args.impl, "oom")


def main():

    # Parse Args

    # Write a benchmarking script using triton.testing.do_bench that compares the performance
    # of your (partially) Triton implementation of FlashAttention-2 forward and backward passes with
    # a regular PyTorch implementation (i.e., not using FlashAttention).
    # Specifically, you will report a table that includes latencies for forward, backward, and the endto-end forward-backward pass, for both your Triton and PyTorch implementations. Randomly
    # generate any necessary inputs before you start benchmarking, and run the benchmark on a single
    # H100. Always use batch size 1 and causal masking. Sweep over the cartesian product of sequence
    # lengths of various powers of 2 from 128 up to 65536, embedding dimension sizes of various powers
    # of 2 from 16 up to size 128, and precisions of torch.bfloat16 and torch.float32. You will
    # likely need to adjust tile sizes depending on the input sizes.
    # Deliverable: A table of results comparing your implementation of FlashAttention-2 with the
    # PyTorch implementation, using the settings above and reporting forward, backward, and end-to-end latencies.

    # Parse arguments from cml
    parser = argparse.ArgumentParser(
                    prog='Flash vs Pytorch Attention Benchmark',
                    description='Basic time tracker')
    
    # Data parameter
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16"])

    # Reporter
    parser.add_argument("--out-jsonl", type=str, default=None)
    parser.add_argument("--out-md", type=str, default=None)


    args = parser.parse_args()


    if not torch.cuda.is_available():
        raise RuntimeError("Cuda required")


    torch.manual_seed(args.seed)


    reporter = FlashBenchmarkReporter(args.jsonl_path, args.out_md)

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda")

    SEQ_LENGTHS = pow_2_list(128, 65536)
    D_LENGTHS = pow_2_list(16, 128)

    # Iterate through seq and d lengths
    for seq in SEQ_LENGTHS:
        for d in D_LENGTHS:

            args.seq = seq
            args.d = d

            for impl in ["pytorch", "triton"]:
                args.impl = impl
                q, k, v = create_inputs(seq, d, dtype, device)

                # Run one benchmark
                run_one_setting(args, q, k, v, reporter)


                print(f">>> Benchmark complete for seq_len: {seq} and d: {d}")

    reporter.render_markdown()



if __name__ == "__main__":
    main()