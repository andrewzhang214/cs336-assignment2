
import argparse
import math
from multiprocessing import Manager
import os
from pathlib import Path
import time


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from cs336_systems.utils import DDPCommRow, DDPCommBenchmarkReporter


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()



def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def worker(rank: int, 
           world_size: int,
           backend: str,
           size_bytes_list: list[int],
           warmup: int,
           iters: int,
           out_rows_proxy, # manager.list()
    ):
    try:
        setup(rank, world_size)

        # Use cuda
        use_cuda = (backend == "nccl")
        if use_cuda:
            assert torch.cuda.is_available(), "CUDA is not available"
            assert world_size <= torch.cuda.device_count(), (f"world_size={world_size} > cuda_device_count={torch.cuda.device_count()}")
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        dtype = torch.float32
        elem_size = torch.tensor([], dtype=dtype).element_size() # size in bytes

        # Setup seed
        torch.manual_seed(1234 + rank)

        # Iterate through byte sizes
        for size_bytes in size_bytes_list:
            numel = size_bytes // elem_size

            # Create random tensor for this rank
            x = torch.rand((numel,), dtype=dtype, device=device)

            # Warmup
            for _ in range(warmup):
                sync_if_cuda(device)
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
                sync_if_cuda(device)
            

            # Iters
            times_ms = []
            for _ in range(iters):
                sync_if_cuda(device)
                t0 = time.perf_counter()
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
                t1 = time.perf_counter()
                sync_if_cuda(device)
                times_ms.append((t1-t0)*1e3) # This is the time 
            
            # Gather the times to rank 0
            gathered : list[list[float]] = [None for _ in range(world_size)] # type: ignore
            dist.all_gather_object(gathered, times_ms) # Rank 0 now has all the times for each rank

            if rank == 0:
                # Get the per iter max
                per_iter_max = [max(gathered[r][i] for r in range(world_size)) for i in range(iters)] # For each iteration, which rank took the longest to reduce? That is the best measure of how long all_reduce took

                # Get stats: mean, std, max
                mean = sum(per_iter_max) / len(per_iter_max)
                var = sum((t - mean)**2 for t in per_iter_max) / max(len(per_iter_max)-1, 1)
                std = math.sqrt(var)
                max_ms = max(per_iter_max)

                # Attach to out_rows

                out_rows_proxy.append(
                    dict(
                        backend=backend,
                        device=("cuda" if use_cuda else "cpu"),
                        data_size_bytes=size_bytes,
                        world_size=world_size,
                        warmup_steps=warmup,
                        measure_steps=iters,
                        mean_ms=float(mean),
                        std_ms=float(std),
                        max_ms=float(max_ms),
                    )
                )
        dist.barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        

        


if __name__ == "__main__":


    # Parse arguments from cml
    parser = argparse.ArgumentParser(
                    prog='Benchmark',
                    description='All Reduce Benchmark')

    # Add arguments
    parser.add_argument("--backend", type=str, choices=["gloo", "nccl"], required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="runs/ddp_comm_test")

    args = parser.parse_args()


    # Iterate through data sizes: 1 MB / 10 MB / 100 MB / 1 GB
    size_bytes_list = [
        1 * 1024 * 1024,
        10 * 1024 * 1024,
        100 * 1024 * 1024,
        1024 * 1024 * 1024
    ]

    # Iterate through world sizes (number of processes / workers / worker processes): 2, 4, 6

    out_dir = Path(args.out_dir)

    # Create reporter
    reporter = DDPCommBenchmarkReporter(
        jsonl_path = out_dir / "metrics.jsonl",
        md_path = out_dir / "table.md"
    )

    # Run benchmark

    with Manager() as manager:
        out_rows = manager.list()

        mp.spawn(
            fn=worker,
            args=(
                args.world_size,
                args.backend,
                size_bytes_list,
                args.warmup,
                args.iters,
                out_rows
            ),
            nprocs=args.world_size,
            join=True,
        )

        out_rows = list(out_rows)

        for r in out_rows:
            reporter.append(DDPCommRow(**r))
        
        reporter.render_markdown()
