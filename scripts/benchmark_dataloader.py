"""
Benchmark dataloader throughput.

Usage:
    python scripts/benchmark_dataloader.py
"""

import argparse
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import LatentSteeringDataset
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)

    batch_size = args.batch_size or config["training"]["dataloader"]["batch_size"]
    num_workers = args.num_workers or config["training"]["dataloader"]["num_workers"]
    prefetch_factor = args.prefetch_factor or config["training"]["dataloader"]["prefetch_factor"]

    dataset_dir = config["data"]["dataset_dir"]
    print(f"Loading dataset from {dataset_dir}")

    dataset = LatentSteeringDataset(config)
    print(f"Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )

    print(
        f"Benchmarking: batch_size={batch_size}, "
        f"num_workers={num_workers}, "
        f"prefetch_factor={prefetch_factor}"
    )
    print(f"Running {args.num_batches} batches...")

    # Warmup
    it = iter(dataloader)
    for _ in range(min(8, args.num_batches)):
        next(it)

    # Benchmark
    start = time.perf_counter()
    for i in range(args.num_batches):
        print(f"Batch {i + 1}/{args.num_batches}...", end="\r")
        try:
            next(it)
        except StopIteration:
            it = iter(dataloader)
            next(it)
    elapsed = time.perf_counter() - start

    total_samples = args.num_batches * batch_size
    print(f"\n{total_samples} samples in {elapsed:.2f}s")
    print(f"{total_samples / elapsed:.1f} samples/sec")
    print(f"{args.num_batches / elapsed:.1f} batches/sec")
    print(f"{elapsed / args.num_batches * 1000:.1f} ms/batch")


if __name__ == "__main__":
    main()
