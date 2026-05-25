"""
Convert a distributed checkpoint to a single .pt file.

Usage:
    python convert_checkpoint.py <checkpoint_dir> [--output model.pt]
"""

import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = args.output or str(checkpoint_dir / "model.pt")

    dcp_to_torch_save(str(checkpoint_dir), output_path)
    state = torch.load(output_path, weights_only=True)
    model_state = state["model_state"]
    torch.save(model_state, output_path)
    print(f"Saved model state to {output_path}")


if __name__ == "__main__":
    main()
