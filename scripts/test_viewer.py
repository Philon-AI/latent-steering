"""
Visualize a single dataset sample in Foxglove.

Usage:
    python scripts/test_viewer.py
    python scripts/test_viewer.py --sample_idx 42 --port 8765
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config
from utils.viewer import LatentSteeringViewer
from data.dataset import LatentSteeringDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = LatentSteeringDataset(config)
    sample = dataset[args.sample_idx]

    viewer = LatentSteeringViewer(sample, fps=dataset.meta.info["fps"], port=args.port)
    viewer.run()


if __name__ == "__main__":
    main()
