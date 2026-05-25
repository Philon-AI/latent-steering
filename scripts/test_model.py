"""
Test model loading and print summary.

Usage:
    python scripts/test_model.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config
from model.latent_steering import LatentSteering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Initializing model")
    model = LatentSteering(config)

    print(f"Model:")
    print(f"{model}")


if __name__ == "__main__":
    main()
