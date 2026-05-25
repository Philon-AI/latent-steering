"""
Test dataloader output shapes.

Usage:
    python scripts/test_dataloader.py
"""

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config
from data.dataset import LatentSteeringDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_dir = config["data"]["dataset_dir"]
    print(f"Loading dataset from {dataset_dir}")

    dataset = LatentSteeringDataset(config)
    print(f"Dataset splits: {dataset.meta.info["splits"]}")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["dataloader"]["num_workers"],
    )

    batch = next(iter(dataloader))

    import torch
    images = torch.cat([batch["observation.images.cam.past"], batch["observation.images.cam.future"]], dim=1)
    mouse_position_delta = torch.cat([batch["action.mouse.position_delta.past"], batch["action.mouse.position_delta.future"]], dim=1)
    mouse_buttons = torch.cat([batch["action.mouse.buttons.past"], batch["action.mouse.buttons.future"]], dim=1)
    keyboard_buttons = torch.cat([batch["action.keyboard.buttons.past"], batch["action.keyboard.buttons.future"]], dim=1)

    print(f"Dataset: {len(dataset)} samples")
    print(f"  images:           {tuple(images.shape)}  {images.dtype}  min={images.min():.3f}  max={images.max():.3f}")
    print(f"  keyboard_buttons: {tuple(keyboard_buttons.shape)}  {keyboard_buttons.dtype}  min={keyboard_buttons.min():.3f}  max={keyboard_buttons.max():.3f}")
    print(f"  mouse_buttons:    {tuple(mouse_buttons.shape)}  {mouse_buttons.dtype}  min={mouse_buttons.min():.3f}  max={mouse_buttons.max():.3f}")
    print(f"  mouse_position_delta:      {tuple(mouse_position_delta.shape)}  {mouse_position_delta.dtype}  min={mouse_position_delta.min():.3f}  max={mouse_position_delta.max():.3f}")


if __name__ == "__main__":
    main()
