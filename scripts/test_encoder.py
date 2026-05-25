"""
Test encoder loading and print summary.

Usage:
    python scripts/test_encoder.py
"""

import argparse
import sys
from pathlib import Path

import torch

import vjepa2.models_v2_1.vision_transformer as vit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    encoder_config = config["model"]["vision_encoder_past_cfg"]

    encoder_name = encoder_config["model_name"]
    encoder = vit.__dict__[encoder_name](**encoder_config)

    pretrained_path = encoder_config["pretrained_path"]
    pretrained = torch.load(pretrained_path, map_location="cpu")

    pretrained_dict = pretrained["ema_encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            print(f'"{k}" key could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            print(f'"{k}" key is of different shape in model and loaded state dict')
            pretrained_dict[k] = v

    print(f"Encoder:")
    print(f"{encoder}")


if __name__ == "__main__":
    main()
