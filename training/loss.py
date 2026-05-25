import torch
from torch import nn


class WeightedLoss(nn.Module):
    def __init__(self, weights_tensor):
        super().__init__()
        self.register_buffer("weights", weights_tensor)

    def forward(self, raw_loss: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(dtype=raw_loss.dtype).view(1, 1, -1)
        return (raw_loss * weights).mean()
