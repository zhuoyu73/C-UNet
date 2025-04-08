# artifactnet.py
# ZS
import torch
from torch import nn
from torch.nn import Module
from .cunet import CUNet


class ArtifactNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = CUNet(
            in_channels=2,   # real + imag
            out_channels=2,  # predict real + imag artifact
            layer_channels=[32, 64, 128, 256, 512],
            attention=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)