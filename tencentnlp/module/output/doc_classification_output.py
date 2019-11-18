import torch
import torch.nn as nn

from tencentnlp.config.config import OutputConfigBase
from ..layer.mlp import MLP


class ClassificationOutput(nn.Module):
    class Config(OutputConfigBase):
        linear: MLP.Config = None

    def __init__(self, config: Config, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = MLP(config.linear, input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)
