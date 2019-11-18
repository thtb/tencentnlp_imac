import torch
import torch.nn as nn

from tencentnlp.config.config import OutputConfigBase
from tencentnlp.config.config_property import Property
from ..activations import Activation
from ..activations import get_activation


class MLP(nn.Module):
    class Config(OutputConfigBase):
        intermediate: Property = Property(value_type=list, optional=True)
        dropout: Property = Property(value_type=float, default_value=0.0)
        layer_norm: Property = Property(value_type=bool, default_value=False)
        activation: Property = Property(value_type=Activation,
                                        default_value=Activation.NONE)

    def __init__(self, config: Config, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential()
        for dim in config.intermediate:
            self.mlp.append(nn.Linear(input_dim, dim))
            if config.activation != Activation.NONE:
                self.mlp.append(get_activation(config.activation))
            if config.layer_norm:
                self.mlp.append(nn.LayerNorm(dim))
            if config.dropout > 0:
                self.mlp.append(nn.Dropout(config.dropout))
            input_dim = dim
        self.mlp.append(nn.Linear(input_dim, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)
