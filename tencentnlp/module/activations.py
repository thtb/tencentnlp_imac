import math
from enum import Enum

import torch
import torch.nn as nn


class Activation(Enum):
    NONE = "none"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    GELU = "gelu"
    GLU = "glu"


class GeLU(nn.Module):
    """
    Gaussian Error Linear Units (GELUs)
    Reference: https://arxiv.org/pdf/1606.08415.pdf
    # TODO: Delete when PyTorch provide the class
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * (x * x * x))))


def get_activation(name):
    if name == Activation.RELU:
        return nn.ReLU()
    elif name == Activation.LEAKYRELU:
        return nn.LeakyReLU()
    elif name == Activation.TANH:
        return nn.Tanh()
    elif name == Activation.GELU:
        return GeLU()
    elif name == Activation.GLU:
        return nn.GLU(dim=1)
    else:
        raise RuntimeError(f"{name} is not supported")
