from __future__ import annotations

import numpy as np
import match

from math import sqrt
from match import Tensor
from .module import Module


class Linear(Module):
    """y = x W^T + b"""

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        # Kaiming He initialization
        self.W = Tensor.randn(out_features, in_features) * sqrt((2 / out_features) / 3)
        self.b = Tensor.randn(out_features, 1) * sqrt((2 / out_features) / 3)

    def forward(self, x: Tensor) -> Tensor:
        # Returns a new Tensor
        return x @ self.W.T + self.b.T

    def __repr__(self) -> str:
        return f"A: {self.W}\nb: {self.b}"
