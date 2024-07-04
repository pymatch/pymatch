from __future__ import annotations

import match
import numpy as np

from match import Tensor
from module import Module

class ReLU(Module):
    """ReLU(x) = max(0, x)"""

    def forward(self, x: Tensor) -> Tensor:
        # Returns a new Tensor
        return x.relu()

class Sigmoid(Module):
    """Sigmoid(x) = 1 / (1 + e^(-x))"""

    def forward(self, x: Tensor) -> Tensor:
        # Returns a new Tensor
        return x.sigmoid()