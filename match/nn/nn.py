from __future__ import annotations

from math import sqrt, prod
from typing import Optional

import numpy as np

import numpy as np

import match
from match import Tensor, TensorData, use_numpy

from linear import Linear
from module import Module

from copy import deepcopy







class MSELoss(Module):
    """loss = (1/N) * Σ (yhati - yi)^2"""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # Returns a new Tensor
        return ((target - prediction) ** 2).mean()
