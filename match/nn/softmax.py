from __future__ import annotations

from match import Tensor
from module import Module

# TODO: Fix division shape
class Softmax(Module):
    """Adapted from https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        tensor_exp = x.exp()
        tensor_exp_sum = tensor_exp.sum(dim=self.dim, keepdims=True)
        print(f"tensor_exp shape: {tensor_exp.shape}")
        print(f"tensor_exp_sum shape: {tensor_exp_sum.shape}")
        return tensor_exp / tensor_exp_sum