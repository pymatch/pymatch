"""
TODO(AJC): things for me to add in the future.

- Conv2d, MaxPool2d, AdaptiveMaxPool2d, Flatten
- BatchNorm2d
- Dropout
- RNN, LSTM
- Embeddings
- Transformer
- BCELoss, CrossEntropyLoss
"""


from __future__ import annotations

from math import sqrt

import match
from match import Matrix


class Module:
    """Base class for all neural network modules.

    All custom models should subclass this class. Here is an example
    usage of the Module class.

        class MatchNetwork(match.nn.Module):
            def __init__(self, n0, n1, n2) -> None:
                super().__init__()
                self.linear1 = match.nn.Linear(n0, n1)
                self.relu = match.nn.ReLU()
                self.linear2 = match.nn.Linear(n1, n2)
                self.sigmoid = match.nn.Sigmoid()

            def forward(self, x) -> Matrix:
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x
    """

    def __call__(self, *args) -> Matrix:
        """Enable calling the module like a function."""
        return self.forward(*args)

    def forward(self) -> Matrix:
        """Forward must be implemented by the subclass."""
        raise NotImplementedError("Implement in the subclass.")

    def parameters(self) -> list[Matrix]:
        """Return a list of all parameters in the module."""

        # Collect all parameters by searching attributes for Module objects.
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Linear):
                params.append(attr.W)
                params.append(attr.b)
            elif isinstance(attr, Matrix):
                params.append(attr)
        return params

    def zero_grad(self) -> None:
        """Set gradients for all parameters to zero."""
        for param in self.parameters():
            param.grad.zeros_()


class Linear(Module):
    """y = x W^T + b"""

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        # Kaiming He initialization
        self.W = match.randn(out_features, in_features) * sqrt((2 / out_features) / 3)
        self.b = match.randn(out_features, 1) * sqrt((2 / out_features) / 3)

    def forward(self, x: Matrix) -> Matrix:
        # Returns a new Matrix
        return x @ self.W.T + self.b.T

    def __repr__(self) -> str:
        return f"A: {self.W}\nb: {self.b}"


class ReLU(Module):
    """ReLU(x) = max(0, x)"""

    def forward(self, x: Matrix) -> Matrix:
        # Returns a new Matrix
        return x.relu()


class LeakyReLU(Module):
    """LeakyReLU(x) = max(0,x)+0.1∗min(0,x) """
    def forward(self, x: Matrix) -> Matrix:
        # Returns a new Matrix
        return x.leakyReLU()

class Sigmoid(Module):
    """Sigmoid(x) = 1 / (1 + e^(-x))"""

    def forward(self, x: Matrix) -> Matrix:
        # Returns a new Matrix
        return x.sigmoid()


class MSELoss(Module):
    """loss = (1/N) * Σ (yhati - yi)^2"""

    def forward(self, prediction: Matrix, target: Matrix) -> Matrix:
        # Returns a new Matrix
        return ((target - prediction) ** 2).mean()

class MAELoss(Module):
    """loss = (1/N0 * Σ |yhati - yi|"""

    def forward(self, prediction: Matrix, target: Matrix) -> Matrix:
        # Returns a new Matrix
        return ((target - prediction).abs()).mean()
