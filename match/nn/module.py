from __future__ import annotations

import numpy as np

from match import Tensor

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

            def forward(self, x) -> Tensor:
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x
    """

    def __call__(self, *args) -> Tensor:
        """Enable calling the module like a function."""
        return self.forward(*args)

    def forward(self) -> Tensor:
        """Forward must be implemented by the subclass."""
        raise NotImplementedError("Implement in the subclass.")

    # def parameters(self) -> list[Tensor]:
    #     """Return a list of all parameters in the module."""

    #     # Collect all parameters by searching attributes for Module objects.
    #     params = []
    #     for attr_name in dir(self):
    #         attr = getattr(self, attr_name)
    #         if isinstance(attr, Linear):
    #             params.append(attr.W)
    #             params.append(attr.b)
    #         elif isinstance(attr, Tensor):
    #             params.append(attr)
    #     return params

    def zero_grad(self) -> None:
        """Set gradients for all parameters to zero."""
        for param in self.parameters():
            param.grad.zeros_()
