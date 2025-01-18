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

    def parameters(self) -> list[Tensor]:
        """Return a list of all parameters in the module.

        Collect all parameters by searching attributes for Module objects via a DFS to account for nested parameters.
        """
        params = []
        seen_ids = set()
        stack = [self]

        while stack:
            current_attr = stack.pop()

            # Iterable attributes (e.g., list, tuple, set) could be nested structures containing other Modules or Tensors.
            # Traverse into the nested structure to search for parameters using a Depth First Search.
            if isinstance(current_attr, (list, tuple, set)):
                stack.extend(current_attr)

            # Add dictionary values to the queue to search.
            elif isinstance(current_attr, dict):
                stack.extend(current_attr.values())

            else:
                # Traverse all attributes of the current object
                for attr_name in dir(current_attr):
                    # Skip private/protected and callable attributes
                    if attr_name.startswith("__"):
                        continue

                    attr = getattr(current_attr, attr_name)

                    # Tensors are leaves of the parameter tree, so append to list of parameters.
                    if isinstance(attr, Tensor):
                        if id(attr) not in seen_ids:
                            params.append(attr)
                            seen_ids.add(id(attr))

                    # Modules are intermediary nodes that could contain other modules or Tensors.
                    # Call paramaters on the Module attribute to seek more Tensor paramaters.
                    elif isinstance(attr, Module):
                        for param in attr.parameters():
                            if id(param) not in seen_ids:
                                params.append(param)
                                seen_ids.add(id(param))

                    elif isinstance(attr, (list, tuple, set)):
                        stack.extend(attr)

                    elif isinstance(attr, dict):
                        stack.extend(attr.values())

        return params

    def zero_grad(self) -> None:
        """Set gradients for all parameters to zero."""
        for param in self.parameters():
            param.grad.zeros_()
