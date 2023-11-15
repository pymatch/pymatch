from __future__ import annotations

from copy import deepcopy
from logging import info

from .tensordata import TensorData


class Tensor(object):
    def __init__(self, data: TensorData, children: tuple = ()) -> None:
        """A Tensor object that tracks computations for computing gradients."""
        super().__init__()
        self.shape = data.shape
        self.data = data
        self.grad = TensorData(*self.shape)

        # Backpropagation compute graph
        self._gradient = lambda: None
        self._children = set(children)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def backward(self) -> None:
        """Compute all gradients using backpropagation."""

        sorted_nodes: list[Tensor] = []
        visited: set[Tensor] = set()

        # Sort all elements in the compute graph using a topological ordering (DFS)
        # (Creating a closure here for convenience; capturing sorted_nodes and visited)
        def topological_sort(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    topological_sort(child)
                sorted_nodes.append(node)

        # Perform the topological sort
        topological_sort(self)

        # Initialize all gradients with ones
        self.grad.ones_()

        # Update gradients from output to input (backwards)
        info("Computing gradients using backpropagation.")
        for node in reversed(sorted_nodes):
            node._gradient()

    @property
    def T(self) -> Tensor:
        """Return a transposed version of this Tensor."""
        result = Tensor(self.data.T, children=(self,))

        def _gradient() -> None:
            info(f"Gradient of transpose. Shape: {self.shape}")
            self.grad += result.grad.T

        result._gradient = _gradient
        return result

    def sum(self) -> Tensor:
        """Return the sum of all values across both dimensions."""
        result = Tensor(TensorData(value = self.data.sum()), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of summation. Shape: {self.shape}")
            self.grad += TensorData(*self.shape, value=result.data.item())

        result._gradient = _gradient
        return result

    def mean(self) -> Tensor:
        """Return the mean of all values across both dimensions."""
        result = Tensor(TensorData(value = self.data.mean()), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of mean. Shape: {self.shape}")
            n = len(self.data._data) # Equivalent to prod(self.shape), just cheaper.
            self.grad += TensorData(*self.shape, value=result.data.item() / n)

        result._gradient = _gradient
        return result

    def relu(self) -> Tensor:
        """Element-wise rectified linear unit (ReLU)."""
        result = Tensor(self.data.relu(), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of ReLU. Shape: {self.shape}")
            self.grad += (result.data > 0) * result.grad

        result._gradient = _gradient
        return result

    def sigmoid(self) -> Tensor:
        """Element-wise sigmoid."""
        result = Tensor(self.data.sigmoid(), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of sigmoid. Shape: {self.shape}")
            self.grad += result.data * (1 - result.data) * result.grad

        result._gradient = _gradient
        return result

    def __add__(self, rhs: float | int | Tensor) -> Tensor:
        """Element-wise addition."""
        assert isinstance(rhs, (float, int, Tensor)), f"Wrong type: {type(rhs)}"

        rhs_vals = rhs.data if isinstance(rhs, Tensor) else rhs
        children = (self, rhs) if isinstance(rhs, Tensor) else (self,)
        result = Tensor(self.data + rhs_vals, children=children)

        def _gradient() -> None:
            info(f"Gradient of addition (LHS). Shape: {self.shape}")
            self.grad += result.grad.unbroadcast(*self.shape)
            if isinstance(rhs, Tensor):
                info(f"Gradient of addition (RHS). Shape: {self.shape}")
                rhs.grad += result.grad.unbroadcast(*rhs.shape)

        result._gradient = _gradient
        return result

    def __mul__(self, rhs: float | int | Tensor) -> Tensor:
        """Element-wise multiplication."""
        assert isinstance(rhs, (float, int, Tensor)), f"Wrong type: {type(rhs)}"

        rhs_vals = rhs.data if isinstance(rhs, Tensor) else rhs
        children = (self, rhs) if isinstance(rhs, Tensor) else (self,)
        result = Tensor(self.data * rhs_vals, children=children)

        def _gradient() -> None:
            info(f"Gradient of multiplication (LHS). Shape: {self.shape}")
            self.grad += (rhs_vals * result.grad).unbroadcast(*self.shape)
            if isinstance(rhs, Tensor):
                info(f"Gradient of multiplication (RHS). Shape: {self.shape}")
                rhs.grad += (self.data * result.grad).unbroadcast(*rhs.shape)

        result._gradient = _gradient
        return result

    def __pow__(self, rhs: float | int) -> Tensor:
        """Element-wise exponentiation: self^rhs."""
        assert isinstance(rhs, (float, int)), f"Wrong type: {type(rhs)}"

        result = Tensor(self.data**rhs, children=(self,))

        def _gradient() -> None:
            # rhs_vals will be a number (not Tensor)
            info(f"Gradient of exponentiation. Shape: {self.shape}")
            g = rhs * self.data ** (rhs - 1) * result.grad
            self.grad += g.unbroadcast(*self.shape)

        result._gradient = _gradient
        return result

    def __matmul__(self, rhs: Tensor) -> Tensor:
        """Tensor multiplication: self @ rhs."""
        assert isinstance(rhs, Tensor), f"Wrong type: {type(rhs)}"
        
        result = Tensor(self.data @ rhs.data, children=(self, rhs))

        def _gradient() -> None:
            info(f"Gradient of Tensor multiplication (LHS). Shape: {self.shape}")
            self.grad += result.grad @ rhs.data.T
            info(f"Gradient of Tensor multiplication (RHS). Shape: {self.shape}")
            rhs.grad += self.data.T @ result.grad

        result._gradient = _gradient
        return result

    def __radd__(self, lhs: float | int) -> Tensor:
        """Element-wise addition is commutative: lhs + self."""
        return self + lhs

    def __sub__(self, rhs: float | int | Tensor) -> Tensor:
        """Element-wise subtraction: self - rhs is equivalent to self + (-rhs)."""
        return self + (-rhs)

    def __rsub__(self, lhs: float | int) -> Tensor:
        """Self as RHS in element-wise subtraction: lhs - self."""
        return -self + lhs

    def __rmul__(self, lhs: float | int) -> Tensor:
        """Element-wise multiplication is commutative: lhs * self."""
        return self * lhs

    def __truediv__(self, rhs: float | int) -> Tensor:
        """Element-wise division: self / rhs."""
        return self * rhs**-1

    def __rtruediv__(self, lhs: float | int) -> Tensor:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self**-1

    def __neg__(self) -> Tensor:
        """Element-wise unary negation: -self."""
        return self * -1
    
    def __getitem__(self, coords) -> Tensor:
        # What about children here???
        return Tensor(data=self.data[coords])

    def __setitem__(self,coords, value) -> None:
        self.data[coords] = value