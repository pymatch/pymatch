from __future__ import annotations

from copy import deepcopy
from logging import info

from .tensordata import TensorData


class Tensor(object):
    def __init__(self, data: TensorData, children: tuple = ()) -> None:
        """A matrix object that tracks computations for computing gradients."""
        super().__init__()
        self.nrow = data.nrow
        self.ncol = data.ncol
        self.shape = (self.nrow, self.ncol)

        self.data = data
        self.grad = List2D(self.nrow, self.ncol, 0.0)

        # Backpropagation compute graph
        self._gradient = lambda: None
        self._children = set(children)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def backward(self) -> None:
        """Compute all gradients using backpropagation."""

        sorted_nodes: list[Matrix] = []
        visited: set[Matrix] = set()

        # Sort all elements in the compute graph using a topological ordering (DFS)
        # (Creating a closure here for convenience; capturing sorted_nodes and visited)
        def topological_sort(node: Matrix) -> None:
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
    def T(self) -> Matrix:
        """Return a transposed version of this matrix."""
        result = Matrix(self.data.T, children=(self,))

        def _gradient() -> None:
            info(f"Gradient of transpose. Shape: {self.shape}")
            self.grad += result.grad.T

        result._gradient = _gradient
        return result

    def sum(self) -> Matrix:
        """Return the sum of all values across both dimensions."""
        result = Matrix(List2D(1, 1, self.data.sum()), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of summation. Shape: {self.shape}")
            self.grad += List2D(self.nrow, self.ncol, result.grad.vals[0][0])

        result._gradient = _gradient
        return result

    def mean(self) -> Matrix:
        """Return the mean of all values across both dimensions."""
        result = Matrix(List2D(1, 1, self.data.mean()), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of mean. Shape: {self.shape}")
            n = self.nrow * self.ncol
            self.grad += List2D(self.nrow, self.ncol, result.grad.vals[0][0] / n)

        result._gradient = _gradient
        return result

    def relu(self) -> Matrix:
        """Element-wise rectified linear unit (ReLU)."""
        result = Matrix(self.data.relu(), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of ReLU. Shape: {self.shape}")
            self.grad += (result.data > 0) * result.grad

        result._gradient = _gradient
        return result

    def sigmoid(self) -> Matrix:
        """Element-wise sigmoid."""
        result = Matrix(self.data.sigmoid(), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of sigmoid. Shape: {self.shape}")
            self.grad += result.data * (1 - result.data) * result.grad

        result._gradient = _gradient
        return result

    def __add__(self, rhs: float | int | Matrix) -> Matrix:
        """Element-wise addition."""
        assert isinstance(rhs, (float, int, Matrix)), f"Wrong type: {type(rhs)}"

        rhs_vals = rhs.data if isinstance(rhs, Matrix) else rhs
        children = (self, rhs) if isinstance(rhs, Matrix) else (self,)
        result = Matrix(self.data + rhs_vals, children=children)

        def _gradient() -> None:
            info(f"Gradient of addition (LHS). Shape: {self.shape}")
            self.grad += result.grad.unbroadcast(*self.shape)
            if isinstance(rhs, Matrix):
                info(f"Gradient of addition (RHS). Shape: {self.shape}")
                rhs.grad += result.grad.unbroadcast(*rhs.shape)

        result._gradient = _gradient
        return result

    def __mul__(self, rhs: float | int | Matrix) -> Matrix:
        """Element-wise multiplication."""
        assert isinstance(rhs, (float, int, Matrix)), f"Wrong type: {type(rhs)}"

        rhs_vals = rhs.data if isinstance(rhs, Matrix) else rhs
        children = (self, rhs) if isinstance(rhs, Matrix) else (self,)
        result = Matrix(self.data * rhs_vals, children=children)

        def _gradient() -> None:
            info(f"Gradient of multiplication (LHS). Shape: {self.shape}")
            self.grad += (rhs_vals * result.grad).unbroadcast(*self.shape)
            if isinstance(rhs, Matrix):
                info(f"Gradient of multiplication (RHS). Shape: {self.shape}")
                rhs.grad += (self.data * result.grad).unbroadcast(*rhs.shape)

        result._gradient = _gradient
        return result

    def __pow__(self, rhs: float | int) -> Matrix:
        """Element-wise exponentiation: self^rhs."""
        assert isinstance(rhs, (float, int)), f"Wrong type: {type(rhs)}"

        result = Matrix(self.data**rhs, children=(self,))

        def _gradient() -> None:
            # rhs_vals will be a number (not matrix)
            info(f"Gradient of exponentiation. Shape: {self.shape}")
            g = rhs * self.data ** (rhs - 1) * result.grad
            self.grad += g.unbroadcast(*self.shape)

        result._gradient = _gradient
        return result

    def __matmul__(self, rhs: Matrix) -> Matrix:
        """Matrix multiplication: self @ rhs."""
        assert isinstance(rhs, Matrix), f"Wrong type: {type(rhs)}"
        assert self.ncol == rhs.nrow, f"Wrong shapes: {self.shape} and {rhs.shape}"

        result = Matrix(self.data @ rhs.data, children=(self, rhs))

        def _gradient() -> None:
            info(f"Gradient of matrix multiplication (LHS). Shape: {self.shape}")
            self.grad += result.grad @ rhs.data.T
            info(f"Gradient of matrix multiplication (RHS). Shape: {self.shape}")
            rhs.grad += self.data.T @ result.grad

        result._gradient = _gradient
        return result

    def __radd__(self, lhs: float | int) -> Matrix:
        """Element-wise addition is commutative: lhs + self."""
        return self + lhs

    def __sub__(self, rhs: float | int | Matrix) -> Matrix:
        """Element-wise subtraction: self - rhs is equivalent to self + (-rhs)."""
        return self + (-rhs)

    def __rsub__(self, lhs: float | int) -> Matrix:
        """Self as RHS in element-wise subtraction: lhs - self."""
        return -self + lhs

    def __rmul__(self, lhs: float | int) -> Matrix:
        """Element-wise multiplication is commutative: lhs * self."""
        return self * lhs

    def __truediv__(self, rhs: float | int) -> Matrix:
        """Element-wise division: self / rhs."""
        return self * rhs**-1

    def __rtruediv__(self, lhs: float | int) -> Matrix:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self**-1

    def __neg__(self) -> Matrix:
        """Element-wise unary negation: -self."""
        return self * -1