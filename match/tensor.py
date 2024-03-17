from __future__ import annotations

from copy import deepcopy
from logging import info

from .tensordata import TensorData

from operator import add, ge, gt, le, lt, mul, pow
from math import exp, ceil, prod
from random import gauss

import numpy as np


class Tensor(object):
    def __init__(self, data: TensorData, children: tuple = ()) -> None:
        """A Tensor object that tracks computations for computing gradients."""
        # super().__init__()
        self.shape: tuple = data.shape
        self.data: TensorData = data
        self.use_numpy: bool = self.data.use_numpy
        self.grad = TensorData(*self.shape, use_numpy=self.use_numpy)

        # Backpropagation compute graph
        self._gradient = lambda: None
        self._children = set(children)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def randn(*shape, generator=lambda: gauss(0, 1), use_numpy=False) -> Tensor:
        if isinstance(shape[0], tuple):
            shape = shape[0]

        if use_numpy:
            rng = np.random.default_rng(seed=47)
            data = TensorData(
                *shape,
                use_numpy=True,
                numpy_data=rng.random(shape),
            )
            return Tensor(data=data)

        if not shape:
            return Tensor(TensorData(value=generator()))

        rand_tensordata = TensorData(0)
        data = [TensorData(value=generator()) for _ in range(prod(shape))]
        rand_tensordata._data = data
        rand_tensordata.reshape_(shape)
        return Tensor(rand_tensordata)

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

    @property
    def numel(self) -> int:
        return len(self.data._data)

    def sum(self, dim: tuple | int = None, keepdims: bool = False) -> Tensor:
        """Return the sum of all values across dimensions"""
        result = Tensor(self.data.sum(dim, keepdims), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of summation. Shape: {self.shape}")
            self.grad += 1 * result.data * result.grad

        result._gradient = _gradient
        return result

    def mean(self, dim: tuple | int = None, keepdims: bool = False) -> Tensor:
        """Return the mean of all values across both dimensions."""
        result = Tensor(self.data.mean(dim, keepdims), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of mean. Shape: {self.shape}")
            self.grad += 1 * result.data * result.grad

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
            # g is the same shape as self so we don't need to unbroadcast?? Check this.
            # self.grad += rhs * self.data ** (rhs - 1) * result.grad

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
        return Tensor(data=self.data[coords], children=())

    def __setitem__(self, coords, value) -> None:
        self.data[coords] = value

    # Implement gradient for these functions
    # unpermuting and un-reshaping

    def reshape(self, *shape: int) -> Tensor:
        """Helper method to reshape and return a new TensorData object without changing the data"""
        # The reshape method can accept either a variadict or a tuple.
        result: Tensor = Tensor(self.data.reshape(tuple(shape)), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of reshape. Shape: {self.shape}")
            # Permuting the result with the same parameter (reverse)
            self.grad += result.grad.reshape(self.shape)

        result._gradient = _gradient
        return result

    def permute(self, *dims: int) -> Tensor:

        result: Tensor = Tensor(self.data.permute(*dims), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of permute. Shape: {self.shape}")
            # Permuting the result with the same parameter (reverse)
            new_dims = tuple(dims.index(i) for i in range(len(dims)))
            self.grad += result.grad.permute(*new_dims)

        result._gradient = _gradient
        return result

    def exp(self) -> Tensor:
        """Performs element-wise exp"""
        result: Tensor = Tensor(self.data.exp(), children=(self,))

        def _gradient() -> None:
            self.grad += self.data.exp() * result.grad

        result._gradient = _gradient
        return result
