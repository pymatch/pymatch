from __future__ import annotations

from logging import info
from math import prod
from random import gauss
from typing import Callable, List, Set
import numpy as np

from icecream import ic

use_numpy = True  # False to use the python implementation of TensorData.

if use_numpy:
    from .tensordata_numpy import TensorData
else:
    from .tensordata import TensorData

LOG = True


class Tensor:
    def __init__(self, data: TensorData, children: tuple = ()) -> None:
        """
        Initialize a Tensor object with given data and optional children, supporting autodifferentiation.

        Args:
            data (TensorData): The data for the tensor, typically a NumPy array or similar structure.
            children (tuple, optional): A tuple of child tensors that this tensor depends on in the
                                        computational graph. Defaults to an empty tuple.
        """
        self.data: TensorData = data
        self.grad: TensorData = TensorData(*self.shape)

        # Backpropagation compute graph
        self._gradient: Callable = lambda: None
        self._children: Set[Tensor] = set(children)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    # TODO: Remove this
    def randn(*shape, generator=lambda: gauss(0, 1)) -> Tensor:
        if isinstance(shape[0], tuple):
            shape = shape[0]

        if use_numpy:
            data = TensorData(
                *shape,
                numpy_data=np.random.default_rng().normal(0, 1, size=shape),
            )
            return Tensor(data=data)

        if not shape:
            return Tensor(TensorData(value=generator()))

        rand_tensordata = TensorData(0)
        rand_tensordata._data = [TensorData(value=generator()) for _ in range(prod(shape))]
        rand_tensordata.reshape_(shape)
        return Tensor(rand_tensordata)

    def backward(self) -> None:
        """Compute all gradients using backpropagation.

        This method performs backpropagation to compute the gradient of each tensor
        in the computational graph. It does this by first performing a topological
        sort of the nodes in the graph and then computing the gradients in reverse
        order of the sort.
        """

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
    
    def dim(self) -> int:
        """Return the dimension of the tensor."""
        return len(self.data.shape)

    @property
    def numel(self) -> int:
        """Return the number of elements in a tensor."""
        return self.data.numel()

    @property
    def shape(self) -> int:
        """Return the shape of the tensor."""
        return self.data.shape

    def sum(self, dim: tuple | int = None, keepdims: bool = False) -> Tensor:
        """
        Return the sum of all values across specified dimensions.

        Args:
            dim (tuple | int, optional): The dimensions over which to sum. If None,
                sums over all dimensions. Defaults to None.
            keepdims (bool, optional): Whether to keep the dimensions of the result
                the same as the input tensor, with reduced dimensions set to size 1.
                Defaults to False.

        Returns:
            Tensor: A new tensor containing the sum of all values across the specified dimensions.

        Note:
            The gradient of the sum operation is handled by assigning a gradient
            function to the resulting tensor, which propagates the gradient to the
            original tensor.
        """
        result = Tensor(self.data.sum(dims=dim, keepdims=keepdims), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of summation. Shape: {self.shape}")
            self.grad += 1 * result.grad

        result._gradient = _gradient
        return result

    def mean(self, dim: tuple | int = None, keepdims: bool = False) -> Tensor:
        """
        Return the mean of all values across specified dimensions.

        Args:
            dim (tuple | int, optional): The dimensions over which to calculate the mean.
                If None, calculates the mean over all dimensions. Defaults to None.
            keepdims (bool, optional): Whether to keep the dimensions of the result
                the same as the input tensor, with reduced dimensions set to size 1.
                Defaults to False.

        Returns:
            Tensor: A new tensor containing the mean of all values across the specified dimensions.

        Note:
            The gradient of the mean operation is handled by assigning a gradient
            function to the resulting tensor, which propagates the gradient to the
            original tensor.
        """
        result = Tensor(self.data.mean(dims=dim, keepdims=keepdims), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of mean. Shape: {self.shape}")
            # Calculate the number of elements in the mean
            num_elements_in_mean = self.numel / result.numel
            self.grad += (1 / num_elements_in_mean) * result.grad

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
            info(f"Gradient of exponentiation. Shape: {self.shape}")
            self.grad += rhs * self.data ** (rhs - 1) * result.grad

        result._gradient = _gradient
        return result

    def __matmul__(self, rhs: Tensor) -> Tensor:
        """Tensor multiplication: self @ rhs."""
        assert isinstance(rhs, Tensor), f"Wrong type: {type(rhs)}"
        lhs_dims, rhs_dims = len(self.shape), len(rhs.shape)
        # Go through each of the cases

        if lhs_dims == 1 and rhs_dims == 1:
            # Return the dot product, has gradient logic encoded
            return (self * rhs).sum()

        result = Tensor(self.data @ rhs.data, children=(self, rhs))
        if lhs_dims >= 2 and rhs_dims >= 2:  # Like normal
            self_permutation = tuple(range(lhs_dims - 2)) + (lhs_dims - 1, lhs_dims - 2)
            rhs_permutation = tuple(range(rhs_dims - 2)) + (rhs_dims - 1, rhs_dims - 2)

            def _gradient() -> None:
                info(f"Gradient of Tensor multiplication (LHS). Shape: {self.shape}")
                g = result.grad @ rhs.data.permute(*rhs_permutation)
                self.grad += g.unbroadcast(*self.shape)
                info(f"Gradient of Tensor multiplication (RHS). Shape: {self.shape}")
                g = self.data.permute(*self_permutation) @ result.grad
                rhs.grad += g.unbroadcast(*rhs.shape)

        elif lhs_dims == 1 and rhs_dims >= 2:
            rhs_permutation = tuple(range(rhs_dims - 2)) + (rhs_dims - 1, rhs_dims - 2)

            def _gradient() -> None:
                info(f"Gradient of Tensor multiplication (LHS). Shape: {self.shape}")
                g = result.grad.reshape(
                    result.shape[:-1] + (1,) + result.shape[-1:]
                ) @ rhs.data.permute(*rhs_permutation)
                # The broadcast will take of the 1 in the second last position
                self.grad += g.unbroadcast(*self.shape)
                info(f"Gradient of Tensor multiplication (RHS). Shape: {self.shape}")
                g = self.data.reshape((self.shape[0], 1)) @ result.grad.reshape(
                    result.shape[:-1] + (1,) + result.shape[-1:]
                )
                rhs.grad += g.unbroadcast(*rhs.shape)

        elif lhs_dims >= 2 and rhs_dims == 1:
            # If the RHS has dim 1, rhs is technically a column vector
            # append a 1 to its dimension (4,) -> (4,1)
            self_permutation = tuple(range(lhs_dims - 2)) + (lhs_dims - 1, lhs_dims - 2)

            def _gradient() -> None:
                info(f"Gradient of Tensor multiplication (LHS). Shape: {self.shape}")
                g = result.grad.reshape(result.shape + (1,)) @ rhs.data.reshape(
                    (1, rhs.shape[0])
                )
                self.grad += g.unbroadcast(*self.shape)
                info(f"Gradient of Tensor multiplication (RHS). Shape: {self.shape}")
                # Add a one to result grad dimension
                g = self.data.permute(*self_permutation) @ result.grad.reshape(
                    result.shape + (1,)
                )
                # Then remove the last "1" dimension it after multiplication
                g.reshape_(g.shape[:-1])
                rhs.grad += g.unbroadcast(*rhs.shape)

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
        """
        Reshape the tensor to the specified shape and return a new Tensor object.

        Args:
            shape (int): The new shape dimensions. Can be passed as separate integers
                or as a tuple of integers.

        Returns:
            Tensor: A new tensor with the specified shape.
        """
        # The reshape method can accept either a variadict or a tuple.
        result: Tensor = Tensor(self.data.reshape(tuple(shape)), children=(self,))

        def _gradient() -> None:
            info(f"Gradient of reshape. Shape: {self.shape}")
            # Permuting the result with the same parameter (reverse)
            self.grad += result.grad.reshape(self.shape)

        result._gradient = _gradient
        return result

    def permute(self, *dims: int) -> Tensor:
        """
        Permute the dimensions of the tensor and return a new Tensor object.

        Args:
            *dims (int): The new order of dimensions. Can be passed as separate integers.

        Returns:
            Tensor: A new tensor with the dimensions permuted according to the specified order.
        """
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

    def var(
        self, dim: tuple | int = None, correction=1, keepdims: bool = False
    ) -> Tensor:
        """
        Calculate the variance over the dimensions specified by `dim`.

        Args:
            dim (tuple | int, optional): The dimension or dimensions over which to
                calculate the variance. Can be a single dimension, a list of dimensions,
                or None to reduce over all dimensions. Defaults to None.
            correction (int, optional): An integer specifying whether to apply a
                correction factor to the variance calculation. Defaults to 1.
            keepdims (bool, optional): Whether to keep the dimensions of the result
                the same as the input tensor, with reduced dimensions set to size 1.
                Defaults to False.

        Returns:
            Tensor: A new tensor containing the variance over the specified dimensions.

        Note:
            The variance is the average squared distance from the mean.
            The gradient of the variance operation is not handled explicitly here. Rather,
            in the other elementary functions that make this one.
        """
        squared_deviation_from_mean = (self - self.mean(dim, keepdims)) ** 2
        # The var is the average squared distance from the mean.
        # TODO: account for degrees of freedom
        var = squared_deviation_from_mean.mean(dim, keepdims)
        return var
