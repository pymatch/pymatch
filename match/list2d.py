from __future__ import annotations

from math import exp
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from typing import Callable

# Type alias for two-dimensional lists of floats
list2d = list[list[float]]


def sigmoid(z):
    """Sigmoid activation function."""
    return (1 / (1 + exp(-z))) if z > 0 else (1 - 1 / (1 + exp(z)))


class List2D(object):
    """A storage and arithmetic object for matrix data.

    This is an inefficient, but easy-to-understand implementation
    of many matrix multiplication operations. It would be better
    to store this data using a list with a single dimension and
    strides, but that may make it too hard to understand for
    an educational implementation.
    """

    def __init__(self, nrow: int, ncol: int, val: float | list2d = 0.0) -> None:
        """Create a new List2D object to store 2d lists of floats.

        Args:
            nrow (int): number of rows
            ncol (int): number of columns
            val (float | list2d, optional): either an initial value or a list to store
                (no copied). Defaults to 0.0.

        Raises:
            TypeError: raised if val is not a float, int, or list of lists of floats or
                ints
        """
        super().__init__()

        self.nrow = nrow
        self.ncol = ncol
        self.shape = (nrow, ncol)

        if isinstance(val, (float, int)):
            self.vals = [[val] * ncol for _ in range(nrow)]
        elif (
            isinstance(val, list)
            and isinstance(val[0], list)
            and isinstance(val[0][0], (float, int))
        ):
            self.vals = val
        else:
            raise TypeError("Cannot create List2D from", type(val))

    def __repr__(self) -> str:
        return "\n".join(" ".join(f"{val: 2.4f}" for val in row) for row in self.vals)

    def __str__(self) -> str:
        return self.__repr__()

    @staticmethod
    def from_2d_list(vals: list2d) -> List2D:
        """Helper method to quickly create a List2D object from a list of lists."""
        return List2D(len(vals), len(vals[0]), vals)

    @staticmethod
    def randn(nrow: int, ncol: int) -> List2D:
        """Helper method to quickly create a List2D object with random values."""
        vals = [[gauss(0, 1) for _ in range(ncol)] for _ in range(nrow)]
        return List2D.from_2d_list(vals)

    def ones_(self) -> None:
        """Modify all values in the matrix to be 1.0."""
        self.__set(1.0)

    def zeros_(self) -> None:
        """Modify all values in the matrix to be 0.0."""
        self.__set(0.0)

    def sum(self) -> float:
        """Compute the sum of all values in the matrix."""
        return sum(self.vals[i][j] for j in range(self.ncol) for i in range(self.nrow))

    def mean(self) -> float:
        """Compute the mean of all values in the matrix."""
        return self.sum() / (self.nrow * self.ncol)

    def abs(self) -> List2D:
        """Compute the absolute value of each entry of the matrix"""
        vals = [
            [abs(self.vals[i][j]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        return List2D(*self.shape, vals)

    def relu(self) -> List2D:
        """Return a new List2D object with the ReLU of each element."""
        vals = [
            [max(0.0, self.vals[i][j]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        return List2D(*self.shape, vals)

    def leakyrelu(self) -> List2D:
        """Return a new List 2D object with the LeakyReLU of each element."""
        vals = [
            [max(0.1 * self.vals[i][j], self.vals[i][j]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        return List2D(*self.shape, vals)

    def abs(self) -> List2D:
        """Return a new List 2D object with the abs of each element."""
        vals = [
            [abs(self.vals[i][j]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        return List2D(*self.shape, vals)
    
        
    def sigmoid(self) -> List2D:
        """Return a new List2D object with the sigmoid of each element."""
        vals = [
            [sigmoid(self.vals[i][j]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        return List2D(*self.shape, vals)

    def broadcast(self, nrow: int, ncol: int) -> List2D:
        """Return a new List2D broadcast from current shape to (nrow, ncol)."""
        if self.nrow == nrow and self.ncol == ncol:
            return self
        elif self.nrow == 1 and self.ncol == 1:
            return List2D(nrow, ncol, self.vals[0][0])
        elif self.nrow == 1:
            # Copy the row nrow times
            return List2D.from_2d_list(self.vals * nrow)
        else:  # self.ncol == 1
            # Copy the column ncol times (rows are single element lists)
            vals = [row * ncol for row in self.vals]
            return List2D.from_2d_list(vals)

    def unbroadcast(self, nrow: int, ncol: int) -> List2D:
        """Return a new List2D unbroadcast from current shape to (nrow, ncol)."""
        if self.nrow == nrow and self.ncol == ncol:
            return self
        elif nrow == 1 and ncol == 1:
            return List2D(nrow, ncol, self.sum())
        elif nrow == 1:
            # Sum values in each column to collapse to a single row
            return List2D.from_2d_list([list(map(sum, zip(*self.vals)))])
        else:  # self.ncol == 1
            # Sum values in each row to collapse to a single column
            return List2D.from_2d_list([[sum(row)] for row in self.vals])

    @property
    def T(self) -> List2D:
        """Return a new List2D object with the transpose of the matrix."""
        out = List2D(self.ncol, self.nrow)
        for i in range(out.nrow):
            for j in range(out.ncol):
                out.vals[i][j] = self.vals[j][i]
        return out

    def __set(self, val) -> None:
        """Internal method to set all values in the matrix to val."""
        self.vals = [[val] * self.ncol for _ in range(self.nrow)]

    def __binary_op(self, op: Callable, rhs: float | int | List2D) -> List2D:
        """Internal method to perform a binary operation on the matrix.

        This method will automatically broadcast inputs when necessary.
        """

        # Handle the case where rhs is a scalar
        if isinstance(rhs, (float, int)):
            rhs = List2D(*self.shape, rhs)

        # Check for invalid types and shapes
        if isinstance(rhs, List2D):
            if self.nrow != rhs.nrow and self.nrow != 1 and rhs.nrow != 1:
                raise TypeError(f"Wrong shapes: {self.shape} and {rhs.shape}")
            if self.ncol != rhs.ncol and self.ncol != 1 and rhs.ncol != 1:
                raise TypeError(f"Wrong shapes: {self.shape} and {rhs.shape}")
        else:
            raise TypeError(f"Wrong type: {type(rhs)}")

        nrow = max(self.nrow, rhs.nrow)
        ncol = max(self.ncol, rhs.ncol)

        # Create empty output
        out = List2D(nrow, ncol)

        # The broadcast method returns a new object
        lhs = self.broadcast(nrow, ncol)
        rhs = rhs.broadcast(nrow, ncol)

        # Compute the binary operation
        for i in range(out.nrow):
            for j in range(out.ncol):
                out.vals[i][j] = op(lhs.vals[i][j], rhs.vals[i][j])

        return out

    def __add__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise addition: self + rhs."""
        return self.__binary_op(add, rhs)

    def __radd__(self, lhs: float | int | List2D) -> List2D:
        """Element-wise addition is commutative: lhs + self."""
        return self + lhs

    def __sub__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise subtraction: self - rhs."""
        return -rhs + self

    def __rsub__(self, lhs: float | int | List2D) -> List2D:
        """Self as RHS in element-wise subtraction: lhs - self."""
        return -self + lhs

    def __mul__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise multiplication: self * rhs."""
        return self.__binary_op(mul, rhs)

    def __rmul__(self, lhs: float | int | List2D) -> List2D:
        """Element-wise multiplication is commutative: lhs * self."""
        return self * lhs

    def __truediv__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise division: self / rhs."""
        return self * rhs**-1

    def __rtruediv__(self, lhs: float | int | List2D) -> List2D:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self**-1

    def __pow__(self, rhs: float | int) -> List2D:
        """Element-wise exponentiation: self ** rhs."""
        assert isinstance(rhs, (float, int)), "Exponent must be a number."
        exponent = List2D(self.nrow, self.ncol, rhs)
        return self.__binary_op(pow, exponent)

    def __neg__(self) -> List2D:
        """Element-wise unary negation: -self."""
        negative_ones = List2D(self.nrow, self.ncol, -1)
        return self * negative_ones

    def __matmul__(self, rhs: List2D) -> List2D:
        """Two-dimensional matrix multiplication."""
        assert self.ncol == rhs.nrow, "Mismatched shapes in matmul."

        out = List2D(self.nrow, rhs.ncol)

        for i in range(out.nrow):
            for j in range(out.ncol):
                for k in range(self.ncol):
                    out.vals[i][j] += self.vals[i][k] * rhs.vals[k][j]

        return out

    def __gt__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise comparison: self > rhs."""
        return self.__binary_op(gt, rhs)

    def __le__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise comparison: self <= rhs."""
        return self.__binary_op(le, rhs)
