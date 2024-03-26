from __future__ import annotations

import os

backend = os.environ.get("MATRIX_BACKEND", "listcomp")

if backend == "listcomp":
    # print(f"Using backend: {backend}")
    from matrix_listcomp import (
        add_scalar,
        add_values,
        exp_values,
        matrix_multiply,
        negate_values,
        rdiv_scalar,
    )
    from matrix_listcomp import randn as randn_base
    from matrix_listcomp import sigmoid as sigmoid_base

else:
    # print(f"Using backend: {backend}")
    from matrix_loops import (
        add_scalar,
        add_values,
        exp_values,
        matrix_multiply,
        negate_values,
        rdiv_scalar,
    )
    from matrix_loops import randn as randn_base
    from matrix_loops import sigmoid as sigmoid_base


def randn(rows: int, cols: int) -> Matrix:
    return Matrix(randn_base(rows, cols), (rows, cols))


def sigmoid(x: Matrix) -> Matrix:
    return Matrix(sigmoid_base(x.data), x.shape)


class Matrix:
    def __init__(self, data: list[float], shape: tuple[int, int]):
        self.data = data[::]
        self.shape = shape

    def __add__(self, other: Matrix | float) -> Matrix:
        if isinstance(other, float):
            result_data = add_scalar(self.data, other)
        else:
            result_data = add_values(self.data, other.data)

        return Matrix(result_data, self.shape)

    def __radd__(self, other: Matrix | float) -> Matrix:
        return self + other

    def __rtruediv__(self, other: float) -> Matrix:
        return Matrix(rdiv_scalar(self.data, other), self.shape)

    def __matmul__(self, other: Matrix) -> Matrix:
        new_shape = (self.shape[0], other.shape[1])
        new_data = matrix_multiply(self.data, other.data, self.shape, other.shape)
        return Matrix(new_data, new_shape)

    def __neg__(self) -> Matrix:
        return Matrix(negate_values(self.data), self.shape)

    def exp(self) -> Matrix:
        return Matrix(exp_values(self.data), self.shape)
