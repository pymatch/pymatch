from __future__ import annotations

from math import exp
from random import random

from numba import jit

# TODO: fastmath, parallel
# @jit
# @jit(fastmath=True)
# @jit(parallel=True)
# @jit(fastmath=True, parallel=True)


@jit
def randn(rows: int, cols: int) -> list[float]:
    return [random() for _ in range(rows * cols)]


@jit
def sigmoid(x: list[float]) -> list[float]:
    return [1 / (1 + exp(-value)) for value in x]


@jit
def add_values(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


@jit
def add_scalar(m: list[float], s: float) -> list[float]:
    return [x + s for x in m]


@jit
def rdiv_scalar(m: list[float], s: float) -> list[float]:
    return [s / x for x in m]


@jit
def negate_values(m: list[float]) -> list[float]:
    return [-x for x in m]


@jit
def exp_values(m: list[float]) -> list[float]:
    return [exp(x) for x in m]


@jit
def matrix_multiply(
    a: list[float], b: list[float], a_shape: tuple[int, int], b_shape: tuple[int, int]
) -> list[float]:
    result = [0.0] * (a_shape[0] * b_shape[1])

    # TODO: make cache friendly
    for i in range(a_shape[0]):
        for j in range(b_shape[1]):
            values = [
                a[i * a_shape[1] + k] * b[k * b_shape[1] + j] for k in range(a_shape[1])
            ]
            result[i * b_shape[1] + j] = sum(values)

    return result
