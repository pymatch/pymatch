from __future__ import annotations

from math import exp
from random import random

from numba import jit

# TODO: fastmath
# TODO: numba jit (and ability to disable) (NUMBA_DISABLE_JIT)


@jit
def randn(rows: int, cols: int) -> list[float]:
    values = []
    for _ in range(rows * cols):
        values.append(random())
    return values


@jit
def sigmoid(x: list[float]) -> list[float]:
    values = []
    for value in x:
        values.append(1 / (1 + exp(-value)))
    return values


@jit
def add_values(a: list[float], b: list[float]) -> list[float]:
    values = []
    for x, y in zip(a, b):
        values.append(x + y)
    return values


@jit
def add_scalar(m: list[float], s: float) -> list[float]:
    values = []
    for x in m:
        values.append(x + s)
    return values


@jit
def rdiv_scalar(m: list[float], s: float) -> list[float]:
    values = []
    for x in m:
        values.append(s / x)
    return values


@jit
def negate_values(m: list[float]) -> list[float]:
    values = []
    for x in m:
        values.append(-x)
    return values


@jit
def exp_values(m: list[float]) -> list[float]:
    values = []
    for x in m:
        values.append(exp(x))
    return values


@jit
def matrix_multiply(
    a: list[float], b: list[float], a_shape: tuple[int, int], b_shape: tuple[int, int]
) -> list[float]:
    result = [0.0] * (a_shape[0] * b_shape[1])

    for i in range(a_shape[0]):
        for j in range(b_shape[1]):
            for k in range(a_shape[1]):
                result[i * b_shape[1] + j] += a[i * a_shape[1] + k] * b[k * b_shape[1] + j]

    return result
