from __future__ import annotations

from math import exp
from random import random

from numba import jit

# No benefit from range


@jit(fastmath=True)
def randn(rows: int, cols: int) -> list[float]:
    values = [0.0] * (rows * cols)
    for i in range(rows * cols):
        values[i] = random()
    return values


@jit(fastmath=True)
def sigmoid(x: list[float]) -> list[float]:
    n = len(x)
    values = [0.0] * n
    for i in range(n):
        values[i] = (1 / (1 + exp(-x[i])))
    return values


@jit(fastmath=True)
def add_values(a: list[float], b: list[float]) -> list[float]:
    n = len(a)
    values = [0.0] * n
    for i in range(n):
        values[i] = a[i] + b[i]
    return values


@jit(fastmath=True)
def add_scalar(m: list[float], s: float) -> list[float]:
    n = len(m)
    values = [0.0] * n
    for i in range(n):
        values[i] = m[i] + s
    return values


@jit(fastmath=True)
def rdiv_scalar(m: list[float], s: float) -> list[float]:
    n = len(m)
    values = [0.0] * n
    for i in range(n):
        values[i] = s / m[i]
    return values


@jit(fastmath=True)
def negate_values(m: list[float]) -> list[float]:
    n = len(m)
    values = [0.0] * n
    for i in range(n):
        values[i] = -m[i]
    return values


@jit(fastmath=True)
def exp_values(m: list[float]) -> list[float]:
    n = len(m)
    values = [0.0] * n
    for i in range(n):
        values[i] = exp(m[i])
    return values


@jit(fastmath=True)
def matrix_multiply(
    a: list[float], b: list[float], a_shape: tuple[int, int], b_shape: tuple[int, int]
) -> list[float]:

    result = [0.0] * (a_shape[0] * b_shape[1])

    for i in range(a_shape[0]):
        for j in range(b_shape[1]):
            for k in range(a_shape[1]):
                result[i * b_shape[1] + j] += (
                    a[i * a_shape[1] + k] * b[k * b_shape[1] + j]
                )

    return result
