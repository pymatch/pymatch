import itertools
from math import exp, ceil, prod
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from copy import deepcopy
from typing import Callable, Union, Iterable
from collections import Counter
from itertools import zip_longest

Number: type = Union[float, int]


def all_coordinates(shape: tuple):
    possible_indices = [range(dim) for dim in shape]
    return itertools.product(*possible_indices)


def relu(n: Number) -> Number:
    return max(0, n)


def sigmoid(n: Number) -> float:
    return 1.0 / (1.0 + exp(-1 * n))


def leakyrelu(n: Number, beta: Number):
    return n if n > 0 else beta * n


def is_permutation(iter1: Iterable, iter2: Iterable) -> bool:
    """Returns true of iter1 is a permutation of iter2"""
    return Counter(iter1) == Counter(iter2)


def get_common_broadcast_shape(shape1, shape2):
    """Determine the common broadcast shape between two input shapes."""
    shape1 = [1] * (max(len(shape2) - len(shape1), 0)) + list(shape1)
    shape2 = [1] * (max(len(shape1) - len(shape2), 0)) + list(shape2)

    new_shape = []

    for dim1, dim2 in zip_longest(shape1, shape2, fillvalue=1):
        if dim1 == 1:
            new_shape.append(dim2)
        elif dim2 == 1:
            new_shape.append(dim1)
        elif dim1 == dim2:
            new_shape.append(dim1)
        else:
            raise ValueError("Incompatible dimensions for broadcasting")

    return new_shape


def matmul_2d(l1: Iterable, shape1: tuple, l2: Iterable, shape2: tuple) -> tuple:
    """Compute l1 @ l2"""
    # return the new list of data, and the new shape
    if len(shape1) != 2 or len(shape2) != 2 or shape1[-1] != shape2[0]:
        raise ValueError("Inconsistent shapes for 2d matrix multiplication")
    result_shape = (shape1[0], shape2[1])
    result_data = [0] * (result_shape[0] * result_shape[1])

    for i in range(result_shape[0]):
        for j in range(result_shape[1]):
            # Compute the dot product of the i-th row of l1 and the j-th column of l2
            dot_product = 0
            for k in range(shape1[1]):
                # TODO(Sam): Reimplement so it uses ._data directly instead of calling the getitem and setitem methods
                dot_product += l1[i * shape1[1] + k] * l2[k * shape2[1] + j]
            result_data[i * result_shape[1] + j] = dot_product

    return result_shape, result_data
