import itertools
from math import exp, ceil, prod
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from copy import deepcopy
from typing import Callable, Union, Iterable
from collections import Counter
from itertools import zip_longest
import numpy as np

Number: type = float | int


def all_coordinates(shape: tuple):
    possible_indices = [range(dim) for dim in shape]
    return itertools.product(*possible_indices)


def relu(n: Number) -> Number:
    return max(0, n)


def sigmoid(n: Union[Number, np.ndarray]) -> float:
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


def dot(l1: Iterable, l2: Iterable):
    """Compute the inner product of two iterable objects, a*b"""
    if len(l1) != len(l2):
        raise ValueError("Size of tensors do not match")
    return sum(i * j for i, j in zip(l1, l2))

def matmul_2d(l1: list, shape1: tuple, l2: list, shape2: tuple) -> tuple:
    """Compute l1 @ l2"""
    # return the new list of data, and the new shape
    if len(shape1) != 2 or len(shape2) != 2 or shape1[-1] != shape2[0]:
        raise ValueError(f"Inconsistent shapes for 2d matrix multiplication: {shape1} @ {shape2}")

    result_shape = (shape1[0], shape2[1])
    result_data = [0] * (result_shape[0] * result_shape[1])

    for i in range(result_shape[0]):
        p = i * shape1[1]
        q = i * result_shape[1]
        for j in range(result_shape[1]):
            # Compute the dot product of the i-th row of l1 and the j-th column of l2
            dot_product = sum(l1[p + k] * l2[k * shape2[1] + j] for k in range(shape1[1]))
            result_data[q + j] = dot_product

    return result_shape, result_data


def get_kernel_position_slices_conv2d(
    tensor_shape: tuple[int],
    kernel_shape: tuple[int],
    stride: tuple,
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
) -> tuple[slice]:
    N = 1
    if len(tensor_shape) == 4:
        N, _, height_in, width_in = tensor_shape
    elif len(tensor_shape) == 3:
        _, height_in, width_in = tensor_shape
    else:
        raise ValueError(
            "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
        )

    # Calculate the positions for each instance in the batch.
    kernel_channels, kernel_height, kernel_width = kernel_shape

    instance_kernel_positions = []
    height_out = len(range(0, height_in - kernel_height + 1, stride[0]))
    width_out = len(range(0, width_in - kernel_width + 1, stride[1]))
    for h in range(0, height_in - kernel_height + 1, stride[0]):
        for c in range(0, width_in - kernel_width + 1, stride[1]):
            instance_kernel_positions.append(
                (
                    slice(0, kernel_channels),  # Number of channels
                    slice(h, h + kernel_height),  # The height of the area
                    slice(c, c + kernel_width),  # The width of the area
                )
            )

#     instance_kernel_positions = [
#         (
#             slice(0, kernel_channels),
#             slice(h, h + kernel_height - 1),
#             slice(c, c + kernel_width - 1),
#         )
#         for h in range(0, height_in - kernel_height + 1, stride[0])
#         for c in range(0, width_in - kernel_width + 1, stride[1])
#         
#     ]

    if len(tensor_shape) == 4:
        instance_kernel_positions = [
            (n,) + position for n in range(N) for position in instance_kernel_positions 
        ]

    return tuple(instance_kernel_positions), height_out, width_out
