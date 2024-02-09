import itertools
from collections import Counter
from copy import deepcopy
from itertools import zip_longest
from math import ceil, exp, prod
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from typing import Callable, Iterable, Union

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
            raise ValueError(f"Incompatible dimensions for broadcasting: {dim1} and {dim2}")

    return new_shape


def dot(l1: Iterable, l2: Iterable):
    """Compute the inner product of two iterable objects, a*b"""
    return sum(i * j for i, j in zip(l1, l2))


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
                dot_product += l1[i * shape1[1] + k] * l2[k * shape2[1] + j]
            result_data[i * result_shape[1] + j] = dot_product

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
        N, channels, height_in, width_in = tensor_shape
    elif len(tensor_shape) == 3:
        channels, height_in, width_in = tensor_shape
    else:
        raise ValueError(
            "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
        )

    # Calculate the positions for each instance in the batch.
    kernel_channels, kernel_height, kernel_width = kernel_shape
    #     height_out = (
    #         height_in + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1
    #     ) / stride[0] - 1

    #     width_out = (
    #         width_in + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1
    #     ) / stride[1] - 1

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
