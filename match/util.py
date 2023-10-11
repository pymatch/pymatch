import itertools
from math import exp, ceil, prod
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from copy import deepcopy
from typing import Callable, Union, Iterable
from collections import Counter
from itertools import zip_longest

Number: type = Union[float, int]


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


    
