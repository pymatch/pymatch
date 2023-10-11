import itertools
from math import exp, ceil, prod
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from copy import deepcopy
from typing import Callable, Union, Iterable
from collections import Counter

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
    
