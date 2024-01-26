from .tensor import Tensor
from .tensordata import TensorData
from .nn import *
from operator import add, ge, gt, le, lt, mul, pow
from math import exp, ceil, prod
from random import gauss


def randn(*shape, generator=lambda: gauss(0, 1)) -> Tensor:
    if isinstance(shape[0], tuple):
        shape = shape(0)
    if not shape:
        return Tensor(TensorData(value=generator()))

    rand_tensordata = TensorData(0)
    data = [TensorData(value=generator()) for _ in range(prod(shape))]
    rand_tensordata._data = data
    rand_tensordata.reshape_(shape)
    return Tensor(rand_tensordata)

