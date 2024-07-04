from .tensor import Tensor, use_numpy

if use_numpy:
    from .tensordata_numpy import TensorData
else:
    from .tensordata import TensorData

from .nn import *
from math import prod
from random import gauss
import numpy as np


def randn(*shape, generator=lambda: gauss(0, 1)) -> Tensor:
    if isinstance(shape[0], tuple):
        shape = shape[0]

    if use_numpy:
        rng = np.random.default_rng(seed=47)
        data = TensorData(
            *shape,
            numpy_data=rng.random(shape),
        )
        return Tensor(data=data)

    if not shape:
        return Tensor(TensorData(value=generator()))

    rand_tensordata = TensorData(0)
    data = [TensorData(value=generator()) for _ in range(prod(shape))]
    rand_tensordata._data = data
    rand_tensordata.reshape_(shape)
    return Tensor(rand_tensordata)
