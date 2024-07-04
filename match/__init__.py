"""Pytorch Remake
"""

from math import prod
from random import gauss
from .tensor import Tensor, use_numpy
import numpy as np

if use_numpy:
    from .tensordata_numpy import TensorData
else:
    from .tensordata import TensorData


def cat(tensors: list[Tensor], dim=0) -> Tensor:
    """_summary_

    Args:
        tensors (list[Tensor]): _description_
        dim (int, optional): _description_. Defaults to 0.

    Returns:
        Tensor: _description_
    """
    # Store the underlying TensorData objects.
    tensordata_objects = [tensor.data for tensor in tensors]
    # Concatenate the TensorData objects into a single object.
    concatenated_tensordata_objects = TensorData.concatenate(
        tensordatas=tensordata_objects, dim=dim
    )
    return Tensor(data=concatenated_tensordata_objects)


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
