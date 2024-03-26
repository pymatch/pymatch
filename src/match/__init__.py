# from math import ceil, exp, prod
# from operator import add, ge, gt, le, lt, mul, pow
# from random import gauss

# from .nn import Module
from .tensor import Tensor

__all__ = [
    "Tensor",
    # 'randn',
    # 'add',
    # 'mul',
    # 'pow',
    # 'exp',
    # 'lt',
    # 'le',
    # 'gt',
    # 'ge',
    # 'conv2d',
    # 'max_pool2d',
    # 'relu',
    # 'cross_entropy',
    # 'softmax',
    # 'log_softmax',
    # 'nll_loss',
    # 'mse_loss',
    # 'sgd',
    # 'adam',
    # 'randn'
]


# def randn(*shape, generator=lambda: gauss(0, 1)) -> Tensor:
#     if isinstance(shape[0], tuple):
#         shape = shape(0)
#     if not shape:
#         return Tensor(TensorBase(value=generator()))

#     rand_tensorbase = TensorBase(0)
#     data = [TensorBase(value=generator()) for _ in range(prod(shape))]
#     rand_tensorbase._data = data
#     rand_tensorbase.reshape_(shape)
#     return Tensor(rand_tensorbase)
