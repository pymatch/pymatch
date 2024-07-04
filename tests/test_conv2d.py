import unittest
import torch
from match.nn import Conv2d
from match import tensordata, tensor, randn, use_numpy
import itertools
import numpy as np
import random


def almost_equal(
    match_tensor: tensor.Tensor,
    pytorch_tensor: torch.Tensor,
    check_grad=False,
    debug: bool = False,
) -> bool:
    m = to_tensor(match_tensor, get_grad=check_grad)
    t = pytorch_tensor.grad if check_grad else pytorch_tensor
    if t.ndim == 1:
        m.squeeze_()
    res = torch.allclose(m, t, rtol=1e-02, atol=1e-05)
    if not res or debug:
        print("Match: ", m)
        print("Torch: ", t)
    return res


def to_tensor(
    match_tensor: tensor.Tensor, requires_grad=False, get_grad=False
) -> torch.Tensor:
    match_tensor_data = match_tensor.grad if get_grad else match_tensor.data
    torch_tensor = None
    if use_numpy:
        torch_tensor = torch.from_numpy(np.array(match_tensor_data._numpy_data)).float()

    else:
        if match_tensor_data._data == None:
            torch_tensor = torch.tensor(data=match_tensor_data.item()).float()
        else:
            torch_tensor = torch.zeros(match_tensor_data.shape).float()
            for index in itertools.product(
                *[range(dim) for dim in match_tensor_data.shape]
            ):
                torch_tensor[index] = match_tensor_data[index].item()

    torch_tensor.requires_grad = requires_grad
    return torch_tensor


def mat_and_ten(shape: tuple[int] = None):
    # Generate a random dimension
    if not shape:
        dim = random.randint(2, 5)
        shape = (random.randint(1, 5) for _ in range(dim))
    mat = randn(*shape)
    ten = to_tensor(mat, requires_grad=True)
    return mat, ten


class TestConv2d(unittest.TestCase):
    def test_conv(self):
        # Define parameters
        in_channels = 2
        out_channels = 3
        kernel_height, kernel_width = 3, 3
        stride = 1
        h, w = 5, 5
        # Initialize the same kernel for both match and torch by grabbing the kernel from the match conv2d.
        match_conv2d = Conv2d(in_channels, out_channels, (kernel_height, kernel_width), stride)
        # (#elem, #kernels) -> T -> (#kernels, #elem) -> reshape -> (#kernels, in_channels, kernel_height, kernel_width)
        kernels = match_conv2d._trainable_kernels.T.reshape(out_channels, in_channels, kernel_height, kernel_width)
        mat_x, ten_x = mat_and_ten((in_channels, h, w))
        self.assertTrue(almost_equal(match_conv2d(mat_x), torch.nn.functional.conv2d(ten_x, to_tensor(kernels))))
