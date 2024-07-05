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
    def test_conv2d_output_with_bias(self):
        """Gemini Generated, then modified."""
        in_channels = 3
        out_channels = 5
        kernel_size = (3, 3)
        bias = True

        match_conv2d = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
        )
        pytorch_conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
        )

        # Initialize both layers with the same weights and biases
        pytorch_conv2d.weight.data = torch.nn.Parameter(
            to_tensor(
                match_conv2d._trainable_kernels.T.reshape(
                    out_channels, in_channels, *kernel_size
                )
            ),
            False,
        )
        pytorch_conv2d.bias.data = torch.nn.Parameter(
            to_tensor(match_conv2d._trainable_bias), False
        )

        mat_x, ten_x = mat_and_ten((in_channels, 8, 8))  # Batch size = 2

        match_output = match_conv2d(mat_x)
        pytorch_output = pytorch_conv2d(ten_x)
        self.assertTrue(almost_equal(match_output, pytorch_output))

    def test_conv2d_output_without_bias(self):
        """Gemini Generated, then modified."""
        in_channels = 3
        out_channels = 5
        kernel_size = (3, 3)
        bias = False

        match_conv2d = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
        )
        pytorch_conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
        )

        # Initialize both layers with the same weights (no bias this time)
        pytorch_conv2d.weight.data = torch.nn.Parameter(
            to_tensor(
                match_conv2d._trainable_kernels.T.reshape(
                    out_channels, in_channels, *kernel_size
                )
            ),
            False,
        )

        mat_x, ten_x = mat_and_ten((2, in_channels, 8, 8))

        match_output = match_conv2d(mat_x)
        pytorch_output = pytorch_conv2d(ten_x)
        self.assertTrue(almost_equal(match_output, pytorch_output))

    # TODO(Sam): Fix gradient functionality.
    def test_conv2d_gradients(self):
        """Gemini Generated, then modified."""
        in_channels = 2
        out_channels = 2
        kernel_size = (2,2)

        for bias in [True, False]:
            with self.subTest(bias=bias):
                match_conv2d = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    bias=bias,
                )
                pytorch_conv2d = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    bias=bias,
                )

                # Initialize layers with same weights and biases (if applicable)
                pytorch_conv2d.weight.data = torch.nn.Parameter(
                    to_tensor(
                        match_conv2d._trainable_kernels.T.reshape(
                            out_channels, in_channels, *kernel_size
                        )
                    ),
                    True,
                )
                if bias:
                    pytorch_conv2d.bias.data = torch.nn.Parameter(
                        to_tensor(match_conv2d._trainable_bias), True
                    )

                mat_x, ten_x = mat_and_ten((1, in_channels, 4, 4))

                match_output = match_conv2d(mat_x)
                pytorch_output = pytorch_conv2d(ten_x)

                # Backpropagation to calculate gradients
                match_output.sum().backward()
                pytorch_output.sum().backward()

                # Compare gradients (weights and input)
                self.assertTrue(
                    almost_equal(
                        match_conv2d._trainable_kernels.T.reshape(
                            out_channels, in_channels, *kernel_size
                        ),
                        pytorch_conv2d.weight,
                        check_grad=True,
                    )
                )
                if bias:
                    self.assertTrue(
                        almost_equal(
                            match_conv2d._trainable_bias,
                            pytorch_conv2d.bias,
                            check_grad=True,
                        )
                    )

    def test_conv2d_various_shapes_and_strides(self):
        """Gemini Generated, then modified."""
        configurations = [
            (1, 2, (3, 3), (1, 1), (0, 0)),
            (3, 4, (5, 5), (2, 2), (1, 1)),
            (1, 1, (1, 1), (1, 1), (0, 0)),  # 1x1 kernel, no padding
            # ... (add more configurations as needed)
        ]

        for in_channels, out_channels, kernel_size, stride, padding in configurations:
            with self.subTest(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ):
                match_conv2d = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )
                pytorch_conv2d = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )

                # Initialize with same weights
                pytorch_conv2d.weight.data = to_tensor(
                    match_conv2d._trainable_kernels.T.reshape(
                        out_channels, in_channels, *kernel_size
                    )
                )

                h, w = random.randint(5, 10), random.randint(5, 10)
                mat_x, ten_x = mat_and_ten((1, in_channels, h, w))  # Batch size = 1

                match_output = match_conv2d(mat_x)
                pytorch_output = pytorch_conv2d(ten_x)

                self.assertTrue(almost_equal(match_output, pytorch_output))

    def test_conv_base(self):
        # Define parameters
        in_channels = 2
        out_channels = 3
        kernel_height, kernel_width = 3, 3
        stride = 1
        h, w = 5, 5
        mat_x, ten_x = mat_and_ten((in_channels, h, w))

        # Initialize the same kernel for both match and torch by grabbing the kernel from the match conv2d.
        match_conv2d = Conv2d(
            in_channels, out_channels, (kernel_height, kernel_width), stride
        )
        # (#elem, #kernels) -> T -> (#kernels, #elem) -> reshape -> (#kernels, in_channels, kernel_height, kernel_width)
        kernels = match_conv2d._trainable_kernels.T.reshape(
            out_channels, in_channels, kernel_height, kernel_width
        )
        self.assertTrue(
            almost_equal(
                match_conv2d(mat_x),
                torch.nn.functional.conv2d(input=ten_x, weight=to_tensor(kernels)),
            )
        )
