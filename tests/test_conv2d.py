from math import prod
import unittest
import torch
from match.nn import Conv2d
from match import tensordata, tensor, randn, use_numpy
import itertools
import numpy as np
import random
from .base import BaseUnitTest


class TestConv2d(BaseUnitTest):
    def test_conv2d_1d_tensor_failure(self):
        match_tensor, _ = self.generate_tensor_pair((5,))
        match_conv2d = Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=(3, 3),
            bias=False,
        )
        with self.assertRaises(ValueError):
            match_conv2d(match_tensor)

    def test_conv2d_initialization_failure(self):
        # Configuration format: (in_channels, out_channels, kernel_size, stride, padding, dilation)
        configurations = [
            (-1, 3, (3, 3), 1, 0, 1),  # Faulty in_channels (cannot be negative)
            (3, -1, (3, 3), 1, 0, 1),  # Faulty out_channels (cannot be negative)
            (3, 3, (0, 3), 1, 0, 1),  # Faulty kernel_size (0 in one dimension)
            (3, 3, (-3, 3), 1, 0, 1),  # Faulty kernel_size (negative dimension)
            (3, 3, (3, 3), 0, 0, 1),  # Faulty stride (0 is invalid)
            (3, 3, (3, 3), -1, 0, 1),  # Faulty stride (negative value)
            (3, 3, (3, 3), 1, -1, 1),  # Faulty padding (negative value)
            (3, 3, (3, 3), 1, 0, 0),  # Faulty dilation (0 is invalid)
            (3, 3, (3, 3), 1, 0, -1),  # Faulty dilation (negative value)
            (3, 3, (3,), -1, 0, 1),  # Faulty kernel_size (tuple of length 1)
            (3, 3, (3, 3), (1,), -1, 1),  # Faulty string (tuple of length 1)
            (3, 3, (3, 3), 1, (1,), 0),  # Faulty padding (tuple of length 1)
            (3, 3, (3, 3), 1, 0, (1,)),  # Faulty dilation (tuple of length 1)
        ]
        for (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        ) in configurations:
            with self.subTest(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ):
                with self.assertRaises(RuntimeError):
                    Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    )

    def test_conv2d_with_batch(self):
        """Gemini Generated, then modified."""
        in_channels = 3
        out_channels = 5
        batch_size = 4
        kernel_size = (3, 3)

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

                # Initialize layers with same weights and biases.
                pytorch_conv2d.weight.data = torch.nn.Parameter(
                    self.to_tensor(
                        match_conv2d._trainable_kernels.T.reshape(
                            out_channels, in_channels, *kernel_size
                        )
                    ),
                    False,
                )
                if bias:
                    pytorch_conv2d.bias.data = torch.nn.Parameter(
                        self.to_tensor(match_conv2d._trainable_bias), False
                    )

                match_tensor, torch_tensor = self.generate_tensor_pair(
                    (batch_size, in_channels, 8, 8)
                )

                match_output = match_conv2d(match_tensor)
                torch_output = pytorch_conv2d(torch_tensor)

                self.assertTrue(self.almost_equal(match_output, torch_output))

    def test_conv2d_no_batch(self):
        """Gemini Generated, then modified."""
        in_channels = 3
        out_channels = 5
        kernel_size = (3, 3)

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

                # Initialize layers with same weights and biases.
                pytorch_conv2d.weight.data = torch.nn.Parameter(
                    self.to_tensor(
                        match_conv2d._trainable_kernels.T.reshape(
                            out_channels, in_channels, *kernel_size
                        )
                    ),
                    False,
                )
                if bias:
                    pytorch_conv2d.bias.data = torch.nn.Parameter(
                        self.to_tensor(match_conv2d._trainable_bias), False
                    )

                match_tensor, torch_tensor = self.generate_tensor_pair(
                    (in_channels, 8, 8)
                )

                match_output = match_conv2d(match_tensor)
                torch_output = pytorch_conv2d(torch_tensor)

                self.assertTrue(self.almost_equal(match_output, torch_output))

    def test_conv2d_gradients(self):
        """Gemini Generated, then modified."""
        in_channels = 2
        out_channels = 2
        kernel_size = (2, 2)

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
                    self.to_tensor(
                        match_conv2d._trainable_kernels.T.reshape(
                            out_channels, in_channels, *kernel_size
                        )
                    ),
                    True,
                )
                if bias:
                    pytorch_conv2d.bias.data = torch.nn.Parameter(
                        self.to_tensor(match_conv2d._trainable_bias), True
                    )

                mat_x, ten_x = self.generate_tensor_pair((1, in_channels, 4, 4))

                match_output = match_conv2d(mat_x)
                pytorch_output = pytorch_conv2d(ten_x)

                # Backpropagation to calculate gradients
                match_sum = match_output.sum()
                pytorch_sum = pytorch_output.sum()
                match_sum.backward()
                pytorch_sum.backward()

                # Compare gradients (weights and input)
                match_conv2d_kernel_grad = tensor.Tensor(
                    data=match_conv2d._trainable_kernels.grad
                )
                pytorch_conv2d_kernel_grad = pytorch_conv2d.weight.grad

                # Check grads directly
                self.assertTrue(
                    self.almost_equal(
                        match_conv2d_kernel_grad.T.reshape(
                            out_channels, in_channels, *kernel_size
                        ),
                        pytorch_conv2d_kernel_grad,
                        check_grad=False,
                    )
                )
                if bias:
                    self.assertTrue(
                        self.almost_equal(
                            match_conv2d._trainable_bias,
                            pytorch_conv2d.bias,
                            check_grad=True,
                        )
                    )

    def test_conv2d_various_shapes_and_strides_no_padding(self):
        """Gemini Generated, then modified."""
        configurations = [
            (2, 3, 5, (3, 3), 1, 0, 1),
            (1, 3, 5, (12, 12), 1, 0, 1),
            (2, 3, 5, (3, 3), 1, 0, (2, 3)),
            (3, 4, 2, (2, 2), 1, 0, 1),
            (1, 3, 5, (3, 2), 4, 0, 1),
            (2, 3, 5, (1, 3), 1, 0, 2),
            (5, 3, 5, (1, 1), 1, 0, 1),
            (2, 3, 5, (5, 4), (3, 2), 0, 2),
        ]

        for (
            N,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        ) in configurations:
            with self.subTest(
                N=N,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ):
                match_conv2d = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
                pytorch_conv2d = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )

                # Initialize with same weights
                pytorch_conv2d.weight.data = torch.nn.Parameter(
                    self.to_tensor(
                        match_conv2d._trainable_kernels.T.reshape(
                            out_channels, in_channels, *kernel_size
                        )
                    ),
                    False,
                )

                mat_x, ten_x = self.generate_tensor_pair((N, in_channels, 12, 12))

                pytorch_output = pytorch_conv2d(ten_x)
                match_output = match_conv2d(mat_x)

                self.assertTrue(self.almost_equal(match_output, pytorch_output))

    # def test_conv2d_various_shapes_and_strides_with_padding(self):
    #     """Gemini Generated, then modified."""
    #     configurations = [
    #         (2, 3, 5, (3, 3), 1, 1, 1),
    #         (1, 3, 5, (12, 12), 1, 2, 1),
    #         (2, 3, 5, (3, 3), 1, 2, (2, 3)),
    #         (3, 4, 2, (2, 2), 1, 1, 1),
    #         (1, 3, 5, (3, 2), 4, 1, 1),
    #         (2, 3, 5, (1, 3), 1, 5, 2),
    #     ]

    #     for (
    #         N,
    #         in_channels,
    #         out_channels,
    #         kernel_size,
    #         stride,
    #         padding,
    #         dilation,
    #     ) in configurations:
    #         with self.subTest(
    #             N=N,
    #             in_channels=in_channels,
    #             out_channels=out_channels,
    #             kernel_size=kernel_size,
    #             stride=stride,
    #             padding=padding,
    #             dilation=dilation,
    #         ):
    #             match_conv2d = Conv2d(
    #                 in_channels,
    #                 out_channels,
    #                 kernel_size,
    #                 stride=stride,
    #                 padding=padding,
    #                 dilation=dilation,
    #                 bias=False,
    #             )
    #             pytorch_conv2d = torch.nn.Conv2d(
    #                 in_channels,
    #                 out_channels,
    #                 kernel_size,
    #                 stride=stride,
    #                 padding=padding,
    #                 dilation=dilation,
    #                 bias=False,
    #             )

    #             # Initialize with same weights
    #             pytorch_conv2d.weight.data = torch.nn.Parameter(
    #                 self.to_tensor(
    #                     match_conv2d._trainable_kernels.T.reshape(
    #                         out_channels, in_channels, *kernel_size
    #                     )
    #                 ),
    #                 False,
    #             )

    #             mat_x, ten_x = self.generate_tensor_pair((N, in_channels, 12, 12))

    #             pytorch_output = pytorch_conv2d(ten_x)
    #             match_output = match_conv2d(mat_x)

    #             self.assertTrue(self.almost_equal(match_output, pytorch_output))
