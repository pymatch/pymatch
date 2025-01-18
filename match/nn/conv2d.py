from __future__ import annotations
from typing import Optional

import numpy as np
import match

from math import prod
from match import Tensor, TensorData, use_numpy
from .module import Module


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple | int,
        stride: tuple | int = 1,
        padding: tuple | int = 0,
        dilation: tuple | int = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        if self.in_channels < 0:
            raise RuntimeError("in_channels must be non negative")

        self.out_channels: int = out_channels
        if self.out_channels < 0:
            raise RuntimeError("out_channels must be non negative")

        self.__initialize_kernels(kernel_size)

        self.stride: tuple | int = self.__initialize_position_variable(stride)
        if any(s <= 0 for s in self.stride):
            raise RuntimeError(f"stride must be greater than 0, but got {self.stride}")

        self.padding: tuple | int = self.__initialize_position_variable(padding)
        if any(p != 0 for p in self.padding):
            raise NotImplementedError(f"padding is not supported yet. padding must be set to 0.")

        self.dilation: tuple | int = self.__initialize_position_variable(dilation)
        if any(d < 1 for d in self.dilation):
            raise RuntimeError(
                f"dilation must be greater than 0, but got {self.dilation}"
            )

        self.groups: int = groups
        self.padding_mode = padding_mode
        self.__initialize_bias(bias)

    def __initialize_position_variable(self, val: tuple | int):
        val = val if isinstance(val, tuple) else (val, val)
        if len(val) != 2:
            raise RuntimeError(
                "stride, padding, dilation should be a tuple of two ints. the first int is used for the height dimension, and the second int for the width dimension."
            )
        return val

    def __initialize_kernels(self, kernel_size: tuple | int) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if len(kernel_size) != 2:
            raise RuntimeError(
                "kernel_size should be a tuple of two ints. the first int is used for the height dimension, and the second int for the width dimension."
            )
        if any(kernel_dim <= 0 for kernel_dim in kernel_size):
            raise RuntimeError(
                f"kernel size should be greater than zero, but got shape {kernel_size}"
            )
        # Out channels is the number of kernels, so the true kernel is shape (filters, kernel_size, kernel_size)
        self._single_kernel_shape = (self.in_channels,) + kernel_size
        # Each column will be a single kernel, and we have out_channel columns.
        # The number of rows will be the number of elements in each kernel.
        self._trainable_kernels: Tensor = Tensor.randn(
            prod(self._single_kernel_shape), self.out_channels
        )

    def __initialize_bias(self, bias: bool) -> None:
        self.bias: bool = bias
        if bias:
            self._trainable_bias = match.randn(self.out_channels)

    def __create_tensor_with_duplicate_values(
        self, x: Tensor, kernel_positions: list[slice], N: int
    ) -> Tensor:
        # Store all subtensors cooresponding to all kernel positions in an array
        # and flatten them into a single dimension.
        single_kernel_size = prod(self._single_kernel_shape)

        # duplicate_values_array_flattened = [x.data[kernel_position].reshape(single_kernel_size) for kernel_position in kernel_positions]

        duplicate_values_array_flattened = []
        for kernel_position in kernel_positions:
            # Grab subtensor.
            subtensor = x.data[kernel_position]
            # Flatten subtensor into single dimension.
            flattened_subtensor = subtensor.reshape((single_kernel_size,))
            # Add flattened subtensor into array.
            duplicate_values_array_flattened.append(flattened_subtensor)

        # Concatenate all of the Tensors into a single matrix. Each row is a single kernel position.
        tensordata_with_duplicate_values = TensorData.concatenate(
            tensordatas=duplicate_values_array_flattened, dim=0
        )

        if len(x.shape) == 4:
            tensordata_with_duplicate_values.reshape_(
                (
                    N,
                    int(len(kernel_positions) / N),
                    prod(self._single_kernel_shape),
                )  # Divide by N because kernel positions includes those for all N instances in the batch
            )
            print(
                f"Reshaping to {(N,int(len(kernel_positions) / N),prod(self._single_kernel_shape))}"
            )
        else:
            tensordata_with_duplicate_values.reshape_(
                (
                    len(kernel_positions),
                    prod(self._single_kernel_shape),
                )  # Only single batch so N=1
            )
            print(
                f"Reshaping to {(len(kernel_positions),prod(self._single_kernel_shape))}"
            )

        return Tensor(data=tensordata_with_duplicate_values)

    def get_expected_output_dimensions(self, x: Tensor) -> tuple[int]:
        N = None
        if len(x.shape) == 4:
            N, _, height_in, width_in = x.shape
        elif len(x.shape) == 3:
            _, height_in, width_in = x.shape
        else:
            raise ValueError(
                "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
            )

        height_out = int(
            (
                height_in
                + 2 * self.padding[0]
                - self.dilation[0] * (self._single_kernel_shape[1] - 1)
                - 1
            )
            / self.stride[0]
            + 1
        )

        width_out = int(
            (
                width_in
                + 2 * self.padding[1]
                - self.dilation[1] * (self._single_kernel_shape[2] - 1)
                - 1
            )
            / self.stride[1]
            + 1
        )

        if N:
            return N, self.out_channels, height_out, width_out
        else:
            return self.out_channels, height_out, width_out

    def forward(self, x: Tensor) -> Tensor:
        # Assume tensor is shape (N, in_channels, H, W) or (in_channels, H W)
        N = None
        if x.dim() == 4:
            N, _, height_in, width_in = x.shape
        elif x.dim() == 3:
            _, height_in, width_in = x.shape
        else:
            raise ValueError(
                "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
            )

        print(f"Shape of tensor input: {x.shape}")

        # Flatten kernel positions.
        # Each row represents a single placement of the kernel on the input tensor.
        # This flattens the 2D spatial positions into a single row per kernel placement.
        # The resulting shape is: (number of kernel positions in the input tensor, number of elements in the kernel)

        kernel_positions, h, w = self.__get_kernel_position_slices_conv2d(
            height_in, width_in, N
        )

        expected_output_dimensions = self.get_expected_output_dimensions(x)
        height_out, width_out = expected_output_dimensions[-2:]

        print(f"Actual height_out: {h} ... should be {height_out}")
        print(f"Actual width_out: {w} ... should be {width_out}")
        print(
            f"Number of kernel positions: {len(kernel_positions)} ... should be {(N or 1)*height_out*width_out}\n"
        )
        tensor_with_duplicate_values = self.__create_tensor_with_duplicate_values(
            x, kernel_positions, N
        )

        # (9 positions, 32 kernels)
        print(
            f"Multiplying tensor w/ duplicates and kernels ... {tensor_with_duplicate_values.shape} @ {self._trainable_kernels.shape} "
        )
        convolution_tensor: Tensor = (
            tensor_with_duplicate_values @ self._trainable_kernels
        )
        print(
            f"Convolution tensor (after product) shape: {convolution_tensor.shape} ... should be {(N, int(len(kernel_positions)/(N or 1)), self.out_channels)}"
        )
        # Add bias here before reshaping into desired tensor
        ct1 = convolution_tensor
        if self.bias:
            print("Adding bias...")
            ct1 = convolution_tensor + self._trainable_bias

        # (32 kernels, 9 positions)
        # We only want to transpose the last two dimensions...permute!
        if len(convolution_tensor.shape) == 3:
            permute_shape = (0, 2, 1)
        else:
            permute_shape = (1, 0)

        # What about gradient here?
        #  convolution_tensor.data = convolution_tensor.data.permute(*permute_shape)
        #  convolution_tensor.shape = convolution_tensor.data.shape
        ct2 = ct1.permute(*permute_shape)
        print(
            f"Convolution tensor transpose shape: {ct2.shape} ... should be {(N, self.out_channels, int(len(kernel_positions)/(N or 1)))}"
        )

        # do reshape (N, 32, H*W) -> (N, 32, H, W)

        # What about th gradient here?
        if len(x.shape) == 4:
            ct3 = ct2.reshape(N, self.out_channels, height_out, width_out)
        else:
            ct3 = ct2.reshape(self.out_channels, height_out, width_out)

        print(f"Final shape: {ct3.shape} ... should be {expected_output_dimensions}")

        return ct3

    def __get_kernel_position_slices_conv2d(
        self,
        height_in: int,
        width_in: int,
        N: Optional[int] = None,
    ) -> tuple[slice]:

        # Unpack kernel dimensions and convolution parameters into individual parameters.
        kernel_channels, kernel_height, kernel_width = self._single_kernel_shape
        stride_height, stride_width = self.stride
        dilation_height, dilation_width = self.dilation
        padding_height, padding_width = self.padding

        # Calculate effective kernel size with dilation.
        dilated_kernel_height = (kernel_height - 1) * dilation_height + 1
        dilated_kernel_width = (kernel_width - 1) * dilation_width + 1

        # Calculate starting and ending positions for kernel placement.
        starting_height = -padding_height
        starting_width = -padding_width
        ending_height = height_in + padding_height - dilated_kernel_height + 1
        ending_width = width_in + padding_width - dilated_kernel_width + 1

        # Build kernel position slices with padding and dilation.
        instance_kernel_positions = []
        for h in range(starting_height, ending_height, stride_height):
            for w in range(starting_width, ending_width, stride_width):
                instance_kernel_positions.append(
                    (
                        slice(0, kernel_channels),  # Channel slice
                        slice(
                            h, h + dilated_kernel_height, dilation_height
                        ),  # Height slice with dilation
                        slice(
                            w, w + dilated_kernel_width, dilation_width
                        ),  # Width slice with dilation
                    )
                )

        if N:  # if N is not None, the tensor is 4 dimensional
            instance_kernel_positions = [
                (n,) + position
                for n in range(N)
                for position in instance_kernel_positions
            ]

        # Calculate actual output dimensions for verification
        height_out = len(range(starting_height, ending_height, stride_height))
        width_out = len(range(starting_width, ending_width, stride_width))

        return tuple(instance_kernel_positions), height_out, width_out
