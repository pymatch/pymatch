from __future__ import annotations

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
        use_numpy: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.stride: tuple | int = self.__initialize_position_variable(stride)
        self.padding: tuple | int = self.__initialize_position_variable(padding)
        self.dilation: tuple | int = self.__initialize_position_variable(dilation)
        self.groups: int = groups
        self.padding_mode = "zeros"
        self.use_numpy = use_numpy
        self.__initialize_kernels(kernel_size)
        self.__initialize_bias(bias)

    def __initialize_position_variable(self, val: tuple | int):
        return val if isinstance(val, tuple) else (val, val)

    def __initialize_kernels(self, kernel_size: tuple | int) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        # Out channels is the number of kernels, so the true kernel is shape (filters, kernel_size, kernel_size)
        self._single_kernel_shape = (self.in_channels,) + kernel_size
        # Each column will be a single kernel, and we have out_channel columns.
        # The number of rows will be the number of elements in each kernel.
        self._trainable_kernels: Tensor = Tensor.randn(
            prod(self._single_kernel_shape), self.out_channels
        )

        print(
            f"Shape of a single kernel (#channels, height, width): {self._single_kernel_shape}"
        )
        print(
            f"Shape of a trainable kernel matrix (#elements in each kernel, #kernels): {self._trainable_kernels.shape}"
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

    def forward(self, x: Tensor) -> Tensor:
        # Assume tensor is shape (N, in_channels, H, W) or (in_channels, H W)

        N = 1
        if len(x.shape) == 4:
            N, _, height_in, width_in = x.shape
        elif len(x.shape) == 3:
            _, height_in, width_in = x.shape
        else:
            raise ValueError(
                "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
            )

        print(f"Shape of tensor input: {x.shape}")

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

        # Flatten kernel positions.
        # Each row represents a single placement of the kernel on the input tensor.
        # This flattens the 2D spatial positions into a single row per kernel placement.
        # The resulting shape is: (number of kernel positions in the input tensor, number of elements in the kernel)

        kernel_positions, h, w = self.__get_kernel_position_slices_conv2d(x.shape)
        print(f"Actual height_out: {h} ... should be {height_out}")
        print(f"Actual width_out: {w} ... should be {width_out}")
        print(
            f"Number of kernel positions: {len(kernel_positions)} ... should be {N*height_out*width_out}\n"
        )
        print("The first 5 are...")
        for i in range(5):
            print(kernel_positions[i])
        print()
        print("The last 5 are...")
        for i in range(-5, 0):
            print(kernel_positions[i])
        print()

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
            f"Convolution tensor (after product) shape: {convolution_tensor.shape} ... should be {(N, int(len(kernel_positions)/N), self.out_channels)}"
        )
        # Add bias here before reshaping into desired tensor
        if self.bias:
            print("Adding bias...")
            convolution_tensor += self._trainable_bias
            

        # (32 kernels, 9 positions)
        # We only want to transpose the last two dimensions...permute!
        if len(convolution_tensor.shape) == 3:
            permute_shape = (0, 2, 1)
        else:
            permute_shape = (1, 0)

        # What about gradient here?
        #  convolution_tensor.data = convolution_tensor.data.permute(*permute_shape)
        #  convolution_tensor.shape = convolution_tensor.data.shape
        convolution_tensor = convolution_tensor.permute(*permute_shape)
        print(
            f"Convolution tensor transpose shape: {convolution_tensor.shape} ... should be {(N, self.out_channels, int(len(kernel_positions)/N))}"
        )

        # do reshape (N, 32, H*W) -> (N, 32, H, W)

        # What about th gradient here?
        if len(x.shape) == 4:
            convolution_tensor = convolution_tensor.reshape(
                N, self.out_channels, height_out, width_out
            )
        else:
            convolution_tensor = convolution_tensor.reshape(
                self.out_channels, height_out, width_out
            )

        print(
            f"Final shape: {convolution_tensor.shape} ... should be {(N, self.out_channels, height_out, width_out)}"
        )

        return convolution_tensor

    # TODO: Account for padding and dilation.
    def __get_kernel_position_slices_conv2d(
        self,
        tensor_shape: tuple[int],
    ) -> tuple[slice]:
        if len(tensor_shape) == 4:
            N, _, height_in, width_in = tensor_shape
        elif len(tensor_shape) == 3:
            _, height_in, width_in = tensor_shape
        else:
            raise ValueError(
                "Incorrect shape: Either (N, in_channels, H, W) or (in_channels, H W)"
            )

        # Calculate the positions for each instance in the batch.
        kernel_channels, kernel_height, kernel_width = self._single_kernel_shape

        instance_kernel_positions = []
        for h in range(0, height_in - kernel_height + 1, self.stride[0]):
            for c in range(0, width_in - kernel_width + 1, self.stride[1]):
                instance_kernel_positions.append(
                    (
                        slice(0, kernel_channels),  # Number of channels
                        slice(h, h + kernel_height),  # The height of the area
                        slice(c, c + kernel_width),  # The width of the area
                    )
                )

        # instance_kernel_positions = [
        #     (
        #         slice(0, kernel_channels),  # Number of channels
        #         slice(h, h + kernel_height),  # The height of the area
        #         slice(c, c + kernel_width),  # The width of the area
        #     )
        #     for h in range(0, height_in - kernel_height + 1, self.stride[0])
        #     for c in range(0, width_in - kernel_width + 1, self.stride[1])
        # ]

        if len(tensor_shape) == 4:
            instance_kernel_positions = [
                (n,) + position
                for n in range(N)
                for position in instance_kernel_positions
            ]

        height_out = len(range(0, height_in - kernel_height + 1, self.stride[0]))
        width_out = len(range(0, width_in - kernel_width + 1, self.stride[1]))

        return tuple(instance_kernel_positions), height_out, width_out
