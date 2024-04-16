from __future__ import annotations

from math import sqrt, prod

import numpy as np

import match
from match import Tensor, TensorData
from match.util import get_kernel_position_slices_conv2d


class Module:
    """Base class for all neural network modules.

    All custom models should subclass this class. Here is an example
    usage of the Module class.

        class MatchNetwork(match.nn.Module):
            def __init__(self, n0, n1, n2) -> None:
                super().__init__()
                self.linear1 = match.nn.Linear(n0, n1)
                self.relu = match.nn.ReLU()
                self.linear2 = match.nn.Linear(n1, n2)
                self.sigmoid = match.nn.Sigmoid()

            def forward(self, x) -> Tensor:
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x
    """

    def __call__(self, *args) -> Tensor:
        """Enable calling the module like a function."""
        return self.forward(*args)

    def forward(self) -> Tensor:
        """Forward must be implemented by the subclass."""
        raise NotImplementedError("Implement in the subclass.")

    def parameters(self) -> list[Tensor]:
        """Return a list of all parameters in the module."""

        # Collect all parameters by searching attributes for Module objects.
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Linear):
                params.append(attr.W)
                params.append(attr.b)
            elif isinstance(attr, Tensor):
                params.append(attr)
        return params

    def zero_grad(self) -> None:
        """Set gradients for all parameters to zero."""
        for param in self.parameters():
            param.grad.zeros_()


class Linear(Module):
    """y = x W^T + b"""

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        # Kaiming He initialization
        self.W = Tensor.randn(out_features, in_features) * sqrt((2 / out_features) / 3)
        self.b = Tensor.randn(out_features, 1) * sqrt((2 / out_features) / 3)

    def forward(self, x: Tensor) -> Tensor:
        # Returns a new Tensor
        return x @ self.W.T + self.b.T

    def __repr__(self) -> str:
        return f"A: {self.W}\nb: {self.b}"


class Conv2d(Module):
    def __create_tensordata_with_duplicate_values(
        self, x, kernel_positions, N
    ) -> TensorData:
        if self.use_numpy:
            np_duplicate_values_array = np.array([])
            printed = False
            for kernel_position_slice in kernel_positions:
                # Grab the sub tensor
                sub_tensordata = x.data[kernel_position_slice]
                # Represent the data as a row vector, we can pass this by value
                sub_tensordata_row_vector = sub_tensordata._numpy_data.flatten()
                np_duplicate_values_array = np.append(np_duplicate_values_array, sub_tensordata_row_vector)
                if not printed:
                    print(
                        f"Length of single subtensor_data: {len(sub_tensordata_row_vector)} ... this should be equal to {prod(self._single_kernel_shape)}"
                    )
                    printed = True

            print(
                f"Total number of elements in duplicate tensor: {len(np_duplicate_values_array)}"
            )

            if len(x.shape) == 4:
                np_duplicate_values_array = np_duplicate_values_array.reshape(
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
                np_duplicate_values_array = np_duplicate_values_array.reshape(
                    (
                        len(kernel_positions),
                        prod(self._single_kernel_shape),
                    )  # Only single batch so N=1
                )
                print(
                    f"Reshaping to {(len(kernel_positions),prod(self._single_kernel_shape))}"
                )
            return TensorData(
                *np_duplicate_values_array.shape,
                use_numpy=True,
                numpy_data=np_duplicate_values_array,
            )

        temp_tensordata_with_duplicate_values = TensorData(0)
        # This assumes that the kernel positions are in sorted order of rows then columns.
        printed = False
        for kernel_position_slice in kernel_positions:
            # Grab the sub tensor
            sub_tensordata = x.data[kernel_position_slice]
            # Represent the data as a row vector, we can pass this by value
            temp_tensordata_with_duplicate_values._data += sub_tensordata._data
            if not printed:
                print(
                    f"Length of single subtensor_data: {len(sub_tensordata._data)} ... this should be equal to {prod(self._single_kernel_shape)}"
                )
                printed = True

        print(
            f"Total number of elements in duplicate tensor: {len(temp_tensordata_with_duplicate_values._data)}"
        )

        if len(x.shape) == 4:
            temp_tensordata_with_duplicate_values.reshape_(
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
            temp_tensordata_with_duplicate_values.reshape_(
                (
                    len(kernel_positions),
                    prod(self._single_kernel_shape),
                )  # Only single batch so N=1
            )
            print(
                f"Reshaping to {(len(kernel_positions),prod(self._single_kernel_shape))}"
            )
        return temp_tensordata_with_duplicate_values

    def forward(self, x: Tensor) -> Tensor:
        # Assume tensor is shape (N, in_channels, H, W) or (in_chnnels, H W)

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

        # Calculate positions
        # Put each position of the kernel in a row,
        # shape should be (#positions of kernel in tensor, num elements in kernel (prod(singlekernel_shape)))

        kernel_positions, h, w = get_kernel_position_slices_conv2d(
            x.shape, self._single_kernel_shape, self.stride
        )
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

        temp_tensordata_with_duplicate_values = (
            self.__create_tensordata_with_duplicate_values(x, kernel_positions, N)
        )

        temp_tensor_with_duplicate_values = Tensor(
            data=temp_tensordata_with_duplicate_values
        )

        # (9 positions, 32 kernels)
        print(
            f"Multiplying tensor w/ duplicates and kernels ... {temp_tensor_with_duplicate_values.shape} @ {self._trainable_kernels.shape} "
        )
        convolution_tensor: Tensor = (
            temp_tensor_with_duplicate_values @ self._trainable_kernels
        )
        print(
            f"Convolution tensor (after product) shape: {convolution_tensor.shape} ... should be {(N, int(len(kernel_positions)/N), self.out_channels)}"
        )

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

        if self.bias:
            convolution_tensor += self._trainable_bias

        return convolution_tensor

    def __initialize_kernels(self, kernel_size: tuple | int) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        # Out channels is the number of kernels, so the true kernel is shape (filters, kernel_size, kernel_size)
        self._single_kernel_shape = (self.in_channels,) + kernel_size
        # Each column will be a single kernel, and we have out_channel columns
        self._trainable_kernels: Tensor = match.randn(
            prod(self._single_kernel_shape), self.out_channels, use_numpy=self.use_numpy
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
            self._trainable_bias = match.randn(
                self.out_channels, use_numpy=self.use_numpy
            )

    def __initialize_position_variable(self, val: tuple | int):
        return val if isinstance(val, tuple) else (val, val)

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


class ReLU(Module):
    """ReLU(x) = max(0, x)"""

    def forward(self, x: Tensor) -> Tensor:
        # Returns a new Tensor
        return x.relu()


class Sigmoid(Module):
    """Sigmoid(x) = 1 / (1 + e^(-x))"""

    def forward(self, x: Tensor) -> Tensor:
        # Returns a new Tensor
        return x.sigmoid()


class MSELoss(Module):
    """loss = (1/N) * Σ (yhati - yi)^2"""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # Returns a new Tensor
        return ((target - prediction) ** 2).mean()
