from __future__ import annotations

from math import sqrt, prod
from typing import Optional

import numpy as np

import match
from match import Tensor, TensorData
from match.util import get_kernel_position_slices_conv2d

from copy import deepcopy


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

    def __init__(self, in_features, out_features, use_numpy=True) -> None:
        super().__init__()
        # Kaiming He initialization
        self.W = Tensor.randn(out_features, in_features, use_numpy=use_numpy) * sqrt(
            (2 / out_features) / 3
        )
        self.b = Tensor.randn(out_features, 1, use_numpy=use_numpy) * sqrt(
            (2 / out_features) / 3
        )

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
                np_duplicate_values_array = np.append(
                    np_duplicate_values_array, sub_tensordata_row_vector
                )
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


class MultiheadAttention(Module):
    """Multi Head Self-Attention"""

    def __init__(self, embed_dim: int, num_heads: int, use_numpy=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Assign important variables
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        # Initialize Query, Key, Value tensors for multihead attention.
        self.query_weights = Linear(embed_dim, embed_dim, use_numpy=use_numpy)
        self.key_weights = Linear(embed_dim, embed_dim, use_numpy=use_numpy)
        self.value_weights = Linear(embed_dim, embed_dim, use_numpy=use_numpy)
        self.concat_weights = Linear(embed_dim, embed_dim, use_numpy=use_numpy)

        # Define softmax layer
        self.softmax = Softmax(dim=3)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor = None
    ) -> Tensor:
        # The shape of query, key, value are all (batch_size, sequence_length, embedding_dimension).
        # Compute Q, K, V.
        batch_size = query.shape[0]
        sequence_length = query.shape[1]

        query_vectors = self.query_weights(query)
        key_vectors = self.key_weights(query)
        value_vectors = self.value_weights(query)
        print(f"query/key/value vector shape: {query_vectors.shape}")

        # Reshape into many heads
        # Value @ Value_weights = (batch_size, sequence_length, embedding_dimension) @ (num_heads, embedding_dimension, d_head)
        # We then want to reshape this into (batches, num_heads, sequence_length, d_head)
        query_vectors = query_vectors.reshape(
            batch_size, self.num_heads, sequence_length, self.d_head
        )
        key_vectors = key_vectors.reshape(
            batch_size, self.num_heads, sequence_length, self.d_head
        )
        value_vectors = value_vectors.reshape(
            batch_size, self.num_heads, sequence_length, self.d_head
        )
        print(f"query/key/value vector shape (after reshape): {query_vectors.shape}")

        # Apply attention
        key_vectors_transpose = key_vectors.permute(0, 1, 3, 2)
        print(f"key vector shape (after transpose): {key_vectors_transpose.shape}")
        attn_scores = query_vectors @ key_vectors_transpose
        print(f"attn score shape: {attn_scores.shape}")
        attn_scores /= sqrt(self.d_head)
        if attn_mask:
            attn_scores += attn_mask
        attn_pattern = self.softmax(attn_scores)
        attn = attn_pattern @ value_vectors

        # Concat using reshape
        attn = attn.reshape(query.shape[0], query.shape[1], self.embed_dim)
        attn = self.concat_weights(attn)

        return attn


class LayerNorm(Module):
    """The mean and standard-deviation are calculated over the last D dimensions, where
    D is the dimension of normalized_shape. For example, if normalized_shape is (3, 5)
    (a 2-dimensional shape), the mean and standard-deviation are computed over the last
    2 dimensions of the input (i.e. input.mean((-2, -1))).

    γ and β are learnable parameters applied only if `elementwise_affine` is True.

    Description adapted from: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(
        self,
        normalized_shape: tuple[int] | int,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        use_numpy: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.include_bias = bias
        self.dimensions_to_normalize = tuple(
            -1 * dim for dim in range(len(normalized_shape), 0, -1)
        )

        if elementwise_affine:
            self.weight: Tensor = Tensor(
                data=TensorData(*normalized_shape, value=1, use_numpy=use_numpy)
            )  # γ
            if bias:
                self.bias: Tensor = Tensor(
                    data=TensorData(*normalized_shape, value=0, use_numpy=use_numpy)
                )  # β

    def forward(self, x: Tensor):
        # Calculate mean and variance over the last len(normalized_shape) dimensions.
        self.dimensions_to_normalize = tuple(
            dim if dim >= 0 else dim + len(x.shape)
            for dim in self.dimensions_to_normalize
        )

        mean = x.mean(dim=self.dimensions_to_normalize, keepdims=True)
        variance = x.var(dim=self.dimensions_to_normalize, keepdims=True)

        # Normalize the tensor (add eps to prevent divide by 0).
        normalized_tensor = (x - mean) / ((variance + self.eps) ** (0.5))

        # Apply the affine transformation if specified.
        if self.elementwise_affine:
            normalized_tensor = normalized_tensor * self.weight
            if self.include_bias:
                normalized_tensor += self.bias

        return normalized_tensor


# TODO: Fix division shape
class Softmax(Module):
    """Adapted from https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        tensor_exp = x.exp()
        tensor_exp_sum = tensor_exp.sum(dim=self.dim, keepdims=True)
        print(f"tensor_exp shape: {tensor_exp.shape}")
        print(f"tensor_exp_sum shape: {tensor_exp_sum.shape}")
        return tensor_exp / tensor_exp_sum


# TODO: Implement Embedding
class Embedding(Module):
    # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embeddings = Tensor.randn(num_embeddings, embedding_dim)

    def forward(self, x: Tensor): ...


class TransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-05,
        use_numpy=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = nhead
        self.use_numpy = use_numpy

        self.self_attention = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, use_numpy=use_numpy
        )
        self.feed_forward = PositionWiseFeedForward(
            d_model, dim_feedforward, use_numpy=use_numpy
        )
        self.first_layer_norm = LayerNorm(
            normalized_shape=d_model, eps=layer_norm_eps, use_numpy=use_numpy
        )
        self.second_layer_norm = LayerNorm(
            normalized_shape=d_model, eps=layer_norm_eps, use_numpy=use_numpy
        )

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # Apply self-attention mechanism to input x
        attention_result = self.self_attention(x, x, x, mask)

        # Combine original input x with the attention result
        x_plus_attention = x + attention_result

        # Normalize the combined result (normalizing each embedding)
        normalized_x = self.first_layer_norm(x_plus_attention)

        # Feed the normalized result through a feed-forward network
        feed_forward_result = self.feed_forward(normalized_x)

        # Combine the normalized result with the feed-forward output
        combined_result = normalized_x + feed_forward_result

        # Normalize the combined result again to produce the final output
        output = self.second_layer_norm(combined_result)

        return output


class PositionalEncoding(Module): ...


class PositionWiseFeedForward(Module):
    "Implements FFN equation."

    def __init__(self, d_model: int, ff_dim: int, use_numpy: bool = True):
        super().__init__()
        self.w1 = Linear(d_model, ff_dim, use_numpy=use_numpy)
        self.w2 = Linear(ff_dim, d_model, use_numpy=use_numpy)

    def forward(self, x: Tensor) -> Tensor:
        # Apply first linear transformation
        linear1_output = self.w1(x)

        # Apply ReLU activation function to the output of the first transformation
        relu_output = linear1_output.relu()

        # Apply second linear transformation
        final_output = self.w2(relu_output)

        # Return the final output
        return final_output


class GPT2(Module):
    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Module = None,
        use_numpy: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Create num_layers clones of the specified decoder layer
        self.decoder_layers = []
        for _ in range(num_layers):
            self.decoder_layers.append(deepcopy(decoder_layer))

        self.norm = norm

    def forward(self, x: Tensor, mask: Tensor = None):
        # Apply the decoder layers
        print(f"GPT2 Input Tensor Shape: {x.shape}")
        output = x
        for transformer_decoder_layer in self.decoder_layers:
            output = transformer_decoder_layer(output, None)

        # Apply a final normalization layer
        if self.norm:
            output = self.norm(output)

        print(f"GPT2 Output Tensor Shape: {output.shape}")

        return output


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
