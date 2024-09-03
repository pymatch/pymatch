from __future__ import annotations

from math import sqrt
from typing import Optional
from match import Tensor, TensorData
from .linear import Linear
from .module import Module
from .softmax import Softmax
from copy import deepcopy

class MultiheadAttention(Module):
    """Multi Head Self-Attention"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Assign important variables
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        # Initialize Query, Key, Value tensors for multihead attention.
        self.query_weights = Linear(embed_dim, embed_dim)
        self.key_weights = Linear(embed_dim, embed_dim)
        self.value_weights = Linear(embed_dim, embed_dim)
        self.concat_weights = Linear(embed_dim, embed_dim)

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
        # first, we have to reshape from (batch_size, sequence_length, embedding_dimension) to (batch_size, sequence_length, num_heads, d_head) to avoid splitting the embeddings themselves
        # then we have to permute the dimensions 1,2 so get (batches, num_heads, sequence_length, d_head) so all the values are in the right place
        # we can't just reshape into ( batch_size, sequence_length, self.num_heads, self.d_head) directly
        query_vectors = query_vectors.reshape(
            batch_size, sequence_length, self.num_heads, self.d_head
        ).permute(0, 2, 1, 3)
        key_vectors = key_vectors.reshape(
            batch_size, sequence_length, self.num_heads, self.d_head
        ).permute(0, 2, 1, 3)
        value_vectors = value_vectors.reshape(
            batch_size, sequence_length, self.num_heads, self.d_head
        ).permute(0, 2, 1, 3)
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
                data=TensorData(*normalized_shape, value=1)
            )  # γ
            if bias:
                self.bias: Tensor = Tensor(
                    data=TensorData(*normalized_shape, value=0)
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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = nhead

        self.self_attention = MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward)
        self.first_layer_norm = LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)
        self.second_layer_norm = LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

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

    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.w1 = Linear(d_model, ff_dim)
        self.w2 = Linear(ff_dim, d_model)

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
    """https://www.youtube.com/watch?v=ISNdQcPhsts"""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Module = None,
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
    
