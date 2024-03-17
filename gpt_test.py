import match

use_numpy = False
embed_dim = 10
num_heads = 2
x = match.randn(16, 9, embed_dim, generator=lambda: 1, use_numpy=use_numpy)

multi_head_attn = match.nn.MultiheadAttention(
    embed_dim=embed_dim, num_heads=num_heads, use_numpy=use_numpy
)

if __name__ == "__main__":
    multi_head_attn(x, x, x, None)
