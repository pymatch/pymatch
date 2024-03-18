import match

use_numpy = True
embed_dim = 10
num_heads = 5
x = match.randn(16, 9, embed_dim, generator=lambda: 1, use_numpy=use_numpy)

decoder = match.nn.TransformerDecoderLayer(embed_dim, num_heads, 128, use_numpy=use_numpy)
gpt2 = match.nn.GPT2(decoder, 4, use_numpy=use_numpy)

if __name__ == "__main__":
    gpt2(x)
