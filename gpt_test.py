import match

embed_dim = 64
num_heads = 8
x = match.randn(16, 9, embed_dim, generator=lambda: 1)

decoder = match.nn.TransformerDecoderLayer(embed_dim, num_heads, 128)
gpt2 = match.nn.GPT2(decoder, 4)

if __name__ == "__main__":
    gpt2(x)
