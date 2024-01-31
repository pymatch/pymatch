import match

use_numpy = True
x = match.randn(16, 3, 32, 32, generator=lambda: 1, use_numpy=use_numpy)

conv2d_layer = match.nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, use_numpy=use_numpy)

if __name__ == "__main__":
       conv2d_layer(x)
