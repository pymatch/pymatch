import match

# x = match.randn(4, 3, 32, 16, generator=lambda: 1)
x = match.randn(16, 3, 32, 32, generator=lambda: 1)

conv2d_layer = match.nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(2,3))

if __name__ == "__main__":
       conv2d_layer(x)
