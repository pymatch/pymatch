from match.nn import *

x = match.randn(4,3,16,16)
conv = Conv2d(3, 8, 3,2)

if __name__ == "__main__":
    conv(x)
