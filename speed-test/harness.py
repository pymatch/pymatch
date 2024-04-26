#!/usr/bin/env python

from argparse import ArgumentParser
from time import perf_counter

arg_parser = ArgumentParser("Backend performance test")
arg_parser.add_argument("backend", choices=["numpy", "torch", "matrix", "extension"])
arg_parser.add_argument("name", type=str)
arg_parser.add_argument("--mnist-size", type=int, default=60000)
arg_parser.add_argument("--print-header", action="store_true")
# TODO: debug by comparing with numpy or pytorch
args = arg_parser.parse_args()

if args.backend == "numpy":
    from numpy import exp
    from numpy.random import randn

    def sigmoid(x):
        return 1 / (1 + exp(-x))

elif args.backend == "torch":
    from torch import randn, sigmoid

elif args.backend == "matrix":
    from matrix import randn, sigmoid

elif args.backend == "extension":
    from match.tensorbase import randn, sigmoid

else:
    raise ValueError(f"Unknown backend: {args.backend}")


def main():
    mnist_n = args.mnist_size
    mnist_nx = 28 * 28
    mnist_ny = 10

    layer_size_tests = [
        (mnist_nx, mnist_ny),
        # (mnist_nx, 10, mnist_ny),
        # (mnist_nx, 10, 10, 10, 10, mnist_ny),
        # (mnist_nx, 100, 100, mnist_ny),
        # (mnist_nx, 10000, 10000, mnist_ny),
        # (mnist_nx, 100000, 10000, 1000, mnist_ny),
        # (mnist_nx, 1000000, 1000000, 1000000, mnist_ny),
    ]

    if args.print_header:
        print("Name,TimeCreateMNIST", end=",")
        for ls in layer_size_tests:
            print("TimeCreateLayers", end=",")
            layers = [(mnist_n, mnist_nx)] + [(ni, no) for ni, no in zip(ls, ls[1:])]
            isize = layers[0]
            for osize in layers[1:]:
                print(f"TimeFor({isize[0]}×{isize[1]})@({osize[0]}×{osize[1]})", end=",")
                isize = (isize[0], osize[1])
            print("TimeForward", end=",")
        print()

    name = f"{args.backend}-{args.name}" if args.name else args.backend
    print(name, end=",")

    start = perf_counter()
    fake_mnist = randn(mnist_n, mnist_nx)
    print(perf_counter() - start, end=",")

    for layer_sizes in layer_size_tests:
        start = perf_counter()
        layers = [
            randn(num_in, num_out)
            for num_in, num_out in zip(layer_sizes, layer_sizes[1:])
        ]
        print(perf_counter() - start, end=",")

        start = perf_counter()
        layer_start = start
        x = fake_mnist
        for layer in layers:
            x = sigmoid(x @ layer)
            # x = x @ layer
            print(perf_counter() - layer_start, end=",")
            layer_start = perf_counter()
        print(perf_counter() - start, end=",")
    print()


if __name__ == "__main__":
    main()
