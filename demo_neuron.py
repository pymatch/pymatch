"""A simple single-neuron demo.

The demo creates a fake dataset and shows how to use the match logging feature.
"""

import match
import logging


# Single neuron example
class Neuron(match.nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        # Just one neuron for this model
        self.linear = match.nn.Linear(in_features=num_features, out_features=1)
        self.sigmoid = match.nn.Sigmoid()

    def forward(self, x) -> match.nn.Matrix:
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def main():
    # Set logging level to INFO to display match information
    logging.basicConfig(level=logging.INFO)

    # Number of training examples
    N = 1000

    # Number of input features and output values
    nx = 100
    ny = 1

    # Fake input data
    X = match.randn(N, nx)
    Y = match.randn(N, ny)

    # Create a model
    model = Neuron(num_features=nx)

    # Create a loss function
    mse_loss = match.nn.MSELoss()

    # Compute loss and gradients
    Y_hat = model(X)
    loss = mse_loss(Y_hat, Y)
    loss.backward()


if __name__ == "__main__":
    main()
