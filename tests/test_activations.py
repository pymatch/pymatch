import random
import match.nn
import torch.nn
from .base import BaseUnitTest
from match import tensor


class TestActivations(BaseUnitTest):
    """
    Unit tests for activation functions.
    """

    def test_relu(self):
        """
        Test the ReLU activation function.
        """
        # Generate match and torch tensor pair.
        match_tensor, torch_tensor = self.generate_tensor_pair(shape=(2, 4, 3))

        # Compare raw outputs.
        match_relu = match.nn.ReLU()(match_tensor)
        torch_relu = torch.nn.ReLU()(torch_tensor)
        self.assertTrue(self.almost_equal(match_relu, torch_relu, check_grad=False))

        # Backpropagation to calculate gradients.
        match_sum = match_relu.sum()
        torch_sum = torch_relu.sum()
        match_sum.backward()
        torch_sum.backward()
        self.assertTrue(self.almost_equal(match_tensor, torch_tensor, check_grad=True))

    def test_sigmoid(self):
        """
        Test the Sigmoid activation function.
        """
        # Generate match and torch tensor pair.
        match_tensor, torch_tensor = self.generate_tensor_pair(shape=(2, 4, 3))

        # Compare raw outputs.
        match_sigmoid = match.nn.Sigmoid()(match_tensor)
        torch_sigmoid = torch.nn.Sigmoid()(torch_tensor)
        self.assertTrue(
            self.almost_equal(match_sigmoid, torch_sigmoid, check_grad=False)
        )

        # Backpropagation to calculate gradients.
        match_sum = match_sigmoid.sum()
        torch_sum = torch_sigmoid.sum()
        match_sum.backward()
        torch_sum.backward()
        self.assertTrue(self.almost_equal(match_tensor, torch_tensor, check_grad=True))
