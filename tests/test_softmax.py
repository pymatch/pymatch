import random
import match.nn
import torch.nn
from .base import BaseUnitTest
from match import tensor


class TestSoftmax(BaseUnitTest):

    def test_softmax(self):
        """
        Test the Softmax function with different shapes and dimensions.
        """
        for shape, dim in [
            ((2, 4, 3), -1),  # Last dimension
            ((5, 3), 0),  # First dimension
            ((3, 4, 5), 1),  # Middle dimension
        ]:
            # Generate match and torch tensor pairs.
            match_tensor, torch_tensor = self.generate_tensor_pair(shape=shape)

            # Compare raw outputs.
            match_softmax = match.nn.Softmax(dim=dim)(match_tensor)
            torch_softmax = torch.nn.Softmax(dim=dim)(torch_tensor)
            self.assertTrue(
                self.almost_equal(match_softmax, torch_softmax, check_grad=False)
            )

            # Backpropagation to calculate gradients.
            match_sum = match_softmax.sum()
            torch_sum = torch_softmax.sum()
            match_sum.backward()
            torch_sum.backward()
            self.assertTrue(
                self.almost_equal(match_tensor, torch_tensor, check_grad=True)
            )
