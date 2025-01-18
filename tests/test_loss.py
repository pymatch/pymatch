import random
import match.nn
import torch.nn
from .base import BaseUnitTest
from match import tensor

class TestLossFunctions(BaseUnitTest):
    """
    Unit tests for loss functions. (Modified output from ChatGPT-4o)
    """

    def test_mse_loss(self):
        """
        Test the MSELoss function.
        """
        # Generate match and torch tensor pairs for prediction and target.
        match_prediction, torch_prediction = self.generate_tensor_pair(shape=(2, 4, 3))
        match_target, torch_target = self.generate_tensor_pair(shape=(2, 4, 3))

        # Compare raw outputs.
        match_mse_loss = match.nn.MSELoss()(match_prediction, match_target)
        torch_mse_loss = torch.nn.MSELoss()(torch_prediction, torch_target)
        self.assertTrue(
            self.almost_equal(match_mse_loss, torch_mse_loss, check_grad=False)
        )

        # Backpropagation to calculate gradients.
        match_mse_loss.backward()
        torch_mse_loss.backward()
        self.assertTrue(
            self.almost_equal(match_prediction, torch_prediction, check_grad=True)
        )
        self.assertTrue(
            self.almost_equal(match_target, torch_target, check_grad=True)
        )
