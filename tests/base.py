import unittest
import random
import itertools
import torch
import numpy as np
import logging
from match import tensordata, tensor, randn, use_numpy

# Create a logger
logger = logging.getLogger(__name__)


class BaseUnitTest(unittest.TestCase):
    """
    A base class for unit testing custom tensor implementations.

    This class provides utility methods for comparing custom tensors with PyTorch tensors,
    converting custom tensors to PyTorch tensors, and generating random test data. It is
    designed to be inherited by other test classes to simplify and standardize tensor-based
    unit tests.
    """

    def almost_equal(
        self,
        match_tensor: tensor.Tensor,
        pytorch_tensor: torch.Tensor,
        check_grad=False,
        debug: bool = False,
    ) -> bool:
        """
        Compares a custom tensor implementation with a PyTorch tensor for equality within a tolerance.

        This method is useful in scenarios where you need to verify that a custom tensor
        implementation produces the same results as a PyTorch tensor. For example, it can be
        used to validate the output of custom operations or layers against their PyTorch
        equivalents. The method can also compare gradients if required.

        :param match_tensor: Custom tensor to compare
        :param pytorch_tensor: PyTorch tensor to compare
        :param check_grad: If True, compare gradients
        :param debug: If True, print tensors for debugging
        :return: True if tensors are almost equal, False otherwise
        """
        m = self.to_tensor(match_tensor, get_grad=check_grad)
        t = pytorch_tensor.grad if check_grad else pytorch_tensor

        if t.ndim == 1:
            m.squeeze_()

        is_close = torch.allclose(m, t, rtol=1e-02, atol=1e-05)
        if debug:
            logger.info("Match: ", m)
            logger.info("Torch: ", t)
        return is_close

    def to_tensor(
        self, match_tensor: tensor.Tensor, requires_grad=False, get_grad=False
    ) -> torch.Tensor:
        """
        Converts a custom tensor to a PyTorch tensor.

        :param match_tensor: Custom tensor to convert
        :param requires_grad: If True, the resulting PyTorch tensor will require gradients
        :param get_grad: If True, use the gradient of the custom tensor
        :return: Converted PyTorch tensor
        """
        match_tensor_data = match_tensor.grad if get_grad else match_tensor.data
        torch_tensor = None

        if use_numpy:
            torch_tensor = torch.from_numpy(
                np.array(match_tensor_data._numpy_data)
            ).float()
        else:
            if match_tensor_data._data is None:
                torch_tensor = torch.tensor(data=match_tensor_data.item()).float()
            else:
                torch_tensor = torch.zeros(match_tensor_data.shape).float()
                for index in itertools.product(
                    *[range(dim) for dim in match_tensor_data.shape]
                ):
                    torch_tensor[index] = match_tensor_data[index].item()

        torch_tensor.requires_grad = requires_grad
        return torch_tensor

    def generate_tensor_pair(self, shape: tuple[int] = None):
        """
        Generates a random tensor and its PyTorch equivalent for testing.

        :param shape: Tuple defining the shape of the tensor. If None, a random shape is generated.
        :return: A tuple containing a custom tensor and a PyTorch tensor
        """
        if not shape:
            dim = random.randint(2, 5)
            shape = tuple(random.randint(1, 5) for _ in range(dim))

        mat = randn(*shape)
        ten = self.to_tensor(mat, requires_grad=True)
        return mat, ten

    def same_references(
        self, match_tensor1: tensordata.TensorData, match_tensor2: tensordata.TensorData
    ) -> bool:
        """
        Checks if two tensors reference the same data elements.

        This method compares the underlying data of two tensors to determine
        if they share the same memory references for each element.

        :param match_tensor1: The first tensor to compare
        :param match_tensor2: The second tensor to compare
        :return: True if all elements in the tensors reference the same memory, False otherwise
        """
        if match_tensor1._data == None and match_tensor2._data == None:
            return match_tensor1 is match_tensor2
        
        # Ensure both tensors have the same number of elements
        if len(match_tensor1._data) != len(match_tensor2._data):
            return False

        # Check if every element references the same memory location
        return all(e1 is e2 for e1, e2 in zip(match_tensor1._data, match_tensor2._data))
