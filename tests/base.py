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
    """

    def almost_equal(
        self,
        match_tensor: tensor.Tensor,
        pytorch_tensor: torch.Tensor,
        check_grad: bool = False,
        debug: bool = False,
    ) -> bool:
        """Compares the custom tensor implementation to Pytorch's tensor implementation.

        Args:
            match_tensor (tensor.Tensor): The PyMatch Tensor
            pytorch_tensor (torch.Tensor): PyTorch Tensor
            check_grad (bool, optional): If true, compare the gradient tensors of the match and pytorch Tensor. Defaults to False.
            debug (bool, optional): Verbose logging. Defaults to False.

        Returns:
            bool: Returns True if the match and pytorch are equivalent. False otherwise.
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
        """Converts a match tensor to a Pytorch tensor.

        Args:
            match_tensor (tensor.Tensor): The custom match tensor to convert
            requires_grad (bool, optional): If True, the resulting PyTorch tensor will require grad. Defaults to False.
            get_grad (bool, optional): If True, convert the grad of the provided match tensor to convert. Defaults to False.

        Returns:
            torch.Tensor: The equivalent PyTorch implementation of the provided match Match tensor.
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
        """Generates a random Match tensor and its PyTorch equivalent.

        Args:
            shape (tuple[int], optional): The shape of the random tensor pair. Defaults to None. If None, a random shape is used.

        Returns:
            match.Tensor, torch.Tensor: The random tensor tuple
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
        """Checks of two Match tensors reference the same data elements.

        Args:
            match_tensor1 (tensordata.TensorData): The first tensor to compare.
            match_tensor2 (tensordata.TensorData): The second tensor to compare

        Returns:
            bool: Returns True if the two tensors' base data reference the same memory. False otherwise.
        """
        if match_tensor1._data == None and match_tensor2._data == None:
            return match_tensor1 is match_tensor2

        # Ensure both tensors have the same number of elements
        if len(match_tensor1._data) != len(match_tensor2._data):
            return False

        # Check if every element references the same memory location
        return all(e1 is e2 for e1, e2 in zip(match_tensor1._data, match_tensor2._data))
