import unittest
import torch
from match import tensordata, tensor, randn
import itertools
import numpy as np


def almost_equal(
    match_tensor: tensor.Tensor, pytorch_tensor: torch.Tensor, check_grad=False
) -> bool:
    m = to_tensor(match_tensor, get_grad=check_grad)
    t = torch.Tensor(pytorch_tensor.grad) if check_grad else pytorch_tensor
    if t.ndim == 1:
        m.squeeze_()
    return torch.allclose(m, t, rtol=1e-02, atol=1e-05)


def to_tensor(
    match_tensor: tensor.Tensor, requires_grad=False, get_grad=False
) -> torch.Tensor:
    match_tensor_data = match_tensor.grad if get_grad else match_tensor.data
    torch_tensor = None
    if match_tensor_data.use_numpy:
        torch_tensor = torch.from_numpy(match_tensor_data._numpy_data).float()

    else:
        if match_tensor_data._data == None:
            torch_tensor = torch.tensor(data=match_tensor_data.item()).float()
        else:
            torch_tensor = torch.zeros(match_tensor_data.shape).float()
            for index in itertools.product(
                *[range(dim) for dim in match_tensor_data.shape]
            ):
                torch_tensor[index] = match_tensor_data[index].item()

    torch_tensor.requires_grad = requires_grad
    return torch_tensor


class TestTensorDataTest(unittest.TestCase):
    def test_toTensor_numpy(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3).float()

        match_tensor_data = tensordata.TensorData(
            numpy_data=np.arange(24).reshape(2, 4, 3), use_numpy=True
        )
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_toTensor_singleton_numpy(self):
        torch_tensor = torch.tensor(47).float()

        match_tensor_data = tensordata.TensorData(value=47, use_numpy=True)
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_toTensor_singleton(self):
        torch_tensor = torch.tensor(47).float()

        match_tensor_data = tensordata.TensorData(value=47, use_numpy=False)
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))

    def test_toTensor(self):
        torch_tensor = torch.arange(24).reshape(2, 4, 3).float()

        match_tensor_data = tensordata.TensorData(2, 4, 3, use_numpy=False)
        match_tensor_data._data = [tensordata.TensorData(value=i) for i in range(24)]
        match_tensor = tensor.Tensor(data=match_tensor_data)

        self.assertTrue(almost_equal(match_tensor, torch_tensor))
