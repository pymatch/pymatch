#!/usr/bin/env python

from __future__ import annotations

import unittest

# Use torch to compute correct output for comparison
import torch
from torch import Tensor

import match
from match import Matrix


def almostEqual(matrix: Matrix, tensor: Tensor, check_grad=False) -> bool:
    m = to_tensor(matrix, get_grad=check_grad)
    t = Tensor(tensor.grad) if check_grad else tensor
    if t.ndim == 1:
        m.squeeze_()
    return torch.allclose(m, t, rtol=1e-02, atol=1e-05)


def to_tensor(matrix: Matrix, requires_grad=False, get_grad=False) -> Tensor:
    mdata = matrix.grad.vals if get_grad else matrix.data.vals
    return torch.tensor(mdata, requires_grad=requires_grad)


def mat_and_ten(dim1, dim2) -> tuple[Matrix, Tensor]:
    mat = match.randn(dim1, dim2)
    ten = to_tensor(mat, requires_grad=True)
    return mat, ten


def neuron(a, w, b, relu=True):
    z = a @ w.T + b.T
    a = z.relu() if relu else z.sigmoid()
    return z, a


class TestMatch(unittest.TestCase):
    def test_arithmetic(self):
        """Test the output and gradient of arbitrary arithmetic."""

        mat1, ten1 = mat_and_ten(3, 2)
        mat2, ten2 = mat_and_ten(3, 2)

        mat3 = mat1 * mat2 * -1 + 5
        ten3 = ten1 * ten2 * -1 + 5
        self.assertTrue(almostEqual(mat3, ten3))

        mat4 = mat3.sigmoid()
        ten4 = ten3.sigmoid()
        self.assertTrue(almostEqual(mat4, ten4))

        mat5 = (mat4 / mat1) ** 3
        ten5 = (ten4 / ten1) ** 3
        self.assertTrue(almostEqual(mat5, ten5))

        mat6 = mat5.sigmoid()
        ten6 = ten5.sigmoid()
        self.assertTrue(almostEqual(mat6, ten6))

        mat7 = mat6.sum()
        ten7 = ten6.sum()
        self.assertTrue(almostEqual(mat7, ten7))

        mat7.backward()
        ten7.backward()
        self.assertTrue(almostEqual(mat1, ten1, check_grad=True))
        self.assertTrue(almostEqual(mat2, ten2, check_grad=True))

    def test_relu(self):
        """Test the relu activation function."""
        match_relu = match.nn.ReLU()
        torch_relu = torch.nn.ReLU()
    
        match_tensor, torch_tensor = mat_and_ten(31, 17)
    
        # Check forward
        match_relu_output = match_relu(match_tensor)
        torch_relu_output = torch_relu(torch_tensor)
        self.assertTrue(almostEqual(match_relu_output, torch_relu_output))
    
        # Check backward
        match_mean = match_relu_output.mean()
        match_mean.backward()
    
        torch_mean = torch_relu_output.mean()
        torch_mean.backward()
    
        self.assertTrue(almostEqual(match_tensor, torch_tensor, check_grad=True))

    def test_mse(self):
        """Test the MSE loss function."""
        match_mse = match.nn.MSELoss()
        torch_mse = torch.nn.MSELoss()
    
        match_y, torch_y = mat_and_ten(4, 3)
        match_yhat, torch_yhat = mat_and_ten(4, 3)
    
        # Check forward
        match_mse_output = match_mse(match_y, match_yhat)
        torch_mse_output = torch_mse(torch_y, torch_yhat)
        self.assertTrue(almostEqual(match_mse_output, torch_mse_output))
    
        # Check backward
        mtest = match_mse_output.mean()
        mtest.backward()
    
        ttest = torch_mse_output.mean()
        ttest.backward()
    
        # Note: outputs really shouldn't have derivatives, but we can still test them here
        self.assertTrue(almostEqual(match_y, torch_y, check_grad=True))
        self.assertTrue(almostEqual(match_yhat, torch_yhat, check_grad=True))

    def test_nn(self):
        """Test the neural network layer objects."""
        N, n0, n1 = 7, 10, 14

        mat_linr = match.nn.Linear(n0, n1)
        mat_relu = match.nn.ReLU()

        ten_linr = torch.nn.Linear(n0, n1)
        ten_relu = torch.nn.ReLU()

        # Manually set the tensor to the same values as the matrix
        ten_linr.weight = torch.nn.Parameter(to_tensor(mat_linr.W))
        ten_linr.bias = torch.nn.Parameter(to_tensor(mat_linr.b).squeeze())

        mat_x, ten_x = mat_and_ten(N, n0)

        mat_z = mat_linr(mat_x)
        mat_a = mat_relu(mat_z)

        ten_z = ten_linr(ten_x)
        ten_a = ten_relu(ten_z)

        self.assertTrue(almostEqual(mat_z, ten_z))
        self.assertTrue(almostEqual(mat_a, ten_a))

    def test_module(self):
        """Test the neural network module class."""
        N, n0, n1, n2 = 7, 10, 14, 7

        class MatchNetwork(match.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = match.nn.Linear(n0, n1)
                self.relu = match.nn.ReLU()
                self.linear2 = match.nn.Linear(n1, n2)
                self.sigmoid = match.nn.Sigmoid()

            def forward(self, x) -> Matrix:
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x

        class TorchNetwork(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(n0, n1)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(n1, n2)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x) -> Matrix:
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x

        match_net = MatchNetwork()
        torch_net = TorchNetwork()

        # Set parameter values equal to one another
        with torch.no_grad():
            for mparam, tparam in zip(match_net.parameters(), torch_net.parameters()):
                t = torch.tensor(mparam.data.vals).squeeze()
                tparam.copy_(t)

        mat_x, ten_x = mat_and_ten(N, n0)

        mat_y = match_net(mat_x)
        ten_y = torch_net(ten_x)

        self.assertTrue(almostEqual(mat_y, ten_y))

        mat_y_mean = mat_y.mean()
        ten_y_mean = ten_y.mean()

        mat_y_mean.backward()
        ten_y_mean.backward()

        for mparam, tparam in zip(match_net.parameters(), torch_net.parameters()):
            self.assertTrue(almostEqual(mparam, tparam, check_grad=True))

    def test_3layer(self):
        """Test the output and gradient of a three layer network."""

        N = 5
        n0 = 4
        n1 = 3
        n2 = 6
        n3 = 1

        # Fake input and output
        x = mat_and_ten(N, n0)
        y = mat_and_ten(N, 1)

        # Parameters
        W = []
        b = []

        # Layer 1
        W.append(mat_and_ten(n1, n0))
        b.append(mat_and_ten(n1, 1))

        # Layer 2
        W.append(mat_and_ten(n2, n1))
        b.append(mat_and_ten(n2, 1))

        # Layer 3
        W.append(mat_and_ten(n3, n2))
        b.append(mat_and_ten(n3, 1))

        # Forward
        mat_a, ten_a = x
        for i, ((mat_W, ten_W), (mat_b, ten_b)) in enumerate(zip(W, b)):
            mat_z, mat_a = neuron(mat_a, mat_W, mat_b, relu=(i < len(W) - 1))
            ten_z, ten_a = neuron(ten_a, ten_W, ten_b, relu=(i < len(W) - 1))
            self.assertTrue(almostEqual(mat_z, ten_z))
            self.assertTrue(almostEqual(mat_a, ten_a))

        # MSE Loss
        mat_y, ten_y = y
        mat_loss = ((mat_a - mat_y) ** 2).mean()
        ten_loss = ((ten_a - ten_y) ** 2).mean()
        self.assertTrue(almostEqual(mat_loss, ten_loss))

        # Backward
        mat_loss.backward()
        ten_loss.backward()

        # Check all gradients (even input and output)
        grads = [y] + W + b + [x]
        for mat_g, ten_g in grads:
            self.assertTrue(almostEqual(mat_g, ten_g, check_grad=True))


if __name__ == "__main__":
    unittest.main()
