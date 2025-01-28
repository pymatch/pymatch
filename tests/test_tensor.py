import unittest
import torch
from match import tensordata, tensor, randn, use_numpy
import itertools
import numpy as np
import random
from .base import BaseUnitTest

class TestTensor(BaseUnitTest):
    def test_dim(self):
        for shape in [(), (7,), (4, 3), (6, 1, 3), (1, 1, 1)]:
            mat, ten = self.generate_tensor_pair(shape)
            self.assertEqual(mat.dim(), ten.dim())

    def test_exp(self):
        mat1, ten1 = self.generate_tensor_pair((5, 4, 3))

        mat_res = mat1.exp()
        ten_res = ten1.exp()
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.sum()
        ten_mean = ten_res.sum()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))

    def test_permute(self):
        mat1, ten1 = self.generate_tensor_pair((5, 4, 3))

        mat_res = mat1.permute(0, 2, 1)
        ten_res = ten1.permute((0, 2, 1))
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.sum()
        ten_mean = ten_res.sum()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))

    def test_reshape(self):
        mat1, ten1 = self.generate_tensor_pair((5, 4, 3))

        mat_res = mat1.reshape(6, 5, 2)
        ten_res = ten1.reshape((6, 5, 2))
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.sum()
        ten_mean = ten_res.sum()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))

    def test_matmul_1d_1d(self):
        mat1, ten1 = self.generate_tensor_pair((7,))
        mat2, ten2 = self.generate_tensor_pair((7,))

        mat_res = mat1 @ mat2
        ten_res = ten1 @ ten2
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.sum()
        ten_mean = ten_res.sum()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(self.almost_equal(mat2, ten2, check_grad=True))

    def test_matmul_1d_nd(self):
        mat1, ten1 = self.generate_tensor_pair((4,))
        mat2, ten2 = self.generate_tensor_pair((3, 4, 3))

        mat_res = mat1 @ mat2
        ten_res = ten1 @ ten2

        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(self.almost_equal(mat2, ten2, check_grad=True))

    def test_matmul_nd_1d(self):
        mat1, ten1 = self.generate_tensor_pair((2, 2, 3))
        mat2, ten2 = self.generate_tensor_pair((3,))

        mat_res = mat1 @ mat2
        ten_res = ten1 @ ten2

        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(self.almost_equal(mat2, ten2, check_grad=True))

    def test_matmul(self):
        mat1, ten1 = self.generate_tensor_pair((3, 3, 2, 2, 3))
        mat2, ten2 = self.generate_tensor_pair((3, 1, 3, 4))

        mat_res = mat1 @ mat2
        ten_res = ten1 @ ten2
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(self.almost_equal(mat2, ten2, check_grad=True))

    def test_pow_int(self):
        mat, ten = self.generate_tensor_pair()

        random_exponent = 2  # For some reason this works only with integers
        mat_res = pow(mat, random_exponent)
        ten_res = pow(ten, random_exponent)

        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat, ten, check_grad=True))

    def test_mul(self):
        mat1, ten1 = self.generate_tensor_pair((2, 1, 5, 6))
        mat2, ten2 = self.generate_tensor_pair((1, 1, 1))

        mat_res = mat1 * mat2
        ten_res = ten1 * ten2
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(self.almost_equal(mat2, ten2, check_grad=True))

    def test_add(self):
        mat1, ten1 = self.generate_tensor_pair((3, 4, 5))
        mat2, ten2 = self.generate_tensor_pair((3, 4, 1))

        mat_res = mat1 + mat2
        ten_res = ten1 + ten2
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat1, ten1, check_grad=True))
        self.assertTrue(self.almost_equal(mat2, ten2, check_grad=True))

    def test_sigmoid(self):
        mat, ten = self.generate_tensor_pair()

        mat_res = mat.sigmoid()
        ten_res = ten.sigmoid()
        self.assertTrue(self.almost_equal(mat_res, ten_res))

        # Use the mean to compute backward
        mat_mean = mat_res.mean()
        ten_mean = ten_res.mean()
        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat, ten, check_grad=True))

    def test_mean(self):
        mat, ten = self.generate_tensor_pair()

        mat_mean = mat.mean()
        ten_mean = ten.mean()
        self.assertTrue(self.almost_equal(mat_mean, ten_mean))

        mat_mean.backward()
        ten_mean.backward()

        self.assertTrue(self.almost_equal(mat, ten, check_grad=True))

    def test_relu(self):
        mat, ten = self.generate_tensor_pair()

        mat_relu = mat.relu()
        ten_relu = ten.relu()
        self.assertTrue(self.almost_equal(mat_relu, ten_relu))

        # Check backward with sum
        match_val = mat_relu.sum()
        ten_val = ten_relu.sum()
        match_val.backward()
        ten_val.backward()

        self.assertTrue(self.almost_equal(mat, ten, check_grad=True))

    def test_sum_dim(self):
        mat, ten = self.generate_tensor_pair((2, 3, 4))

        mat_sum = mat.sum((0, 1))
        ten_sum = ten.sum((0, 1))
        self.assertTrue(self.almost_equal(mat_sum, ten_sum))

        # Check backward with sum
        match_val = mat_sum.sum()
        ten_val = ten_sum.sum()
        match_val.backward()
        ten_val.backward()

        self.assertTrue(self.almost_equal(mat, ten, check_grad=True))

    def test_sum_nodim(self):
        mat, ten = self.generate_tensor_pair()

        mat_sum = mat.sum()
        ten_sum = ten.sum()
        self.assertTrue(self.almost_equal(mat_sum, ten_sum))

        mat_sum.backward()
        ten_sum.backward()

        self.assertTrue(self.almost_equal(mat, ten, check_grad=True))
