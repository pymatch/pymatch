import unittest
import torch
from match.tensordata import TensorData
from .base import BaseUnitTest
import itertools
from match import prod
from random import gauss


class TestTensorDataTest(BaseUnitTest):

    def to_tensor(
        self, match_tensor_data: TensorData, requires_grad=False, get_grad=False
    ) -> torch.Tensor:
        """
        Overrides BaseUnitTest.to_tensor
        """
        torch_tensor = None

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

    def randn(self, shape: tuple = None, generator=lambda: gauss(0, 1)):
        if not shape:
            return TensorData(value=generator())

        rand_tensordata = TensorData(0)
        data = [TensorData(value=generator()) for _ in range(prod(shape))]
        rand_tensordata._data = data
        rand_tensordata.reshape_(shape)
        return rand_tensordata

    def generate_tensor_pair(self, shape: tuple[int] = None):
        """
        Generates a random tensor and its PyTorch equivalent for testing.

        :param shape: Tuple defining the shape of the tensor. If None, a random shape is generated.
        :return: A tuple containing a custom tensor and a PyTorch tensor
        """
        if not shape:
            dim = random.randint(2, 5)
            shape = tuple(random.randint(1, 5) for _ in range(dim))

        mat = self.randn(shape)
        ten = self.to_tensor(mat, requires_grad=False)
        return mat, ten

    def test_concatenate_default(self):
        torch_tensor = torch.arange(6).reshape(2, 3).float()
        match_tensor = TensorData(2, 3)
        match_tensor._data = [TensorData(value=i) for i in range(6)]

        self.assertTrue(
            self.almost_equal(
                TensorData.concatenate((match_tensor, match_tensor, match_tensor)),
                torch.cat((torch_tensor, torch_tensor, torch_tensor)),
            )
        )

    # def test_concatenate_column(self):
    #     torch_tensor = torch.arange(6).reshape(2, 3).float()
    #     match_tensor = TensorData(2, 3)
    #     match_tensor._data = [TensorData(value=i) for i in range(6)]

    #     self.assertTrue(
    #         self.almost_equal(
    #             TensorData.concatenate((match_tensor, match_tensor, match_tensor), 1),
    #             torch.cat((torch_tensor, torch_tensor, torch_tensor), 1),
    #         )
    #     )

    # def test_concatenate_nd(self):
    #     torch_tensor = torch.arange(24).reshape(2, 4, 3).float()
    #     match_tensor = TensorData(2, 3, 4)
    #     match_tensor._data = [TensorData(value=i) for i in range(24)]

    #     self.assertTrue(
    #         self.almost_equal(
    #             TensorData.concatenate((match_tensor, match_tensor, match_tensor), 1),
    #             torch.cat((torch_tensor, torch_tensor, torch_tensor), 1),
    #         )
    #     )

    def test_mean_nodim(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((2, 4, 3))
        self.assertTrue(self.almost_equal(match_tensor.mean(), torch_tensor.mean()))

    def test_mean_keepdim(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((2, 4, 3))
        self.assertTrue(
            self.almost_equal(
                match_tensor.mean((1, 2), keepdims=True),
                torch_tensor.mean((1, 2), keepdim=True),
            )
        )

    def test_mean(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((2, 4, 3))
        self.assertTrue(
            self.almost_equal(match_tensor.mean((0,)), torch_tensor.mean((0,)))
        )

    def test_sum_nodim(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((2, 4, 3))
        self.assertTrue(self.almost_equal(match_tensor.sum(), torch_tensor.sum()))

    def test_sum_keepdim(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((2, 4, 3))
        self.assertTrue(
            self.almost_equal(
                match_tensor.sum((1, 2), keepdims=True),
                torch_tensor.sum((1, 2), keepdim=True),
            )
        )

    def test_sum(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3))
        self.assertTrue(
            self.almost_equal(match_tensor.sum((0,)), torch_tensor.sum((0,)))
        )
        self.assertTrue(
            self.almost_equal(match_tensor.sum((0, 1)), torch_tensor.sum((0, 1)))
        )

    # TODO: Implement configurations that are intended to fail.
    def test_matmul_various_shapes_failure(self):
        self.assertTrue(True)

    def test_matmul_various_shapes(self):
        configurations = {
            "1d@1d": [(9,), (9,)],
            "1d@2d": [(9,), (9,)],
            "2d@1d": [(8,), (8, 2)],
            "2d@2d": [(7, 8), (8, 2)],
            "1d@nd": [(5,), (2, 5, 3)],
            "nd@1d": [(3, 2, 5, 8, 5), (5,)],
            "nd@nd": [(2, 1, 7, 4), (2, 4, 3)],
        }
        for msg, shapes in configurations.items():
            with self.subTest(msg=msg):
                match_tensor1, torch_tensor1 = self.generate_tensor_pair(shapes[0])
                match_tensor2, torch_tensor2 = self.generate_tensor_pair(shapes[1])
                self.assertTrue(
                    self.almost_equal(
                        match_tensor1 @ match_tensor2, torch_tensor1 @ torch_tensor2
                    )
                )

    # TODO: Add condition to test the same references.
    def test_transpose(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3))
        self.assertTrue(self.almost_equal(match_tensor.T, torch_tensor.T))

    # TODO: Add condition to test the same references.
    def test_permute(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3))
        self.assertTrue(
            self.almost_equal(
                match_tensor.permute(2, 0, 1), torch_tensor.permute(2, 0, 1)
            )
        )

    def test_reshape_1d_to_singleton(self):
        match_tensor = TensorData(1)
        match_tensor_reshaped = match_tensor.reshape(())

        self.assertEqual(match_tensor.shape, (1,))
        self.assertEqual(match_tensor_reshaped.shape, ())
        self.assertEqual(match_tensor_reshaped.item(), 0)
        self.assertIsNone(match_tensor_reshaped._data)

        # They must have the same reference
        self.assertTrue(self.same_references(match_tensor[0], match_tensor_reshaped))

    def test_reshape_singleton_to_1d(self):
        match_tensor = TensorData(value=47)
        match_tensor_reshaped = match_tensor.reshape((1,))

        # Ensure original object didn't change
        self.assertEqual(match_tensor.shape, ())
        self.assertEqual(match_tensor.item(), 47)
        self.assertIsNone(match_tensor._data)

        self.assertEqual(len(match_tensor_reshaped._data), 1)
        self.assertEqual(match_tensor_reshaped.shape, (1,))
        self.assertEqual(match_tensor_reshaped.item(), 47)
        self.assertIsNone(match_tensor_reshaped._item, None)

        # They must have the same reference
        self.assertTrue(self.same_references(match_tensor, match_tensor_reshaped[0]))

    def test_reshape_failure(self):
        match_tensor = TensorData(2, 3, 4)
        self.assertRaises(RuntimeError, lambda: match_tensor.reshape((5, 5, 5)))

    def test_reshape(self):
        match_tensor = TensorData(2, 3, 4)
        match_tensor_reshaped = match_tensor.reshape((4, 3, 2))
        self.assertEqual(match_tensor.shape, (2, 3, 4))
        self.assertEqual(match_tensor_reshaped.shape, (4, 3, 2))
        self.assertEqual(len(match_tensor._data), len(match_tensor_reshaped._data))
        self.assertTrue(self.same_references(match_tensor, match_tensor_reshaped))
        self.assertRaises(IndexError, lambda: match_tensor_reshaped[1, 2, 3])

    def test_broadcast_singleton(self):
        # Make torch singleton and corresponding match singleton.
        match_tensor = TensorData(value=0)
        torch_tensor = self.to_tensor(match_tensor)

        torch_tensor_broadcasted = torch.broadcast_to(torch_tensor, (3, 3))
        match_tensor_broadcasted = match_tensor.broadcast(3, 3)

        self.assertTrue(
            self.almost_equal(match_tensor_broadcasted, torch_tensor_broadcasted)
        )

    def test_broadcast(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair((3, 1, 3))

            torch_tensor_broadcasted = torch.broadcast_to(torch_tensor, (2, 2, 3, 3, 3))
            match_tensor_broadcasted = match_tensor.broadcast(2, 2, 3, 3, 3)

            self.assertTrue(
                self.almost_equal(match_tensor_broadcasted, torch_tensor_broadcasted)
            )
        with self.subTest(msg="failure"):
            match_tensor = TensorData(3, 1, 3)
            self.assertRaises(ValueError, lambda: match_tensor.broadcast(2, 2, 1, 3, 3))
            self.assertRaises(RuntimeError, lambda: match_tensor.broadcast(3, 3))

    def test_getitem_partial_index(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair((3, 3, 3))
            self.assertTrue(self.almost_equal(match_tensor[1:], torch_tensor[1:]))
        with self.subTest(msg="extreme"):
            match_tensor, torch_tensor = self.generate_tensor_pair((2, 4))
            self.assertTrue(self.almost_equal(match_tensor[4:], torch_tensor[4:]))

    def test_getitem_slice(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair((2, 4))
            self.assertTrue(self.almost_equal(match_tensor[:, 1], torch_tensor[:, 1]))
            self.assertTrue(
                self.almost_equal(match_tensor[:, 1::2], torch_tensor[:, 1::2])
            )
        with self.subTest(msg="slice_zero_failure"):
            match_tensor, _ = self.generate_tensor_pair((2, 4))
            self.assertRaises(ValueError, lambda: match_tensor[:, 1::0])

    def test_getitem_reference(self):
        tensor = TensorData(3, 3, 3)
        tensor_subset = tensor[2, 0, :]
        self.assertTrue(self.same_references(tensor_subset, tensor[2, 0, :]))
        self.assertFalse(
            self.same_references(TensorData(3, 3, 3)[2, 0, :], tensor[2, 0, :])
        )

    def test_getitem_single(self):
        with self.subTest(msg="normal"):
            self.assertEqual(TensorData(2, 3, 4, 5)[0, 1, 2, 3].item(), 0)
        with self.subTest(msg="extreme"):
            self.assertEqual(TensorData(2, 3, 4, 5)[1, 2, 3, 4].item(), 0)
            self.assertEqual(TensorData(2, 3, 4, 5)[0, 0, 0, 0].item(), 0)
        with self.subTest(msg="oob_failure"):
            self.assertRaises(IndexError, lambda: TensorData(2, 3, 4, 5)[2, 0, 0, 0])

    def test_setitem_single_value_index(self):
        match_tensor, torch_tensor = self.generate_tensor_pair((3, 3, 3))
        torch_tensor[2:] = 0
        match_tensor[2:] = 0
        self.assertTrue(self.almost_equal(match_tensor, torch_tensor))

    def test_setitem_partial_index(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair((3, 3, 3))
            torch_tensor[2:] = torch.zeros((1, 3, 3))
            match_tensor[2:] = TensorData(1, 3, 3)
            self.assertTrue(self.almost_equal(match_tensor, torch_tensor))
        with self.subTest(msg="shape_mismatch_failure"):
            match_tensor, _ = self.generate_tensor_pair((3, 3, 3))

            def setitem_helper():
                match_tensor[2:] = TensorData(3, 3, 1)

            self.assertRaises(RuntimeError, setitem_helper)

    def test_setitem_slice(self):
        with self.subTest(msg="normal"):
            match_tensor, torch_tensor = self.generate_tensor_pair((2, 4))
            torch_tensor[:, 1::2] = torch.zeros((2, 2))
            match_tensor[:, 1::2] = TensorData(2, 2)
            self.assertTrue(self.almost_equal(match_tensor, torch_tensor))

        with self.subTest(msg="shape_mismatch_failure"):
            match_tensor, _ = self.generate_tensor_pair((2, 4))

            def setitem_helper():
                match_tensor[:, 1::2] = TensorData(2, 3)

            self.assertRaises(RuntimeError, setitem_helper)

    def test_setitem_single_number(self):
        with self.subTest(msg="normal"):
            tensor = TensorData(2, 3, 4, 5)
            tensor[0, 0, 0, 3] = 47.0
            self.assertEqual(tensor._data[3].item(), 47.0)
        with self.subTest(msg="extreme"):
            tensor = TensorData(2, 3, 4, 5)
            tensor[0, 0, 0, 0] = 47.0
            self.assertEqual(tensor._data[0].item(), 47.0)

            tensor = TensorData(2, 3, 4, 5)
            tensor[1, 2, 3, 4] = 47.0
            self.assertEqual(tensor._data[-1].item(), 47.0)
        with self.subTest(msg="oob_failure"):
            tensor = TensorData(2, 3, 4, 5)

            def setitem_helper():
                tensor[1, 2, 3, 5] = 47.0

            self.assertRaises(IndexError, setitem_helper)

    def test_setitem_single_tensordata(self):
        tensor = TensorData(2, 3, 4, 5)
        tensor[0, 0, 0, 0] = TensorData(value=47)
        self.assertEqual(type(tensor[0, 0, 0, 0]), TensorData)
        self.assertEqual(tensor._data[0].item(), 47.0)

    def test_create_tensor_data_no_data(self):
        tensor = TensorData(5, 5, 0)
        self.assertEqual(tensor._data, [])
        self.assertEqual(tensor._item, None)
        self.assertRaises(ValueError, lambda: tensor.item())

    def test_create_tensor_data_singleton(self):
        tensor = TensorData(value=47.0)
        self.assertEqual(tensor.item(), 47.0)
        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor._data, None)

    def test_create_tensor_data(self):
        tensor = TensorData(5, 2, 4)
        self.assertRaises(ValueError, lambda: tensor.item())
        self.assertEqual(tensor._item, None)
        self.assertEqual(tensor._data[0].item(), 0)

    def test_initialize_strides(self):
        self.assertEqual(TensorData(1, 2, 1)._strides, (2, 1, 1))
        self.assertEqual(TensorData(6, 4, 2, 5, 7)._strides, (280, 70, 35, 7, 1))

    def test_single_to_multi_rank_translation(self):
        tensor = TensorData(4, 3, 5)
        with self.subTest(msg="normal"):
            self.assertEqual(
                tensor._TensorData__single_to_multi_rank_translation(26), (1, 2, 1)
            )
        with self.subTest(msg="extreme"):
            self.assertEqual(
                tensor._TensorData__single_to_multi_rank_translation(0), (0, 0, 0)
            )
            self.assertEqual(
                tensor._TensorData__single_to_multi_rank_translation(59), (3, 2, 4)
            )
        with self.subTest(msg="fail"):
            self.assertRaises(
                IndexError,
                lambda: tensor._TensorData__single_to_multi_rank_translation(-1),
            )
            self.assertRaises(
                IndexError,
                lambda: tensor._TensorData__single_to_multi_rank_translation(60),
            )
            self.assertRaises(
                IndexError,
                lambda: tensor._TensorData__single_to_multi_rank_translation(90),
            )

    def test_multi_to_single_rank_translation(self):
        tensor = TensorData(4, 3, 5)
        with self.subTest(msg="normal"):
            self.assertEqual(
                tensor._TensorData__multi_to_single_rank_translation((1, 2, 1)), 26
            )
        with self.subTest(msg="extreme"):
            self.assertEqual(
                tensor._TensorData__multi_to_single_rank_translation((0, 0, 0)), 0
            )

            self.assertEqual(
                tensor._TensorData__multi_to_single_rank_translation((3, 2, 4)), 59
            )
        with self.subTest(msg="fail"):
            self.assertRaises(
                IndexError,
                lambda: tensor._TensorData__multi_to_single_rank_translation(
                    (-1, 0, -1)
                ),
            )
            self.assertRaises(
                IndexError,
                lambda: tensor._TensorData__multi_to_single_rank_translation((4, 2, 4)),
            )


if __name__ == "__main__":
    unittest.main()
