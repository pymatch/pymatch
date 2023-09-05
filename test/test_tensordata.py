import unittest
import torch
from tensordata import TensorData


class TestTensorDataTest(unittest.TestCase):
    def test_getitem(self):
        tensor = TensorData(2, 3, 4, 5)
        self.assertEqual(type(tensor[0, 0, 0, 0]), TensorData)
        self.assertEqual(tensor[1, 2, 3, 4].item(), 0)
        self.assertRaises(IndexError, lambda: tensor[2, 0, 0, 0])
        self.assertRaises(IndexError, lambda: tensor[0, 0])

    def test_setitem(self):
        tensor = TensorData(2, 3, 4, 5)
        tensor[0, 0, 0, 0] = 47.0
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
        self.assertEqual(tensor._data, [])

    def test_create_tensor_data(self):
        tensor = TensorData(5, 2, 4)
        self.assertRaises(ValueError, lambda: tensor.item())
        self.assertEqual(tensor._item, None)
        self.assertEqual(tensor._data[0].item(), 0)

    def test_initialize_strides(self):
        tensor = TensorData(1, 2, 1)
        self.assertEqual(tensor._strides, (2, 1, 1))
        tensor = TensorData(6, 4, 2, 5, 7)
        self.assertEqual(tensor._strides, (280, 70, 35, 7, 1))

    def test_single_to_multi_rank_translation(self):
        tensor = TensorData(4, 3, 5)
        self.assertEqual(
            tensor._TensorData__single_to_multi_rank_translation(26), (1, 2, 1)
        )
        self.assertEqual(
            tensor._TensorData__single_to_multi_rank_translation(0), (0, 0, 0)
        )
        self.assertRaises(
            IndexError, lambda: tensor._TensorData__single_to_multi_rank_translation(-1)
        )
        self.assertRaises(
            IndexError, lambda: tensor._TensorData__single_to_multi_rank_translation(60)
        )

    def test_multi_to_single_rank_translation(self):
        tensor = TensorData(4, 3, 5)
        self.assertEqual(
            tensor._TensorData__multi_to_single_rank_translation((1, 2, 1)), 26
        )
        self.assertEqual(
            tensor._TensorData__multi_to_single_rank_translation((0, 0, 0)), 0
        )
        self.assertRaises(
            IndexError,
            lambda: tensor._TensorData__multi_to_single_rank_translation((-1, 0, -1)),
        )
        self.assertRaises(
            IndexError,
            lambda: tensor._TensorData__multi_to_single_rank_translation((4, 2, 4)),
        )


if __name__ == "__main__":
    unittest.main()
