import unittest
import torch
from tensordata import TensorData


class TestTensorDataTest(unittest.TestCase):
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
