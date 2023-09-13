import unittest
import torch
from match.tensordata import TensorData


class TestTensorDataTest(unittest.TestCase):
    def test_getitem_partial_index(self):
        # make torch tensor
        torch_tensor = torch.arange(27).reshape(3, 3, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 3, 3)
        match_tensor._data = [TensorData(value=i) for i in range(27)]

        torch_tensor_slice = torch_tensor[1:]
        match_tensor_slice = match_tensor[1:]

        self.assertEqual(match_tensor_slice.shape, (2, 3, 3))
        for x in range(2):
            for y in range(3):
                for z in range(3):
                    self.assertEqual(
                        torch_tensor_slice[x, y, z].item(),
                        match_tensor_slice[x, y, z].item(),
                    )

    def test_setitem_partial_index(self):
        # make torch tensor
        torch_tensor = torch.arange(27).reshape(3, 3, 3)
        # make corresponding match tensor
        match_tensor = TensorData(3, 3, 3)
        match_tensor._data = [TensorData(value=i) for i in range(27)]

        torch_tensor[2:] = torch.zeros((1,3,3))
        match_tensor[2:] = TensorData(1,3,3)

        for x in range(2):
            for y in range(3):
                for z in range(3):
                    self.assertEqual(
                        torch_tensor[x, y, z].item(),
                        match_tensor[x, y, z].item(),
                    )

    def test_setitem_slice(self):
        # make torch tensor
        torch_tensor = torch.arange(1, 9).reshape(2, 4)
        # make corresponding match tensor
        match_tensor = TensorData(2, 4)
        match_tensor._data = [TensorData(value=i + 1) for i in range(8)]

        torch_tensor[:, 1::2] = torch.zeros((2, 2))
        match_tensor[:, 1::2] = TensorData(2, 2)

        for x in range(2):
            for y in range(4):
                self.assertEqual(torch_tensor[x, y].item(), match_tensor[x, y].item())

    def test_getitem_slice(self):
        match_tensor = TensorData(2, 4)
        match_tensor._data = [TensorData(value=i + 1) for i in range(8)]
        match_tensor_slice = match_tensor[:, 1]
        self.assertEqual(match_tensor_slice.shape, (2,))
        self.assertEqual(match_tensor_slice._data[0].item(), 2)
        self.assertEqual(match_tensor_slice._data[1].item(), 6)

        match_tensor_slice = match_tensor[:, 1::2]
        self.assertEqual(match_tensor_slice.shape, (2, 2))
        self.assertEqual(match_tensor_slice._data[0].item(), 2)
        self.assertEqual(match_tensor_slice._data[1].item(), 4)
        self.assertEqual(match_tensor_slice._data[2].item(), 6)
        self.assertEqual(match_tensor_slice._data[3].item(), 8)

    def test_getitem_reference(self):
        tensor = TensorData(3, 3, 3)
        tensor_subset = tensor[2, 0, :]
        self.assertEqual(type(tensor_subset), TensorData)
        self.assertEqual(tensor_subset.shape, (3,))
        for i in range(3):
            self.assertEqual(tensor_subset._data[i].item(), 0)

        self.assertEqual(tensor._data[18].item(), 0)
        tensor_subset[0] = 47
        self.assertEqual(tensor_subset._data[0].item(), 47)
        self.assertEqual(tensor._data[18].item(), 47)

    def test_getitem_single(self):
        tensor = TensorData(2, 3, 4, 5)
        self.assertEqual(type(tensor[0, 0, 0, 0]), TensorData)
        self.assertEqual(tensor[1, 2, 3, 4].item(), 0)
        self.assertRaises(IndexError, lambda: tensor[2, 0, 0, 0])

    def test_setitem_single_number(self):
        tensor = TensorData(2, 3, 4, 5)
        tensor[0, 0, 0, 0] = 47.0
        self.assertEqual(type(tensor[0, 0, 0, 0]), TensorData)
        self.assertEqual(tensor._data[0].item(), 47.0)

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
