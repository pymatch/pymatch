import random
import unittest
import match.nn
from match import tensor, randn


class TestModule(unittest.TestCase):
    """
    Unit tests for Base Module
    """

    def test_module_parameters_nested_iterable(self):
        param1 = randn(1)
        param2 = randn(2)
        param3 = randn(3)
        param4 = randn(4)

        class MatchNetwork1(match.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = param1
                self.w2 = set((param2,))

        class MatchNetwork2(match.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w3 = [[MatchNetwork1()], param3]
                self.w4 = param4

        parameters = MatchNetwork2().parameters()
        self.assertEqual(len(parameters), 4)
        self.assertTrue(all(type(p) == tensor.Tensor for p in parameters))

    def test_module_parameters_iterable_simple(self):
        param1 = randn(1)
        param2 = randn(2)

        class MatchNetwork(match.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = [param1, param2]

        parameters = MatchNetwork().parameters()
        self.assertEqual(len(parameters), 2)
        self.assertTrue(all(type(p) == tensor.Tensor for p in parameters))

    def test_module_parameters_same_reference_iterable(self):
        param1 = randn(1)
        param2 = param1

        class MatchNetwork(match.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = param1
                self.w2 = [
                    param2
                ]  # param1 and param2 aren't the same reference anymore.

        parameters = MatchNetwork().parameters()
        self.assertEqual(len(parameters), 2)
        self.assertTrue(all(type(p) == tensor.Tensor for p in parameters))

    def test_module_parameters_same_reference(self):
        param1 = randn(1)
        param2 = param1

        class MatchNetwork(match.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = param1
                self.w2 = param2

        self.assertCountEqual(MatchNetwork().parameters(), [param1])

    def test_module_parameters_normal(self):
        param1 = randn(1)
        param2 = randn(2)

        class MatchNetwork(match.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w1 = param1
                self.w2 = param2

        self.assertCountEqual(MatchNetwork().parameters(), [param1, param2])
