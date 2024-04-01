from __future__ import annotations
from typing import Union
import numpy as np


class TensorData(object):
    """A storage and arithmetic object for n-dimensional tensor data.

    TensorData is an inefficient, but easy-to-understand implementation
    of many n-dimensional tensor operations.

    Like the PyTorch Tensor object, the pymatch TensorData objects are
    recursive structures that hold either a list of TensorData objects,
    or a single value. For instance, a TensorData object's data list could
    look like [TensorData(0), TensorData(1), raise NotImplementedError, TensorData(47)].
    There are therefore two implicit types of TensorData objects. Ones
    that store only a single value, accesible by .item(), and ones that
    store a list of these `singleton` TensorData objects.

    Using the provided shape of the TensorData object, accessing data using
    the standard coordinate system, for instance x[1,3,2], involves translation
    of the coordinates (1,3,2) to the corresponding index in the data list
    to fetch the intended data.
    """

    def __init__(
        self,
        *size: int,
        value: float = 0.0,
        numpy_data: np.ndarray = None,
    ) -> None:
        """Create a new TensorData object to store an n-dimensional tensor of values.

        Args:
            size (tuple[int]): The shape of the Tensor
            value (float): The default value of each element in the Tensor
            dtype (type): The type of the values in the Tensor
        """
        super().__init__()
        assert all(
            isinstance(dim, int) for dim in size
        ), f"Size {size} must be a variadict of only integers"
        self._numpy_data = numpy_data
        self.__initialize_tensor_data(value)

    def __initialize_tensor_data(self, value) -> None:
        """Helper method that initializes all the values in the TensorData object."""
        if self._numpy_data is None:
            self._numpy_data: np.ndarray = np.full(self.shape, value)

    def item(self) -> Union[int, float]:
        """Returns the item of a singleton, or single element TensorData object.

        Raises:
            ValueError: raises value error if attempting to access the item of a
            TensorData object with more than one element.
        """
        return self._numpy_data.item(0)

    def reshape_(self, shape: tuple):
        """Helper method to reshape the TensorData object inplace, without changing the data.

        Raises:
            RuntimeError: raises runtime error if the product of the dimensions of
            the new shape is not equal to the existing number of elements.
        """
        self._numpy_data = self._numpy_data.reshape(shape)

    def reshape(self, shape: tuple) -> TensorData:
        """Helper method to reshape and return a new TensorData object without changing the data"""
        return TensorData(
            *shape,
            numpy_data=self._numpy_data.reshape(shape),
        )

    def __getitem__(self, coords):
        """Retrieves a subtensor from the current tensor using the given coordinates.

        Args:
            coords: A tuple of slices or integers.

        Returns:
            A new tensor containing the subtensor specified by the given coordinates.
        """
        if not isinstance(coords, tuple):
            coords = (coords,)
        res = self._numpy_data[*coords]
        return TensorData(*res.shape, numpy_data=res)

    def __setitem__(self, coords, value):
        """Sets the value of a subtensor at the given coordinates.

        Args:
            coords: A tuple of slices or integers.
            value: A new value to assign to the subtensor.

        Raises:
            TypeError: If the value is a list or an invalid type.
        """
        if isinstance(value, list):
            raise TypeError("Unable to assign a list to a TensorData")

        if not isinstance(coords, tuple):
            coords = (coords,)

        if isinstance(value, TensorData):
            raise TypeError("Incompatible TensorData options. Numpy and TensorData")
        self._numpy_data[*coords] = value

    def __repr__(self):
        return self._numpy_data.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def sum(self, dims: tuple | int = None, keepdims: bool = False) -> TensorData:
        if isinstance(dims, int):
            dims = (dims,)

        res = self._numpy_data.sum(axis=dims, keepdims=keepdims)
        return TensorData(*res.shape, numpy_data=res)

    def mean(self, dims: tuple | int = None, keepdims: bool = None) -> TensorData:
        """Compute the mean of all values in the tensor."""
        if isinstance(dims, int):
            dims = (dims,)

        res_mean = self._numpy_data.mean(axis=dims, keepdims=keepdims)
        return TensorData(*res_mean.shape, use_numpy=True, numpy_data=res_mean)

    def unbroadcast(self, *shape: int) -> TensorData:
        """Return a new TensorData unbroadcast from current shape to desired shape.

        Reference to this: https://mostafa-samir.github.io/auto-diff-pt2/#unbroadcasting-adjoints
        """
        # TODO(SAM): Change to shallow copy.
        correct_adjoint = self

        if self.shape != shape:
            dim_diff = abs(len(self.shape) - len(shape))
            if dim_diff:  # != 0
                summation_dims = tuple(range(dim_diff))
                correct_adjoint = self.sum(dims=summation_dims)
            # If the shape was (3,4,5,6) and we want to unbroadcast it to (3,4,1,1), we need to sum the 2nd and 3rd dimension with keepdim True

            originally_ones = tuple(
                [axis for axis, size in enumerate(shape) if size == 1]
            )
            if len(originally_ones) != 0:
                correct_adjoint = correct_adjoint.sum(
                    dims=originally_ones, keepdims=True
                )

        return correct_adjoint

    def ones_(self) -> None:
        """Modify all values in the tensor to be 1.0."""
        self.__set(1.0)

    def zeros_(self) -> None:
        """Modify all values in the tensor to be 0.0."""
        self.__set(0.0)

    def numel(self) -> int:
        return self._numpy_data.size

    def relu(self) -> Union[TensorData, np.ndarray]:
        """Return a new TensorData object with the ReLU of each element."""
        return TensorData(
            *self._numpy_data.shape,
            numpy_data=np.maximum(self._numpy_data, 0),
        )

    def sigmoid(self) -> TensorData:
        """Return a new TensorData object with the sigmoid of each element."""
        return TensorData(
            *self._numpy_data.shape,
            numpy_data=1 / (1 + np.exp(-self._numpy_data)),
        )

    def permute(self, *dims: int) -> TensorData:
        """Return an aliased TensorData object with a permutation of its original dimensions permuted"""
        return TensorData(
            *self._numpy_data.shape,
            numpy_data=np.transpose(self._numpy_data, tuple(dims)),
        )

    @property
    def T(self) -> TensorData:
        """Return an aliased TensorData object with the transpose of the tensor."""

        return TensorData(
            *self._numpy_data.shape,
            numpy_data=np.transpose(self._numpy_data),
        )

    def __set(self, val) -> None:
        """Internal method to set all values in the TensorData to val."""
        self._numpy_data = np.full(self._numpy_data.shape, val)

    def __add__(self, rhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise addition: self + rhs."""
        if isinstance(rhs, TensorData):
            res = self._numpy_data + rhs._numpy_data
            return TensorData(
                *res.shape,
                numpy_data=res,
            )
        else:
            return TensorData(
                *self._numpy_data.shape,
                numpy_data=self._numpy_data + rhs,
            )

    def __radd__(self, lhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise addition is commutative: lhs + self."""
        return self + lhs

    def __sub__(self, rhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise subtraction: self - rhs."""
        return -rhs + self

    def __rsub__(self, lhs: Union[float, int, TensorData]) -> TensorData:
        """Self as RHS in element-wise subtraction: lhs - self."""
        return -self + lhs

    def __mul__(self, rhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise multiplication: self * rhs."""
        if isinstance(rhs, TensorData):
            if not rhs.use_numpy:
                raise TypeError("Incompatible TensorData Options")
            res = self._numpy_data * rhs._numpy_data
            return TensorData(
                *res.shape,
                numpy_data=res,
            )
        else:
            return TensorData(
                *self._numpy_data.shape,
                numpy_data=self._numpy_data * rhs,
            )

    def __rmul__(self, lhs: float | int | TensorData) -> TensorData:
        """Element-wise multiplication is commutative: lhs * self."""
        return self * lhs

    def __pow__(self, rhs: float | int) -> TensorData:
        """Element-wise exponentiation: self ** rhs."""
        return TensorData(
            *self._numpy_data.shape,
            numpy_data=self._numpy_data**rhs,
        )

    def __truediv__(self, rhs: float | int | TensorData) -> TensorData:
        """Element-wise division: self / rhs."""
        return self * rhs**-1

    def __rtruediv__(self, lhs: float | int | TensorData) -> TensorData:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self**-1

    def __neg__(self) -> TensorData:
        """Element-wise unary negation: -self."""
        return self * -1

    def __gt__(self, rhs: float | int | TensorData) -> TensorData:
        """Element-wise comparison: self > rhs."""
        if isinstance(rhs, TensorData):
            res = self._numpy_data > rhs._numpy_data
            return TensorData(
                *res.shape,
                numpy_data=res,
            )
        else:
            return TensorData(
                *self._numpy_data.shape,
                numpy_data=self._numpy_data > rhs,
            )

    def exp(self) -> TensorData:
        return TensorData(
            *self._numpy_data.shape,
            numpy_data=np.exp(self._numpy_data),
        )

    def __matmul__(self, rhs: TensorData) -> TensorData:
        res = self._numpy_data @ rhs._numpy_data
        return TensorData(
            *res.shape,
            numpy_data=res,
        )
