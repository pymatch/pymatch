from __future__ import annotations
import itertools
import math
from math import ceil, prod
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from copy import deepcopy
from typing import Callable, Union
from .util import (
    relu,
    sigmoid,
    is_permutation,
    get_common_broadcast_shape,
    matmul_2d,
    all_coordinates,
    dot,
)
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
        dtype: type = float,
        use_numpy: bool = False,
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
        self.use_numpy: bool = use_numpy
        self.shape: tuple[int] = size
        self.dtype: type = dtype
        if self.use_numpy:
            self._numpy_data = numpy_data
        self.__initialize_tensor_data(value)
        if not self.use_numpy:
            self.__initialize_strides()

    def __initialize_strides(self) -> None:
        """Initializes the strides of the dimensions of the TensorData.

        The stride of a dimension is the number of elements to skip in
        order to access the next element in that dimension. For instance,
        if the shape of a Tensor was (3,2,3), the stride of the 1st
        dimension would be the number of elements between [0,0,0] and [1,0,0],
        the stride of the 2nd dimension would be the number of elements
        between [0,0,0] and [0,1,0], and so on. Computationally, the
        stride of a dimension is the product of all following dimensions.

        For example, if the shape of the TensorData object was (3,2,3),
        the desired tuple of strides would be (6,3,1).
        """
        if not self._data:
            self._strides = ()
            return

        strides = []
        current_stride = len(self._data)
        for dim in self.shape:
            current_stride /= dim
            strides.append(current_stride)
        self._strides = tuple(strides)

    def __initialize_tensor_data(self, value) -> None:
        """Helper method that initializes all the values in the TensorData object."""
        if self.use_numpy:
            self._item = None
            if self._numpy_data is None:
                self._numpy_data: np.ndarray = np.full(self.shape, value)
        if self.shape:
            self._item = None
            self._data = [TensorData(value=value) for _ in range(prod(self.shape))]
        else:
            self._item = value
            self._data = None

    def __out_of_bounds_coords(self, coords: tuple):
        """Helper method that checks if a set of coordinates is out of bounds of the
        shape of the TensorData object.

        Raises:
            IndexError: raises index error if the number of dimensions so not match the
            shape, or if any of the individual coordinates are less than 0 or greater
            than or equal to the shape's dimension.
        """
        if len(coords) != len(self.shape):
            raise IndexError(f"Too many dimensions, expected {len(self.shape)}.")
        if any(i < 0 or i >= j for i, j in zip(coords, self.shape)):
            raise IndexError("Index out of bounds")

    def __out_of_bounds_index(self, index: int):
        """Helper method to check if index is out of bounds for self._data

        Raises:
            IndexError: raises index error if index provided is less than 0 or greater
            than or equal to the length of the data list.
        """
        if index < 0 or index >= len(self._data):
            raise IndexError("Index out of bounds")

    def __single_to_multi_rank_translation(self, index: int) -> tuple:
        """Helper method to translate index into the data list into the corresponding
        coordinates of the Tensor."""
        self.__out_of_bounds_index(index)
        coordinates = []
        for shape_dim in reversed(self.shape):
            temp_index = int(index % shape_dim)
            coordinates.append(temp_index)
            index -= temp_index
            index /= shape_dim

        return tuple(reversed(coordinates))

    def __multi_to_single_rank_translation(self, coords: tuple) -> int:
        """Helper method to translate the coordinates into an index into the data list."""
        self.__out_of_bounds_coords(coords)
        return int(sum(dim * stride for dim, stride in zip(coords, self._strides)))

    def item(self) -> Union[int, float]:
        """Returns the item of a singleton, or single element TensorData object.

        Raises:
            ValueError: raises value error if attempting to access the item of a
            TensorData object with more than one element.
        """
        if self.use_numpy:
            return self._numpy_data.item(0)
        if self._item != None:
            return self._item
        elif len(self._data) == 1:
            return self._data[0]._item
        raise ValueError(
            "only one element tensors can be converted into python scalars"
        )

    def reshape_(self, shape: tuple):
        """Helper method to reshape the TensorData object inplace, without changing the data.

        Raises:
            RuntimeError: raises runtime error if the product of the dimensions of
            the new shape is not equal to the existing number of elements.
        """
        if self.use_numpy:
            self._numpy_data = self._numpy_data.reshape(shape)
            return
        # If the current shape is equal to the desired shape, then do nothing.
        if shape == self.shape:
            return

        # Reshape a singleton into a 1D Tensor.
        if self._data is None:
            if shape != (1,):
                raise RuntimeError(f"shape {shape} is invalid for input of size 1")
            self.shape = (1,)
            self._data = [TensorData(value=self._item)]
            self._item = None
        # Reshape a 1D Tensor into a singleton where shape = ().
        elif len(self._data) == 1 and not shape:
            self.shape = ()
            self._item = self._data[0]._item
            self._data = None
        else:
            if prod(shape) != len(self._data):
                raise RuntimeError(
                    f"shape {shape} is invalid for input of size {len(self._data)}"
                )
            self.shape = shape
            self._item = None
        # The strides change when the shape does, so they must be reinitialized.
        self.__initialize_strides()

    def reshape(self, shape: tuple) -> TensorData:
        """Helper method to reshape and return a new TensorData object without changing the data"""
        if self.use_numpy:
            return TensorData(
                *shape,
                dtype=self.dtype,
                use_numpy=True,
                numpy_data=self._numpy_data.reshape(shape),
            )
        # Reshape a singleton into a 1D Tensor.
        if self._data is None:
            if shape != (1,):
                raise RuntimeError(f"shape {shape} is invalid for input of size 1")
            new_tensor = TensorData(1)
            new_tensor._data[0] = self
            return new_tensor

        # Reshape a 1D Tensor into a singleton.
        if len(self._data) == 1 and not shape:
            return self._data[0]

        # An instantiation of a TensorData object will make prod(self.shape)
        # singleton TensorData objects. As we do not need to create these
        # extra objects, we initialize the new tensor to have a shape of (0,)
        # so no singleton is created in new_tensor._data.
        new_tensor = TensorData(0)
        new_tensor._data = self._data
        new_tensor.reshape_(shape)
        return new_tensor

    def __convert_slice_to_index_list(self, coords):
        """Converts a list of slices or integers to a list of possible indices for each dimension.

        Args:
            coords: A list of slices or integers.

        Raises:
            ValueError: raises value error if the index is type other than int or slice

        Returns:
            A tuple of two lists:
                * The first list contains the output shape for each dimension.
                * The second list contains a list of possible indices for each dimension.
        """
        output_shape = []
        possible_indices = []
        for i in range(len(self.shape)):
            if i >= len(coords):
                output_shape.append(self.shape[i])
                possible_indices.append(range(self.shape[i]))
                continue

            coordinate = coords[i]
            if isinstance(coordinate, slice):
                start = coordinate.start or 0
                stop = coordinate.stop or self.shape[i]
                step = coordinate.step or 1

                output_shape.append(ceil((stop - start) / step))
                possible_indices.append(range(start, stop, step))

            elif isinstance(coordinate, int):
                # We are not updating the output shape because we are slicing the input tensor with a single integer.
                # This means that we are reducing the dimensionality of the tensor.
                possible_indices.append([coordinate])
            else:
                raise ValueError("can only be ints or slices")

        return output_shape, possible_indices

    def __getitem__(self, coords):
        """Retrieves a subtensor from the current tensor using the given coordinates.

        Args:
            coords: A tuple of slices or integers.

        Returns:
            A new tensor containing the subtensor specified by the given coordinates.
        """
        if not isinstance(coords, tuple):
            coords = (coords,)

        if self.use_numpy:
            res = self._numpy_data[*coords]
            return TensorData(*res.shape, use_numpy=True, numpy_data=res)

        output_shape, possible_indices = self.__convert_slice_to_index_list(coords)

        if not output_shape:
            # The output_shape is empty, [], if and only if all indices in `coords` were integers
            # indicating only one item should be retrieved.
            self.__out_of_bounds_coords(coords)
            return self._data[self.__multi_to_single_rank_translation(coords)]

        output_data = []
        for index in itertools.product(*possible_indices):
            self.__out_of_bounds_coords(index)
            output_data.append(
                self._data[self.__multi_to_single_rank_translation(index)]
            )

        output_tensor = TensorData(*output_shape)
        output_tensor._data = output_data
        return output_tensor

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

        if self.use_numpy:
            if isinstance(value, TensorData):
                raise TypeError("Incompatible TensorData options. Numpy and TensorData")
            self._numpy_data[*coords] = value

        output_shape, possible_indices = self.__convert_slice_to_index_list(coords)

        if not output_shape:
            self.__out_of_bounds_coords(coords)
            if isinstance(value, TensorData):
                self._data[self.__multi_to_single_rank_translation(coords)]._item = (
                    value.item()
                )
            elif isinstance(value, (int, float)):
                self._data[self.__multi_to_single_rank_translation(coords)]._item = (
                    self.dtype(value)
                )
            else:
                raise TypeError("invalid type to set value in tensor data")
        else:
            if isinstance(value, (int, float)):
                for i, index in enumerate(itertools.product(*possible_indices)):
                    self.__out_of_bounds_coords(index)
                    self._data[self.__multi_to_single_rank_translation(index)]._item = (
                        self.dtype(value)
                    )
            elif isinstance(value, TensorData):
                for i, index in enumerate(itertools.product(*possible_indices)):
                    self.__out_of_bounds_coords(index)
                    self._data[self.__multi_to_single_rank_translation(index)]._item = (
                        value._data[i]._item
                    )

    # TODO(SRM47) Update the repr to make it look like PyTorch
    def __repr__(self):
        if self.use_numpy:
            return self._numpy_data.__repr__()

        return self._data.__repr__() if self._item is None else self._item.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def __translate(self, *coord: int) -> tuple:
        """Helper method for broadcasting for mapping a coordinate in a new tensor to the existing tensor"""
        coord = coord[0]

        res = [0] * len(self.shape)

        for i in range(len(res) - 1, -1, -1):
            # If the shape at the current index is 1, then access the 0th element in that dimension.
            # If the shape isn't one at that dimension, then we grab the coordinate at that dimension as usual.
            res[i] = (
                0 if self.shape[i] == 1 else coord[i + len(coord) - len(self.shape)]
            )

        return tuple(res)

    def __validate_broadcast(self, shape):
        """Helper function to determine whether self TensorData can be broadcasted
        to desired shape."""
        for s1, s2 in zip(reversed(self.shape), reversed(shape)):
            # We exclude when s2 == 1 because self is broadcastable to shape iff
            # any dimension in self.shape is 1. If a dimension in the desired shape
            # is 1 and self.shape is more than 1, then self can't be broadcasted
            # to the new shape.
            if not (s1 == 1 or s1 == s2):
                raise ValueError("Incompatible dimensions for broadcasting")

    # TODO(SAM): Create an explanation for this algorithm.
    def broadcast(self, *shape: int):
        """Broadcasts the TensorData to the desired shape.

        Args:
            shape: The desired shape of the tensor.

        Returns:
            A new tensor with the desired shape.

        Raises:
            ValueError: If the tensor cannot be broadcast to the desired shape.
        """
        if self.shape == shape:
            return self

        if len(shape) < len(self.shape):
            raise RuntimeError(
                "The number of sizes provided must be greater or equal to the number of dimensions in the tensor."
            )

        # Account for the case when we're broadcasting a singleton
        if not self._data:
            value = self._item
            new_tensor = TensorData(*shape)
            for elem in new_tensor._data:
                elem._item = value
            return new_tensor

        self.__validate_broadcast(shape)

        new_tensor = TensorData(*shape[-1 * len(self.shape) :], dtype=self.dtype)
        # Loop through all coordinates in the new tensor
        for i, new_tensor_index in enumerate(new_tensor.__all_coordinates()):
            # For every new coordinate in the new tensor, see what coordinate in the original tensor has its supposed value
            translated_index = self.__translate(new_tensor_index)
            single_index = self.__multi_to_single_rank_translation(translated_index)
            # new_tensor[new_coordinate] = old_tensor[new_coordinate_translated_to_old_coordinate_system].
            new_tensor._data[i]._item = self._data[single_index].item()

        remianing_dimensions = shape[: -1 * len(self.shape)]

        new_data = []
        for _ in range(prod(remianing_dimensions)):
            new_data.extend(deepcopy(new_tensor._data))

        new_tensor._data = new_data
        new_tensor.reshape_(shape)

        return new_tensor

    def __all_coordinates(self):
        return all_coordinates(self.shape)

    def sum(self, dims: tuple | int = None, keepdims: bool = False) -> TensorData:
        """Return a new TensorData object summed along the axis
         if coord is (2,3) and sum alon axis=1, then we increment the value at (2,0) in the new tensor by the
        value at (2,3) in the original tensor. initialize new tensor to be all zeros with the new shape.
        keepdims = True by default (as far as algorithm is cocnerned), then keepdims = false will simply call a
        reshape
        for a given coordinate, set all dimensins to 0 for every axis we're trying to sum over

        initialize new tensor with new shape (depending on keepdims) to all 0's
        for (x1, x2, ...) in orig.all_coordinate()
        new_coordinate = [0 if dim in dims else val for dim, val in enumerate((x1, x2, ...))]
        new_tensor[new_coordinate] += orig[(x1, x2, ...)]

        return new_tensor
        """
        if isinstance(dims, int):
            dims = (dims,)
        if self.use_numpy:
            res = self._numpy_data.sum(axis=dims, keepdims=keepdims)
            return TensorData(*res.shape, use_numpy=True, numpy_data=res)

        if not dims:
            # Handle case where dims is None. If None, all dimensions are reduced into a singleton tensor
            return TensorData(value=sum(td._item for td in self._data))
            # Case where dims is an int or tuple
        if any(ax < 0 or ax >= len(self.shape) for ax in dims):
            raise ValueError(f"dims should be in bounds 0 and {len(self.shape)}-1")

        new_shape = tuple(
            1 if dim in dims else val for dim, val in enumerate(self.shape)
        )
        new_tensor = TensorData(*new_shape, value=0)  # Must initialize to 0
        # for every coordinate in the original tensor
        for orig_index, orig_coord in enumerate(self.__all_coordinates()):
            new_coord = tuple(
                0 if dim in dims else val for dim, val in enumerate(orig_coord)
            )
            new_coord_index = new_tensor.__multi_to_single_rank_translation(new_coord)
            new_tensor._data[new_coord_index]._item += self._data[orig_index]._item

        if not keepdims:
            # remove the dimensions that were summed out (changed to 1's), i.e., the dimensions in dims
            new_shape = []
            for dim, val in enumerate(self.shape):
                if dim not in dims:
                    new_shape.append(val)
            new_tensor.reshape(tuple(new_shape))

        return new_tensor

    def mean(self, dims: tuple | int = None, keepdims: bool = None) -> TensorData:
        """Compute the mean of all values in the tensor."""
        return self.sum(dims, keepdims) / self.numel

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
        return self._numpy_data.size() if self.use_numpy else len(self._data)

    def relu(self) -> Union[TensorData, np.ndarray]:
        """Return a new TensorData object with the ReLU of each element."""
        if self.use_numpy:
            return TensorData(
                *self._numpy_data.shape,
                dtype=self.dtype,
                use_numpy=True,
                # This is the ReLU
                numpy_data=np.maximum(self._numpy_data, 0),
            )
        new_tensor = TensorData(*self.shape)
        for i in range(len(new_tensor._data)):
            new_tensor._data[i]._item = relu(self._data[i]._item)
        return new_tensor

    def sigmoid(self) -> TensorData:
        """Return a new TensorData object with the sigmoid of each element."""
        if self.use_numpy:
            return TensorData(
                *self._numpy_data.shape,
                dtype=self.dtype,
                use_numpy=True,
                # This is the ReLU
                numpy_data=sigmoid(self._numpy_data),
            )
        new_tensor = TensorData(*self.shape)
        for i in range(len(new_tensor._data)):
            new_tensor._data[i]._item = sigmoid(self._data[i]._item)
        return new_tensor

    def permute(self, *dims: int) -> TensorData:
        """Return an aliased TensorData object with a permutation of its original dimensions permuted"""
        if self.use_numpy:
            return TensorData(
                *self._numpy_data.shape,
                dtype=self.dtype,
                use_numpy=True,
                # This is the ReLU
                numpy_data=np.transpose(self._numpy_data, tuple(dims)),
            )

        if not is_permutation([i for i in range(len(self.shape))], dims):
            print(dims)
            raise RuntimeError(
                "provided dimension tuple is not a valid permutation of the column indices of this tensor"
            )

        # Make the new shape.
        # If permuting (3,4,5) with the new permutation (2,0,1) would be (5,3,4).
        new_shape = [self.shape[dim] for dim in dims]
        # Make a new tensor with that shape.
        new_tensor = TensorData(*new_shape)
        # Iterate through all of the element in this tensor.
        for index, coord in enumerate(self.__all_coordinates()):
            # If the original coordinate was [1,2,3], and we permute it to (2,0,1),
            # then the new, translated coordinate is [3,1,2].
            translated_coord = tuple(coord[dim] for dim in dims)

            # Look up which index the translated coordinate maps to in new_tensor._data.
            translated_index = new_tensor.__multi_to_single_rank_translation(
                translated_coord
            )
            new_tensor._data[translated_index] = self._data[index]

        return new_tensor

    @property
    def T(self) -> TensorData:
        """Return an aliased TensorData object with the transpose of the tensor."""
        # Transpose is the same as permuting the tensor with the reverse of its dimensions
        if self.use_numpy:
            return TensorData(
                *self._numpy_data.shape,
                dtype=self.dtype,
                use_numpy=True,
                # This is the ReLU
                numpy_data=np.transpose(self._numpy_data),
            )
        return self.permute(*reversed(range(len(self.shape))))

    def __set(self, val) -> None:
        """Internal method to set all values in the TensorData to val."""
        if self.use_numpy:
            self._numpy_data = np.full(self._numpy_data.shape, val)
            return
        if not self._data:
            self._item = val
        else:
            for td in self._data:
                td._item = val

    def __binary_op(
        self, op: Callable, rhs: Union[float, int, TensorData]
    ) -> TensorData:
        """Internal method to perform an element-wise binary operation on the TensorData object.

        This method will automatically broadcast inputs when necessary.
        """
        # Handle the case where rhs is a scalar or is a singleton.
        if isinstance(rhs, (float, int)) or not rhs._data:
            # If rhs isn't a number, then it's a singleton TensorData object, so the value should be its item.
            value = rhs if isinstance(rhs, (float, int)) else rhs._item
            # Handle case where self is a singleton
            if not self._data:
                return TensorData(value=op(self._item, value))
            new_tensor = TensorData(*self.shape)
            for i, elem in enumerate(new_tensor._data):
                elem._item = op(self._data[i]._item, value)
            return new_tensor

        broadcast_to = get_common_broadcast_shape(self.shape, rhs.shape)

        # Create empty output
        out = TensorData(*broadcast_to)
        # The broadcast method returns a new object
        lhs = self.broadcast(*broadcast_to)
        rhs = rhs.broadcast(*broadcast_to)
        # Compute the binary operation for every element
        for i, elem in enumerate(out._data):
            elem._item = op(lhs._data[i]._item, rhs._data[i]._item)

        return out

    def __add__(self, rhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise addition: self + rhs."""
        if self.use_numpy:
            if isinstance(rhs, TensorData):
                if not rhs.use_numpy:
                    raise TypeError("Incompatible TensorData Options")
                res = self._numpy_data + rhs._numpy_data
                return TensorData(
                    *res.shape,
                    dtype=self.dtype,
                    use_numpy=True,
                    numpy_data=res,
                )
            else:
                return TensorData(
                    *self._numpy_data.shape,
                    dtype=self.dtype,
                    use_numpy=True,
                    numpy_data=self._numpy_data + rhs,
                )

        return self.__binary_op(add, rhs)

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
        if self.use_numpy:
            if isinstance(rhs, TensorData):
                if not rhs.use_numpy:
                    raise TypeError("Incompatible TensorData Options")
                res = self._numpy_data * rhs._numpy_data
                return TensorData(
                    *res.shape,
                    dtype=self.dtype,
                    use_numpy=True,
                    numpy_data=res,
                )
            else:
                return TensorData(
                    *self._numpy_data.shape,
                    dtype=self.dtype,
                    use_numpy=True,
                    numpy_data=self._numpy_data * rhs,
                )

        return self.__binary_op(mul, rhs)

    def __rmul__(self, lhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise multiplication is commutative: lhs * self."""
        return self * lhs

    def __truediv__(self, rhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise division: self / rhs."""
        return self * rhs**-1

    def __rtruediv__(self, lhs: Union[float, int, TensorData]) -> TensorData:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self**-1

    def __pow__(self, rhs: Union[float, int]) -> TensorData:
        """Element-wise exponentiation: self ** rhs."""
        if self.use_numpy:
            return TensorData(
                *self._numpy_data.shape,
                dtype=self.dtype,
                use_numpy=True,
                numpy_data=self._numpy_data**rhs,
            )
        return self.__binary_op(pow, rhs)

    def __neg__(self) -> TensorData:
        """Element-wise unary negation: -self."""
        return self * -1

    def __gt__(self, rhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise comparison: self > rhs."""
        if self.use_numpy:
            if isinstance(rhs, TensorData):
                if not rhs.use_numpy:
                    raise TypeError("Incompatible TensorData Options")
                res = self._numpy_data > rhs._numpy_data
                return TensorData(
                    *res.shape,
                    dtype=self.dtype,
                    use_numpy=True,
                    numpy_data=res,
                )
            else:
                return TensorData(
                    *self._numpy_data.shape,
                    dtype=self.dtype,
                    use_numpy=True,
                    numpy_data=self._numpy_data > rhs,
                )
        return self.__binary_op(gt, rhs)

    def exp(self) -> TensorData:
        if self.use_numpy:
            return TensorData(
                *self._numpy_data.shape,
                dtype=self.dtype,
                use_numpy=True,
                numpy_data=np.exp(self._numpy_data),
            )
        else:
            # Handle case where self is a singleton
            if not self._data:
                return TensorData(value=math.exp(self._item))
            # Handle case where self is a non-singleton tensordata
            new_tensor = TensorData(*self.shape)
            for i, elem in enumerate(new_tensor._data):
                elem._item = math.exp(self._data[i]._item)
            return new_tensor

    @property
    def vals(self) -> list:
        if not self._data:
            return [self._item]

        return [td.item() for td in self._data]

    def __matmul__(self, rhs: TensorData) -> TensorData:
        """N-dimensional tensor multiplication

        1. If both tensors are 1-dimensional, the dot product (scalar) is returned.

        2. If both arguments are 2-dimensional, the matrix-matrix product is returned.

        3. If the first argument is 1-dimensional and the second argument is 2-dimensional,
           a 1 is prepended to its dimension for the purpose of the matrix multiply.
           After the matrix multiply, the prepended dimension is removed.

        4. If the first argument is 2-dimensional and the second argument is 1-dimensional,
           the matrix-vector product is returned.

        5. If both arguments are at least 1-dimensional and at least one argument is N-dimensional
           (where N > 2), then a batched matrix multiply is returned. If the first argument
           is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched
           matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is
           appended to its dimension for the purpose of the batched matrix multiple and removed after.
           The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable).

        See https://pytorch.org/docs/stable/generated/torch.matmul.html for more information
        """
        if self.use_numpy:
            if not rhs.use_numpy:
                raise TypeError("Incompatible TensorData Options")
            res = self._numpy_data @ rhs._numpy_data
            return TensorData(
                *res.shape,
                dtype=self.dtype,
                use_numpy=True,
                numpy_data=res,
            )
        lhs = self
        lhs_shape, rhs_shape = self.shape, rhs.shape
        lhs_dims, rhs_dims = len(lhs_shape), len(rhs_shape)

        # If both tensors are 1-dimensional, the dot product (scalar) is returned.
        if lhs_dims == 1 and rhs_dims == 1:
            if len(lhs._data) == len(rhs._data):
                # dot product
                return dot(self._data, rhs._data)

        # If both arguments are 2-dimensional, the matrix-matrix product is returned.
        elif lhs_dims == 2 and rhs_dims == 2:
            result_shape, result_data = matmul_2d(
                lhs._data, lhs.shape, rhs._data, rhs.shape
            )
            new_tensor = TensorData(0)
            new_tensor._data = result_data
            new_tensor.reshape_(result_shape)
            return new_tensor

        # If the first argument is 1-dimensional and the second argument is 2-dimensional...
        elif lhs_dims == 1 and rhs_dims == 2:
            # ... a 1 is prepended to its dimension for the purpose of the matrix multiply...
            result_shape, result_data = matmul_2d(
                lhs._data, (1,) + lhs.shape, rhs._data, rhs.shape
            )
            new_tensor = TensorData(0)
            new_tensor._data = result_data
            # ... and after the matrix multiply, the prepended dimension is removed.
            new_tensor.reshape_((result_shape[1],))
            return new_tensor

        # If the first argument is 2-dimensional and the second argument is 1-dimensional,
        # the matrix-vector product is returned.
        elif lhs_dims == 2 and rhs_dims == 1:
            # A 1 is appended to its dimension for the purpose of the matrix multiply...
            result_shape, result_data = matmul_2d(
                lhs._data, lhs.shape, rhs._data, rhs.shape + (1,)
            )
            new_tensor = TensorData(0)
            new_tensor._data = result_data
            # ... and after the matrix multiply, the appended dimension is removed.
            new_tensor.reshape_((result_shape[0],))
            return new_tensor

        # If both arguments are at least 1-dimensional and at least one argument is
        # N-dimensional (where N > 2), then a batched matrix multiply is returned.
        elif (lhs_dims >= 1 and rhs_dims >= 1) and (lhs_dims > 2 or rhs_dims > 2):
            # If the first argument is 1-dimensional, a 1 is prepended to its dimension
            # TODO(SAM): make a copy of lhs and rhs and use that instead of shaping the original object
            if lhs_dims == 1:
                lhs.reshape_((1,) + lhs_shape)
            # If the second argument is 1-dimensional, a 1 is appended to its dimension
            if rhs_dims == 1:
                rhs.reshape_(rhs_shape + (1,))

            # The non-matrix (i.e. batch) dimensions are broadcasted
            lhs_non_matrix_dims, lhs_matrix_dims = lhs.shape[:-2], lhs.shape[-2:]
            rhs_non_matrix_dims, rhs_matrix_dims = rhs.shape[:-2], rhs.shape[-2:]

            common_broadcast_shape = get_common_broadcast_shape(
                lhs_non_matrix_dims, rhs_non_matrix_dims
            )

            # The number of elements in each matrix
            lhs_matrix_size = prod(lhs_matrix_dims)
            rhs_matrix_size = prod(rhs_matrix_dims)
            result_data = []
            for coord in all_coordinates(common_broadcast_shape):
                # Look up the index in lhs._data where the matrix we're multiplying should start
                # Of the dimensions except for the last two, non matrix dimensions, see what that coordinate would translate to in the lhs coordinate system
                # if the coord is (0,1) in the new combined tensor, and the lhs dimensions are (4,1), then I have to translate the (0,1) into (0,0)
                coord_unbroadcasted_lhs = tuple(
                    reversed(
                        tuple(
                            coord[i] if lhs_non_matrix_dims[i] != 1 else 0
                            for i in range(-1, -len(lhs_non_matrix_dims) - 1, -1)
                        )
                    )
                )
                # Get combine the previous translated coordinate with 0,0, to grab the starting point of the matrix
                # then grab the index
                # we dont want to use get set item for increased efficiency
                lhs_matrix_start = lhs.__multi_to_single_rank_translation(
                    coord_unbroadcasted_lhs + (0, 0)
                )

                coord_unbroadcasted_rhs = tuple(
                    reversed(
                        tuple(
                            coord[i] if rhs_non_matrix_dims[i] != 1 else 0
                            for i in range(-1, -len(rhs_non_matrix_dims) - 1, -1)
                        )
                    )
                )
                rhs_matrix_start = rhs.__multi_to_single_rank_translation(
                    coord_unbroadcasted_rhs + (0, 0)
                )

                lhs_matrix = lhs._data[
                    lhs_matrix_start : lhs_matrix_start + lhs_matrix_size
                ]
                rhs_matrix = rhs._data[
                    rhs_matrix_start : rhs_matrix_start + rhs_matrix_size
                ]
                _, local_matrix_prod = matmul_2d(
                    lhs_matrix, lhs_matrix_dims, rhs_matrix, rhs_matrix_dims
                )
                result_data += local_matrix_prod

            # Revert lhs and rhs back to orginal dimension if changed previously and remove the extra dimension
            new_shape = tuple(common_broadcast_shape)

            if lhs_dims == 1:
                lhs.reshape_(lhs_shape)
            else:
                new_shape += (lhs_matrix_dims[0],)

            if rhs_dims == 1:
                rhs.reshape_(rhs_shape)
            else:
                new_shape += (rhs_matrix_dims[1],)

            new_tensor = TensorData(0)
            new_tensor._data = result_data
            new_tensor.reshape_(new_shape)

            return new_tensor
