from __future__ import annotations
import itertools
import math
from math import ceil, prod
from operator import add, gt, mul, pow
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


class TensorData:
    """A storage and arithmetic object for n-dimensional tensor.

    The TensorData class provides a basic implementation for storing and performing
    operations on  n-dimensional tensors. While not optimized for performance, it's
    useful for understanding fundamental tensor concepts.

    TensorData objects are recursive:
        Leaf Nodes: A TensorData object can hold a single value, accessible via .item().
        Internal Nodes: A TensorData object can hold a list of other Leaf TensorData
        objects, allowing the representation of multi-dimensional tensors.

    Accessing Elements
    Use standard coordinate notation to retrieve data (e.g., x[1, 3, 2]). The TensorData
    class internally translates these coordinates into the correct index within its data list.
    """

    def __init__(
        self,
        *size: int,
        value: float = 0.0,
    ) -> None:
        """Create a new TensorData object to store an n-dimensional tensor of values.

        Args:
            size (tuple[int]): The shape of the Tensor
            value (float): The default value of each element in the Tensor
            dtype (type): The type of the values in the Tensor
        """
        # Ensures that 'size' contains only integer dimensions, raising an error otherwise.
        assert all(
            isinstance(dim, int) for dim in size
        ), f"Size {size} must be a variadict of only integers"
        # Stores the tensor's shape as a tuple of integers for future reference.
        self.shape: tuple[int] = size
        # Initializes the tensor with data from 'value'.
        self.__initialize_tensor_data(value)
        # Calculates memory access strides for efficient operations based on the shape.
        self.__initialize_strides()

    @staticmethod
    def create_tensor_from_data(data: list, shape: tuple) -> TensorData:
        """_summary_

        Args:
            data (list): _description_
            shape (tuple): _description_

        Returns:
            TensorData: _description_
        """
        new_tensor = TensorData()
        new_tensor._data = data
        new_tensor.reshape_(shape)
        return new_tensor

    def __initialize_tensor_data(self, value) -> None:
        """Helper method that initializes all the values in the TensorData object.

        Args:
            value (Number): The numerical value the tensor will be initialized to.
        """
        if self.shape:
            # Case 1: Tensor has a defined shape (not a single value).
            self._item = None  # Indicate this node doesn't store a single value.

            # Create a list containing child TensorData objects. The total number of
            # child nodes is the product of shape dimensions. Each child node is
            # initialized with the provided 'value'.
            self._data = [TensorData(value=value) for _ in range(prod(self.shape))]
        else:
            # Case 2: Tensor represents a single value.
            self._item = value  # Store the value directly.
            self._data = None  # Indicate there's no list of child nodes.

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

        See `__multi_to_single_rank_translation` for example stride usage.
        """
        # Handle the case where the tensor holds a single value
        if not self._data:
            # No strides needed for a single value.
            self._strides = None
            return

        strides = []
        # Start with the total number of elements.
        current_stride = len(self._data)

        # Iterate over each dimension to calculate strides.
        for dim in self.shape:
            # Divide by the current dimension's size.
            current_stride /= dim
            # Add the calculated stride to the list.
            strides.append(current_stride)
        # Store the strides as a tuple.
        self._strides = tuple(strides)

    def __out_of_bounds_coords(self, coords: tuple[int]):
        """Helper method that checks if a set of coordinates is out of bounds of the
        shape of the TensorData object.

        Raises:
            IndexError: raises index error if the number of dimensions so not match the
            shape, or if any of the individual coordinates are less than 0 or greater
            than or equal to the shape's dimension.
        """
        if len(coords) != len(self.shape):
            # Error check: Ensure the number of coordinates matches the number of dimensions
            raise IndexError(f"Too many dimensions, expected {len(self.shape)}.")
        if any(i < 0 or i >= j for i, j in zip(coords, self.shape)):
            # Error check: Ensure each coordinate is within the bounds of its corresponding dimension
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

    def __multi_to_single_rank_translation(self, coords: tuple[int]) -> int:
        """Helper method to translate the n-dimensional coordinates into an index into the data list.

        Args:
            coords (tuple[int]): The coordinates to convert.

        Returns:
            int: The index into the data list.
        """
        # Verify that the provided coordinates are not out of bounds.
        self.__out_of_bounds_coords(coords)
        return int(sum(dim * stride for dim, stride in zip(coords, self._strides)))

    def item(self) -> int | float:
        """Returns the item of a singleton, or single element TensorData object.

        Raises:
            ValueError: raises value error if attempting to access the item of a
            TensorData object with more than one element.
        """
        if self._item != None:
            return self._item
        elif len(self._data) == 1:
            return self._data[0]._item
        raise ValueError(
            f"only single element tensors can be converted into python scalars."
        )

    def reshape_(self, shape: tuple):
        """Helper method to reshape the TensorData object inplace, without changing the data.

        Args:
            shape (tuple): The shape to change to.

        Raises:
            RuntimeError: raises runtime error if the product of the dimensions of
            the new shape is not equal to the existing number of elements.
        """
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
        """Helper method to reshape and return a new TensorData object without changing the data.

        Args:
            shape (tuple): The shape of the new tensor

        Raises:
            RuntimeError: raises runtime error if the product of the dimensions of
            the new shape is not equal to the existing number of elements.

        Returns:
            TensorData: The new tensor with a shallow copy of the data, but the new shape
        """
        # Reshape a singleton into a 1D, single-element TensorData.
        if self._data is None:
            if shape != (1,):
                raise RuntimeError(f"shape {shape} is invalid for input of size 1")
            new_tensor = TensorData(1)
            new_tensor._data[0] = self
            return new_tensor

        # Reshape a single-element TensorData into a singleton.
        if len(self._data) == 1 and not shape:
            return self._data[0]

        # Create a new tensor from specified data and shape.
        return TensorData.create_tensor_from_data(self._data, shape)

    def __convert_slice_to_index_list(self, coordinates):
        """Converts a list of slices or integers to a list of possible indices for each dimension.

        Args:
            coordinates: A list of slices or integers.

        Raises:
            ValueError: If the index is not an integer or slice.

        Returns:
            A tuple containing:
                - A list representing the output shape for each dimension.
                - A list of possible indices for each dimension.
        """
        output_shape = []  # List to store the output shape for each dimension.
        possible_indices = []  # List to store the possible indices for each dimension.

        # Loop through each dimension of the tensor.
        for dim in range(len(self.shape)):
            # If the number of provided coordinates is less than the number of dimensions,
            # add the remaining dimensions to the output shape and all possible indices.
            if dim >= len(coordinates):
                output_shape.append(self.shape[dim])
                possible_indices.append(range(self.shape[dim]))
                continue

            coordinate = coordinates[dim]  # Current coordinate for this dimension.
            if isinstance(coordinate, slice):
                # Do not allow step size of 0.
                if coordinate.step == 0:
                    raise ValueError("slice step cannot be zero")
                # If the coordinate is a slice, calculate the start, stop, and step values.
                start = coordinate.start or 0
                stop = coordinate.stop or self.shape[dim]
                step = coordinate.step or 1

                # Calculate the size of this dimension in the output shape.
                output_shape.append(max(ceil((stop - start) / step), 0))
                # Generate possible indices within the slice range.
                possible_indices.append(range(start, stop, step))

            elif isinstance(coordinate, int):
                # If the coordinate is an integer, it means slicing with a single integer,
                # which reduces the dimensionality of the tensor.
                # Only the specified index will be considered.
                possible_indices.append([coordinate])
            else:
                raise ValueError("Indices must be integers or slices")

        return (
            tuple(output_shape),
            possible_indices,
        )  # Return the output shape and possible indices.

    def __getitem__(self, coords: tuple[int]):
        """Retrieves a subtensor from the current tensor using the given coordinates.

        Args:
            coords: A tuple of slices or integers.

        Returns:
            A new tensor containing the subtensor specified by the given coordinates.
        """
        if not isinstance(coords, tuple):
            coords = (coords,)

        output_shape, possible_indices = self.__convert_slice_to_index_list(coords)

        if not output_shape:
            # The output_shape is empty, [], if and only if all indices in `coords` were integers
            # indicating only one item should be retrieved.
            self.__out_of_bounds_coords(coords)
            return self._data[self.__multi_to_single_rank_translation(coords)]

        # Retrieve data for all possible index combinations using itertools.product.
        # itertools.product generates all possible combinations of the elements of the input iterables.
        # In this case, possible_indices is a list of lists, where each inner list represents possible indices for one dimension.
        # By passing *possible_indices to itertools.product, we unpack these lists and generate all combinations.
        # Each combination represents a unique set of indices for retrieving data from the tensor.
        output_data = [
            self._data[self.__multi_to_single_rank_translation(index)]
            for index in itertools.product(*possible_indices)
        ]

        # Create a new tensor from the retrieved data and shape.
        return TensorData.create_tensor_from_data(output_data, output_shape)

    def __setitem__(self, coordinates: tuple, value: TensorData | int | float) -> None:
        """Sets the value of a subtensor at the given coordinates.

        Args:
            coordinates: A tuple of slices or integers specifying the location of the subtensor.
            value: The new value to assign to the subtensor.

        Raises:
            TypeError: If the value is a list or an invalid type.
        """
        # Ensure the value is not a list, as lists cannot be assigned directly to TensorData.
        if isinstance(value, list):
            raise TypeError("Lists cannot be assigned to a TensorData.")

        # Ensure coordinates is a tuple, even if it's a single integer.
        if not isinstance(coordinates, tuple):
            coordinates = (coordinates,)

        # Convert slice coordinates to index lists.
        output_shape, possible_indices = self.__convert_slice_to_index_list(coordinates)

        if not output_shape:
            # If output_shape is empty, it means only one item should be set.
            self.__out_of_bounds_coords(coordinates)
            # Assign the value to the appropriate location.
            if isinstance(value, TensorData):
                self._data[
                    self.__multi_to_single_rank_translation(coordinates)
                ]._item = value.item()
            elif isinstance(value, (int, float)):
                self._data[
                    self.__multi_to_single_rank_translation(coordinates)
                ]._item = value
            else:
                raise TypeError("Invalid type to set value in TensorData.")
        else:
            # If output_shape is not empty, it means multiple items should be set.
            if isinstance(value, (int, float)):
                # Assign the same value to multiple locations.
                for index in itertools.product(*possible_indices):
                    self.__out_of_bounds_coords(index)
                    self._data[self.__multi_to_single_rank_translation(index)]._item = (
                        value
                    )
            elif isinstance(value, TensorData):
                if output_shape != value.shape:
                    raise RuntimeError(
                        f"The expanded size of the tensor must match the existing non-singleton dimension size. Target sizes: {output_shape}.  Tensor sizes: {value.shape}"
                    )
                # Assign values from another TensorData to multiple locations.
                for index, item in zip(
                    itertools.product(*possible_indices), value._data
                ):
                    self.__out_of_bounds_coords(index)
                    self._data[self.__multi_to_single_rank_translation(index)]._item = (
                        item._item
                    )

    # TODO(SRM47) Update the repr to make it look like PyTorch
    def __repr__(self):
        return self._data.__repr__() if self._item is None else self._item.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def __translate_broadcasted_coordinate_into_self_coordinate(
        self, new_tensor_coordinates: tuple
    ) -> tuple:
        """Helper method for broadcasting to map a coordinate in a new tensor to the corresponding coordinate
        in the existing tensor. Assumes that the provided coordinate is for broadcasting into the current tensor.

        Args:
            new_tensor_coordinates (tuple): Coordinates in the new tensor to be mapped.

        Returns:
            tuple: Coordinates in the existing tensor mapped from the provided new tensor coordinates.
        """

        self_coordinates = [0] * len(
            self.shape
        )  # Initialize a list to store the translated coordinates.

        # Iterate over the dimensions of the existing tensor in reverse order.
        for existing_dimension_index in range(len(self.shape) - 1, -1, -1):
            # If the shape at the current index of the existing tensor is 1, set the coordinate to 0 in that dimension.
            # Otherwise, use the corresponding coordinate from the new tensor coordinates.
            if self.shape[existing_dimension_index] == 1:
                self_coordinates[existing_dimension_index] = 0
            else:
                # Calculate the index in the new tensor coordinates corresponding to the current dimension of the existing tensor.
                new_tensor_dimension_index = (
                    existing_dimension_index
                    + len(new_tensor_coordinates)
                    - len(self.shape)
                )
                self_coordinates[existing_dimension_index] = new_tensor_coordinates[
                    new_tensor_dimension_index
                ]

        return tuple(
            self_coordinates
        )  # Return the translated coordinates of the existing tensor.

    def __validate_broadcast(self, desired_shape: tuple[int]) -> None:
        """Helper function to determine whether the shape of self TensorData can be broadcasted
        to the desired shape.

        Args:
            desired_shape (tuple): The shape to which self TensorData is attempted to be broadcasted.

        Raises:
            ValueError: If the dimensions of self TensorData are incompatible for broadcasting to the desired shape.
        """
        # Iterate over the reversed dimensions of both shapes.
        for self_dim, desired_dim in zip(reversed(self.shape), reversed(desired_shape)):
            # If the dimension in desired shape is not 1 and doesn't match the dimension in self.shape,
            # then broadcasting is not possible.
            if not (self_dim == 1 or self_dim == desired_dim):
                raise ValueError("Incompatible dimensions for broadcasting")

    def broadcast(self, *desired_shape: int) -> TensorData:
        """Broadcasts the TensorData to the desired shape.

        Args:
            desired_shape: The desired shape of the tensor.

        Returns:
            A new tensor with the desired shape.

        Raises:
            ValueError: If the tensor cannot be broadcast to the desired shape.
        """
        # If the current shape already matches the desired shape, return the tensor as is.
        if self.shape == desired_shape:
            return self

        # Ensure the number of dimensions provided is sufficient for broadcasting.
        if len(desired_shape) < len(self.shape):
            raise RuntimeError(
                "The number of sizes provided must be greater or equal to the number of dimensions in the tensor."
            )

        # If the tensor has no elements (e.g., it's a singleton), create a new tensor with the desired shape and fill it with the same value.
        if not self._data:
            value = self._item
            broadcasted_tensor = TensorData(*desired_shape)
            for elem in broadcasted_tensor._data:
                elem._item = value
            return broadcasted_tensor

        # Validate if broadcasting is possible to the desired shape.
        self.__validate_broadcast(desired_shape)

        # Create a new tensor with the desired shape and the same data type.
        broadcasted_tensor = TensorData(*desired_shape[-len(self.shape) :])

        # Loop through all coordinates in the new tensor.
        for i, broadcasted_tensor_index in enumerate(
            broadcasted_tensor.__all_coordinates()
        ):
            # Translate each coordinate in the new tensor to the corresponding coordinate in the original tensor.
            translated_index = (
                self.__translate_broadcasted_coordinate_into_self_coordinate(
                    broadcasted_tensor_index
                )
            )
            # Compute the index in self._data.
            single_index = self.__multi_to_single_rank_translation(translated_index)
            # Assign the value from the original tensor to the corresponding coordinate in the new tensor.
            broadcasted_tensor._data[i]._item = self._data[single_index].item()

        # Duplicate data to match the remaining dimensions in the desired shape.
        # This is the "broadcasting".
        remaining_dimensions = desired_shape[: -len(self.shape)]
        new_data = []
        for _ in range(prod(remaining_dimensions)):
            new_data.extend(deepcopy(broadcasted_tensor._data))

        broadcasted_tensor._data = new_data
        broadcasted_tensor.reshape_(desired_shape)

        return broadcasted_tensor

    def __all_coordinates(self):
        return all_coordinates(self.shape)

    def sum(self, dims: tuple | int = None, keepdims: bool = False) -> TensorData:
        """Return a new TensorData object summed along the specified dimensions.

        Args:
            dims (tuple or int, optional): The dimensions along which to sum. If None, sum over all dimensions.
            keep_dims (bool, optional): Whether to keep the dimensions of the summed axes. Defaults to True.

        Returns:
            TensorData: A new TensorData object with the summed values.

        Raises:
            ValueError: If any dimension index is out of bounds.

        Notes:
            - If `dims` is None, all dimensions are reduced into a singleton tensor.
            - For a given coordinate, set all dimensions to 0 for every dimension we're trying to sum over.
            - Initialize the new tensor with the new shape (depending on `keep_dims`) to all zeros.
            - For each coordinate in the original tensor:
                * Create a new coordinate with dimensions set to 0 for the dimensions being summed over.
                * Add the corresponding value from the original tensor to the new tensor.

        """
        if isinstance(dims, int):
            dims = (dims,)
        if self._item:
            # If the tensor is a singleton, return a new tensor with the same value.
            return TensorData(value=self._item)

        if not dims:
            # Handle case where dims is None. If None, sum over all dimensions.
            return TensorData(value=sum(td._item for td in self._data))

        # Ensure all dimensions are within bounds.
        if any(dim < 0 or dim >= len(self.shape) for dim in dims):
            raise ValueError(
                f"Dimension indices should be in bounds 0 and {len(self.shape)-1}"
            )

        # Compute the shape of the new tensor after summing along the specified dimensions.
        new_shape = tuple(
            1 if dim in dims else dimension for dim, dimension in enumerate(self.shape)
        )
        # Initialize the new tensor with zeros.
        new_tensor = TensorData(*new_shape, value=0)

        # Iterate over every coordinate in the original tensor.
        for orig_index, orig_coord in enumerate(self.__all_coordinates()):
            # Create a new coordinate with dimensions set to 0 for the dimensions being summed over.
            new_coord = tuple(
                0 if dim in dims else val for dim, val in enumerate(orig_coord)
            )
            # Translate the new coordinate into a single index in the new tensor.
            new_coord_index = new_tensor.__multi_to_single_rank_translation(new_coord)
            # Add the value from the original tensor to the corresponding index in the new tensor.
            new_tensor._data[new_coord_index]._item += self._data[orig_index]._item

        if not keepdims:
            # Remove the dimensions that were summed out (changed to 1's).
            new_shape = [
                dimension for dim, dimension in enumerate(self.shape) if dim not in dims
            ]
            new_tensor.reshape_(tuple(new_shape))

        return new_tensor

    def mean(self, dims: tuple | int = None, keepdims: bool = False) -> TensorData:
        """Compute the mean of values in the tensor along the specified dimensions.

        Args:
            dims (tuple or int, optional): The dimensions along which to compute the mean. If None, compute the mean over all dimensions.
            keep_dims (bool, optional): Whether to keep the dimensions of the computed mean. Defaults to None.

        Returns:
            TensorData: A new TensorData object containing the mean values.

        Notes:
            - If `dims` is None, compute the mean over all dimensions.
            - If `keep_dims` is None, the default behavior is to keep the dimensions of the computed mean.
            - If `keep_dims` is False, the dimensions of the computed mean are reduced (broadcasted) to 1.

        """
        if self._item:
            # If the tensor is a singleton, return a new tensor with the same value.
            return TensorData(value=self._item)

        # Compute the sum along the specified dimensions.
        res_sum = self.sum(dims, keepdims)

        if not dims:
            # If dims is None, compute the mean over all dimensions.
            return res_sum / self.numel()

        # Compute the product of dimensions specified in dims.
        quotient = prod(self.shape[dim] for dim in dims)

        # Return the mean by dividing the sum by the product of dimensions.
        return res_sum / quotient

    # TODO(SAM): Document
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
        return len(self._data) if self._data else 1

    def relu(self) -> Union[TensorData, np.ndarray]:
        """Return a new TensorData object with the ReLU of each element."""
        new_tensor = TensorData(*self.shape)
        for i in range(len(new_tensor._data)):
            new_tensor._data[i]._item = relu(self._data[i]._item)
        return new_tensor

    def sigmoid(self) -> TensorData:
        """Return a new TensorData object with the sigmoid of each element."""
        new_tensor = TensorData(*self.shape)
        for i in range(len(new_tensor._data)):
            new_tensor._data[i]._item = sigmoid(self._data[i]._item)
        return new_tensor

    def permute(self, *dims: int) -> TensorData:
        """Return a new TensorData object with dimensions permuted according to the provided permutation.

        Args:
            *dims (int): The new order of dimensions for the tensor.

        Returns:
            TensorData: A new TensorData object with the specified permutation of dimensions.

        Raises:
            RuntimeError: If the provided dimension tuple is not a valid permutation of the column indices of this tensor.

        """
        if not is_permutation([i for i in range(len(self.shape))], dims):
            raise RuntimeError(
                "The provided dimension tuple is not a valid permutation of the column indices of this tensor."
            )

        # Create the new shape based on the provided permutation.
        new_shape = [self.shape[dim] for dim in dims]
        # Create a new tensor with the permuted shape.
        new_tensor = TensorData(*new_shape)
        # Iterate through all elements in the original tensor.
        for index, coord in enumerate(self.__all_coordinates()):
            # Translate the coordinates according to the permutation.
            translated_coord = tuple(coord[dim] for dim in dims)
            # Look up the index where the translated coordinate maps to in the new tensor's data list.
            translated_index = new_tensor.__multi_to_single_rank_translation(
                translated_coord
            )
            # Assign the corresponding element from the original tensor to the new tensor.
            new_tensor._data[translated_index] = self._data[index]

        return new_tensor

    @property
    def T(self) -> TensorData:
        """Return an aliased TensorData object with the transpose of the tensor."""
        # Transpose is the same as permuting the tensor with the reverse of its dimensions
        return self.permute(*reversed(range(len(self.shape))))

    def __set(self, val) -> None:
        """Internal method to set all values in the TensorData to the specified value.

        Args:
            val: The value to set for all elements in the TensorData.

        Returns:
            None
        """
        if not self._data:
            # If the TensorData is a singleton, set its item value directly.
            self._item = val
        else:
            # Iterate over all elements and set their values to the specified value.
            for td in self._data:
                td._item = val

    def __binary_op(self, op: Callable, rhs: float | int | TensorData) -> TensorData:
        """Internal method to perform an element-wise binary operation on the TensorData object.

        Args:
            op (Callable): The binary operation to be applied.
            rhs (float, int, or TensorData): The right-hand side operand of the operation.

        Returns:
            TensorData: A new TensorData object containing the result of the binary operation.

        Notes:
            - This method assumes that numpy mode is False if invoked.
            - This method automatically broadcasts inputs when necessary.

        """
        # Handle the case where rhs is a scalar or is a singleton.
        if isinstance(rhs, (float, int)) or not rhs._data:
            # If rhs isn't a number, then it's a singleton TensorData object, so the value should be its item.
            value = rhs if isinstance(rhs, (float, int)) else rhs._item
            # Handle case where self is a singleton
            if not self._data:
                return TensorData(value=op(self._item, value))
            # Create a new tensor with the same shape as self
            new_tensor = TensorData(*self.shape)
            # Apply the binary operation element-wise
            for i, elem in enumerate(new_tensor._data):
                elem._item = op(self._data[i]._item, value)
            return new_tensor

        # Determine the broadcast shape
        broadcast_to = get_common_broadcast_shape(self.shape, rhs.shape)

        # Create an empty output tensor with the broadcast shape
        out = TensorData(*broadcast_to)
        # Broadcast self and rhs to the common shape
        lhs = self.broadcast(*broadcast_to)
        rhs = rhs.broadcast(*broadcast_to)
        # Compute the binary operation for every element
        for i, elem in enumerate(out._data):
            elem._item = op(lhs._data[i]._item, rhs._data[i]._item)

        return out

    def __add__(self, rhs: Union[float, int, TensorData]) -> TensorData:
        """Element-wise addition: self + rhs."""
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
        return self.__binary_op(mul, rhs)

    def __rmul__(self, lhs: float | int | TensorData) -> TensorData:
        """Element-wise multiplication is commutative: lhs * self."""
        return self * lhs

    def __pow__(self, rhs: float | int) -> TensorData:
        """Element-wise exponentiation: self ** rhs."""
        return self.__binary_op(pow, rhs)

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
        return self.__binary_op(gt, rhs)

    def exp(self) -> TensorData:
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
        # Handle case where self or rhs is a singleton
        if self._item or rhs._item:
            raise ValueError("Both arguments to matmul need to be at least 1D")

        lhs = self
        lhs_shape, rhs_shape = self.shape, rhs.shape
        lhs_dims, rhs_dims = len(lhs_shape), len(rhs_shape)

        # If both tensors are 1-dimensional, the dot product (scalar) is returned.
        if lhs_dims == 1 and rhs_dims == 1:
            # dot will return a tensordata because _data is comprised of Tensordata objects.
            return dot(self._data, rhs._data)

        # If both arguments are 2-dimensional, the matrix-matrix product is returned.
        elif lhs_dims == 2 and rhs_dims == 2:
            result_shape, result_data = matmul_2d(
                lhs._data, lhs.shape, rhs._data, rhs.shape
            )

            return TensorData.create_tensor_from_data(result_data, result_shape)

        # If the first argument is 1-dimensional and the second argument is 2-dimensional...
        elif lhs_dims == 1 and rhs_dims == 2:
            # ... a 1 is prepended to its dimension for the purpose of the matrix multiply...
            result_shape, result_data = matmul_2d(
                lhs._data, (1,) + lhs.shape, rhs._data, rhs.shape
            )
            return TensorData.create_tensor_from_data(
                result_data,
                (
                    result_shape[1],
                ),  # ... and after the matrix multiply, the appended dimension is removed.
            )

        # If the first argument is 2-dimensional and the second argument is 1-dimensional,
        # the matrix-vector product is returned.
        elif lhs_dims == 2 and rhs_dims == 1:
            # A 1 is appended to its dimension for the purpose of the matrix multiply...
            result_shape, result_data = matmul_2d(
                lhs._data, lhs.shape, rhs._data, rhs.shape + (1,)
            )

            return TensorData.create_tensor_from_data(
                result_data,
                (
                    result_shape[0],
                ),  # ... and after the matrix multiply, the appended dimension is removed.
            )

        # If both arguments are at least 1-dimensional and at least one argument is
        # N-dimensional (where N > 2), then a batched matrix multiply is returned.
        elif (lhs_dims >= 1 and rhs_dims >= 1) and (lhs_dims > 2 or rhs_dims > 2):
            # If the first argument is 1-dimensional, a 1 is prepended to its dimension
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

            return TensorData.create_tensor_from_data(result_data, new_shape)

    @staticmethod
    def concatenate(tensordatas: tuple[TensorData], dim: int = 0) -> TensorData:
        """Concatenates a sequence of tensors along a given dimension.

        Args:
            tensors: A list of TensorData objects to concatenate.
            dim: The dimension along which to concatenate.

        Returns:
            The concatenated TensorData object.

        Raises:
            ValueError: If tensors have incompatible shapes or dim is invalid.
        """

        if not tensordatas:
            raise ValueError("match.cat(): input tensors cannot be empty")

        if dim < 0 or dim >= len(tensordatas[0].shape):
            raise ValueError(f"Invalid dimension: {dim}")

        # Check shape compatibility (all dimensions except the concatenation dim must match)
        for i in range(1, len(tensordatas)):
            if (
                tensordatas[i].shape[:dim] + tensordatas[i].shape[dim + 1 :]
                != tensordatas[0].shape[:dim] + tensordatas[0].shape[dim + 1 :]
            ):
                raise ValueError(
                    "match.cat(): tensors must have the same shape, except along the concatenation dimension"
                )

        # Calculate the new shape after concatenation
        new_shape = list(tensordatas[0].shape)
        new_shape[dim] = sum(t.shape[dim] for t in tensordatas)
        new_shape = tuple(new_shape)

        """CURRENTLY ONLY DIM=0 (HARDCODING IT) IS SUPPORT FOR EFFICIENCY"""
        data_objects = [td._data for td in tensordatas]
        new_data = list(itertools.chain.from_iterable(data_objects))
        return TensorData.create_tensor_from_data(new_data, new_shape)

        # # Initialize the new tensor
        # new_tensor = TensorData(*new_shape)

        # # Perform concatenation by iterating through the tensors and copying their data
        # current_index = 0
        # for tensor in tensordatas:
        #     size = tensor.numel()
        #     for i in range(current_index, current_index + size):
        #         new_tensor._data[i]._item = tensor._data[i - current_index]._item
        #     current_index += size

        # return new_tensor
