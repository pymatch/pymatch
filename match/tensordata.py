import itertools
from math import exp, ceil, prod
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from copy import deepcopy
from typing import Callable, Union


class TensorData(object):
    """A storage and arithmetic object for n-dimensional tensor data.

    TensorData is an inefficient, but easy-to-understand implementation
    of many n-dimensional tensor operations.

    Like the PyTorch Tensor object, the pymatch TensorData objects are
    recursive structures that hold either a list of TensorData objects,
    or a single value. For instance, a TensorData object's data list could
    look like [TensorData(0), TensorData(1), ..., TensorData(47)].
    There are therefore two implicit types of TensorData objects. Ones
    that store only a single value, accesible by .item(), and ones that
    store a list of these `singleton` TensorData objects.

    Using the provided shape of the TensorData object, accessing data using
    the standard coordinate system, for instance x[1,3,2], involves translation
    of the coordinates (1,3,2) to the corresponding index in the data list
    to fetch the intended data.
    """

    def __init__(self, *size: int, value: float = 0.0, dtype: type = float) -> None:
        """Create a new TensorData object to store an n-dimensional tensor of values.

        Args:
            size (tuple[int]): The shape of the Tensor
            value (float): The default value of each element in the Tensor
            dtype (type): The type of the values in the Tensor
        """
        super().__init__()

        self.shape: tuple[int] = size
        self.dtype: type = dtype
        self.__initialize_tensor_data(value)
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
        if self.shape:
            self._item = None
            self._data = [TensorData(value=value) for _ in range(prod(self.shape))]
        else:
            self._item = value
            self._data = []

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

    def item(self):
        """Returns the item of a singleton, or single element TensorData object.

        Raises:
            ValueError: raises value error if attempting to access the item of a
            TensorData object with more than one element.
        """
        if self._item != None:
            return self._item
        if len(self._data) == 1:
            self._data[0]._item
        raise ValueError(
            "only one element tensors can be converted into python scalars"
        )

    def __reshape(self, shape: tuple):
        """Helper method to reshape the TensorData object inplace, without changing the data.

        Raises:
            RuntimeError: raises runtime error if the product of the dimensions of
            the new shape is not equal to the existing number of elements.
        """
        if prod(shape) != len(self._data):
            raise RuntimeError(
                f"shape {shape} is invalid for input of size {len(self._data)}"
            )
        self.shape = shape
        # The strides change when the shape does, so they must be reinitialized.
        self.__initialize_strides()

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

        Example:
            j = TensorData([5, 3, 6, 13])
            j._data = np.arange(0, 216).reshape(5, 3, 6, 13)

            k = j[2:3, 5, 2:5:2, 9]

            print(k.shape)
            # (1, 2)

            print(k)
            # [[111, 113]]
        """
        if not isinstance(coords, tuple):
            coords = (coords,)

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

        Example:
            j = TensorData([5, 3, 6, 13])
            j._data = np.arange(0, 216).reshape(5, 3, 6, 13)

            # Set the value of a subtensor with shape (1, 2)
            j[2:3, 5, 2:5:2, 9] = 100

            print(j[2:3, 5, 2:5:2, 9])
            # [[100, 100]]
        """
        if isinstance(value, list):
            raise TypeError("can't assign a list to a TensorData")

        if not isinstance(coords, tuple):
            coords = (coords,)

        output_shape, possible_indices = self.__convert_slice_to_index_list(coords)

        if not output_shape:
            self.__out_of_bounds_coords(coords)
            if isinstance(value, TensorData):
                self._data[
                    self.__multi_to_single_rank_translation(coords)
                ]._item = value.item()
            elif isinstance(value, (int, float)):
                self._data[
                    self.__multi_to_single_rank_translation(coords)
                ]._item = self.dtype(value)
            else:
                raise TypeError("invalid type to set value in tensor data")
        else:
            if isinstance(value, (int, float)):
                for i, index in enumerate(itertools.product(*possible_indices)):
                    self.__out_of_bounds_coords(index)
                    self._data[
                        self.__multi_to_single_rank_translation(index)
                    ]._item = self.dtype(value)
            elif isinstance(value, TensorData):
                for i, index in enumerate(itertools.product(*possible_indices)):
                    self.__out_of_bounds_coords(index)
                    self._data[
                        self.__multi_to_single_rank_translation(index)
                    ]._item = value._data[i]._item

    def __repr__(self):
        return self._data.__repr__() if self._item is None else self._item.__repr__()

    def __translate(self, *shape: int) -> tuple:
        shape = shape[0]
        res = [0] * len(self.shape)
        for i in range(len(res) - 1, -1, -1):
            res[i] = (
                0 if self.shape[i] == 1 else shape[i + len(shape) - len(self.shape)]
            )
        return tuple(res)

    def __validate_broadcast(self, shape):
        """Helper function to determine whether self TensorData can be broadcasted 
        to desired shape."""
        for s1, s2 in zip(reversed(self.shape), reversed(shape)):
            if not (s1 == 1 or s2 == 1 or s1 == s2):
                raise ValueError("Can't broadcast")

    def broadcast(self, *shape: int):
        """Broadcasts the TensorData to the desired shape.

        Args:
            shape: The desired shape of the tensor.

        Returns:
            A new tensor with the desired shape.

        Raises:
            ValueError: If the tensor cannot be broadcast to the desired shape.
        """
        self.__validate_broadcast(shape)

        if self.shape == shape:
            return self
        
        new_tensor = TensorData(*shape[len(self.shape) - 1 :], dtype=self.dtype)
        possible_indices = [range(dim) for dim in new_tensor.shape]
        for i, new_tensor_index in enumerate(itertools.product(*possible_indices)):
            translated_index = self.__translate(new_tensor_index)
            single_index = self.__multi_to_single_rank_translation(translated_index)
            new_tensor._data[i]._item = self._data[single_index].item()

        remianing_dimensions = shape[: len(self.shape) - 1]

        new_data = []
        for _ in range(prod(remianing_dimensions)):
            new_data.extend(deepcopy(new_tensor._data))

        new_tensor._data = new_data
        new_tensor.__reshape(shape)

        return new_tensor

    def unbroadcast(self, *shape: int):
        """Return a new TensorData unbroadcast from current shape to desired shape."""
        ...
