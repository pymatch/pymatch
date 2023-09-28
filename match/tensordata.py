import itertools
from math import exp, ceil
from operator import add, ge, gt, le, lt, mul, pow
from random import gauss
from copy import deepcopy
from typing import Callable, Union


class TensorData(object):
    def __init__(self, *size: int, value: float = 0.0, dtype: type = float):
        super().__init__()

        self.shape: tuple[int] = size
        self.dtype: type = dtype
        self.__initialize_tensor_data(value)
        self.__initialize_strides()

    def __initialize_strides(self): 
        if not self._data:
            self._strides = ()
            return

        strides = []
        current_stride = len(self._data)
        for dim in self.shape:
            current_stride /= dim
            strides.append(current_stride)
        self._strides = tuple(strides)

    def __initialize_tensor_data(self, value):
        num_elements = 1 if self.shape else 0
        for dim in self.shape:
            num_elements *= dim
        self._item = None if self.shape else value
        self._data = [TensorData(value=value) for _ in range(num_elements)]

    def __out_of_bounds_coords(self, coords: tuple):
        if len(coords) != len(self.shape):
            raise IndexError(f"Too many dimensions, expected {len(self.shape)}.")
        for i, j in zip(coords, self.shape):
            if i < 0 or i >= j:
                raise IndexError("Index out of bounds")

    def __out_of_bounds_index(self, index: int):
        if index < 0 or index >= len(self._data):
            raise IndexError("Index out of bounds")

    def __single_to_multi_rank_translation(self, index: int) -> tuple:
        self.__out_of_bounds_index(index)
        coordinates = []
        for shape_dim in reversed(self.shape):
            temp_index = int(index % shape_dim)
            coordinates.append(temp_index)
            index -= temp_index
            index /= shape_dim

        return tuple(reversed(coordinates))

    def __multi_to_single_rank_translation(self, coords: tuple) -> int:
        self.__out_of_bounds_coords(coords)
        return int(sum(dim * stride for dim, stride in zip(coords, self._strides)))

    def item(self):
        if self._item != None:
            return self._item
        if len(self._data) != 1:
            raise ValueError(
                "only one element tensors can be converted into python scalars"
            )
        return self._data[0]._item
    
    def __reshape(self, shape: tuple):
        # check to see if shape matches with data
        p = 1
        for k in shape:
            p *= k
        assert p == len(self._data)

        self.shape = shape
        self.__initialize_strides()

    def __convert_slice_to_index_list(self, coords):
        output_shape = []
        possible_indices = []
        for i in range(len(self.shape)):
            if i >= len(coords):
                possible_indices.append(range(self.shape[i]))
                output_shape.append(self.shape[i])
                continue

            coordinate = coords[i]
            if isinstance(coordinate, slice):
                start = coordinate.start or 0
                stop = coordinate.stop or self.shape[i]
                step = coordinate.step or 1
                possible_indices.append(range(start, stop, step))
                # like convolution formula? [(W−K+2P)/S]+1
                output_shape.append(ceil((stop - start) / step))
            elif isinstance(coordinate, int):
                # if int, we aren't goign to add to shape
                # j.shape = [3,4,5]; j[:, 0, :].shape = [3,5], j[0, :, :] = [4,5]...and so on
                possible_indices.append([coordinate])
            else:
                raise ValueError("can only be ints or slices")

        return output_shape, possible_indices

    def __getitem__(self, coords):
        # go through each of items in the tuple and check if it is a `slice` object
        # slice objects have start, stop, step, so the range of coordinates for that index is range(start, stop, step)
        # original shape [5,3,6,13]
        # [2:3, 5, 2:5:2, 9]
        # [range(2,3), 5, range(2,5,2), 9]
        # final shape = (1,2)
        if not isinstance(coords, tuple):
            coords = (coords,)

        output_shape, possible_indices = self.__convert_slice_to_index_list(coords)

        if not output_shape:
            # output_shape is empty, [], if and only if all indices in `coords` were integers
            # indicating only one item should be retrieved
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
        # setitem doesn't copy
        # if we do j[1] = x, and j and x are both tensors, j will change, but if we change x, j will not change
        if isinstance(value, list):
            raise TypeError("can't assign a list to a TensorData")

        if not isinstance(coords, tuple):
            coords = (coords,)

        output_shape, possible_indices = self.__convert_slice_to_index_list(coords)

        if not output_shape:
            self.__out_of_bounds_coords(coords)
            if isinstance(value, TensorData):
                # value.item() will check for invalid gets
                self._data[
                    self.__multi_to_single_rank_translation(coords)
                ]._item = value.item()
            elif isinstance(value, (int, float)):
                self._data[
                    self.__multi_to_single_rank_translation(coords)
                ]._item = value  # self.dtype(value)
            else:
                raise TypeError("invalid type to set value in tensor data")
        else:
            for i, index in enumerate(itertools.product(*possible_indices)):
                self.__out_of_bounds_coords(index)
                self._data[
                    self.__multi_to_single_rank_translation(index)
                ]._item = value._data[i]._item

    def __repr__(self):
        return self._data.__repr__() if self._item is None else self._item.__repr__()
    

    def __translate(self, *shape: int) -> tuple:
        """
        EXAMPLE
        current shape = (2,1,3)
        new shape = (3,2,5,3)

        we get an index of [2,0,2,2]
        the output should me 0,0,2
        """
        shape = shape[0]
        res = [0]*len(self.shape)
        for i in range(len(res)-1, -1, -1):
            res[i] = 0 if self.shape[i] == 1 else shape[i+len(shape)-len(self.shape)]
        return tuple(res)

    def __validate_broadcast(self, shape):
        for s1, s2 in zip(reversed(self.shape), reversed(shape)):
            if not (s1==1 or s2 == 1 or s1 == s2):
                raise ValueError("Can't broadcast")
    
    def broadcast(self, *shape: int):
        """
        Consider all these cases
        Current Shape (8,1,2), broadcast shape (1,1,8,1,2)
        Current Shape (8,1,2), broadcast shape (1,1,8,1,2)
        Current Shape (8,1,2,1), broadcast shape (8,9,2,5)

        this algorithm goes backward from the end of the new tensor
        take this for an example 
        Current Shape (8,1,2), broadcast shape (5,1,8,3,2)
        For we consider the last positions that deals with (8,1,2) and (8,3,2)
        we first make a new tensor of shape (8,3,2) and try to populate that new tensor with data. what data?
        we go index by index, translating the integer index into a multidimensional index (for the new tensor)
        then seeing which value to grab from the original tensor.
        then we grab that value from the original tensor and put it in the new tensor

        after than, we use the remaining indices that were 'Added on', the (5,1).
        because we can just copy the array over by the product of the dimensions
        then we change the data by copying and setting the new dimension
        """
        self.__validate_broadcast(shape)

        if self.shape == shape:
            return self
        # very naive solution, brute force. 
        new_tensor = TensorData(*shape[len(self.shape)-1:], dtype=self.dtype)
        print(self.shape, new_tensor.shape)
        # is this more efficient, or it itertools.product() more efficient?
        for i in range(len(new_tensor._data)):
            # Convert i into the coordinates of the new tensor
            new_tensor_index = new_tensor.__single_to_multi_rank_translation(i)
            # Determine what value that new coordinate would be in the current tensor
            translated_index = self.__translate(new_tensor_index)
            # Convert that into an integer index into the ._data array (it's faster than the __getitem__ because we know there are no slices)
            single_index = self.__multi_to_single_rank_translation(translated_index)
            # Set the new value
            new_tensor._data[i]._item = self._data[single_index].item()
        
        remianing_dimensions = shape[:len(self.shape)-1]
        p = 1
        for j in remianing_dimensions:
            p *= j
        
        l = []
        for _ in range(p):
            l.extend(deepcopy(new_tensor._data))

        new_tensor._data = l
        new_tensor.__reshape(shape)

        return new_tensor


    def unbroadcast(self, *shape: int):
        """Return a new TensorData unbroadcast from current shape to (nrow, ncol)."""
        ...


    """
    notes:
    we'd want to deepcopy each list we adding in the broadcast function
    test using python3 -m unittest in the pymatch root directory

    rewrite the broadcast
    iterate the dimension

    play aroundw ith pytorch view and shape, look for patterns
    


    """
