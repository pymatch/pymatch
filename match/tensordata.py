import itertools # check with prof clark

class TensorData:
    def __init__(self, *size: int, value: float = 0.0):
        self.shape = size
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

    
    def __getitem__(self, coords):
        # go through each of items in the tuple and check if it is a `slice` object
        # slice objects have start, stop, step, so the range of coordinates for that index is range(start, stop, step)
        # original shape [5,3,6,13]
        # [2:3, 5, 2:5:2, 9]
        # [range(2,3), 5, range(2,5,2), 9]
        # final shape = (1,2)
        output_shape = []
        possible_indices = [] 
        for i, coordinate in enumerate(coords):
            if isinstance(coordinate, slice):
                start = coordinate.start or 0
                stop = coordinate.stop or self.shape[i]
                step = coordinate.step or 1
                possible_indices.append(range(start, stop, step))
                # like convolution formula? [(W−K+2P)/S]+1
                output_shape.append( ((stop-start)//step) )
            elif isinstance(coordinate, int):
                # if int, we aren't goign to add to shape
                # j.shape = [3,4,5]; j[:, 0, :].shape = [3,5], j[0, :, :] = [4,5]...and so on
                possible_indices.append([coordinate])
            else:
                raise ValueError("can only be ints or slices")
            
        if not output_shape:
            # output_shape is empty, [], if and only if all indices in `coords` were integers
            # indicating only one item should be retrieved
            self.__out_of_bounds_coords(coords)
            return self._data[self.__multi_to_single_rank_translation(coords)]
        
        output_data = []
        for index in itertools.product(*possible_indices):
            self.__out_of_bounds_coords(index)
            output_data.append(self._data[self.__multi_to_single_rank_translation(index)])

        output_tensor = TensorData(*output_shape)
        assert len(output_data) == len(output_tensor._data)
        output_tensor._data = output_data
        return output_tensor

    def __setitem__(self, coords, value):
        self.__out_of_bounds_coords(coords)
        self._data[self.__multi_to_single_rank_translation(coords)]._item = value

    def __repr__(self):
        return self._data.__repr__() if self._item is None else self._item.__repr__()



if __name__ == "__main__":
    t = TensorData(2,2,2)
    print(t.shape, t[:,1:,0].shape)