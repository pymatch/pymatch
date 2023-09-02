class TensorData:
    def __init__(self, *size: int):
        self.shape = size
        self.__initialize_tensor_data()
        self.__initialize_strides()

    def __initialize_strides(self):
        strides = []
        current_stride = len(self._data)
        for dim in self.shape:
            current_stride /= dim
            strides.append(current_stride)
        self._strides = tuple(strides)

    def __initialize_tensor_data(self):
        num_elements = 1
        for dim in self.shape:
            num_elements *= dim
        self._data = list(range(num_elements))

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
        return sum(dim * stride for dim, stride in zip(coords, self._strides))
