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
        self.__out_of_bounds_coords(coords)
        return self._data[self.__multi_to_single_rank_translation(coords)]

    def __setitem__(self, coords, value):
        self.__out_of_bounds_coords(coords)
        self._data[self.__multi_to_single_rank_translation(coords)]._item = value
