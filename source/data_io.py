from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from source.utils import get_or_create_zarr_array


class DataIO(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Return the shape of the data array."""

    @abstractmethod
    def get_data(self, data_slice):
        """Retrieve data from the specified slice."""

    @abstractmethod
    def write_data(self, data, data_slice):
        """Write data to the specified slice."""


class Zarr2DataIO(DataIO):
    def __init__(self, zarr_name, chunk_size: Optional[np.ndarray] = None,
                 array_shape: Optional[tuple] = None):
        self.zarr_path = Path(zarr_name)
        if self.zarr_path.exists():
            self.zarr_array = get_or_create_zarr_array(zarr_name, opening_mode='r+')
        else:
            assert chunk_size is not None and array_shape is not None, \
                "both chunk_size and array_shape need to be given when a new array is created"
            self.zarr_array = get_or_create_zarr_array(zarr_name, chunk_size, array_shape,
                                                       dtype_=np.uint64)

    @property
    def shape(self):
        return self.zarr_array.shape

    def get_data(self, data_slice):
        return self.zarr_array[data_slice]

    def write_data(self, data, data_slice):
        self.zarr_array[data_slice] = data
