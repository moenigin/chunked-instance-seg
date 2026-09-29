"""Storage abstraction the pipeline reads/writes through.

Every pass in the pipeline talks to volumes only via the DataIO
interface (get_data/write_data over a slice, plus a shape property) --
never directly against zarr, N5, HDF5, etc. Zarr2DataIO is the only
implementation provided. To support another storage backend, write a
new DataIO subclass rather than adding branches elsewhere: implement
the same three members and thread it through parse_cfg's `io_func`
dispatch. Chunk-alignment, resumability and parallelism all live in the
pipeline and don't care which backend is behind the interface.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from source.utils import get_or_create_zarr_array


class DataIO(ABC):
    """Minimal read/write-by-slice contract every storage backend must
    implement."""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Full shape of the underlying array, e.g. (Z, Y, X)."""

    @abstractmethod
    def get_data(self, data_slice):
        """Return the array data at `data_slice` (a tuple of slices)."""

    @abstractmethod
    def write_data(self, data, data_slice):
        """Write `data` into `data_slice` (a tuple of slices)."""


class Zarr2DataIO(DataIO):
    """DataIO backed by a Zarr v2 array (via get_or_create_zarr_array).
    Opens the array at zarr_name if it already exists; otherwise creates
    it, in which case chunk_size and array_shape are required."""

    def __init__(self, zarr_name, chunk_size: Optional[np.ndarray] = None,
                 array_shape: Optional[tuple] = None,
                 add_zarr_metadata: bool = True,
                 voxel_size: tuple = (25, 9, 9),
                 fill_value: Optional[int] = 0,
                 write_empty_chunks: Optional[bool] = False
                 ):
        self.zarr_path = Path(zarr_name)
        if self.zarr_path.exists():
            self.zarr_array = get_or_create_zarr_array(zarr_name,
                                                       opening_mode='r+')
        else:
            assert chunk_size is not None and array_shape is not None, \
                "both chunk_size and array_shape need to be given when a new array is created"
            self.zarr_array = get_or_create_zarr_array(zarr_name, chunk_size,
                                                       array_shape,
                                                       dtype_=np.uint64,
                                                       add_zarr_metadata=add_zarr_metadata,
                                                       voxel_size=voxel_size,
                                                       fill_value=fill_value,
                                                       write_empty_chunks=write_empty_chunks
                                                       )

    @property
    def shape(self):
        return self.zarr_array.shape

    def get_data(self, data_slice):
        return self.zarr_array[data_slice]

    def write_data(self, data, data_slice):
        self.zarr_array[data_slice] = data