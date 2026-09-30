"""Storage abstraction the pipeline and postprocessing read/write
through.

Every pass in the pipeline, and every postprocessing step, talks to
volumes only via the DataIO interface -- never directly against zarr,
N5, HDF5, etc. Zarr2DataIO is the only implementation provided. The
contract has two parts:

- get_data/write_data/shape: the read/write-by-slice interface the main
  pipeline needs.
- path/open_or_create_like: identity and construction, needed by
  postprocess_instances.py so it can compare "is this the same store?",
  test whether a destination already has data, and open-or-create a new
  store of the SAME backend at a new location -- all without ever
  naming a concrete backend class itself.

To support another storage backend, write a new DataIO subclass
implementing all five members and thread it through parse_cfg's
`io_func` dispatch. Nothing outside data_io.py should need to know
which backend is in use, including postprocessing.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np

from source.utils import get_or_create_zarr_array


class DataIO(ABC):
    """Minimal contract every storage backend must implement: read/write
    by slice, plus enough identity to let generic code (in particular
    postprocess_instances.py) create a same-backend destination and
    check whether it's the same store as an existing one."""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Full shape of the underlying array, e.g. (Z, Y, X)."""

    @property
    @abstractmethod
    def path(self) -> Path:
        """Stable, comparable location of the underlying store (e.g. a
        Zarr store's root directory). Used for identity checks (`is
        this the same store as that other DataIO?`) and, since every
        backend here is filesystem-based, for `.exists()` checks on a
        prospective destination."""

    @abstractmethod
    def get_data(self, data_slice):
        """Return the array data at `data_slice` (a tuple of slices)."""

    @abstractmethod
    def write_data(self, data, data_slice):
        """Write `data` into `data_slice` (a tuple of slices)."""

    @abstractmethod
    def open_or_create_like(self, path: Union[str, Path],
                            chunk_size: Optional[tuple] = None) -> "DataIO":
        """Return a DataIO of this SAME backend at `path`: open it if
        data already exists there, otherwise create a fresh store
        matching this instance's shape and creation settings (chunking,
        voxel size, etc). This is how postprocessing steps write to a
        new destination without hardcoding a specific backend class."""


class Zarr2DataIO(DataIO):
    """DataIO backed by a Zarr v2 array (via get_or_create_zarr_array).
    Opens the array at zarr_name if it already exists; otherwise creates
    it, in which case chunk_size and array_shape are required. The
    creation settings (voxel_size, add_zarr_metadata, fill_value,
    write_empty_chunks) are remembered on the instance so
    open_or_create_like can carry them forward to a new destination --
    note that if zarr_name already existed, these settings were never
    applied (see get_or_create_zarr_array) and stay at their passed-in
    or default values rather than whatever the existing store actually
    used."""

    def __init__(self, zarr_name, chunk_size: Optional[np.ndarray] = None,
                 array_shape: Optional[tuple] = None,
                 add_zarr_metadata: bool = True,
                 voxel_size: tuple = (25, 9, 9),
                 fill_value: Optional[int] = 0,
                 write_empty_chunks: Optional[bool] = False
                 ):
        self.zarr_path = Path(zarr_name)
        self.add_zarr_metadata = add_zarr_metadata
        self.voxel_size = voxel_size
        self.fill_value = fill_value
        self.write_empty_chunks = write_empty_chunks
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

    @property
    def path(self) -> Path:
        return self.zarr_path

    def get_data(self, data_slice):
        return self.zarr_array[data_slice]

    def write_data(self, data, data_slice):
        self.zarr_array[data_slice] = data

    def open_or_create_like(self, path: Union[str, Path],
                            chunk_size: Optional[tuple] = None) -> "Zarr2DataIO":
        chunks = tuple(int(c) for c in chunk_size) if chunk_size is not None \
            else tuple(self.zarr_array.chunks)
        return type(self)(path, chunk_size=chunks, array_shape=self.shape,
                          add_zarr_metadata=self.add_zarr_metadata,
                          voxel_size=self.voxel_size,
                          fill_value=self.fill_value,
                          write_empty_chunks=self.write_empty_chunks)