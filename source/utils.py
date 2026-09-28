import numpy as np
import zarr

from numcodecs import Blosc
from ome_zarr.io import parse_url
from pathlib import Path
from typing import Sequence, Optional, Union

TOTAL_ID_BITS = 63  # keep every id within the positive range of int64


class IDScheme:
    """chunk_key = C-order index of the chunk in the chunk grid.
    Identical to the key the old scheme got from sorting all origins,
    because generate_chunks() enumerates the grid in C order."""

    def __init__(self, stack_dim, chunk_size):
        stack_dim = np.asarray(stack_dim, dtype=np.int64)
        self._start = stack_dim[0].copy()
        self._chunk = np.asarray(chunk_size, dtype=np.int64)
        extent = stack_dim[1] - stack_dim[0]
        self._grid = tuple(int(-(-e // c)) for e, c in zip(extent, self._chunk))
        self.n_chunks = int(np.prod(self._grid, dtype=np.int64))
        if self.n_chunks == 0:
            raise ValueError("IDScheme built from an empty chunk grid")

        self.chunk_key_bits = max(1, (self.n_chunks - 1).bit_length())
        self.local_bits = TOTAL_ID_BITS - self.chunk_key_bits
        if self.local_bits < 1:
            raise ValueError(
                f"Chunk grid has {self.n_chunks} chunks, which alone needs "
                f"{self.chunk_key_bits} bits and leaves no room for object "
                f"ids within a chunk. Use a larger chunk_size.")
        self.max_objects_per_chunk = 1 << self.local_bits

    def chunk_key(self, origin) -> int:
        rel = np.asarray(origin, dtype=np.int64) - self._start
        idx = rel // self._chunk
        if (np.any(rel % self._chunk) or np.any(idx < 0)
                or np.any(idx >= np.array(self._grid))):
            raise KeyError(f"origin {tuple(int(c) for c in origin)} is not on "
                           f"the chunk grid this IDScheme was built for")
        return int(np.ravel_multi_index(tuple(int(i) for i in idx), self._grid))

    def origins_for_keys(self, keys) -> np.ndarray:
        """Vectorised: array of keys -> (n, 3) array of origins."""
        idx = np.unravel_index(np.asarray(keys, dtype=np.int64), self._grid)
        return self._start + np.stack(idx, axis=-1) * self._chunk

    def origin_for_key(self, chunk_key: int) -> np.ndarray:
        return self.origins_for_keys(np.array([chunk_key]))[0]

    def pack(self, chunk_key: int, local_label):
        """Scalar or array of chunk-local labels -> array of global ids."""
        labels = np.atleast_1d(np.asarray(local_label, dtype=np.int64))
        if labels.size and (labels.min() < 1
                            or labels.max() >= self.max_objects_per_chunk):
            raise RuntimeError(
                f"chunk_key={chunk_key} produced local label "
                f"{int(labels.max())}, which exceeds the assumed maximum of "
                f"{self.max_objects_per_chunk} objects per chunk "
                f"({self.local_bits}-bit local id space, derived from "
                f"{self.n_chunks} total chunks). Reduce chunk_size to lower "
                f"the number of objects per chunk -- the pipeline cannot "
                f"guarantee unique ids past this point and is stopping "
                f"rather than silently corrupting labels.")
        return (np.int64(chunk_key) << self.local_bits) | labels

    def unpack_chunk_key(self, object_id):
        """Works for python ints and int64 arrays alike."""
        return object_id >> self.local_bits

    def describe(self) -> dict:
        return {"n_chunks": self.n_chunks,
                "chunk_key_bits": self.chunk_key_bits,
                "local_bits": self.local_bits,
                "max_objects_per_chunk": self.max_objects_per_chunk}


def generate_chunks(stack_dim: np.ndarray[int], chunk_size: np.ndarray[int]) -> \
        np.ndarray[list[np.ndarray]]:
    """Generate chunk origins and far corners"""
    ranges = [np.arange(stack_dim[0, dim], stack_dim[1, dim], chunk_size[dim])
              for dim in range(3)]
    z, y, x = np.meshgrid(*ranges, indexing='ij')
    origins = np.stack([z, y, x], axis=-1).reshape(-1, 3)
    far_corners = np.minimum(origins + chunk_size, stack_dim[1, :])
    return np.array(list(zip(origins, far_corners)))


def open_or_create_zarr_array(output_fn: Union[str, Path],
                              zarr_chunks: Optional[list[int]] = None,
                              data_dim: Optional[tuple[int, int, int]] = None,
                              dtype_: Optional[type] = None,
                              opening_mode: str = 'a',
                              voxel_size: Optional[tuple[int, int, int]] = (25, 9, 9)) -> zarr.Array:
    """
    :param output_fn:
    :param zarr_chunks:
    :param data_dim:
    :param dtype_: type of the output dataset 'uint8' for images, 'uint64' for segmentations
    :param voxel_size:
    :return:
    """
    store = parse_url(output_fn, mode=opening_mode).store
    zarr_group = zarr.group(store=store)
    if '0' in zarr_group:
        return zarr_group['0']

    assert zarr_chunks is not None and data_dim is not None and dtype_ is not None, \
        "both zarr_chunks, data_dim and dtype_ need to be given when a new array is created"

    output_dataset = zarr_group.create_dataset(
        name="0",
        shape=data_dim,
        chunks=zarr_chunks,
        dtype=dtype_,
        cache_attrs=False,
        compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=0,
        write_empty_chunks=False,
        overwrite=False,
        dimension_separator='/',
    )
    metadata = {
        "multiscales": [
            {"axes": [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"}
            ],
                "datasets": [{"coordinateTransformations": [
                    {"scale": voxel_size, "type": "scale"}],
                    "path": "0"}], "version": "0.4"}]
    }
    zarr_group.attrs.put(metadata)
    return output_dataset


def _chunk_id(chunk_coords) -> str:
    """Human-readable id for a chunk bbox."""
    return "_".join(str(int(c)) for c in np.asarray(chunk_coords).flatten())


def mark_completed(chunk_coords, progress_dir):
    Path(progress_dir).mkdir(exist_ok=True)
    Path(progress_dir, f"{_chunk_id(chunk_coords)}.done").touch()


def filter_remaining_chunks(all_chunks: list, progress_dir: Union[Path, str]) -> list:
    progress_dir = Path(progress_dir)
    if not progress_dir.exists():
        return all_chunks
    completed = {f.stem for f in progress_dir.glob("*.done")}
    return [c for c in all_chunks if _chunk_id(c) not in completed]


def mk_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def resolve_path(val: str | None, default_path: Path, base_dir: Path) -> Path:
    if not val:
        return default_path
    p = Path(val)
    return p if p.is_absolute() else base_dir / p
