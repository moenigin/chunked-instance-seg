"""Grid arithmetic, global object-id scheme, Zarr array helper, and the
filesystem-based progress markers used for resumability throughout the
pipeline.

Design notes:
- Chunk grids are always defined by (stack_dim, chunk_size); every
  function here that needs "which chunk is this" recomputes it from
  those two small arrays rather than looking it up in a table, so it
  stays cheap to pass into parallel workers (see IDScheme).
- Resumability is filesystem-based: mark_completed/filter_remaining_chunks
  use one empty marker file per finished chunk. This is simple, needs
  no locking (each chunk only ever writes its own marker), and survives
  interpreter/process restarts for free.
"""
import numpy as np
import zarr

from numcodecs import Blosc
from ome_zarr.io import parse_url
from pathlib import Path
from typing import Sequence, Optional, Union

TOTAL_ID_BITS = 63  # keep every id within the positive range of int64


class IDScheme:
    """Deterministic, collision-free global object ids without a shared
    counter or a stored lookup table.

    object_id = (chunk_key << local_bits) | local_label

    chunk_key is the chunk's flattened C-order index in the
    (stack_dim, chunk_size) grid, computed arithmetically -- so an
    IDScheme instance holds only a few small numbers and is cheap to
    ship to parallel workers (unlike a per-chunk lookup table). Every
    chunk must be built from the same (stack_dim, chunk_size) for keys
    to stay consistent across a run, including on resume.
    """

    def __init__(self, stack_dim, chunk_size):
        """stack_dim: [[z0,y0,x0],[z1,y1,x1]] volume bounds. chunk_size:
        (dz,dy,dx). Derives how many bits are needed for the chunk key
        vs. the per-chunk local label; raises if the grid is empty or
        leaves no room for local labels (use a larger chunk_size)."""
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
        """Integer grid key for the chunk starting at `origin`. Raises
        KeyError if `origin` doesn't align with this grid."""
        rel = np.asarray(origin, dtype=np.int64) - self._start
        idx = rel // self._chunk
        if (np.any(rel % self._chunk) or np.any(idx < 0)
                or np.any(idx >= np.array(self._grid))):
            raise KeyError(f"origin {tuple(int(c) for c in origin)} is not on "
                           f"the chunk grid this IDScheme was built for")
        return int(np.ravel_multi_index(tuple(int(i) for i in idx), self._grid))

    def origins_for_keys(self, keys) -> np.ndarray:
        """Vectorised inverse of chunk_key: array of keys -> (n, 3) array
        of chunk origins."""
        idx = np.unravel_index(np.asarray(keys, dtype=np.int64), self._grid)
        return self._start + np.stack(idx, axis=-1) * self._chunk

    def origin_for_key(self, chunk_key: int) -> np.ndarray:
        """Scalar convenience wrapper around origins_for_keys."""
        return self.origins_for_keys(np.array([chunk_key]))[0]

    def pack(self, chunk_key: int, local_label):
        """Combine a chunk_key with one local label or an array of local
        labels (each >= 1) into global id(s). Raises RuntimeError rather
        than silently colliding if a label would overflow local_bits --
        reduce chunk_size (fewer objects per chunk) if this happens."""
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
        """Recover the chunk_key an id was created in. Works for a
        python int or an int64 array."""
        return object_id >> self.local_bits

    def describe(self) -> dict:
        """Small summary dict, worth logging once at pipeline start."""
        return {"n_chunks": self.n_chunks,
                "chunk_key_bits": self.chunk_key_bits,
                "local_bits": self.local_bits,
                "max_objects_per_chunk": self.max_objects_per_chunk}


def generate_chunks(stack_dim: np.ndarray[int], chunk_size: np.ndarray[int]) -> \
        np.ndarray[list[np.ndarray]]:
    """Enumerate every (origin, far_corner) pair tiling stack_dim with
    chunk_size, in C order (z outermost, x innermost). Edge chunks are
    clipped to stack_dim, so they may be smaller than chunk_size."""
    ranges = [np.arange(stack_dim[0, dim], stack_dim[1, dim], chunk_size[dim])
              for dim in range(3)]
    z, y, x = np.meshgrid(*ranges, indexing='ij')
    origins = np.stack([z, y, x], axis=-1).reshape(-1, 3)
    far_corners = np.minimum(origins + chunk_size, stack_dim[1, :])
    return np.array(list(zip(origins, far_corners)))


def get_or_create_zarr_array(output_fn: Union[str, Path],
                             zarr_chunks: Optional[list[int]] = None,
                             data_dim: Optional[tuple[int, int, int]] = None,
                             dtype_: Optional[type] = None,
                             opening_mode: Optional[str] = 'a',
                             fill_value: Optional[int] = 0,
                             write_empty_chunks: Optional[bool] = False,
                             add_zarr_metadata: Optional[bool] = True,
                             voxel_size: Optional[tuple[int, int, int]] = (25, 9, 9)) -> zarr.Array:
    """Return the '0' array at output_fn, creating it (plus OME-NGFF 0.4
    group metadata) if missing. zarr_chunks/data_dim/dtype_ are only
    required on creation; opening an existing array ignores every other
    argument, including voxel_size -- change it only by creating a new
    array, not by re-running this against an existing one."""
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

    store = parse_url(output_fn, mode=opening_mode).store
    zarr_group = zarr.group(store=store)
    if '0' in zarr_group:
        return zarr_group['0']

    assert zarr_chunks is not None and data_dim is not None and dtype_ is not None, \
        "zarr_chunks, data_dim and dtype_ must all be given when creating a new array"

    output_dataset = zarr_group.create_dataset(
        name="0", shape=data_dim, chunks=zarr_chunks, dtype=dtype_,
        cache_attrs=False, compressor=compressor, fill_value=fill_value,
        write_empty_chunks=write_empty_chunks, overwrite=False,
        dimension_separator='/',
    )

    if add_zarr_metadata:
        zarr_group.attrs.put({
            "multiscales": [{
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [{
                    "coordinateTransformations": [
                        {"scale": voxel_size, "type": "scale"}],
                    "path": "0",
                }],
                "version": "0.4",
            }]
        })

    return output_dataset


def _chunk_id(chunk_coords) -> str:
    """Human-readable, filesystem-safe id for a chunk, from its
    [[origin],[far_corner]] coordinates -- used to name both metadata
    files and progress markers."""
    return "_".join(str(int(c)) for c in np.asarray(chunk_coords).flatten())


def mark_completed(chunk_coords, progress_dir):
    """Touch an empty marker file for this chunk under progress_dir.
    One file per chunk, so concurrent workers marking different chunks
    never contend."""
    Path(progress_dir).mkdir(exist_ok=True)
    Path(progress_dir, f"{_chunk_id(chunk_coords)}.done").touch()


def filter_remaining_chunks(all_chunks: list, progress_dir: Union[Path, str]) -> list:
    """Drop any chunk that already has a marker file under progress_dir
    -- the basis for resuming an interrupted pass without redoing
    finished work."""
    progress_dir = Path(progress_dir)
    if not progress_dir.exists():
        return all_chunks
    completed = {f.stem for f in progress_dir.glob("*.done")}
    return [c for c in all_chunks if _chunk_id(c) not in completed]


def mk_dir(path: Union[str, Path]) -> Path:
    """mkdir -p, returning the Path."""
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def resolve_path(val: str | None, default_path: Path, base_dir: Path) -> Path:
    """Resolve a user-supplied, possibly-relative config path against
    base_dir, or fall back to default_path if val is empty/None."""
    if not val:
        return default_path
    p = Path(val)
    return p if p.is_absolute() else base_dir / p