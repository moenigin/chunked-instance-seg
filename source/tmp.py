"""
Global, non-contiguous, collision-free object ID scheme via bit-packing.

WHY THIS EXISTS
----------------
The old pipeline handed out object IDs from one shared, lock-protected
counter, so every worker had to coordinate through that single piece of
state. That made resuming an interrupted run unsafe (the counter has to
be perfectly restored or IDs collide) and tied the implementation to a
model (threads in one process) where sharing that state is easy.

Object IDs only need to be *unique*, not contiguous or small. So instead
of asking a shared counter "what's the next free id?", each chunk simply
computes its own ids from information it already has, with no
coordination:

    object_id = (chunk_key << K) | local_label

- chunk_key: a unique integer per chunk. Assigned once, up front, from
  the deterministic list of chunk origins produced by generate_chunks()
  -- the same chunk always gets the same key on every run, which is what
  makes re-processing an interrupted chunk safe (it reproduces the exact
  same ids, never new ones).
- local_label: the label scipy.ndimage.label() assigned within that
  chunk (always >= 1; 0 is background and is never used as an object id).
- K ("local_bits" below): the number of bits reserved for local_label,
  derived automatically from the total number of chunks in the volume,
  so chunk_key and local_label can never overlap.

Because both halves are computed independently per chunk, Pass 1 needs
no shared counter, no lock, and no cross-worker coordination at all --
which is what makes it safe under threads, multiple processes on one
node, or separate SLURM array tasks without any code changes.

HARD ASSUMPTION -- CHECKED, NOT SILENT
----------------------------------------
No single chunk may produce more than `max_objects_per_chunk` distinct
objects (after small-object filtering). This is *not* a fixed constant
you have to guess: it is derived at pipeline start from the number of
chunks in the volume (see IDScheme.__init__), and for realistic chunk
counts it leaves room for many millions of objects per chunk. If a
chunk ever exceeds it, `IDScheme.pack()` raises immediately with a
clear message rather than silently producing a colliding id. If you
hit this in practice, reduce chunk_size (fewer objects per chunk) --
the bit budget re-derives itself automatically on the next run.

IDs are kept within the positive range of a signed 64-bit integer
(< 2**63) so they remain safe to store as a plain int64 column in
pandas / Parquet / anything else that assumes signed integers, even
though the segmentation volume itself can still use uint64 if you want.
"""
import numpy as np
import zarr

from numcodecs import Blosc
from ome_zarr.io import parse_url
from pathlib import Path
from typing import Sequence, Optional, Union

TOTAL_ID_BITS = 63  # keep every id within the positive range of int64


class IDScheme:
    """
    Assigns each chunk a stable integer key (from its position in a
    fixed, deterministically-ordered chunk list) and packs
    (chunk_key, local_label) pairs into single global object ids.

    Build ONE instance from the full chunk list at pipeline start and
    reuse it everywhere (Pass 1 packing, Pass 3 unpacking) -- the
    origin<->key lookup table only exists in this one object, and every
    chunk must agree on the same key for the scheme to be resumable.
    """

    def __init__(self, chunk_origins: Sequence[np.ndarray]):
        # Sort so the same chunk always gets the same key regardless of
        # what order generate_chunks() happens to return them in on a
        # given run -- required for resuming to reproduce identical ids.
        ordered = sorted(
            (tuple(int(c) for c in origin) for origin in chunk_origins)
        )
        self._origin_to_key = {origin: i for i, origin in enumerate(ordered)}
        self._key_to_origin = {i: np.array(o) for o, i in
                               self._origin_to_key.items()}

        self.n_chunks = len(ordered)
        if self.n_chunks == 0:
            raise ValueError("IDScheme built from an empty chunk list")

        # Bits needed to represent chunk_key in [0, n_chunks - 1].
        self.chunk_key_bits = max(1, (self.n_chunks - 1).bit_length())
        self.local_bits = TOTAL_ID_BITS - self.chunk_key_bits
        if self.local_bits < 1:
            raise ValueError(
                f"Chunk grid has {self.n_chunks} chunks, which alone needs "
                f"{self.chunk_key_bits} bits and leaves no room for object "
                f"ids within a chunk. Use a coarser chunk grid (larger "
                f"chunk_size) so fewer, bigger chunks are needed."
            )
        self.max_objects_per_chunk = 1 << self.local_bits

        # Note: this lookup table is small enough to pickle cheaply for
        # realistic chunk counts (up to several million). If you ever
        # tile a volume into far more chunks than that, replace this
        # table with arithmetic on the chunk grid coordinates instead.

    def chunk_key(self, origin: np.ndarray) -> int:
        """Look up the stable integer key for a chunk's origin."""
        key = tuple(int(c) for c in origin)
        try:
            return self._origin_to_key[key]
        except KeyError as exc:
            raise KeyError(
                f"origin {key} is not one of the chunks this IDScheme was "
                f"built from -- IDScheme must be constructed from the "
                f"complete, current chunk list."
            ) from exc

    def origin_for_key(self, chunk_key: int) -> np.ndarray:
        """Recover a chunk's origin from its integer key."""
        return self._key_to_origin[chunk_key]

    def pack(self, chunk_key: int, local_label: int) -> int:
        """Combine a chunk_key and a chunk-local label into a global id."""
        if not (1 <= local_label < self.max_objects_per_chunk):
            raise RuntimeError(
                f"chunk_key={chunk_key} produced local label "
                f"{local_label}, which exceeds the assumed maximum of "
                f"{self.max_objects_per_chunk} objects per chunk "
                f"({self.local_bits}-bit local id space, derived from "
                f"{self.n_chunks} total chunks). Reduce chunk_size to "
                f"lower the number of objects per chunk -- the pipeline "
                f"cannot guarantee unique ids past this point and is "
                f"stopping rather than silently corrupting labels."
            )
        return (chunk_key << self.local_bits) | local_label

    def unpack_chunk_key(self, object_id: int) -> int:
        """Recover which chunk an object id was created in."""
        return object_id >> self.local_bits

    def describe(self) -> dict:
        """Small summary worth logging once at pipeline start."""
        return {
            "n_chunks": self.n_chunks,
            "chunk_key_bits": self.chunk_key_bits,
            "local_bits": self.local_bits,
            "max_objects_per_chunk": self.max_objects_per_chunk,
        }


def generate_chunks(stack_dim: np.ndarray[int], chunk_size: np.ndarray[int]) -> \
        np.ndarray[
            list[np.ndarray]]:
    """Generate chunk origins and far corners"""
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
