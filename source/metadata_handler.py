"""Per-chunk Parquet metadata storage.

Each chunk writes its own small Parquet file during Pass 1
(chunk_<chunk_id>.parquet) -- no shared file, so no lock is needed on
the write path, under threads or separate processes alike. Once Pass 3
has resolved every cross-chunk merge, consolidate() streams every
per-chunk file plus the small merge table into ONE final Parquet file,
written once, atomically (temp-file-then-rename), so is_consolidated()
only ever sees a complete file or none at all. The per-chunk files and
the merge table are the durable, resumable intermediate state; the
consolidated file is a derived, write-once artifact built from them.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class ObjectMetadata:
    """One segmented object's identity, chunk of origin, bounding box
    (z1,y1,x1,z2,y2,x2, inclusive) and voxel count."""
    object_id: int
    chunk_id: str
    bbox: tuple  # z1, y1, x1, z2, y2, x2
    nvoxels: int


_SCHEMA = pa.schema([
    ("object_id", pa.int64()),
    ("chunk_id", pa.string()),
    ("bbox_z_min", pa.int64()), ("bbox_y_min", pa.int64()), ("bbox_x_min", pa.int64()),
    ("bbox_z_max", pa.int64()), ("bbox_y_max", pa.int64()), ("bbox_x_max", pa.int64()),
    ("nvoxels", pa.int64()),
])

_COLUMNS = [f.name for f in _SCHEMA]


def _empty_table() -> pa.Table:
    """Zero-row table with the canonical schema -- used so callers never
    have to special-case "no objects" vs. "some objects"."""
    return pa.Table.from_arrays([pa.array([], type=f.type) for f in _SCHEMA], schema=_SCHEMA)


def _rows_to_table(rows) -> pa.Table:
    """Convert a list of row-dicts or a DataFrame to a Table with the
    canonical schema/column order."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if len(df) == 0:
        return _empty_table()
    return pa.Table.from_pandas(df[_COLUMNS], schema=_SCHEMA, preserve_index=False)


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    """Write to a .tmp file and rename over `path`, so readers never see
    a partially-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


class ParquetChunkMetadataStore:
    """One Parquet file per chunk, a small merge table, and a single
    consolidated file built from both once every chunk is done."""

    def __init__(self, meta_dir: Union[str, Path]):
        self.meta_dir = Path(meta_dir)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.merges_file = self.meta_dir / "_merges.parquet"
        self.consolidated_file = self.meta_dir / "_consolidated.parquet"

    # ---------------------------------------------------------------
    # Pass 1: one independent write per chunk, no shared state at all
    # ---------------------------------------------------------------
    def chunk_file(self, chunk_id: str) -> Path:
        """Path of the per-chunk Parquet file for `chunk_id`."""
        return self.meta_dir / f"chunk_{chunk_id}.parquet"

    def chunk_metadata_exists(self, chunk_id: str) -> bool:
        """Whether Pass 1 has already written this chunk's metadata."""
        return self.chunk_file(chunk_id).exists()

    def write_chunk_metadata(self, chunk_id: str, objects: list) -> None:
        """Write one chunk's object metadata as its own Parquet file.
        Safe to call concurrently from any number of threads/processes
        writing DIFFERENT chunk_ids -- each call only ever touches the
        one file for its own chunk_id. Called even for an empty
        `objects` list, so downstream reads never have to distinguish
        "not processed yet" from "processed, nothing here"."""
        rows = [{
            "object_id": o.object_id, "chunk_id": o.chunk_id,
            "bbox_z_min": o.bbox[0], "bbox_y_min": o.bbox[1], "bbox_x_min": o.bbox[2],
            "bbox_z_max": o.bbox[3], "bbox_y_max": o.bbox[4], "bbox_x_max": o.bbox[5],
            "nvoxels": o.nvoxels,
        } for o in objects]
        _atomic_write_table(_rows_to_table(rows), self.chunk_file(chunk_id))

    def read_chunk_metadata(self, chunk_id: str) -> pd.DataFrame:
        """This chunk's object rows, or an empty (correctly-typed)
        DataFrame if the chunk hasn't been processed yet."""
        path = self.chunk_file(chunk_id)
        if not path.exists():
            return _empty_table().to_pandas()
        return pq.read_table(path).to_pandas()

    # ---------------------------------------------------------------
    # Pass 3: the (usually much smaller) table of merged root objects
    # ---------------------------------------------------------------
    def merges_computed(self) -> bool:
        """Whether Pass 3's merge table has already been written."""
        return self.merges_file.exists()

    def write_merges(self, merged_rows) -> None:
        """One row per object that absorbed one or more others across a
        chunk boundary. object_id is the root id of the merged group
        (see IDScheme / union-by-min in the pipeline)."""
        _atomic_write_table(_rows_to_table(merged_rows), self.merges_file)

    def read_merges(self) -> pd.DataFrame:
        """The merge table, or an empty DataFrame if none has been
        written yet."""
        if not self.merges_file.exists():
            return _empty_table().to_pandas()
        return pq.read_table(self.merges_file).to_pandas()

    # ---------------------------------------------------------------
    # Final consolidation: write-once, streamed, never partial
    # ---------------------------------------------------------------
    def is_consolidated(self) -> bool:
        """Whether the final consolidated file has been built."""
        return self.consolidated_file.exists()

    def consolidate(self, absorbed_ids) -> None:
        """Stream every per-chunk file into one final Parquet file,
        dropping rows whose object_id was absorbed into a merge (its
        replacement lives in the merge table), then append the merge
        rows. Written to a temp file and renamed atomically, so
        is_consolidated() only ever sees a complete file or none."""
        tmp = self.consolidated_file.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(tmp, _SCHEMA)
        try:
            absorbed = np.asarray(absorbed_ids, dtype=np.int64)
            absorbed_array = pa.array(absorbed) if absorbed.size else None
            for path in sorted(self.meta_dir.glob("chunk_*.parquet")):
                table = pq.read_table(path)
                if absorbed_array is not None and table.num_rows:
                    keep = pc.invert(pc.is_in(table["object_id"], value_set=absorbed_array))
                    table = table.filter(keep)
                if table.num_rows:
                    writer.write_table(table)

            merges = pq.read_table(self.merges_file) if self.merges_file.exists() else _empty_table()
            if merges.num_rows:
                writer.write_table(merges)
        finally:
            writer.close()
        tmp.replace(self.consolidated_file)

    def read_consolidated(self) -> pd.DataFrame:
        """The final, deduplicated metadata table (requires
        is_consolidated())."""
        return pq.read_table(self.consolidated_file).to_pandas()

    def cleanup_chunk_files(self, max_workers: int = 8) -> int:
        """Delete the per-chunk Parquet files now that their data lives
        in the consolidated file. Refuses if is_consolidated() is False,
        since the per-chunk files would then be the only copy of that
        data. Idempotent -- a second call finds nothing left and returns
        0. Deletions run on a thread pool: unlink() releases the GIL
        while it waits, and on a network filesystem that wait is a real
        round trip, so several can be in flight at once; tune
        max_workers to your filesystem. One-way in practice: after
        cleanup, rebuilding a chunk's metadata means re-running Pass 1
        for it (clear its progress marker first) -- fully recoverable
        from the original semantic volume, just recomputed rather than
        restored."""
        if not self.is_consolidated():
            raise RuntimeError(
                "refusing to delete per-chunk metadata before the "
                "consolidated file exists at "
                f"{self.consolidated_file} -- that data has no other copy yet."
            )
        paths = list(self.meta_dir.glob("chunk_*.parquet"))
        if not paths:
            return 0

        def _safe_unlink(path: Path) -> bool:
            try:
                # missing_ok=True: another concurrent cleanup call (or a
                # retried run) may have already removed this file --
                # that's success, not an error, for an idempotent cleanup.
                path.unlink(missing_ok=True)
                return True
            except OSError as exc:
                print(f"Warning: failed to remove {path}: {exc}")
                return False

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_safe_unlink, paths))
        return sum(results)