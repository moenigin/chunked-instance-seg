import datetime
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import time
import yaml

from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from joblib import Parallel, delayed
from pathlib import Path
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from typing import Optional, Union

from source.data_io import DataIO, Zarr2DataIO
from source.metadata_handler import ObjectMetadata, ParquetChunkMetadataStore
from source.utils import IDScheme, generate_chunks, mark_completed, \
    filter_remaining_chunks, mk_dir, resolve_path


@dataclass
class InstanceParams:
    input_data: DataIO
    output_data: DataIO
    scratch_dir: Union[
        str, Path]  # resumable intermediate state (per-chunk metadata, progress markers) -- safe to point at fast/local storage, cleaned up explicitly via pipeline.cleanup_scratch() once you trust the result
    metadata_path: Union[
        str, Path]  # final, permanent consolidated metadata file -- promoted out of scratch once Pass 3 finishes
    chunk_size: tuple
    stack_dim: Optional[list] = None
    src_origin: Optional[np.ndarray] = None
    target_label: int = 1
    min_object_size: int = 300
    parallel_backend: str = 'loky'  # 'loky' = multiprocessing-safe (default); 'threading' also supported
    cleanup_chunk_metadata: bool = True  # delete per-chunk Parquet files once consolidated (see note in run_pipeline_parallel)
    verbose: bool = False
    generate_report: bool = True
    report_dir: Optional[
        Union[str, Path]] = None  # defaults to metadata_path.parent / "report"
    list_empty_chunks: bool = False
    delete_scratch: bool = False


def _pairs_from_faces(face1: Optional[np.ndarray],
                      face2: Optional[np.ndarray]) -> np.ndarray:
    """Vectorized replacement for the old pixel-by-pixel Python double
    loop. Returns an (n, 2) array of unique (id1, id2) pairs of objects
    that touch across this boundary. Pure function, no shared state --
    safe under any parallelism model."""
    if face1 is None or face2 is None:
        return np.empty((0, 2), dtype=np.int64)
    if face1.shape != face2.shape:
        print(
            f"Warning: face shape mismatch {face1.shape} vs {face2.shape}; truncating to overlap")
        min_shape = tuple(min(a, b) for a, b in zip(face1.shape, face2.shape))
        face1 = face1[: min_shape[0], : min_shape[1]]
        face2 = face2[: min_shape[0], : min_shape[1]]
    mask = (face1 > 0) & (face2 > 0) & (face1 != face2)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int64)
    stacked = np.stack([face1[mask], face2[mask]], axis=1).astype(np.int64)
    return np.unique(stacked, axis=0)


def get_absorbed_ids(resolved_mapping: dict) -> set:
    """Every object id that ends up represented by a merged row instead
    of its own original per-chunk row -- i.e. every key and every value
    appearing anywhere in the final, fully-resolved id mapping."""
    if not resolved_mapping:
        return set()
    return set(resolved_mapping.keys()) | set(resolved_mapping.values())


class InstanceSegmentationPipeline:
    def __init__(self, params: InstanceParams):
        """
        to crop a subvolume form a bigger stack that does not start at [0,0,0]
        set the scr_origin to the desired origin in the big stack and set
        stack_dim to the desired crop size [[0,0,0], ]
        :param params:
        """
        self.semanticIO = params.input_data
        self.instancesIO = params.output_data
        self.chunk_size = np.array(params.chunk_size)
        self.target_label = params.target_label
        self.min_object_size = params.min_object_size
        self.verbose = params.verbose
        self.src_origin = params.src_origin
        self.parallel_backend = params.parallel_backend
        self.cleanup_chunk_metadata = params.cleanup_chunk_metadata
        self.scratch_dir = mk_dir(params.scratch_dir)
        self.metadata_path = Path(params.metadata_path)
        self.meta_store = ParquetChunkMetadataStore(self.scratch_dir / "meta")
        self.generate_report = params.generate_report
        self.report_dir = params.report_dir
        self.list_empty_chunks = params.list_empty_chunks
        self.delete_scratch = params.delete_scratch

        if params.stack_dim is None:
            self.stack_dim = np.array([[0, 0, 0], self.semanticIO.shape])
        else:
            self.stack_dim = np.array(params.stack_dim)

        # All resumable intermediate state lives under scratch_dir --
        # nothing here has value once the run is done and metadata_path
        # is trusted; see cleanup_scratch().
        self.progress_dir_pass1 = mk_dir(self.scratch_dir / "progress_pass1")
        self.progress_dir_cc_analysis = mk_dir(
            self.scratch_dir / "progress_pass2")
        self.progress_dir_pass3 = mk_dir(self.scratch_dir / "progress_pass3")

        # id_mapping.parquet holds the final, fully-resolved (old_id -> root_id)
        # mapping once Pass 2 is complete -- its presence is the completion
        # marker for the whole boundary-analysis stage.
        self.id_mapping_file = self.meta_store.meta_dir / "_id_mapping.parquet"

        # One IDScheme built from the full, deterministic chunk list --
        # every chunk must agree on the same chunk_key <-> origin mapping
        # for resuming to be safe, however building it once resulted
        self.id_scheme = IDScheme(self.stack_dim, self.chunk_size) #
        if self.verbose:
            print(f"ID scheme: {self.id_scheme.describe()}", flush=True)

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def get_chunk_list(self, dir_) -> list:
        """Retrieve list of chunks from the progress directory"""
        chunk_list = generate_chunks(self.stack_dim, self.chunk_size)
        start = time.time()
        n_chunks = len(chunk_list)
        chunk_list = filter_remaining_chunks(chunk_list, dir_)
        stop = time.time()
        print(
            f"filtering {n_chunks - len(chunk_list)} items took {stop - start} sec",
            flush=True)
        return chunk_list

    def get_chunk_slice(self, origin: np.ndarray,
                        far_corner: np.ndarray) -> tuple:
        """Convert origin and far_corner to slice tuple"""
        return tuple(slice(o, f) for o, f in zip(origin, far_corner))

    def _chunk_id(self, origin: np.ndarray) -> str:
        return f"{origin[0]}_{origin[1]}_{origin[2]}"

    def _chunk_id_for_object(self, object_id: int) -> str:
        origin = self.id_scheme.origin_for_key(
            self.id_scheme.unpack_chunk_key(object_id))
        return self._chunk_id(origin)

    # ------------------------------------------------------------------
    # Pass 1: per-chunk semantic -> instance segmentation
    # ------------------------------------------------------------------
    def mk_chunk_instances(self, origin: np.ndarray, far_corner: np.ndarray):
        """Process a single chunk: semantic -> instance segmentation.
        Fully self-contained -- the only external state it reads is the
        (already-finalized, never-modified-by-this-pass) semantic input
        and this chunk's own coordinates, and the only state it writes
        is this chunk's own zarr region and its own metadata file. That
        means re-running this on an already-completed chunk (e.g. after
        a crash right before the completion marker was written) is
        always safe: it reproduces identical output, not new ids."""
        start_time = datetime.datetime.now()
        chunk_id = self._chunk_id(origin)
        targ_slice = self.get_chunk_slice(origin, far_corner)
        if self.src_origin is not None:
            src_slice = self.get_chunk_slice(origin + self.src_origin,
                                             far_corner + self.src_origin)
        else:
            src_slice = targ_slice

        semantic_chunk = self.semanticIO.get_data(src_slice)
        binary_mask = (semantic_chunk == self.target_label)

        labeled, n_objects = ndimage.label(binary_mask)
        objects_metadata = []
        if n_objects:
            sizes = np.bincount(labeled.ravel(), minlength=n_objects + 1)
            kept = np.flatnonzero(sizes >= self.min_object_size)
            kept = kept[kept > 0]
            if kept.size:
                chunk_key = self.id_scheme.chunk_key(origin)
                ids = self.id_scheme.pack(chunk_key, np.arange(1,kept.size + 1))  # vectorised, must still raise on overflow
                lut = np.zeros(n_objects + 1, dtype=np.int64)
                lut[kept] = ids
                final_label_chunk = lut[labeled]
                boxes = ndimage.find_objects(labeled) # bboxes for all labels in one pass
                for old, oid in zip(kept, ids):
                    sl = boxes[old - 1]
                    bbox = tuple(int(origin[i] + sl[i].start) for i in range(3)) \
                           + tuple(int(origin[i] + sl[i].stop - 1) for i in range(3))
                    objects_metadata.append(
                        ObjectMetadata(int(oid), chunk_id, bbox, int(sizes[old])))
                self.instancesIO.write_data(final_label_chunk, targ_slice)

        self.meta_store.write_chunk_metadata(chunk_id, objects_metadata)

        if self.verbose:
            duration = (datetime.datetime.now() - start_time).total_seconds()
            print(
                f"Created instances for chunk {origin} with {n_objects} objects in {duration:.2f} seconds",
                flush=True)

        mark_completed(chunk_coords=np.array([origin, far_corner]),
                       progress_dir=self.progress_dir_pass1)


    def run_instance_pass1(self, n_jobs: int = -1):
        print("Starting parallelized instance segmentation pipeline...")
        chunks = self.get_chunk_list(self.progress_dir_pass1)
        print(f"Pass 1: Processing {len(chunks)} chunks in parallel...")

        Parallel(n_jobs=n_jobs, backend=self.parallel_backend)(
            delayed(self.mk_chunk_instances)(origin, far_corner)
            for origin, far_corner in tqdm(chunks, desc="Processing chunks")
        )

    # ------------------------------------------------------------------
    # Pass 2: find objects connected across chunk boundaries
    # ------------------------------------------------------------------
    def get_neighboring_chunks(self, origin: np.ndarray,
                               far_corner: np.ndarray) -> list:
        """Get neighboring chunks that share boundaries (+z, +y, +x only,
        which is sufficient to find every adjacent pair exactly once)."""
        neighbors = []
        neighbor_offsets = [
            [self.chunk_size[0], 0, 0],
            [0, self.chunk_size[1], 0],
            [0, 0, self.chunk_size[2]],
        ]
        for offset in neighbor_offsets:
            neighbor_origin = origin + offset
            if np.all(neighbor_origin < self.stack_dim[1]):
                neighbor_far_corner = np.minimum(
                    neighbor_origin + self.chunk_size, self.stack_dim[1])
                neighbors.append((neighbor_origin, neighbor_far_corner))
        return neighbors

    def _get_boundary_faces(self, origin1, far_corner1, origin2, far_corner2):
        """Extract boundary faces between two neighboring chunks."""
        diff = origin2 - origin1
        boundary_axis = int(np.argmax(np.abs(diff)))

        slice1 = self.get_chunk_slice(origin1, far_corner1)
        slice2 = self.get_chunk_slice(origin2, far_corner2)

        if boundary_axis == 0:
            if diff[0] > 0:
                face1 = self.instancesIO.get_data(
                    (slice(far_corner1[0] - 1, far_corner1[0]), slice1[1],
                     slice1[2]))
                face2 = self.instancesIO.get_data(
                    (slice(origin2[0], origin2[0] + 1), slice2[1], slice2[2]))
            else:
                face1 = self.instancesIO.get_data(
                    (slice(origin1[0], origin1[0] + 1), slice1[1], slice1[2]))
                face2 = self.instancesIO.get_data(
                    (slice(far_corner2[0] - 1, far_corner2[0]), slice2[1],
                     slice2[2]))
        elif boundary_axis == 1:
            if diff[1] > 0:
                face1 = self.instancesIO.get_data(
                    (slice1[0], slice(far_corner1[1] - 1, far_corner1[1]),
                     slice1[2]))
                face2 = self.instancesIO.get_data(
                    (slice2[0], slice(origin2[1], origin2[1] + 1), slice2[2]))
            else:
                face1 = self.instancesIO.get_data(
                    (slice1[0], slice(origin1[1], origin1[1] + 1), slice1[2]))
                face2 = self.instancesIO.get_data(
                    (slice2[0], slice(far_corner2[1] - 1, far_corner2[1]),
                     slice2[2]))
        else:
            if diff[2] > 0:
                face1 = self.instancesIO.get_data((slice1[0], slice1[1],
                                                   slice(far_corner1[2] - 1,
                                                         far_corner1[2])))
                face2 = self.instancesIO.get_data(
                    (slice2[0], slice2[1], slice(origin2[2], origin2[2] + 1)))
            else:
                face1 = self.instancesIO.get_data(
                    (slice1[0], slice1[1], slice(origin1[2], origin1[2] + 1)))
                face2 = self.instancesIO.get_data((slice2[0], slice2[1],
                                                   slice(far_corner2[2] - 1,
                                                         far_corner2[2])))

        return np.squeeze(face1), np.squeeze(face2)

    def _pair_file(self, origin1: np.ndarray, origin2: np.ndarray) -> Path:
        name = "pairs_" + "_".join(map(str, origin1)) + "__" + "_".join(
            map(str, origin2)) + ".parquet"
        return self.progress_dir_cc_analysis / name

    def _find_connected_across_boundary(self, origin1, far_corner1, origin2,
                                        far_corner2) -> None:
        """Find connected components across one chunk-pair boundary and
        persist the result to its own file. No shared state is written
        here -- each call only ever touches its own pairs file, so this
        needs no lock and is safe under multiple processes."""
        pair_path = self._pair_file(origin1, origin2)
        if pair_path.exists():
            return  # already computed on a previous, interrupted run

        start_time = datetime.datetime.now()
        face1, face2 = self._get_boundary_faces(origin1, far_corner1, origin2,
                                                far_corner2)
        pairs = _pairs_from_faces(face1, face2)

        if pairs.size == 0:
            df = pd.DataFrame({"id1": pd.Series(dtype="int64"),
                               "id2": pd.Series(dtype="int64")})
        else:
            df = pd.DataFrame(pairs, columns=["id1", "id2"]).astype("int64")
        tmp = pair_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(pair_path)

        if self.verbose:
            duration = (datetime.datetime.now() - start_time).total_seconds()
            print(
                f"Found {len(pairs)} boundary pairs for {origin1} & {origin2} in {duration:.2f}s",
                flush=True)

    def _build_chunk_pairs(self, chunks: list) -> list:
        chunk_pairs = []
        for origin, far_corner in chunks:
            for neighbor_origin, neighbor_far_corner in self.get_neighboring_chunks(
                    origin, far_corner):
                chunk_pairs.append(((origin, far_corner),
                                    (neighbor_origin, neighbor_far_corner)))
        return chunk_pairs

    @staticmethod
    def _find_root(obj_id: int, id_mapping: dict) -> int:
        """Union-find root lookup with path compression."""
        if obj_id not in id_mapping:
            return obj_id
        if id_mapping[obj_id] != obj_id:
            id_mapping[obj_id] = InstanceSegmentationPipeline._find_root(
                id_mapping[obj_id], id_mapping)
        return id_mapping[obj_id]

    def _read_pair_files(self, max_workers: int = 16):
        files = sorted(self.progress_dir_cc_analysis.glob("pairs_*.parquet"))

        def _read(path):
            t = pq.read_table(path)
            return (t["id1"].to_numpy().astype(np.int64),
                    t["id2"].to_numpy().astype(np.int64))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            parts = list(pool.map(_read, files))
        if not parts:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return (np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]))

    def _consolidate_id_mapping(self):
        """Returns (old, new): int64 arrays sorted by old.  root = smallest id
        of each merged group; only non-root ids appear in `old`."""
        if self.id_mapping_file.exists():
            t = pq.read_table(self.id_mapping_file)
            return (t["old_id"].to_numpy().astype(np.int64),
                    t["new_id"].to_numpy().astype(np.int64))

        id1, id2 = self._read_pair_files()
        if id1.size == 0:
            old = new = np.empty(0, np.int64)
        else:
            # dense 0..m-1 index per distinct id; `ids` is sorted ascending
            ids, inv = np.unique(np.concatenate([id1, id2]),
                                 return_inverse=True)
            inv = inv.ravel()
            a, b = inv[:id1.size], inv[id1.size:]
            graph = coo_matrix((np.ones(a.size, dtype=np.int32), (a, b)),
                               shape=(ids.size, ids.size))
            _, comp = connected_components(graph, directed=False)
            # first occurrence of each component in `comp` = lowest index =
            # lowest id (because ids is sorted) -> that is the root
            _, first = np.unique(comp, return_index=True)
            root = ids[first][comp]
            keep = ids != root
            old, new = ids[keep], root[keep]

        out = pa.table({"old_id": pa.array(old, type=pa.int64()),
                        "new_id": pa.array(new, type=pa.int64())})
        tmp = self.id_mapping_file.with_suffix(".parquet.tmp")
        pq.write_table(out, tmp)
        tmp.replace(self.id_mapping_file)
        return old, new

    def run_connected_component_analysis(self, n_jobs: int = -1):
        """Same as before, but returns (old, new) arrays."""
        from source.utils import generate_chunks  # local import: patch file
        print("Pass 2: Finding connected components across boundaries...")
        if self.id_mapping_file.exists():
            print("id_mapping already computed, skipping boundary search")
            return self._consolidate_id_mapping()

        chunks = generate_chunks(self.stack_dim, self.chunk_size)
        chunk_pairs = self._build_chunk_pairs(chunks)
        chunk_pairs = [p for p in chunk_pairs
                       if not self._pair_file(p[0][0], p[1][0]).exists()]
        if chunk_pairs:
            Parallel(n_jobs=n_jobs, backend=self.parallel_backend)(
                delayed(self._find_connected_across_boundary)(o1, f1, o2, f2)
                for (o1, f1), (o2, f2) in
                tqdm(chunk_pairs, desc="find objects across chunk boundaries"))
        else:
            print("All chunk pairs already processed")
        return self._consolidate_id_mapping()

    # ------------------------------------------------------------------
    # Pass 3: relabel affected chunks + compute merged metadata
    # ------------------------------------------------------------------
    def _mapping_blocks(self, old):
        """`old` is sorted, and the chunk key sits in the HIGH bits of every
        id, so all ids of one chunk are adjacent.  Returns, per chunk that has
        entries: (chunk_keys, block_start, block_end) into `old`/`new`."""
        keys = self.id_scheme.unpack_chunk_key(old)
        chunk_keys, first = np.unique(keys, return_index=True)
        ends = np.append(first[1:], old.size)
        return chunk_keys, first, ends

    def get_affected_chunks(self, chunk_keys) -> list:
        origins = self.id_scheme.origins_for_keys(chunk_keys)
        far = np.minimum(origins + self.chunk_size, self.stack_dim[1])
        return [np.array([o, f]) for o, f in zip(origins, far)]

    def _apply_relabelling(self, origin, far_corner, old_block, new_block):
        """Relabel one chunk with only ITS entries (old_block sorted)."""
        import datetime
        from source.utils import mark_completed
        start_time = datetime.datetime.now()
        if old_block.size:
            chunk_slice = self.get_chunk_slice(origin, far_corner)
            chunk_data = self.instancesIO.get_data(chunk_slice)
            old_b = old_block.astype(chunk_data.dtype, copy=False)
            new_b = new_block.astype(chunk_data.dtype, copy=False)
            nz = chunk_data != 0                 # background never matches
            vals = chunk_data[nz]
            pos = np.searchsorted(old_b, vals)   # binary search per voxel
            pos[pos == old_b.size] = 0           # ids above every entry
            hit = old_b[pos] == vals
            vals[hit] = new_b[pos[hit]]
            chunk_data[nz] = vals
            self.instancesIO.write_data(chunk_data, chunk_slice)
        if self.verbose:
            d = (datetime.datetime.now() - start_time).total_seconds()
            print(f"Updated labels in chunk {origin} in {d:.2f} seconds",
                  flush=True)
        mark_completed(chunk_coords=np.array([origin, far_corner]),
                       progress_dir=self.progress_dir_pass3)

    def run_relabeling(self, old, new, n_jobs: int = -1):
        from source.utils import filter_remaining_chunks
        print("Pass 3: Applying relabeling...")
        chunk_keys, first, ends = self._mapping_blocks(old)
        affected = self.get_affected_chunks(chunk_keys)
        affected = filter_remaining_chunks(affected, self.progress_dir_pass3)
        if not affected:
            print("All affected chunks already relabeled")
            return

        def _tasks():
            for origin, far_corner in tqdm(affected, desc="Updating labels"):
                i = np.searchsorted(chunk_keys, self.id_scheme.chunk_key(origin))
                sl = slice(first[i], ends[i])
                yield origin, far_corner, old[sl], new[sl]

        Parallel(n_jobs=n_jobs, backend=self.parallel_backend)(
            delayed(self._apply_relabelling)(o, f, ob, nb)
            for o, f, ob, nb in _tasks())

    def compute_merged_metadata(self, old, new, max_workers: int = 16):
        if self.meta_store.merges_computed():
            return
        if old.size == 0:
            self.meta_store.write_merges([])
            return

        roots = np.unique(new)
        ids_all = np.concatenate([old, roots])       # every merged fragment
        root_all = np.concatenate([new, roots])      # ... and its root
        order = np.argsort(ids_all)
        ids_all, root_all = ids_all[order], root_all[order]

        # only chunks that contain at least one involved object
        keys = np.unique(self.id_scheme.unpack_chunk_key(ids_all))
        origins = self.id_scheme.origins_for_keys(keys)

        def _load(origin):
            df = self.meta_store.read_chunk_metadata(self._chunk_id(origin))
            oid = df["object_id"].to_numpy()
            pos = np.searchsorted(ids_all, oid)
            pos[pos == ids_all.size] = 0
            hit = ids_all[pos] == oid
            if not hit.any():
                return None
            sub = df.loc[hit].drop(columns=["object_id", "chunk_id"])
            sub["root"] = root_all[pos[hit]]
            return sub

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            parts = [p for p in pool.map(_load, origins) if p is not None]
        if not parts:            # every member filtered out upstream
            self.meta_store.write_merges([])
            return

        out = (pd.concat(parts, ignore_index=True)
               .groupby("root", sort=True)
               .agg(bbox_z_min=("bbox_z_min", "min"),
                    bbox_y_min=("bbox_y_min", "min"),
                    bbox_x_min=("bbox_x_min", "min"),
                    bbox_z_max=("bbox_z_max", "max"),
                    bbox_y_max=("bbox_y_max", "max"),
                    bbox_x_max=("bbox_x_max", "max"),
                    nvoxels=("nvoxels", "sum"))
               .reset_index().rename(columns={"root": "object_id"}))
        o = self.id_scheme.origins_for_keys(
            self.id_scheme.unpack_chunk_key(out["object_id"].to_numpy()))
        out["chunk_id"] = (pd.Series(o[:, 0]).astype(str) + "_"
                           + pd.Series(o[:, 1]).astype(str) + "_"
                           + pd.Series(o[:, 2]).astype(str))
        self.meta_store.write_merges(out)

    # ------------------------------------------------------------------
    # promotion + scratch lifecycle
    # ------------------------------------------------------------------
    def _promote_consolidated(self):
        """Copy the scratch-internal consolidated file out to the
        permanent, public metadata_path. Cheap and idempotent -- safe
        to call every run; the scratch copy remains the resumability
        marker checked by is_consolidated()."""
        tmp = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        shutil.copy2(self.meta_store.consolidated_file, tmp)
        tmp.replace(self.metadata_path)

    def cleanup_scratch(self):
        """Delete the entire scratch directory (per-chunk metadata still
        present, progress markers, pairs files, id_mapping, the
        scratch-internal consolidated copy). NOT called automatically --
        call this yourself once you've verified the promoted
        metadata_path and the output volume are correct. After this,
        the run cannot be resumed; a re-run starts Pass 1 from scratch."""
        shutil.rmtree(self.scratch_dir)

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def run_pipeline_parallel(self, n_jobs: int = -1):
        """Run the complete pipeline. Safe to call again after an
        interruption at any point -- every stage checks what's already
        done before doing it again.

        Does NOT run postprocessing (expand_labels, size filtering,
        etc.) -- that's a separate step over the finalized output of
        this pipeline; see postprocess_instances.py."""
        self.run_instance_pass1(n_jobs)
        old, new = self.run_connected_component_analysis(n_jobs)

        if old.size:
            self.run_relabeling(old, new, n_jobs)
        self.compute_merged_metadata(old, new)

        if not self.meta_store.is_consolidated():
            self.meta_store.consolidate(np.union1d(old, new))

        self._promote_consolidated()

        # cleanup_chunk_files() only deletes files once that final copy of the
        # data is confirmed present.
        if self.cleanup_chunk_metadata:
            n_removed = self.meta_store.cleanup_chunk_files()
            if self.verbose and n_removed:
                print(
                    f"Removed {n_removed} per-chunk metadata files after consolidation",
                    flush=True)

        if self.generate_report:
            from metadata_report import generate_report
            all_chunk_ids = [self._chunk_id(origin) for origin, _ in
                             generate_chunks(self.stack_dim, self.chunk_size)]
            generate_report(pd.read_parquet(self.metadata_path), all_chunk_ids,
                            out_dir=self.report_dir or self.metadata_path.parent / "report",
                            list_empty_chunks=self.list_empty_chunks)

        if self.delete_scratch:
            self.cleanup_scratch()

        print(f"Pipeline completed! Metadata: {self.metadata_path}")


def parse_cfg(cfg_params):
    project_dir = mk_dir(cfg_params["project_dir"])
    output_volume = Path(cfg_params['instance_vol'])
    if output_volume.parent == Path("."):
        output_volume = project_dir.joinpath(output_volume)

    instance_vol_path = Path(cfg_params["instance_vol"])
    default_meta_name = f"{instance_vol_path.stem}_metadata.parquet"
    cfg_params["metadata_path"] = resolve_path(cfg_params.get("metadata_path"),
                                               default_path=project_dir / default_meta_name,
                                               base_dir=project_dir)

    cfg_params["scratch_dir"] = resolve_path(cfg_params.get("scratch_dir"),
                                             default_path=project_dir / "scratch",
                                             base_dir=project_dir)

    if cfg_params['io_func'] == 'zarr2':
        input_data = Zarr2DataIO(cfg_params['semantic_vol'])
        array_shape = cfg_params.get("volume_size", input_data.shape)

        output_data = Zarr2DataIO(output_volume, cfg_params['chunk_size'],
                                  array_shape)
    else:
        raise NotImplementedError(f"{cfg_params['io_func']} is not implemented")

    valid_keys = {f.name for f in fields(InstanceParams)}
    filtered_params = {k: v for k, v in cfg_params.items() if k in valid_keys}
    return InstanceParams(input_data=input_data, output_data=output_data,
                          **filtered_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="")
    parser.add_argument("--generate-report", action=BooleanOptionalAction,
                        default=None)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg_params = yaml.safe_load(f)

    if args.generate_report is not None:
        cfg_params["generate_report"] = args.generate_report

    params = parse_cfg(cfg_params)

    if params.verbose:
        print(f"__main__: Starting instance segmentation pipeline with following parameters: {cfg_params}")

    pipeline = InstanceSegmentationPipeline(params)
    pipeline.run_pipeline_parallel()
