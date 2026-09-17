"""
Postprocessing over a finalized instance volume + its consolidated
metadata -- run as a separate step, after instance_segmentation_pipeline.py
has produced <metadata_path> and the labeled Zarr volume.

WHY THIS IS ITS OWN SCRIPT
----------------------------
Any operation that changes which voxels belong to which object also
invalidates that object's bbox/voxel-count metadata. Folding that
bookkeeping into each individual operation (as the original
`postprocess_expand` did implicitly, by never updating it at all) means
every new postprocessing step has to reinvent its own, easy-to-get-wrong
metadata-repair logic. Here, that responsibility is split cleanly by
what a step actually needs:

- RescanStep: mutates voxels in a way that can change any object's
  extent unpredictably (expand_labels is the example here). The runner
  always follows one of these with a full volume rescan that recomputes
  metadata from the ground truth of what's actually in the volume --
  no step has to implement its own metadata patching.
- TargetedStep: only removes or otherwise resolves whole objects whose
  affected region is already exactly known from the CURRENT metadata
  (e.g. dropping objects outside a size range). This never needs a
  volume-wide rescan -- it edits the metadata table directly and only
  touches the specific voxels of the objects it's removing.

The runner (`run_postprocessing`) only pays for the expensive full
rescan when a RescanStep actually ran since the last one, and only once
even if several RescanSteps are chained back to back.

THE HALO REQUIREMENT -- WHY expand_labels CAN'T RUN CHUNK-LOCAL, IN PLACE
----------------------------------------------------------------------------
Two things have to both be true for chunked processing here to be safe:

1. Each chunk's expand_labels call needs to SEE beyond its own chunk --
   otherwise an object a few voxels from a chunk boundary can't
   correctly compete for/claim background voxels that its actual
   (whole-volume) expansion would reach, and a neighboring object on
   the other side of that boundary is invisible to it entirely. That's
   the "run on overlapping chunks" fix: read each chunk padded by a
   halo of at least `distance` voxels, run expand_labels on the padded
   region, but only write back the unpadded core -- core regions across
   all chunks exactly tile the volume with no overlap, so every voxel
   is written by exactly one worker even though many workers' READS
   overlap.

2. Less obvious: the source volume being read from must not be mutated
   DURING this step. If chunk A's halo includes chunk B's core, and B
   has already been expanded and written back by the time A reads it,
   A sees a partially-expanded neighbor instead of the original
   segmentation -- the result becomes dependent on worker execution
   order, which is silently wrong and not reproducible between runs.
   That's why `ExpandLabelsStep` reads from `ctx.source_io` and always
   writes to a DIFFERENT destination array, never in place. Any future
   RescanStep that looks beyond its own chunk's un-grown footprint needs
   the same separation; a step that only ever touches exactly the voxels
   already recorded for known objects (i.e. a TargetedStep) does not,
   since there's no neighbor-halo dependency to race on.
"""
import numpy as np
import pandas as pd
import time
import yaml

from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from joblib import Parallel, delayed
from pathlib import Path
from skimage.segmentation import expand_labels
from tqdm import tqdm
from typing import Optional, Union

from source.data_io import DataIO, Zarr2DataIO
from source.instance_segmentation_pipeline import parse_cfg
from source.metadata_handler import _COLUMNS  # same schema/column names as the main pipeline -- single source of truth
from source.utils import generate_chunks, mark_completed, \
    filter_remaining_chunks, mk_dir, resolve_path



def get_slice(origin: np.ndarray, far_corner: np.ndarray) -> tuple:
    return tuple(slice(int(o), int(f)) for o, f in zip(origin, far_corner))


def _new_zarr_like(source_io: DataIO, path: Union[str, Path], chunk_size: np.ndarray) -> Zarr2DataIO:
    """Create a fresh Zarr array with the same shape/dtype as the
    source, chunked the same way the pipeline chunks it. RescanSteps
    write here, never back into the source they're reading from."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists -- refusing to overwrite a RescanStep "
            f"destination. Delete it first if you intend to redo this step "
            f"from scratch, or point dest_path somewhere new."
        )
    return Zarr2DataIO(path, chunk_size=tuple(int(c) for c in chunk_size), array_shape=source_io.shape)


@dataclass
class PostprocessContext:
    io_func: DataIO
    chunk_size: np.ndarray
    stack_dim: np.ndarray
    scratch_dir: Path
    parallel_backend: str = 'loky'
    n_jobs: int = -1
    verbose: bool = False


class RescanStep(ABC):
    """A step that mutates voxels in a way that can change any object's
    extent unpredictably. The runner always triggers a full metadata
    rescan after one or more of these run consecutively, before the
    next TargetedStep (or the end of the pipeline)."""

    name: str = "rescan_step"

    @abstractmethod
    def apply_to_volume(self, ctx: PostprocessContext) -> DataIO:
        """Read from ctx.source_io, write the result to a NEW
        destination (never back into ctx.source_io -- see module
        docstring), and return the DataIO for that destination. The
        runner adopts the return value as the new ctx.source_io for
        any subsequent step."""


class TargetedStep(ABC):
    """A step whose voxel changes are already exactly known from the
    CURRENT metadata (e.g. "remove object ids X, Y, Z") -- no
    volume-wide rescan needed. Safe to mutate ctx.source_io in place,
    since it never depends on a neighbor's un-mutated state."""

    name: str = "targeted_step"

    @abstractmethod
    def apply(self, ctx: PostprocessContext, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the edit to ctx.source_io and return the updated
        metadata table (e.g. with removed objects' rows dropped)."""


# ----------------------------------------------------------------------
# RescanStep: expand_labels on haloed, non-overlapping-write chunks
# ----------------------------------------------------------------------

class ExpandLabelsStep(RescanStep):
    name = "expand_labels"

    def __init__(self, distance: int, dest_path: Union[str, Path], halo: Optional[int] = None):
        self.distance = distance
        self.dest_path = Path(dest_path)
        # Must be >= distance for correctness (see module docstring);
        # a little extra margin is cheap insurance, not required.
        self.halo = halo if halo is not None else distance

    def apply_to_volume(self, ctx: PostprocessContext) -> DataIO:
        progress_dir = mk_dir(ctx.scratch_dir / f"progress_postprocess_{self.name}")
        dest_io = (
            Zarr2DataIO(self.dest_path)
            if self.dest_path.exists()
            else _new_zarr_like(ctx.io_func, self.dest_path, ctx.chunk_size)
        )

        chunks = generate_chunks(ctx.stack_dim, ctx.chunk_size)
        remaining = filter_remaining_chunks(chunks, progress_dir)
        print(f"{self.name}: {len(chunks) - len(remaining)}/{len(chunks)} chunks already done")
        if remaining:
            Parallel(n_jobs=ctx.n_jobs, backend=ctx.parallel_backend)(
                delayed(self._process_chunk)(chunk[0], chunk[1], ctx, dest_io, progress_dir)
                for chunk in tqdm(remaining, desc=f"{self.name} (haloed)")
            )
        return dest_io

    def _process_chunk(self, origin, far_corner, ctx: PostprocessContext, dest_io: DataIO, progress_dir: Path):
        halo = np.full(3, self.halo, dtype=np.int64)
        padded_origin = np.maximum(origin - halo, ctx.stack_dim[0])
        padded_far = np.minimum(far_corner + halo, ctx.stack_dim[1])

        # Read from the IMMUTABLE source -- never dest_io -- so this
        # chunk's neighbors can't have been mutated yet no matter what
        # order workers finish in.
        padded = ctx.io_func.get_data(get_slice(padded_origin, padded_far))
        expanded = expand_labels(padded, distance=self.distance)

        core_start = origin - padded_origin
        core_end = core_start + (far_corner - origin)
        core = expanded[core_start[0]:core_end[0], core_start[1]:core_end[1], core_start[2]:core_end[2]]
        dest_io.write_data(core, get_slice(origin, far_corner))

        mark_completed(chunk_coords=np.array([origin, far_corner]), progress_dir=progress_dir)


# ----------------------------------------------------------------------
# TargetedStep: remove objects outside a voxel-count range
# ----------------------------------------------------------------------

def _chunks_overlapping_bbox(bbox, chunk_size: np.ndarray, stack_dim: np.ndarray) -> list:
    """Every pipeline-chunk origin whose region intersects this bbox --
    a bbox from a merged object can legitimately span several chunks."""
    lo = np.array(bbox[:3])
    hi = np.array(bbox[3:])
    grid_lo = np.maximum((lo - stack_dim[0]) // chunk_size, 0)
    grid_hi = (hi - stack_dim[0]) // chunk_size
    origins = []
    for gz in range(grid_lo[0], grid_hi[0] + 1):
        for gy in range(grid_lo[1], grid_hi[1] + 1):
            for gx in range(grid_lo[2], grid_hi[2] + 1):
                origin = stack_dim[0] + np.array([gz, gy, gx]) * chunk_size
                if np.all(origin < stack_dim[1]):
                    origins.append(origin)
    return origins


class SizeFilterStep(TargetedStep):
    name = "size_filter"

    def __init__(self, min_nvoxels: Optional[int] = None, max_nvoxels: Optional[int] = None):
        if min_nvoxels is None and max_nvoxels is None:
            raise ValueError("SizeFilterStep needs at least one of min_nvoxels/max_nvoxels")
        self.min_nvoxels = min_nvoxels
        self.max_nvoxels = max_nvoxels

    def apply(self, ctx: PostprocessContext, metadata_df: pd.DataFrame) -> pd.DataFrame:
        keep = pd.Series(True, index=metadata_df.index)
        if self.min_nvoxels is not None:
            keep &= metadata_df["nvoxels"] >= self.min_nvoxels
        if self.max_nvoxels is not None:
            keep &= metadata_df["nvoxels"] <= self.max_nvoxels

        to_remove = metadata_df.loc[~keep]
        print(f"{self.name}: removing {len(to_remove)}/{len(metadata_df)} objects")
        if len(to_remove):
            self._remove_from_volume(ctx, to_remove)
        return metadata_df.loc[keep].reset_index(drop=True)

    @staticmethod
    def _remove_from_volume(ctx: PostprocessContext, rows: pd.DataFrame):
        # Group removed ids by every physical chunk their bbox touches,
        # so each chunk region is only ever written by one worker --
        # the same "one writer per region" rule Pass 3 uses, needed
        # here because two removed objects' bboxes can overlap the same
        # chunk.
        chunk_to_ids = defaultdict(list)
        for _, row in rows.iterrows():
            bbox = (row.bbox_z_min, row.bbox_y_min, row.bbox_x_min,
                    row.bbox_z_max, row.bbox_y_max, row.bbox_x_max)
            for origin in _chunks_overlapping_bbox(bbox, ctx.chunk_size, ctx.stack_dim):
                chunk_to_ids[tuple(origin.tolist())].append(int(row.object_id))

        def _process(origin_key, ids):
            origin = np.array(origin_key)
            far_corner = np.minimum(origin + ctx.chunk_size, ctx.stack_dim[1])
            sl = get_slice(origin, far_corner)
            data = ctx.io_func.get_data(sl)
            mask = np.isin(data, ids)
            if np.any(mask):
                data[mask] = 0
                ctx.io_func.write_data(data, sl)

        Parallel(n_jobs=ctx.n_jobs, backend=ctx.parallel_backend)(
            delayed(_process)(k, v) for k, v in tqdm(chunk_to_ids.items(), desc=f"{SizeFilterStep.name}: removing")
        )


# ----------------------------------------------------------------------
# Full metadata rescan -- ground truth from the volume, for use after
# any RescanStep
# ----------------------------------------------------------------------

def _rescan_chunk(origin, far_corner, source_io: DataIO) -> pd.DataFrame:
    data = source_io.get_data(get_slice(origin, far_corner))
    nz = np.nonzero(data)
    if len(nz[0]) == 0:
        return pd.DataFrame(columns=["object_id", "bbox_z_min", "bbox_y_min", "bbox_x_min",
                                      "bbox_z_max", "bbox_y_max", "bbox_x_max", "nvoxels"])
    ids_flat = np.asarray(data[nz]).astype(np.int64)
    # Vectorized groupby instead of a per-id np.where loop (what Pass 1
    # itself still does) -- much faster when a chunk holds many objects,
    # and there's no local-label bookkeeping to preserve here since ids
    # are already final.
    coords = pd.DataFrame({
        "object_id": ids_flat,
        "z": nz[0] + origin[0], "y": nz[1] + origin[1], "x": nz[2] + origin[2],
    })
    agg = coords.groupby("object_id").agg(
        bbox_z_min=("z", "min"), bbox_y_min=("y", "min"), bbox_x_min=("x", "min"),
        bbox_z_max=("z", "max"), bbox_y_max=("y", "max"), bbox_x_max=("x", "max"),
        nvoxels=("z", "size"),
    ).reset_index()
    return agg


def rescan_metadata(ctx: PostprocessContext) -> pd.DataFrame:
    """Ground-truth metadata recompute, scanning the current volume
    directly -- the only correct way to know an object's extent after
    an operation (like expand_labels) that can grow it into chunks it
    never originally touched. Two-level aggregate: per-chunk partial
    bbox/voxel-count contributions computed in parallel (each chunk
    only holds its own small summary in memory, not every voxel
    coordinate in the volume at once), then a second, cheap groupby
    over those small per-chunk summaries."""
    start = time.time()
    chunks = generate_chunks(ctx.stack_dim, ctx.chunk_size)
    partials = Parallel(n_jobs=ctx.n_jobs, backend=ctx.parallel_backend)(
        delayed(_rescan_chunk)(chunk[0], chunk[1], ctx.io_func)
        for chunk in tqdm(chunks, desc="rescanning metadata")
    )
    partials = [p for p in partials if len(p)]
    if not partials:
        result = pd.DataFrame(columns=["object_id", "chunk_id", *[c for c in _COLUMNS if c not in ("object_id", "chunk_id")]])
    else:
        full = pd.concat(partials, ignore_index=True)
        result = full.groupby("object_id").agg(
            bbox_z_min=("bbox_z_min", "min"), bbox_y_min=("bbox_y_min", "min"), bbox_x_min=("bbox_x_min", "min"),
            bbox_z_max=("bbox_z_max", "max"), bbox_y_max=("bbox_y_max", "max"), bbox_x_max=("bbox_x_max", "max"),
            nvoxels=("nvoxels", "sum"),
        ).reset_index()
        # chunk_id isn't meaningful post-rescan for objects spanning
        # multiple chunks (same documented limitation as the main
        # pipeline's merge rows) -- leave it unset rather than pick an
        # arbitrary one that looks more authoritative than it is.
        result["chunk_id"] = ""
    if ctx.verbose:
        print(f"rescan_metadata: {len(result)} objects in {time.time() - start:.1f}s", flush=True)
    return result[_COLUMNS]

# todo> check whether metafile gets iverwritten, if volume dies not neither should the metafile
# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def run_postprocessing(steps: list, ctx: PostprocessContext, metadata_path: Union[str, Path], output_metadata_path:Optional[str]=None) -> tuple:
    """Run steps in order. Metadata is only rescanned when a RescanStep
    has actually run since the last accurate metadata was available --
    consecutive RescanSteps share a single rescan afterward rather than
    one each."""
    metadata_df = pd.read_parquet(metadata_path)
    dirty = False

    for step in steps:
        if isinstance(step, RescanStep):
            print(f"--- {step.name} (volume) ---")
            ctx.io_func = step.apply_to_volume(ctx)
            dirty = True
        elif isinstance(step, TargetedStep):
            if dirty:
                metadata_df = rescan_metadata(ctx)
                dirty = False
            print(f"--- {step.name} (metadata-targeted) ---")
            metadata_df = step.apply(ctx, metadata_df)
        else:
            raise TypeError(f"step {step!r} is neither a RescanStep nor a TargetedStep")

    if dirty:
        metadata_df = rescan_metadata(ctx)

    metadata_path = Path(output_metadata_path or metadata_path)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_df.to_parquet(tmp, index=False)
    tmp.replace(metadata_path)

    return ctx.io_func, metadata_df


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

STEP_REGISTRY = {
    "expand_labels": ExpandLabelsStep,
    "size_filter": SizeFilterStep,
}


def build_steps(step_configs: list) -> list:
    steps = []
    for cfg in step_configs:
        cfg = dict(cfg)
        step_type = cfg.pop("type")
        if step_type not in STEP_REGISTRY:
            raise NotImplementedError(
                f"unknown postprocessing step type '{step_type}'; "
                f"registered types: {sorted(STEP_REGISTRY)}"
            )
        steps.append(STEP_REGISTRY[step_type](**cfg))
    return steps


def parse_postprocess_cfg(pp_cfg: dict) -> tuple:
    """
    Loads the original pipeline run configuration and builds the postprocessing
    context directly from it to guarantee analytical consistency.
    """
    if "pipeline_config" not in pp_cfg:
        raise KeyError(
            "Post-processing config must include 'pipeline_config' pointing to the original run's YAML.")

    # 1. Load and parse the original pipeline configuration
    run_cfg_path = Path(pp_cfg["pipeline_config"])
    with open(run_cfg_path, 'r') as f:
        run_cfg = yaml.safe_load(f)

    # Use the pipeline's exact logic to resolve paths, defaults, and data IO
    pipeline_params = parse_cfg(run_cfg)
    project_dir = Path(run_cfg["project_dir"])

    # 2. Build context strictly from the pipeline's validated parameters
    # The pipeline's output_data (instance_vol) automatically becomes the post-processor's source io_func.
    context = PostprocessContext(
        io_func=pipeline_params.output_data,
        chunk_size=pipeline_params.chunk_size,
        stack_dim=pipeline_params.stack_dim,
        scratch_dir= pp_cfg["scratch_dir"],
        parallel_backend=pipeline_params.parallel_backend,
        n_jobs=pp_cfg.get("n_jobs", run_cfg.get("n_jobs", -1)),
        verbose=pipeline_params.verbose
    )

    # 3. Resolve post-processing specific variables
    metadata_path = pipeline_params.metadata_path

    out_meta = pp_cfg.get("output_metadata_path")
    output_metadata_path = resolve_path(out_meta, default_path=metadata_path,
                                        base_dir=project_dir) if out_meta else None

    steps = pp_cfg.get("steps", [])

    return context, metadata_path, output_metadata_path, steps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to the postprocessing configuration YAML.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(
        f"Running postprocessing utilizing pipeline configuration: {cfg.get('pipeline_config')}")

    context, metadata_path, output_metadata_path, step_configs = parse_postprocess_cfg(
        cfg)
    postprocess_steps = build_steps(step_configs)

    final_io, final_metadata = run_postprocessing(
        postprocess_steps, context, metadata_path, output_metadata_path
    )

    print(f"Postprocessing completed! {len(final_metadata)} objects remain. "
          f"Final volume: {final_io.zarr_path if hasattr(final_io, 'zarr_path') else final_io}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    source_io = Zarr2DataIO(cfg["source_instance_vol"])
    context = PostprocessContext(
        io_func=source_io,
        chunk_size=np.array(cfg["chunk_size"]),
        stack_dim=np.array(cfg.get("stack_dim", [[0, 0, 0], source_io.shape])),
        scratch_dir=mk_dir(cfg["scratch_dir"]),
        parallel_backend=cfg.get("parallel_backend", "loky"),
        n_jobs=cfg.get("n_jobs", -1),
        verbose=cfg.get("verbose", False),
    )
    postprocess_steps = build_steps(cfg["steps"])

    final_io, final_metadata = run_postprocessing(
        postprocess_steps, context, cfg["metadata_path"],
        cfg.get("output_metadata_path",None),
    )
    print(f"Postprocessing completed! {len(final_metadata)} objects remain. "
          f"Final volume: {final_io.zarr_path if isinstance(final_io, Zarr2DataIO) else final_io}")
