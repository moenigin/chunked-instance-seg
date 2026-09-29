"""Chained, resumable post-processing over an already-segmented volume
(the output of instance_segmentation_pipeline.py): label expansion,
size-based filtering, and a metadata rescan tying them together.

Two kinds of step, because they need different correctness handling:
- RescanStep: mutates voxels in a way that can change any object's
  extent unpredictably (e.g. expand_labels can grow an object into a
  chunk it never touched before). Always writes to a NEW destination,
  never in place, and the runner triggers a full metadata rescan after
  one or more of these before the next TargetedStep (or the end).
- TargetedStep: the voxel changes are already exactly known from the
  CURRENT metadata (e.g. "zero out object ids X, Y, Z"), so it's safe
  to mutate in place and update the metadata table directly -- no
  rescan needed.

Steps are chained by the runner (run_postprocessing), each reading from
the previous step's output DataIO and (unless allow_overwrite) writing
to its own private destination, so a later step's halo reads can never
race an earlier step's in-flight writes to the same store. Progress
markers and intermediate per-step stores live under scratch_dir and are
cleaned up on success unless cleanup_scratch=False.
"""
import numpy as np
import pandas as pd
import shutil
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
from source.metadata_handler import \
    _COLUMNS  # same schema/column names as the main pipeline -- single source of truth
from source.utils import generate_chunks, mark_completed, \
    filter_remaining_chunks, mk_dir, resolve_path


class OutputVolumeNotWritten(RuntimeError):
    """Raised when the pipeline finishes but no volume exists at
    ctx.output_vol_path -- e.g. every step turned out to be a no-op
    (a SizeFilterStep with nothing to remove and no step after it).
    actual_path is where the data really is, so a caller can redirect
    a downstream step there instead of treating this as pure failure."""

    def __init__(self, expected_path: Path, actual_path: Path):
        self.expected_path = Path(expected_path)
        self.actual_path = Path(actual_path)
        super().__init__(
            f"no volume written to {self.expected_path} -- every step was a "
            f"no-op; data is unchanged at {self.actual_path}"
        )


def get_slice(origin: np.ndarray, far_corner: np.ndarray) -> tuple:
    """(origin, far_corner) -> a tuple of slices usable on a DataIO."""
    return tuple(slice(int(o), int(f)) for o, f in zip(origin, far_corner))


def _new_zarr_like(source_io: DataIO, path: Union[str, Path],
                   chunk_size: np.ndarray) -> Zarr2DataIO:
    """Create a fresh Zarr array with the same shape/dtype as
    source_io, chunked the same way the pipeline chunks it. Raises if
    `path` already exists -- RescanSteps write here, never back into
    the source they're reading from, so an existing path almost always
    means a stale destination from a previous attempt."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists -- refusing to overwrite a RescanStep "
            f"destination. Delete it first if you intend to redo this step "
            f"from scratch, or point dest_path somewhere new."
        )
    return Zarr2DataIO(path, chunk_size=tuple(int(c) for c in chunk_size),
                       array_shape=source_io.shape)


@dataclass
class PostprocessParams:
    """Shared context threaded through every step of one postprocessing
    run. instance_vol/chunk_size/stack_dim describe the volume being
    processed (instance_vol is reassigned to each step's output as the
    chain progresses); output_vol_path is the single final destination
    once every step has run; allow_overwrite mutates the original
    input in place instead (only valid for a single-step run -- see
    run_postprocessing)."""
    instance_vol: DataIO
    chunk_size: np.ndarray
    stack_dim: np.ndarray
    scratch_dir: Path
    output_vol_path: Optional[Path] = None
    allow_overwrite: bool = False
    parallel_backend: str = 'loky'
    n_jobs: int = -1
    verbose: bool = False
    cleanup_scratch: bool = True  # remove intermediate step stores + progress dirs on success
    require_output_at_path: bool = True  # raise OutputVolumeNotWritten if output_vol_path was never created


class RescanStep(ABC):
    """A step that mutates voxels in a way that can change any object's
    extent unpredictably. The runner always triggers a full metadata
    rescan after one or more of these run consecutively, before the
    next TargetedStep (or the end of the pipeline)."""

    name: str = "rescan_step"

    @abstractmethod
    def apply_to_volume(self, ctx: PostprocessParams) -> DataIO:
        """Read from ctx.instance_vol, write the result to a NEW
        destination (never back into ctx.instance_vol -- see module
        docstring), and return the DataIO for that destination. The
        runner adopts the return value as the new ctx.instance_vol for
        any subsequent step."""


class TargetedStep(ABC):
    """A step whose voxel changes are already exactly known from the
    CURRENT metadata (e.g. "remove object ids X, Y, Z") -- no
    volume-wide rescan needed. Safe to mutate ctx.instance_vol in
    place, since it never depends on a neighbor's un-mutated state."""

    name: str = "targeted_step"
    dest_path: Optional[Path] = None
    wrote_volume: bool = False  # apply() sets this; tells the runner whether ctx.instance_vol should move

    @abstractmethod
    def apply(self, ctx: PostprocessParams,
              metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the edit to ctx.instance_vol and return the updated
        metadata table (e.g. with removed objects' rows dropped)."""


# ----------------------------------------------------------------------
# RescanStep: expand_labels on haloed, non-overlapping-write chunks
# ----------------------------------------------------------------------
class ExpandLabelsStep(RescanStep):
    """Grow every label outward by `distance` voxels (skimage's
    expand_labels), processed chunk-by-chunk with a halo so a growth
    near a chunk boundary is computed correctly. halo must be >=
    distance for correctness; it defaults to distance, and a little
    extra margin is cheap insurance, not required."""

    name = "expand_labels"

    def __init__(self, distance: int, halo: Optional[int] = None,
                 dest_path: Optional[Union[str, Path]] = None):
        self.distance = distance
        self.halo = halo if halo is not None else distance
        self.dest_path = Path(dest_path) if dest_path is not None else None

    def apply_to_volume(self, ctx: PostprocessParams) -> DataIO:
        """Run expand_labels over every not-yet-completed chunk (each
        read with a halo from the immutable source) in parallel,
        writing into a fresh destination array."""
        progress_dir = mk_dir(
            ctx.scratch_dir / f"progress_postprocess_{self.name}")
        project_dir = ctx.scratch_dir.parent
        self.dest_path = resolve_path(self.dest_path,
                                      default_path=project_dir / f"instances_metadata_{self.name}.parquet",
                                      base_dir=ctx.scratch_dir.parent)

        if self.dest_path.exists():
            dest_io = Zarr2DataIO(
                self.dest_path,
                chunk_size=tuple(int(c) for c in ctx.chunk_size),
            )
        else:
            dest_io = _new_zarr_like(ctx.instance_vol, self.dest_path,
                                     ctx.chunk_size)

        chunks = generate_chunks(ctx.stack_dim, ctx.chunk_size)
        remaining = filter_remaining_chunks(chunks, progress_dir)
        print(
            f"{self.name}: {len(chunks) - len(remaining)}/{len(chunks)} chunks already done")
        if remaining:
            Parallel(n_jobs=ctx.n_jobs, backend=ctx.parallel_backend)(
                delayed(self._process_chunk)(chunk[0], chunk[1], ctx, dest_io,
                                             progress_dir)
                for chunk in tqdm(remaining, desc=f"{self.name} (haloed)")
            )
        return dest_io

    def _process_chunk(self, origin, far_corner, ctx: PostprocessParams,
                       dest_io: DataIO, progress_dir: Path):
        """Expand labels within one haloed read, then write back only
        the chunk's own (non-halo) core region."""
        halo = np.full(3, self.halo, dtype=np.int64)
        padded_origin = np.maximum(origin - halo, ctx.stack_dim[0])
        padded_far = np.minimum(far_corner + halo, ctx.stack_dim[1])

        # Read from the IMMUTABLE source -- never dest_io -- so this
        # chunk's neighbors can't have been mutated yet no matter what
        # order workers finish in.
        padded = ctx.instance_vol.get_data(get_slice(padded_origin, padded_far))
        expanded = expand_labels(padded, distance=self.distance)

        core_start = origin - padded_origin
        core_end = core_start + (far_corner - origin)
        core = expanded[core_start[0]:core_end[0], core_start[1]:core_end[1],
               core_start[2]:core_end[2]]
        dest_io.write_data(core, get_slice(origin, far_corner))

        mark_completed(chunk_coords=np.array([origin, far_corner]),
                       progress_dir=progress_dir)


# ----------------------------------------------------------------------
# TargetedStep: remove objects outside a voxel-count range
# ----------------------------------------------------------------------
def _chunks_overlapping_bbox(bbox, chunk_size: np.ndarray,
                             stack_dim: np.ndarray) -> list:
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
    """Zero out every object whose voxel count falls outside
    [min_nvoxels, max_nvoxels] (either bound may be omitted, not both)
    and drop its row from the metadata table."""

    name = "size_filter"

    def __init__(self, dest_path: Optional[Union[str, Path]] = None,
                 min_nvoxels: Optional[int] = None,
                 max_nvoxels: Optional[int] = None):
        self.dest_path = dest_path
        if min_nvoxels is None and max_nvoxels is None:
            raise ValueError(
                "SizeFilterStep needs at least one of min_nvoxels/max_nvoxels")
        self.min_nvoxels = min_nvoxels
        self.max_nvoxels = max_nvoxels

    def apply(self, ctx: PostprocessParams,
              metadata_df: pd.DataFrame) -> pd.DataFrame:
        keep = pd.Series(True, index=metadata_df.index)
        if self.min_nvoxels is not None:
            keep &= metadata_df["nvoxels"] >= self.min_nvoxels
        if self.max_nvoxels is not None:
            keep &= metadata_df["nvoxels"] <= self.max_nvoxels

        to_remove = metadata_df.loc[~keep]
        print(
            f"{self.name}: removing {len(to_remove)}/{len(metadata_df)} objects")

        dest = Path(self.dest_path)
        in_place = dest == Path(ctx.instance_vol.zarr_path)

        if len(to_remove) == 0:
            if in_place:
                print(
                    f"{self.name}: nothing to remove -- volume already correct, no write needed")
            else:
                print(
                    f"{self.name}: nothing to remove -- skipping copy; volume stays at "
                    f"{ctx.instance_vol.zarr_path} instead of {dest}")
            self.wrote_volume = False
            return metadata_df.loc[keep].reset_index(drop=True)

        self._remove_from_volume(ctx, to_remove, dest)
        self.wrote_volume = True
        return metadata_df.loc[keep].reset_index(drop=True)

    @staticmethod
    def _remove_from_volume(ctx: PostprocessParams, rows: pd.DataFrame,
                            dest_path: Path):
        """Zero every voxel belonging to an object in `rows`, chunk by
        chunk in parallel. In place (dest_path == the source), only
        chunks that actually contain a to-be-removed object are
        touched; writing to a new destination instead copies every
        chunk across, edited or not."""
        in_place = dest_path == Path(ctx.instance_vol.zarr_path)

        if dest_path.exists():
            output_volume = Zarr2DataIO(
                dest_path, chunk_size=tuple(int(c) for c in ctx.chunk_size))
        else:
            output_volume = _new_zarr_like(ctx.instance_vol, dest_path,
                                           ctx.chunk_size)

        chunk_to_ids = defaultdict(list)
        for _, row in rows.iterrows():
            bbox = (row.bbox_z_min, row.bbox_y_min, row.bbox_x_min,
                    row.bbox_z_max, row.bbox_y_max, row.bbox_x_max)
            for origin in _chunks_overlapping_bbox(bbox, ctx.chunk_size,
                                                   ctx.stack_dim):
                chunk_to_ids[tuple(origin.tolist())].append(int(row.object_id))

        # Progress dir keyed on dest_path's own name -- each step
        # instance already writes to a distinct location (per the
        # runner's dest resolution), so this can't collide between
        # two SizeFilterStep instances in the same pipeline run.
        progress_dir = mk_dir(
            ctx.scratch_dir / f"progress_postprocess_{SizeFilterStep.name}_{dest_path.stem}")
        all_chunks = generate_chunks(ctx.stack_dim, ctx.chunk_size)
        remaining = filter_remaining_chunks(all_chunks, progress_dir)
        print(
            f"{SizeFilterStep.name}: {len(all_chunks) - len(remaining)}/{len(all_chunks)} chunks already done")

        def _process(origin, far_corner):
            ids = chunk_to_ids.get(tuple(origin.tolist()), [])
            sl = get_slice(origin, far_corner)
            if ids:
                data = ctx.instance_vol.get_data(sl)
                mask = np.isin(data, ids)
                if np.any(mask):
                    data[mask] = 0
                if not in_place or np.any(mask):
                    output_volume.write_data(data, sl)
            elif not in_place:
                output_volume.write_data(ctx.instance_vol.get_data(sl), sl)
            mark_completed(chunk_coords=np.array([origin, far_corner]),
                           progress_dir=progress_dir)

        if remaining:
            Parallel(n_jobs=ctx.n_jobs, backend=ctx.parallel_backend)(
                delayed(_process)(o, f) for o, f in
                tqdm(remaining, desc=f"{SizeFilterStep.name}: writing chunks"))
        else:
            print(f"{SizeFilterStep.name}: all chunks already written")


# ----------------------------------------------------------------------
# Full metadata rescan -- ground truth from the volume, for use after
# any RescanStep
# ----------------------------------------------------------------------
def _rescan_chunk(origin, far_corner, source_io: DataIO) -> pd.DataFrame:
    """Per-chunk partial bbox/voxel-count summary for every object
    touching this chunk, via a vectorised groupby (not a per-id
    np.where loop) -- the building block for rescan_metadata's
    two-level aggregate."""
    data = source_io.get_data(get_slice(origin, far_corner))
    nz = np.nonzero(data)
    if len(nz[0]) == 0:
        return pd.DataFrame(
            columns=["object_id", "bbox_z_min", "bbox_y_min", "bbox_x_min",
                     "bbox_z_max", "bbox_y_max", "bbox_x_max", "nvoxels"])
    ids_flat = np.asarray(data[nz]).astype(np.int64)
    coords = pd.DataFrame({
        "object_id": ids_flat,
        "z": nz[0] + origin[0], "y": nz[1] + origin[1], "x": nz[2] + origin[2],
    })
    agg = coords.groupby("object_id").agg(
        bbox_z_min=("z", "min"), bbox_y_min=("y", "min"),
        bbox_x_min=("x", "min"),
        bbox_z_max=("z", "max"), bbox_y_max=("y", "max"),
        bbox_x_max=("x", "max"),
        nvoxels=("z", "size"),
    ).reset_index()
    return agg


def rescan_metadata(ctx: PostprocessParams) -> pd.DataFrame:
    """Ground-truth metadata recompute, scanning the current volume
    directly -- the only correct way to know an object's extent after
    an operation (like expand_labels) that can grow it into chunks it
    never originally touched. Two-level aggregate: per-chunk partial
    contributions computed in parallel (each chunk only holds its own
    small summary in memory, not every voxel coordinate in the volume
    at once), then a cheap second groupby over those summaries.
    chunk_id is left unset ("") on the result, since it isn't
    meaningful for an object now spanning multiple chunks -- the same
    documented limitation as the main pipeline's merge rows."""
    start = time.time()
    chunks = generate_chunks(ctx.stack_dim, ctx.chunk_size)
    partials = Parallel(n_jobs=ctx.n_jobs, backend=ctx.parallel_backend)(
        delayed(_rescan_chunk)(chunk[0], chunk[1], ctx.instance_vol)
        for chunk in tqdm(chunks, desc="rescanning metadata")
    )
    partials = [p for p in partials if len(p)]
    if not partials:
        result = pd.DataFrame(columns=["object_id", "chunk_id",
                                       *[c for c in _COLUMNS if
                                         c not in ("object_id", "chunk_id")]])
    else:
        full = pd.concat(partials, ignore_index=True)
        result = full.groupby("object_id").agg(
            bbox_z_min=("bbox_z_min", "min"), bbox_y_min=("bbox_y_min", "min"),
            bbox_x_min=("bbox_x_min", "min"),
            bbox_z_max=("bbox_z_max", "max"), bbox_y_max=("bbox_y_max", "max"),
            bbox_x_max=("bbox_x_max", "max"),
            nvoxels=("nvoxels", "sum"),
        ).reset_index()
        result["chunk_id"] = ""
    if ctx.verbose:
        print(
            f"rescan_metadata: {len(result)} objects in {time.time() - start:.1f}s",
            flush=True)
    return result[_COLUMNS]


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run_postprocessing(steps: list, ctx: PostprocessParams,
                       metadata_path: Union[str, Path],
                       output_metadata_path: Optional[str] = None) -> tuple:
    """Run `steps` in order against `ctx`, starting from the metadata at
    metadata_path. Rescans metadata before any TargetedStep that
    follows one or more RescanSteps, and once more at the end if the
    volume is still "dirty" when the loop finishes. Writes the final
    metadata to output_metadata_path (or metadata_path if omitted),
    cleans up scratch on success (unless disabled), and returns
    (final DataIO, final metadata DataFrame). allow_overwrite is only
    accepted for a single-step run, since with several chained steps
    an in-place step's write could race a later step's halo read of
    the same store."""
    if ctx.allow_overwrite and len(steps) > 1:
        raise ValueError(
            "allow_overwrite=True is only valid for a single-step run -- "
            "with several chained steps, each intermediate needs its own "
            "store or a later step's halo reads can race an earlier "
            "step's in-flight writes to the same one."
        )

    metadata_df = pd.read_parquet(metadata_path)
    dirty = False

    for idx, step in enumerate(steps):
        is_last = idx == len(steps) - 1

        if isinstance(step, TargetedStep) and dirty:
            metadata_df = rescan_metadata(ctx)
            dirty = False

        step.dest_path = _resolve_step_dest(step, ctx, idx, is_last)

        if isinstance(step, RescanStep):
            print(f"--- {step.name} (volume) ---")
            ctx.instance_vol = step.apply_to_volume(ctx)
            dirty = True

        elif isinstance(step, TargetedStep):
            print(f"--- {step.name} (metadata-targeted) ---")
            metadata_df = step.apply(ctx, metadata_df)
            if step.wrote_volume and Path(step.dest_path) != Path(
                    ctx.instance_vol.zarr_path):
                ctx.instance_vol = Zarr2DataIO(Path(step.dest_path),
                                               chunk_size=ctx.chunk_size)
        else:
            raise TypeError(
                f"step {step!r} is neither a RescanStep nor a TargetedStep")

    if dirty:
        metadata_df = rescan_metadata(ctx)

    metadata_path = Path(output_metadata_path or metadata_path)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_df.to_parquet(tmp, index=False)
    tmp.replace(metadata_path)

    output_exists = Path(ctx.output_vol_path) == Path(
        ctx.instance_vol.zarr_path) or \
                    Path(ctx.output_vol_path).exists()

    if ctx.cleanup_scratch:
        removed = cleanup_intermediate(ctx, ctx.output_vol_path)
        if ctx.verbose and removed:
            print(f"Removed {len(removed)} intermediate scratch artifacts",
                  flush=True)

    if not output_exists and ctx.require_output_at_path:
        raise OutputVolumeNotWritten(ctx.output_vol_path,
                                     ctx.instance_vol.zarr_path)

    return ctx.instance_vol, metadata_df


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
STEP_REGISTRY = {
    "expand_labels": ExpandLabelsStep,
    "size_filter": SizeFilterStep,
}


def _default_step_dest(ctx: PostprocessParams, idx: int,
                       step_name: str) -> Path:
    """Private scratch destination for one step -- distinct from both
    its own source and any other step's output, so halo reads can
    never race against in-flight writes."""
    return ctx.scratch_dir / f"step{idx:02d}_{step_name}.zarr"


def _resolve_step_dest(step, ctx: PostprocessParams, idx: int,
                       is_last: bool) -> Path:
    """Where a step should write: its own explicit dest_path if set,
    else the original volume in place if ctx.allow_overwrite, else
    ctx.output_vol_path if it's the last step, else a private scratch
    path."""
    if step.dest_path is not None:
        return resolve_path(step.dest_path,
                            default_path=ctx.instance_vol.zarr_path,
                            base_dir=ctx.scratch_dir.parent)
    if ctx.allow_overwrite:
        return Path(ctx.instance_vol.zarr_path)
    if is_last and ctx.output_vol_path is not None:
        return Path(ctx.output_vol_path)
    return _default_step_dest(ctx, idx, step.name)


def build_steps(step_configs: list) -> list:
    """Instantiate each step from its config dict via STEP_REGISTRY,
    keyed by its 'type' field."""
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


def _default_output_vol_path(instance_vol_path: Path, steps: list,
                             project_dir: Path) -> Path:
    """Default final-output path when none is configured: the input
    volume's name plus every step's type, so it's traceable from the
    filename alone."""
    step_names = "_".join(s["type"] for s in steps)
    return project_dir / f"{instance_vol_path.stem}_{step_names}.zarr"


def cleanup_intermediate(ctx: PostprocessParams, final_vol_path: Path) -> list:
    """Remove per-step scratch artifacts once the run has finished and
    the final volume + metadata are confirmed written: progress-marker
    directories for every step, and any private intermediate zarr store
    created by _default_step_dest. Never touches anything outside
    ctx.scratch_dir, so the original input volume and the promoted
    final_vol_path (which normally live in project_dir, not scratch)
    are never candidates for removal even if matched by accident."""
    final_vol_path = Path(final_vol_path).resolve()
    removed = []
    for pattern in ("progress_postprocess_*", "step*.zarr"):
        for p in ctx.scratch_dir.glob(pattern):
            if p.resolve() == final_vol_path:
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(p)
    return removed


def parse_postprocess_cfg(pp_cfg: dict) -> tuple:
    """Build (PostprocessParams, metadata_path, output_metadata_path,
    step_configs) from a postprocessing config dict, which must point
    back at the original pipeline run's config via 'pipeline_config'
    (that's where the volume, chunk_size, stack_dim etc. come from)."""
    if "pipeline_config" not in pp_cfg:
        raise KeyError(
            "Post-processing config must include 'pipeline_config' pointing to the original run's YAML.")

    run_cfg_path = Path(pp_cfg["pipeline_config"])
    with open(run_cfg_path, 'r') as f:
        run_cfg = yaml.safe_load(f)

    instance_params = parse_cfg(run_cfg)
    project_dir = Path(run_cfg["project_dir"])

    stack_dim = instance_params.stack_dim
    if stack_dim is None:
        stack_dim = np.array([[0, 0, 0], instance_params.output_data.shape],
                             dtype=np.int64)

    steps = pp_cfg.get("steps", [])
    allow_overwrite = pp_cfg.get("allow_overwrite", False)
    instance_vol_path = Path(instance_params.output_data.zarr_path)

    out_vol = pp_cfg.get("output_vol_path")
    if out_vol is not None:
        output_vol_path = resolve_path(out_vol, default_path=None,
                                       base_dir=project_dir)
    elif allow_overwrite:
        output_vol_path = instance_vol_path
    else:
        output_vol_path = _default_output_vol_path(instance_vol_path, steps,
                                                   project_dir)

    context = PostprocessParams(
        instance_vol=instance_params.output_data,
        chunk_size=instance_params.chunk_size,
        stack_dim=stack_dim,
        scratch_dir=instance_params.scratch_dir,
        output_vol_path=output_vol_path,
        allow_overwrite=allow_overwrite,
        cleanup_scratch=pp_cfg.get("cleanup_scratch", True),
        parallel_backend=instance_params.parallel_backend,
        n_jobs=pp_cfg.get("n_jobs", run_cfg.get("n_jobs", -1)),
        verbose=instance_params.verbose,
    )

    metadata_path = instance_params.metadata_path
    out_meta = pp_cfg.get("output_metadata_path")
    if out_meta:
        output_metadata_path = resolve_path(out_meta, default_path=None,
                                            base_dir=project_dir)
    elif output_vol_path == instance_vol_path:
        # True in-place overwrite: keep updating the original metadata
        # file regardless of what naming convention it happened to use.
        output_metadata_path = metadata_path
    else:
        output_metadata_path = output_vol_path.parent / f"{output_vol_path.stem}_metadata.parquet"

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