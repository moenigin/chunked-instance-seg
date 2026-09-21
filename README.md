# Chunked Instance Segmentation

chunked-instance-seg is a high-performance pipeline for converting large 3D
semantic segmentations into distinct instance segmentations. Designed
specifically for datasets that are too large to fit into system memory, such as
large Zarr volumes, it processes your data in parallel chunks to deliver fast,
scalable results. 
### Key Features:
* Out-of-Core Processing: Safely handle massive
volumes by processing and saving data in manageable chunks. Fail-Safe
* Highly Parallelized: Optimized for multi-core
environments to process massive datasets quickly across your available hardware.
* Resumability: If a long-running job is interrupted by a crash or timeout, the
pipeline tracks progress and resumes exactly where it left off, ensuring you
never lose computed work. 
* Built-in Postprocessing: Includes scalable tools for targeted size filtering and
label expansion to clean up and refine your final instance masks

## Installation

### Using Pixi (Recommended)

Clone the repository and install dependencies with `pixi`:

```bash
git clone https://github.com/your-org/chunked-instance-seg.git
cd chunked-instance-seg
pixi install
```

### Using Pip

You can also install the repository directly as a Python package:

```bash
pip install git+https://github.com/your-org/chunked-instance-seg.git
```

---

## Configuration Setup

The pipeline and postprocessing workflows are configured via YAML files. Copies
of template configurations are stored in the `templates/` directory.

## Running the Pipeline

Execute the 3-pass instance segmentation pipeline by providing a configuration
file:

```bash
pixi run python -m source.instance_segmentation_pipeline --config path/to/my_pipeline_config.yaml
```

### Pipeline Summary

1. **Pass 1 (Per-Chunk Segmentation)**: Runs local connected-component labeling
   and assigns deterministic, lock-free global IDs derived from bit-packed chunk
   coordinates.
2. **Pass 2 (Boundary Analysis)**: Vectorized scanning across neighboring chunk
   boundaries to compute unified ID mappings.
3. **Pass 3 (Relabeling & Metadata)**: Relabels boundary-spanning objects,
   consolidates metadata into a single Parquet file, and outputs summary
   reports.

### QA Reports

When `generate_report: true` is set, quality assurance reports are automatically
placed in `<project_dir>/report/`:

* **`report.json`**: Contains total object count, non-empty chunk metrics, and
  summary voxel statistics.
* **`nvoxels_histogram.png`**: Log-scale distribution plot of object sizes
  across the volume.

---

## Running Postprocessing on Diverse Data Structures

The postprocessing module separates non-local voxel expansion (which requires
halo-padded reads) from targeted metadata edits (like object removal).

To apply postprocessing on differently organized data or independent runs:

1. Copy `templates/postprocess_config.yaml` to your working directory.
2. Update `pipeline_config` to reference the configuration of the dataset you
   wish to process.
3. Specify step definitions explicitly to build custom workflows.

Run the postprocessing step:

```bash
pixi run python -m source.postprocess_instances --config path/to/my_postprocess_config.yaml
```

### Step Types

| Step                | Parameters                               | Description                                                                                                                                                       |
|:--------------------|:-----------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`expand_labels`** | `distance` (int), `halo` (int)           | Grows instances into background voxels using non-overlapping core writes with padded halo reads to prevent boundary races. Triggers an automatic metadata rescan. |
| **`size_filter`**   | `min_nvoxels` (int), `max_nvoxels` (int) | Drops objects exceeding size thresholds directly in metadata and zeroes corresponding volume voxels without requiring a full volume rescan.                       |

---

## Resumability & Scratch Lifecycle

All intermediate states (Pass progress markers, boundary pair tables, step
outputs) are stored inside `<project_dir>/scratch/`.

* **Interrupted Runs**: Re-running any pipeline or postprocessing command
  automatically skips previously completed chunks.
* **Cleanup**: Pass `delete_scratch: true` in your YAML or call
  `pipeline.cleanup_scratch()` to purge temporary progress markers once final
  outputs and Parquet metadata are verified.