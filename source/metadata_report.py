import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def compute_report_stats(metadata_df: pd.DataFrame, total_chunks: int) -> dict:
    n_objects = len(metadata_df)
    n_nonempty_chunks = metadata_df["chunk_id"].nunique()
    return {
        "n_objects": n_objects,
        "n_total_chunks": total_chunks,
        "n_nonempty_chunks": n_nonempty_chunks,
        "n_empty_chunks": total_chunks - n_nonempty_chunks,
        "avg_objects_per_chunk": n_objects / total_chunks if total_chunks else float("nan"),
        "nvoxels_summary": metadata_df["nvoxels"].describe().to_dict(),
    }


def find_empty_chunk_ids(metadata_df: pd.DataFrame, all_chunk_ids: list) -> list:
    present = set(metadata_df["chunk_id"].unique())
    return [c for c in all_chunk_ids if c not in present]


def plot_nvoxels_histogram(metadata_df: pd.DataFrame, out_path: Union[str, Path]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = metadata_df["nvoxels"].to_numpy()
    bins = np.logspace(np.log10(max(values.min(), 1)), np.log10(values.max()), 50)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_xscale("log")  # object sizes are typically heavy-tailed; adjust if yours aren't
    ax.set_xlabel("voxels per object")
    ax.set_ylabel("count")
    ax.set_title("Object size distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_report(metadata_df, all_chunk_ids: list, out_dir: Union[str, Path],
                     list_empty_chunks: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_report_stats(metadata_df, total_chunks=len(all_chunk_ids))
    if list_empty_chunks:
        stats["empty_chunk_ids"] = find_empty_chunk_ids(metadata_df, all_chunk_ids)
    plot_nvoxels_histogram(metadata_df, out_dir / "nvoxels_histogram.png")
    with open(out_dir / "report.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[report] {stats['n_objects']} objects, "
          f"{stats['avg_objects_per_chunk']:.2f} objects/chunk "
          f"({stats['n_empty_chunks']}/{stats['n_total_chunks']} chunks empty)")
    return stats