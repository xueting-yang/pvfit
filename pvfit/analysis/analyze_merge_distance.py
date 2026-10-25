"""Signed separations of true-vertex pairs participating in track-based merges."""

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analyze_merge_rate_track import _true_vertices_with_recall_above
from .binned_rate import iter_events


def collect_merge_pairs(hdf5_path, recall_threshold):
    """Count each unordered true pair once per event, even if several recs merge it.

    Use precisely the existing merge-rate recall test. For a merge of k true
    vertices, include all k*(k-1)/2 pairs. Sign is higher true-row index minus
    lower index, not ordering by z or t. Event ids are file-local HDF5 groups.
    """
    rows = []
    for event in iter_events(hdf5_path):
        pairs = set()
        for tracks in event.rec_tracks:
            if tracks:
                matched = _true_vertices_with_recall_above(
                    tracks, event.true_tracks, recall_threshold
                )
                pairs.update(combinations(sorted(matched), 2))
        for first, second in sorted(pairs):
            rows.append(
                {
                    "event_group": event.event_id,
                    "true_index_1": first,
                    "true_index_2": second,
                    "delta_z_mm": float(event.true_z[second] - event.true_z[first]),
                    "delta_t_ns": float(event.true_t[second] - event.true_t[first]),
                }
            )
    return rows


def analyze_merge_distance(hdf5_path, output_dir, recall_threshold, method_name):
    """Write raw pairs and the z, joint (z,t), and horizontal t count projections.

    Plot windows follow the reference figure: +/-3 mm and +/-0.1 ns. CSV keeps
    every pair, including tails/nonfinite coordinates; JSON records exclusions.
    No track-multiplicity plotting cut is imposed on this pair diagnostic.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"merge_distance_{method_name.lower()}_track_based"

    def output_file(extension):
        # Method names may contain dots; with_suffix would truncate the name.
        return output / f"{stem}.{extension}"

    rows = collect_merge_pairs(hdf5_path, recall_threshold)
    fields = ["event_group", "true_index_1", "true_index_2", "delta_z_mm", "delta_t_ns"]
    with output_file("csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    z = np.array([row["delta_z_mm"] for row in rows])
    t = np.array([row["delta_t_ns"] for row in rows])
    finite = np.isfinite(z) & np.isfinite(t)
    z, t = z[finite], t[finite]
    z_edges = np.linspace(-3, 3, 41)
    t_edges = np.linspace(-0.1, 0.1, 41)
    in_z = (z >= z_edges[0]) & (z <= z_edges[-1])
    in_t = (t >= t_edges[0]) & (t <= t_edges[-1])
    summary = {
        "input_hdf5": str(Path(hdf5_path).resolve()),
        "matching": "track",
        "recall_threshold": recall_threshold,
        "pair_count": len(rows),
        "finite_pair_count": int(finite.sum()),
        "nonfinite_pairs": int((~finite).sum()),
        "shown_z_pairs": int(in_z.sum()),
        "shown_t_pairs": int(in_t.sum()),
        "shown_joint_pairs": int((in_z & in_t).sum()),
        "z_edges_mm": z_edges.tolist(),
        "t_edges_ns": t_edges.tolist(),
        "pair_counting": "unique unordered true pairs per event; all pairs in multi-merges",
        "sign": "higher true-row index minus lower true-row index",
        "event_group": "file-local HDF5 group; source identity remains in input HDF5",
    }
    output_file("json").write_text(json.dumps(summary, indent=2) + "\n")

    with plt.style.context("default"):
        fig = plt.figure(figsize=(11, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 4)
        ax_z = fig.add_subplot(grid[0, :2])
        ax_joint = fig.add_subplot(grid[0, 2:])
        ax_t = fig.add_subplot(grid[1, 1:3])
        ax_z.hist(z, bins=z_edges, color="steelblue", edgecolor="black", alpha=0.7)
        joint = ax_joint.hist2d(z, t, bins=(z_edges, t_edges), cmap="viridis")
        fig.colorbar(joint[3], ax=ax_joint, label="Counts")
        ax_t.hist(
            t,
            bins=t_edges,
            orientation="horizontal",
            color="coral",
            edgecolor="black",
            alpha=0.7,
        )
        ax_z.set(
            xlabel=r"$\Delta z$ (mm)",
            ylabel="Counts",
            xlim=(-3, 3),
            title="(a) Longitudinal separation",
        )
        ax_joint.set(
            xlabel=r"$\Delta z$ (mm)",
            ylabel=r"$\Delta t$ (ns)",
            title="(b) Joint separation distribution",
        )
        ax_t.set(
            xlabel="Counts",
            ylabel=r"$\Delta t$ (ns)",
            ylim=(-0.1, 0.1),
            title="(c) Temporal separation",
        )
        for ax in (ax_z, ax_joint, ax_t):
            ax.grid(alpha=0.2)
            if not len(z):
                ax.text(
                    0.5,
                    0.5,
                    "No finite merged pairs",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        fig.suptitle(
            f"Merged true-vertex pairs — {method_name}, track-based "
            f"(recall ≥ {recall_threshold:g})\n"
            f"{len(z)} finite pairs; counts, not probability density"
        )
        fig.savefig(output_file("pdf"))
        fig.savefig(output_file("png"), dpi=160)
        plt.close(fig)
    print(f"Merge distance: {len(rows)} pairs; outputs in {output / stem}.*")
    return summary
