"""Convert Allen's monitoring ROOT output into the PVFit HDF5 event layout.

Allen's ``upgrade2_tv_cheated_tracking`` sequence dumps six trees: the fitted
VELO tracks with their MC truth (``velo_kalman_filter/monitor_tree``), the
effective true vertices (``mc_pv_dumper/MCPV_effective_tree``), the sparse
seeding histogram (``pv_beamline_histo/histo_tree``), the histogram peaks
(``pv_beamline_peak/peak_tree``) and the merged, fitted vertices with the MC
keys of their member tracks (``pv_beamline_merge_vertices/merge_vtx_tree``).
This module rewrites them as one HDF5 file per the layout documented in
``docs/data-format.md`` so ``pvfit analyze`` evaluates the GPU reconstruction
with exactly the code used for PVFit's own fit.

Conventions, chosen to match PVFit's reconstruction output:

- Events are identified by ``(run_number, event_number)`` across all input
  files (``pvfit.fit.event_identity``); every tree must carry both branches.
- The ``tracks/`` table holds every monitor-tree row of the event, in dump
  order. ``selected_for_seeding`` is the histogram's z/t window only.
- ``true_vertices/`` membership is built by :func:`build_true_vertex_info` on
  the full track table, with the same four-row minimum as ``pvfit fit``.
- ``rec_vertices/track_indices`` are the rows whose ``mc_key`` appears in the
  vertex's ``track_mc_keys``; Allen records member keys, not row numbers, so a
  key shared by several rows selects all of them.
- The reconstruction parameters are read from ``allen_configuration.json``
  next to each ROOT file (the campaign writes it from the effective Allen
  sequence) and recorded as the same ``reco_<name>`` attributes ``pvfit fit``
  writes (``pvfit.reconstruction_parameters``). Without that file only the
  histogram binning, read back from the dump, is recorded; nothing is assumed.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from .allen import (
    CONFIGURATION_NAME,
    configuration_attributes,
    load_configuration,
    parameters_from_configuration,
)
from .fit.event_identity import require_run_numbers, unify_event_identity
from .fit.save_results_to_hdf5 import (
    TRACK_COLUMNS,
    TRUE_COLUMNS,
    _write,
    _write_vlen_int32,
)
from .fit.truth_matching import (
    TRUTH_MIN_TRACKS,
    build_true_vertex_info,
    read_mcpv_dataframe,
)
from .provenance import HDF5_SCHEMA_VERSION, implementation_fingerprint
from .reconstruction_parameters import write_parameters

DEFAULT_TREES = {
    "tracks": "velo_kalman_filter/monitor_tree",
    "truth": "mc_pv_dumper/MCPV_effective_tree",
    "histogram": "pv_beamline_histo/histo_tree",
    "peaks": "pv_beamline_peak/peak_tree",
    "vertices": "pv_beamline_merge_vertices/merge_vtx_tree",
}
# Allen's bin widths (mm, ns); used only when the configuration is missing and
# an axis of the dumped histogram has a single populated bin.
DEFAULT_WIDTH = {"z": 0.5, "t": 0.025}
IDENTITY = ["run_number", "event_number"]
COORDS = ("x", "y", "z", "t")
AXES = ("z", "t")


def _require_branches(tree, names, label):
    missing = sorted(set(names) - set(tree.keys()))
    if missing:
        raise ValueError(f"{label} lacks branches {missing}")


def _histogram_axes(histogram, n_bins):
    """Bin origin and width per axis, from the dump's (bin index, bin centre) pairs.

    The centre of bin ``i`` is ``low + (i + 0.5) * width``; two distinct
    populated indices fix ``width`` and ``low`` without assuming that
    neighbouring bins are populated. An axis with a single populated index
    falls back to Allen's compiled bin width.
    """
    axes = {}
    for axis, index, centre in (("z", "iz", "z_bin"), ("t", "it", "t_bin")):
        count = int(n_bins[axis])
        idx = histogram[index].to_numpy().astype(np.int64)
        pos = histogram[centre].to_numpy().astype(np.float64)
        if len(idx) == 0:
            raise ValueError("Histogram tree is empty; cannot infer the binning")
        if idx.min() < 0 or idx.max() >= count:
            raise ValueError(
                f"Histogram {index} runs from {idx.min()} to {idx.max()}, outside the "
                f"configured {count} {axis} bins"
            )
        lo, hi = int(np.argmin(idx)), int(np.argmax(idx))
        if idx[hi] > idx[lo]:
            width = float((pos[hi] - pos[lo]) / (idx[hi] - idx[lo]))
        else:
            width = DEFAULT_WIDTH[axis]
        if not np.isfinite(width) or width <= 0:
            raise ValueError(f"Cannot infer the {axis} bin width from the histogram")
        origin = pos - (idx + 0.5) * width
        low = float(np.median(origin))
        if np.abs(origin - low).max() > 1e-3 * width:
            raise ValueError(f"Histogram {centre} values are not on a regular grid")
        # Bin centres are float32 in the dump; remove their rounding noise from the edges.
        width = float(round(width, 6))
        low = float(round(low, 6))
        axes[axis] = {
            "min": low,
            "max": float(round(low + count * width, 6)),
            "n_bins": count,
            "width": width,
        }
    return axes


def _configured_axes(parameters):
    return {
        axis: {
            "min": parameters[f"{axis}_min"],
            "max": parameters[f"{axis}_max"],
            "n_bins": parameters[f"n_bins_{axis}"],
            "width": (parameters[f"{axis}_max"] - parameters[f"{axis}_min"])
            / parameters[f"n_bins_{axis}"],
        }
        for axis in AXES
    }


def _check_axes_against_dump(axes, histogram):
    """The configured binning must reproduce the bin centres Allen dumped."""
    inferred = _histogram_axes(histogram, {a: axes[a]["n_bins"] for a in AXES})
    for axis in AXES:
        width = axes[axis]["width"]
        if (
            abs(inferred[axis]["width"] - width) > 1e-5 * width
            or abs(inferred[axis]["min"] - axes[axis]["min"]) > 1e-3 * width
        ):
            raise ValueError(
                f"The configured {axis} binning ({axes[axis]['min']} + i * {width}) "
                f"does not match the dumped histogram bin centres "
                f"({inferred[axis]['min']} + i * {inferred[axis]['width']})"
            )


def _dense_histogram(rows, axes):
    H = np.zeros((axes["z"]["n_bins"], axes["t"]["n_bins"]), dtype=np.float64)
    if len(rows):
        np.add.at(
            H,
            (
                rows["iz"].to_numpy().astype(np.int64),
                rows["it"].to_numpy().astype(np.int64),
            ),
            rows["weight"].to_numpy().astype(np.float64),
        )
    return H


def _member_rows(keys, member_keys):
    """Row indices whose mc_key is one of the vertex's member keys."""
    member_keys = np.asarray(member_keys, dtype=np.int64)
    member_keys = member_keys[member_keys >= 0]
    mask = (keys >= 0) & np.isin(keys, member_keys)
    return np.flatnonzero(mask).tolist()


def _read_file(path, trees):
    """All five trees of one ROOT file as DataFrames, keyed by run/event."""
    with uproot.open(path) as file:
        tracks = file[trees["tracks"]]
        _require_branches(
            tracks,
            ["mc_key", "backward", "run_number", "event_number"],
            trees["tracks"],
        )
        tracks_df = tracks.arrays(library="pd")
        truth = file[trees["truth"]]
        _require_branches(
            truth,
            [
                "effective_pv_key",
                "track_keys",
                "n_valid_tracks",
                "run_number",
                "event_number",
            ],
            trees["truth"],
        )
        truth_df = read_mcpv_dataframe(truth)
        histogram = file[trees["histogram"]]
        _require_branches(
            histogram,
            ["iz", "it", "z_bin", "t_bin", "weight", *IDENTITY],
            trees["histogram"],
        )
        histogram_df = pd.DataFrame(
            histogram.arrays(
                ["iz", "it", "z_bin", "t_bin", "weight", *IDENTITY], library="np"
            )
        )
        peaks = file[trees["peaks"]]
        _require_branches(peaks, ["z_peak", "t_peak", *IDENTITY], trees["peaks"])
        peaks_df = pd.DataFrame(
            peaks.arrays(["z_peak", "t_peak", *IDENTITY], library="np")
        )
        vertices = file[trees["vertices"]]
        vertex_branches = [
            *COORDS,
            "cov00",
            "cov11",
            "cov22",
            "cov33",
            "nTracks",
            "chi2",
            "track_mc_keys",
            *IDENTITY,
        ]
        _require_branches(vertices, vertex_branches, trees["vertices"])
        vertices_df = pd.DataFrame(vertices.arrays(vertex_branches, library="np"))
    require_run_numbers(tracks_df, truth_df)
    return tracks_df, truth_df, histogram_df, peaks_df, vertices_df


def _remap(frame, lookup, source_file):
    """Replace run/event by the unified event id; rows of unknown events are an error."""
    if not len(frame):
        return frame.assign(event_id=np.zeros(0, dtype=np.int64))
    keys = list(
        zip(
            [str(source_file)] * len(frame),
            frame["run_number"].astype(np.int64),
            frame["event_number"].astype(np.int64),
            strict=True,
        )
    )
    ids = np.fromiter(
        (lookup.get(k, -1) for k in keys), dtype=np.int64, count=len(keys)
    )
    if (ids < 0).any():
        bad = keys[int(np.flatnonzero(ids < 0)[0])]
        raise ValueError(
            f"{source_file}: run {bad[1]} event {bad[2]} appears in a PV tree but not "
            "in the track or truth tree"
        )
    return frame.assign(event_id=ids)


def _resolve_parameters(configuration, histogram, n_bins, allow_missing):
    """The reconstruction parameters to record and the histogram axes.

    With a configuration every parameter comes from it and the binning is
    checked against the dumped bin centres; explicit bin counts must agree.
    Without one, only the binning is recorded: counts from ``n_bins``, origin
    and width from the dump.
    """
    if configuration is not None:
        parameters = parameters_from_configuration(configuration)
        for axis in AXES:
            given = n_bins.get(axis)
            if given is not None and int(given) != parameters[f"n_bins_{axis}"]:
                raise ValueError(
                    f"--n-bins-{axis} {given} contradicts the Allen configuration "
                    f"({parameters[f'n_bins_{axis}']} bins)"
                )
        axes = _configured_axes(parameters)
        _check_axes_against_dump(axes, histogram)
        return parameters, axes
    if not allow_missing:
        raise ValueError(
            f"No {CONFIGURATION_NAME} next to the input, so the reconstruction "
            "parameters are unknown. Pass --allen-config FILE, or "
            "--allow-missing-configuration to record only the histogram binning."
        )
    counts = {axis: int(n_bins.get(axis) or 0) for axis in AXES}
    if min(counts.values()) < 1:
        raise ValueError(
            "Without an Allen configuration the histogram bin counts must be given "
            "(--n-bins-z, --n-bins-t)"
        )
    axes = _histogram_axes(histogram, counts)
    parameters = {}
    for axis in AXES:
        parameters[f"{axis}_min"] = axes[axis]["min"]
        parameters[f"{axis}_max"] = axes[axis]["max"]
        parameters[f"n_bins_{axis}"] = axes[axis]["n_bins"]
    return parameters, axes


def convert_allen_root(
    inputs,
    output,
    allen_config=None,
    n_bins_z=None,
    n_bins_t=None,
    trees=None,
    min_truth_tracks=TRUTH_MIN_TRACKS,
    allow_missing_configuration=False,
):
    """Write one PVFit HDF5 file from one or more Allen ROOT files.

    Returns a summary dict (events, tracks, vertices, parameters).
    """
    inputs = [Path(p) for p in inputs]
    if not inputs:
        raise ValueError("No input ROOT files given")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"ROOT file not found: {path}")
    trees = {**DEFAULT_TREES, **(trees or {})}
    n_bins = {"z": n_bins_z, "t": n_bins_t}

    configuration, configuration_paths = load_configuration(inputs, allen_config)
    if configuration is None:
        print(
            f"WARNING: no {CONFIGURATION_NAME} next to the input; the HDF5 "
            "records only the histogram binning. The other reconstruction "
            "parameters are unknown, not assumed."
        )

    per_file, other = [], []
    for path in inputs:
        print(f"Reading {path}")
        tracks_df, truth_df, histogram_df, peaks_df, vertices_df = _read_file(
            path, trees
        )
        per_file.append((tracks_df, truth_df, path))
        other.append((histogram_df, peaks_df, vertices_df, path))
    tracks, truth, identity = unify_event_identity(per_file)
    lookup = {
        (str(row.source_file), int(row.run_number), int(row.source_event_number)): int(
            row.event_number
        )
        for row in identity.itertuples(index=False)
    }
    histogram = pd.concat(
        [_remap(h, lookup, p) for h, _, _, p in other], ignore_index=True
    )
    peaks = pd.concat([_remap(k, lookup, p) for _, k, _, p in other], ignore_index=True)
    vertices = pd.concat(
        [_remap(v, lookup, p) for _, _, v, p in other], ignore_index=True
    )
    parameters, axes = _resolve_parameters(
        configuration, histogram, n_bins, allow_missing_configuration
    )

    identity = identity.set_index("event_number")
    tracks_by_event = tracks.groupby("event_number")
    truth_by_event = truth.groupby("event_number")
    histogram_by_event = histogram.groupby("event_id")
    peaks_by_event = peaks.groupby("event_id")
    vertices_by_event = vertices.groupby("event_id")

    def group(grouped, key, template):
        return grouped.get_group(key) if key in grouped.groups else template.iloc[:0]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_ids = sorted(identity.index.tolist())
    summary = {
        "inputs": [str(p.resolve()) for p in inputs],
        "output": str(output.resolve()),
        "events": len(event_ids),
        "tracks": 0,
        "true_vertices": 0,
        "rec_vertices": 0,
        "seeds": 0,
        "allen_configuration": "present" if configuration is not None else "missing",
        "parameters": parameters,
    }

    with h5py.File(output, "w") as f:
        f.attrs["n_events"] = len(event_ids)
        f.attrs["source"] = "Allen"
        f.attrs["truth_track_selection"] = "all_input"
        f.attrs["assignment_track_selection"] = "allen_mc_key"
        f.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        for key, value in implementation_fingerprint().items():
            f.attrs[key] = value
        f.attrs["truth_min_tracks"] = int(min_truth_tracks)
        f.attrs["converter"] = "pvfit convert-allen"
        f.attrs["converter_trees"] = json.dumps(trees, sort_keys=True)
        write_parameters(f.attrs, parameters, complete=configuration is not None)
        if configuration is not None:
            for key, value in configuration_attributes(configuration_paths[0]).items():
                f.attrs[key] = value

        for event_id in tqdm(event_ids, desc="Converting events", unit="event"):
            event_tracks = group(tracks_by_event, event_id, tracks).reset_index(
                drop=True
            )
            event_truth = group(truth_by_event, event_id, truth).reset_index(drop=True)
            event_hist = group(histogram_by_event, event_id, histogram)
            event_peaks = group(peaks_by_event, event_id, peaks)
            event_vertices = group(vertices_by_event, event_id, vertices)
            keys = event_tracks["mc_key"].to_numpy().astype(np.int64)
            n_tracks = len(event_tracks)

            true_info = (
                build_true_vertex_info(
                    event_tracks, event_truth, min_tracks=min_truth_tracks
                )
                if n_tracks
                else []
            )
            selected = (
                event_tracks["beamPOCA_z"].between(axes["z"]["min"], axes["z"]["max"])
                & event_tracks["beamPOCA_t"].between(axes["t"]["min"], axes["t"]["max"])
            ).to_numpy()

            origin = identity.loc[event_id]
            event = f.create_group(str(event_id))
            event.attrs["n_tracks"] = n_tracks
            event.attrs["n_seeding_tracks"] = int(selected.sum())
            event.attrs["truth_track_selection"] = "all_input"
            event.attrs["assignment_track_selection"] = "allen_mc_key"
            event.attrs["n_true_vertices"] = len(true_info)
            event.attrs["n_rec_vertices"] = len(event_vertices)
            event.attrs["n_seedings"] = len(event_peaks)
            event.attrs["truth_matching"] = "key"
            event.attrs["run_number"] = int(origin["run_number"])
            event.attrs["source_event_number"] = int(origin["source_event_number"])
            event.attrs["source_file"] = str(origin["source_file"])

            histogram_ds = event.create_dataset(
                "histogram",
                data=_dense_histogram(event_hist, axes),
                compression="gzip",
                compression_opts=4,
            )
            for axis in AXES:
                histogram_ds.attrs[f"{axis}_min"] = axes[axis]["min"]
                histogram_ds.attrs[f"{axis}_max"] = axes[axis]["max"]
                histogram_ds.attrs[f"n_bins_{axis}"] = axes[axis]["n_bins"]

            seeds = event.create_group("seeds")
            seeds.create_dataset(
                "z", data=event_peaks["z_peak"].to_numpy().astype(np.float32)
            )
            seeds.create_dataset(
                "t", data=event_peaks["t_peak"].to_numpy().astype(np.float32)
            )

            true = event.create_group("true_vertices")
            for name, (key, dtype) in TRUE_COLUMNS.items():
                _write(true, name, [v[key] for v in true_info], dtype)
            _write_vlen_int32(
                true, "track_indices", [v["track_indices"] for v in true_info]
            )

            rec = event.create_group("rec_vertices")
            for coord, cov in zip(
                COORDS, ("cov00", "cov11", "cov22", "cov33"), strict=True
            ):
                _write(rec, coord, event_vertices[coord].to_numpy(), np.float32)
                _write(
                    rec,
                    f"{coord}_err",
                    np.sqrt(
                        np.clip(
                            event_vertices[cov].to_numpy().astype(np.float64), 0.0, None
                        )
                    ),
                    np.float32,
                )
            _write(
                rec,
                "n_tracks_weighted",
                event_vertices["nTracks"].to_numpy(),
                np.float32,
            )
            _write(rec, "chi2", event_vertices["chi2"].to_numpy(), np.float32)
            _write_vlen_int32(
                rec,
                "track_indices",
                [
                    _member_rows(keys, member)
                    for member in event_vertices["track_mc_keys"]
                ],
            )

            table = event.create_group("tracks")
            for name, (column, dtype) in TRACK_COLUMNS.items():
                if column is None:
                    _write(table, name, np.arange(n_tracks), dtype)
                else:
                    _write(table, name, event_tracks[column].to_numpy(), dtype)
            _write(table, "selected_for_seeding", selected, np.bool_)

            summary["tracks"] += n_tracks
            summary["true_vertices"] += len(true_info)
            summary["rec_vertices"] += len(event_vertices)
            summary["seeds"] += len(event_peaks)

    print(
        f"Converted {summary['events']} events, {summary['tracks']} tracks, "
        f"{summary['true_vertices']} true and {summary['rec_vertices']} reconstructed vertices "
        f"to {output} (Allen configuration: {summary['allen_configuration']})"
    )
    return summary
