"""Write PVfit results to HDF5 for the analysis stage.

The layout (one group per event with histogram, seeds/, true_vertices/,
rec_vertices/ and tracks/) is documented in docs/data-format.md under "HDF5 layout";
the tables below are its definition. Every dataset is gzip-compressed.
"""

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ..provenance import HDF5_SCHEMA_VERSION, implementation_fingerprint
from ..reconstruction_parameters import write_parameters
from .truth_matching import TRUTH_MIN_TRACKS

# tracks/<name>: (column of the full input-track DataFrame, dtype)
TRACK_COLUMNS = {
    "track_id": (None, np.int32),  # row number; set separately
    "mc_key": ("mc_key", np.int64),
    "x": ("beamPOCA_x", np.float32),
    "y": ("beamPOCA_y", np.float32),
    "z": ("beamPOCA_z", np.float32),
    "t": ("beamPOCA_t", np.float32),
    "tx": ("beamPOCA_tx", np.float32),
    "ty": ("beamPOCA_ty", np.float32),
    "backward": ("backward", np.bool_),
    "cov_xx": ("c00", np.float32),
    "cov_yy": ("c11", np.float32),
    "cov_tt": ("c55", np.float32),
    "ovtx_x": ("ovtx_x", np.float32),
    "ovtx_y": ("ovtx_y", np.float32),
    "ovtx_z": ("ovtx_z", np.float32),
    "ovtx_t": ("ovtx_t", np.float32),
    "p": ("p", np.float32),
    "pt": ("pt", np.float32),
    "chi2": ("chi2", np.float32),
    "fromPV": ("fromPV", np.bool_),
}

# true_vertices/<name>: (key of the truth dict, dtype)
TRUE_COLUMNS = {
    "vertex_id": ("vertex_id", np.int32),
    "effective_pv_key": ("effective_pv_key", np.int64),
    "x": ("x", np.float32),
    "y": ("y", np.float32),
    "z": ("z", np.float32),
    "t": ("t", np.float32),
    "expected_ntracks": ("expected_ntracks", np.int32),
}

COORDS = ("x", "y", "z", "t")


def _rec_values(vertex):
    """rec_vertices/ fields of one vertex: fitted position, errors and seed."""
    values = {}
    for i, coord in enumerate(COORDS):
        seed = vertex.get(f"rec_{coord}", 0.0)
        values[coord] = vertex.get(f"Params_rec_{coord}", seed)
        values[f"{coord}_err"] = np.sqrt(vertex.get(f"cov_{i}{i}", 0.0))
        values[f"{coord}_seed"] = seed
    return values


def _write(group, name, data, dtype=None):
    data = np.asarray(data, dtype=dtype)
    group.create_dataset(name, data=data, compression="gzip", compression_opts=4)


def _write_vlen_int32(group, name, rows):
    """A ragged int32 dataset; built row by row because np.array(rows,
    dtype=object) becomes 2D when every row has the same length."""
    out = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        out[i] = np.asarray(row, dtype=np.int32)
    group.create_dataset(
        name,
        data=out,
        dtype=h5py.vlen_dtype(np.int32),
        compression="gzip",
        compression_opts=4,
    )


def save_event_to_hdf5(event_result, event_data, output_file, params):
    """Append one event's results to the HDF5 file.

    Args:
        event_result: Event processing results from process_event()
        event_data: Full input track data for the event (pandas DataFrame)
        output_file: Path to output HDF5 file
        params: Algorithm parameters (for histogram metadata)
    """
    true_info = event_result["vertex_true_info"]
    rec_info = event_result["vertex_rec_info"]
    n_tracks = len(event_data)
    selected = np.asarray(event_result["selected_for_seeding"], dtype=bool)
    if selected.shape != (n_tracks,):
        raise ValueError("Seeding selection must have one flag per input track")

    with h5py.File(output_file, "a") as f:
        name = str(event_result["event_number"])
        if name in f:
            del f[name]
        event = f.create_group(name)
        event.attrs["n_tracks"] = n_tracks
        event.attrs["n_seeding_tracks"] = int(selected.sum())
        event.attrs["truth_track_selection"] = "all_input"
        event.attrs["assignment_track_selection"] = "all_input"
        event.attrs["n_true_vertices"] = len(true_info)
        event.attrs["n_rec_vertices"] = len(rec_info)
        event.attrs["n_seedings"] = event_result["n_seedings"]
        event.attrs["truth_matching"] = "key"
        # The dump's own identity of the event, attached by run_fit (see
        # event_identity); results built elsewhere may not carry it
        for attr in ("run_number", "source_event_number", "source_file"):
            if attr in event_result:
                event.attrs[attr] = event_result[attr]

        histogram = _write_dataset_returning(
            event, "histogram", event_result["histogram"]
        )
        for attr in ("z_min", "z_max", "t_min", "t_max", "n_bins_z", "n_bins_t"):
            histogram.attrs[attr] = params[attr]

        # Every seed, including those later dropped or merged
        seeds = np.asarray(event_result["seeds"], dtype=np.float32).reshape(-1, 2)
        seeds_group = event.create_group("seeds")
        seeds_group.create_dataset("z", data=seeds[:, 0])
        seeds_group.create_dataset("t", data=seeds[:, 1])

        true = event.create_group("true_vertices")
        for name, (key, dtype) in TRUE_COLUMNS.items():
            _write(true, name, [v[key] for v in true_info], dtype)
        _write_vlen_int32(
            true, "track_indices", [v["track_indices"] for v in true_info]
        )

        rec = event.create_group("rec_vertices")
        rec_values = [_rec_values(v) for v in rec_info]
        for coord in COORDS:
            for suffix in ("", "_err", "_seed"):
                name = f"{coord}{suffix}"
                _write(rec, name, [v[name] for v in rec_values], np.float32)
        _write_vlen_int32(rec, "track_indices", [v["track_index"] for v in rec_info])

        tracks = event.create_group("tracks")
        _write(tracks, "selected_for_seeding", selected, np.bool_)
        for name, (column, dtype) in TRACK_COLUMNS.items():
            if column is None:
                _write(tracks, name, np.arange(n_tracks), dtype)
            else:
                _write(tracks, name, event_data[column].to_numpy(), dtype)


def _write_dataset_returning(group, name, data):
    return group.create_dataset(name, data=data, compression="gzip", compression_opts=4)


def save_all_events_to_hdf5(
    fit_results,
    output_file,
    params,
    *,
    provenance_path=None,
    track_input_attributes=None,
):
    """Save all events' results to HDF5 file with dynamic sizes per event.

    Args:
        fit_results: List of event processing results from process_event()
        output_file: Path to output HDF5 file
        params: the reconstruction parameters (ReconstructionConfig.as_params());
            every one named in pvfit.reconstruction_parameters is recorded as a
            root attribute reco_<name>, track-input parameters as input_<name>
        provenance_path: CLI stage record, when run from the CLI
        track_input_attributes: attributes naming the configuration file the
            track-input parameters were read from, when there is one
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(fit_results)} events to HDF5 file: {output_file}")

    # Clear existing file
    with h5py.File(output_file, "w") as f:
        f.attrs["n_events"] = len(fit_results)
        f.attrs["source"] = "PVFit"
        f.attrs["truth_track_selection"] = "all_input"
        f.attrs["assignment_track_selection"] = "all_input"
        f.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        for key, value in implementation_fingerprint().items():
            f.attrs[key] = value
        f.attrs["truth_min_tracks"] = TRUTH_MIN_TRACKS
        write_parameters(f.attrs, params)
        for key, value in (track_input_attributes or {}).items():
            f.attrs[key] = value
        if provenance_path is not None:
            f.attrs["provenance_manifest"] = str(Path(provenance_path).resolve())

    for event_result in tqdm(fit_results, desc="Saving events to HDF5"):
        save_event_to_hdf5(
            event_result, event_result["event_data"], output_file, params
        )

    print(f"Successfully saved {len(fit_results)} events to {output_file}")
