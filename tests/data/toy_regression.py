"""Read frozen synthetic rows and summarize reconstruction without plot rendering."""

import csv
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import awkward as ak
import h5py
import numpy as np
import uproot

from pvfit.analysis.run_analysis import run_analysis
from pvfit.config_loader import AnalysisConfig, ReconstructionConfig
from pvfit.fit import run_fit

DATA = Path(__file__).parent


def write_root(data, output):
    with uproot.recreate(output) as root:
        for event_index, event in enumerate(data["events"]):
            for kind, tree in (
                ("tracks", "velo_kalman_filter/monitor_tree"),
                ("vertices", "mc_pv_dumper/MCPV_effective_tree"),
            ):
                table = event[kind]
                arrays = {}
                for index, column in enumerate(table["columns"]):
                    values = [row[index] for row in table["data"]]
                    arrays[column] = (
                        ak.Array(values)
                        if column == "track_keys"
                        else np.asarray(values, dtype=table["dtypes"][column])
                    )
                if event_index == 0:
                    types = {
                        k: "var * int64" if k == "track_keys" else v.dtype
                        for k, v in arrays.items()
                    }
                    root.mktree(tree, types)
                root[tree].extend(arrays)


def reconstruct(work, workers=1):
    data = json.loads((DATA / "toy_regression_input.json").read_text())
    sample = work / "synthetic.root"
    write_root(data, sample)
    config = ReconstructionConfig.from_mapping(data["reconstruction"])
    output = run_fit(
        [str(sample)],
        config.track_tree_name,
        config.mc_vertex_tree_name,
        config.as_params(),
        -1,
        workers,
        str(work / "fit"),
        config.hdf5_file,
    )
    return output, AnalysisConfig.from_mapping(data["analysis"])


def summarize(output, settings, directory):
    events = {}
    with h5py.File(output, "r") as source:
        for name in sorted(source, key=int):
            event = source[name]
            result = {
                k: int(event.attrs[k])
                for k in ("run_number", "source_event_number", "n_tracks", "n_seedings")
            }
            for path in (
                "tracks/mc_key",
                "true_vertices/effective_pv_key",
                "true_vertices/track_indices",
                "rec_vertices/track_indices",
                "seeds/z",
                "seeds/t",
                *[
                    f"rec_vertices/{coord}{suffix}"
                    for coord in "xyzt"
                    for suffix in ("", "_err")
                ],
            ):
                values = event[path][:]
                result[path] = (
                    [list(map(int, row)) for row in values]
                    if path.endswith("track_indices")
                    else values.tolist()
                )
            result["histogram_sum"] = float(event["histogram"][:].sum())
            events[name] = result
    with ExitStack() as stack:
        for function in ("plot_efficiency_lhcb2", "plot_rate_lhcb2"):
            stack.enter_context(patch(f"pvfit.analysis.driver.{function}"))
        run_analysis(output, str(directory), settings)
    rates = {}
    for path in sorted(directory.glob("*.csv")):
        # Pair-level diagnostics are tested separately, not frozen rate tables.
        if path.name.startswith("merge_distance_"):
            continue
        with path.open() as stream:
            rates[path.name] = list(csv.DictReader(stream))
    return {"events": events, "rates": rates}
