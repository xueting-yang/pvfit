"""Allen's beamline PV sequence as a source of PVFit's parameter record.

The campaign writes the effective Allen sequence configuration as
``allen_configuration.json`` next to ``histograms.root``, as
``algorithm instance -> property -> value`` plus a ``sequence`` entry. This
module maps those properties onto the parameters of
:mod:`pvfit.reconstruction_parameters`: for ``pvfit convert-allen`` the whole
reconstruction record, for ``pvfit fit`` the parameters of the track input
(``paths.track_input_configuration``). Nothing outside this module knows
Allen's algorithm or property names.
"""

import json
import math
from pathlib import Path

import numpy as np

from .reconstruction_parameters import (
    BY_NAME,
    PARAMETERS,
    TRACK_INPUT_PARAMETERS,
    configuration_file_attributes,
)

CONFIGURATION_NAME = "allen_configuration.json"
# parameter -> Allen (algorithm, property) pairs; every listed source must agree
SOURCES = {
    "z_min": (("pv_beamline_histo", "zmin"), ("pv_beamline_peak", "zmin")),
    "z_max": (("pv_beamline_histo", "zmax"),),
    "n_bins_z": (("pv_beamline_histo", "Nbins_z"), ("pv_beamline_peak", "Nbins_z")),
    "t_min": (("pv_beamline_histo", "tmin"), ("pv_beamline_peak", "tmin")),
    "t_max": (("pv_beamline_histo", "tmax"),),
    "n_bins_t": (("pv_beamline_histo", "Nbins_t"), ("pv_beamline_peak", "Nbins_t")),
    "smear_window_nbins": (("pv_beamline_histo", "smear_window_nbins"),),
    "max_track_beamline_chi2_xy": (("pv_beamline_histo", "maxTrackBLChi2XY"),),
    "max_track_z0_err": (("pv_beamline_histo", "pp_maxTrackZ0Err"),),
    "min_tracks_in_seed": (("pv_beamline_peak", "pp_minTracks"),),
    "max_track_seed_chi2": (("pv_beamline_assign_tracks", "maxTrackSeedChi2XYT"),),
    "min_tracks_per_seed": (("pv_beamline_assign_tracks", "pp_min_tracks_per_seed"),),
    "max_chi2_tukey": (
        ("pv_beamline_fitter", "maxChi2Tukey"),
        ("pv_beamline_merge_vertices", "maxChi2Tukey"),
    ),
    "max_fit_iterations": (
        ("pv_beamline_fitter", "maxFitIter"),
        ("pv_beamline_merge_vertices", "maxFitIter"),
    ),
    "max_delta_z_converged": (
        ("pv_beamline_fitter", "maxDeltaZConverged"),
        ("pv_beamline_merge_vertices", "maxDeltaZConverged"),
    ),
    "max_vertex_rho2": (("pv_beamline_fitter", "maxVertexRho2"),),
    "min_chi2_dist_merge": (("pv_beamline_merge_vertices", "minChi2Dist"),),
    "hit_time_resolution_ns": (("velo_kalman_filter", "hit_time_resolution_ns"),),
}
assert set(SOURCES) == {p.name for p in PARAMETERS}
# Allen stores the bin width as its own property; it must agree with the edges.
BIN_WIDTHS = {
    "z": (("pv_beamline_histo", "dz"), ("pv_beamline_peak", "dz")),
    "t": (("pv_beamline_histo", "dt"), ("pv_beamline_peak", "dt")),
}
# Allen configures floats in single precision.
FLOAT32_RELATIVE_TOLERANCE = 1e-5


def _read_json(path):
    return json.loads(Path(path).read_text())


def _same(configurations, what):
    if any(c != configurations[0] for c in configurations[1:]):
        raise ValueError(
            f"Allen configurations differ between the {what}; convert or fit them "
            "separately so each HDF5 carries one reconstruction configuration"
        )
    return configurations[0]


def load_configuration(inputs, explicit=None):
    """The Allen configuration shared by all input ROOT files, or ``None``.

    Returns ``(configuration, paths)``. One explicit path applies to every
    input; otherwise each file's sibling ``allen_configuration.json`` is read
    and all of them must agree. Mixing inputs with and without one is an
    error, because the result could not say which events were reconstructed
    with which parameters.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Allen configuration not found: {path}")
        return _read_json(path), [path]
    paths = [Path(p).with_name(CONFIGURATION_NAME) for p in inputs]
    present = [p for p in paths if p.is_file()]
    if not present:
        return None, paths
    if len(present) != len(paths):
        missing = [str(p) for p in paths if not p.is_file()]
        raise ValueError(
            "Some inputs have no Allen configuration next to them while others do; "
            f"missing: {missing}"
        )
    return _same([_read_json(p) for p in present], "input files"), present


def configuration_attributes(path):
    """Attributes naming the Allen configuration a converted record came from."""
    return configuration_file_attributes(path, "allen_configuration")


def _instance(configuration, algorithm):
    """The one algorithm instance named ``algorithm`` or ``algorithm_<suffix>``."""
    if algorithm in configuration and isinstance(configuration[algorithm], dict):
        return configuration[algorithm]
    matches = [
        name
        for name, properties in configuration.items()
        if name.startswith(algorithm + "_") and isinstance(properties, dict)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Allen configuration has {len(matches)} instances of {algorithm}; "
            "cannot tell which reconstruction parameters apply"
        )
    return configuration[matches[0]]


def _value(configuration, algorithm, prop, kind):
    properties = _instance(configuration, algorithm)
    if prop not in properties:
        raise ValueError(f"Allen configuration lacks {algorithm}.{prop}")
    value = properties[prop]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{algorithm}.{prop} must be a number, not {value!r}")
    if kind is int:
        if not float(value).is_integer():
            raise ValueError(f"{algorithm}.{prop} must be an integer, not {value!r}")
        return int(value)
    # Allen properties are floats; drop the noise of their float32 storage.
    return float(str(np.float32(value)))


def _agree(first, second, kind):
    if kind is int or math.isinf(first) or math.isinf(second):
        return first == second
    return math.isclose(first, second, rel_tol=FLOAT32_RELATIVE_TOLERANCE, abs_tol=0.0)


def _parameter(configuration, name, *, required):
    """One parameter from all its Allen sources, which must agree; ``None`` if
    an optional one is absent."""
    parameter = BY_NAME[name]
    found = []
    for algorithm, prop in SOURCES[name]:
        try:
            found.append(
                (
                    algorithm,
                    prop,
                    _value(configuration, algorithm, prop, parameter.kind),
                )
            )
        except ValueError:
            if required:
                raise
    if not found:
        return None
    first = found[0][2]
    for algorithm, prop, value in found[1:]:
        if not _agree(first, value, parameter.kind):
            raise ValueError(
                f"{name}: {found[0][0]}.{found[0][1]} = {first} but {algorithm}.{prop} = {value}"
            )
    return first


def parameters_from_configuration(configuration):
    """The full parameter record of an Allen run, from its configuration JSON.

    Every Allen source of a parameter must carry the same value (Allen repeats
    the binning in ``pv_beamline_peak`` and the fit settings in
    ``pv_beamline_merge_vertices``), and the stored bin widths must match the
    edges and counts. Nothing is assumed for a missing property.
    """
    values = {}
    for parameter in PARAMETERS:
        value = _parameter(
            configuration, parameter.name, required=not parameter.optional
        )
        if value is not None:
            values[parameter.name] = value
    for axis, sources in BIN_WIDTHS.items():
        expected = (values[f"{axis}_max"] - values[f"{axis}_min"]) / values[
            f"n_bins_{axis}"
        ]
        for algorithm, prop in sources:
            width = _value(configuration, algorithm, prop, float)
            if not _agree(width, expected, float):
                raise ValueError(
                    f"{algorithm}.{prop} = {width} does not match "
                    f"({axis}_max - {axis}_min) / n_bins_{axis} = {expected}"
                )
    return values


def track_input_parameters(configuration):
    """The track-input parameters an Allen configuration names, if any."""
    values = {}
    for name in TRACK_INPUT_PARAMETERS:
        value = _parameter(configuration, name, required=False)
        if value is not None:
            values[name] = value
    return values


def load_track_input(inputs, specification):
    """The track-input parameters for ``pvfit fit`` and the attributes naming
    their source, from the Allen configuration ``paths.track_input_configuration``
    points to.

    A relative ``specification`` is resolved against each input file's
    directory, an absolute one applies to every input. All files must exist
    and agree.
    """
    spec = Path(specification).expanduser()
    paths = []
    for path in inputs:
        candidate = spec if spec.is_absolute() else Path(path).parent / spec
        if candidate not in paths:
            paths.append(candidate)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Track input configuration not found: {missing}")
    configuration = _same([_read_json(p) for p in paths], "input files")
    parameters = track_input_parameters(configuration)
    if not parameters:
        raise ValueError(
            f"{paths[0]} names none of the track-input parameters "
            f"{list(TRACK_INPUT_PARAMETERS)}"
        )
    return parameters, configuration_file_attributes(
        paths[0], "track_input_configuration"
    )
