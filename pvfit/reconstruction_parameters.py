"""The reconstruction parameters every PVFit HDF5 file records.

PVFit's reconstruction follows the beamline PV algorithm step by step (seeding
histogram, peak finding, track-seed assignment, vertex fit, merging), so one
parameter set describes it and any other implementation of the same algorithm.
``pvfit fit`` takes the values from its YAML, using one ``max_chi2`` for
assignment and Tukey fitting and recording it for both stages. An importer
such as ``pvfit convert-allen`` records the producer's actual stage values,
which may differ. Both store each
parameter as the root attribute ``reco_<name>`` so analyses and comparisons
read one record. Parameters of the track input (group ``input``) are not set by
the vertex reconstruction; they are recorded as ``input_<name>`` when known.

The algorithm has no on/off switches, only thresholds, and this table keeps
that: a disabled track cut is an infinite threshold, disabled merging a zero
chi2 distance and ``min_tracks_per_seed = 0`` keeps every seed.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

ATTRIBUTE_PREFIXES = {"reconstruction": "reco_", "input": "input_"}
# Reference binning; the seed threshold scales with the bin area relative to it.
REFERENCE_BIN_SIZE_Z = 0.5  # mm
REFERENCE_BIN_SIZE_T = 0.025  # ns


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: type
    description: str
    group: str = "reconstruction"

    @property
    def optional(self):
        """Track-input parameters are recorded only when known."""
        return self.group == "input"

    @property
    def attribute(self):
        return ATTRIBUTE_PREFIXES[self.group] + self.name


PARAMETERS = (
    # seeding histogram
    Parameter(
        "z_min",
        float,
        "mm; lower edge of the seeding histogram and of the seeding track window",
    ),
    Parameter("z_max", float, "mm; upper z edge"),
    Parameter("n_bins_z", int, "z bins"),
    Parameter("t_min", float, "ns; lower t edge"),
    Parameter("t_max", float, "ns; upper t edge"),
    Parameter("n_bins_t", int, "t bins"),
    Parameter(
        "smear_window_nbins",
        int,
        "bins on each side of a track over which its Gaussian is spread",
    ),
    Parameter(
        "max_track_beamline_chi2_xy",
        float,
        "tracks with a larger xy chi2 to the beamline do not enter the histogram",
    ),
    Parameter(
        "max_track_z0_err",
        float,
        "mm; tracks with a larger z error at the beamline do not enter the histogram",
    ),
    # peak finding
    Parameter(
        "min_tracks_in_seed",
        float,
        "3x3 integral a histogram maximum needs, at the reference binning",
    ),
    # track-seed assignment
    Parameter(
        "max_track_seed_chi2",
        float,
        "a track joins its closest seed only below this xyt chi2",
    ),
    Parameter(
        "min_tracks_per_seed",
        int,
        "seeds with fewer tracks are dropped and their tracks reassigned",
    ),
    # vertex fit
    Parameter(
        "max_chi2_tukey", float, "Tukey bisquare threshold of the track chi2 in the fit"
    ),
    Parameter("max_fit_iterations", int, "iterations of the vertex fit"),
    Parameter(
        "max_delta_z_converged", float, "mm; the fit stops once a z step is smaller"
    ),
    Parameter(
        "max_vertex_rho2",
        float,
        "mm^2; fitted vertices farther from the beamline are dropped",
    ),
    # merging
    Parameter(
        "min_chi2_dist_merge",
        float,
        "vertices closer than this (z, t) chi2 distance are merged and refitted",
    ),
    # track input
    Parameter(
        "hit_time_resolution_ns",
        float,
        "ns; per-hit time resolution assumed by the track fit",
        group="input",
    ),
)
PARAMETER_NAMES = tuple(p.name for p in PARAMETERS)
BY_NAME = {p.name: p for p in PARAMETERS}
TRACK_INPUT_PARAMETERS = tuple(p.name for p in PARAMETERS if p.group == "input")


def attribute_name(parameter):
    return BY_NAME[parameter].attribute


def scaled_min_tracks_in_seed(min_tracks_in_seed, bin_size_z, bin_size_t):
    """The peak threshold at this binning, scaled by the bin area relative to
    the reference binning so it stays physically comparable."""
    return (
        min_tracks_in_seed
        * (bin_size_z * bin_size_t)
        / (REFERENCE_BIN_SIZE_Z * REFERENCE_BIN_SIZE_T)
    )


def write_parameters(attrs, values, *, complete=True):
    """Store ``values[name]`` as its attribute for every known parameter.

    With ``complete`` every reconstruction parameter must be present, so a
    file never silently lacks part of its record; track-input parameters are
    written when present.
    """
    for parameter in PARAMETERS:
        if parameter.name not in values:
            if complete and not parameter.optional:
                raise KeyError(f"Reconstruction parameter {parameter.name} is missing")
            continue
        value = values[parameter.name]
        if parameter.kind is int:
            if not float(value).is_integer():
                raise ValueError(f"{parameter.name} must be an integer, not {value!r}")
            attrs[parameter.attribute] = int(value)
        else:
            attrs[parameter.attribute] = float(value)


def read_parameters(attrs):
    """The recorded parameters of an HDF5 file, by name."""
    return {
        parameter.name: parameter.kind(attrs[parameter.attribute])
        for parameter in PARAMETERS
        if parameter.attribute in attrs
    }


def configuration_file_attributes(path, prefix):
    """``<prefix>_path`` and ``<prefix>_sha256`` of a configuration file a
    record was taken from."""
    path = Path(path)
    return {
        f"{prefix}_path": str(path.resolve()),
        f"{prefix}_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
