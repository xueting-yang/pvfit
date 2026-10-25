"""Configuration for reconstruction and independent analysis.

Each stage has one frozen dataclass. Its fields carry the defaults, its
``from_mapping`` is the only code that knows the YAML layout and validates it,
and ``to_mapping`` writes that layout back. Everything else asks the dataclass.
"""

import math
import multiprocessing as mp
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from .reconstruction_parameters import (
    REFERENCE_BIN_SIZE_T,
    REFERENCE_BIN_SIZE_Z,
    scaled_min_tracks_in_seed,
)

ANALYSIS_NAMES = (
    "efficiency_track",
    "efficiency_geo",
    "fake_rate_track",
    "fake_rate_geo",
    "merge_rate",
    "split_rate",
    "merge_rate_geo",
    "split_rate_geo",
    "resolution",
    "plot_histogram",
    "track_pull",
    "vertex_pull",
)
# Analyses run when neither the YAML nor the command line selects any.
DEFAULT_PERFORMANCE_ANALYSES = (
    "efficiency_track",
    "efficiency_geo",
    "fake_rate_track",
    "fake_rate_geo",
    "merge_rate",
    "split_rate",
)

# Reference bin sizes: z ∈ [-200, 200] / 800 and t ∈ [-0.8, 0.8] / 64.
REF_BIN_SIZE_Z = REFERENCE_BIN_SIZE_Z  # mm
REF_BIN_SIZE_T = REFERENCE_BIN_SIZE_T  # ns


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_file}")
    return config


def _section(config, name):
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return section


def _reject_unknown(section, name, known):
    unknown = sorted(set(section) - set(known))
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(unknown)}")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    """A finite int or float; NaN and infinities are not numbers here."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_bool(value):
    return isinstance(value, bool)


def _check_bools(config, names, section):
    """Flags must be real booleans: the string "false" would count as true."""
    for name in names:
        if not _is_bool(getattr(config, name)):
            raise ValueError(f"{section}.{name} must be true or false")


def _check_positive(config, names, section, *, allow_inf=False):
    """Positive finite numbers; thresholds whose cut can be disabled allow +inf."""
    for name in names:
        value = getattr(config, name)
        if allow_inf and isinstance(value, float) and value == math.inf:
            continue
        if not _is_number(value) or value <= 0:
            kind = "positive number or .inf" if allow_inf else "positive finite number"
            raise ValueError(f"{section}.{name} must be a {kind}")


def _check_nonnegative(config, names, section):
    for name in names:
        value = getattr(config, name)
        if not _is_number(value) or value < 0:
            raise ValueError(f"{section}.{name} must be a nonnegative finite number")


def _check_ints(config, names, section, *, minimum):
    for name in names:
        value = getattr(config, name)
        if not _is_int(value) or value < minimum:
            bound = "positive" if minimum > 0 else "nonnegative"
            raise ValueError(f"{section}.{name} must be a {bound} integer")


def _check_path(value, name):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")


def _default_workers():
    return max(1, mp.cpu_count() // 2)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Settings of ``pvfit fit``: the YAML's paths, workflow and algorithm."""

    # paths (data paths are relative to the working directory)
    input_root: str
    track_tree_name: str
    mc_vertex_tree_name: str
    output_dir: str
    hdf5_file: str = "fit_results.h5"
    # configuration of the program that produced the input tracks (an Allen
    # configuration JSON); relative paths resolve against each input file's
    # directory. None: record no track-input parameters.
    track_input_configuration: str | None = None

    # workflow
    event_number: int = -1  # -1: all events
    max_workers: int = field(default_factory=_default_workers)

    # algorithm. The names and defaults are Allen's beamline PV parameters
    # (pvfit/reconstruction_parameters.py); the HDF5 output records every one
    # as reco_<name>. A cut is disabled by its threshold (.inf or 0), as in
    # Allen, not by a flag.
    #
    # seeding histogram (pv_beamline_histo); the z/t window also selects the
    # tracks that enter it
    z_min: float = -200.0  # mm
    z_max: float = 200.0
    n_bins_z: int = 800
    t_min: float = -0.8  # ns
    t_max: float = 0.8
    n_bins_t: int = 64
    smear_window_nbins: int = 2
    max_track_beamline_chi2_xy: float = 10.0  # .inf: no beamline chi2 cut
    max_track_z0_err: float = 1.5  # mm; .inf: no z0 error cut
    # peak finding (pv_beamline_peak); scaled with the bin area, see below
    min_tracks_in_seed: float = 2.5
    # track-seed assignment (pv_beamline_assign_tracks): the chi2 to the seed
    # decides which tracks enter a seed's fit
    max_track_seed_chi2: float = 10.0
    min_tracks_per_seed: int = 4  # 0: keep every seed
    # vertex fit (pv_beamline_fitter): the Tukey threshold weights the tracks
    # in the fit and decides which of them remain members of the vertex
    max_chi2_tukey: float = 12.0
    max_fit_iterations: int = 20
    max_delta_z_converged: float = 0.0005  # mm
    max_vertex_rho2: float = 0.09  # mm^2; .inf: keep vertices off the beamline
    # merging (pv_beamline_merge_vertices)
    min_chi2_dist_merge: float = 25.0  # 0: no merging
    # PVFit-only debugging switch; Allen always recomputes the chi2 (true)
    is_adaptive_tukey: bool = True

    _PATH_KEYS = (
        "input_root",
        "track_tree_name",
        "mc_vertex_tree_name",
        "output_dir",
        "hdf5_file",
        "track_input_configuration",
    )
    _WORKFLOW_KEYS = ("event_number", "max_workers")
    _ALGORITHM_KEYS = (
        "z_min", "z_max", "n_bins_z", "t_min", "t_max", "n_bins_t",
        "smear_window_nbins", "max_track_beamline_chi2_xy", "max_track_z0_err",
        "min_tracks_in_seed", "max_track_seed_chi2", "min_tracks_per_seed",
        "max_chi2_tukey", "max_fit_iterations", "max_delta_z_converged",
        "max_vertex_rho2", "min_chi2_dist_merge", "is_adaptive_tukey",
    )  # fmt: skip

    def __post_init__(self):
        for name in (
            "input_root",
            "track_tree_name",
            "mc_vertex_tree_name",
            "output_dir",
        ):
            _check_path(getattr(self, name), f"paths.{name}")
        filename = Path(self.hdf5_file)
        if (
            not str(self.hdf5_file).strip()
            or filename.is_absolute()
            or len(filename.parts) != 1
            or filename.name in (".", "..")
        ):
            raise ValueError("paths.hdf5_file must be a filename, not a directory path")
        if self.track_input_configuration is not None:
            _check_path(
                self.track_input_configuration, "paths.track_input_configuration"
            )
        if not _is_int(self.max_workers) or self.max_workers < 1:
            raise ValueError("workflow.max_workers must be a positive integer")
        if not _is_int(self.event_number) or self.event_number < -1:
            raise ValueError(
                "workflow.event_number must be -1 or a nonnegative integer"
            )
        _check_ints(
            self, ("n_bins_z", "n_bins_t", "max_fit_iterations"), "algorithm", minimum=1
        )
        _check_ints(
            self, ("smear_window_nbins", "min_tracks_per_seed"), "algorithm", minimum=0
        )
        for axis in ("z", "t"):
            low, high = getattr(self, f"{axis}_min"), getattr(self, f"{axis}_max")
            if not (_is_number(low) and _is_number(high) and low < high):
                raise ValueError(
                    f"algorithm.{axis}_min must be below algorithm.{axis}_max"
                )
        _check_positive(
            self,
            (
                "min_tracks_in_seed",
                "max_track_seed_chi2",
                "max_chi2_tukey",
                "max_delta_z_converged",
            ),
            "algorithm",
        )
        _check_positive(
            self,
            ("max_track_beamline_chi2_xy", "max_track_z0_err", "max_vertex_rho2"),
            "algorithm",
            allow_inf=True,
        )
        _check_nonnegative(self, ("min_chi2_dist_merge",), "algorithm")
        _check_bools(self, ("is_adaptive_tukey",), "algorithm")

    @classmethod
    def from_mapping(cls, config):
        """Build from a parsed reconstruction YAML; missing sections use defaults."""
        paths = _section(config, "paths")
        workflow = _section(config, "workflow")
        algorithm = _section(config, "algorithm")
        _reject_unknown(paths, "paths", cls._PATH_KEYS)
        _reject_unknown(workflow, "workflow", cls._WORKFLOW_KEYS)
        _reject_unknown(algorithm, "algorithm", cls._ALGORITHM_KEYS)

        values = {}
        for name in (
            "input_root",
            "track_tree_name",
            "mc_vertex_tree_name",
            "output_dir",
        ):
            if name not in paths:
                raise KeyError(f"paths.{name}")
            values[name] = paths[name]
        for name in ("hdf5_file", "track_input_configuration"):
            if name in paths:
                values[name] = paths[name]
        values.update(workflow)
        values.update(algorithm)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path):
        return cls.from_mapping(load_config(path))

    def to_mapping(self):
        """The YAML layout ``from_mapping`` reads."""
        d = asdict(self)
        return {
            "paths": {name: d[name] for name in self._PATH_KEYS},
            "workflow": {name: d[name] for name in self._WORKFLOW_KEYS},
            "algorithm": {name: d[name] for name in self._ALGORITHM_KEYS},
        }

    @property
    def bin_size_z(self):
        return (self.z_max - self.z_min) / self.n_bins_z

    @property
    def bin_size_t(self):
        return (self.t_max - self.t_min) / self.n_bins_t

    @property
    def scaled_min_tracks_in_seed(self):
        """min_tracks_in_seed at this binning: scaled by the bin area relative
        to the reference binning, as Allen's pv_beamline_peak does, so the peak
        threshold stays physically consistent across binnings."""
        return scaled_min_tracks_in_seed(
            self.min_tracks_in_seed, self.bin_size_z, self.bin_size_t
        )

    def as_params(self):
        """The parameter dict the reconstruction modules read: every algorithm
        key under its own name, plus the derived bin sizes and peak threshold."""
        return {
            **{name: getattr(self, name) for name in self._ALGORITHM_KEYS},
            "bin_size_z": self.bin_size_z,
            "bin_size_t": self.bin_size_t,
            "scaled_min_tracks_in_seed": self.scaled_min_tracks_in_seed,
        }


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings of ``pvfit analyze``: the YAML's analysis and paths sections."""

    analyses: tuple = DEFAULT_PERFORMANCE_ANALYSES
    method_name: str = "PVfit"

    # Execution only; all workers use identical event-level analysis definitions.
    workers: int = 1
    batches: int | None = None  # automatic: four batches per worker
    work_dir: str | None = None  # parent for this run's private scratch directory
    keep_columns: bool = False

    # reconstructed-to-true vertex matching
    recall_threshold: float = 0.7
    purity_threshold: float = 0.7
    m_r: float = 0.8
    nsigma: float = 5.0
    z_scale: float = 1.5
    t_scale: float = 0.06

    # None follows the threshold recorded by reconstruction; a number overrides it.
    track_pull_chi2: float | None = None

    # vertex-resolution selection windows
    x_residual_range: float = 0.1  # mm
    y_residual_range: float = 0.1  # mm
    z_residual_range: float = 0.7  # mm
    t_residual_range: float = 0.06  # ns

    # binning in track multiplicity for the performance plots
    plot_min_ntracks: int = 4
    plot_max_ntracks: int = 200
    plot_bin_width_ntracks: int = 5
    # Display limit for all LHCb2 efficiency figures.
    efficiency_plot_max_ntracks: int = 150
    rate_plot_max_ntracks: int = 150
    # Optional y limits keyed by e.g. "split_rate_geo_rec"; empty means automatic.
    rate_plot_ymax: dict[str, float] = field(default_factory=dict)

    # 2D histogram plot
    # None selects the first stored event; explicit selection uses source identity.
    plot_run_number: int | None = None
    plot_event_number: int | None = None
    draw_fixed_ellipses: bool = False
    draw_fitted_ellipses: bool = False

    # optional paths (relative to the working directory)
    input_hdf5: str | None = None
    output_dir: str | None = None

    _PATH_KEYS = ("input_hdf5", "output_dir")

    def __post_init__(self):
        analyses = (
            tuple(self.analyses) if isinstance(self.analyses, (list, tuple)) else None
        )
        if not analyses:
            raise ValueError("analyses must be a nonempty list of analysis names")
        unknown = [name for name in analyses if name not in ANALYSIS_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown analyses: {unknown}; choose from {', '.join(ANALYSIS_NAMES)}"
            )
        object.__setattr__(self, "analyses", analyses)
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("analysis.method_name must be a nonempty string")
        for name in ("workers", "batches"):
            value = getattr(self, name)
            if name == "batches" and value is None:
                continue
            if not _is_int(value) or value < 1:
                raise ValueError(f"analysis.{name} must be a positive integer")
        _check_bools(self, ("keep_columns",), "analysis")
        if self.work_dir is not None:
            _check_path(self.work_dir, "analysis.work_dir")
        for name in ("recall_threshold", "purity_threshold"):
            value = getattr(self, name)
            if not _is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"analysis.{name} must be a number between 0 and 1")
        _check_positive(
            self,
            (
                "m_r",
                "nsigma",
                "z_scale",
                "t_scale",
                "x_residual_range",
                "y_residual_range",
                "z_residual_range",
                "t_residual_range",
            ),
            "analysis",
        )
        if self.track_pull_chi2 is not None:
            _check_positive(self, ("track_pull_chi2",), "analysis")
        _check_bools(self, ("draw_fixed_ellipses", "draw_fitted_ellipses"), "analysis")
        for name in (
            "plot_min_ntracks",
            "plot_max_ntracks",
            "plot_bin_width_ntracks",
            "efficiency_plot_max_ntracks",
            "rate_plot_max_ntracks",
        ):
            if not _is_int(getattr(self, name)) or getattr(self, name) < 1:
                raise ValueError(f"analysis.{name} must be a positive integer")
        if not isinstance(self.rate_plot_ymax, dict):
            raise ValueError("analysis.rate_plot_ymax must be a mapping")
        rate_plot_keys = {
            f"{metric}_{matching}_{binned_by}"
            for metric, binnings in (
                ("fake_rate", ("rec",)),
                ("merge_rate", ("true", "rec")),
                ("split_rate", ("true", "rec")),
            )
            for matching in ("track", "geo", "reso")
            for binned_by in binnings
        }
        _reject_unknown(self.rate_plot_ymax, "analysis.rate_plot_ymax", rate_plot_keys)
        for key, value in self.rate_plot_ymax.items():
            if not _is_number(value) or value <= 0:
                raise ValueError(
                    f"analysis.rate_plot_ymax.{key} must be a positive finite number"
                )
        if (self.plot_run_number is None) != (self.plot_event_number is None):
            raise ValueError(
                "analysis.plot_run_number and analysis.plot_event_number "
                "must both be set or both be null"
            )
        for name in ("plot_run_number", "plot_event_number"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise ValueError(
                    f"analysis.{name} must be a nonnegative integer or null"
                )
        if self.plot_min_ntracks > self.plot_max_ntracks:
            raise ValueError(
                "analysis.plot_min_ntracks must not exceed plot_max_ntracks"
            )
        for name in self._PATH_KEYS:
            if getattr(self, name) is not None:
                _check_path(getattr(self, name), f"paths.{name}")

    @classmethod
    def analysis_keys(cls):
        return tuple(f.name for f in fields(cls) if f.name not in cls._PATH_KEYS)

    @classmethod
    def from_mapping(cls, config):
        """Build from a parsed analysis YAML; missing keys use defaults."""
        section = _section(config, "analysis")
        paths = _section(config, "paths")
        if "plot_event_id" in section:
            raise ValueError(
                "analysis.plot_event_id has been removed; use plot_run_number and "
                "plot_event_number with the original run/event identity, or leave "
                "both null to plot the first stored event"
            )
        _reject_unknown(section, "analysis", cls.analysis_keys())
        _reject_unknown(paths, "paths", cls._PATH_KEYS)
        return cls(**section, **paths)

    @classmethod
    def from_yaml(cls, path):
        return cls.from_mapping(load_config(path))

    def to_mapping(self):
        d = asdict(self)
        d["analyses"] = list(d["analyses"])
        mapping = {"analysis": {name: d[name] for name in self.analysis_keys()}}
        paths = {name: d[name] for name in self._PATH_KEYS if d[name] is not None}
        if paths:
            mapping["paths"] = paths
        return mapping
