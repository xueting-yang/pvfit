"""Optimize Allen runtime properties, then score its ROOT dump through PVFit.

The baseline is a complete, executable Allen sequence JSON, not a PVFit
reconstruction YAML. Each trial edits a private copy. GPU kernels already expose
these properties, so changing them never edits or rebuilds the user's stack.
"""

import copy
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import yaml

from ...allen import BIN_WIDTHS, SOURCES, _instance, parameters_from_configuration
from ...reconstruction_parameters import scaled_min_tracks_in_seed
from .base import Backend, run_logged

# LD_LIBRARY_PATH must be restored before starting Python: updating it inside
# Python is too late for the dynamic loader used by ROOT/Allen shared libraries.
_GPU_ENVIRONMENT_SHELL = """set -eu
if [ "$1" != __PVFIT_UNSET__ ]; then export CUDA_VISIBLE_DEVICES="$1"; fi
export CUDA_DEVICE_ORDER="$2"
if [ -n "$3" ]; then export PATH="$3:$PATH"; fi
export LD_LIBRARY_PATH="${4:+$4:}/.singularity.d/libs:${LD_LIBRARY_PATH:-}"
shift 4
exec python "$@"
"""


PARAMETER_TYPES = {
    "min_tracks_in_seed": "float",
    "smear_window_nbins": "int",
    "n_bins_z": "int",
    "n_bins_t": "int",
    "max_track_seed_chi2": "float",
    "max_chi2_tukey": "float",
    "min_chi2_dist_merge": "float",
    "max_track_beamline_chi2_xy": "float",
    "max_track_z0_err": "float",
}
PATH_KEYS = {"input_mdf", "allen_run", "entrypoint", "sequence", "runtime_files"}
RUNTIME_DEFAULTS = {
    "n_events": 1000,
    "events_per_slice": 200,
    "device": "0",
    "timeout_seconds": 1800,
    "tags": "dd4hep:run5/branch-baseline,run5-ideal",
    "params": "${PARAMFILESROOT}",
    "detector_list": ["/world", "TV"],
    "memory_mb": 1000,
    "host_memory_mb": 200,
    "cuda_bin": None,
    "cuda_lib": None,
}


def _path(value):
    """CWD-relative, with environment/home expansion; preserve Allen/run symlinks.

    lb-stack-setup's run wrapper uses its invocation path to select the project;
    resolving the symlink here would lose that information.
    """
    value = os.path.expandvars(str(value))
    if "$" in value:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return Path(value).expanduser().absolute()


def _settings(base_cfg):
    if not isinstance(base_cfg, dict) or set(base_cfg) - {"paths", "runtime"}:
        raise ValueError("Allen configuration accepts only paths and runtime")
    paths = base_cfg.get("paths")
    if not isinstance(paths, dict) or set(paths) != PATH_KEYS:
        raise ValueError(
            f"Allen paths must define exactly: {', '.join(sorted(PATH_KEYS))}"
        )
    for key in PATH_KEYS - {"runtime_files"}:
        if not isinstance(paths[key], str) or not paths[key].strip():
            raise ValueError(f"paths.{key} must be a nonempty path")
    assets = paths["runtime_files"]
    if (
        not isinstance(assets, list)
        or not assets
        or any(not isinstance(p, str) or not p.strip() for p in assets)
    ):
        raise ValueError(
            "paths.runtime_files must list the installed Allen libraries/assets"
        )
    given = base_cfg.get("runtime", {})
    if not isinstance(given, dict) or set(given) - set(RUNTIME_DEFAULTS):
        raise ValueError("Unknown Allen runtime options")
    runtime = {**RUNTIME_DEFAULTS, **given}
    for name in ("n_events", "events_per_slice", "memory_mb", "host_memory_mb"):
        value = runtime[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"runtime.{name} must be an integer >= 1")
    timeout = runtime["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("runtime.timeout_seconds must be finite and > 0")
    for name in ("device", "tags", "params"):
        if not isinstance(runtime[name], str) or not runtime[name].strip():
            raise ValueError(f"runtime.{name} must be a nonempty string")
    detectors = runtime["detector_list"]
    if not isinstance(detectors, list) or any(
        not isinstance(d, str) or not d.strip() for d in detectors
    ):
        raise ValueError("runtime.detector_list must be a list of detector names")
    for name in ("cuda_bin", "cuda_lib"):
        if runtime[name] is not None:
            if not isinstance(runtime[name], str) or not runtime[name].strip():
                raise ValueError(f"runtime.{name} must be a path or null")
            runtime[name] = str(_path(runtime[name]))
    return paths, runtime


def _sequence(base_cfg):
    paths, _ = _settings(base_cfg)
    configuration = json.loads(_path(paths["sequence"]).read_text())
    if (
        not isinstance(configuration, dict)
        or not isinstance(configuration.get("sequence"), dict)
        or not configuration["sequence"].get("configured_algorithms")
    ):
        raise ValueError(
            "paths.sequence must be a complete executable Allen sequence JSON"
        )
    parameters = parameters_from_configuration(configuration)
    if any(parameters[f"n_bins_{axis}"] < 3 for axis in ("z", "t")):
        raise ValueError("Allen peak finding needs at least 3 bins on each axis")
    # The default TV sequence enables the required monitor trees. Reject an
    # explicitly disabled tree; truth/track monitors are unconditional in this stack.
    for algorithm in (
        "pv_beamline_histo",
        "pv_beamline_peak",
        "pv_beamline_merge_vertices",
    ):
        if _instance(configuration, algorithm).get("enable_monitor") is not True:
            raise ValueError(f"{algorithm}.enable_monitor must be true for evaluation")
    return configuration


def trial_sequence(base_cfg, sampled):
    """Map PV parameter names to all Allen consumers, without double scaling.

    pp_minTracks is the *unscaled* threshold. Allen's peak kernel already
    multiplies it by dz*dt/(0.5 mm * 0.025 ns). The histo and peak must agree on
    both bin counts and widths; fitter and merge refitter share the Tukey cut.
    """
    configuration = copy.deepcopy(_sequence(base_cfg))
    for name, value in sampled.items():
        if name not in PARAMETER_TYPES:
            raise ValueError(f"Unsupported Allen search parameter: {name}")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or (PARAMETER_TYPES[name] == "int" and not isinstance(value, int))
        ):
            raise ValueError(f"Invalid Allen parameter {name}: {value!r}")
        if name in {"n_bins_z", "n_bins_t"} and value < 3:
            raise ValueError("Allen peak finding needs at least 3 bins on each axis")
        for algorithm, prop in SOURCES[name]:
            _instance(configuration, algorithm)[prop] = value
    # Use validated physical edges, including float32 cleanup, before changing
    # widths. parameters_from_configuration would reject intermediate binning.
    histo = _instance(configuration, "pv_beamline_histo")
    for axis in ("z", "t"):
        width = (histo[f"{axis}max"] - histo[f"{axis}min"]) / histo[f"Nbins_{axis}"]
        for algorithm, prop in BIN_WIDTHS[axis]:
            _instance(configuration, algorithm)[prop] = width
    parameters_from_configuration(configuration)
    return configuration


def _runtime_files(paths):
    files = [_path(paths[key]) for key in ("allen_run", "entrypoint", "sequence")]
    for pattern in paths["runtime_files"]:
        matches = sorted(glob.glob(str(_path(pattern)), recursive=True))
        if not matches:
            raise FileNotFoundError(f"No Allen runtime files matched: {pattern}")
        for match in matches:
            path = Path(match)
            files.extend(
                sorted(p for p in path.rglob("*") if p.is_file())
                if path.is_dir()
                else [path]
            )
    return sorted(set(files))


class AllenBackend(Backend):
    name = "allen"
    analysis_method_name = "Allen"
    max_parallel_trials = 1
    max_trial_workers = 1

    def validate_config(self, base_cfg):
        paths, runtime = _settings(base_cfg)
        _sequence(base_cfg)
        for name in ("cuda_bin", "cuda_lib"):
            if runtime[name] is not None and not Path(runtime[name]).is_dir():
                raise FileNotFoundError(
                    f"runtime.{name} directory not found: {runtime[name]}"
                )
        for path in _runtime_files(paths):
            if not path.is_file():
                raise FileNotFoundError(f"Allen runtime file not found: {path}")
        if not os.access(_path(paths["allen_run"]), os.X_OK):
            raise ValueError("paths.allen_run must be executable")

    def input_pattern(self, base_cfg):
        paths, _ = _settings(base_cfg)
        return str(_path(paths["input_mdf"]))

    def fingerprint(self, base_cfg, root):
        """Pin the sequence, launch scripts and installed libraries, not just Python."""
        paths, _ = _settings(base_cfg)
        records = []
        for path in _runtime_files(paths):
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            records.append(
                {"path": str(path), "target": str(path.resolve()), "sha256": digest}
            )
        return {"files": records}

    def validate_search_space(self, search_space):
        for name, spec in search_space.items():
            if name not in PARAMETER_TYPES:
                raise ValueError(f"Unsupported Allen search parameter: {name}")
            if spec["type"] != PARAMETER_TYPES[name]:
                raise ValueError(
                    f"search_space.{name}.type must be {PARAMETER_TYPES[name]}"
                )
            if name in {"n_bins_z", "n_bins_t"} and spec["low"] < 3:
                raise ValueError(
                    "Allen peak finding needs at least 3 bins on each axis"
                )

    def derived(self, sampled, base_cfg):
        p = parameters_from_configuration(trial_sequence(base_cfg, sampled))
        dz = (p["z_max"] - p["z_min"]) / p["n_bins_z"]
        dt = (p["t_max"] - p["t_min"]) / p["n_bins_t"]
        return {
            "derived_bin_size_z": dz,
            "derived_bin_size_t": dt,
            "derived_scaled_min_tracks_in_seed": scaled_min_tracks_in_seed(
                p["min_tracks_in_seed"], dz, dt
            ),
        }

    def write_config(self, base_cfg, sampled, trial_dir, trial_workers, input_root):
        if trial_workers != 1:
            raise ValueError("Allen monitoring requires execution.trial_workers: 1")
        paths, runtime = _settings(base_cfg)
        trial_dir = Path(trial_dir).absolute()
        trial_dir.mkdir(parents=True, exist_ok=True)
        sequence_path = trial_dir / "allen_configuration.json"
        sequence_path.write_text(
            json.dumps(trial_sequence(base_cfg, sampled), indent=2) + "\n"
        )
        inputs = sorted(glob.glob(str(_path(input_root))))
        if not inputs or any(not Path(p).is_file() for p in inputs):
            raise FileNotFoundError(f"No MDF files matched: {input_root}")
        if any("," in p for p in inputs):
            raise ValueError(
                "Allen's comma-separated MDF input cannot contain commas in paths"
            )
        monitoring = trial_dir / "histograms.root"
        arguments = [
            "--sequence",
            str(sequence_path),
            "--mdf",
            ",".join(inputs),
            "-n",
            str(runtime["n_events"]),
            "-t",
            "1",
            "-s",
            "1",
            "-r",
            "1",
            "--events-per-slice",
            str(runtime["events_per_slice"]),
            "--device",
            runtime["device"],
            "--params",
            runtime["params"],
            "--tags",
            runtime["tags"],
            "-m",
            str(runtime["memory_mb"]),
            "--host-memory",
            str(runtime["host_memory_mb"]),
            "--register-monitoring-counters",
            "1",
            "--monitoring-filename",
            str(monitoring),
        ]
        manifest = {
            "allen_run": str(_path(paths["allen_run"])),
            "entrypoint": str(_path(paths["entrypoint"])),
            "arguments": arguments,
            "detector_list": runtime["detector_list"],
            "timeout_seconds": runtime["timeout_seconds"],
            "inputs": inputs,
            "sequence": str(sequence_path),
            "root_output": str(monitoring),
            "hdf5_output": str(trial_dir / "fit_results.h5"),
            "sampled_parameters": sampled,
            # Allen/run resets the environment. Restore only GPU visibility and
            # the requested CUDA paths before starting its Python.
            "gpu_environment": {
                key: os.environ[key]
                for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
                if key in os.environ
            },
            "cuda_bin": runtime["cuda_bin"],
            "cuda_lib": runtime["cuda_lib"],
        }
        config_path = trial_dir / "allen_trial.json"
        config_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (trial_dir / "reconstruction.yaml").write_text(
            yaml.safe_dump(base_cfg, sort_keys=False)
        )
        return config_path

    def reconstruct(self, config_path, trial_dir, root):
        manifest = json.loads(Path(config_path).read_text())
        trial_dir = Path(trial_dir).absolute()
        root_output = Path(manifest["root_output"])
        hdf5_output = Path(manifest["hdf5_output"])
        if root_output.exists() or hdf5_output.exists():
            raise ValueError(
                "Refusing to evaluate stale Allen outputs in an existing trial directory"
            )
        runner = Path(__file__).with_name("allen_runner.py")
        run_logged(
            [
                manifest["allen_run"],
                "bash",
                "--norc",
                "-c",
                _GPU_ENVIRONMENT_SHELL,
                "pvfit-allen",
                manifest["gpu_environment"].get(
                    "CUDA_VISIBLE_DEVICES", "__PVFIT_UNSET__"
                ),
                manifest["gpu_environment"].get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
                manifest["cuda_bin"] or "",
                manifest["cuda_lib"] or "",
                runner,
                config_path,
            ],
            trial_dir,
            trial_dir,
            "Allen reconstruction",
            timeout=manifest["timeout_seconds"],
        )
        if not root_output.is_file():
            raise FileNotFoundError(
                f"Allen succeeded without its monitoring output: {root_output}"
            )
        run_logged(
            [
                sys.executable,
                "-m",
                "pvfit",
                "convert-allen",
                "--input",
                root_output,
                "--output",
                hdf5_output,
                "--allen-config",
                manifest["sequence"],
            ],
            trial_dir,
            root,
            "pvfit convert-allen",
            timeout=manifest["timeout_seconds"],
        )
        if not hdf5_output.is_file():
            raise FileNotFoundError(f"Allen conversion did not produce {hdf5_output}")
        return hdf5_output
