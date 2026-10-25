"""Read the optimizer's own settings and resolve references to stage configs."""

import copy
import glob
import math
from pathlib import Path

from ..config_loader import load_config
from ..provenance import implementation_fingerprint as implementation_fingerprint
from .backends import get_backend

# What a study's recorded configuration must contain before trials are reused.
FINGERPRINT_KEYS = ("inputs", "implementation")


def input_fingerprint(input_pattern, root):
    """Identity of the input files ``input_pattern`` resolves to right now.

    The path string alone does not tell whether a file was regenerated in
    place or whether a glob picked up new files, so size and modification time
    of every matched file are recorded too.
    """
    matches = sorted(glob.glob(str(Path(root) / Path(input_pattern).expanduser())))
    if not matches:
        raise FileNotFoundError(f"No input files matched: {input_pattern}")
    files = []
    for match in matches:
        path = Path(match).resolve()
        stat = path.stat()
        try:
            shown = str(path.relative_to(Path(root).resolve()))
        except ValueError:
            shown = str(path)
        files.append(
            {"path": shown, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    return files


def validate_search_space(space, backend):
    """Check the generic shape of every entry, then the backend's own rules."""
    if not isinstance(space, dict) or not space:
        raise ValueError("search_space must be a nonempty mapping")
    for name, spec in space.items():
        if not isinstance(spec, dict) or spec.get("type") not in ("int", "float"):
            raise ValueError(f"search_space.{name}.type must be int or float")
        if set(spec) - {"type", "low", "high", "step"}:
            raise ValueError(f"Unknown search-space options for {name}")
        for key in ("low", "high", "step"):
            value = spec.get(key, 1 if key == "step" else None)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"search_space.{name}.{key} must be a finite number")
            if spec["type"] == "int" and not isinstance(value, int):
                raise ValueError(f"search_space.{name}.{key} must be an integer")
        if spec["low"] <= 0 or spec["high"] < spec["low"] or spec.get("step", 1) <= 0:
            raise ValueError(
                f"search_space.{name} requires 0 < low <= high and step > 0"
            )
    backend.validate_search_space(space)


def load_optimization_settings(config_path, overrides=None):
    """CLI overrides YAML; referenced YAML paths are relative to config_path."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    for name in ("study", "execution"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Optimization configuration requires a '{name}' mapping")
    study, execution = config["study"], config["execution"]
    settings = {
        name: study[name]
        for name in (
            "output_dir",
            "n_trials",
            "n_jobs",
            "population_size",
            "sampler_seed",
        )
    }
    settings["study_name"] = study["name"]
    settings["trial_workers"] = execution.get("trial_workers", 1)
    settings["all_input_files"] = execution["all_input_files"]
    settings["backend"] = execution.get("backend", "pvfit")
    for name, value in (overrides or {}).items():
        if name in settings and value is not None:
            settings[name] = value
    for name, minimum in (
        ("n_trials", 0 if (overrides or {}).get("export_only") else 1),
        ("n_jobs", 1),
        ("population_size", 2),
        ("sampler_seed", 0),
        ("trial_workers", 1),
    ):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(settings["all_input_files"], bool):
        raise ValueError("execution.all_input_files must be true or false")
    for name in ("study_name", "output_dir", "backend"):
        if (
            not isinstance(settings[name], (str, Path))
            or not str(settings[name]).strip()
        ):
            raise ValueError(f"{name} must be nonempty")
    settings["output_dir"] = Path(settings["output_dir"]).expanduser().resolve()
    for name in ("reconstruction_config", "analysis_config"):
        value = execution[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"execution.{name} must be a nonempty path")
        reference = Path(value).expanduser()
        settings[name] = (path.parent / reference).resolve()
    backend = get_backend(settings["backend"])
    limit = backend.max_parallel_trials
    if limit is not None and settings["n_jobs"] > limit:
        raise ValueError(
            f"backend '{backend.name}' allows at most {limit} parallel trial(s); "
            f"n_jobs is {settings['n_jobs']}"
        )
    worker_limit = backend.max_trial_workers
    if worker_limit is not None and settings["trial_workers"] > worker_limit:
        raise ValueError(
            f"backend '{backend.name}' allows at most {worker_limit} trial_workers"
        )
    validate_search_space(config.get("search_space"), backend)
    settings["search_space"] = config["search_space"]
    return settings


def check_study_configuration(study, configuration):
    """Do not reuse metrics obtained with a different input or evaluation setup.

    ``configuration`` must carry the input file identities and the pvfit
    implementation fingerprint (see :func:`input_fingerprint` and
    :func:`implementation_fingerprint`); a study recorded without them cannot
    prove its trials were scored on the same data and code, so it is not
    resumed.
    """
    missing = [key for key in FINGERPRINT_KEYS if key not in configuration]
    if missing:
        raise ValueError(f"task configuration lacks {', '.join(missing)}")
    previous = study.user_attrs.get("task_configuration")
    if previous is None:
        if study.trials:
            raise ValueError(
                "This study predates separate configs; use a new study name"
            )
        study.set_user_attr("task_configuration", copy.deepcopy(configuration))
        return
    # Studies recorded before backends existed were all run with PVFit.
    previous = {"backend": "pvfit", **previous}
    if any(key not in previous for key in FINGERPRINT_KEYS):
        raise ValueError(
            "This study predates input and code fingerprinting, so its trials "
            "cannot be checked against the current data; use a new study name"
        )
    changed = sorted(
        key
        for key in set(previous) | set(configuration)
        if previous.get(key) != configuration.get(key)
    )
    if changed:
        raise ValueError(
            f"Study configuration changed ({', '.join(changed)}); use a new study name"
        )
