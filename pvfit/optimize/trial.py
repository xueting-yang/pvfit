"""Trial sampling, execution, metric parsing, and metadata.

A trial hands the sampled parameters to a backend (see :mod:`.backends`),
which writes the reconstruction configuration and produces an HDF5 file. The
trial then runs ``pvfit analyze`` on that file and reads the objectives from
the analysis CSVs, so every backend is scored identically.
"""

import copy
import csv
import glob
import json
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import optuna
import yaml

from ..config_loader import AnalysisConfig
from .backends import run_logged
from .constants import AUDIT_SPECS, OBJECTIVE_SPECS, REQUIRED_ANALYSES


def resolve_trial_input_root(
    input_pattern: str, use_first_input_file: bool, root: Path
) -> str:
    """Resolve the backend input pattern (PVFit ROOT or Allen MDF) for trials.

    By default we use the first matching input file for faster exploratory scans.
    Pass --all-input-files to keep the full baseline pattern.
    """
    if not use_first_input_file:
        return input_pattern

    matches = sorted(glob.glob(str(root / input_pattern)))
    if not matches:
        raise FileNotFoundError(f"No input files matched pattern: {input_pattern}")

    first_match = Path(matches[0]).resolve()
    try:
        return str(first_match.relative_to(root))
    except ValueError:
        return str(first_match)


def sanitize_study_name(study_name: str) -> str:
    """Convert study_name into a filesystem-safe directory name."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", study_name).strip("._")
    return sanitized or "study"


def get_study_dir(study_name: str, optuna_root: Path) -> Path:
    return optuna_root / sanitize_study_name(study_name)


def write_failure_metadata(
    trial_dir: Path,
    trial_number: int,
    stage: str,
    exc: Exception,
    sampled: dict,
    derived: dict,
):
    """Persist structured failure details for a failed trial."""
    failure = {
        "trial_number": trial_number,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "sampled_params": sampled,
        **derived,
        "trial_output_dir": str(trial_dir),
        "traceback": traceback.format_exc(),
    }
    (trial_dir / "failure.json").write_text(json.dumps(failure, indent=2))


def param_signature(params: dict) -> str:
    """Create a stable signature for exact-parameter duplicate detection."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def generation_attr_key(study: optuna.Study) -> str:
    return f"{study.sampler.__class__.__name__}:generation"


def get_live_trial_generation(trial: optuna.Trial) -> int | None:
    frozen = trial.study._storage.get_trial(trial._trial_id)
    return frozen.system_attrs.get(generation_attr_key(trial.study))


def find_duplicate_completed_trial(
    study: optuna.Study,
    sampled: dict,
) -> optuna.trial.FrozenTrial | None:
    """Return a completed trial with identical params, if any."""
    target_sig = param_signature(sampled)
    for frozen in study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
    ):
        if param_signature(frozen.params) == target_sig:
            return frozen
    return None


def persist_metrics(
    trial_dir: Path,
    *,
    trial_number: int,
    sampled: dict,
    derived: dict,
    obj_values: dict[str, float],
    audit_values: dict[str, float],
    input_root: str,
    generation: int | None,
    param_sig: str,
    reused_from_trial: int | None,
):
    metrics = {
        "trial_number": trial_number,
        "sampled_params": sampled,
        **derived,
        "objectives": obj_values,
        "audit_metrics": audit_values,
        "trial_output_dir": str(trial_dir),
        "trial_input_root": input_root,
        "generation": generation,
        "param_signature": param_sig,
        "reused_from_trial": reused_from_trial,
    }
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def write_reuse_artifacts(
    trial_dir: Path, duplicate_trial: optuna.trial.FrozenTrial
) -> None:
    """Leave breadcrumbs when a trial reuses an already-computed result."""
    (trial_dir / "stdout.log").write_text(
        f"Reused results from completed trial {duplicate_trial.number}.\n"
    )
    (trial_dir / "stderr.log").write_text("")
    (trial_dir / "exit_code.txt").write_text("reused")


# ── Parameter sampling ───────────────────────────────────────────────────────
def sample_params(trial: optuna.Trial, search_space: dict) -> dict:
    sampled = {}
    for name, spec in search_space.items():
        suggest = trial.suggest_int if spec["type"] == "int" else trial.suggest_float
        kwargs = {key: value for key, value in spec.items() if key != "type"}
        sampled[name] = suggest(name, **kwargs)
    return sampled


def backend_analysis_config(analysis_cfg, backend):
    """Share evaluation cuts while letting the backend name its own outputs."""
    result = copy.deepcopy(analysis_cfg)
    if backend.analysis_method_name is not None:
        result.setdefault("analysis", {})["method_name"] = backend.analysis_method_name
    return result


def write_trial_analysis_config(analysis_cfg: dict, trial_dir: Path) -> Path:
    """Keep evaluation thresholds fixed while varying reconstruction parameters."""
    settings = replace(
        AnalysisConfig.from_mapping(analysis_cfg), analyses=REQUIRED_ANALYSES
    )
    path = trial_dir / "analysis.yaml"
    path.write_text(yaml.safe_dump(settings.to_mapping(), sort_keys=False))
    return path


# ── CSV reader ───────────────────────────────────────────────────────────────
def read_csv_value(csv_path: Path, column: str) -> float:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return float(row[column])


def read_metrics(analysis_dir: Path, analysis_cfg: dict, specs, kind: str) -> dict:
    """Read one value per spec from the analysis CSVs of a finished trial."""
    label = AnalysisConfig.from_mapping(analysis_cfg).method_name
    method_slug = f"{label}_Track_Based".lower().replace(" ", "_")
    values = {}
    for fname, col, export_col in specs:
        csv_path = analysis_dir / fname.replace("pvfit_track_based", method_slug)
        if not csv_path.exists():
            raise RuntimeError(f"missing {kind} output {csv_path}")
        values[export_col] = read_csv_value(csv_path, col)
    return values


# ── Objective function factory ───────────────────────────────────────────────
def make_objective(
    backend,
    base_cfg: dict,
    trial_workers: int,
    input_root: str,
    study_dir: Path,
    root: Path,
    *,
    analysis_cfg: dict,
    search_space: dict,
):
    analysis_cfg = backend_analysis_config(analysis_cfg, backend)

    def objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        sampled = sample_params(trial, search_space)
        trial_dir = study_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        for name in ("stdout.log", "stderr.log"):
            (trial_dir / name).write_text("")
        derived = {}
        try:
            config_path = backend.write_config(
                base_cfg, sampled, trial_dir, trial_workers, input_root
            )
            analysis_config_path = write_trial_analysis_config(analysis_cfg, trial_dir)
            derived = backend.derived(sampled, base_cfg)
        except (RuntimeError, OSError, KeyError, ValueError) as exc:
            write_failure_metadata(
                trial_dir, trial.number, "configuration", exc, sampled, derived
            )
            raise
        generation = get_live_trial_generation(trial)
        param_sig = param_signature(sampled)
        for key, value in derived.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("trial_output_dir", str(trial_dir))
        trial.set_user_attr("trial_input_root", input_root)
        trial.set_user_attr("generation", generation)
        trial.set_user_attr("param_signature", param_sig)

        try:
            duplicate_trial = find_duplicate_completed_trial(trial.study, sampled)
            if duplicate_trial is not None:
                trial.set_user_attr("duplicate_of_trial_number", duplicate_trial.number)

                obj_values = {
                    export_col: val
                    for (_, _, export_col), val in zip(
                        OBJECTIVE_SPECS, duplicate_trial.values, strict=True
                    )
                }
                audit_values = {
                    export_col: duplicate_trial.user_attrs.get(export_col, "")
                    for _, _, export_col in AUDIT_SPECS
                }
                for key, value in audit_values.items():
                    if value != "":
                        trial.set_user_attr(key, value)

                write_reuse_artifacts(trial_dir, duplicate_trial)
                persist_metrics(
                    trial_dir,
                    trial_number=trial.number,
                    sampled=sampled,
                    derived=derived,
                    obj_values=obj_values,
                    audit_values=audit_values,
                    input_root=input_root,
                    generation=generation,
                    param_sig=param_sig,
                    reused_from_trial=duplicate_trial.number,
                )

                print(
                    f"[Trial {trial.number}] reused Trial {duplicate_trial.number} "
                    f"(generation={generation})"
                )
                return tuple(duplicate_trial.values)

            # ── Reconstruct, then evaluate with the shared analysis ───────────
            hdf5_path = backend.reconstruct(config_path, trial_dir, root)
            analysis_dir = trial_dir / "analysis"
            run_logged(
                [
                    sys.executable,
                    "-m",
                    "pvfit",
                    "analyze",
                    "--input",
                    hdf5_path,
                    "--config",
                    analysis_config_path,
                    "--output",
                    analysis_dir,
                ],
                trial_dir,
                root,
                "pvfit analyze",
            )

            obj_values = read_metrics(
                analysis_dir, analysis_cfg, OBJECTIVE_SPECS, "objective"
            )
            audit_values = read_metrics(
                analysis_dir, analysis_cfg, AUDIT_SPECS, "audit"
            )

            # Store audit metrics as user attrs so they survive to export
            for k, v in audit_values.items():
                trial.set_user_attr(k, v)

            persist_metrics(
                trial_dir,
                trial_number=trial.number,
                sampled=sampled,
                derived=derived,
                obj_values=obj_values,
                audit_values=audit_values,
                input_root=input_root,
                generation=generation,
                param_sig=param_sig,
                reused_from_trial=None,
            )

            eff = obj_values["efficiency_true_overall"]
            fake = obj_values["fake_rate_rec_overall"]
            merge = obj_values["merge_rate_true_overall"]
            split = obj_values["split_rate_rec_overall"]

            derived_text = "".join(
                f"  {key.removeprefix('derived_')}={value:.4f}"
                for key, value in derived.items()
            )
            print(
                f"[Trial {trial.number}] "
                f"eff={eff:.4f}  fake={fake:.4f}  "
                f"merge={merge:.4f}  split={split:.4f}{derived_text}"
            )
            return eff, fake, merge, split
        except (
            RuntimeError,
            FileNotFoundError,
            KeyError,
            ValueError,
            StopIteration,
        ) as exc:
            write_failure_metadata(
                trial_dir=trial_dir,
                trial_number=trial.number,
                stage="objective",
                exc=exc,
                sampled=sampled,
                derived=derived,
            )
            print(f"[Trial {trial.number}] FAILED: {exc}")
            raise

    return objective
