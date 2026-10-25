"""Prepare or submit one shared-filesystem GPU job for a whole Allen study."""

import copy
import json
import re
import shlex
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..config_loader import AnalysisConfig, load_config
from .backends import get_backend
from .backends.allen import _path, _settings
from .config import (
    implementation_fingerprint,
    input_fingerprint,
    load_optimization_settings,
)
from .trial import get_study_dir, resolve_trial_input_root

CONDOR_DEFAULTS = {
    "request_cpus": 4,
    "request_memory_mb": 16384,
    "request_disk_mb": 12288,
    "gpu_model": "L40S",
    "use_os": "el9",
    "job_category": "medium",
    "max_runtime_seconds": 21600,
}


def condor_settings(config):
    given = config.get("condor", {})
    if not isinstance(given, dict) or set(given) - set(CONDOR_DEFAULTS):
        raise ValueError("Unknown condor configuration options")
    settings = {**CONDOR_DEFAULTS, **given}
    for name in (
        "request_cpus",
        "request_memory_mb",
        "request_disk_mb",
        "max_runtime_seconds",
    ):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"condor.{name} must be an integer >= 1")
    for name in ("gpu_model", "use_os", "job_category"):
        value = settings[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"condor.{name} must be a nonempty string")
        _quote(value)
    return settings


def _quote(value):
    value = str(value)
    if any(c in value for c in ("\n", "\r", "\0", "$(")):
        raise ValueError("Condor values cannot contain newlines or submit macros")
    return json.dumps(value)


def _submit_path(value):
    # Submit path directives take literal values (including spaces), unlike
    # ClassAd string expressions such as +UseOS or regexp arguments.
    _quote(value)  # reject newlines and macro expansion
    return str(value)


def execution_identity(backend, base_cfg, input_pattern, root):
    return {
        "implementation": implementation_fingerprint(),
        "backend_implementation": backend.fingerprint(base_cfg, root),
        "inputs": input_fingerprint(input_pattern, root),
    }


def write_submit_file(folder, settings):
    """Exactly one GPU allocation, with all trials inside its worker."""
    lines = [
        "universe = vanilla",
        f"executable = {_submit_path(folder / 'job.sh')}",
        f"initialdir = {_submit_path(folder)}",
        f"output = {_submit_path(folder / 'condor.out')}",
        f"error = {_submit_path(folder / 'condor.err')}",
        f"log = {_submit_path(folder / 'condor.log')}",
        f"request_cpus = {settings['request_cpus']}",
        "request_gpus = 1",
        f"request_memory = {settings['request_memory_mb']} MB",
        f"request_disk = {settings['request_disk_mb']} MB",
        f"requirements = regexp({_quote(settings['gpu_model'])}, TARGET.GPUs_DeviceName)",
        "getenv = False",
        "should_transfer_files = NO",
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        f"periodic_hold = (JobStatus == 2) && (time() - JobCurrentStartDate > {settings['max_runtime_seconds']})",
        "kill_sig = SIGTERM",
        "kill_sig_timeout = 30",
        f"+UseOS = {_quote(settings['use_os'])}",
        f"+JobCategory = {_quote(settings['job_category'])}",
        '+PVFitPurpose = "allen_optimization"',
        f"+PVFitSubmissionId = {_quote(folder.name)}",
        f"+JobBatchName = {_quote(folder.parent.parent.name)}",
        "queue 1",
    ]
    path = folder / "job.sub"
    path.write_text("\n".join(lines) + "\n")
    return path


def prepare_submission(config_path, overrides, *, submit=False):
    """Snapshot configs, validate with condor_submit, optionally enqueue one job.

    The software, input data and result directories must be visible from worker
    nodes at the same paths. No GPU reconstruction runs on the submission host.
    """
    if overrides.get("export_only"):
        raise ValueError("--export-only cannot be combined with Condor submission")
    settings = load_optimization_settings(config_path, overrides)
    if settings["backend"] != "allen":
        raise ValueError("Condor GPU submission currently requires backend: allen")
    backend = get_backend("allen")
    resources = condor_settings(load_config(config_path))
    root = Path.cwd()
    base_cfg = copy.deepcopy(load_config(settings["reconstruction_config"]))
    paths, runtime = _settings(base_cfg)
    # Resolve submission-host path variables now. Only PARAMFILESROOT (params)
    # remains a stack-environment variable, resolved by Allen's own Python.
    base_cfg["paths"] = {
        name: [str(_path(p)) for p in value]
        if name == "runtime_files"
        else str(_path(value))
        for name, value in paths.items()
    }
    base_cfg["runtime"] = runtime
    if runtime["device"] != "0":
        raise ValueError("Condor allocates one visible GPU; set runtime.device to '0'")
    backend.validate_config(base_cfg)
    analysis_cfg = load_config(settings["analysis_config"])
    AnalysisConfig.from_mapping(analysis_cfg)
    input_pattern = resolve_trial_input_root(
        backend.input_pattern(base_cfg), not settings["all_input_files"], root
    )
    expected = execution_identity(backend, base_cfg, input_pattern, root)
    study_dir = get_study_dir(settings["study_name"], settings["output_dir"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = study_dir / "submissions" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    (folder / "allen.yaml").write_text(yaml.safe_dump(base_cfg, sort_keys=False))
    (folder / "analysis.yaml").write_text(yaml.safe_dump(analysis_cfg, sort_keys=False))
    config = {
        "study": {
            "name": settings["study_name"],
            "output_dir": str(settings["output_dir"]),
            **{
                key: settings[key]
                for key in ("n_trials", "n_jobs", "population_size", "sampler_seed")
            },
        },
        "execution": {
            "backend": "allen",
            "reconstruction_config": "allen.yaml",
            "analysis_config": "analysis.yaml",
            "all_input_files": settings["all_input_files"],
        },
        "search_space": settings["search_space"],
    }
    snapshot = folder / "optimization.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "config": str(snapshot),
        "working_directory": str(root),
        "input_pattern": input_pattern,
        "expected_identity": expected,
        "study_dir": str(study_dir),
        "submission_dir": str(folder),
        "resources": resources,
        "config_snapshots": {
            name: (folder / name).read_text()
            for name in ("optimization.yaml", "allen.yaml", "analysis.yaml")
        },
    }
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    package_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "pvfit.optimize.condor_worker", str(manifest_path)]
    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "export PATH=/usr/bin:/bin",
            "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1",
            'export MPLCONFIGDIR="${_CONDOR_SCRATCH_DIR:-/tmp}/pvfit-mpl"',
            f"export PYTHONPATH={shlex.quote(str(package_root))}",
            f"cd {shlex.quote(str(root))}",
            "exec " + shlex.join(command),
            "",
        ]
    )
    job_script = folder / "job.sh"
    job_script.write_text(script)
    job_script.chmod(0o755)
    submit_file = write_submit_file(folder, resources)
    commands = ["condor_submit", "-maxjobs", "1", "-single-cluster"]
    result = subprocess.run(
        [*commands, "-dry-run", str(folder / "job.ad"), str(submit_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (folder / "prepare.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"Condor validation failed; see {folder / 'prepare.log'}\n{result.stderr[-800:]}"
        )
    receipt = {"state": "prepared", "submission_dir": str(folder)}
    receipt_path = folder / "submission.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if submit:
        # Never silently retry a submission whose response was lost: retain an
        # explicit in-flight record for the user to check against the scheduler.
        receipt["state"] = "submitting"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        try:
            result = subprocess.run(
                [*commands, "-terse", str(submit_file)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            receipt.update(state="submission_unknown", error=str(exc))
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            raise RuntimeError(
                f"Submission response timed out. Check condor_q for PVFitSubmissionId "
                f"{folder.name} before submitting again; receipt: {receipt_path}"
            ) from exc
        receipt.update(
            stdout=result.stdout, stderr=result.stderr, returncode=result.returncode
        )
        receipt["state"] = (
            "submitted" if result.returncode == 0 else "submission_failed"
        )
        job_id = re.search(r"\b\d+\.\d+\b", result.stdout)
        if result.returncode == 0 and job_id:
            receipt["job_id"] = job_id.group(0)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(
                f"condor_submit failed; see {receipt_path}\n{result.stderr[-800:]}"
            )
        print(f"Submitted one GPU job for the whole study: {result.stdout.strip()}")
    else:
        print("Prepared and validated one GPU job; no job submitted.")
    print(f"Submission files and logs: {folder}")
    return folder
