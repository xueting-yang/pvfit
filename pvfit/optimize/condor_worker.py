"""Run an entire Allen optimization inside one Condor GPU allocation."""

import json
import os
import signal
import sys
import traceback
from pathlib import Path

from ..config_loader import load_config
from .backends import get_backend
from .condor import execution_identity
from .run import run_optimization


def run_worker(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    folder = Path(manifest["submission_dir"])
    status_path = folder / "status.json"

    def status(state, **extra):
        status_path.write_text(
            json.dumps({"state": state, "pid": os.getpid(), **extra}, indent=2) + "\n"
        )

    status("starting")
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or visible in {"-1", "NoDevFiles"}:
            raise RuntimeError(
                "No Condor-assigned CUDA_VISIBLE_DEVICES; refusing GPU execution"
            )
        os.chdir(manifest["working_directory"])
        for name, content in manifest["config_snapshots"].items():
            if (folder / name).read_text() != content:
                raise ValueError(f"Submitted configuration changed: {name}")
        base_cfg = load_config(folder / "allen.yaml")
        identity = execution_identity(
            get_backend("allen"), base_cfg, manifest["input_pattern"], Path.cwd()
        )
        if identity != manifest["expected_identity"]:
            raise ValueError(
                "Code, Allen runtime or inputs changed since submission; prepare a new job"
            )
        status("running", cuda_visible_devices=visible)
        run_optimization(manifest["config"], {"_submission_dir": str(folder)})
        status("complete", cuda_visible_devices=visible)
    except BaseException as exc:
        status(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Condor terminated the worker (signal {signum})")

    signal.signal(signal.SIGTERM, terminate)
    run_worker(sys.argv[1])


if __name__ == "__main__":
    main()
