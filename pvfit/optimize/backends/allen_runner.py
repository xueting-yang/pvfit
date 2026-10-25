"""Executed by Allen/run's Python, not the PVFit environment.

Only the standard library and Allen's own Configurables are imported here.
Keeping this as a script lets a PVFit virtualenv drive a different stack Python.
"""

import json
import os
import runpy
import sys
from pathlib import Path


def main():
    manifest_path = Path(sys.argv[1])
    manifest = json.loads(manifest_path.read_text())
    arguments = [os.path.expandvars(value) for value in manifest["arguments"]]
    if any("$" in value for value in arguments):
        raise ValueError("Unresolved environment variable in Allen arguments")
    params = Path(arguments[arguments.index("--params") + 1])
    if not params.is_dir():
        raise FileNotFoundError(f"Allen parameter directory not found: {params}")
    if manifest["detector_list"]:
        from Configurables import LHCb__Det__LbDD4hep__DD4hepSvc as DD4hepSvc

        DD4hepSvc().DetectorList = manifest["detector_list"]
    command = [manifest["entrypoint"], *arguments]
    manifest_path.with_name("allen_command.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "argv": command,
                "gpu_environment": manifest.get("gpu_environment", {}),
            },
            indent=2,
        )
        + "\n"
    )
    sys.argv = command
    runpy.run_path(manifest["entrypoint"], run_name="__main__")


if __name__ == "__main__":
    main()
