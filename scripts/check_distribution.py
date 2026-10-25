"""Check sdist/wheel contents and run an installed wheel outside the checkout.

Run after `uv build`; use --offline to restrict dependency installs to uv's cache.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--unlocked",
        action="store_true",
        help="Resolve current compatible dependencies instead of using uv.lock",
    )
    args = parser.parse_args()
    wheels = list(args.dist.glob("*.whl"))
    sources = list(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit(
            "Expected exactly one wheel and one sdist; use a fresh build directory"
        )
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        for source in (ROOT / "configs").glob("*.yaml"):
            assert archive.read(f"pvfit/_defaults/{source.name}") == source.read_bytes()
        assert not any(
            name.startswith(("dev/", "tests/")) for name in archive.namelist()
        ), "Development validation and tests must not be installed in the wheel"
        assert "pvfit/toy/profile.json" in archive.namelist()
        assert "pvfit/analysis/track_pull_config.py" in archive.namelist()
        assert not any(
            name.startswith("pvfit/analysis/analyze_") for name in archive.namelist()
        ), "Retired serial analyzers must not be included in the wheel"
    with tarfile.open(sources[0]) as archive:
        members = {name.split("/", 1)[-1] for name in archive.getnames()}
        for name in (
            "Snakefile",
            "configs/reconstruction.yaml",
            "docs/usage.md",
            "docs/analysis.md",
            "CHANGELOG.md",
            "tests/data/toy_regression_input.json",
        ):
            assert name in members, f"Missing sdist member: {name}"
        assert not any(
            name == "dev"
            or name.startswith(("dev/", "tests/data/analysis_serial/"))
            or name
            in {
                "tests/analysis_baseline.py",
                "tests/analysis_comparison.py",
                "tests/test_parallel_analysis.py",
            }
            for name in members
        ), "Development analysis validation must not be included in the sdist"
    with tempfile.TemporaryDirectory(prefix="pvfit-installed-") as directory:
        work = Path(directory)
        env_dir = work / "venv"
        offline = ["--offline"] if args.offline else []
        subprocess.run(
            ["uv", "venv", *offline, "--python", sys.executable, str(env_dir)],
            check=True,
        )
        python = env_dir / "bin" / "python"
        requirements = work / "requirements.txt"
        subprocess.run(
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                str(requirements),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                *offline,
                "--python",
                str(python),
                *([] if args.unlocked else ["-r", str(requirements)]),
                str(wheel),
            ],
            check=True,
        )
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(work / "mpl"),
            "OPENBLAS_NUM_THREADS": "1",
        }

        def invoke(*command):
            subprocess.run([str(python), "-I", *command], cwd=work, env=env, check=True)

        invoke(
            "-c",
            "from pathlib import Path; import pvfit; assert Path(pvfit.__file__).is_relative_to(Path.cwd() / 'venv'); print(pvfit.__file__)",
        )
        invoke("-m", "pvfit", "toy", "--events", "2", "--seed", "18")
        # No checkout configs or PYTHONPATH: these defaults must come from the wheel.
        invoke(
            "-m",
            "pvfit",
            "run",
            "--max-workers",
            "1",
            "--analysis-workers",
            "2",
            "--analysis-batches",
            "2",
            "--analyses",
            "efficiency_track",
            "efficiency_geo",
            "track_pull",
        )
        output = work / "output" / "baseline"
        assert (output / "analysis" / "analysis.done").is_file()
        for base in (output, output / "analysis"):
            manifests = list(base.glob("provenance/*/manifest.json"))
            assert len(manifests) == 1
            assert json.loads(manifests[0].read_text())["status"] == "completed"
        invoke("-m", "pvfit", "init-config")
        for name in ("reconstruction.yaml", "analysis.yaml", "optimization.yaml"):
            assert (work / "configs" / name).is_file()
    print("Installed wheel smoke test passed; source and wheel resources are complete.")


if __name__ == "__main__":
    main()
