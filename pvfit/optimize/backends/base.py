"""Interface between the optimizer and one reconstruction program.

A trial samples parameters; the backend turns them into that program's
configuration, runs it, and returns an HDF5 file in the PVFit event layout.
The optimizer then evaluates that file with ``pvfit analyze``, so every backend
is scored by the same analysis code.
"""

import os
import signal
import subprocess
from pathlib import Path


class Backend:
    """Base class; subclasses implement the methods that raise below."""

    name = ""
    analysis_method_name = None
    # None: any number of trials may run at once. A GPU backend sets this to
    # the number of trials its device can serve concurrently.
    max_parallel_trials = None
    max_trial_workers = None

    def validate_config(self, base_cfg):
        """Fail before any trial when the base configuration is unusable."""

    def fingerprint(self, base_cfg, root):
        """External runtime identity, in addition to the common PVFit source digest."""
        return {}

    def input_pattern(self, base_cfg):
        """Glob pattern of the input files named in the base configuration."""
        raise NotImplementedError

    def validate_search_space(self, search_space):
        """Reject parameter names or types this program cannot vary."""

    def derived(self, sampled, base_cfg):
        """Values computed from the sampled parameters.

        Stored with the trial and exported; keys must start with ``derived_``.
        """
        return {}

    def write_config(self, base_cfg, sampled, trial_dir, trial_workers, input_root):
        """Write the program's configuration for this trial; return its path."""
        raise NotImplementedError

    def reconstruct(self, config_path, trial_dir, root):
        """Run the reconstruction; return the produced HDF5 path."""
        raise NotImplementedError


def run_logged(command, trial_dir, root, stage, *, timeout=None):
    """Run one stage of a trial, appending its output to the trial logs.

    Raises RuntimeError with the tail of stderr when the stage fails.
    """
    command = [str(part) for part in command]
    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    with open(stdout_path, "a") as stdout_f, open(stderr_path, "a") as stderr_f:
        stdout_f.write(f"=== {stage}: {' '.join(command)}\n")
        stdout_f.flush()
        if timeout is None:
            result = subprocess.run(
                command, stdout=stdout_f, stderr=stderr_f, text=True, cwd=root
            )
        else:
            # Allen can launch child processes. Kill the whole trial process
            # group on timeout/interruption, not just its wrapper.
            with subprocess.Popen(
                command,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                cwd=root,
                start_new_session=True,
            ) as process:
                try:
                    process.wait(timeout=timeout)
                except BaseException as exc:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    if isinstance(exc, subprocess.TimeoutExpired):
                        (trial_dir / "exit_code.txt").write_text("timeout")
                        raise RuntimeError(
                            f"{stage} exceeded {timeout} seconds"
                        ) from exc
                    raise
                result = subprocess.CompletedProcess(command, process.returncode)
    (trial_dir / "exit_code.txt").write_text(str(result.returncode))
    if result.returncode != 0:
        tail_text = stderr_path.read_text(errors="replace")[-800:]
        raise RuntimeError(f"{stage} exited with rc={result.returncode}\n{tail_text}")
    return Path(trial_dir)
