"""Effective CLI settings and failed stages remain attributable after reruns."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import yaml

from pvfit.cli import main
from pvfit.defaults import bundled_config_dir, default_config_path, export_configs
from pvfit.fit.save_results_to_hdf5 import save_all_events_to_hdf5
from pvfit.provenance import HDF5_SCHEMA_VERSION, StageRecord
from pvfit.reconstruction_parameters import read_parameters
from tests.reco_params import default_params


class ProvenanceTests(unittest.TestCase):
    def test_fit_records_effective_cli_overrides_and_input_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            sample = work / "sample.root"
            sample.write_bytes(b"mock input")
            output = work / "output"
            with (
                patch(
                    "pvfit.cli._run_fit", return_value=str(output / "fit_results.h5")
                ) as fit,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "fit",
                        "--input",
                        str(sample),
                        "--output",
                        str(output),
                        "--max-workers",
                        "2",
                        "--event-number",
                        "3",
                    ]
                )
            self.assertEqual(code, 0)
            path = Path(fit.call_args.args[0]["provenance_path"])
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["status"], "completed")
            cfg = yaml.safe_load(path.with_name("reconstruction.yaml").read_text())
            self.assertEqual(cfg, manifest["configuration"])
            self.assertEqual(cfg["workflow"], {"event_number": 3, "max_workers": 2})
            self.assertEqual(cfg["paths"]["input_root"], str(sample))
            self.assertEqual(manifest["inputs"][0]["size"], sample.stat().st_size)
            self.assertEqual(len(manifest["implementation"]["source_sha256"]), 64)
            self.assertIn("numpy", manifest["runtime"]["dependencies"])

    def test_reruns_preserve_records_including_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            sample = work / "input.h5"
            sample.touch()
            with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                with StageRecord(
                    work, "analyze", {"analysis": {}}, [sample], ["analyze"]
                ) as failed:
                    raise RuntimeError("analysis failed")
            with StageRecord(
                work, "analyze", {"analysis": {}}, [sample], ["analyze"]
            ) as success:
                pass
            self.assertNotEqual(failed.path, success.path)
            self.assertEqual(json.loads(failed.path.read_text())["status"], "failed")
            self.assertEqual(
                json.loads(success.path.read_text())["status"], "completed"
            )

    def test_hdf5_writer_carries_version_and_algorithm_parameters(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            output = Path(directory) / "results.h5"
            params = default_params(max_chi2_tukey=8.0)
            save_all_events_to_hdf5(
                [],
                output,
                {**params, "hit_time_resolution_ns": 0.05},
                provenance_path="manifest.json",
                track_input_attributes={"track_input_configuration_sha256": "0" * 64},
            )
            with h5py.File(output) as source:
                self.assertEqual(source.attrs["schema_version"], HDF5_SCHEMA_VERSION)
                # Every reconstruction parameter is an explicit reco_* attribute
                recorded = read_parameters(source.attrs)
                self.assertEqual(recorded["max_chi2_tukey"], 8.0)
                self.assertEqual(recorded["max_track_seed_chi2"], 10.0)
                self.assertEqual(recorded["hit_time_resolution_ns"], 0.05)
                self.assertEqual(
                    set(recorded),
                    {name for name in params if name not in (
                        "bin_size_z", "bin_size_t", "scaled_min_tracks_in_seed",
                        "is_adaptive_tukey",
                    )} | {"hit_time_resolution_ns"},
                )  # fmt: skip
                self.assertEqual(
                    source.attrs["track_input_configuration_sha256"], "0" * 64
                )
                self.assertEqual(source.attrs["input_hit_time_resolution_ns"], 0.05)
                self.assertEqual(source.attrs["truth_min_tracks"], 4)
                self.assertNotIn("reconstruction_parameters", source.attrs)
            # Incomplete parameters are refused rather than silently omitted
            with self.assertRaisesRegex(KeyError, "max_vertex_rho2"):
                incomplete = {k: v for k, v in params.items() if k != "max_vertex_rho2"}
                save_all_events_to_hdf5([], output, incomplete)
                self.assertEqual(len(source.attrs["source_sha256"]), 64)
                self.assertTrue(
                    source.attrs["provenance_manifest"].endswith("manifest.json")
                )


class DefaultConfigTests(unittest.TestCase):
    def test_templates_work_outside_checkout_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                shipped = default_config_path("reconstruction.yaml")
                self.assertTrue(shipped.is_file())
                self.assertEqual(shipped.parent, bundled_config_dir())
                export_configs("configs")
                local = default_config_path("reconstruction.yaml")
                self.assertEqual(local, Path("configs/reconstruction.yaml"))
                self.assertEqual(local.read_text(), shipped.read_text())
                local.write_text("custom content")
                with self.assertRaises(FileExistsError):
                    export_configs("configs")
                self.assertEqual(local.read_text(), "custom content")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["fit", "--config", "missing.yaml"]), 2)
            finally:
                os.chdir(previous)
