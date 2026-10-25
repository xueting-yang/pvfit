"""Allen parameter mapping and optimizer/runner/converter integration.

The integration test runs an external Python executable producing a small ROOT
fixture; it tests orchestration and evaluation, not GPU reconstruction physics.
"""

import copy
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import optuna
import yaml

from pvfit.allen import parameters_from_configuration
from pvfit.optimize.backends import get_backend
from pvfit.optimize.backends.allen import trial_sequence
from pvfit.optimize.backends.base import run_logged
from pvfit.optimize.config import (
    check_study_configuration,
    load_optimization_settings,
)
from pvfit.optimize.run import run_optimization
from tests.test_convert_allen import CONFIG


class AllenOptimizationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="allen optimizer ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.backend = get_backend("allen")
        self.configuration = copy.deepcopy(CONFIG)
        self.configuration["sequence"]["configured_algorithms"] = [
            ["test", "test", "test"]
        ]
        for name in (
            "pv_beamline_histo",
            "pv_beamline_peak",
            "pv_beamline_merge_vertices",
        ):
            self.configuration[name]["enable_monitor"] = True
        self.sequence = self.root / "sequence.json"
        self.sequence.write_text(json.dumps(self.configuration))
        self.runner = self.root / "run"
        self.runner.write_text(
            f'#!/bin/sh\nexport PATH={shlex.quote(str(Path(sys.executable).parent))}:"$PATH"\nunset CUDA_VISIBLE_DEVICES\nexec "$@"\n'
        )
        self.runner.chmod(0o755)
        self.entrypoint = self.root / "entrypoint.py"
        self.entrypoint.write_text("# test fixture entrypoint\n")
        self.library = self.root / "libAllenAlgorithms.so"
        self.library.write_bytes(b"test-build-1")
        self.input = self.root / "sample.mdf"
        self.input.write_bytes(b"synthetic-MDF-placeholder")
        self.params = self.root / "params"
        self.params.mkdir()
        self.cfg = {
            "paths": {
                "input_mdf": str(self.input),
                "allen_run": str(self.runner),
                "entrypoint": str(self.entrypoint),
                "sequence": str(self.sequence),
                "runtime_files": [str(self.library)],
            },
            "runtime": {
                "n_events": 2,
                "params": str(self.params),
                "detector_list": [],
                "timeout_seconds": 60,
            },
        }
        self.trial_dir = self.root / "trial"

    def test_mapping_updates_both_consumers_and_does_not_double_scale(self):
        sampled = {
            "n_bins_z": 400,
            "n_bins_t": 28,
            "min_tracks_in_seed": 2.0,
            "smear_window_nbins": 3,
            "max_track_seed_chi2": 9.0,
            "max_chi2_tukey": 13.0,
            "min_chi2_dist_merge": 30.0,
            "max_track_beamline_chi2_xy": 8.0,
            "max_track_z0_err": 2.0,
        }
        before = self.sequence.read_bytes()
        sequence = trial_sequence(self.cfg, sampled)
        p = parameters_from_configuration(sequence)
        for name, value in sampled.items():
            self.assertEqual(p[name], value)
        self.assertEqual(sequence["pv_beamline_peak"]["pp_minTracks"], 2.0)
        self.assertEqual(sequence["pv_beamline_histo"]["dz"], 1.0)
        self.assertAlmostEqual(sequence["pv_beamline_peak"]["dt"], 0.05)
        self.assertEqual(sequence["pv_beamline_merge_vertices"]["maxChi2Tukey"], 13.0)
        self.assertEqual(sequence["pv_beamline_fitter"]["maxChi2Tukey"], 13.0)
        self.assertEqual(sequence["sequence"], self.configuration["sequence"])
        self.assertAlmostEqual(
            self.backend.derived(sampled, self.cfg)[
                "derived_scaled_min_tracks_in_seed"
            ],
            8.0,
        )
        self.assertEqual(self.sequence.read_bytes(), before)

    def test_subset_search_and_invalid_parameters(self):
        self.backend.validate_search_space(
            {"max_chi2_tukey": {"type": "float", "low": 4, "high": 15}}
        )
        for space in (
            {"typo": {"type": "float", "low": 1}},
            {"n_bins_z": {"type": "float", "low": 400}},
            {"n_bins_t": {"type": "int", "low": 2}},
        ):
            with self.subTest(space=space), self.assertRaises(ValueError):
                self.backend.validate_search_space(space)
        for values in (
            {"n_bins_t": 2},
            {"max_chi2_tukey": float("nan")},
            {"smear_window_nbins": True},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                trial_sequence(self.cfg, values)

    def test_invalid_runtime_and_incomplete_sequence_fail_early(self):
        for key, value in (
            ("n_events", 0),
            ("timeout_seconds", float("nan")),
            ("detector_list", "TV"),
        ):
            cfg = copy.deepcopy(self.cfg)
            cfg["runtime"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.backend.validate_config(cfg)
        self.configuration["pv_beamline_peak"]["enable_monitor"] = False
        self.sequence.write_text(json.dumps(self.configuration))
        with self.assertRaisesRegex(ValueError, "enable_monitor"):
            self.backend.validate_config(self.cfg)
        self.configuration.pop("sequence")
        self.sequence.write_text(json.dumps(self.configuration))
        with self.assertRaisesRegex(ValueError, "complete executable"):
            self.backend.validate_config(self.cfg)

    def test_suffix_instances_are_supported_and_ambiguous_instances_rejected(self):
        sequence = {
            name + "_test" if name != "sequence" else name: props
            for name, props in self.configuration.items()
        }
        self.sequence.write_text(json.dumps(sequence))
        result = trial_sequence(self.cfg, {"max_chi2_tukey": 11.0})
        self.assertEqual(result["pv_beamline_fitter_test"]["maxChi2Tukey"], 11.0)
        sequence["pv_beamline_fitter_another"] = sequence["pv_beamline_fitter_test"]
        self.sequence.write_text(json.dumps(sequence))
        with self.assertRaisesRegex(ValueError, "2 instances"):
            trial_sequence(self.cfg, {})

    def test_manifest_preserves_run_symlink_and_pins_sorted_mdf_list(self):
        link = self.root / "Allen" / "run"
        link.parent.mkdir()
        link.symlink_to(self.runner)
        self.cfg["paths"]["allen_run"] = str(link)
        second = self.root / "a sample.mdf"
        second.write_bytes(b"second")
        path = self.backend.write_config(
            self.cfg, {}, self.trial_dir, 1, str(self.root / "*.mdf")
        )
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["allen_run"], str(link))
        self.assertEqual(manifest["inputs"], [str(second), str(self.input)])
        argv = manifest["arguments"]
        self.assertEqual(argv[argv.index("--mdf") + 1], f"{second},{self.input}")
        self.assertEqual(
            argv[argv.index("--sequence") + 1],
            str(self.trial_dir / "allen_configuration.json"),
        )
        with self.assertRaisesRegex(ValueError, "trial_workers"):
            self.backend.write_config(self.cfg, {}, self.trial_dir, 2, str(self.input))

    def test_build_changes_invalidate_study_reuse(self):
        self.backend.validate_config(self.cfg)
        fingerprint = self.backend.fingerprint(self.cfg, self.root)
        configuration = {
            "inputs": [],
            "implementation": {},
            "backend_implementation": fingerprint,
        }
        study = optuna.create_study()
        check_study_configuration(study, configuration)
        self.library.write_bytes(b"test-build-2")
        changed = {
            **configuration,
            "backend_implementation": self.backend.fingerprint(self.cfg, self.root),
        }
        with self.assertRaisesRegex(ValueError, "backend_implementation"):
            check_study_configuration(study, changed)

    def test_concurrency_limits_reject_before_trial_start(self):
        source = Path(__file__).parents[1] / "configs/optimization_allen.yaml"
        config = yaml.safe_load(source.read_text())
        path = self.root / "optimization.yaml"
        for section, key, value in (
            ("study", "n_jobs", 2),
            ("execution", "trial_workers", 2),
        ):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            path.write_text(yaml.safe_dump(changed))
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "allows at most"),
            ):
                load_optimization_settings(path)

    def test_missing_or_stale_outputs_cannot_be_scored(self):
        path = self.backend.write_config(
            self.cfg, {}, self.trial_dir, 1, str(self.input)
        )
        with patch("pvfit.optimize.backends.allen.run_logged"):
            with self.assertRaisesRegex(FileNotFoundError, "without its monitoring"):
                self.backend.reconstruct(path, self.trial_dir, self.root)
        (self.trial_dir / "histograms.root").write_bytes(b"old output")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.backend.reconstruct(path, self.trial_dir, self.root)

    def test_logged_failure_and_timeout(self):
        self.trial_dir.mkdir()
        with self.assertRaisesRegex(RuntimeError, "rc=7"):
            run_logged(
                [
                    sys.executable,
                    "-c",
                    "import sys;sys.stderr.write('bad run');sys.exit(7)",
                ],
                self.trial_dir,
                self.root,
                "test",
                timeout=5,
            )
        self.assertIn("bad run", (self.trial_dir / "stderr.log").read_text())
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            run_logged(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                self.trial_dir,
                self.root,
                "test",
                timeout=0.05,
            )
        self.assertEqual((self.trial_dir / "exit_code.txt").read_text(), "timeout")

    def test_external_runner_conversion_analysis_and_duplicate_reuse(self):
        # This executable stands in for Allen: consume the real generated argv
        # and produce real monitoring ROOT trees. Conversion/analysis are real.
        repo = Path(__file__).parents[1].resolve()
        self.entrypoint.write_text(
            "import sys, json\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(repo)!r})\n"
            "from tests.test_convert_allen import write_allen_file\n"
            "argv = sys.argv\n"
            "sequence = json.loads(Path(argv[argv.index('--sequence')+1]).read_text())\n"
            "assert sequence['pv_beamline_fitter']['maxChi2Tukey'] == 13.0\n"
            "write_allen_file(Path(argv[argv.index('--monitoring-filename')+1]), [123], 2)\n"
        )
        self.cfg["runtime"]["params"] = "${PVFIT_TEST_ALLEN_PARAMS}"
        reco_path = self.root / "allen.yaml"
        reco_path.write_text(yaml.safe_dump(self.cfg))
        analysis_path = self.root / "analysis.yaml"
        analysis_path.write_text(yaml.safe_dump({"analysis": {"method_name": "Allen"}}))
        study_dir = self.root / "output" / "allen_test"
        config = {
            "study": {
                "name": "allen_test",
                "output_dir": str(self.root / "output"),
                "n_trials": 2,
                "n_jobs": 1,
                "population_size": 2,
                "sampler_seed": 42,
            },
            "execution": {
                "backend": "allen",
                "reconstruction_config": str(reco_path),
                "analysis_config": str(analysis_path),
                "trial_workers": 1,
                "all_input_files": True,
            },
            "search_space": {
                "max_chi2_tukey": {"type": "float", "low": 13.0, "high": 13.0}
            },
        }
        config_path = self.root / "optimization.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with patch.dict(os.environ, {"PVFIT_TEST_ALLEN_PARAMS": str(self.params)}):
            run_optimization(config_path, {})
        study = optuna.load_study(
            study_name="allen_test",
            storage=f"sqlite:///{study_dir / 'optuna_study.db'}",
        )
        self.assertEqual(
            [t.state for t in study.trials], [optuna.trial.TrialState.COMPLETE] * 2
        )
        self.assertEqual(study.trials[0].values, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(study.trials[1].user_attrs["duplicate_of_trial_number"], 0)
        with h5py.File(study_dir / "trial_0/fit_results.h5") as handle:
            self.assertEqual(handle.attrs["source"], "Allen")
            self.assertEqual(handle.attrs["reco_max_chi2_tukey"], 13.0)
        self.assertFalse((study_dir / "trial_1/histograms.root").exists())
        self.assertTrue((study_dir / "all_trials.csv").is_file())
        # A changed external build at the same pathname cannot reuse the study.
        self.library.write_bytes(b"another-build")
        with self.assertRaisesRegex(ValueError, "backend_implementation"):
            run_optimization(config_path, {"n_trials": 1})


if __name__ == "__main__":
    unittest.main()
