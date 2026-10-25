"""Separate config resolution, trial execution, and evaluation consistency."""

import copy
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
import yaml

from pvfit.optimize.backends import get_backend
from pvfit.optimize.config import (
    check_study_configuration,
    implementation_fingerprint,
    input_fingerprint,
    load_optimization_settings,
)
from pvfit.optimize.constants import AUDIT_SPECS, OBJECTIVE_SPECS, REQUIRED_ANALYSES
from pvfit.optimize.export import export_results
from pvfit.optimize.trial import make_objective


class OptimizationConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.folder = self.root / "settings"
        self.folder.mkdir()
        source = Path(__file__).parents[1] / "configs"
        self.config = yaml.safe_load((source / "optimization.yaml").read_text())
        self.reconstruction = yaml.safe_load(
            (source / "reconstruction.yaml").read_text()
        )
        self.analysis = yaml.safe_load((source / "analysis.yaml").read_text())
        self.path = self.folder / "optimization.yaml"
        self.save()

    def save(self):
        self.path.write_text(yaml.safe_dump(self.config))

    def test_references_are_relative_to_optimization_yaml_and_cli_overrides_win(self):
        settings = load_optimization_settings(
            self.path,
            {
                "n_trials": 2,
                "n_jobs": 1,
                "all_input_files": True,
                "population_size": None,
            },
        )
        self.assertEqual(
            settings["reconstruction_config"], self.folder / "reconstruction.yaml"
        )
        self.assertEqual(settings["analysis_config"], self.folder / "analysis.yaml")
        self.assertEqual(settings["n_trials"], 2)
        self.assertEqual(settings["n_jobs"], 1)
        self.assertTrue(settings["all_input_files"])
        self.assertEqual(
            settings["population_size"], self.config["study"]["population_size"]
        )
        self.assertEqual(settings["backend"], "pvfit")

    def test_backend_must_exist_and_own_the_search_space(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend 'missing'"):
            get_backend("missing")
        self.config["execution"]["backend"] = "missing"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            load_optimization_settings(self.path)
        self.config["execution"]["backend"] = "pvfit"
        del self.config["search_space"]["n_bins_t"]
        self.save()
        with self.assertRaisesRegex(ValueError, "must define exactly"):
            load_optimization_settings(self.path)

    def test_invalid_search_and_execution_settings_fail_before_trials(self):
        original = copy.deepcopy(self.config)
        for section, key, value in (
            ("study", "n_jobs", 0),
            ("execution", "trial_workers", True),
            ("execution", "all_input_files", "false"),
            ("search_space", "n_bins_z", {"type": "int", "low": 1.5, "high": 10}),
            ("search_space", "max_chi2_tukey", {"type": "float", "low": 15, "high": 4}),
        ):
            with self.subTest(section=section, key=key):
                self.config = copy.deepcopy(original)
                self.config[section][key] = value
                self.save()
                with self.assertRaises(ValueError):
                    load_optimization_settings(self.path)

    def test_trial_reads_search_ranges_and_passes_separate_stage_files(self):
        self.analysis["analysis"]["method_name"] = "External Reco"
        self.analysis["analysis"]["purity_threshold"] = 0.95
        original = copy.deepcopy(self.analysis)
        self.config["search_space"]["max_chi2_tukey"].update(low=6.5, high=6.5)
        self.save()
        settings = load_optimization_settings(self.path)
        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize", "minimize"]
        )
        trial = study.ask()
        trial_dir = self.root / "study" / "trial_0"

        def execute(command, **kwargs):
            stage = command[command.index("pvfit") + 1]
            config_path = Path(command[command.index("--config") + 1])
            if stage == "fit":
                self.assertEqual(config_path, trial_dir / "reconstruction.yaml")
                reco = yaml.safe_load(config_path.read_text())
                self.assertEqual(reco["algorithm"]["max_chi2_tukey"], 6.5)
                self.assertNotIn("purity_threshold", reco["algorithm"])
                self.assertNotIn("analyses", reco["workflow"])
                self.assertEqual(reco["paths"]["output_dir"], str(trial_dir))
                return subprocess.CompletedProcess(command, 0)
            self.assertEqual(stage, "analyze")
            self.assertEqual(config_path, trial_dir / "analysis.yaml")
            self.assertEqual(
                Path(command[command.index("--input") + 1]),
                trial_dir / self.reconstruction["paths"]["hdf5_file"],
            )
            analysis = yaml.safe_load(config_path.read_text())
            self.assertEqual(analysis["analysis"]["purity_threshold"], 0.95)
            self.assertEqual(analysis["analysis"]["analyses"], REQUIRED_ANALYSES)
            output = Path(command[command.index("--output") + 1])
            self.assertEqual(output, trial_dir / "analysis")
            output.mkdir()
            for filename, column, _ in OBJECTIVE_SPECS + AUDIT_SPECS:
                filename = filename.replace(
                    "pvfit_track_based", "external_reco_track_based"
                )
                with (output / filename).open("w") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([column])
                    writer.writerow([0.5])
            return subprocess.CompletedProcess(command, 0)

        objective = make_objective(
            get_backend("pvfit"),
            self.reconstruction,
            1,
            "sample.root",
            self.root / "study",
            self.root,
            analysis_cfg=self.analysis,
            search_space=settings["search_space"],
        )
        with patch(
            "pvfit.optimize.backends.base.subprocess.run", side_effect=execute
        ) as run:
            self.assertEqual(objective(trial), (0.5, 0.5, 0.5, 0.5))
        self.assertEqual(run.call_count, 2)
        self.assertAlmostEqual(
            trial.user_attrs["derived_scaled_min_tracks_in_seed"],
            trial.params["min_tracks_in_seed"]
            * (400 / trial.params["n_bins_z"])
            * (1.6 / trial.params["n_bins_t"])
            / (0.5 * 0.025),
        )
        self.assertEqual(self.analysis, original)
        self.assertEqual(self.reconstruction["algorithm"]["max_chi2_tukey"], 12.0)

    def test_export_handles_trials_without_a_sampler_generation(self):
        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize", "minimize"]
        )
        trial = study.ask()
        trial.set_user_attr("generation", None)
        study.tell(trial, [0.8, 0.1, 0.01, 0.02])
        export_results(study, self.root)
        with (self.root / "all_trials.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "COMPLETE")
        self.assertEqual(rows[0]["generation"], "")
        with (self.root / "generation_summary.csv").open() as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])

    def configuration(self):
        sample = self.root / "sample.root"
        if not sample.exists():
            sample.write_bytes(b"root")
        return {
            "backend": "pvfit",
            "analysis": self.analysis,
            "search_space": self.config["search_space"],
            "input_root": "sample.root",
            "inputs": input_fingerprint("sample.root", self.root),
            "implementation": implementation_fingerprint(),
        }

    def test_resume_rejects_changed_evaluation_or_search_configuration(self):
        study = optuna.create_study()
        configuration = self.configuration()
        check_study_configuration(study, configuration)
        check_study_configuration(study, copy.deepcopy(configuration))
        # A study recorded before backends existed still resumes with PVFit.
        study.set_user_attr(
            "task_configuration",
            {key: value for key, value in configuration.items() if key != "backend"},
        )
        check_study_configuration(study, copy.deepcopy(configuration))
        with self.assertRaisesRegex(ValueError, "changed \\(backend\\)"):
            check_study_configuration(study, {**configuration, "backend": "allen"})
        for key in ("analysis", "search_space"):
            with self.subTest(key=key):
                changed = copy.deepcopy(configuration)
                changed[key] = {}
                with self.assertRaisesRegex(ValueError, f"changed \\({key}\\)"):
                    check_study_configuration(study, changed)

    def test_resume_rejects_inputs_regenerated_at_the_same_path(self):
        study = optuna.create_study()
        configuration = self.configuration()
        check_study_configuration(study, configuration)
        trial = study.ask()
        study.tell(trial, 1.0)
        # Same pathname, different content: the recorded metrics no longer
        # describe this file.
        (self.root / "sample.root").write_bytes(b"regenerated")
        with self.assertRaisesRegex(ValueError, "changed \\(inputs\\)"):
            check_study_configuration(study, self.configuration())
        # A glob that now matches an extra file is a changed input too.
        (self.root / "sample.root").write_bytes(b"root")
        pattern = {**self.configuration(), "input_root": "*.root"}
        pattern["inputs"] = input_fingerprint("*.root", self.root)
        study = optuna.create_study()
        check_study_configuration(study, pattern)
        (self.root / "extra.root").write_bytes(b"more")
        pattern["inputs"] = input_fingerprint("*.root", self.root)
        with self.assertRaisesRegex(ValueError, "changed \\(inputs\\)"):
            check_study_configuration(study, pattern)

    def test_resume_rejects_changed_implementation(self):
        study = optuna.create_study()
        configuration = self.configuration()
        check_study_configuration(study, configuration)
        changed = copy.deepcopy(configuration)
        changed["implementation"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed \\(implementation\\)"):
            check_study_configuration(study, changed)
        fingerprint = implementation_fingerprint()
        self.assertEqual(len(fingerprint["source_sha256"]), 64)
        self.assertEqual(fingerprint, implementation_fingerprint())

    def test_study_without_fingerprints_is_not_resumed(self):
        study = optuna.create_study()
        configuration = self.configuration()
        study.set_user_attr(
            "task_configuration",
            {
                key: value
                for key, value in configuration.items()
                if key not in ("inputs", "implementation")
            },
        )
        with self.assertRaisesRegex(ValueError, "predates input and code"):
            check_study_configuration(study, configuration)
        with self.assertRaisesRegex(ValueError, "lacks inputs, implementation"):
            check_study_configuration(
                optuna.create_study(), {"analysis": self.analysis}
            )

    def test_input_fingerprint_needs_a_match(self):
        with self.assertRaises(FileNotFoundError):
            input_fingerprint("missing.root", self.root)

    def test_old_study_with_trials_cannot_silently_reuse_unrecorded_configuration(self):
        study = optuna.create_study()
        trial = study.ask()
        study.tell(trial, 1.0)
        with self.assertRaisesRegex(ValueError, "predates separate configs"):
            check_study_configuration(study, self.configuration())


if __name__ == "__main__":
    unittest.main()
