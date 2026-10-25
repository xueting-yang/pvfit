"""Whole-study Condor submission, shared analysis and restart behaviour."""

import copy
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
import yaml

from pvfit.cli import main
from pvfit.defaults import CONFIG_NAMES, export_configs
from pvfit.optimize.condor import prepare_submission
from pvfit.optimize.condor_worker import run_worker
from pvfit.optimize.run import study_lock, submission_trials_remaining
from pvfit.optimize.trial import backend_analysis_config
from tests import test_allen_optimization as fixtures


class CondorOptimizationTests(unittest.TestCase):
    def setUp(self):
        fixtures.AllenOptimizationTests.setUp(self)
        self.reco = self.root / "allen.yaml"
        self.reco.write_text(yaml.safe_dump(self.cfg))
        self.analysis = self.root / "analysis.yaml"
        self.analysis_cfg = {
            "analysis": {
                "method_name": "PVfit",
                "purity_threshold": 0.8,
                "recall_threshold": 0.7,
                "nsigma": 5,
            }
        }
        self.analysis.write_text(yaml.safe_dump(self.analysis_cfg))
        self.config = {
            "study": {
                "name": "condor_test",
                "output_dir": str(self.root / "output"),
                "n_trials": 2,
                "n_jobs": 1,
                "population_size": 2,
                "sampler_seed": 42,
            },
            "execution": {
                "backend": "allen",
                "reconstruction_config": str(self.reco),
                "analysis_config": str(self.analysis),
                "all_input_files": True,
            },
            "search_space": {
                "max_chi2_tukey": {"type": "float", "low": 13.0, "high": 13.0}
            },
        }
        self.path = self.root / "optimization.yaml"
        self.path.write_text(yaml.safe_dump(self.config))
        self.previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.previous_cwd)

    def prepare_mock(self, submit=False):
        with patch(
            "pvfit.optimize.condor.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "12345.0 - 12345.0\n", ""),
        ) as run:
            folder = prepare_submission(self.path, {}, submit=submit)
        return folder, run.call_args_list

    def test_shared_evaluation_only_changes_backend_label(self):
        original = copy.deepcopy(self.analysis_cfg)
        result = backend_analysis_config(self.analysis_cfg, self.backend)
        self.assertEqual(result["analysis"]["method_name"], "Allen")
        self.assertEqual(result["analysis"]["purity_threshold"], 0.8)
        self.assertEqual(self.analysis_cfg, original)
        templates = self.root / "templates"
        export_configs(templates)
        self.assertIn("allen.yaml", CONFIG_NAMES)
        self.assertIn("optimization_allen.yaml", CONFIG_NAMES)
        self.assertFalse((templates / "analysis_allen.yaml").exists())
        config = yaml.safe_load((templates / "optimization_allen.yaml").read_text())
        self.assertEqual(config["execution"]["analysis_config"], "analysis.yaml")
        self.assertNotIn("trial_workers", config["execution"])

    def test_one_submit_runs_the_whole_study_and_saves_snapshots(self):
        folder, calls = self.prepare_mock(submit=True)
        self.assertEqual(len(calls), 2)
        self.assertIn("-dry-run", calls[0].args[0])
        self.assertIn("-terse", calls[1].args[0])
        submit = (folder / "job.sub").read_text()
        self.assertIn("request_gpus = 1\n", submit)
        self.assertEqual(submit.count("queue 1"), 1)
        self.assertIn("should_transfer_files = NO", submit)
        self.assertIn('regexp("L40S", TARGET.GPUs_DeviceName)', submit)
        self.assertIn("getenv = False", submit)
        self.assertIn('+JobCategory = "medium"', submit)
        self.assertIn("pvfit.optimize.condor_worker", (folder / "job.sh").read_text())
        saved = yaml.safe_load((folder / "optimization.yaml").read_text())
        self.assertEqual(saved["study"]["n_trials"], 2)
        self.assertEqual(
            yaml.safe_load((folder / "analysis.yaml").read_text()), self.analysis_cfg
        )
        self.assertEqual(
            json.loads((folder / "submission.json").read_text())["state"], "submitted"
        )

    def test_prepare_does_not_enqueue(self):
        folder, calls = self.prepare_mock()
        self.assertEqual(len(calls), 1)
        self.assertNotIn("-terse", calls[0].args[0])
        self.assertEqual(
            json.loads((folder / "submission.json").read_text())["state"], "prepared"
        )

    @unittest.skipUnless(shutil.which("condor_submit"), "HTCondor not installed")
    def test_real_condor_dry_run_accepts_submit_with_spaced_paths(self):
        folder = prepare_submission(self.path, {})
        self.assertTrue((folder / "job.ad").is_file())
        self.assertRegex((folder / "job.ad").read_text(), r"RequestGPUs\s*=\s*1")

    def test_bad_resources_and_nonzero_device_fail_before_submission(self):
        self.config["condor"] = {"request_cpus": 0}
        self.path.write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(ValueError, "request_cpus"):
            prepare_submission(self.path, {})
        self.config.pop("condor")
        self.path.write_text(yaml.safe_dump(self.config))
        self.cfg["runtime"]["device"] = "1"
        self.reco.write_text(yaml.safe_dump(self.cfg))
        with self.assertRaisesRegex(ValueError, "one visible GPU"):
            prepare_submission(self.path, {})

    def test_worker_requires_gpu_and_rejects_changed_snapshots(self):
        folder, _ = self.prepare_mock()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(RuntimeError, "No Condor-assigned"):
                run_worker(folder / "manifest.json")
        (folder / "analysis.yaml").write_text("changed")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test"}):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run_worker(folder / "manifest.json")
        self.assertEqual(
            json.loads((folder / "status.json").read_text())["state"], "failed"
        )

    def test_restart_consumes_only_remaining_trial_budget(self):
        study = optuna.create_study()
        self.assertEqual(submission_trials_remaining(study, self.root, 3), 3)
        study.tell(study.ask(), 1.0)
        interrupted = study.ask()
        self.assertEqual(submission_trials_remaining(study, self.root, 3), 1)
        self.assertEqual(
            study.trials[interrupted.number].state, optuna.trial.TrialState.FAIL
        )
        study.tell(study.ask(), 0.5)
        self.assertEqual(submission_trials_remaining(study, self.root, 200), 0)
        self.assertEqual(len(study.trials), 3)

    def test_concurrent_writers_are_rejected(self):
        with study_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another optimizer"):
                with study_lock(self.root):
                    self.fail("concurrent lock was granted")
        with study_lock(self.root):
            pass

    def test_cli_dispatches_submission_without_running_local_optimization(self):
        with (
            patch("pvfit.optimize.condor.prepare_submission") as submit,
            patch("pvfit.optimize.run.run_optimization") as local,
        ):
            self.assertEqual(
                main(["optimize", "--config", str(self.path), "--submit-condor"]), 0
            )
            self.assertTrue(submit.call_args.kwargs["submit"])
            local.assert_not_called()
        with self.assertRaisesRegex(ValueError, "export-only"):
            prepare_submission(self.path, {"export_only": True})

    def test_worker_runs_whole_study_once_and_preserves_allocated_gpu(self):
        repo = Path(__file__).parents[1].resolve()
        self.entrypoint.write_text(
            "import os, sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(repo)!r})\n"
            "from tests.test_convert_allen import write_allen_file\n"
            "assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-condor-test'\n"
            "assert '/.singularity.d/libs' in os.environ['LD_LIBRARY_PATH']\n"
            "write_allen_file(Path(sys.argv[sys.argv.index('--monitoring-filename')+1]), [123], 2)\n"
        )
        folder, _ = self.prepare_mock()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-condor-test"}):
            run_worker(folder / "manifest.json")
            run_worker(folder / "manifest.json")  # simulate a starter restart
        study_dir = Path(
            json.loads((folder / "manifest.json").read_text())["study_dir"]
        )
        study = optuna.load_study(
            study_name="condor_test",
            storage=f"sqlite:///{study_dir / 'optuna_study.db'}",
        )
        self.assertEqual(len(study.trials), 2)
        self.assertTrue(
            all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        )
        self.assertEqual(study.trials[0].values, [1.0, 0.0, 0.0, 0.0])
        analysis = yaml.safe_load((study_dir / "trial_0/analysis.yaml").read_text())
        self.assertEqual(analysis["analysis"]["method_name"], "Allen")
        self.assertEqual(analysis["analysis"]["purity_threshold"], 0.8)
        self.assertEqual(
            json.loads((folder / "status.json").read_text())["state"], "complete"
        )
        self.assertEqual(
            json.loads((folder / "trial_budget.json").read_text())["target_trials"], 2
        )


if __name__ == "__main__":
    unittest.main()
