"""Regression checks for truth association, transport, and residual chi2 selection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from pvfit.analysis.accumulators import TrackPull
from pvfit.analysis.columns import ColumnReader
from pvfit.analysis.run_analysis import run_analysis
from pvfit.analysis.track_pull_config import resolve_track_pull_chi2
from pvfit.config_loader import AnalysisConfig
from pvfit.fit.save_results_to_hdf5 import save_all_events_to_hdf5
from pvfit.fit.truth_matching import build_true_vertex_info
from tests.reco_params import default_params


class TrackPullMembershipTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "input.h5"

    def write_tracks(self, n=2, indices=None, states=None, truth=None, max_chi2=10.0):
        defaults = {name: np.zeros(n) for name in ("x", "y", "z", "t", "tx", "ty")}
        defaults["backward"] = np.zeros(n, dtype=bool)
        defaults.update({name: np.ones(n) for name in ("cov_xx", "cov_yy", "cov_tt")})
        defaults.update(states or {})
        with h5py.File(self.path, "w") as f:
            f.attrs["n_events"] = 1
            if max_chi2 is not None:
                f.attrs["reco_max_chi2_tukey"] = max_chi2
            event = f.create_group("0")
            true = event.create_group("true_vertices")
            for name in ("x", "y", "z", "t"):
                true[name] = [(truth or {}).get(name, 0.0)]
            membership = true.create_dataset(
                "track_indices", (1,), dtype=h5py.vlen_dtype(np.int32)
            )
            membership[0] = np.asarray(
                np.arange(n) if indices is None else indices, dtype=np.int32
            )
            tracks = event.create_group("tracks")
            for name, values in defaults.items():
                tracks[name] = values

    def analyze(self, max_chi2=None):
        settings = AnalysisConfig(
            analyses=("track_pull",), track_pull_chi2=max_chi2, keep_columns=True
        )
        with patch("pvfit.analysis.residual_accumulators._plot_track_pulls") as plot:
            execution = run_analysis(str(self.path), self.directory.name, settings)
        plot.assert_called_once()
        reader = ColumnReader(
            sorted(Path(execution["work_dir"]).glob("batch_*/columns.json"))
        )
        data = {name: reader.array(f"track/{name}") for name in TrackPull.FIELDS}
        return plot.call_args.args[0]["summary"], data

    def test_uses_track_indices_without_origin_coordinate_fields(self):
        self.write_tracks(indices=[1], states={"x": [100.0, 0.1]})
        summary, data = self.analyze()
        np.testing.assert_array_equal(data["residual_x"], [0.1])
        self.assertEqual(summary["n_associated"], 1)

    def test_transport_to_truth_plane_for_forward_and_backward_tracks(self):
        # Two exact lines through the same truth point. Opposite time slopes
        # require opposite propagation signs; both must give zero residuals.
        flight_time = 2 * np.sqrt(2) / 299.792458
        self.write_tracks(
            states={
                "x": [8.0, 10.0],
                "y": [-5.0, -7.0],
                "z": [0.0, 0.0],
                "t": [7 - flight_time, 7 + flight_time],
                "tx": [1.0, 0.0],
                "ty": [0.0, 1.0],
                "backward": [False, True],
            },
            truth={"x": 10.0, "y": -5.0, "z": 2.0, "t": 7.0},
        )
        _, data = self.analyze()
        for axis in ("x", "y", "t"):
            np.testing.assert_allclose(data[f"residual_{axis}"], 0, atol=1e-12)
        np.testing.assert_array_equal(data["delta_z"], [2.0, 2.0])

    def test_large_residual_is_rejected_from_plot_and_statistics(self):
        self.write_tracks(states={"x": [0.5, 8.0]})
        summary, data = self.analyze()
        np.testing.assert_array_equal(data["pull_x"], [0.5])
        self.assertEqual(summary["n_valid"], 2)
        self.assertEqual(summary["n_selected"], 1)
        self.assertEqual(summary["n_rejected_by_chi2"], 1)
        self.assertEqual(summary["statistics"]["pull_x"]["n"], 1)
        self.assertEqual(summary["statistics"]["pull_x"]["mean"], 0.5)
        stored = json.loads(
            (
                Path(self.directory.name) / "pull_analysis_input_track_summary.json"
            ).read_text()
        )
        self.assertEqual(stored, summary)

    def test_cut_is_applied_after_transport_and_handles_no_survivors(self):
        # The raw residual is zero, but transport produces a 4-sigma residual.
        self.write_tracks(n=1, states={"tx": [2.0]}, truth={"z": 2.0})
        summary, data = self.analyze()
        self.assertEqual(summary["n_valid"], 1)
        self.assertEqual(summary["n_selected"], 0)
        self.assertEqual(summary["n_rejected_by_chi2"], 1)
        self.assertEqual(len(data["pull_x"]), 0)
        self.assertIsNone(summary["statistics"]["pull_x"]["mean"])

    def test_cut_includes_time_and_accepts_the_threshold(self):
        self.write_tracks(
            n=3,
            states={"x": [3.0, 3.0, 0.0], "y": [1.0, 1.0, 0.0], "t": [0.0, 0.1, 4.0]},
        )
        summary, data = self.analyze()
        np.testing.assert_array_equal(data["chi2"], [10.0])
        self.assertEqual(summary["n_selected"], 1)
        self.assertEqual(summary["n_rejected_by_chi2"], 2)

    def test_configured_threshold_controls_the_selection(self):
        self.write_tracks(states={"x": [0.5, 2.0]})
        summary, data = self.analyze(max_chi2=1.0)
        np.testing.assert_array_equal(data["pull_x"], [0.5])
        self.assertEqual(summary["max_chi2"], 1.0)

    def test_default_analysis_follows_stored_reconstruction_threshold(self):
        self.write_tracks(states={"x": [2.0, 3.0]}, max_chi2=4.0)
        with patch("pvfit.analysis.residual_accumulators._plot_track_pulls") as plot:
            run_analysis(
                str(self.path),
                self.directory.name,
                AnalysisConfig(analyses=("track_pull",)),
            )
        plot.assert_called_once()
        selected = plot.call_args.args[0]["summary"]["statistics"]["pull_x"]
        self.assertEqual(selected["n"], 1)
        self.assertEqual(selected["mean"], 2.0)
        summary = json.loads(
            (
                Path(self.directory.name) / "pull_analysis_input_track_summary.json"
            ).read_text()
        )
        self.assertEqual(summary["max_chi2"], 4.0)
        self.assertIn("HDF5 attribute", summary["max_chi2_source"])

    def test_legacy_file_uses_adjacent_reconstruction_snapshot(self):
        self.write_tracks(states={"x": [0.5, 2.0]}, max_chi2=None)
        self.path.with_name("reconstruction.yaml").write_text(
            "algorithm:\n  max_chi2_tukey: 1.0\n"
        )
        summary, data = self.analyze()
        np.testing.assert_array_equal(data["pull_x"], [0.5])
        self.assertEqual(summary["max_chi2"], 1.0)
        self.assertIn("reconstruction.yaml", summary["max_chi2_source"])

    def test_stored_value_takes_precedence_over_a_different_snapshot(self):
        self.write_tracks(max_chi2=4.0)
        self.path.with_name("reconstruction.yaml").write_text(
            "algorithm:\n  max_chi2_tukey: 9\n"
        )
        value, source = resolve_track_pull_chi2(self.path)
        self.assertEqual(value, 4.0)
        self.assertIn("HDF5 attribute", source)

    def test_missing_threshold_has_no_silent_numeric_default(self):
        self.write_tracks(max_chi2=None)
        with self.assertRaisesRegex(ValueError, "Cannot determine the reconstruction"):
            self.analyze()
        self.path.with_name("reconstruction.yaml").write_text("algorithm: {}\n")
        with self.assertRaisesRegex(ValueError, "positive finite max_chi2"):
            self.analyze()
        summary, _ = self.analyze(max_chi2=6.5)
        self.assertEqual(summary["max_chi2"], 6.5)
        self.assertEqual(summary["max_chi2_source"], "explicit analysis override")

    def test_invalid_stored_threshold_is_not_ignored(self):
        for value in (0, -1, np.nan, np.inf, True, "4"):
            with self.subTest(value=value):
                self.write_tracks(max_chi2=value)
                with self.assertRaisesRegex(ValueError, "positive finite max_chi2"):
                    self.analyze()

    def test_invalid_states_and_covariances_are_excluded_and_counted(self):
        self.write_tracks(
            n=6,
            states={
                "cov_xx": [1.0, 0.0, -1.0, np.nan, np.inf, 1.0],
                "x": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            },
        )
        summary, data = self.analyze()
        self.assertEqual(summary["n_invalid"], 5)
        self.assertEqual(len(data["pull_x"]), 1)

    def test_legacy_hdf5_requires_regeneration(self):
        self.write_tracks()
        with h5py.File(self.path, "a") as f:
            del f["0/tracks/tx"]
        with self.assertRaisesRegex(ValueError, "re-run reconstruction"):
            self.analyze()

    def test_empty_membership_has_defined_empty_statistics(self):
        self.write_tracks(indices=[])
        summary, data = self.analyze()
        self.assertEqual(summary["n_valid"], 0)
        self.assertEqual(len(data["pull_x"]), 0)
        self.assertIsNone(summary["statistics"]["pull_x"]["std"])

    def test_writer_preserves_directions_in_selected_row_order(self):
        # Two truth memberships with different multiplicities exercise the
        # writer's ragged arrays as in the original toy-based regression case.
        # Coordinates coincide deliberately: the associations must follow keys.
        z = np.array([-2.0, 3.0, -1.0, 5.0, -3.0, 2.0, 4.0, -4.0, 6.0])
        tx = np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1, 0.3, -0.2])
        ty = np.array([0.2, 0.1, -0.1, -0.2, 0.3, -0.2, 0.1, 0.2, -0.1])
        backward = np.array([False, True, False, True, True, False, True, False, True])
        time_slope = np.where(backward, -1, 1) * np.sqrt(1 + tx**2 + ty**2) / 299.792458
        n = len(z)
        tracks = pd.DataFrame(
            {
                "event_number": np.zeros(n, dtype=int),
                "mc_key": [12, 20, 10, 23, 13, 21, 11, 24, 22],
                "beamPOCA_x": z * tx,
                "beamPOCA_y": z * ty,
                "beamPOCA_z": z,
                "beamPOCA_t": z * time_slope,
                "beamPOCA_tx": tx,
                "beamPOCA_ty": ty,
                "backward": backward,
                "c00": np.ones(n),
                "c11": np.ones(n),
                "c55": np.ones(n),
                "ovtx_x": np.zeros(n),
                "ovtx_y": np.zeros(n),
                "ovtx_z": np.zeros(n),
                "ovtx_t": np.zeros(n),
                "p": np.full(n, 1000.0),
                "pt": 1000 * np.hypot(tx, ty) / np.sqrt(1 + tx**2 + ty**2),
                "chi2": np.zeros(n),
                "fromPV": np.ones(n, dtype=bool),
            },
            index=[22, 7, 14, 5, 30, 4, 17, 2, 8],
        )
        vertices = pd.DataFrame(
            {
                "event_number": [0, 0],
                "effective_pv_key": [99, 100],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "time": [0.0, 0.0],
                "n_valid_tracks": [4, 5],
                "track_keys": [
                    np.array([10, 11, 12, 13]),
                    np.array([20, 21, 22, 23, 24]),
                ],
            }
        )
        truth = build_true_vertex_info(tracks, vertices)
        result = {
            "event_number": 0,
            "vertex_true_info": truth,
            "vertex_rec_info": [],
            "histogram": np.zeros((2, 2)),
            "n_seedings": 0,
            "seeds": [],
            "event_data": tracks,
            "selected_for_seeding": np.ones(len(tracks), dtype=bool),
        }
        params = default_params(
            t_min=-0.6, t_max=0.8, n_bins_z=2, n_bins_t=2, max_chi2_tukey=6.5
        )
        save_all_events_to_hdf5([result], self.path, params)
        with h5py.File(self.path, "r") as f:
            self.assertEqual(f.attrs["reco_max_chi2_tukey"], 6.5)
            saved = f["0/tracks"]
            for name in ("tx", "ty"):
                np.testing.assert_array_equal(
                    saved[name][:],
                    tracks[f"beamPOCA_{name}"].to_numpy(dtype=np.float32),
                )
            np.testing.assert_array_equal(saved["backward"][:], tracks.backward)
        summary, _ = self.analyze()
        self.assertEqual(
            summary["n_associated"], sum(len(v["track_indices"]) for v in truth)
        )
        self.assertEqual(summary["n_invalid"], 0)
        self.assertEqual(summary["max_chi2"], 6.5)


if __name__ == "__main__":
    unittest.main()
