import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from pvfit.fit.event_processing import process_event
from pvfit.fit.save_results_to_hdf5 import TRACK_COLUMNS, save_all_events_to_hdf5
from pvfit.fit.truth_matching import build_true_vertex_info, select_tracks
from pvfit.fit.vertex_fitter import (
    calculate_track_vertex_chi2,
    extract_track_info_for_chi2,
    track_arrays,
)
from tests.reco_params import default_params


class TruthMatchingTests(unittest.TestCase):
    def setUp(self):
        self.tracks = pd.DataFrame(
            {
                "event_number": [7] * 5,
                "mc_key": [10, 11, 12, 13, -1],
                "ovtx_x": [0.0] * 5,
                "ovtx_y": [0.0] * 5,
                "ovtx_z": [0.0] * 5,
                "ovtx_t": [0.0] * 5,
                "beamPOCA_z": [0.0] * 5,
                "beamPOCA_t": [0.0] * 5,
            },
            index=[20, 40, 50, 80, 90],
        )
        self.vertices = pd.DataFrame(
            {
                "event_number": [7],
                "effective_pv_key": [99],
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "time": [0.0],
                "n_valid_tracks": [20],
                "track_keys": [np.array([10, 11, 12, 13])],
            }
        )

    def test_key_does_not_match_unrelated_track_at_same_position(self):
        key = build_true_vertex_info(self.tracks, self.vertices)
        self.assertEqual(key[0]["track_indices"], [0, 1, 2, 3])
        self.assertEqual(key[0]["effective_pv_key"], 99)

    def test_key_matches_displaced_origin(self):
        self.tracks.loc[:, "ovtx_z"] = 1.0
        self.assertEqual(len(build_true_vertex_info(self.tracks, self.vertices)), 1)

    def test_keys_are_event_local(self):
        self.vertices.loc[:, "event_number"] = 8
        with self.assertRaisesRegex(ValueError, "same single event"):
            build_true_vertex_info(self.tracks, self.vertices)

    def test_minimum_counts_supplied_rows_not_dump_count(self):
        self.tracks.loc[20, "beamPOCA_z"] = 500.0
        selected = select_tracks(
            self.tracks, dict(z_min=-200, z_max=200, t_min=-1, t_max=1)
        )
        self.assertEqual(build_true_vertex_info(selected, self.vertices), [])
        info = build_true_vertex_info(selected, self.vertices, min_tracks=0)
        self.assertEqual(info[0]["track_indices"], [0, 1, 2])

    def test_process_keeps_full_rows_through_fitting_merging_and_storage(self):
        # Exclude middle rows from seeding, with duplicate/nonconsecutive input labels.
        self.tracks.index = [8, 8, 90, 2, 3]
        self.tracks.iloc[1, self.tracks.columns.get_loc("beamPOCA_t")] = 2.0
        self.tracks.iloc[3, self.tracks.columns.get_loc("beamPOCA_z")] = 500.0
        original = self.tracks.copy(deep=True)
        params = default_params(t_min=-1.0, t_max=1.0)

        def fit(tracks, vertices, *_, **__):
            self.assertEqual(tracks.mc_key.tolist(), [10, 11, 12, 13, -1])
            vertices[0]["track_index"] = [1, 2]
            vertices[0]["Params_rec_x"] = 0.0

        def merge(tracks, vertices, *_, **__):
            self.assertEqual(tracks.mc_key.tolist(), [10, 11, 12, 13, -1])
            self.assertEqual(vertices[0]["track_index"], [1, 2])
            vertices[0]["track_index"] = [3, 1]

        with ExitStack() as stack:
            module = "pvfit.fit.event_processing."
            stack.enter_context(
                patch(
                    module + "fill_histogram_for_seedings",
                    return_value=np.zeros((2, 2)),
                )
            )
            stack.enter_context(patch(module + "find_seedings", return_value=[(0, 0)]))
            stack.enter_context(
                patch(module + "fit_reconstructed_vertices", side_effect=fit)
            )
            stack.enter_context(
                patch(module + "merge_close_vertices", side_effect=merge)
            )
            result = process_event(7, self.tracks, self.vertices, params)
        self.assertEqual(result["vertex_true_info"][0]["track_indices"], [0, 1, 2, 3])
        self.assertEqual(result["vertex_rec_info"][0]["track_index"], [3, 1])
        self.assertEqual(
            result["selected_for_seeding"].tolist(), [True, False, True, False, True]
        )
        self.assertEqual(len(result["event_data"]), 5)
        pd.testing.assert_frame_equal(self.tracks, original)
        # Round-trip truth and reconstruction through the full-row HDF5 schema.
        for column, _ in TRACK_COLUMNS.values():
            if column is not None and column not in result["event_data"]:
                result["event_data"][column] = 0
        params.update(n_bins_z=2, n_bins_t=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.h5"
            save_all_events_to_hdf5([result], path, params)
            with h5py.File(path, "r") as stored:
                event = stored["7"]
                self.assertEqual(
                    stored.attrs["assignment_track_selection"], "all_input"
                )
                self.assertEqual(event.attrs["assignment_track_selection"], "all_input")
                self.assertEqual(event.attrs["n_tracks"], 5)
                self.assertEqual(event.attrs["n_seeding_tracks"], 3)
                self.assertEqual(
                    event["true_vertices/track_indices"][0].tolist(), [0, 1, 2, 3]
                )
                self.assertEqual(
                    event["rec_vertices/track_indices"][0].tolist(), [3, 1]
                )
                self.assertEqual(
                    event["tracks/mc_key"][:].tolist(), [10, 11, 12, 13, -1]
                )
                self.assertEqual(
                    event["tracks/selected_for_seeding"][:].tolist(),
                    [True, False, True, False, True],
                )

    def test_process_retains_truth_when_no_tracks_pass_seeding_window(self):
        self.tracks["beamPOCA_t"] = 2.0
        params = default_params(t_min=-1.0, t_max=1.0, min_chi2_dist_merge=0.0)
        with ExitStack() as stack:
            module = "pvfit.fit.event_processing."
            histogram = stack.enter_context(
                patch(
                    module + "fill_histogram_for_seedings",
                    return_value=np.zeros((2, 2)),
                )
            )
            stack.enter_context(patch(module + "find_seedings", return_value=[]))
            fit = stack.enter_context(patch(module + "fit_reconstructed_vertices"))
            result = process_event(7, self.tracks, self.vertices, params)
        self.assertTrue(histogram.call_args.args[0].empty)
        self.assertEqual(len(fit.call_args.args[0]), len(self.tracks))
        self.assertEqual(result["n_true"], 1)
        self.assertEqual(result["n_fitted"], 0)
        self.assertEqual(result["vertex_rec_info"], [])
        self.assertFalse(result["selected_for_seeding"].any())
        self.assertEqual(len(result["event_data"]), 5)

    def test_window_excluded_tracks_can_keep_a_seed_alive_by_chi2(self):
        # The fourth compatible track is just outside the seeding window. It
        # must still be assigned, keeping this seed above the four-track minimum.
        # A fifth track far away in time must still fail the assignment chi2 cut.
        for boundary in ("t", "z"):
            with self.subTest(boundary=boundary):
                tracks = self.tracks.copy(deep=True)
                tracks["beamPOCA_x"] = 0.0
                tracks["beamPOCA_y"] = 0.0
                tracks["beamPOCA_tx"] = [0.01, -0.02, 0.03, -0.04, 0.05]
                tracks["beamPOCA_ty"] = [0.02, 0.03, -0.02, 0.01, -0.03]
                tracks["backward"] = False
                tracks["c00"] = tracks["c11"] = 0.0001
                tracks["c55"] = 0.025**2
                seed_z, seed_t = 0.0, 0.79
                tracks["beamPOCA_z"] = 0.0
                tracks["beamPOCA_t"] = [0.785, 0.79, 0.795, 0.81, 1.2]
                if boundary == "z":
                    seed_z = 199.99
                    tracks["beamPOCA_z"] = [199.98, 199.985, 199.99, 200.01, 199.99]
                    tracks.iloc[3, tracks.columns.get_loc("beamPOCA_t")] = seed_t
                params = default_params(min_chi2_dist_merge=0.0)
                with ExitStack() as stack:
                    module = "pvfit.fit.event_processing."
                    histogram = stack.enter_context(
                        patch(
                            module + "fill_histogram_for_seedings",
                            return_value=np.zeros((2, 2)),
                        )
                    )
                    stack.enter_context(
                        patch(module + "find_seedings", return_value=[(seed_z, seed_t)])
                    )
                    result = process_event(7, tracks, self.vertices, params)
                self.assertEqual(
                    histogram.call_args.args[0].mc_key.tolist(), [10, 11, 12]
                )
                self.assertEqual(
                    result["selected_for_seeding"].tolist(),
                    [True, True, True, False, False],
                )
                self.assertEqual(result["n_fitted"], 1)
                vertex = result["vertex_rec_info"][0]
                self.assertEqual(vertex["track_index"], [0, 1, 2, 3])
                final_position = {
                    f"rec_{coord}": vertex[f"Params_rec_{coord}"] for coord in "xyzt"
                }
                self.assertTrue(
                    all(np.isfinite(value) for value in final_position.values())
                )
                outside_track = extract_track_info_for_chi2(
                    track_arrays(result["event_data"]), 3
                )
                self.assertLess(
                    calculate_track_vertex_chi2(outside_track, final_position)["chi2"],
                    params["max_track_seed_chi2"],
                )

    def test_missing_key_fields_never_fall_back(self):
        with self.assertRaisesRegex(ValueError, "mc_key"):
            build_true_vertex_info(self.tracks.drop(columns="mc_key"), self.vertices)
        with self.assertRaisesRegex(ValueError, "track_keys"):
            build_true_vertex_info(
                self.tracks, self.vertices.drop(columns="track_keys")
            )

    def test_duplicate_reco_keys_preserve_track_rows(self):
        self.tracks.loc[90, "mc_key"] = 10
        self.assertEqual(
            build_true_vertex_info(self.tracks, self.vertices)[0]["track_indices"],
            list(range(5)),
        )

    def test_conflicting_truth_owners_fail(self):
        other = self.vertices.copy()
        other["effective_pv_key"] = 100
        with self.assertRaisesRegex(ValueError, "multiple truth vertices"):
            build_true_vertex_info(self.tracks, pd.concat([self.vertices, other]))

    def test_empty_truth_and_required_count(self):
        self.assertEqual(
            build_true_vertex_info(self.tracks, self.vertices.iloc[:0]), []
        )
        with self.assertRaisesRegex(ValueError, "n_valid_tracks"):
            build_true_vertex_info(
                self.tracks, self.vertices.drop(columns="n_valid_tracks")
            )

    def test_membership_does_not_require_origin_coordinates(self):
        tracks = self.tracks.drop(columns=["ovtx_x", "ovtx_y", "ovtx_z", "ovtx_t"])
        self.assertEqual(
            build_true_vertex_info(tracks, self.vertices)[0]["track_indices"],
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
