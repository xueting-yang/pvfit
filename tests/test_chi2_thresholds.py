"""Assignment and Tukey thresholds are separate, and members are the assigned
tracks with nonzero Tukey weight at the fitted vertex (Allen's convention)."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd

from pvfit.fit.event_processing import process_event
from pvfit.fit.vertex_fitter import (
    calculate_track_vertex_chi2,
    extract_track_info_for_chi2,
    track_arrays,
)
from tests.reco_params import default_params

SIGMA_XY, SIGMA_T = 0.01, 0.025


def make_tracks(seed_times, outlier_chi2, outlier_time=0.0):
    """Four crossing tracks per seed constrain every coordinate; one extra track
    at the first seed has only an x residual, so its chi2 to that seed is
    exactly ``outlier_chi2`` and the fitted x is a weighted mean."""
    n_tracks = 4 * len(seed_times) + 1
    return pd.DataFrame(
        {
            "event_number": [7] * n_tracks,
            "mc_key": np.arange(n_tracks),
            "beamPOCA_x": [0.0] * (n_tracks - 1) + [SIGMA_XY * np.sqrt(outlier_chi2)],
            "beamPOCA_y": [0.0] * n_tracks,
            "beamPOCA_z": [0.0] * n_tracks,
            "beamPOCA_t": list(np.repeat(seed_times, 4)) + [outlier_time],
            "beamPOCA_tx": [-0.1, 0.1, 0.0, 0.0] * len(seed_times) + [0.0],
            "beamPOCA_ty": [0.0, 0.0, -0.1, 0.1] * len(seed_times) + [0.0],
            "backward": [False] * n_tracks,
            "c00": [SIGMA_XY**2] * n_tracks,
            "c11": [SIGMA_XY**2] * n_tracks,
            "c55": [SIGMA_T**2] * n_tracks,
        }
    )


def process(tracks, seeds, params):
    truth = pd.DataFrame(
        {
            "event_number": [7],
            "effective_pv_key": [0],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "time": [0.0],
            "n_valid_tracks": [len(tracks)],
            "track_keys": [tracks.mc_key.to_numpy()],
        }
    )
    with ExitStack() as stack:
        module = "pvfit.fit.event_processing."
        stack.enter_context(
            patch(module + "fill_histogram_for_seedings", return_value=np.zeros((2, 2)))
        )
        stack.enter_context(patch(module + "find_seedings", return_value=seeds))
        return process_event(7, tracks, truth, params)


def chi2_at_fit(tracks, vertex, index):
    position = {f"rec_{c}": vertex[f"Params_rec_{c}"] for c in "xyzt"}
    arrays = track_arrays(tracks)
    return calculate_track_vertex_chi2(
        extract_track_info_for_chi2(arrays, index), position
    )["chi2"]


class Chi2ThresholdTests(unittest.TestCase):
    def test_assignment_uses_max_track_seed_chi2(self):
        # chi2 11 to the seed: above the assignment cut of 10, below Tukey's 12
        tracks = make_tracks([0.0], outlier_chi2=11.0)
        vertex = process(tracks, [(0.0, 0.0)], default_params())["vertex_rec_info"][0]
        self.assertEqual(vertex["assigned_index"], [0, 1, 2, 3])
        self.assertEqual(vertex["track_index"], [0, 1, 2, 3])
        self.assertAlmostEqual(vertex["Params_rec_x"], 0.0)
        # Loosening only the assignment cut lets it in, weighted by Tukey's 12
        params = default_params(max_track_seed_chi2=12.0, max_fit_iterations=1)
        vertex = process(tracks, [(0.0, 0.0)], params)["vertex_rec_info"][0]
        weight = (1.0 - 11.0 / 12.0) ** 2
        self.assertEqual(vertex["assigned_index"], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(
            vertex["Params_rec_x"], tracks.beamPOCA_x.iloc[-1] * weight / (4.0 + weight)
        )

    def test_tukey_weight_uses_max_chi2_tukey(self):
        tracks = make_tracks([0.0], outlier_chi2=9.0)
        for tukey in (12.0, 10.0):
            with self.subTest(max_chi2_tukey=tukey):
                params = default_params(max_chi2_tukey=tukey, max_fit_iterations=1)
                vertex = process(tracks, [(0.0, 0.0)], params)["vertex_rec_info"][0]
                weight = (1.0 - 9.0 / tukey) ** 2
                self.assertEqual(vertex["assigned_index"], [0, 1, 2, 3, 4])
                self.assertAlmostEqual(
                    vertex["Params_rec_x"],
                    tracks.beamPOCA_x.iloc[-1] * weight / (4.0 + weight),
                )
                self.assertAlmostEqual(vertex["cov_00"], SIGMA_XY**2 / (4.0 + weight))
                self.assertEqual(vertex["track_index"], [0, 1, 2, 3, 4])

    def test_members_are_assigned_tracks_with_nonzero_weight_at_the_fit(self):
        # The four tracks sit 0.05 ns from the seed, so the fitted time moves
        # towards them; the outlier (chi2 9.9 to the seed) is assigned, but at
        # the fitted position its chi2 exceeds the Tukey threshold.
        tracks = make_tracks([0.05], outlier_chi2=9.9, outlier_time=0.0)
        params = default_params(max_fit_iterations=1)
        result = process(tracks, [(0.0, 0.0)], params)
        vertex = result["vertex_rec_info"][0]
        self.assertEqual(vertex["assigned_index"], [0, 1, 2, 3, 4])
        chi2 = [chi2_at_fit(tracks, vertex, i) for i in vertex["assigned_index"]]
        self.assertGreater(chi2[4], params["max_chi2_tukey"])
        self.assertTrue(all(c < params["max_chi2_tukey"] for c in chi2[:4]))
        self.assertEqual(vertex["track_index"], [0, 1, 2, 3])
        # A looser Tukey threshold keeps it
        loose = default_params(max_fit_iterations=1, max_chi2_tukey=20.0)
        vertex = process(tracks, [(0.0, 0.0)], loose)["vertex_rec_info"][0]
        self.assertEqual(vertex["track_index"], [0, 1, 2, 3, 4])

    def test_merged_vertex_members_are_filtered_at_the_merged_fit(self):
        seed_times = [0.0, 0.01]
        tracks = make_tracks(seed_times, outlier_chi2=9.0)
        params = default_params(min_chi2_dist_merge=25.0)
        result = process(tracks, [(0.0, t) for t in seed_times], params)
        self.assertEqual(result["n_seedings"], 2)
        self.assertEqual(len(result["vertex_rec_info"]), 1)
        vertex = result["vertex_rec_info"][0]
        self.assertEqual(sorted(vertex["assigned_index"]), list(range(len(tracks))))
        expected = [
            i
            for i in vertex["assigned_index"]
            if chi2_at_fit(tracks, vertex, i) < params["max_chi2_tukey"]
        ]
        self.assertEqual(vertex["track_index"], expected)
        self.assertEqual(sorted(expected), list(range(len(tracks))))

    def test_seeds_without_a_fit_produce_no_vertex(self):
        # A second seed far away collects no track; with min_tracks_per_seed 0
        # it reaches the fit, which is singular, and Allen writes no vertex.
        tracks = make_tracks([0.0], outlier_chi2=9.0)
        params = default_params(min_tracks_per_seed=0)
        result = process(tracks, [(0.0, 0.0), (150.0, 0.5)], params)
        self.assertEqual(result["n_seedings"], 2)
        self.assertEqual(len(result["vertex_rec_info"]), 1)
        self.assertEqual(result["n_fitted"], 1)
        self.assertEqual(result["vertex_rec_info"][0]["rec_vertex_id"], 0)


if __name__ == "__main__":
    unittest.main()
