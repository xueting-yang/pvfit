import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from pvfit.fit.truth_matching import build_true_vertex_info, read_mcpv_dataframe
from pvfit.toy import (
    default_profile,
    fit_momentum_distribution,
    generate_event,
    generate_sample,
    sample_momentum_distribution,
)


class ToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = default_profile()

    def test_seed_is_reproducible(self):
        a, av = generate_event(np.random.default_rng(12), self.profile, 1, 0)
        b, bv = generate_event(np.random.default_rng(12), self.profile, 1, 0)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(
            av.drop(columns="track_keys"), bv.drop(columns="track_keys")
        )
        for first, second in zip(av.track_keys, bv.track_keys, strict=False):
            np.testing.assert_array_equal(first, second)
        c, _ = generate_event(np.random.default_rng(13), self.profile, 1, 0)
        self.assertFalse(a.equals(c))

    def test_geometry_momentum_and_covariance(self):
        t, _ = generate_event(np.random.default_rng(14), self.profile, 1, 0)
        dot = t.beamPOCA_x * t.beamPOCA_tx + t.beamPOCA_y * t.beamPOCA_ty
        np.testing.assert_allclose(dot, 0, atol=1e-10)
        theta = np.hypot(t.beamPOCA_tx, t.beamPOCA_ty)
        np.testing.assert_allclose(t.pt, t.p * theta / np.sqrt(1 + theta**2))
        self.assertTrue(np.all(np.isfinite(t.to_numpy(dtype=float))))
        self.assertTrue((t[["c00", "c11", "c22", "c33", "c55"]] > 0).all().all())
        self.assertTrue(t.mc_key.is_unique)

    def test_keys_recover_full_vertex_membership_after_shuffle(self):
        t, vertices = generate_event(np.random.default_rng(15), self.profile, 1, 9)
        truth = build_true_vertex_info(t, vertices)
        self.assertEqual(len(truth), len(vertices))
        used = set()
        for info, keys in zip(truth, vertices.track_keys, strict=False):
            indices = info["track_indices"]
            self.assertFalse(used.intersection(indices))
            used.update(indices)
            self.assertEqual(set(t.iloc[indices].mc_key), set(keys))
            self.assertEqual(len(indices), info["expected_ntracks"])

    def test_root_roundtrip_without_reference_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.root"
            result = generate_sample(self.profile, path, events=2, seed=16)
            with uproot.open(path) as root:
                self.assertEqual(
                    json.loads(str(root["toy_metadata"]))["sample_type"],
                    "synthetic_parametric_toy",
                )
                t = pd.DataFrame(
                    root["velo_kalman_filter/monitor_tree"].arrays(library="np")
                )
                v = read_mcpv_dataframe(root["mc_pv_dumper/MCPV_effective_tree"])
            self.assertEqual(len(t), result["tracks"])
            self.assertEqual(set(t.event_number), {0, 1})
            self.assertEqual(set(t.run_number), {16})
            self.assertEqual(set(v.run_number), {16})
            self.assertEqual(result["run_number"], 16)
            for event in (0, 1):
                truth = build_true_vertex_info(
                    t[t.event_number == event], v[v.event_number == event]
                )
                self.assertEqual(len(truth), int((v.event_number == event).sum()))
            with self.assertRaises(FileExistsError):
                generate_sample(self.profile, path)

    def test_time_errors_retain_discrete_levels(self):
        tracks, _ = generate_event(np.random.default_rng(17), self.profile, 1, 0)
        levels = set()
        for population in self.profile["populations"].values():
            for direction in ("forward", "backward"):
                for band in population[direction]["momentum_bins"]:
                    levels.update(
                        band["distribution"]["discrete_marginals"]["3"]["values"]
                    )
        observed = set(np.round(np.log(np.sqrt(tracks.c55)), 6))
        self.assertTrue(observed.issubset(levels))

    def test_momentum_dependent_width_survives_sampling(self):
        # Signed residuals have almost no rank correlation with momentum here,
        # but their width differs by a factor of four. A single copula loses it.
        rng = np.random.default_rng(312)
        features = rng.normal(size=(4000, 8))
        features[:1600, 1] = np.log(rng.uniform(100, 400, 1600))
        features[1600:, 1] = np.log(rng.uniform(11000, 20000, 2400))
        features[:, 3] = np.log(rng.choice([0.015, 0.020, 0.025], len(features)))
        features[:1600, 7] *= 4
        profile = fit_momentum_distribution(features)
        sampled = sample_momentum_distribution(rng, profile, 20000)
        momentum = np.exp(sampled[:, 1])
        low = momentum < 500
        self.assertAlmostEqual(float(low.mean()), 0.4, delta=0.02)
        self.assertTrue(np.all(momentum[~low] > 10000))
        for selected, expected_width in [(low, 4.0), (~low, 1.0)]:
            q16, q84 = np.quantile(sampled[selected, 7], [0.16, 0.84])
            self.assertAlmostEqual(
                (q84 - q16) / 2, expected_width, delta=0.15 * expected_width
            )

    def test_old_profile_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = {**self.profile, "schema_version": 1}
            path = Path(directory) / "old.root"
            with self.assertRaisesRegex(ValueError, "version 2"):
                generate_sample(profile, path)
            self.assertFalse(path.exists())

    def test_generated_sample_runs_the_whole_pipeline(self):
        """pvfit toy -> pvfit run with every analysis, without reference data."""
        import contextlib
        import io

        from pvfit.cli import main

        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            sample = work / "toy.root"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "toy",
                        "--output",
                        str(sample),
                        "--events",
                        "2",
                        "--seed",
                        "18",
                    ]
                )
                self.assertEqual(code, 0)
                code = main(
                    [
                        "run",
                        "--config",
                        str(root / "configs/reconstruction.yaml"),
                        "--input",
                        str(sample),
                        "--output-dir",
                        str(work / "out"),
                        "--max-workers",
                        "1",
                        "--analysis-config",
                        str(root / "configs/analysis.yaml"),
                    ]
                )
            self.assertEqual(code, 0)
            outputs = {p.name for p in (work / "out" / "analysis").iterdir()}
            for name in (
                "analysis.done",
                "efficiency_overall_true_binned_pvfit_track_based.csv",
                "fake_rate_overall_rec_binned_pvfit_reso_based.csv",
                "merge_rate_true_binned_pvfit_geo_based.csv",
                "resolution_z_overall_pvfit_track_based.csv",
                "pull_analysis_input_track_summary.json",
                "histogram_run_18_event_0.pdf",
            ):
                self.assertIn(name, outputs)

    def test_invalid_event_count(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_sample(self.profile, Path(directory) / "toy.root", events=0)


if __name__ == "__main__":
    unittest.main()
