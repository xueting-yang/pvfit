"""The histogram plot reads any producer's HDF5 in the PVFit event layout."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pvfit.analysis.plot_histogram import load_event_histogram, plot_event_histogram
from pvfit.config_loader import AnalysisConfig
from pvfit.fit.save_results_to_hdf5 import save_all_events_to_hdf5
from tests.reco_params import default_params


class PlotHistogramTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "input.h5"

    def write(self, histogram, groups=None, source=None, n_bins=None):
        with h5py.File(self.path, "w") as f:
            if source:
                f.attrs["source"] = source
            event = f.create_group("0")
            ds = event.create_dataset("histogram", data=histogram)
            n_z, n_t = n_bins or histogram.shape
            ds.attrs.update(
                z_min=-10.0,
                z_max=10.0,
                t_min=-1.0,
                t_max=1.0,
                n_bins_z=n_z,
                n_bins_t=n_t,
            )
            for name, fields in (groups or {}).items():
                group = event.create_group(name)
                for field, values in fields.items():
                    group[field] = values

    def plot(self, **flags):
        output = plot_event_histogram(
            self.path, self.root / "plots", AnalysisConfig(**flags)
        )
        self.assertIsNotNone(output)
        self.assertTrue(output.exists())

    def write_events(self, identities):
        """Create distinct histograms keyed independently of source identities."""
        self.write(np.ones((4, 2)))
        with h5py.File(self.path, "a") as f:
            for group_id, run, event in identities:
                f.copy("0", str(group_id))
                group = f[str(group_id)]
                group.attrs.update(run_number=run, source_event_number=event)
                group["histogram"][:] = int(group_id)
            del f["0"]

    def test_default_selects_first_numeric_group_not_zero_or_lexical_order(self):
        self.write_events([(10, 1, 9), (2, 1, 8)])
        data = load_event_histogram(self.path)
        self.assertEqual(data["event_id"], 2)
        self.assertEqual((data["run_number"], data["event_number"]), (1, 8))
        output = plot_event_histogram(self.path, self.root, AnalysisConfig())
        self.assertEqual(output.name, "histogram_run_1_event_8.pdf")
        self.assertTrue(output.exists())

    def test_identity_selects_original_event_and_disambiguates_runs(self):
        self.write_events([(2, 1, 10), (10, 2, 10), (11, 2, 2)])
        data = load_event_histogram(self.path, run_number=2, event_number=10)
        self.assertEqual(data["event_id"], 10)
        np.testing.assert_array_equal(data["histogram"], np.full((4, 2), 10))
        # Event 2 is an original number here, not HDF5 group 2.
        data = load_event_histogram(self.path, run_number=2, event_number=2)
        self.assertEqual(data["event_id"], 11)
        output = plot_event_histogram(
            self.path, self.root, AnalysisConfig(), run_number=2, event_number=2
        )
        self.assertEqual(output.name, "histogram_run_2_event_2.pdf")
        self.assertTrue(output.exists())

    def test_zero_and_large_integer_identities_are_exact(self):
        large = 2**53 + 1
        self.write_events([(2, 0, 0), (3, 0, large), (4, 0, large + 1)])
        for event, group_id in ((0, 2), (large, 3), (large + 1, 4)):
            with self.subTest(event=event):
                data = load_event_histogram(self.path, run_number=0, event_number=event)
                self.assertEqual(data["event_id"], group_id)

    def test_duplicate_identity_is_rejected(self):
        self.write_events([(2, 1, 8), (10, 1, 8)])
        with self.assertRaisesRegex(ValueError, "Duplicate event identity"):
            load_event_histogram(self.path, run_number=1, event_number=8)

    def test_missing_identity_is_not_an_internal_id_fallback(self):
        self.write_events([(2, 1, 8)])
        with self.assertRaisesRegex(ValueError, "not found"):
            load_event_histogram(self.path, run_number=1, event_number=2)

    def test_explicit_identity_requires_metadata(self):
        self.write(np.ones((4, 2)))
        with self.assertRaisesRegex(ValueError, "lacks run_number/source_event_number"):
            load_event_histogram(self.path, run_number=0, event_number=0)
        # The default still supports histogram-only files from other producers.
        data = load_event_histogram(self.path)
        self.assertIsNone(data["run_number"])
        self.assertIsNone(data["event_number"])

    def test_invalid_or_incomplete_selection_is_rejected(self):
        self.write(np.ones((4, 2)))
        for kwargs in (
            {"run_number": 1},
            {"event_number": 1},
            {"run_number": True, "event_number": 1},
            {"run_number": 1, "event_number": 1.5},
            {"run_number": -1, "event_number": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_event_histogram(self.path, **kwargs)

    def test_pvfit_writer_stores_all_seeds_and_source(self):
        tracks = pd.DataFrame(
            {
                name: np.zeros(1)
                for name in (
                    "beamPOCA_x",
                    "beamPOCA_y",
                    "beamPOCA_z",
                    "beamPOCA_t",
                    "beamPOCA_tx",
                    "beamPOCA_ty",
                    "c00",
                    "c11",
                    "c55",
                    "ovtx_x",
                    "ovtx_y",
                    "ovtx_z",
                    "ovtx_t",
                    "p",
                    "pt",
                    "chi2",
                )
            }
            | {"mc_key": [1], "backward": [False], "fromPV": [True]}
        )
        rec = {"rec_z": 1.0, "rec_t": 0.1, "track_index": [0]}
        result = {
            "event_number": 0,
            "run_number": 123,
            "source_event_number": 456,
            "vertex_true_info": [],
            "vertex_rec_info": [rec],
            "histogram": np.ones((4, 2)),
            "n_seedings": 3,
            # Two seeds were dropped or merged; only one became a vertex.
            "seeds": [(1.0, 0.1), (5.0, -0.2), (-3.0, 0.3)],
            "event_data": tracks,
            "selected_for_seeding": np.ones(len(tracks), dtype=bool),
        }
        params = default_params(
            z_min=-10.0, z_max=10.0, t_min=-1.0, t_max=1.0, n_bins_z=4, n_bins_t=2
        )
        save_all_events_to_hdf5([result], self.path, params)

        data = load_event_histogram(self.path)
        self.assertEqual(data["source"], "PVFit")
        self.assertEqual((data["run_number"], data["event_number"]), (123, 456))
        np.testing.assert_allclose(data["seeds"]["z"], [1.0, 5.0, -3.0])
        self.assertEqual(len(data["rec"]["z"]), 1)
        self.plot(draw_fixed_ellipses=True, draw_fitted_ellipses=True)

    def test_other_producer_with_peaks_and_no_errors_or_membership(self):
        self.write(
            np.arange(8, dtype=np.int32).reshape(4, 2),
            groups={
                "seeds": {"z": [1.0, 2.0, 3.0], "t": [0.0, 0.1, 0.2]},
                "rec_vertices": {"z": [1.5], "t": [0.05]},
                "true_vertices": {"z": [1.4], "t": [0.04]},
            },
            source="Allen",
        )
        data = load_event_histogram(self.path)
        self.assertEqual(data["source"], "Allen")
        self.assertEqual(len(data["seeds"]["z"]), 3)
        self.plot(draw_fixed_ellipses=True, draw_fitted_ellipses=True)

    def test_older_files_fall_back_to_rec_seed_positions(self):
        self.write(
            np.ones((4, 2)),
            groups={
                "rec_vertices": {
                    "z": [1.5],
                    "t": [0.05],
                    "z_seed": [1.0],
                    "t_seed": [0.0],
                }
            },
        )
        data = load_event_histogram(self.path)
        np.testing.assert_allclose(data["seeds"]["z"], [1.0])
        self.assertIsNone(data["source"])

    def test_histogram_only(self):
        self.write(np.ones((4, 2)))
        data = load_event_histogram(self.path)
        self.assertIsNone(data["true"])
        self.assertIsNone(data["seeds"])
        self.assertIsNone(data["rec"])
        self.plot(draw_fitted_ellipses=True)

    def test_transposed_histogram_is_rejected(self):
        self.write(np.ones((2, 4)), n_bins=(4, 2))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_event_histogram(self.path)


if __name__ == "__main__":
    unittest.main()
