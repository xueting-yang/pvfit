"""Frozen public synthetic reconstruction: identities, memberships, fits and rates."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from tests.data.toy_regression import DATA, reconstruct, summarize


class SyntheticRegressionTests(unittest.TestCase):
    def test_serial_and_parallel_match_reviewed_baseline(self):
        expected = json.loads((DATA / "toy_regression_expected.json").read_text())
        for workers in (1, 2):
            with (
                self.subTest(workers=workers),
                tempfile.TemporaryDirectory() as directory,
            ):
                work = Path(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    output, settings = reconstruct(work, workers)
                    actual = summarize(output, settings, work / "analysis")
                with h5py.File(output, "r") as stored:
                    self.assertEqual(stored.attrs["truth_track_selection"], "all_input")
                    self.assertEqual(
                        stored.attrs["assignment_track_selection"], "all_input"
                    )
                    for event in stored.values():
                        mask = event["tracks/selected_for_seeding"][:]
                        self.assertEqual(len(mask), event.attrs["n_tracks"])
                        self.assertEqual(mask.sum(), event.attrs["n_seeding_tracks"])
                        self.assertFalse(mask.all())
                        for members in event["rec_vertices/track_indices"]:
                            self.assertTrue(
                                ((members >= 0) & (members < len(mask))).all()
                            )
                self.assertEqual(actual["events"].keys(), expected["events"].keys())
                for event, reference in expected["events"].items():
                    for field, value in reference.items():
                        observed = actual["events"][event][field]
                        if (
                            field.startswith(("seeds/", "rec_vertices/"))
                            and not field.endswith("track_indices")
                            or field == "histogram_sum"
                        ):
                            np.testing.assert_allclose(
                                observed,
                                value,
                                rtol=2e-6,
                                atol=1e-7,
                                err_msg=f"event {event}: {field}",
                            )
                        else:
                            self.assertEqual(observed, value, f"event {event}: {field}")
                self.assertEqual(actual["rates"].keys(), expected["rates"].keys())
                for name, rows in expected["rates"].items():
                    observed = actual["rates"][name]
                    self.assertEqual(len(observed), len(rows), name)
                    for got, want in zip(observed, rows, strict=True):
                        self.assertEqual(got.keys(), want.keys(), name)
                        for key, value in want.items():
                            if key in ("rate", "error_lower", "error_upper"):
                                np.testing.assert_allclose(
                                    float(got[key] or "nan"),
                                    float(value or "nan"),
                                    rtol=1e-10,
                                    atol=1e-12,
                                    equal_nan=True,
                                    err_msg=f"{name}: {key}",
                                )
                            else:
                                self.assertEqual(got[key], value, f"{name}: {key}")
