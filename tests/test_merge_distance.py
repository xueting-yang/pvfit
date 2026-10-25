"""Merge-pair combinatorics and signed projections of known true vertices."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pvfit.analysis.analyze_merge_distance import (
    analyze_merge_distance,
    collect_merge_pairs,
)
from pvfit.analysis.run_analysis import run_analysis
from pvfit.config_loader import AnalysisConfig
from tests.test_rate_analyses import write_events


def event(event_id, true, rec, z, t):
    return SimpleNamespace(
        event_id=event_id,
        true_tracks=true,
        rec_tracks=rec,
        true_z=np.array(z),
        true_t=np.array(t),
    )


class MergeDistanceTests(unittest.TestCase):
    def test_pipeline_serial_and_parallel_use_batch_columns(self):
        true = [
            {"z": 0.5, "t": 0.01, "tracks": {0, 1, 2, 3}},
            {"z": -0.5, "t": -0.01, "tracks": {4, 5, 6, 7}},
        ]
        rec = [{"z": 0, "t": 0, "tracks": set(range(8))}] * 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.h5"
            write_events(path, [(true, rec), (true, rec)])
            outputs = []
            for workers in (1, 2):
                output = root / str(workers)
                settings = AnalysisConfig(
                    analyses=("merge_rate",), workers=workers, batches=2
                )
                with patch(
                    "pvfit.analysis.analyze_merge_distance.collect_merge_pairs",
                    side_effect=AssertionError("must not rescan HDF5"),
                ):
                    run_analysis(str(path), str(output), settings)
                stem = output / "merge_distance_pvfit_track_based"
                summary = json.loads(stem.with_suffix(".json").read_text())
                self.assertEqual(summary["pair_count"], 2)
                outputs.append(stem.with_suffix(".csv").read_bytes())
            self.assertEqual(*outputs)

    def test_multi_merge_deduplicated_per_event_and_signed_by_index(self):
        events = [
            event(
                0,
                [{0}, {1}, {2}],
                [{0, 1, 2}, {0, 1, 2}, {0}],
                [-0.5, 0.5, 2],
                [0.01, -0.01, 0.15],
            ),
            event(1, [{0}, {1}], [{0, 1}], [1, -1], [-0.01, 0.01]),
        ]
        with patch(
            "pvfit.analysis.analyze_merge_distance.iter_events", return_value=events
        ):
            rows = collect_merge_pairs("unused.h5", 0.7)
        self.assertEqual(len(rows), 4)  # three pairs, not six, plus another event
        self.assertEqual(
            [(r["true_index_1"], r["true_index_2"]) for r in rows[:3]],
            [(0, 1), (0, 2), (1, 2)],
        )
        np.testing.assert_allclose([r["delta_z_mm"] for r in rows], [1, 2.5, 1.5, -2])
        np.testing.assert_allclose(
            [r["delta_t_ns"] for r in rows], [-0.02, 0.14, 0.16, 0.02]
        )

    def test_existing_recall_threshold_is_inclusive(self):
        e = event(
            0,
            [{0, 1, 2, 3}, {4, 5, 6, 7}],
            [{0, 1, 2, 4, 5}, {0, 1, 2, 4, 5, 6}],
            [0, 1],
            [0, 0.01],
        )
        with patch(
            "pvfit.analysis.analyze_merge_distance.iter_events", return_value=[e]
        ):
            self.assertEqual(len(collect_merge_pairs("unused.h5", 0.75)), 1)
            self.assertEqual(len(collect_merge_pairs("unused.h5", 0.76)), 0)

    def test_exports_keep_tails_and_report_nonfinite_and_empty(self):
        e = event(0, [{0}, {1}, {2}], [{0, 1, 2}], [0, 4, np.nan], [0, 0.2, 0])
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "pvfit.analysis.analyze_merge_distance.iter_events", return_value=[e]
            ):
                summary = analyze_merge_distance("unused.h5", directory, 0.7, "PVfit")
            self.assertEqual(summary["pair_count"], 3)
            self.assertEqual(summary["finite_pair_count"], 1)
            self.assertEqual(summary["nonfinite_pairs"], 2)
            self.assertEqual(summary["shown_joint_pairs"], 0)
            stem = Path(directory) / "merge_distance_pvfit_track_based"
            with stem.with_suffix(".csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertTrue(stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"))
            self.assertTrue(stem.with_suffix(".png").is_file())
            with patch(
                "pvfit.analysis.analyze_merge_distance.iter_events", return_value=[]
            ):
                empty = analyze_merge_distance("unused.h5", directory, 0.7, "empty.v2")
            self.assertEqual(empty["pair_count"], 0)
            for suffix in ("csv", "json", "pdf", "png"):
                self.assertTrue(
                    (
                        Path(directory)
                        / f"merge_distance_empty.v2_track_based.{suffix}"
                    ).is_file()
                )


if __name__ == "__main__":
    unittest.main()
