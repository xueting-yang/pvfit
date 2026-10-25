"""Overall rates and their errors must not depend on the plotting range."""

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pvfit.analysis.binned_rate import BinnedCounts, make_bins
from pvfit.analysis.driver import RATE_ORDER
from pvfit.analysis.run_analysis import run_analysis
from pvfit.config_loader import AnalysisConfig
from tests.test_rate_analyses import write_events


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class OverallPopulationTests(unittest.TestCase):
    def test_overall_errors_include_events_wholly_outside_bins(self):
        events = [
            [(2, True), (205, True)],
            [(3, False), (204, False)],
            [(4, True), (9, True)],
            [],
        ]
        narrow = BinnedCounts(make_bins(4, 13, 5))
        wide = BinnedCounts(make_bins(0, 210, 1))
        parts = []
        for event in events:
            part = BinnedCounts(narrow.bins)
            for n, success in event:
                narrow.add(n, success)
                wide.add(n, success)
                part.add(n, success)
            for counter in (narrow, wide, part):
                counter.end_event()
            parts.append(part.to_dict())
        self.assertEqual(narrow.overall(), wide.overall())
        self.assertEqual(narrow.overall()[3:], (4, 6))
        self.assertEqual(
            (narrow.total_ss, narrow.total_sn, narrow.total_nn), (8, 8, 12)
        )
        self.assertAlmostEqual(narrow.overall_design_effect(), np.sqrt(2))
        self.assertEqual(sum(narrow.denominator.values()), 2)
        self.assertEqual(
            narrow.coverage(),
            {
                "bin_min_ntracks": 4,
                "bin_max_ntracks": 13,
                "binned_numerator": 2,
                "binned_denominator": 2,
                "underflow_numerator": 1,
                "underflow_denominator": 2,
                "underflow_rate": 0.5,
                "underflow_fraction": 1 / 3,
                "overflow_numerator": 1,
                "overflow_denominator": 2,
                "overflow_rate": 0.5,
                "overflow_fraction": 1 / 3,
            },
        )
        for ordered in (parts, list(reversed(parts))):
            merged = BinnedCounts.from_dict(
                narrow.bins, BinnedCounts.merge_dicts(ordered)
            )
            self.assertEqual(merged.to_dict(), narrow.to_dict())
            self.assertEqual(merged.overall(), narrow.overall())
            self.assertEqual(merged.coverage(), narrow.coverage())

    def test_coverage_uses_actual_bin_edges_and_keeps_zero_successes(self):
        counts = BinnedCounts(make_bins(4, 200, 5))
        for n in (0, 2, 3, 4, 200, 201, 203, 204):
            counts.add(n, False)
        counts.end_event()
        self.assertEqual(counts.overall()[3:], (0, 8))
        coverage = counts.coverage()
        self.assertEqual(coverage["bin_max_ntracks"], 203)
        self.assertEqual(coverage["underflow_denominator"], 3)
        self.assertEqual(coverage["overflow_denominator"], 1)
        self.assertEqual(coverage["binned_denominator"], 4)
        self.assertEqual(coverage["underflow_rate"], 0)
        empty = BinnedCounts([(4, 4)]).coverage()
        self.assertTrue(np.isnan(empty["underflow_rate"]))
        self.assertTrue(np.isnan(empty["underflow_fraction"]))

    def test_bins_must_partition_one_contiguous_range(self):
        for bins in ([], [(4, 3)], [(4, 5), (5, 8)], [(4, 5), (7, 8)]):
            with (
                self.subTest(bins=bins),
                self.assertRaisesRegex(ValueError, "contiguous"),
            ):
                BinnedCounts(bins)


class OverallOutputTests(unittest.TestCase):
    def test_workers_and_bins_agree_on_full_population(self):
        def vertex(tracks, z=0):
            return dict(tracks=tracks, z=z, t=0, z_err=0.1, t_err=0.01)

        events = [
            (
                [vertex(range(4)), vertex(range(10, 16), 5), vertex(range(20, 40), 10)],
                [
                    vertex(range(3)),
                    vertex(range(10, 16), 5),
                    vertex(range(20, 40), 10),
                    vertex([100, 101], 50),
                    vertex(range(200, 404), 100),
                    vertex([]),
                ],
            ),
            (
                [vertex(range(10), 1)],
                [
                    vertex(range(5), 0.99),
                    vertex(range(5, 10), 1.01),
                    vertex(range(100, 103), 99),
                ],
            ),
            (
                [vertex(range(5)), vertex(range(5, 10), 0.01)],
                [vertex(range(10), 0.005)],
            ),
            ([], [vertex([0, 1]), vertex(range(10, 215), 5)]),
            ([vertex([]), vertex(range(200, 410))], []),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdf5 = root / "events.h5"
            write_events(hdf5, events)
            settings = AnalysisConfig(
                analyses=RATE_ORDER,
                method_name="Population",
                plot_min_ntracks=4,
                plot_max_ntracks=8,
                plot_bin_width_ntracks=1,
                workers=1,
                batches=1,
            )
            # Hand-counted fixture: 8 true vertices (including one empty),
            # 11 nonempty recs; 3 matched true, 5 fake recs, 2 merged true
            # in 1 rec, and 1 split true supplying 2 recs. These expectations
            # are independent of both the counter and the optimizer.
            expected = {
                "efficiency_true_overall": (3, 8),
                "fake_rate_rec_overall": (5, 11),
                "merge_rate_true_overall": (2, 8),
                "merge_rate_rec_overall": (1, 11),
                "split_rate_true_overall": (1, 8),
                "split_rate_rec_overall": (2, 11),
            }
            variants = {
                "serial": settings,
                "parallel": replace(settings, workers=2, batches=3),
                "wide": replace(
                    settings, plot_min_ntracks=1, plot_max_ntracks=220, batches=2
                ),
            }
            # Exercise real reading, spawned workers, reduction and CSV writing.
            # Plot invariance follows from the unchanged binned tables; real CLI
            # rendering is checked separately on the existing physics sample.
            with (
                patch("pvfit.analysis.driver.plot_efficiency_lhcb2"),
                patch("pvfit.analysis.driver.plot_rate_lhcb2"),
                patch("pvfit.analysis.driver.plot_fake_rate_comparison_lhcb2"),
            ):
                for label, config in variants.items():
                    run_analysis(str(hdf5), str(root / label), config)
            files = sorted((root / "serial").glob("*.csv"))
            self.assertEqual(len(files), 38)  # rate tables plus merge-pair diagnostic
            pairs = root / "serial" / "merge_distance_population_track_based.csv"
            self.assertEqual(len(read_csv(pairs)), 1)
            for path in files:
                self.assertEqual(
                    path.read_bytes(),
                    (root / "parallel" / path.name).read_bytes(),
                    path.name,
                )
            for path in (root / "serial").glob("*_overall_*.csv"):
                row = read_csv(path)[0]
                wide = read_csv(root / "wide" / path.name)[0]
                for key in (
                    "numerator",
                    "denominator",
                    "rate",
                    "error_lower",
                    "error_upper",
                    "design_effect",
                    "population_scope",
                ):
                    self.assertEqual(row[key], wide[key], (path.name, key))
                for column in ("numerator", "denominator"):
                    self.assertEqual(
                        int(row[column]),
                        sum(
                            int(row[f"{part}_{column}"])
                            for part in ("binned", "underflow", "overflow")
                        ),
                    )
            for rate, (numerator, denominator) in expected.items():
                prefix, side, _ = rate.rsplit("_", 2)
                path = (
                    root
                    / "serial"
                    / f"{prefix}_overall_{side}_binned_population_track_based.csv"
                )
                row = read_csv(path)[0]
                self.assertEqual(int(row["numerator"]), numerator)
                self.assertEqual(int(row["denominator"]), denominator)
                self.assertEqual(float(row["rate"]), numerator / denominator)
            # Existing in-range fake bins still exclude the 2/3-member recs.
            path = root / "serial" / "fake_rate_rec_binned_population_track_based.csv"
            binned = read_csv(path)
            self.assertEqual(sum(int(row["denominator"]) for row in binned), 3)
            self.assertEqual(sum(int(row["numerator"]) for row in binned), 0)


if __name__ == "__main__":
    unittest.main()
