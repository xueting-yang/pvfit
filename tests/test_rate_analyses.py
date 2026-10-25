"""Each rate analysis counts hand-built events the way its definition says."""

import csv
import tempfile
import unittest
from copy import deepcopy
from functools import partial
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from pvfit.analysis.accumulators import REGISTRY
from pvfit.analysis.binned_rate import BinnedCounts, make_bins
from pvfit.analysis.events import list_event_ids, read_events
from pvfit.analysis.run_analysis import run_analysis
from pvfit.analysis.utils import (
    claim_vertex_matches,
    match_vertices_elliptical_zt,
    match_vertices_resolution_zt,
)
from pvfit.config_loader import AnalysisConfig

BINS = make_bins(4, 13, 5)  # (4, 8) and (9, 13)
LOW, HIGH = BINS


def _rate(name, path, bins, *thresholds):
    """Exercise one rate accumulator on the hand-built events, without plotting."""
    if name.endswith("_geo"):
        keys = ("m_r", "z_scale", "t_scale", "nsigma")
    elif name == "merge_rate":
        keys = ("recall_threshold",)
    elif name == "split_rate":
        keys = ("purity_threshold",)
    else:
        keys = ("recall_threshold", "purity_threshold")
    settings = AnalysisConfig(
        analyses=(name,), **dict(zip(keys, thresholds, strict=True))
    )
    accumulator = REGISTRY[name](settings, bins)
    for event in read_events(path, list_event_ids(path)):
        accumulator.update(event, None)
    return accumulator.finalize(accumulator.partial(), None)


efficiency_track = partial(_rate, "efficiency_track")
efficiency_geo = partial(_rate, "efficiency_geo")
fake_track = partial(_rate, "fake_rate_track")
fake_geo = partial(_rate, "fake_rate_geo")
merge_track = partial(_rate, "merge_rate")
split_track = partial(_rate, "split_rate")
merge_geo = partial(_rate, "merge_rate_geo")
split_geo = partial(_rate, "split_rate_geo")


def write_events(path, events):
    """events: list of (true_vertices, rec_vertices); a vertex is a dict with
    z, t, tracks and, for rec vertices, z_err/t_err."""
    with h5py.File(path, "w") as f:
        for event_id, (true, rec) in enumerate(events):
            group = f.create_group(str(event_id))
            for name, vertices in (("true_vertices", true), ("rec_vertices", rec)):
                g = group.create_group(name)
                for field in ("x", "y", "z", "t", "z_err", "t_err"):
                    if name == "true_vertices" and field.endswith("_err"):
                        continue
                    g[field] = np.array([v.get(field, 0.0) for v in vertices])
                ds = g.create_dataset(
                    "track_indices", (len(vertices),), dtype=h5py.vlen_dtype(np.int32)
                )
                for i, v in enumerate(vertices):
                    ds[i] = np.array(sorted(v["tracks"]), dtype=np.int32)


def counts(result):
    """{bin: (numerator, denominator)} of a RateResult."""
    c = result.counts
    return {b: (c.numerator[b], c.denominator[b]) for b in c.bins}


def by_key(results):
    return {(r.matching, r.binned_by): r for r in results}


class BinnedCountsTests(unittest.TestCase):
    def test_bins_counts_and_overall(self):
        self.assertEqual(BINS, [(4, 8), (9, 13)])
        c = BinnedCounts(BINS)
        for n_tracks, success in ((4, True), (8, False), (13, True), (14, True)):
            c.add(n_tracks, success)  # 14 contributes only to the overall count
        self.assertEqual(c.numerator, {LOW: 1, HIGH: 1})
        self.assertEqual(c.denominator, {LOW: 2, HIGH: 1})
        rate, lower, upper, numerator, denominator = c.overall()
        self.assertEqual((rate, numerator, denominator), (3 / 4, 3, 4))
        self.assertGreater(lower, 0)
        self.assertGreater(upper, 0)
        self.assertEqual(list(c.table()["rate"]), [0.5, 1.0])

    def test_design_effect_widens_errors_for_correlated_pairs(self):
        # Independent trials: one object per event, design effect stays 1.
        independent = BinnedCounts(BINS)
        for i in range(400):
            independent.add(5, i % 10 == 0)
            independent.end_event()
        self.assertEqual(independent.bin_design_effect(LOW), 1.0)
        # Perfectly correlated pairs: each event contributes two identical
        # outcomes (a merge marks two true vertices, a split two fragments).
        paired = BinnedCounts(BINS)
        for i in range(200):
            for _ in range(2):
                paired.add(5, i % 10 == 0)
            paired.end_event()
        self.assertEqual(paired.numerator, independent.numerator)
        self.assertEqual(paired.denominator, independent.denominator)
        self.assertAlmostEqual(paired.bin_design_effect(LOW), 2**0.5, places=6)
        self.assertAlmostEqual(paired.overall_design_effect(), 2**0.5, places=6)
        rate_i, low_i, up_i = independent.rate(LOW)
        rate_p, low_p, up_p = paired.rate(LOW)
        self.assertEqual(rate_i, rate_p)
        self.assertAlmostEqual(low_p / low_i, 2**0.5, places=6)
        self.assertAlmostEqual(up_p / up_i, 2**0.5, places=6)
        self.assertEqual(
            list(paired.table()["design_effect"]), [paired.bin_design_effect(LOW), 1.0]
        )
        # Alternating outcomes within an event are under-dispersed; the factor
        # is clamped at 1 rather than narrowing the binomial interval.
        alternating = BinnedCounts(BINS)
        for _ in range(100):
            alternating.add(5, True)
            alternating.add(5, False)
            alternating.end_event()
        self.assertEqual(alternating.bin_design_effect(LOW), 1.0)
        # Without end_event the plain Clopper-Pearson interval is reported.
        plain = BinnedCounts(BINS)
        for i in range(20):
            plain.add(5, i % 2 == 0)
        self.assertEqual(plain.rate(LOW), BinnedCounts._rate(10, 20))

    def test_counts_serialise_and_merge_exactly(self):
        parts = []
        for seed in (1, 2, 3):
            c = BinnedCounts(BINS)
            for i in range(seed * 7):
                c.add(5 + (i % 9), (i + seed) % 3 == 0)
                c.add(5 + (i % 9), (i + seed) % 3 == 0)  # a correlated pair
                c.end_event()
            parts.append(c)
        merged = BinnedCounts.from_dict(
            BINS, BinnedCounts.merge_dicts(p.to_dict() for p in parts)
        )
        whole = BinnedCounts(BINS)
        for seed in (1, 2, 3):
            for i in range(seed * 7):
                whole.add(5 + (i % 9), (i + seed) % 3 == 0)
                whole.add(5 + (i % 9), (i + seed) % 3 == 0)
                whole.end_event()
        self.assertEqual(merged.to_dict(), whole.to_dict())
        self.assertEqual(merged.table().to_dict(), whole.table().to_dict())
        self.assertGreater(merged.overall_design_effect(), 1.3)


class TrackBasedRateTests(unittest.TestCase):
    """Thresholds 0.7/0.7. Event 0: true0 (tracks 0-9), true1 (10-15),
    true2 (16-21); rec A (0-7) matches true0, rec B (10-21) covers true1 and
    true2 with purity 0.5, rec C (22-25) shares nothing. Event 1: true0 (0-9)
    is cut in half by rec D (0-4) and rec E (5-9)."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "track.h5"
        write_events(
            cls.path,
            [
                (
                    [
                        {"tracks": range(0, 10)},
                        {"tracks": range(10, 16)},
                        {"tracks": range(16, 22)},
                    ],
                    [
                        {"tracks": range(0, 8)},
                        {"tracks": range(10, 22)},
                        {"tracks": range(22, 26)},
                    ],
                ),
                (
                    [{"tracks": range(0, 10)}],
                    [{"tracks": range(0, 5)}, {"tracks": range(5, 10)}],
                ),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_efficiency(self):
        (result,) = efficiency_track(self.path, BINS, 0.7, 0.7)
        # only true0 of event 0 is matched (rec A: recall 0.8, purity 1.0);
        # rec B fails purity, D and E fail recall
        self.assertEqual(counts(result), {LOW: (0, 2), HIGH: (1, 2)})
        self.assertEqual(result.parameters["recall_threshold"], 0.7)
        # The spectrum includes unmatched and split rec vertices by their own
        # multiplicity, independently of the true-vertex efficiency counts.
        self.assertEqual(result.reconstructed_counts, {LOW: 4, HIGH: 1})

    def test_fake_rate(self):
        (result,) = fake_track(self.path, BINS, 0.7, 0.7)
        # Only C lacks a qualifying truth candidate. B passes recall, while
        # both pure halves D and E pass purity despite failing efficiency.
        self.assertEqual(counts(result), {LOW: (1, 4), HIGH: (0, 1)})

    def test_each_rec_uses_its_own_candidates(self):
        true = [
            {"tracks": range(10)},
            {"tracks": range(10, 20)},
            {"tracks": range(20, 30)},
        ]
        rec = [
            {"tracks": range(6)},
            {"tracks": range(6, 10)},
            {"tracks": [10, 11, 12, 20, 21, 22, 30, 31, 32, 33]},
        ]
        # True0 overlaps X more than Y, but each pure 6+4 split piece has its
        # own purity match. The mixed rec reaches neither threshold to any truth.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "independent_candidates.h5"
            for ordered_true in (true, list(reversed(true))):
                for ordered_rec in permutations(rec):
                    write_events(path, [(ordered_true, ordered_rec)])
                    for bins in (BINS, [(4, 4)]):
                        with self.subTest(
                            truth_order=list(ordered_true[0]["tracks"]),
                            rec_order=[len(r["tracks"]) for r in ordered_rec],
                            bins=bins,
                        ):
                            (result,) = fake_track(path, bins, 0.7, 0.7)
                            expected = (
                                {LOW: (0, 2), HIGH: (1, 1)}
                                if bins == BINS
                                else {(4, 4): (0, 1)}
                            )
                            self.assertEqual(counts(result), expected)
            (efficiency,) = efficiency_track(path, BINS, 0.7, 0.7)
            self.assertEqual(counts(efficiency), {LOW: (0, 0), HIGH: (0, 3)})

    def test_fake_requires_shared_tracks_even_at_zero_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_shared_tracks.h5"
            write_events(
                path,
                [
                    ([{"tracks": range(10)}], [{"tracks": range(20, 24)}]),
                    ([{"tracks": []}], [{"tracks": range(4)}]),
                    ([], [{"tracks": range(4)}]),
                    ([{"tracks": range(10)}], []),
                    ([{"tracks": range(10)}], [{"tracks": []}]),
                ],
            )
            (result,) = fake_track(path, BINS, 0.0, 0.0)
            self.assertEqual(counts(result), {LOW: (3, 3), HIGH: (0, 0)})

    def test_merge_rate(self):
        results = by_key(merge_track(self.path, BINS, 0.7))
        # rec B has recall 1.0 to both true1 and true2
        self.assertEqual(counts(results["track", "rec"]), {LOW: (0, 4), HIGH: (1, 1)})
        self.assertEqual(counts(results["track", "true"]), {LOW: (2, 2), HIGH: (0, 2)})

    def test_split_rate(self):
        results = by_key(split_track(self.path, BINS, 0.7))
        # true0 of event 1 has two pure pieces D and E
        self.assertEqual(counts(results["track", "true"]), {LOW: (0, 2), HIGH: (1, 2)})
        self.assertEqual(counts(results["track", "rec"]), {LOW: (2, 4), HIGH: (0, 1)})


class GeometricMatchingTests(unittest.TestCase):
    """Fake uses candidate existence; efficiency retains unique claiming."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "claims.h5"
        self.args = (self.path, BINS, 0.8, 1.5, 0.06, 5.0)

    @staticmethod
    def rec(z, ntracks, **overrides):
        return {
            "z": z,
            "t": 0.0,
            "tracks": range(ntracks),
            "z_err": 0.1,
            "t_err": 0.005,
            **overrides,
        }

    def efficiency_claims(self, true, rec, matching):
        """Expose the efficiency assignment for winner-specific assertions."""
        true_vertices = [
            {"vertex_id": i, "z": v["z"], "t": v.get("t", 0.0)}
            for i, v in enumerate(true)
        ]
        rec_vertices = [
            {**v, "rec_vertex_id": i, "rec_z": v["z"], "rec_t": v["t"]}
            for i, v in enumerate(rec)
        ]
        if matching == "geo":
            match_vertices_elliptical_zt(rec_vertices, true_vertices, *self.args[2:5])
        else:
            match_vertices_resolution_zt(rec_vertices, true_vertices, self.args[5])
        return claim_vertex_matches(rec_vertices)

    def test_all_split_copies_with_candidates_are_nonfake(self):
        true = [{"z": 0.0, "tracks": range(10)}]
        rec = [self.rec(0.2, 5), self.rec(0.1, 10), self.rec(-0.2, 5)]
        write_events(self.path, [(true, rec)])
        fake = by_key(fake_geo(*self.args))
        efficiency = by_key(efficiency_geo(*self.args))
        split = by_key(split_geo(*self.args))
        for matching in ("geo", "reso"):
            with self.subTest(matching=matching):
                self.assertEqual(
                    counts(fake[matching, "rec"]), {LOW: (0, 2), HIGH: (0, 1)}
                )
                self.assertEqual(
                    counts(efficiency[matching, "true"]), {LOW: (0, 0), HIGH: (1, 1)}
                )
                self.assertEqual(
                    counts(split[matching, "true"]), {LOW: (0, 0), HIGH: (1, 1)}
                )
                self.assertEqual(
                    counts(split[matching, "rec"]), {LOW: (2, 2), HIGH: (1, 1)}
                )

    def test_efficiency_retries_without_erasing_merge_candidates(self):
        true = [{"z": 0.0, "tracks": range(5)}, {"z": 0.4, "tracks": range(10)}]
        rec = [self.rec(0.1, 5), self.rec(0.15, 10)]
        write_events(self.path, [(true, rec)])
        fake = by_key(fake_geo(*self.args))
        efficiency = by_key(efficiency_geo(*self.args))
        merge = by_key(merge_geo(*self.args))
        for matching in ("geo", "reso"):
            with self.subTest(matching=matching):
                # Efficiency forms two distinct pairs; fake only needs candidates.
                self.assertEqual(
                    counts(fake[matching, "rec"]), {LOW: (0, 1), HIGH: (0, 1)}
                )
                self.assertEqual(
                    counts(efficiency[matching, "true"]), {LOW: (1, 1), HIGH: (1, 1)}
                )
                # Both candidate ellipses still contain two true vertices.
                for binned_by in ("rec", "true"):
                    self.assertEqual(
                        counts(merge[matching, binned_by]),
                        {LOW: (1, 1), HIGH: (1, 1)},
                    )

    def test_efficiency_keeps_closer_pairs_and_all_candidates_are_nonfake(self):
        true = [{"z": 0.0, "tracks": range(5)}, {"z": 0.6, "tracks": range(10)}]
        rec = [
            self.rec(0.0, 5, z_err=0.12),
            self.rec(0.24, 10, z_err=0.12),
            self.rec(1.02, 6, z_err=0.12),
        ]
        # A wins true0 at distance 0. B then beats C for true1: its second
        # choice has d^2=0.36, smaller than C's first choice at d^2=0.49.
        # Efficiency leaves C unclaimed, but all three recs have a candidate.
        for ordered_rec in permutations(rec):
            write_events(self.path, [(true, ordered_rec)])
            for result in fake_geo(*self.args):
                with self.subTest(
                    matching=result.matching,
                    order=[len(r["tracks"]) for r in ordered_rec],
                ):
                    self.assertEqual(counts(result), {LOW: (0, 2), HIGH: (0, 1)})
                    expected_claims = {
                        i: 0 if len(r["tracks"]) == 5 else 1
                        for i, r in enumerate(ordered_rec)
                        if len(r["tracks"]) != 6
                    }
                    self.assertEqual(
                        self.efficiency_claims(true, ordered_rec, result.matching),
                        expected_claims,
                    )
            for result in efficiency_geo(*self.args):
                self.assertEqual(counts(result), {LOW: (1, 1), HIGH: (1, 1)})

    def test_efficiency_winner_stays_and_unclaimed_rec_is_nonfake(self):
        true = [{"z": 0.0, "tracks": range(5)}, {"z": 0.78, "tracks": range(10)}]
        rec = [self.rec(0.24, 5, z_err=0.12), self.rec(-0.3, 10, z_err=0.12)]
        write_events(self.path, [(true, rec)])
        # Efficiency keeps A with true0, leaving B unclaimed and true1
        # unmatched. Both A and B still have candidates for the fake test.
        for result in fake_geo(*self.args):
            self.assertEqual(counts(result), {LOW: (0, 1), HIGH: (0, 1)})
        for result in efficiency_geo(*self.args):
            self.assertEqual(counts(result), {LOW: (1, 1), HIGH: (0, 1)})

    def test_later_candidates_preserve_metadata_and_ignore_iteration_order(self):
        true = [
            {"vertex_id": i, "z": z, "t": 0.0} for i, z in enumerate((0.0, 0.2, 0.4))
        ]
        for matching in ("geo", "reso"):
            rec = [
                {
                    "rec_vertex_id": i,
                    "rec_z": z,
                    "rec_t": 0.0,
                    "z_err": 0.1,
                    "t_err": 0.005,
                }
                for i, z in enumerate((0.0, 0.05, 0.09))
            ]
            if matching == "geo":
                match_vertices_elliptical_zt(rec, true, 0.8, 1.5, 0.06)
            else:
                match_vertices_resolution_zt(rec, true, 5.0)
            snapshot = deepcopy(rec)
            # All prefer true0. A wins there; C beats B for true1 at a
            # smaller distance; B can still match its third candidate.
            for ordered_rec in (rec, list(reversed(rec))):
                with self.subTest(
                    matching=matching, order=ordered_rec[0]["rec_vertex_id"]
                ):
                    self.assertEqual(
                        claim_vertex_matches(ordered_rec), {0: 0, 2: 1, 1: 2}
                    )
            self.assertEqual(rec, snapshot)

    def test_later_true_displaces_owner_who_retries_and_can_be_rejected(self):
        rec = [
            {"rec_vertex_id": 10, "matched_true_distances": {0: 0.3, 1: 0.1, 3: 0.8}},
            {"rec_vertex_id": 20, "matched_true_distances": {0: 0.4, 2: 0.2}},
            {"rec_vertex_id": 30, "matched_true_distances": {0: 0.5}},
            {"rec_vertex_id": 40, "matched_true_distances": {}},
        ]
        # True0 initially claims rec10, but the later, closer true1 replaces it.
        # True0 then loses to true2 at rec20 and succeeds at its third candidate.
        # True3 has only rec10, loses there, and exhausts its candidate list.
        self.assertEqual(claim_vertex_matches(rec), {10: 1, 20: 2, 30: 0})

    def test_displaced_true_can_retry_more_than_a_hundred_candidates(self):
        candidate_count = 129
        rec = [
            {"rec_vertex_id": i, "matched_true_distances": {0: 0.75}}
            for i in range(candidate_count)
        ]
        for i in range(candidate_count - 1):
            rec[i]["matched_true_distances"][i + 1] = 0.25
        # Every rec except the last has a closer true. After displacement,
        # true0 must keep applying until it reaches candidate 129.
        expected = {i: i + 1 for i in range(candidate_count - 1)}
        expected[candidate_count - 1] = 0
        self.assertEqual(claim_vertex_matches(rec), expected)

    def test_queue_claims_match_distance_greedy_on_random_candidate_graphs(self):
        rng = np.random.default_rng(20260930)
        for case in range(300):
            true_ids = [200 + 5 * i for i in range(int(rng.integers(0, 9)))]
            rec = []
            for i in range(int(rng.integers(0, 9))):
                # Discrete distances exercise ties as well as sparse graphs.
                distances = {
                    true_id: float(rng.integers(0, 5)) / 4
                    for true_id in rng.permutation(true_ids)
                    if rng.random() < 0.45
                }
                rec.append(
                    {"rec_vertex_id": 100 + 3 * i, "matched_true_distances": distances}
                )

            # Independent reference: accept globally shortest eligible edges.
            edges = sorted(
                (distance, vertex["rec_vertex_id"], true_id)
                for vertex in rec
                for true_id, distance in vertex["matched_true_distances"].items()
            )
            expected = {}
            claimed_true = set()
            for _, rec_id, true_id in edges:
                if rec_id not in expected and true_id not in claimed_true:
                    expected[rec_id] = true_id
                    claimed_true.add(true_id)

            orders = (
                rec,
                list(reversed(rec)),
                [rec[i] for i in rng.permutation(len(rec))],
            )
            for order_id, ordered_rec in enumerate(orders):
                with self.subTest(case=case, order=order_id):
                    self.assertEqual(claim_vertex_matches(ordered_rec), expected)

    def test_efficiency_claim_uses_each_rec_uncertainty(self):
        true = [{"z": 0.0, "tracks": range(10)}]
        rec = [self.rec(0.05, 5, z_err=0.02), self.rec(0.1, 10)]
        write_events(self.path, [(true, rec)])
        fake = by_key(fake_geo(*self.args))
        # Efficiency: fixed prefers the smaller absolute distance; resolution
        # prefers 1 sigma over 2.5 sigma. Both recs pass each fake window.
        self.assertEqual(self.efficiency_claims(true, rec, "geo"), {0: 0})
        self.assertEqual(self.efficiency_claims(true, rec, "reso"), {1: 0})
        for result in fake.values():
            self.assertEqual(counts(result), {LOW: (0, 1), HIGH: (0, 1)})

    def test_efficiency_claim_distance_includes_time(self):
        true = [{"z": 0.0, "tracks": range(10)}]
        rec = [self.rec(0.0, 5, t=0.01), self.rec(0.1, 10)]
        write_events(self.path, [(true, rec)])
        for result in fake_geo(*self.args):
            self.assertEqual(counts(result), {LOW: (0, 1), HIGH: (0, 1)})
            self.assertEqual(self.efficiency_claims(true, rec, result.matching), {1: 0})

    def test_efficiency_equal_distances_keep_smaller_rec_id(self):
        true = [{"z": 0.0, "tracks": range(10)}]
        rec = [self.rec(-0.1, 5), self.rec(0.1, 10)]
        write_events(self.path, [(true, rec)])
        for result in fake_geo(*self.args):
            self.assertEqual(counts(result), {LOW: (0, 1), HIGH: (0, 1)})
            self.assertEqual(self.efficiency_claims(true, rec, result.matching), {0: 0})

    def test_fake_candidates_are_independent_of_binning(self):
        true = [{"z": 0.0, "tracks": range(10)}]
        rec = [self.rec(0.0, 14), self.rec(0.1, 5)]
        write_events(self.path, [(true, rec)])
        for bins in (BINS, make_bins(4, 18, 5)):
            for result in fake_geo(self.path, bins, *self.args[2:]):
                # The closer rec's inclusion in the plot cannot change whether
                # the 5-track rec has an in-window truth candidate.
                self.assertEqual(counts(result)[LOW], (0, 1))

    def test_bad_resolution_errors_leave_no_resolution_candidate(self):
        true = [{"z": 0.0, "tracks": range(10)}]
        for errors in (
            {"z_err": 0.0},
            {"t_err": 0.0},
            {"z_err": -0.1},
            {"t_err": float("nan")},
        ):
            with self.subTest(errors=errors):
                rec = [self.rec(0.0, 10, **errors), self.rec(0.1, 5)]
                write_events(self.path, [(true, rec)])
                fake = by_key(fake_geo(*self.args))
                self.assertEqual(
                    counts(fake["geo", "rec"]), {LOW: (0, 1), HIGH: (0, 1)}
                )
                self.assertEqual(
                    counts(fake["reso", "rec"]), {LOW: (0, 1), HIGH: (1, 1)}
                )

    def test_matching_is_event_local_and_handles_empty_vertices(self):
        true = [{"z": 0.0, "tracks": range(10)}]
        rec = [self.rec(0.1, 5), self.rec(0.0, 0)]
        write_events(
            self.path,
            [
                (true, rec),
                (true, rec),
                (true, []),
                ([], rec),
                (true, [self.rec(0.0, 0)]),
            ],
        )
        for result in fake_geo(*self.args):
            # Repeated event-local IDs retain their candidates. The fourth
            # event has no truth; empty recs never enter the denominator.
            self.assertEqual(counts(result), {LOW: (1, 3), HIGH: (0, 0)})
        for result in efficiency_geo(*self.args):
            self.assertEqual(counts(result), {LOW: (0, 0), HIGH: (2, 4)})


class DistanceBasedRateTests(unittest.TestCase):
    """Fixed ellipse: m_r=0.8, z_scale=1.5, t_scale=0.06 -> semi-axes 0.6 mm,
    0.024 ns. Resolution ellipse: 3 sigma with z_err=0.05 -> 0.15 mm.
    true0 at z=0 (10 tracks) and true1 at z=0.5 (6 tracks) sit inside one
    fixed ellipse; rec A (z=0.1, 8 tracks) and rec C (z=0.45, 5 tracks)
    each contain both in the fixed ellipse but only their nearest in the
    3 sigma one; rec B (z=10, 4 tracks) is far from everything."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "geo.h5"
        err = {"z_err": 0.05, "t_err": 0.005}
        write_events(
            cls.path,
            [
                (
                    [
                        {"z": 0.0, "t": 0.0, "tracks": range(0, 10)},
                        {"z": 0.5, "t": 0.0, "tracks": range(10, 16)},
                    ],
                    [
                        {"z": 0.1, "t": 0.0, "tracks": range(0, 8), **err},
                        {"z": 10.0, "t": 0.0, "tracks": range(20, 24), **err},
                        {"z": 0.45, "t": 0.0, "tracks": range(10, 15), **err},
                    ],
                )
            ],
        )
        cls.args = (cls.path, BINS, 0.8, 1.5, 0.06, 3.0)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_efficiency(self):
        results = by_key(efficiency_geo(*self.args))
        # A's closest is true0, C's closest is true1, under both definitions
        for matching in ("geo", "reso"):
            self.assertEqual(
                counts(results[matching, "true"]), {LOW: (1, 1), HIGH: (1, 1)}
            )
            # Three rec vertices (including unmatched B) all have 4-8 tracks;
            # truth has one vertex in each bin. Do not substitute truth counts.
            self.assertEqual(
                results[matching, "true"].reconstructed_counts, {LOW: 3, HIGH: 0}
            )

    def test_fake_rate(self):
        results = by_key(fake_geo(*self.args))
        for matching in ("geo", "reso"):
            self.assertEqual(
                counts(results[matching, "rec"]), {LOW: (1, 3), HIGH: (0, 0)}
            )
        self.assertEqual(results["reso", "rec"].parameters, {"nsigma": 3.0})

    def test_merge_rate(self):
        results = by_key(merge_geo(*self.args))
        # fixed ellipse: A and C each contain both true vertices
        self.assertEqual(counts(results["geo", "rec"]), {LOW: (2, 3), HIGH: (0, 0)})
        self.assertEqual(counts(results["geo", "true"]), {LOW: (1, 1), HIGH: (1, 1)})
        # 3 sigma: each rec contains only its nearest true vertex
        self.assertEqual(counts(results["reso", "rec"]), {LOW: (0, 3), HIGH: (0, 0)})
        self.assertEqual(counts(results["reso", "true"]), {LOW: (0, 1), HIGH: (0, 1)})

    def test_split_rate(self):
        results = by_key(split_geo(*self.args))
        # fixed ellipse: each true vertex lies inside A's and C's ellipse
        self.assertEqual(counts(results["geo", "true"]), {LOW: (1, 1), HIGH: (1, 1)})
        self.assertEqual(counts(results["geo", "rec"]), {LOW: (2, 3), HIGH: (0, 0)})
        self.assertEqual(counts(results["reso", "true"]), {LOW: (0, 1), HIGH: (0, 1)})
        self.assertEqual(counts(results["reso", "rec"]), {LOW: (0, 3), HIGH: (0, 0)})

    def test_standalone_distribution_is_raw_and_independent_of_analysis(self):
        selections = (
            ("efficiency_track",),
            ("efficiency_geo",),
            ("fake_rate_geo",),
            ("merge_rate", "split_rate"),
            (
                "efficiency_track",
                "efficiency_geo",
                "fake_rate_track",
                "fake_rate_geo",
                "merge_rate",
                "split_rate",
                "merge_rate_geo",
                "split_rate_geo",
            ),
        )
        for analyses in selections:
            with (
                self.subTest(analyses=analyses),
                tempfile.TemporaryDirectory() as directory,
                patch("pvfit.analysis.driver.plot_efficiency_lhcb2"),
                patch("pvfit.analysis.driver.plot_rate_lhcb2"),
                patch(
                    "pvfit.analysis.driver.plot_fake_rate_comparison_lhcb2"
                ) as compare_fake,
            ):
                run_analysis(
                    str(self.path),
                    directory,
                    AnalysisConfig(
                        analyses=analyses,
                        method_name="External Reco",
                        plot_min_ntracks=4,
                        plot_max_ntracks=18,
                        plot_bin_width_ntracks=5,
                        # Display limits must not truncate the saved distribution.
                        efficiency_plot_max_ntracks=8,
                        rate_plot_max_ntracks=8,
                    ),
                )
                if {"fake_rate_track", "fake_rate_geo"}.issubset(analyses):
                    compare_fake.assert_called_once()
                    track, reso, _, method, display_max = compare_fake.call_args.args
                    self.assertEqual((track.matching, reso.matching), ("track", "reso"))
                    self.assertEqual(track.counts.denominator, reso.counts.denominator)
                    self.assertEqual((method, display_max), ("External Reco", 8))
                else:
                    compare_fake.assert_not_called()
                files = list(
                    Path(directory).glob("vertex_track_count_distribution_*.csv")
                )
                self.assertEqual(
                    [p.name for p in files],
                    ["vertex_track_count_distribution_external_reco.csv"],
                )
                with files[0].open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(
                    list(rows[0]),
                    [
                        "bin_start",
                        "bin_end",
                        "bin_center",
                        "true_count",
                        "reconstructed_count",
                    ],
                )
                # Includes unmatched rec B and the empty final bin, without
                # matching cuts, display scaling, or true/rec multiplicity mixing.
                self.assertEqual(
                    [[row[key] for key in rows[0]] for row in rows],
                    [
                        ["4", "8", "6.0", "1", "3"],
                        ["9", "13", "11.0", "1", "0"],
                        ["14", "18", "16.0", "0", "0"],
                    ],
                )
                self.assertEqual(
                    len(list(Path(directory).glob("efficiency_*binned*.csv"))),
                    2 * int("efficiency_track" in analyses)
                    + 4 * int("efficiency_geo" in analyses),
                )


if __name__ == "__main__":
    unittest.main()
