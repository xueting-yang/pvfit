"""Uncertainties reported by the analyses are the documented estimators."""

import unittest

import numpy as np
from scipy.stats import norm

from pvfit.analysis.moments import Moments
from pvfit.analysis.residual_accumulators import _summary_from
from pvfit.analysis.utils import (
    ONE_SIGMA_TAIL,
    calculate_clopper_pearson_errors,
    mean_with_error,
    std_with_error,
)


class StdWithErrorTests(unittest.TestCase):
    def test_gaussian_sample_matches_sigma_over_sqrt_2n(self):
        rng = np.random.default_rng(3)
        n, sigma = 20000, 0.05
        std, error = std_with_error(rng.normal(0.0, sigma, n))
        self.assertAlmostEqual(std, sigma, delta=0.002)
        self.assertAlmostEqual(error, sigma / np.sqrt(2 * n), delta=0.05 * error)

    def test_heavy_tails_widen_the_uncertainty(self):
        rng = np.random.default_rng(4)
        n = 20000
        _, gaussian_error = std_with_error(rng.normal(0.0, 1.0, n))
        _, heavy_error = std_with_error(rng.standard_t(5, n) / np.sqrt(5 / 3))
        self.assertGreater(heavy_error, 1.3 * gaussian_error)

    def test_bias_is_not_folded_into_the_std(self):
        rng = np.random.default_rng(5)
        residuals = rng.normal(0.02, 0.05, 50000)
        std, _ = std_with_error(residuals)
        mean, mean_error = mean_with_error(residuals)
        self.assertAlmostEqual(std, 0.05, delta=0.001)
        self.assertAlmostEqual(mean, 0.02, delta=0.001)
        self.assertAlmostEqual(mean_error, 0.05 / np.sqrt(50000), delta=1e-5)
        self.assertGreater(np.sqrt(np.mean(residuals**2)), std)

    def test_degenerate_and_tiny_samples(self):
        self.assertEqual(std_with_error([]), (0.0, 0.0))
        self.assertEqual(std_with_error([1.0]), (0.0, 0.0))
        self.assertEqual(std_with_error([2.0, 2.0, 2.0]), (0.0, 0.0))
        self.assertEqual(mean_with_error([]), (None, None))
        self.assertEqual(mean_with_error([1.5]), (1.5, None))


class ClopperPearsonTests(unittest.TestCase):
    def test_one_sigma_tail_is_exact(self):
        self.assertAlmostEqual(ONE_SIGMA_TAIL, norm.cdf(-1), places=12)
        self.assertAlmostEqual(ONE_SIGMA_TAIL, 0.158655, places=6)

    def test_bounds_and_edges(self):
        lower, upper = calculate_clopper_pearson_errors(0, 10, 0.0)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)
        lower, upper = calculate_clopper_pearson_errors(10, 10, 1.0)
        self.assertGreater(lower, 0.0)
        self.assertEqual(upper, 0.0)
        self.assertEqual(calculate_clopper_pearson_errors(0, 0, 0.0), (0.0, 0.0))
        # Large n: the interval approaches the normal approximation.
        k, n = 7000, 10000
        lower, upper = calculate_clopper_pearson_errors(k, n, k / n)
        expected = np.sqrt(0.7 * 0.3 / n)
        self.assertAlmostEqual(lower, expected, delta=0.02 * expected)
        self.assertAlmostEqual(upper, expected, delta=0.02 * expected)


def pull_summary(values, *, population_std=False):
    values = np.asarray(values)
    quantiles = np.quantile(values, [0.16, 0.84]) if len(values) else (None, None)
    return _summary_from(Moments.of(values), quantiles, population_std=population_std)


class PullSummaryTests(unittest.TestCase):
    def test_track_pull_statistics_carry_errors(self):
        stats = pull_summary([1.0, 2.0, 3.0, 4.0], population_std=True)
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["mean_error"], np.std([1, 2, 3, 4], ddof=1) / 2)
        self.assertIsNotNone(stats["std_error"])
        single = pull_summary([4.0], population_std=True)
        self.assertEqual(single["mean"], 4.0)
        self.assertIsNone(single["mean_error"])
        self.assertIsNone(single["std_error"])
        empty = pull_summary([], population_std=True)
        self.assertIsNone(empty["mean_error"])
        self.assertIsNone(empty["std_error"])

    def test_vertex_pull_summary_uses_all_entries(self):
        rng = np.random.default_rng(6)
        pulls = np.concatenate([rng.normal(0, 1, 5000), [50.0]])  # one far outlier
        summary = pull_summary(pulls)
        self.assertEqual(summary["n"], 5001)
        self.assertIsNotNone(summary["std_error"])
        self.assertAlmostEqual(summary["central68_half_width"], 1.0, delta=0.05)
        self.assertGreater(summary["std"], 1.0)  # the outlier is kept


if __name__ == "__main__":
    unittest.main()
