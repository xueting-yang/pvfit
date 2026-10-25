"""Central moments up to fourth order that merge exactly across batches.

A :class:`Moments` holds ``n``, the mean and the central power sums M2, M3, M4
of the values seen so far. Two of them combine with the pairwise update
formulas (Chan, Golub, LeVeque; Pébay 2008), so a sample split over any
number of batches, merged in any order, gives the same mean, sample variance
and fourth moment as the whole sample up to floating-point rounding. Nothing
here averages per-batch standard deviations.

The derived quantities reproduce ``pvfit.analysis.utils``:

- ``mean_with_error``: mean and s/sqrt(n) with ddof = 1
- ``std_with_error``: s (ddof = 1) and sqrt(Var(s^2) / (4 s^2)) with
  Var(s^2) = (1/n) [m4 - ((n-3)/(n-1)) s^4], m4 = M4 / n
- ``population_std``: sqrt(M2 / n) (ddof = 0), the ``std`` of the track pull
  summary
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0

    @classmethod
    def of(cls, values):
        """Moments of an array, computed in float64 with a two-pass formula."""
        values = np.asarray(values, dtype=np.float64).ravel()
        n = int(values.size)
        if n == 0:
            return cls()
        mean = float(np.mean(values))
        centered = values - mean
        squared = centered * centered
        return cls(
            n=n,
            mean=mean,
            m2=float(np.sum(squared)),
            m3=float(np.sum(squared * centered)),
            m4=float(np.sum(squared * squared)),
        )

    def update(self, values):
        """Fold an array into this accumulator (merge with its moments)."""
        self.merge(Moments.of(values))
        return self

    def merge(self, other):
        """Combine with the moments of a disjoint sample, in place."""
        if other.n == 0:
            return self
        if self.n == 0:
            self.n, self.mean = other.n, other.mean
            self.m2, self.m3, self.m4 = other.m2, other.m3, other.m4
            return self
        na, nb = float(self.n), float(other.n)
        n = na + nb
        delta = other.mean - self.mean
        delta2 = delta * delta
        m2 = self.m2 + other.m2 + delta2 * na * nb / n
        m3 = (
            self.m3
            + other.m3
            + delta2 * delta * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * other.m2 - nb * self.m2) / n
        )
        m4 = (
            self.m4
            + other.m4
            + delta2 * delta2 * na * nb * (na * na - na * nb + nb * nb) / (n**3)
            + 6.0 * delta2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * delta * (na * other.m3 - nb * self.m3) / n
        )
        self.mean = self.mean + delta * nb / n
        self.n, self.m2, self.m3, self.m4 = int(n), m2, m3, m4
        return self

    @classmethod
    def merged(cls, parts):
        total = cls()
        for part in parts:
            total.merge(part)
        return total

    # ---- derived quantities, matching pvfit.analysis.utils ----------------

    def mean_with_error(self):
        """(mean, s / sqrt(n)); error None below two values, both None if empty."""
        if self.n == 0:
            return None, None
        if self.n < 2:
            return self.mean, None
        sample_std = np.sqrt(self.m2 / (self.n - 1))
        return self.mean, float(sample_std / np.sqrt(self.n))

    def std_with_error(self):
        """Sample standard deviation and its fourth-moment uncertainty.

        Fewer than two values give (0.0, 0.0); a degenerate sample (s, 0.0).
        """
        n = self.n
        if n < 2:
            return 0.0, 0.0
        sample_var = self.m2 / (n - 1)
        m4 = self.m4 / n
        var_of_sample_var = (1 / n) * (m4 - ((n - 3) / (n - 1)) * sample_var**2)
        if sample_var > 0 and var_of_sample_var > 0:
            return float(np.sqrt(sample_var)), float(
                np.sqrt(var_of_sample_var / (4 * sample_var))
            )
        return float(np.sqrt(sample_var)), 0.0

    def population_std(self):
        """sqrt(M2 / n), the ddof = 0 standard deviation; None if empty."""
        if self.n == 0:
            return None
        return float(np.sqrt(self.m2 / self.n))

    def to_dict(self):
        return {
            "n": self.n,
            "mean": self.mean,
            "m2": self.m2,
            "m3": self.m3,
            "m4": self.m4,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(int(data["n"]), data["mean"], data["m2"], data["m3"], data["m4"])
