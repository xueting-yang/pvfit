"""Rates binned in track multiplicity: the machinery every rate analysis shares.

An analysis module decides which vertices count as successes (that is the
physics: overlap-based or distance-based matching) and returns RateResult
objects. This module provides the bins, the counters with Clopper-Pearson
errors, the HDF5 event access, and the plot / CSV / summary writers that
run_analysis applies to every result in the same way.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.ticker import (
    AutoMinorLocator,
    MaxNLocator,
    MultipleLocator,
    StrMethodFormatter,
)

from .utils import calculate_clopper_pearson_errors, find_best_match


def make_bins(min_ntracks, max_ntracks, width):
    """Equal-width, inclusive (start, end) bins in track multiplicity."""
    num_bins = (max_ntracks - min_ntracks) // width + 1
    return [
        (min_ntracks + i * width, min_ntracks + (i + 1) * width - 1)
        for i in range(num_bins)
    ]


def find_bin(bins, n_tracks):
    """The (start, end) bin containing n_tracks, or None if outside all bins."""
    for bin_ in bins:
        if bin_[0] <= n_tracks <= bin_[1]:
            return bin_
    return None


@dataclass
class EventVertices:
    """One event's true and reconstructed vertices as the analyses see them.

    Coordinates and errors are arrays indexed like the track sets; a field the
    file does not store is None (only z, t and track_indices are required).
    """

    event_id: int
    true_tracks: list  # set of track row indices per true vertex
    true_x: np.ndarray
    true_y: np.ndarray
    true_z: np.ndarray
    true_t: np.ndarray
    rec_tracks: list  # set of track row indices per reconstructed vertex
    rec_x: np.ndarray
    rec_y: np.ndarray
    rec_z: np.ndarray
    rec_t: np.ndarray
    rec_x_err: np.ndarray
    rec_y_err: np.ndarray
    rec_z_err: np.ndarray
    rec_t_err: np.ndarray


def track_matched_pairs(event, recall_threshold, purity_threshold):
    """(true_id, rec_id) for every reconstructed vertex with an overlap match.

    Each reconstructed vertex pairs with at most one true vertex (see
    utils.find_best_match); a true vertex may appear in several pairs.
    """
    true_vertex_info = {
        true_id: {"track_indices": tracks}
        for true_id, tracks in enumerate(event.true_tracks)
    }
    for rec_id, rec_tracks in enumerate(event.rec_tracks):
        if not rec_tracks:
            continue
        true_id = find_best_match(
            rec_tracks, true_vertex_info, recall_threshold, purity_threshold
        )
        if true_id is not None:
            yield true_id, rec_id


class BinnedCounts:
    """Full-population and binned counts, with event-level error statistics.

    Overall counts include every object passed to ``add``, independently of
    the plotting bins. Underflow and overflow counts document the populations
    outside those bins without changing the per-bin tables or figures.

    Vertices of one event are not independent Bernoulli trials: a merge marks
    two true vertices at once, a split two reconstructed fragments, a busy
    event can produce several fakes. ``add`` counts objects; ``end_event``
    closes the current event and accumulates, per bin and overall, the sums
    Σs_e², Σs_e·n_e and Σn_e² of its successes s_e and trials n_e. From them
    the ratio-estimator (cluster) variance Σ(s_e − p̂·n_e)² / N² is compared
    with the binomial variance p̂(1−p̂)/N; the ratio of their square roots is
    the *design effect*, by which the Clopper-Pearson interval is widened (never
    narrowed). Without ``end_event`` calls the design effect is 1 and the
    intervals are the plain binomial ones.
    """

    _STATS = ("sum_ss", "sum_sn", "sum_nn")
    _TOTALS = (
        "total_numerator",
        "total_denominator",
        "total_ss",
        "total_sn",
        "total_nn",
        "underflow_numerator",
        "underflow_denominator",
        "overflow_numerator",
        "overflow_denominator",
    )

    def __init__(self, bins):
        self.bins = list(bins)
        if not self.bins or any(
            end < start or (i and start != self.bins[i - 1][1] + 1)
            for i, (start, end) in enumerate(self.bins)
        ):
            raise ValueError("Rate bins must be nonempty, ordered and contiguous")
        self.denominator = dict.fromkeys(self.bins, 0)
        self.numerator = dict.fromkeys(self.bins, 0)
        self.sum_ss = dict.fromkeys(self.bins, 0)
        self.sum_sn = dict.fromkeys(self.bins, 0)
        self.sum_nn = dict.fromkeys(self.bins, 0)
        for name in self._TOTALS:
            setattr(self, name, 0)
        self._event_total_s = self._event_total_n = 0
        self._event_s = {}
        self._event_n = {}

    def add(self, n_tracks, success):
        """Count every object overall, and in its bin when one exists."""
        success = int(bool(success))
        self.total_denominator += 1
        self.total_numerator += success
        self._event_total_n += 1
        self._event_total_s += success
        if n_tracks < self.bins[0][0]:
            self.underflow_denominator += 1
            self.underflow_numerator += success
            return
        if n_tracks > self.bins[-1][1]:
            self.overflow_denominator += 1
            self.overflow_numerator += success
            return
        bin_ = find_bin(self.bins, n_tracks)
        if bin_ is not None:
            self.denominator[bin_] += 1
            self._event_n[bin_] = self._event_n.get(bin_, 0) + 1
            if success:
                self.numerator[bin_] += 1
                self._event_s[bin_] = self._event_s.get(bin_, 0) + 1

    def end_event(self):
        """Close the current event: fold its per-bin successes and trials into
        the cluster sums. Call once per event after all ``add`` calls."""
        for bin_, n in self._event_n.items():
            s = self._event_s.get(bin_, 0)
            self.sum_ss[bin_] += s * s
            self.sum_sn[bin_] += s * n
            self.sum_nn[bin_] += n * n
        # Include events wholly outside the bins and cross-bin correlations.
        total_s, total_n = self._event_total_s, self._event_total_n
        self.total_ss += total_s * total_s
        self.total_sn += total_s * total_n
        self.total_nn += total_n * total_n
        self._event_total_s = self._event_total_n = 0
        self._event_s = {}
        self._event_n = {}

    @staticmethod
    def design_effect(numerator, denominator, ss, sn, nn):
        """sqrt(cluster variance / binomial variance), at least 1."""
        if denominator <= 0 or numerator <= 0 or numerator >= denominator:
            return 1.0
        p = numerator / denominator
        cluster = (ss - 2.0 * p * sn + p * p * nn) / denominator**2
        binomial = p * (1.0 - p) / denominator
        if cluster <= 0.0 or binomial <= 0.0:
            return 1.0
        return max(1.0, float(np.sqrt(cluster / binomial)))

    @staticmethod
    def _rate(numerator, denominator, deff=1.0):
        rate = numerator / denominator if denominator > 0 else 0.0
        lower, upper = calculate_clopper_pearson_errors(numerator, denominator, rate)
        return rate, deff * lower, deff * upper

    def bin_design_effect(self, bin_):
        return self.design_effect(
            self.numerator[bin_],
            self.denominator[bin_],
            self.sum_ss[bin_],
            self.sum_sn[bin_],
            self.sum_nn[bin_],
        )

    def overall_design_effect(self):
        return self.design_effect(
            self.total_numerator,
            self.total_denominator,
            self.total_ss,
            self.total_sn,
            self.total_nn,
        )

    def rate(self, bin_):
        """(rate, error_lower, error_upper) of one bin, errors widened by the
        bin's design effect."""
        return self._rate(
            self.numerator[bin_], self.denominator[bin_], self.bin_design_effect(bin_)
        )

    def overall(self):
        """Rate, errors and counts for all objects, including under/overflow."""
        numerator = self.total_numerator
        denominator = self.total_denominator
        return (
            *self._rate(numerator, denominator, self.overall_design_effect()),
            numerator,
            denominator,
        )

    def coverage(self):
        """Counts and rates outside the actual inclusive bin boundaries.

        ``fraction`` is the share of the full denominator in that region;
        ``rate`` is its own success fraction, undefined for an empty region.
        """
        result = {
            "bin_min_ntracks": self.bins[0][0],
            "bin_max_ntracks": self.bins[-1][1],
            "binned_numerator": sum(self.numerator.values()),
            "binned_denominator": sum(self.denominator.values()),
        }
        for region in ("underflow", "overflow"):
            numerator = getattr(self, f"{region}_numerator")
            denominator = getattr(self, f"{region}_denominator")
            result.update(
                {
                    f"{region}_numerator": numerator,
                    f"{region}_denominator": denominator,
                    f"{region}_rate": numerator / denominator
                    if denominator
                    else float("nan"),
                    f"{region}_fraction": denominator / self.total_denominator
                    if self.total_denominator
                    else float("nan"),
                }
            )
        return result

    def table(self):
        """One row per bin, including empty bins."""
        rows = []
        for start, end in self.bins:
            rate, lower, upper = self.rate((start, end))
            rows.append(
                {
                    "bin_start": start,
                    "bin_end": end,
                    "bin_center": (start + end) / 2,
                    "rate": rate,
                    "error_lower": lower,
                    "error_upper": upper,
                    "denominator": self.denominator[start, end],
                    "numerator": self.numerator[start, end],
                    "design_effect": self.bin_design_effect((start, end)),
                }
            )
        return pd.DataFrame(rows)

    # ---- batch serialisation -------------------------------------------------

    def to_dict(self):
        """Integer sums per bin and overall, in bin order; JSON-serialisable."""
        return {
            "numerator": [self.numerator[b] for b in self.bins],
            "denominator": [self.denominator[b] for b in self.bins],
            **{
                name: [getattr(self, name)[b] for b in self.bins]
                for name in self._STATS
            },
            "totals": [int(getattr(self, name)) for name in self._TOTALS],
        }

    @classmethod
    def from_dict(cls, bins, data):
        counts = cls(bins)
        for name in ("numerator", "denominator", *cls._STATS):
            values = data[name]
            target = getattr(counts, name)
            for bin_, value in zip(counts.bins, values, strict=True):
                target[bin_] = int(value)
        for name, value in zip(cls._TOTALS, data["totals"], strict=True):
            setattr(counts, name, int(value))
        return counts

    @staticmethod
    def merge_dicts(parts):
        """Sum the serialised sums of several batches (all exact integers)."""
        parts = list(parts)
        merged = {}
        for name in ("numerator", "denominator", *BinnedCounts._STATS, "totals"):
            merged[name] = [
                int(sum(values))
                for values in zip(*(p[name] for p in parts), strict=True)
            ]
        return merged


@dataclass
class RateResult:
    """A binned rate plus what it measures, ready to be written out."""

    metric: str  # e.g. "efficiency", "fake_rate"; used in file names
    binned_by: str  # "true" or "rec": which vertices the bins count
    counts: BinnedCounts
    parameters: dict = field(default_factory=dict)  # matching settings, for the CSV
    label: str = ""  # plot title / legend text for this definition
    matching: str = "track"  # "track" (overlap), "geo" or "reso" (distance)

    # Optional population histogram, independent of rate numerator/denominator.
    reconstructed_counts: dict[tuple[int, int], int] | None = None

    @property
    def stem(self):
        return f"{self.metric}_{self.binned_by}_binned"

    def method_name(self, base_name):
        """Method label carried in file names, e.g. PVfit_Track_Based."""
        return f"{base_name}_{self.matching.title()}_Based"

    def summary_line(self, base_name):
        rate, lower, upper, numerator, denominator = self.counts.overall()
        return (
            f"{self.metric:<11} {self.binned_by:>4}-binned  {self.matching:<5}  "
            f"{rate:.3f} +{upper:.3f}/-{lower:.3f}  ({numerator}/{denominator})"
        )


def _slug(method_name):
    return method_name.lower().replace(" ", "_")


def write_rate_csv(result, output_dir, base_name):
    """Write <stem>_<slug>.csv (per bin) and <metric>_overall_<...>.csv."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_name = result.method_name(base_name)
    slug = _slug(method_name)

    result.counts.table().to_csv(output_path / f"{result.stem}_{slug}.csv", index=False)

    rate, lower, upper, numerator, denominator = result.counts.overall()
    overall = {
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "error_lower": lower,
        "error_upper": upper,
        "design_effect": result.counts.overall_design_effect(),
        **result.parameters,
        "method_name": method_name,
        "population_scope": "all_stored_true"
        if result.binned_by == "true"
        else "all_nonempty_rec",
        **result.counts.coverage(),
    }
    pd.DataFrame([overall]).to_csv(
        output_path / f"{result.metric}_overall_{result.binned_by}_binned_{slug}.csv",
        index=False,
    )


def write_vertex_track_count_csv(
    true_counts, reconstructed_counts, output_dir, base_name
):
    """Save both raw populations once, with no matching suffix or display scaling."""
    if true_counts.keys() != reconstructed_counts.keys():
        raise ValueError(
            "True and reconstructed track-count distributions need the same bins"
        )
    columns = [
        "bin_start",
        "bin_end",
        "bin_center",
        "true_count",
        "reconstructed_count",
    ]
    rows = [
        (
            start,
            end,
            (start + end) / 2,
            true_counts[start, end],
            reconstructed_counts[start, end],
        )
        for start, end in sorted(true_counts)
    ]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / f"vertex_track_count_distribution_{_slug(base_name)}.csv"
    pd.DataFrame(rows, columns=columns).to_csv(csv_file, index=False)
    return csv_file


_VERTEX_WORD = {"true": "true", "rec": "reconstructed"}


def plot_efficiency_lhcb2(result, output_dir, base_name, max_ntracks=150):
    """Efficiency with asymmetric 1-sigma bars and a reconstructed-vertex spectrum."""
    return plot_rate_lhcb2(result, output_dir, base_name, max_ntracks)


def _rate_display_ymax(table):
    """Leave headroom above central values, independently of long intervals."""
    target = max(0.001, 1.25 * float(table["rate"].max()))
    step = 10 ** np.floor(np.log10(target)) / 2
    return min(1.05, float(np.ceil(target / step) * step))


def plot_rate_lhcb2(result, output_dir, base_name, max_ntracks=150, ymax=None):
    """Compact rate plot with CP bars and an optional vertex spectrum.

    Efficiency uses a separately counted reconstructed-vertex spectrum;
    other true-binned rates show their truth population. Keep complete bins.
    Large intervals remain intact and get an overflow marker when they
    extend beyond the displayed y range.
    """
    table = result.counts.table()
    table = table[(table["denominator"] > 0) & (table["bin_end"] <= max_ntracks)]
    if table.empty:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"{result.stem}_{_slug(result.method_name(base_name))}_lhcb2"
    is_efficiency = result.metric == "efficiency"
    spectrum = None
    if is_efficiency:
        if result.reconstructed_counts is None:
            raise ValueError("Efficiency plots require reconstructed-vertex counts")
        spectrum = pd.DataFrame(
            [
                (start, end, count)
                for (start, end), count in result.reconstructed_counts.items()
            ],
            columns=["bin_start", "bin_end", "count"],
        )
        spectrum = spectrum[
            (spectrum["count"] > 0) & (spectrum["bin_end"] <= max_ntracks)
        ].sort_values("bin_start")
        spectrum_label = "Reconstructed-vertex distribution (scaled)"
    elif result.binned_by == "true":
        spectrum = table.rename(columns={"denominator": "count"})
        spectrum_label = "True-vertex distribution (scaled)"
    label = f"{base_name}: {result.label}"
    if result.matching == "geo" and "m_r" in result.parameters:
        label = rf"{base_name}: geometric matching ($m_r={result.parameters['m_r']:g}$)"

    return _plot_rate_series_lhcb2(
        [(table, label, "#2446b5", "o")],
        output_path,
        stem,
        result.metric,
        result.binned_by,
        max_ntracks,
        ymax,
        spectrum,
        spectrum_label if spectrum is not None else None,
    )


def plot_fake_rate_comparison_lhcb2(
    track_result, reso_result, output_dir, base_name, max_ntracks=150, ymax=None
):
    """Overlay track-overlap and resolution-scaled fake rates for one sample.

    Both results must count the same reconstructed vertices in the same bins.
    Each result counts rec vertices with no qualifying truth candidate under
    its own definition, without excluding track-based merges from geometric fake.
    Each curve retains its own asymmetric
    Clopper-Pearson bars; neither curve gets a population histogram. Display
    limits leave the complete counts and the individual rate CSVs unchanged.
    """
    for result, matching in ((track_result, "track"), (reso_result, "reso")):
        if (result.metric, result.binned_by, result.matching) != (
            "fake_rate",
            "rec",
            matching,
        ):
            raise ValueError("Fake comparison requires track and reso rec fake rates")
    if (
        track_result.counts.bins != reso_result.counts.bins
        or track_result.counts.denominator != reso_result.counts.denominator
    ):
        raise ValueError("Fake comparison requires identical rec populations and bins")

    nsigma = reso_result.parameters["nsigma"]
    series = []
    for result, label, color, marker in (
        (track_result, "Track overlap", "#c53c38", "s"),
        (
            reso_result,
            rf"${nsigma:g}\sigma$ geometric (no truth in window)",
            "#2446b5",
            "o",
        ),
    ):
        table = result.counts.table()
        table = table[(table["denominator"] > 0) & (table["bin_end"] <= max_ntracks)]
        if table.empty:
            return None
        series.append((table, label, color, marker))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return _plot_rate_series_lhcb2(
        series,
        output_path,
        f"fake_rate_comparison_{_slug(base_name)}_lhcb2",
        "fake_rate",
        "rec",
        max_ntracks,
        ymax,
    )


def _plot_rate_series_lhcb2(
    series,
    output_path,
    stem,
    metric,
    binned_by,
    max_ntracks,
    ymax,
    spectrum=None,
    spectrum_label=None,
):
    """Shared LHCb2 rendering for a single rate or a same-population overlay."""
    is_efficiency = metric == "efficiency"
    table = pd.concat([entry[0] for entry in series], ignore_index=True)
    if ymax is None:
        ymax = 1.10 if is_efficiency else _rate_display_ymax(table)
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError("Rate plot ymax must be positive and finite")
    if ymax < table["rate"].max():
        raise ValueError("Rate plot ymax must include all displayed central values")
    plot_file = output_path / f"{stem}.pdf"

    with plt.rc_context():
        hep.style.use(hep.style.LHCb2)
        # Compact paper styling: thin frames and short inward ticks.
        plt.rcParams.update(
            {
                "axes.linewidth": 0.7,
                "font.weight": "normal",
                "xtick.top": False,
                "ytick.right": False,
                "xtick.major.size": 4.5,
                "ytick.major.size": 4.5,
                "xtick.minor.size": 2.2,
                "ytick.minor.size": 2.2,
                "xtick.major.width": 0.65,
                "ytick.major.width": 0.65,
                "xtick.minor.width": 0.55,
                "ytick.minor.width": 0.55,
                "xtick.major.pad": 3,
                "ytick.major.pad": 3,
            }
        )
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        if spectrum is not None and not spectrum.empty:
            # Scale only the background shape, independently of the rate.
            peak = 0.65 if is_efficiency else 0.6 * ymax
            distribution = peak * spectrum["count"] / spectrum["count"].max()
            ax.bar(
                (spectrum["bin_start"] + spectrum["bin_end"]) / 2,
                distribution,
                width=spectrum["bin_end"] - spectrum["bin_start"] + 1,
                color="0.8",
                edgecolor="none",
                linewidth=0,
                zorder=0,
                label=spectrum_label,
            )
        overflow_labeled = False
        for table, label, color, marker in series:
            ax.errorbar(
                table["bin_center"],
                table["rate"],
                yerr=table[["error_lower", "error_upper"]].to_numpy().T,
                fmt=marker,
                color=color,
                markersize=3.0,
                markerfacecolor="none",
                markeredgewidth=0.65,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.8,
                barsabove=True,
                label=label,
            )
            above = table["rate"] + table["error_upper"] > ymax
            if above.any():
                ax.plot(
                    table.loc[above, "bin_center"],
                    np.full(above.sum(), 0.97 * ymax),
                    linestyle="none",
                    marker="^",
                    markersize=3,
                    color=color,
                    label="Interval extends above axis"
                    if not overflow_labeled
                    else "_nolegend_",
                )
                overflow_labeled = True
        ax.set_xlim(0, max_ntracks)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(
            "Number of tracks per vertex"
            if is_efficiency
            else f"Number of tracks per {_VERTEX_WORD[binned_by]} vertex",
            fontsize=16,
            labelpad=3,
        )
        ax.set_ylabel(metric.replace("_", " ").capitalize(), fontsize=17, labelpad=4)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(
            MultipleLocator(0.2) if is_efficiency else MaxNLocator(nbins=5)
        )
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(False)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.025),
            fontsize=11.5,
            borderaxespad=0,
            handlelength=1.4,
            handletextpad=0.45,
        )
        fig.tight_layout(pad=0.7)
        fig.savefig(plot_file, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(
            output_path / f"{stem}.png",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        plt.close(fig)
    return plot_file
