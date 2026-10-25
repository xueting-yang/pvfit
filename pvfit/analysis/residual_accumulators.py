"""Residual accumulators: vertex resolution, fitted-vertex pulls and input-track pulls.

They keep moments per column in memory and spill the raw observations to the
batch's column store in their source dtype. ``finalize`` derives means,
standard deviations and their errors from the merged moments, exact quantiles
and histogram counts from the columns; ``write`` draws the figures from those
counts and writes the CSV/JSON summaries.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .accumulator_base import Accumulator
from .binned_rate import find_bin, track_matched_pairs
from .moments import Moments
from .resolution_output import (
    COORDS,
    plot_resolutions,
    prepare_plot_data,
    write_resolution_csv,
)

C_MM_PER_NS = 299.792458


def _merge_moments(parts):
    return Moments.merged(Moments.from_dict(p) for p in parts).to_dict()


def _pairs(event, settings):
    pairs = list(
        track_matched_pairs(event, settings.recall_threshold, settings.purity_threshold)
    )
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    true_ids, rec_ids = zip(*pairs, strict=True)
    return np.asarray(true_ids, dtype=np.int64), np.asarray(rec_ids, dtype=np.int64)


def _bin_key(bin_):
    return f"{bin_[0]}_{bin_[1]}"


class Resolution(Accumulator):
    """Residuals rec − true of overlap-matched pairs, binned by true track count
    and windowed per coordinate, as analyze_resolution does.

    State: moments per (coordinate, bin); raw residuals per (coordinate, bin)
    in the column ``resolution/<coord>/<start>_<end>``, source dtype.
    """

    name = "resolution"

    def __init__(self, settings, bins, **extra):
        super().__init__(settings, bins, **extra)
        self.moments = {coord: {b: Moments() for b in self.bins} for coord in COORDS}
        self.ranges = {
            coord: getattr(settings, f"{coord}_residual_range") for coord in COORDS
        }

    @staticmethod
    def column(coord, bin_):
        return f"resolution/{coord}/{_bin_key(bin_)}"

    def update(self, event, store):
        true_ids, rec_ids = _pairs(event, self.settings)
        if not len(true_ids):
            return
        bins_of_pair = [
            find_bin(self.bins, len(event.true_tracks[t])) for t in true_ids
        ]
        keep = np.array([b is not None for b in bins_of_pair], dtype=bool)
        if not keep.any():
            return
        true_ids, rec_ids = true_ids[keep], rec_ids[keep]
        pair_bins = [b for b, k in zip(bins_of_pair, keep, strict=True) if k]
        for coord in COORDS:
            residual = (
                getattr(event, f"rec_{coord}")[rec_ids]
                - getattr(event, f"true_{coord}")[true_ids]
            )
            limit = self.ranges[coord]
            inside = (-limit <= residual) & (residual <= limit)
            for bin_ in self.bins:
                mask = inside & np.array([b == bin_ for b in pair_bins], dtype=bool)
                if mask.any():
                    values = residual[mask]
                    self.moments[coord][bin_].update(values)
                    store.append(self.column(coord, bin_), values)

    def partial(self):
        return {
            coord: {_bin_key(b): self.moments[coord][b].to_dict() for b in self.bins}
            for coord in COORDS
        }

    @classmethod
    def merge(cls, partials):
        partials = list(partials)
        return {
            coord: {
                key: _merge_moments([p[coord][key] for p in partials])
                for key in partials[0][coord]
            }
            for coord in COORDS
        }

    def finalize(self, partial, reader):
        resolutions, errors, biases, counts = {}, {}, {}, {}
        overall = {}
        for coord in COORDS:
            resolutions[coord], errors[coord], biases[coord], counts[coord] = (
                {},
                {},
                {},
                {},
            )
            per_bin = {}
            for bin_ in self.bins:
                moments = Moments.from_dict(partial[coord][_bin_key(bin_)])
                per_bin[bin_] = moments
                resolutions[coord][bin_], errors[coord][bin_] = moments.std_with_error()
                mean, error = moments.mean_with_error()
                biases[coord][bin_] = (
                    (0.0, 0.0) if mean is None or error is None else (mean, error)
                )
                counts[coord][bin_] = moments.n
            total = Moments.merged(per_bin.values())
            resolution, error = total.std_with_error()
            mean, mean_error = total.mean_with_error()
            overall[coord] = {
                "resolution": resolution,
                "error": error,
                "bias": 0.0 if mean is None or mean_error is None else mean,
                "bias_error": 0.0 if mean is None or mean_error is None else mean_error,
                "n_vertices": total.n if total.n > 1 else 0,
            }
        plot_data = prepare_plot_data(self.bins, resolutions, errors, counts)
        histograms = {
            coord: {
                bin_: reader.histogram(self.column(coord, bin_), 100)
                for bin_ in self.bins
                if counts[coord][bin_] > 0
            }
            for coord in COORDS
        }
        return {
            "resolutions": resolutions,
            "errors": errors,
            "biases": biases,
            "counts": counts,
            "overall": overall,
            "plot_data": plot_data,
            "histograms": histograms,
        }

    def write(self, results, output_dir, settings):
        s = self.settings
        method_name = f"{settings.method_name}_Track_Based"
        for coord in COORDS:
            stats = results["overall"][coord]
            unit = "ns" if coord == "t" else "mm"
            print(
                f"resolution  {coord}  {stats['resolution']:.4f} ± {stats['error']:.4f} {unit}"
                f"  bias {stats['bias']:.4f} ± {stats['bias_error']:.4f} {unit}"
                f"  ({stats['n_vertices']} vertices)"
            )
        if results["plot_data"] is None:
            print("No resolution data to plot")
            return
        ranges = tuple(self.ranges[coord] for coord in COORDS)
        plot_resolutions(
            results["plot_data"],
            s.recall_threshold,
            s.purity_threshold,
            *ranges,
            output_dir,
            method_name,
        )
        _plot_residual_histograms(
            self.bins,
            results["histograms"],
            results["resolutions"],
            results["biases"],
            results["counts"],
            self.ranges,
            output_dir,
            method_name,
        )
        write_resolution_csv(
            self.bins,
            results["resolutions"],
            results["errors"],
            results["biases"],
            results["counts"],
            results["overall"],
            s.recall_threshold,
            s.purity_threshold,
            output_dir,
            method_name,
        )


def _plot_residual_histograms(
    bins, histograms, resolutions, biases, counts, ranges, output_dir, method_name
):
    """The residual histogram pages of analyze_resolution, drawn from the
    precomputed ``np.histogram(residuals, bins=100)`` of each bin."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")
    colors = {"x": "red", "y": "green", "z": "blue", "t": "purple"}
    for coord in COORDS:
        unit = "ns" if coord == "t" else "mm"
        bins_with_data = [b for b in bins if counts[coord][b] > 0]
        if not bins_with_data:
            continue
        n_cols = 2
        n_rows = (len(bins_with_data) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = np.asarray(axes).reshape(n_rows, n_cols)
        for plot_idx, bin_ in enumerate(bins_with_data):
            ax = axes[plot_idx // n_cols, plot_idx % n_cols]
            hist_counts, edges = histograms[coord][bin_]
            n = counts[coord][bin_]
            density = hist_counts / (n * np.diff(edges))
            ax.stairs(density, edges, fill=True, alpha=0.7, color=colors[coord])
            ax.axvline(0, color="black", linestyle="--", alpha=0.7)
            ax.set_xlabel(f"{coord.upper()} Residual ({unit})", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(
                f"{coord.upper()} Residuals - [{bin_[0]}, {bin_[1]}] tracks\n"
                f"std={resolutions[coord][bin_]:.4f} {unit}, "
                f"bias={biases[coord][bin_][0]:.4f} {unit}, n={n}",
                fontsize=11,
            )
            ax.set_xlim(-ranges[coord], ranges[coord])
            ax.grid(True, alpha=0.3)
        for i in range(len(bins_with_data), n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)
        plt.tight_layout()
        hist_file = output_path / f"residual_histograms_{coord}_{method_slug}.pdf"
        plt.savefig(hist_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {coord.upper()} residual histograms to: {hist_file}")


def _summary_from(moments, quantiles, population_std=False):
    """The per-column summary of the pull analyses from moments and quantiles.

    ``population_std`` selects the ddof = 0 ``std`` of the track pull summary;
    the vertex pull summary reports the sample standard deviation.
    """
    if moments.n == 0:
        return {
            "n": 0,
            "mean": None,
            "mean_error": None,
            "std": None,
            "std_error": None,
            "central68_half_width": None,
        }
    mean, mean_error = moments.mean_with_error()
    if population_std:
        std = moments.population_std()
        std_error = moments.std_with_error()[1] if moments.n > 1 else None
    else:
        std, std_error = moments.std_with_error() if moments.n > 1 else (None, None)
    low, high = quantiles
    return {
        "n": int(moments.n),
        "mean": mean,
        "mean_error": mean_error,
        "std": std,
        "std_error": std_error,
        "central68_half_width": float((high - low) / 2),
    }


def _pm_from_moments(moments):
    """Format ``mean=a ± b, std=c ± d`` from moments."""
    mean, mean_error = moments.mean_with_error()
    if mean is None:
        return "no entries"
    std, std_error = moments.std_with_error()
    if mean_error is None:
        return f"mean={mean:.3f}, std={std:.3f}"
    return f"mean={mean:.3f} ± {mean_error:.3f}, std={std:.3f} ± {std_error:.3f}"


class VertexPull(Accumulator):
    """Residuals, errors and pulls of overlap-matched vertices with valid errors
    (analyze_vertex_pull). Columns ``vertex/<name>`` in source dtype; moments
    of every column and of the values inside each plot window."""

    name = "vertex_pull"
    FIELDS = tuple(
        f"{kind}_{coord}" for kind in ("pull", "error", "residual") for coord in COORDS
    )
    WINDOWS = {
        **{f"pull_{c}": (-5.0, 5.0) for c in COORDS},
        "residual_x": (-0.1, 0.1),
        "residual_y": (-0.1, 0.1),
        "residual_z": (-0.7, 0.7),
        "residual_t": (-0.06, 0.06),
    }

    def __init__(self, settings, bins, **extra):
        super().__init__(settings, bins, **extra)
        self.moments = {name: Moments() for name in self.FIELDS}
        self.windowed = {name: Moments() for name in self.WINDOWS}

    @staticmethod
    def column(name):
        return f"vertex/{name}"

    def update(self, event, store):
        true_ids, rec_ids = _pairs(event, self.settings)
        if not len(true_ids):
            return
        errors = {
            coord: getattr(event, f"rec_{coord}_err")[rec_ids] for coord in COORDS
        }
        valid = np.ones(len(rec_ids), dtype=bool)
        for err in errors.values():
            valid &= ~((err <= 0) | np.isnan(err))
        if not valid.any():
            return
        true_ids, rec_ids = true_ids[valid], rec_ids[valid]
        for coord in COORDS:
            err = errors[coord][valid]
            residual = (
                getattr(event, f"rec_{coord}")[rec_ids]
                - getattr(event, f"true_{coord}")[true_ids]
            )
            for name, values in (
                (f"residual_{coord}", residual),
                (f"error_{coord}", err),
                (f"pull_{coord}", residual / err),
            ):
                self.moments[name].update(values)
                store.append(self.column(name), values)
                if name in self.WINDOWS:
                    low, high = self.WINDOWS[name]
                    self.windowed[name].update(
                        values[(values >= low) & (values <= high)]
                    )

    def partial(self):
        return {
            "moments": {name: m.to_dict() for name, m in self.moments.items()},
            "windowed": {name: m.to_dict() for name, m in self.windowed.items()},
        }

    @classmethod
    def merge(cls, partials):
        partials = list(partials)
        return {
            key: {
                name: _merge_moments([p[key][name] for p in partials])
                for name in partials[0][key]
            }
            for key in ("moments", "windowed")
        }

    def finalize(self, partial, reader):
        moments = {name: Moments.from_dict(d) for name, d in partial["moments"].items()}
        windowed = {
            name: Moments.from_dict(d) for name, d in partial["windowed"].items()
        }
        n_vertices = moments["pull_x"].n
        statistics = {}
        histograms = {}
        for name in self.FIELDS:
            column = self.column(name)
            quantiles = (
                reader.quantile(column, [0.16, 0.84])
                if moments[name].n
                else (None, None)
            )
            statistics[name] = _summary_from(moments[name], quantiles)
            if n_vertices:
                histograms[name] = reader.histogram(
                    column, 50, range=self.WINDOWS.get(name)
                )
        return {
            "n_vertices": int(n_vertices),
            "statistics": statistics,
            "histograms": histograms,
            "titles": {
                name: _pm_from_moments(
                    windowed[name] if name in windowed else moments[name]
                )
                for name in self.FIELDS
            },
        }

    def write(self, results, output_dir, settings):
        n = results["n_vertices"]
        print(f"vertex_pull  {n} matched vertices with valid errors")
        if n == 0:
            print("No valid matched vertices found for pull analysis")
            return
        summary = {
            "statistics_scope": "All matched vertices with valid errors, including "
            "values outside the plot windows",
            "n_vertices": n,
            "statistics": results["statistics"],
        }
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "pull_analysis_fitted_vertex_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        for coord in COORDS:
            stats = summary["statistics"][f"pull_{coord}"]
            mean_error = (
                stats["mean_error"] if stats["mean_error"] is not None else float("nan")
            )
            std = stats["std"] if stats["std"] is not None else float("nan")
            std_error = (
                stats["std_error"] if stats["std_error"] is not None else float("nan")
            )
            print(
                f"vertex_pull  {coord}  mean {stats['mean']:.3f} ± {mean_error:.3f}"
                f"  std {std:.3f} ± {std_error:.3f}"
            )
        _plot_vertex_pulls(results, output_path)


def _plot_vertex_pulls(results, output_path):
    """analyze_vertex_pull._plot_vertex_pulls from histogram counts."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("Fitted Vertex Analysis: Residuals, Errors, and Pulls", fontsize=16)
    units = {"x": "mm", "y": "mm", "z": "mm", "t": "ns"}
    labels = {"residual": "Residual", "error": "Error", "pull": "Pull"}
    for row, kind in enumerate(("residual", "error", "pull")):
        for col, coord in enumerate(COORDS):
            name = f"{kind}_{coord}"
            ax = axes[row, col]
            counts, edges = results["histograms"][name]
            # ax.hist(..., range=..., density=True) normalises to the entries
            # inside the displayed range, not to all observations
            shown = counts.sum()
            if shown:
                ax.stairs(
                    counts / (shown * np.diff(edges)), edges, fill=True, alpha=0.7
                )
            unit = f" ({units[coord]})" if kind != "pull" else ""
            ax.set_xlabel(f"{labels[kind]} {coord.upper()}{unit}")
            ax.set_ylabel("Density")
            ax.set_title(f"{labels[kind]} {coord.upper()} ({results['titles'][name]})")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_path / "pull_analysis_fitted_vertex.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Fitted vertex pull analysis plot saved as {output_path / 'pull_analysis_fitted_vertex.pdf'}"
    )


class TrackPull(Accumulator):
    """Truth-plane track residuals after the residual chi2 selection
    (analyze_track_pull). Reads the track states of every event; selected
    tracks go to columns ``track/<field>`` (float64, as computed)."""

    name = "track_pull"
    needs_tracks = True
    needs_rec_vertices = False
    FIELDS = (
        "pull_x",
        "pull_y",
        "pull_t",
        "error_x",
        "error_y",
        "error_t",
        "residual_x",
        "residual_y",
        "residual_t",
        "delta_z",
        "chi2",
    )
    WINDOWS = {
        "residual_x": (-0.75, 0.75),
        "residual_y": (-0.75, 0.75),
        "residual_t": (-0.15, 0.15),
        "pull_x": (-5, 5),
        "pull_y": (-5, 5),
        "pull_t": (-5, 5),
        "delta_z": (-5, 5),
        "chi2": (0, 30),
    }

    def __init__(self, settings, bins, *, max_chi2, threshold_source, **extra):
        super().__init__(settings, bins, **extra)
        self.max_chi2 = float(max_chi2)
        self.threshold_source = threshold_source
        self.counts = {"n_associated": 0, "n_invalid": 0, "n_valid": 0, "n_selected": 0}
        self.moments = {name: Moments() for name in self.FIELDS}

    @staticmethod
    def column(name):
        return f"track/{name}"

    def update(self, event, store):
        tracks = event.tracks
        owner = np.full(len(tracks["x"]), -1, dtype=np.int64)
        for vertex_index, indices in enumerate(event.true_track_arrays):
            owner[indices] = vertex_index
        indices = np.flatnonzero(owner >= 0)
        self.counts["n_associated"] += len(indices)
        if not len(indices):
            return

        def state(name):
            return tracks[name][indices].astype(np.float64)

        matched = owner[indices]
        target = {
            coord: getattr(event, f"true_{coord}")[matched].astype(np.float64)
            for coord in COORDS
        }
        dz = target["z"] - state("z")
        tx, ty = state("tx"), state("ty")
        backward = state("backward")
        time_slope = (
            np.where(backward, -1.0, 1.0) * np.sqrt(1 + tx**2 + ty**2) / C_MM_PER_NS
        )
        residual = np.column_stack(
            [
                state("x") + dz * tx - target["x"],
                state("y") + dz * ty - target["y"],
                state("t") + dz * time_slope - target["t"],
            ]
        )
        covariance = np.column_stack(
            [state("cov_xx"), state("cov_yy"), state("cov_tt")]
        )
        valid = (
            np.isfinite(residual).all(axis=1)
            & np.isfinite(covariance).all(axis=1)
            & (covariance > 0).all(axis=1)
            & np.isfinite(dz)
            & np.isfinite(backward)
        )
        self.counts["n_invalid"] += int((~valid).sum())
        error = np.sqrt(covariance[valid])
        residual = residual[valid]
        scaled = residual / error
        chi2 = np.sum(scaled**2, axis=1)
        selected = chi2 <= self.max_chi2
        self.counts["n_valid"] += int(len(chi2))
        self.counts["n_selected"] += int(selected.sum())
        columns = {
            **{f"residual_{n}": residual[:, i] for i, n in enumerate(("x", "y", "t"))},
            **{f"error_{n}": error[:, i] for i, n in enumerate(("x", "y", "t"))},
            **{f"pull_{n}": scaled[:, i] for i, n in enumerate(("x", "y", "t"))},
            "delta_z": dz[valid],
            "chi2": chi2,
        }
        for name in self.FIELDS:
            values = columns[name][selected]
            self.moments[name].update(values)
            store.append(self.column(name), values)

    def partial(self):
        return {
            "counts": dict(self.counts),
            "moments": {name: m.to_dict() for name, m in self.moments.items()},
        }

    @classmethod
    def merge(cls, partials):
        partials = list(partials)
        return {
            "counts": {
                key: int(sum(p["counts"][key] for p in partials))
                for key in partials[0]["counts"]
            },
            "moments": {
                name: _merge_moments([p["moments"][name] for p in partials])
                for name in partials[0]["moments"]
            },
        }

    def finalize(self, partial, reader):
        counts = partial["counts"]
        moments = {name: Moments.from_dict(d) for name, d in partial["moments"].items()}
        statistics, histograms = {}, {}
        for name in self.FIELDS:
            column = self.column(name)
            quantiles = (
                reader.quantile(column, [0.16, 0.84])
                if moments[name].n
                else (None, None)
            )
            statistics[name] = _summary_from(
                moments[name], quantiles, population_std=True
            )
            if moments[name].n:
                histograms[name] = reader.histogram(
                    column, 60, range=self.WINDOWS.get(name)
                )
        return {
            "summary": {
                "normalization": "Truth-plane residual / reported POCA uncertainty; covariance not propagated",
                "transport": "Straight line to matched truth z; forward/backward sign; beta = 1",
                "matching": "Stored key-based true_vertices/track_indices",
                "selection": "pull_x**2 + pull_y**2 + pull_t**2 <= max_chi2, evaluated at truth z; no Tukey weights",
                "max_chi2": self.max_chi2,
                "max_chi2_source": self.threshold_source,
                "statistics_scope": "All tracks passing the residual chi2 selection, including values outside plot limits",
                "n_associated": counts["n_associated"],
                "n_invalid": counts["n_invalid"],
                "n_valid": counts["n_valid"],
                "n_selected": counts["n_selected"],
                "n_rejected_by_chi2": counts["n_valid"] - counts["n_selected"],
                "statistics": statistics,
            },
            "histograms": histograms,
        }

    def write(self, results, output_dir, settings):
        summary = results["summary"]
        print("\nStarting truth-plane input track residual analysis...")
        print(
            f"Residual chi2 <= {self.max_chi2:g}; threshold from {self.threshold_source}"
        )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "pull_analysis_input_track_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        _plot_track_pulls(results, output, max_chi2=self.max_chi2)
        print(
            f"Key-associated: {summary['n_associated']}; invalid: {summary['n_invalid']}; "
            f"valid: {summary['n_valid']}; chi2 <= {self.max_chi2:g}: {summary['n_selected']}; "
            f"rejected: {summary['n_rejected_by_chi2']}"
        )
        print(f"Track residual plots and summary saved to {output}/")


def _pm(value, error):
    return f"{value:.3g}" if error is None else f"{value:.3g} ± {error:.2g}"


def _plot_track_pulls(results, output_dir, *, max_chi2):
    """analyze_track_pull._plot_track_pulls from histogram counts and the
    full-sample statistics; same panels, same normalisation by all entries."""
    summary = results["summary"]
    n_total = summary["n_selected"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(20, 13), layout="constrained")
    fig.suptitle(
        f"Tracks transported to truth z | Residual chi2 <= {max_chi2:g} | N = {n_total:,}\n"
        "Scaled by reported POCA uncertainties (covariance not propagated)",
        fontsize=15,
    )

    def histogram(ax, name, label, limits=None):
        ax.set_xlabel(label)
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.25)
        stats = summary["statistics"][name]
        if not stats["n"]:
            ax.text(
                0.5,
                0.5,
                "No tracks pass the selection",
                ha="center",
                transform=ax.transAxes,
            )
            return
        counts, edges = results["histograms"][name]
        ax.stairs(counts / (stats["n"] * np.diff(edges)), edges, fill=True, alpha=0.6)
        outside = 100 * (1 - counts.sum() / stats["n"])
        ax.set_title(
            f"Mean {_pm(stats['mean'], stats['mean_error'])}\n"
            f"std {_pm(stats['std'], stats['std_error'])} | "
            f"68% half-width {stats['central68_half_width']:.3g}\n"
            f"Outside display: {outside:.2f}%",
            fontsize=10,
        )
        if limits is not None:
            ax.set_xlim(limits)

    for column, name in enumerate(("x", "y", "t")):
        unit = "ns" if name == "t" else "mm"
        histogram(
            axes[0, column],
            f"residual_{name}",
            f"Residual {name} at truth z [{unit}]",
            TrackPull.WINDOWS[f"residual_{name}"],
        )
        histogram(
            axes[1, column], f"error_{name}", f"Reported POCA sigma({name}) [{unit}]"
        )
        histogram(
            axes[2, column],
            f"pull_{name}",
            f"Residual {name} / POCA sigma({name})",
            TrackPull.WINDOWS[f"pull_{name}"],
        )
    histogram(
        axes[0, 3], "delta_z", "Truth z - POCA z [mm]", TrackPull.WINDOWS["delta_z"]
    )
    histogram(
        axes[1, 3], "chi2", "Sum of squared scaled residuals", TrackPull.WINDOWS["chi2"]
    )
    axes[2, 3].axis("off")
    axes[2, 3].text(
        0.0,
        0.9,
        "Statistics include all selected tracks,\nincluding values outside the\n"
        "displayed ranges.\n\n"
        "68% half-width = (q84 - q16) / 2.\n\n"
        "The residual chi2 cut includes x, y, t.\n"
        "No Tukey weights are applied.",
        va="top",
        fontsize=11,
        transform=axes[2, 3].transAxes,
    )
    for extension in ("pdf", "png"):
        fig.savefig(output / f"pull_analysis_input_track.{extension}", dpi=120)
    plt.close(fig)
