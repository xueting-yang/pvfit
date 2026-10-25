"""Map analysis names to modules; the analysis stage of ``pvfit analyze``.

The rate analyses only count (see binned_rate.RateResult); this module writes
their CSVs, plots and one summary line each, so every rate is reported the
same way. The residual analyses (resolution, pulls, histogram) write their
own outputs.
"""

import os

from ..config_loader import AnalysisConfig
from .analyze_efficiency_geo import (
    analyze_vertex_efficiency as analyze_vertex_efficiency_geo,
)
from .analyze_efficiency_track import (
    analyze_vertex_efficiency as analyze_vertex_efficiency_track,
)
from .analyze_fake_rate_geo import (
    analyze_vertex_fake_rate as analyze_vertex_fake_rate_geo,
)
from .analyze_fake_rate_track import (
    analyze_vertex_fake_rate as analyze_vertex_fake_rate_track,
)
from .analyze_merge_distance import analyze_merge_distance
from .analyze_merge_rate_track import (
    analyze_vertex_merge_rate as analyze_vertex_merge_rate_track,
)
from .analyze_merge_split_geo import (
    analyze_vertex_merge_rate as analyze_vertex_merge_rate_geo,
)
from .analyze_merge_split_geo import (
    analyze_vertex_split_rate as analyze_vertex_split_rate_geo,
)
from .analyze_resolution import analyze_vertex_resolution
from .analyze_split_rate_track import (
    analyze_vertex_split_rate as analyze_vertex_split_rate_track,
)
from .analyze_track_pull import analyze_track_pull
from .analyze_vertex_pull import analyze_vertex_pull
from .binned_rate import (
    count_vertex_tracks,
    make_bins,
    plot_efficiency_lhcb2,
    plot_rate_lhcb2,
    write_rate_csv,
    write_vertex_track_count_csv,
)
from .plot_histogram import plot_event_histogram


def run_analysis(hdf5_file: str, output_dir: str, settings: AnalysisConfig):
    """Run the analyses ``settings.analyses`` selects on an HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file (e.g. fit_results.h5)
        output_dir: Directory to save plots / CSVs
        settings: analysis settings, independent of any reconstruction config
    """
    analyses = settings.analyses
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analyzing {hdf5_file} -> {output_dir}: {', '.join(analyses)}")

    bins = make_bins(
        settings.plot_min_ntracks,
        settings.plot_max_ntracks,
        settings.plot_bin_width_ntracks,
    )
    track_based = f"{settings.method_name}_Track_Based"

    # ------------------------------------------------------------
    # Rates binned in track multiplicity: count, then report uniformly
    # ------------------------------------------------------------
    results = []
    if "efficiency_track" in analyses:
        results += analyze_vertex_efficiency_track(
            hdf5_file, bins, settings.recall_threshold, settings.purity_threshold
        )
    if "efficiency_geo" in analyses:
        geo_results = analyze_vertex_efficiency_geo(
            hdf5_file,
            bins,
            settings.m_r,
            settings.z_scale,
            settings.t_scale,
            settings.nsigma,
        )
        results += geo_results
    if "fake_rate_track" in analyses:
        results += analyze_vertex_fake_rate_track(
            hdf5_file, bins, settings.recall_threshold, settings.purity_threshold
        )
    if "fake_rate_geo" in analyses:
        geo_results = analyze_vertex_fake_rate_geo(
            hdf5_file,
            bins,
            settings.m_r,
            settings.z_scale,
            settings.t_scale,
            settings.nsigma,
        )
        results += geo_results
    if "merge_rate" in analyses:
        results += analyze_vertex_merge_rate_track(
            hdf5_file, bins, settings.recall_threshold
        )
    if "split_rate" in analyses:
        results += analyze_vertex_split_rate_track(
            hdf5_file, bins, settings.purity_threshold
        )
    ellipses = (settings.m_r, settings.z_scale, settings.t_scale, settings.nsigma)
    for name, analyze in (
        ("merge_rate_geo", analyze_vertex_merge_rate_geo),
        ("split_rate_geo", analyze_vertex_split_rate_geo),
    ):
        if name in analyses:
            geo_results = analyze(hdf5_file, bins, *ellipses)
            results += geo_results

    if results:
        # Efficiency already carries both unconditioned populations. Reuse
        # them without another HDF5 scan; other analysis selections also
        # get the standalone distribution, independently of matching.
        efficiency = next((r for r in results if r.metric == "efficiency"), None)
        if efficiency is not None:
            true_counts = efficiency.counts.denominator
            reconstructed_counts = efficiency.reconstructed_counts
        else:
            true_counts, reconstructed_counts = count_vertex_tracks(hdf5_file, bins)
        write_vertex_track_count_csv(
            true_counts, reconstructed_counts, output_dir, settings.method_name
        )

    for result in results:
        print(result.summary_line(settings.method_name))
        write_rate_csv(result, output_dir, settings.method_name)
        if result.metric == "efficiency":
            plot_efficiency_lhcb2(
                result,
                output_dir,
                settings.method_name,
                settings.efficiency_plot_max_ntracks,
            )
        else:
            key = f"{result.metric}_{result.matching}_{result.binned_by}"
            plot_rate_lhcb2(
                result,
                output_dir,
                settings.method_name,
                settings.rate_plot_max_ntracks,
                settings.rate_plot_ymax.get(key),
            )

    # ------------------------------------------------------------
    # Residual analyses and the event display
    # ------------------------------------------------------------
    if "merge_rate" in analyses:
        analyze_merge_distance(
            hdf5_file, output_dir, settings.recall_threshold, settings.method_name
        )

    if "resolution" in analyses:
        analyze_vertex_resolution(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=settings.recall_threshold,
            purity_threshold=settings.purity_threshold,
            bins=bins,
            residual_ranges={
                coord: getattr(settings, f"{coord}_residual_range") for coord in "xyzt"
            },
            method_name=track_based,
        )

    if "plot_histogram" in analyses:
        plot_event_histogram(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            settings=settings,
            event_id=settings.plot_event_id,
        )

    if "track_pull" in analyses:
        analyze_track_pull(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            max_chi2=settings.track_pull_chi2,
        )

    if "vertex_pull" in analyses:
        analyze_vertex_pull(
            hdf5_path=hdf5_file,
            output_dir=output_dir,
            recall_threshold=settings.recall_threshold,
            purity_threshold=settings.purity_threshold,
        )

    print(f"Analysis outputs in {output_dir}")
