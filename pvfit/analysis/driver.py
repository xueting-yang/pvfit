"""Run every selected analysis in one pass over the events, in parallel batches.

    read one event
        -> accumulator.update(event, store)      (per batch, in a worker)
    worker returns partials (counts, moments) and its column manifest
        -> Accumulator.merge(partials)           (main process)
        -> accumulator.finalize(merged, reader)  (statistics; columns read once)
        -> write files, draw figures             (from results only)

Workers never return arrays; raw observations reach the main process through
the column files named in the manifests. ``workers=1`` runs the batches in the
calling process using exactly the same accumulators as the parallel path.
The outputs, their names and the console lines follow
``pvfit.analysis.run_analysis.run_analysis``.
"""

import concurrent.futures
import multiprocessing
import os
import shutil
import tempfile
from pathlib import Path

from pvfit.analysis.binned_rate import (
    make_bins,
    plot_efficiency_lhcb2,
    plot_fake_rate_comparison_lhcb2,
    plot_rate_lhcb2,
    write_rate_csv,
    write_vertex_track_count_csv,
)
from pvfit.analysis.plot_histogram import plot_event_histogram

from .accumulators import MAIN_PROCESS_ONLY, REGISTRY, VertexTrackCounts
from .analyze_merge_distance import MergeDistance
from .columns import ColumnReader, ColumnStore
from .events import list_event_ids, make_batches, read_events
from .track_pull_config import resolve_track_pull_chi2

# Report order of run_analysis: rates first, then the residual analyses.
RATE_ORDER = (
    "efficiency_track",
    "efficiency_geo",
    "fake_rate_track",
    "fake_rate_geo",
    "merge_rate",
    "split_rate",
    "merge_rate_geo",
    "split_rate_geo",
)
RESIDUAL_ORDER = ("resolution", "plot_histogram", "track_pull", "vertex_pull")
WORK_PREFIX = "pvfit_parallel_"


def build_specs(hdf5_file, settings, analyses):
    """(name, extra kwargs) of every accumulator the selection needs.

    Unknown analysis names are refused here, before any event is read; the
    population counter is added when a rate is requested without an
    efficiency (which supplies the populations itself, as in run_analysis).
    """
    unknown = sorted(set(analyses) - set(REGISTRY) - MAIN_PROCESS_ONLY)
    if unknown:
        raise ValueError(f"No accumulator for analyses: {unknown}")
    specs = []
    for name in RATE_ORDER:
        if name in analyses:
            specs.append((name, {}))
    if "merge_rate" in analyses:
        specs.append((MergeDistance.name, {"hdf5_path": str(hdf5_file)}))
    if any(name in analyses for name in RATE_ORDER) and not any(
        name.startswith("efficiency") for name in analyses
    ):
        specs.append((VertexTrackCounts.name, {}))
    if "resolution" in analyses:
        specs.append(("resolution", {}))
    if "track_pull" in analyses:
        max_chi2, source = resolve_track_pull_chi2(hdf5_file, settings.track_pull_chi2)
        specs.append(("track_pull", {"max_chi2": max_chi2, "threshold_source": source}))
    if "vertex_pull" in analyses:
        specs.append(("vertex_pull", {}))
    return specs


def _make_accumulators(settings, bins, specs):
    return [REGISTRY[name](settings, bins, **extra) for name, extra in specs]


def run_batch(task):
    """One batch in one process: update every accumulator over its events."""
    hdf5_file, event_ids, settings, bins, specs, batch_dir, flush_values = task
    accumulators = _make_accumulators(settings, bins, specs)
    needs_tracks = any(acc.needs_tracks for acc in accumulators)
    store = ColumnStore(batch_dir, flush_values=flush_values)
    for event in read_events(
        hdf5_file,
        event_ids,
        with_tracks=needs_tracks,
        with_rec_vertices=any(acc.needs_rec_vertices for acc in accumulators),
    ):
        for acc in accumulators:
            acc.update(event, store)
    manifest = store.close()
    return {
        "events": len(event_ids),
        "partials": {acc.name: acc.partial() for acc in accumulators},
        "manifest": str(manifest),
    }


def _run_main_process_analyses(hdf5_file, output_dir, settings, analyses):
    """The analyses that are not accumulations: they open the file themselves."""
    if "plot_histogram" in analyses:
        plot_event_histogram(
            hdf5_path=str(hdf5_file),
            output_dir=str(output_dir),
            settings=settings,
            run_number=settings.plot_run_number,
            event_number=settings.plot_event_number,
        )


def make_work_directory(work_dir, output_dir):
    """A fresh directory of this run only, inside ``work_dir`` (or the output).

    Only this directory is ever removed, never ``work_dir`` itself or anything
    else it contains.
    """
    parent = (
        Path(work_dir).expanduser() if work_dir else Path(output_dir) / ".parallel_work"
    )
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=WORK_PREFIX, dir=parent))


def reduce_events(
    hdf5_file,
    settings,
    *,
    workers=1,
    batches=None,
    work=None,
    output_dir=None,
    flush_values=1 << 20,
):
    """Read, accumulate in batches and merge; return the finalized results.

    Returns ``(finalized, accumulators, reader, summary)``: results by analysis
    name, the main-process accumulators (for ``write``), the column reader over
    the batch manifests and a summary dict. ``work`` is the run's own column
    directory (see :func:`make_work_directory`); it is created under
    ``output_dir`` when not given.
    """
    hdf5_file = str(hdf5_file)
    analyses = tuple(settings.analyses)
    bins = make_bins(
        settings.plot_min_ntracks,
        settings.plot_max_ntracks,
        settings.plot_bin_width_ntracks,
    )
    specs = build_specs(hdf5_file, settings, analyses)
    event_ids = list_event_ids(hdf5_file)
    workers = max(1, int(workers))
    n_batches = int(batches) if batches else workers * 4
    batch_ids = make_batches(event_ids, n_batches)
    work = Path(work) if work else make_work_directory(None, output_dir)
    tasks = [
        (
            hdf5_file,
            ids,
            settings,
            bins,
            specs,
            str(work / f"batch_{i:04d}"),
            flush_values,
        )
        for i, ids in enumerate(batch_ids)
    ]

    if workers == 1 or len(tasks) <= 1:
        batch_results = [run_batch(task) for task in tasks]
    else:
        # Children render nothing and must not multiply BLAS threads.
        os.environ.setdefault("MPLBACKEND", "Agg")
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=context
        ) as pool:
            batch_results = list(pool.map(run_batch, tasks))  # batch (= event) order

    accumulators = _make_accumulators(settings, bins, specs)
    merged = {}
    for acc in accumulators:
        partials = [result["partials"][acc.name] for result in batch_results]
        merged[acc.name] = type(acc).merge(partials or [acc.partial()])
    reader = ColumnReader([result["manifest"] for result in batch_results])
    finalized = {
        acc.name: acc.finalize(merged[acc.name], reader) for acc in accumulators
    }
    summary = {
        "events": len(event_ids),
        "batches": len(tasks),
        "workers": min(workers, max(1, len(tasks))),
        "work_dir": str(work),
    }
    return finalized, accumulators, reader, summary


def run_parallel_analysis(
    hdf5_file,
    output_dir,
    settings,
    *,
    workers=1,
    batches=None,
    work_dir=None,
    keep_columns=False,
    flush_values=1 << 20,
):
    """``run_analysis`` with one event pass, ``workers`` processes and a reduction.

    Args:
        hdf5_file, output_dir, settings: as for run_analysis
        workers: processes; 1 runs everything in this process
        batches: number of contiguous event batches (default 4 per worker)
        work_dir: parent of this run's column directory (default
            <output>/.parallel_work); the run gets its own fresh subdirectory
        keep_columns: keep the run's column directory after writing the outputs
        flush_values: values a worker buffers per column before writing a part

    Returns a small summary dict; ``work_dir`` in it names the kept column
    directory, or None when it was removed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = tuple(settings.analyses)
    print(f"Analyzing {hdf5_file} -> {output_dir}: {', '.join(analyses)}")
    specs = build_specs(hdf5_file, settings, analyses)
    if not specs:
        # Only main-process analyses (the event display) were selected: they
        # read what they need themselves, so no event pass and no columns.
        _run_main_process_analyses(hdf5_file, output_dir, settings, analyses)
        print(f"Analysis outputs in {output_dir}")
        return {
            "events": None,
            "batches": 0,
            "workers": 0,
            "work_dir": None,
            "output_dir": str(output_dir),
        }
    work = make_work_directory(work_dir, output_dir)
    try:
        finalized, accumulators, reader, summary = reduce_events(
            hdf5_file,
            settings,
            workers=workers,
            batches=batches,
            work=work,
            flush_values=flush_values,
        )
        by_name = {acc.name: acc for acc in accumulators}

        # ---- report, in run_analysis order ---------------------------------------
        results = []
        for name in RATE_ORDER:
            if name in finalized:
                results += finalized[name]
        if results:
            efficiency = next((r for r in results if r.metric == "efficiency"), None)
            if efficiency is not None:
                true_counts = efficiency.counts.denominator
                reconstructed_counts = efficiency.reconstructed_counts
            else:
                true_counts, reconstructed_counts = finalized[VertexTrackCounts.name]
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
        fake_results = {
            result.matching: result
            for result in results
            if result.metric == "fake_rate" and result.binned_by == "rec"
        }
        if "track" in fake_results and "reso" in fake_results:
            plot_fake_rate_comparison_lhcb2(
                fake_results["track"],
                fake_results["reso"],
                output_dir,
                settings.method_name,
                settings.rate_plot_max_ntracks,
            )

        if MergeDistance.name in finalized:
            by_name[MergeDistance.name].write(
                finalized[MergeDistance.name], str(output_dir), settings
            )

        for name in RESIDUAL_ORDER:
            if name not in analyses:
                continue
            if name in MAIN_PROCESS_ONLY:
                _run_main_process_analyses(hdf5_file, output_dir, settings, (name,))
            else:
                by_name[name].write(finalized[name], str(output_dir), settings)

    finally:
        # Whatever happened, the run's own column directory is removed unless
        # the caller asked to keep it; the parent work_dir is never touched.
        if not keep_columns:
            shutil.rmtree(work, ignore_errors=True)
    if not keep_columns:
        summary["work_dir"] = None
    print(f"Analysis outputs in {output_dir}")
    return {**summary, "output_dir": str(output_dir)}
