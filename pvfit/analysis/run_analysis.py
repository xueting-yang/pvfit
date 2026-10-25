"""Public analysis entry point: stream events through the selected accumulators."""

from ..config_loader import AnalysisConfig
from .driver import run_parallel_analysis


def run_analysis(hdf5_file: str, output_dir: str, settings: AnalysisConfig):
    """Analyze existing HDF5 results with the configured process count.

    Both one and multiple workers use the same accumulator/reduction pipeline.
    Matching definitions, residual cuts and output names are unchanged.
    """
    return run_parallel_analysis(
        hdf5_file,
        output_dir,
        settings,
        workers=settings.workers,
        batches=settings.batches,
        work_dir=settings.work_dir,
        keep_columns=settings.keep_columns,
    )
