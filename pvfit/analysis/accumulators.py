"""Registry of the analysis accumulators (see ``accumulator_base.Accumulator``).

The rate accumulators live in ``rate_accumulators``, the residual ones in
``residual_accumulators``; this module maps every analysis name of
``config_loader.ANALYSIS_NAMES`` to its class. Analyses that are not
accumulations over events (the event display) run in the main process.
"""

from .accumulator_base import Accumulator
from .analyze_merge_distance import MergeDistance
from .rate_accumulators import (
    EfficiencyGeo,
    EfficiencyTrack,
    FakeGeo,
    FakeTrack,
    MergeGeo,
    MergeTrack,
    SplitGeo,
    SplitTrack,
    VertexTrackCounts,
)
from .residual_accumulators import Resolution, TrackPull, VertexPull

REGISTRY = {
    cls.name: cls
    for cls in (
        EfficiencyTrack,
        EfficiencyGeo,
        FakeTrack,
        FakeGeo,
        MergeTrack,
        MergeDistance,
        SplitTrack,
        MergeGeo,
        SplitGeo,
        VertexTrackCounts,
        Resolution,
        VertexPull,
        TrackPull,
    )
}
# Analyses that are not accumulations over events; the driver runs them in the
# main process after the reduction.
MAIN_PROCESS_ONLY = frozenset({"plot_histogram"})

__all__ = [
    "Accumulator",
    "EfficiencyGeo",
    "EfficiencyTrack",
    "FakeGeo",
    "FakeTrack",
    "MAIN_PROCESS_ONLY",
    "MergeDistance",
    "MergeGeo",
    "MergeTrack",
    "REGISTRY",
    "Resolution",
    "SplitGeo",
    "SplitTrack",
    "TrackPull",
    "VertexPull",
    "VertexTrackCounts",
]
