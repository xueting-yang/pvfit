"""One pass over the HDF5 events, for every analysis at once.

``read_events`` yields one :class:`Event` per HDF5 group, in increasing event
id. :class:`Event` extends ``EventVertices`` with the stored member arrays and
the optional track states. The track states the
track-pull analysis needs are read in the same group visit, only when asked
for, so the file is traversed once however many analyses run.

Batches are contiguous ranges of the sorted event ids; a batch is the unit of
work of one process. Every event belongs to exactly one batch, so per-event
decisions (matching, claiming, tie-breaking) are identical in serial and
parallel runs.
"""

from dataclasses import dataclass, field

import h5py
import numpy as np

from pvfit.analysis.binned_rate import EventVertices

# tracks/ datasets the track-pull analysis transports to the truth plane
TRACK_STATE_FIELDS = (
    "x",
    "y",
    "z",
    "t",
    "tx",
    "ty",
    "backward",
    "cov_xx",
    "cov_yy",
    "cov_tt",
)


@dataclass
class Event(EventVertices):
    """One event as the accumulators see it: the vertex data of
    :class:`EventVertices` plus ``true_track_arrays``, the stored member index
    arrays in their stored order (the sets lose it), and ``tracks``, the
    track-state columns, or None when no analysis asked for them."""

    true_track_arrays: list = field(default_factory=list)
    tracks: dict | None = None


def _optional(group, name):
    return group[name][:] if group is not None and name in group else None


def list_event_ids(hdf5_path):
    """Internal event ids of the file, ascending."""
    with h5py.File(hdf5_path, "r") as f:
        return sorted(int(key) for key in f)


def make_batches(event_ids, n_batches):
    """Split the ids into at most ``n_batches`` contiguous, non-empty ranges."""
    event_ids = list(event_ids)
    if not event_ids:
        return []
    n_batches = max(1, min(int(n_batches), len(event_ids)))
    size, remainder = divmod(len(event_ids), n_batches)
    batches, start = [], 0
    for index in range(n_batches):
        stop = start + size + (1 if index < remainder else 0)
        batches.append(event_ids[start:stop])
        start = stop
    return batches


def read_events(hdf5_path, event_ids, *, with_tracks=False, with_rec_vertices=True):
    """Yield :class:`Event` for the given ids, in the given order.

    Reconstructed vertices are read only with ``with_rec_vertices``; input
    track diagnostics need only truth and track states.

    Track states are read only with ``with_tracks``; a file without a
    ``tracks`` group then raises, as the track-pull analysis does.
    """
    with h5py.File(hdf5_path, "r") as f:
        for event_id in event_ids:
            group = f[str(event_id)]
            true = group["true_vertices"]
            rec = group["rec_vertices"] if with_rec_vertices else None
            true_track_arrays = [np.asarray(idx) for idx in true["track_indices"][:]]
            event = Event(
                event_id=int(event_id),
                true_tracks=[set(idx) for idx in true_track_arrays],
                true_x=_optional(true, "x"),
                true_y=_optional(true, "y"),
                true_z=true["z"][:],
                true_t=true["t"][:],
                rec_tracks=[set(idx) for idx in rec["track_indices"][:]]
                if rec is not None
                else [],
                rec_x=_optional(rec, "x"),
                rec_y=_optional(rec, "y"),
                rec_z=rec["z"][:] if rec is not None else np.empty(0),
                rec_t=rec["t"][:] if rec is not None else np.empty(0),
                rec_x_err=_optional(rec, "x_err"),
                rec_y_err=_optional(rec, "y_err"),
                rec_z_err=_optional(rec, "z_err"),
                rec_t_err=_optional(rec, "t_err"),
                true_track_arrays=true_track_arrays,
            )
            tracks = None
            if with_tracks:
                if "tracks" not in group:
                    raise ValueError(
                        f"Event {event_id}: track pull analysis requires a tracks group; "
                        "this file stores no track states"
                    )
                table = group["tracks"]
                missing = sorted({"tx", "ty", "backward"} - set(table))
                if missing:
                    raise ValueError(
                        f"Event {event_id}: track pull analysis requires tracks/"
                        f"{', '.join(missing)} for transport to the truth z plane. "
                        "This HDF5 predates those fields; re-run reconstruction "
                        "with the updated code (pvfit fit --config <your-config.yaml>)."
                    )
                tracks = {name: table[name][:] for name in TRACK_STATE_FIELDS}
            event.tracks = tracks
            yield event
