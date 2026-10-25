"""The contract every analysis accumulator follows.

``update(event, store)`` consumes one event; counts and moments stay in memory,
raw observations go to the batch's column store. ``partial()`` returns that
state as a small JSON-serialisable dict, ``merge(partials)`` combines the
partials of all batches, ``finalize(partial, reader)`` turns the merged state
and the disk columns into results, and ``write(results, output_dir, settings)``
writes files and figures from the results only.
"""


class Accumulator:
    name = ""
    needs_tracks = False
    needs_rec_vertices = True

    def __init__(self, settings, bins, **extra):
        self.settings = settings
        self.bins = list(bins)
        self.extra = extra

    def update(self, event, store):
        raise NotImplementedError

    def partial(self):
        raise NotImplementedError

    @classmethod
    def merge(cls, partials):
        raise NotImplementedError

    def finalize(self, partial, reader):
        raise NotImplementedError

    def write(self, results, output_dir, settings):
        """Residual analyses write their own outputs; rate results are written
        by the driver."""
