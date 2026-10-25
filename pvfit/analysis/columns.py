"""Raw observations spilled to disk in chunks, read back one column at a time.

Each worker owns a :class:`ColumnStore`: ``append(name, values)`` buffers a
column's values and writes a part file as soon as the buffer holds at least
``flush_values`` numbers (a part is that many values plus the chunk that
crossed the threshold), so worker memory stays bounded by the buffer size and
one event's chunk, not by the sample. ``close`` writes a manifest naming every part, its count and
the column's source dtype, which is preserved as stored (float32 residuals stay
float32), because NumPy's automatic histogram edges depend on it.

The reducer opens a :class:`ColumnReader` over the batch manifests in batch
(= event) order. ``array`` materialises one column; ``histogram`` with a fixed
range streams the parts and never holds the column; ``quantile`` and automatic
histogram ranges need the whole column once. Nothing crosses process
boundaries except file names and counts.
"""

import json
from pathlib import Path

import numpy as np

MANIFEST_NAME = "columns.json"


def _file_stem(name):
    return name.replace("/", "__")


class ColumnStore:
    def __init__(self, directory, flush_values=1 << 20):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_values = int(flush_values)
        self._buffers = {}
        self._buffered = {}
        self._dtypes = {}
        self._parts = {}
        self._counts = {}
        self.closed = False

    def declare(self, name, dtype):
        """Register a column that may stay empty, so its dtype is still known."""
        dtype = np.dtype(dtype).str
        if name in self._dtypes and self._dtypes[name] != dtype:
            raise ValueError(
                f"Column {name}: dtype {dtype} differs from {self._dtypes[name]}"
            )
        self._dtypes.setdefault(name, dtype)
        self._buffers.setdefault(name, [])
        self._buffered.setdefault(name, 0)
        self._parts.setdefault(name, [])
        self._counts.setdefault(name, 0)

    def append(self, name, values):
        if self.closed:
            raise RuntimeError("ColumnStore is closed")
        values = np.asarray(values)
        if values.ndim != 1:
            values = values.ravel()
        if values.size == 0:
            return
        self.declare(name, values.dtype)
        if values.dtype.str != self._dtypes[name]:
            raise ValueError(
                f"Column {name}: values of dtype {values.dtype.str} appended to a "
                f"{self._dtypes[name]} column"
            )
        self._buffers[name].append(values)
        self._buffered[name] += values.size
        if self._buffered[name] >= self.flush_values:
            self._flush(name)

    def _flush(self, name):
        if not self._buffered[name]:
            return
        chunk = np.concatenate(self._buffers[name])
        index = len(self._parts[name])
        file_name = f"{_file_stem(name)}.{index:04d}.bin"
        chunk.tofile(self.directory / file_name)
        self._parts[name].append({"file": file_name, "count": int(chunk.size)})
        self._counts[name] += int(chunk.size)
        self._buffers[name] = []
        self._buffered[name] = 0

    def close(self):
        """Flush every buffer and write the manifest; returns its path."""
        for name in list(self._buffers):
            self._flush(name)
        manifest = {
            name: {
                "dtype": self._dtypes[name],
                "count": self._counts[name],
                "parts": self._parts[name],
            }
            for name in sorted(self._dtypes)
        }
        path = self.directory / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=1) + "\n")
        self.closed = True
        return path


class ColumnReader:
    """Columns of several batches, concatenated in the given batch order."""

    def __init__(self, manifest_paths):
        self._sources = []  # (directory, manifest dict), batch order
        for path in manifest_paths:
            path = Path(path)
            self._sources.append((path.parent, json.loads(path.read_text())))

    def names(self):
        found = set()
        for _, manifest in self._sources:
            found.update(manifest)
        return sorted(found)

    def dtype(self, name):
        """The column's dtype; batches with values must agree; float64 if none."""
        dtypes = {
            manifest[name]["dtype"]
            for _, manifest in self._sources
            if name in manifest and manifest[name]["count"]
        }
        if len(dtypes) > 1:
            raise ValueError(
                f"Column {name} has mixed dtypes across batches: {sorted(dtypes)}"
            )
        if dtypes:
            return np.dtype(dtypes.pop())
        declared = {
            manifest[name]["dtype"] for _, manifest in self._sources if name in manifest
        }
        return np.dtype(declared.pop()) if len(declared) == 1 else np.dtype(np.float64)

    def count(self, name):
        return sum(
            manifest[name]["count"] for _, manifest in self._sources if name in manifest
        )

    def iter_chunks(self, name):
        """Yield the stored parts as read-only memory maps, in order."""
        dtype = self.dtype(name)
        for directory, manifest in self._sources:
            if name not in manifest:
                continue
            for part in manifest[name]["parts"]:
                if part["count"]:
                    yield np.memmap(
                        directory / part["file"],
                        dtype=dtype,
                        mode="r",
                        shape=(part["count"],),
                    )

    def array(self, name):
        """The whole column in memory, in its source dtype.

        Parts are copied one at a time into a preallocated array and each
        memory map is dropped before the next one is opened, so the number of
        open files stays constant however many parts a column has.
        """
        out = np.empty(self.count(name), dtype=self.dtype(name))
        offset = 0
        for chunk in self.iter_chunks(name):
            out[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
            del chunk
        return out

    def histogram(self, name, bins, range=None):
        """``np.histogram`` of the column: streamed with a fixed range, else on
        the materialised column (automatic edges need its min and max)."""
        if range is None:
            values = self.array(name)
            if values.size == 0:
                return np.zeros(bins, dtype=np.int64), np.linspace(0.0, 1.0, bins + 1)
            return np.histogram(values, bins=bins)
        edges = np.histogram_bin_edges(
            np.empty(0, dtype=self.dtype(name)), bins=bins, range=range
        )
        counts = np.zeros(len(edges) - 1, dtype=np.int64)
        for chunk in self.iter_chunks(name):
            counts += np.histogram(chunk, bins=edges)[0]
        return counts, edges

    def quantile(self, name, qs):
        """Exact quantiles; materialises the column once (float64 copy inside NumPy)."""
        values = self.array(name)
        if values.size == 0:
            return [None for _ in qs]
        # The serial summaries cast to float64 before np.quantile; do the same.
        return [
            float(q) for q in np.quantile(values.astype(np.float64, copy=False), qs)
        ]

    def masked_values(self, name, low, high):
        """Values inside [low, high], streamed; for windowed plot annotations."""
        kept = [
            np.asarray(chunk[(chunk >= low) & (chunk <= high)])
            for chunk in self.iter_chunks(name)
        ]
        if not kept:
            return np.empty(0, dtype=self.dtype(name))
        return np.concatenate(kept)
