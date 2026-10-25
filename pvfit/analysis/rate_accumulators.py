"""Rate accumulators: efficiency, fake, merge and split, binned in track count.

Each one decides per event which vertices count as successes (the physics of
overlap- or distance-based matching) and keeps ``BinnedCounts``; the driver
reports every ``RateResult`` with the shared CSV writers and figure functions
of ``binned_rate``. Every event is closed on every counter after it is counted,
so the rate errors can account for correlated outcomes within an event.
"""

from .accumulator_base import Accumulator
from .analyze_merge_distance import _true_vertices_with_recall_above
from .binned_rate import BinnedCounts, RateResult, find_bin
from .utils import (
    claim_vertex_matches,
    find_best_match,
    match_vertices_elliptical_zt,
    match_vertices_resolution_zt,
)


def _sum_lists(lists):
    return [int(sum(values)) for values in zip(*lists, strict=True)]


def _population_dict(bins, population):
    return [population[b] for b in bins]


def _population_from_list(bins, values):
    return {b: int(v) for b, v in zip(bins, values, strict=True)}


def _true_vertices_geo(event):
    return [
        {"vertex_id": i, "z": z, "t": t, "num_tracks": len(tracks)}
        for i, (z, t, tracks) in enumerate(
            zip(event.true_z, event.true_t, event.true_tracks, strict=True)
        )
    ]


def _rec_vertices_geo(event, with_num_tracks):
    rows = []
    for i, (z, t, z_err, t_err, tracks) in enumerate(
        zip(
            event.rec_z,
            event.rec_t,
            event.rec_z_err,
            event.rec_t_err,
            event.rec_tracks,
            strict=True,
        )
    ):
        if len(tracks) > 0:
            row = {
                "rec_vertex_id": i,
                "rec_z": z,
                "rec_t": t,
                "z_err": z_err,
                "t_err": t_err,
            }
            if with_num_tracks:
                row["num_tracks"] = len(tracks)
            rows.append(row)
    return rows


def _geo_definitions(settings):
    """(matching, parameters, label, matcher) of the two ellipse definitions."""
    m_r, z_scale, t_scale, nsigma = (
        settings.m_r,
        settings.z_scale,
        settings.t_scale,
        settings.nsigma,
    )
    return [
        (
            "geo",
            {"m_r": m_r, "z_scale": z_scale, "t_scale": t_scale},
            f"Geo matching (m_r={m_r})",
            lambda rec, true: match_vertices_elliptical_zt(
                rec, true, m_r=m_r, z_scale=z_scale, t_scale=t_scale
            ),
        ),
        (
            "reso",
            {"nsigma": nsigma},
            f"Resolution matching ({nsigma}σ)",
            lambda rec, true: match_vertices_resolution_zt(rec, true, nsigma=nsigma),
        ),
    ]


class _RateAccumulator(Accumulator):
    """Common serialisation of named BinnedCounts and population lists."""

    counters = ()  # names of BinnedCounts attributes
    populations = ()  # names of {bin: int} attributes

    def __init__(self, settings, bins, **extra):
        super().__init__(settings, bins, **extra)
        for name in self.counters:
            setattr(self, name, BinnedCounts(self.bins))
        for name in self.populations:
            setattr(self, name, dict.fromkeys(self.bins, 0))

    def update(self, event, store):
        """Count one event, then close it on every counter so the rate errors
        can account for correlated outcomes within an event."""
        self.update_event(event, store)
        for name in self.counters:
            getattr(self, name).end_event()

    def update_event(self, event, store):
        raise NotImplementedError

    def partial(self):
        return {
            **{name: getattr(self, name).to_dict() for name in self.counters},
            **{
                name: _population_dict(self.bins, getattr(self, name))
                for name in self.populations
            },
        }

    @classmethod
    def merge(cls, partials):
        partials = list(partials)
        return {
            **{
                name: BinnedCounts.merge_dicts([p[name] for p in partials])
                for name in cls.counters
            },
            **{
                name: _sum_lists([p[name] for p in partials])
                for name in cls.populations
            },
        }

    def _counts(self, partial, name):
        return BinnedCounts.from_dict(self.bins, partial[name])

    def _population(self, partial, name):
        return _population_from_list(self.bins, partial[name])


class EfficiencyTrack(_RateAccumulator):
    name = "efficiency_track"
    counters = ("counts",)
    populations = ("reconstructed_counts",)

    def update_event(self, event, store):
        s = self.settings
        true_vertex_info = {
            vertex_id: {"track_indices": tracks, "num_tracks": len(tracks)}
            for vertex_id, tracks in enumerate(event.true_tracks)
        }
        matched_true_ids = set()
        for rec_tracks in event.rec_tracks:
            if not rec_tracks:
                continue
            bin_ = find_bin(self.bins, len(rec_tracks))
            if bin_ is not None:
                self.reconstructed_counts[bin_] += 1
            best_match_id = find_best_match(
                rec_tracks, true_vertex_info, s.recall_threshold, s.purity_threshold
            )
            if best_match_id is not None:
                matched_true_ids.add(best_match_id)
        for vertex_id, tracks in enumerate(event.true_tracks):
            self.counts.add(len(tracks), vertex_id in matched_true_ids)

    def finalize(self, partial, reader):
        s = self.settings
        return [
            RateResult(
                metric="efficiency",
                binned_by="true",
                counts=self._counts(partial, "counts"),
                parameters={
                    "recall_threshold": s.recall_threshold,
                    "purity_threshold": s.purity_threshold,
                },
                label=f"Recall≥{s.recall_threshold}, Purity≥{s.purity_threshold}",
                reconstructed_counts=self._population(partial, "reconstructed_counts"),
            )
        ]


class EfficiencyGeo(_RateAccumulator):
    name = "efficiency_geo"
    counters = ("by_geo", "by_reso")
    populations = ("reconstructed_counts",)

    def update_event(self, event, store):
        s = self.settings
        for tracks in event.rec_tracks:
            if tracks:
                bin_ = find_bin(self.bins, len(tracks))
                if bin_ is not None:
                    self.reconstructed_counts[bin_] += 1
        true_vertices = _true_vertices_geo(event)
        matched_geo = matched_reso = set()
        if len(event.rec_tracks) > 0:
            rec_geo = _rec_vertices_geo(event, with_num_tracks=False)
            match_vertices_elliptical_zt(
                rec_geo, true_vertices, m_r=s.m_r, z_scale=s.z_scale, t_scale=s.t_scale
            )
            matched_geo = set(claim_vertex_matches(rec_geo).values())
            rec_reso = _rec_vertices_geo(event, with_num_tracks=False)
            match_vertices_resolution_zt(rec_reso, true_vertices, nsigma=s.nsigma)
            matched_reso = set(claim_vertex_matches(rec_reso).values())
        for vertex in true_vertices:
            self.by_geo.add(vertex["num_tracks"], vertex["vertex_id"] in matched_geo)
            self.by_reso.add(vertex["num_tracks"], vertex["vertex_id"] in matched_reso)

    def finalize(self, partial, reader):
        s = self.settings
        population = self._population(partial, "reconstructed_counts")
        return [
            RateResult(
                metric="efficiency",
                binned_by="true",
                counts=self._counts(partial, "by_geo"),
                parameters={"m_r": s.m_r, "z_scale": s.z_scale, "t_scale": s.t_scale},
                label=f"Geo matching (m_r={s.m_r})",
                matching="geo",
                reconstructed_counts=dict(population),
            ),
            RateResult(
                metric="efficiency",
                binned_by="true",
                counts=self._counts(partial, "by_reso"),
                parameters={"nsigma": s.nsigma},
                label=f"Resolution matching ({s.nsigma}σ)",
                matching="reso",
                # the same population dict, as in the original module
                reconstructed_counts=dict(population),
            ),
        ]


class FakeTrack(_RateAccumulator):
    name = "fake_rate_track"
    counters = ("counts",)

    def update_event(self, event, store):
        s = self.settings
        for rec_tracks in event.rec_tracks:
            if not rec_tracks:
                continue
            num_rec_tracks = len(rec_tracks)
            has_candidate = False
            for true_tracks in event.true_tracks:
                num_shared = len(rec_tracks & true_tracks)
                if num_shared == 0:
                    continue
                purity = num_shared / num_rec_tracks
                recall = num_shared / len(true_tracks)
                if purity >= s.purity_threshold or recall >= s.recall_threshold:
                    has_candidate = True
                    break
            self.counts.add(num_rec_tracks, not has_candidate)

    def finalize(self, partial, reader):
        s = self.settings
        return [
            RateResult(
                metric="fake_rate",
                binned_by="rec",
                counts=self._counts(partial, "counts"),
                parameters={
                    "recall_threshold": s.recall_threshold,
                    "purity_threshold": s.purity_threshold,
                },
                label=(
                    f"No overlap match (Recall≥{s.recall_threshold} or "
                    f"Purity≥{s.purity_threshold})"
                ),
            )
        ]


class FakeGeo(_RateAccumulator):
    name = "fake_rate_geo"
    counters = ("by_geo", "by_reso")

    def update_event(self, event, store):
        s = self.settings
        true_vertices = [
            {"vertex_id": i, "z": z, "t": t}
            for i, (z, t) in enumerate(zip(event.true_z, event.true_t, strict=True))
        ]
        rec_geo = _rec_vertices_geo(event, with_num_tracks=True)
        match_vertices_elliptical_zt(
            rec_geo, true_vertices, m_r=s.m_r, z_scale=s.z_scale, t_scale=s.t_scale
        )
        for rec in rec_geo:
            self.by_geo.add(rec["num_tracks"], rec["n_matches"] == 0)
        rec_reso = _rec_vertices_geo(event, with_num_tracks=True)
        match_vertices_resolution_zt(rec_reso, true_vertices, nsigma=s.nsigma)
        for rec in rec_reso:
            self.by_reso.add(rec["num_tracks"], rec["n_matches"] == 0)

    def finalize(self, partial, reader):
        s = self.settings
        return [
            RateResult(
                metric="fake_rate",
                binned_by="rec",
                counts=self._counts(partial, "by_geo"),
                parameters={"m_r": s.m_r, "z_scale": s.z_scale, "t_scale": s.t_scale},
                label=f"Geo matching (m_r={s.m_r})",
                matching="geo",
            ),
            RateResult(
                metric="fake_rate",
                binned_by="rec",
                counts=self._counts(partial, "by_reso"),
                parameters={"nsigma": s.nsigma},
                label=f"Resolution matching ({s.nsigma}σ)",
                matching="reso",
            ),
        ]


class MergeTrack(_RateAccumulator):
    name = "merge_rate"
    counters = ("by_true", "by_rec")

    def update_event(self, event, store):
        threshold = self.settings.recall_threshold
        merged_true_ids = set()
        for rec_tracks in event.rec_tracks:
            if not rec_tracks:
                continue
            overlapping = _true_vertices_with_recall_above(
                rec_tracks, event.true_tracks, threshold
            )
            is_merged = len(overlapping) > 1
            self.by_rec.add(len(rec_tracks), is_merged)
            if is_merged:
                merged_true_ids |= overlapping
        for true_id, true_tracks in enumerate(event.true_tracks):
            self.by_true.add(len(true_tracks), true_id in merged_true_ids)

    def finalize(self, partial, reader):
        parameters = {"recall_threshold": self.settings.recall_threshold}
        label = f"Recall threshold: {self.settings.recall_threshold}"
        return [
            RateResult(
                "merge_rate",
                "true",
                self._counts(partial, "by_true"),
                parameters,
                label,
            ),
            RateResult(
                "merge_rate", "rec", self._counts(partial, "by_rec"), parameters, label
            ),
        ]


class SplitTrack(_RateAccumulator):
    name = "split_rate"
    counters = ("by_true", "by_rec")

    def update_event(self, event, store):
        threshold = self.settings.purity_threshold
        split_rec_ids = set()
        for true_tracks in event.true_tracks:
            pieces = (
                {
                    rec_id
                    for rec_id, rec_tracks in enumerate(event.rec_tracks)
                    if rec_tracks
                    and len(true_tracks & rec_tracks) / len(rec_tracks) >= threshold
                }
                if true_tracks
                else set()
            )
            is_split = len(pieces) > 1
            self.by_true.add(len(true_tracks), is_split)
            if is_split:
                split_rec_ids |= pieces
        for rec_id, rec_tracks in enumerate(event.rec_tracks):
            if not rec_tracks:
                continue
            self.by_rec.add(len(rec_tracks), rec_id in split_rec_ids)

    def finalize(self, partial, reader):
        parameters = {"purity_threshold": self.settings.purity_threshold}
        label = f"Purity threshold: {self.settings.purity_threshold}"
        return [
            RateResult(
                "split_rate",
                "true",
                self._counts(partial, "by_true"),
                parameters,
                label,
            ),
            RateResult(
                "split_rate", "rec", self._counts(partial, "by_rec"), parameters, label
            ),
        ]


class _GeoMergeSplit(_RateAccumulator):
    counters = ("geo_true", "geo_rec", "reso_true", "reso_rec")
    metric = ""

    def _matched_event(self, event, matcher):
        true_vertices = _true_vertices_geo(event)
        rec_vertices = _rec_vertices_geo(event, with_num_tracks=True)
        matcher(rec_vertices, true_vertices)
        return true_vertices, rec_vertices

    def update_event(self, event, store):
        for matching, _, _, matcher in _geo_definitions(self.settings):
            true_vertices, rec_vertices = self._matched_event(event, matcher)
            self._count(
                true_vertices,
                rec_vertices,
                getattr(self, f"{matching}_true"),
                getattr(self, f"{matching}_rec"),
            )

    def _count(self, true_vertices, rec_vertices, by_true, by_rec):
        raise NotImplementedError

    def finalize(self, partial, reader):
        results = []
        for matching, parameters, label, _ in _geo_definitions(self.settings):
            results += [
                RateResult(
                    self.metric,
                    "true",
                    self._counts(partial, f"{matching}_true"),
                    parameters,
                    label,
                    matching,
                ),
                RateResult(
                    self.metric,
                    "rec",
                    self._counts(partial, f"{matching}_rec"),
                    parameters,
                    label,
                    matching,
                ),
            ]
        return results


class MergeGeo(_GeoMergeSplit):
    name = "merge_rate_geo"
    metric = "merge_rate"

    def _count(self, true_vertices, rec_vertices, by_true, by_rec):
        merged_true_ids = set()
        for rec in rec_vertices:
            is_merged = rec["n_matches"] > 1
            by_rec.add(rec["num_tracks"], is_merged)
            if is_merged:
                merged_true_ids |= {
                    tv["vertex_id"] for tv in rec["matched_true_vertices"]
                }
        for tv in true_vertices:
            by_true.add(tv["num_tracks"], tv["vertex_id"] in merged_true_ids)


class SplitGeo(_GeoMergeSplit):
    name = "split_rate_geo"
    metric = "split_rate"

    def _count(self, true_vertices, rec_vertices, by_true, by_rec):
        containing_recs = {tv["vertex_id"]: 0 for tv in true_vertices}
        for rec in rec_vertices:
            for tv in rec["matched_true_vertices"]:
                containing_recs[tv["vertex_id"]] += 1
        split_true_ids = {
            vertex_id for vertex_id, n in containing_recs.items() if n > 1
        }
        for tv in true_vertices:
            by_true.add(tv["num_tracks"], tv["vertex_id"] in split_true_ids)
        for rec in rec_vertices:
            is_split = any(
                tv["vertex_id"] in split_true_ids for tv in rec["matched_true_vertices"]
            )
            by_rec.add(rec["num_tracks"], is_split)


class VertexTrackCounts(_RateAccumulator):
    """Raw true/rec populations by track count;
    used for the population CSV when no efficiency analysis supplies them."""

    name = "_vertex_track_counts"
    populations = ("true_counts", "reconstructed_counts")

    def update_event(self, event, store):
        for vertices, counts in (
            (event.true_tracks, self.true_counts),
            (event.rec_tracks, self.reconstructed_counts),
        ):
            for tracks in vertices:
                if not tracks:
                    continue
                bin_ = find_bin(self.bins, len(tracks))
                if bin_ is not None:
                    counts[bin_] += 1

    def finalize(self, partial, reader):
        return (
            self._population(partial, "true_counts"),
            self._population(partial, "reconstructed_counts"),
        )
