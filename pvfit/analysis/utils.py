"""Utility functions for vertex reconstruction performance analysis."""

from collections import deque

import numpy as np
from scipy.stats import beta, norm

# Two-sided 68.27% (one-sigma) interval: the tail probability on each side.
ONE_SIGMA_TAIL = float(norm.cdf(-1.0))


def mean_with_error(values):
    """Sample mean and its standard error s / sqrt(n).

    Fewer than two values give an error of ``None``; no values give ``(None,
    None)``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return None, None
    mean = float(np.mean(values))
    if n < 2:
        return mean, None
    return mean, float(np.std(values, ddof=1) / np.sqrt(n))


def std_with_error(values):
    """Sample standard deviation (ddof=1) and its uncertainty.

    The variance of the sample variance is taken from the fourth central
    moment, Var(s²) ≈ (1/n)[m₄ − ((n−3)/(n−1)) s⁴], with no Gaussian
    assumption; the delta method then gives Var(s) ≈ Var(s²)/(4 s²). For a
    Gaussian sample m₄ = 3σ⁴ and the error reduces to σ/√(2n). Fewer than two
    values give ``(0.0, 0.0)``; a degenerate sample gives ``(s, 0.0)``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    sample_var = float(np.var(values, ddof=1))
    centered = values - np.mean(values)
    m4 = float(np.mean(centered**4))
    var_of_sample_var = (1 / n) * (m4 - ((n - 3) / (n - 1)) * sample_var**2)
    if sample_var > 0 and var_of_sample_var > 0:
        return float(np.sqrt(sample_var)), float(
            np.sqrt(var_of_sample_var / (4 * sample_var))
        )
    return float(np.sqrt(sample_var)), 0.0


def calculate_clopper_pearson_errors(k, n, rate):
    """Calculate Clopper-Pearson 1-sigma confidence interval errors.

    Args:
        k: Number of successes
        n: Total number of trials
        rate: Observed rate (k/n)

    Returns:
        tuple: (error_lower, error_upper)
    """
    if n == 0:
        return 0.0, 0.0

    lower_bound = beta.ppf(ONE_SIGMA_TAIL, k, n - k + 1) if k > 0 else 0.0
    upper_bound = beta.ppf(1 - ONE_SIGMA_TAIL, k + 1, n - k) if k < n else 1.0

    if np.isnan(lower_bound) or np.isnan(upper_bound):
        return 0.0, 0.0

    return rate - lower_bound, upper_bound - rate


def match_vertices_elliptical_zt(
    rec_vertices: list[dict],
    true_vertices: list[dict],
    m_r: float,
    z_scale: float = 1.5,
    t_scale: float = 0.06,
) -> None:
    """
    Match reconstructed vertices to true vertices using an elliptical
    cut in (z, t) space.

    The matching follows the criterion:

        (dz / sigma_z)^2 + (dt / sigma_t)^2 <= 1

    where:
        dz = z_true - z_rec
        dt = t_true - t_rec
        sigma_z = z_scale * m_r / 2
        sigma_t = t_scale * m_r / 2

    Among all matched true vertices, the closest one is selected using
    a scaled distance metric.

    Parameters
    ----------
    rec_vertices : list of dict
        Each dict must contain:
            - 'rec_z'
            - 'rec_t'

        This function adds the following keys in-place:
            - 'matched_true_vertices'
            - 'closest_true_vertex'
            - 'matched_true_distances' (true ID to squared normalized distance)
            - 'n_matches'

    true_vertices : list of dict
        Each dict must contain:
            - 'vertex_id'
            - 'z'
            - 't'

    m_r : float
        Matching radius parameter (mm).

    z_scale : float, optional
        Scale factor for z ellipse diameter. The ellipse semi-axis in z is
        sigma_z = z_scale * m_r / 2 (default: 1.5).

    t_scale : float, optional
        Scale factor for t ellipse diameter. The ellipse semi-axis in t is
        sigma_t = t_scale * m_r / 2 (default: 0.06).
    """

    sigma_z = z_scale * m_r / 2.0
    sigma_t = t_scale * m_r / 2.0

    for rec in rec_vertices:
        rec_z = rec["rec_z"]
        rec_t = rec["rec_t"]

        matched = []
        distances = {}
        closest: dict | None = None
        min_dist = float("inf")

        for tru in true_vertices:
            dz = tru["z"] - rec_z
            dt = tru["t"] - rec_t

            # Elliptical cut
            if (dz * dz) / (sigma_z * sigma_z) + (dt * dt) / (sigma_t * sigma_t) <= 1.0:
                matched.append(tru)

                # Distance metric for closest match
                dist = (dz / sigma_z) ** 2 + (dt / sigma_t) ** 2
                distances[tru["vertex_id"]] = dist
                if dist < min_dist:
                    min_dist = dist
                    closest = tru

        rec["matched_true_vertices"] = matched
        rec["closest_true_vertex"] = closest
        rec["matched_true_distances"] = distances
        rec["n_matches"] = len(matched)


def match_vertices_resolution_zt(
    rec_vertices: list[dict],
    true_vertices: list[dict],
    nsigma: float = 5.0,
) -> None:
    """
    Match reconstructed vertices to true vertices using a resolution-based
    elliptical cut in (z, t) space.

    The matching criterion is:

        (dz / (nsigma * z_err))^2 + (dt / (nsigma * t_err))^2 <= 1

    i.e. an ellipse with semi-axes:
        a_z = nsigma * z_err
        a_t = nsigma * t_err

    Adds the same match fields as ``match_vertices_elliptical_zt``, including
    ``matched_true_distances`` for all in-ellipse candidates.

    Parameters
    ----------
    rec_vertices : list of dict
        Each dict must contain:
            - 'rec_z'
            - 'rec_t'
            - 'z_err'
            - 't_err'

    true_vertices : list of dict
        Each dict must contain:
            - 'vertex_id'
            - 'z'
            - 't'

    nsigma : float
        Size of matching window in units of resolution (default: 5σ)
    """

    for rec in rec_vertices:
        rec_z = rec["rec_z"]
        rec_t = rec["rec_t"]

        z_err = rec.get("z_err", None)
        t_err = rec.get("t_err", None)

        # --- Safety: skip bad fits ---
        if z_err is None or t_err is None or z_err <= 0 or t_err <= 0:
            rec["matched_true_vertices"] = []
            rec["closest_true_vertex"] = None
            rec["matched_true_distances"] = {}
            rec["n_matches"] = 0
            continue

        sigma_z = nsigma * z_err
        sigma_t = nsigma * t_err

        matched = []
        distances = {}
        closest: dict | None = None
        min_dist = float("inf")

        for tru in true_vertices:
            dz = tru["z"] - rec_z
            dt = tru["t"] - rec_t

            # --- Elliptical cut (resolution-based) ---
            val = (dz / sigma_z) ** 2 + (dt / sigma_t) ** 2

            if val <= 1.0:
                matched.append(tru)
                distances[tru["vertex_id"]] = val

                # --- normalized distance ---
                if val < min_dist:
                    min_dist = val
                    closest = tru

        rec["matched_true_vertices"] = matched
        rec["closest_true_vertex"] = closest
        rec["matched_true_distances"] = distances
        rec["n_matches"] = len(matched)


def claim_vertex_matches(rec_vertices):
    """Return event-local {rec_id: true_id} claims by true-vertex proposals.

    Each unmatched true vertex applies to its next reconstructed candidate in
    increasing squared normalized distance. A rec keeps its closest applicant,
    replacing its current owner if a better true arrives later. Rejected or
    displaced true vertices continue until matched or out of candidates.

    Equal distances prefer the smaller rec ID for a true's candidates and the
    smaller true ID for a rec's applicants. Both sides thus use the same pair
    priority (distance, rec_id, true_id). The result equals globally sorted
    greedy matching, but does not maximize the number of pairs.

    Every candidate pair is proposed at most once, with no iteration/retry cap.
    Input candidate metadata is read only, preserving merge/split associations.
    """
    # candidates[true_id]: that true's eligible (distance, rec_id) options.
    # distances_by_rec[rec_id]: read-only true_id -> distance lookup, used to
    # compare a new applicant with the rec's current owner.
    candidates = {}
    distances_by_rec = {}
    for rec in rec_vertices:  # One reconstructed vertex and its candidate data.
        rec_id = rec["rec_vertex_id"]  # Event-local reconstructed vertex ID.
        distances = rec["matched_true_distances"]  # Precomputed in-window scores.
        distances_by_rec[rec_id] = distances
        for true_id, distance in distances.items():
            # true_id identifies a true vertex; distance is the squared
            # normalized separation of this particular true/rec pair.
            candidates.setdefault(true_id, []).append((distance, rec_id))
    for options in candidates.values():
        # Each true tries closer recs first; equal scores prefer smaller IDs.
        options.sort()

    # next_index[true_id]: first candidate this true has not yet tried.
    next_index = dict.fromkeys(candidates, 0)
    # pending contains only unmatched true IDs waiting for another attempt.
    # Trues with no eligible candidate need no attempt and are absent here.
    pending = deque(sorted(candidates))
    # matches[rec_id]: the true provisionally held by this rec; a missing key
    # means the rec is free. An accepted true leaves pending until displaced.
    matches = {}

    while pending:
        true_id = pending.popleft()  # Unmatched true making the next proposal.
        index = next_index[true_id]  # Its next untried candidate's position.
        if index == len(candidates[true_id]):
            continue  # Exhausted candidates: leave this true unmatched.

        distance, rec_id = candidates[true_id][index]
        # Advance before applying, so rejection or displacement never causes
        # this true to repeat a proposal it has already made.
        next_index[true_id] += 1
        previous_true_id = matches.get(rec_id)  # Current owner, or None if free.

        if previous_true_id is None:
            matches[rec_id] = true_id  # Free rec accepts; true stops queuing.
        else:
            previous_distance = distances_by_rec[rec_id][previous_true_id]
            # rec_id is identical for both applicants, so their common pair
            # priority reduces to (distance, true_id) for this comparison.
            if (distance, true_id) < (previous_distance, previous_true_id):
                matches[rec_id] = true_id  # Later, better applicant takes over.
                pending.append(previous_true_id)  # Displaced owner tries next.
            else:
                pending.append(true_id)  # Rejected applicant tries next.

    return matches


def find_best_match(
    rec_track_indices, true_vertex_info, recall_threshold, purity_threshold
):
    """Find the best matching true vertex for a reconstructed vertex.

    A match requires BOTH:
    - Recall = intersection / num_true_tracks >= recall_threshold
    - Purity = intersection / num_rec_tracks >= purity_threshold

    Args:
        rec_track_indices: set of track indices for the reconstructed vertex
        true_vertex_info: dict {vertex_id: {"track_indices": set, ...}}
        recall_threshold: minimum recall for a match
        purity_threshold: minimum purity for a match

    Returns:
        Best matching vertex ID or None if no match found
    """
    best_match_id = None
    best_recall = 0.0

    num_rec_tracks = len(rec_track_indices)
    if num_rec_tracks == 0:
        return None

    for vertex_id, vertex_data in true_vertex_info.items():
        true_track_indices = vertex_data["track_indices"]
        num_true_tracks = len(true_track_indices)

        if num_true_tracks == 0:
            continue

        # Calculate overlap metrics
        num_shared = len(rec_track_indices & true_track_indices)
        recall = num_shared / num_true_tracks
        purity = num_shared / num_rec_tracks

        # Both recall and purity must exceed their thresholds, and recall should be best so far
        if (
            recall >= recall_threshold
            and purity >= purity_threshold
            and recall > best_recall
        ):
            best_recall = recall
            best_match_id = vertex_id

    return best_match_id
