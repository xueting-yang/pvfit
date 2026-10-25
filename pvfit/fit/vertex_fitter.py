"""
vertex_fitter.py - Vertex fitting functions for PVfit

The loops and formulas mirror the GPU implementation on purpose (including the
closed-form 4x4 inverse). Track columns are read once into numpy arrays and
indexed positionally; every arithmetic step is the same as before.
"""

import numpy as np

from .utils import getTukeyWeight

# Track columns the fit reads, keyed by the short names used in the formulas.
_TRACK_COLUMNS = {
    "backward": "backward",
    "tx": "beamPOCA_tx",
    "ty": "beamPOCA_ty",
    "z": "beamPOCA_z",
    "x": "beamPOCA_x",
    "y": "beamPOCA_y",
    "t": "beamPOCA_t",
    "c00": "c00",
    "c11": "c11",
    "c55": "c55",
}
_CACHE_KEYS = (
    "HWH00_cache",
    "HWH11_cache",
    "HWH22_cache",
    "HWH02_cache",
    "HWH12_cache",
    "HWH23_cache",
    "HWH33_cache",
)


def track_arrays(event_data):
    """Positional numpy views of the track columns, original dtypes kept."""
    return {
        name: event_data[column].to_numpy() for name, column in _TRACK_COLUMNS.items()
    }


def extract_track_info_for_chi2(tracks, track_index):
    """Track parameters and covariance elements of one track, for chi2."""
    return {name: values[track_index] for name, values in tracks.items()}


def calculate_track_vertex_chi2(track_info, vertex_info):
    """
    Calculate chi-squared between track and vertex positions.

    Args:
        track_info: Dictionary with track parameters and covariance elements
        vertex_info: Dictionary with vertex position (rec_z, rec_x, rec_y, rec_t)

    Returns:
        dict: Chi2 value, covariances, and residuals
    """
    # Extract data from dictionaries
    backward = track_info["backward"]
    tx = track_info["tx"]
    ty = track_info["ty"]
    track_z = track_info["z"]
    track_x = track_info["x"]
    track_y = track_info["y"]
    track_t = track_info["t"]
    c00 = track_info["c00"]  # x_x covariance
    c11 = track_info["c11"]  # y_y covariance
    c55 = track_info["c55"]  # t_t covariance

    vertex_rec_z = vertex_info["rec_z"]
    vertex_rec_x = vertex_info["rec_x"]
    vertex_rec_y = vertex_info["rec_y"]
    vertex_rec_t = vertex_info["rec_t"]

    # Calculate the sign and time slope
    sgn = -1 if backward else 1
    tb_tmp = sgn * np.sqrt(1 + tx**2 + ty**2) / 299.792458  # mm/ns

    # Calculate delta z and residuals
    delta_z = vertex_rec_z - track_z
    res_x = track_x - vertex_rec_x + delta_z * tx
    res_y = track_y - vertex_rec_y + delta_z * ty
    res_t = track_t - vertex_rec_t + delta_z * tb_tmp

    # Track covariances are not extrapolated to the vertex (the GPU code's
    # "switch 2"); c22/c20/c33/c31 would enter here otherwise.
    Track_cov_xx = c00
    Track_cov_yy = c11
    Track_cov_tt = c55

    # Calculate chi-squared
    trackvtxchi2 = (
        res_x**2 / Track_cov_xx + res_y**2 / Track_cov_yy + res_t**2 / Track_cov_tt
    )

    return {
        "chi2": trackvtxchi2,
        "cov_xx": Track_cov_xx,
        "cov_yy": Track_cov_yy,
        "cov_tt": Track_cov_tt,
        "tb_tmp": tb_tmp,
        "res_x": res_x,
        "res_y": res_y,
        "res_t": res_t,
    }


def _new_cache(n_tracks):
    cache = {"trackvtxchi2_cache": np.full(n_tracks, np.inf)}
    for key in _CACHE_KEYS:
        cache[key] = np.zeros(n_tracks)
    return cache


def _fill_cache(cache, tracks, i, result):
    """Store the chi2 and the HWH matrix elements of track i."""
    tb_tmp = result["tb_tmp"]
    c00, c11, c55 = tracks["c00"][i], tracks["c11"][i], tracks["c55"][i]
    tx, ty = tracks["tx"][i], tracks["ty"][i]
    cache["trackvtxchi2_cache"][i] = result["chi2"]
    cache["HWH00_cache"][i] = 1 / c00
    cache["HWH11_cache"][i] = 1 / c11
    cache["HWH22_cache"][i] = tx**2 / c00 + ty**2 / c11 + tb_tmp**2 / c55
    cache["HWH02_cache"][i] = -tx / c00
    cache["HWH12_cache"][i] = -ty / c11
    cache["HWH23_cache"][i] = -tb_tmp / c55
    cache["HWH33_cache"][i] = 1 / c55


def _assign_track_to_closest_vertex(
    tracks, i, vertex_rec_info, max_track_seed_chi2, cache
):
    """Find closest vertex for track i, assign it, and update cache.

    Returns True if the track was assigned, False otherwise.
    """
    c00, c11, c55 = tracks["c00"][i], tracks["c11"][i], tracks["c55"][i]
    if c00 <= 0 or c11 <= 0 or c55 <= 0:
        print(
            f"Warning: Non-positive covariance at index {i}: c00={c00}, c11={c11}, c55={c55}"
        )
        return False

    track_info = extract_track_info_for_chi2(tracks, i)
    vertex_closest = None
    result_closest = None
    trackvtxchi2_min = max_track_seed_chi2

    for vertex in vertex_rec_info:
        result = calculate_track_vertex_chi2(track_info, vertex)
        if result["chi2"] < trackvtxchi2_min:
            trackvtxchi2_min = result["chi2"]
            vertex_closest = vertex
            result_closest = result

    if vertex_closest is None:
        return False

    vertex_closest["track_index"].append(i)
    _fill_cache(cache, tracks, i, result_closest)
    return True


def _build_cache_for_vertex(tracks, vertex):
    """Build HWH cache for a single vertex using its existing track_index.

    Unlike assign_tracks_to_vertices, this does NOT re-assign tracks.
    It only computes the cached matrix elements needed by fit_vertices.
    """
    cache = _new_cache(len(tracks["z"]))
    for i in vertex["track_index"]:
        if tracks["c00"][i] <= 0 or tracks["c11"][i] <= 0 or tracks["c55"][i] <= 0:
            continue
        track_info = extract_track_info_for_chi2(tracks, i)
        _fill_cache(cache, tracks, i, calculate_track_vertex_chi2(track_info, vertex))
    return cache


def assign_tracks_to_vertices(
    tracks, vertex_rec_info, *, max_track_seed_chi2, min_tracks_per_seed
):
    """
    Assign tracks to closest vertices using chi2 distance (Allen's
    pv_beamline_assign_tracks).

    Args:
        tracks: track arrays from track_arrays()
        vertex_rec_info: List of reconstructed vertex dictionaries
        max_track_seed_chi2: a track joins its closest seed only below this chi2
        min_tracks_per_seed: seeds with fewer tracks are removed and their
                             tracks reassigned to the nearest remaining seed;
                             0 keeps every seed

    Returns:
        dict: Cache containing chi2 values and HWH matrices for fitting
    """
    n_tracks = len(tracks["z"])
    cache = _new_cache(n_tracks)

    # Initial track-to-vertex assignment
    for i in range(n_tracks):
        _assign_track_to_closest_vertex(
            tracks, i, vertex_rec_info, max_track_seed_chi2, cache
        )

    # Remove seeds with too few tracks, reassign orphaned tracks
    if min_tracks_per_seed > 0:
        orphaned_tracks = []
        vertices_to_remove = [
            v for v in vertex_rec_info if len(v["track_index"]) < min_tracks_per_seed
        ]
        for vertex in vertices_to_remove:
            orphaned_tracks.extend(vertex["track_index"])
            vertex_rec_info.remove(vertex)

        # Clear cache and reassign orphaned tracks to remaining vertices
        for i in orphaned_tracks:
            cache["trackvtxchi2_cache"][i] = np.inf
            for key in _CACHE_KEYS:
                cache[key][i] = 0

        if vertex_rec_info:
            for i in orphaned_tracks:
                _assign_track_to_closest_vertex(
                    tracks, i, vertex_rec_info, max_track_seed_chi2, cache
                )

    return cache


def fit_vertices(
    tracks,
    vertex_rec_info,
    cache_data,
    *,
    max_chi2_tukey,
    max_fit_iterations,
    max_delta_z_converged,
    is_adaptive_tukey=True,
    max_vertex_rho2=None,
):
    """
    Fit vertex parameters using assigned tracks with Tukey reweighting
    (Allen's pv_beamline_fitter).

    Args:
        tracks: track arrays from track_arrays()
        vertex_rec_info: List of reconstructed vertex dictionaries
        cache_data: Dictionary with cached chi2 values and HWH matrices
        max_chi2_tukey: Tukey bisquare threshold of the track chi2
        max_fit_iterations: iterations of the fit
        max_delta_z_converged: the fit stops once a z step is smaller (mm)
        is_adaptive_tukey: If True, recompute chi2 each iteration for Tukey weight;
                        if False, use cached chi2 from initial track assignment
        max_vertex_rho2: fitted vertices with x^2 + y^2 at or above this (mm^2)
                         are removed from vertex_rec_info, as Allen drops
                         vertices off the beamline; None keeps them all

    Returns:
        None: Modifies vertex_rec_info in-place with fitted parameters; seeds
        whose fit is singular or off the beamline are removed from it
    """
    trackvtxchi2_cache = cache_data["trackvtxchi2_cache"]
    HWH00_cache = cache_data["HWH00_cache"]
    HWH11_cache = cache_data["HWH11_cache"]
    HWH22_cache = cache_data["HWH22_cache"]
    HWH02_cache = cache_data["HWH02_cache"]
    HWH12_cache = cache_data["HWH12_cache"]
    HWH23_cache = cache_data["HWH23_cache"]
    HWH33_cache = cache_data["HWH33_cache"]

    c00 = tracks["c00"]
    c11 = tracks["c11"]
    c55 = tracks["c55"]
    tx = tracks["tx"]
    ty = tracks["ty"]

    rejected = []
    for vertex in vertex_rec_info:
        Params_rec_x = vertex["rec_x"]  # Initial values
        Params_rec_y = vertex["rec_y"]  # Initial values
        Params_rec_z = vertex["rec_z"]  # Initial values
        Params_rec_t = vertex["rec_t"]  # Initial values

        maxIter = max_fit_iterations
        minIter = 1
        Iter = 0
        maxDeltaZ = max_delta_z_converged
        converged = False
        cov_00 = cov_11 = cov_22 = cov_33 = None

        # Iterative fitting
        while Iter < minIter or (Iter < maxIter and (not converged)):
            sum_weight = 0
            halfD2Chi2DX2_00 = 0.0
            halfD2Chi2DX2_02 = 0.0
            halfD2Chi2DX2_11 = 0.0
            halfD2Chi2DX2_12 = 0.0
            halfD2Chi2DX2_22 = 0.0
            halfD2Chi2DX2_23 = 0.0
            halfD2Chi2DX2_33 = 0.0
            halfDChi2DX_0 = 0
            halfDChi2DX_1 = 0
            halfDChi2DX_2 = 0
            halfDChi2DX_3 = 0
            totol_chi2 = 0

            vertex_data = {
                "rec_z": Params_rec_z,
                "rec_x": Params_rec_x,
                "rec_y": Params_rec_y,
                "rec_t": Params_rec_t,
            }
            for i in vertex[
                "track_index"
            ]:  # Loop over all tracks assigned to the vertex
                track_data = extract_track_info_for_chi2(tracks, i)
                result = calculate_track_vertex_chi2(track_data, vertex_data)

                if is_adaptive_tukey:
                    weight = getTukeyWeight(result["chi2"], max_chi2=max_chi2_tukey)
                else:
                    weight = getTukeyWeight(
                        trackvtxchi2_cache[i], max_chi2=max_chi2_tukey
                    )
                sum_weight += weight
                totol_chi2 += weight * result["chi2"]

                halfD2Chi2DX2_00 += weight * HWH00_cache[i]
                halfD2Chi2DX2_02 += weight * HWH02_cache[i]
                halfD2Chi2DX2_11 += weight * HWH11_cache[i]
                halfD2Chi2DX2_12 += weight * HWH12_cache[i]
                halfD2Chi2DX2_22 += weight * HWH22_cache[i]
                halfD2Chi2DX2_23 += weight * HWH23_cache[i]
                halfD2Chi2DX2_33 += weight * HWH33_cache[i]

                # Calculate gradient
                halfDChi2DX_0 += weight * (-result["res_x"] / c00[i])
                halfDChi2DX_1 += weight * (-result["res_y"] / c11[i])
                halfDChi2DX_2 += weight * (
                    result["res_y"] * ty[i] / c11[i]
                    + result["res_x"] * tx[i] / c00[i]
                    + result["res_t"] * result["tb_tmp"] / c55[i]
                )
                halfDChi2DX_3 += weight * (-result["res_t"] / c55[i])

            # Matrix inversion
            a00 = halfD2Chi2DX2_00
            a11 = halfD2Chi2DX2_11
            a22 = halfD2Chi2DX2_22
            a33 = halfD2Chi2DX2_33
            a02 = halfD2Chi2DX2_02
            a12 = halfD2Chi2DX2_12
            a23 = halfD2Chi2DX2_23

            # Gradient vector
            grad_x = halfDChi2DX_0
            grad_y = halfDChi2DX_1
            grad_z = halfDChi2DX_2
            grad_t = halfDChi2DX_3

            # Calculate determinant
            det = (
                a00 * a11 * a22 * a33
                - a00 * a11 * a23**2
                - a00 * a12**2 * a33
                - a02**2 * a11 * a33
            )

            if abs(det) < 1e-12:
                break

            # Inverse matrix elements
            cov_00 = (a11 * a22 * a33 - a11 * a23**2 - a12**2 * a33) / det
            cov_10 = a02 * a12 * a33 / det
            cov_11 = (a00 * a22 * a33 - a00 * a23**2 - a02**2 * a33) / det
            cov_20 = -a02 * a11 * a33 / det
            cov_21 = -a00 * a12 * a33 / det
            cov_22 = a00 * a11 * a33 / det
            cov_30 = a02 * a11 * a23 / det
            cov_31 = a00 * a12 * a23 / det
            cov_32 = -a00 * a11 * a23 / det
            cov_33 = (a00 * a11 * a22 - a00 * a12**2 - a02**2 * a11) / det

            delta_x = (
                cov_00 * grad_x + cov_10 * grad_y + cov_20 * grad_z + cov_30 * grad_t
            )
            delta_y = (
                cov_10 * grad_x + cov_11 * grad_y + cov_21 * grad_z + cov_31 * grad_t
            )
            delta_z = (
                cov_20 * grad_x + cov_21 * grad_y + cov_22 * grad_z + cov_32 * grad_t
            )
            delta_t = (
                cov_30 * grad_x + cov_31 * grad_y + cov_32 * grad_z + cov_33 * grad_t
            )

            # Update parameters
            Params_rec_x -= delta_x
            Params_rec_y -= delta_y
            Params_rec_z -= delta_z
            Params_rec_t -= delta_t

            converged = abs(delta_z) < maxDeltaZ
            Iter += 1

        if cov_00 is None:
            # Singular on the first iteration: no fit and no vertex, as Allen's
            # fit_vertex reports failure and the seed is not written out.
            rejected.append(vertex)
            continue

        # Vertices off the beamline are not primary vertices (Allen: rho2 cut
        # after the fit, relative to the beamline, which is the origin here).
        if (
            max_vertex_rho2 is not None
            and Params_rec_x**2 + Params_rec_y**2 >= max_vertex_rho2
        ):
            rejected.append(vertex)
            continue

        # Save fitted parameters
        vertex["Params_rec_x"] = Params_rec_x
        vertex["Params_rec_y"] = Params_rec_y
        vertex["Params_rec_z"] = Params_rec_z
        vertex["Params_rec_t"] = Params_rec_t
        vertex["cov_00"] = cov_00
        vertex["cov_11"] = cov_11
        vertex["cov_22"] = cov_22
        vertex["cov_33"] = cov_33

    for vertex in rejected:
        vertex_rec_info.remove(vertex)


def fit_reconstructed_vertices(
    event_data_selected,
    vertex_rec_info,
    *,
    max_track_seed_chi2,
    min_tracks_per_seed,
    max_chi2_tukey,
    max_fit_iterations,
    max_delta_z_converged,
    max_vertex_rho2,
    is_adaptive_tukey=True,
):
    """
    Assign tracks to the seeds and fit every seed of the event.

    Args:
        event_data_selected: Input track DataFrame, including rows outside the seed window
        vertex_rec_info: List of reconstructed vertex dictionaries
        max_track_seed_chi2, min_tracks_per_seed: see assign_tracks_to_vertices
        max_chi2_tukey, max_fit_iterations, max_delta_z_converged,
        max_vertex_rho2, is_adaptive_tukey: see fit_vertices

    Returns:
        None: Modifies vertex_rec_info in-place with fitted parameters
    """
    tracks = track_arrays(event_data_selected)

    # Step 1: Assign tracks to vertices, we cache some data here for fitting
    cache_data = assign_tracks_to_vertices(
        tracks,
        vertex_rec_info,
        max_track_seed_chi2=max_track_seed_chi2,
        min_tracks_per_seed=min_tracks_per_seed,
    )

    # Step 2: Fit vertices
    fit_vertices(
        tracks,
        vertex_rec_info,
        cache_data,
        max_chi2_tukey=max_chi2_tukey,
        max_fit_iterations=max_fit_iterations,
        max_delta_z_converged=max_delta_z_converged,
        is_adaptive_tukey=is_adaptive_tukey,
        max_vertex_rho2=max_vertex_rho2,
    )


def merge_close_vertices(
    event_data_selected,
    vertex_rec_info,
    *,
    min_chi2_dist_merge,
    max_chi2_tukey,
    max_fit_iterations,
    max_delta_z_converged,
    is_adaptive_tukey=True,
):
    """
    Merge reconstructed vertices that are close in (z, t) space after fitting
    (Allen's pv_beamline_merge_vertices).

    For each pair of fitted vertices, compute a chi2 distance:
        chi2 = (dz / sigma_z)^2 + (dt / sigma_t)^2
    where sigma_z = sqrt(cov_zz_i + cov_zz_j), sigma_t = sqrt(cov_tt_i + cov_tt_j).
    Pairs with chi2 < min_chi2_dist_merge are merged: their track lists are
    combined, and the merged vertex is re-fitted. A threshold of 0 merges nothing.

    Single pass: find all pairs within threshold, merge each pair directly.

    Args:
        event_data_selected: Input track DataFrame, including rows outside the seed window
        vertex_rec_info: List of reconstructed vertex dictionaries (modified in-place)
        min_chi2_dist_merge: chi2 distance below which two vertices are one
        max_chi2_tukey, max_fit_iterations, max_delta_z_converged,
        is_adaptive_tukey: fit settings of the re-fit (see fit_vertices)

    Returns:
        None: Modifies vertex_rec_info in-place
    """
    # Only consider fitted vertices (those with covariance information)
    fitted = [
        v
        for v in vertex_rec_info
        if "Params_rec_z" in v and "cov_22" in v and "cov_33" in v
    ]

    if len(fitted) < 2:
        return

    chi2_threshold = min_chi2_dist_merge

    # ------------------------------------------------------------------
    # Find pairs to merge
    # ------------------------------------------------------------------
    pairs_to_merge = []
    for idx_a in range(len(fitted)):
        for idx_b in range(idx_a + 1, len(fitted)):
            vi = fitted[idx_a]
            vj = fitted[idx_b]

            dz = vi["Params_rec_z"] - vj["Params_rec_z"]
            dt = vi["Params_rec_t"] - vj["Params_rec_t"]
            sigma_z_sq = vi["cov_22"] + vj["cov_22"]
            sigma_t_sq = vi["cov_33"] + vj["cov_33"]

            if sigma_z_sq <= 0 or sigma_t_sq <= 0:
                continue

            chi2 = dz**2 / sigma_z_sq + dt**2 / sigma_t_sq

            if chi2 < chi2_threshold:
                pairs_to_merge.append((vi, vj, chi2))

    if not pairs_to_merge:
        return

    tracks = track_arrays(event_data_selected)

    # ------------------------------------------------------------------
    # Merge each pair: combine tracks, keep the one with more tracks as seed
    # ------------------------------------------------------------------
    already_merged = set()  # rec_vertex_id's that have been consumed

    for vi, vj, _chi2 in sorted(pairs_to_merge, key=lambda x: x[2]):
        if (
            vi["rec_vertex_id"] in already_merged
            or vj["rec_vertex_id"] in already_merged
        ):
            continue

        # Determine which vertex has more tracks → keep as seed
        if len(vi["track_index"]) >= len(vj["track_index"]):
            keep, drop = vi, vj
        else:
            keep, drop = vj, vi

        # Combine tracks (deduplicate)
        combined = list(set(keep["track_index"] + drop["track_index"]))

        # Build merged vertex using the kept vertex's fitted position as seed
        merged = {
            "rec_vertex_id": keep["rec_vertex_id"],
            "rec_z": keep["Params_rec_z"],
            "rec_t": keep["Params_rec_t"],
            "rec_x": keep["Params_rec_x"],
            "rec_y": keep["Params_rec_y"],
            "track_index": combined,
        }

        # Replace keep with merged, remove drop
        idx_keep = vertex_rec_info.index(keep)
        vertex_rec_info[idx_keep] = merged
        vertex_rec_info.remove(drop)

        already_merged.add(vi["rec_vertex_id"])
        already_merged.add(vj["rec_vertex_id"])

        # Re-fit only this merged vertex (Allen applies no rho2 cut here)
        cache_data = _build_cache_for_vertex(tracks, merged)
        fit_vertices(
            tracks,
            [merged],
            cache_data,
            max_chi2_tukey=max_chi2_tukey,
            max_fit_iterations=max_fit_iterations,
            max_delta_z_converged=max_delta_z_converged,
            is_adaptive_tukey=is_adaptive_tukey,
        )
        if "Params_rec_x" not in merged:
            # The merged fit was singular: no vertex, as for a single seed
            vertex_rec_info.remove(merged)


def select_members(event_data_selected, vertex_rec_info, *, max_chi2_tukey):
    """Final membership of every fitted vertex, as Allen's
    pv_beamline_merge_vertices writes it: of the tracks assigned to the seed
    (and absorbed by merging), those whose chi2 at the fitted position is below
    the Tukey threshold, i.e. with nonzero weight in the final fit. The
    assignment list is kept as ``assigned_index``; ``track_index`` becomes the
    member list the output stores. Vertices without a fit are left unchanged.
    """
    fitted = [v for v in vertex_rec_info if all(f"Params_rec_{c}" in v for c in "xyzt")]
    if not fitted:
        return
    tracks = track_arrays(event_data_selected)
    for vertex in fitted:
        position = {f"rec_{c}": vertex[f"Params_rec_{c}"] for c in "xyzt"}
        assigned = list(vertex["track_index"])
        vertex["assigned_index"] = assigned
        vertex["track_index"] = [
            i
            for i in assigned
            if calculate_track_vertex_chi2(
                extract_track_info_for_chi2(tracks, i), position
            )["chi2"]
            < max_chi2_tukey
        ]
