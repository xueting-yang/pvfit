"""
Event-level processing for PVfit (track-based version).

This module contains:
  - process_event: physics logic for a single event
  - process_event_wrapper: multiprocessing-compatible wrapper
"""

import pandas as pd

# Seed finding & vertex fitting
from .seedings_finder import fill_histogram_for_seedings, find_seedings
from .truth_matching import build_true_vertex_info, select_tracks
from .vertex_fitter import (
    fit_reconstructed_vertices,
    merge_close_vertices,
    select_members,
)


def process_event_wrapper(args):
    """
    Wrapper for multiprocessing.

    Args:
        args (tuple): (event_number, event_track_data, event_mcpv_data, params)

    Returns:
        dict: output of process_event
    """
    event_number, event_track_data, event_mcpv_data, params = args
    return process_event(event_number, event_track_data, event_mcpv_data, params)


def process_event(
    event_number: int,
    event_track_data: pd.DataFrame,
    event_mcpv_data: pd.DataFrame,
    params: dict,
) -> dict:
    """
    Process a single event:
      - read true vertices from MCPV data
      - apply z/t cuts only to seeding tracks
      - run seeding
      - assign and fit using all input tracks
      - build true / reco vertex structures

    NOTE:
      - Truth membership only; no reconstructed-to-true vertex matching
      - No efficiency / fake-rate analysis
      - Pure reconstruction stage

    Args:
        event_number: event ID
        event_track_data: pandas DataFrame with per-track information from monitor_tree
        event_mcpv_data: pandas DataFrame with MC true vertices from MCPV_effective_tree
        params: configuration dictionary

    Returns:
        dict containing:
          - event_number
          - n_true
          - n_seedings
          - seeds: every (z, t) seed, including ones later dropped or merged
          - n_fitted
          - histogram
          - vertex_true_info
          - vertex_rec_info
    """

    # ------------------------------------------------------------
    # 0. Fit settings shared by the first fit and the merge re-fit
    # ------------------------------------------------------------
    fit_settings = dict(
        max_chi2_tukey=params["max_chi2_tukey"],
        max_fit_iterations=params["max_fit_iterations"],
        max_delta_z_converged=params["max_delta_z_converged"],
        is_adaptive_tukey=params["is_adaptive_tukey"],
    )

    # ------------------------------------------------------------
    # 1. Build truth on all input rows; select tracks only for seeding
    # ------------------------------------------------------------
    event_data = event_track_data.reset_index(drop=True)
    vertex_true_info = build_true_vertex_info(
        event_data,
        event_mcpv_data,
    )
    seeding_tracks = select_tracks(event_data, params)

    # ------------------------------------------------------------
    # 3. Seeding
    # ------------------------------------------------------------
    H = fill_histogram_for_seedings(seeding_tracks, params)
    seedings = find_seedings(H, params)

    # ------------------------------------------------------------
    # 4. Initialize reconstructed vertices
    # ------------------------------------------------------------
    vertex_rec_info = []

    for i, (rec_z, rec_t) in enumerate(seedings):
        vertex_rec_info.append(
            {
                "rec_vertex_id": i,
                "rec_z": rec_z,
                "rec_t": rec_t,
                "rec_x": 0.0,
                "rec_y": 0.0,
                "track_index": [],
            }
        )

    # ------------------------------------------------------------
    # 5. Vertex fitting
    # ------------------------------------------------------------
    # Window-excluded tracks can still match an existing seed by chi2.
    fit_reconstructed_vertices(
        event_data,
        vertex_rec_info,
        max_track_seed_chi2=params["max_track_seed_chi2"],
        min_tracks_per_seed=params["min_tracks_per_seed"],
        max_vertex_rho2=params["max_vertex_rho2"],
        **fit_settings,
    )

    # ------------------------------------------------------------
    # 5b. Merge close vertices in (z, t) space (a threshold of 0 merges nothing)
    # ------------------------------------------------------------
    if params["min_chi2_dist_merge"] > 0:
        merge_close_vertices(
            event_data,
            vertex_rec_info,
            min_chi2_dist_merge=params["min_chi2_dist_merge"],
            **fit_settings,
        )

    # ------------------------------------------------------------
    # 6. Members: assigned tracks with nonzero Tukey weight at the fitted
    #    position (Allen's output convention); the assignment list stays
    #    available as assigned_index
    # ------------------------------------------------------------
    select_members(event_data, vertex_rec_info, max_chi2_tukey=params["max_chi2_tukey"])

    # ------------------------------------------------------------
    # 7. Collect summary numbers
    # ------------------------------------------------------------
    n_fitted = len([v for v in vertex_rec_info if "Params_rec_x" in v])

    # Truth, assignment, fitting and storage share the full row order;
    # reconstructed member indices therefore need no remapping.

    return {
        "event_number": event_number,
        "n_true": len(vertex_true_info),
        "n_seedings": len(seedings),
        "seeds": seedings,
        "n_fitted": n_fitted,
        "histogram": H,
        "vertex_true_info": vertex_true_info,
        "vertex_rec_info": vertex_rec_info,
        "event_data": event_data,
        "selected_for_seeding": event_data.index.isin(seeding_tracks.index),
    }
