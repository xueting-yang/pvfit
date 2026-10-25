"""
seedings_finder.py - Seedings finder functions for PVfit
"""

import numpy as np

from .utils import gauss_integral, integ_for_threshold_cut


def find_seedings(histogram, params):
    """
    Find vertex seeding candidates by identifying local maxima in the 2D (z,t) histogram.

    Searches for bins with sufficient track density that represent local maxima
    compared to their 8 neighboring bins. These peaks indicate potential vertex positions.

    Args:
        histogram (numpy.ndarray): 2D histogram array of track density in (z,t) space
        params (dict): Configuration parameters including:
            - z_min, t_min: Minimum z and t values
            - n_bins_z, n_bins_t: Number of bins in z and t dimensions
            - bin_size_z, bin_size_t: Bin widths
            - scaled_min_tracks_in_seed: 3x3 integral a maximum needs at this binning

    Returns:
        list: List of (z, t) tuples representing seeding candidates in physical coordinates
    """
    seedings = []
    # Extract parameters
    zMin = params["z_min"]
    tMin = params["t_min"]
    nZbins = params["n_bins_z"]
    nTbins = params["n_bins_t"]
    minTracks = params["scaled_min_tracks_in_seed"]

    # Calculate bin width
    zW = params["bin_size_z"]
    tW = params["bin_size_t"]
    # Iterate over the histogram (skip boundaries)
    for z in range(1, nZbins - 1):
        for t in range(1, nTbins - 1):
            v = histogram[z][t]
            # Check if it is a local maximum in a 3x3 region
            is_maximum = (
                v > histogram[z + 1][t]
                and v > histogram[z - 1][t]
                and v > histogram[z][t + 1]
                and v > histogram[z][t - 1]
                and v > histogram[z + 1][t + 1]
                and v > histogram[z + 1][t - 1]
                and v > histogram[z - 1][t + 1]
                and v > histogram[z - 1][t - 1]
            )
            if not is_maximum:
                continue
            # Refine the seed position with quadratic interpolation.
            zNi_p1 = histogram[z + 1][t]
            zNi = histogram[z][t]
            zNi_m1 = histogram[z - 1][t]
            # the first 0.5 is to center the position in the bin
            zp = z + 0.5 + (zNi_p1 - zNi_m1) / (2 * zNi - zNi_p1 - zNi_m1) * 0.5

            tNi_p1 = histogram[z][t + 1]
            tNi = histogram[z][t]
            tNi_m1 = histogram[z][t - 1]
            # the first 0.5 is to center the position in the bin
            tp = t + 0.5 + (tNi_p1 - tNi_m1) / (2 * tNi - tNi_p1 - tNi_m1) * 0.5

            # Calculate the integral of the 3x3 region
            s9, max_val = integ_for_threshold_cut(histogram, z, t, 1)
            if s9 >= minTracks and v == max_val:
                seedings.append((zp * zW + zMin, tp * tW + tMin))
    return seedings


def fill_histogram_for_seedings(event_data, params):
    """
    Create a 2D histogram of track density in (z,t) space for vertex seeding.

    Each track contributes to the histogram based on its beam point-of-closest-approach
    position and uncertainties. The contribution is Gaussian-smeared using widths
    derived from the track covariance, but only within a fixed window of neighboring
    bins around the track position.

    Args:
        event_data (pd.DataFrame): Track data containing columns:
            - beamPOCA_z, beamPOCA_t: Track positions
            - c00, c11, c55: Covariance matrix elements
            - tx, ty: Track slopes
        params (dict): Configuration parameters including:
            - z_min, t_min: Histogram boundaries
            - n_bins_z, n_bins_t: Number of histogram bins
            - bin_size_z, bin_size_t: Bin sizes
            - smear_window_nbins: Half-width of the fixed bin window used for Gaussian smearing
            - max_track_beamline_chi2_xy: tracks with a larger xy chi2 to the
              beamline are skipped (inf: no cut), as in Allen's pv_beamline_histo
            - max_track_z0_err: tracks with a larger z error at the beamline are
              skipped (inf: no cut)

    Returns:
        numpy.ndarray: 2D histogram array with track density values
    """
    zmin = params["z_min"]
    tmin = params["t_min"]
    Nbins_z = params["n_bins_z"]
    Nbins_t = params["n_bins_t"]
    bin_size_z = params["bin_size_z"]
    bin_size_t = params["bin_size_t"]
    smear_window_size = params["smear_window_nbins"]
    max_track_bl_chi2 = params["max_track_beamline_chi2_xy"]
    pp_max_track_z0_err = params["max_track_z0_err"]

    # Track columns as positional arrays (original dtypes kept)
    z = event_data["beamPOCA_z"].to_numpy()
    t = event_data["beamPOCA_t"].to_numpy()
    W_00 = (1.0 / event_data["c00"]).to_numpy()
    W_11 = (1.0 / event_data["c11"]).to_numpy()
    c55 = event_data["c55"].to_numpy()
    tx = event_data["beamPOCA_tx"].to_numpy()
    ty = event_data["beamPOCA_ty"].to_numpy()
    x = event_data["beamPOCA_x"].to_numpy()
    y = event_data["beamPOCA_y"].to_numpy()

    # Create histogram
    H = np.zeros((Nbins_z, Nbins_t))

    for i in range(len(z)):
        # Beamline chi2 cut (an infinite threshold keeps every track)
        blchi2 = x[i] ** 2 * W_00[i] + y[i] ** 2 * W_11[i]
        if blchi2 >= max_track_bl_chi2:
            continue

        zbin = (z[i] - zmin) / bin_size_z
        tbin = (t[i] - tmin) / bin_size_t

        # Debug: check for problematic values
        weight_term = W_00[i] * tx[i] ** 2 + W_11[i] * ty[i] ** 2
        if weight_term <= 0:
            print(f"Warning: Non-positive weight term at index {i}: {weight_term}")
            continue

        zerr = 1 / np.sqrt(weight_term)
        terr = np.sqrt(c55[i])

        if zerr >= pp_max_track_z0_err:
            continue

        # Restrict the Gaussian smearing to a fixed window in bin units.
        minbin_z = max(0, int(zbin - smear_window_size))
        maxbin_z = min(Nbins_z - 1, int(zbin + smear_window_size))
        minbin_t = max(0, int(tbin - smear_window_size))
        maxbin_t = min(Nbins_t - 1, int(tbin + smear_window_size))

        # calculate the integral of the gaussian function for z
        cache_integral_z_list = []
        for zBinp in range(minbin_z, maxbin_z + 1):
            renorm_z = (zmin + (zBinp) * bin_size_z - z[i]) / zerr
            this_integral_z = gauss_integral(renorm_z)
            cache_integral_z_list.append(this_integral_z)
        cache_integral_z_diffs = []
        for it in range(len(cache_integral_z_list) - 1):
            diff = cache_integral_z_list[it + 1] - cache_integral_z_list[it]
            cache_integral_z_diffs.append(diff)

        # calculate the integral of the gaussian function for t
        cache_integral_t_list = []
        for tBinp in range(minbin_t, maxbin_t + 1):
            renorm_t = (tmin + (tBinp) * bin_size_t - t[i]) / terr
            this_integral_t = gauss_integral(renorm_t)
            cache_integral_t_list.append(this_integral_t)
        cache_integral_t_diffs = []
        for it in range(len(cache_integral_t_list) - 1):
            diff = cache_integral_t_list[it + 1] - cache_integral_t_list[it]
            cache_integral_t_diffs.append(diff)

        # fill the histogram
        for index_z in range(len(cache_integral_z_diffs)):
            for index_t in range(len(cache_integral_t_diffs)):
                H[minbin_z + index_z][minbin_t + index_t] += (
                    cache_integral_z_diffs[index_z] * cache_integral_t_diffs[index_t]
                )

    return H
