"""Plot the 2D (z, t) seeding histogram with vertex overlays.

Works for any HDF5 file that follows the PVFit event layout, whichever program
produced it. Only ``histogram`` is required; each overlay is drawn when present:

    <event_id>/   @run_number, @source_event_number (original event identity)
    ├── histogram [n_bins_z, n_bins_t]
    │   @z_min, @z_max, @t_min, @t_max, @n_bins_z, @n_bins_t
    ├── seeds/          z, t                              (all seeds / peaks)
    ├── true_vertices/  z, t  [vertex_id, track_indices]  (labels)
    └── rec_vertices/   z, t  [z_err, t_err]              (error ellipses)

Files written before ``seeds/`` existed fall back to ``rec_vertices/{z,t}_seed``.
The optional file attribute ``@source`` (e.g. "PVFit", "Allen") labels the plot.
"""

from pathlib import Path

import h5py
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def _read_points(event_group, group_name, *optional, z="z", t="t"):
    """Read (z, t) and any optional per-point fields; None if not stored."""
    group = event_group.get(group_name)
    if group is None or z not in group or t not in group:
        return None
    points = {"z": group[z][:], "t": group[t][:]}
    for name in optional:
        if name in group:
            points[name] = group[name][:]
    return points


def _event_identity(event_group, *, required=False):
    """Read the original run/event pair; group names are internal event IDs."""
    names = ("run_number", "source_event_number")
    if any(name not in event_group.attrs for name in names):
        if required:
            raise ValueError(
                f"Event group {event_group.name} lacks run_number/source_event_number "
                "attributes required for selection by run/event"
            )
        return None
    values = tuple(event_group.attrs[name] for name in names)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
        for value in values
    ):
        raise ValueError(
            f"Event group {event_group.name}: run_number/source_event_number "
            "must be nonnegative integers"
        )
    return tuple(int(value) for value in values)


def load_event_histogram(hdf5_path, *, run_number=None, event_number=None):
    """Load one event's histogram and optional overlays from an HDF5 file.

    ``run_number`` and ``event_number`` select the original event identity,
    stored as the group's ``run_number`` and ``source_event_number`` attributes.
    Supply both, or omit both to load the first event (smallest internal ID).
    An explicit selection raises ValueError if the identity metadata is missing,
    the requested pair is absent, or more than one group has that pair.

    Returns a dict with the internal ``event_id``, original ``run_number`` /
    ``event_number`` (None for legacy files), source, histogram, z_edges, t_edges
    and true / seeds / rec overlays (None when absent). Returns None for an empty
    file with default selection, or if the selected event has no histogram.
    """
    if (run_number is None) != (event_number is None):
        raise ValueError("run_number and event_number must be provided together")
    for name, value in (("run_number", run_number), ("event_number", event_number)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer")

    with h5py.File(hdf5_path, "r") as f:
        available = sorted(f.keys(), key=int)
        if run_number is None:
            if not available:
                print("No events found in HDF5 file")
                return None
            event_group = f[available[0]]
        else:
            requested = (run_number, event_number)
            matches = [
                key
                for key in available
                if _event_identity(f[key], required=True) == requested
            ]
            if not matches:
                raise ValueError(
                    f"Event (run_number={run_number}, event_number={event_number}) "
                    "not found in HDF5 file"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Duplicate event identity (run_number={run_number}, "
                    f"event_number={event_number}) in HDF5 groups {matches}"
                )
            event_group = f[matches[0]]

        event_id = int(event_group.name.rsplit("/", 1)[-1])
        identity = _event_identity(event_group)
        if identity is None:
            print(f"Using first event: internal ID {event_id} (no run/event metadata)")
        else:
            print(
                f"Using run {identity[0]}, event {identity[1]} (internal ID {event_id})"
            )
        if "histogram" not in event_group:
            print(f"No histogram data found for event {event_id}")
            return None

        H = event_group["histogram"][:]
        hist_attrs = event_group["histogram"].attrs
        z_bins = int(hist_attrs["n_bins_z"])
        t_bins = int(hist_attrs["n_bins_t"])
        if H.shape != (z_bins, t_bins):
            raise ValueError(
                f"Event {event_id}: histogram shape {H.shape} does not match "
                f"(n_bins_z, n_bins_t) = {(z_bins, t_bins)}"
            )

        seeds = _read_points(event_group, "seeds")
        if seeds is None:
            seeds = _read_points(event_group, "rec_vertices", z="z_seed", t="t_seed")

        source = f.attrs.get("source")
        return {
            "event_id": event_id,
            "run_number": identity[0] if identity is not None else None,
            "event_number": identity[1] if identity is not None else None,
            "source": source.decode() if isinstance(source, bytes) else source,
            "histogram": H,
            "z_edges": np.linspace(
                hist_attrs["z_min"], hist_attrs["z_max"], z_bins + 1
            ),
            "t_edges": np.linspace(
                hist_attrs["t_min"], hist_attrs["t_max"], t_bins + 1
            ),
            "true": _read_points(
                event_group, "true_vertices", "vertex_id", "track_indices"
            ),
            "seeds": seeds,
            "rec": _read_points(event_group, "rec_vertices", "z_err", "t_err"),
        }


def plot_event_histogram(
    hdf5_path, output_dir, settings, *, run_number=None, event_number=None, label=None
):
    """Plot the seeding histogram and vertex overlays in the LHCb2 style.

    Plots the track density histogram used for vertex seeding, with overlaid markers
    for seeds, fitted vertices and true vertices when the file stores them.

    Args:
        hdf5_path: Path to the HDF5 file (PVFit event layout, any producer)
        output_dir: Directory for saving plots
        settings: AnalysisConfig; uses the matching scales and the draw_* flags
        run_number: Original run number; provide together with event_number
        event_number: Original event number, not the internal HDF5 group ID.
            Omit both numbers to plot the first stored event (smallest internal ID)
        label: Producer label; defaults to the file's @source attribute

    Returns:
        Path of the saved PDF, or None if the event could not be plotted
    """
    print("Plotting 2D Histogram...")

    draw_fixed_ellipses = settings.draw_fixed_ellipses
    if draw_fixed_ellipses:
        m_r, z_scale, t_scale = settings.m_r, settings.z_scale, settings.t_scale
        print(f"Using matching scales: m_r={m_r}, z_scale={z_scale}, t_scale={t_scale}")

    data = load_event_histogram(
        hdf5_path, run_number=run_number, event_number=event_number
    )
    if data is None:
        return None

    z_edges, t_edges = data["z_edges"], data["t_edges"]
    print(
        f"Histogram parameters: z_range=({z_edges[0]:g}, {z_edges[-1]:g}), "
        f"t_range=({t_edges[0]:g}, {t_edges[-1]:g}), "
        f"z_bins={len(z_edges) - 1}, t_bins={len(t_edges) - 1}"
    )

    with plt.rc_context():
        hep.style.use(hep.style.LHCb2)
        # Match the compact typography and inward ticks of the rate figures.
        plt.rcParams.update(
            {
                "axes.linewidth": 0.7,
                "font.weight": "normal",
                "xtick.top": False,
                "ytick.right": False,
                "xtick.major.size": 4.5,
                "ytick.major.size": 4.5,
                "xtick.minor.size": 2.2,
                "ytick.minor.size": 2.2,
                "xtick.major.width": 0.65,
                "ytick.major.width": 0.65,
                "xtick.minor.width": 0.55,
                "ytick.minor.width": 0.55,
                "xtick.major.pad": 3,
                "ytick.major.pad": 3,
            }
        )
        return _draw_event_histogram(data, output_dir, settings, label)


def _draw_event_histogram(data, output_dir, settings, label):
    """Render inside the caller's temporary LHCb2 style context."""
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    # Keep a conventional 4:3 data area; reserve compact space for the legend.
    fig.subplots_adjust(left=0.13, right=0.83, bottom=0.14, top=0.78)
    ax.set_box_aspect(0.75)
    z_edges, t_edges = data["z_edges"], data["t_edges"]

    # Gaussian bin integrals are fractional track weights. Preserve the linear
    # scale and hide only empty bins; rasterize the mesh, not text or overlays.
    H_plot = data["histogram"].astype(float).T
    H_plot[H_plot == 0] = np.nan
    im = ax.pcolormesh(
        z_edges, t_edges, H_plot, cmap="jet", shading="auto", rasterized=True
    )
    true, seeds, rec = data["true"], data["seeds"], data["rec"]
    true_color, seed_color, rec_color = "#bd1f36", "#218c47", "#d97706"
    # Thin white outlines separate the overlays from similarly coloured bins.
    marker_outline = [path_effects.withStroke(linewidth=1.6, foreground="white")]

    if true is not None and len(true["z"]) > 0:
        ax.scatter(
            true["z"],
            true["t"],
            s=24,
            facecolors="none",
            edgecolors=true_color,
            linewidths=0.8,
            marker="o",
            label=f"True vertices ({len(true['z'])})",
            path_effects=marker_outline,
            zorder=5,
        )
        ids = true.get("vertex_id", np.arange(len(true["z"])))
        for i in range(len(true["z"])):
            text = f"{int(ids[i])}"
            if "track_indices" in true:
                text += f"({len(true['track_indices'][i])})"
            ax.annotate(
                text,
                xy=(true["z"][i], true["t"][i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=true_color,
                fontweight="normal",
                path_effects=[path_effects.withStroke(linewidth=1, foreground="white")],
                zorder=6,
            )

    if seeds is not None and len(seeds["z"]) > 0:
        ax.scatter(
            seeds["z"],
            seeds["t"],
            color=seed_color,
            s=22,
            marker="x",
            linewidths=0.85,
            label=f"Seeds ({len(seeds['z'])})",
            path_effects=marker_outline,
            zorder=4,
        )

    if rec is not None and len(rec["z"]) > 0:
        ax.scatter(
            rec["z"],
            rec["t"],
            s=18,
            facecolors="none",
            edgecolors=rec_color,
            linewidths=0.8,
            marker="s",
            label=f"Reconstructed vertices ({len(rec['z'])})",
            path_effects=marker_outline,
            zorder=5,
        )

    if settings.draw_fixed_ellipses and seeds is not None:
        for i in range(len(seeds["z"])):
            ax.add_patch(
                patches.Ellipse(
                    xy=(seeds["z"][i], seeds["t"][i]),
                    width=settings.z_scale * settings.m_r,
                    height=settings.t_scale * settings.m_r,
                    fill=False,
                    edgecolor="0.4",
                    linestyle="--",
                    linewidth=0.7,
                    label="Fixed matching window" if i == 0 else "_nolegend_",
                    zorder=3,
                )
            )

    if (
        settings.draw_fitted_ellipses
        and rec is not None
        and "z_err" in rec
        and "t_err" in rec
    ):
        for i in range(len(rec["z"])):
            ax.add_patch(
                patches.Ellipse(
                    xy=(rec["z"][i], rec["t"][i]),
                    width=2 * settings.nsigma * rec["z_err"][i],
                    height=2 * settings.nsigma * rec["t_err"][i],
                    fill=False,
                    edgecolor="#a51b9c",
                    linewidth=0.65,
                    alpha=0.8,
                    label=rf"${settings.nsigma:g}\sigma$ uncertainty window"
                    if i == 0
                    else "_nolegend_",
                    zorder=3,
                )
            )

    # A separate colour-bar axis follows the actual data box after aspect is set.
    position = ax.get_position()
    cbar_axes = fig.add_axes([position.x1 + 0.025, position.y0, 0.022, position.height])
    cbar = fig.colorbar(im, cax=cbar_axes)
    cbar.set_label("Track density", fontsize=16, labelpad=7)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    cbar.outline.set_linewidth(0.7)
    ax.set_xlabel(r"$z_{\mathrm{POCA}}$ [mm]", fontsize=18, labelpad=4)
    ax.set_ylabel(r"$t_{\mathrm{POCA}}$ [ns]", fontsize=18, labelpad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(False)

    if data["run_number"] is not None:
        identity = f"Run {data['run_number']}, event {data['event_number']}"
        filename = (
            f"histogram_run_{data['run_number']}_event_{data['event_number']}.pdf"
        )
    else:
        identity = f"Event {data['event_id']}"
        filename = f"histogram_event_{data['event_id']}.pdf"
    producer = label or data["source"]
    header = f"{producer}  |  {identity}" if producer else identity
    handles, labels = ax.get_legend_handles_labels()
    ax.text(
        0,
        1.045 + 0.055 * ((len(handles) + 1) // 2) if handles else 1.025,
        header,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
    )
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0, 1.015),
            ncols=2,
            fontsize=10.5,
            frameon=False,
            borderaxespad=0,
            handlelength=1.4,
            handletextpad=0.45,
            columnspacing=1.2,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        output_file.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.04
    )
    plt.close(fig)
    print(f"Plot saved as {output_file}")
    return output_file
