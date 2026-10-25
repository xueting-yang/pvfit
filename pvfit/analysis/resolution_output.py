"""Resolution plots and CSVs from finalized statistics and integer counts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COORDS = ("x", "y", "z", "t")


def prepare_plot_data(bins, resolutions, resolution_errors, counts):
    """Prepare data for plotting, filtering out bins with no data."""
    bin_centers = [(start + end) / 2 for start, end in bins]

    # Prepare data for each coordinate
    plot_data = {coord: {} for coord in ["x", "y", "z", "t"]}

    for coord in COORDS:
        x_values, res_values, err_values, n_vertices_list = [], [], [], []

        for idx, bin_tuple in enumerate(bins):
            n_vertices = counts[coord][bin_tuple]
            if n_vertices > 0:
                x_values.append(bin_centers[idx])
                res_values.append(resolutions[coord][bin_tuple])
                err_values.append(resolution_errors[coord][bin_tuple])
                n_vertices_list.append(n_vertices)

        if x_values:
            plot_data[coord] = {
                "x": x_values,
                "resolution": res_values,
                "error": err_values,
                "n_vertices": n_vertices_list,
            }

    # Check if we have any data
    if not any(plot_data[coord] for coord in ["x", "y", "z", "t"]):
        return None

    return plot_data


def plot_resolutions(
    plot_data,
    recall_threshold,
    purity_threshold,
    x_residual_range,
    y_residual_range,
    z_residual_range,
    t_residual_range,
    output_dir,
    method_name,
):
    """Create and save resolution vs track count plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")

    # Create a figure with four subplots (one for each coordinate)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    coords = ["x", "y", "z", "t"]
    colors = ["red", "green", "blue", "purple"]
    residual_ranges = [
        x_residual_range,
        y_residual_range,
        z_residual_range,
        t_residual_range,
    ]
    units = ["μm", "μm", "μm", "ps"]  # micrometers for x,y,z; picoseconds for t
    scale_factors = [1000, 1000, 1000, 1000]  # mm to μm, ns to ps

    for idx, (coord, color, res_range, unit, scale) in enumerate(
        zip(coords, colors, residual_ranges, units, scale_factors, strict=True)
    ):
        ax = axes[idx]

        if coord not in plot_data or not plot_data[coord]:
            ax.text(
                0.5,
                0.5,
                f"No {coord.upper()} resolution data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        data = plot_data[coord]
        x = data["x"]
        resolution = data["resolution"]
        error = data["error"]
        n_vertices = data["n_vertices"]

        # Convert to display units (micrometers or picoseconds)
        resolution_scaled = [r * scale for r in resolution]
        error_scaled = [e * scale for e in error]

        # Plot with error bars
        ax.errorbar(
            x,
            resolution_scaled,
            yerr=error_scaled,
            fmt="o-",
            color=color,
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            label=f"{coord.upper()} Resolution (std of residuals)",
        )

        ax.set_xlabel("Number of tracks per primary vertex", fontsize=12)
        ax.set_ylabel(f"{coord.upper()} Resolution ({unit})", fontsize=12)

        # Format range display based on coordinate
        if coord == "t":
            range_str = f"[-{res_range}, {res_range}] ns"
        else:
            range_str = f"[-{res_range}, {res_range}] mm"

        ax.set_title(
            f"{coord.upper()} Resolution vs Track Count\n"
            f"(std of residuals in {range_str}; bias reported separately)\n"
            r"Error: $\sqrt{Var(s^2)/(4s^2)}$",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=0)

        # Add annotations
        max_res = max(resolution_scaled) if resolution_scaled else 1
        for x_pos, res, n_vtx in zip(x, resolution_scaled, n_vertices, strict=True):
            if n_vtx > 0:
                ax.text(
                    x_pos,
                    res + max_res * 0.05,
                    f"n={n_vtx}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    alpha=0.7,
                )

    plt.suptitle(
        f"Vertex Position Resolution Analysis\nRecall≥{recall_threshold}, Purity≥{purity_threshold}",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    plot_file = output_path / f"resolution_vs_ntracks_{method_slug}.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved resolution plot to: {plot_file}")


def write_resolution_csv(
    bins,
    resolutions,
    resolution_errors,
    biases,
    counts,
    overall_stats,
    recall_threshold,
    purity_threshold,
    output_dir,
    method_name,
):
    """Save resolution results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.lower().replace(" ", "_")

    # Save per-bin results for each coordinate
    for coord in COORDS:
        bin_data = []
        unit = "ns" if coord == "t" else "mm"

        for bin_tuple in bins:
            bin_start, bin_end = bin_tuple
            bin_center = (bin_start + bin_end) / 2
            resolution = resolutions[coord][bin_tuple]
            error = resolution_errors[coord][bin_tuple]
            bias, bias_error = biases[coord][bin_tuple]
            n_vertices = counts[coord][bin_tuple]

            bin_data.append(
                {
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "bin_center": bin_center,
                    f"resolution_{unit}": resolution,
                    f"resolution_error_{unit}": error,
                    f"bias_{unit}": bias,
                    f"bias_error_{unit}": bias_error,
                    "n_vertices": n_vertices,
                }
            )

        df = pd.DataFrame(bin_data)
        csv_path = output_path / f"resolution_{coord}_binned_{method_slug}.csv"
        df.to_csv(csv_path, index=False)

    # Save overall summary for each coordinate
    for coord in COORDS:
        stats = overall_stats[coord]
        unit = "ns" if coord == "t" else "mm"

        summary_df = pd.DataFrame(
            [
                {
                    "coordinate": coord,
                    f"overall_resolution_{unit}": stats["resolution"],
                    f"overall_error_{unit}": stats["error"],
                    f"overall_bias_{unit}": stats["bias"],
                    f"overall_bias_error_{unit}": stats["bias_error"],
                    "total_vertices": stats["n_vertices"],
                    "recall_threshold": recall_threshold,
                    "purity_threshold": purity_threshold,
                    "method_name": method_name,
                }
            ]
        )

        overall_path = output_path / f"resolution_{coord}_overall_{method_slug}.csv"
        summary_df.to_csv(overall_path, index=False)
