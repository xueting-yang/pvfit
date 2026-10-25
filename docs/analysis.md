# Analysis

[Back to README](../README.md)

Select modules with `pvfit analyze --analyses ...` or `analysis.analyses` in
[configs/analysis.yaml](../configs/analysis.yaml). See [usage](usage.md) for commands.

## Analysis modules

| Analysis key | Module | Description |
| --- | --- | --- |
| `efficiency_track` | `analyze_efficiency_track.py` | reconstruction efficiency, track-based matching |
| `efficiency_geo` | `analyze_efficiency_geo.py` | reconstruction efficiency, fixed and resolution-scaled `(z, t)` ellipses |
| `fake_rate_track` | `analyze_fake_rate_track.py` | fake vertex rate, track-based matching |
| `fake_rate_geo` | `analyze_fake_rate_geo.py` | fake vertex rate, fixed and resolution-scaled `(z, t)` ellipses |
| `merge_rate` | `analyze_merge_rate_track.py` | vertex merge rate |
| `split_rate` | `analyze_split_rate_track.py` | vertex split rate |
| `merge_rate_geo` | `analyze_merge_split_geo.py` | merge rate, true vertices inside one rec ellipse (fixed and resolution-scaled) |
| `split_rate_geo` | `analyze_merge_split_geo.py` | split rate, one true vertex inside several rec ellipses (fixed and resolution-scaled) |
| `resolution` | `analyze_resolution.py` | position resolution (std of residuals inside the window) and bias (their mean) vs track multiplicity, each with its uncertainty |
| `track_pull` | `analyze_track_pull.py` | input track pulls, track vs true vertex |
| `vertex_pull` | `analyze_vertex_pull.py` | fitted vertex pulls, rec vs true vertex |
| `plot_histogram` | `plot_histogram.py` | 2D `(z, t)` histogram with vertex overlays |

The eight rate analyses only decide which vertices count as successes and return
`RateResult` objects; `binned_rate.py` holds the shared bins, counters,
Clopper-Pearson errors and the writers `run_analysis` applies to each result.
All rate figures use the shared LHCb2 renderer. Legacy count-annotation,
comparison and ratio figures are no longer generated. Existing figures in
previous output directories are retained.

### Uncertainties

Rate uncertainties are one-sigma Clopper-Pearson intervals. The `resolution`
analysis reports the standard deviation of the residuals rec − true and their
mean (the bias) separately rather than one RMS about zero, which would mix the
two: the std uncertainty comes from the fourth central moment without a Gaussian
assumption (see [Error_Of_RMS.md](derivations/Error_Of_RMS.md)) and the bias
uncertainty is `s/√n`. Both values describe the core of the distribution inside
the configured residual window. The two pull analyses attach the same
uncertainties (`mean_error`, `std_error`) to their reported means and widths, in
the plot titles and in their JSON summaries.

### LHCb2 efficiency figure

`efficiency_track` and `efficiency_geo` write
`efficiency_true_binned_<method>_lhcb2.pdf` and `.png` for track, geometric and
resolution-scaled matching. These figures use `mplhep.style.LHCb2`, efficiency
points with asymmetric one-sigma Clopper-Pearson error bars, and a gray
reconstructed-vertex track-count distribution behind the points. The analysis
counts all nonempty reconstructed vertices by their own track multiplicity,
including unmatched vertices, independently of efficiency matching. Its peak
is scaled to 0.65 for display. The efficiency points remain binned by true
vertex multiplicity, with their original true-vertex denominator and errors;
the background scaling does not affect efficiencies or uncertainties. No bin-count
annotations are drawn. Compact paper styling uses thin frames and
ticks, a legend above the axes, and 3-point hollow markers with error bars drawn
above them so the markers do not hide the uncertainties.
`analysis.efficiency_plot_max_ntracks` sets
its horizontal display limit (default 150). It shows complete populated bins
within that limit, preserving the original binning and counts. CSV files and
overall rates retain their full configured range.

### LHCb2 fake, merge and split rate figures

Each fake, merge and split rate result gets `_lhcb2.pdf` and `.png`
figures with the same compact styling. Reconstructed-vertex distributions are
omitted. True-binned merge/split figures keep a gray true-vertex spectrum,
scaled to 60% of the displayed y maximum. No bin counts are annotated.

`analysis.rate_plot_max_ntracks` controls the horizontal display limit
(default 150). The automatic y maximum uses 1.25 times the largest displayed
central value, a minimum range of 0.001, and rounding upward to a readable
limit, capped at 1.05. It does not expand to contain every confidence interval.
Upward triangles mark intervals extending above the axis; the underlying
Clopper-Pearson errors and CSV values stay unchanged. An optional
`analysis.rate_plot_ymax` mapping overrides individual limits, for example
`{merge_rate_geo_rec: 0.25}`. Keys have the form
`<fake_rate|merge_rate|split_rate>_<track|geo|reso>_<true|rec>`
(fake rate only has `rec`). An override must still include all central values.

### Merge-distance distributions

Selecting `merge_rate` also writes `merge_distance_<method>_track_based.pdf`
and `.png`: the signed true-pair Δz projection (mm), joint Δz–Δt counts, and
horizontal Δt projection (ns). It reuses the track-based merge definition and
`recall_threshold`; it does not change the rate or matching definitions. These
are separations **between true vertices sharing a reconstructed merge**, not
reconstructed-minus-true residuals or distances between arbitrary neighbours.

Each unordered true pair counts once per event, including all k(k−1)/2 pairs
in a merge of k true vertices; repeated merges of the same pair are deduplicated.
The sign is higher true-row index minus lower index, without sorting by z or
mirroring pairs. This diagnostic has no track-multiplicity bin cut. The `.csv`
keeps all pairs and their file-local HDF5 event groups/true indices. The `.json`
records input path, counting/sign convention, thresholds, nonfinite exclusions,
bin edges and visible counts. Figure windows are ±3 mm and ±0.1 ns; projection
counts include finite pairs outside the *other* coordinate's window, while the
joint panel includes only pairs inside both. Tails remain in the CSV. Empty
samples produce labelled empty plots. This plot uses track matching only;
`merge_rate_geo` retains its existing geometric/resolution-based rate outputs.

### CSV files

Whenever any rate analysis runs, it also writes one
`vertex_track_count_distribution_<base_method>.csv` (for example,
`vertex_track_count_distribution_pvfit.csv`), shared by all matching methods.
Its columns are `bin_start, bin_end, bin_center, true_count,
reconstructed_count`. Both populations come from the same input events and
use each vertex's own stored track multiplicity. Counts include unmatched
vertices and every configured bin, including empty bins; they have no display
scaling or matching cuts. The distribution retains the full analysis binning
regardless of either figure's horizontal display limit. Efficiency CSVs keep
their separate numerators, denominators and errors. Together these files allow
replotting both the efficiency and its reconstructed-vertex spectrum without
reading the HDF5 input again.

Every rate is written the same way. `<metric>_<true|rec>_binned_<method>.csv`
has one row per track-multiplicity bin with `bin_start, bin_end, bin_center,
rate, error_lower, error_upper, denominator, numerator`; the matching
`<metric>_overall_<true|rec>_binned_<method>.csv` has one row with `numerator,
denominator, rate, error_lower, error_upper`, the matching parameters and
`method_name`. `<method>` is `<method_name>_Track_Based`, `_Geo_Based` or
`_Reso_Based` after the matching used.

`resolution_<coord>_binned_<method>.csv` has one row per bin with
`resolution_<unit>, resolution_error_<unit>, bias_<unit>, bias_error_<unit>,
n_vertices`; `resolution_<coord>_overall_<method>.csv` holds the same four
values over all bins.

## Matching and rate definitions

Truth track membership follows the [event-local keys](data-format.md#truth-association).
Track-based efficiency matches a reconstructed vertex to the true vertex with
highest recall among those satisfying both `recall >= recall_threshold` and
`purity >= purity_threshold`. Recall is shared tracks / true-vertex tracks;
purity is shared tracks / reconstructed-vertex tracks.

Track-based fake rate uses a different claiming rule: each true vertex claims
the reconstructed vertex with highest recall among those sharing tracks and
satisfying either threshold. The higher-recall claim wins a conflict; unclaimed
reconstructed vertices count as fake. Track-based merge rate counts a merge
when one reconstructed vertex reaches the recall threshold for multiple true
vertices. Split rate counts a split when multiple reconstructed vertices reach
the purity threshold for one true vertex.

Geometric analyses evaluate two ellipses in `(z, t)`. Both require
`(dz / sigma_z)^2 + (dt / sigma_t)^2 <= 1`:

- Fixed ellipse: `sigma_z = z_scale * m_r / 2`, `sigma_t = t_scale * m_r / 2`.
- Resolution-scaled ellipse: `sigma_z = nsigma * z_err`, `sigma_t = nsigma * t_err`.

Efficiency uses true vertices as its denominator; fake rate uses reconstructed
vertices. Merge and split analyses each report both true-vertex and
reconstructed-vertex fractions. The `true` or `rec` in each filename identifies
that denominator and the vertex track count used for binning. Track counts use
the stored membership indices; reconstructed vertices with no tracks do not
enter these rate calculations. See [truth selection](data-format.md#truth-association)
for the minimum multiplicity and acceptance applied to true vertices.

## Output

The shipped configuration produces the following files. Selecting fewer modules
produces only their outputs. Standalone `analyze --output` sets the plot directory
directly; the tree below shows a combined `run`.

```
output/<experiment>/
├── fit_results.h5
└── analysis/
    ├── analysis.done
    ├── vertex_track_count_distribution_<base_method>.csv
    ├── <metric>_<true|rec>_binned_<method>_lhcb2.pdf / .png   # one per rate result
    ├── <metric>_<true|rec>_binned_<method>.csv
    ├── <metric>_overall_<true|rec>_binned_<method>.csv
    ├── merge_distance_<base_method>_track_based.pdf / .png / .csv / .json
    ├── resolution_*.pdf
    ├── pull_analysis_input_track.pdf / .png
    ├── pull_analysis_input_track_summary.json
    ├── pull_analysis_fitted_vertex.pdf
    ├── pull_analysis_fitted_vertex_summary.json
    └── histogram_event_*.pdf
```

## Input track residual diagnostics

The `track_pull` analysis transports each key-associated track from its POCA to
its matched true vertex's z plane before subtracting x, y, and t. Time transport
uses the forward/backward sign and beta = 1, as in the fitter. The denominators
remain the reported POCA uncertainties; these plots show **scaled residuals**,
not pulls based on a fully propagated covariance.

The analysis keeps finite key-associated tracks with positive variances and
`pull_x**2 + pull_y**2 + pull_t**2 <= max_chi2`.
It evaluates this cut **after** transporting the means to truth z, with the
reported POCA variances and no Tukey weights. By default,
`analysis.track_pull_chi2: null` follows reconstruction's `algorithm.max_chi2`:
new HDF5 files record it in the root attribute `reconstruction_max_chi2`.
For older files, the analysis reads `algorithm.max_chi2` from the original
`reconstruction.yaml` snapshot beside the HDF5. It never silently falls back to
10 or reads the current working directory's reconstruction config.
A numeric `analysis.track_pull_chi2` explicitly overrides this behavior for a
diagnostic comparison. Missing or invalid reconstruction metadata requires an
original snapshot or an explicit override. This truth-based diagnostic selection
need not reproduce the tracks and weights of each reconstruction iteration.

- `pull_analysis_input_track.pdf` / `.png`: tracks passing this fixed selection.
- `pull_analysis_input_track_summary.json`: `n_valid` before the chi-square cut,
  `n_selected`, `n_rejected_by_chi2`, the resolved threshold `max_chi2`, and its
  provenance `max_chi2_source`. The
  `statistics` mapping reports selected-sample means, standard deviations,
  their uncertainties (`mean_error`, `std_error`) and central 68% half-widths.
  Statistics include selected values outside the plotting window; plot titles
  also report the overflow fraction.

The input still has the reconstruction's z/t acceptance and minimum truth-vertex
multiplicity requirements. A width near one does not establish uncertainty
calibration because the analysis uses the reported POCA uncertainties and
selects on the same scaled residuals it plots.

New HDF5 results include `tracks/tx`, `tracks/ty`, and `tracks/backward`. For older
files missing these fields, regenerate results before running this analysis:

```bash
uv run pvfit fit --config configs/reconstruction.yaml
uv run pvfit analyze --input output/baseline/fit_results.h5 \
  --config configs/analysis.yaml --analyses track_pull
```

## Validation observations

A development check of 10,000 minimum-bias events at luminosity
`1.0e34`, beam origin `(0, 0)` and nominal 50 ps timing found the following
with the existing default matching and reconstruction settings. These are
sample-specific observations, not calibration corrections or guarantees for
other inputs.

- For 206,939 track-matched vertices with valid fit errors, the full-sample
  z-pull standard deviation was about 0.745 and the mean reconstructed-minus-true
  time residual was +1.52 ps. The summary includes residuals outside plot windows.
- In the 4–8 selected-track bin, track-based efficiency was 11,100/29,603
  (37.5%); geometric efficiency was 13,027/29,603 (44.0%). The two definitions
  answer different matching questions and must not be labeled interchangeably.

These observations motivate further checks of covariance calibration, time
bias and track assignment. They do not by themselves establish a universal
error rescaling or the cause of the low-multiplicity inefficiency. This release
keeps geo, track and reso rate definitions and optimization objectives unchanged.
