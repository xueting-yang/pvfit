# Analysis

[Back to README](../README.md)

Select modules with `pvfit analyze --analyses ...` or `analysis.analyses` in
[configs/analysis.yaml](../configs/analysis.yaml). See [usage](usage.md) for commands.

Analysis reads stored true and reconstructed vertices through the
[HDF5 interface](data-format.md#hdf5-layout), independently of the reconstruction
program. Select metrics supported by the fields in your input and keep truth
selection and matching settings consistent when comparing producers.

One accumulator pipeline handles both `workers=1` and multiple processes
(rate accumulators in `pvfit/analysis/rate_accumulators.py`, residual ones in
`residual_accumulators.py`, the registry in `accumulators.py`).
See [parallel analysis](usage.md#parallel-analysis) for execution and memory
settings, and [validation scope](#parallel-analysis-validation) for the
implementation checks.

## Analysis modules

| Analysis key | Accumulator class (`rate_accumulators.py` / `residual_accumulators.py`) | Description |
| --- | --- | --- |
| `efficiency_track` | `EfficiencyTrack` | reconstruction efficiency, track-based matching |
| `efficiency_geo` | `EfficiencyGeo` | reconstruction efficiency, fixed and resolution-scaled `(z, t)` ellipses |
| `fake_rate_track` | `FakeTrack` | fake vertex rate, track-based matching |
| `fake_rate_geo` | `FakeGeo` | fake vertex rate, fixed and resolution-scaled `(z, t)` ellipses |
| `merge_rate` | `MergeTrack` | vertex merge rate |
| `split_rate` | `SplitTrack` | vertex split rate |
| `merge_rate_geo` | `MergeGeo` | merge rate, true vertices inside one rec ellipse (fixed and resolution-scaled) |
| `split_rate_geo` | `SplitGeo` | split rate, one true vertex inside several rec ellipses (fixed and resolution-scaled) |
| `resolution` | `Resolution` | position resolution (std of residuals inside the window) and bias (their mean) vs track multiplicity, each with its uncertainty |
| `track_pull` | `TrackPull` | input track pulls, track vs true vertex |
| `vertex_pull` | `VertexPull` | fitted vertex pulls, rec vs true vertex |
| `plot_histogram` | `plot_histogram.py` | 2D `(z, t)` histogram with vertex overlays |

The two efficiency and six fake/merge/split analyses share the same binning,
counters, uncertainty calculation and output writers. Each analysis defines
which vertices enter its numerator and denominator; see
[matching and rate definitions](#matching-and-rate-definitions).
All rate figures use the shared LHCb2 styling. Legacy count-annotation and
ratio figures are no longer generated. Existing figures in previous output
directories are retained.

### Overall population and plotting bins

Overall **rate** results include every stored true vertex on the true side
and every nonempty reconstructed vertex on the rec side. In particular,
reconstructed vertices with two or three final members still count. Matching
uses the stored final members and the existing criteria below; this does not
add a minimum reconstructed multiplicity cut or refit any vertices.

`plot_min_ntracks`, `plot_max_ntracks` and `plot_bin_width_ntracks` control the
binned tables and figures, not the overall rate or its uncertainty. Counters
accumulate full-population totals and event-level statistics independently of
the bins. The overall rate is the total numerator divided by the total
denominator, not an average of bin rates. The optimizer uses this same
population for its track-based objectives.

The actual bin edges are inclusive: with minimum 4, maximum 200 and width 5,
the last bin is 199–203. With width 1 the range is 4–200. Each overall CSV
records these actual bounds and the underflow/overflow counts, so the
unplotted populations remain visible. Horizontal display limits can further
shorten a figure without changing its saved bins or overall result.

Earlier releases summed only the configured bins for rate overall results.
Re-run analysis on the existing HDF5 to obtain the complete totals; no new
reconstruction or conversion is needed. Residual selections and resolution /
pull summaries are unchanged by this rate-counting change.

### Uncertainties

Rate uncertainties are one-sigma Clopper-Pearson intervals widened by a
*design effect* that accounts for correlated outcomes within an event: a merge
marks two true vertices at once, a split two reconstructed fragments, a busy
event can produce several fakes, so vertices are not independent trials. Each
counter closes every event and accumulates Σs², Σs·n and Σn² of the event's
successes and trials, per bin and overall; the ratio-estimator variance
Σ(s_e − p̂·n_e)² / N² is compared with the binomial p̂(1−p̂)/N and the square root
of their ratio, clamped at 1, multiplies both Clopper-Pearson errors.

The multiplier is reported as `design_effect` in the CSV files. It is computed
from each sample's event counts, not fixed by the metric name. If a bin has no
trials, no successes or all successes, the multiplier is 1. A value near √2 can
occur when correlated pairs are counted; it is not a universal correction.

The `resolution` analysis reports the standard deviation of the residuals
rec − true and their mean (the bias) separately. An RMS about zero would mix
the two. The std uncertainty comes from the fourth central moment without a
Gaussian assumption (see [Error_Of_RMS.md](derivations/Error_Of_RMS.md)) and the bias
uncertainty is `s/√n`. Both values describe the core of the distribution inside
the configured residual window. The two pull analyses attach the same
uncertainties (`mean_error`, `std_error`) to their reported means and widths, in
the plot titles and in their JSON summaries.

### LHCb2 efficiency figure

`efficiency_track` and `efficiency_geo` write
`efficiency_true_binned_<method>_lhcb2.pdf` and `.png` for track, geometric and
resolution-scaled matching. These figures use `mplhep.style.LHCb2`, efficiency
points with asymmetric one-sigma error bars using the
[uncertainty prescription above](#uncertainties), and a gray
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
within that limit, preserving the original binning and counts. Per-bin CSV
files retain the full configured bin range; overall rates include the complete
population, including vertices outside those bins.

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
computed errors and CSV values stay unchanged. An optional
`analysis.rate_plot_ymax` mapping overrides individual limits, for example
`{merge_rate_geo_rec: 0.25}`. Keys have the form
`<fake_rate|merge_rate|split_rate>_<track|geo|reso>_<true|rec>`
(fake rate only has `rec`). An override must still include all central values.

When both `fake_rate_track` and `fake_rate_geo` run, the analysis also writes
`fake_rate_comparison_<base_method>_lhcb2.pdf` and `.png`. These overlay the
track-overlap fake rate and the resolution-scaled geometric fake rate (5 sigma
by default), using distinct hollow markers and asymmetric one-sigma errors.
Both curves use the same reconstructed-vertex bins and denominators, with no
background distribution. The comparison uses `analysis.rate_plot_max_ntracks`
and derives its y range from the central values of both curves. The individual
CSV files retain the complete counts and errors for each definition.

### Merge-distance distributions

Selecting `merge_rate` also collects pairs in the existing single-pass
accumulator pipeline (including parallel batches) and writes `merge_distance_<method>_track_based.pdf`
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
rate, error_lower, error_upper, denominator, numerator, design_effect`; the
matching `<metric>_overall_<true|rec>_binned_<method>.csv` has one row with
`numerator, denominator, rate, error_lower, error_upper, design_effect`, the
matching parameters and `method_name`. `<method>` is `<method_name>_Track_Based`, `_Geo_Based` or
`_Reso_Based` after the matching used.

Overall CSVs also record the population and bin coverage:

| Columns | Meaning |
| --- | --- |
| `population_scope` | `all_stored_true` or `all_nonempty_rec` |
| `bin_min_ntracks`, `bin_max_ntracks` | actual inclusive boundaries of the saved bins |
| `binned_numerator`, `binned_denominator` | totals within those bins |
| `underflow_numerator`, `underflow_denominator` | counts below the first bin |
| `overflow_numerator`, `overflow_denominator` | counts above the last bin |
| `underflow_rate`, `overflow_rate` | metric numerator / denominator within that region; blank for an empty region |
| `underflow_fraction`, `overflow_fraction` | region denominator / overall denominator; blank for an empty overall population |

All rates and fractions use the 0–1 scale. Binned, underflow and overflow
counts sum to the overall counts. The underflow/overflow rates describe the
same metric as that CSV, not a separate matching rule. The standalone
track-count distribution CSV remains a binned table.

`resolution_<coord>_binned_<method>.csv` has one row per bin with
`resolution_<unit>, resolution_error_<unit>, bias_<unit>, bias_error_<unit>,
n_vertices`; `resolution_<coord>_overall_<method>.csv` holds the same four
values over all bins.

## Matching and rate definitions

Truth track membership follows the [event-local keys](data-format.md#truth-association).
New PVFit files count truth tracks in the complete input table before the
reconstruction z/t window. Recall and true-track multiplicity therefore include
tracks excluded from seed-histogram filling by that window. Reconstructed
track counts use the stored final members, not every track originally assigned
to a seed. Legacy HDF5 files retain their stored populations and memberships.
Track-based efficiency matches a reconstructed vertex to the true vertex with
highest recall among those satisfying both `recall >= recall_threshold` and
`purity >= purity_threshold`. Recall is shared tracks / true-vertex tracks;
purity is shared tracks / reconstructed-vertex tracks.

Track-based fake rate counts a reconstructed vertex as fake exactly when no
true vertex shares at least one track with it and satisfies either
`recall >= recall_threshold` or `purity >= purity_threshold`. Every qualifying
pair remains a candidate: one true vertex can support several reconstructed
vertices, and one reconstructed vertex can have several true candidates.
Fake classification uses no best-candidate choice, competition or retries.
Track-based merge rate counts a merge when one reconstructed vertex reaches
the recall threshold for multiple true vertices. Split rate counts a split
when multiple reconstructed vertices reach the purity threshold for one true
vertex.

Geometric analyses evaluate two ellipses in `(z, t)`. Both require
`(dz / sigma_z)^2 + (dt / sigma_t)^2 <= 1`:

- Fixed ellipse: `sigma_z = z_scale * m_r / 2`, `sigma_t = t_scale * m_r / 2`.
- Resolution-scaled ellipse: `sigma_z = nsigma * z_err`, `sigma_t = nsigma * t_err`.

`nsigma` defaults to 5.0 in both the analysis configuration template and
configuration-free analysis. An explicit configuration can override it.

Both geometric efficiency definitions use the same unique-claiming procedure.
Each true vertex ranks its in-ellipse reconstructed candidates by
increasing squared normalized distance, then joins a queue to apply to its best
candidate. A reconstructed vertex provisionally accepts its closest applicant.
If a later applicant is closer, it replaces the current owner. Rejected and
displaced true vertices return to the queue and try their next candidates;
already occupied recs still accept a better applicant. For the resolution-scaled
case, each pair's distance uses that reconstructed vertex's own errors.

Equal distances prefer the smaller rec ID when a true orders its candidates,
and the smaller true ID when a rec compares applicants. These rules implement
the common pair priority `(distance, rec_id, true_id)`. Every candidate pair is
tried at most once, so the queue empties without an iteration or retry limit.
Both sides rank pairs by the same score: the final assignment is equivalent to
sorting all pairs by this priority and greedily accepting pairs whose endpoints
are still free. The queue implementation therefore preserves the distance-first
result; it does not maximize the total number of pairs.

Efficiency counts the true vertices in the final pairs. Claiming is event-local
and happens before track-count binning, so changing the displayed or saved bins
cannot change who wins an efficiency claim.

Geometric fake rate counts a reconstructed vertex as fake exactly when its
ellipse contains no true vertex, separately for the fixed and resolution-scaled
ellipses. It uses all candidate associations, without unique claiming,
best-candidate selection, competition or retries. Geometric merge and split
also use all in-ellipse candidate associations; their definitions remain
unchanged.

For each fake definition, qualifying split copies do not count as fake. A
physical split can still count as fake if it fails that definition's candidate
criterion. A track-defined merge can still count as geometric fake if its
ellipse contains no true vertex. These rates do not form a disjoint partition.

Rerun `pvfit analyze` on an existing HDF5 file to apply these fake definitions;
this analysis change does not require rerunning reconstruction. Previous output
files and slides retain their historical definitions.

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
    ├── fake_rate_comparison_<base_method>_lhcb2.pdf / .png   # when both fake analyses run
    ├── resolution_vs_ntracks_<method>.pdf
    ├── residual_histograms_<coord>_<method>.pdf              # one per coordinate
    ├── resolution_<coord>_binned_<method>.csv
    ├── resolution_<coord>_overall_<method>.csv
    ├── pull_analysis_input_track.pdf / .png
    ├── pull_analysis_input_track_summary.json
    ├── pull_analysis_fitted_vertex.pdf
    ├── pull_analysis_fitted_vertex_summary.json
    └── histogram_run_<run>_event_<event>.pdf / .png          # histogram_event_<id> without identity metadata
```

## Event histogram selection

`plot_histogram` plots one event's stored `(z, t)` histogram with vertex
overlays in `mplhep.style.LHCb2`. It uses a linear coloured track-density map,
a 4:3 data area, small vertex markers and a legend above the axes. Fitted
uncertainty ellipses use `analysis.nsigma`. Run/event identity appears above
the plot and overlay counts appear in the legend. Both PDF and PNG are saved.

To select an event:

- Set both `analysis.plot_run_number` and `analysis.plot_event_number` to
  nonnegative integers. They match the stored `run_number` and
  `source_event_number` attributes. A missing or duplicate identity is an error.
- Leave both values `null` (the default) to plot the smallest numeric HDF5
  group ID. Group `0` need not exist.
- Setting just one of the two values is invalid.

When the event has identity metadata, the output is
`histogram_run_<run>_event_<event>.pdf`. The default selection also supports
older or third-party files without that metadata and writes
`histogram_event_<internal_id>.pdf` instead. Explicit run/event selection
requires the identity attributes. The former `plot_event_id` option is no
longer accepted; use the original run/event pair rather than a group number.

## Input track residual diagnostics

The `track_pull` analysis transports each key-associated track from its POCA to
its matched true vertex's z plane before subtracting x, y, and t. Time transport
uses the forward/backward sign and beta = 1, as in the fitter. The denominators
remain the reported POCA uncertainties; these plots show **scaled residuals**,
not pulls based on a fully propagated covariance.

The analysis keeps finite key-associated tracks with positive variances and
`pull_x**2 + pull_y**2 + pull_t**2 <= max_chi2`.
It evaluates this cut **after** transporting the means to truth z, with the
reported POCA variances and no Tukey weights. The threshold is resolved in this
order:

1. A numeric `analysis.track_pull_chi2` explicitly sets the diagnostic cut.
2. Otherwise (`null`, the default), use the HDF5 root attribute
   `reco_max_chi2_tukey`.
3. Only when that attribute is absent, read `algorithm.max_chi2_tukey` from
   `reconstruction.yaml` beside the HDF5.

The resolved threshold must be positive and finite. A present but invalid HDF5
value is an error, not a reason to try the snapshot. The analysis never falls
back to a built-in value or the current working directory's reconstruction
config. See [older HDF5 files](data-format.md#older-hdf5-files) for unsupported
legacy keys and regeneration requirements.

This truth-based diagnostic selection need not reproduce the tracks and
weights of each reconstruction iteration.

- `pull_analysis_input_track.pdf` / `.png`: tracks passing this fixed selection.
- `pull_analysis_input_track_summary.json`: `n_valid` before the chi-square cut,
  `n_selected`, `n_rejected_by_chi2`, the resolved threshold `max_chi2`, and its
  provenance `max_chi2_source`. The
  `statistics` mapping reports selected-sample means, standard deviations,
  their uncertainties (`mean_error`, `std_error`) and central 68% half-widths.
  Statistics include selected values outside the plotting window; plot titles
  also report the overflow fraction.

The diagnostic uses the truth memberships stored in the HDF5; it does not
select a new truth population. PVFit and the Allen converter build these from
all input tracks according to their [truth selection](data-format.md#truth-association).
The seeding z/t window does not prefilter this diagnostic. Legacy files retain
their older track selection. A width near one does not establish
uncertainty calibration because the analysis uses the reported POCA uncertainties
and selects on the same scaled residuals it plots.

New HDF5 results include `tracks/tx`, `tracks/ty`, and `tracks/backward`. For older
files missing these fields, regenerate results before running this analysis:

```bash
uv run pvfit fit --config configs/reconstruction.yaml
uv run pvfit analyze --input output/baseline/fit_results.h5 \
  --config configs/analysis.yaml --analyses track_pull
```

## Parallel analysis validation

During the accumulator migration, a 500,000-event comparison against the
original serial implementation found exact integer counts and agreement of
45 CSV files and two pull-summary JSON files within `rtol=1e-10`, `atol=1e-12`.
Residual/pull histogram bin edges matched exactly; plotted densities agreed
within float32 roundoff (`rtol=2e-7`, `atol=1e-12`). Some figure styles changed,
so this was not a claim of identical PDF pixels. A separate 50,000-event check
compared the raw residual and pull observations individually.

These checks establish the scope of that implementation migration. They do not
validate physics calibration, guarantee a speedup or bound memory for another
sample. Later intentional changes, including event-level rate uncertainties,
are described in the [changelog](../CHANGELOG.md) and are outside that historical
comparison. Development datasets and detailed reports are local and are not
part of the public repository. For current resource guidance, see
[parallel analysis](usage.md#parallel-analysis).
