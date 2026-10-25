# Data formats

[Back to README](../README.md)

PVFit separates reconstruction inputs from the analysis interface:

| Workflow | Input contract |
| --- | --- |
| `pvfit fit` | ROOT track and truth trees described below |
| `pvfit analyze` | event groups in the [HDF5 layout](#hdf5-layout), from PVFit or another compatible producer |
| `pvfit convert-allen` | monitoring trees from the supported Allen sequence; see [the adapter](#importing-allen-output) |

An external reconstruction program can write the HDF5 interface directly or use
its own converter. Analysis uses the stored populations and memberships; it does
not rerun the producer's reconstruction or harmonize different truth selections.

## ROOT input

The Python fitter expects the two trees and branches listed below. The
[toy generator](toy.md) and supported Allen monitoring dumps provide this
schema; other ROOT layouts need conversion. Set `paths.input_root` or pass
`--input` to `fit` or `run`; a glob reads all matching files together. The
optimizer has a separate [input selection policy](optimization.md#input-selection).

### Event identity

Allen monitoring dumps preserve the run and event numbers from MDF ODIN.
Event numbers can repeat across runs, so an event is identified by the pair
`(run_number, event_number)`, and both trees **must** carry a `run_number`
branch; a dump without it is rejected. PVFit numbers the events densely in
`(run_number, event_number)` order across all input files. That internal ID
identifies HDF5 groups and reconstruction's `workflow.event_number` selection.
Each HDF5 event group records the original `run_number`, `source_event_number`
and `source_file` in its attributes. The same `(run_number, event_number)`
appearing in two input files is an error.

For plotting an event by its original run/event pair, see
[event histogram selection](analysis.md#event-histogram-selection).

### ROOT trees

Both trees are required. Positions use mm, times use ns and momenta use MeV.

**Track tree** (default `velo_kalman_filter/monitor_tree`), one row per track:

| Branch | Meaning |
| --- | --- |
| `event_number` | event id, used to group rows |
| `run_number` | run id — **required**; events are keyed by (run, event) |
| `beamPOCA_x`, `beamPOCA_y`, `beamPOCA_z`, `beamPOCA_t` | point of closest approach to the beamline |
| `beamPOCA_tx`, `beamPOCA_ty` | track slopes at the POCA |
| `backward` | track direction flag used for time transport; required |
| `c00`, `c11`, `c55` | diagonal covariance entries for `x`, `y`, `t` |
| `chi2` | chi2 of the VELO Kalman track fit, carried through to `tracks/chi2`; the beamline chi2 cut does not read it but recomputes `x²/c00 + y²/c11` from the POCA state |
| `p`, `pt`, `fromPV` | kinematics / provenance, carried through to the output |
| `mc_key` | MC particle key — **required** for truth association |
| `ovtx_x`, `ovtx_y`, `ovtx_z`, `ovtx_t` | MC origin vertex of the track |

**Truth vertex tree** (default `mc_pv_dumper/MCPV_effective_tree`), one row per
true PV:

| Branch | Meaning |
| --- | --- |
| `event_number` | event id |
| `run_number` | run id — **required** |
| `x`, `y`, `z`, `time` | true vertex position |
| `effective_pv_key` | unique key of the vertex within the event |
| `track_keys` | variable-length array of the `mc_key`s belonging to this vertex |
| `n_valid_tracks` | track count from the dumper, stored as `expected_ntracks` |

### Truth association

A reconstructed track row belongs to a true vertex when its `mc_key` appears in
that vertex's `track_keys`, **within the same event**. Association is purely by
key; track and vertex coordinates never decide membership.

Selection and validation:

- `pvfit fit` stores a true vertex when at least **4** rows in the complete
  input track table belong to it, **before** seeding POCA `z`/`t` cuts.
  `convert-allen` uses the same default, configurable with
  [`--min-truth-tracks`](usage.md#evaluating-allen-output). Both writers record
  the chosen minimum as `truth_min_tracks`.
- These are available input tracks, not all generated particles.
  `n_valid_tracks` is recorded as `expected_ntracks`; it does not replace the
  count obtained by intersecting the input rows with the truth keys.
  Analysis uses the stored true vertices and applies its configured multiplicity
  bins; it does not independently enforce a universal four-track minimum.
  Keep the producer's truth selection consistent when comparing results.
- The configured POCA `z`/`t` window limits seed-histogram filling only.
  Track-to-vertex assignment considers the complete input table, including
  window-excluded tracks, and selects members by track-vertex chi-square.
  Fitting and merging use the same full row order.
- Missing `mc_key` / `effective_pv_key` / `track_keys` / `n_valid_tracks` raises
  an error. There is no silent fallback.
- A truth particle owned by two effective vertices is an error, not a warning.
- Truth association supplies the truth overlays and performance evaluation.
  Reconstruction uses input track states independently of truth membership.

## HDF5 layout

The tree below describes output from the current PVFit writer. Other producers
need only the fields used by their selected analyses, with consistent membership
indices. Positions use mm and times use ns; uncertainties use the same units
as their coordinates, and variances use squared units. Required fields depend
on the selected analyses:

- Event group names are numeric internal IDs. Rate analyses read `z`, `t` and
  `track_indices` in both `true_vertices/` and `rec_vertices/`.
- Geometric rate modules produce both fixed and resolution-scaled results, so
  they also require reconstructed `z_err` and `t_err`.
- Resolution and vertex-pull analyses use the coordinate fields; vertex pulls
  also need their reconstructed errors. Input-track diagnostics need track
  states, variances, direction and truth memberships; see
  [input-track diagnostics](analysis.md#input-track-residual-diagnostics).
- `plot_histogram` needs the histogram and its axis attributes. Vertex and seed
  overlays are optional; see [event histogram selection](analysis.md#event-histogram-selection).

Each `track_indices` value refers to a row of the same event's common track
table, not an MC key. Track-based rates can use these shared row identities
without loading track states. Set `analysis.method_name` to label an external
producer's rate and resolution outputs; `source` labels the event histogram.

```
fit_results.h5
├── @n_events
├── @source                        "PVFit" (producer label for the event histogram)
├── @truth_track_selection         "all_input" (before seeding z/t cuts)
├── @assignment_track_selection    "all_input" (window applies only to seeding)
├── @schema_version                1 (new writers; legacy files may omit it)
├── @pvfit_version, @source_sha256  imported package identity
├── @truth_min_tracks              producer's minimum associated rows per true vertex (default 4)
├── @reco_<name>                   one attribute per reconstruction parameter, see below
├── @input_<name>                  parameters of the track input, when known (see below)
├── @track_input_configuration_path, @track_input_configuration_sha256
│                                  the file paths.track_input_configuration named, when set
├── @provenance_manifest           CLI fit record path, when supplied
└── <event_id>/
    ├── @n_tracks, @n_true_vertices, @n_rec_vertices, @n_seedings, @truth_matching
    ├── @n_seeding_tracks, @truth_track_selection, @assignment_track_selection
    ├── @run_number, @source_event_number, @source_file   (the dump's own identity)
    │
    ├── histogram [n_bins_z, n_bins_t] float64
    │   @z_min, @z_max, @t_min, @t_max, @n_bins_z, @n_bins_t
    │
    ├── seeds/
    │   └── z, t                   [n_seedings] float32  (all seeds, incl. dropped/merged)
    │
    ├── true_vertices/
    │   ├── vertex_id              [n_true] int32
    │   ├── effective_pv_key       [n_true] int64
    │   ├── x, y, z, t             [n_true] float32
    │   ├── expected_ntracks       [n_true] int32
    │   └── track_indices          [n_true] vlen<int32>
    │
    ├── rec_vertices/
    │   ├── x, y, z, t                     [n_rec] float32
    │   ├── x_err, y_err, z_err, t_err     [n_rec] float32
    │   ├── x_seed, y_seed, z_seed, t_seed [n_rec] float32  (pvfit fit only)
    │   └── track_indices                  [n_rec] vlen<int32>
    │
    └── tracks/
        ├── track_id               [n_tracks] int32
        ├── selected_for_seeding  [n_tracks] bool (initial POCA z/t window only)
        ├── x, y, z, t             [n_tracks] float32
        ├── tx, ty                 [n_tracks] float32
        ├── backward               [n_tracks] bool
        ├── cov_xx, cov_yy, cov_tt [n_tracks] float32
        ├── chi2, p, pt            [n_tracks] float32
        ├── fromPV                 [n_tracks] bool
        ├── mc_key                 [n_tracks] int64
        └── ovtx_x, ovtx_y, ovtx_z, ovtx_t [n_tracks] float32
```

### Reconstruction parameters

Current `pvfit fit` results record reconstruction parameters as explicit root
attributes `reco_<name>`, one per parameter. `pvfit convert-allen` writes the same
names when the original Allen configuration is available
(`pvfit/reconstruction_parameters.py` is the shared list), so the two producers
can be compared attribute by attribute. Legacy and external files may omit these
attributes. Conversion with `--allow-missing-configuration` records only the
known binning; it does not invent the missing fit parameters.

| Attribute | Type | Allen source |
| --- | --- | --- |
| `reco_z_min`, `reco_z_max`, `reco_n_bins_z` | float, float, int | `pv_beamline_histo.zmin/zmax/Nbins_z` (also `pv_beamline_peak`) |
| `reco_t_min`, `reco_t_max`, `reco_n_bins_t` | float, float, int | `pv_beamline_histo.tmin/tmax/Nbins_t` (also `pv_beamline_peak`) |
| `reco_smear_window_nbins` | int | `pv_beamline_histo.smear_window_nbins` |
| `reco_max_track_beamline_chi2_xy` | float | `pv_beamline_histo.maxTrackBLChi2XY` |
| `reco_max_track_z0_err` | float (mm) | `pv_beamline_histo.pp_maxTrackZ0Err` |
| `reco_min_tracks_in_seed` | float | `pv_beamline_peak.pp_minTracks` (unscaled; both sides scale it with the bin area) |
| `reco_max_track_seed_chi2` | float | `pv_beamline_assign_tracks.maxTrackSeedChi2XYT` |
| `reco_min_tracks_per_seed` | int | `pv_beamline_assign_tracks.pp_min_tracks_per_seed` |
| `reco_max_chi2_tukey` | float | `pv_beamline_fitter.maxChi2Tukey` (also `pv_beamline_merge_vertices`) |
| `reco_max_fit_iterations` | int | `pv_beamline_fitter.maxFitIter` (also merge) |
| `reco_max_delta_z_converged` | float (mm) | `pv_beamline_fitter.maxDeltaZConverged` (also merge) |
| `reco_max_vertex_rho2` | float (mm²) | `pv_beamline_fitter.maxVertexRho2` |
| `reco_min_chi2_dist_merge` | float | `pv_beamline_merge_vertices.minChi2Dist` |
| `input_hit_time_resolution_ns` | float (ns) | `velo_kalman_filter.hit_time_resolution_ns`; a parameter of the track input, not of the vertex reconstruction, recorded when known |

`pvfit fit` takes the values from its `algorithm` section, one key per
attribute. The beamline xy-chi2, z0-error and vertex-rho2 cuts accept infinite
thresholds to disable them; a zero merge distance disables merging. Assignment
and Tukey thresholds must remain finite.

`rec_vertices/track_indices` lists a vertex's members the way Allen's
`pv_beamline_merge_vertices` writes them: of the tracks assigned to the seed
(chi2 to the seed below `max_track_seed_chi2`) and absorbed by merging, those
whose chi2 at the fitted position is below `max_chi2_tukey`, i.e. with nonzero
weight in the final fit. PVFit applies the same rule, so a track assigned at
the seed but rejected by the fit is not a member, and track-based metrics of
the two producers compare like with like. Seeds whose fit is singular or whose
fitted position is off the beamline (`max_vertex_rho2`) produce no vertex.

Parameters of the track input are not PVFit's to set: it records them only when
`paths.track_input_configuration` names the
configuration of the program that produced the tracks (an Allen configuration
JSON, read by `pvfit/allen.py`; relative paths resolve against each input
file's directory), together with that file's path and sha256. The `track_pull`
analysis reads its default cut from `reco_max_chi2_tukey`. PVFit's
`is_adaptive_tukey` debugging switch is not recorded; Allen always recomputes
the chi2. The Allen property names live only in `pvfit/allen.py`; the core
parameter list in `pvfit/reconstruction_parameters.py` knows no producer.

`track_indices` are positional indices into the `tracks/` arrays of the same
event, so truth and reconstruction can be compared on identical row ordering.
The track table contains every input row. `selected_for_seeding` records only
the initial POCA z/t window, before the histogram's beamline and error cuts.
It does not gate assignment or fitting: a reconstructed member can have a
false seeding flag. Truth, assignment, fitting, merging and storage use the
same full row order, so member indices need no remapping.

`assignment_track_selection="all_input"` identifies this reconstruction
convention. Earlier full-truth files may contain
`selected_for_reconstruction`/`n_selected_tracks`; those record the older
window selection that also gated assignment. They remain readable, but
re-analysis cannot recover tracks that never entered the old fit. New
reconstructed positions and memberships require rerunning reconstruction.

Older PVFit files stored only window-selected tracks and built truth from them.
They remain readable, but their denominators retain that older meaning; running
analysis again cannot restore omitted tracks. Rebuilding truth requires the
original ROOT input. The `truth_track_selection="all_input"` attribute identifies
the new convention; do not assume it for unmarked files.

The `plot_histogram` analysis needs only `histogram` and its attributes; it draws
`seeds/`, `true_vertices/` and `rec_vertices/` (and `z_err`/`t_err` ellipses) when
present, so other producers can write just the parts they have. Files without
`seeds/` fall back to `rec_vertices/{z,t}_seed`.

Schema version 1 adds provenance attributes without changing the event dataset
layout. The full-input truth convention adds selection metadata and a track mask
without changing the existing dataset types. Existing analyses continue to
accept compatible legacy files without a schema attribute; individual diagnostics still require their documented fields.
See [run provenance](usage.md#run-provenance) for effective configuration snapshots.

## Older HDF5 files

A missing `schema_version` or `reco_*` attribute does **not** make an entire
file incompatible. Compatibility depends on the selected analyses and their
required datasets. Old files retain the track population, memberships and
reconstruction decisions with which they were written.

| Situation | What to do |
| --- | --- |
| Compatible vertex datasets but no schema or reconstruction metadata | Rate analyses can still run; supply the coordinates, memberships and, for resolution-scaled matching, fit errors required by the chosen analyses. |
| `track_pull` has its required track fields but no `reco_max_chi2_tukey` | Supply an adjacent `reconstruction.yaml` containing the original `algorithm.max_chi2_tukey`, or explicitly set `analysis.track_pull_chi2`. |
| Track directions, variances or truth memberships needed by a diagnostic are absent | Regenerate HDF5 from the original inputs, through reconstruction or Allen conversion as appropriate. A metadata edit cannot restore missing data. |
| Tracks were discarded by the old z/t window, or stored memberships still represent seed assignment | Reprocess the original inputs to obtain current truth, fitted positions and final-Tukey memberships. Re-analysis cannot apply a new reconstruction convention to old fitted results. |
| Only the analysis definitions or plotting settings changed | Rerun `pvfit analyze` on the compatible HDF5; reconstruction need not be repeated. |

The [input-track diagnostic](analysis.md#input-track-residual-diagnostics)
defines threshold precedence and validation. The old `reconstruction_max_chi2`
and reconstruction JSON attributes, and the old YAML key `algorithm.max_chi2`,
are not threshold fallbacks. Translate historical settings using the original
run configuration; current defaults are not evidence of how an old result was
produced.

For Allen, conversion preserves the reconstruction stored in the ROOT file.
Rerun Allen itself if the desired memberships or fitted results are absent from
that dump; converting again cannot change Allen's original reconstruction.

## Importing Allen output

`pvfit convert-allen` maps the trees written by Allen's
`upgrade2_tv_cheated_tracking` sequence onto the HDF5 layout above:

| HDF5 | Allen tree | Notes |
| --- | --- | --- |
| `tracks/` | `velo_kalman_filter/monitor_tree` | every row of the event, in dump order; `x,y,z,t` from `beamPOCA_*`, `cov_*` from `c00,c11,c55` |
| `tracks/selected_for_seeding` | histogram window | POCA `z`/`t` inside the seeding histogram range only |
| `true_vertices/` | `mc_pv_dumper/MCPV_effective_tree` | membership by `mc_key` on the full track table; minimum defaults to 4, overridden by `--min-truth-tracks` |
| `rec_vertices/` | `pv_beamline_merge_vertices/merge_vtx_tree` | errors are `sqrt(cov00..cov33)`; `track_indices` are the rows whose `mc_key` is in the vertex's `track_mc_keys`; extra datasets `n_tracks_weighted` (Allen's Tukey weight sum) and `chi2`; no per-vertex `*_seed` datasets, since Allen's merge tree keeps no seed per vertex (the seeds are in `seeds/`) |
| `seeds/` | `pv_beamline_peak/peak_tree` | all peaks of the event |
| `histogram` | `pv_beamline_histo/histo_tree` | sparse bins rebuilt into `[n_bins_z, n_bins_t]`; edges and bin counts from the Allen configuration, checked against the dumped bin centres. Only with `--allow-missing-configuration` do the counts come from `--n-bins-z/--n-bins-t`, with width and origin read back from the dump |

Events are identified by `(run_number, event_number)` across all inputs and
every tree must carry both branches. File attributes: `source = "Allen"`,
`assignment_track_selection = "allen_mc_key"` (memberships come from Allen's
member keys, so a key shared by several rows selects all of them),
`converter = "pvfit convert-allen"`, `converter_trees` (the tree names read),
`truth_min_tracks` and the `reco_*` record below.

### Reconstruction parameters from the Allen configuration

Supply the effective Allen sequence configuration used to produce the ROOT
file as `allen_configuration.json` next to `histograms.root`, or select it with
`--allen-config`. Its format is
`algorithm instance -> property -> value` plus a `sequence` entry. The converter
reads `root_path.with_name("allen_configuration.json")` for every input (or one
`--allen-config` file for all of them) and fills the `reco_*` attributes of the
table above from it: each parameter from its Allen property, cleaned of float32
storage noise. Allen repeats the binning in `pv_beamline_peak` and the fit
settings in `pv_beamline_merge_vertices`; the copies must agree, the stored bin
widths `dz`/`dt` must equal `(max - min) / n_bins`, and the configured binning
must reproduce the bin centres of the dumped histogram, or the conversion is
refused. `allen_configuration_path` and `allen_configuration_sha256` name the
file the values came from. Properties that are not reconstruction parameters
(`block_dim`, `enable_monitor`, the SMOG2 region) are not recorded.

Without a configuration the conversion stops. With
`--allow-missing-configuration` (and `--n-bins-z`/`--n-bins-t`) it records only
`reco_z_min` … `reco_n_bins_t`, read back from the dump, and reports on the
console that the other parameters are unknown; it never substitutes defaults.
Inputs with differing configurations, or a mix of inputs with and without one,
are refused: convert them separately so each HDF5 file describes one
reconstruction.
