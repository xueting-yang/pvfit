# Usage and configuration

[Back to README](../README.md)

Run the examples from the project root. For your first run, use the
[toy quick start](../README.md#quick-start). The reconstruction YAML
expects `input/toy_sample.root`; set `paths.input_root` or pass `--input` to
`fit` or `run` for your own sample.

Alternatively, install the checkout with `python -m pip install .` in a virtual
environment and invoke `pvfit` directly. `python -m pvfit` uses the same CLI.
Wheels include the same default YAML templates. With no explicit config option,
`fit`, `run` and `optimize` prefer a local `configs/<stage>.yaml`, then use the
shipped template. An explicitly named missing file remains an error. Run
`pvfit init-config` to copy all five templates into `configs/` for editing;
`--output DIRECTORY` chooses another directory, and existing files are never
overwritten. The wheel supports the CLI; use the source archive or checkout for
Snakemake, notebooks and documentation.

## Command-line interface

```bash
# reconstruct, analyze an existing result, or run both stages
uv run pvfit fit --config configs/reconstruction.yaml
uv run pvfit analyze --input output/baseline/fit_results.h5 \
  --config configs/analysis.yaml --output output/reanalysis
uv run pvfit run --config configs/reconstruction.yaml \
  --analysis-config configs/analysis.yaml

# analyze compatible HDF5 output without any reconstruction configuration
uv run pvfit analyze --input /path/to/results.h5 \
  --output output/external --analyses efficiency_track resolution

# one event, one worker
uv run pvfit fit --config configs/reconstruction.yaml --event-number 1 --max-workers 1

uv run pvfit analyze --help
uv run pvfit --version
```

`fit` reads ROOT and writes HDF5. `analyze` reads an existing HDF5 file without
loading or running the fitter. `run` executes both stages. External results must
follow the [HDF5 layout](data-format.md#hdf5-layout); this command does not
convert arbitrary ROOT or HDF5 formats automatically.

Output paths depend on the command:

| Command | Output option | Destination |
| --- | --- | --- |
| `fit`, `run` | `--output` / `--output-dir` | reconstruction directory; `run` puts analysis in its `analysis/` subdirectory |
| `analyze` | `--output` / `--output-dir` | directory for plots and tables |
| `convert-allen` | `--output` | HDF5 **file**, e.g. `output/allen/results.h5` |
| `toy` | `--output` | ROOT **file**, e.g. `input/toy_sample.root` |
| `init-config` | `--output` | directory for editable YAML templates |

Standalone analysis accepts `--input` or `paths.input_hdf5` in its own YAML.
Its optional `paths.output_dir` sets the plot destination. It does not derive
input paths from a reconstruction config. Relative data/output paths and CLI
config paths are relative to the working directory.

Common short aliases are `-c` for `--config-file`, `-i` for `--input`, `-e` for
`--event-number` and `-a` for `--analyses`, where the command supports them.
Worker options distinguish reconstruction from analysis:

| Command | Option | Controls |
| --- | --- | --- |
| `fit`, `run` | `-w`, `--max-workers` | reconstruction workers |
| `analyze` | `-w`, `--workers`, `--analysis-workers` | analysis workers |
| `run` | `--analysis-workers` | analysis workers; separate from `-w` |

Standalone analysis needs no config unless overriding its defaults. See
[`configs/analysis.yaml`](../configs/analysis.yaml). The `analysis:` section holds
matching thresholds, selected analyses and `method_name` for output labels;
`--analyses` overrides that selection. `run` reads the reconstruction YAML via
`--config` and the analysis YAML via `--analysis-config`, checking both before
fitting. The defaults are `configs/reconstruction.yaml` and
`configs/analysis.yaml`, respectively.

For `fit`, `analyze` and `run`, malformed arguments or configuration return
exit code 2; execution failures return 1 and a concise error; successful runs
return 0. Use `--debug` after the subcommand to retain the full traceback.
See [troubleshooting](#troubleshooting) for common input and configuration errors.

Reconstruction and analysis also remain independently callable Python modules.

### Reconstruction workers and memory

`fit --max-workers N` and `workflow.max_workers` control reconstruction
processes. The shipped value of 30 is a configurable example, not a recommended
worker count for every machine or sample.

The fitter reads the input ROOT tables into memory, prepares event tasks and
retains their results until HDF5 writing. It does not stream the whole sample
with bounded memory. More workers add concurrent event processing and data
transfer costs; reducing workers does not remove the initial full-input load.

Start with a small input file and a few workers, measure the whole process tree's
memory, and leave room for larger events before increasing concurrency. Selecting
one `--event-number` does not avoid reading the full input tables. For large
samples, reconstruct smaller, disjoint sets of input files in separate runs.

### Parallel analysis

`analyze` and the analysis stage of `run` use one accumulator pipeline for both
one and multiple processes. Each worker reads an event once and updates every
selected analysis. The default uses one process; increase it explicitly:

```bash
uv run pvfit analyze --input output/baseline/fit_results.h5 --config configs/analysis.yaml \
  --output output/reanalysis --workers 4 --batches 16
uv run pvfit run --max-workers 4 --analysis-workers 8 --analysis-batches 32
```

`--workers` (`--analysis-workers`, `-w`) and `--batches` (`--analysis-batches`)
override `analysis.workers` and `analysis.batches` for `analyze`. For `run`, use
`--analysis-workers` and `--analysis-batches`; `--max-workers` still controls
reconstruction. With `batches: null`, analysis creates up to four batches per
worker, capped by the event count. A complete event always stays in one batch.

Workers write temporary residual/pull columns to a private directory beneath
`<output>/.parallel_work`. Set `analysis.work_dir` or `--analysis-work-dir` to
choose another parent. Only the run's own subdirectory is removed, including
on failure. `analysis.keep_columns: true` or `--keep-analysis-columns` retains
it for diagnostics; `--no-keep-analysis-columns` overrides a true YAML value.
All workers must see the input HDF5, its linked shards and the scratch directory.

Counts add exactly and means/standard deviations merge through central moments;
no rate or standard deviation is averaged over batches. Exact quantiles still
load one complete column, so memory is not constant in sample size. Workers
also consume memory; the main process's peak is not the whole job's requirement.
Allow scratch disk for the residual and pull observations. As a lower-bound
storage estimate, each retained float64 value takes 8 bytes, per column;
statistics and plotting may need additional working arrays. Start with a small
sample and a few workers, then measure memory and disk use with the selected
analyses before scaling up. Plotting preserves the input dtype to keep histogram
bin edges consistent. See [validation scope](analysis.md#parallel-analysis-validation)
for the checks performed when this pipeline replaced the original implementation.

The CLI saves the effective settings and actual event/batch/worker counts in
its provenance record. It writes `analysis.done` only after all outputs succeed.
Python callers use `run_analysis(path, output, AnalysisConfig(workers=8))`;
scripts that spawn workers must call it under `if __name__ == "__main__":`.

### Evaluating Allen output

`pvfit convert-allen` rewrites the monitoring ROOT file of Allen's
`upgrade2_tv_cheated_tracking` sequence into the HDF5 layout, so the GPU
reconstruction is scored by the same analyses. For standalone analysis, set
`analysis.method_name: Allen` in the analysis YAML to label the plots and output
files correctly; the HDF5 `source` attribute does not override that setting.
The event histogram uses `source` for its producer label independently:

```bash
uv run pvfit convert-allen --input /path/to/histograms.root --output output/allen/results.h5
uv run pvfit analyze --input output/allen/results.h5 --config configs/analysis.yaml \
  --output output/allen/analysis
```

Several `--input` files are keyed together by `(run_number, event_number)`.
The reconstruction parameters are taken from `allen_configuration.json` next
to each input (or `--allen-config`) and recorded under the same `reco_*`
attribute names `pvfit fit` writes; the binning is checked against the dumped
histogram. Without that file the conversion stops, unless
`--allow-missing-configuration` is given together with `--n-bins-z` and
`--n-bins-t`; then only the binning, read back from the dump, is recorded. See
[importing Allen output](data-format.md#importing-allen-output).

Conversion options (`pvfit convert-allen --help` lists them as well):

| Option | Default / purpose |
| --- | --- |
| `--input FILE [FILE ...]` | required monitoring ROOT input file(s) |
| `--output FILE` | required destination HDF5 file |
| `--allen-config FILE` | by default, read `allen_configuration.json` beside each input; this option supplies one configuration for all inputs |
| `--allow-missing-configuration` | allow conversion without that configuration; requires both bin counts below |
| `--n-bins-z`, `--n-bins-t` | use configuration values by default; explicit values must agree with it, or supply the counts when it is missing |
| `--min-truth-tracks` | default `4`; minimum associated input-track rows per stored true vertex |
| `--track-tree` | default `velo_kalman_filter/monitor_tree` |
| `--truth-tree` | default `mc_pv_dumper/MCPV_effective_tree` |
| `--histogram-tree` | default `pv_beamline_histo/histo_tree` |
| `--peak-tree` | default `pv_beamline_peak/peak_tree` |
| `--vertex-tree` | default `pv_beamline_merge_vertices/merge_vtx_tree` |
| `--debug` | show a full traceback on failure |

Tree overrides change paths, not the required branch schema.
`--min-truth-tracks` changes the stored truth population and thus the analysis
denominators; the converter records it as `truth_min_tracks`. It does not remove
input track rows or events, or alter Allen's reconstructed vertices. Use the same
truth selection when comparing producers. See [truth association](data-format.md#truth-association).

## Snakemake

```bash
uv run snakemake -n      # show the DAG
uv run snakemake -c 1    # run fit + analysis

# override the two stage configurations
uv run snakemake -c 1 --config \
  reconstruction_config=configs/reconstruction.yaml \
  analysis_config=configs/analysis.yaml
```

Snakemake reads input paths from the reconstruction YAML. Its default input
matches `pvfit toy`; update the YAML for another sample. Snakemake writes `logs/fit.log` and
`logs/analysis.log` beneath the reconstruction output directory.

## Configuration

The configuration loader rejects invalid values before execution and names the
key in its error message:

- Numbers must be finite unless a documented cut accepts `.inf` to disable it.
- Flags must be YAML booleans (`true`/`false`), not quoted strings or `0`/`1`.
- Recall and purity thresholds must lie in `[0, 1]`.
- Scales and residual windows must be positive. Other cuts accept zero only
  where the tables below explicitly allow it.

Each stage owns its configuration:

| File | Contents | Consumer |
| --- | --- | --- |
| [`configs/reconstruction.yaml`](../configs/reconstruction.yaml) | ROOT input, reconstruction output, workers/events, histogram and fit parameters | `pvfit fit`; reconstruction stage of `pvfit run` |
| [`configs/analysis.yaml`](../configs/analysis.yaml) | analysis selection, truth matching, plotting and resolution settings | `pvfit analyze`; analysis stage of `pvfit run` |
| [`configs/optimization.yaml`](../configs/optimization.yaml) | PVFit study settings, concurrency, search ranges and references to the two stage configs | `pvfit optimize --config ...` |
| [`configs/allen.yaml`](../configs/allen.yaml) | MDF inputs, existing Allen build, executable sequence and runtime settings | Allen optimization backend |
| [`configs/optimization_allen.yaml`](../configs/optimization_allen.yaml) | Allen study, search ranges, stage config references and Condor resources | `pvfit optimize --config ...` with the Allen backend |

The former root `config.yaml` has been replaced by these files. Put analysis
settings under `analysis:` in the analysis file, not under reconstruction's
`algorithm:` or `workflow:`. No stage merges the two parameter dictionaries.

The following `paths`, `workflow` and `algorithm` sections belong to
`configs/reconstruction.yaml`.

### `paths`

| Key | Meaning |
| --- | --- |
| `input_root` | input ROOT file or glob |
| `track_tree_name` | track tree path inside the file |
| `mc_vertex_tree_name` | truth vertex tree path inside the file |
| `output_dir` | output root; acts as the experiment id |
| `hdf5_file` | name of the HDF5 written under `output_dir` |
| `track_input_configuration` | optional; the configuration of the program that produced the input tracks (an Allen configuration JSON), whose `hit_time_resolution_ns` is recorded as `input_hit_time_resolution_ns`. Relative paths resolve against each input file's directory, e.g. `allen_configuration.json`; `null` records nothing |

### `workflow`

| Key | Meaning |
| --- | --- |
| `event_number` | internal event ID to process; `-1` selects all events |
| `max_workers` | parallel worker processes |

### `algorithm`

These parameters control PVFit's histogram seeding, track assignment, adaptive
fit and vertex merging. The table lists the shipped configuration values.
Set `max_track_beamline_chi2_xy`, `max_track_z0_err` or `max_vertex_rho2` to `.inf`
to disable that cut, or `min_chi2_dist_merge` to `0` to disable merging.
Assignment and Tukey thresholds must remain finite. The
[reconstruction metadata table](data-format.md#reconstruction-parameters)
also maps these parameters to the supported Allen sequence.

| Key | Shipped value | Meaning |
| --- | --- | --- |
| `z_min`, `z_max` | `-200`, `200` mm | seeding histogram extent in z; also the seeding track window |
| `n_bins_z` | `800` | z bins |
| `t_min`, `t_max` | `-0.8`, `0.8` ns | extent in t |
| `n_bins_t` | `64` | t bins |
| `smear_window_nbins` | `2` | bins on each side over which a track's Gaussian is spread |
| `max_track_beamline_chi2_xy` | `10.0` | skip tracks with a larger xy chi2 to the beamline |
| `max_track_z0_err` | `1.5` mm | skip tracks with a larger z error at the beamline |
| `min_tracks_in_seed` | `2.5` | 3×3 integral a histogram maximum needs — **see the scaling note below** |
| `max_track_seed_chi2` | `10.0` | a track joins its closest seed only below this chi2 to the seed; decides who enters a fit |
| `min_tracks_per_seed` | `4` | drop seeds with fewer tracks and reassign them; `0` keeps every seed |
| `max_chi2_tukey` | `12.0` | Tukey bisquare threshold of the track chi2 in the fit; assigned tracks above it at the fitted position get zero weight and are not members of the vertex |
| `max_fit_iterations` | `20` | iterations of the vertex fit |
| `max_delta_z_converged` | `0.0005` mm | the fit stops once a z step is smaller |
| `max_vertex_rho2` | `0.09` mm² | drop fitted vertices farther from the beamline |
| `min_chi2_dist_merge` | `25.0` | merge and refit vertex pairs closer than this `(z, t)` chi2 distance; `0` disables merging |
| `is_adaptive_tukey` | `true` | PVFit-only debugging switch: recompute the chi2 each iteration (Allen's behaviour) or reuse the assignment's |

> **PVFit scales `min_tracks_in_seed` with the histogram bin area**, as Allen's
> `pv_beamline_peak` does. It multiplies the configured threshold by the bin
> area relative to `bin_size_z = 0.5` mm and `bin_size_t = 0.025` ns (that is,
> `n_bins_z: 800` over `[-200, 200]` and `n_bins_t: 64` over `[-0.8, 0.8]`), so
> that a peak threshold stays physically comparable when the binning changes.
> At the reference binning the factor is exactly 1. See
> `pvfit/reconstruction_parameters.py`.

`ReconstructionConfig` and `AnalysisConfig` in `pvfit/config_loader.py`
hold defaults and derived values. Both reject unknown keys before execution,
and older YAML keys are not aliases for the current names (the former
`algorithm.max_chi2` maps to neither `max_track_seed_chi2` nor
`max_chi2_tukey`). Migrating settings and results from earlier versions is
described once, under [older HDF5 files](data-format.md#older-hdf5-files).

### Analysis settings

These keys belong to `analysis:` in `configs/analysis.yaml`:

| Key | Shipped value | Meaning |
| --- | --- | --- |
| `analyses` | list in the shipped YAML | analysis modules to run |
| `method_name` | `PVfit` | labels and output filename prefix |
| `workers` | `1` | analysis worker processes; one uses the same accumulator pipeline without spawning |
| `batches` | `null` | automatic event batches (four per worker), or a positive explicit count |
| `work_dir` | `null` | parent of the run's temporary column directory; defaults to `<output>/.parallel_work` |
| `keep_columns` | `false` | retain the run's temporary residual/pull columns after execution |
| `recall_threshold`, `purity_threshold` | `0.7`, `0.7` | track-based matching thresholds |
| `track_pull_chi2` | `null` | use the recorded threshold; a positive finite number overrides the [input-track diagnostic cut](analysis.md#input-track-residual-diagnostics) |
| `z_scale`, `t_scale`, `m_r`, `nsigma` | `1.5`, `0.06`, `0.8`, `5.0` | geo (elliptical) matching in `(z, t)` |
| `plot_run_number`, `plot_event_number` | `null`, `null` | original run/event identity to plot; set both, or leave both null for the first stored event |
| `draw_fixed_ellipses`, `draw_fitted_ellipses` | `false`, `true` | overlays on that plot |
| `plot_min_ntracks`, `plot_max_ntracks`, `plot_bin_width_ntracks` | `4`, `200`, `5` | track-multiplicity bins; overall rates include vertices outside them ([population](analysis.md#overall-population-and-plotting-bins)) |
| `efficiency_plot_max_ntracks` | `150` | horizontal display limit of all LHCb2 efficiency figures; keeps complete bins and does not change overall rates |
| `rate_plot_max_ntracks` | `150` | horizontal display limit of the LHCb2 fake/merge/split rate figures |
| `rate_plot_ymax` | `{}` | optional per-plot y maxima, e.g. `{merge_rate_geo_rec: 0.25}`; otherwise use central values without expanding for long error bars |
| `x_residual_range`, `y_residual_range`, `z_residual_range`, `t_residual_range` | `0.1`, `0.1`, `0.7` mm; `0.06` ns | half-width of the residual window the `resolution` analysis keeps per coordinate; the reported std and bias describe the core inside it |

Reconstruction's `workflow.event_number` uses the
[dense internal ID](data-format.md#event-identity). The histogram instead selects
an event by its original `(run_number, event_number)` pair:

```yaml
analysis:
  plot_run_number: 22001
  plot_event_number: 3
```

See [event histogram selection](analysis.md#event-histogram-selection) for the
default, identity requirements and migration from `plot_event_id`.
[Analysis definitions](analysis.md#matching-and-rate-definitions) describe the
matching criteria and the populations behind each metric.

## Run provenance

Each CLI `fit` and `analyze` stage (including the two stages of `run`) writes a
unique `provenance/<UTC timestamp>_<id>_<stage>/` directory under its output
directory. It contains `manifest.json` and an effective `reconstruction.yaml`
or `analysis.yaml`, including CLI overrides. Reruns retain previous records.
The manifest records the invocation and working directory, input paths/sizes/
modification times, package version and source SHA-256, Git commit and dirty
flag when available, Python/dependency versions, and completion or failure.
Input identities use file metadata, not content checksums of large ROOT files.
A hard process kill may leave the status `running`; this is not completion.

HDF5 files store the schema version, package/source identity, effective fitter
parameters and, for CLI fits, the provenance manifest path. Copy the provenance
directory with the results to retain the full record; absolute recorded paths
describe the original run and may no longer resolve after moving it. Analysis
does not modify its input HDF5. Direct Python calls retain HDF5 writer metadata;
the per-stage manifests are a CLI feature. Optimizer studies retain their
existing configuration and input/source fingerprint checks.

`uv.lock` fixes the tested environment. Package dependency ranges allow
compatible installations (including `mplhep>=1.1.2,<2`); use `uv sync --locked`
to reproduce the locked plotting stack.

## Troubleshooting

The messages below are identifying fragments; details include the affected file,
key or event. Use `--debug` for a traceback.

| Message or symptom | Cause | Action |
| --- | --- | --- |
| Missing `run_number` branch | ROOT event identity is incomplete. | Produce a dump containing run and event numbers in each required tree; do not infer run identity from event numbers alone. See [event identity](data-format.md#event-identity). |
| Duplicate `(run_number, event_number)` across files | The same event identity occurs in multiple inputs. | Check the input list for overlapping dumps and use disjoint inputs. Do not merge distinct events by inventing new identities. |
| Missing `allen_configuration.json` | Conversion cannot recover all reconstruction settings from monitoring trees. | Supply the original `--allen-config`; if it is unavailable, explicitly use `--allow-missing-configuration` with both bin counts and retain the resulting incomplete provenance. |
| Different Allen configurations or mixed configured/unconfigured inputs | One output cannot describe several reconstruction configurations. | Convert each configuration separately. See [Allen import metadata](data-format.md#reconstruction-parameters-from-the-allen-configuration). |
| `track_pull` cannot resolve a positive finite threshold | Required diagnostic configuration is absent or invalid. | Follow the [threshold resolution rules](analysis.md#input-track-residual-diagnostics), using the original run settings or an explicit diagnostic override. |
| Missing `tracks/tx`, `tracks/ty`, `tracks/backward` or variances | The chosen diagnostic needs fields absent from this HDF5. | Regenerate from original inputs, or select analyses whose fields are present. See [HDF5 compatibility](data-format.md#older-hdf5-files). |
| Unknown configuration key | The YAML uses an obsolete name or a key from another stage. | Compare with `pvfit init-config` templates in a new directory; migrate the original values rather than replacing them with current defaults. |
| Requested histogram identity is absent or ambiguous | The run/event selection does not identify one stored event. | Check the event attributes and supply both identity values; see [histogram selection](analysis.md#event-histogram-selection). |
| Memory exhaustion or a full scratch disk | Input loading, concurrent workers or retained diagnostic columns exceed resources. | Read [reconstruction memory](#reconstruction-workers-and-memory) or [analysis memory](#parallel-analysis), reduce the sample/concurrency as appropriate, and choose scratch storage with enough space. |
