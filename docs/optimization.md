# Parameter optimization

[Back to README](../README.md)

`pvfit optimize` runs a multi-objective scan with Optuna's NSGA-II sampler over
four **track-based** objectives (the reconstruction backend does not change the scoring definition):

| Objective | Direction | Denominator |
| --- | --- | --- |
| Efficiency | maximize | true vertices |
| Fake rate | minimize | rec vertices |
| Merge rate | minimize | true vertices |
| Split rate | minimize | rec vertices |

[configs/optimization.yaml](../configs/optimization.yaml) defines the study
name/output, trial count, parallel jobs, sampler seed, population size, workers
per trial, and the search ranges.
The `execution.reconstruction_config` and `execution.analysis_config` references
are resolved relative to the optimization YAML's directory. Data and output
paths remain relative to the working directory. CLI options override YAML
settings.

## Search space

Nine parameters are varied together, named as in `reconstruction.yaml`:
`min_tracks_in_seed`, `smear_window_nbins`, `n_bins_z`, `n_bins_t`,
`max_track_seed_chi2`, `max_chi2_tukey`, `min_chi2_dist_merge`,
`max_track_beamline_chi2_xy`, `max_track_z0_err`. The track-seed assignment
cut and the Tukey threshold are separate parameters, so a study can change
which tracks enter a fit independently of how it weights outliers. Their
bounds, steps and types are in `search_space`.

## Input selection

The optimizer defaults to the **first file in sorted glob order** from its
backend's input setting: `paths.input_root` for PVFit or `paths.input_mdf` for
Allen. To use every matching file, set `execution.all_input_files: true` in the
optimization YAML or pass `--all-input-files`. This differs from `fit` and `run`,
which read all ROOT matches.

## Commands

Set the reconstruction YAML's `paths.input_root` to an existing sample first;
the toy quick start's `--input` override does not change that YAML.

```bash
# use the study settings and search ranges from YAML
uv run pvfit optimize --config configs/optimization.yaml

# smoke test
uv run pvfit optimize --config configs/optimization.yaml \
  --n-trials 3 --n-jobs 1 --trial-workers 1 --study-name pvfit_smoke

# include all matching ROOT files in a separate study
uv run pvfit optimize --config configs/optimization.yaml \
  --all-input-files --study-name pvfit_all_inputs

# re-export the Pareto front from an existing study, no new trials
uv run pvfit optimize --config configs/optimization.yaml --export-only
```

## Resuming a study

The same study name continues the same SQLite study and keeps completed trials.
The study records the backend, reconstruction and analysis settings, search
space, working directory, selected input paths and each selected file's size
and modification time. It also records the installed package version and a
digest of the `pvfit` Python source and JSON files.

Changing these recorded values requires a new study name; the error names what
changed. With all inputs enabled, adding a matching file changes the fingerprint.
With the default first-file selection, only a change to the selected file or
its identity does so; adding a later-sorting file leaves the selection unchanged.
Changes to Python source include comments and docstrings. Markdown documentation
outside `pvfit/` does not affect the digest.

The sampler seed and population size must also stay fixed. You can change the
additional trial count or worker/job counts. Studies created before the stage
configuration split or input/code fingerprinting support export only.

> Optuna preserves the study state across a resume but **not** the NSGA-II
> sampler RNG state, so a resumed run can re-suggest a point it already sampled.
> `pvfit optimize` detects exact duplicate parameter signatures and reuses the
> stored objective values instead of re-running the pipeline. That reuse is safe
> only because the study's inputs and code are pinned as described above.

## Results

Results land in `output/optuna/<study_name>/`, with `all_trials.csv`,
`pareto_front.csv`, `generation_summary.csv` and a `trial_N/` directory per trial
holding the backend's configuration, `analysis.yaml`, logs, parsed `metrics.json`
and analysis CSVs. Each trial changes only the reconstruction parameters. The
analysis configuration retains its thresholds and runs the four required metric
modules.

Use [analysis_optuna.ipynb](../notebooks/analysis_optuna.ipynb) to inspect these
results. It builds on the helpers in `pvfit/optimize/analysis.py`. Notebook outputs
are stripped in the repository; run the cells yourself.

## Adding a backend

`execution.backend` names the reconstruction program each trial runs. A backend
(`pvfit/optimize/backends/`) turns the sampled parameters into that program's
configuration and produces an HDF5 file in the
[PVFit layout](data-format.md#hdf5-layout). The trial scores it with
`pvfit analyze`, so every backend uses the same evaluation code.
`pvfit` and `allen` are available. Adding another backend means implementing the `Backend` methods in `backends/base.py` — the parameter
names in `search_space` and any derived values are the backend's to define —
and registering it in `backends/__init__.py`. A backend can also cap how many
trials run at once, which `n_jobs` must respect.


## Optimizing Allen

Use [configs/optimization_allen.yaml](../configs/optimization_allen.yaml), which
selects `execution.backend: allen`, references
[configs/allen.yaml](../configs/allen.yaml) for reconstruction, and shares
[configs/analysis.yaml](../configs/analysis.yaml) with PVFit. The Allen backend
sets the output method name to `Allen`; evaluation thresholds and the four
track-based objectives remain the same.

The backend targets the `upgrade2_tv_cheated_tracking` beamline PV sequence
with the monitoring extensions described by the
[Allen conversion interface](data-format.md#importing-allen-output). It does
not support arbitrary Allen sequences. You need:

- a compiled Allen installation providing that sequence and the required monitors;
- an allocated GPU;
- **MDF** inputs containing truth banks;
- a **complete executable sequence JSON** from that build.

A parameter-only JSON or reconstructed ROOT file cannot replace the MDF and
executable sequence. The monitors must dump tracks, effective truth, histogram,
peaks and final vertices with event identity and vertex member keys.

1. Build and validate the Allen stack once using its normal workflow.
2. Set `ALLEN_ROOT` to the Allen checkout containing `run` and `InstallArea`.
3. Edit `configs/allen.yaml`: select the MDF glob, the sequence JSON, geometry
   tags, event count, GPU device and runtime files. A production
   `allen_configuration.json` is suitable if it contains the `sequence` graph
   and comes from this build. Enable the histogram, peak and merged-vertex
   monitors in that base sequence.
4. Start a small study on the GPU worker:

   ```bash
   export ALLEN_ROOT=/path/to/stack/Allen
   pvfit optimize --config configs/optimization_allen.yaml --n-trials 2
   ```

Stage configuration references are relative to `optimization_allen.yaml`, as
for PVFit. Paths *inside* `allen.yaml` are relative to the working directory and
accept `~` and environment variables. `runtime.params` is expanded inside
`Allen/run` so `${PARAMFILESROOT}` comes from the stack environment. The default
DD4hep detector list is `[/world, TV]`; use an empty list only if the entrypoint
should retain its own detector configuration.

For each trial the backend:

1. Copies the full baseline sequence to `trial_N/allen_configuration.json` and
   writes the sampled runtime properties into it.
2. Runs the configured `allen.py` through `Allen/run`, in the trial directory,
   with that JSON, the sorted MDF file list and fixed event count.
3. Converts `histograms.root` into `fit_results.h5`, using that same trial JSON
   as reconstruction provenance.
4. Evaluates the HDF5 with the shared analysis modules and records the usual
   objective/audit CSVs and `metrics.json`.

The backend supports the same nine parameter names as the PVFit search. Allen
studies may vary a subset; omitted parameters keep their baseline JSON values.

| Parameter | Allen property |
| --- | --- |
| `min_tracks_in_seed` | peak `pp_minTracks` |
| `smear_window_nbins` | histo `smear_window_nbins` |
| `n_bins_z`, `n_bins_t` | histo and peak `Nbins_z`, `Nbins_t`; also recomputes `dz`, `dt` |
| `max_track_seed_chi2` | assign_tracks `maxTrackSeedChi2XYT` |
| `max_chi2_tukey` | fitter and merge_vertices `maxChi2Tukey` |
| `min_chi2_dist_merge` | merge_vertices `minChi2Dist` |
| `max_track_beamline_chi2_xy` | histo `maxTrackBLChi2XY` |
| `max_track_z0_err` | histo `pp_maxTrackZ0Err` |

Each algorithm name above has the `pv_beamline_` prefix. The mapper also accepts
a single suffixed instance of each algorithm and rejects ambiguous instances.
Bin edges stay fixed. `min_tracks_in_seed` remains **unscaled** in the JSON:
the Allen peak kernel applies the bin-area factor itself. The exported
`derived_scaled_min_tracks_in_seed` uses the current stack's reference bin widths
of 0.5 mm and 0.025 ns; a stack with a different reference convention needs a
corresponding adapter change. No C++ files or shared build configuration are
edited per trial, and these runtime properties require no per-trial compilation.

Use `n_jobs: 1`. Allen defaults `trial_workers` to 1, so its YAML omits that
fixed setting. It runs with one thread, slice and repetition to keep monitoring
dumps unique; the backend rejects larger trial concurrency. `runtime.timeout_seconds` bounds reconstruction and conversion
separately. A timeout kills the stage process group. A failed command, absent
output or stale output fails the trial rather than scoring another trial's ROOT.
For direct execution, run on an allocated GPU worker. The Condor submission
mode below allocates one GPU for the whole study.

`paths.runtime_files` lists the installed libraries and any external scripts,
local geometry or parameter assets that affect reconstruction. Globs and
recursive directory contents are supported. The optimizer hashes these files,
the launcher, entrypoint and baseline sequence before opening/resuming a study;
changes require a new study name, just like input or evaluation changes. List
the actual runtime assets of the selected build and keep them fixed throughout
a study. The default glob covers Allen libraries, not the whole LHCb software
stack or remote conditions database.

Each trial additionally saves `allen_trial.json` (input list, commands and sampled
parameters) and `allen_command.json` (arguments after expansion in the Allen
Python environment). The generic `reconstruction.yaml` snapshot remains the
baseline runtime settings; `allen_configuration.json` contains the actual trial
reconstruction properties. Duplicate parameter points reuse completed metrics
under the same pinned study configuration.


## One Condor job per optimization

For Allen optimization on an HTCondor pool with a shared filesystem, submit
the **whole study as one job**. The job runs Optuna and all Allen trials
sequentially on its allocated GPU. Each trial retains its own reconstruction
and analysis files; trials do not create additional Condor jobs.

After filling the Allen paths and input in `configs/allen.yaml`:

```bash
# Prepare snapshots and validate the submit file; do not enqueue anything.
pvfit optimize --config configs/optimization_allen.yaml --prepare-condor --n-trials 2

# Prepare and submit one GPU job for these two trials.
pvfit optimize --config configs/optimization_allen.yaml --submit-condor --n-trials 2
```

Without these flags, `pvfit optimize` still runs directly. `--export-only` runs
on the submission host and cannot be combined with the Condor flags.
`pvfit init-config` exports the Allen runtime and optimization templates along
with the shared analysis configuration.

The `condor` section sets CPUs, RAM, scratch disk, GPU model, OS and maximum
run time for the **whole job**. Review it for your installation before submitting:
the shipped values are a [Nikhef deployment example](#site-example-nikhef), not
portable hardware requirements.

GPU selection and environment:

- Set `runtime.device: "0"`, meaning the first GPU visible through Condor's
  `CUDA_VISIBLE_DEVICES`, not physical GPU zero on the worker.
- The launcher restores GPU visibility and CUDA/driver library paths after
  `Allen/run` resets its environment.
- If the stack needs an external CUDA toolkit, set `runtime.cuda_bin` and
  `runtime.cuda_lib` in `allen.yaml`, for example `${CUDA_ROOT}/bin` and
  `${CUDA_ROOT}/lib64` on the submission host.

This mode uses `should_transfer_files = NO`. The worker must see PVFit's Python
environment, Allen installation, MDF inputs, local assets and output directory
at the same paths as the submission host. The generated script uses explicit
paths rather than copying the host's entire environment.

Every invocation creates
`<study_dir>/submissions/<timestamp>_<id>/` containing:

- frozen optimization, reconstruction and analysis YAML snapshots;
- `manifest.json`, including input and implementation identities;
- `job.sh`, `job.sub` and dry-run `job.ad`;
- `submission.json` with submission state and the scheduler's job ID output;
- `condor.out`, `condor.err`, `condor.log`, and worker `status.json`;
- `trial_budget.json`, created on first execution to track this job's budget.

The worker checks that software, inputs and submitted snapshots have not changed
while the job was queued. It acquires a study write lock before optimizing, so
another local optimizer or Condor job cannot simultaneously write the same
study. The scheduler holds jobs that exit unsuccessfully or exceed the wall-time
budget; holding releases their GPU allocation. Inspect the recorded error before
releasing or replacing a held job.

A Condor restart resumes the saved study with the **remaining number of trial
attempts**, rather than adding the requested `n_trials` again. Completed trials
retain their values. An interrupted RUNNING attempt becomes FAILED and consumes
one attempt; a new submission can request additional trials after inspection.
A separate new `--submit-condor` invocation intentionally requests another batch
of additional trials, so it should not overlap an active submission for the same
study. The worker does not automatically resubmit failed jobs.

The generated submit directives use the [HTCondor submit-file interface](https://htcondor.readthedocs.io/en/25.0/man-pages/condor_submit.html).
The preparation command validates them with the locally installed
`condor_submit -dry-run` before any real submission.

### Site example: Nikhef

The shipped `configs/optimization_allen.yaml` requests one L40S, 4 CPUs,
16 GiB RAM and EL9. These settings describe the deployment used for development;
choose a GPU compatible with your Allen build and size memory/disk requests for
your inputs and selected analyses.

`+UseOS = "el9"`, `+JobCategory = "medium"` (from `condor.job_category`) and the
GPU model expression are pool-specific. Nikhef requires a job category for
submission; `condor_submit -dry-run` does not check that scheduler policy.
Review these settings with your pool's requirements before using the template
elsewhere. If another pool requires unsupported submit directives or file
transfer, the current submission helper needs adaptation; use direct execution
on an allocated GPU worker in the meantime.
