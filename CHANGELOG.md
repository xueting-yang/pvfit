# Changelog

## [Unreleased]

- Count all stored true vertices and all nonempty reconstructed vertices in
  rate overall results, independently of plotting bins. Merge complete totals
  and event-level uncertainty statistics across workers. Report bin bounds,
  binned counts, underflow/overflow counts, rates and population fractions in
  the existing overall CSVs. Match the optimizer's counted populations while
  preserving per-bin results, figures and reconstruction. Existing HDF5 files
  need analysis only to update the overall results.

- Widen the rate uncertainties by an event-level design effect: every rate
  counter closes each event and accumulates the sums needed for the
  ratio-estimator (cluster) variance, and the Clopper-Pearson errors are
  multiplied by `sqrt(cluster / binomial)`, clamped at 1. Merge per true
  vertex and split per reconstructed vertex count correlated pairs and were
  understated by about √2; fake, split per true and merge per rec are
  unchanged. The rate CSVs gain a `design_effect` column; counts and rates
  are unchanged. The synthetic regression baseline was re-frozen for the new
  column.
- Retire the original multi-pass serial analysis modules and development
  prototype. Single-process analysis remains available through `--workers 1`,
  using the same accumulators as parallel execution. Preserve independent
  migration checks with frozen serial CSV/JSON and residual/pull arrays under
  `dev/analysis_validation/` locally; exclude them from version control,
  default tests and both release distributions.

- Run `pvfit analyze` and the analysis stage of `pvfit run` through a shared
  event-streaming accumulator pipeline. Add configurable analysis workers,
  batches and private scratch storage; keep matching definitions and residual
  windows unchanged. Merge counts and central moments, retain exact quantiles
  and preserve source dtypes for histograms. Record actual execution counts in
  provenance and test the installed CLI with spawned workers.

- Share one analysis YAML between Allen and PVFit optimization; derive Allen's
  output label from its backend and omit its fixed worker-count setting. Add
  `--prepare-condor` and `--submit-condor`: one GPU job runs a complete study,
  with frozen configurations, queued-job provenance checks, GPU environment
  restoration, per-study locking and restart-aware trial budgets.

- Add an `allen` optimization backend: sample runtime PV properties into a
  private executable sequence JSON, run Allen on MDF inputs, convert its ROOT
  monitoring output to HDF5, and score it with the existing track-based
  efficiency/fake/merge/split objectives. Synchronize binning and Tukey
  properties across their consumers without double scaling the seed threshold.
  Add Allen example configs, external-build fingerprints for safe study resume,
  single-trial/thread limits, stage timeouts and isolated output checks.

- Name the reconstruction parameters after Allen's beamline PV sequence and
  record each as an HDF5 root attribute `reco_<name>`, written identically by
  `pvfit fit` and `pvfit convert-allen` (list in
  `pvfit/reconstruction_parameters.py`, Allen mapping in `pvfit/allen.py`).
  The `algorithm` YAML keys take the same names; the on/off flags are gone
  (an infinite threshold disables a track or vertex cut, `0` disables merging;
  assignment and Tukey thresholds stay finite). The optimizer search space
  uses the same names and varies the assignment cut and the Tukey threshold
  separately. The JSON attributes `reconstruction_parameters`,
  `allen_configuration` and `reconstruction_max_chi2` are no longer written;
  `track_pull` reads `reco_max_chi2_tukey`. Old YAML keys are not aliases, and
  the old attribute is not a threshold fallback: see
  [older HDF5 files](docs/data-format.md#older-hdf5-files).
- Match Allen's fit in PVFit: separate assignment cut (`max_track_seed_chi2`,
  10) and Tukey threshold (`max_chi2_tukey`, 12); configurable iteration count
  and convergence step; seeds whose fit is singular or whose fitted position
  lies farther than `sqrt(max_vertex_rho2)` from the beamline produce no
  vertex; a vertex's members are the assigned tracks with nonzero Tukey weight
  at the fitted position, as Allen writes them. Fitted values and memberships
  change; the synthetic regression baseline was re-frozen after review.
- Add `pvfit convert-allen`: Allen's monitoring ROOT output (tracks, effective
  truth, seeding histogram, peaks, merged vertices) becomes a PVFit HDF5 file,
  keyed by `(run_number, event_number)`, with truth membership from the full
  track table and reconstructed memberships from Allen's `track_mc_keys`. The
  reconstruction parameters come from the `allen_configuration.json` next to
  each ROOT file, checked for consistency and against the dumped histogram;
  without it the conversion stops unless `--allow-missing-configuration`
  records only the binning. `pvfit fit` records the track input's
  `hit_time_resolution_ns` as `input_hit_time_resolution_ns` when
  `paths.track_input_configuration` names that file.
- Apply the POCA z/t window only to seed-histogram input. Assign, fit and
  merge using all input tracks, with the existing chi-square and Tukey cuts.
  Window-excluded tracks can join an existing seed; fitted vertices and their
  memberships can change. Save the full track table, `selected_for_seeding`,
  `n_seeding_tracks` and `assignment_track_selection="all_input"`.
- Build truth membership and the four-track truth minimum from all input
  tracks. Truth and reconstruction share the full row order. Truth
  multiplicities, recall and performance denominators can change relative
  to window-selected truth. Existing HDF5 files retain their original
  semantics; rerun reconstruction to use the new assignment convention.

- Draw 2D event histograms in LHCb2 style, with a coloured density map,
  a 4:3 data area, compact vertex markers and external event labels/legend. Export PDF
  and PNG; fit-error ellipses follow the analysis sigma setting.

- Select event histograms by the original run/event identity with
  `plot_run_number` and `plot_event_number`, replacing `plot_event_id`.
  Both default to null to plot the first stored event by numeric group ID.
  Name plots with the run/event pair when metadata is available; retain
  default plotting for older files without identity metadata.
- Set the resolution-scaled analysis window to 5 sigma by default. When both
  fake analyses run, add an LHCb2 comparison of track-overlap and geometric
  fake rates with their original asymmetric uncertainties.
- Use unique claiming for both fixed and resolution-scaled geometric efficiency.
  True vertices propose to rec candidates by increasing normalized distance;
  recs retain their closest applicants, replacing owners when a better applicant
  arrives. Rejected and displaced true vertices try their next candidates with
  no retry cap. Deterministic ties preserve the result of globally sorted
  distance-first matching. Efficiency counts true vertices in the final pairs.
  This changes geometric efficiency relative to v0.1.0; rerun analysis to obtain
  the new efficiency rates. Track-based efficiency and merge/split candidate
  relations remain unchanged.
- Define fake rates by the absence of any qualifying true candidate, without
  unique claiming, best-candidate selection, competition or retries. A track
  candidate shares at least one track and reaches either the purity or recall
  threshold; a geometric candidate lies in the fixed or resolution-scaled
  ellipse (5 sigma by default). Qualifying split copies no longer count as
  fake. Physical splits that fail the candidate criterion can still count as
  fake, and track-defined merges can still count as geometric fake. Efficiency
  and merge/split definitions remain unchanged by this fake-rate update.
  Rerun `pvfit analyze` on existing HDF5 files to obtain the new fake rates;
  this analysis change does not require rerunning reconstruction. Previous
  output files and slides retain their historical definitions.

## [0.1.0] - 2026-09-29

Initial versioned release of the public PVFit workflow.

- Reconstruct 4D primary vertices, analyze compatible HDF5 files independently,
  and run Optuna parameter studies.
- Identify input events by `(run_number, event_number)` and associate truth
  tracks by event-local keys.
- Provide a synthetic ROOT generator and a frozen synthetic reconstruction
  regression with exact membership checks and numerical tolerances.
- Report residual standard deviations, bias and statistical uncertainties;
  let input-track diagnostics inherit reconstruction's chi-square threshold.
- Use LHCb2 rate plots and export true/reconstructed multiplicity distributions
  separately from the efficiency numerators and denominators.
- Record effective CLI configurations, input identities, source and dependency
  versions, and stage status; mark newly written HDF5 files as schema version 1.
- Include default YAML configurations in wheels and provide `pvfit init-config`.
  Default reconstruction input is the generated `input/toy_sample.root`.
- Add CI for Python 3.12 and 3.13, distribution installation checks, and tagged
  GitHub releases with wheel and source archives.

The existing geo, track and reso matching/rate definitions and optimization
objectives remain unchanged. See [analysis documentation](docs/analysis.md)
for their definitions and sample-specific validation observations.
