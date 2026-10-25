# Synthetic reconstruction regression

`toy_regression_input.json` contains **generated synthetic rows**, not recorded
simulation events. It freezes two small events (91 input tracks; 83 within the seed window)
from the bundled aggregate toy profile, seed `20260929`, with the multiplicity
overrides and event ordering recorded in the file. Freezing rows also avoids
making this reconstruction test depend on changes in NumPy's random sampler.
The reconstruction and analysis configurations are stored with the input.

`toy_regression_expected.json` was first frozen using the pre-release source
snapshot at commit `f19f09e`, before adding packaging and provenance, and
re-frozen on 2026-10-02 when the reconstruction parameters took Allen's names
and defaults (Tukey threshold 12 instead of 10, separate track-seed chi2 cut,
beamline distance cut). That review found only the fitted coordinates and
errors changed (z by at most 0.12 mm, errors by up to 20%); seeds, memberships,
counts and all rate tables were unchanged. The baseline has two reconstructed
vertices in each event and 37 CSV outputs:
18 per-bin rate tables, their 18 overall tables, and one population table.
On 2026-10-05 the configuration keys became `max_track_seed_chi2: 10` and
`max_chi2_tukey: 12` (separate assignment and Tukey thresholds, as in Allen) and
a vertex's stored members became the assigned tracks with nonzero Tukey weight
at the fitted position. The frozen expected results did not change: in these
two events no assigned track exceeds the Tukey threshold at the fit, so the
memberships, fitted values and all rate tables are the same as before.
`tests/test_chi2_thresholds.py` covers the two thresholds and the membership
rule on hand-built events. On 2026-10-14 the rate tables gained a
`design_effect` column (event-cluster widening of the Clopper-Pearson errors);
the baseline was re-frozen for that column only, with every count, rate and
error value unchanged (the two events hold no correlated outcomes).
The test compares event identities, input MC keys, truth/reconstructed track
memberships, seeds, fitted coordinates/errors, histogram sums and every CSV.
It runs with both one and two reconstruction workers.

Identities, memberships and counts must match exactly. Fitted values use
`rtol=2e-6, atol=1e-7` (stored HDF5 coordinates/errors are float32); rates and
confidence intervals use `rtol=1e-10, atol=1e-12`. The test does not compare
rendered files or serialize interpreter-specific objects. The hand-built rate
tests separately cover fake, merge and split cases absent from this tiny sample.

Update a baseline only after reviewing the numerical and membership changes.
Use `tests.data.toy_regression.reconstruct` and `summarize` to produce a candidate
in a temporary directory, then review its diff against the committed JSON.
Tests never regenerate or silently accept expected results.

## Development analysis validation

The old serial comparison suite and its frozen outputs are maintained separately
locally in `dev/analysis_validation/`. They are run explicitly and are excluded
from version control, this test-data directory and release distributions.
