# PVFit

PVFit is a Python package for reconstructing primary vertices in three spatial
coordinates and time, evaluating reconstruction performance, and optimizing
algorithm parameters. It was developed for LHCb Upgrade-II studies.

## How it works

The pipeline has two stages joined by a single HDF5 file:

1. **Fit** — read tracks and MC truth vertices from ROOT, fill a 2D `(z, t)`
   histogram of track-beamline intersections, find seeds as histogram peaks,
   fit each seed with an adaptive Tukey-weighted vertex fit, optionally merge
   close vertices, and write the results to HDF5.
2. **Analysis** — read that HDF5 and produce performance plots and summary
   tables: efficiency, fake, merge and split rates, resolution, and pulls.

The HDF5 file lets you rerun analyses without repeating the fit. Analysis
supports one or [multiple worker processes](docs/usage.md#parallel-analysis).

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).
Install uv if needed, then clone the project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/shenzh15/pvfit.git
cd pvfit
```

## Quick start

From the project root, install the locked dependencies, generate ten synthetic
events and run reconstruction plus analysis:

```bash
uv sync --locked
uv run pvfit toy --events 10
uv run pvfit run --config configs/reconstruction.yaml --input input/toy_sample.root \
  --output-dir output/toy --analysis-config configs/analysis.yaml --max-workers 2
```

The results are `output/toy/fit_results.h5` and plots and tables under
`output/toy/analysis/`. The toy generator refuses to overwrite an existing
sample; choose another `--output` and pass it to `run --input` when repeating it.
Toy results demonstrate the workflow; they are not detector performance.
See the [toy model and limitations](docs/toy.md).
Each fit/analysis stage also saves its effective configuration and
[run provenance](docs/usage.md#run-provenance).

For installation from a release wheel, use `python -m pip install <wheel-file>`.
Bundled defaults let `pvfit toy` followed by `pvfit run --max-workers 2` work
outside a checkout; `pvfit init-config` exports editable YAML templates.

## Using your own data

For Python reconstruction, pass your ROOT file or a quoted glob to `fit --input`,
or set `paths.input_root` in [configs/reconstruction.yaml](configs/reconstruction.yaml).
The input must follow the documented [track and truth format](docs/data-format.md#root-input).
To evaluate results from another reconstruction program, write them in the
[HDF5 analysis format](docs/data-format.md#hdf5-layout) and run `pvfit analyze`
without running the PVFit fitter. Choose metrics and plotting settings in the
separate [analysis configuration](configs/analysis.yaml).

## Working with Allen

PVFit can also evaluate and optimize Allen's GPU reconstruction:

- **Import results** — `pvfit convert-allen` converts Allen's reconstructed
  vertices from monitoring ROOT files to HDF5 for the same performance analysis.
  See [commands and input requirements](docs/usage.md#evaluating-allen-output).
- **Optimize parameters** — `pvfit optimize` tunes runtime PV parameters in an
  existing Allen build, with support for running a complete study in one Condor
  GPU job. See [setup and execution](docs/optimization.md#optimizing-allen).

## Documentation

- [Usage and configuration](docs/usage.md): CLI commands, Snakemake and YAML settings.
- [Data formats](docs/data-format.md): ROOT inputs, event identity and the HDF5 interface for other producers.
- [Synthetic example](docs/toy.md): generation, calibration and limitations.
- [Analysis](docs/analysis.md): matching, metric definitions, output files and track diagnostics.
- [Optimization](docs/optimization.md): scans, input selection, study resume and backends.
- [Derivations](docs/derivations/README.md): mathematical reference notes and notebooks.

## Development

```bash
uv sync --locked --group dev
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_docs.py
```

The implementation lives in `pvfit/fit/`, `pvfit/analysis/`, `pvfit/toy/` and
`pvfit/optimize/`, with the Allen import in `pvfit/convert_allen.py`, the Allen
configuration mapping in `pvfit/allen.py` and the shared reconstruction
parameter record in `pvfit/reconstruction_parameters.py`; `pvfit/cli.py`
provides the command-line entry point.
CI runs these checks and verifies a wheel installation outside the checkout.
The [synthetic regression](tests/data/README.md) freezes numerical behavior
without shipping reference events. See the [changelog](CHANGELOG.md) for releases
and [CITATION.cff](CITATION.cff) for citation metadata.

## Acknowledgements

PVFit reads track and truth data from monitoring outputs of
[Allen](https://gitlab.cern.ch/lhcb/Allen), the LHCb GPU trigger framework
(© CERN for the benefit of the LHCb Collaboration, Apache-2.0).

## License

MIT — see [LICENSE](LICENSE).
