"""Command-line interface for reconstruction and independent HDF5 analysis."""

import argparse
import glob
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from . import __version__
from .config_loader import (
    ANALYSIS_NAMES,
    AnalysisConfig,
    ReconstructionConfig,
    load_config,
)
from .defaults import default_config_path, export_configs
from .provenance import StageRecord


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _event_number(value):
    value = int(value)
    if value < -1:
        raise argparse.ArgumentTypeError(
            "must be -1 (all events) or a nonnegative event id"
        )
    return value


def _add_options(parser, command):
    parser.add_argument(
        "--config",
        "--config-file",
        "-c",
        dest="config_file",
        type=Path,
        default=None,
        help="Analysis YAML for analyze (optional); reconstruction YAML for fit/run",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        "-o",
        type=Path,
        help="Output directory; for analyze, write plots/tables directly here",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on failure"
    )
    if command == "run":
        parser.add_argument(
            "--analysis-config",
            type=Path,
            default=None,
            help="Independent analysis YAML (default: configs/analysis.yaml)",
        )
    if command in ("fit", "run"):
        parser.add_argument(
            "--max-workers",
            "-w",
            type=_positive_int,
            help="Override reconstruction worker count",
        )
        parser.add_argument(
            "--event-number",
            "-e",
            type=_event_number,
            help="Reconstruct one event; -1 selects all events",
        )
    if command == "analyze":
        parser.add_argument(
            "--input", "-i", type=Path, help="Existing HDF5 reconstruction output"
        )
    else:
        parser.add_argument(
            "--input",
            "-i",
            type=Path,
            help="Input ROOT file or glob (overrides paths.input_root)",
        )
    if command in ("analyze", "run"):
        worker_flags = (
            ("--workers", "--analysis-workers", "-w")
            if command == "analyze"
            else ("--analysis-workers",)
        )
        batch_flags = (
            ("--batches", "--analysis-batches")
            if command == "analyze"
            else ("--analysis-batches",)
        )
        parser.add_argument(
            *worker_flags,
            dest="analysis_workers",
            type=_positive_int,
            help="Analysis processes (default: analysis.workers, 1)",
        )
        parser.add_argument(
            *batch_flags,
            dest="analysis_batches",
            type=_positive_int,
            help="Analysis event batches (default: four per worker)",
        )
        parser.add_argument(
            "--analysis-work-dir",
            type=Path,
            help="Parent directory for temporary analysis columns",
        )
        parser.add_argument(
            "--keep-analysis-columns",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Retain temporary analysis columns",
        )
    if command != "fit":
        parser.add_argument(
            "--analyses",
            "-a",
            nargs="+",
            choices=ANALYSIS_NAMES,
            help="Analyses to run (overrides configuration)",
        )


def _add_optimize_options(parser):
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Optimization YAML (default: configs/optimization.yaml)",
    )
    parser.add_argument("--study-name", help="Override study.name")
    parser.add_argument("--output-dir", type=Path, help="Override study.output_dir")
    parser.add_argument("--n-trials", type=int, help="Number of additional trials")
    parser.add_argument("--n-jobs", type=int, help="Parallel trial jobs")
    parser.add_argument("--population-size", type=int, help="NSGA-II population size")
    parser.add_argument("--sampler-seed", type=int, help="NSGA-II sampler seed")
    parser.add_argument(
        "--trial-workers", type=int, help="Reconstruction workers per trial"
    )
    parser.add_argument(
        "--all-input-files",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use all matching backend inputs (ROOT or MDF) instead of the first file",
    )
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument(
        "--submit-condor",
        action="store_true",
        help="Submit the whole Allen optimization as one Condor GPU job",
    )
    submission.add_argument(
        "--prepare-condor",
        action="store_true",
        help="Write and dry-run validate the Condor job without submitting",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export an existing study without new trials",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on failure"
    )


def _add_toy_options(parser):
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Aggregate toy profile JSON (default: the one shipped with pvfit)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("input/toy_sample.root"),
        help="ROOT file to write; refuses to overwrite (default: input/toy_sample.root)",
    )
    parser.add_argument(
        "--events", type=_positive_int, default=10, help="Events to generate"
    )
    parser.add_argument("--seed", type=int, default=20260922, help="Random seed")
    parser.add_argument(
        "--debug", action="store_true", help="Show tracebacks on failure"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pvfit",
        description="Reconstruct primary vertices and evaluate HDF5 reconstruction results.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"PVFit {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("fit", "Reconstruct vertices from ROOT input"),
        ("analyze", "Analyze existing HDF5 results without running reconstruction"),
        ("run", "Run reconstruction followed by analysis"),
        ("optimize", "Run an NSGA-II parameter study over a reconstruction backend"),
        ("toy", "Generate a synthetic ROOT sample that needs no reference data"),
        (
            "convert-allen",
            "Convert Allen monitoring ROOT output into the PVFit HDF5 layout",
        ),
        ("init-config", "Copy the bundled default YAML files for editing"),
    ):
        command = commands.add_parser(
            name, help=help_text, description=help_text, allow_abbrev=False
        )
        if name == "init-config":
            command.add_argument("--output", type=Path, default=Path("configs"))
        elif name == "optimize":
            _add_optimize_options(command)
        elif name == "toy":
            _add_toy_options(command)
        elif name == "convert-allen":
            _add_convert_allen_options(command)
        else:
            _add_options(command, name)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _prepare(args):
    config_path = args.config_file
    if config_path is None and args.command in ("fit", "run"):
        config_path = default_config_path("reconstruction.yaml")
    config = load_config(config_path) if config_path else {}
    fit_options = analysis_options = effective_reconstruction = None

    if args.command in ("fit", "run"):
        # Command-line values override the YAML; the dataclass validates both.
        overrides = {
            name: value
            for name, value in (
                ("input_root", args.input and str(args.input)),
                ("max_workers", args.max_workers),
                ("event_number", args.event_number),
                ("output_dir", args.output_dir and str(args.output_dir)),
            )
            if value is not None
        }
        cfg = replace(ReconstructionConfig.from_mapping(config), **overrides)
        effective_reconstruction = cfg.to_mapping()
        inputs = sorted(glob.glob(str(Path(cfg.input_root).expanduser())))
        if not inputs:
            raise ValueError(f"No input ROOT files matched: {cfg.input_root}")
        track_input = None
        if cfg.track_input_configuration is not None:
            from .allen import load_track_input

            track_input = load_track_input(inputs, cfg.track_input_configuration)
        fit_options = dict(
            input_files=inputs,
            track_tree_name=cfg.track_tree_name,
            mc_vertex_tree_name=cfg.mc_vertex_tree_name,
            params=cfg.as_params(),
            event_number=cfg.event_number,
            max_workers=cfg.max_workers,
            output_dir=str(Path(cfg.output_dir).expanduser()),
            hdf5_file=cfg.hdf5_file,
            track_input=track_input,
        )

    if args.command in ("analyze", "run"):
        analysis_config_path = (
            (args.analysis_config or default_config_path("analysis.yaml"))
            if args.command == "run"
            else args.config_file
        )
        analysis_config = (
            load_config(analysis_config_path) if args.command == "run" else config
        )
        if analysis_config_path and "analysis" not in analysis_config:
            raise ValueError(
                "Analysis configuration requires an 'analysis' mapping; use a separate analysis YAML"
            )
        settings = AnalysisConfig.from_mapping(analysis_config)
        overrides = {
            name: value
            for name, value in (
                ("workers", args.analysis_workers),
                ("batches", args.analysis_batches),
                (
                    "work_dir",
                    str(args.analysis_work_dir.expanduser())
                    if args.analysis_work_dir
                    else None,
                ),
                ("keep_columns", args.keep_analysis_columns),
            )
            if value is not None
        }
        if args.analyses is not None:
            overrides["analyses"] = tuple(args.analyses)
        settings = replace(settings, **overrides)
        if fit_options:
            input_hdf5 = Path(fit_options["output_dir"]) / fit_options["hdf5_file"]
            analysis_dir = Path(fit_options["output_dir"]) / "analysis"
        else:
            input_hdf5 = args.input or settings.input_hdf5
            if input_hdf5 is None:
                raise ValueError(
                    "analyze requires --input FILE.h5 or paths.input_hdf5 in the analysis YAML"
                )
            input_hdf5 = Path(input_hdf5).expanduser()
            if not input_hdf5.is_file():
                raise ValueError(f"HDF5 input does not exist: {input_hdf5}")
            analysis_dir = args.output_dir or settings.output_dir or "output/analysis"
        analysis_options = dict(
            hdf5_file=str(input_hdf5),
            output_dir=str(Path(analysis_dir).expanduser()),
            settings=settings,
        )
    return fit_options, analysis_options, effective_reconstruction


def _run_fit(options):
    from .fit import run_fit

    return run_fit(**options)


def _run_analysis(options):
    from .analysis import run_analysis

    return run_analysis(**options)


def _run_toy(args):
    import json

    from .toy import default_profile, generate_sample

    profile = (
        json.loads(Path(args.profile).read_text())
        if args.profile
        else default_profile()
    )
    summary = generate_sample(profile, args.output, args.events, args.seed)
    print(json.dumps(summary, indent=2))


def _add_convert_allen_options(parser):
    from .convert_allen import CONFIGURATION_NAME, DEFAULT_TREES
    from .fit.truth_matching import TRUTH_MIN_TRACKS

    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Allen histograms.root file(s); events are keyed by (run, event) across files",
    )
    parser.add_argument("--output", type=Path, required=True, help="HDF5 file to write")
    parser.add_argument(
        "--allen-config",
        type=Path,
        default=None,
        help=f"Allen configuration JSON (default: {CONFIGURATION_NAME} next to each input)",
    )
    parser.add_argument(
        "--allow-missing-configuration",
        action="store_true",
        help=(
            "convert without an Allen configuration; only the histogram binning is "
            "recorded, the other reconstruction parameters stay unknown"
        ),
    )
    parser.add_argument(
        "--n-bins-z",
        type=int,
        default=None,
        help=(
            "z bins of the seeding histogram; required without a configuration, "
            "must agree with it otherwise"
        ),
    )
    parser.add_argument(
        "--n-bins-t",
        type=int,
        default=None,
        help="t bins of the seeding histogram (as --n-bins-z)",
    )
    for key, flag in (
        ("tracks", "--track-tree"),
        ("truth", "--truth-tree"),
        ("histogram", "--histogram-tree"),
        ("peaks", "--peak-tree"),
        ("vertices", "--vertex-tree"),
    ):
        parser.add_argument(flag, dest=f"tree_{key}", default=DEFAULT_TREES[key])
    parser.add_argument(
        "--min-truth-tracks",
        type=int,
        default=TRUTH_MIN_TRACKS,
        help="rows a true vertex needs to enter the denominators (as in pvfit fit)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show the full traceback on errors"
    )


def _run_convert_allen(args):
    from .convert_allen import convert_allen_root

    trees = {
        key: getattr(args, f"tree_{key}")
        for key in ("tracks", "truth", "histogram", "peaks", "vertices")
    }
    return convert_allen_root(
        args.input,
        args.output,
        allen_config=args.allen_config,
        n_bins_z=args.n_bins_z,
        n_bins_t=args.n_bins_t,
        trees=trees,
        min_truth_tracks=args.min_truth_tracks,
        allow_missing_configuration=args.allow_missing_configuration,
    )


def _run_optimization(args):
    from .optimize.run import run_optimization

    overrides = {key: value for key, value in vars(args).items() if key != "config"}
    config = args.config or default_config_path("optimization.yaml")
    if args.submit_condor or args.prepare_condor:
        from .optimize.condor import prepare_submission

        return prepare_submission(config, overrides, submit=args.submit_condor)
    return run_optimization(config, overrides)


def main(argv=None):
    """Run the CLI; return 0 on success, 2 for bad inputs, or 1 on execution failure."""
    args = parse_args(argv)
    invocation = list(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "init-config":
            export_configs(args.output)
            print(f"Configuration templates written to {args.output}")
            return 0
        if args.command == "optimize":
            _run_optimization(args)
            return 0
        if args.command == "toy":
            _run_toy(args)
            return 0
        if args.command == "convert-allen":
            _run_convert_allen(args)
            return 0
        fit_options, analysis_options, reconstruction = _prepare(args)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        if getattr(args, "debug", False):
            raise
        detail = (
            f"Missing configuration key: {exc}"
            if isinstance(exc, KeyError)
            else str(exc)
        )
        print(f"pvfit: error: {detail}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        if getattr(args, "debug", False):
            raise
        print(f"pvfit: error: {exc}", file=sys.stderr)
        return 1
    try:
        if fit_options:
            with StageRecord(
                fit_options["output_dir"],
                "fit",
                reconstruction,
                fit_options["input_files"],
                invocation,
            ) as record:
                output = _run_fit({**fit_options, "provenance_path": str(record.path)})
                record.data["outputs"] = [str(Path(output).resolve())]
            print(f"Fitting complete: {output}")
            if analysis_options:
                analysis_options["hdf5_file"] = str(output)
        if analysis_options:
            output_dir = Path(analysis_options["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            done = output_dir / "analysis.done"
            done.unlink(missing_ok=True)
            settings = replace(
                analysis_options["settings"],
                input_hdf5=analysis_options["hdf5_file"],
                output_dir=str(output_dir),
            )
            with StageRecord(
                output_dir,
                "analyze",
                settings.to_mapping(),
                [analysis_options["hdf5_file"]],
                invocation,
            ) as record:
                execution = _run_analysis(analysis_options)
                if isinstance(execution, dict):
                    record.data["execution"] = execution
                record.data["outputs"] = [str(output_dir.resolve())]
            done.touch()
            print(f"Analysis complete: {output_dir}")
        return 0
    except KeyboardInterrupt:
        print("pvfit: interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if getattr(args, "debug", False):
            raise
        print(f"pvfit: error: {exc} (use --debug for a traceback)", file=sys.stderr)
        return 1
