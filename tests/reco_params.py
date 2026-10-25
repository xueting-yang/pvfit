"""The shipped reconstruction parameters, for tests that write HDF5 directly."""

from dataclasses import replace
from pathlib import Path

from pvfit.config_loader import ReconstructionConfig

CONFIGS = Path(__file__).parents[1] / "configs"


def shipped_config(**overrides):
    return replace(
        ReconstructionConfig.from_yaml(CONFIGS / "reconstruction.yaml"), **overrides
    )


def default_params(**overrides):
    """ReconstructionConfig.as_params() of configs/reconstruction.yaml, with
    algorithm fields overridden."""
    return shipped_config(**overrides).as_params()
