"""Resolve the input-track residual cut from analysis or reconstruction metadata."""

from numbers import Real
from pathlib import Path

import h5py
import numpy as np

from ..config_loader import load_config
from ..reconstruction_parameters import attribute_name


def resolve_track_pull_chi2(hdf5_path, override=None):
    """The chi2 cut: an explicit analysis override, else the reconstruction's
    recorded Tukey threshold (root attribute ``reco_max_chi2_tukey``), else the
    ``algorithm.max_chi2_tukey`` of a ``reconstruction.yaml`` beside the file."""
    value = override
    source = "explicit analysis override"
    if value is None:
        with h5py.File(hdf5_path, "r") as f:
            value = f.attrs.get(attribute_name("max_chi2_tukey"))
        source = f"HDF5 attribute {attribute_name('max_chi2_tukey')}"
        if value is None:
            snapshot = Path(hdf5_path).expanduser().with_name("reconstruction.yaml")
            if not snapshot.is_file():
                raise ValueError(
                    "Cannot determine the reconstruction's Tukey threshold: the HDF5 "
                    f"has no {attribute_name('max_chi2_tukey')} attribute and no "
                    "reconstruction.yaml exists beside it. Supply the original "
                    "reconstruction.yaml snapshot or set analysis.track_pull_chi2 "
                    "explicitly."
                )
            algorithm = load_config(snapshot).get("algorithm", {})
            value = (
                algorithm.get("max_chi2_tukey") if isinstance(algorithm, dict) else None
            )
            source = f"{snapshot}: algorithm.max_chi2_tukey"
    if not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{source} must specify a positive finite max_chi2")
    return float(value), source
