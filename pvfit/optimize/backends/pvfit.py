"""PVFit's own reconstruction as an optimizer backend."""

import sys
from dataclasses import replace
from pathlib import Path

import yaml

from ...config_loader import ReconstructionConfig
from .base import Backend, run_logged

# The ReconstructionConfig fields a study varies, in export order.
PARAM_COLS = [
    "min_tracks_in_seed",
    "smear_window_nbins",
    "n_bins_z",
    "n_bins_t",
    "max_track_seed_chi2",
    "max_chi2_tukey",
    "min_chi2_dist_merge",
    "max_track_beamline_chi2_xy",
    "max_track_z0_err",
]
INT_PARAMETERS = {"smear_window_nbins", "n_bins_z", "n_bins_t"}


class PVFitBackend(Backend):
    name = "pvfit"

    def validate_config(self, base_cfg):
        ReconstructionConfig.from_mapping(base_cfg)

    def input_pattern(self, base_cfg):
        return ReconstructionConfig.from_mapping(base_cfg).input_root

    def validate_search_space(self, search_space):
        if set(search_space) != set(PARAM_COLS):
            raise ValueError(
                f"search_space must define exactly: {', '.join(PARAM_COLS)}"
            )
        for name, spec in search_space.items():
            expected = "int" if name in INT_PARAMETERS else "float"
            if spec["type"] != expected:
                raise ValueError(f"search_space.{name}.type must be {expected}")

    def derived(self, sampled, base_cfg):
        trial_cfg = replace(ReconstructionConfig.from_mapping(base_cfg), **sampled)
        return {
            "derived_scaled_min_tracks_in_seed": trial_cfg.scaled_min_tracks_in_seed
        }

    def write_config(self, base_cfg, sampled, trial_dir, trial_workers, input_root):
        """Apply the sampled parameters to the base config; write reconstruction.yaml."""
        cfg = replace(
            ReconstructionConfig.from_mapping(base_cfg),
            **sampled,
            output_dir=str(trial_dir),
            input_root=input_root,
            max_workers=trial_workers,
        )
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "reconstruction.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg.to_mapping(), f, default_flow_style=False)
        return config_path

    def reconstruct(self, config_path, trial_dir, root):
        run_logged(
            [sys.executable, "-m", "pvfit", "fit", "--config", config_path],
            trial_dir,
            root,
            "pvfit fit",
        )
        cfg = ReconstructionConfig.from_yaml(config_path)
        return Path(cfg.output_dir) / cfg.hdf5_file
