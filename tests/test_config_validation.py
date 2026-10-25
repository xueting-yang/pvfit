"""Configuration values that would silently distort results are rejected."""

import copy
import math
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from pvfit.config_loader import AnalysisConfig, ReconstructionConfig
from pvfit.reconstruction_parameters import PARAMETER_NAMES, TRACK_INPUT_PARAMETERS

CONFIGS = Path(__file__).parents[1] / "configs"


class ReconstructionConfigValidationTests(unittest.TestCase):
    def setUp(self):
        self.base = yaml.safe_load((CONFIGS / "reconstruction.yaml").read_text())

    def build(self, **algorithm):
        mapping = copy.deepcopy(self.base)
        mapping["algorithm"].update(algorithm)
        return ReconstructionConfig.from_mapping(mapping)

    def test_shipped_configuration_is_valid(self):
        self.build()

    def test_round_trips_and_derives_bin_quantities(self):
        cfg = self.build()
        self.assertEqual(ReconstructionConfig.from_mapping(cfg.to_mapping()), cfg)
        self.assertAlmostEqual(cfg.bin_size_z, 0.5)
        self.assertAlmostEqual(cfg.bin_size_t, 0.025)
        self.assertAlmostEqual(cfg.scaled_min_tracks_in_seed, cfg.min_tracks_in_seed)
        halved = replace(cfg, n_bins_z=400)
        self.assertAlmostEqual(
            halved.scaled_min_tracks_in_seed, 2 * cfg.min_tracks_in_seed
        )
        with self.assertRaisesRegex(ValueError, "n_bins_z must be a positive"):
            replace(cfg, n_bins_z=0)

    def test_shipped_values_are_allens_and_every_parameter_is_recorded(self):
        # The YAML template carries Allen's beamline PV defaults, with the
        # assignment cut and the Tukey threshold as separate keys, so the
        # recorded reco_* attributes of PVFit and Allen outputs are comparable.
        cfg = self.build()
        self.assertEqual(cfg.max_track_seed_chi2, 10.0)
        self.assertEqual(cfg.max_chi2_tukey, 12.0)
        self.assertEqual(cfg.max_vertex_rho2, 0.09)
        self.assertEqual(cfg.min_chi2_dist_merge, 25.0)
        algorithm = cfg.to_mapping()["algorithm"]
        self.assertIn("max_track_seed_chi2", algorithm)
        self.assertIn("max_chi2_tukey", algorithm)
        params = cfg.as_params()
        for name in PARAMETER_NAMES:
            if name not in TRACK_INPUT_PARAMETERS:
                self.assertIn(name, params)
        self.assertIn("scaled_min_tracks_in_seed", params)

    def test_track_input_configuration_is_optional(self):
        self.assertIsNone(self.build().track_input_configuration)
        mapping = copy.deepcopy(self.base)
        mapping["paths"]["track_input_configuration"] = "allen_configuration.json"
        cfg = ReconstructionConfig.from_mapping(mapping)
        self.assertEqual(cfg.track_input_configuration, "allen_configuration.json")
        self.assertEqual(
            cfg.to_mapping()["paths"]["track_input_configuration"],
            "allen_configuration.json",
        )
        mapping["paths"]["track_input_configuration"] = ""
        with self.assertRaisesRegex(ValueError, "paths.track_input_configuration"):
            ReconstructionConfig.from_mapping(mapping)

    def test_analysis_keys_are_rejected_and_never_reach_fit_params(self):
        fit_params = self.build().as_params()
        self.assertNotIn("purity_threshold", fit_params)
        self.assertNotIn("m_r", fit_params)
        with self.assertRaisesRegex(ValueError, "Unknown algorithm keys: m_r"):
            self.build(m_r=0.8)

    def test_non_finite_numbers_are_rejected(self):
        for name in (
            "max_chi2_tukey",
            "max_track_seed_chi2",
            "min_tracks_in_seed",
            "min_chi2_dist_merge",
            "max_delta_z_converged",
        ):
            for value in (math.nan, math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"algorithm.{name}"):
                        self.build(**{name: value})
        with self.assertRaisesRegex(ValueError, "z_min must be below"):
            self.build(z_min=-math.inf)

    def test_cuts_are_disabled_by_their_thresholds_not_flags(self):
        # As in Allen: an infinite threshold keeps every track or vertex,
        # a zero merge distance merges nothing, and there are no flags.
        for name in (
            "max_track_beamline_chi2_xy",
            "max_track_z0_err",
            "max_vertex_rho2",
        ):
            self.assertEqual(getattr(self.build(**{name: math.inf}), name), math.inf)
            with self.assertRaisesRegex(ValueError, f"algorithm.{name}"):
                self.build(**{name: math.nan})
            with self.assertRaisesRegex(ValueError, f"algorithm.{name}"):
                self.build(**{name: 0})
        self.assertEqual(self.build(min_chi2_dist_merge=0).min_chi2_dist_merge, 0)
        self.assertEqual(self.build(min_tracks_per_seed=0).min_tracks_per_seed, 0)
        with self.assertRaisesRegex(ValueError, "min_tracks_per_seed"):
            self.build(min_tracks_per_seed=-1)
        for name in (
            "beamline_chi2_cut",
            "z0_err_cut",
            "merge_vertices",
            "max_chi2",
        ):
            with self.assertRaisesRegex(ValueError, f"Unknown algorithm keys: {name}"):
                self.build(**{name: True})

    def test_flags_must_be_real_booleans(self):
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be true or false"):
                    self.build(is_adaptive_tukey=value)
        self.assertFalse(self.build(is_adaptive_tukey=False).is_adaptive_tukey)


class AnalysisConfigValidationTests(unittest.TestCase):
    def test_execution_settings_are_validated_and_round_trip(self):
        settings = AnalysisConfig(
            workers=2, batches=7, work_dir="scratch", keep_columns=True
        )
        self.assertEqual(AnalysisConfig.from_mapping(settings.to_mapping()), settings)
        self.assertEqual(AnalysisConfig().workers, 1)
        self.assertIsNone(AnalysisConfig().batches)
        for name in ("workers", "batches"):
            for bad in (0, -1, True, 1.5, "2"):
                with (
                    self.subTest(name=name, bad=bad),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    AnalysisConfig(**{name: bad})
        for kwargs in ({"keep_columns": "false"}, {"work_dir": ""}):
            with self.assertRaises(ValueError):
                AnalysisConfig(**kwargs)

    def test_shipped_configuration_is_valid(self):
        settings = AnalysisConfig.from_mapping(
            yaml.safe_load((CONFIGS / "analysis.yaml").read_text())
        )
        self.assertEqual(settings.nsigma, 5.0)
        self.assertIsNone(settings.plot_run_number)
        self.assertIsNone(settings.plot_event_number)

    def test_histogram_identity_requires_both_run_and_event(self):
        for selection in (
            {"plot_run_number": 22001},
            {"plot_event_number": 3},
            {"plot_run_number": 0},
            {"plot_event_number": 0},
        ):
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(
                    ValueError, "must both be set or both be null"
                ):
                    AnalysisConfig.from_mapping({"analysis": selection})

        for run, event in ((0, 0), (22001, 3), (22001, 2**40)):
            with self.subTest(run=run, event=event):
                settings = AnalysisConfig(plot_run_number=run, plot_event_number=event)
                self.assertEqual(
                    AnalysisConfig.from_mapping(settings.to_mapping()), settings
                )
                self.assertNotIn("plot_event_id", settings.to_mapping()["analysis"])

    def test_histogram_identity_rejects_invalid_numbers(self):
        for name in ("plot_run_number", "plot_event_number"):
            for value in (True, False, 1.0, -1, "1", math.nan, math.inf):
                selection = {"plot_run_number": 22001, "plot_event_number": 3}
                selection[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"analysis.{name}"):
                        AnalysisConfig.from_mapping({"analysis": selection})

    def test_histogram_legacy_selection_has_migration_guidance(self):
        with self.assertRaisesRegex(
            ValueError, "plot_event_id has been removed; use plot_run_number and"
        ):
            AnalysisConfig.from_mapping({"analysis": {"plot_event_id": 0}})

    def test_fit_keys_are_rejected_and_track_cut_defaults_to_metadata(self):
        settings = AnalysisConfig.from_mapping({"analysis": {"m_r": 0.5}})
        self.assertEqual(settings.purity_threshold, 0.7)
        self.assertEqual(settings.nsigma, 5.0)
        self.assertEqual(
            AnalysisConfig.from_mapping({"analysis": {"nsigma": 3.0}}).nsigma,
            3.0,
        )
        self.assertIsNone(settings.track_pull_chi2)
        self.assertFalse(hasattr(settings, "max_chi2"))
        with self.assertRaisesRegex(ValueError, "Unknown analysis keys: max_chi2"):
            AnalysisConfig.from_mapping({"analysis": {"max_chi2": 4}})

    def test_thresholds_are_fractions(self):
        for name in ("recall_threshold", "purity_threshold"):
            for value in (1.5, -0.1, math.nan, "0.7", True):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                        AnalysisConfig(**{name: value})
            AnalysisConfig(**{name: 0.0})
            AnalysisConfig(**{name: 1.0})

    def test_scales_and_windows_are_positive_and_finite(self):
        for name in (
            "m_r",
            "nsigma",
            "z_scale",
            "t_scale",
            "track_pull_chi2",
            "x_residual_range",
            "y_residual_range",
            "z_residual_range",
            "t_residual_range",
        ):
            for value in (0, -1, math.nan, math.inf, "1"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        AnalysisConfig(**{name: value})

    def test_flags_and_labels(self):
        for name in ("draw_fixed_ellipses", "draw_fitted_ellipses"):
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                AnalysisConfig(**{name: "false"})
        with self.assertRaisesRegex(ValueError, "method_name"):
            AnalysisConfig(method_name="  ")
        with self.assertRaisesRegex(ValueError, "plot_min_ntracks"):
            AnalysisConfig(plot_min_ntracks=10, plot_max_ntracks=5)


if __name__ == "__main__":
    unittest.main()
