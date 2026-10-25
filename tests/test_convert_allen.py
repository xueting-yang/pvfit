"""Allen monitoring ROOT files convert into the HDF5 layout the analyses read."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot

from pvfit.allen import (
    load_configuration,
    load_track_input,
    parameters_from_configuration,
)
from pvfit.cli import main
from pvfit.config_loader import AnalysisConfig
from pvfit.convert_allen import convert_allen_root
from pvfit.reconstruction_parameters import read_parameters

Z_MIN, DZ, NZ = -200.0, 0.5, 800
T_MIN, DT, NT = -0.6, 0.025, 56
# The effective Allen sequence as the campaign writes it: float properties
# carry their float32 storage noise.
CONFIG = {
    "velo_kalman_filter": {
        "hit_time_resolution_ns": 0.05000000074505806,
        "block_dim": [256, 1, 1],
    },
    "pv_beamline_histo": {
        "zmin": Z_MIN,
        "zmax": Z_MIN + NZ * DZ,
        "dz": DZ,
        "Nbins_z": NZ,
        "tmin": -0.6000000238418579,
        "tmax": 0.800000011920929,
        "dt": 0.02500000037252903,
        "Nbins_t": NT,
        "smear_window_nbins": 2,
        "maxTrackBLChi2XY": 10.0,
        "pp_maxTrackZ0Err": 1.5,
        "SMOG2_maxTrackZ0Err": 10.0,
        "SMOG2_pp_separation": -334.0,
        "block_dim": [128, 1, 1],
    },  # fmt: skip
    "pv_beamline_peak": {
        "zmin": Z_MIN,
        "dz": DZ,
        "Nbins_z": NZ,
        "tmin": -0.6000000238418579,
        "dt": 0.02500000037252903,
        "Nbins_t": NT,
        "pp_minTracks": 2.5,
        "SMOG2_minTracks": 1.75,
        "SMOG2_pp_separation": -334.0,
    },  # fmt: skip
    "pv_beamline_assign_tracks": {
        "maxTrackSeedChi2XYT": 10.0,
        "pp_min_tracks_per_seed": 4,
        "SMOG2_min_tracks_per_seed": 3,
    },
    "pv_beamline_fitter": {
        "maxChi2Tukey": 12.0,
        "maxFitIter": 20,
        "maxDeltaZConverged": 0.0005000000237487257,
        "maxVertexRho2": 0.09000000357627869,
    },
    "pv_beamline_merge_vertices": {
        "maxChi2Tukey": 12.0,
        "maxFitIter": 20,
        "maxDeltaZConverged": 0.0005000000237487257,
        "minChi2Dist": 25.0,
    },
    "sequence": {"configured_algorithms": []},
}
EXPECTED_PARAMETERS = {
    "z_min": -200.0, "z_max": 200.0, "n_bins_z": 800,
    "t_min": -0.6, "t_max": 0.8, "n_bins_t": 56,
    "smear_window_nbins": 2, "max_track_beamline_chi2_xy": 10.0, "max_track_z0_err": 1.5,
    "min_tracks_in_seed": 2.5, "max_track_seed_chi2": 10.0, "min_tracks_per_seed": 4,
    "max_chi2_tukey": 12.0, "max_fit_iterations": 20, "max_delta_z_converged": 0.0005,
    "max_vertex_rho2": 0.09, "min_chi2_dist_merge": 25.0, "hit_time_resolution_ns": 0.05,
}  # fmt: skip


def write_allen_file(path, runs, events_per_run, seed=3):
    """A small Allen-like file: two PVs per event, tracks on the beamline."""
    rng = np.random.default_rng(seed)
    tracks, truth, histo, peaks, vertices = [], [], [], [], []
    for run in runs:
        for event in range(events_per_run):
            pvs = [(rng.normal(0, 30), rng.normal(0, 0.1)) for _ in range(2)]
            key = 10
            for ipv, (z, t) in enumerate(pvs):
                keys = list(range(key, key + 6))
                key += 6
                for k in keys:
                    tracks.append(
                        dict(
                            run_number=run,
                            event_number=event,
                            mc_key=k,
                            beamPOCA_x=rng.normal(0, 0.01),
                            beamPOCA_y=rng.normal(0, 0.01),
                            beamPOCA_z=z + rng.normal(0, 0.05),
                            beamPOCA_t=t + rng.normal(0, 0.01),
                            beamPOCA_tx=rng.normal(0, 0.05),
                            beamPOCA_ty=rng.normal(0, 0.05),
                            backward=False,
                            c00=1e-4,
                            c11=1e-4,
                            c55=4e-4,
                            chi2=1.0,
                            p=5000.0,
                            pt=500.0,
                            fromPV=True,
                            ovtx_x=0.0,
                            ovtx_y=0.0,
                            ovtx_z=z,
                            ovtx_t=t,
                        )
                    )
                truth.append(
                    dict(
                        run_number=run,
                        event_number=event,
                        effective_pv_key=ipv,
                        x=0.0,
                        y=0.0,
                        z=z,
                        time=t,
                        n_valid_tracks=6,
                        track_keys=np.array(keys, dtype=np.int32),
                    )
                )
                iz, it = int((z - Z_MIN) / DZ), int((t - T_MIN) / DT)
                histo.append(
                    dict(
                        run_number=run,
                        event_number=event,
                        iz=iz,
                        it=it,
                        z_bin=Z_MIN + (iz + 0.5) * DZ,
                        t_bin=T_MIN + (it + 0.5) * DT,
                        weight=6.0,
                    )
                )
                peaks.append(
                    dict(run_number=run, event_number=event, z_peak=z, t_peak=t)
                )
                vertices.append(
                    dict(
                        run_number=run,
                        event_number=event,
                        n_vtx=2,
                        x=0.0,
                        y=0.0,
                        z=z + 0.01 + rng.normal(0, 0.02),
                        t=t + 0.002 + rng.normal(0, 0.003),
                        cov00=1e-6,
                        cov11=1e-6,
                        cov22=1e-4,
                        cov33=1e-6,
                        nTracks=5.5,
                        chi2=3.0,
                        track_mc_keys=np.array(keys[:5], dtype=np.int32),
                    )
                )
    # one track with no truth and one stray key, to exercise the guards
    tracks.append({**tracks[0], "mc_key": -1})

    def columns(rows):
        return {k: ak.Array([r[k] for r in rows]) for k in rows[0]}

    with uproot.recreate(path) as f:
        f["velo_kalman_filter/monitor_tree"] = columns(tracks)
        f["mc_pv_dumper/MCPV_effective_tree"] = columns(truth)
        f["pv_beamline_histo/histo_tree"] = columns(histo)
        f["pv_beamline_peak/peak_tree"] = columns(peaks)
        f["pv_beamline_merge_vertices/merge_vtx_tree"] = columns(vertices)
    return len(tracks), len(truth), len(vertices)


class ConvertAllenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.input = self.root / "histograms.root"
        self.counts = write_allen_file(self.input, runs=[101, 102], events_per_run=3)
        (self.root / "allen_configuration.json").write_text(json.dumps(CONFIG))

    def convert(self, **kwargs):
        output = self.root / "out" / "results.h5"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = convert_allen_root([self.input], output, **kwargs)
        return output, summary

    def test_layout_identity_and_memberships(self):
        output, summary = self.convert()
        self.assertEqual(summary["events"], 6)
        self.assertEqual(summary["allen_configuration"], "present")
        with h5py.File(output) as f:
            self.assertEqual(f.attrs["source"], "Allen")
            # The same reco_* record pvfit fit writes, cleaned of float32 noise
            self.assertEqual(read_parameters(f.attrs), EXPECTED_PARAMETERS)
            self.assertEqual(f.attrs["reco_max_chi2_tukey"], 12.0)
            # Imported provenance keeps the real Allen cut, not PVFit's default.
            self.assertEqual(f.attrs["reco_max_track_seed_chi2"], 10.0)
            self.assertEqual(f.attrs["truth_min_tracks"], 4)
            self.assertEqual(
                f.attrs["allen_configuration_path"],
                str((self.root / "allen_configuration.json").resolve()),
            )
            self.assertEqual(len(f.attrs["allen_configuration_sha256"]), 64)
            for stale in ("allen_configuration", "reconstruction_parameters"):
                self.assertNotIn(stale, f.attrs)
            self.assertEqual(sorted(int(k) for k in f), list(range(6)))
            # (run, event) order: run 101 events 0..2 then run 102
            self.assertEqual(
                [f[str(i)].attrs["run_number"] for i in range(6)], [101] * 3 + [102] * 3
            )
            self.assertEqual(
                [f[str(i)].attrs["source_event_number"] for i in range(6)],
                [0, 1, 2] * 2,
            )
            g = f["0"]
            n = g.attrs["n_tracks"]
            self.assertEqual(n, 13)  # 12 PV tracks + the keyless one
            self.assertEqual(g.attrs["n_true_vertices"], 2)
            self.assertEqual(g.attrs["n_rec_vertices"], 2)
            self.assertEqual(g.attrs["n_seedings"], 2)
            self.assertEqual(g["histogram"].shape, (NZ, NT))
            self.assertAlmostEqual(g["histogram"].attrs["z_min"], Z_MIN)
            self.assertAlmostEqual(
                g["histogram"].attrs["t_max"], T_MIN + NT * DT, places=6
            )
            self.assertAlmostEqual(float(g["histogram"][:].sum()), 12.0)
            true_members = [sorted(idx) for idx in g["true_vertices/track_indices"][:]]
            self.assertEqual(true_members, [list(range(0, 6)), list(range(6, 12))])
            rec_members = [sorted(idx) for idx in g["rec_vertices/track_indices"][:]]
            self.assertEqual(rec_members, [list(range(0, 5)), list(range(6, 11))])
            self.assertTrue(np.allclose(g["rec_vertices/z_err"][:], 1e-2))
            self.assertEqual(g["tracks/mc_key"][-1], -1)
            self.assertEqual(len(g["seeds/z"]), 2)
            self.assertTrue(g["tracks/selected_for_seeding"][:].all())

    def test_analyses_run_on_the_converted_file(self):
        from pvfit.analysis import run_analysis

        output, _ = self.convert()
        settings = AnalysisConfig(
            analyses=(
                "efficiency_track",
                "fake_rate_track",
                "merge_rate",
                "split_rate",
                "resolution",
                "vertex_pull",
            ),
            plot_min_ntracks=4,
            plot_max_ntracks=20,
            plot_bin_width_ntracks=4,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run_analysis(
                hdf5_file=str(output),
                output_dir=str(self.root / "analysis"),
                settings=settings,
            )
        names = {p.name for p in (self.root / "analysis").iterdir()}
        self.assertTrue(any(n.startswith("efficiency_overall") for n in names))
        self.assertTrue(any(n.startswith("fake_rate_overall") for n in names))

    def test_missing_configuration_is_refused_unless_allowed(self):
        (self.root / "allen_configuration.json").unlink()
        with self.assertRaisesRegex(ValueError, "parameters are unknown"):
            with contextlib.redirect_stdout(io.StringIO()):
                convert_allen_root([self.input], self.root / "out.h5")
        with self.assertRaisesRegex(ValueError, "bin counts must be given"):
            with contextlib.redirect_stdout(io.StringIO()):
                convert_allen_root(
                    [self.input], self.root / "out.h5", allow_missing_configuration=True
                )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = convert_allen_root(
                [self.input],
                self.root / "out.h5",
                n_bins_z=NZ,
                n_bins_t=NT,
                allow_missing_configuration=True,
            )
        self.assertEqual(summary["allen_configuration"], "missing")
        self.assertIn("records only the histogram binning", buffer.getvalue())
        with h5py.File(self.root / "out.h5") as f:
            # Only the binning, read back from the dump; nothing assumed
            self.assertEqual(
                read_parameters(f.attrs),
                {k: EXPECTED_PARAMETERS[k] for k in (
                    "z_min", "z_max", "n_bins_z", "t_min", "t_max", "n_bins_t"
                )},
            )  # fmt: skip
            self.assertNotIn("allen_configuration_path", f.attrs)

    def test_configuration_must_be_consistent_with_itself_and_the_dump(self):
        self.assertEqual(parameters_from_configuration(CONFIG), EXPECTED_PARAMETERS)
        peak_disagrees = json.loads(json.dumps(CONFIG))
        peak_disagrees["pv_beamline_peak"]["tmin"] = -0.8
        with self.assertRaisesRegex(ValueError, "t_min: pv_beamline_histo.tmin"):
            parameters_from_configuration(peak_disagrees)
        width_disagrees = json.loads(json.dumps(CONFIG))
        width_disagrees["pv_beamline_histo"]["dz"] = 0.25
        with self.assertRaisesRegex(ValueError, "pv_beamline_histo.dz"):
            parameters_from_configuration(width_disagrees)
        incomplete = json.loads(json.dumps(CONFIG))
        del incomplete["pv_beamline_fitter"]["maxVertexRho2"]
        with self.assertRaisesRegex(
            ValueError, "lacks pv_beamline_fitter.maxVertexRho2"
        ):
            parameters_from_configuration(incomplete)
        # The optional track-input parameter may be absent
        without_tracks = {k: v for k, v in CONFIG.items() if k != "velo_kalman_filter"}
        expected = {
            k: v
            for k, v in EXPECTED_PARAMETERS.items()
            if k != "hit_time_resolution_ns"
        }
        self.assertEqual(parameters_from_configuration(without_tracks), expected)
        # A binning that does not reproduce the dumped bin centres is refused
        shifted = json.loads(json.dumps(CONFIG))
        for algorithm in ("pv_beamline_histo", "pv_beamline_peak"):
            shifted[algorithm]["zmin"] = -199.0
        shifted["pv_beamline_histo"]["zmax"] = 201.0
        (self.root / "allen_configuration.json").write_text(json.dumps(shifted))
        with self.assertRaisesRegex(ValueError, "does not match the dumped histogram"):
            self.convert()
        (self.root / "allen_configuration.json").write_text(json.dumps(CONFIG))
        with self.assertRaisesRegex(ValueError, "contradicts the Allen configuration"):
            self.convert(n_bins_z=400)

    def test_track_input_for_pvfit_fit(self):
        parameters, attributes = load_track_input(
            [self.input], "allen_configuration.json"
        )
        self.assertEqual(parameters, {"hit_time_resolution_ns": 0.05})
        self.assertEqual(
            attributes["track_input_configuration_path"],
            str((self.root / "allen_configuration.json").resolve()),
        )
        self.assertEqual(len(attributes["track_input_configuration_sha256"]), 64)
        with self.assertRaisesRegex(FileNotFoundError, "nope.json"):
            load_track_input([self.input], "nope.json")
        other = self.root / "second" / "histograms.root"
        other.parent.mkdir()
        write_allen_file(other, runs=[201], events_per_run=1)
        (other.parent / "allen_configuration.json").write_text(
            json.dumps(
                {**CONFIG, "velo_kalman_filter": {"hit_time_resolution_ns": 0.1}}
            )
        )
        with self.assertRaisesRegex(ValueError, "differ"):
            load_track_input([self.input, other], "allen_configuration.json")
        no_tracks = self.root / "no_tracks.json"
        no_tracks.write_text(
            json.dumps({k: v for k, v in CONFIG.items() if k != "velo_kalman_filter"})
        )
        with self.assertRaisesRegex(ValueError, "names none of the track-input"):
            load_track_input([self.input], str(no_tracks))

    def test_mixed_or_differing_configurations_are_refused(self):
        other = self.root / "second" / "histograms.root"
        other.parent.mkdir()
        write_allen_file(other, runs=[201], events_per_run=1)
        with self.assertRaisesRegex(ValueError, "no Allen configuration"):
            load_configuration([self.input, other])
        (other.parent / "allen_configuration.json").write_text(
            json.dumps({**CONFIG, "extra": 1})
        )
        with self.assertRaisesRegex(ValueError, "differ"):
            load_configuration([self.input, other])
        (other.parent / "allen_configuration.json").write_text(json.dumps(CONFIG))
        configuration, paths = load_configuration([self.input, other])
        self.assertEqual(configuration, CONFIG)
        self.assertEqual(len(paths), 2)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = convert_allen_root([self.input, other], self.root / "both.h5")
        self.assertEqual(summary["events"], 7)

    def test_duplicate_identity_across_files_is_an_error(self):
        copy = self.root / "copy" / "histograms.root"
        copy.parent.mkdir()
        copy.write_bytes(self.input.read_bytes())
        (copy.parent / "allen_configuration.json").write_text(json.dumps(CONFIG))
        with self.assertRaisesRegex(ValueError, "more than one input file"):
            with contextlib.redirect_stdout(io.StringIO()):
                convert_allen_root([self.input, copy], self.root / "dup.h5")

    def test_wrong_bin_count_is_rejected(self):
        (self.root / "allen_configuration.json").unlink()
        with self.assertRaisesRegex(ValueError, "outside the configured"):
            with contextlib.redirect_stdout(io.StringIO()):
                convert_allen_root(
                    [self.input],
                    self.root / "bad.h5",
                    n_bins_z=100,
                    n_bins_t=NT,
                    allow_missing_configuration=True,
                )

    def test_cli_round_trip(self):
        output = self.root / "cli.h5"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(
                ["convert-allen", "--input", str(self.input), "--output", str(output)]
            )
        self.assertEqual(code, 0)
        self.assertTrue(output.is_file())
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = main(
                [
                    "convert-allen",
                    "--input",
                    str(self.root / "nope.root"),
                    "--output",
                    str(output),
                ]
            )
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
