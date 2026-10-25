import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import uproot
from tqdm import tqdm

from .event_identity import require_run_numbers, unify_event_identity
from .event_processing import process_event, process_event_wrapper
from .save_results_to_hdf5 import save_all_events_to_hdf5
from .truth_matching import read_mcpv_dataframe


def run_fit(
    input_files,
    track_tree_name,
    mc_vertex_tree_name,
    params: dict,
    event_number: int,
    max_workers: int,
    output_dir: str,
    hdf5_file: str,
    *,
    provenance_path: str | None = None,
    track_input=None,
):
    """
    Run the full PVfit reconstruction pipeline and save results to HDF5.

    This function handles:
      - ROOT loading (track data + MC truth vertices)
      - event grouping
      - multiprocessing
      - HDF5 writing

    Args:
        input_files (list[str]): ROOT files
        track_tree_name (str): tree name for track data (monitor_tree)
        mc_vertex_tree_name (str): tree name for MC true vertices (MCPV_effective_tree)
        params (dict): configuration parameters
        event_number (int): event to run (-1 = all)
        max_workers (int): number of workers
        output_dir (str): output directory
        hdf5_file (str): HDF5 file name
        provenance_path (str): CLI stage record, when run from the CLI
        track_input: ``(parameters, attributes)`` of the track input, as
            :func:`pvfit.allen.load_track_input` returns them for
            ``paths.track_input_configuration``; recorded with the result
    Returns:
        str: path to produced HDF5 file
    """

    # ------------------------------------------------------------
    # Load ROOT data
    # ------------------------------------------------------------
    print("\nLoading ROOT files:")
    per_file = []
    total_events = 0

    if not input_files:
        raise ValueError("No input ROOT files matched the configured input_root")
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ROOT file not found: {file_path}")

        print(f"  - {file_path}")
        with uproot.open(file_path) as file:
            # Load track data from monitor_tree
            track_tree = file[track_tree_name]
            required_pv = {"effective_pv_key", "track_keys", "n_valid_tracks"}
            if "mc_key" not in track_tree or not required_pv.issubset(
                file[mc_vertex_tree_name].keys()
            ):
                raise ValueError(
                    "key matching requires monitor_tree.mc_key and "
                    "MCPV_effective_tree with effective_pv_key, track_keys, n_valid_tracks"
                )
            track_df = track_tree.arrays(library="pd")

            # Load MC true vertices from MCPV_effective_tree
            mcpv_tree = file[mc_vertex_tree_name]
            mcpv_df = read_mcpv_dataframe(mcpv_tree)

            # Events are identified by (run_number, event_number)
            require_run_numbers(track_df, mcpv_df)
            per_file.append((track_df, mcpv_df, file_path))

            keys = ["run_number", "event_number"]
            n_evt = track_df.groupby(keys).ngroups
            total_events += n_evt
            print(f"    runs: {track_df['run_number'].nunique()}")
            print(f"    track events: {n_evt}")
            print(f"    MCPV events: {mcpv_df.groupby(keys).ngroups}")

    # One dense id per (run_number, event_number) across all files; the
    # dump's own numbers stay in source_event_number and the identity table.
    track_data, mcpv_data, identity = unify_event_identity(per_file)
    identity = identity.set_index("event_number")

    # ------------------------------------------------------------
    # Group by event
    # ------------------------------------------------------------
    track_grouped = track_data.groupby("event_number")
    mcpv_grouped = mcpv_data.groupby("event_number")
    print(f"\nTotal events available: {len(track_grouped)}")

    # Select events
    events_to_process = []
    for evt_id, evt_track_data in track_grouped:
        if event_number != -1 and evt_id != event_number:
            continue
        # Get corresponding MCPV data for this event
        evt_mcpv_data = (
            mcpv_grouped.get_group(evt_id)
            if evt_id in mcpv_grouped.groups
            else mcpv_data.iloc[:0]
        )
        events_to_process.append((evt_id, evt_track_data, evt_mcpv_data, params))

    if not events_to_process:
        raise RuntimeError("No events selected for processing")

    print(f"Events to process: {len(events_to_process)}")

    # ------------------------------------------------------------
    # Run event processing
    # ------------------------------------------------------------
    fit_results = []

    if len(events_to_process) == 1 or max_workers == 1:
        print("\nRunning in single-thread mode")
        for args in tqdm(events_to_process, desc="Processing events", unit="event"):
            fit_results.append(process_event(*args))
    else:
        print(f"\nRunning with {max_workers} parallel workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_event_wrapper, args): args[0]
                for args in events_to_process
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing events",
                unit="event",
            ):
                evt_id = futures[future]
                try:
                    result = future.result()
                    fit_results.append(result)
                except Exception as e:
                    raise RuntimeError(f"Event {evt_id} failed") from e

    print(f"\nFinished processing {len(fit_results)} events")

    # Carry the original identity of every event into the output
    for result in fit_results:
        origin = identity.loc[result["event_number"]]
        result["run_number"] = int(origin["run_number"])
        result["source_event_number"] = int(origin["source_event_number"])
        result["source_file"] = str(origin["source_file"])

    # Parameters of the track input are recorded, not set, by this fit
    track_input_attributes = None
    if track_input is not None:
        input_parameters, track_input_attributes = track_input
        params = {**params, **input_parameters}

    # ------------------------------------------------------------
    # Save to HDF5
    # ------------------------------------------------------------
    hdf5_path = os.path.join(output_dir, hdf5_file)
    print(f"\nSaving results to HDF5:\n  {hdf5_path}")

    save_all_events_to_hdf5(
        fit_results,
        hdf5_path,
        params,
        provenance_path=provenance_path,
        track_input_attributes=track_input_attributes,
    )

    return hdf5_path
