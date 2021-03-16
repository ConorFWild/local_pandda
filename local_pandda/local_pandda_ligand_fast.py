# First party
from typing import *
from pathlib import Path

# Third party
import fire
import numpy as np
import gemmi

# Custom
from local_pandda.datatypes import (
    Dataset,
    DatasetAffinityResults,
    ResidueAffinityResults,
    PanDDAAffinityResults,
    Alignment,
    Params,
    AffinityMaxima,
    AffinityEvent,
    Marker,
)
from local_pandda.functions import (
    get_comparator_datasets,
    get_not_enough_comparator_dataset_affinity_result,
    get_mean,
    get_std,
    get_z,
    iterate_markers,
    get_comparator_samples,
    get_datasets,
    get_truncated_datasets,
    get_alignments,
    sample_datasets,
    get_distance_matrix,
    write_result_html,
    write_event_map,
    get_reference,
    cluster_strong_density,
    smooth_datasets,
    get_fragment_affinity_map,
    get_linkage_from_correlation_matrix,
    get_dataset_apo_mask,
    get_fragment_z_maxima,
    get_background_corrected_density_from_affinity,
    get_backtransformed_map,
    is_affinity_event,
    get_affinity_event_map_path,
    get_affinity_event,
    get_fragment_maps,
    print_dataset_summary,
    print_params,
    get_failed_affinity_event,
    save_mtz,
    get_markers,
analyse_residue,
)


def run_pandda(data_dir: str,
               out_dir: str,
               known_apos: List[str] = None,
               reference_dtag: Optional[str] = None,
               markers: Optional[List[Tuple[float, float, float]]] = None,
               **kwargs):
    # Update the Parameters
    params: Params = Params()
    params.update(**kwargs)
    if params.debug:
        print_params(params)

    # Type the input
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    if params.debug:
        print(
            (
                "Input:\n"
                f"\tdata_dir: {data_dir}\n"
                f"\tout dir: {out_dir}\n"
                f"\tknown apos: {known_apos}\n"
                f"\treference_dtag: {reference_dtag}\n"
                f"\tmarkers: {markers}\n"
            )
        )

    # Load the datasets
    datasets: MutableMapping[str, Dataset] = get_datasets(
        data_dir,
        params.structure_regex,
        params.reflections_regex,
        params.smiles_regex,
        params.pruning_threshold,
    )
    if params.debug:
        print_dataset_summary(datasets)

    if not known_apos:
        known_apos = [dataset.dtag for dtag, dataset in datasets.items() if dataset.fragment_path]
        if params.debug:
            print(f"Got {len(known_apos)} known apos")

    # Get a reference dataset against which to sample things
    reference_dataset: Dataset = get_reference(datasets, reference_dtag, known_apos)
    if params.debug:
        print(f"Reference dataset for alignment is: {reference_dataset.dtag}")

    # B factor smooth the datasets
    smoothed_datasets: MutableMapping[str, Dataset] = smooth_datasets(
        datasets,
        reference_dataset,
        params.structure_factors,
    )
    if params.output_smoothed_mtzs:
        for dtag, smoothed_dataset in smoothed_datasets.items():
            save_mtz(smoothed_dataset.reflections, out_dir / f"{smoothed_dataset.dtag}_smoothed.mtz")

    # Get the markers to sample around
    markers: List[Marker] = get_markers(reference_dataset, markers)

    # Find the alignments between the reference and all other datasets
    alignments: MutableMapping[str, Alignment] = get_alignments(smoothed_datasets, reference_dataset, markers)

    # Loop over the residues, sampling the local electron density
    pandda_results: PanDDAAffinityResults = {}
    for marker, residue_datasets in iterate_markers(datasets, markers):
        if params.debug:
            print(f"Processing residue: {marker}")

        residue_results = analyse_residue_fast(
                residue_datasets,
                marker,
                alignments,
                reference_dataset,
                known_apos,
                out_dir,
                params,
        )


        # Update the program log
        pandda_results[marker.resid] = residue_results

    # End loop over residues

    # Write the summary and graphs of the output
    write_result_html(pandda_results)


if __name__ == "__main__":
    fire.Fire(run_pandda)
