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
)
from local_pandda.functions import (
    get_comparator_datasets,
    get_not_enough_comparator_dataset_affinity_result,
    get_mean,
    get_std,
    get_z,
    iterate_residues,
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
)


def run_pandda(data_dir: str, out_dir: str, known_apos: List[str], reference_dtag: Optional[str], **kwargs):
    # Update the Parameters
    params: Params = Params()
    params.update(**kwargs)
    if params.debug:
        print_params(params)

    # Type the input
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    # Load the datasets
    datasets: MutableMapping[str, Dataset] = get_datasets(
        data_dir,
        params.structure_regex,
        params.reflections_regex,
        params.smiles_regex,
    )
    if params.debug:
        print_dataset_summary(datasets)

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

    # Find the alignments between the reference and all other datasets
    alignments: MutableMapping[str, Alignment] = get_alignments(smoothed_datasets, reference_dataset)

    # Loop over the residues, sampling the local electron density
    pandda_results: PanDDAAffinityResults = PanDDAAffinityResults()
    for residue_id, residue_datasets in iterate_residues(datasets, reference_dataset):
        # TODO: REMOVE THIS DEBUG CODE
        if params.debug:
            if residue_id.insertion != 1943:
                continue

        if params.debug:
            print(f"Found {len(residue_datasets)} residue datasets")

        # Truncate the datasets to the same reflections
        truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(residue_datasets,
                                                                                  reference_dataset,
                                                                                  params.structure_factors)

        # Truncated dataset apos
        truncated_dataset_apo_mask: np.ndarray = get_dataset_apo_mask(truncated_datasets, known_apos)

        # resolution
        resolution: float = list(truncated_datasets.values())[0].reflections.resolution_high()
        if params.debug:
            print(f"Resolution is: {resolution}")

        # Sample the datasets to ndarrays
        sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
            truncated_datasets,
            residue_id,
            alignments,
            params.structure_factors,
            params.sample_rate,
            params.grid_size,
            params.grid_spacing,
        )

        # Get the distance matrix
        distance_matrix: np.ndarray = get_distance_matrix(sample_arrays)
        if params.debug:
            print(f"First line of distance matrix: {distance_matrix[0, :]}")

        # Get the distance matrix linkage
        linkage: np.ndarray = get_linkage_from_correlation_matrix(distance_matrix)

        # Cluster the available density
        dataset_clusters: np.ndarray = cluster_strong_density(
            linkage,
            params.strong_density_cluster_cutoff,
        )

        # For every dataset, find the datasets of the closest known apo cluster
        # If none can be found, make a note of it, and proceed to next dataset
        residue_results: ResidueAffinityResults = ResidueAffinityResults()
        for dataset_index, dtag in enumerate(truncated_datasets):
            # TODO: REMOVE THIS DEBUG CODE
            if params.debug:
                if dtag != "BAZ2BA-x434":
                    continue

            dataset = truncated_datasets[dtag]

            # Get a result object
            dataset_results: DatasetAffinityResults = DatasetAffinityResults(
                dataset.dtag,
                residue_id,
                dataset.structure_path,
                dataset.reflections_path,
                dataset.fragment_path,
            )

            # Get the fragment
            dataset_fragment_structures: Optional[MutableMapping[str, gemmi.Structure]] = dataset.fragment_structures

            # No need to analyse if no fragment present
            if not dataset_fragment_structures:
                continue

            # Get the sample associated with the dataset of interest
            dataset_sample: np.ndarray = sample_arrays[dataset.dtag]

            # Select the comparator datasets
            comparator_datasets: Optional[MutableMapping[str, Dataset]] = get_comparator_datasets(
                linkage,
                dataset_clusters,
                dataset_index,
                truncated_dataset_apo_mask,
                truncated_datasets,
                params.min_dataset_cluster_size,
            )

            # Handle no comparators
            if not comparator_datasets:
                dataset_results: DatasetAffinityResults = get_not_enough_comparator_dataset_affinity_result(
                    dataset,
                    residue_id,
                )
                residue_results[dataset.dtag] = dataset_results
                continue

            # Get the local density samples associated with the comparator datasets
            comparator_samples: MutableMapping[str, np.ndarray] = get_comparator_samples(
                sample_arrays,
                comparator_datasets,
            )
            if params.debug:
                print(f"Got {len(comparator_samples)} comparater samples")

            # Characterise the local distribution
            mean: np.nparray = get_mean(comparator_samples)
            std: np.ndarray = get_std(comparator_samples)
            z: np.ndarray = get_z(dataset_sample, mean, std)

            # Get the comparator affinity maps
            for fragment_id, fragment_structure in dataset_fragment_structures.items():
                fragment_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = get_fragment_maps(
                    fragment_structure,
                    resolution,
                    params.num_fragment_pose_samples,
                    params.sample_rate,
                )

                # Get affinity maps for various orientations
                fragment_affinity_z_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = {}
                for rotation_index, fragment_map in fragment_maps.items():

                    # Get the affinity map for each dataset at this orientation
                    fragment_affinity_maps: MutableMapping[str, np.ndarray] = {}
                    for comparator_dtag, comparator_sample in comparator_samples.items():
                        affinity_map: np.ndarray = get_fragment_affinity_map(
                            z,
                            fragment_map,
                        )
                        fragment_affinity_maps[comparator_dtag] = affinity_map

                    # Get the current dataset affinity maps
                    dataset_affinity_map = get_fragment_affinity_map(dataset_sample, fragment_map)

                    # Characterise the local distribution of affinity scores
                    fragment_affinity_mean: np.nparray = get_mean(fragment_affinity_maps)
                    fragment_affinity_std: np.ndarray = get_std(fragment_affinity_maps)
                    fragment_affinity_z: np.ndarray = get_z(
                        dataset_affinity_map,
                        fragment_affinity_mean,
                        fragment_affinity_std,
                    )
                    fragment_affinity_z_maps[rotation_index] = fragment_affinity_z
                # End loop over versions of fragment

                # Extract the maxima indexes
                maxima: AffinityMaxima = get_fragment_z_maxima(fragment_affinity_z_maps)

                # Check if the maxima is an event: if so
                if is_affinity_event(maxima, params.min_correlation):

                    # Produce the corrected density by subtracting (1-affinity) * ED mean map
                    corrected_density: np.ndarray = get_background_corrected_density_from_affinity(
                        dataset_sample,
                        maxima,
                        mean,
                    )

                    # Resample the corrected density onto the original map
                    event_map: gemmi.FloatGrid = get_backtransformed_map(
                        corrected_density,
                        reference_dataset,
                        dataset,
                        alignments[dataset.dtag][residue_id],
                        residue_id,
                        params.grid_size,
                        params.grid_spacing,
                        params.structure_factors,
                        params.sample_rate,
                    )

                    # Write the event map
                    event_map_path: Path = get_affinity_event_map_path(
                        out_dir,
                        dataset,
                        residue_id,
                    )
                    write_event_map(
                        event_map,
                        event_map_path,
                    )

                    # Record event
                    event: AffinityEvent = get_affinity_event(
                        dataset,
                        maxima,
                        residue_id,
                    )

                else:
                    # Record a failed event
                    event: AffinityEvent = get_failed_affinity_event(
                        dataset,
                        residue_id,
                    )

                # Record the event
                dataset_results.events[0] = event
            # End loop over fragment builds

            # Record the dataset results
            residue_results[dataset.dtag] = dataset_results
        # End loop over truncated datasets

        # Update the program log
        pandda_results[residue_id] = residue_results
    # End loop over residues

    # Write the summary and graphs of the output
    write_result_html(pandda_results)


if __name__ == "__main__":
    fire.Fire(run_pandda)
