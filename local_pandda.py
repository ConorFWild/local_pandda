# First party
from typing import *
from pathlib import Path

# Third party
import fire
import numpy as np
import gemmi

# Custom
from local_pandda.constants import Constants
from local_pandda.datatypes import (
    Dataset,
    DatasetResults,
    ResidueResults,
    PanDDAResults,
    Alignment,
    Cluster,
    Event,
)
from local_pandda.functions import (
    get_comparator_datasets,
    get_not_enough_comparator_dataset_result,
    is_event,
    get_mean,
    get_std,
    get_z,
    iterate_residues,
    get_apo_mask,
    cluster_z_array,
    get_comparator_samples,
    get_datasets,
    get_truncated_datasets,
    get_alignments,
    sample_datasets,
    get_distance_matrix,
    get_z_clusters,
    get_event_map,
    write_result_html,
    get_background_corrected_density,
    write_event_map,
    get_event,
    get_failed_event,
    get_reference,
    cluster_strong_density,
    smooth_datasets,
    get_event_map_path,
)


def run_pandda(data_dir: str, out_dir: str, known_apos: List[str], reference_dtag: Optional[str], **kwargs):
    # Update the constants
    Constants.update(kwargs)

    # Type the input
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    # Load the datasets
    datasets: MutableMapping[str, Dataset] = get_datasets(data_dir)

    # Get a reference dataset against which to sample things
    reference_dataset: Dataset = get_reference(datasets, reference_dtag, known_apos)

    # B factor smooth the datasets
    smoothed_datasets: MutableMapping[str, Dataset] = smooth_datasets(
        datasets,
        reference_dataset,
        Constants.structure_factors,
    )

    # Find the alignments between the reference and all other datasets
    alignments: MutableMapping[str, Alignment] = get_alignments(smoothed_datasets, reference_dataset)

    # Loop over the residues, sampling the local electron density
    pandda_results: PanDDAResults = PanDDAResults()
    for residue_id, residue_datasets in iterate_residues(datasets):

        # Truncate the datasets to the same reflections
        truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(residue_datasets)

        # Sample the datasets to ndarrays
        sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
            truncated_datasets,
            residue_id,
            alignments,
        )

        # Get the distance matrix
        distance_matrix: np.ndarray = get_distance_matrix(sample_arrays)

        # Cluster the available density
        dataset_clusters: np.ndarray = cluster_strong_density(
            distance_matrix,
            get_apo_mask(
                truncated_datasets,
                known_apos,
            ),
        )

        # For every dataset, find the datasets of the closest known apo cluster
        # If none can be found, make a note of it, and proceed to next dataset
        residue_results: ResidueResults = ResidueResults()
        for dataset_index, dataset in truncated_datasets:
            dataset_results: DatasetResults = DatasetResults(
                dataset.dtag,
                residue_id,
                dataset.structure_path,
                dataset.reflections_path,
                dataset.fragment_path,
            )

            # Get the sample associated with the dataset of interest
            dataset_sample: np.ndarray = sample_arrays[dataset.dtag]

            # Select the comparator datasets
            comparator_datasets: MutableMapping[str, Dataset] = get_comparator_datasets(
                dataset_clusters,
                dataset_index,
                dataset_clusters,
                truncated_datasets,
            )

            # Check if there are enough comparator datasets to characterise a distribution
            if len(comparator_datasets) < Constants.min_comparator_datasets:
                dataset_results: DatasetResults = get_not_enough_comparator_dataset_result(dataset)
                residue_results[dataset.dtag] = dataset_results
                continue

            # Get the local density samples associated with the comparator datasets
            comparator_samples: MutableMapping[str, np.ndarray] = get_comparator_samples(
                sample_arrays,
                comparator_datasets,
            )

            # Characterise the local distribution
            mean: np.nparray = get_mean(comparator_samples)
            std: np.ndarray = get_std(comparator_samples)
            z: np.ndarray = get_z(dataset_sample, mean, std)

            # Extract the clusters
            clusters: MutableMapping[int, Cluster] = get_z_clusters(z, mean)

            # Analyse each cluster
            for cluster_num, cluster in clusters.items():

                # Check each cluster is a valid event
                if is_event(cluster):
                    # Produce the corrected density
                    corrected_density: np.ndarray = get_background_corrected_density(
                        dataset_sample,
                        cluster,
                        mean,
                        z,
                    )

                    # Resample the corrected density onto the original map
                    event_map: gemmi.FloatGrid = get_event_map(
                        corrected_density,
                        reference,
                        dataset,
                        alignments[dataset.dtag][residue_id],
                    )

                    # Write the event map
                    event_map_path: Path = get_event_map_path(
                        out_dir,
                        dataset,
                        cluster_num,
                        residue_id,
                    )
                    write_event_map(
                        event_map,
                        event_map_path,
                    )

                    # Record event
                    event: Event = get_event(
                        dataset,
                        cluster,
                    )

                else:
                    # Record a failed event
                    event: Event = get_failed_event(
                        dataset,
                    )

                # Record the event
                dataset_results.events[cluster_num] = event

            # Record the dataset results
            residue_results[dataset.dtag] = dataset_results

        # Update the program log
        pandda_results[residue_id] = residue_results

    # Write the summary and graphs of the output
    write_result_html(pandda_results)


if __name__ == "__main__":
    fire.Fire(run_pandda)

x = (
    lambda dir: (
        lambda datasets: (
            lambda reference:
                map(
                    process_residues,
                    reference
                )
        )(get_reference(datasets)
    )
    )(get_datasets(dir))
)("dir")
