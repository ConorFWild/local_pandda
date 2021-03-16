# First party
from typing import *
from pathlib import Path

# Third party
import fire

# Custom
from local_pandda.datatypes import (
    Dataset,
    MarkerAffinityResults,
    PanDDAAffinityResults,
    Alignment,
    Params,
    Marker,
)
from local_pandda.functions import (
    iterate_markers,
    get_datasets,
    get_alignments,
    get_reference,
    smooth_datasets,
    print_dataset_summary,
    print_params,
    save_mtz,
    get_markers,
    analyse_residue_gpu,
    make_database,
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
    for marker, marker_datasets in iterate_markers(datasets, markers):
        if params.debug:
            print(f"Processing residue: {marker}")

        marker_results: MarkerAffinityResults = analyse_residue_gpu(
            marker_datasets,
            marker,
            alignments,
            reference_dataset,
            known_apos,
            out_dir,
            params,
        )

        # Update the program log
        pandda_results[marker] = marker_results

    # End loop over residues

    # Merge hits
    make_database(
        datasets,
        pandda_results,
        out_dir / params.database_file
    )

    # Write the summary and graphs of the output
    # write_result_html(pandda_results)


if __name__ == "__main__":
    fire.Fire(run_pandda)
