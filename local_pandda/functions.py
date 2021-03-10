import os
from typing import *
from local_pandda.datatypes import *
from pathlib import Path
import re
import itertools
import dataclasses

# 3rd party
import numpy as np
import scipy
from scipy import spatial as spsp, ndimage as spn, signal as spsi, cluster as spc
import pandas as pd
import gemmi
from sklearn import neighbors
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib

# Custom
from local_pandda.constants import Constants


#
# def cell_to_python(unit_cell: gemmi.UnitCell, tmp_dir: Path = Path("./tmp")) -> PyCell:
#
#     return PyCell()
#
# def python_to_cell(PyCell) -> gemmi.UnitCell:
#
#
# def spacegroup_to_python(gemmi.Spacegroup) -> PySpacegroup:
#
# def python_to_spacegroup(PySpacegroup) -> gemmi.Spacegroup:
#
#
#
# def mtz_to_python(gemmi.Mtz) -> PyMtz:
#
#
# def python_to_mtz(PyMtz) -> gemmi.Mtz:
#
#
# def structure_to_python(gemmi.Structure) -> PyStructure:
#
# def python_to_structure(PyStructure) -> gemmi.Structure:


def try_make(path: Path):
    if not path.exists():
        os.mkdir(str(path))


def mtz_to_path(mtz: gemmi.Mtz, out_dir: Path) -> Path:
    return None


def python_to_mtz(path: Path) -> gemmi.Mtz:
    return None


def structure_to_python(structure: gemmi.Structure, out_dir: Path) -> Path:
    return None


def python_to_structure(path: Path) -> gemmi.Structure:
    return None


def print_dataset_summary(datasets: MutableMapping[str, Dataset]):
    string = "Dataset summary\n"
    string = string + f"\tNumber of datasets: {len(datasets)}"
    string = string + "\tNumber of datasets with smiles: {}".format(
        len([dtag for dtag in datasets if datasets[dtag].fragment_path])
    )

    print(string)


def print_params(params: Params):
    params_string = "program parameters\n"
    for key, value in Params.__dict__.items():
        params_string = params_string + f"\t{key}: {value}\n"
    print(params_string)


def print_dataclass(dc: dataclass, title: str = ""):
    if title:
        string = f"{title}\n"

    else:
        string = f"{dc.__name__}\n"

    for field in dataclasses.fields(dc):
        field_value = dc.__dict__[field.name]
        string = string + f"\t{field.name}: {field.type} = {field_value}"

    print(string)


def get_dataset_apo_mask(truncated_datasets: MutableMapping[str, Dataset], known_apos: List[str]) -> np.ndarray:
    apo_mask: List[bool] = []

    for dtag in truncated_datasets:
        if dtag in known_apos:
            apo_mask.append(True)
        else:
            apo_mask.append(False)

    apo_mask_array = np.full(len(apo_mask), False)

    for i, val in enumerate(apo_mask):
        apo_mask_array[i] = val

    return apo_mask_array


def get_structures_from_mol(mol: Chem.Mol) -> MutableMapping[int, gemmi.Structure]:
    fragment_structures: MutableMapping[int, gemmi.Structure] = {}
    for i, conformer in enumerate(mol.GetConformers()):
        positions: np.ndarray = conformer.GetPositions()

        structure: gemmi.Structure = gemmi.Structure()
        model: gemmi.Model = gemmi.Model(f"{i}")
        chain: gemmi.Chain = gemmi.Chain(f"{i}")
        residue: gemmi.Residue = gemmi.Residue()

        # Get the sequence id
        # seqid: gemmi.SeqId = gemmi.SeqId(j, ' ')
        # gemmi_atom.seqid = seqid
        # gemmi_atom.seqid = seqid

        # Loop over atoms, adding them to a gemmi residue
        for j, atom in enumerate(mol.GetAtoms()):
            # Get the atomic symbol
            atom_symbol: str = atom.GetSymbol()
            gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)

            # Get the position as a gemmi type
            pos: np.ndarray = positions[j, :]
            gemmi_pos: gemmi.Position = gemmi.Position(pos[0], pos[1], pos[2])

            # Get the
            gemmi_atom: gemmi.Atom = gemmi.Atom()
            gemmi_atom.name = atom_symbol
            gemmi_atom.pos = gemmi_pos
            gemmi_atom.element = gemmi_element

            # Add atom to residue
            residue.add_atom(gemmi_atom)

        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)

        fragment_structures[i] = structure

    return fragment_structures


def get_fragment_structures(smiles_path: Path, pruning_threshold: float) -> MutableMapping[int, Chem.Mol]:
    # Get smiels string
    with open(str(smiles_path), "r") as f:
        smiles_string: str = str(f.read())

    # Load the mol
    m: Chem.Mol = Chem.MolFromSmiles(smiles_string)

    # Generate conformers
    m2: Chem.Mol = Chem.AddHs(m)

    # Generate conformers
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=50, pruneRmsThresh=pruning_threshold)

    # Translate to structures
    fragment_structures: MutableMapping[int, gemmi.Structure] = get_structures_from_mol(m2)

    return fragment_structures


def get_fragment_affinity_map(dataset_sample: np.ndarray, fragment_map: np.ndarray) -> np.ndarray:
    # Convolve the fragment map with the sample
    affinity_map: np.ndarray = spsi.correlate(dataset_sample, fragment_map)

    return affinity_map


def get_fragment_map(structure: gemmi.Structure, resolution: float) -> np.ndarray:
    dencalc: gemmi.DensityCalculatorE = gemmi.DensityCalculatorE()

    dencalc.d_min = resolution
    dencalc.rate = 3.0

    dencalc.set_grid_cell_and_spacegroup(structure)
    dencalc.put_model_density_on_grid(structure[0])

    grid: gemmi.FloatGrid = dencalc.grid
    array: np.ndarray = np.array(grid, copy=True)

    return array


def rotate_translate_structure(fragment_structure: gemmi.Structure, rotation_matrix,
                               margin: float = 1.5) -> gemmi.Structure:
    structure_copy = fragment_structure.clone()
    transform: gemmi.Transform = gemmi.Transform()
    transform.mat.fromlist(rotation_matrix.tolist())
    transform.vec.fromlist([0.0, 0.0, 0.0])

    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    rotated_vec = transform.apply(pos)
                    rotated_position = gemmi.Position(rotated_vec.x, rotated_vec.y, rotated_vec.z)
                    atom.pos = rotated_position

    box = structure_copy.calculate_box()
    box.add_margin(margin)
    min_pos: gemmi.Position = box.min

    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    new_x = pos.x - min_pos.x
                    new_y = pos.y - min_pos.y
                    new_z = pos.z - min_pos.z
                    atom.pos = gemmi.Position(new_x, new_y, new_z)

    return structure_copy


def get_fragment_maps(fragment_structure: gemmi.Structure, resolution: float, num_samples: int, sample_rate: float):
    sample_angles = np.linspace(0, 360, num=10, endpoint=False).tolist()

    fragment_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = {}
    for x, y, z in itertools.product(sample_angles, sample_angles, sample_angles):
        rotation_index = (x, y, z)
        rotation = spsp.transform.Rotation.from_euler("xyz", [x, y, x], degrees=True)
        rotation_matrix: np.ndarray = rotation.as_matrix()
        rotated_structure: gemmi.Structure = rotate_translate_structure(fragment_structure, rotation_matrix)
        fragment_map: np.ndarray = get_fragment_map(rotated_structure, resolution)
        fragment_maps[rotation_index] = fragment_map

    return fragment_maps


def get_residue_id(model: gemmi.Model, chain: gemmi.Chain, insertion: str):
    return ResidueID(model.name, chain.name, str(insertion))


def get_residue(structure: gemmi.Structure, residue_id: ResidueID) -> gemmi.Residue:
    return structure[residue_id.model][residue_id.chain][residue_id.insertion][0]


def get_comparator_datasets(
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        apo_mask: np.ndarray,
        datasets: MutableMapping[str, Dataset],
        min_cluster_size: int,
) -> Optional[MutableMapping[str, Dataset]]:
    #
    apo_cluster_indexes: np.ndarray = np.unique(dataset_clusters[apo_mask])

    apo_clusters: MutableMapping[int, np.ndarray] = {}
    for apo_cluster_index in apo_cluster_indexes:
        cluster: np.ndarray = dataset_clusters[dataset_clusters == apo_cluster_index]

        #
        if cluster.size > min_cluster_size:
            apo_clusters[apo_cluster_index] = cluster

    if len(apo_clusters) == 0:
        return None

    # Get the cophenetic distances
    cophenetic_distances_reduced: np.ndarray = scipy.cluster.hierarchy.cophenet(linkage)
    cophenetic_distances: np.ndarray = spsp.distance.squareform(cophenetic_distances_reduced)
    distances_from_dataset: np.ndarray = cophenetic_distances[dataset_index, :]

    # Find the closest apo cluster
    cluster_distances: MutableMapping[int, float] = {}
    for apo_cluster_index, apo_cluster in apo_clusters.items():
        distances_from_cluster: np.ndarray = distances_from_dataset[dataset_clusters == apo_cluster_index]
        mean_distance: float = np.mean(distances_from_cluster)
        cluster_distances[apo_cluster_index] = mean_distance

    # Find closest n datasets in cluster
    closest_cluster_index: int = min(cluster_distances, key=lambda x: cluster_distances[x])
    closest_cluster_dtag_array: np.ndarray = np.array(list(datasets.keys()))[dataset_clusters == closest_cluster_index]

    closest_cluster_datasets: MutableMapping[str, Dataset] = {dtag: datasets[dtag]
                                                              for dtag
                                                              in closest_cluster_dtag_array
                                                              }

    return closest_cluster_datasets


def get_not_enough_comparator_dataset_result(dataset: Dataset, residue_id: ResidueID) -> DatasetResults:
    dataset_result: DatasetResults = DatasetResults(
        dataset.dtag,
        residue_id,
        structure_path=dataset.structure_path,
        reflections_path=dataset.reflections_path,
        fragment_path=dataset.fragment_path,
        events={},
        comparators=[],
    )

    return dataset_result


def is_event(cluster: Cluster, min_z_cluster_size: int) -> bool:
    # Check if the density is large enough to be considered an event
    if cluster.size() > min_z_cluster_size:
        return True
    else:
        return False


def get_mean(comparator_samples: MutableMapping[str, np.ndarray]) -> np.ndarray:
    # Get all the samples as an array
    samples = [sample for dtag, sample in comparator_samples.items()]
    samples_array = np.stack(samples, axis=0)

    # Get the mean
    mean = np.mean(samples_array, axis=0)

    return mean


def get_std(comparator_samples: MutableMapping[str, np.ndarray]) -> np.ndarray:
    # Get all the samples as an array
    samples = [sample for dtag, sample in comparator_samples.items()]
    samples_array = np.stack(samples, axis=0)

    # Get the standard deviation
    std = np.std(samples_array, axis=0)

    return std


def get_z(dataset_sample: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # Calculate the z value as (dataset_sample - mean) / std
    z: np.ndarray = (dataset_sample - mean) / std

    return z


def iterate_residues(
        datasets: MutableMapping[str, Dataset],
        reference: Dataset,
        debug: bool = True,
) -> Iterator[Tuple[ResidueID, MutableMapping[str, Dataset]]]:
    # Get all unique ResidueIDs from all datasets
    # Order them: Sort by model, then chain, then residue insertion
    # yield them

    reference_structure: gemmi.Structure = reference.structure

    for model in reference_structure:
        for chain in model:
            for residue in chain.get_polymer():
                residue_id: ResidueID = ResidueID(model.name, chain.name, str(residue.seqid.num))

                residue_datasets: MutableMapping[str, Dataset] = {}
                for dtag, dataset in datasets.items():
                    structure: gemmi.Structure = dataset.structure

                    try:
                        res = get_residue(structure, residue_id)
                        res_ca = res["CA"][0]
                        residue_datasets[dtag] = dataset
                    except Exception as e:
                        if debug:
                            print(e)
                        continue

                yield residue_id, residue_datasets


def iterate_markers(
        datasets: MutableMapping[str, Dataset],
        markers: List[Marker],
        debug: bool = True,
) -> Iterator[Tuple[Marker, MutableMapping[str, Dataset]]]:
    for marker in markers:
        yield marker, datasets


def get_apo_mask(
        truncated_datasets: MutableMapping[str, Dataset],
        known_apos: List[str],
) -> np.ndarray:
    # Make a dummy mask
    # Iterate over the truncated datasets
    # If they are in known apos, mask them
    return None


def cluster_z_array(z: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return None


def get_comparator_samples(
        sample_arrays: MutableMapping[str, np.ndarray],
        comparator_datasets: MutableMapping[str, Dataset],
) -> MutableMapping[str, np.ndarray]:
    comparator_samples: MutableMapping[str, np.ndarray] = {}
    for dtag in comparator_datasets:
        comparator_samples[dtag] = sample_arrays[dtag]

    return comparator_samples


def get_path_from_regex(directory: Path, regex: str) -> Optional[Path]:
    for path in directory.glob(f"{regex}"):
        return path

    else:
        return None


def get_structure(structure_path: Path) -> gemmi.Structure:
    structure: gemmi.Structure = gemmi.read_structure(str(structure_path))
    structure.setup_entities()
    return structure


def get_reflections(reflections_path: Path) -> gemmi.Mtz:
    reflections: gemmi.Mtz = gemmi.read_mtz_file(str(reflections_path))
    return reflections


def get_dataset_from_dir(
        directory: Path,
        structure_regex: str,
        reflections_regex: str,
        smiles_regex: str,
        pruning_threshold: float,
        debug: bool = True,
) -> Optional[Dataset]:
    if debug:
        print(f"\tChecking directoy {directory} for data...")

    if directory.is_dir():
        if debug:
            print(
                f"\t\t{directory} is a directory. Checking for regexes: {structure_regex}, {reflections_regex} and {smiles_regex}")
        dtag = directory.name
        structure_path: Optional[Path] = get_path_from_regex(directory, structure_regex)
        reflections_path: Optional[Path] = get_path_from_regex(directory, reflections_regex)
        smiles_path: Optional[Path] = get_path_from_regex(directory, smiles_regex)

        if structure_path and reflections_path:

            if smiles_path:
                fragment_structures: Optional[MutableMapping[int, Chem.Mol]] = get_fragment_structures(
                    smiles_path,
                    pruning_threshold,
                )
                if debug:
                    print(
                        f"\t\tGenerated {len(fragment_structures)} after pruning")

            else:
                fragment_structures = None

            dataset: Dataset = Dataset(
                dtag=dtag,
                structure=get_structure(structure_path),
                reflections=get_reflections(reflections_path),
                structure_path=structure_path,
                reflections_path=reflections_path,
                fragment_path=smiles_path,
                fragment_structures=fragment_structures,
            )

            return dataset

        else:
            if debug:
                print(f"\t\t{directory} Lacks either a structure or reflections. Skipping")
            return None
    else:
        return None


def get_datasets(
        data_dir: Path,
        structure_regex: str,
        reflections_regex: str,
        smiles_regex: str,
        pruning_threshold: float,
        debug: bool = True,
) -> MutableMapping[str, Dataset]:
    # Iterate over the paths
    directories = list(data_dir.glob("*"))

    datasets_list: List[Optional[Dataset]] = joblib.Parallel(
        n_jobs=20,
        verbose=50,
        # backend="multiprocessing",
    )(
        joblib.delayed(
            get_dataset_from_dir)(
            directory,
            structure_regex,
            reflections_regex,
            smiles_regex,
            pruning_threshold,
            debug,
        )
        for directory
        in directories

    )

    datasets: MutableMapping[str, Dataset] = {dataset.dtag: dataset for dataset in datasets_list if dataset is not None}

    return datasets


def truncate_resolution(reflections: gemmi.Mtz, resolution: float) -> gemmi.Mtz:
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = reflections.spacegroup
    new_reflections.set_cell_for_all(reflections.cell)

    # Add dataset
    new_reflections.add_dataset("truncated")

    # Add columns
    for column in reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Get data
    data_array = np.array(reflections, copy=True)
    data = pd.DataFrame(data_array,
                        columns=reflections.column_labels(),
                        )
    data.set_index(["H", "K", "L"], inplace=True)

    # add resolutions
    data["res"] = reflections.make_d_array()

    # Truncate by resolution
    data_truncated = data[data["res"] >= resolution]

    # Rem,ove res colum
    data_dropped = data_truncated.drop("res", "columns")

    # To numpy
    data_dropped_array = data_dropped.to_numpy()

    # new data
    new_data = np.hstack([data_dropped.index.to_frame().to_numpy(),
                          data_dropped_array,
                          ]
                         )

    # Update
    new_reflections.set_data(new_data)

    # Update resolution
    new_reflections.update_reso()

    return new_reflections


def get_truncated_datasets(datasets: MutableMapping[str, Dataset],
                           reference_dataset: Dataset,
                           structure_factors: StructureFactors) -> MutableMapping[str, Dataset]:
    resolution_truncated_datasets = {}

    # Get the lowest common resolution
    resolution: float = min([dataset.reflections.resolution_high() for dtag, dataset in datasets.items()])

    # Truncate by common resolution
    for dtag, dataset in datasets.items():
        dataset_reflections: gemmi.Mtz = dataset.reflections
        truncated_reflections: gemmi.Mtz = truncate_resolution(dataset_reflections, resolution)
        truncated_dataset: Dataset = Dataset(dataset.dtag,
                                             dataset.structure,
                                             truncated_reflections,
                                             dataset.structure_path,
                                             dataset.reflections_path,
                                             dataset.fragment_path,
                                             dataset.fragment_structures,
                                             dataset.smoothing_factor,
                                             )

        resolution_truncated_datasets[dtag] = truncated_dataset

    # Get common set of reflections
    common_reflections = get_all_common_reflections(resolution_truncated_datasets, structure_factors)

    # truncate on reflections
    new_datasets_reflections: MutableMapping[str, Dataset] = {}
    for dtag in resolution_truncated_datasets:
        resolution_truncated_dataset: Dataset = resolution_truncated_datasets[dtag]
        reflections = resolution_truncated_dataset.reflections
        reflections_array = np.array(reflections)
        print(f"{dtag}")
        print(f"{reflections_array.shape}")

        truncated_reflections: gemmi.Mtz = truncate_reflections(
            reflections,
            common_reflections,
        )

        reflections_array = np.array(truncated_reflections)
        print(f"{dtag}: {reflections_array.shape}")

        new_dataset: Dataset = Dataset(
            resolution_truncated_dataset.dtag,
            resolution_truncated_dataset.structure,
            truncated_reflections,
            resolution_truncated_dataset.structure_path,
            resolution_truncated_dataset.reflections_path,
            resolution_truncated_dataset.fragment_path,
            resolution_truncated_dataset.fragment_structures,
            resolution_truncated_dataset.smoothing_factor,
        )

        new_datasets_reflections[dtag] = new_dataset

    return new_datasets_reflections


def transform_from_translation_rotation(translation, rotation):
    transform = gemmi.Transform()
    transform.vec.fromlist(translation.tolist())
    transform.mat.fromlist(rotation.as_matrix().tolist())

    return Transform(transform)


def get_transform_from_atoms(
        moving_selection,
        reference_selection,
) -> Transform:
    """
    Get the transform FROM the moving TO the reference
    :param moving_selection:
    :param reference_selection:
    :return:
    """

    # Get the means
    mean_moving = np.mean(moving_selection, axis=0)
    mean_reference = np.mean(reference_selection, axis=0)

    # Het the transation FROM the moving TO the reference
    vec = np.array(mean_reference - mean_moving)

    de_meaned_moving = moving_selection - mean_moving
    de_meaned_referecnce = reference_selection - mean_reference

    rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned_moving, de_meaned_referecnce)

    return transform_from_translation_rotation(vec, rotation)


def get_markers(
        reference_dataset: Dataset,
        markers: Optional[List[Tuple[float, float, float]]],
        debug: bool = True,
) -> List[Marker]:
    new_markers: List[Marker] = []

    if markers:
        for marker in markers:
            new_markers.append(
                Marker(
                    marker[0],
                    marker[1],
                    marker[2],
                    None,
                )
            )
        return new_markers

    else:
        for model in reference_dataset.structure:
            for chain in model:
                for ref_res in chain.get_polymer():
                    print(f"\t\tGetting transform for residue: {ref_res}")

                    # if ref_res.name.upper() not in Constants.residue_names:
                    #     continue
                    try:

                        # Get ca pos from reference
                        current_res_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
                        reference_ca_pos = ref_res["CA"][0].pos
                        new_markers.append(
                            Marker(
                                reference_ca_pos.x,
                                reference_ca_pos.y,
                                reference_ca_pos.z,
                                current_res_id,
                            )
                        )

                    except Exception as e:
                        if debug:
                            print(f"\t\tAlignment exception: {e}")
                        continue

        if debug:
            print(f"Found {len(new_markers)}: {new_markers}")

        return new_markers


def get_alignment(
        reference: Dataset,
        dataset: Dataset,
        markers: List[Marker],
        debug: bool = True,
) -> Alignment:
    # Find the common atoms as an array
    dataset_pos_list = []
    reference_pos_list = []
    for model in reference.structure:
        for chain in model:
            for ref_res in chain.get_polymer():
                # if ref_res.name.upper() not in Constants.residue_names:
                #     continue
                try:

                    # Get ca pos from reference
                    print("Getting ref ca")
                    current_res_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
                    print(type(ref_res))
                    print(ref_res)
                    reference_ca_pos = ref_res["CA"][0].pos

                    print("Getting dataset ca")
                    # Get the ca pos from the dataset
                    dataset_res = get_residue(dataset.structure, current_res_id)
                    print(type(dataset_res))
                    print(dataset_res)
                    dataset_ca_pos = dataset_res["CA"][0].pos
                except Exception as e:
                    if debug:
                        print(f"\t\tAlignment exception: {e}")
                    continue

                residue_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)

                dataset_res: gemmi.Residue = get_residue(dataset.structure, residue_id)

                for atom_ref, atom_dataset in zip(ref_res, dataset_res):
                    dataset_pos_list.append([atom_dataset.pos.x, atom_dataset.pos.y, atom_dataset.pos.z, ])
                    reference_pos_list.append([atom_ref.pos.x, atom_ref.pos.y, atom_ref.pos.z, ])
    dataset_atom_array = np.array(dataset_pos_list)
    reference_atom_array = np.array(reference_pos_list)

    if debug:
        print(f"\t\tdataset atom array size: {dataset_atom_array.shape}")
        print(f"\t\treference atom array size: {reference_atom_array.shape}")

    # dataset kdtree
    # dataset_tree = spsp.KDTree(dataset_atom_array)
    reference_tree = spsp.KDTree(reference_atom_array)

    # Get the transform for each
    # alignment: Alignment = {}
    # for model in reference.structure:
    #     for chain in model:
    #         for ref_res in chain.get_polymer():
    #             print(f"\t\tGetting transform for residue: {ref_res}")
    #
    #             # if ref_res.name.upper() not in Constants.residue_names:
    #             #     continue
    #             try:
    #
    #                 # Get ca pos from reference
    #                 current_res_id: ResidueID = get_residue_id(model, chain, ref_res.seqid.num)
    #                 reference_ca_pos = ref_res["CA"][0].pos
    #
    #                 # Get the ca pos from the dataset
    #                 dataset_res = get_residue(dataset.structure, current_res_id)
    #                 dataset_ca_pos = dataset_res["CA"][0].pos
    #             except Exception as e:
    #                 if debug:
    #                     print(f"\t\tAlignment exception: {e}")
    #                 continue
    #
    #             # dataset selection
    #             if debug:
    #                 print("\t\tQuerying")
    #
    #             dataset_indexes = dataset_tree.query_ball_point(
    #                 [dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z],
    #                 7.0,
    #             )
    #             # dataset_indexes = reference_tree.query_ball_point(
    #             #     [dataset_ca_pos.x, dataset_ca_pos.y, dataset_ca_pos.z],
    #             #     7.0,
    #             # )
    #             dataset_selection = dataset_atom_array[dataset_indexes]
    #
    #             # Reference selection
    #             reference_selection = reference_atom_array[dataset_indexes]
    #
    #             # Get transform
    #             if debug:
    #                 print("\t\tGetting transform")
    #             alignment[current_res_id] = get_transform_from_atoms(
    #                 dataset_selection,
    #                 reference_selection,
    #             )
    #             if debug:
    #                 print(
    #                     (
    #                         f"\t\t\tTransform is:\n"
    #                         f"\t\t\t\tMat: {alignment[current_res_id].transform.mat}\n"
    #                         f"\t\t\t\tVec: {alignment[current_res_id].transform.vec}\n"
    #                     )
    #                 )
    alignment: Alignment = {}
    for marker in markers:
        # dataset selection
        if debug:
            print("\t\tQuerying")

        reference_indexes = reference_tree.query_ball_point(
            [marker.x, marker.y, marker.z],
            7.0,
        )
        dataset_selection = dataset_atom_array[reference_indexes]

        # Reference selection
        reference_selection = reference_atom_array[reference_indexes]

        # Get transform
        if debug:
            print("\t\tGetting transform")
        alignment[marker] = get_transform_from_atoms(
            dataset_selection,
            reference_selection,
        )
        if debug:
            print(
                (
                    f"\t\t\tTransform is:\n"
                    f"\t\t\t\tMat: {alignment[marker].transform.mat}\n"
                    f"\t\t\t\tVec: {alignment[marker].transform.vec}\n"
                )
            )

    if debug:
        print("Returning alignment...")
    return alignment


def get_alignments(
        datasets: MutableMapping[str, Dataset],
        reference: Dataset,
        markers: List[Marker],
        debug: bool = True,
) -> MutableMapping[str, Alignment]:
    alignment_list: List[Alignment] = joblib.Parallel(
        n_jobs=20,
        verbose=50,
        # backend="multiprocessing",
    )(
        joblib.delayed(get_alignment)(
            reference,
            dataset,
            markers,
        )
        for dataset
        in list(datasets.values())
    )

    alignments: MutableMapping[str, Alignment] = {
        dtag: alignment
        for dtag, alignment
        in zip(list(datasets.keys()), alignment_list)
    }

    # alignments = {}
    # for dtag, dataset in datasets.items():
    #     if debug:
    #         print(f"\tAligning {dtag} against reference {reference.dtag}")
    #     alignment: Alignment = get_alignment(reference, dataset)
    #     alignments[dtag] = alignment

    return alignments


def sample_dataset(
        dataset: Dataset,
        transform: Transform,
        structure_factors: StructureFactors,
        sample_rate: float,
        grid_size: int,
        grid_spacing: float,
) -> np.ndarray:
    reflections: gemmi.Mtz = dataset.reflections
    unaligned_xmap: gemmi.FloatGrid = reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )

    unaligned_xmap_array = np.array(unaligned_xmap, copy=False)

    std = np.std(unaligned_xmap_array)
    unaligned_xmap_array[:, :, :] = unaligned_xmap_array[:, :, :] / std

    transform_inverse = transform.transform.inverse()

    transform_vec = np.array(transform_inverse.vec.tolist())
    transform_mat = np.array(transform_inverse.mat.tolist())

    transform_mat = np.matmul(transform_mat, np.eye(3) * grid_spacing)
    offset = np.matmul(transform_mat, np.array([grid_size / 2, grid_size / 2, grid_size / 2]).reshape(3, 1)).flatten()
    offset_tranform_vec = transform_vec - offset

    tr = gemmi.Transform()
    tr.mat.fromlist(transform_mat.tolist())
    tr.vec.fromlist(offset_tranform_vec.tolist())

    arr = np.zeros([grid_size, grid_size, grid_size], dtype=np.float32)

    unaligned_xmap.interpolate_values(arr, tr)

    return arr


def sample_datasets(
        truncated_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        structure_factors: StructureFactors,
        sample_rate: float,
        grid_size: int,
        grid_spacing: float,
) -> MutableMapping[str, np.ndarray]:
    samples: MutableMapping[str, np.ndarray] = {}
    for dtag, dataset in truncated_datasets.items():
        alignment: Alignment = alignments[dtag]
        residue_transform: Transform = alignment[marker]

        sample: np.ndarray = sample_dataset(
            dataset,
            residue_transform,
            structure_factors,
            sample_rate,
            grid_size,
            grid_spacing,
        )
        samples[dtag] = sample

    return samples


def get_corr(reference_sample_mask, sample_mask, diag):
    reference_mask_size = np.sum(reference_sample_mask)
    sample_mask_size = np.sum(sample_mask)

    denominator = max(sample_mask_size, reference_mask_size)

    if denominator == 0.0:
        if diag:
            corr = 1.0
        else:
            corr = 0.0

    else:

        corr = np.sum(sample_mask[reference_sample_mask == 1]) / denominator

    return corr


def get_distance_matrix(samples: MutableMapping[str, np.ndarray]) -> np.ndarray:
    # Make a pairwise matrix
    correlation_matrix = np.zeros((len(samples), len(samples)))

    for x, reference_sample in enumerate(samples.values()):

        reference_sample_copy_unscaled = reference_sample

        reference_sample_copy = reference_sample_copy_unscaled

        reference_sample_mask = np.zeros(reference_sample_copy.shape)
        reference_sample_mask[reference_sample_copy > 1.0] = 1

        for y, sample in enumerate(samples.values()):
            sample_copy = sample.copy()

            sample_mask = np.zeros(sample_copy.shape)
            sample_mask[sample_copy > 1.0] = 1

            correlation_matrix[x, y] = get_corr(reference_sample_mask, sample_mask, x == y)
    # For every sample
    # Find the distance to every other sample
    # Enter it into the matrix
    return correlation_matrix


def get_z_clusters(z: np.ndarray, mean: np.ndarray) -> MutableMapping[int, Cluster]:
    # Mask large z values
    # Mask small Z values
    # Mask intermediate ones
    # Mask large mean values
    # Mask (Large mean and not small z) and (large z)
    # Cluster adjacent voxels on this mask with skimage.measure.label
    return None


def get_event_map(
        corrected_density: np.ndarray,
        reference: gemmi.Structure,
        dataset: Dataset,
        alignment: Transform,
) -> gemmi.FloatGrid:
    return None


def get_background_corrected_density(
        dataset_sample: np.ndarray,
        cluster: Cluster,
        mean: np.ndarray,
        z: np.ndarray,
) -> np.ndarray:
    # Select the cluster density
    # Set it to some relatively large sigma
    return None


def write_event_map(event_map: gemmi.FloatGrid, out_path: Path):
    ccp4 = gemmi.CCP4Map()
    ccp4.grid = event_map
    ccp4.setup()
    ccp4.set_header(True, 2)

    ccp4.write_ccp4_map(str(out_path))


def get_event(dataset: Dataset, cluster: Cluster) -> Event:
    return None


def get_failed_event(dataset: Dataset) -> Event:
    return None


def get_reference(datasets: MutableMapping[str, Dataset], reference_dtag: Optional[str],
                  apo_dtags: List[str]) -> gemmi.Structure:
    # If reference dtag given, select it
    if reference_dtag:
        for dtag in datasets:
            if dtag == reference_dtag:
                return datasets[dtag]

        raise Exception("Reference dtag not in datasets!")

    # Otherwise, select highest resolution structure
    else:
        reference_dtag = min(
            apo_dtags,
            key=lambda dataset_dtag: datasets[dataset_dtag].reflections.resolution_high(),
        )

        return datasets[reference_dtag]


def get_linkage_from_correlation_matrix(correlation_matrix):
    condensed = spsp.distance.squareform(1.0 - correlation_matrix)
    linkage = spc.hierarchy.linkage(condensed, method='complete')
    # linkage = spc.linkage(condensed, method='ward')

    return linkage


def cluster_linkage(linkage, cutoff):
    idx = spc.hierarchy.fcluster(linkage, cutoff, 'distance')

    return idx


def cluster_strong_density(linkage: np.ndarray, cutoff: float) -> np.ndarray:
    # Get the linkage matrix
    # Cluster the datasets
    clusters: np.ndarray = cluster_linkage(linkage, cutoff)
    # Determine which clusters have known apos in them

    return clusters


def get_common_reflections(
        moving_reflections: gemmi.Mtz,
        reference_reflections: gemmi.Mtz,
        structure_factors: StructureFactors,
):
    # Get own reflections
    moving_reflections_array = np.array(moving_reflections, copy=True)
    moving_reflections_table = pd.DataFrame(
        moving_reflections_array,
        columns=moving_reflections.column_labels(),
    )
    moving_reflections_table.set_index(["H", "K", "L"], inplace=True)
    dtag_flattened_index = moving_reflections_table[
        ~moving_reflections_table[structure_factors.f].isna()].index.to_flat_index()

    # Get reference
    reference_reflections_array = np.array(reference_reflections, copy=True)
    reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                               columns=reference_reflections.column_labels(),
                                               )
    reference_reflections_table.set_index(["H", "K", "L"], inplace=True)
    reference_flattened_index = reference_reflections_table[
        ~reference_reflections_table[structure_factors.f].isna()].index.to_flat_index()

    running_index = dtag_flattened_index.intersection(reference_flattened_index)

    return running_index.to_list()


def get_all_common_reflections(datasets: MutableMapping[str, Dataset], structure_factors: StructureFactors,
                               tol=0.000001):
    running_index = None

    for dtag, dataset in datasets.items():
        reflections = dataset.reflections
        reflections_array = np.array(reflections, copy=True)
        reflections_table = pd.DataFrame(reflections_array,
                                         columns=reflections.column_labels(),
                                         )
        reflections_table.set_index(["H", "K", "L"], inplace=True)

        is_na = reflections_table[structure_factors.f].isna()
        is_zero = reflections_table[structure_factors.f].abs() < tol
        mask = ~(is_na | is_zero)

        flattened_index = reflections_table[mask].index.to_flat_index()
        if running_index is None:
            running_index = flattened_index
        running_index = running_index.intersection(flattened_index)
    return running_index.to_list()


def truncate_reflections(reflections: gemmi.Mtz, index=None) -> gemmi.Mtz:
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = reflections.spacegroup
    new_reflections.set_cell_for_all(reflections.cell)

    # Add dataset
    new_reflections.add_dataset("truncated")

    # Add columns
    for column in reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Get data
    data_array = np.array(reflections, copy=True)
    data = pd.DataFrame(data_array,
                        columns=reflections.column_labels(),
                        )
    data.set_index(["H", "K", "L"], inplace=True)

    # Truncate by index
    data_indexed = data.loc[index]

    # To numpy
    data_dropped_array = data_indexed.to_numpy()

    # new data
    new_data = np.hstack([data_indexed.index.to_frame().to_numpy(),
                          data_dropped_array,
                          ]
                         )

    # Update
    new_reflections.set_data(new_data)

    # Update resolution
    new_reflections.update_reso()

    return new_reflections


def smooth(reference: Dataset, moving: Dataset, structure_factors: StructureFactors):
    # Get common set of reflections
    common_reflections = get_common_reflections(
        reference.reflections,
        moving.reflections,
        structure_factors,
    )

    # Truncate
    truncated_reference: gemmi.Mtz = truncate_reflections(reference.reflections, common_reflections)
    truncated_dataset: gemmi.Mtz = truncate_reflections(moving.reflections, common_reflections)

    # Refference array
    reference_reflections: gemmi.Mtz = truncated_reference
    reference_reflections_array: np.ndarray = np.array(reference_reflections,
                                                       copy=True,
                                                       )
    reference_reflections_table = pd.DataFrame(reference_reflections_array,
                                               columns=reference_reflections.column_labels(),
                                               )
    reference_f_array = reference_reflections_table[structure_factors.f].to_numpy()

    # Dtag array
    dtag_reflections = truncated_dataset
    dtag_reflections_array = np.array(dtag_reflections,
                                      copy=True,
                                      )
    dtag_reflections_table = pd.DataFrame(dtag_reflections_array,
                                          columns=dtag_reflections.column_labels(),
                                          )
    dtag_f_array = dtag_reflections_table[structure_factors.f].to_numpy()

    # Resolution array
    resolution_array = reference_reflections.make_1_d2_array()

    # Prepare optimisation
    x = reference_f_array
    y = dtag_f_array

    r = resolution_array

    sample_grid = np.linspace(min(r), max(r), 100)
    #
    # knn_x = neighbors.RadiusNeighborsRegressor(0.01)
    # knn_x.fit(r.reshape(-1, 1),
    #           x.reshape(-1, 1),
    #           )
    # x_f = knn_x.predict(sample_grid[:, np.newaxis]).reshape(-1)
    #
    sorting = np.argsort(r)
    r_sorted = r[sorting]
    x_sorted = x[sorting]
    y_sorted = y[sorting]

    scales = []
    rmsds = []

    # Approximate x_f
    former_sample_point = sample_grid[0]
    x_f_list = []
    for sample_point in sample_grid[1:]:
        mask = (r_sorted < sample_point) * (r_sorted > former_sample_point)
        x_vals = x_sorted[mask]
        former_sample_point = sample_point
        x_f_list.append(np.mean(x_vals))
    x_f = np.array(x_f_list)

    # Optimise the scale factor
    for scale in np.linspace(-10, 10, 100):
        # y_s = y_s * np.exp(scale * r)
        #
        # knn_y = neighbors.RadiusNeighborsRegressor(0.01)
        # knn_y.fit(r.reshape(-1, 1),
        #           y_s.reshape(-1, 1),
        #           )
        # y_f = knn_y.predict(sample_grid[:, np.newaxis]).reshape(-1)

        y_s_sorted = y_sorted * np.exp(scale * r_sorted)

        # approximate y_f
        former_sample_point = sample_grid[0]
        y_f_list = []
        for sample_point in sample_grid[1:]:
            mask = (r_sorted < sample_point) * (r_sorted > former_sample_point)
            y_vals = y_s_sorted[mask]
            former_sample_point = sample_point
            y_f_list.append(np.mean(y_vals))
        y_f = np.array(y_f_list)

        rmsd = np.sum(np.abs(x_f - y_f))

        scales.append(scale)
        rmsds.append(rmsd)

    min_scale = scales[np.argmin(rmsds)]

    # Get the original reflections
    original_reflections = moving.reflections

    original_reflections_array = np.array(original_reflections,
                                          copy=True,
                                          )

    original_reflections_table = pd.DataFrame(original_reflections_array,
                                              columns=reference_reflections.column_labels(),
                                              )

    f_array = original_reflections_table[structure_factors.f]

    f_scaled_array = f_array * np.exp(min_scale * original_reflections.make_1_d2_array())

    original_reflections_table[structure_factors.f] = f_scaled_array

    # New reflections
    new_reflections = gemmi.Mtz(with_base=False)

    # Set dataset properties
    new_reflections.spacegroup = original_reflections.spacegroup
    new_reflections.set_cell_for_all(original_reflections.cell)

    # Add dataset
    new_reflections.add_dataset("scaled")

    # Add columns
    for column in original_reflections.columns:
        new_reflections.add_column(column.label, column.type)

    # Update
    new_reflections.set_data(original_reflections_table.to_numpy())

    # Update resolution
    new_reflections.update_reso()

    # Create new dataset
    smoothed_dataset = Dataset(
        moving.dtag,
        moving.structure,
        new_reflections,
        moving.structure_path,
        moving.reflections_path,
        moving.fragment_path,
        moving.fragment_structures,
        min_scale,
    )

    return smoothed_dataset


def smooth_datasets(
        datasets: MutableMapping[str, Dataset],
        reference_dataset: Dataset,
        structure_factors: StructureFactors,
        debug: bool = True,
) -> MutableMapping[str, Dataset]:
    # For dataset reflections

    # smoothed_datasets: MutableMapping[str, Dataset] = {}
    # for dtag, dataset in datasets.items():
    #     if debug:
    #         print(f"\tSmoothing {dtag} against reference {reference_dataset.dtag}")
    #     # Minimise distance to reference reflections
    #     smoothed_datasets[dtag] = smooth(reference_dataset, dataset, structure_factors)
    #
    #     if debug:
    #         print(f"\t\tSmoothinging factor: {smoothed_datasets[dtag].smoothing_factor}")

    datasets_list: List[Optional[Dataset]] = joblib.Parallel(
        n_jobs=20,
        verbose=50,
        # backend="multiprocessing",
    )(
        joblib.delayed(smooth)(
            reference_dataset, dataset, structure_factors
        )
        for dataset
        in list(datasets.values())
    )
    smoothed_datasets = {dtag: smoothed_dataset for dtag, smoothed_dataset in zip(list(datasets.keys()), datasets_list)}

    return smoothed_datasets


def get_event_map_path(out_dir_path: Path, dataset: Dataset, cluster_num: int, residue_id: ResidueID, ) -> Path:
    # Construct a unique file name: {model}_{chain}_{insertion}_{dtag}_{cluster_num}.pdb
    # Append to out directory
    return None


def write_result_html(pandda_results: PanDDAResults) -> Path:
    return None


def get_fragment_z_maxima(
        fragment_affinity_z_maps: MutableMapping[Tuple[float, float, float], np.ndarray]
) -> AffinityMaxima:
    maximas: MutableMapping[Tuple[float, float, float], AffinityMaxima] = {}
    for rotation_index, affinity_map in fragment_affinity_z_maps.items():
        #
        max_index: np.ndarray = np.argmax(affinity_map)
        max_value: float = affinity_map[max_index]
        maxima: AffinityMaxima = AffinityMaxima(
            max_index,
            max_value,
            rotation_index,
        )
        maximas[rotation_index] = maxima

    best_maxima: AffinityMaxima = max(
        list(maximas.values()),
        key=lambda _maxima: _maxima.correlation,
    )

    return best_maxima


def is_affinity_event(affinity_maxima: AffinityMaxima, min_correlation: float) -> bool:
    if affinity_maxima.correlation > min_correlation:
        return True

    else:
        return False


def get_background_corrected_density_from_affinity(
        dataset_sample: np.ndarray,
        maxima: AffinityMaxima,
        mean: np.ndarray,
) -> np.ndarray:
    # Get the maxima excess correlation

    # Subtract 1-excess correlation * the mean map from the sample
    background_corrected_density: np.ndarray = (dataset_sample - maxima.correlation * mean) / (1 - maxima.correlation)

    return background_corrected_density


def get_backtransformed_map(
        corrected_density: np.ndarray,
        reference_dataset: Dataset,
        dataset: Dataset,
        transform: Transform,
        marker: Marker,
        grid_size: int,
        grid_spacing: float,
        structure_factors: StructureFactors,
        sample_rate: float,
) -> gemmi.FloatGrid:
    # Embed corrected density in grid at origin
    corrected_density_grid: gemmi.FloatGrid = gemmi.FloatGrid(*corrected_density.shape)
    unit_cell: gemmi.UnitCell = gemmi.UnitCell(grid_size * grid_spacing, grid_size * grid_spacing,
                                               grid_size * grid_spacing,
                                               90, 90, 90)
    corrected_density_grid.set_unit_cell(unit_cell)
    corrected_density_grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')

    # FFT
    grid: gemmi.FloatGrid = dataset.reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )

    # mask
    mask: gemmi.Int8Grid = gemmi.Int8Grid()
    # residue_ca: gemmi.Atom = dataset.structure[residue_id.model][residue_id.chain][residue_id.insertion][0]["CA"]
    # dataset_centroid: gemmi.Pos = residue_ca.pos
    dataset_centroid: gemmi.Position = gemmi.Position(marker.x, marker.y, marker.z)
    mask.set_points_around(dataset_centroid, radius=6, value=1)

    # Get indexes of grid points around moving residue
    mask_array: np.ndarray = np.array(mask, copy=False)
    indexes: np.ndarray = np.argwhere(mask_array == 1)

    # Loop over those indexes, transforming them to grid at origin, assigning 0 to all points outside cell (0,0,0)
    for index in indexes:
        index_position: gemmi.Position = grid.get_position(index[0], index[1], index[2])
        index_relative_position: gemmi.Position = gemmi.Position(
            index_position.x - dataset_centroid.x,
            index_position.y - dataset_centroid.y,
            index_position.z - dataset_centroid.z,
        )
        transformed_position: gemmi.Position = transform.transform.inverse().apply(index_relative_position)
        interpolated_value: float = corrected_density_grid.interpolate_value(transformed_position)
        grid.set_value(index[0], index[1], index[2], interpolated_value)

    return grid


def get_affinity_event_map_path(
        out_dir,
        dataset,
        marker: Marker,
) -> Path:
    if marker.resid:
        path: Path = out_dir / Constants.affinity_event_map_res_path.format(
            dtag=dataset.dtag,
            model=marker.resid.model,
            chain=marker.resid.chain,
            insertion=marker.resid.insertion,
        )
    else:
        path: Path = out_dir / Constants.affinity_event_map_res_path.format(
            dtag=dataset.dtag,
            x=round(marker.x, 3),
            y=round(marker.y, 3),
            z=round(marker.z, 3),
        )

    return path


def get_affinity_event(
        dataset: Dataset,
        maxima: AffinityMaxima,
        marker: Marker,
) -> AffinityEvent:
    return AffinityEvent(
        dataset.dtag,
        marker,
        maxima.correlation,
    )


def get_failed_affinity_event(dataset: Dataset, marker: Marker) -> AffinityEvent:
    return AffinityEvent(
        dataset.dtag,
        marker,
        0,
    )


def get_not_enough_comparator_dataset_affinity_result(dataset: Dataset,
                                                      marker: Marker) -> DatasetAffinityResults:
    dataset_result: DatasetAffinityResults = DatasetAffinityResults(
        dataset.dtag,
        marker,
        structure_path=dataset.structure_path,
        reflections_path=dataset.reflections_path,
        fragment_path=dataset.fragment_path,
        events={},
        comparators=[],
    )

    return dataset_result


def save_mtz(mtz: gemmi.Mtz, path: Path):
    mtz.write_to_file(str(path))
