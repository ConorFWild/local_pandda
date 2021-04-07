import os
from typing import *
from local_pandda.datatypes import *
from pathlib import Path
import re
import itertools
import dataclasses
import math
import gc

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
from scipy import signal as spsi
from scipy import ndimage
from scipy.signal import fftconvolve, oaconvolve
from skimage.transform import rescale, resize, downscale_local_mean

try:
    import torch
except Exception as e:
    print(e)

# Custom
from local_pandda.constants import Constants
from local_pandda.database import *
from local_pandda.ncc import NCC

import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cudnn.benchmark = False


# torch.cuda.set_device(0)

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


def get_fragment_map(
        structure: gemmi.Structure,
        resolution: float,
        grid_spacing: float,
        sample_rate: float,
        b_factor,
        margin: float = 1.5,
) -> np.ndarray:
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.b_iso = b_factor

    dencalc: gemmi.DensityCalculatorE = gemmi.DensityCalculatorE()

    dencalc.d_min = resolution
    dencalc.rate = resolution / (2 * grid_spacing)

    # print(resolution)
    # print(grid_spacing)
    # print(structure.spacegroup_hm)
    # print(structure.cell)
    dencalc.set_grid_cell_and_spacegroup(structure)
    dencalc.put_model_density_on_grid(structure[0])

    grid: gemmi.FloatGrid = dencalc.grid
    # print(grid)
    # array: np.ndarray = np.array(grid, copy=True)

    # box = structure.calculate_box()
    # box.add_margin(margin)
    # min_pos: gemmi.Position = box.minimum
    # max_pos = box.maximum
    #
    # distance = max_pos - min_pos
    unit_cell = grid.unit_cell
    min_pos = [0.0, 0.0, 0.0]

    tr = gemmi.Transform()
    tr.mat.fromlist([[1 * grid_spacing, 0, 0], [0, 1 * grid_spacing, 0], [0, 0, 1 * grid_spacing]])
    tr.vec.fromlist([min_pos[0], min_pos[1], min_pos[2]])

    arr = np.zeros(
        [
            int(unit_cell.a / grid_spacing) + 1,
            int(unit_cell.b / grid_spacing) + 1,
            int(unit_cell.c / grid_spacing) + 1,
        ],
        dtype=np.float32
    )

    grid.interpolate_values(arr, tr)
    # print(arr.shape)

    # Mask
    mask_grid = gemmi.FloatGrid(
        grid.nu,
        grid.nv,
        grid.nw,
    )

    mask_grid.set_unit_cell(grid.unit_cell)
    mask_grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == "H":
                        # print("Skipping H")
                        continue
                    pos: gemmi.Position = atom.pos
                    # mask_grid.set_points_around(pos, 1.0, 1.0)
                    mask_grid.set_points_around(pos, 0.75, 1.0)


    mask_arr = np.zeros(
        [
            int(unit_cell.a / grid_spacing) + 1,
            int(unit_cell.b / grid_spacing) + 1,
            int(unit_cell.c / grid_spacing) + 1,
        ],
        dtype=np.float32
    )

    mask_grid.interpolate_values(mask_arr, tr)

    # mask the array
    arr[mask_arr < 0.5] = 0

    return arr


def rotate_translate_structure(fragment_structure: gemmi.Structure, rotation_matrix, max_dist: float,
                               margin: float = 3.0) -> gemmi.Structure:
    # print(rotation_matrix)
    structure_copy = fragment_structure.clone()
    transform: gemmi.Transform = gemmi.Transform()
    transform.mat.fromlist(rotation_matrix.tolist())
    transform.vec.fromlist([0.0, 0.0, 0.0])

    # Get fragment mean
    xs = []
    ys = []
    zs = []
    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # print(atom.pos)
                    pos: gemmi.Position = atom.pos
                    xs.append(pos.x)
                    ys.append(pos.y)
                    zs.append(pos.z)

    mean_x = np.mean(np.array(xs))
    mean_y = np.mean(np.array(ys))
    mean_z = np.mean(np.array(zs))

    # demean
    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    new_x = pos.x - mean_x
                    new_y = pos.y - mean_y
                    new_z = pos.z - mean_z
                    atom.pos = gemmi.Position(new_x, new_y, new_z)

    # rotate
    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # print(atom.pos)
                    pos: gemmi.Position = atom.pos
                    rotated_vec = transform.apply(pos)
                    # print(rotated_vec)
                    rotated_position = gemmi.Position(rotated_vec.x, rotated_vec.y, rotated_vec.z)
                    atom.pos = rotated_position
                    # print(atom.pos)

    # remean to max_dist/2 + margin
    for model in structure_copy:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos: gemmi.Position = atom.pos
                    new_x = pos.x + ((max_dist / 2) + margin)
                    new_y = pos.y + ((max_dist / 2) + margin)
                    new_z = pos.z + ((max_dist / 2) + margin)
                    atom.pos = gemmi.Position(new_x, new_y, new_z)

    #
    # box = structure_copy.calculate_box()
    # box.add_margin(margin)
    # min_pos: gemmi.Position = box.minimum
    #
    # for model in structure_copy:
    #     for chain in model:
    #         for residue in chain:
    #             for atom in residue:
    #                 pos: gemmi.Position = atom.pos
    #                 new_x = pos.x - min_pos.x
    #                 new_y = pos.y - min_pos.y
    #                 new_z = pos.z - min_pos.z
    #                 atom.pos = gemmi.Position(new_x, new_y, new_z)

    structure_copy.cell = gemmi.UnitCell(
        max_dist + (2 * margin),
        max_dist + (2 * margin),
        max_dist + (2 * margin),
        90.0, 90.0, 90.0)

    structure_copy.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").xhm()

    return structure_copy


def sample_fragment(rotation_index, path, resolution, grid_spacing, sample_rate, b_factor):
    fragment_structure = path_to_structure(path)
    rotation = spsp.transform.Rotation.from_euler("xyz",
                                                  [rotation_index[0],
                                                   rotation_index[1],
                                                   rotation_index[2]],
                                                  degrees=True)
    rotation_matrix: np.ndarray = rotation.as_matrix()

    max_dist = get_max_dist(fragment_structure)

    rotated_structure: gemmi.Structure = rotate_translate_structure(fragment_structure, rotation_matrix, max_dist)
    fragment_map: np.ndarray = get_fragment_map(rotated_structure, resolution, grid_spacing, sample_rate, b_factor)

    return fragment_map


def get_fragment_mask(rotated_structure, grid_spacing, radii):
    unit_cell = rotated_structure.cell

    grid = gemmi.FloatGrid(
        int(unit_cell.a / grid_spacing),
        int(unit_cell.b / grid_spacing),
        int(unit_cell.c / grid_spacing),
    )

    grid.set_unit_cell(unit_cell)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    for i, radius in enumerate(radii):
        for model in rotated_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element.name == "H":
                            print("Skipping H")
                            continue
                        pos: gemmi.Position = atom.pos
                        grid.set_points_around(pos, radius, float(i) + 1.0)

    arr = np.array(grid, copy=True)
    return arr


def sample_fragment_mask(rotation_index, path, max_dist, grid_spacing, radii):
    fragment_structure = path_to_structure(path)
    rotation = spsp.transform.Rotation.from_euler("xyz",
                                                  [rotation_index[0],
                                                   rotation_index[1],
                                                   rotation_index[2]],
                                                  degrees=True)
    rotation_matrix: np.ndarray = rotation.as_matrix()
    rotated_structure: gemmi.Structure = rotate_translate_structure(fragment_structure, rotation_matrix, max_dist)
    fragment_map: np.ndarray = get_fragment_mask(rotated_structure, grid_spacing, radii)

    return fragment_map


def get_fragment_maps(
        fragment_structure: gemmi.Structure,
        resolution: float,
        num_poses: int,
        sample_rate: float,
        grid_spacing: float,
        b_factor: float):
    sample_angles = np.linspace(0, 360, num=10, endpoint=False).tolist()

    rotations = [(x, y, z) for x, y, z in itertools.product(sample_angles, sample_angles, sample_angles)]

    fragment_samples = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
    )(
        joblib.delayed(sample_fragment)(
            rotation_index, structure_to_path(fragment_structure), resolution, grid_spacing, sample_rate, b_factor,
        )
        for rotation_index
        in rotations
    )

    fragment_maps = {rotation: fragment_sample for rotation, fragment_sample in zip(rotations, fragment_samples)}

    return fragment_maps


def get_max_dist(fragment_structure):  #
    distances = []
    for model in fragment_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    pos: gemmi.Position = atom.pos

                    for atom_2 in residue:
                        if atom.element.name == "H":
                            continue
                        pos_2: gemmi.Position = atom_2.pos
                        distances.append(pos.dist(pos_2))

    return max(distances)


def get_fragment_masks(
        fragment_structure: gemmi.Structure,
        num_poses: int,
        grid_spacing: float,
        radii: List[float],
):
    sample_angles = np.linspace(0, 360, num=10, endpoint=False).tolist()

    rotations = [(x, y, z) for x, y, z in itertools.product(sample_angles, sample_angles, sample_angles)]

    max_dist = get_max_dist(fragment_structure)

    fragment_samples = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
    )(
        joblib.delayed(sample_fragment_mask)(
            rotation_index, structure_to_path(fragment_structure), max_dist, grid_spacing, radii
        )
        for rotation_index
        in rotations
    )

    fragment_maps = {rotation: fragment_sample for rotation, fragment_sample in zip(rotations, fragment_samples)}

    return fragment_maps


def get_residue_id(model: gemmi.Model, chain: gemmi.Chain, insertion: str):
    return ResidueID(model.name, chain.name, str(insertion))


def get_residue(structure: gemmi.Structure, residue_id: ResidueID) -> gemmi.Residue:
    return structure[residue_id.model][residue_id.chain][residue_id.insertion][0]


def get_comparator_datasets(
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        target_dtag: str,
        apo_mask: np.ndarray,
        datasets: MutableMapping[str, Dataset],
        min_cluster_size: int,
        num_datasets: int,
) -> Optional[MutableMapping[str, Dataset]]:
    #
    # apo_cluster_indexes: np.ndarray = np.unique(dataset_clusters[apo_mask])
    #
    # apo_clusters: MutableMapping[int, np.ndarray] = {}
    # for apo_cluster_index in apo_cluster_indexes:
    #     cluster: np.ndarray = dataset_clusters[dataset_clusters == apo_cluster_index]
    #
    #     #
    #     if cluster.size > min_cluster_size:
    #         apo_clusters[apo_cluster_index] = cluster

    apo_clusters: MutableMapping[int, np.ndarray] = {}
    for apo_cluster_index in np.unique(dataset_clusters):
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

    # Sort by resolution
    closest_cluster_dtag_resolutions = {dtag: datasets[dtag].reflections.resolution_high()
                                        for dtag in closest_cluster_dtag_array}
    print(f"Got {len(closest_cluster_dtag_resolutions)} comparatprs")
    sorted_resolution_dtags = sorted(closest_cluster_dtag_resolutions,
                                     key=lambda dtag: closest_cluster_dtag_resolutions[dtag])
    resolution_cutoff = max(datasets[target_dtag].reflections.resolution_high(),
                            datasets[sorted_resolution_dtags[
                                min(len(sorted_resolution_dtags), num_datasets)]].reflections.resolution_high()
                            )
    # sorted_resolution_dtags_cutoff = [dtag for dtag in sorted_resolution_dtags if datasets[dtag].reflections.resolution_high() < resolution_cutoff]

    # highest_resolution_dtags = sorted_resolution_dtags_cutoff[-min(len(sorted_resolution_dtags), num_datasets):]

    closest_cluster_datasets: MutableMapping[str, Dataset] = {dtag: datasets[dtag]
                                                              for dtag
                                                              in closest_cluster_dtag_array
                                                              if datasets[
                                                                  dtag].reflections.resolution_high() < resolution_cutoff
                                                              }
    print(closest_cluster_datasets)

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
        verbose=50,
        n_jobs=-1,
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
    resolution: float = max([dataset.reflections.resolution_high() for dtag, dataset in datasets.items()])

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

    return resolution_truncated_datasets

    # # Get common set of reflections
    # common_reflections = get_all_common_reflections(resolution_truncated_datasets, structure_factors)
    #
    # # truncate on reflections
    # new_datasets_reflections: MutableMapping[str, Dataset] = {}
    # for dtag in resolution_truncated_datasets:
    #     resolution_truncated_dataset: Dataset = resolution_truncated_datasets[dtag]
    #     reflections = resolution_truncated_dataset.reflections
    #     reflections_array = np.array(reflections)
    #     print(f"{dtag}")
    #     print(f"{reflections_array.shape}")
    #
    #     truncated_reflections: gemmi.Mtz = truncate_reflections(
    #         reflections,
    #         common_reflections,
    #     )
    #
    #     reflections_array = np.array(truncated_reflections)
    #     print(f"{dtag}: {reflections_array.shape}")
    #
    #     new_dataset: Dataset = Dataset(
    #         resolution_truncated_dataset.dtag,
    #         resolution_truncated_dataset.structure,
    #         truncated_reflections,
    #         resolution_truncated_dataset.structure_path,
    #         resolution_truncated_dataset.reflections_path,
    #         resolution_truncated_dataset.fragment_path,
    #         resolution_truncated_dataset.fragment_structures,
    #         resolution_truncated_dataset.smoothing_factor,
    #     )
    #
    #     new_datasets_reflections[dtag] = new_dataset
    #
    # return new_datasets_reflections


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

    rotation, rmsd = scipy.spatial.transform.Rotation.align_vectors(de_meaned_referecnce, de_meaned_moving)

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
        verbose=50,
        n_jobs=-1,
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
        marker: Marker,
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

    # transform_vec = np.array(transform_inverse.vec.tolist())
    transform_vec = -np.array(transform.transform.vec.tolist())

    transform_mat = np.array(transform_inverse.mat.tolist()).T
    transform_mat = np.matmul(transform_mat, np.eye(3) * grid_spacing)

    offset = np.matmul(transform_mat, np.array([grid_size / 2, grid_size / 2, grid_size / 2]).reshape(3, 1)).flatten()
    print(f"Offset from: {offset}")
    print(f"transform_vec from: {transform_vec}")

    offset_tranform_vec = transform_vec - offset
    marker_offset_tranform_vec = offset_tranform_vec + np.array([marker.x, marker.y, marker.z])
    print(f"Sampling from: {marker_offset_tranform_vec}")

    tr = gemmi.Transform()
    tr.mat.fromlist(transform_mat.tolist())
    # transform_non_inv_mat = np.array(transform.transform.mat.tolist()).T
    # transform_non_inv_mat = np.matmul(transform_non_inv_mat, np.eye(3) * grid_spacing)
    # tr.mat.fromlist(transform_non_inv_mat.tolist())
    tr.vec.fromlist(marker_offset_tranform_vec.tolist())

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
    arrays = joblib.Parallel(
        verbose=50,
        n_jobs=-1,
        # backend="multiprocessing",
    )(
        joblib.delayed(sample_dataset)(
            dataset,
            alignments[dtag][marker],
            marker,
            structure_factors,
            sample_rate,
            grid_size,
            grid_spacing,
        )
        for dtag, dataset
        in truncated_datasets.items()
    )
    samples = {dtag: result for dtag, result in zip(truncated_datasets, arrays)}

    #
    # for dtag, dataset in truncated_datasets.items():
    #     alignment: Alignment = alignments[dtag]
    #     residue_transform: Transform = alignment[marker]
    #
    #     sample: np.ndarray = sample_dataset(
    #         dataset,
    #         residue_transform,
    #         marker,
    #         structure_factors,
    #         sample_rate,
    #         grid_size,
    #         grid_spacing,
    #     )
    #     samples[dtag] = sample

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


# def get_z_clusters(z: np.ndarray, mean: np.ndarray) -> MutableMapping[int, Cluster]:
#     # Mask large z values
#     # Mask small Z values
#     # Mask intermediate ones
#     # Mask large mean values
#     # Mask (Large mean and not small z) and (large z)
#     # Cluster adjacent voxels on this mask with skimage.measure.label
#     return None


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


def get_affinity_background_corrected_density(
        dataset_sample: np.ndarray,
        fragment_map: np.ndarray,
        maxima: AffinityMaxima,
        mean: np.ndarray,
) -> Tuple[float, np.ndarray]:
    # Select the cluster density
    # Set it to some relatively large sigma
    index = maxima.index
    # np.unravel_index(1621, (6, 7, 8, 9))

    dataset_shape_tuple = dataset_sample.shape
    fragment_shape_tuple = fragment_map.shape
    dataset_shape = np.array([dataset_shape_tuple[0], dataset_shape_tuple[1], dataset_shape_tuple[2]])
    fragment_shape = np.array([fragment_shape_tuple[0], fragment_shape_tuple[1], fragment_shape_tuple[2]])

    print(index)
    print(np.array([index[0], index[1], index[2]]))
    offset_dataset_to_fragment = np.array([index[0], index[1], index[2]]) - np.floor((fragment_shape - 1) / 2)
    print(f"Offset dataset to fragment: {offset_dataset_to_fragment}")
    dataset_frame_fragment_max = offset_dataset_to_fragment + fragment_shape

    dataset_min_x = int(np.max([0, offset_dataset_to_fragment[0]]))
    print([0, -offset_dataset_to_fragment[0]])
    dataset_min_y = int(np.max([0, offset_dataset_to_fragment[1]]))
    dataset_min_z = int(np.max([0, offset_dataset_to_fragment[2]]))
    dataset_max_x = int(np.min([dataset_shape[0], dataset_frame_fragment_max[0]]))
    print([dataset_shape[0], dataset_frame_fragment_max[0]])
    dataset_max_y = int(np.min([dataset_shape[1], dataset_frame_fragment_max[1]]))
    dataset_max_z = int(np.min([dataset_shape[2], dataset_frame_fragment_max[2]]))

    sample_overlap = dataset_sample[dataset_min_x:dataset_max_x, dataset_min_y:dataset_max_y,
                     dataset_min_z:dataset_max_z, ]
    mean_overlap = mean[dataset_min_x:dataset_max_x, dataset_min_y:dataset_max_y, dataset_min_z:dataset_max_z, ]

    offset_fragment_to_dataset = -offset_dataset_to_fragment
    fragment_frame_dataset_max = offset_fragment_to_dataset + dataset_shape
    fragment_min_x = int(np.max([0, offset_fragment_to_dataset[0]]))
    print([0, offset_fragment_to_dataset[0]])
    fragment_min_y = int(np.max([0, offset_fragment_to_dataset[1]]))
    fragment_min_z = int(np.max([0, offset_fragment_to_dataset[2]]))
    fragment_max_x = int(np.min([fragment_shape[0], fragment_frame_dataset_max[0]]))
    print([fragment_shape[0], fragment_frame_dataset_max[0]])
    fragment_max_y = int(np.min([fragment_shape[1], fragment_frame_dataset_max[1]]))
    fragment_max_z = int(np.min([fragment_shape[2], fragment_frame_dataset_max[2]]))

    fragment_overlap = fragment_map[fragment_min_x:fragment_max_x, fragment_min_y:fragment_max_y,
                       fragment_min_z:fragment_max_z, ]
    print(dataset_sample.shape)
    print(sample_overlap.shape)
    print(fragment_map.shape)
    print(fragment_overlap.shape)

    fragment_mask = fragment_overlap.copy()
    fragment_mask[fragment_overlap < fragment_overlap.mean()] = 0.0
    fragment_mask[fragment_overlap >= fragment_overlap.mean()] = 1.0

    sum_absolute_differances = {}
    for b in np.linspace(0.0, 1.0, 100):
        residual_map = sample_overlap - (b * mean_overlap)
        scaled_fragment_map = (1 - b) * fragment_overlap
        # sum_absolute_differance = np.sum(
        #     np.abs(residual_map[fragment_mask > 0] - scaled_fragment_map[fragment_mask > 0]))

        sum_absolute_differance = np.sum(
            np.square(residual_map[fragment_mask > 0] - scaled_fragment_map[fragment_mask > 0]))
        print(f"For b: {b}: sum absolute diff: {sum_absolute_differance}")

        print(f"\tmax: {np.max(residual_map[fragment_mask > 0])}, {np.max(scaled_fragment_map[fragment_mask > 0])}")
        print(f"\tmin: {np.min(residual_map[fragment_mask > 0])}, {np.min(scaled_fragment_map[fragment_mask > 0])}")

        masked_residual_map = residual_map[fragment_mask > 0]
        masked_scaled_fragment_map = scaled_fragment_map[fragment_mask > 0]

        residual_map_quantile_low = np.quantile(masked_residual_map, 0.25)
        residual_map_quantile_high = np.quantile(masked_residual_map, 0.75)

        #
        # rescaled_masked_residual_map = (masked_residual_map - np.mean(masked_residual_map)) / np.std(masked_residual_map)
        # rescaled_masked_scaled_fragment_map = (masked_scaled_fragment_map - np.mean(masked_scaled_fragment_map)) / np.std(masked_scaled_fragment_map)
        #
        # rescaled_sum_absolute_differance = np.sum(
        #     np.square(rescaled_masked_residual_map - rescaled_masked_scaled_fragment_map))
        # print(f"For b: {b}: rescaled sum absolute diff: {rescaled_sum_absolute_differance}")

        correlation, intercept = np.polyfit(
            masked_residual_map[
                (masked_residual_map > residual_map_quantile_low) * (masked_residual_map < residual_map_quantile_high)],
            masked_scaled_fragment_map[
                (masked_residual_map > residual_map_quantile_low) * (masked_residual_map < residual_map_quantile_high)],
            deg=1,
        )
        print(f"Correlation is: {correlation}")
        print(f"Intercept is: {intercept}")

        sum_absolute_differances[b] = sum_absolute_differance

    bcd = min(sum_absolute_differances, key=lambda x: sum_absolute_differances[x])

    print(bcd)

    bcd_map = (dataset_sample - (bcd * mean)) / (1 - bcd)

    return bcd, bcd_map


def write_event_map(event_map: gemmi.FloatGrid, out_path: Path, marker: Marker, dataset: Dataset, resolution: float):
    st: gemmi.Structure = gemmi.Structure()
    model: gemmi.Model = gemmi.Model(f"{1}")
    chain: gemmi.Chain = gemmi.Chain(f"{1}")
    residue: gemmi.Residue = gemmi.Residue()

    # Get the
    gemmi_atom: gemmi.Atom = gemmi.Atom()
    gemmi_atom.name = "H"
    gemmi_atom.pos = gemmi.Position(marker.x, marker.y, marker.z)
    gemmi_atom.element = gemmi.Element("H")

    # Add atom to residue
    residue.add_atom(gemmi_atom)

    st.cell = dataset.structure.cell
    st.spacegroup_hm = dataset.structure.spacegroup_hm

    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)
    #
    # box = st.calculate_fractional_box(margin=32)
    #
    # ccp4 = gemmi.Ccp4Map()
    # ccp4.grid = event_map
    # ccp4.setup()
    # ccp4.update_ccp4_header(2, True)
    #
    # ccp4.set_extent(box)
    #
    # ccp4.setup()
    # ccp4.update_ccp4_header(2, True)
    #
    # ccp4.write_ccp4_map(str(out_path))

    sf = gemmi.transform_map_to_f_phi(event_map)
    data = sf.prepare_asu_data(dmin=resolution)

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data)
    mtz.write_to_file(str(out_path))


# def get_event(dataset: Dataset, cluster: Cluster) -> Event:
#     return None
#
#
# def get_failed_event(dataset: Dataset) -> Event:
#     return None


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
        verbose=50,
        n_jobs=-1,
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


def write_result_html(pandda_results: PanDDAResults) -> Optional[Path]:
    return None


def get_fragment_z_maxima(
        fragment_affinity_z_maps: MutableMapping[Tuple[float, float, float], np.ndarray]
) -> AffinityMaxima:
    # maximas: MutableMapping[Tuple[float, float, float], AffinityMaxima] = {}
    # for rotation_index, affinity_map in fragment_affinity_z_maps.items():
    #     #
    #     max_index: np.ndarray = np.argmax(affinity_map)
    #     max_value: float = affinity_map.flatten()[max_index]
    #     maxima: AffinityMaxima = AffinityMaxima(
    #         max_index,
    #         max_value,
    #         rotation_index,
    #     )
    #     maximas[rotation_index] = maxima
    #
    # best_maxima: AffinityMaxima = max(
    #     list(maximas.values()),
    #     key=lambda _maxima: _maxima.correlation,
    # )
    #
    # return best_maxima

    #
    fragment_affinity_z_maps_rotation_list = list(fragment_affinity_z_maps.keys())
    fragment_affinity_z_maps_list = list(fragment_affinity_z_maps.values())

    max_fragment_affinity_z_map_rotation = fragment_affinity_z_maps_rotation_list[0]
    max_fragment_affinity_z_map = fragment_affinity_z_maps_list[0]

    for rotation, fragment_affinity_z_map in zip(fragment_affinity_z_maps_rotation_list[1:],
                                                 fragment_affinity_z_maps_list[1:]):
        #     print([rotation, np.max(fragment_affinity_z_map)])
        if np.max(fragment_affinity_z_map) > np.max(max_fragment_affinity_z_map):
            max_fragment_affinity_z_map = fragment_affinity_z_map
            max_fragment_affinity_z_map_rotation = rotation

    # Extract the maxima indexes
    maxima = AffinityMaxima(
        [2 * x for x in
         np.unravel_index(np.argmax(max_fragment_affinity_z_map), max_fragment_affinity_z_map.shape)],
        np.max(max_fragment_affinity_z_map),
        max_fragment_affinity_z_map_rotation,
    )

    return maxima


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
    unit_cell: gemmi.UnitCell = gemmi.UnitCell(grid_size * grid_spacing,
                                               grid_size * grid_spacing,
                                               grid_size * grid_spacing,
                                               90, 90, 90)
    print(f"Corrected density unit cell: {unit_cell}")
    corrected_density_grid.set_unit_cell(unit_cell)
    corrected_density_grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')

    for index, value in np.ndenumerate(corrected_density):
        corrected_density_grid.set_value(index[0], index[1], index[2], value)

    # FFT
    grid: gemmi.FloatGrid = dataset.reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
    grid.fill(0)

    # reference to moving
    inverse_transform = transform.transform.inverse()

    # mask
    mask: gemmi.Int8Grid = gemmi.Int8Grid(grid.nu, grid.nv, grid.nw)
    mask.set_unit_cell(grid.unit_cell)
    mask.spacegroup = gemmi.find_spacegroup_by_name('P 1')
    # residue_ca: gemmi.Atom = dataset.structure[residue_id.model][residue_id.chain][residue_id.insertion][0]["CA"]
    # dataset_centroid: gemmi.Pos = residue_ca.pos

    # dataset_centroid_vec: gemmi.Position = inverse_transform.apply(gemmi.Position(marker.x, marker.y, marker.z))
    # dataset_centroid = gemmi.Position(dataset_centroid_vec[0], dataset_centroid_vec[1], dataset_centroid_vec[2])
    tr = transform.transform.vec.tolist()
    dataset_centroid = gemmi.Position(marker.x - tr[0], marker.y - tr[1], marker.z - tr[2])
    print(f"Dataset centriod: {dataset_centroid}")
    # dataset_origin = gemmi.Position(dataset_centroid.x - (grid_size * grid_spacing),
    #                                 dataset_centroid.y - (grid_size * grid_spacing),
    #                                 dataset_centroid.z - (grid_size * grid_spacing))
    # mask.set_points_around(dataset_centroid, radius=10, value=1)

    dataset_centroid_np = np.array([dataset_centroid.x, dataset_centroid.y, dataset_centroid.z])
    box_min = dataset_centroid_np - ((grid_size * grid_spacing) / 2)
    print(f"Box min: {box_min}")
    box_max = dataset_centroid_np + ((grid_size * grid_spacing) / 2)
    print(f"Box max: {box_max}")
    min_pos = gemmi.Position(*box_min)
    max_pos = gemmi.Position(*box_max)
    min_pos_frac = grid.unit_cell.fractionalize(min_pos)
    max_pos_frac = grid.unit_cell.fractionalize(max_pos)
    dataset_centroid_frac = grid.unit_cell.fractionalize(dataset_centroid)
    print(f"min_pos_frac: {min_pos_frac}")
    print(f"max_pos_frac: {max_pos_frac}")
    print(f"dataset_centroid_frac: {dataset_centroid_frac}")

    if (dataset_centroid_frac.x // 1) != (min_pos_frac.x // 1):
        min_pos_frac.x = 0.0
    if (dataset_centroid_frac.y // 1) != (min_pos_frac.y // 1):
        min_pos_frac.y = 0.0
    if (dataset_centroid_frac.z // 1) != (min_pos_frac.z // 1):
        min_pos_frac.z = 0.0

    if (dataset_centroid_frac.x // 1) != (max_pos_frac.x // 1):
        max_pos_frac.x = 1.0 - 0.000001
    if (dataset_centroid_frac.y // 1) != (max_pos_frac.y // 1):
        max_pos_frac.y = 1.0 - 0.000001
    if (dataset_centroid_frac.z // 1) != (max_pos_frac.z // 1):
        max_pos_frac.z = 1.0 - 0.000001

    print(f"min_pos_frac: {min_pos_frac}")
    print(f"max_pos_frac: {max_pos_frac}")

    min_pos_frac_np_mod = np.mod(np.array([min_pos_frac.x, min_pos_frac.y, min_pos_frac.z]), 1)
    print(f"min_pos_frac_np_mod: {min_pos_frac_np_mod}")
    max_pos_frac_np_mod = np.mod(np.array([max_pos_frac.x, max_pos_frac.y, max_pos_frac.z]), 1)
    print(f"max_pos_frac_np_mod: {max_pos_frac_np_mod}")

    min_wrapped_frac = gemmi.Fractional(max(0.0, min_pos_frac_np_mod[0]),
                                        max(0.0, min_pos_frac_np_mod[1]),
                                        max(0.0, min_pos_frac_np_mod[2]), )
    print(f"min_wrapped_frac: {min_wrapped_frac}")

    max_wrapped_frac = gemmi.Fractional(min(1.0, max_pos_frac_np_mod[0]),
                                        min(1.0, max_pos_frac_np_mod[1]),
                                        min(1.0, max_pos_frac_np_mod[2]), )
    print(f"max_wrapped_frac: {max_wrapped_frac}")

    min_wrapped_coord = np.array([min_wrapped_frac.x * grid.nu,
                                  min_wrapped_frac.y * grid.nv,
                                  min_wrapped_frac.z * grid.nw,
                                  ])
    print(f"Min wrapped coord: {min_wrapped_coord}")

    max_wrapped_coord = np.array([max_wrapped_frac.x * grid.nu,
                                  max_wrapped_frac.y * grid.nv,
                                  max_wrapped_frac.z * grid.nw,
                                  ])
    print(f"Max wrapped coord: {max_wrapped_coord}")

    r = gemmi.Transform()
    r.mat.fromlist(transform.transform.inverse().mat.tolist())
    r.vec.fromlist([0.0, 0.0, 0.0])

    # Get indexes of grid points around moving residue
    # mask_array: np.ndarray = np.array(mask, copy=False)
    # indexes: np.ndarray = np.argwhere(mask_array == 1)
    indexes = list(
        itertools.product(
            [x for x in range(int(min_wrapped_coord[0]), int(max_wrapped_coord[0]))],
            [y for y in range(int(min_wrapped_coord[1]), int(max_wrapped_coord[1]))],
            [z for z in range(int(min_wrapped_coord[2]), int(max_wrapped_coord[2]))],
        )
    )
    print(f"Num non-zero indexes: {len(indexes)}")

    fractional_centroid = grid.unit_cell.fractionalize(dataset_centroid)
    wrapped_centroid_frac = gemmi.Fractional(
        fractional_centroid.x % 1,
        fractional_centroid.y % 1,
        fractional_centroid.z % 1,
    )
    wrapped_centroid_orth = grid.unit_cell.orthogonalize(wrapped_centroid_frac)

    # Loop over those indexes, transforming them to grid at origin, assigning 0 to all points outside cell (0,0,0)
    for index in indexes:
        # print(f"Index: {index}")
        # Get the 3d position of the point to sample on the
        index_position: gemmi.Position = grid.point_to_position(grid.get_point(index[0], index[1], index[2]))
        # print(f"index position: {index_position}")
        # Get the position relative to the box centroid
        index_relative_position: gemmi.Position = gemmi.Position(
            index_position.x - wrapped_centroid_orth.x,
            index_position.y - wrapped_centroid_orth.y,
            index_position.z - wrapped_centroid_orth.z,
        )
        # print(f"index_relative_position: {index_relative_position}")
        # Rotate it translate it to reference frame
        # transformed_vec: gemmi.Vec3 = transform.transform.apply(index_relative_position)
        transformed_vec: gemmi.Vec3 = r.apply(index_relative_position)

        # print(f"transformed_vec: {transformed_vec}")
        # transformed_position: gemmi.Position = gemmi.Position(transformed_vec.x - marker.x,
        #                                                       transformed_vec.y - marker.y,
        #                                                       transformed_vec.z - marker.z, )
        transformed_position: gemmi.Position = gemmi.Position(transformed_vec.x,
                                                              transformed_vec.y,
                                                              transformed_vec.z,
                                                              )
        # print(f"transformed_position: {transformed_position}")
        transformed_sample_position = gemmi.Position(
            transformed_position.x + (grid_size * grid_spacing) / 2,
            transformed_position.y + (grid_size * grid_spacing) / 2,
            transformed_position.z + (grid_size * grid_spacing) / 2,
        )
        # print(f"transformed_sample_position: {transformed_sample_position}")
        interpolated_value: float = corrected_density_grid.interpolate_value(transformed_sample_position)
        grid.set_value(index[0], index[1], index[2], interpolated_value)

    return grid


def get_backtransformed_map_mtz(
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
    unit_cell: gemmi.UnitCell = gemmi.UnitCell(grid_size * grid_spacing,
                                               grid_size * grid_spacing,
                                               grid_size * grid_spacing,
                                               90, 90, 90)
    print(f"Corrected density unit cell: {unit_cell}")
    corrected_density_grid.set_unit_cell(unit_cell)
    corrected_density_grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')

    for index, value in np.ndenumerate(corrected_density):
        corrected_density_grid.set_value(index[0], index[1], index[2], value)

    # FFT
    grid: gemmi.FloatGrid = dataset.reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
    grid.fill(0)

    # reference to moving

    # mask
    mask: gemmi.Int8Grid = gemmi.Int8Grid(grid.nu, grid.nv, grid.nw)
    mask.set_unit_cell(grid.unit_cell)
    mask.spacegroup = gemmi.find_spacegroup_by_name('P 1')

    tr = transform.transform.vec.tolist()
    dataset_centroid = gemmi.Position(marker.x - tr[0], marker.y - tr[1], marker.z - tr[2])
    print(f"Dataset centriod: {dataset_centroid}")

    dataset_centroid_np = np.array([dataset_centroid.x, dataset_centroid.y, dataset_centroid.z])
    box_min = dataset_centroid_np - ((grid_size * grid_spacing) / 2)
    print(f"Box min: {box_min}")
    box_max = dataset_centroid_np + ((grid_size * grid_spacing) / 2)
    print(f"Box max: {box_max}")
    min_pos = gemmi.Position(*box_min)
    max_pos = gemmi.Position(*box_max)
    min_pos_frac = grid.unit_cell.fractionalize(min_pos)
    max_pos_frac = grid.unit_cell.fractionalize(max_pos)
    dataset_centroid_frac = grid.unit_cell.fractionalize(dataset_centroid)
    print(f"min_pos_frac: {min_pos_frac}")
    print(f"max_pos_frac: {max_pos_frac}")
    print(f"dataset_centroid_frac: {dataset_centroid_frac}")

    print(f"min_pos_frac: {min_pos_frac}")
    print(f"max_pos_frac: {max_pos_frac}")

    min_wrapped_coord = np.array([min_pos_frac.x * grid.nu,
                                  min_pos_frac.y * grid.nv,
                                  min_pos_frac.z * grid.nw,
                                  ])
    print(f"Min wrapped coord: {min_wrapped_coord}")

    max_wrapped_coord = np.array([max_pos_frac.x * grid.nu,
                                  max_pos_frac.y * grid.nv,
                                  max_pos_frac.z * grid.nw,
                                  ])
    print(f"Max wrapped coord: {max_wrapped_coord}")

    r = gemmi.Transform()
    r.mat.fromlist(transform.transform.inverse().mat.tolist())
    r.vec.fromlist([0.0, 0.0, 0.0])

    # Get indexes of grid points around moving residue
    indexes = list(
        itertools.product(
            [x for x in range(int(min_wrapped_coord[0]), int(max_wrapped_coord[0]))],
            [y for y in range(int(min_wrapped_coord[1]), int(max_wrapped_coord[1]))],
            [z for z in range(int(min_wrapped_coord[2]), int(max_wrapped_coord[2]))],
        )
    )
    print(f"Num non-zero indexes: {len(indexes)}")

    fractional_centroid = grid.unit_cell.fractionalize(dataset_centroid)
    centroid_orth = grid.unit_cell.orthogonalize(fractional_centroid)

    # Loop over those indexes, transforming them to grid at origin, assigning 0 to all points outside cell (0,0,0)
    for index in indexes:
        # Get the 3d position of the point to sample on the
        index_position: gemmi.Position = grid.point_to_position(grid.get_point(index[0], index[1], index[2]))
        # Get the position relative to the box centroid
        index_relative_position: gemmi.Position = gemmi.Position(
            index_position.x - centroid_orth.x,
            index_position.y - centroid_orth.y,
            index_position.z - centroid_orth.z,
        )

        # Rotate it translate it to reference frame
        transformed_vec: gemmi.Vec3 = r.apply(index_relative_position)

        transformed_position: gemmi.Position = gemmi.Position(transformed_vec.x,
                                                              transformed_vec.y,
                                                              transformed_vec.z,
                                                              )
        transformed_sample_position = gemmi.Position(
            transformed_position.x + (grid_size * grid_spacing) / 2,
            transformed_position.y + (grid_size * grid_spacing) / 2,
            transformed_position.z + (grid_size * grid_spacing) / 2,
        )

        interpolated_value: float = corrected_density_grid.interpolate_value(transformed_sample_position)
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


def analyse_dataset(
        dataset: Dataset,
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> Optional[DatasetAffinityResults]:
    # Get a result object
    dataset_results: DatasetAffinityResults = DatasetAffinityResults(
        dataset.dtag,
        marker,
        dataset.structure_path,
        dataset.reflections_path,
        dataset.fragment_path,
    )

    # Get the fragment
    dataset_fragment_structures: Optional[MutableMapping[str, gemmi.Structure]] = dataset.fragment_structures

    # No need to analyse if no fragment present
    if not dataset_fragment_structures:
        return None

    # Select the comparator datasets
    comparator_datasets: Optional[MutableMapping[str, Dataset]] = get_comparator_datasets(
        linkage,
        dataset_clusters,
        dataset_index,
        get_dataset_apo_mask(residue_datasets, known_apos),
        residue_datasets,
        params.min_dataset_cluster_size,
        params.min_dataset_cluster_size,
    )

    if params.debug:
        print(f"\tComparator datasets are: {list(comparator_datasets.keys())}")

    # Handle no comparators
    if not comparator_datasets:
        dataset_results: DatasetAffinityResults = get_not_enough_comparator_dataset_affinity_result(
            dataset,
            marker,
        )
        return dataset_results
        # continue

    # Get the truncated comparator datasets
    comparator_truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        comparator_datasets,
        dataset,
        params.structure_factors,
    )
    comparator_truncated_datasets[dataset.dtag] = dataset
    print(len(comparator_truncated_datasets))

    # Get the local density samples associated with the comparator datasets
    comparator_sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        comparator_truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        params.grid_size,
        params.grid_spacing,
    )
    print(max([comparator_truncated_dataset.reflections.resolution_high() for comparator_truncated_dataset in
               comparator_truncated_datasets.values()]))

    # Get the sample associated with the dataset of interest
    dataset_sample: np.ndarray = comparator_sample_arrays[dataset.dtag]
    del comparator_sample_arrays[dataset.dtag]

    if params.debug:
        print(f"\tGot {len(comparator_sample_arrays)} comparater samples")

    resolution: float = list(comparator_truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Characterise the local distribution
    sample_mean: np.ndarray = get_mean(comparator_sample_arrays)
    sample_std: np.ndarray = get_std(comparator_sample_arrays)
    sample_z: np.ndarray = get_z(dataset_sample, sample_mean, sample_std)
    if params.debug:
        print(f"\tGot mean: max {np.max(sample_mean)}, min: {np.min(sample_mean)}")
        print(f"\tGot std: max {np.max(sample_std)}, min: {np.min(sample_std)}")
        print(f"\tGot z: max {np.max(sample_z)}, min: {np.min(sample_z)}")

    # Get the comparator affinity maps
    downscaled_comparator_samples = {dtag: downscale_local_mean(comparator_sample, (2, 2, 2)) for
                                     dtag, comparator_sample in
                                     comparator_sample_arrays.items()}
    downsampled_dataset_sample = downscale_local_mean(comparator_sample_arrays[dataset.dtag], (2, 2, 2))
    downsampled_sample_mean = downscale_local_mean(sample_mean, (2, 2, 2))
    downsampled_sample_std = downscale_local_mean(sample_std, (2, 2, 2))

    for fragment_id, fragment_structure in dataset_fragment_structures.items():
        if params.debug:
            print(f"\t\tProcessing fragment: {fragment_id}")

        fragment_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = get_fragment_maps(
            fragment_structure,
            resolution,
            params.num_fragment_pose_samples,
            params.sample_rate,
            params.grid_spacing,
        )
        downscaled_fragment_maps = {rotation_index: downscale_local_mean(fragment_map, (2, 2, 2)) for
                                    rotation_index, fragment_map in fragment_maps.items()}

        fragment_masks = {}
        for rotation, fragment_map in downscaled_fragment_maps.items():
            arr = fragment_map.copy()
            mean = np.mean(arr)
            great_mask = arr > mean
            less_mask = arr <= mean
            arr[great_mask] = 1.0
            arr[less_mask] = 0.0
            fragment_masks[rotation] = arr

        if params.debug:
            print(f"\t\tGot {len(fragment_maps)} fragment maps")

        stack = np.stack(([get_z(comparator_sample, downsampled_sample_mean, downsampled_sample_std) for
                           comparator_dtag, comparator_sample in downscaled_comparator_samples.items()]), axis=0)
        print(stack.shape)

        # Get affinity maps for various orientations
        fragment_affinity_z_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = {}
        for rotation_index, fragment_mask in fragment_masks.items():
            if params.debug:
                print(f"\t\t\tProcessing rotation: {rotation_index}")
                print(
                    f"\tGot fragment map: max {np.max(fragment_mask)}, min: {np.min(fragment_mask)}; shape: {fragment_mask.shape}")

            # Get the affinity map for each dataset at this orientation
            fragment_affinity_maps: MutableMapping[str, np.ndarray] = {}

            filters = fragment_mask[np.newaxis, :, :, :]
            print(filters.shape)
            affinity_maps = oaconvolve(stack, filters, mode="same")
            print(affinity_maps.shape)
            fragment_affinity_maps = {dtag: affinity_maps[i, :, :, :] for i, dtag in
                                      enumerate(downscaled_comparator_samples)}
            print([x.shape for x in fragment_affinity_maps.values()])

            # Get the current dataset affinity maps
            dataset_affinity_map = oaconvolve(
                get_z(downsampled_dataset_sample, downsampled_sample_mean, downsampled_sample_std),
                fragment_mask,
                mode="same"
            )

            print(
                f"\tGot dataset affinity map: max {np.max(dataset_affinity_map)}, min: {np.min(dataset_affinity_map)}")

            # Characterise the local distribution of affinity scores
            fragment_affinity_mean: np.ndarray = get_mean(fragment_affinity_maps)
            fragment_affinity_std: np.ndarray = get_std(fragment_affinity_maps)
            fragment_affinity_z: np.ndarray = get_z(
                dataset_affinity_map,
                downsampled_sample_mean,
                downsampled_sample_std,
            )
            fragment_affinity_z_maps[rotation_index] = fragment_affinity_z
            if params.debug:
                print(f"\t\tGot mean: max {np.max(fragment_affinity_mean)}, min: {np.min(fragment_affinity_mean)}")
                print(f"\t\tGot std: max {np.max(fragment_affinity_std)}, min: {np.min(fragment_affinity_std)}")
                print(f"\t\tGot z: max {np.max(fragment_affinity_z)}, min: {np.min(fragment_affinity_z)}")

        # End loop over versions of fragment

        # Get the maxima
        maxima: AffinityMaxima = get_fragment_z_maxima(fragment_affinity_z_maps)
        if params.debug:
            print(f"\t\t\tGot maxima: {maxima}")

        # Check if the maxima is an event: if so
        # if is_affinity_event(maxima, params.min_correlation):

        # Produce the corrected density by subtracting (1-affinity) * ED mean map
        bcd, corrected_density = get_affinity_background_corrected_density(
            dataset_sample,
            fragment_maps[maxima.rotation_index],
            maxima,
            sample_mean,
        )

        # Resample the corrected density onto the original map
        event_map: gemmi.FloatGrid = get_backtransformed_map(
            corrected_density,
            reference_dataset,
            dataset,
            alignments[dataset.dtag][marker],
            marker,
            params.grid_size,
            params.grid_spacing,
            params.structure_factors,
            params.sample_rate,
        )

        # Write the event map
        event_map_path: Path = get_affinity_event_map_path(
            out_dir,
            dataset,
            marker,
        )
        write_event_map(
            event_map,
            event_map_path,
            marker,
            dataset,
        )

        # Record event
        event: AffinityEvent = get_affinity_event(
            dataset,
            maxima,
            marker,
        )

        # else:
        #     # Record a failed event
        #     event: AffinityEvent = get_failed_affinity_event(
        #         dataset,
        #         marker,
        #     )

        # Record the event
        dataset_results.events[0] = event
        break

        # End loop over fragment builds

    return dataset_results


def analyse_residue(
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> MarkerAffinityResults:
    if params.debug:
        print(f"Found {len(residue_datasets)} residue datasets")

    # Truncate the datasets to the same reflections
    truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        residue_datasets,
        reference_dataset,
        params.structure_factors,
    )

    # Truncated dataset apos
    truncated_dataset_apo_mask: np.ndarray = get_dataset_apo_mask(truncated_datasets, known_apos)

    # resolution
    resolution: float = list(truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Sample the datasets to ndarrays
    if params.debug:
        print(f"Getting sample arrays...")
    sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        int(params.grid_size / 2),
        params.grid_spacing * 2,
        #     1.0,
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
    residue_results: MarkerAffinityResults = {}
    for dataset_index, dtag in enumerate(truncated_datasets):
        if params.debug:
            print(f"\tProcessing dataset: {dtag}")

        dataset = residue_datasets[dtag]

        dataset_results: DatasetAffinityResults = analyse_dataset(
            dataset,
            residue_datasets,
            marker,
            alignments,
            reference_dataset,
            linkage,
            dataset_clusters,
            dataset_index,
            known_apos,
            out_dir,
            params,
        )

        # Record the dataset results
        residue_results[dtag] = dataset_results

    # End loop over truncated datasets

    return residue_results
    #
    # # Update the program log
    # pandda_results[residue_id] = residue_results
    # # End loop over residues
    #
    # # Write the summary and graphs of the output
    # write_result_html(pandda_results)


def fragment_search_gpu(xmap_np, fragment_maps_np, fragment_masks_np, mean_map_rscc, min_correlation,
                        max_mean_map_correlation, fragment_size_np, fragment_map_value_list):
    reference_fragment = fragment_maps_np[0, 0, :, :, :]
    print(f"reference_fragment: {reference_fragment.shape}")

    reference_mask = fragment_masks_np[0, 0, :, :, :]
    print(f"reference_mask: {reference_mask.shape}")

    padding = (int((reference_fragment.shape[0]) / 2),
               int((reference_fragment.shape[1]) / 2),
               int((reference_fragment.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    size = torch.tensor(fragment_size_np, dtype=torch.float).cuda()
    print(f"size: {size.shape}")

    reference_map_sum_np = np.array(
        [np.sum(fragment_map_value) for fragment_map_value in fragment_map_value_list]).reshape(
        1,
        len(fragment_map_value_list),
        1,
        1,
        1,
    )
    reference_map_sum = torch.tensor(reference_map_sum_np, dtype=torch.float).cuda()
    print(f"reference_map_sum: {reference_map_sum.shape}")

    # size = torch.tensor(np.sum(reference_mask > 0.0), dtype=torch.float).cuda()
    # print(f"size: {size}")
    #
    # reference_map_masked_values = reference_fragment[reference_mask > 0]
    # print(f"reference_map_masked_values: {reference_map_masked_values.shape}")
    #
    # reference_map_sum = np.sum(reference_map_masked_values)
    # print(f"reference_map_sum: {reference_map_sum}")

    # Tensors
    rho_o = torch.tensor(xmap_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_c = torch.tensor(fragment_maps_np, dtype=torch.float).cuda()
    print(f"rho_c: {rho_c.shape}")

    masks = torch.tensor(fragment_masks_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # Means
    rho_o_mu = torch.nn.functional.conv3d(rho_o, masks, padding=padding)
    print(f"rho_o_mu: {rho_o_mu.shape} {torch.max(rho_o_mu)} {torch.min(rho_o_mu)}")

    # rho_o_mu = (rho_o_mu / size).reshape((1,-1,1,1,1))

    # rho_c_mu = torch.tensor(np.mean(reference_map_masked_values), dtype=torch.float).cuda()
    # print(f"rho_c_mu: {rho_c_mu.shape}; {rho_c_mu}")
    rho_c_mu_np = np.array([np.mean(fragment_map_values) for fragment_map_values in fragment_map_value_list]).reshape(
        len(fragment_map_value_list),
        1,
        1,
        1,
        1,
    )
    rho_c_mu = torch.tensor(rho_c_mu_np, dtype=torch.float).cuda()
    print(f"rho_c_mu: {rho_c_mu.shape}; ")

    rho_c_mu = rho_c_mu.reshape((1, -1, 1, 1, 1))

    # Nominator
    conv_rho_o_rho_c = torch.nn.functional.conv3d(rho_o, rho_c, padding=padding)
    print(
        f"conv_rho_o_rho_c: {conv_rho_o_rho_c.shape} {torch.max(conv_rho_o_rho_c)} {torch.min(conv_rho_o_rho_c)}")

    conv_rho_o_rho_c_mu = torch.nn.functional.conv3d(rho_o, masks, padding=padding) * rho_c_mu
    print(
        f"conv_rho_o_rho_c_mu: {conv_rho_o_rho_c_mu.shape} {torch.max(conv_rho_o_rho_c_mu)} {torch.min(conv_rho_o_rho_c_mu)}")

    conv_rho_o_mu_rho_c = rho_o_mu * reference_map_sum
    print(
        f"conv_rho_o_mu_rho_c: {conv_rho_o_mu_rho_c.shape} {torch.max(conv_rho_o_mu_rho_c)} {torch.min(conv_rho_o_mu_rho_c)}")

    conv_rho_o_mu_rho_c_mu = rho_o_mu * rho_c_mu * size
    print(
        f"conv_rho_o_mu_rho_c_mu: {conv_rho_o_mu_rho_c_mu.shape} {torch.max(conv_rho_o_mu_rho_c_mu)} {torch.min(conv_rho_o_mu_rho_c_mu)}")

    nominator = conv_rho_o_rho_c - conv_rho_o_rho_c_mu - conv_rho_o_mu_rho_c + conv_rho_o_mu_rho_c_mu
    print(
        f"nominator: {nominator.shape} {torch.max(nominator)} {torch.min(nominator)} {nominator[0, 0, 32, 32, 32]}")

    # Denominator
    # # # o
    rho_o_squared = torch.nn.functional.conv3d(torch.square(rho_o), masks, padding=padding)
    print(f"rho_o_squared: {rho_o_squared.shape} {torch.max(rho_o_squared)} {torch.min(rho_o_squared)}")

    conv_rho_o_rho_o_mu = torch.nn.functional.conv3d(rho_o, masks, padding=padding) * rho_o_mu
    print(
        f"conv_rho_o_rho_o_mu: {conv_rho_o_rho_o_mu.shape} {torch.max(conv_rho_o_rho_o_mu)} {torch.min(conv_rho_o_rho_o_mu)}")

    rho_o_mu_squared = torch.square(rho_o_mu) * size
    print(
        f"rho_o_mu_squared: {rho_o_mu_squared.shape} {torch.max(rho_o_mu_squared)} {torch.min(rho_o_mu_squared)}")

    denominator_rho_o = rho_o_squared - 2 * conv_rho_o_rho_o_mu + rho_o_mu_squared
    print(
        f"denominator_rho_o: {denominator_rho_o.shape} {torch.max(denominator_rho_o)} {torch.min(denominator_rho_o)}")

    # # # c
    denominator_rho_c_np = np.array(
        [
            np.sum(np.square(fragment_map_value - np.mean(fragment_map_value)))
            for fragment_map_value
            in fragment_map_value_list
        ]
    ).reshape(
        1,
        len(fragment_map_value_list),
        1,
        1,
        1
    )
    denominator_rho_c = torch.tensor(
        denominator_rho_c_np,
        dtype=torch.float).cuda()
    print(
        f"denominator_rho_c: {denominator_rho_c.shape}; {torch.max(denominator_rho_c)} {torch.min(denominator_rho_c)}")

    # denominator_rho_c = torch.tensor(
    #     np.sum(np.square(reference_map_masked_values - np.mean(reference_map_masked_values))),
    #     dtype=torch.float).cuda()
    # print(
    #     f"denominator_rho_c: {denominator_rho_c.shape}; {torch.max(denominator_rho_c)} {torch.min(denominator_rho_c)}")

    denominator = torch.sqrt(denominator_rho_c) * torch.sqrt(denominator_rho_o)
    print(
        f"denominator: {denominator.shape} {torch.max(denominator)} {torch.min(denominator)} {denominator[0, 0, 32, 32, 32]}")

    rscc = nominator / denominator
    print(f"RSCC: {rscc.shape} {rscc[0, 0, 32, 32, 32]}")

    rscc = torch.nan_to_num(rscc, nan=0.0, posinf=0.0, neginf=0.0, )

    delta_rscc = rscc - mean_map_rscc

    rscc_mask = (rscc > min_correlation)

    mean_map_rscc_mask = (mean_map_rscc < max_mean_map_correlation)

    # delta_rscc[rscc_mask] = 0

    # rscc_mask_float = rscc_mask.float()

    delta_rscc.mul_(rscc_mask)

    delta_rscc.mul_(mean_map_rscc_mask)

    # max_delta_correlation = torch.max(delta_rscc).cpu()
    # print(f"max_delta_correlation: {max_delta_correlation}")

    max_delta_correlation = torch.max(rscc).cpu()
    print(f"max_delta_correlation: {max_delta_correlation}")

    max_index = np.unravel_index(torch.argmax(delta_rscc).cpu(), delta_rscc.shape)

    max_correlation = rscc[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]
    print(f"max_correlation: {max_correlation}")

    print(
        f"max_correlation nominator: {nominator[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]}")
    print(
        f"max_correlation denominator: {denominator[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]}")

    # print(rscc[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]].cpu())

    mean_map_correlation = mean_map_rscc[0, max_index[1], max_index[2], max_index[3], max_index[4]].cpu()

    # max_array = torch.max(delta_rscc, 1)[0].cpu().numpy()
    # print(f"max_array shape: {max_array.shape}")

    del rho_o
    del rho_c
    del masks
    del rho_o_mu
    del rho_c_mu

    del conv_rho_o_rho_c
    del conv_rho_o_rho_c_mu
    del conv_rho_o_mu_rho_c
    del conv_rho_o_mu_rho_c_mu

    del nominator

    del rho_o_squared
    del conv_rho_o_rho_o_mu
    del rho_o_mu_squared

    del denominator_rho_o
    del denominator_rho_c
    del denominator

    del rscc
    del delta_rscc
    # del rscc_mask

    return max_correlation.item(), max_index, mean_map_correlation.item(), max_delta_correlation.item()  # , max_array


def fragment_search_rmsd_scaled_gpu(xmap_np, fragment_maps_np, fragment_masks_np, mean_map_rscc, min_correlation,
                                    max_mean_map_correlation):
    # Reference fragment
    reference_fragment = fragment_maps_np[0, 0, :, :, :]
    print(f"reference_fragment: {reference_fragment.shape}")

    reference_mask = fragment_masks_np[0, 0, :, :, :]
    print(f"reference_mask: {reference_mask.shape}")

    padding = (int((reference_fragment.shape[0]) / 2),
               int((reference_fragment.shape[1]) / 2),
               int((reference_fragment.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    size = torch.tensor(np.sum(reference_mask), dtype=torch.float).cuda()
    print(f"size: {size}")

    reference_map_masked_values = reference_fragment[reference_mask > 0]
    print(f"reference_map_masked_values: {reference_map_masked_values.shape}")

    reference_map_sum = np.sum(reference_map_masked_values)
    print(f"reference_map_sum: {reference_map_sum}")

    # Basic data
    rho_o = torch.tensor(xmap_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_c = torch.tensor(fragment_maps_np, dtype=torch.float).cuda()
    print(f"rho_c: {rho_c.shape}")

    masks = torch.tensor(fragment_masks_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # terms
    rho_o_u = torch.nn.functional.conv3d(rho_o, masks, padding=padding) / size
    rho_c_u = torch.tensor(np.mean(reference_map_masked_values), dtype=torch.float).cuda()

    rho_c_rho_c = torch.sum(torch.square(torch.tensor(reference_map_masked_values, dtype=torch.float).cuda()))
    rho_c_u_rho_c_u = torch.square(rho_c_u) * size
    rho_c_rho_c_u = torch.sum(torch.tensor(reference_map_masked_values, dtype=torch.float).cuda() * rho_c_u)

    rho_o_rho_o = torch.nn.functional.conv3d(torch.square(rho_o), masks, padding=padding)
    rho_o_u_rho_o_u = torch.square(rho_o_u) * size
    rho_o_rho_o_u = torch.nn.functional.conv3d(rho_o, masks, padding=padding) * rho_o_u

    rho_c_rho_o = torch.nn.functional.conv3d(rho_o, rho_c, padding=padding)
    rho_c_u_rho_o_u = rho_o_u * rho_c_u * size
    rho_c_rho_o_u = rho_o_u * reference_map_sum
    rho_c_u_rho_o = torch.nn.functional.conv3d(rho_o, masks, padding=padding) * rho_c_u

    # Sigmas
    sigma_c = torch.sqrt((rho_c_rho_c + rho_c_u_rho_c_u - 2 * rho_c_rho_c_u) / size)
    sigma_o = torch.sqrt((rho_o_rho_o + rho_o_u_rho_o_u - 2 * rho_o_rho_o_u) / size)

    # t1
    t1 = torch.square(1 / sigma_c) * (rho_c_rho_c + rho_c_u_rho_c_u - 2 * rho_c_rho_c_u)

    # t2
    t2 = torch.square(1 / sigma_o) * (rho_o_rho_o + rho_o_u_rho_o_u - 2 * rho_o_rho_o_u)

    # t3
    t3 = 2 * (1 / (sigma_o * sigma_c)) * (rho_c_rho_o + rho_c_u_rho_o_u - rho_c_rho_o_u - rho_c_u_rho_o)

    # Terms
    rmsd = t1 + t2 - t3

    # outliers
    # maxval = torch.max(rmsd)
    rmsd = torch.nan_to_num(rmsd, nan=float('inf'), posinf=float('inf'), neginf=float('inf'), )

    return rmsd


def fragment_search_rmsd_gpu(xmap_np, fragment_maps_np, fragment_masks_np,
                             fragment_size_np,
                             fragment_map_value_list):
    reference_fragment = fragment_maps_np[0, 0, :, :, :]
    print(f"reference_fragment: {reference_fragment.shape}")

    reference_mask = fragment_masks_np[0, 0, :, :, :]
    print(f"reference_mask: {reference_mask.shape}")

    padding = (int((reference_fragment.shape[0]) / 2),
               int((reference_fragment.shape[1]) / 2),
               int((reference_fragment.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    # size = torch.tensor(np.sum(reference_mask), dtype=torch.float).cuda()
    # print(f"size: {size}")

    size = torch.tensor(fragment_size_np, dtype=torch.float).cuda()
    print(f"size: {size.shape} {size[0,0,0,0,0]}")

    reference_map_sum_np = np.array(
        [np.sum(fragment_map_value) for fragment_map_value in fragment_map_value_list]).reshape(
        1,
        len(fragment_map_value_list),
        1,
        1,
        1,
    )
    reference_map_sum = torch.tensor(reference_map_sum_np, dtype=torch.float).cuda()
    print(f"reference_map_sum: {reference_map_sum.shape}")

    # reference_map_masked_values = reference_fragment[reference_mask > 0]
    # print(f"reference_map_masked_values: {reference_map_masked_values.shape}")

    # reference_map_sum = np.sum(reference_map_masked_values)
    # print(f"reference_map_sum: {reference_map_sum}")

    # reference_map_squared_masked_values = np.square(reference_map_masked_values)
    # print(f"reference_map_squared_masked_values: {reference_map_squared_masked_values.shape}")
    #
    # reference_map_squared_sum = np.sum(reference_map_squared_masked_values)
    # print(f"reference_map_squared_masked_values: {reference_map_squared_masked_values}")

    rho_c_rho_c_np = np.array(
        [
            np.sum(np.square(fragment_map_value))
            for fragment_map_value
            in fragment_map_value_list
        ]
    ).reshape(
        1,
        len(fragment_map_value_list),
        1,
        1,
        1
    )
    rho_c_rho_c = torch.tensor(
        rho_c_rho_c_np,
        dtype=torch.float).cuda()
    print(
        f"rho_c_rho_c: {rho_c_rho_c.shape}; {torch.max(rho_c_rho_c)} {torch.min(rho_c_rho_c)} {rho_c_rho_c[0,0,0,0,0]}")

    # Tensors
    rho_o = torch.tensor(xmap_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_c = torch.tensor(fragment_maps_np, dtype=torch.float).cuda()
    print(f"rho_c: {rho_c.shape}")

    masks = torch.tensor(fragment_masks_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # Convolutions
    conv_rho_o_rho_c = torch.nn.functional.conv3d(rho_o, rho_c, padding=padding)
    print(
        f"conv_rho_o_rho_c: {conv_rho_o_rho_c.shape} {torch.max(conv_rho_o_rho_c)} {torch.min(conv_rho_o_rho_c)} {conv_rho_o_rho_c[0,0,24,24,24]}")

    rho_o_squared = torch.nn.functional.conv3d(torch.square(rho_o), masks, padding=padding)
    print(f"rho_o_squared: {rho_o_squared.shape} {torch.max(rho_o_squared)} {torch.min(rho_o_squared)} {rho_o_squared[0,0,24,24,24]}")

    rmsd_unsacled = (rho_o_squared + rho_c_rho_c - 2 * conv_rho_o_rho_c)
    print(f"rmsd: {rmsd_unsacled.shape} {rmsd_unsacled[0, 0, 24, 24, 24]}")

    rmsd = rmsd_unsacled / size
    print(f"rmsd: {rmsd.shape} {rmsd[0, 0, 24, 24, 24]}")

    rmsd = torch.nan_to_num(rmsd, nan=0.0, posinf=0.0, neginf=0.0, )

    del rho_o
    del rho_c
    del masks

    del conv_rho_o_rho_c

    del rho_o_squared

    return rmsd


def fragment_search_mask_gpu(xmap_np, fragment_masks_np, cutoff):
    reference_mask = fragment_masks_np[0, 0, :, :, :]
    print(f"reference_mask: {reference_mask.shape}")

    size = torch.tensor(np.sum(reference_mask), dtype=torch.float).cuda()
    print(f"size: {size}")

    padding = (int((reference_mask.shape[0]) / 2),
               int((reference_mask.shape[1]) / 2),
               int((reference_mask.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    # Tensors
    rho_o = torch.tensor(xmap_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_o_greater_cutoff = rho_o > cutoff

    rho_o_less_cutoff = rho_o < cutoff

    rho_o[rho_o_greater_cutoff] = 1.0

    rho_o[rho_o_less_cutoff] = 0.0

    masks = torch.tensor(fragment_masks_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # Convolutions
    conv_rho_o_mask_overlap = torch.nn.functional.conv3d(rho_o, masks, padding=padding) / size
    print(
        f"conv_rho_o_rho_c: {conv_rho_o_mask_overlap.shape} {torch.max(conv_rho_o_mask_overlap)} {torch.min(conv_rho_o_mask_overlap)}")

    print(f"RSCC: {conv_rho_o_mask_overlap.shape} {conv_rho_o_mask_overlap[0, 0, 32, 32, 32]}")

    conv_rho_o_mask_overlap = torch.nan_to_num(conv_rho_o_mask_overlap, nan=0.0, posinf=0.0, neginf=0.0, )

    del rho_o
    del rho_o_greater_cutoff
    del rho_o_less_cutoff
    del masks

    return conv_rho_o_mask_overlap


def fragment_search_mask_unnormalised_gpu(xmap_np, fragment_masks_np, cutoff):
    reference_mask = fragment_masks_np[0, 0, :, :, :]
    print(f"reference_mask: {reference_mask.shape}")

    size = torch.tensor(np.sum(reference_mask), dtype=torch.float).cuda()
    print(f"size: {size}")

    padding = (int((reference_mask.shape[0]) / 2),
               int((reference_mask.shape[1]) / 2),
               int((reference_mask.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    # Tensors
    rho_o = torch.tensor(xmap_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_o_greater_cutoff = rho_o > cutoff

    rho_o_less_cutoff = rho_o < cutoff

    rho_o[rho_o_greater_cutoff] = 1.0

    rho_o[rho_o_less_cutoff] = 0.0

    masks = torch.tensor(fragment_masks_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # Convolutions
    conv_rho_o_mask_overlap = torch.nn.functional.conv3d(rho_o, masks, padding=padding)
    print(
        f"conv_rho_o_rho_c: {conv_rho_o_mask_overlap.shape} {torch.max(conv_rho_o_mask_overlap)} {torch.min(conv_rho_o_mask_overlap)}")

    print(f"RSCC: {conv_rho_o_mask_overlap.shape} {conv_rho_o_mask_overlap[0, 0, 32, 32, 32]}")

    conv_rho_o_mask_overlap = torch.nan_to_num(conv_rho_o_mask_overlap, nan=0.0, posinf=0.0, neginf=0.0, )

    del rho_o
    del rho_o_greater_cutoff
    del rho_o_less_cutoff
    del masks

    return conv_rho_o_mask_overlap


def peak_search(reference_map, target_map):
    delta_map = target_map - reference_map

    max_delta = torch.min(delta_map).cpu()
    print(f"max_delta: {max_delta}")

    max_index = np.unravel_index(torch.argmin(delta_map).cpu(), delta_map.shape)

    max_map_val = target_map[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]
    print(f"max_map_val: {max_map_val}")

    reference_map_val = reference_map[0, max_index[1], max_index[2], max_index[3], max_index[4]].cpu()

    return max_map_val.item(), max_index, reference_map_val.item(), max_delta.item()


def peak_search_mask_dep(reference_map, target_map):
    delta_map = target_map - reference_map

    max_delta = torch.max(delta_map).cpu()
    print(f"max_delta: {max_delta}")

    max_index = np.unravel_index(torch.argmax(delta_map).cpu(), delta_map.shape)

    max_map_val = target_map[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]
    print(f"max_map_val: {max_map_val}")

    reference_map_val = reference_map[0, max_index[1], max_index[2], max_index[3], max_index[4]].cpu()

    return max_map_val.item(), max_index, reference_map_val.item(), max_delta.item()


def peak_search_mask(target_map):
    max_delta = 0.0
    print(f"max_delta: {max_delta}")

    max_index = np.unravel_index(torch.argmax(target_map).cpu(), target_map.shape)

    max_map_val = target_map[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]
    print(f"max_map_val: {max_map_val}")

    return [max_map_val.item(), max_index, 0.0, 0.0]


def peak_search_rmsd(target_map):
    max_delta = 0.0
    print(f"max_delta: {max_delta}")

    min_index = np.unravel_index(torch.argmin(target_map).cpu(), target_map.shape)

    min_map_val = target_map[min_index[0], min_index[1], min_index[2], min_index[3], min_index[4]]
    print(f"min_map_val: {min_map_val}")

    return [min_map_val.item(), min_index, 0.0, 0.0]


def get_mean_rscc(sample_mean, fragment_maps_np, fragment_masks_np):
    # Get mean map RSCC
    mean_map_np = np.stack([sample_mean], axis=0)
    mean_map_np = mean_map_np.reshape(
        1,
        1,
        mean_map_np.shape[1],
        mean_map_np.shape[2],
        mean_map_np.shape[3])
    print(f"mean_map_np: {mean_map_np.shape}")

    reference_fragment = fragment_maps_np[0, 0, :, :, :]
    print(f"reference_fragment: {reference_fragment.shape}")

    reference_mask = fragment_masks_np[0, 0, :, :, :]
    print(f"reference_mask: {reference_mask.shape}")

    padding = (int((reference_fragment.shape[0]) / 2),
               int((reference_fragment.shape[1]) / 2),
               int((reference_fragment.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    size = torch.tensor(np.sum(reference_mask), dtype=torch.float).cuda()
    print(f"size: {size}")

    reference_map_masked_values = reference_fragment[reference_mask > 0]
    print(f"reference_map_masked_values: {reference_map_masked_values.shape}")

    reference_map_sum = np.sum(reference_map_masked_values)
    print(f"reference_map_sum: {reference_map_sum}")

    # Tensors
    rho_o = torch.tensor(mean_map_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_c = torch.tensor(fragment_maps_np, dtype=torch.float).cuda()
    print(f"rho_c: {rho_c.shape}")

    masks = torch.tensor(fragment_masks_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # Tensors
    rho_o = torch.tensor(mean_map_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_c = torch.tensor(fragment_maps_np, dtype=torch.float).cuda()
    print(f"rho_c: {rho_c.shape}")

    masks = torch.tensor(fragment_masks_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # Means
    rho_o_mu = torch.nn.functional.conv3d(rho_o, masks, padding=padding) / size
    print(f"rho_o_mu: {rho_o_mu.shape} {torch.max(rho_o_mu)} {torch.min(rho_o_mu)}")

    rho_c_mu = torch.tensor(np.mean(reference_map_masked_values), dtype=torch.float).cuda()
    print(f"rho_c_mu: {rho_c_mu.shape}; {rho_c_mu}")

    # Nominator
    conv_rho_o_rho_c = torch.nn.functional.conv3d(rho_o, rho_c, padding=padding)
    print(
        f"conv_rho_o_rho_c: {conv_rho_o_rho_c.shape} {torch.max(conv_rho_o_rho_c)} {torch.min(conv_rho_o_rho_c)}")

    conv_rho_o_rho_c_mu = torch.nn.functional.conv3d(rho_o, masks, padding=padding) * rho_c_mu
    print(
        f"conv_rho_o_rho_c_mu: {conv_rho_o_rho_c_mu.shape} {torch.max(conv_rho_o_rho_c_mu)} {torch.min(conv_rho_o_rho_c_mu)}")

    conv_rho_o_mu_rho_c = rho_o_mu * reference_map_sum
    print(
        f"conv_rho_o_mu_rho_c: {conv_rho_o_mu_rho_c.shape} {torch.max(conv_rho_o_mu_rho_c)} {torch.min(conv_rho_o_mu_rho_c)}")

    conv_rho_o_mu_rho_c_mu = rho_o_mu * rho_c_mu * size
    print(
        f"conv_rho_o_mu_rho_c_mu: {conv_rho_o_mu_rho_c_mu.shape} {torch.max(conv_rho_o_mu_rho_c_mu)} {torch.min(conv_rho_o_mu_rho_c_mu)}")

    nominator = conv_rho_o_rho_c - conv_rho_o_rho_c_mu - conv_rho_o_mu_rho_c + conv_rho_o_mu_rho_c_mu
    print(
        f"nominator: {nominator.shape} {torch.max(nominator)} {torch.min(nominator)} {nominator[0, 0, 32, 32, 32]}")

    # Denominator
    # # # o
    rho_o_squared = torch.nn.functional.conv3d(torch.square(rho_o), masks, padding=padding)
    print(f"rho_o_squared: {rho_o_squared.shape} {torch.max(rho_o_squared)} {torch.min(rho_o_squared)}")

    conv_rho_o_rho_o_mu = torch.nn.functional.conv3d(rho_o, masks, padding=padding) * rho_o_mu
    print(
        f"conv_rho_o_rho_o_mu: {conv_rho_o_rho_o_mu.shape} {torch.max(conv_rho_o_rho_o_mu)} {torch.min(conv_rho_o_rho_o_mu)}")

    rho_o_mu_squared = torch.square(rho_o_mu) * size
    print(
        f"rho_o_mu_squared: {rho_o_mu_squared.shape} {torch.max(rho_o_mu_squared)} {torch.min(rho_o_mu_squared)}")

    denominator_rho_o = rho_o_squared - 2 * conv_rho_o_rho_o_mu + rho_o_mu_squared
    print(
        f"denominator_rho_o: {denominator_rho_o.shape} {torch.max(denominator_rho_o)} {torch.min(denominator_rho_o)}")

    # # # c
    denominator_rho_c = torch.tensor(
        np.sum(np.square(reference_map_masked_values - np.mean(reference_map_masked_values))),
        dtype=torch.float).cuda()
    print(
        f"denominator_rho_c: {denominator_rho_c.shape}; {torch.max(denominator_rho_c)} {torch.min(denominator_rho_c)}")

    denominator = torch.sqrt(denominator_rho_c) * torch.sqrt(denominator_rho_o)
    print(
        f"denominator: {denominator.shape} {torch.max(denominator)} {torch.min(denominator)} {denominator[0, 0, 32, 32, 32]}")

    mean_map_rscc = nominator / denominator
    print(f"mean_map_rscc: {mean_map_rscc.shape} {mean_map_rscc[0, 0, 32, 32, 32]}")

    mean_map_rscc = torch.nan_to_num(mean_map_rscc, nan=0.0, posinf=0.0, neginf=0.0, )

    mean_map_max_correlation = torch.max(mean_map_rscc).cpu()

    max_index = np.unravel_index(torch.argmax(mean_map_rscc).cpu(), mean_map_rscc.shape)

    print(
        f"max_correlation nominator: {nominator[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]}")
    print(
        f"max_correlation denominator: {denominator[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]}")

    print(f"mean_map_max_correlation: {mean_map_max_correlation}")
    del rho_o
    del rho_c
    del masks
    del rho_o_mu
    del rho_c_mu

    del conv_rho_o_rho_c
    del conv_rho_o_rho_c_mu
    del conv_rho_o_mu_rho_c
    del conv_rho_o_mu_rho_c_mu

    del nominator

    del rho_o_squared
    del conv_rho_o_rho_o_mu
    del rho_o_mu_squared

    del denominator_rho_o
    del denominator_rho_c
    del denominator

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()

    return mean_map_rscc


def save_example_fragment_map(fragment_map, grid_spacing, path):
    shape = fragment_map.shape
    grid = gemmi.FloatGrid(*shape)
    unit_cell = gemmi.UnitCell(
        shape[0] * grid_spacing,
        shape[1] * grid_spacing,
        shape[2] * grid_spacing,
        90,
        90,
        90
    )
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    grid.set_unit_cell(unit_cell)

    for index, value in np.ndenumerate(fragment_map):
        grid.set_value(index[0], index[1], index[2], value)

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.setup()
    ccp4.update_ccp4_header(2, True)

    ccp4.write_ccp4_map(str(path))


def max_coord_to_position(max_index_mask_coord, fragment_maps, max_rotation, grid_size, grid_spacing, max_x, max_y,
                          max_z, alignments, dataset, marker):
    # max_index_fragment_coord = [
    #     max_index_mask_coord[0] - (max_x / 2) + (fragment_maps[max_rotation].shape[0] / 2),
    #     max_index_mask_coord[1] - (max_y / 2) + (fragment_maps[max_rotation].shape[1] / 2),
    #     max_index_mask_coord[2] - (max_z / 2) + (fragment_maps[max_rotation].shape[2] / 2),
    # ]
    # print(f"max_index_fragment_coord: {max_index_fragment_coord}")
    #
    # max_index_fragment_relative_coord = [max_index_fragment_coord[0] - grid_size / 2,
    #                                      max_index_fragment_coord[1] - grid_size / 2,
    #                                      max_index_fragment_coord[2] - grid_size / 2,
    #                                      ]
    # print(f"max_index_fragment_relative_coord: {max_index_fragment_relative_coord}")
    #
    # max_index_fragment_relative_position = gemmi.Position(
    #     max_index_fragment_relative_coord[0] * grid_spacing,
    #     max_index_fragment_relative_coord[1] * grid_spacing,
    #     max_index_fragment_relative_coord[2] * grid_spacing,
    # )
    # print(f"max_index_fragment_relative_position: {max_index_fragment_relative_position}")
    #
    # transform = alignments[dataset.dtag][marker].transform
    # inverse_transform = transform.inverse()
    # rotation_tr = gemmi.Transform()
    # rotation_tr.mat.fromlist(inverse_transform.mat.tolist())
    #
    # max_index_fragment_relative_position_dataset_frame = rotation_tr.apply(max_index_fragment_relative_position)
    # print(
    #     f"max_index_fragment_relative_position_dataset_frame: {max_index_fragment_relative_position_dataset_frame}")
    #
    # max_index_fragment_position_dataset_frame = [
    #     max_index_fragment_relative_position_dataset_frame.x + (marker.x - transform.vec.x),
    #     max_index_fragment_relative_position_dataset_frame.y + (marker.y - transform.vec.y),
    #     max_index_fragment_relative_position_dataset_frame.z + (marker.z - transform.vec.z),
    # ]
    # print(f"max_index_fragment_position_dataset_frame: {max_index_fragment_position_dataset_frame}")

    max_index_fragment_coord = [
        max_index_mask_coord[0] - (max_x / 2) + (fragment_maps[max_rotation].shape[0] / 2),
        max_index_mask_coord[1] - (max_y / 2) + (fragment_maps[max_rotation].shape[1] / 2),
        max_index_mask_coord[2] - (max_z / 2) + (fragment_maps[max_rotation].shape[2] / 2),
    ]
    print(f"max_index_fragment_coord: {max_index_fragment_coord}")

    max_index_fragment_relative_coord = [max_index_fragment_coord[0] - grid_size / 2,
                                         max_index_fragment_coord[1] - grid_size / 2,
                                         max_index_fragment_coord[2] - grid_size / 2,
                                         ]
    print(f"max_index_fragment_relative_coord: {max_index_fragment_relative_coord}")

    max_index_fragment_relative_position = gemmi.Position(
        max_index_fragment_relative_coord[0] * grid_spacing,
        max_index_fragment_relative_coord[1] * grid_spacing,
        max_index_fragment_relative_coord[2] * grid_spacing,
    )
    print(f"max_index_fragment_relative_position: {max_index_fragment_relative_position}")

    transform = alignments[dataset.dtag][marker].transform
    inverse_transform = transform.inverse()
    rotation_tr = gemmi.Transform()
    rotation_tr.mat.fromlist(inverse_transform.mat.tolist())

    max_index_fragment_relative_position_dataset_frame = rotation_tr.apply(max_index_fragment_relative_position)
    print(
        f"max_index_fragment_relative_position_dataset_frame: {max_index_fragment_relative_position_dataset_frame}")

    max_index_fragment_position_dataset_frame = [
        max_index_fragment_relative_position_dataset_frame.x + (marker.x - transform.vec.x),
        max_index_fragment_relative_position_dataset_frame.y + (marker.y - transform.vec.y),
        max_index_fragment_relative_position_dataset_frame.z + (marker.z - transform.vec.z),
    ]
    print(f"max_index_fragment_position_dataset_frame: {max_index_fragment_position_dataset_frame}")

    return max_index_fragment_position_dataset_frame


def get_protein_scaling(dataset: Dataset, structure_factors, sample_rate):
    structure = dataset.structure

    xmap = dataset.reflections.transform_f_phi_to_map(
        structure_factors.f,
        structure_factors.phi,
        sample_rate=sample_rate,
    )
    xmap_np = np.array(xmap)

    mask = gemmi.FloatGrid(xmap.nu, xmap.nv, xmap.nw)
    mask.set_unit_cell(xmap.unit_cell)

    for model in structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    mask.set_points_around(atom.pos, 0.75, 1.0)

    mask_np = np.array(mask)

    masked_points = xmap_np[mask_np == 1.0]

    location = np.mean(masked_points)
    scale = np.std(masked_points)
    print(f"scaling is: {scale}")

    return location, scale


def analyse_dataset_gpu(
        dataset: Dataset,
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> Optional[DatasetAffinityResults]:
    # Get the fragment
    dataset_fragment_structures: Optional[MutableMapping[str, gemmi.Structure]] = dataset.fragment_structures

    # No need to analyse if no fragment present
    if not dataset_fragment_structures:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )

    # Select the comparator datasets
    comparator_datasets: Optional[MutableMapping[str, Dataset]] = get_comparator_datasets(
        linkage,
        dataset_clusters,
        dataset_index,
        dataset.dtag,
        get_dataset_apo_mask(residue_datasets, known_apos),
        residue_datasets,
        params.min_dataset_cluster_size,
        params.min_dataset_cluster_size,
    )

    if params.debug:
        print(f"\tComparator datasets are: {list(comparator_datasets.keys())}")

    # Handle no comparators
    if not comparator_datasets:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )
        # continue

    # Get the truncated comparator datasets
    comparator_truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        comparator_datasets,
        dataset,
        params.structure_factors,
    )
    comparator_truncated_datasets[dataset.dtag] = dataset
    print(len(comparator_truncated_datasets))

    # Get the local density samples associated with the comparator datasets
    comparator_sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        comparator_truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        params.grid_size,
        params.grid_spacing,
    )
    print(max([comparator_truncated_dataset.reflections.resolution_high() for comparator_truncated_dataset in
               comparator_truncated_datasets.values()]))

    # Get the sample associated with the dataset of interest
    dataset_sample: np.ndarray = comparator_sample_arrays[dataset.dtag]
    del comparator_sample_arrays[dataset.dtag]

    if params.debug:
        print(f"\tGot {len(comparator_sample_arrays)} comparater samples")

    resolution: float = list(comparator_truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Characterise the local distribution
    sample_mean: np.ndarray = get_mean(comparator_sample_arrays)
    sample_std: np.ndarray = get_std(comparator_sample_arrays)
    sample_z: np.ndarray = get_z(dataset_sample, sample_mean, sample_std)
    if params.debug:
        print(f"\tGot mean: max {np.max(sample_mean)}, min: {np.min(sample_mean)}")
        print(f"\tGot std: max {np.max(sample_std)}, min: {np.min(sample_std)}")
        print(f"\tGot z: max {np.max(sample_z)}, min: {np.min(sample_z)}")

    # Get the comparator affinity maps
    for fragment_id, fragment_structure in dataset_fragment_structures.items():
        if params.debug:
            print(f"\t\tProcessing fragment: {fragment_id}")

        fragment_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = get_fragment_maps(
            fragment_structure,
            resolution,
            params.num_fragment_pose_samples,
            params.sample_rate,
            params.grid_spacing,
        )

        save_example_fragment_map(list(fragment_maps.values())[0], params.grid_spacing,
                                  out_dir / "example_fragment_map.ccp4")

        max_x = max([fragment_map.shape[0] for fragment_map in fragment_maps.values()])
        max_y = max([fragment_map.shape[1] for fragment_map in fragment_maps.values()])
        max_z = max([fragment_map.shape[2] for fragment_map in fragment_maps.values()])
        if max_x % 2 == 0: max_x = max_x + 1
        if max_y % 2 == 0: max_y = max_y + 1
        if max_z % 2 == 0: max_z = max_z + 1

        fragment_masks_list = []

        fragment_maps_list = []
        fragment_masks = {}
        for rotation, fragment_map in fragment_maps.items():
            arr = fragment_map.copy()

            arr_mask = fragment_map > 0.0

            print(f"arr_mask: {np.sum(arr_mask)}")

            arr[~arr_mask] = 0.0

            fragment_mask_arr = np.zeros(fragment_map.shape)
            fragment_mask_arr[arr_mask] = 1.0

            fragment_map = np.zeros((max_x, max_y, max_z,))
            fragment_mask = np.zeros((max_x, max_y, max_z,))

            fragment_map[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr[:, :, :]
            fragment_mask[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = fragment_mask_arr[:, :, :]

            fragment_maps_list.append(fragment_map)
            fragment_masks_list.append(fragment_mask)

            fragment_masks[rotation] = fragment_mask

        if params.debug:
            print(f"\t\tGot {len(fragment_maps)} fragment maps")

        # Get affinity maps for various orientations
        event_map_list = []
        bdcs = np.linspace(0, 0.90, 10)
        for b in bdcs:
            event_map = (dataset_sample - (b * sample_mean)) / (1 - b)

            event_map_list.append(event_map)

        with torch.no_grad():

            # Fragment maps
            fragment_maps_np = np.stack(fragment_maps_list, axis=0)
            fragment_maps_np = fragment_maps_np.reshape(fragment_maps_np.shape[0],
                                                        1,
                                                        fragment_maps_np.shape[1],
                                                        fragment_maps_np.shape[2],
                                                        fragment_maps_np.shape[3])
            print(f"fragment_maps_np: {fragment_maps_np.shape}")

            fragment_masks_np = np.stack(fragment_masks_list, axis=0)
            fragment_masks_np = fragment_masks_np.reshape(fragment_masks_np.shape[0],
                                                          1,
                                                          fragment_masks_np.shape[1],
                                                          fragment_masks_np.shape[2],
                                                          fragment_masks_np.shape[3])
            print(f"fragment_masks_np: {fragment_masks_np.shape}")

            mean_map_rscc = get_mean_rscc(sample_mean, fragment_maps_np, fragment_masks_np)
            mean_map_max_correlation = torch.max(mean_map_rscc).cpu().item()

            rsccs = {}
            for b_index in range(len(event_map_list)):
                print(f"\tBDC: {bdcs[b_index]}")
                event_maps_np = np.stack([event_map_list[b_index]], axis=0)
                event_maps_np = event_maps_np.reshape(event_maps_np.shape[0],
                                                      1,
                                                      event_maps_np.shape[1],
                                                      event_maps_np.shape[2],
                                                      event_maps_np.shape[3])
                print(f"event_maps_np: {event_maps_np.shape}")

                rsccs[bdcs[b_index]] = fragment_search_gpu(event_maps_np, fragment_maps_np, fragment_masks_np,
                                                           mean_map_rscc, 0.5, 0.4)

                max_index = rsccs[bdcs[b_index]][1]
                max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
                max_rotation = list(fragment_maps.keys())[max_index[1]]
                max_position = max_coord_to_position(
                    max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing,
                    max_x,
                    max_y,
                    max_z,
                    alignments, dataset, marker)

                print(f"max position: {max_position}")

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass

            max_rscc_bdc = max(rsccs, key=lambda x: rsccs[x][0])
            max_rscc_correlation_index = rsccs[max_rscc_bdc]
            max_correlation = max_rscc_correlation_index[0]
            max_index = max_rscc_correlation_index[1]
            max_mean_map_correlation = max_rscc_correlation_index[2]
            max_delta_correlation = max_rscc_correlation_index[3]

            max_bdc = max_rscc_bdc
            max_rotation = list(fragment_maps.keys())[max_index[1]]
            max_index_fragment_map = fragment_maps[max_rotation]
            max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
            max_index_fragment_map_shape = max_index_fragment_map.shape

            max_index_fragment_position_dataset_frame = max_coord_to_position(
                max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing, max_x, max_y,
                max_z,
                alignments, dataset, marker
            )

            # get affinity maxima
            maxima: AffinityMaxima = AffinityMaxima(
                index=max_index,
                correlation=max_correlation,
                rotation_index=max_rotation,
                position=max_index_fragment_position_dataset_frame,
                bdc=max_bdc,
                mean_map_correlation=max_mean_map_correlation,
                mean_map_max_correlation=mean_map_max_correlation,
                max_delta_correlation=max_delta_correlation,
            )
            print(maxima)

            # if max_correlation > params.min_correlation:
            event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
                (dataset_sample - (maxima.bdc * sample_mean)) / (1 - maxima.bdc),
                # max_array[0,:,:,:],
                reference_dataset,
                dataset,
                alignments[dataset.dtag][marker],
                marker,
                params.grid_size,
                params.grid_spacing,
                params.structure_factors,
                params.sample_rate,
            )

            dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
                                          marker.y - alignments[dataset.dtag][marker].transform.vec.y,
                                          marker.z - alignments[dataset.dtag][marker].transform.vec.z,
                                          None,
                                          )

            write_event_map(
                event_map,
                out_dir / f"{dataset.dtag}_{max_index_fragment_position_dataset_frame[0]}_{max_index_fragment_position_dataset_frame[1]}_{max_index_fragment_position_dataset_frame[2]}_{fragment_id}.mtz",
                dataset_event_marker,
                dataset,
                resolution,
            )

    # End loop over fragment builds

    # Get a result object
    dataset_results: DatasetAffinityResults = DatasetAffinityResults(
        dataset.dtag,
        marker,
        dataset.structure_path,
        dataset.reflections_path,
        dataset.fragment_path,
        maxima,
    )

    return dataset_results


def analyse_dataset_b_factor_gpu(
        dataset: Dataset,
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> Optional[DatasetAffinityResults]:
    # Get the fragment
    dataset_fragment_structures: Optional[MutableMapping[str, gemmi.Structure]] = dataset.fragment_structures

    # No need to analyse if no fragment present
    if not dataset_fragment_structures:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )

    # Select the comparator datasets
    comparator_datasets: Optional[MutableMapping[str, Dataset]] = get_comparator_datasets(
        linkage,
        dataset_clusters,
        dataset_index,
        dataset.dtag,
        get_dataset_apo_mask(residue_datasets, known_apos),
        residue_datasets,
        params.min_dataset_cluster_size,
        params.min_dataset_cluster_size,
    )

    if params.debug:
        print(f"\tComparator datasets are: {list(comparator_datasets.keys())}")

    # Handle no comparators
    if not comparator_datasets:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )
        # continue

    # Get the truncated comparator datasets
    comparator_truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        comparator_datasets,
        dataset,
        params.structure_factors,
    )
    comparator_truncated_datasets[dataset.dtag] = dataset
    print(len(comparator_truncated_datasets))

    # Get the local density samples associated with the comparator datasets
    comparator_sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        comparator_truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        params.grid_size,
        params.grid_spacing,
    )
    print(max([comparator_truncated_dataset.reflections.resolution_high() for comparator_truncated_dataset in
               comparator_truncated_datasets.values()]))

    # Get the sample associated with the dataset of interest
    dataset_sample: np.ndarray = comparator_sample_arrays[dataset.dtag]
    del comparator_sample_arrays[dataset.dtag]

    if params.debug:
        print(f"\tGot {len(comparator_sample_arrays)} comparater samples")

    resolution: float = list(comparator_truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Characterise the local distribution
    sample_mean: np.ndarray = get_mean(comparator_sample_arrays)
    sample_std: np.ndarray = get_std(comparator_sample_arrays)
    sample_z: np.ndarray = get_z(dataset_sample, sample_mean, sample_std)
    if params.debug:
        print(f"\tGot mean: max {np.max(sample_mean)}, min: {np.min(sample_mean)}")
        print(f"\tGot std: max {np.max(sample_std)}, min: {np.min(sample_std)}")
        print(f"\tGot z: max {np.max(sample_z)}, min: {np.min(sample_z)}")

    # Get the comparator affinity maps

    results = []
    for b_factor in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0):
        print(f"############ B FACTOR: {b_factor} ########")
        for fragment_id, fragment_structure in dataset_fragment_structures.items():
            if params.debug:
                print(f"\t\tProcessing fragment: {fragment_id}")

            fragment_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = get_fragment_maps(
                fragment_structure,
                resolution,
                params.num_fragment_pose_samples,
                params.sample_rate,
                params.grid_spacing,
                b_factor
            )

            save_example_fragment_map(list(fragment_maps.values())[0], params.grid_spacing,
                                      out_dir / "example_fragment_map.ccp4")

            max_x = max([fragment_map.shape[0] for fragment_map in fragment_maps.values()])
            max_y = max([fragment_map.shape[1] for fragment_map in fragment_maps.values()])
            max_z = max([fragment_map.shape[2] for fragment_map in fragment_maps.values()])
            if max_x % 2 == 0: max_x = max_x + 1
            if max_y % 2 == 0: max_y = max_y + 1
            if max_z % 2 == 0: max_z = max_z + 1

            fragment_masks_list = []

            fragment_maps_list = []
            fragment_map_size_list = []
            fragment_map_value_list = []
            fragment_masks = {}
            for rotation, fragment_map in fragment_maps.items():
                arr = fragment_map.copy()

                arr_mask = fragment_map > 0.0

                fragment_map_size_list.append(np.sum(arr_mask))

                # print(f"arr_mask: {np.sum(arr_mask)}")

                arr[~arr_mask] = 0.0
                fragment_map_value_list.append(arr[arr_mask])

                fragment_mask_arr = np.zeros(fragment_map.shape)
                fragment_mask_arr[arr_mask] = 1.0

                fragment_map = np.zeros((max_x, max_y, max_z,))
                fragment_mask = np.zeros((max_x, max_y, max_z,))

                fragment_map[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr[:, :, :]
                fragment_mask[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = fragment_mask_arr[:, :, :]

                fragment_maps_list.append(fragment_map)
                fragment_masks_list.append(fragment_mask)

                fragment_masks[rotation] = fragment_mask

            if params.debug:
                print(f"\t\tGot {len(fragment_maps)} fragment maps")

            # Get affinity maps for various orientations
            event_map_list = []
            bdcs = np.linspace(0, 0.90, 10)
            for b in bdcs:
                event_map = (dataset_sample - (b * sample_mean)) / (1 - b)

                event_map_list.append(event_map)

            with torch.no_grad():

                # Fragment maps
                fragment_maps_np = np.stack(fragment_maps_list, axis=0)
                fragment_maps_np = fragment_maps_np.reshape(fragment_maps_np.shape[0],
                                                            1,
                                                            fragment_maps_np.shape[1],
                                                            fragment_maps_np.shape[2],
                                                            fragment_maps_np.shape[3])
                print(f"fragment_maps_np: {fragment_maps_np.shape}")

                fragment_masks_np = np.stack(fragment_masks_list, axis=0)
                fragment_masks_np = fragment_masks_np.reshape(fragment_masks_np.shape[0],
                                                              1,
                                                              fragment_masks_np.shape[1],
                                                              fragment_masks_np.shape[2],
                                                              fragment_masks_np.shape[3])
                print(f"fragment_masks_np: {fragment_masks_np.shape}")

                fragment_size_np = np.array(fragment_map_size_list).reshape(
                    1,
                    len(fragment_map_size_list),
                    1,
                    1,
                    1)

                mean_map_rscc = get_mean_rscc(sample_mean, fragment_maps_np, fragment_masks_np)
                mean_map_max_correlation = torch.max(mean_map_rscc).cpu().item()

                rsccs = {}
                for b_index in range(len(event_map_list)):
                    print(f"\tBDC: {bdcs[b_index]}")
                    event_maps_np = np.stack([event_map_list[b_index]], axis=0)
                    event_maps_np = event_maps_np.reshape(event_maps_np.shape[0],
                                                          1,
                                                          event_maps_np.shape[1],
                                                          event_maps_np.shape[2],
                                                          event_maps_np.shape[3])
                    print(f"event_maps_np: {event_maps_np.shape}")

                    rsccs[bdcs[b_index]] = fragment_search_gpu(event_maps_np, fragment_maps_np, fragment_masks_np,
                                                               mean_map_rscc, 0.5, 0.4, fragment_size_np,
                                                               fragment_map_value_list)

                    print(f"\tresults: {rsccs[bdcs[b_index]]}")

                    max_index = rsccs[bdcs[b_index]][1]
                    max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
                    max_rotation = list(fragment_maps.keys())[max_index[1]]
                    max_position = max_coord_to_position(
                        max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing,
                        max_x,
                        max_y,
                        max_z,
                        alignments, dataset, marker)

                    print(f"max position: {max_position}")

                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()

                    results.append((b_factor, bdcs[b_index], rsccs[bdcs[b_index]], max_position))

                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            print(type(obj), obj.size())
                    except:
                        pass

                max_rscc_bdc = max(rsccs, key=lambda x: rsccs[x][0])
                max_rscc_correlation_index = rsccs[max_rscc_bdc]
                max_correlation = max_rscc_correlation_index[0]
                max_index = max_rscc_correlation_index[1]
                max_mean_map_correlation = max_rscc_correlation_index[2]
                max_delta_correlation = max_rscc_correlation_index[3]

                max_bdc = max_rscc_bdc
                max_rotation = list(fragment_maps.keys())[max_index[1]]
                max_index_fragment_map = fragment_maps[max_rotation]
                max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
                max_index_fragment_map_shape = max_index_fragment_map.shape

                max_index_fragment_position_dataset_frame = max_coord_to_position(
                    max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing, max_x,
                    max_y,
                    max_z,
                    alignments, dataset, marker
                )

                # get affinity maxima
                maxima: AffinityMaxima = AffinityMaxima(
                    index=max_index,
                    correlation=max_correlation,
                    rotation_index=max_rotation,
                    position=max_index_fragment_position_dataset_frame,
                    bdc=max_bdc,
                    mean_map_correlation=max_mean_map_correlation,
                    mean_map_max_correlation=mean_map_max_correlation,
                    max_delta_correlation=max_delta_correlation,
                )
                print(maxima)

                # if max_correlation > params.min_correlation:
                event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
                    (dataset_sample - (maxima.bdc * sample_mean)) / (1 - maxima.bdc),
                    # max_array[0,:,:,:],
                    reference_dataset,
                    dataset,
                    alignments[dataset.dtag][marker],
                    marker,
                    params.grid_size,
                    params.grid_spacing,
                    params.structure_factors,
                    params.sample_rate,
                )

                dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
                                              marker.y - alignments[dataset.dtag][marker].transform.vec.y,
                                              marker.z - alignments[dataset.dtag][marker].transform.vec.z,
                                              None,
                                              )

                write_event_map(
                    event_map,
                    out_dir / f"{dataset.dtag}_{max_index_fragment_position_dataset_frame[0]}_{max_index_fragment_position_dataset_frame[1]}_{max_index_fragment_position_dataset_frame[2]}_{fragment_id}.mtz",
                    dataset_event_marker,
                    dataset,
                    resolution,
                )

        # End loop over fragment builds
    # end loop over b factors

    for result in sorted(
            results,
            key=lambda result: result[2][0],
    ):
        print(result)

    exit()

    # Get a result object
    dataset_results: DatasetAffinityResults = DatasetAffinityResults(
        dataset.dtag,
        marker,
        dataset.structure_path,
        dataset.reflections_path,
        dataset.fragment_path,
        maxima,
    )

    return dataset_results


def analyse_dataset_rmsd_protein_scaled_gpu(
        dataset: Dataset,
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> Optional[DatasetAffinityResults]:
    # Get the fragment
    dataset_fragment_structures: Optional[MutableMapping[str, gemmi.Structure]] = dataset.fragment_structures

    # No need to analyse if no fragment present
    if not dataset_fragment_structures:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )

    # Select the comparator datasets
    comparator_datasets: Optional[MutableMapping[str, Dataset]] = get_comparator_datasets(
        linkage,
        dataset_clusters,
        dataset_index,
        dataset.dtag,
        get_dataset_apo_mask(residue_datasets, known_apos),
        residue_datasets,
        params.min_dataset_cluster_size,
        params.min_dataset_cluster_size,
    )

    if params.debug:
        print(f"\tComparator datasets are: {list(comparator_datasets.keys())}")

    # Handle no comparators
    if not comparator_datasets:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )
        # continue

    # Get the truncated comparator datasets
    comparator_truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        comparator_datasets,
        dataset,
        params.structure_factors,
    )
    comparator_truncated_datasets[dataset.dtag] = dataset
    print(len(comparator_truncated_datasets))

    # Get the local density samples associated with the comparator datasets
    comparator_sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        comparator_truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        params.grid_size,
        params.grid_spacing,
    )
    print(max([comparator_truncated_dataset.reflections.resolution_high() for comparator_truncated_dataset in
               comparator_truncated_datasets.values()]))

    # Get the sample associated with the dataset of interest
    dataset_sample: np.ndarray = comparator_sample_arrays[dataset.dtag]
    del comparator_sample_arrays[dataset.dtag]

    if params.debug:
        print(f"\tGot {len(comparator_sample_arrays)} comparater samples")

    resolution: float = list(comparator_truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Characterise the local distribution
    sample_mean: np.ndarray = get_mean(comparator_sample_arrays)
    sample_std: np.ndarray = get_std(comparator_sample_arrays)
    sample_z: np.ndarray = get_z(dataset_sample, sample_mean, sample_std)
    if params.debug:
        print(f"\tGot mean: max {np.max(sample_mean)}, min: {np.min(sample_mean)}")
        print(f"\tGot std: max {np.max(sample_std)}, min: {np.min(sample_std)}")
        print(f"\tGot z: max {np.max(sample_z)}, min: {np.min(sample_z)}")

    # Find the scaling for the fragments
    protein_location, protein_scale = get_protein_scaling(dataset, params.structure_factors, params.sample_rate)
    print(f"protein_location: {protein_location}")

    print(f"protein_scale: {protein_scale}")

    # Get the comparator affinity maps
    results = []

    for b_factor in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0):
        print(f"############ B FACTOR: {b_factor} ########")

        for fragment_id, fragment_structure in dataset_fragment_structures.items():
            if params.debug:
                print(f"\t\tProcessing fragment: {fragment_id}")
            fragment_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = get_fragment_maps(
                fragment_structure,
                resolution,
                params.num_fragment_pose_samples,
                params.sample_rate,
                params.grid_spacing,
                b_factor
            )

            save_example_fragment_map(list(fragment_maps.values())[0], params.grid_spacing,
                                      out_dir / "example_fragment_map.ccp4")

            max_x = max([fragment_map.shape[0] for fragment_map in fragment_maps.values()])
            max_y = max([fragment_map.shape[1] for fragment_map in fragment_maps.values()])
            max_z = max([fragment_map.shape[2] for fragment_map in fragment_maps.values()])
            if max_x % 2 == 0: max_x = max_x + 1
            if max_y % 2 == 0: max_y = max_y + 1
            if max_z % 2 == 0: max_z = max_z + 1

            fragment_masks_list = []

            fragment_maps_list = []
            fragment_masks = {}
            fragment_map_size_list = []
            fragment_map_value_list = []
            for rotation, fragment_map in fragment_maps.items():
                arr = fragment_map.copy()

                arr_mask = fragment_map > 0.0

                # print(f"arr_mask: {np.sum(arr_mask)}")

                arr[~arr_mask] = 0.0

                fragment_map_size_list.append(np.sum(arr_mask))
                fragment_map_value_list.append(arr[arr_mask])

                fragment_mask_arr = np.zeros(fragment_map.shape)
                fragment_mask_arr[arr_mask] = 1.0

                fragment_map = np.zeros((max_x, max_y, max_z,))
                fragment_mask = np.zeros((max_x, max_y, max_z,))

                fragment_map[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr[:, :, :]
                fragment_mask[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = fragment_mask_arr[:, :, :]

                fragment_maps_list.append(fragment_map)
                fragment_masks_list.append(fragment_mask)

                fragment_masks[rotation] = fragment_mask

            if params.debug:
                print(f"\t\tGot {len(fragment_maps)} fragment maps")

            # Get affinity maps for various orientations
            event_map_list = []
            bdcs = np.linspace(0, 0.90, 10)
            for b in bdcs:
                event_map = (dataset_sample - (b * sample_mean)) / (1 - b)

                # Mask the significant density in the event map
                event_map[sample_z < 1.5] = 0.0

                event_map_list.append(event_map)

            with torch.no_grad():

                # Scale
                reference_map = fragment_maps_list[0]
                reference_mask = fragment_masks_list[0]
                reference_map_points = reference_map[reference_mask > 0.0]
                reference_location = np.mean(reference_map_points)
                reference_scale = np.std(reference_map_points)
                print(f"reference_loc: {reference_location}")
                print(f"reference_scale: {reference_scale}")

                for j, fragment_map in enumerate(fragment_maps_list):
                    fragment_map = fragment_maps_list[j]
                    fragment_mask = fragment_masks_list[j]

                    masked_points = fragment_map[fragment_mask > 0.0]

                    masked_points = (((
                                                  masked_points - reference_location) / reference_scale) * protein_scale) + protein_location

                    fragment_maps_list[j][fragment_masks_list[j] > 0.0] = masked_points

                print(
                    f"exmaple map stats: {np.max(fragment_maps_list[0])} {np.min(fragment_maps_list[0])} {np.mean(fragment_maps_list[0])} {np.std(fragment_maps_list[0])}")

                # Fragment maps
                fragment_maps_np = np.stack(fragment_maps_list, axis=0)
                fragment_maps_np = fragment_maps_np.reshape(
                    fragment_maps_np.shape[0],
                    1,
                    fragment_maps_np.shape[1],
                    fragment_maps_np.shape[2],
                    fragment_maps_np.shape[3])
                print(f"fragment_maps_np: {fragment_maps_np.shape}")

                fragment_masks_np = np.stack(fragment_masks_list, axis=0)
                fragment_masks_np = fragment_masks_np.reshape(
                    fragment_masks_np.shape[0],
                    1,
                    fragment_masks_np.shape[1],
                    fragment_masks_np.shape[2],
                    fragment_masks_np.shape[3])
                print(f"fragment_masks_np: {fragment_masks_np.shape}")

                fragment_size_np = np.array(fragment_map_size_list).reshape(
                    1,
                    len(fragment_map_size_list),
                    1,
                    1,
                    1)

                # Mask the significant density in the sample mean
                sample_mean_masked = sample_mean.copy()
                sample_mean_masked[sample_z < 1.5] = 0

                # Perform the search
                background_rmsd_map = fragment_search_rmsd_gpu(
                    sample_mean_masked.reshape(1, 1, sample_mean.shape[0], sample_mean.shape[1], sample_mean.shape[2]),
                    fragment_maps_np,
                    fragment_masks_np,
                    fragment_size_np,
                    fragment_map_value_list
                )

                peak = peak_search_rmsd(background_rmsd_map)
                print(f"peak: {peak}")

                max_index = peak[1]
                max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
                max_rotation = list(fragment_maps.keys())[max_index[1]]
                max_position = max_coord_to_position(
                    max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing,
                    max_x,
                    max_y,
                    max_z,
                    alignments, dataset, marker)

                print(f"max reference position: {max_position}")

                rsccs = {}
                for b_index in range(len(event_map_list)):
                    print(f"\tBDC: {bdcs[b_index]}")
                    event_maps_np = np.stack([event_map_list[b_index]], axis=0)
                    event_maps_np = event_maps_np.reshape(event_maps_np.shape[0],
                                                          1,
                                                          event_maps_np.shape[1],
                                                          event_maps_np.shape[2],
                                                          event_maps_np.shape[3])
                    print(f"event_maps_np: {event_maps_np.shape}")

                    rmsd_map = fragment_search_rmsd_gpu(event_maps_np, fragment_maps_np, fragment_masks_np,
                                                        fragment_size_np,
                                                        fragment_map_value_list
                                                        )

                    peak = peak_search_rmsd(rmsd_map)
                    print(f"\tpeak: {peak}")
                    max_index = peak[1]
                    peak[2] = background_rmsd_map[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4],].item()

                    rsccs[bdcs[b_index]] = peak

                    results.append((b_factor, bdcs[b_index], rsccs[bdcs[b_index]], max_position))

                    # rmsd_map_np = torch.min(rmsd_map, 1)[0].cpu().numpy()[0, :, :, :]
                    # inverse_rmsd_map_np = 1 / rmsd_map_np
                    # inverse_rmsd_map_np = np.nan_to_num(inverse_rmsd_map_np, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

                    # event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
                    #     inverse_rmsd_map_np,
                    #     reference_dataset,
                    #     dataset,
                    #     alignments[dataset.dtag][marker],
                    #     marker,
                    #     params.grid_size,
                    #     params.grid_spacing,
                    #     params.structure_factors,
                    #     params.sample_rate,
                    # )
                    #
                    # dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
                    #                               marker.y - alignments[dataset.dtag][marker].transform.vec.y,
                    #                               marker.z - alignments[dataset.dtag][marker].transform.vec.z,
                    #                               None,
                    #                               )
                    #
                    # write_event_map(
                    #     event_map,
                    #     out_dir / f"{dataset.dtag}_{b_index}_inverse_rmsd.mtz",
                    #     dataset_event_marker,
                    #     dataset,
                    #     resolution,
                    # )
                    #
                    # rmsd_delta_map = torch.min(rmsd_map, 1)[0].cpu().numpy()[0, :, :, :]
                    # inverse_rmsd_delta_map = 1 / rmsd_delta_map
                    # inverse_rmsd_delta_map = np.nan_to_num(inverse_rmsd_delta_map, copy=True, nan=0.0, posinf=0.0,
                    #                                        neginf=0.0)
                    #
                    # event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
                    #     inverse_rmsd_delta_map,
                    #     reference_dataset,
                    #     dataset,
                    #     alignments[dataset.dtag][marker],
                    #     marker,
                    #     params.grid_size,
                    #     params.grid_spacing,
                    #     params.structure_factors,
                    #     params.sample_rate,
                    # )
                    #
                    # dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
                    #                               marker.y - alignments[dataset.dtag][marker].transform.vec.y,
                    #                               marker.z - alignments[dataset.dtag][marker].transform.vec.z,
                    #                               None,
                    #                               )
                    #
                    # write_event_map(
                    #     event_map,
                    #     out_dir / f"{dataset.dtag}_{b_index}_inverse_rmsd.mtz",
                    #     dataset_event_marker,
                    #     dataset,
                    #     resolution,
                    # )
                    #
                    # max_index = rsccs[bdcs[b_index]][1]
                    # max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
                    # max_rotation = list(fragment_maps.keys())[max_index[1]]
                    # max_position = max_coord_to_position(
                    #     max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing,
                    #     max_x,
                    #     max_y,
                    #     max_z,
                    #     alignments, dataset, marker)

                    print(f"max position: {max_position}")

                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()

                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            print(type(obj), obj.size())
                    except:
                        pass

                max_rscc_bdc = max(rsccs, key=lambda x: rsccs[x][0])
                max_rscc_correlation_index = rsccs[max_rscc_bdc]
                max_correlation = max_rscc_correlation_index[0]
                max_index = max_rscc_correlation_index[1]
                max_mean_map_correlation = max_rscc_correlation_index[2]
                max_delta_correlation = max_rscc_correlation_index[3]

                max_bdc = max_rscc_bdc
                max_rotation = list(fragment_maps.keys())[max_index[1]]
                max_index_fragment_map = fragment_maps[max_rotation]
                max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
                max_index_fragment_map_shape = max_index_fragment_map.shape

                max_index_fragment_position_dataset_frame = max_coord_to_position(
                    max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing, max_x,
                    max_y,
                    max_z,
                    alignments, dataset, marker
                )

                # get affinity maxima
                maxima: AffinityMaxima = AffinityMaxima(
                    index=max_index,
                    correlation=max_correlation,
                    rotation_index=max_rotation,
                    position=max_index_fragment_position_dataset_frame,
                    bdc=max_bdc,
                    mean_map_correlation=max_mean_map_correlation,
                    mean_map_max_correlation=0.0,
                    max_delta_correlation=max_delta_correlation,
                )
                print(maxima)

                # if max_correlation > params.min_correlation:
                event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
                    (dataset_sample - (maxima.bdc * sample_mean)) / (1 - maxima.bdc),
                    # max_array[0,:,:,:],
                    reference_dataset,
                    dataset,
                    alignments[dataset.dtag][marker],
                    marker,
                    params.grid_size,
                    params.grid_spacing,
                    params.structure_factors,
                    params.sample_rate,
                )

                dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
                                              marker.y - alignments[dataset.dtag][marker].transform.vec.y,
                                              marker.z - alignments[dataset.dtag][marker].transform.vec.z,
                                              None,
                                              )

                write_event_map(
                    event_map,
                    out_dir / f"{dataset.dtag}_{max_index_fragment_position_dataset_frame[0]}_{max_index_fragment_position_dataset_frame[1]}_{max_index_fragment_position_dataset_frame[2]}_{fragment_id}.mtz",
                    dataset_event_marker,
                    dataset,
                    resolution,
                )

    for result in sorted(
            results,
            key=lambda result: result[2][0],
    ):
        print(result)

    exit()

    # End loop over fragment builds

    # Get a result object
    dataset_results: DatasetAffinityResults = DatasetAffinityResults(
        dataset.dtag,
        marker,
        dataset.structure_path,
        dataset.reflections_path,
        dataset.fragment_path,
        maxima,
    )

    return dataset_results


def analyse_dataset_rmsd_gpu(
        dataset: Dataset,
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> Optional[DatasetAffinityResults]:
    # Get the fragment
    dataset_fragment_structures: Optional[MutableMapping[str, gemmi.Structure]] = dataset.fragment_structures

    # No need to analyse if no fragment present
    if not dataset_fragment_structures:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )

    # Select the comparator datasets
    comparator_datasets: Optional[MutableMapping[str, Dataset]] = get_comparator_datasets(
        linkage,
        dataset_clusters,
        dataset_index,
        dataset.dtag,
        get_dataset_apo_mask(residue_datasets, known_apos),
        residue_datasets,
        params.min_dataset_cluster_size,
        params.min_dataset_cluster_size,
    )

    if params.debug:
        print(f"\tComparator datasets are: {list(comparator_datasets.keys())}")

    # Handle no comparators
    if not comparator_datasets:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )
        # continue

    # Get the truncated comparator datasets
    comparator_truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        comparator_datasets,
        dataset,
        params.structure_factors,
    )
    comparator_truncated_datasets[dataset.dtag] = dataset
    print(len(comparator_truncated_datasets))

    # Get the local density samples associated with the comparator datasets
    comparator_sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        comparator_truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        params.grid_size,
        params.grid_spacing,
    )
    print(max([comparator_truncated_dataset.reflections.resolution_high() for comparator_truncated_dataset in
               comparator_truncated_datasets.values()]))

    # Get the sample associated with the dataset of interest
    dataset_sample: np.ndarray = comparator_sample_arrays[dataset.dtag]
    del comparator_sample_arrays[dataset.dtag]

    if params.debug:
        print(f"\tGot {len(comparator_sample_arrays)} comparater samples")

    resolution: float = list(comparator_truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Characterise the local distribution
    sample_mean: np.ndarray = get_mean(comparator_sample_arrays)
    sample_std: np.ndarray = get_std(comparator_sample_arrays)
    sample_z: np.ndarray = get_z(dataset_sample, sample_mean, sample_std)
    if params.debug:
        print(f"\tGot mean: max {np.max(sample_mean)}, min: {np.min(sample_mean)}")
        print(f"\tGot std: max {np.max(sample_std)}, min: {np.min(sample_std)}")
        print(f"\tGot z: max {np.max(sample_z)}, min: {np.min(sample_z)}")

    # Get the comparator affinity maps
    for fragment_id, fragment_structure in dataset_fragment_structures.items():
        if params.debug:
            print(f"\t\tProcessing fragment: {fragment_id}")
        fragment_maps: MutableMapping[Tuple[float, float, float], np.ndarray] = get_fragment_maps(
            fragment_structure,
            resolution,
            params.num_fragment_pose_samples,
            params.sample_rate,
            params.grid_spacing,
        )

        save_example_fragment_map(list(fragment_maps.values())[0], params.grid_spacing,
                                  out_dir / "example_fragment_map.ccp4")

        max_x = max([fragment_map.shape[0] for fragment_map in fragment_maps.values()])
        max_y = max([fragment_map.shape[1] for fragment_map in fragment_maps.values()])
        max_z = max([fragment_map.shape[2] for fragment_map in fragment_maps.values()])
        if max_x % 2 == 0: max_x = max_x + 1
        if max_y % 2 == 0: max_y = max_y + 1
        if max_z % 2 == 0: max_z = max_z + 1

        fragment_masks_list = []

        fragment_maps_list = []
        fragment_masks = {}
        for rotation, fragment_map in fragment_maps.items():
            arr = fragment_map.copy()

            arr_mask = fragment_map > 0.0

            print(f"arr_mask: {np.sum(arr_mask)}")

            arr[~arr_mask] = 0.0

            fragment_mask_arr = np.zeros(fragment_map.shape)
            fragment_mask_arr[arr_mask] = 1.0

            fragment_map = np.zeros((max_x, max_y, max_z,))
            fragment_mask = np.zeros((max_x, max_y, max_z,))

            fragment_map[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr[:, :, :]
            fragment_mask[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = fragment_mask_arr[:, :, :]

            fragment_maps_list.append(fragment_map)
            fragment_masks_list.append(fragment_mask)

            fragment_masks[rotation] = fragment_mask

        if params.debug:
            print(f"\t\tGot {len(fragment_maps)} fragment maps")

        # Get affinity maps for various orientations
        event_map_list = []
        bdcs = np.linspace(0, 0.90, 10)
        for b in bdcs:
            event_map = (dataset_sample - (b * sample_mean)) / (1 - b)

            event_map_list.append(event_map)

        with torch.no_grad():

            # Fragment maps
            fragment_maps_np = np.stack(fragment_maps_list, axis=0)
            fragment_maps_np = fragment_maps_np.reshape(fragment_maps_np.shape[0],
                                                        1,
                                                        fragment_maps_np.shape[1],
                                                        fragment_maps_np.shape[2],
                                                        fragment_maps_np.shape[3])
            print(f"fragment_maps_np: {fragment_maps_np.shape}")

            fragment_masks_np = np.stack(fragment_masks_list, axis=0)
            fragment_masks_np = fragment_masks_np.reshape(fragment_masks_np.shape[0],
                                                          1,
                                                          fragment_masks_np.shape[1],
                                                          fragment_masks_np.shape[2],
                                                          fragment_masks_np.shape[3])
            print(f"fragment_masks_np: {fragment_masks_np.shape}")

            mean_map_rscc = get_mean_rscc(sample_mean, fragment_maps_np, fragment_masks_np)
            mean_map_max_correlation = torch.max(mean_map_rscc).cpu().item()

            rsccs = {}
            for b_index in range(len(event_map_list)):
                print(f"\tBDC: {bdcs[b_index]}")
                event_maps_np = np.stack([event_map_list[b_index]], axis=0)
                event_maps_np = event_maps_np.reshape(event_maps_np.shape[0],
                                                      1,
                                                      event_maps_np.shape[1],
                                                      event_maps_np.shape[2],
                                                      event_maps_np.shape[3])
                print(f"event_maps_np: {event_maps_np.shape}")

                rmsd_map = fragment_search_rmsd_scaled_gpu(event_maps_np, fragment_maps_np, fragment_masks_np,
                                                           mean_map_rscc, 0.5, 0.4)

                peak = peak_search_rmsd(rmsd_map)

                max_index = peak[1]
                max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
                max_rotation = list(fragment_maps.keys())[max_index[1]]
                max_position = max_coord_to_position(
                    max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing,
                    max_x,
                    max_y,
                    max_z,
                    alignments, dataset, marker)

                print(f"Max position: {max_position}")

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass

            exit()

            max_rscc_bdc = max(rsccs, key=lambda x: rsccs[x][0])
            max_rscc_correlation_index = rsccs[max_rscc_bdc]
            max_correlation = max_rscc_correlation_index[0]
            max_index = max_rscc_correlation_index[1]
            max_mean_map_correlation = max_rscc_correlation_index[2]
            max_delta_correlation = max_rscc_correlation_index[3]

            max_bdc = max_rscc_bdc
            max_rotation = list(fragment_maps.keys())[max_index[1]]
            max_index_fragment_map = fragment_maps[max_rotation]
            max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
            max_index_fragment_map_shape = max_index_fragment_map.shape

            max_index_fragment_position_dataset_frame = max_coord_to_position(
                max_index_mask_coord, fragment_maps, max_rotation, params.grid_size, params.grid_spacing, max_x, max_y,
                max_z,
                alignments, dataset, marker
            )

            # get affinity maxima
            maxima: AffinityMaxima = AffinityMaxima(
                index=max_index,
                correlation=max_correlation,
                rotation_index=max_rotation,
                position=max_index_fragment_position_dataset_frame,
                bdc=max_bdc,
                mean_map_correlation=max_mean_map_correlation,
                mean_map_max_correlation=mean_map_max_correlation,
                max_delta_correlation=max_delta_correlation,
            )
            print(maxima)

            # if max_correlation > params.min_correlation:
            event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
                (dataset_sample - (maxima.bdc * sample_mean)) / (1 - maxima.bdc),
                # max_array[0,:,:,:],
                reference_dataset,
                dataset,
                alignments[dataset.dtag][marker],
                marker,
                params.grid_size,
                params.grid_spacing,
                params.structure_factors,
                params.sample_rate,
            )

            dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
                                          marker.y - alignments[dataset.dtag][marker].transform.vec.y,
                                          marker.z - alignments[dataset.dtag][marker].transform.vec.z,
                                          None,
                                          )

            write_event_map(
                event_map,
                out_dir / f"{dataset.dtag}_{max_index_fragment_position_dataset_frame[0]}_{max_index_fragment_position_dataset_frame[1]}_{max_index_fragment_position_dataset_frame[2]}_{fragment_id}.mtz",
                dataset_event_marker,
                dataset,
                resolution,
            )

    # End loop over fragment builds

    # Get a result object
    dataset_results: DatasetAffinityResults = DatasetAffinityResults(
        dataset.dtag,
        marker,
        dataset.structure_path,
        dataset.reflections_path,
        dataset.fragment_path,
        maxima,
    )

    return dataset_results


def analyse_dataset_masks_gpu(
        dataset: Dataset,
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        linkage: np.ndarray,
        dataset_clusters: np.ndarray,
        dataset_index: int,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> Optional[DatasetAffinityResults]:
    # contours = [2.0, 3.0, 4.0, 5.0]
    # contours = [1.5, 2.0, 2.5, 3.0]
    contours = [2.0]

    # Get the fragment
    dataset_fragment_structures: Optional[MutableMapping[str, gemmi.Structure]] = dataset.fragment_structures

    # No need to analyse if no fragment present
    if not dataset_fragment_structures:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )

    # Select the comparator datasets
    print(f"dataset clusters: {dataset_clusters}")

    comparator_datasets: Optional[MutableMapping[str, Dataset]] = get_comparator_datasets(
        linkage,
        dataset_clusters,
        dataset_index,
        dataset.dtag,
        get_dataset_apo_mask(residue_datasets, known_apos),
        residue_datasets,
        params.min_dataset_cluster_size,
        params.min_dataset_cluster_size,
    )

    if params.debug:
        print(f"\tComparator datasets are: {list(comparator_datasets.keys())}")

    # Handle no comparators
    if not comparator_datasets:
        return DatasetAffinityResults(
            dtag=dataset.dtag,
            marker=marker,
            structure_path=dataset.structure_path,
            reflections_path=dataset.reflections_path,
            fragment_path=dataset.fragment_path,
            maxima=AffinityMaxima(
                (0, 0, 0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                0.0,
                0.0,
                0.0
            )
        )
        # continue

    # Get the truncated comparator datasets
    comparator_truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        comparator_datasets,
        dataset,
        params.structure_factors,
    )
    comparator_truncated_datasets[dataset.dtag] = dataset
    print(len(comparator_truncated_datasets))

    # Get the local density samples associated with the comparator datasets
    comparator_sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        comparator_truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        params.grid_size,
        params.grid_spacing,
    )

    print(max([comparator_truncated_dataset.reflections.resolution_high() for comparator_truncated_dataset in
               comparator_truncated_datasets.values()]))

    # Get the sample associated with the dataset of interest
    dataset_sample: np.ndarray = comparator_sample_arrays[dataset.dtag]
    del comparator_sample_arrays[dataset.dtag]

    if params.debug:
        print(f"\tGot {len(comparator_sample_arrays)} comparater samples")

    resolution: float = list(comparator_truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Characterise the local distribution
    sample_mean: np.ndarray = get_mean(comparator_sample_arrays)
    sample_std: np.ndarray = get_std(comparator_sample_arrays)
    sample_z: np.ndarray = get_z(dataset_sample, sample_mean, sample_std)
    sample_adjusted = sample_z + dataset_sample

    if params.debug:
        print(f"\tGot mean: max {np.max(sample_mean)}, min: {np.min(sample_mean)}")
        print(f"\tGot std: max {np.max(sample_std)}, min: {np.min(sample_std)}")
        print(f"\tGot z: max {np.max(sample_z)}, min: {np.min(sample_z)}")

    # Get the comparator affinity maps
    for fragment_id, fragment_structure in dataset_fragment_structures.items():
        if params.debug:
            print(f"\t\tProcessing fragment: {fragment_id}")
        initial_fragment_masks: MutableMapping[Tuple[float, float, float], np.ndarray] = get_fragment_masks(
            fragment_structure,
            params.num_fragment_pose_samples,
            params.grid_spacing,
            [2.0, 1.25, 0.75]
        )

        max_x = max([fragment_map.shape[0] for fragment_map in initial_fragment_masks.values()])
        max_y = max([fragment_map.shape[1] for fragment_map in initial_fragment_masks.values()])
        max_z = max([fragment_map.shape[2] for fragment_map in initial_fragment_masks.values()])
        if max_x % 2 == 0: max_x = max_x + 1
        if max_y % 2 == 0: max_y = max_y + 1
        if max_z % 2 == 0: max_z = max_z + 1

        fragment_masks_list = []
        fragment_masks_low_list = []

        fragment_masks = {}
        for rotation, initial_fragment_mask in initial_fragment_masks.items():
            arr = initial_fragment_mask.copy()

            arr_mask = initial_fragment_mask >= 3.0
            # arr_mask_low = (initial_fragment_mask >= 1.0) * (initial_fragment_mask < 2.0)
            arr_mask_low = initial_fragment_mask >= 1.0

            print(f"arr_mask: {np.sum(arr_mask)}")
            print(f"arr_mask_low: {np.sum(arr_mask_low)}")

            arr[~arr_mask] = 0.0
            fragment_mask_arr = np.zeros(initial_fragment_mask.shape)
            fragment_mask_arr[arr_mask] = 1.0

            fragment_mask_low_arr = np.zeros(initial_fragment_mask.shape)
            fragment_mask_low_arr[arr_mask_low] = 1.0

            fragment_mask = np.zeros((max_x, max_y, max_z,))
            fragment_mask_low = np.zeros((max_x, max_y, max_z,))

            fragment_mask[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = fragment_mask_arr[:, :, :]
            fragment_mask_low[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = fragment_mask_low_arr[:, :, :]

            fragment_masks_list.append(fragment_mask)
            fragment_masks_low_list.append(fragment_mask_low)

            fragment_masks[rotation] = fragment_mask

        if params.debug:
            print(f"\t\tGot {len(fragment_masks)} fragment maps")

        # Get affinity maps for various orientations
        event_map_list = []
        bdcs = np.linspace(0, 0.90, 10)
        for b in bdcs:
            event_map = (dataset_sample - (b * sample_mean)) / (1 - b)

            event_map_list.append(event_map)

        with torch.no_grad():

            # Fragment masks
            fragment_masks_np = np.stack(fragment_masks_list, axis=0)
            fragment_masks_np = fragment_masks_np.reshape(fragment_masks_np.shape[0],
                                                          1,
                                                          fragment_masks_np.shape[1],
                                                          fragment_masks_np.shape[2],
                                                          fragment_masks_np.shape[3])
            print(f"fragment_masks_np: {fragment_masks_np.shape}")

            fragment_masks_low_np = np.stack(fragment_masks_low_list, axis=0)
            fragment_masks_low_np = fragment_masks_low_np.reshape(fragment_masks_low_np.shape[0],
                                                                  1,
                                                                  fragment_masks_low_np.shape[1],
                                                                  fragment_masks_low_np.shape[2],
                                                                  fragment_masks_low_np.shape[3])
            print(f"fragment_masks_low_np: {fragment_masks_low_np.shape}")

            fragment_mask_size = np.sum(fragment_masks_np[0, 0, :, :, :])
            print(f"fragment_mask_size: {fragment_mask_size}")

            fragment_mask_low_size = np.sum(fragment_masks_low_np[0, 0, :, :, :])
            print(f"fragment_mask_low_size: {fragment_mask_low_size}")

            mean_map_np = np.stack([sample_mean], axis=0)
            mean_map_np = mean_map_np.reshape(
                1,
                1,
                mean_map_np.shape[1],
                mean_map_np.shape[2],
                mean_map_np.shape[3],
            )

            reference_maps = {}
            for contour in contours:
                reference_map = fragment_search_mask_unnormalised_gpu(mean_map_np, fragment_masks_np, contour)
                reference_map_low = fragment_search_mask_unnormalised_gpu(mean_map_np, fragment_masks_low_np, contour)

                mean_map_max_correlation = torch.max(reference_map).cpu().item()
                reference_maps[contour] = reference_map + (fragment_mask_low_size - reference_map_low)

            # rsccs = {}
            rmsds = {}
            # for b_index in range(len(event_map_list)):
            #
            #     for contour in contours:
            #         reference_map = reference_maps[contour]
            #
            #         event_maps_np = np.stack([event_map_list[b_index]], axis=0)
            #         event_maps_np = event_maps_np.reshape(event_maps_np.shape[0],
            #                                               1,
            #                                               event_maps_np.shape[1],
            #                                               event_maps_np.shape[2],
            #                                               event_maps_np.shape[3])
            #         print(f"event_maps_np: {event_maps_np.shape}")
            #
            #         target_map = fragment_search_mask_unnormalised_gpu(event_maps_np, fragment_masks_np, contour)
            #         target_map_low = fragment_search_mask_unnormalised_gpu(event_maps_np, fragment_masks_low_np,
            #                                                                contour)
            #
            #         # Get the score of each point as the number of contoured points in the inner mask +
            #         # The number of outer mask points not in the contour
            #         search_map = target_map + (fragment_mask_low_size-target_map_low)
            #
            #         # Censor points where the inner mask is a bad fit
            #         search_map[(target_map / fragment_mask_size) < 0.5] = 0.0
            #
            #         # Find the strongest match
            #         rmsds[(bdcs[b_index], contour)] = peak_search_mask(
            #             search_map,
            #         )
            #         print(f"\tContour {contour}: {rmsds[(bdcs[b_index], contour)]}")
            #
            #         gc.collect()
            #         torch.cuda.empty_cache()
            #         torch.cuda.synchronize()
            #         torch.cuda.ipc_collect()

            # Searching z map
            for contour in contours:
                reference_map = reference_maps[contour]

                event_maps_np = np.stack([sample_z], axis=0)
                event_maps_np = event_maps_np.reshape(event_maps_np.shape[0],
                                                      1,
                                                      event_maps_np.shape[1],
                                                      event_maps_np.shape[2],
                                                      event_maps_np.shape[3])
                print(f"event_maps_np: {event_maps_np.shape}")

                target_map = fragment_search_mask_unnormalised_gpu(event_maps_np, fragment_masks_np, contour)
                target_map_low = fragment_search_mask_unnormalised_gpu(event_maps_np, fragment_masks_low_np,
                                                                       contour)

                # Get the score of each point as the number of contoured points in the inner mask +
                # The number of outer mask points not in the contour
                # search_map = target_map + (fragment_mask_low_size-target_map_low)
                # search_map = (target_map / fragment_mask_size) * ((fragment_mask_low_size-target_map_low)/fragment_mask_low_size)

                # search_map = (fragment_mask_low_size-target_map_low)/fragment_mask_low_size

                # search_map[(target_map / fragment_mask_size) < 0.8] = 0.0

                # search_map = target_map * (target_map / ( target_map_low))
                search_map = ((target_map / fragment_mask_size) * (target_map / fragment_mask_size)) / (
                        target_map_low / fragment_mask_low_size)
                search_map = torch.nan_to_num(search_map, nan=0.0, posinf=0.0, neginf=0.0, )

                # Censor points where the inner mask is a bad fit
                # search_map[(target_map / fragment_mask_size) < 0.5] = 0.0

                # Find the strongest match
                peak = peak_search_mask(
                    search_map,
                )

                rmsds[(0, contour)] = peak
                print(f"\tContour {contour}: {rmsds[(0, contour)]}")

                print(torch.nonzero(search_map > (0.7 * peak[0])).shape)

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass

            max_rscc_bdc_contour = max(rmsds, key=lambda x: rmsds[x][0])
            print(f"max_rscc_bdc_contour: {max_rscc_bdc_contour}")
            max_rscc_bdc = max_rscc_bdc_contour[0]
            max_rscc_contour = max_rscc_bdc_contour[0]
            max_rscc_correlation_index = rmsds[max_rscc_bdc_contour]
            max_correlation = max_rscc_correlation_index[0]
            max_index = max_rscc_correlation_index[1]
            max_mean_map_correlation = max_rscc_correlation_index[2]
            max_delta_correlation = max_rscc_correlation_index[3]

            max_bdc = max_rscc_bdc
            max_rotation = list(fragment_masks.keys())[max_index[1]]
            max_index_fragment_mask = fragment_masks[max_rotation]
            max_index_mask_coord = [max_index[2], max_index[3], max_index[4]]
            max_index_fragment_map_shape = max_index_fragment_mask.shape
            max_index_fragment_coord = [
                max_index_mask_coord[0] - (max_x / 2) + (fragment_masks[max_rotation].shape[0] / 2),
                max_index_mask_coord[1] - (max_y / 2) + (fragment_masks[max_rotation].shape[1] / 2),
                max_index_mask_coord[2] - (max_z / 2) + (fragment_masks[max_rotation].shape[2] / 2),
            ]
            print(f"max_index_fragment_coord: {max_index_fragment_coord}")

            max_index_fragment_relative_coord = [max_index_fragment_coord[0] - params.grid_size / 2,
                                                 max_index_fragment_coord[1] - params.grid_size / 2,
                                                 max_index_fragment_coord[2] - params.grid_size / 2,
                                                 ]
            print(f"max_index_fragment_relative_coord: {max_index_fragment_relative_coord}")

            max_index_fragment_relative_position = gemmi.Position(
                max_index_fragment_relative_coord[0] * params.grid_spacing,
                max_index_fragment_relative_coord[1] * params.grid_spacing,
                max_index_fragment_relative_coord[2] * params.grid_spacing,
            )
            print(f"max_index_fragment_relative_position: {max_index_fragment_relative_position}")

            transform = alignments[dataset.dtag][marker].transform
            inverse_transform = transform.inverse()
            rotation_tr = gemmi.Transform()
            rotation_tr.mat.fromlist(inverse_transform.mat.tolist())

            max_index_fragment_relative_position_dataset_frame = rotation_tr.apply(max_index_fragment_relative_position)
            print(
                f"max_index_fragment_relative_position_dataset_frame: {max_index_fragment_relative_position_dataset_frame}")

            max_index_fragment_position_dataset_frame = [
                max_index_fragment_relative_position_dataset_frame.x + (marker.x - transform.vec.x),
                max_index_fragment_relative_position_dataset_frame.y + (marker.y - transform.vec.y),
                max_index_fragment_relative_position_dataset_frame.z + (marker.z - transform.vec.z),
            ]
            print(f"max_index_fragment_position_dataset_frame: {max_index_fragment_position_dataset_frame}")

            # get affinity maxima
            maxima: AffinityMaxima = AffinityMaxima(
                index=max_index,
                correlation=max_correlation,
                rotation_index=max_rotation,
                position=max_index_fragment_position_dataset_frame,
                bdc=max_bdc,
                mean_map_correlation=max_mean_map_correlation,
                mean_map_max_correlation=mean_map_max_correlation,
                max_delta_correlation=max_delta_correlation,
            )
            print(maxima)

            # if max_correlation > params.min_correlation:
            # event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
            #     (dataset_sample - (maxima.bdc * sample_mean)) / (1 - maxima.bdc),
            #     # max_array[0,:,:,:],
            #     reference_dataset,
            #     dataset,
            #     alignments[dataset.dtag][marker],
            #     marker,
            #     params.grid_size,
            #     params.grid_spacing,
            #     params.structure_factors,
            #     params.sample_rate,
            # )
            #
            # dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
            #                               marker.y - alignments[dataset.dtag][marker].transform.vec.y,
            #                               marker.z - alignments[dataset.dtag][marker].transform.vec.z,
            #                               None,
            #                               )
            #
            # write_event_map(
            #     event_map,
            #     out_dir / f"{dataset.dtag}_{max_index_fragment_position_dataset_frame[0]}_{max_index_fragment_position_dataset_frame[1]}_{max_index_fragment_position_dataset_frame[2]}_{fragment_id}.mtz",
            #     dataset_event_marker,
            #     dataset,
            #     resolution,
            # )
            #
            # event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
            #     torch.max(target_map_low, 1)[0].cpu().numpy()[0, :, :, :],
            #     reference_dataset,
            #     dataset,
            #     alignments[dataset.dtag][marker],
            #     marker,
            #     params.grid_size,
            #     params.grid_spacing,
            #     params.structure_factors,
            #     params.sample_rate,
            # )
            #
            # dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
            #                               marker.y - alignments[dataset.dtag][marker].transform.vec.y,
            #                               marker.z - alignments[dataset.dtag][marker].transform.vec.z,
            #                               None,
            #                               )
            #
            # write_event_map(
            #     event_map,
            #     out_dir / f"{dataset.dtag}_target_map_low.mtz",
            #     dataset_event_marker,
            #     dataset,
            #     resolution,
            # )
            #
            # event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
            #     torch.max(target_map, 1)[0].cpu().numpy()[0, :, :, :],
            #     reference_dataset,
            #     dataset,
            #     alignments[dataset.dtag][marker],
            #     marker,
            #     params.grid_size,
            #     params.grid_spacing,
            #     params.structure_factors,
            #     params.sample_rate,
            # )
            #
            # dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
            #                               marker.y - alignments[dataset.dtag][marker].transform.vec.y,
            #                               marker.z - alignments[dataset.dtag][marker].transform.vec.z,
            #                               None,
            #                               )
            #
            # write_event_map(
            #     event_map,
            #     out_dir / f"{dataset.dtag}_target_map.mtz",
            #     dataset_event_marker,
            #     dataset,
            #     resolution,
            # )

            event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
                sample_z,
                # max_array[0,:,:,:],
                reference_dataset,
                dataset,
                alignments[dataset.dtag][marker],
                marker,
                params.grid_size,
                params.grid_spacing,
                params.structure_factors,
                params.sample_rate,
            )

            dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
                                          marker.y - alignments[dataset.dtag][marker].transform.vec.y,
                                          marker.z - alignments[dataset.dtag][marker].transform.vec.z,
                                          None,
                                          )

            write_event_map(
                event_map,
                out_dir / f"{dataset.dtag}_{max_index_fragment_position_dataset_frame[0]}_{max_index_fragment_position_dataset_frame[1]}_{max_index_fragment_position_dataset_frame[2]}_{fragment_id}_z.mtz",
                dataset_event_marker,
                dataset,
                resolution,
            )

            # event_map: gemmi.FloatGrid = get_backtransformed_map_mtz(
            #     torch.max(search_map, 1)[0].cpu().numpy()[0, :, :, :],                # max_array[0,:,:,:],
            #     reference_dataset,
            #     dataset,
            #     alignments[dataset.dtag][marker],
            #     marker,
            #     params.grid_size,
            #     params.grid_spacing,
            #     params.structure_factors,
            #     params.sample_rate,
            # )
            #
            # dataset_event_marker = Marker(marker.x - alignments[dataset.dtag][marker].transform.vec.x,
            #                               marker.y - alignments[dataset.dtag][marker].transform.vec.y,
            #                               marker.z - alignments[dataset.dtag][marker].transform.vec.z,
            #                               None,
            #                               )
            #
            # write_event_map(
            #     event_map,
            #     out_dir / f"search_map.mtz",
            #     dataset_event_marker,
            #     dataset,
            #     resolution,
            # )

    # End loop over fragment builds

    # Get a result object
    dataset_results: DatasetAffinityResults = DatasetAffinityResults(
        dataset.dtag,
        marker,
        dataset.structure_path,
        dataset.reflections_path,
        dataset.fragment_path,
        maxima,
    )

    return dataset_results


def analyse_residue_gpu(
        residue_datasets: MutableMapping[str, Dataset],
        marker: Marker,
        alignments: MutableMapping[str, Alignment],
        reference_dataset: Dataset,
        known_apos: List[str],
        out_dir: Path,
        params: Params,
) -> MarkerAffinityResults:
    if params.debug:
        print(f"Found {len(residue_datasets)} residue datasets")

    # Truncate the datasets to the same reflections
    truncated_datasets: MutableMapping[str, Dataset] = get_truncated_datasets(
        residue_datasets,
        reference_dataset,
        params.structure_factors,
    )

    # Truncated dataset apos
    truncated_dataset_apo_mask: np.ndarray = get_dataset_apo_mask(truncated_datasets, known_apos)

    # resolution
    resolution: float = list(truncated_datasets.values())[0].reflections.resolution_high()
    if params.debug:
        print(f"Resolution is: {resolution}")

    # Sample the datasets to ndarrays
    if params.debug:
        print(f"Getting sample arrays...")
    sample_arrays: MutableMapping[str, np.ndarray] = sample_datasets(
        truncated_datasets,
        marker,
        alignments,
        params.structure_factors,
        params.sample_rate,
        int(params.grid_size / 2),
        params.grid_spacing * 2,
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
    residue_results: MarkerAffinityResults = {}
    for dataset_index, dtag in enumerate(truncated_datasets):
        if params.debug:
            print(f"\tProcessing dataset: {dtag}")

        # if dtag != "HAO1A-x0381":
        #     continue

        if dtag != "HAO1A-x0604":
            continue

        # if dtag != "HAO1A-x0964":
        #     continue

        # if dtag != "HAO1A-x0132":
        #     continue

        # if dtag != "HAO1A-x0808":
        #     continue

        # if dtag != "HAO1A-x0707":
        #     continue

        # if dtag != "HAO1A-x1003":
        #     continue

        dataset = residue_datasets[dtag]

        dataset_results: DatasetAffinityResults = analyse_dataset_rmsd_protein_scaled_gpu(
            dataset,
            residue_datasets,
            marker,
            alignments,
            reference_dataset,
            linkage,
            dataset_clusters,
            dataset_index,
            known_apos,
            out_dir,
            params,
        )

        # Record the dataset results
        residue_results[dtag] = dataset_results

    # End loop over truncated datasets

    return residue_results


def analyse_gpu_hits(pandda_results: PanDDAAffinityResults):
    ...


def make_database(datasets: MutableMapping[str, Dataset], results: PanDDAAffinityResults, database_path: Path):
    database = Database(
        database_path,
        True,
    )

    dataset_record_ids = {}
    for dtag, dataset in datasets.items():
        reflections_record = ReflectionsRecord(

        )
        database.session.add(reflections_record)

        smiles_record = SmilesRecord()
        database.session.add(smiles_record)

        model_record = ModelRecord()
        database.session.add(model_record)

        dataset_record = DatasetRecord(
            dtag=dtag,
            reflections=reflections_record,
            smiles=smiles_record,
            model=model_record,
        )

        dataset_record_ids[dtag] = dataset_record.id
        database.session.add(dataset_record)

    # Add the marker results
    for marker, marker_result in results.items():

        marker_record = MarkerRecord(
            position_x=marker.x,
            position_y=marker.y,
            position_z=marker.z,
        )
        database.session.add(marker_record)

        for dtag, dataset_results in marker_result.items():
            maxima = dataset_results.maxima
            maxima_rotation = maxima.rotation_index
            maxima_position = maxima.position

            maxima_record = MaximaRecord(
                bdc=maxima.bdc,
                correlation=maxima.correlation,
                rotation_x=maxima_rotation[0],
                rotation_y=maxima_rotation[1],
                rotation_z=maxima_rotation[2],
                position_x=maxima_position[0],
                position_y=maxima_position[1],
                position_z=maxima_position[2],
                dataset=database.session.query(DatasetRecord).filter(DatasetRecord.dtag == dtag).first(),
                marker=marker_record,
                mean_map_correlation=maxima.mean_map_correlation,
                mean_map_max_correlation=maxima.mean_map_max_correlation,
            )
            database.session.add(maxima_record)

    database.session.commit()
