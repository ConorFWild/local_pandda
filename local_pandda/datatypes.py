from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import gemmi
from rdkit import Chem


@dataclass
class StructureFactors:
    f: str
    phi: str


@dataclass
class Dataset:
    dtag: str
    structure: gemmi.Structure
    reflections: gemmi.Mtz
    structure_path: Path
    reflections_path: Path
    fragment_path: Optional[Path]
    fragment_structures: Optional[MutableMapping[int, Chem.Mol]]
    smoothing_factor: Optional[float] = None


@dataclass()
class Data:
    datasets: Dict


@dataclass()
class Event:
    centroid: Tuple[float, float, float]
    size: int


@dataclass()
class AffinityEvent:
    dtag: str
    residue_id: ResidueID
    correlation: float

@dataclass()
class AffinityMaxima:
    index: Tuple[int, int, int]
    correlation: float
    rotation_index: Tuple[float, float, float]
    # centroid: Tuple[float, float, float]


@dataclass()
class ResidueID:
    model: str
    chain: str
    insertion: str


@dataclass()
class DatasetResults:
    dtag: str
    residue_id: ResidueID
    structure_path: Path
    reflections_path: Path
    fragment_path: Path
    events: Dict[int, Event] = field(default_factory=dict)
    comparators: List[Dataset] = field(default_factory=list)

@dataclass()
class DatasetAffinityResults:
    dtag: str
    residue_id: ResidueID
    structure_path: Path
    reflections_path: Path
    fragment_path: Path
    events: Dict[int, AffinityEvent] = field(default_factory=dict)
    comparators: List[Dataset] = field(default_factory=list)


@dataclass()
class ResidueAffinityResults(MutableMapping[str, DatasetResults]):
    _dataset_results: Dict[str, DatasetAffinityResults] = field(default_factory=dict)

@dataclass()
class ResidueResults(MutableMapping[str, DatasetResults]):
    _dataset_results: Dict[str, DatasetResults] = field(default_factory=dict)


@dataclass()
class PanDDAResults(MutableMapping[ResidueID, ResidueResults]):
    _pandda_results: Dict[ResidueID, ResidueResults] = field(default_factory=dict)


@dataclass()
class Transform:
    transform: gemmi.Transform


@dataclass()
class Alignment(MutableMapping[ResidueID, Transform]):
    _residue_alignments: MutableMapping[ResidueID, Transform] = field(default_factory=dict)


@dataclass()
class Cluster(object):
    _indexes: Tuple[np.ndarray, np.ndarray, np.ndarray]

    def size(self):
        return len(self._indexes)


@dataclass()
class DatasetAffinityResults:
    dtag: str
    residue_id: ResidueID
    structure_path: Path
    reflections_path: Path
    fragment_path: Path
    events: Dict[int, AffinityEvent] = field(default_factory=dict)
    comparators: List[Dataset] = field(default_factory=list)


@dataclass()
class ResidueAffinityResults(MutableMapping[str, DatasetAffinityResults]):
    _dataset_results: Dict[str, DatasetAffinityResults] = field(default_factory=dict)


@dataclass()
class PanDDAAffinityResults(MutableMapping[ResidueID, ResidueAffinityResults]):
    _pandda_results: Dict[ResidueID, ResidueAffinityResults] = field(default_factory=dict)



class Params:
    debug: bool = True

    # Data loading
    structure_regex: str = "*.pdb"
    reflections_regex: str = "*.mtz"
    smiles_regex: str = "*.smiles"

    # Diffraction handling
    structure_factors: StructureFactors = StructureFactors("FWT", "PHWT")
    sample_rate: float = 3.0

    # Grid sampling
    grid_size: int = 32
    grid_spacing: float = 0.5

    # Dataset clusterings
    strong_density_cluster_cutoff: float = 0.6
    min_dataset_cluster_size: int = 30

    # Fragment searching
    num_fragment_pose_samples: int = 10
    min_correlation: float = 0.1
    pruning_threshold: float = 1.5

    def update(self, **kwargs):
        for key, value in kwargs.items():

            if key == "debug":
                self.debug = value

            # Diffraction handling
            elif key == "structure_factors":
                self.structure_factors = StructureFactors(*value.split(","))
            elif key == "sample_rate":
                self.sample_rate = value

            # Grid samping
            elif key == "grid_size":
                self.grid_size = value
            elif key == "grid_spacing":
                self.grid_spacing = value

            # Dataset clusterings
            elif key == "strong_density_cluster_cutoff":
                self.strong_density_cluster_cutoff = value
            elif key == "min_dataset_cluster_size":
                self.min_dataset_cluster_size = value

            # Fragment searching
            elif key == "num_fragment_pose_samples":
                self.num_fragment_pose_samples = value
            elif key == "min_correlation":
                self.min_correlation = value
            elif key == "pruning_threshold":
                self.pruning_threshold = value

            # Unknown argument handling
            else:
                raise Exception(f"Unknown paramater: {key} = {value}")


