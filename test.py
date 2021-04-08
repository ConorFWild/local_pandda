# First party
from typing import *
from pathlib import Path

# Third party
import fire
import gemmi
import numpy as np
import torch

# Custom
from local_pandda.datatypes import (
    Dataset,
    MarkerAffinityResults,
    PanDDAAffinityResults,
    Alignment,
    Params,
    Marker,
)


def main():

    # Params
    resolution = 2.0
    spacing = 0.5
    shape = np.array([20,20,20])
    size = shape * spacing

    # Create a unit cell
    shape = shape
    grid = gemmi.FloatGrid(*shape)
    unit_cell = gemmi.UnitCell(
        size[0],
        size[1],
        size[2],
        90,
        90,
        90
    )
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

    grid.set_unit_cell(unit_cell)

    # Create a test structure
    structure: gemmi.Structure = gemmi.Structure()
    model: gemmi.Model = gemmi.Model(f"{i}")
    chain: gemmi.Chain = gemmi.Chain(f"{i}")
    residue: gemmi.Residue = gemmi.Residue()

    # Get the sequence id
    # seqid: gemmi.SeqId = gemmi.SeqId(j, ' ')
    # gemmi_atom.seqid = seqid
    # gemmi_atom.seqid = seqid

    # ####### Loop over atoms, adding them to a gemmi residue
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

    # Sample the grid

    # Sample a mask
    dencalc: gemmi.DensityCalculatorE = gemmi.DensityCalculatorE()

    dencalc.d_min = resolution
    dencalc.rate = resolution / (2 * spacing)

    dencalc.set_grid_cell_and_spacegroup(structure)
    dencalc.put_model_density_on_grid(structure[0])

    grid: gemmi.FloatGrid = dencalc.grid

    unit_cell = grid.unit_cell
    min_pos = [0.0, 0.0, 0.0]

    tr = gemmi.Transform()
    tr.mat.fromlist([[1 * spacing, 0, 0], [0, 1 * spacing, 0], [0, 0, 1 * spacing]])
    tr.vec.fromlist([min_pos[0], min_pos[1], min_pos[2]])

    arr = np.zeros(
        [
            int(unit_cell.a / spacing) + 1,
            int(unit_cell.b / spacing) + 1,
            int(unit_cell.c / spacing) + 1,
        ],
        dtype=np.float32
    )

    grid.interpolate_values(arr, tr)

    # ########## Perform an fft search
    padding = (int((reference_fragment.shape[0]) / 2),
               int((reference_fragment.shape[1]) / 2),
               int((reference_fragment.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    size = torch.tensor(fragment_size_np, dtype=torch.float).cuda()
    print(f"size: {size.shape} {size[0, 0, 0, 0, 0]}")

    # Tensors
    rho_o = torch.tensor(xmap_np, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    rho_c = torch.tensor(fragment_map_np, dtype=torch.float).cuda()
    print(f"rho_c: {rho_c.shape}")

    masks = torch.tensor(fragment_mask_np, dtype=torch.float).cuda()
    print(f"masks: {masks.shape}")

    # Convolutions
    signal_unscaled = torch.nn.functional.conv3d(torch.square(rho_o), masks, padding=padding)
    print(f"rmsd: {signal_unscaled.shape} {signal_unscaled[0, 0, 24, 24, 24]}")

    signal = signal_unscaled / size
    print(f"signal: {signal.shape} {signal[0, 0, 24, 24, 24]}")

    signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0, )

