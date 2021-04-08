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
    st: gemmi.Structure = gemmi.Structure()
    model: gemmi.Model = gemmi.Model(f"1")
    chain: gemmi.Chain = gemmi.Chain(f"1")
    residue: gemmi.Residue = gemmi.Residue()

    # Get the sequence id


    # ####### Loop over atoms, adding them to a gemmi residue
    # Get the atomic symbol
    atom_symbol: str = "C"
    gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)

    # Get the position as a gemmi type
    gemmi_pos: gemmi.Position = gemmi.Position(size/2, size/2, size/2)

    # Get the
    gemmi_atom: gemmi.Atom = gemmi.Atom()
    gemmi_atom.name = atom_symbol
    gemmi_atom.pos = gemmi_pos
    gemmi_atom.element = gemmi_element

    # Add atom to residue
    residue.add_atom(gemmi_atom)

    st.cell = unit_cell
    st.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").xhm()

    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)

    # Sample the grid

    # Sample a mask
    dencalc: gemmi.DensityCalculatorE = gemmi.DensityCalculatorE()

    dencalc.d_min = resolution
    dencalc.rate = resolution / (2 * spacing)

    dencalc.set_grid_cell_and_spacegroup(st)
    dencalc.put_model_density_on_grid(st[0])

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

    fragment_size_np = np.sum(mask_arr > 0.5)

    mask_np = mask_arr > 0.5

    # ########## Perform an fft search
    padding = (int((mask_np.shape[0]) / 2),
               int((mask_np.shape[1]) / 2),
               int((mask_np.shape[2]) / 2),
               )
    print(f"Padding: {padding}")

    size = torch.tensor(fragment_size_np, dtype=torch.float).cuda()
    print(f"size: {size.shape} {size[0, 0, 0, 0, 0]}")

    # Tensors
    rho_o = torch.tensor(arr, dtype=torch.float).cuda()
    print(f"rho_o: {rho_o.shape}")

    masks = torch.tensor(mask_np, dtype=torch.float).cuda().reshape(1,1, shape[0], shape[1], shape[2])
    print(f"masks: {masks.shape}")

    # Convolutions
    signal_unscaled = torch.nn.functional.conv3d(rho_o, masks, padding=padding)
    print(f"rmsd: {signal_unscaled.shape} {signal_unscaled[0, 0, 24, 24, 24]}")

    signal = signal_unscaled / size
    print(f"signal: {signal.shape} {signal[0, 0, 24, 24, 24]}")

    signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0, )

    max_signal = torch.min(signal).cpu()
    print(f"max_signal: {max_signal}")

    max_index = np.unravel_index(torch.argmin(max_signal).cpu(), max_signal.shape)
    print(f"max_index: {max_index}")

    max_map_val = signal[max_index[0], max_index[1], max_index[2], max_index[3], max_index[4]]
    print(f"max_map_val: {max_map_val}")