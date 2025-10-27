# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.
from dataclasses import dataclass
from logging import config
from pathlib import Path
from typing import Optional

import scipy.sparse as sp

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.utils.gpu_utils import get_host
from qttools.utils.input_utils import create_hamiltonian, cutoff_hr
from qttools.utils.mpi_utils import distributed_load, get_section_sizes
from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig

""" hold input data for photon calculations.

Attributes:
    distance_unit_cells: Tensor of distances between orbitals in the device.
    small_block_sizes: Sizes of small blocks in the device Hamiltonian.
    hamiltonian_sparray: Sparse representation of the device Hamiltonian.
    block_sizes: Sizes of blocks in the device Hamiltonian.
"""


# temporary class to hold device configuration
@dataclass(frozen=True)
class DeviceConfig:
    construct_from_unit_cell: bool
    number_of_supercells: int
    transport_direction: str  # 'x'/'y'/'z'
    unit_cell_per_supercell: dict  # {'x':..,'y':..,'z':..}


# declaring the shape of the class
@dataclass(frozen=True)  # Immutable class (safe for parallel computations)
class IOConfig:
    input_dir: Path
    device: DeviceConfig
    example_input_dir: Optional[Path] = None


def load_distances(config: IOConfig) -> NDArray:
    """Load the distance tensor between orbitals in the device, of shape (N, N, 3).

    - if the device is constructed_from_unit_cell, load from file.
    - else, load from the example input directory.

    Args:
        config: Configuration for input/output.

    Returns:
        Tensor of distances between orbitals in the device.
    """
    if config.device.construct_from_unit_cell:
        distance_unit_cells = distributed_load(
            config.input_dir / "carbon-nanotube-dist.npy"
        ).astype(xp.complex128, copy=False)

        small_block_sizes = xp.asarray(
            [
                distance_unit_cells.shape[-1]
                * config.device.unit_cell_per_supercell[
                    "xyz".index(config.device.transport_direction)
                ]
            ]
            * config.device.number_of_supercells
        )
    # for later maybe
    # --- IGNORE ---
    # else:
    #     # Load block sizes.
    #     self.small_block_sizes = get_host(
    #         distributed_load(quatrex_config.input_dir / "block_sizes.xpy").astype(
    #             xp.int32
    #         )
    #     )
    else:
        print(
            "example_input_dir was required because we are not constructing from thes unit cell yet"
        )
        orbital_positions = xp.load(config.example_input_dir / "grid.npy")
        distance_unit_cells = (
            orbital_positions[:, None, :] - orbital_positions[None, :, :]
        ).astype(xp.complex128, copy=False)

    return distance_unit_cells


def load_hamiltonian_sparse(config: IOConfig) -> tuple[DSDBSparse, NDArray]:
    """
    Returns (hamiltonian_sparray[DSDBSparse complex128], block_sizes[xp.int32]).
    """

    if config.device.construct_from_unit_cell:
        hamiltonian_unit_cells = distributed_load(
            config.input_dir / "hamiltonian_unit_cells.npy"
        ).astype(xp.complex128, copy=False)

        section_sizes, __ = get_section_sizes(
            config.device.number_of_supercells, comm.block.size
        )
        section_offsets = xp.hstack(([0], xp.cumsum(section_sizes)))
        start_block = section_offsets[comm.block.rank]
        end_block = section_offsets[comm.block.rank + 1]

        hamiltonian_sparray, block_sizes = create_hamiltonian(
            cutoff_hr(
                hamiltonian_unit_cells,
                R_cutoff=config.device.unit_cell_per_supercell,
            ),
            config.device.number_of_supercells,
            config.device.transport_direction,
            config.device.unit_cell_per_supercell,
            block_start=start_block,
            block_end=end_block,
            return_sparse=True,
        )
        # normalize types
        hamiltonian_sparray = hamiltonian_sparray.astype(xp.complex128)
        hamiltonian_sparray.sum_duplicates()

        block_sizes = get_host(block_sizes)
        block_sizes = xp.asarray([block_sizes[0]] * config.device.number_of_supercells)

    else:
        print(
            "example_input_dir was required because we are not constructing from the unit cell yet"
        )
        hamiltonian_sparray = (
            sp.load_npz(config.example_input_dir / "hamiltonian.npz")
            .tocsr()
            .astype(xp.complex128, copy=False)
        )
        N = hamiltonian_sparray.shape[0]
        block_sizes = xp.array([N], dtype=xp.int32)

    return hamiltonian_sparray, block_sizes
