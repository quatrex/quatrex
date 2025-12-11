# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import numpy as np
from functools import partial

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.utils.gpu_utils import get_array_module_name, get_device, get_host
from scipy.optimize import minimize_scalar

from quatrex.bandstructure.band_edges import get_block, local_band_edges
from quatrex.core.statistics import fermi_dirac


def contact_band_structure(
    h_10: NDArray,
    h_00: NDArray,
    h_01: NDArray,
    num_k_points: int | None = None,
) -> NDArray:
    """Computes the band structure of a device contact.

    Parameters
    ----------
    h_10 : NDArray
        The off-diagonal element of the Hamiltonian.
    h_00 : NDArray
        The diagonal element of the Hamiltonian.
    h_01 : NDArray
        The off-diagonal element of the Hamiltonian.
    num_k_points : int, optional
        The number of k points. If not given, only the Gamma point is
        considered.

    Returns
    -------
    e_k : NDArray
        The sorted eigenvalues in energy and k.

    """
    k = (
        xp.linspace(-xp.pi, xp.pi, num_k_points)
        if num_k_points is not None
        else xp.array([0])
    )

    h_k = (
        h_01 * xp.exp(-1j * k)[:, xp.newaxis, xp.newaxis]
        + h_00
        + h_10 * xp.exp(1j * k)[:, xp.newaxis, xp.newaxis]
    )
    if get_array_module_name(h_k) == "cupy":
        e_k = get_device(np.linalg.eigvals(get_host(h_k)))
    else:
        e_k = np.linalg.eigvals(h_k)
    return xp.sort(e_k.real, axis=1)


def contact_dos(e_k: NDArray, energies: NDArray) -> NDArray:
    """Computes the density of states of a device contact.

    Parameters
    ----------
    e_k : NDArray
        The sorted eigenvalues in energy and k.
    energies : NDArray
        The energies.

    Returns
    -------
    dos : NDArray
        The density of states.

    """
    dos = np.zeros_like(energies)
    dos[:-1] = np.histogram(e_k, energies)[0]

    # Normalize the density of states.
    dos /= e_k.shape[0] * (energies[1] - energies[0])

    return dos


def extract_blocks(matrix, small_block_size):
    """
    Reshape a matrix into blocks.

    Parameters:
    -----------
    matrix : ndarray
        Input matrix of shape (..., big_block_size, big_block_size) where ... can be any additional dimensions
    small_block_size : int
        Size of each small block

    Returns:
    --------
    blocks : ndarray
        Reshaped blocks of shape (..., small_blocks_per_side, small_blocks_per_side, small_block_size, small_block_size)
    """
    big_block_size = matrix.shape[-1]
    if big_block_size % small_block_size != 0:
        raise ValueError("Big block size must be divisible by small block size.")
    small_blocks_per_side = big_block_size // small_block_size

    # Reshape and transpose to get blocks
    # Handle arbitrary leading dimensions
    leading_shape = matrix.shape[:-2]
    blocks = matrix.reshape(
        *leading_shape,
        small_blocks_per_side,
        small_block_size,
        small_blocks_per_side,
        small_block_size,
    )

    # Transpose to move block indices before element indices
    # Move axes: (..., i, si, j, sj) -> (..., i, j, si, sj)
    n_leading = len(leading_shape)
    axes = list(range(n_leading)) + [
        n_leading,
        n_leading + 2,
        n_leading + 1,
        n_leading + 3,
    ]
    blocks = blocks.transpose(axes)
    return blocks


def extract_sub_arrow_blocks(mat_nm, mat_mm, mat_mn, small_block_size, side="left"):
    """
    Extract and tile blocks according to the pattern shown.
        'left' side:        'right' side:
         ------- -------     ------- -------
        | x x x | x x x |   | o o o | o o x |
        | x o o | o o o |   | o o o | o o x |
        | x o o | o o o |   | o o o | o o x |
         ------- -------     ------- -------
        | x o o | o o o |   | o o o | o o x |
        | x o o | o o o |   | o o o | o o x |
        | x o o | o o o |   | x x x | x x x |
         ------- -------     ------- -------
    Parameters:
    -----------
    mat_nm, mat_mm, mat_mn : ndarray
        Input blocks of shape (..., big_block_size, big_block_size)
    small_block_size : int
        Size of each small block

    Returns:
    --------
    tiled_blocks : ndarray
        Tiled blocks of shape (..., total_small_blocks, small_block_size, small_block_size)
    """
    # Extract blocks using reshape_and_extract
    blocks_nm = extract_blocks(mat_nm, small_block_size)
    blocks_mm = extract_blocks(mat_mm, small_block_size)
    blocks_mn = extract_blocks(mat_mn, small_block_size)

    big_block_size = mat_mm.shape[-1]
    small_blocks_per_side = big_block_size // small_block_size

    leading_shape = mat_mm.shape[:-2]
    total_small_blocks = 4 * small_blocks_per_side - 1
    tiled_shape = (
        *leading_shape,
        total_small_blocks,
        small_block_size,
        small_block_size,
    )
    tiled_blocks = xp.zeros(tiled_shape, dtype=mat_mm.dtype)

    if side == "left":
        # Low left block (reversed vertically)
        tiled_blocks[..., :small_blocks_per_side, :, :] = blocks_nm[..., ::-1, 0, :, :]
        # Top left block (reversed vertically)
        tiled_blocks[..., small_blocks_per_side : 2 * small_blocks_per_side, :, :] = (
            blocks_mm[..., ::-1, 0, :, :]
        )
        # Top left block (horizontal, skip first)
        tiled_blocks[
            ..., 2 * small_blocks_per_side : 3 * small_blocks_per_side - 1, :, :
        ] = blocks_mm[..., 0, 1:, :, :]
        # Top right block (horizontal)
        tiled_blocks[..., 3 * small_blocks_per_side - 1 :, :, :] = blocks_mn[
            ..., 0, :, :, :
        ]
    elif side == "right":
        # Bottom left block (horizontal)
        tiled_blocks[..., :small_blocks_per_side, :, :] = blocks_mn[..., -1, :, :, :]
        # Bottom right block (horizontal)
        tiled_blocks[..., small_blocks_per_side : 2 * small_blocks_per_side, :, :] = (
            blocks_mm[..., -1, :, :, :]
        )
        # Bottom right block (reversed vertically, skip first)
        tiled_blocks[
            ..., 2 * small_blocks_per_side : 3 * small_blocks_per_side - 1, :, :
        ] = blocks_mm[..., -2::-1, -1, :, :]
        # Top right block (reversed vertically)
        tiled_blocks[..., 3 * small_blocks_per_side - 1 :, :, :] = blocks_nm[
            ..., ::-1, -1, :, :
        ]
    return tiled_blocks


def assemble_kpoint_matrix(tiled_blocks):
    """
    Assemble the k-point matrix from tiled blocks, where tiled blocks are in
    the 3rd dimension from the end.

    Parameters:
    -----------
    tiled_blocks : ndarray
        Tiled blocks of shape (..., total_small_blocks, small_block_size, small_block_size)
    Returns:
    --------
    kpoint_matrix : ndarray
        Assembled k-point matrix.
    """
    leading_shape = tiled_blocks.shape[:-3]
    total_small_blocks = tiled_blocks.shape[-3]
    num_kpoints = 16*total_small_blocks

    kpoint_matrix = xp.zeros(leading_shape + (num_kpoints, tiled_blocks.shape[-2], tiled_blocks.shape[-1]), dtype=tiled_blocks.dtype)

    # R shifts in x-direction
    x = xp.arange(-(total_small_blocks // 2), total_small_blocks // 2 + 1)
    # Reshape x for broadcasting
    x = x.reshape((1,) * len(leading_shape) + x.shape + (1, 1))
    # kpoints in transport direction
    kpoints = xp.linspace(-0.5, 0.5, num_kpoints, endpoint=False)
    for i, kp in enumerate(kpoints):
        phase_factors = xp.exp(2j * xp.pi * kp * x)
        kpoint_matrix[..., i, :, :] = xp.sum(
            tiled_blocks * phase_factors,
            axis=-3,
        )
    return kpoint_matrix


def contact_greens_function(
    hamiltonian: NDArray,
    overlap: NDArray,
    potential: NDArray,
    sigma_retarded: NDArray,
    energies: NDArray,
    eta: float = 1e-3,
) -> NDArray:
    """Computes the retarded Green's function of a device contact.

    Parameters
    ----------
    hamiltonian : NDArray
        The Hamiltonian of the contact.
    overlap : NDArray
        The overlap matrix of the contact.
    potential : NDArray
        The potential of the contact.
    sigma_retarded : NDArray
        The retarded self-energy of the contact.
    energies : NDArray
        The energies.

    Returns
    -------
    g_retarded : NDArray
        The retarded Green's function.

    """
    num_energies = energies.shape[0]
    g_retarded = np.zeros_like(sigma_retarded)

    for i in range(num_energies):
        energy_matrix = (energies[i]+1j * eta) * overlap + potential
        g_retarded[i] = np.linalg.inv(energy_matrix - hamiltonian - sigma_retarded[i])

    return g_retarded


def contact_fermi_level(
    temperature: float,
    dos: NDArray,
    energies: NDArray,
    doping_density: float,
    midgap_energy: float,
) -> float:
    """Computes the Fermi level of a device contact.

    This is done by minimizing the excess charge difference, while
    taking doping into account.

    Parameters
    ----------
    temperature : float
        The temperature.
    dos : NDArray
        The density of states.
    energies : NDArray
        The energies.
    doping_density : float
        The doping density.
    midgap_energy : float
        The energy at the middle of the band gap. This is used to
        separate conduction from valence bands.

    Returns
    -------
    float
        The Fermi level.

    """
    dE = energies[1] - energies[0]

    def objective_function(fermi_level):
        f = fermi_dirac(energies - fermi_level, temperature)
        n = (f * dos)[energies >= midgap_energy].sum() * dE
        p = ((1 - f) * dos)[energies < midgap_energy].sum() * dE
        # Apparently scipy minimize_scalar requires a numpy float
        return get_host(((n - p) - doping_density) ** 2)

    result = minimize_scalar(
        objective_function,
        bounds=(energies.min(), energies.max()),
        method="bounded",
    )

    return result.x


def find_charge_neutral_fermi_level(
    hamiltonian: DSDBSparse,
    overlap: DSDBSparse,
    potential: NDArray,
    sigma_retarded: DSDBSparse,
    local_energies: NDArray,
    energies: NDArray,
    temperature: float,
    target_charge: float,
    mid_gap_energy: float,
    block_sections: int = 1,
    side: str = "left",
) -> tuple[float, float]:
    """Finds the charge neutrality Fermi levels for left and right contacts.

    Parameters
    ----------
    hamiltonian : sparse.spmatrix
        The Hamiltonian.
    overlap : sparse.spmatrix
        The overlap matrix.
    sigma_retarded : DSDBSparse
        The retarded self-energy.
    local_energies : NDArray
        The local energies of each rank.
    energies : NDArray
        The energies.
    temperature : float
        The temperature in Kelvin.
    target_charge : float
        The target charge for the contact, should be in "unit cell" units,
        i.e., number of electrons per contact unit cell.
    mid_gap_energy : float
        The mid-gap energy for the contact.
    block_sections : int, optional
        The number of block sections to use, usually the
        number of blocks per contact, by default 1.
    side : str, optional
        The side to extract blocks from, either 'left' or 'right',

    Returns
    -------
    fermi_levels : tuple[float, float]
        The charge neutrality Fermi levels for left and right contacts.

    """
    big_blocksize = sigma_retarded.block_sizes[0]
    small_blocksize = big_blocksize // block_sections

    _get_block = partial(
        get_block,
        block_sizes=sigma_retarded.block_sizes,
        block_offsets=sigma_retarded.block_offsets,
    )

    if side == "left":
        blocks = [(1, 0), (0, 0), (0, 1)]
        potential = xp.diag(potential[:small_blocksize])
    elif side == "right":
        blocks = [(-1, -2), (-1, -1), (-2, -1)]
        potential = xp.diag(potential[-small_blocksize:])
    else:
        raise ValueError(f"Unknown side '{side}'.")

    h_R = extract_sub_arrow_blocks(
        _get_block(hamiltonian, index=blocks[0]),
        _get_block(hamiltonian, index=blocks[1]),
        _get_block(hamiltonian, index=blocks[2]),
        small_blocksize,
        side=side,
    )
    h_k = assemble_kpoint_matrix(h_R)

    s_R = extract_sub_arrow_blocks(
        _get_block(overlap, index=blocks[0]),
        _get_block(overlap, index=blocks[1]),
        _get_block(overlap, index=blocks[2]),
        small_blocksize,
        side=side,
    )
    s_k = assemble_kpoint_matrix(s_R)

    sigma_R = extract_sub_arrow_blocks(
        _get_block(sigma_retarded, index=blocks[0]),
        _get_block(sigma_retarded, index=blocks[1]),
        _get_block(sigma_retarded, index=blocks[2]),
        small_blocksize,
        side=side,
    )
    sigma_k = assemble_kpoint_matrix(sigma_R)

    g_k = contact_greens_function(
        hamiltonian=h_k,
        overlap=s_k,
        potential=potential,
        sigma_retarded=sigma_k,
        energies=local_energies,
    )

    
    dos_k = -(1 / np.pi) * np.imag(np.trace(g_k, axis1=-2, axis2=-1))
    # Mean over k-points (all axis except energy axis, which is the first axis)
    dos = dos_k.mean(axis=tuple(range(1, dos_k.ndim)))

    # Allgather dos from all ranks
    dos = comm.stack.all_gather_v(dos, axis=0)

    # Update the mid band gap from the dos
    vb_edge, cb_edge = local_band_edges(dos[:, None], energies, xp.array([mid_gap_energy,]))
    mid_gap_energy = float(0.5 * (vb_edge + cb_edge))

    fermi_level = contact_fermi_level(
        temperature=temperature,
        dos=dos,
        energies=energies,
        doping_density=target_charge,
        midgap_energy=mid_gap_energy,
    )

    return fermi_level, mid_gap_energy
