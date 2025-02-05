# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from functools import partial

from mpi4py.MPI import COMM_WORLD as comm
from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.utils.gpu_utils import get_device, get_host
from qttools.utils.mpi_utils import get_section_sizes
from scipy import linalg as spla

if xp.__name__ == "numpy":
    from scipy.signal import find_peaks
elif xp.__name__ == "cupy":
    from cupyx.scipy.signal import find_peaks
else:
    raise ImportError("Unknown backend.")


def get_block(
    coo: sparse.coo_matrix,
    block_sizes: NDArray,
    block_offsets: NDArray,
    index: tuple,
) -> NDArray:
    """Gets a block from a COO matrix.

    Parameters
    ----------
    coo : sparse.coo_matrix
        The COO matrix.
    block_sizes : NDArray
        The block sizes.
    block_offsets : NDArray
        The block offsets.
    index : tuple
        The index of the block to extract.

    Returns
    -------
    block : NDArray
        The requested, dense block.

    """
    row, col = index
    row = row + len(block_sizes) if row < 0 else row
    col = col + len(block_sizes) if col < 0 else col

    mask = (
        (block_offsets[row] <= coo.row)
        & (coo.row < block_offsets[row + 1])
        & (block_offsets[col] <= coo.col)
        & (coo.col < block_offsets[col + 1])
    )
    block = xp.zeros((int(block_sizes[row]), int(block_sizes[col])), dtype=coo.dtype)
    block[
        coo.row[mask] - block_offsets[row],
        coo.col[mask] - block_offsets[col],
    ] = coo.data[mask]

    return block


def find_dos_peaks(dos: NDArray, energies: NDArray) -> NDArray:
    """Computes the band edges from the density of states.

    Parameters
    ----------
    dos : NDArray
        The density of states.
    energies : NDArray
        The energies corresponding to the DOS.

    Returns
    -------
    e_0 : NDArray
        Suspected band edges sorted by energy in ascending order.

    """
    peaks = find_peaks(dos, height=1e-8)[0]
    return energies[peaks]


def _compute_eigenvalues(
    hamiltonian: sparse.spmatrix,
    overlap: sparse.spmatrix,
    potential: NDArray,
    sigma_retarded: DSBSparse,
    ind: int,
    side: str,
):
    """Computes the eigenvalues for the left or right contact."""
    if side == "left":
        blocks = [(0, 0), (0, 1), (1, 0)]
        potential = xp.diag(potential[: sigma_retarded.block_sizes[0]])
    elif side == "right":
        blocks = [(-1, -1), (-1, -2), (-2, -1)]
        potential = xp.diag(potential[-sigma_retarded.block_sizes[-1] :])
    else:
        raise ValueError(f"Unknown side '{side}'.")

    _get_block = partial(
        get_block,
        block_sizes=sigma_retarded.block_sizes,
        block_offsets=sigma_retarded.block_offsets,
    )

    h_0 = (
        sum(
            _get_block(hamiltonian, index=block) + sigma_retarded.blocks[*block][ind]
            for block in blocks
        )
        + potential
    )
    s_0 = sum(_get_block(overlap, index=block) for block in blocks)
    e_0 = get_device(spla.eigvals(get_host(h_0), get_host(s_0)))
    return xp.sort(e_0.real)


def find_renormalized_eigenvalues(
    hamiltonian: sparse.spmatrix,
    overlap: sparse.spmatrix,
    potential: NDArray,
    sigma_retarded: DSBSparse,
    energies: NDArray,
    conduction_band_guesses: tuple[float, float],
    mid_gap_energies: tuple[float, float],
    num_ref_iterations: int = 2,
) -> tuple[NDArray, NDArray]:
    """Computes renormalized eigenvalues for left and right contacts.

    Parameters
    ----------
    hamiltonian : sparse.spmatrix
        The Hamiltonian.
    overlap : sparse.spmatrix
        The overlap matrix.
    sigma_lesser : DSBSparse
        The lesser self-energy.
    sigma_greater : DSBSparse
        The greater self-energy.
    sigma_retarded : DSBSparse
        The retarded self-energy.
    energies : NDArray
        The energies.
    local_energies : NDArray
        The local energies.
    conduction_band_guess : float
        A guess for the conduction band edge.
    num_ref_iterations : int, optional
        The number of refinement iterations, by default 2.

    Returns
    -------
    e_0_left : NDArray
        The renormalized eigenvalues for the left contact.
    e_0_right : NDArray
        The renormalized eigenvalues for the right contact.

    """

    # Find the rank that holds the energies corresponding to the initial
    # energy guess.
    left_conduction_band_guess, right_conduction_band_guess = conduction_band_guesses
    left_mid_gap_energy, right_mid_gap_energy = mid_gap_energies

    section_sizes, __ = get_section_sizes(energies.size, comm.size)
    section_sizes = xp.array(section_sizes)
    section_offsets = xp.hstack(([0], xp.cumsum(section_sizes)))

    e_0_left = None
    e_0_right = None

    for __ in range(num_ref_iterations):
        ind_left = xp.argmin(xp.abs(energies - left_conduction_band_guess))
        rank_left = xp.digitize(ind_left, section_offsets) - 1

        ind_right = xp.argmin(xp.abs(energies - right_conduction_band_guess))
        rank_right = xp.digitize(ind_right, section_offsets) - 1

        if rank_left == comm.rank:
            local_ind = ind_left - section_offsets[rank_left]
            e_0_left = _compute_eigenvalues(
                hamiltonian, overlap, potential, sigma_retarded, local_ind, "left"
            )
            left_valence_band, left_conduction_band_guess = find_band_edges(
                e_0_left, left_mid_gap_energy
            )
            left_mid_gap_energy = (left_valence_band + left_conduction_band_guess) / 2

        if rank_right == comm.rank:
            local_ind = ind_right - section_offsets[rank_right]
            e_0_right = _compute_eigenvalues(
                hamiltonian, overlap, potential, sigma_retarded, local_ind, "right"
            )
            right_valence_band, right_conduction_band_guess = find_band_edges(
                e_0_right, right_mid_gap_energy
            )
            right_mid_gap_energy = (
                right_valence_band + right_conduction_band_guess
            ) / 2

        left_conduction_band_guess = comm.bcast(left_conduction_band_guess, rank_left)
        left_mid_gap_energy = comm.bcast(left_mid_gap_energy, rank_left)
        right_conduction_band_guess = comm.bcast(
            right_conduction_band_guess, rank_right
        )
        right_mid_gap_energy = comm.bcast(right_mid_gap_energy, rank_right)

    e_0_left = comm.bcast(e_0_left, rank_left)
    e_0_right = comm.bcast(e_0_right, rank_right)

    return e_0_left, e_0_right


def find_band_edges(e_0: NDArray, mid_gap_energy: float) -> NDArray:
    """Partitions the band edges into valence and conduction bands.

    Parameters
    ----------
    e_0 : NDArray
        Eigenvalues at Gamma or the suspected band edges sorted by
        energy in ascending order.
    mid_gap_energy : float
        An energy in the band gap. This is used to separate conduction
        from valence bands.

    Returns
    -------
    band_edges : NDArray
        The valence and conduction band edges.

    """
    mask = (e_0 - mid_gap_energy) < 0
    valence_band_edge = e_0[mask].max()
    conduction_band_edge = e_0[~mask].min()
    return xp.array([valence_band_edge, conduction_band_edge])


def local_band_edges(
    electron_ldos: NDArray, energies: NDArray, mid_gap_energies: NDArray
) -> tuple[NDArray, NDArray]:
    """Computes the band edges from the local density of states.

    Parameters
    ----------
    ldos : NDArray
        The local density of states.
    energies : NDArray
        The energies corresponding to the LDOS.
    mid_gap_energies : NDArray
        The mid-gap energies through the whole device.

    Returns
    -------
    valence_band_edges : NDArray
        The valence band edges.
    conduction_band_edges : NDArray
        The conduction band edges.

    """
    conduction_band_edges = xp.zeros_like(mid_gap_energies)
    valence_band_edges = xp.zeros_like(mid_gap_energies)
    for i in range(electron_ldos.shape[1]):
        e_0 = find_dos_peaks(xp.abs(electron_ldos[:, i]), energies)
        valence_band_edges[i], conduction_band_edges[i] = find_band_edges(
            e_0, mid_gap_energies[i]
        )

    return valence_band_edges, conduction_band_edges
