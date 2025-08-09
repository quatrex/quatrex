# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time
from functools import partial

import numpy as np
from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.kernels.linalg import eigvalsh
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import synchronize_device
from qttools.utils.mpi_utils import get_section_sizes

from quatrex.core.compute_config import BandEdgeConfig

profiler = Profiler()


if xp.__name__ == "numpy":
    from scipy.signal import find_peaks
elif xp.__name__ == "cupy":
    from cupyx.scipy.signal import find_peaks
else:
    raise ImportError("Unknown backend.")


def get_block(
    coo: sparse.coo_matrix | DSDBSparse,
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

    if isinstance(coo, DSDBSparse):
        start_block = coo.block_section_offsets[comm.block.rank]
        return coo.blocks[row - start_block, col - start_block]

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


@profiler.profile(level="api")
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
    peaks = find_peaks(dos, height=1e-3)[0]
    return energies[peaks]


@profiler.profile(level="debug")
def _compute_eigenvalues(
    hamiltonian: sparse.spmatrix | DSDBSparse,
    overlap: sparse.spmatrix,
    potential: NDArray,
    sigma_retarded: DSDBSparse,
    sigma_retarded_blocks_interp: tuple[NDArray, NDArray] | None,
    inds: int,
    weights: NDArray,
    side: str,
    band_edge_config: BandEdgeConfig = BandEdgeConfig(),
):
    """Computes the eigenvalues for the left or right contact."""
    big_blocksize = sigma_retarded.block_sizes[0]
    block_sections = band_edge_config.block_sections
    small_blocksize = big_blocksize // block_sections

    if side == "left":
        blocks = [(0, 0), (0, 1)]  # , (1, 0)]
        potential = xp.diag(potential[:small_blocksize])
        row_slice = slice(0, small_blocksize)
    elif side == "right":
        blocks = [(-1, -1), (-1, -2)]  # , (-2, -1)]
        potential = xp.diag(potential[-small_blocksize:])
        row_slice = slice(big_blocksize - small_blocksize, big_blocksize)
    else:
        raise ValueError(f"Unknown side '{side}'.")

    _get_block = partial(
        get_block,
        block_sizes=sigma_retarded.block_sizes,
        block_offsets=sigma_retarded.block_offsets,
    )

    h_00 = _get_block(hamiltonian, index=blocks[0])[0, row_slice]
    h_01 = _get_block(hamiltonian, index=blocks[1])[0, row_slice]
    s_00 = _get_block(overlap, index=blocks[0])[row_slice]
    s_01 = _get_block(overlap, index=blocks[1])[row_slice]

    if sigma_retarded_blocks_interp is None:
        # Interpolate the self-energy to the conduction band edge guess.
        sigma_00 = xp.real(_get_block(sigma_retarded, index=blocks[0]))[inds, row_slice]
        sigma_01 = xp.real(_get_block(sigma_retarded, index=blocks[1]))[inds, row_slice]
        # Average these along the energy axis to get the interpolated self-energy.
        sigma_00 = xp.average(sigma_00, axis=0, weights=weights)
        sigma_01 = xp.average(sigma_01, axis=0, weights=weights)
    else:
        # Get the local self-energy blocks.
        sigma_00 = xp.real(_get_block(sigma_retarded, index=blocks[0]))[
            inds[0], row_slice
        ]
        sigma_01 = xp.real(_get_block(sigma_retarded, index=blocks[1]))[
            inds[0], row_slice
        ]

        # Get the blocks that were sent from the other rank.
        sigma_00_interp, sigma_01_interp = sigma_retarded_blocks_interp
        sigma_00_interp = sigma_00_interp[row_slice]
        sigma_01_interp = sigma_01_interp[row_slice]

        # Interpolate the self-energy blocks.
        sigma_00 = sigma_00 * weights[0] + sigma_00_interp * weights[1]
        sigma_01 = sigma_01 * weights[0] + sigma_01_interp * weights[1]

    h_0 = sum(
        h_00[:, i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(1, block_sections)
    )
    h_0 += sum(
        h_01[:, i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(block_sections)
    )
    h_0 += sum(
        sigma_00[:, i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(1, block_sections)
    )
    h_0 += sum(
        sigma_01[:, i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(block_sections)
    )
    h_0 += h_0.conj().swapaxes(-2, -1)
    h_0 += h_00[:, :small_blocksize]
    h_0 += sigma_00[:, :small_blocksize]
    h_0 += potential

    s_0 = sum(
        s_00[:, i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(1, block_sections)
    )
    s_0 += sum(
        s_01[:, i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(block_sections)
    )
    s_0 += s_0.conj().swapaxes(-2, -1)
    s_0 += s_00[:, :small_blocksize]

    # NOTE: In this case we use only the real part of the retarded
    # self-energy.
    e_0 = eigvalsh(
        # NOTE: Prevent eigvalsh from calling a batched routine (slow).
        xp.squeeze(h_0),
        xp.squeeze(s_0),
        compute_module=band_edge_config.eigvalsh_compute_location,
        use_pinned_memory=band_edge_config.use_pinned_memory,
    )
    return xp.sort(e_0.real)


@profiler.profile(level="api")
def find_renormalized_eigenvalues(
    hamiltonian: sparse.spmatrix | DSDBSparse,
    overlap: sparse.spmatrix,
    potential: NDArray,
    sigma_retarded: DSDBSparse,
    energies: NDArray,
    conduction_band_guesses: tuple[float, float],
    mid_gap_energies: tuple[float, float],
    num_ref_iterations: int = 2,
    band_edge_config: BandEdgeConfig = BandEdgeConfig(),
) -> tuple[NDArray, NDArray]:
    """Computes renormalized eigenvalues for left and right contacts.

    Parameters
    ----------
    hamiltonian : sparse.spmatrix
        The Hamiltonian.
    overlap : sparse.spmatrix
        The overlap matrix.
    sigma_lesser : DSDBSparse
        The lesser self-energy.
    sigma_greater : DSDBSparse
        The greater self-energy.
    sigma_retarded : DSDBSparse
        The retarded self-energy.
    energies : NDArray
        The energies.
    local_energies : NDArray
        The local energies.
    conduction_band_guess : float
        A guess for the conduction band edge.
    num_ref_iterations : int, optional
        The number of refinement iterations, by default 2.
    use_eigvalsh : bool, optional
        Whether to assume the eigenvalue problem is Hermitian.
    eigvalsh_compute_location : str, optional
        The compute module to use in the eigvalsh call. By default, the
        Hermitian eigenvalue problem is solved on the GPU.

    Returns
    -------
    left_band_edges : NDArray
        The left band edges.
    right_band_edges : NDArray
        The right band edges.

    """

    # Find the rank that holds the energies corresponding to the initial
    # energy guess.
    left_conduction_band_guess, right_conduction_band_guess = conduction_band_guesses
    left_mid_gap_energy, right_mid_gap_energy = mid_gap_energies

    section_sizes, __ = get_section_sizes(energies.size, comm.stack.size)
    section_sizes = np.array(section_sizes)
    section_offsets = np.hstack(([0], np.cumsum(section_sizes)))

    left_band_edges = xp.empty(2, dtype=float)
    right_band_edges = xp.empty(2, dtype=float)

    if comm.block.rank == 0:
        for __ in range(num_ref_iterations):
            # We want to interpolate the self-energy to the conduction
            # band edge guess. So determine the indices of the energy
            # below and above the guess.
            ind_left = xp.argmin(xp.abs(energies - left_conduction_band_guess))
            if energies[ind_left] <= left_conduction_band_guess:
                inds_left = np.array([ind_left, ind_left + 1])
            else:
                inds_left = np.array([ind_left - 1, ind_left])

            # Ensure that the indices are within bounds.
            if inds_left[0] < 0 or inds_left[1] >= energies.size:
                raise ValueError(
                    "The conduction band edge guess is outside the range of the energies."
                    "Something has gone severely wrong."
                )

            # Compute the weights for the interpolation.
            dE = energies[inds_left[1]] - energies[inds_left[0]]
            weights = xp.array(
                [
                    (energies[inds_left[1]] - left_conduction_band_guess) / dE,
                    (left_conduction_band_guess - energies[inds_left[0]]) / dE,
                ]
            )

            # Determine which ranks hold the indices of the conduction
            # band edge guess.
            ranks_left = np.digitize(inds_left, section_offsets) - 1

            sigma_retarded_blocks_interp = None
            if ranks_left[0] != ranks_left[1]:
                # We need to send the self-energy blocks at the energy
                # above to the rank that holds the energy below.
                sigma_00 = xp.empty(
                    (sigma_retarded.block_sizes[0], sigma_retarded.block_sizes[0]),
                    dtype=xp.float64,
                )
                sigma_01 = xp.empty(
                    (sigma_retarded.block_sizes[0], sigma_retarded.block_sizes[1]),
                    dtype=xp.float64,
                )
                if ranks_left[1] == comm.stack.rank:
                    # NOTE: If we interpolate accross ranks, it will
                    # always be between the last local energy on the
                    # first rank and the first local energy on the
                    # second rank.
                    sigma_00 = xp.ascontiguousarray(
                        xp.real(sigma_retarded.blocks[0, 0][0])
                    )
                    sigma_01 = xp.ascontiguousarray(
                        xp.real(sigma_retarded.blocks[0, 1][0])
                    )

                    comm.stack._mpi_comm.Send(
                        sigma_00,
                        dest=ranks_left[0],
                        tag=comm.stack.rank,
                    )
                    comm.stack._mpi_comm.Send(
                        sigma_01,
                        dest=ranks_left[0],
                        tag=comm.stack.rank + 1,
                    )
                if ranks_left[0] == comm.stack.rank:
                    comm.stack._mpi_comm.Recv(
                        sigma_00,
                        source=ranks_left[1],
                        tag=ranks_left[1],
                    )
                    comm.stack._mpi_comm.Recv(
                        sigma_01,
                        source=ranks_left[1],
                        tag=ranks_left[1] + 1,
                    )
                    sigma_retarded_blocks_interp = (sigma_00, sigma_01)

            rank_left = ranks_left[0]

            if rank_left == comm.stack.rank:
                local_inds = inds_left - section_offsets[rank_left]
                e_0_left = _compute_eigenvalues(
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded=sigma_retarded,
                    sigma_retarded_blocks_interp=sigma_retarded_blocks_interp,
                    inds=local_inds,
                    weights=weights,
                    side="left",
                    band_edge_config=band_edge_config,
                )
                left_band_edges = find_band_edges(e_0_left, left_mid_gap_energy)
                left_mid_gap_energy = xp.mean(left_band_edges)
                __, left_conduction_band_guess = left_band_edges

            left_packed = xp.array([left_conduction_band_guess, left_mid_gap_energy])
            comm.stack.bcast(
                left_packed,
                root=rank_left,
            )
            left_conduction_band_guess, left_mid_gap_energy = left_packed

        synchronize_device()
        comm.stack.barrier()
        t_band_edge_start = time.perf_counter()
        comm.stack.bcast(left_band_edges, root=rank_left)
        synchronize_device()
        t_band_edge_end = time.perf_counter()
        comm.stack.barrier()
        t_band_edge_end_all = time.perf_counter()
        if comm.rank == 0:
            print(
                f"        Band edge comm time: {t_band_edge_end - t_band_edge_start:.3f} s",
                flush=True,
            )
            print(
                f"        Band edge comm all time: {t_band_edge_end_all - t_band_edge_start:.3f} s",
                flush=True,
            )

    if comm.block.rank == comm.block.size - 1:
        for __ in range(num_ref_iterations):
            # We want to interpolate the self-energy to the conduction
            # band edge guess. So determine the indices of the energy
            # below and above the guess.
            ind_right = xp.argmin(xp.abs(energies - right_conduction_band_guess))
            if energies[ind_right] <= right_conduction_band_guess:
                inds_right = np.array([ind_right, ind_right + 1])
            else:
                inds_right = np.array([ind_right - 1, ind_right])

            # Ensure that the indices are within bounds.
            if inds_right[0] < 0 or inds_right[1] >= energies.size:
                raise ValueError(
                    "The conduction band edge guess is outside the range of the energies."
                    "Something has gone severely wrong."
                )

            # Compute the weights for the interpolation.
            dE = energies[inds_right[1]] - energies[inds_right[0]]
            weights = xp.array(
                [
                    (energies[inds_right[1]] - right_conduction_band_guess) / dE,
                    (right_conduction_band_guess - energies[inds_right[0]]) / dE,
                ]
            )

            # Determine which ranks hold the indices of the conduction
            # band edge guess.
            ranks_right = np.digitize(inds_right, section_offsets) - 1

            sigma_retarded_blocks_interp = None
            if ranks_right[0] != ranks_right[1]:
                # We need to send the self-energy blocks at the energy
                # above to the rank that holds the energy below.
                sigma_00 = xp.empty(
                    (sigma_retarded.block_sizes[-1], sigma_retarded.block_sizes[-1]),
                    dtype=xp.float64,
                )
                sigma_01 = xp.empty(
                    (sigma_retarded.block_sizes[-1], sigma_retarded.block_sizes[-2]),
                    dtype=xp.float64,
                )
                if ranks_right[1] == comm.stack.rank:
                    # NOTE: If we interpolate accross ranks, it will
                    # always be between the last local energy on the
                    # first rank and the first local energy on the
                    # second rank.
                    sigma_00 = xp.real(sigma_retarded.blocks[-1, -1][0])
                    sigma_01 = xp.real(sigma_retarded.blocks[-1, -2][0])

                    comm.stack._mpi_comm.Send(
                        sigma_00,
                        dest=ranks_right[0],
                        tag=comm.stack.rank,
                    )
                    comm.stack._mpi_comm.Send(
                        sigma_01,
                        dest=ranks_right[0],
                        tag=comm.stack.rank + 1,
                    )
                if ranks_right[0] == comm.stack.rank:
                    comm.stack._mpi_comm.Recv(
                        sigma_00,
                        source=ranks_right[1],
                        tag=ranks_right[1],
                    )
                    comm.stack._mpi_comm.Recv(
                        sigma_01,
                        source=ranks_right[1],
                        tag=ranks_right[1] + 1,
                    )
                    sigma_retarded_blocks_interp = (sigma_00, sigma_01)

            rank_right = ranks_right[0]

            if rank_right == comm.stack.rank:
                local_inds = inds_right - section_offsets[rank_right]
                e_0_right = _compute_eigenvalues(
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded=sigma_retarded,
                    sigma_retarded_blocks_interp=sigma_retarded_blocks_interp,
                    inds=local_inds,
                    weights=weights,
                    side="right",
                    band_edge_config=band_edge_config,
                )
                right_band_edges = find_band_edges(e_0_right, right_mid_gap_energy)
                right_mid_gap_energy = xp.mean(right_band_edges)
                __, right_conduction_band_guess = right_band_edges

            right_packed = xp.array([right_conduction_band_guess, right_mid_gap_energy])
            comm.stack.bcast(right_packed, root=rank_right)
            right_conduction_band_guess, right_mid_gap_energy = right_packed

        comm.stack.bcast(right_band_edges, root=rank_right)

    comm.block.bcast(left_band_edges, root=0)
    comm.block.bcast(right_band_edges, root=comm.block.size - 1)

    return left_band_edges, right_band_edges


@profiler.profile(level="api")
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


@profiler.profile(level="api")
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
