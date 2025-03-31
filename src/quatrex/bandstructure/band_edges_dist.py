# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

import time
from functools import partial

from qttools import (
    NCCL_AVAILABLE,
    NDArray,
    block_comm,
    global_comm,
    nccl_block_comm,
    nccl_stack_comm,
    sparse,
    stack_comm,
    xp,
)
from qttools.datastructures import DSDBSparse
from qttools.kernels.linalg import eigvalsh
from qttools.profiling import Profiler
from qttools.utils.gpu_utils import get_device, get_host, synchronize_device
from qttools.utils.mpi_utils import check_gpu_aware_mpi, get_section_sizes
from scipy import linalg as spla

from quatrex.core.compute_config import BandEdgeConfig

profiler = Profiler()

GPU_AWARE_MPI = check_gpu_aware_mpi()

if xp.__name__ == "numpy":
    from scipy.signal import find_peaks
elif xp.__name__ == "cupy":
    from cupyx.scipy.signal import find_peaks
else:
    raise ImportError("Unknown backend.")


def _bcast(values: list | NDArray, num_values: int, root: int, block: bool) -> float:
    """Broadcasts a list values to all ranks.

    Parameters
    ----------
    value : float
        The value to broadcast.
    root : int
        The rank to broadcast from.
    """

    if values is None:
        buf = xp.empty(num_values, dtype=xp.float64)
    else:
        buf = xp.asarray(values)

    if block:
        mpi_comm, nccl_comm = block_comm, nccl_block_comm
    else:
        mpi_comm, nccl_comm = stack_comm, nccl_stack_comm

    if NCCL_AVAILABLE:
        nccl_comm.broadcast(buf, root)
        synchronize_device()
    elif xp.__name__ == "numpy" or GPU_AWARE_MPI:
        mpi_comm.Bcast(buf, root)
    else:
        mpi_comm.bcast(buf, root)

    return buf


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
        start_block = coo.block_section_offsets[block_comm.rank]
        return coo.local_blocks[row - start_block, col - start_block]

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
    peaks = find_peaks(dos, height=1e-8)[0]
    return energies[peaks]


@profiler.profile(level="debug")
def _compute_eigenvalues(
    hamiltonian: sparse.spmatrix | DSDBSparse,
    overlap: sparse.spmatrix,
    potential: NDArray,
    sigma_retarded: DSDBSparse,
    ind: int,
    side: str,
    band_edge_config: BandEdgeConfig = BandEdgeConfig(),
):
    """Computes the eigenvalues for the left or right contact."""
    if side == "left":
        blocks = [(0, 0), (0, 1), (1, 0)]
        sigma_blocks = blocks
        potential = xp.diag(potential[: sigma_retarded.block_sizes[0]])
    elif side == "right":
        blocks = [(-1, -1), (-1, -2), (-2, -1)]
        n = sigma_retarded.num_local_blocks - 1
        m = n - 1
        sigma_blocks = [(n, n), (n, m), (m, n)]
        potential = xp.diag(potential[-sigma_retarded.block_sizes[-1] :])
    else:
        raise ValueError(f"Unknown side '{side}'.")

    _get_block = partial(
        get_block,
        block_sizes=sigma_retarded.block_sizes,
        block_offsets=sigma_retarded.block_offsets,
    )

    s_0 = sum(_get_block(overlap, index=block) for block in blocks)

    h_0 = sum(_get_block(hamiltonian, index=block) for block in blocks) + potential
    if band_edge_config.use_eigvalsh:
        # NOTE: In this case we use only the real part of the retarded
        # self-energy.
        h_0 += sum(
            xp.real(sigma_retarded.local_blocks[*block][ind]) for block in sigma_blocks
        )
        e_0 = eigvalsh(
            # NOTE: Prevent eigvalsh from calling a batched routine (slow).
            xp.squeeze(h_0),
            xp.squeeze(s_0),
            compute_module=band_edge_config.eigvalsh_compute_location,
            use_pinned_memory=band_edge_config.use_pinned_memory,
        )
        return xp.sort(e_0.real)

    h_0 += sum(sigma_retarded.local_blocks[*block][ind] for block in sigma_blocks)
    e_0 = get_device(spla.eigvals(get_host(h_0), get_host(s_0)))
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
    e_0_left : NDArray
        The renormalized eigenvalues for the left contact.
    e_0_right : NDArray
        The renormalized eigenvalues for the right contact.

    """

    # Find the rank that holds the energies corresponding to the initial
    # energy guess.
    left_conduction_band_guess, right_conduction_band_guess = conduction_band_guesses
    left_mid_gap_energy, right_mid_gap_energy = mid_gap_energies

    section_sizes, __ = get_section_sizes(energies.size, stack_comm.size)
    section_sizes = xp.array(section_sizes)
    section_offsets = xp.hstack(([0], xp.cumsum(section_sizes)))

    e_0_left = None
    e_0_right = None

    if block_comm.rank == 0:
        for __ in range(num_ref_iterations):
            ind_left = xp.argmin(xp.abs(energies - left_conduction_band_guess))
            rank_left = xp.digitize(ind_left, section_offsets) - 1

            if rank_left == stack_comm.rank:
                local_ind = ind_left - section_offsets[rank_left]
                e_0_left = _compute_eigenvalues(
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded=sigma_retarded,
                    ind=local_ind,
                    side="left",
                    band_edge_config=band_edge_config,
                )
                left_valence_band, left_conduction_band_guess = find_band_edges(
                    e_0_left, left_mid_gap_energy
                )
                left_mid_gap_energy = (
                    left_valence_band + left_conduction_band_guess
                ) / 2

            # left_conduction_band_guess = stack_comm.bcast(
            #     left_conduction_band_guess, rank_left
            # )
            # left_mid_gap_energy = stack_comm.bcast(left_mid_gap_energy, rank_left)
            left_conduction_band_guess, left_mid_gap_energy = _bcast(
                [left_conduction_band_guess, left_mid_gap_energy],
                2,
                rank_left,
                block=False,
            )

        # e_0_left = stack_comm.bcast(e_0_left, rank_left)
        synchronize_device()
        stack_comm.Barrier()
        t_eigvals_start = time.perf_counter()
        e_0_left = _bcast(
            e_0_left, sigma_retarded.block_sizes[0], rank_left, block=False
        )
        synchronize_device()
        t_eigvals_end = time.perf_counter()
        stack_comm.Barrier()
        t_eigvals_end_all = time.perf_counter()
        if global_comm.rank == 0:
            print(
                f"        Eigvals comm time: {t_eigvals_end - t_eigvals_start:.3f} s",
                flush=True,
            )
            print(
                f"        Eigvals comm all time: {t_eigvals_end_all - t_eigvals_start:.3f} s",
                flush=True,
            )

    if block_comm.rank == block_comm.size - 1:
        for __ in range(num_ref_iterations):
            ind_right = xp.argmin(xp.abs(energies - right_conduction_band_guess))
            rank_right = xp.digitize(ind_right, section_offsets) - 1

            if rank_right == stack_comm.rank:
                local_ind = ind_right - section_offsets[rank_right]
                e_0_right = _compute_eigenvalues(
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded=sigma_retarded,
                    ind=local_ind,
                    side="right",
                    band_edge_config=band_edge_config,
                )
                right_valence_band, right_conduction_band_guess = find_band_edges(
                    e_0_right, right_mid_gap_energy
                )
                right_mid_gap_energy = (
                    right_valence_band + right_conduction_band_guess
                ) / 2

            # right_conduction_band_guess = stack_comm.bcast(
            #     right_conduction_band_guess, rank_right
            # )
            # right_mid_gap_energy = stack_comm.bcast(right_mid_gap_energy, rank_right)
            right_conduction_band_guess, right_mid_gap_energy = _bcast(
                [right_conduction_band_guess, right_mid_gap_energy],
                2,
                rank_right,
                block=False,
            )

        # e_0_right = stack_comm.bcast(e_0_right, rank_right)
        e_0_right = _bcast(
            e_0_right, sigma_retarded.block_sizes[-1], rank_right, block=False
        )

    # e_0_left = block_comm.bcast(e_0_left, 0)
    # e_0_right = block_comm.bcast(e_0_right, block_comm.size - 1)
    e_0_left = _bcast(e_0_left, sigma_retarded.block_sizes[0], 0, block=True)
    e_0_right = _bcast(
        e_0_right, sigma_retarded.block_sizes[-1], block_comm.size - 1, block=True
    )

    return e_0_left, e_0_right


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
