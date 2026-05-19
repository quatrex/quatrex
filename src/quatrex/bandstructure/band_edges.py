# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import warnings

import numpy as np

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _block_view
from qttools.kernels.linalg import eigvalsh
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.core.config import BandEdgeConfig
from quatrex.device.contact import order_block, order_vector

if xp.__name__ == "numpy":
    from scipy.signal import find_peaks
elif xp.__name__ == "cupy":
    from cupyx.scipy.signal import find_peaks
else:
    raise ImportError("Unknown backend.")


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


def _slice_sigma(
    target_energy: float,
    energies: NDArray,
    ind_lower: tuple[int, ...],
    ind_upper: tuple[int, ...],
    rank_lower: int,
    rank_upper: int,
    sigma_00: NDArray,
    sigma_01: NDArray,
    section_offsets: NDArray,
    sigma_slice: tuple[int, ...],
):
    """Slices the self-energy blocks at the given energy indices and performs
    interpolation if needed.

    Parameters
    ----------
    target_energy : float
        The energy at which the self-energy should be sliced.
    energies : NDArray
        The full energy grid.
    ind_lower : tuple[int, ...]
        The local (E,) index corresponding to the energy just below the target
        energy.
    ind_upper : tuple[int, ...]
        The local (E,) index corresponding to the energy just above the target
        energy.
    rank_lower : int
        The rank that holds the local index corresponding to the energy just
        below the target energy.
    rank_upper : int
        The rank that holds the local index corresponding to the energy just
        above the target energy.
    sigma_00 : NDArray
        The diagonal block of the self-energy at the given k-point.
    sigma_01 : NDArray
        The upper off-diagonal block of the self-energy at the given k-point.
    section_offsets : NDArray
        The offsets of the energy sections for each rank. This is needed to
        determine the global energy index.
    sigma_slice : tuple[int, ...]
        The slice corresponding to the k-point and block layer of interest.

    Returns
    -------
    sigma_00 : NDArray
        The diagonal block of the self-energy at the target energy and given
        k-point.
    sigma_01 : NDArray
        The upper off-diagonal block of the self-energy at the target energy and
        given k-point

    """

    if comm.stack.rank not in [rank_lower, rank_upper]:
        raise ValueError("Rank must be either rank_lower or rank_upper.")

    if comm.stack.rank == rank_lower:
        sigma_slice_lower = (ind_lower - section_offsets[rank_lower],) + sigma_slice
        sigma_00_lower = sigma_00[sigma_slice_lower].copy()
        sigma_01_lower = sigma_01[sigma_slice_lower].copy()
    if comm.stack.rank == rank_upper:
        sigma_slice_upper = (ind_upper - section_offsets[rank_upper],) + sigma_slice
        sigma_00_upper = sigma_00[sigma_slice_upper].copy()
        sigma_01_upper = sigma_01[sigma_slice_upper].copy()

    if rank_lower != rank_upper:
        if comm.stack.rank == rank_lower:
            sigma_00_upper = xp.empty_like(sigma_00_lower)
            sigma_01_upper = xp.empty_like(sigma_01_lower)

            partner_rank = rank_upper

        elif comm.stack.rank == rank_upper:
            sigma_00_lower = xp.empty_like(sigma_00_upper)
            sigma_01_lower = xp.empty_like(sigma_01_upper)

            partner_rank = rank_lower

        comm.stack.send_recv(
            sendbuf=sigma_00_lower if comm.stack.rank == rank_lower else sigma_00_upper,
            dest=partner_rank,
            recvbuf=sigma_00_upper if comm.stack.rank == rank_lower else sigma_00_lower,
            source=partner_rank,
        )

        comm.stack.send_recv(
            sendbuf=sigma_01_lower if comm.stack.rank == rank_lower else sigma_01_upper,
            dest=partner_rank,
            recvbuf=sigma_01_upper if comm.stack.rank == rank_lower else sigma_01_lower,
            source=partner_rank,
        )

    # Interpolate
    if ind_upper != ind_lower:
        energy_lower = energies[ind_lower]
        energy_upper = energies[ind_upper]
        sigma_00 = (sigma_00_upper - sigma_00_lower) * (
            target_energy - energy_lower
        ) / (energy_upper - energy_lower) + sigma_00_lower
        sigma_01 = (sigma_01_upper - sigma_01_lower) * (
            target_energy - energy_lower
        ) / (energy_upper - energy_lower) + sigma_01_lower
    else:
        sigma_00 = sigma_00_lower
        sigma_01 = sigma_01_lower

    return sigma_00, sigma_01


def _compute_eigenvalues(
    target_energy: float,
    energies: NDArray,
    hamiltonian: DSDBSparse,
    overlap: DSDBSparse | None,
    potential: NDArray,
    sigma_retarded_hermitian: DSDBSparse,
    ind_lower: tuple[int, ...],
    ind_upper: tuple[int, ...],
    rank_lower: int,
    rank_upper: int,
    section_offsets: NDArray,
    diagonal_inds: tuple,
    upper_inds: tuple,
    order: str | NDArray | None = None,
    block_sections: int = 1,
    use_eigvalsh: bool = True,
    eigvalsh_compute_location: str = "numpy",
    use_pinned_memory: bool = True,
):
    r"""Computes the eigenvalues for the left or right contact.

    Block sectioning is done in the following way:
    Assume 2 sections with
    ||  h00  || ||  h01  ||
    || a | b || || c | d ||
    || e | f || || g | h ||

    First $h = (b + c + d)$ is constructed, then
    $h += (b + c + d)^{\dagger}$
    and finally $h += a$.
    With the same logic, sigma and the potential are added.

    Parameters
    ----------
    target_energy : float
        The energy at which the eigenvalues should be computed.
        This is a guess that should be as close as possible to where the
        band edge is to determine the correct band edge in this
        non-linear EVP. An outer loop should be used to refine this
        guess.
    energies : NDArray
        The full energy grid.
    hamiltonian : DSDBSparse
        The Hamiltonian.
    overlap : DSDBSparse | None
        The overlap matrix. If None, the basis is assumed to be
        orthogonal.
    potential : NDArray
        The potential.
    sigma_retarded_hermitian : DSDBSparse
        The hermitian part of the retarded self-energy.
    ind_lower : tuple[int, ...]
        The local (E, k) index corresponding to the energy just below the target energy.
    ind_upper : tuple[int, ...]
        The local (E, k) index corresponding to the energy just above the target energy.
    rank_lower : int
        The rank that holds the local index corresponding to the energy just below the target energy.
    rank_upper : int
        The rank that holds the local index corresponding to the energy just above the target energy.
    section_offsets : NDArray
        The offsets of the energy sections for each rank.
        This is need to determine the global energy index.
    diagonal_inds : tuple
        The indices of the diagonal blocks corresponding to the contact.
    upper_inds : tuple
        The indices of the upper off-diagonal blocks corresponding to the contact.
    order : str | NDArray | None, optional
        The permutation of the blocks to achieve the same order as the canonical left contact.
        If None, the left contact order is assumed.
        Instead of an explicit permutation, the string "reverse" can be passed
        to reverse the order of the blocks, which is equivalent to the right contact order.
    block_sections : int, optional
        The number of block sections to assume in the computation. This is used to
        reduce the size of the eigenvalue problem by utilizing periodicity.
    use_eigvalsh : bool, optional
        Whether to assume the eigenvalue problem is Hermitian. By default, this is True.
        This is an approximation in the case of scattering since the self-energy is not Hermitian.
    eigvalsh_compute_location : str, optional
        The compute module to use in the eigvalsh call.
        Can be "numpy" or "cupy".
    use_pinned_memory : bool, optional
        Whether to use pinned memory in the eigvalsh call.
        This can speed up the computation when using `cupy`.

    Returns
    -------
    e_0 : NDArray
        The eigenvalues at the given (E, k) index sorted by energy in
        ascending order.

    """

    if not use_eigvalsh:
        raise NotImplementedError("Only use_eigvalsh=True is supported.")

    if comm.stack.rank not in [rank_lower, rank_upper]:
        raise ValueError("Rank must be either rank_lower or rank_upper.")

    big_blocksize = sigma_retarded_hermitian.block_sizes[diagonal_inds[0]]
    small_blocksize = big_blocksize // block_sections

    ind_k = tuple([s // 2 for s in sigma_retarded_hermitian.shape[1:-2]])

    h_slice = (np.s_[:],) + tuple(ind_k) + np.s_[:small_blocksize, :]
    sigma_slice = tuple(ind_k) + np.s_[:small_blocksize, :]

    # Sigma is extract at a certain energy and k-point
    sigma_00 = order_block(sigma_retarded_hermitian.blocks[*diagonal_inds], order)
    sigma_01 = order_block(sigma_retarded_hermitian.blocks[*upper_inds], order)

    sigma_00, sigma_01 = _slice_sigma(
        target_energy,
        energies,
        ind_lower,
        ind_upper,
        rank_lower,
        rank_upper,
        sigma_00,
        sigma_01,
        section_offsets,
        sigma_slice,
    )

    # Hamiltonian is only extracted at a certain k-point
    # as it is not energy dependent.
    h_00 = order_block(hamiltonian.blocks[*diagonal_inds], order)[h_slice]
    h_01 = order_block(hamiltonian.blocks[*upper_inds], order)[h_slice]

    # Extract the blocks corresponding to the first block layer
    h_00 = _block_view(h_00, axis=-1, num_blocks=block_sections)
    h_01 = _block_view(h_01, axis=-1, num_blocks=block_sections)

    sigma_00 = _block_view(sigma_00, axis=-1, num_blocks=block_sections)
    sigma_01 = _block_view(sigma_01, axis=-1, num_blocks=block_sections)

    h_0 = xp.zeros(h_00.shape[1:], dtype=h_00.dtype)
    h_0 += xp.sum(h_00[1:], axis=0)
    h_0 += xp.sum(h_01, axis=0)
    h_0 += xp.sum(sigma_00[1:], axis=0)
    h_0 += xp.sum(sigma_01, axis=0)

    potential = order_vector(potential, order)
    if overlap is not None:
        potential_0 = potential[:big_blocksize]
        potential_1 = potential[big_blocksize : 2 * big_blocksize]

        # match the shape of the hamiltonian blocks
        # this is done for the multiplication with the overlap blocks later on.
        potential_0 = potential_0.reshape(
            (block_sections, *((1,) * (len(h_00.shape) - 3)), small_blocksize)
        )
        potential_1 = potential_1.reshape(
            (block_sections, *((1,) * (len(h_00.shape) - 3)), small_blocksize)
        )

        # The overlap is only extracted at a certain k-point
        # as it is not energy dependent.
        s_00 = order_block(overlap.blocks[*diagonal_inds], order)[h_slice]
        s_01 = order_block(overlap.blocks[*upper_inds], order)[h_slice]

        s_00 = _block_view(s_00, axis=-1, num_blocks=block_sections)
        s_01 = _block_view(s_01, axis=-1, num_blocks=block_sections)

        s_0 = xp.zeros(s_00.shape[1:], dtype=s_00.dtype)
        s_0 += xp.sum(s_00[1:], axis=0)
        s_0 += xp.sum(s_01, axis=0)

        s_0 += s_0.conj().swapaxes(-2, -1)
        s_0 += s_00[0]

        ps_0 = 0.5 * (
            s_00 * potential_0[..., None, :] + s_00 * potential_0[..., :, None]
        )
        ps_1 = 0.5 * (
            s_01 * potential_1[..., None, :] + s_01 * potential_1[..., :, None]
        )

        h_0 += xp.sum(ps_0[1:], axis=0)
        h_0 += xp.sum(ps_1, axis=0)

    h_0 += h_0.conj().swapaxes(-2, -1)
    h_0 += h_00[0]
    h_0 += sigma_00[0]

    if overlap is None:
        s_0 = None
        h_0 += xp.diag(potential[:small_blocksize])
    else:
        h_0 += ps_0[0]

    # NOTE: Prevent eigvalsh from calling a batched routine (slow).
    h_0 = xp.squeeze(h_0)
    if s_0 is not None:
        s_0 = xp.squeeze(s_0)

    w = eigvalsh(
        h_0,
        s_0,
        compute_module=eigvalsh_compute_location,
        use_pinned_memory=use_pinned_memory,
    )
    return xp.sort(w.real)


def find_renormalized_eigenvalues(
    hamiltonian: DSDBSparse,
    overlap: DSDBSparse | None,
    potential: NDArray,
    sigma_retarded_hermitian: DSDBSparse,
    energies: NDArray,
    conduction_band_guesses: tuple[float, float],
    mid_gap_energies: tuple[float, float],
    num_ref_iterations: int = 2,
    band_edge_config: BandEdgeConfig = BandEdgeConfig(),
) -> tuple[NDArray, NDArray]:
    """Computes renormalized eigenvalues for left and right contacts.

    Parameters
    ----------
    hamiltonian : DSDBSparse
        The Hamiltonian.
    overlap : DSDBSparse | None
        The overlap matrix. If None, the basis is assumed to be
        orthogonal.
    sigma_retarded_hermitian : DSDBSparse
        The hermitian part of the retarded self-energy.
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
    section_sizes = xp.array(section_sizes)
    section_offsets = xp.hstack(([0], xp.cumsum(section_sizes)))

    left_band_edges = xp.empty(2, dtype=float)
    right_band_edges = xp.empty(2, dtype=float)

    if comm.block.rank == 0:
        for __ in range(num_ref_iterations):
            ind_left = xp.argmin(xp.abs(energies - left_conduction_band_guess))

            if energies[ind_left] < left_conduction_band_guess:
                ind_left_lower = ind_left
                ind_left_upper = ind_left + 1
            else:
                ind_left_lower = ind_left - 1
                ind_left_upper = ind_left

            # Sanity checks when the energy grid is unphysical,
            # but one still wants to benchmark
            if ind_left_upper >= len(energies):
                ind_left_upper = len(energies) - 1
                if comm.rank == 0:
                    warnings.warn(
                        "The initial guess for the conduction band edge is above the maximum energy. "
                        "Using the maximum energy for the upper index."
                    )

            if ind_left_lower < 0:
                ind_left_lower = 0
                if comm.rank == 0:
                    warnings.warn(
                        "The initial guess for the conduction band edge is below the minimum energy. "
                        "Using the minimum energy for the lower index."
                    )

            rank_left_lower = xp.digitize(ind_left_lower, section_offsets) - 1
            rank_left_upper = xp.digitize(ind_left_upper, section_offsets) - 1

            if comm.stack.rank in [rank_left_lower, rank_left_upper]:
                # NOTE: This assumes that each rank has all k-points and that the band edge
                # is at the Gamma point.
                # TODO: Generalize this to arbitrary k-points (and maybe change gamma point index).
                e_0_left = _compute_eigenvalues(
                    target_energy=left_conduction_band_guess,
                    energies=energies,
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded_hermitian=sigma_retarded_hermitian,
                    ind_lower=ind_left_lower,
                    ind_upper=ind_left_upper,
                    rank_lower=rank_left_lower,
                    rank_upper=rank_left_upper,
                    section_offsets=section_offsets,
                    diagonal_inds=(0, 0),
                    upper_inds=(0, 1),
                    block_sections=band_edge_config.block_sections,
                    use_eigvalsh=band_edge_config.use_eigvalsh,
                    eigvalsh_compute_location=band_edge_config.eigvalsh_compute_location,
                    use_pinned_memory=band_edge_config.use_pinned_memory,
                )

                # NOTE: Only the lower rank broadcasts and overwrites the guess of
                # the upper rank (which should be identical).
                left_band_edges = find_band_edges(e_0_left, left_mid_gap_energy)
                left_mid_gap_energy = xp.mean(left_band_edges)
                __, left_conduction_band_guess = left_band_edges

            left_packed = xp.array([left_conduction_band_guess, left_mid_gap_energy])
            comm.stack.bcast(
                left_packed,
                root=rank_left_lower,
            )
            left_conduction_band_guess, left_mid_gap_energy = left_packed

        comm.stack.bcast(left_band_edges, root=rank_left_lower)

    if comm.block.rank == comm.block.size - 1:
        for __ in range(num_ref_iterations):
            ind_right = xp.argmin(xp.abs(energies - right_conduction_band_guess))

            if energies[ind_right] < right_conduction_band_guess:
                ind_right_lower = ind_right
                ind_right_upper = ind_right + 1
            else:
                ind_right_lower = ind_right - 1
                ind_right_upper = ind_right

            if ind_right_upper >= len(energies):
                ind_right_upper = len(energies) - 1
                if comm.rank == comm.block.rank:
                    warnings.warn(
                        "The initial guess for the conduction band edge is above the maximum energy. "
                        "Using the maximum energy for the upper index."
                    )
            if ind_right_lower < 0:
                ind_right_lower = 0
                if comm.rank == comm.block.rank:
                    warnings.warn(
                        "The initial guess for the conduction band edge is below the minimum energy. "
                        "Using the minimum energy for the lower index."
                    )

            rank_right_lower = xp.digitize(ind_right_lower, section_offsets) - 1
            rank_right_upper = xp.digitize(ind_right_upper, section_offsets) - 1

            if comm.stack.rank in [rank_right_lower, rank_right_upper]:
                # NOTE: This assumes that each rank has all k-points and that the band edge
                # is at the Gamma point.
                # TODO: Generalize this to arbitrary k-points (and maybe change gamma point index).
                n = hamiltonian.num_local_blocks - 1
                m = n - 1
                e_0_right = _compute_eigenvalues(
                    target_energy=right_conduction_band_guess,
                    energies=energies,
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded_hermitian=sigma_retarded_hermitian,
                    ind_lower=ind_right_lower,
                    ind_upper=ind_right_upper,
                    rank_lower=rank_right_lower,
                    rank_upper=rank_right_upper,
                    section_offsets=section_offsets,
                    diagonal_inds=(n, n),
                    upper_inds=(n, m),
                    order="reverse",
                    block_sections=band_edge_config.block_sections,
                    use_eigvalsh=band_edge_config.use_eigvalsh,
                    eigvalsh_compute_location=band_edge_config.eigvalsh_compute_location,
                    use_pinned_memory=band_edge_config.use_pinned_memory,
                )

                # NOTE: Only the lower rank broadcasts and overwrites the guess of
                # the upper rank (which should be identical).
                right_band_edges = find_band_edges(e_0_right, right_mid_gap_energy)
                right_mid_gap_energy = xp.mean(right_band_edges)
                __, right_conduction_band_guess = right_band_edges

            right_packed = xp.array([right_conduction_band_guess, right_mid_gap_energy])
            comm.stack.bcast(right_packed, root=rank_right_lower)
            right_conduction_band_guess, right_mid_gap_energy = right_packed

        comm.stack.bcast(right_band_edges, root=rank_right_lower)

    comm.block.bcast(left_band_edges, root=0)
    comm.block.bcast(right_band_edges, root=comm.block.size - 1)

    return left_band_edges, right_band_edges


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
