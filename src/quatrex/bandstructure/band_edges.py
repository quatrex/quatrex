# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes band edges related functions."""

import warnings

import numpy as np

from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.datastructures.dsdbsparse import _block_view
from qttools.kernels.linalg import eigvalsh
from qttools.utils.mpi_utils import get_section_sizes
from quatrex.contact.scba import SCBAContact, order_block, order_vector
from quatrex.core.config import BandEdgeConfig
from quatrex.device import SCBADevice

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
    device: SCBADevice,
    sigma_retarded_hermitian: DSDBSparse,
    ind_lower: tuple[int, ...],
    ind_upper: tuple[int, ...],
    rank_lower: int,
    rank_upper: int,
    section_offsets: NDArray,
    contact: SCBAContact,
    band_edge_config: BandEdgeConfig = BandEdgeConfig(),
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
    device : SCBADevice
        The SCBA device.
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
    contact : SCBAContact
        The contact for which the eigenvalues should be computed.
    band_edge_config : BandEdgeConfig, optional
        The configuration for the band edge computation, by default
        BandEdgeConfig().

    Returns
    -------
    e_0 : NDArray
        The eigenvalues at the given (E, k) index sorted by energy in
        ascending order.

    """
    # TODO: Even after the second small refactoring due to k-point
    # symmetries, this function is a mess. Contact class should have
    # methods to extract blocks and do things like this.

    if not band_edge_config.use_eigvalsh:
        raise NotImplementedError("Only use_eigvalsh=True is supported.")

    if comm.stack.rank not in [rank_lower, rank_upper]:
        raise ValueError("Rank must be either rank_lower or rank_upper.")

    big_blocksize = sigma_retarded_hermitian.block_sizes[contact.diagonal_inds[0]]
    small_blocksize = big_blocksize // contact.transport_repetitions

    # Find k index closest to Gamma point (k=0) for the given kgrid.
    ind_k = np.argmin(np.linalg.norm(device.kgrid.points, axis=-1))

    h_slice = (np.s_[:], ind_k, *np.s_[:small_blocksize, :])
    sigma_slice = (ind_k, *np.s_[:small_blocksize, :])

    # Sigma is extract at a certain energy and k-point
    sigma_00 = order_block(
        sigma_retarded_hermitian.blocks[*contact.diagonal_inds],
        contact.order,
    )
    sigma_01 = order_block(
        sigma_retarded_hermitian.blocks[*contact.upper_inds],
        contact.order,
    )

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
    h_00 = order_block(
        device.hamiltonians.blocks[*contact.diagonal_inds],
        contact.order,
    )[h_slice]
    h_01 = order_block(
        device.hamiltonians.blocks[*contact.upper_inds],
        contact.order,
    )[h_slice]

    # Extract the blocks corresponding to the first block layer
    h_00 = _block_view(h_00, axis=-1, num_blocks=contact.transport_repetitions)
    h_01 = _block_view(h_01, axis=-1, num_blocks=contact.transport_repetitions)

    sigma_00 = _block_view(sigma_00, axis=-1, num_blocks=contact.transport_repetitions)
    sigma_01 = _block_view(sigma_01, axis=-1, num_blocks=contact.transport_repetitions)

    h_0 = xp.zeros(h_00.shape[1:], dtype=h_00.dtype)
    h_0 += xp.sum(h_00[1:], axis=0)
    h_0 += xp.sum(h_01, axis=0)
    h_0 += xp.sum(sigma_00[1:], axis=0)
    h_0 += xp.sum(sigma_01, axis=0)

    potential = order_vector(device.potential, contact.order)
    if device.overlap_matrices is not None:
        potential_0 = potential[:big_blocksize]
        potential_1 = potential[big_blocksize : 2 * big_blocksize]

        # match the shape of the hamiltonian blocks
        # this is done for the multiplication with the overlap blocks later on.
        potential_0 = potential_0.reshape(
            (
                contact.transport_repetitions,
                *((1,) * (len(h_00.shape) - 3)),
                small_blocksize,
            )
        )
        potential_1 = potential_1.reshape(
            (
                contact.transport_repetitions,
                *((1,) * (len(h_00.shape) - 3)),
                small_blocksize,
            )
        )

        # The overlap is only extracted at a certain k-point
        # as it is not energy dependent.
        s_00 = order_block(
            device.overlap_matrices.blocks[*contact.diagonal_inds],
            contact.order,
        )[h_slice]
        s_01 = order_block(
            device.overlap_matrices.blocks[*contact.upper_inds],
            contact.order,
        )[h_slice]

        s_00 = _block_view(s_00, axis=-1, num_blocks=contact.transport_repetitions)
        s_01 = _block_view(s_01, axis=-1, num_blocks=contact.transport_repetitions)

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

    if device.overlap_matrices is None:
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
        compute_module=band_edge_config.eigvalsh_compute_location,
        use_pinned_memory=band_edge_config.use_pinned_memory,
    )
    return xp.sort(w.real)


def find_renormalized_band_edges(
    device: SCBADevice,
    sigma_retarded_hermitian: DSDBSparse,
    energies: NDArray,
    contact: SCBAContact,
    conduction_band_guess: float,
    band_edge_config: BandEdgeConfig = BandEdgeConfig(),
) -> NDArray:
    """Computes renormalized eigenvalues for a specific contact.

    The eigenvalues are computed at the Gamma point (k=0) and are used
    to determine the band edges.

    Note
    ----
    Only the block comm ranks holding the specific contact should call
    this function. Otherwise, there will be unexpected behavior.

    Parameters
    ----------
    device : SCBADevice
        The SCBA device.
    sigma_retarded_hermitian : DSDBSparse
        The hermitian part of the retarded self-energy.
    energies : NDArray
        The energies.
    contact : SCBAContact
        The contact for which the band edges should be computed.
    conduction_band_guess : float
        An initial guess for the conduction band edge.
    band_edge_config : BandEdgeConfig, optional
        The configuration for the band edge computation, by default
        BandEdgeConfig().

    Returns
    -------
    band_edges : NDArray
        The band edges.

    """

    # Find the rank that holds the energies corresponding to the initial
    # energy guess.
    section_sizes, __ = get_section_sizes(energies.size, comm.stack.size)
    section_sizes = xp.array(section_sizes)
    section_offsets = xp.hstack(([0], xp.cumsum(section_sizes)))

    band_edges = xp.empty(2, dtype=float)
    mid_gap_energy = contact.mid_gap_energy

    for __ in range(band_edge_config.num_ref_iterations):
        ind = xp.argmin(xp.abs(energies - conduction_band_guess))

        if energies[ind] < conduction_band_guess:
            ind_lower = ind
            ind_upper = ind + 1
        else:
            ind_lower = ind - 1
            ind_upper = ind

        # Sanity checks when the energy grid is unphysical,
        # but one still wants to benchmark
        if ind_upper >= len(energies):
            ind_upper = len(energies) - 1
            if comm.stack.rank == 0:
                warnings.warn(
                    "The initial guess for the conduction band edge is above the maximum energy. "
                    "Using the maximum energy for the upper index."
                )

        if ind_lower < 0:
            ind_lower = 0
            if comm.stack.rank == 0:
                warnings.warn(
                    "The initial guess for the conduction band edge is below the minimum energy. "
                    "Using the minimum energy for the lower index."
                )

        rank_lower = xp.digitize(ind_lower, section_offsets) - 1
        rank_upper = xp.digitize(ind_upper, section_offsets) - 1

        if comm.stack.rank in [rank_lower, rank_upper]:
            # NOTE: This assumes that each rank has all k-points and that the band edge
            # is at the Gamma point.
            # TODO: Generalize this to arbitrary k-points (and maybe change gamma point index).
            e_0 = _compute_eigenvalues(
                target_energy=conduction_band_guess,
                energies=energies,
                device=device,
                sigma_retarded_hermitian=sigma_retarded_hermitian,
                ind_lower=ind_lower,
                ind_upper=ind_upper,
                rank_lower=rank_lower,
                rank_upper=rank_upper,
                section_offsets=section_offsets,
                contact=contact,
                band_edge_config=band_edge_config,
            )

            # NOTE: Only the lower rank broadcasts and overwrites the guess of
            # the upper rank (which should be identical).
            band_edges = find_band_edges(e_0, mid_gap_energy)
            mid_gap_energy = xp.mean(band_edges)
            __, conduction_band_guess = band_edges

        packed = xp.array([conduction_band_guess, mid_gap_energy])
        comm.stack.bcast(
            packed,
            root=rank_lower,
        )
        conduction_band_guess, mid_gap_energy = packed

    comm.stack.bcast(band_edges, root=rank_lower)

    return band_edges


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
    electron_ldos : NDArray
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
