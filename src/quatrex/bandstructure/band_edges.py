# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.


from qttools import NDArray, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
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


def _compute_eigenvalues(
    hamiltonian: DSDBSparse,
    overlap: DSDBSparse | None,
    potential: NDArray,
    sigma_retarded: DSDBSparse,
    ind: tuple[int, ...],
    diagonal_inds: tuple,
    upper_inds: tuple,
    order: str | NDArray | None = None,
    block_sections: int = 1,
    use_eigvalsh: bool = True,
    eigvalsh_compute_location: str = "numpy",
    use_pinned_memory: bool = True,
):
    """Computes the eigenvalues for the left or right contact.

    Parameters
    ----------
    hamiltonian : DSDBSparse
        The Hamiltonian.
    overlap : DSDBSparse | None
        The overlap matrix. If None, the basis is assumed to be
        orthogonal.
    potential : NDArray
        The potential.
    sigma_retarded : DSDBSparse
        The retarded self-energy.
    ind : tuple[int, ...]
        The local (E, k) index where the eigenvalues should be computed.
        This is a guess that should be as close as possible to where the
        band edge is to determine the correct band edge in this
        non-linear EVP. An outer loop should be used to refine this
        guess.
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

    big_blocksize = sigma_retarded.block_sizes[diagonal_inds[0]]
    small_blocksize = big_blocksize // block_sections

    # Hamiltonian is only extracted at a certain k-point
    # as it is not energy dependent.
    h_00 = order_block(hamiltonian.blocks[*diagonal_inds], order)[
        :, *ind[1:], :small_blocksize, :
    ]
    h_01 = order_block(hamiltonian.blocks[*upper_inds], order)[
        :, *ind[1:], :small_blocksize, :
    ]

    # Sigma is extract at a certain energy and k-point
    # NOTE: In this case we use only the real part of the retarded
    # self-energy.
    sigma_00 = xp.real(
        order_block(sigma_retarded.blocks[*diagonal_inds], order)[
            *ind, :small_blocksize, :
        ]
    )
    sigma_01 = xp.real(
        order_block(sigma_retarded.blocks[*upper_inds], order)[*ind, :small_blocksize]
    )

    h_0 = sum(
        h_00[..., i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(1, block_sections)
    )
    h_0 += sum(
        h_01[..., i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(block_sections)
    )
    h_0 += sum(
        sigma_00[..., i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(1, block_sections)
    )
    h_0 += sum(
        sigma_01[..., i * small_blocksize : (i + 1) * small_blocksize]
        for i in range(block_sections)
    )

    potential = order_vector(potential, order)
    if overlap is None:
        s_0 = None
        h_0 += h_0.conj().swapaxes(-2, -1)
        h_0 += xp.diag(potential[:small_blocksize])

    else:
        potential_0 = potential[:big_blocksize]
        potential_1 = order_vector(potential, order)[big_blocksize : 2 * big_blocksize]

        # The overlap is only extracted at a certain k-point
        # as it is not energy dependent.
        s_00 = order_block(overlap.blocks[*diagonal_inds], order)[
            :, *ind[1:], :small_blocksize, :
        ]
        s_01 = order_block(overlap.blocks[*upper_inds], order)[
            :, *ind[1:], :small_blocksize, :
        ]

        s_0 = sum(
            s_00[..., i * small_blocksize : (i + 1) * small_blocksize]
            for i in range(1, block_sections)
        )
        s_0 += sum(
            s_01[..., i * small_blocksize : (i + 1) * small_blocksize]
            for i in range(block_sections)
        )
        s_0 += s_0.conj().swapaxes(-2, -1)
        s_0 += s_00[..., :small_blocksize]

        h_0 += sum(
            s_00[..., i * small_blocksize : (i + 1) * small_blocksize]
            * potential_0[i * small_blocksize : (i + 1) * small_blocksize, None]
            + s_00[..., i * small_blocksize : (i + 1) * small_blocksize]
            * potential_0[i * small_blocksize : (i + 1) * small_blocksize]
            for i in range(1, block_sections)
        )
        h_0 += sum(
            sigma_01[..., i * small_blocksize : (i + 1) * small_blocksize]
            * potential_1[i * small_blocksize : (i + 1) * small_blocksize, None]
            + sigma_01[..., i * small_blocksize : (i + 1) * small_blocksize]
            * potential_1[i * small_blocksize : (i + 1) * small_blocksize]
            for i in range(block_sections)
        )

        # NOTE: The potential is added to the Hamiltonian as 0.5 * (S V + V S).
        h_0 += h_0.conj().swapaxes(-2, -1)
        h_0 += 0.5 * (
            s_00[..., :small_blocksize] * potential_0[:small_blocksize, None]
            + s_00[..., :small_blocksize] * potential_0[:small_blocksize]
        )

    h_0 += h_00[..., :small_blocksize]
    h_0 += sigma_00[..., :small_blocksize]

    # NOTE: Prevent eigvalsh from calling a batched routine (slow).
    h_0 = xp.squeeze(h_0)
    if s_0 is not None:
        s_0 = xp.squeeze(s_0)

    if use_eigvalsh:
        e_0 = eigvalsh(
            h_0,
            s_0,
            compute_module=eigvalsh_compute_location,
            use_pinned_memory=use_pinned_memory,
        )
        return xp.sort(e_0.real)

    raise NotImplementedError("Only use_eigvalsh=True is supported at the moment.")


def find_renormalized_eigenvalues(
    hamiltonian: DSDBSparse,
    overlap: DSDBSparse | None,
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
    hamiltonian : DSDBSparse
        The Hamiltonian.
    overlap : DSDBSparse | None
        The overlap matrix. If None, the basis is assumed to be
        orthogonal.
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
    section_sizes = xp.array(section_sizes)
    section_offsets = xp.hstack(([0], xp.cumsum(section_sizes)))

    left_band_edges = xp.empty(2, dtype=float)
    right_band_edges = xp.empty(2, dtype=float)

    if comm.block.rank == 0:
        for __ in range(num_ref_iterations):
            ind_left = xp.argmin(xp.abs(energies - left_conduction_band_guess))
            rank_left = xp.digitize(ind_left, section_offsets) - 1

            if rank_left == comm.stack.rank:
                # NOTE: This assumes that each rank has all k-points and that the band edge
                # is at the Gamma point.
                # TODO: Generalize this to arbitrary k-points (and maybe change gamma point index).
                local_ind = (ind_left - section_offsets[rank_left],) + tuple(
                    [s // 2 for s in sigma_retarded.shape[1:-2]]
                )
                e_0_left = _compute_eigenvalues(
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded=sigma_retarded,
                    ind=local_ind,
                    diagonal_inds=(0, 0),
                    upper_inds=(0, 1),
                    block_sections=band_edge_config.block_sections,
                    use_eigvalsh=band_edge_config.use_eigvalsh,
                    eigvalsh_compute_location=band_edge_config.eigvalsh_compute_location,
                    use_pinned_memory=band_edge_config.use_pinned_memory,
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

        comm.stack.bcast(left_band_edges, root=rank_left)

    if comm.block.rank == comm.block.size - 1:
        for __ in range(num_ref_iterations):
            ind_right = xp.argmin(xp.abs(energies - right_conduction_band_guess))
            rank_right = xp.digitize(ind_right, section_offsets) - 1

            if rank_right == comm.stack.rank:
                # NOTE: This assumes that each rank has all k-points and that the band edge
                # is at the Gamma point.
                # TODO: Generalize this to arbitrary k-points (and maybe change gamma point index).
                local_ind = (ind_right - section_offsets[rank_right],) + tuple(
                    [s // 2 for s in sigma_retarded.shape[1:-2]]
                )

                n = hamiltonian.num_local_blocks - 1
                m = n - 1
                e_0_right = _compute_eigenvalues(
                    hamiltonian=hamiltonian,
                    overlap=overlap,
                    potential=potential,
                    sigma_retarded=sigma_retarded,
                    ind=local_ind,
                    diagonal_inds=(n, n),
                    upper_inds=(n, m),
                    order="reverse",
                    block_sections=band_edge_config.block_sections,
                    use_eigvalsh=band_edge_config.use_eigvalsh,
                    eigvalsh_compute_location=band_edge_config.eigvalsh_compute_location,
                    use_pinned_memory=band_edge_config.use_pinned_memory,
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
