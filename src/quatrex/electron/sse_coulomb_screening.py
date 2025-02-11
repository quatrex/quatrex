# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.

from qttools import NDArray, sparse, xp
from qttools.datastructures import DSBSparse
from qttools.utils.mpi_utils import distributed_load

from quatrex.core.compute_config import ComputeConfig
from quatrex.core.quatrex_config import QuatrexConfig
from quatrex.core.sse import ScatteringSelfEnergy


def fft_convolve(a: NDArray, b: NDArray) -> NDArray:
    """Computes the convolution of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The convolution of the two arrays.

    """
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b, (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


def fft_correlate(a: NDArray, b: NDArray) -> NDArray:
    """Computes the correlation of two arrays using FFT.

    Parameters
    ----------
    a : NDArray
        First array.
    b : NDArray
        Second array.

    Returns
    -------
    NDArray
        The cross-correlation of the two arrays.

    """
    n = a.shape[0] + b.shape[0] - 1
    a_fft = xp.fft.fftn(a, (n,), axes=(0,))
    b_fft = xp.fft.fftn(b[::-1], (n,), axes=(0,))
    return xp.fft.ifftn(a_fft * b_fft, axes=(0,))


def hilbert_transform(a: NDArray, energies: NDArray, eta=1e-8) -> NDArray:
    """Computes the Hilbert transform of the array a.

    Assumes that the first axis corresponds to the energy axis.

    Parameters
    ----------
    a : NDArray
        The array to transform.
    energies : NDArray
        The energy values corresponding to the first axis of a.
    eta : float, optional
        For the principle part. Small part to avoid singularity, by
        default 1e-8.

    Returns
    -------
    NDArray
         The Hilbert transform of a.

    """
    # Add a small imaginary part to avoid singularity.
    energy_differences = (energies - energies[0]).reshape(-1, 1)
    ne = energies.size
    # eta for removing the singularity. See Cauchy principal value.
    b = (
        fft_convolve(a, 1 / (energy_differences + eta))[:ne]
        + fft_convolve(a, 1 / (-energy_differences[::-1] - eta))[ne - 1 :]
    )
    # Not sure about the prefactor. Currently gives the same value as the old code.
    return b / (2 * xp.pi) * (energies[1] - energies[0])


class SigmaCoulombScreening(ScatteringSelfEnergy):
    """Computes the scattering self-energy from the Coulomb screening.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        electron_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ):
        """Initializes the scattering self-energy."""
        self.energies = electron_energies
        self.num_energies = self.energies.size
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])

        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        # TODO: This is pretty wasteful memory-wise.
        self.w_lesser_reduced = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=self.energies.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )
        self.w_greater_reduced = compute_config.dsbsparse_type.zeros_like(
            self.w_lesser_reduced
        )

    def compute(
        self,
        g_lesser: DSBSparse,
        g_greater: DSBSparse,
        w_lesser: DSBSparse,
        w_greater: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        g_greater : DSBSparse
            The greater Green's function.
        w_lesser : DSBSparse
            The lesser screened Coulomb interaction.
        w_greater : DSBSparse
            The greater screened Coulomb interaction.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_lesser, sigma_greater, sigma_retarded.

        """
        # If the w_lesser and w_greater don't have the same sparsity
        # pattern as g_lesser and g_greater, we have to reduce them to
        # the same sparsity pattern.
        if w_lesser.nnz != self.w_lesser_reduced.nnz:
            num_blocks = len(self.w_lesser_reduced.block_sizes)
            original_block_sizes = w_lesser.block_sizes
            w_lesser.block_sizes = self.w_lesser_reduced.block_sizes
            w_greater.block_sizes = self.w_greater_reduced.block_sizes
            for i in range(num_blocks):
                for j in range(max(0, i - 1), min(num_blocks, i + 2)):
                    self.w_lesser_reduced.blocks[i, j] = w_lesser.blocks[i, j]
                    self.w_greater_reduced.blocks[i, j] = w_greater.blocks[i, j]

            w_lesser.block_sizes = original_block_sizes
            w_greater.block_sizes = original_block_sizes

            # self.w_lesser_reduced.data = w_lesser[*self.w_lesser_reduced.spy()]
            # self.w_greater_reduced.data = w_greater[*self.w_greater_reduced.spy()]
        else:
            self.w_lesser_reduced.data = w_lesser.data
            self.w_greater_reduced.data = w_greater.data

        sigma_lesser, sigma_greater, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (
            g_lesser,
            g_greater,
            self.w_lesser_reduced,
            self.w_greater_reduced,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "nnz" else None

        # Compute the full self-energy using the convolution theorem.
        # Second term are corrections for negative frequencies that
        # where cut off by the polarization calculation.
        sigma_lesser.data += self.prefactor * (
            fft_convolve(g_lesser.data, self.w_lesser_reduced.data)[: self.num_energies]
            - fft_correlate(g_lesser.data, self.w_greater_reduced.data.conj())[
                self.num_energies - 1 :
            ]
        )
        sigma_greater.data += self.prefactor * (
            fft_convolve(g_greater.data, self.w_greater_reduced.data)[
                : self.num_energies
            ]
            - fft_correlate(g_greater.data, self.w_lesser_reduced.data.conj())[
                self.num_energies - 1 :
            ]
        )

        # Compute retarded self-energy with a Hilbert transform.
        sigma_antihermitian = 1j * xp.imag(sigma_greater.data - sigma_lesser.data)
        sigma_hermitian = hilbert_transform(sigma_antihermitian, self.energies)
        sigma_retarded.data += 1j * sigma_hermitian + sigma_antihermitian / 2

        # Transpose the matrices to stack distribution.
        for m in (
            g_lesser,
            g_greater,
            self.w_lesser_reduced,
            self.w_greater_reduced,
            sigma_lesser,
            sigma_greater,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "stack" else None


class SigmaCoulombScreening_X(ScatteringSelfEnergy):
    """Computes the scattering self-energy from the Coulomb screening.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        energies: NDArray,
        energies_x: NDArray,
        number_of_energies_lesser: int,
        number_of_overlap_energies: int,
        sparsity_pattern: sparse.coo_matrix,
    ):
        """Initializes the scattering self-energy."""
        self.energies = energies
        self.prefactor = 1j / (2 * xp.pi) * (energies_x[1] - energies_x[0])
        self.nel = number_of_energies_lesser
        self.neg = len(energies_x) - number_of_energies_lesser
        self.noe = number_of_overlap_energies
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")
        # TODO: This is pretty wasteful memory-wise.
        self.w_lesser_reduced = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=energies_x.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )

    def compute(
        self,
        g_x: DSBSparse,
        w_lesser: DSBSparse,
        out: tuple[DSBSparse, ...],
    ) -> None:
        """Computes the GW self-energy.

        Parameters
        ----------
        g_x : DSBSparse
            The combines lesser/greater Green's function.
        w_x : DSBSparse
            The greater screened Coulomb interaction.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_x, sigma_retarded.

        """
        nel = self.nel
        neg = self.neg
        noe = self.noe
        # If the w_x don't have the same sparsity
        # pattern as g_x, we have to reduce them to
        # the same sparsity pattern.
        if w_lesser.nnz != self.w_lesser_reduced.nnz:
            num_blocks = len(self.w_lesser_reduced.block_sizes)
            original_block_sizes = w_lesser.block_sizes
            w_lesser.block_sizes = self.w_lesser_reduced.block_sizes
            for i in range(num_blocks):
                for j in range(max(0, i - 1), min(num_blocks, i + 2)):
                    self.w_lesser_reduced.blocks[i, j] = w_lesser.blocks[i, j]

            w_lesser.block_sizes = original_block_sizes
        else:
            self.w_lesser_reduced.data = w_lesser.data

        sigma_x, sigma_retarded = out
        # Transpose the matrices to nnz distribution.
        for m in (
            g_x,
            self.w_lesser_reduced,
            sigma_x,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "nnz" else None

        # Compute the full self-energy using the convolution theorem.
        sigma_x._data[:nel] += self.prefactor * (
            fft_convolve(g_x.data[:nel], self.w_lesser_reduced.data)[
                -nel - noe + 1 : -noe + 1
            ]
        )
        sigma_x._data[nel:] += self.prefactor * (
            -fft_correlate(g_x.data[nel:], self.w_lesser_reduced.data.conj())[
                noe - 1 : neg + noe - 1
            ]
        )

        # Compute retarded self-energy with a Hilbert transform.
        sigma_antihermitian = xp.zeros(
            (len(self.energies), sigma_retarded.data.shape[-1]), dtype=xp.complex128
        )
        sigma_antihermitian[nel - noe :] = 1j * xp.imag(sigma_x.data[nel:])
        sigma_antihermitian[:nel] -= 1j * xp.imag(sigma_x.data[:nel])
        sigma_hermitian = hilbert_transform(sigma_antihermitian, self.energies)
        sigma_retarded._data[:nel] += (
            1j * sigma_hermitian[:nel] + sigma_antihermitian[:nel] / 2
        )
        sigma_retarded._data[nel:] += (
            1j * sigma_hermitian[nel - noe :] + sigma_antihermitian[nel - noe :] / 2
        )

        # Transpose the matrices to stack distribution.
        for m in (
            g_x,
            self.w_lesser_reduced,
            sigma_x,
            sigma_retarded,
        ):
            m.dtranspose() if m.distribution_state != "stack" else None


class SigmaFock(ScatteringSelfEnergy):
    """Computes the bare Fock self-energy.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        electron_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ):
        """Initializes the bare Fock self-energy."""
        self.energies = electron_energies
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        coulomb_matrix_sparray = distributed_load(
            quatrex_config.input_dir / "coulomb_matrix.npz"
        ).astype(xp.complex128)
        # Load block sizes for the coulomb matrix.
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")

        # Create the DSBSparse object.
        # TODO: This is pretty wasteful memory-wise.
        self.coulomb_matrix = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=self.energies.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )
        self.coulomb_matrix.data = 0.0
        self.coulomb_matrix += coulomb_matrix_sparray

    def compute(self, g_lesser: DSBSparse, out: tuple[DSBSparse, ...]) -> None:
        """Computes the Fock self-energy.

        Parameters
        ----------
        g_lesser : DSBSparse
            The lesser Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.

        """
        # TODO: Check again if we really need to transpose the matrices
        # here.
        (sigma_retarded,) = out
        for m in (g_lesser, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "nnz" else None
        # Compute the electron density by summing over energies.
        gl_density = self.prefactor * g_lesser.data.sum(axis=0)
        sigma_retarded.data += xp.real(gl_density * self.coulomb_matrix.data)
        for m in (g_lesser, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "stack" else None


class SigmaFock_X(ScatteringSelfEnergy):
    """Computes the bare Fock self-energy.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The Quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.
    electron_energies : NDArray
        The energies for the electron system.

    """

    def __init__(
        self,
        quatrex_config: QuatrexConfig,
        compute_config: ComputeConfig,
        number_of_energies_lesser: int,
        electron_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix,
    ):
        """Initializes the bare Fock self-energy."""
        self.nel = number_of_energies_lesser
        self.energies = electron_energies
        self.prefactor = 1j / (2 * xp.pi) * (self.energies[1] - self.energies[0])
        coulomb_matrix_sparray = distributed_load(
            quatrex_config.input_dir / "coulomb_matrix.npz"
        ).astype(xp.complex128)
        # Load block sizes for the coulomb matrix.
        block_sizes = distributed_load(quatrex_config.input_dir / "block_sizes.npy")

        # Create the DSBSparse object.
        # TODO: This is pretty wasteful memory-wise.
        self.coulomb_matrix = compute_config.dsbsparse_type.from_sparray(
            sparsity_pattern.astype(xp.complex128),
            block_sizes=block_sizes,
            global_stack_shape=self.energies.shape,
            densify_blocks=[(0, 0), (-1, -1)],  # Densify for OBC.
        )
        self.coulomb_matrix.data = 0.0
        self.coulomb_matrix += coulomb_matrix_sparray

    def compute(self, g_x: DSBSparse, out: tuple[DSBSparse, ...]) -> None:
        """Computes the Fock self-energy.

        Parameters
        ----------
        g_x : DSBSparse
            The combined lesser/greater Green's function.
        out : tuple[DSBSparse, ...]
            The output matrices for the self-energy. The order is
            sigma_retarded.

        """
        # TODO: Check again if we really need to transpose the matrices
        # here.
        nel = self.nel
        (sigma_retarded,) = out
        for m in (g_x, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "nnz" else None
        # Compute the electron density by summing over energies.
        gl_density = self.prefactor * g_x.data[:nel].sum(axis=0)
        sigma_fock = xp.real(gl_density * self.coulomb_matrix.data[0])
        sigma_retarded.data += sigma_fock

        for m in (g_x, sigma_retarded, self.coulomb_matrix):
            m.dtranspose() if m.distribution_state != "stack" else None
