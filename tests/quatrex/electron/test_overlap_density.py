# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.
"""Tests the density in non-orthogonal bases."""

import numpy as np
import pytest
from scipy import sparse

from qttools import xp
from qttools.comm import comm
from qttools.comm.comm import _default_config
from qttools.datastructures import DSDBCSR
from qttools.greens_function_solver import RGF
from qttools.greens_function_solver.solver import OBCBlocks
from qttools.utils.gpu_utils import get_host
from quatrex.electron.solver import OverlapSpillOver, overlap_density

BLOCK_SIZE = 4
ETA = 1e-9
ENERGIES = np.linspace(-1.0, 1.0, 5)
# Different occupancies, i.e., the device is out of equilibrium.
OCCUPANCIES = {"left": np.linspace(1.0, 0.6, 5), "right": np.linspace(0.5, 0.1, 5)}


def _block_tridiagonal(diag, upper, num_blocks):
    """Creates a Hermitian block-tridiagonal matrix from its blocks."""
    ones = np.ones(num_blocks - 1)
    return (
        np.kron(np.eye(num_blocks), diag)
        + np.kron(np.diag(ones, 1), upper)
        + np.kron(np.diag(ones, -1), upper.conj().T)
    )


def _surface_greens_function(m_10, m_00, m_01):
    """Computes the retarded surface Green's function by decimation."""
    eps_s, eps = m_00.copy(), m_00.copy()
    alpha, beta = m_10.copy(), m_01.copy()
    for __ in range(60):
        g = np.linalg.inv(eps)
        eps_s = eps_s - alpha @ g @ beta
        eps = eps - alpha @ g @ beta - beta @ g @ alpha
        alpha, beta = alpha @ g @ alpha, beta @ g @ beta
    return np.linalg.inv(eps_s)


def _to_dsdbsparse(dense, pattern, num_energies):
    """Converts dense (stacked) matrices to a DSDBSparse."""
    num_blocks = dense.shape[-1] // BLOCK_SIZE
    out = DSDBCSR.from_sparray(
        sparse.coo_matrix(pattern.astype(np.complex128)),
        block_sizes=np.full(num_blocks, BLOCK_SIZE),
        global_stack_shape=(num_energies,),
    )
    for i in range(num_blocks):
        for j in range(max(0, i - 1), min(num_blocks, i + 2)):
            out.blocks[i, j] = xp.asarray(
                dense[
                    ...,
                    i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                    j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
                ]
            )
    return out


def _densities(num_blocks, h, s, device_potential, g_is_truncated):
    """Computes the lesser and greater densities of a device."""
    num_energies = len(ENERGIES)
    n = num_blocks * BLOCK_SIZE

    s_dense = _block_tridiagonal(*s, num_blocks)
    h_dense = _block_tridiagonal(*h, num_blocks)
    h_dense = h_dense + 0.5 * s_dense * (
        device_potential[:, None] + device_potential[None, :]
    )
    a = (ENERGIES + 1j * ETA)[:, None, None] * s_dense - h_dense

    obc_blocks = OBCBlocks(num_blocks)
    spill_overs = [[] for __ in range(num_blocks)]
    for contact, ind in (("left", 0), ("right", num_blocks - 1)):
        h_00, h_01 = h
        s_00, s_01 = s
        # For the left contact, the first lead block (rows) couples to
        # the device (columns) with the superdiagonal block.
        if contact == "right":
            h_01, s_01 = h_01.conj().T, s_01.conj().T

        blocks = {"retarded": [], "lesser": [], "greater": []}
        spill = {"retarded": [], "advanced": []}
        for energy, occupancy in zip(ENERGIES, OCCUPANCIES[contact], strict=True):
            z = energy + 1j * ETA
            m_00 = z * s_00 - h_00
            m_01 = z * s_01 - h_01
            m_10 = z * s_01.conj().T - h_01.conj().T
            g = _surface_greens_function(m_10, m_00, m_01)
            sigma = m_10 @ g @ m_01
            gamma = 1j * (sigma - sigma.conj().T)
            blocks["retarded"].append(sigma)
            blocks["lesser"].append(1j * occupancy * gamma)
            blocks["greater"].append(1j * (occupancy - 1) * gamma)
            spill["retarded"].append(m_10 @ g @ s_01)
            spill["advanced"].append(m_01.conj().T @ g.conj().T @ s_01)

        obc_blocks.retarded[ind] = xp.asarray(blocks["retarded"])
        obc_blocks.lesser[ind] = xp.asarray(blocks["lesser"])
        obc_blocks.greater[ind] = xp.asarray(blocks["greater"])
        spill_overs[ind].append(
            OverlapSpillOver(
                retarded=xp.asarray(spill["retarded"]),
                advanced=xp.asarray(spill["advanced"]),
                occupancies=xp.asarray(OCCUPANCIES[contact])[:, None],
            )
        )

    tridiagonal = (
        _block_tridiagonal(
            np.ones((BLOCK_SIZE,) * 2), np.ones((BLOCK_SIZE,) * 2), num_blocks
        )
        > 0
    )
    g_pattern = np.eye(n, dtype=bool) if g_is_truncated else tridiagonal

    a_ = _to_dsdbsparse(a, tridiagonal, num_energies)
    overlap = _to_dsdbsparse(s_dense[None], tridiagonal, 1)
    # Ballistic, i.e., without scattering self-energies.
    zeros = np.zeros_like(a)
    sigma_lesser = _to_dsdbsparse(zeros, tridiagonal, num_energies)
    sigma_greater = _to_dsdbsparse(zeros, tridiagonal, num_energies)
    out = tuple(_to_dsdbsparse(zeros, g_pattern, num_energies) for __ in range(3))

    lesser = xp.zeros((num_energies, n))
    greater = xp.zeros((num_energies, n))
    RGF(max_batch_size=2).selected_solve(
        a=a_,
        sigma_lesser=sigma_lesser,
        sigma_greater=sigma_greater,
        obc_blocks=obc_blocks,
        out=out,
        return_retarded=True,
        callbacks=[overlap_density(lesser, greater, overlap, spill_overs)],
    )

    return get_host(lesser), get_host(greater)


@pytest.fixture(autouse=True, scope="module")
def configure_comm():
    """Configures the communicator for a single rank."""
    comm.configure(
        block_comm_size=1,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        global_comm_config=_default_config,
        override=True,
    )


@pytest.fixture
def blocks():
    """Hamiltonian and overlap blocks of a random periodic lead."""
    rng = np.random.default_rng(0)

    def random(scale):
        shape = (BLOCK_SIZE, BLOCK_SIZE)
        return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))

    h_00, s_00 = random(0.5), random(0.05)
    h = (h_00 + h_00.conj().T, random(0.5))
    s = (np.eye(BLOCK_SIZE) + s_00 + s_00.conj().T, random(0.05))
    return h, s


@pytest.fixture
def potential():
    """A potential in the two central blocks of a four-block device."""
    phi = np.zeros(4 * BLOCK_SIZE)
    phi[BLOCK_SIZE:-BLOCK_SIZE] = np.linspace(0.1, 0.4, 2 * BLOCK_SIZE)
    return phi


@pytest.mark.mpi_skip()
def test_truncated_greens_function(blocks, potential):
    """Tests that the density does not depend on the Green's function's
    sparsity pattern."""
    h, s = blocks
    reference = _densities(4, h, s, potential, g_is_truncated=False)
    truncated = _densities(4, h, s, potential, g_is_truncated=True)

    for x, ref in zip(truncated, reference, strict=True):
        assert np.abs(ref).max() > 1e-2
        assert np.allclose(x, ref, rtol=0, atol=1e-12)


@pytest.mark.mpi_skip()
def test_spill_over(blocks, potential):
    """Tests the lead spill-over against a device that explicitly
    includes the first block of both leads."""
    h, s = blocks
    b = BLOCK_SIZE
    small = _densities(4, h, s, potential, g_is_truncated=True)
    large = _densities(6, h, s, np.pad(potential, b), g_is_truncated=True)

    for x, ref in zip(small, large, strict=True):
        assert np.abs(ref).max() > 1e-2
        assert np.allclose(x, ref[:, b:-b], rtol=0, atol=1e-6 * np.abs(ref).max())
