# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures.csx import CSX
from qttools.datastructures.csx_routines import allgather_csx
from qttools.datastructures.dsdbsparse import symmetry_ops


@pytest.fixture(autouse=True, scope="module", params=[6, 3, 1])
def configure_comm(request):
    """Setup any state specific to the execution of the given module."""
    block_comm_size = request.param

    if global_comm.size < block_comm_size:
        pytest.skip(
            f"Skipping test for block comm size {block_comm_size} with global comm size {global_comm.size}."
        )

    # Configure the comm singleton with the parameterized block_comm_size
    comm.configure(
        block_comm_size=block_comm_size,
        override=True,
    )


def _create_coo(
    size: int,
    symmetry: str | None = None,
) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""

    rng = xp.random.default_rng(seed=42)
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(xp.complex128)
    coo.setdiag(rng.uniform(size=size) + 1j * rng.uniform(size=size))
    coo.data += 1j * rng.uniform(size=coo.nnz)

    # make the sparsity pattern symmetric
    coo_ = coo.copy()
    coo_.data[:] = rng.uniform(size=coo_.nnz) + 1j * rng.uniform(size=coo_.nnz)
    coo = coo + coo_.T
    coo = coo.tocoo()

    if symmetry is not None:
        coo_t = coo.copy()
        coo_t.data[:] = symmetry_ops[symmetry](coo_t.data)
        coo = coo + coo_t.T
        # Keep only the upper triangular part
        coo = sparse.triu(coo, format="coo")
        return coo

    return coo


def _create_coo_csx(
    size: int,
    local_stack_shape: tuple,
    symmetry: str | None = None,
) -> tuple[sparse.coo_matrix, CSX]:
    """Returns a random complex sparse array
    and a CSX matrix with the same sparsity pattern.
    """
    coo = _create_coo(size, symmetry=symmetry)

    csx = CSX.from_sparray(
        sparray=coo,
        local_stack_shape=local_stack_shape,
        symmetry=symmetry,
    )
    csx.data = coo.data
    return coo, csx


class TestAllgather:
    """Tests for the allgather methods on CSX."""

    @pytest.mark.parametrize("axis", [0, 1])
    def test_allgather_csx(
        self,
        size: int,
        local_stack_shape: tuple,
        symmetry: str | None,
        axis: int,
    ):
        """Tests the allgather csx method."""

        coo, csx = _create_coo_csx(
            size=size,
            local_stack_shape=local_stack_shape,
            symmetry=symmetry,
        )
        reference = coo.toarray()
        reference = comm.block.all_gather_v(reference, axis=axis)

        offsets = [0] + list(np.cumsum([size] * comm.block.size))

        csx_allgathered = allgather_csx(csx, comm=comm.block, axis=axis)
        csx_allgathered = csx_allgathered.toarray()

        for i in range(comm.block.size):
            if axis == 0:
                test = csx_allgathered[..., offsets[i] : offsets[i + 1], :]
                ref = reference[offsets[i] : offsets[i + 1], :]
            else:
                test = csx_allgathered[..., :, offsets[i] : offsets[i + 1]]
                ref = reference[:, offsets[i] : offsets[i + 1]]

            ref = xp.broadcast_to(ref, local_stack_shape + (size, size))
            assert xp.allclose(test, ref)


@pytest.mark.mpi(min_size=2)
class TestAllgatherDist(TestAllgather):
    """Tests all tests of TestAllgather in distributed setting."""

    pass
