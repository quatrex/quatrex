# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures.dcsx import DCSX
from qttools.datastructures.dsdbsparse import symmetry_ops
from qttools.utils.mpi_utils import get_section_sizes


@pytest.fixture(autouse=True, scope="module", params=[3, 1])
def configure_comm(request):
    """Setup any state specific to the execution of the given module."""
    block_comm_size = request.param

    # Default configuration setup based on the xp module
    if xp.__name__ == "cupy":
        _default_config = {
            "all_to_all": "host_mpi",
            "all_gather": "host_mpi",
            "all_reduce": "host_mpi",
            "bcast": "host_mpi",
        }
    elif xp.__name__ == "numpy":
        _default_config = {
            "all_to_all": "device_mpi",
            "all_gather": "device_mpi",
            "all_reduce": "device_mpi",
            "bcast": "device_mpi",
        }

    if global_comm.size < block_comm_size:
        pytest.skip(
            f"Skipping test for block comm size {block_comm_size} with global comm size {global_comm.size}."
        )

    # Configure the comm singleton with the parameterized block_comm_size
    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


def _create_coo(
    size: int,
    symmetry: str | None = None,
) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""

    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(xp.complex128)
    coo.setdiag(rng.uniform(size=size) + 1j * rng.uniform(size=size))
    coo.data += 1j * rng.uniform(size=coo.nnz)

    if symmetry is not None:
        coo_t = coo.copy()
        coo_t.data[:] = symmetry_ops[symmetry](coo_t.data)
        coo = coo + coo_t.T
        return coo

    return coo


def _create_coo_dcsx(
    size: int,
    local_stack_shape: tuple,
    symmetry: str | None = None,
) -> tuple[sparse.coo_matrix, sparse.coo_matrix, DCSX]:
    """Returns a random complex sparse array
    and a DCSX matrix with the same sparsity pattern.
    """
    coo = _create_coo(size, symmetry=symmetry) if global_comm.rank == 0 else None
    coo = global_comm.bcast(coo, root=0)

    section_sizes, __ = get_section_sizes(size, comm.block.size)
    offsets = np.cumsum([0] + section_sizes)

    local_sparray = coo.tocsr()[
        offsets[comm.block.rank] : offsets[comm.block.rank + 1], :
    ]

    dcsx = DCSX.from_sparray(
        sparray=local_sparray,
        local_stack_shape=local_stack_shape,
        symmetry=symmetry,
    )
    return local_sparray, coo, dcsx


class TestCreation:
    """Tests the creation methods of DCSX."""

    def test_from_sparray(
        self,
        size: int,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests the creation of DCSX matrices from sparse arrays."""

        __, coo, dcsx = _create_coo_dcsx(
            size=size,
            local_stack_shape=global_stack_shape,
            symmetry=symmetry,
        )
        dense_dcsx = dcsx.to_dense()
        assert xp.array_equiv(coo.toarray(), dense_dcsx)
        if symmetry is not None:
            assert xp.array_equiv(
                dense_dcsx, symmetry_ops[symmetry](dense_dcsx).swapaxes(-1, -2)
            )


@pytest.mark.mpi(min_size=2)
class TestCreationDist(TestCreation):
    """Tests all tests of TestCreation in distributed setting."""

    pass
