# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest
from mpi4py import MPI

from qttools import xp
from qttools.comm import comm
from qttools.datastructures import DSDBCOO
from qttools.utils.input_utils import create_hamiltonian
from qttools.utils.mpi_utils import get_section_sizes


# TODO test for block distributed
@pytest.fixture(
    autouse=True,
    scope="module",
)
def configure_comm():
    """Setup any state specific to the execution of the given module."""
    block_comm_size = MPI.COMM_WORLD.Get_size()

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

    # Configure the comm singleton with the parameterized block_comm_size
    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


@pytest.mark.parametrize(
    "format, num_ranks",
    [
        ("coo", 1),
        ("csr", 1),
        ("coo", 2),
        ("csr", 2),
        ("coo", 3),
        ("csr", 3),
        ("coo", 4),
        ("csr", 4),
        ("coo", 5),
        ("csr", 5),
        ("coo", 6),
        ("csr", 6),
        ("coo", 7),
        ("csr", 7),
        ("coo", 8),
        ("csr", 8),
    ],
    ids=[
        "sparse-coo-1",
        "sparse-csr-1",
        "sparse-coo-2",
        "sparse-csr-2",
        "sparse-coo-3",
        "sparse-csr-3",
        "sparse-coo-4",
        "sparse-csr-4",
        "sparse-coo-5",
        "sparse-csr-5",
        "sparse-coo-6",
        "sparse-csr-6",
        "sparse-coo-7",
        "sparse-csr-7",
        "sparse-coo-8",
        "sparse-csr-8",
    ],
)
def test_create_hamiltonian_dist(
    format: str | None,
    num_ranks: int,
):
    hr = xp.ones((3, 3, 3, 5, 5))
    num_transport_cells = 10
    transport_dir = "x"
    transport_cell = (2, 1, 1)
    cutoff = 2
    coords = xp.zeros((5, 3))
    lat_vecs = xp.eye(3)

    global_ham, _ = create_hamiltonian(
        hr,
        num_transport_cells,
        transport_dir=transport_dir,
        transport_cell=transport_cell,
        block_start=0,
        block_end=num_transport_cells,
        return_sparse=True,
        cutoff=cutoff,
        coords=coords,
        lattice_vectors=lat_vecs,
        format=format,
    )

    global_ham_dense = global_ham.todense()
    global_ham = None

    section_sizes, __ = get_section_sizes(num_transport_cells, num_ranks)
    section_offsets = np.hstack(([0], np.cumsum(section_sizes)))

    tmp = None

    for rank in range(num_ranks):

        block_start = section_offsets[rank]
        block_end = section_offsets[rank + 1]

        local_ham, _ = create_hamiltonian(
            hr,
            num_transport_cells,
            transport_dir=transport_dir,
            transport_cell=transport_cell,
            block_start=block_start,
            block_end=block_end,
            return_sparse=True,
            cutoff=cutoff,
            coords=coords,
            lattice_vectors=lat_vecs,
            format=format,
        )

        local_ham_dense = local_ham.todense()
        local_ham = None

        if tmp is None:
            tmp = local_ham_dense
        else:
            print(
                f"Rank {rank} adding local_ham_dense with shape {local_ham_dense.shape}"
            )
            tmp += local_ham_dense
            local_ham_dense = None

    assert xp.allclose(global_ham_dense, tmp)


@pytest.mark.mpi
@pytest.mark.parametrize("format", ["coo", "csr"])
def test_create_hamiltonian_dsdbcoo(format: str):
    hr = xp.ones((3, 3, 3, 5, 5))
    num_transport_cells = 10
    transport_dir = "x"
    transport_cell = (2, 1, 1)
    cutoff = 2
    coords = xp.zeros((5, 3))
    lat_vecs = xp.eye(3)

    global_ham, bsizes = create_hamiltonian(
        hr,
        num_transport_cells,
        transport_dir=transport_dir,
        transport_cell=transport_cell,
        block_start=0,
        block_end=num_transport_cells,
        return_sparse=True,
        cutoff=cutoff,
        coords=coords,
        lattice_vectors=lat_vecs,
        format=format,
    )

    global_ham_dense = global_ham.todense()
    global_ham = None

    num_ranks = comm.block.size
    rank = comm.block.rank

    section_sizes, __ = get_section_sizes(num_transport_cells, num_ranks)
    section_offsets = np.hstack(([0], np.cumsum(section_sizes)))

    block_start = section_offsets[rank]
    block_end = section_offsets[rank + 1]

    local_ham, _ = create_hamiltonian(
        hr,
        num_transport_cells,
        transport_dir=transport_dir,
        transport_cell=transport_cell,
        block_start=block_start,
        block_end=block_end,
        return_sparse=True,
        cutoff=cutoff,
        coords=coords,
        lattice_vectors=lat_vecs,
        format=format,
    )

    dsdbcoo = DSDBCOO.from_sparray(
        local_ham,
        bsizes,
        (1,),
    )
    dense = dsdbcoo.to_dense()
    assert xp.allclose(global_ham_dense, dense)
