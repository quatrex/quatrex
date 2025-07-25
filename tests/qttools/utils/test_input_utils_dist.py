# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import numpy as np
import pytest

from qttools import xp
from qttools.utils.input_utils import create_hamiltonian
from qttools.utils.mpi_utils import get_section_sizes


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

        start_idx = section_offsets[rank]
        end_idx = section_offsets[rank + 1]

        block_start = start_idx
        block_end = end_idx

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
