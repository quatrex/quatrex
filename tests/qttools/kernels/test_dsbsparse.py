# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import xp
from qttools.kernels import dsbsparse_kernels
from qttools.utils.mpi_utils import get_section_sizes


@pytest.mark.usefixtures("nnz", "comm_size", "num_inds")
def test_find_ranks(nnz: int, comm_size: int, num_inds: int):
    """Tests that the ranks are computed correctly."""
    section_sizes, __ = get_section_sizes(nnz, comm_size, strategy="greedy")
    section_offsets = xp.hstack(([0], xp.cumsum(xp.array(section_sizes))))

    inds = xp.random.randint(0, nnz, num_inds)

    reference_ranks = xp.sum(section_offsets <= inds[:, xp.newaxis], axis=-1) - 1

    ranks = dsbsparse_kernels.find_ranks(section_offsets, inds)
    assert xp.all(ranks == reference_ranks)
