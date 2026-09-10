# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import xp
from qttools.comm import comm, distributed_max


@pytest.mark.mpi(min_size=3)
def test_distributed_max_constant():
    """Test the distributed_max function on an array of equal entries."""

    for test_comm in [comm.block, comm.stack, comm.global_]:
        a = xp.ones((test_comm.size,), dtype=xp.float32) * test_comm.rank
        a_max = distributed_max(a, test_comm)

        reference_max = xp.array([test_comm.size - 1], dtype=xp.float32)

        assert xp.allclose(
            a_max, reference_max
        ), f"a: {a}, a_max: {a_max}, reference_max: {reference_max}"


@pytest.mark.mpi(min_size=3)
def test_distributed_max_random():
    """Test the distributed_max function on a random array."""

    for test_comm in [comm.block, comm.stack, comm.global_]:
        xp.random.seed(0)
        a = xp.random.rand(test_comm.size)
        a_max = distributed_max(a, test_comm)

        found_max = False
        for a_entry in a.flatten():
            assert a_entry <= a_max, "`distributed_max` did not return the maximum"
            if a_entry == a_max:
                found_max = True

        assert (
            found_max
        ), f"The maximum returned by `distributed_max` does not occur in the array: a_max = {a_max}, a = {a}"
