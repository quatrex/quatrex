# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import xp
from qttools.comm import comm, distributed_max


@pytest.mark.mpi(min_size=3)
def test_distributed_max(
    backend_type: str,
    block_comm_size: int,
):
    """Test the distributed_max function."""

    for test_comm in [comm.block]:
        a = xp.ones((test_comm.size,), dtype=xp.float32) * test_comm.rank
        a_max = distributed_max(a)

        found_max = False
        for a_entry in a.flatten():
            assert a_entry <= a_max, "`distributed_max` did not return the maximum"
            if a_entry == a_max:
                found_max = True

        assert (
            found_max
        ), f"The maximum returned by `distributed_max` does not occur in the array: a_max = {a_max}, a = {a}"
