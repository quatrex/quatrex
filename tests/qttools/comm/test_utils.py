# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest

from qttools import xp
from qttools.comm import comm, distributed_max


@pytest.mark.mpi(min_size=3)
def test_distributed_max():
    """Test the distributed_max function."""

    for test_comm in [comm.block, comm.stack, comm.global_]:
        a = xp.ones((test_comm.size,), dtype=xp.float32) * test_comm.rank
        a_max = distributed_max(a, test_comm)

        reference_max = xp.array([test_comm.size - 1], dtype=xp.float32)

        assert xp.allclose(
            a_max, reference_max
        ), f"a: {a}, a_max: {a_max}, reference_max: {reference_max}"
