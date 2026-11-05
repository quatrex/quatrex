# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import xp
from qttools.comm import comm
from qttools.comm.comm import GPU_AWARE_MPI, _backends

BACKEND = [pytest.param(backend, id=backend) for backend in _backends]

BLOCK_COMM_SIZES = [
    pytest.param(1, id="1"),
    pytest.param(2, id="2"),
    pytest.param(3, id="3"),
    pytest.param(4, id="4"),
]


@pytest.fixture(params=BACKEND)
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=BLOCK_COMM_SIZES)
def block_comm_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(autouse=True)
def configure(
    request,
    backend: str,
    block_comm_size: int,
):
    # To specifically test the `configure` function, we can use the
    # `no_autoconf` marker to skip this fixture.
    if "no_autoconf" in request.keywords:
        yield
        return

    if (block_comm_size > global_comm.size) | (global_comm.size % block_comm_size != 0):
        pytest.skip("Config not valid")

    if xp.__name__ == "numpy" and backend in ["nccl", "host_mpi"]:
        pytest.skip("Config not valid")

    if xp.__name__ == "cupy":
        from cupy.cuda import nccl

        if not nccl.available and backend == "nccl":
            pytest.skip("Config not valid")
        if not GPU_AWARE_MPI and backend == "device_mpi":
            pytest.skip("Config not valid")

    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_backend=backend,
        stack_comm_backend=backend,
        global_comm_backend=backend,
        override=True,
    )
    yield
