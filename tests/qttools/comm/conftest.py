# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import xp
from qttools.comm import comm
from qttools.comm.comm import GPU_AWARE_MPI, _backends, _default_config

BACKEND_TYPE = [pytest.param(backend, id=backend) for backend in _backends]

BLOCK_COMM_SIZES = [
    pytest.param(1, id="1"),
    pytest.param(2, id="2"),
    pytest.param(4, id="4"),
]


@pytest.fixture(params=BACKEND_TYPE)
def backend_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=BLOCK_COMM_SIZES)
def block_comm_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(autouse=True)
def configure(
    backend_type: str,
    block_comm_size: int,
):
    # set config to all the same backend type
    config = _default_config.copy()
    config = {key: backend_type for key in config.keys()}

    if (block_comm_size > global_comm.size) | (global_comm.size % block_comm_size != 0):
        with pytest.raises(ValueError):
            comm.configure(
                block_comm_size=block_comm_size,
                block_comm_config=config,
                stack_comm_config=config,
                override=True,
            )
        pytest.skip("Config not valid")

    if xp.__name__ == "numpy" and backend_type in ["nccl", "host_mpi"]:
        with pytest.raises(ValueError):
            comm.configure(
                block_comm_size=block_comm_size,
                block_comm_config=config,
                stack_comm_config=config,
                override=True,
            )
        pytest.skip("Config not valid")

    if xp.__name__ == "cupy":
        from cupy.cuda import nccl

        if not nccl.available and backend_type == "nccl":
            with pytest.raises(RuntimeError):
                comm.configure(
                    block_comm_size=block_comm_size,
                    block_comm_config=config,
                    stack_comm_config=config,
                    override=True,
                )
        pytest.skip("Config not valid")

        if not GPU_AWARE_MPI and backend_type == "device_mpi":
            with pytest.raises(ValueError):
                comm.configure(
                    block_comm_size=block_comm_size,
                    block_comm_config=config,
                    stack_comm_config=config,
                    override=True,
                )
        pytest.skip("Config not valid")

    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_config=config,
        stack_comm_config=config,
        override=True,
    )
