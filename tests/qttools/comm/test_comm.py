import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import xp
from qttools.comm import comm
from qttools.comm.comm import GPU_AWARE_MPI, _default_config, pad_buffer

data_size = 20


@pytest.fixture(scope="function")
def reset_comm():
    # Setup: set all members of the singleton comm to None
    for attr in comm.__dict__:
        setattr(comm, attr, None)
    comm._is_configured = False

    # Yield control to the test function
    yield


def _configure(
    backend_type: str,
    block_comm_size: int,
) -> bool:

    # set config to all the same backend type
    config = _default_config.copy()
    config = {key: backend_type for key in config.keys()}

    if (block_comm_size > global_comm.size) | (global_comm.size % block_comm_size != 0):
        with pytest.raises(ValueError):
            comm.configure(
                block_comm_size=block_comm_size,
                block_comm_config=config,
                stack_comm_config=config,
            )
        return False

    if xp.__name__ == "numpy" and backend_type in ["nccl", "host_mpi"]:
        with pytest.raises(ValueError):
            comm.configure(
                block_comm_size=block_comm_size,
                block_comm_config=config,
                stack_comm_config=config,
            )
        return False

    if xp.__name__ == "cupy":
        from cupy.cuda import nccl

        if not nccl.available and backend_type == "nccl":
            with pytest.raises(RuntimeError):
                comm.configure(
                    block_comm_size=block_comm_size,
                    block_comm_config=config,
                    stack_comm_config=config,
                )
            return False

        if not GPU_AWARE_MPI and backend_type == "device_mpi":
            with pytest.raises(ValueError):
                comm.configure(
                    block_comm_size=block_comm_size,
                    block_comm_config=config,
                    stack_comm_config=config,
                )
            return False

    comm.configure(
        block_comm_size=block_comm_size,
        block_comm_config=config,
        stack_comm_config=config,
    )
    return True


@pytest.mark.mpi(min_size=3)
def test_configure(
    reset_comm,
    backend_type: str,
    block_comm_size: int,
) -> bool:
    """Test the configure function of the comm singleton."""

    _configure(
        backend_type=backend_type,
        block_comm_size=block_comm_size,
    )
    return


@pytest.mark.mpi(min_size=3)
def test_all_to_all(
    reset_comm,
    backend_type: str,
    block_comm_size: int,
):
    """Test the all_to_all function of the comm singleton."""

    if not _configure(
        backend_type=backend_type,
        block_comm_size=block_comm_size,
    ):
        pytest.skip("Config not valid")

    for test_comm in [comm.block, comm.stack]:

        # random sendbuf
        sendbuf = xp.ones((test_comm.size,), dtype=xp.float32) * test_comm.rank

        recvbuf = xp.empty_like(sendbuf)

        test_comm.all_to_all(sendbuf, recvbuf)

        assert xp.allclose(
            xp.arange(test_comm.size, dtype=xp.float32),
            recvbuf,
        ), f"sendbuf: {sendbuf}, recvbuf: {recvbuf}"


@pytest.mark.mpi(min_size=3)
def test_all_gather(
    reset_comm,
    backend_type: str,
    block_comm_size: int,
):
    """Test the all_gather function of the comm singleton."""

    if not _configure(
        backend_type=backend_type,
        block_comm_size=block_comm_size,
    ):
        pytest.skip("Config not valid")

    for test_comm in [comm.block, comm.stack]:

        # random sendbuf
        sendbuf = xp.ones((data_size,), dtype=xp.float32) * test_comm.rank

        recvbuf = xp.empty((data_size * test_comm.size,), dtype=xp.float32)

        test_comm.all_gather(sendbuf, recvbuf)

        for i in range(test_comm.size):
            assert xp.allclose(
                xp.ones((data_size,), dtype=xp.float32) * i,
                recvbuf[i * data_size : (i + 1) * data_size],
            ), f"sendbuf: {sendbuf}, recvbuf: {recvbuf[i * data_size : (i + 1) * data_size]}"


@pytest.mark.mpi(min_size=3)
def test_all_reduce(
    reset_comm,
    backend_type: str,
    block_comm_size: int,
):
    """Test the all_reduce function of the comm singleton."""

    if not _configure(
        backend_type=backend_type,
        block_comm_size=block_comm_size,
    ):
        pytest.skip("Config not valid")

    for test_comm in [comm.block, comm.stack]:

        # random sendbuf
        sendbuf = xp.ones((data_size,), dtype=xp.float32) * test_comm.rank

        recvbuf = xp.empty_like(sendbuf)

        test_comm.all_reduce(sendbuf, recvbuf)

        assert xp.allclose(
            xp.ones((data_size,), dtype=xp.float32)
            * test_comm.size
            * (test_comm.size - 1)
            / 2,
            recvbuf,
        ), f"sendbuf: {sendbuf}, recvbuf: {recvbuf}"


@pytest.mark.mpi(min_size=3)
def test_bcast(
    reset_comm,
    backend_type: str,
    block_comm_size: int,
):
    """Test the bcast function of the comm singleton."""

    if not _configure(
        backend_type=backend_type,
        block_comm_size=block_comm_size,
    ):
        pytest.skip("Config not valid")

    for test_comm in [comm.block, comm.stack]:

        # random sendbuf
        sendbuf = xp.ones((data_size,), dtype=xp.float32) * xp.pi

        test_comm.bcast(sendbuf)

        assert xp.allclose(
            xp.ones((data_size,), dtype=xp.float32) * xp.pi,
            sendbuf,
        ), f"sendbuf: {sendbuf}"


@pytest.mark.mpi(min_size=3)
def test_pad_buffer(
    reset_comm,
    backend_type: str,
    block_comm_size: int,
):
    """Test the pad_buffer function."""

    if not _configure(
        backend_type=backend_type,
        block_comm_size=block_comm_size,
    ):
        pytest.skip("Config not valid")

    for test_comm in [comm.block, comm.stack]:

        # random sendbuf
        sendbuf = (
            xp.ones((data_size - test_comm.rank,), dtype=xp.float32) * test_comm.rank
        )

        padded_sendbuf = pad_buffer(
            sendbuf,
            global_size=data_size * test_comm.size,
            comm_size=test_comm.size,
            axis=0,
        )

        recvbuf = xp.empty((data_size * test_comm.size,), dtype=xp.float32)

        test_comm.all_gather(padded_sendbuf, recvbuf)

        # mask the recvbuf to only include the padded values
        recvbuf = recvbuf.reshape((test_comm.size, data_size))

        for i in range(test_comm.size):
            assert xp.allclose(
                xp.ones((data_size - i,), dtype=xp.float32) * i,
                recvbuf[i, : data_size - i],
            )
            assert xp.allclose(
                xp.zeros((i,), dtype=xp.float32),
                recvbuf[i, data_size - i :],
            )


@pytest.mark.mpi(min_size=3)
def test_all_gather_v(
    reset_comm,
    backend_type: str,
    block_comm_size: int,
):
    """Test the all_gather_v function."""

    if not _configure(
        backend_type=backend_type,
        block_comm_size=block_comm_size,
    ):
        pytest.skip("Config not valid")

    for test_comm in [comm.block, comm.stack]:

        # random sendbuf
        sendbuf = (
            xp.ones(
                (
                    data_size,
                    data_size - test_comm.rank,
                    data_size,
                ),
                dtype=xp.float32,
            )
            * test_comm.rank
        )

        recvbuf = test_comm.all_gather_v(sendbuf, axis=1)

        counts = np.zeros(test_comm.size, dtype=xp.int32)
        for i in range(test_comm.size):
            counts[i] = data_size - i
        displacements = np.cumsum(counts) - counts

        for i in range(test_comm.size):
            assert xp.allclose(
                xp.ones(
                    (
                        data_size,
                        data_size - i,
                        data_size,
                    ),
                    dtype=xp.float32,
                )
                * i,
                recvbuf[:, displacements[i] : displacements[i] + counts[i], :],
            )
