# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse as sparse
from qttools import xp
from qttools.comm import comm
from qttools.utils.hdf5_utils import save_hdf5_dict
from qttools.utils.mpi_utils import distributed_load, get_local_slice, get_section_sizes


def _is_multi_node() -> bool:
    """Checks if the MPI program is running on multiple nodes."""
    host = MPI.Get_processor_name().strip().lower()
    hosts = set(global_comm.allgather(host))
    return len(hosts) > 1


def setup_module():
    """setup any state specific to the execution of the given module."""
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
    # Configure the comm singleton.
    comm.configure(
        block_comm_size=1,
        block_comm_config=_default_config,
        stack_comm_config=_default_config,
        override=True,
    )


@pytest.mark.parametrize(
    "num_elements, num_sections, strategy, expected",
    [
        (10, 2, "balanced", ([5, 5], 10)),
        (10, 3, "greedy", ([4, 4, 2], 12)),
        (7, 3, "balanced", ([3, 2, 2], 9)),
        (7, 3, "greedy", ([3, 3, 1], 9)),
        (7, 7, "balanced", ([1, 1, 1, 1, 1, 1, 1], 7)),
        (7, 7, "greedy", ([1, 1, 1, 1, 1, 1, 1], 7)),
    ],
)
def test_get_section_sizes(
    num_elements: int,
    num_sections: int,
    strategy: str,
    expected: tuple[list[int], int],
):
    assert (
        get_section_sizes(
            num_elements=num_elements,
            num_sections=num_sections,
            strategy=strategy,
        )
        == expected
    )


@pytest.mark.skipif(
    _is_multi_node(),
    reason="This test only works if all ranks see the same file system.",
)
@pytest.mark.mpi(min_size=2)
def test_distributed_load_npy(mpi_tmp_path: Path):
    """Test the distributed_load function."""
    arr = None
    if global_comm.rank == 0:
        arr = xp.random.rand(10)
        xp.save(mpi_tmp_path / "arr.npy", arr)
    arr = global_comm.bcast(arr, root=0)

    loaded_arr = distributed_load(mpi_tmp_path / "arr.npy")
    assert xp.allclose(arr, loaded_arr)


@pytest.mark.skipif(
    _is_multi_node(),
    reason="This test only works if all ranks see the same file system.",
)
@pytest.mark.mpi(min_size=2)
def test_distributed_load_npz(mpi_tmp_path: Path):
    """Test the distributed_load function."""
    coo = None
    if global_comm.rank == 0:
        coo = sps.random(10, 10, density=0.5)
        sps.save_npz(mpi_tmp_path / "coo.npz", coo)
    coo = sparse.coo_matrix(global_comm.bcast(coo, root=0))

    loaded_arr = distributed_load(mpi_tmp_path / "coo.npz")
    assert xp.allclose(coo.toarray(), loaded_arr.toarray())


@pytest.mark.skipif(
    _is_multi_node(),
    reason="This test only works if all ranks see the same file system.",
)
@pytest.mark.mpi(min_size=2)
def test_distributed_load_h5(mpi_tmp_path: Path):
    """Test the distributed_load function."""
    dict = None
    if global_comm.rank == 0:
        dict = {
            "[0,0,0]": np.random.rand(10, 10),
            "[1,0,0]": sps.random(10, 10, density=0.5, format="csr"),
            "[0,1,0]": sps.random(10, 10, density=0.5, format="coo"),
            "[0,0,1]": sps.random(10, 10, density=0.5, format="csc"),
        }
        save_hdf5_dict(mpi_tmp_path / "dict.h5", dict)

    dict = global_comm.bcast(dict, root=0)

    # Distributed_load converts the keys back to tuples of ints, so we need to do the same for the original dict.
    dict = {
        tuple(map(int, r.strip("[]").split(","))): h_r
        for r, h_r in dict.items()
        if r.startswith("[")
    }

    loaded_dict = distributed_load(mpi_tmp_path / "dict.h5")

    for key in dict.keys():
        if isinstance(dict[key], sps.spmatrix):
            assert np.allclose(dict[key].toarray(), loaded_dict[key].toarray())
        else:
            assert np.allclose(dict[key], loaded_dict[key])


@pytest.mark.mpi(min_size=2)
def test_get_local_slice():
    """Test the distributed_load function."""
    global_array = xp.arange(10)
    local_arrays = xp.array_split(global_array, global_comm.size)
    assert xp.allclose(local_arrays[global_comm.rank], get_local_slice(global_array))
