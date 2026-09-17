# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from contextlib import nullcontext

import pytest

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse, _block_view, symmetry_ops


@pytest.fixture(autouse=True, scope="module")
def configure_comm():
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


def _create_coo(
    sizes: NDArray,
    symmetric_sparsity: bool = False,
    symmetry: str | None = None,
) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(xp.complex128)
    coo.setdiag(rng.uniform(size=size) + 1j * rng.uniform(size=size))
    if symmetry is not None:
        coo.data += 1j * rng.uniform(size=coo.nnz)
        coo_t = coo.copy()
        coo_t.data[:] = symmetry_ops[symmetry](coo_t.data)
        coo = coo + coo_t.T
        return coo
    if symmetric_sparsity:
        coo = coo + coo.T
        coo.data[:] = rng.uniform(size=coo.nnz)
    coo.data += 1j * rng.uniform(size=coo.nnz)
    return coo


def _create_coo_dsdbsparse(
    dsdbsparse_type: DSDBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
    symmetry: str | None,
    symmetric_sparsity: bool = False,
) -> tuple[sparse.coo_matrix, DSDBSparse]:
    """Returns a random complex sparse array
    and a DSDBSparse matrix with the same sparsity pattern.
    """
    coo = (
        _create_coo(
            block_sizes,
            symmetry=symmetry,
            symmetric_sparsity=symmetric_sparsity,
        )
        if comm.rank == 0
        else None
    )
    dsdbsparse = dsdbsparse_type.from_sparray(
        sparray=coo,
        block_sizes=block_sizes,
        global_stack_shape=global_stack_shape,
        symmetry=symmetry,
    )
    return coo, dsdbsparse


class TestAccess:
    """Tests for the access methods of DSDBSparse."""

    def test_diagonal_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
        symmetry: str | None,
    ):
        """Tests that we can get the correct diagonal elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        dense = dsdbsparse.to_dense()

        reference = xp.diagonal(dense[stack_index], axis1=-2, axis2=-1)
        assert xp.allclose(reference, dsdbsparse.diagonal(stack_index=stack_index))

    def test_set_diagonal(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can set the correct diagonal elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.fill_diagonal(val=xp.ones_like(dense[..., inds, inds]))
        stack_index = (0,) * len(global_stack_shape)
        inds = dense[*stack_index, inds, inds].nonzero()
        if symmetry is None:
            dense[..., inds, inds] = 1
        else:
            dense[..., inds, inds] = 0.5 * (symmetry_ops[symmetry](1) + 1)
        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_set_diagonal_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
        symmetry: str | None,
    ):
        """Tests that we can set the correct diagonal elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        data_stack = dsdbsparse.data[*stack_index]
        dsdbsparse.fill_diagonal(
            stack_index=stack_index, val=xp.ones((*data_stack.shape[:-1], n))
        )
        tmp_stack_index = (0,) * len(global_stack_shape)
        inds = dense[*tmp_stack_index, inds, inds].nonzero()
        if symmetry is None:
            dense[*stack_index][..., inds, inds] = 1
        else:
            dense[*stack_index][..., inds, inds] = 0.5 * (symmetry_ops[symmetry](1) + 1)
        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_set_diagonal_substack_val(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
        symmetry: str | None,
    ):
        """Tests that we can set the correct diagonal elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.fill_diagonal(stack_index=stack_index, val=2)
        tmp_stack_index = (0,) * len(global_stack_shape)
        inds = dense[*tmp_stack_index, inds, inds].nonzero()
        if symmetry is None:
            dense[*stack_index][..., inds, inds] = 2
        else:
            dense[*stack_index][..., inds, inds] = 0.5 * (symmetry_ops[symmetry](2) + 2)
        assert xp.allclose(dense, dsdbsparse.to_dense())


# Shape of the dense array.
ARRAY_SHAPE = (12, 10, 30)


@pytest.fixture()
def array() -> NDArray:
    """Returns a random dense array."""
    return xp.random.rand(*ARRAY_SHAPE)


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(0, id="axis-0"),
        pytest.param(-1, id="axis-(-1)"),
    ],
)
@pytest.mark.parametrize(
    "num_blocks",
    [
        pytest.param(2, id="2-blocks"),
        pytest.param(3, id="3-blocks"),
        pytest.param(5, id="5-blocks"),
    ],
)
def test_block_view(array: NDArray, axis: int, num_blocks: int):
    """Tests the block view helper function."""
    with (
        pytest.raises(ValueError)
        if ARRAY_SHAPE[axis] % num_blocks != 0
        else nullcontext()
    ):
        view = _block_view(array, axis, num_blocks)
        assert view.shape[0] == num_blocks

        for i in range(num_blocks):
            index = [slice(None)] * array.ndim
            size = array.shape[axis] // num_blocks
            index[axis] = slice(i * size, (i + 1) * size)
            assert (array[*index] == view[i]).all()


if __name__ == "__main__":
    pytest.main([__file__])
