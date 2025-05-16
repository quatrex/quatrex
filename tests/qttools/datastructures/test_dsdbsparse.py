# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from contextlib import nullcontext
from typing import Callable

import numpy as np
import pytest

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse, _block_view


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
    symmetric: bool = False,
    symmetry_op: Callable = xp.conj,
) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(xp.complex128)
    coo.setdiag(rng.uniform(size=size) + 1j * rng.uniform(size=size))
    if symmetric:
        coo.data += 1j * rng.uniform(size=coo.nnz)
        coo_t = coo.copy()
        coo_t.data[:] = symmetry_op(coo_t.data)
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
    symmetry_type: tuple[bool, Callable],
    symmetric_sparsity: bool = False,
) -> tuple[sparse.coo_matrix, DSDBSparse]:
    """Returns a random complex sparse array
    and a DSDBSparse matrix with the same sparsity pattern.
    """
    symmetry, symmetry_op = symmetry_type
    coo = (
        _create_coo(
            block_sizes,
            symmetric=symmetry,
            symmetry_op=symmetry_op,
            symmetric_sparsity=symmetric_sparsity,
        )
        if comm.rank == 0
        else None
    )
    dsdbsparse = dsdbsparse_type.from_sparray(
        coo,
        block_sizes,
        global_stack_shape,
        symmetry=symmetry,
        symmetry_op=symmetry_op,
    )
    return coo, dsdbsparse


def _create_new_block_sizes(
    block_sizes: NDArray, block_change_factor: float
) -> NDArray:
    """Creates new block sizes based on the block change factor."""
    rest = 0
    updated_block_sizes = []
    for bs in block_sizes:
        if sum(updated_block_sizes) < sum(block_sizes):
            # Calculate the new block size.
            el = max(int(bs * block_change_factor), 1)
            # Calculate the number of repetitions and the rest. The rest is added to the next block.
            reps, rest = max(divmod(bs + rest, el), (1, 0))
            # Add the new block size to the list.
            updated_block_sizes = updated_block_sizes + [el] * int(reps)
        else:
            # Break if the sum of the updated block sizes is equal or greater than the sum of the original block sizes.
            break
    if sum(updated_block_sizes) != sum(block_sizes):
        # Add the rest to the last block.
        updated_block_sizes[-1] += sum(block_sizes) - sum(updated_block_sizes)
    return np.asarray(updated_block_sizes)


def _unsign_index(row: int, col: int, num_blocks) -> tuple:
    """Adjusts the sign to allow negative indices and checks bounds."""
    row = num_blocks + row if row < 0 else row
    col = num_blocks + col if col < 0 else col
    in_bounds = 0 <= row < num_blocks and 0 <= col < num_blocks
    return row, col, in_bounds


def _get_block_inds(block: tuple, block_sizes: NDArray) -> tuple:
    """Returns the equivalent dense indices for a block."""
    block_offsets = np.hstack(([0], np.cumsum(block_sizes)), dtype=np.int32)
    num_blocks = len(block_sizes)

    # Normalize negative indices.
    row, col, in_bounds = _unsign_index(*block, num_blocks)
    index = (
        slice(block_offsets[row], block_offsets[row + 1]),
        slice(block_offsets[col], block_offsets[col + 1]),
    )

    return index, in_bounds


class TestAccess:
    """Tests for the access methods of DSDBSparse."""

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
        accessed_element: tuple,
    ):
        """Tests that we can get individual matrix elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        reference = dense[..., *accessed_element]
        assert xp.allclose(reference, dsdbsparse[accessed_element])

    @pytest.mark.usefixtures("num_inds")
    def test_getitem_with_array(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
        num_inds: int,
    ):
        """Tests that we can get multiple matrix elements at once."""
        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )

        # Generate a number of unique indices.
        rows = xp.random.choice(coo.shape[0], size=num_inds, replace=False)
        cols = xp.random.choice(coo.shape[1], size=num_inds, replace=False)

        reference_data = coo.tocsr()[rows, cols]
        if sparse.issparse(reference_data):
            reference_data = reference_data.toarray()
        reference = xp.broadcast_to(
            reference_data, global_stack_shape + reference_data.shape[1:]
        )
        assert xp.allclose(reference, dsdbsparse[rows, cols])

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
        accessed_element: tuple,
    ):
        """Tests that we can set individual matrix elements."""
        symmetry, symmetry_op = symmetry_type
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )

        dense = dsdbsparse.to_dense()

        val = 42 + 42j

        if symmetry:
            sym_val = symmetry_op(val)
            r, c = accessed_element
            if r == c:
                val = (val + sym_val) / 2
                dsdbsparse[accessed_element] = val
                dense[..., *accessed_element][
                    dense[..., *accessed_element].nonzero()
                ] = val
            else:
                dsdbsparse[accessed_element] = val
                dense[..., *accessed_element][
                    dense[..., *accessed_element].nonzero()
                ] = val
                dense[..., *accessed_element[::-1]][
                    dense[..., *accessed_element[::-1]].nonzero()
                ] = sym_val
        else:
            dsdbsparse[accessed_element] = val
            dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = val

        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_diagonal_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests that we can get the correct diagonal elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        reference = xp.diagonal(dense[stack_index], axis1=-2, axis2=-1)
        assert xp.allclose(reference, dsdbsparse.diagonal(stack_index=stack_index))

    def test_set_diagonal(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests that we can set the correct diagonal elements."""
        _, symmetry_op = symmetry_type
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.fill_diagonal(val=xp.ones_like(dense[..., inds, inds]))
        stack_index = (0,) * len(global_stack_shape)
        inds = dense[*stack_index, inds, inds].nonzero()
        dense[..., inds, inds] = 0.5 * (symmetry_op(1) + 1)
        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_set_diagonal_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests that we can set the correct diagonal elements."""
        _, symmetry_op = symmetry_type
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
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
        dense[*stack_index][..., inds, inds] = 0.5 * (symmetry_op(1) + 1)
        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_set_diagonal_substack_val(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests that we can set the correct diagonal elements."""
        _, symmetry_op = symmetry_type
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.fill_diagonal(stack_index=stack_index, val=2)
        tmp_stack_index = (0,) * len(global_stack_shape)
        inds = dense[*tmp_stack_index, inds, inds].nonzero()
        dense[*stack_index][..., inds, inds] = 0.5 * (symmetry_op(2) + 2)
        assert xp.allclose(dense, dsdbsparse.to_dense())


class TestArithmetic:
    """Tests for the arithmetic operations of DSDBSparse."""

    def test_iadd(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        dsdbsparse += dsdbsparse

        assert xp.allclose(dense + dense, dsdbsparse.to_dense())

    def test_iadd_coo(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the in-place addition of a DSDBSparse matrix with a COO matrix."""

        if dsdbsparse_type.__name__ == "DSDBCSR":
            pytest.skip("DSDBCSR does not support in-place addition.")

        if symmetry_type[0]:
            pytest.skip("Symmetric DSDBSparse does not support in-place addition.")

        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )

        dsdbsparse += coo.copy()

        assert xp.allclose(dsdbsparse.to_dense(), 2 * coo.toarray())

    def test_isub(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the in-place subtraction of a DSDBSparse matrix."""
        symmetry, symmetry_op = symmetry_type
        coo, dsdbsparse_1 = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense_1 = dsdbsparse_1.to_dense()

        coo.data[:] = xp.random.uniform(size=coo.nnz)
        dsdbsparse_2 = dsdbsparse_type.from_sparray(
            coo,
            block_sizes,
            global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )
        dense_2 = dsdbsparse_2.to_dense()

        dsdbsparse_1 -= dsdbsparse_2

        assert xp.allclose(dense_1 - dense_2, dsdbsparse_1.to_dense())

    def test_isub_coo(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the in-place subtraction of a DSDBSparse matrix with a COO matrix."""

        if dsdbsparse_type.__name__ == "DSDBCSR":
            pytest.skip("DSDBCSR does not support in-place subtraction.")

        if symmetry_type[0]:
            pytest.skip("Symmetric DSDBSparse does not support in-place subtraction.")

        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        dsdbsparse -= 2 * coo

        assert xp.allclose(dense - 2 * coo.toarray(), dsdbsparse.to_dense())

    def test_imul(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the in-place multiplication of a DSDBSparse matrix."""

        if symmetry_type[0]:
            pytest.skip(
                "Symmetric DSDBSparse does not support in-place multiplication."
            )

        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        dsdbsparse *= dsdbsparse

        assert xp.allclose(dense * dense, dsdbsparse.to_dense())

    def test_neg(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the negation of a DSDBSparse matrix."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type,
            block_sizes,
            global_stack_shape,
            symmetry_type,
        )
        dense = dsdbsparse.to_dense()

        assert xp.allclose(-dense, (-dsdbsparse).to_dense())


# Shape of the dense array.
ARRAY_SHAPE = (12, 10, 30)


@pytest.fixture(autouse=True)
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
