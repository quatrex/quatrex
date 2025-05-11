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
    if symmetric:
        coo.data += 1j * rng.uniform(size=coo.nnz)
        coo_t = coo.copy()
        coo_t.data[:] = symmetry_op(coo_t.data)
        coo = coo + coo_t.T
        # NOTE: The following works only with scipy on the host.
        # coo = coo + symmetry_op(coo.T)
        return coo
    if symmetric_sparsity:
        coo = coo + coo.T
        coo.data[:] = rng.uniform(size=coo.nnz)
    coo.data += 1j * rng.uniform(size=coo.nnz)
    return coo


class TestCreation:
    """Tests the creation methods of DSDBSparse."""

    # def test_from_sparray_hermitian(
    #     self,
    #     dsdbsparse_type: DSDBSparse,
    #     block_sizes: NDArray,
    #     global_stack_shape: int | tuple,
    # ):
    #     """Tests the creation of DSDBSparse matrices from hermitian sparse arrays."""
    #     coo = _create_coo(block_sizes)
    #     coo = (coo + coo.conj().T) / 2
    #     dsdbsparse = dsdbsparse_type.from_sparray(
    #         coo, block_sizes, global_stack_shape, symmetry=True
    #     )
    #     assert xp.array_equiv(coo.toarray(), dsdbsparse.to_dense())

    def test_from_sparray(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: int | tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the creation of DSDBSparse matrices from sparse arrays."""
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes,
            global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )
        assert xp.array_equiv(coo.toarray(), dsdbsparse.to_dense())

    def test_zeros_like(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: int | tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests the creation of a zero DSDBSparse matrix with the same shape as another."""
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes,
            global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )
        zeros = dsdbsparse_type.zeros_like(dsdbsparse)
        assert (zeros.to_dense() == 0).all()


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


class TestConversion:
    """Tests for the conversion methods of DSDBSparse."""

    def test_to_dense(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests that we can convert a DSDBSparse matrix to dense."""
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)

        reference = xp.broadcast_to(coo.toarray(), global_stack_shape + coo.shape)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )

        assert xp.allclose(reference, dsdbsparse.to_dense())

    def test_symmetrize(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        op: Callable[[NDArray, NDArray], NDArray],
    ):
        """Tests that we can transpose a DSDBSparse matrix."""
        coo = _create_coo(block_sizes, symmetric_sparsity=True)

        dense = coo.toarray()
        symmetrized = 0.5 * op(dense, dense.transpose().conj())
        reference = xp.broadcast_to(symmetrized, global_stack_shape + symmetrized.shape)

        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dsdbsparse.symmetrize(op)

        assert xp.allclose(reference, dsdbsparse.to_dense())


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
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
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
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
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
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
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

    @pytest.mark.usefixtures("accessed_block")
    def test_get_block(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        reference_block = dense[..., *inds]

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            assert xp.allclose(reference_block, dsdbsparse.blocks[accessed_block])

    @pytest.mark.skip
    @pytest.mark.usefixtures("accessed_block")
    def test_get_sparse_block(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        symmetry, symmetry_op = False, lambda x: x
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        reference_block = dense[..., *inds]

        # We want to get sparse blocks.
        dsdbsparse.return_dense = False

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if "CSR" in dsdbsparse_type.__name__:
                rowptr, cols, data = dsdbsparse.blocks[accessed_block]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.csr_matrix(
                        (data[ind], cols, rowptr),
                        shape=reference_block.shape[-2:],
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            elif "COO" in dsdbsparse_type.__name__:
                rows, cols, data = dsdbsparse.blocks[accessed_block]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.coo_matrix(
                        (data[ind], (rows, cols)), shape=reference_block.shape[-2:]
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            else:
                raise ValueError("Unknown DSDBSparse type.")

    @pytest.mark.usefixtures("accessed_block")
    def test_set_block(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
        accessed_block: tuple,
    ):
        """Tests that we can set a block and not modify sparsity structure."""
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)

        rng = xp.random.default_rng()
        # val = xp.ones_like(dense[..., *inds]) + 1j * xp.ones_like(dense[..., *inds])
        val = rng.uniform(size=dense[..., *inds].shape) + 1j * rng.uniform(
            size=dense[..., *inds].shape
        )

        if symmetry:
            sym_val = symmetry_op(val).swapaxes(-2, -1)
            r, c = accessed_block
            if r == c:
                val = (val + sym_val) / 2
                with pytest.raises(IndexError) if not in_bounds else nullcontext():
                    dsdbsparse.blocks[accessed_block] = val
                dense[..., *inds][dense[..., *inds].nonzero()] = val[
                    dense[..., *inds].nonzero()
                ]
            else:
                with pytest.raises(IndexError) if not in_bounds else nullcontext():
                    dsdbsparse.blocks[accessed_block] = val
                dense[..., *inds][dense[..., *inds].nonzero()] = val[
                    *dense[..., *inds].nonzero()
                ]
                dense[..., *inds[::-1]][dense[..., *inds[::-1]].nonzero()] = sym_val[
                    dense[..., *inds[::-1]].nonzero()
                ]
        else:
            with pytest.raises(IndexError) if not in_bounds else nullcontext():
                dsdbsparse.blocks[accessed_block] = val
            dense[..., *inds][dense[..., *inds].nonzero()] = val[
                dense[..., *inds].nonzero()
            ]

        assert xp.allclose(dense, dsdbsparse.to_dense())

    # @pytest.mark.usefixtures("accessed_block")
    # def test_set_block_hermitian(
    #     self,
    #     dsdbsparse_type: DSDBSparse,
    #     block_sizes: NDArray,
    #     global_stack_shape: tuple,
    #     accessed_block: tuple,
    # ):
    #     """Tests that we can set a block and not modify sparsity structure."""
    #     coo = _create_coo(block_sizes)
    #     coo = (coo + coo.conj().T) / 2

    #     dsdbsparse = dsdbsparse_type.from_sparray(
    #         coo,
    #         block_sizes=block_sizes,
    #         global_stack_shape=global_stack_shape,
    #         symmetry=True,
    #     )
    #     dense = dsdbsparse.to_dense()

    #     inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
    #     inds_t, in_bounds_t = _get_block_inds(reversed(accessed_block), block_sizes)

    #     with pytest.raises(IndexError) if not in_bounds else nullcontext():
    #         dsdbsparse.blocks[accessed_block] = xp.ones_like(dense[..., *inds])

    #         # Sparsity structure should not be modified.
    #         dense[..., *inds][dense[..., *inds].nonzero()] = 1
    #         dense[..., *inds_t][dense[..., *inds_t].nonzero()] = 1
    #         assert xp.allclose(dense, dsdbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_block_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        # symmetry_type: tuple[bool, Callable],
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct block from a substack."""
        # symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        inds = (
            stack_index
            + (slice(None),) * (len(global_stack_shape) - len(stack_index))
            + inds
        )
        reference_block = dense[inds]
        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            assert xp.allclose(
                reference_block, dsdbsparse.stack[stack_index].blocks[accessed_block]
            )

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_sparse_block_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct block from a substack."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        inds = (
            stack_index
            + (slice(None),) * (len(global_stack_shape) - len(stack_index))
            + inds
        )
        reference_block = dense[inds]

        # We want to get sparse blocks.
        dsdbsparse.return_dense = False

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            if "CSR" in dsdbsparse_type.__name__:
                rowptr, cols, data = dsdbsparse.stack[stack_index].blocks[
                    accessed_block
                ]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.csr_matrix(
                        (data[ind], cols, rowptr),
                        shape=reference_block.shape[-2:],
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            elif "COO" in dsdbsparse_type.__name__:
                rows, cols, data = dsdbsparse.stack[stack_index].blocks[accessed_block]
                for ind in xp.ndindex(reference_block.shape[:-2]):
                    block = sparse.coo_matrix(
                        (data[ind], (rows, cols)), shape=reference_block.shape[-2:]
                    )
                    assert xp.allclose(reference_block[ind], block.toarray())

            else:
                raise ValueError("Unknown DSDBSparse type.")

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_set_block_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can set a block in a substack and not modify sparsity structure."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        inds = (
            stack_index
            + (slice(None),) * (len(global_stack_shape) - len(stack_index))
            + inds
        )

        with pytest.raises(IndexError) if not in_bounds else nullcontext():
            dsdbsparse.stack[stack_index].blocks[accessed_block] = xp.ones_like(
                dense[inds]
            )

        # Sparsity structure should not be modified.
        dense[inds][dense[inds].nonzero()] = 1
        assert xp.allclose(dense, dsdbsparse.to_dense())

    @pytest.mark.usefixtures("block_change_factor")
    def test_block_sizes_setter(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        block_change_factor: float,
    ):
        """Tests that we can update the block sizes correctly."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        # Create new block sizes.
        updated_block_sizes = _create_new_block_sizes(block_sizes, block_change_factor)
        # Create a new DSDBSparse matrix with the updated block sizes.
        dsdbsparse_updated_block_sizes = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=updated_block_sizes,
            global_stack_shape=global_stack_shape,
        )

        # Update the block sizes.
        dsdbsparse.block_sizes = updated_block_sizes

        # Assert that the two DSDBSparse matrices are equivalent.
        assert (dsdbsparse.data == dsdbsparse_updated_block_sizes.data).all()

    def test_spy(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct sparsity pattern."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        inds = xp.lexsort(xp.vstack((coo.col, coo.row)))
        ref_col, ref_row = coo.col[inds], coo.row[inds]

        rows, cols = dsdbsparse.spy()
        inds = xp.lexsort(xp.vstack((cols, rows)))
        col, row = cols[inds], rows[inds]

        assert xp.allclose(ref_col, col)
        assert xp.allclose(ref_row, row)

    def test_diagonal(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct diagonal elements."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        reference = xp.diagonal(dense, axis1=-2, axis2=-1)
        assert xp.allclose(reference, dsdbsparse.diagonal())

    def test_diagonal_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct diagonal elements."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        reference = xp.diagonal(dense[stack_index], axis1=-2, axis2=-1)
        assert xp.allclose(reference, dsdbsparse.diagonal(stack_index=stack_index))

    def test_set_diagonal(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can set the correct diagonal elements."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.fill_diagonal(val=xp.ones_like(dense[..., inds, inds]))
        stack_index = (0,) * len(global_stack_shape)
        inds = dense[*stack_index, inds, inds].nonzero()
        dense[..., inds, inds] = 1
        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_set_diagonal_substack(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
    ):
        """Tests that we can set the correct diagonal elements."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
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
        dense[*stack_index][..., inds, inds] = 1
        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_set_diagonal_substack_val(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        stack_index: tuple,
    ):
        """Tests that we can set the correct diagonal elements."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.fill_diagonal(stack_index=stack_index, val=2)
        tmp_stack_index = (0,) * len(global_stack_shape)
        inds = dense[*tmp_stack_index, inds, inds].nonzero()
        dense[*stack_index][..., inds, inds] = 2
        assert xp.allclose(dense, dsdbsparse.to_dense())


class TestArithmetic:
    """Tests for the arithmetic operations of DSDBSparse."""

    def test_iadd(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()

        dsdbsparse += dsdbsparse

        assert xp.allclose(dense + dense, dsdbsparse.to_dense())

    def test_iadd_coo(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place addition of a DSDBSparse matrix with a COO matrix."""

        if dsdbsparse_type.__name__ == "DSDBCSR":
            pytest.skip("DSDBCSR does not support in-place addition.")

        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)

        dsdbsparse += coo.copy()

        assert xp.allclose(dsdbsparse.to_dense(), 2 * coo.toarray())

    def test_isub(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place subtraction of a DSDBSparse matrix."""
        coo = _create_coo(block_sizes)

        dsdbsparse_1 = dsdbsparse_type.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense_1 = dsdbsparse_1.to_dense()

        dsdbsparse_2 = dsdbsparse_type.from_sparray(
            2 * coo, block_sizes, global_stack_shape
        )
        dense_2 = dsdbsparse_2.to_dense()

        dsdbsparse_1 -= dsdbsparse_2

        assert xp.allclose(dense_1 - dense_2, dsdbsparse_1.to_dense())

    def test_isub_coo(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place subtraction of a DSDBSparse matrix with a COO matrix."""

        if dsdbsparse_type.__name__ == "DSDBCSR":
            pytest.skip("DSDBCSR does not support in-place subtraction.")

        coo = _create_coo(block_sizes)

        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()

        dsdbsparse -= 2 * coo

        assert xp.allclose(dense - 2 * coo.toarray(), dsdbsparse.to_dense())

    def test_imul(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the in-place multiplication of a DSDBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
        dense = dsdbsparse.to_dense()

        dsdbsparse *= dsdbsparse

        assert xp.allclose(dense * dense, dsdbsparse.to_dense())

    def test_neg(
        self,
        dsdbsparse_type: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the negation of a DSDBSparse matrix."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type.from_sparray(coo, block_sizes, global_stack_shape)
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
