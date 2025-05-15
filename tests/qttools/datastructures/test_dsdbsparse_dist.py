from contextlib import nullcontext
from typing import Callable

import numpy as np
import pytest
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse
from qttools.utils.mpi_utils import get_section_sizes


@pytest.fixture(autouse=True, scope="module", params=[3, 1])
def configure_comm(request):
    """Setup any state specific to the execution of the given module."""
    block_comm_size = request.param

    # Default configuration setup based on the xp module
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

    # Configure the comm singleton with the parameterized block_comm_size
    comm.configure(
        block_comm_size=block_comm_size,
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
        return coo
    if symmetric_sparsity:
        coo = coo + coo.T
        coo.data[:] = rng.uniform(size=coo.nnz)
    coo.data += 1j * rng.uniform(size=coo.nnz)
    return coo


@pytest.mark.mpi(min_size=3)
class TestCreation:
    """Tests the creation methods of DSDBSparse."""

    def test_from_sparray(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the creation of DSDBSparse matrices from sparse arrays."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )

        assert xp.array_equiv(coo.toarray(), dsdbsparse.to_dense())

    def test_zeros_like(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the creation of a zero DSDBSparse matrix with the same shape as another."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        zeros = dsdbsparse_type_dist.zeros_like(dsdbsparse)
        assert (zeros.to_dense() == 0).all()
        assert zeros.shape == dsdbsparse.shape


@pytest.mark.mpi(min_size=3)
class TestConversion:
    """Tests for the conversion methods of DSDBSparse."""

    def test_to_dense(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can convert a DSDBSparse matrix to dense."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        reference = xp.broadcast_to(coo.toarray(), dsdbsparse.shape)

        assert xp.allclose(reference, dsdbsparse.to_dense())

    def test_symmetrize(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        op: Callable[[NDArray, NDArray], NDArray],
    ):
        """Tests that we can transpose a DSDBSparse matrix."""
        coo = (
            _create_coo(block_sizes, symmetric_sparsity=True)
            if global_comm.rank == 0
            else None
        )
        coo = global_comm.bcast(coo, root=0)

        dense = coo.toarray()
        symmetrized = 0.5 * op(dense, dense.transpose().conj())

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        reference = xp.broadcast_to(symmetrized, dsdbsparse.shape)
        dsdbsparse.symmetrize(op)

        assert xp.allclose(reference, dsdbsparse.to_dense())


def _create_new_block_sizes(
    block_sizes: NDArray, block_change_factor: float
) -> NDArray:
    """Creates new block sizes based on the block change factor."""
    block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
    block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))
    num_local_blocks = block_section_sizes[comm.block.rank]
    local_block_sizes = block_sizes[block_section_offsets[comm.block.rank] :]

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

    updated_block_sizes = np.asarray(updated_block_sizes)

    block_section_sizes, __ = get_section_sizes(
        len(updated_block_sizes), comm.block.size
    )
    block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

    local_updated_block_sizes = updated_block_sizes[
        block_section_offsets[comm.block.rank] :
    ]
    inconsistent = sum(
        local_updated_block_sizes[: block_section_sizes[comm.block.rank]]
    ) != sum(local_block_sizes[:num_local_blocks])

    return updated_block_sizes, inconsistent


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


@pytest.mark.mpi(min_size=3)
class TestAccess:
    """Tests for the access methods of DSDBSparse."""

    @pytest.mark.usefixtures("accessed_block")
    def test_get_block(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry_type: tuple[bool, Callable],
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        symmetry, symmetry_op = symmetry_type
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        reference_block = dense[..., *inds]

        block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

        start_block, stop_block = (
            block_section_offsets[comm.block.rank],
            block_section_offsets[comm.block.rank + 1],
        )

        if (start_block <= accessed_block[0] and start_block <= accessed_block[1]) and (
            accessed_block[0] < stop_block or accessed_block[1] < stop_block
        ):
            accessed_block = (
                accessed_block[0] - start_block,
                accessed_block[1] - start_block,
            )
            with pytest.raises(IndexError) if not in_bounds else nullcontext():
                # Find the correct rank in block-comm
                assert xp.allclose(
                    reference_block, dsdbsparse.local_blocks[accessed_block]
                )

    @pytest.mark.usefixtures("accessed_block")
    def test_get_sparse_block(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)
        reference_block = dense[..., *inds]

        # We want to get sparse blocks.
        dsdbsparse.return_dense = False

        block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

        start_block, stop_block = (
            block_section_offsets[comm.block.rank],
            block_section_offsets[comm.block.rank + 1],
        )

        if (start_block <= accessed_block[0] and start_block <= accessed_block[1]) and (
            accessed_block[0] < stop_block or accessed_block[1] < stop_block
        ):
            accessed_block = (
                accessed_block[0] - start_block,
                accessed_block[1] - start_block,
            )

            with pytest.raises(IndexError) if not in_bounds else nullcontext():
                if "COO" in dsdbsparse_type_dist.__name__:
                    rows, cols, data = dsdbsparse.local_blocks[accessed_block]
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
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
    ):
        """Tests that we can set a block and not modify sparsity structure."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        inds, in_bounds = _get_block_inds(accessed_block, block_sizes)

        if not in_bounds:
            # If we wouldn't return here, there would be no error raised
            # with the current test setup.
            return

        block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

        start_block, stop_block = (
            block_section_offsets[comm.block.rank],
            block_section_offsets[comm.block.rank + 1],
        )

        if ((start_block <= accessed_block[0]) & (start_block <= accessed_block[1])) & (
            (accessed_block[0] < stop_block) | (accessed_block[1] < stop_block)
        ):
            accessed_block = (
                accessed_block[0] - start_block,
                accessed_block[1] - start_block,
            )

            with pytest.raises(IndexError) if not in_bounds else nullcontext():
                dsdbsparse.local_blocks[accessed_block] = xp.ones_like(
                    dense[..., *inds]
                )

        # Sparsity structure should not be modified.
        dense[..., *inds][dense[..., *inds].nonzero()] = 1
        assert xp.allclose(dense, dsdbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_block_substack(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct block from a substack."""

        # TODO: This test is not working with the current setup.
        # skip if block comm size is not 1
        if comm.block.size == 1:
            pytest.skip("Skipping test for non-block comm size 1.")

        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
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
        block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

        start_block, stop_block = (
            block_section_offsets[comm.block.rank],
            block_section_offsets[comm.block.rank + 1],
        )

        if (start_block <= accessed_block[0] and start_block <= accessed_block[1]) and (
            accessed_block[0] < stop_block or accessed_block[1] < stop_block
        ):
            accessed_block = (
                accessed_block[0] - start_block,
                accessed_block[1] - start_block,
            )
            with pytest.raises(IndexError) if not in_bounds else nullcontext():
                assert xp.allclose(
                    reference_block,
                    dsdbsparse.stack[stack_index].local_blocks[accessed_block],
                )

    @pytest.mark.usefixtures("accessed_block", "stack_index")
    def test_get_sparse_block_substack(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct block from a substack."""

        # TODO: This test is not working with the current setup.
        # skip if block comm size is not 1
        if comm.block.size == 1:
            pytest.skip("Skipping test for non-block comm size 1.")

        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
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

        block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

        start_block, stop_block = (
            block_section_offsets[comm.block.rank],
            block_section_offsets[comm.block.rank + 1],
        )

        if (start_block <= accessed_block[0] and start_block <= accessed_block[1]) and (
            accessed_block[0] < stop_block or accessed_block[1] < stop_block
        ):
            accessed_block = (
                accessed_block[0] - start_block,
                accessed_block[1] - start_block,
            )

            with pytest.raises(IndexError) if not in_bounds else nullcontext():
                if "COO" in dsdbsparse_type_dist.__name__:
                    rows, cols, data = dsdbsparse.stack[stack_index].local_blocks[
                        accessed_block
                    ]
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
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can set a block in a substack and not modify sparsity structure."""

        # TODO: This test is not working with the current setup.
        # skip if block comm size is not 1
        if comm.block.size == 1:
            pytest.skip("Skipping test for non-block comm size 1.")

        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
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

        if not in_bounds:
            # If we wouldn't return here, there would be no error raised
            # with the current test setup.
            return

        block_section_sizes, __ = get_section_sizes(len(block_sizes), comm.block.size)
        block_section_offsets = np.hstack(([0], np.cumsum(block_section_sizes)))

        start_block, stop_block = (
            block_section_offsets[comm.block.rank],
            block_section_offsets[comm.block.rank + 1],
        )

        if ((start_block <= accessed_block[0]) & (start_block <= accessed_block[1])) & (
            (accessed_block[0] < stop_block) | (accessed_block[1] < stop_block)
        ):
            accessed_block = (
                accessed_block[0] - start_block,
                accessed_block[1] - start_block,
            )

            with pytest.raises(IndexError) if not in_bounds else nullcontext():
                dsdbsparse.stack[stack_index].local_blocks[accessed_block] = (
                    xp.ones_like(dense[inds])
                )

        # Sparsity structure should not be modified.
        dense[inds][dense[inds].nonzero()] = 1
        assert xp.allclose(dense, dsdbsparse.to_dense())

    @pytest.mark.usefixtures("block_change_factor")
    def test_block_sizes_setter(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        block_change_factor: float,
    ):
        """Tests that we can update the block sizes correctly."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        # Create new block sizes.
        updated_block_sizes, inconsistent = _create_new_block_sizes(
            block_sizes, block_change_factor
        )

        # Create a new DSDBSparse matrix with the updated block sizes.
        dsdbsparse_updated_block_sizes = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=updated_block_sizes,
            global_stack_shape=global_stack_shape,
        )

        # Update the block sizes.
        with pytest.raises(ValueError) if inconsistent else nullcontext():
            dsdbsparse.block_sizes = updated_block_sizes

        inconsistent = comm.block._mpi_comm.allreduce(inconsistent, op=MPI.LOR)
        if inconsistent:
            # If the block sizes are inconsistent, we cannot compare the
            # two DSDBSparse matrices.
            return

        # Assert that the two DSDBSparse matrices are equivalent.
        assert xp.allclose(dsdbsparse.data, dsdbsparse_updated_block_sizes.data)

    def test_spy(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct sparsity pattern."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
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
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests that we can get the correct diagonal elements."""
        coo = _create_coo(block_sizes)
        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes=block_sizes,
            global_stack_shape=global_stack_shape,
        )
        dense = dsdbsparse.to_dense()

        reference = xp.diagonal(dense, axis1=-2, axis2=-1)
        assert xp.allclose(reference, dsdbsparse.diagonal())


@pytest.mark.mpi(min_size=3)
class TestDistribution:
    """Tests for the distribution methods of DSDBSparse."""

    def test_dtranspose(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests the distributed transpose method."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        assert dsdbsparse.distribution_state == "stack"

        original_data = dsdbsparse._data.copy()

        # Transpose forth.
        dsdbsparse.dtranspose()
        assert dsdbsparse.distribution_state == "nnz"

        # Transpose back.
        dsdbsparse.dtranspose()
        assert dsdbsparse.distribution_state == "stack"

        comm.stack.barrier()

        assert xp.allclose(original_data, dsdbsparse._data)

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_stack(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests distributed access of individual matrix elements."""
        symmetry, symmetry_op = symmetry_type
        coo = (
            _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
            if global_comm.rank == 0
            else None
        )
        coo = global_comm.bcast(coo, root=0)

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )

        reference = coo.tocsr()[*accessed_element]
        test = dsdbsparse[accessed_element]

        # This returns either the correct value or zeros if the element
        # is on a different rank in the comm.block.
        assert xp.allclose(reference, test) or (test == 0).all()

    @pytest.mark.usefixtures("accessed_element")
    def test_getitem_nnz(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
        symmetry_type: tuple[bool, Callable],
    ):
        """Tests distributed access of individual matrix elements."""
        symmetry, symmetry_op = symmetry_type
        coo = (
            _create_coo(block_sizes, symmetric=symmetry, symmetry_op=symmetry_op)
            if global_comm.rank == 0
            else None
        )
        coo = global_comm.bcast(coo, root=0)

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo,
            block_sizes,
            global_stack_shape,
            symmetry=symmetry,
            symmetry_op=symmetry_op,
        )

        reference = coo.tocsr()[*accessed_element]

        dsdbsparse.dtranspose()
        test = dsdbsparse[accessed_element]

        # This returns either the correct value or zeros if the element
        # is on a different rank in the comm.block.
        assert xp.allclose(reference, test) or (test == 0).all()

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem_stack(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsdbsparse.to_dense()

        dsdbsparse[accessed_element] = 42

        dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42
        assert xp.allclose(dense, dsdbsparse.to_dense())

    @pytest.mark.usefixtures("accessed_element")
    def test_setitem_nnz(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        accessed_element: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo = global_comm.bcast(coo, root=0)

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsdbsparse.to_dense()
        rows, cols = dsdbsparse.spy()
        row, col, __ = _unsign_index(*accessed_element, dense.shape[-1])
        ind = xp.where((rows == row) & (cols == col))[0]

        if len(ind) == 0:
            return

        dense[..., *accessed_element][dense[..., *accessed_element].nonzero()] = 42

        dsdbsparse.dtranspose()

        dsdbsparse[accessed_element] = 42

        dsdbsparse.dtranspose()

        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_diagonal_nnz(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests distributed access of individual matrix elements."""

        # TODO: This test is not working with the current setup.
        # skip if block comm size is not 1
        if comm.block.size != 1:
            pytest.skip("Skipping test for non-block comm size 1.")

        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo: sparse.coo_matrix = global_comm.bcast(coo, root=0)

        if comm.rank == 0:
            print(
                f"Diagonal nonzero elements: {xp.diagonal(coo.toarray(), axis1=-2, axis2=-1).nonzero()}"
            )

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = coo.toarray()

        reference = xp.diagonal(dense, axis1=-2, axis2=-1)
        reference = reference[reference.nonzero()]

        dsdbsparse.dtranspose()

        local_diagonal = dsdbsparse.diagonal()
        diagonal = xp.concatenate(global_comm.allgather(local_diagonal), axis=-1)

        if comm.rank == 0:
            print(f"Diagonal test: {diagonal}")
            print(f"Diagonal reference: {reference}")

        assert xp.allclose(reference, diagonal)

    def test_set_diagonal_nnz(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
    ):
        """Tests distributed setting of individual matrix elements."""
        coo = _create_coo(block_sizes) if global_comm.rank == 0 else None
        coo: sparse.coo_matrix = global_comm.bcast(coo, root=0)

        dsdbsparse = dsdbsparse_type_dist.from_sparray(
            coo, block_sizes, global_stack_shape
        )
        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.dtranspose()

        dsdbsparse.fill_diagonal(val=42)
        stack_index = (0,) * len(global_stack_shape)
        inds = dense[*stack_index, inds, inds].nonzero()
        dense[..., inds, inds] = 42

        dsdbsparse.dtranspose()

        assert xp.allclose(dense, dsdbsparse.to_dense())


if __name__ == "__main__":
    pytest.main([__file__])
