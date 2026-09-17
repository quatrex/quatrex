# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.
from contextlib import nullcontext

import numpy as np
import pytest
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import NDArray, sparse, xp
from qttools.comm import comm
from qttools.datastructures.dsdbsparse import DSDBSparse, symmetry_ops
from qttools.utils.gpu_utils import get_array_module_name
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

    if global_comm.size < block_comm_size:
        pytest.skip(
            f"Skipping test for block comm size {block_comm_size} with global comm size {global_comm.size}."
        )

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
    dsdbsparse_type_dist: DSDBSparse,
    block_sizes: NDArray,
    global_stack_shape: tuple,
    symmetry: str | None = None,
    symmetric_sparsity: bool = False,
) -> tuple[sparse.coo_matrix, DSDBSparse]:
    """Returns a random complex sparse array
    and a DSDBSparse matrix with the same sparsity pattern.
    """
    coo = (
        _create_coo(
            block_sizes, symmetry=symmetry, symmetric_sparsity=symmetric_sparsity
        )
        if global_comm.rank == 0
        else None
    )
    coo = global_comm.bcast(coo, root=0)

    dsdbsparse = dsdbsparse_type_dist.from_sparray(
        sparray=coo,
        block_sizes=block_sizes,
        global_stack_shape=global_stack_shape,
        symmetry=symmetry,
    )
    return coo, dsdbsparse


class TestCreation:
    """Tests the creation methods of DSDBSparse."""

    def test_from_sparray(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests the creation of DSDBSparse matrices from sparse arrays."""
        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        assert xp.array_equiv(coo.toarray(), dsdbsparse.to_dense())

    def test_empty_like(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests the creation of a empty DSDBSparse matrix with the same
        shape as another."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        empty = dsdbsparse_type_dist.empty_like(dsdbsparse)
        empty.allocate_data()
        empty.data[:] = 0
        assert (empty.to_dense() == 0).all()
        assert empty.shape == dsdbsparse.shape


@pytest.mark.mpi(min_size=2)
class TestCreationDist(TestCreation):
    """Tests all tests of TestCreation in distributed setting."""

    pass


class TestConversion:
    """Tests for the conversion methods of DSDBSparse."""

    def test_to_dense(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can convert a DSDBSparse matrix to dense."""
        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        reference = xp.broadcast_to(coo.toarray(), dsdbsparse.shape)

        assert xp.allclose(reference, dsdbsparse.to_dense())

    def test_symmetrize(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can transpose a DSDBSparse matrix."""

        if symmetry is None:
            pytest.skip("Skipping test since symmetry is None.")

        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            None,
            symmetric_sparsity=True,
        )

        dense = coo.toarray()
        symmetrized = 0.5 * (symmetry_ops[symmetry](dense.transpose()) + dense)

        reference = xp.broadcast_to(symmetrized, dsdbsparse.shape)
        dsdbsparse.symmetrize(symmetry)

        assert xp.allclose(reference, dsdbsparse.to_dense())


@pytest.mark.mpi(min_size=2)
class TestConversionDist(TestConversion):
    """Tests all tests of TestConversion in distributed setting."""

    pass


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


class TestAccess:
    """Tests for the access methods of DSDBSparse."""

    def test_get_block(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
        accessed_block: tuple,
    ):
        """Tests that we can get the correct block."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
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
                assert xp.allclose(reference_block, dsdbsparse.blocks[accessed_block])

    def test_set_block(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
        accessed_block: tuple,
    ):
        """Tests that we can set a block and not modify sparsity structure."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
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
                dsdbsparse.blocks[accessed_block] = xp.ones_like(dense[..., *inds])

        # Sparsity structure should not be modified.
        if symmetry is None:
            dense[..., *inds][dense[..., *inds].nonzero()] = 1
        else:
            # For symmetric matrices, we need to set the upper and lower
            if accessed_block[0] > accessed_block[1]:
                inds, _ = _get_block_inds(accessed_block[::-1], block_sizes)
                dense[..., *inds][dense[..., *inds].nonzero()] = symmetry_ops[symmetry](
                    1
                )
            else:
                dense[..., *inds][dense[..., *inds].nonzero()] = 1

            dense = xp.triu(dense)
            dense = dense + symmetry_ops[symmetry](dense.swapaxes(-2, -1))
            idx = xp.arange(dense.shape[-1])
            dense[..., idx, idx] = 0.5 * dense[..., idx, idx]

        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_get_block_substack(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can get the correct block from a substack."""

        # TODO: This test is not working with the current setup.
        # skip if block comm size is not 1
        if comm.block.size == 1:
            pytest.skip("Skipping test for non-block comm size 1.")

        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
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
                    dsdbsparse.stack[stack_index].blocks[accessed_block],
                )

    def test_set_block_substack(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
        accessed_block: tuple,
        stack_index: tuple,
    ):
        """Tests that we can set a block in a substack and not modify sparsity structure."""

        # TODO: This test is not working with the current setup.
        # skip if block comm size is not 1
        if comm.block.size == 1:
            pytest.skip("Skipping test for non-block comm size 1.")

        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
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
                dsdbsparse.stack[stack_index].blocks[accessed_block] = xp.ones_like(
                    dense[inds]
                )

        # Sparsity structure should not be modified.
        if symmetry is None:
            dense[inds][dense[inds].nonzero()] = 1
        else:
            # For symmetric matrices, we need to set the upper and lower
            if accessed_block[0] > accessed_block[1]:
                inds, _ = _get_block_inds(accessed_block[::-1], block_sizes)
                inds = (
                    stack_index
                    + (slice(None),) * (len(global_stack_shape) - len(stack_index))
                    + inds
                )
                dense[inds][dense[inds].nonzero()] = symmetry_ops[symmetry](1)
            else:
                dense[inds][dense[inds].nonzero()] = 1

            dense = xp.triu(dense)
            dense = dense + symmetry_ops[symmetry](dense.swapaxes(-2, -1))
            idx = xp.arange(dense.shape[-1])
            dense[..., idx, idx] = 0.5 * dense[..., idx, idx]

        assert xp.allclose(dense, dsdbsparse.to_dense())

    def test_block_sizes_setter(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        block_change_factor: float,
        symmetry: str | None,
    ):
        """Tests that we can update the block sizes correctly."""
        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        dsdbsparse_original = dsdbsparse_type_dist.empty_like(dsdbsparse)
        dsdbsparse_original.allocate_data()
        dsdbsparse_original.data = dsdbsparse.data

        # Create new block sizes.
        updated_block_sizes, inconsistent = _create_new_block_sizes(
            block_sizes, block_change_factor
        )

        # Create a new DSDBSparse matrix with the updated block sizes.
        dsdbsparse_updated_block_sizes = dsdbsparse_type_dist.from_sparray(
            sparray=coo,
            block_sizes=updated_block_sizes,
            global_stack_shape=global_stack_shape,
            symmetry=symmetry,
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
        attributes = [
            "data",
            "num_blocks",
            "block_section_offsets",
            "num_local_blocks",
            "local_block_sizes",
            "local_block_offsets",
            "global_block_offset",
            "block_offsets",
            "_diag_inds",
            "_diag_value_inds",
            "_diag_value_inds_nnz",
            "_diag_value_inds_nnz",
        ]

        for attr in attributes:
            actual = getattr(dsdbsparse, attr)
            expected = getattr(dsdbsparse_updated_block_sizes, attr)
            if actual is not None:
                if get_array_module_name(actual) == "numpy":
                    assert np.allclose(actual, expected)
                else:
                    assert xp.allclose(actual, expected)

        with pytest.raises(ValueError) if inconsistent else nullcontext():
            # Test caching
            dsdbsparse.block_sizes = block_sizes

        for attr in attributes:
            actual = getattr(dsdbsparse, attr)
            expected = getattr(dsdbsparse_original, attr)
            if actual is not None:
                if get_array_module_name(actual) == "numpy":
                    assert np.allclose(actual, expected)
                else:
                    assert xp.allclose(actual, expected)

        with pytest.raises(ValueError) if inconsistent else nullcontext():
            dsdbsparse.block_sizes = updated_block_sizes

        # Assert that the two DSDBSparse matrices are equivalent.
        for attr in attributes:
            actual = getattr(dsdbsparse, attr)
            expected = getattr(dsdbsparse_updated_block_sizes, attr)
            if actual is not None:
                if get_array_module_name(actual) == "numpy":
                    assert np.allclose(actual, expected)
                else:
                    assert xp.allclose(actual, expected)

    def test_spy(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can get the correct sparsity pattern."""
        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        if symmetry is not None:
            coo = sparse.triu(coo)

        inds = xp.lexsort(xp.vstack((coo.col, coo.row)))
        ref_col, ref_row = coo.col[inds], coo.row[inds]

        rows, cols = dsdbsparse.spy()
        rows = comm.block.all_gather_v(rows, axis=0)
        cols = comm.block.all_gather_v(cols, axis=0)

        inds = xp.lexsort(xp.vstack((cols, rows)))
        col, row = cols[inds], rows[inds]

        assert xp.allclose(ref_col, col)
        assert xp.allclose(ref_row, row)

    def test_diagonal(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests that we can get the correct diagonal elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        dense = dsdbsparse.to_dense()

        reference = xp.diagonal(dense, axis1=-2, axis2=-1)
        diagonal = dsdbsparse.diagonal()
        diagonal = comm.block.all_gather_v(diagonal, axis=-1)

        assert xp.allclose(reference, diagonal)


@pytest.mark.mpi(min_size=2)
class TestAccessDist(TestAccess):
    """Tests all tests of TestAccess in distributed setting."""

    pass


@pytest.mark.mpi(min_size=3)
class TestDistribution:
    """Tests for the distribution methods of DSDBSparse."""

    def test_dtranspose(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests the distributed transpose method."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )
        assert dsdbsparse.distribution_state == "stack"

        original_data = dsdbsparse.data.copy()

        # Transpose forth.
        dsdbsparse.dtranspose()
        assert dsdbsparse.distribution_state == "nnz"

        # Transpose back.
        dsdbsparse.dtranspose()
        assert dsdbsparse.distribution_state == "stack"

        comm.stack.barrier()

        assert xp.allclose(original_data, dsdbsparse.data)

    def test_diagonal_nnz(
        self,
        dsdbsparse_type_dist: DSDBSparse,
        block_sizes: NDArray,
        global_stack_shape: tuple,
        symmetry: str | None,
    ):
        """Tests distributed access of individual matrix elements."""

        # TODO: This test is not working with the current setup.
        # skip if block comm size is not 1
        if comm.block.size != 1:
            pytest.skip("Skipping test for non-block comm size 1.")

        coo, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
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
        symmetry: str | None,
    ):
        """Tests distributed setting of individual matrix elements."""
        _, dsdbsparse = _create_coo_dsdbsparse(
            dsdbsparse_type_dist,
            block_sizes,
            global_stack_shape,
            symmetry,
        )

        dense = dsdbsparse.to_dense()

        n = dsdbsparse.shape[-1]
        inds = xp.arange(n)

        dsdbsparse.dtranspose()

        dsdbsparse.fill_diagonal(val=42)
        stack_index = (0,) * len(global_stack_shape)
        inds = dense[*stack_index, inds, inds].nonzero()
        if symmetry is None:
            dense[..., inds, inds] = 42
        else:
            dense[..., inds, inds] = 0.5 * (symmetry_ops[symmetry](42) + 42)

        dsdbsparse.dtranspose()

        assert xp.allclose(dense, dsdbsparse.to_dense())


if __name__ == "__main__":
    pytest.main([__file__])
