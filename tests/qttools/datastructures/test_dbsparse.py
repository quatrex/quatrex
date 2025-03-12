# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from typing import Callable

import pytest
from mpi4py.MPI import COMM_WORLD as comm

from qttools import NDArray, sparse, xp
from qttools.datastructures.dbsparse import DBSparse


def _create_coo(sizes: NDArray, symmetric_sparsity: bool = False) -> sparse.coo_matrix:
    """Returns a random complex sparse array."""
    size = int(xp.sum(sizes))
    rng = xp.random.default_rng()
    density = rng.uniform(low=0.1, high=0.3)
    coo = sparse.random(size, size, density=density, format="coo").astype(xp.complex128)
    if symmetric_sparsity:
        coo = coo + coo.T
        coo.data[:] = rng.uniform(size=coo.nnz)
    coo.data += 1j * rng.uniform(size=coo.nnz)
    return coo


@pytest.mark.mpi
class TestCreation:
    """Tests the creation methods of DBSparse."""

    def test_from_sparray(
        self,
        dbsparse_type: DBSparse,
        block_sizes: NDArray,
    ):
        """Tests the creation of DBSparse matrices from sparse arrays."""
        coo = _create_coo(block_sizes)
        coo = comm.bcast(coo, root=0)
        dbsparse = dbsparse_type.from_sparray(coo, block_sizes)
        assert xp.array_equiv(coo.toarray(), dbsparse.to_dense())

    def test_zeros_like(
        self,
        dbsparse_type: DBSparse,
        block_sizes: NDArray,
    ):
        """Tests the creation of a zero DBSparse matrix with the same shape as another."""
        coo = _create_coo(block_sizes)
        coo = comm.bcast(coo, root=0)
        dbsparse = dbsparse_type.from_sparray(coo, block_sizes)
        zeros = dbsparse_type.zeros_like(dbsparse)
        assert (zeros.to_dense() == 0).all()


@pytest.mark.mpi
class TestConversion:
    """Tests for the conversion methods of DSBSparse."""

    def test_to_dense(
        self,
        dbsparse_type: DBSparse,
        block_sizes: NDArray,
    ):
        """Tests that we can convert a DSBSparse matrix to dense."""
        coo = _create_coo(block_sizes)
        coo = comm.bcast(coo, root=0)
        reference = coo.toarray()
        dbsparse = dbsparse_type.from_sparray(coo, block_sizes=block_sizes)
        assert xp.allclose(reference, dbsparse.to_dense())

    def test_symmetrize(
        self,
        dbsparse_type: DBSparse,
        block_sizes: NDArray,
        op: Callable[[NDArray, NDArray], NDArray],
    ):
        """Tests that we can symmetrize a DBSparse matrix."""
        coo = _create_coo(block_sizes, symmetric_sparsity=True)
        coo = comm.bcast(coo, root=0)
        dense = coo.toarray()
        reference = 0.5 * op(dense, dense.transpose().conj())
        dbsparse = dbsparse_type.from_sparray(coo, block_sizes=block_sizes)
        dbsparse.symmetrize(op)
        assert xp.allclose(reference, dbsparse.to_dense())


if __name__ == "__main__":
    pytest.main(["--only-mpi", __file__])
