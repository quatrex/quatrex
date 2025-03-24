from qttools import sparse, xp
from qttools.kernels.numba.dsbcoo import compute_block_sort_index
from qttools.utils.mpi_utils import distributed_load

from quatrex.coulomb_screening import CoulombScreeningSolver


def _block_canonicalize(rows, cols, block_sizes):
    """Canonicalize and block order for indices to have a consistent ordering."""
    # Canonical ordering of the transpose.
    inds_bcoo2canonical = xp.lexsort(xp.vstack((cols, rows)))
    canonical_rows = rows[inds_bcoo2canonical]
    canonical_cols = cols[inds_bcoo2canonical]
    # Compute index for sorting the transpose by block.
    inds_canonical2bcoo = compute_block_sort_index(
        canonical_rows, canonical_cols, block_sizes
    )
    # Mapping directly from original ordering to transpose
    # block-ordering is achieved by chaining the two mappings.
    inds_bcoo2bcoo = inds_bcoo2canonical[inds_canonical2bcoo]
    # Cache the necessary objects.
    return inds_bcoo2bcoo


def test_solve(
    datadir,
    quatrex_config,
    compute_config,
    block_sizes,
    coulomb_screening_energies,
    iteration,
):
    """Test the computation of the polarization."""
    # Load the data
    pl_data = distributed_load(datadir.joinpath(f"pl_data_iter{iteration}.npy"))
    pg_data = distributed_load(datadir.joinpath(f"pg_data_iter{iteration}.npy"))
    # TODO: Also load data for p_retarded
    wl_data = distributed_load(datadir.joinpath(f"wl_data_iter{iteration}.npy"))
    wg_data = distributed_load(datadir.joinpath(f"wg_data_iter{iteration}.npy"))
    rows = distributed_load(datadir.joinpath("rows.npy"))
    cols = distributed_load(datadir.joinpath("columns.npy"))
    reordering = _block_canonicalize(rows, cols, block_sizes)
    # Reorder the data
    pl_data = pl_data[..., reordering]
    pg_data = pg_data[..., reordering]
    wl_data = wl_data[..., reordering]
    wg_data = wg_data[..., reordering]
    rows = rows[reordering]
    cols = cols[reordering]
    # Create the DSBSparse objects
    number_of_kpoints = quatrex_config.electron.number_of_kpoints
    p_lesser = compute_config.dsbsparse_type(
        pl_data,
        rows,
        cols,
        block_sizes,
        (pl_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    p_greater = compute_config.dsbsparse_type(
        pg_data,
        rows,
        cols,
        block_sizes,
        (pg_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    num_connected_blocks = 3
    coulomb_screening_block_sizes = (
        block_sizes[: len(block_sizes) // num_connected_blocks] * num_connected_blocks
    )
    sparsity_pattern = sparse.coo_matrix((xp.ones_like(rows), (rows, cols)))
    w_lesser = compute_config.dsbsparse_type.from_sparray(
        sparsity_pattern.astype(xp.complex128),
        block_sizes=coulomb_screening_block_sizes,
        global_stack_shape=coulomb_screening_energies.shape
        + tuple([k for k in number_of_kpoints if k > 1]),
    )
    w_greater = compute_config.dsbsparse_type.zeros_like(w_lesser)
    w_retarded = compute_config.dsbsparse_type.zeros_like(w_lesser)
    # Symmetrize the polarization
    p_lesser.data[:] = (p_lesser.data - p_lesser.ltranspose(copy=True).data.conj()) / 2
    p_greater.data[:] = (
        p_greater.data - p_greater.ltranspose(copy=True).data.conj()
    ) / 2
    # Compute the retarded polarization
    p_retarded = compute_config.dsbsparse_type.zeros_like(p_lesser)
    p_retarded.data[:] = (p_greater.data - p_lesser.data) / 2
    # Initialize the coulomb screening solver
    coulomb_screening_solver = CoulombScreeningSolver(
        quatrex_config, compute_config, coulomb_screening_energies, sparsity_pattern
    )
    # Compute the screened interaction
    coulomb_screening_solver.solve(
        p_lesser,
        p_greater,
        p_retarded,
        out=(w_lesser, w_greater, w_retarded),
    )
    w_lesser.block_sizes = block_sizes
    w_greater.block_sizes = block_sizes
    # Compare the results
    # First energy is different, don't know why
    assert xp.allclose(w_lesser.data, wl_data)
    assert xp.allclose(w_greater.data, wg_data)
