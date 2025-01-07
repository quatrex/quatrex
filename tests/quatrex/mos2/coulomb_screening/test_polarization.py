from qttools import xp
from qttools.kernels.numba.dsbcoo import compute_block_sort_index
from qttools.utils.mpi_utils import distributed_load

from quatrex.coulomb_screening import PCoulombScreening


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


def test_compute(
    datadir,
    compute_config,
    block_sizes,
    coulomb_screening_energies,
    number_of_kpoints,
    iteration,
):
    """Test the computation of the polarization."""
    # Load the data
    gl_data = distributed_load(datadir.joinpath(f"gl_data_iter{iteration}.npy"))
    gg_data = distributed_load(datadir.joinpath(f"gg_data_iter{iteration}.npy"))
    pl_data = distributed_load(datadir.joinpath(f"pl_data_iter{iteration}.npy"))
    pg_data = distributed_load(datadir.joinpath(f"pg_data_iter{iteration}.npy"))
    rows = distributed_load(datadir.joinpath("rows.npy"))
    cols = distributed_load(datadir.joinpath("columns.npy"))
    reordering = _block_canonicalize(rows, cols, block_sizes)
    # Reorder the data
    gl_data = gl_data[..., reordering]
    gg_data = gg_data[..., reordering]
    pl_data = pl_data[..., reordering]
    pg_data = pg_data[..., reordering]
    rows = rows[reordering]
    cols = cols[reordering]
    # Create the DSBSparse objects
    g_lesser = compute_config.dbsparse_type(
        gl_data,
        rows,
        cols,
        block_sizes,
        (gl_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    g_greater = compute_config.dbsparse_type(
        gg_data,
        rows,
        cols,
        block_sizes,
        (gg_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    p_lesser = compute_config.dbsparse_type.zeros_like(g_lesser)
    p_greater = compute_config.dbsparse_type.zeros_like(g_greater)
    p_retarded = compute_config.dbsparse_type.zeros_like(g_greater)
    # Create the expected results
    p_lesser_expected = compute_config.dbsparse_type(
        pl_data,
        rows,
        cols,
        block_sizes,
        (pl_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    p_greater_expected = compute_config.dbsparse_type(
        pg_data,
        rows,
        cols,
        block_sizes,
        (pg_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    # Initialize the polarization object
    p_coulomb_screening = PCoulombScreening(
        coulomb_screening_energies, number_of_kpoints
    )
    # Compute the polarization
    p_coulomb_screening.compute(
        g_lesser,
        g_greater,
        out=(p_lesser, p_greater, p_retarded),
    )
    # Compare the results
    assert xp.allclose(p_lesser.data, p_lesser_expected.data)
    assert xp.allclose(p_greater.data, p_greater_expected.data)
