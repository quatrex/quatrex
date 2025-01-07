from qttools import xp
from qttools.kernels import dsbcoo_kernels
from qttools.utils.mpi_utils import distributed_load

from quatrex.electron import ElectronSolver


def _block_canonicalize(rows, cols, block_sizes):
    """Canonicalize and block order for indices to have a consistent ordering."""
    # Canonical ordering of the transpose.
    inds_bcoo2canonical = xp.lexsort(xp.vstack((cols, rows)))
    canonical_rows = rows[inds_bcoo2canonical]
    canonical_cols = cols[inds_bcoo2canonical]
    # Compute index for sorting the transpose by block.
    inds_canonical2bcoo = dsbcoo_kernels.compute_block_sort_index(
        canonical_rows, canonical_cols, block_sizes
    )
    # Mapping directly from original ordering to transpose
    # block-ordering is achieved by chaining the two mappings.
    inds_bcoo2bcoo = inds_bcoo2canonical[inds_canonical2bcoo]
    # Cache the necessary objects.
    return inds_bcoo2bcoo


def test_solve(
    datadir, quatrex_config, compute_config, block_sizes, electron_energies, iteration
):
    """Test the computation of the polarization."""
    # Load the data
    gl_data = distributed_load(datadir.joinpath(f"gl_data_iter{iteration}.npy"))
    gg_data = distributed_load(datadir.joinpath(f"gg_data_iter{iteration}.npy"))
    gr_data = distributed_load(datadir.joinpath(f"gr_data_iter{iteration}.npy"))
    if iteration == 0:
        sl_data = xp.zeros_like(gl_data)
        sg_data = xp.zeros_like(gg_data)
        sr_data = xp.zeros_like(gr_data)
    else:
        sl_data = distributed_load(datadir.joinpath(f"sl_data_iter{iteration-1}.npy"))
        sg_data = distributed_load(datadir.joinpath(f"sg_data_iter{iteration-1}.npy"))
        sr_data = distributed_load(datadir.joinpath(f"sr_data_iter{iteration-1}.npy"))
    rows = distributed_load(datadir.joinpath("rows.npy"))
    cols = distributed_load(datadir.joinpath("columns.npy"))
    reordering = _block_canonicalize(rows, cols, block_sizes)
    # Reorder the data
    sl_data = sl_data[..., reordering] * quatrex_config.scba.mixing_factor
    sg_data = sg_data[..., reordering] * quatrex_config.scba.mixing_factor
    sr_data = sr_data[..., reordering] * quatrex_config.scba.mixing_factor
    gl_data = gl_data[..., reordering]
    gg_data = gg_data[..., reordering]
    gr_data = gr_data[..., reordering]
    rows = rows[reordering]
    cols = cols[reordering]
    # Create the DSBSparse objects
    number_of_kpoints = quatrex_config.electron.number_of_kpoints
    sigma_lesser = compute_config.dbsparse_type(
        sl_data,
        rows,
        cols,
        block_sizes,
        (sl_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    sigma_greater = compute_config.dbsparse_type(
        sg_data,
        rows,
        cols,
        block_sizes,
        (sg_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    sigma_retarded = compute_config.dbsparse_type(
        sr_data,
        rows,
        cols,
        block_sizes,
        (sr_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    # Initialize the output objects
    g_lesser = compute_config.dbsparse_type.zeros_like(sigma_lesser)
    g_greater = compute_config.dbsparse_type.zeros_like(sigma_greater)
    g_retarded = compute_config.dbsparse_type.zeros_like(sigma_retarded)
    # Initialize the solver object
    electron_solver = ElectronSolver(quatrex_config, compute_config, electron_energies)
    # Compute the green's functions
    electron_solver.solve(
        sigma_lesser,
        sigma_greater,
        sigma_retarded,
        out=(g_lesser, g_greater, g_retarded),
    )
    # Compare the results
    assert xp.allclose(g_lesser.data, gl_data, atol=1e-7)
    assert xp.allclose(g_greater.data, gg_data, atol=1e-7)
    # NOTE: Different sparsity patterns for retarded Green's function
    # assert xp.allclose(g_retarded.data, gr_data)
