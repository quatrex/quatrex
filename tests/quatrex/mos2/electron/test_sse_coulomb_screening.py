from qttools import sparse, xp
from qttools.kernels.numba.dsbcoo import compute_block_sort_index
from qttools.utils.mpi_utils import distributed_load

from quatrex.electron import SigmaCoulombScreening, SigmaFock


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
    quatrex_config,
    compute_config,
    block_sizes,
    electron_energies,
    number_of_kpoints,
    iteration,
):
    """Test the computation of the polarization."""
    # Load the data
    gl_data = distributed_load(datadir.joinpath(f"gl_data_iter{iteration}.npy"))
    gg_data = distributed_load(datadir.joinpath(f"gg_data_iter{iteration}.npy"))
    wl_data = distributed_load(datadir.joinpath(f"wl_data_iter{iteration}.npy"))
    wg_data = distributed_load(datadir.joinpath(f"wg_data_iter{iteration}.npy"))
    sl_data = distributed_load(datadir.joinpath(f"sl_data_iter{iteration}.npy"))
    sg_data = distributed_load(datadir.joinpath(f"sg_data_iter{iteration}.npy"))
    sr_data = distributed_load(datadir.joinpath(f"sr_data_iter{iteration}.npy"))
    rows = distributed_load(datadir.joinpath("rows.npy"))
    cols = distributed_load(datadir.joinpath("columns.npy"))
    reordering = _block_canonicalize(rows, cols, block_sizes)
    # Reorder the data
    gl_data = gl_data[..., reordering]
    gg_data = gg_data[..., reordering]
    wl_data = wl_data[..., reordering]
    wg_data = wg_data[..., reordering]
    sl_data = sl_data[..., reordering]
    sg_data = sg_data[..., reordering]
    sr_data = sr_data[..., reordering]
    rows = rows[reordering]
    cols = cols[reordering]
    # Create the DSBSparse objects
    g_lesser = compute_config.dsbsparse_type(
        gl_data,
        rows,
        cols,
        block_sizes,
        (gl_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    g_greater = compute_config.dsbsparse_type(
        gg_data,
        rows,
        cols,
        block_sizes,
        (gg_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    w_lesser = compute_config.dsbsparse_type(
        wl_data,
        rows,
        cols,
        block_sizes,
        (wl_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    w_greater = compute_config.dsbsparse_type(
        wg_data,
        rows,
        cols,
        block_sizes,
        (wg_data.shape[0],) + tuple([k for k in number_of_kpoints if k > 1]),
    )
    s_lesser = compute_config.dsbsparse_type.zeros_like(g_lesser)
    s_greater = compute_config.dsbsparse_type.zeros_like(g_greater)
    s_retarded = compute_config.dsbsparse_type.zeros_like(g_greater)
    # Initialize the self-energy object
    sigma_coulomb_screening = SigmaCoulombScreening(
        quatrex_config,
        compute_config,
        electron_energies,
    )
    sparsity_pattern = sparse.coo_matrix((xp.ones_like(rows), (rows, cols)))
    sigma_fock = SigmaFock(
        quatrex_config, compute_config, electron_energies, sparsity_pattern
    )
    # Compute the GW self-energy
    sigma_coulomb_screening.compute(
        g_lesser,
        g_greater,
        w_lesser,
        w_greater,
        out=(s_lesser, s_greater, s_retarded),
    )
    # Compute the Fock self-energy
    sigma_fock.compute(
        g_lesser,
        out=(s_retarded,),
    )
    # Compare the results
    # They are shifted in energy (should fix old code?)
    assert xp.allclose(s_lesser.data, sl_data)
    assert xp.allclose(s_greater.data, sg_data)
    assert xp.allclose(s_retarded.data, sr_data)
