from qttools.kernels.datastructure.dsdbsparse_kernels import ( find_ranks )
from qttools import NDArray, xp


def find_overlaping_data(nnz_section_offsets, num_diag: int, rows: NDArray, cols: NDArray):
    """
    Figure out the overlaping data on other ranks to gether from based on an interaction range defined by "num_diag"
    """
    num_rank = len(nnz_section_offsets) - 1
    nnz_to_gether = []
    nnz_rank = []
    
    for rank in range(num_rank):

        min_row = rows[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].min() - num_diag
        max_row = rows[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].max() + num_diag
        min_col = cols[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].min() - num_diag
        max_col = cols[nnz_section_offsets[rank] : nnz_section_offsets[rank + 1]].max() + num_diag

        mask = (
            (rows >= min_row)
            & (rows <= max_row)
            & (cols >= min_col)
            & (cols <= max_col)
        )

        ids = xp.where(mask)[0]
        ids_in_rank = find_ranks(nnz_section_offsets, ids) 

        get_nnz = xp.where(ids_in_rank != rank)[0]

        nnz_to_gether.append(ids[get_nnz])
        nnz_rank.append(ids_in_rank[get_nnz])        

    return nnz_to_gether, nnz_rank
