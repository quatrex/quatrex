# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

"""Includes utility functions for HDF5 operations."""

import re
from typing import Literal

import h5py
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from scipy import sparse

from qttools import NDArray
from qttools.utils.mpi_utils import get_section_sizes


def _load_local_nnz(item: h5py.Dataset, comm: MPI.Intracomm = comm):
    """Loads the local non-zero entries of a COO matrix from an HDF5 dataset.

    Parameters
    ----------
    item : h5py.Dataset
        The HDF5 dataset containing the COO matrix data.
    comm : MPI.Intracomm, optional
        The MPI communicator to use for loading the data. Defaults to
        COMM_WORLD.

    Returns
    -------
    local_rows : NDArray
        The row indices loaded on the current rank.
    local_cols : NDArray
        The column indices loaded on the current rank.
    local_data : NDArray
        The data values loaded on the current rank.

    """
    nnz = item["data"].shape[0]
    sizes, __ = get_section_sizes(nnz, comm.size)
    offsets = np.hstack((0, np.cumsum(sizes, dtype=np.int32)))

    local_rows = item["row"][offsets[comm.rank] : offsets[comm.rank + 1]]
    local_cols = item["col"][offsets[comm.rank] : offsets[comm.rank + 1]]
    local_data = item["data"][offsets[comm.rank] : offsets[comm.rank + 1]]

    return local_rows, local_cols, local_data


def _reshuffle_domain_partitions(
    shape: tuple[int, int],
    local_rows: NDArray,
    local_cols: NDArray,
    local_data: NDArray,
    comm: MPI.Intracomm = comm,
    partitioning_scheme: Literal["row", "fishtail"] = "fishtail",
):
    """Reshuffles locally stored COO matrix data.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the global matrix.
    local_rows : NDArray
        The local row indices of the COO matrix.
    local_cols : NDArray
        The local column indices of the COO matrix.
    local_data : NDArray
        The local data of the COO matrix.
    comm : MPI.Intracomm, optional
        The MPI communicator to use for reshuffling the data. Defaults
        to COMM_WORLD.
    partitioning_scheme : Literal["row", "fishtail"], optional
        The partitioning scheme to use for reshuffling the data. Can be
        "row", "fishtail", or None. The row partitioning scheme assigns
        contiguous rows to each rank, while the fishtail partitioning
        scheme assigns rows and columns to each rank in a fishtail
        pattern. The default is "fishtail".

    """
    # Determine where the data actually needs to go based on the
    # partitioning scheme.
    sizes, __ = get_section_sizes(shape[0], comm.size)
    offsets = np.hstack((0, np.cumsum(sizes, dtype=np.int32)))

    row_owner = np.searchsorted(offsets, local_rows, side="right") - 1

    if partitioning_scheme == "row":
        dest = row_owner
    elif partitioning_scheme == "fishtail":
        col_owner = np.searchsorted(offsets, local_cols, side="right") - 1
        dest = np.minimum(row_owner, col_owner).astype(np.int32)

    order = np.argsort(dest, kind="stable")

    dest = dest[order]
    local_rows = local_rows[order]
    local_cols = local_cols[order]
    local_data = local_data[order]

    # NOTE: Another option would be to pad the buffers to the maximum
    # size and then use Alltoall instead. Since the number of non-zeros
    # per rank can vary by more than just a few elements (think about
    # first and last partitions), this is probably not really worth it.
    send_counts = np.bincount(dest, minlength=comm.size).astype(np.int32)
    recv_counts = np.empty(comm.size, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)

    send_displs = np.zeros(comm.size, dtype=np.int32)
    np.cumsum(send_counts[:-1], out=send_displs[1:])
    recv_displs = np.zeros(comm.size, dtype=np.int32)
    np.cumsum(recv_counts[:-1], out=recv_displs[1:])

    total_recv = int(recv_counts.sum())

    rows = np.empty((total_recv,), dtype=local_rows.dtype)
    cols = np.empty((total_recv,), dtype=local_cols.dtype)
    data = np.empty((total_recv,), dtype=local_data.dtype)

    comm.Alltoallv(
        [local_rows, (send_counts, send_displs)],
        [rows, (recv_counts, recv_displs)],
    )
    comm.Alltoallv(
        [local_cols, (send_counts, send_displs)],
        [cols, (recv_counts, recv_displs)],
    )
    comm.Alltoallv(
        [local_data, (send_counts, send_displs)],
        [data, (recv_counts, recv_displs)],
    )

    return rows, cols, data


def load_hdf5_dict(
    filename: str,
    comm: MPI.Intracomm | None = comm,
    partitioning_scheme: str = "fishtail",
) -> dict:
    """Loads the given HDF5 file and returns its contents as a dictionary.

    The dictionary values can be either numpy arrays or scipy sparse
    matrices in CSR, COO, or CSC format. The dictionary keys should be
    strings in the format "[x,y,z]", where x, y, and z are the
    coordinates of the corresponding matrix in the hopping grid.

    Parameters
    ----------
    filename : str
        The path to the HDF5 file to load.
    comm : MPI.Intracomm, optional
        The MPI communicator to use for loading the data. Defaults to
        COMM_WORLD.
    partitioning_scheme : str, optional
        The partitioning scheme to use for reshuffling the data. Can be
        "row" or "fishtail". The default is "fishtail".

    Returns
    -------
    dict
        A dictionary containing the contents of the HDF5 file.


    """
    distributed = comm is not None and comm.size > 1
    if distributed and partitioning_scheme not in ["row", "fishtail"]:
        raise ValueError(
            f"Invalid domain partitioning scheme: {partitioning_scheme}. "
            f"Must be 'row' or 'fishtail'."
        )

    kwargs = {"driver": "mpio", "comm": comm} if distributed else {}

    with h5py.File(filename, "r", **kwargs) as f:

        matrix_dict = {}

        # Iterate over lattice indices and corresponding operator
        # hopping blocks.
        for r, o_r in f.items():

            # Validate the key format (should be a string in the format "[x,y,z]"")
            if not re.fullmatch(r"\[[^,\[\]]+,[^,\[\]]+,[^,\[\]]+\]", r):
                raise ValueError(f"Key '{r}' must be in the format '[x,y,z]'.")

            r = tuple(map(int, r.strip("[]").split(",")))

            # Load the item and determine its format
            fmt = o_r.attrs.get("format")

            if fmt not in ["ndarray", "csr", "csc", "coo"]:
                raise ValueError(
                    f"Unsupported format '{fmt}' for item '{r}' in HDF5 file."
                )

            if distributed and fmt != "coo":
                raise ValueError(
                    f"Distributed loading is only supported for COO matrices, "
                    f"but item '{r}' has format '{fmt}'."
                )

            if fmt == "ndarray":
                matrix_dict[r] = o_r[:]
                continue

            # Must be a sparse matrix in this case, so we check for the
            # required datasets and attributes.
            shape = o_r.attrs.get("shape")

            if shape is None:
                raise ValueError(
                    f"Missing 'shape' attribute for {fmt} matrix '{r}' in HDF5 file."
                )

            # Depending on the format, the correct sparse matrix is
            # constructed from the corresponding datasets.
            if fmt == "coo":
                if not distributed:
                    rows, cols, data = (
                        o_r["row"][:],
                        o_r["col"][:],
                        o_r["data"][:],
                    )
                else:
                    local_rows, local_cols, local_data = _load_local_nnz(o_r, comm=comm)
                    rows, cols, data = _reshuffle_domain_partitions(
                        shape=shape,
                        local_rows=local_rows,
                        local_cols=local_cols,
                        local_data=local_data,
                        comm=comm,
                        partitioning_scheme=partitioning_scheme,
                    )
                matrix_dict[r] = sparse.coo_matrix((data, (rows, cols)), shape=shape)

            elif fmt == "csr":
                matrix_dict[r] = sparse.csr_matrix(
                    (
                        o_r["data"][:],
                        o_r["indices"][:],
                        o_r["indptr"][:],
                    ),
                    shape=shape,
                )
            elif fmt == "csc":
                matrix_dict[r] = sparse.csc_matrix(
                    (
                        o_r["data"][:],
                        o_r["indices"][:],
                        o_r["indptr"][:],
                    ),
                    shape=shape,
                )

    return matrix_dict


def save_hdf5_dict(filename: str, data: dict):
    """Saves a dictionary to an HDF5 file.

    The dictionary values can be either numpy arrays or scipy sparse
    matrices in CSR, COO, or CSC format. The dictionary keys should be
    strings in the format "[x,y,z]", where x, y, and z are the
    coordinates of the corresponding matrix in the hopping grid.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file to save the dictionary to.
    data : dict
        The dictionary to save.

    Returns
    -------
    None

    """

    # Validate if the keys and values in the dictionary are of the correct type and format
    for key, mat in data.items():

        if not isinstance(key, str):
            raise TypeError(
                f"Keys in the dictionary must be strings, got {type(key)} for key '{key}'."
            )

        if not re.fullmatch(r"\[[^,\[\]]+,[^,\[\]]+,[^,\[\]]+\]", key):
            raise ValueError(f"Key '{key}' must be in the format [x,y,z].")

        if not isinstance(
            mat, (sparse.csr_matrix, sparse.coo_matrix, sparse.csc_matrix, np.ndarray)
        ):
            raise TypeError(
                f"Unsupported data type {type(mat)} for key '{key}'. "
                f"Supported types are: scipy.sparse.csr_matrix, scipy.sparse.coo_matrix, "
                f"scipy.sparse.csc_matrix, and numpy.ndarray."
            )

    # Save the dictionary to the HDF5 file, storing the format and shape information as attributes
    with h5py.File(filename, "w") as f:
        for key, mat in data.items():

            if isinstance(
                mat, (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix)
            ):
                # If the matrix is a sparse matrix, a group is created and the format and shape informations
                # are stored as attributes
                grp = f.create_group(key)
                grp.attrs["format"] = mat.format
                grp.attrs["shape"] = mat.shape

                # Depending on the format, the correct datasets are created for the sparse matrix data
                if isinstance(mat, (sparse.csr_matrix, sparse.csc_matrix)):
                    grp.create_dataset("data", data=mat.data, compression="gzip")
                    grp.create_dataset("indices", data=mat.indices, compression="gzip")
                    grp.create_dataset("indptr", data=mat.indptr, compression="gzip")
                else:
                    grp.create_dataset("data", data=mat.data, compression="gzip")
                    grp.create_dataset("row", data=mat.row, compression="gzip")
                    grp.create_dataset("col", data=mat.col, compression="gzip")

            elif isinstance(mat, np.ndarray):
                # If the matrix is a dense numpy array, it is directly stored as a
                # dataset with the format information as an attribute
                dset = f.create_dataset(key, data=mat, compression="gzip")
                dset.attrs["format"] = "ndarray"

            else:
                raise TypeError(
                    f"Unsupported data type {type(mat)} for key '{key}'. "
                    f"Supported types are: scipy.sparse.csr_matrix, scipy.sparse.coo_matrix, "
                    f"scipy.sparse.csc_matrix, and numpy.ndarray."
                )
