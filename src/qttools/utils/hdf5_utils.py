# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import re

import h5py
import numpy as np
from scipy import sparse


def load_hdf5_dict(filename: str) -> dict:
    """Loads the given HDF5 file and returns its contents as a dictionary.
    The dictionary values can be either numpy arrays or scipy sparse matrices in CSR,
    COO, or CSC format. The dictionary keys should be strings in the format "[x,y,z]",
    where x, y, and z are the coordinates of the corresponding matrix in the hopping grid.

    Parameters
    ----------
    filename : str
        The path to the HDF5 file to load.

    Returns
    -------
    dict
        A dictionary containing the contents of the HDF5 file.

    """
    with h5py.File(filename, "r") as f:

        matrix_dict = {}

        for key in f.keys():

            # Validate the key format (should be a string in the format "[x,y,z]"")
            if not re.fullmatch(r"\[[^,\[\]]+,[^,\[\]]+,[^,\[\]]+\]", key):
                raise ValueError(f"Key '{key}' must be in the format '[x,y,z]'.")

            # Load the item and determine its format
            item = f[key]
            fmt = item.attrs.get("format", None)

            if fmt == "ndarray":
                matrix_dict[key] = item[:]

            elif fmt == "csr" or fmt == "csc" or fmt == "coo":

                # If the format is a sparse matrix, we load the shape attribue
                shape = item.attrs.get("shape", None)

                if shape is None:
                    raise ValueError(
                        f"Missing 'shape' attribute for {fmt} matrix '{key}' in HDF5 file."
                    )

                # Depending on the format, the correct sparse matrix is constructed
                # from the corresponding datasets
                if fmt == "csr":
                    matrix_dict[key] = sparse.csr_matrix(
                        (item["data"][:], item["indices"][:], item["indptr"][:]),
                        shape=shape,
                    )
                elif fmt == "coo":
                    matrix_dict[key] = sparse.coo_matrix(
                        (item["data"][:], (item["row"][:], item["col"][:])), shape=shape
                    )
                elif fmt == "csc":
                    matrix_dict[key] = sparse.csc_matrix(
                        (item["data"][:], item["indices"][:], item["indptr"][:]),
                        shape=shape,
                    )

            else:
                raise ValueError(
                    f"Unsupported format '{fmt}' for item '{key}' in HDF5 file."
                )

    return matrix_dict


def save_hdf5_dict(filename: str, data: dict):
    """Saves a dictionary to an HDF5 file.
    The dictionary values can be either numpy arrays or scipy sparse matrices in CSR, COO, or CSC format.
    The dictionary keys should be strings in the format "[x,y,z]", where x, y, and z are the coordinates
    of the corresponding matrix in the hopping grid.

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
