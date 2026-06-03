# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

import h5py
import numpy as np
from scipy import sparse


def load_hdf5_dict(filename: str) -> dict:
    """Loads the given HDF5 file and returns its contents as a dictionary.

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

            item = f[key]
            fmt = item.attrs.get("format", None)

            if fmt == "ndarray":
                matrix_dict[key] = item[:]

            elif fmt == "csr":
                shape = item.attrs.get("shape", None)
                if shape is None:
                    raise ValueError(
                        f"Missing 'shape' attribute for CSR matrix '{key}' in HDF5 file."
                    )

                matrix_dict[key] = sparse.csr_matrix(
                    (item["data"][:], item["indices"][:], item["indptr"][:]),
                    shape=shape,
                )

            elif fmt == "coo":
                shape = item.attrs.get("shape", None)
                if shape is None:
                    raise ValueError(
                        f"Missing 'shape' attribute for COO matrix '{key}' in HDF5 file."
                    )

                matrix_dict[key] = sparse.coo_matrix(
                    (item["data"][:], (item["row"][:], item["col"][:])), shape=shape
                )

            elif fmt == "csc":
                shape = item.attrs.get("shape", None)
                if shape is None:
                    raise ValueError(
                        f"Missing 'shape' attribute for CSC matrix '{key}' in HDF5 file."
                    )

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
    with h5py.File(filename, "w") as f:
        for key, mat in data.items():
            if isinstance(mat, sparse.csr_matrix):
                grp = f.create_group(key)
                fmt = mat.format

                grp.attrs["format"] = fmt
                grp.attrs["shape"] = mat.shape

                grp.create_dataset("data", data=mat.data, compression="gzip")
                grp.create_dataset("indices", data=mat.indices, compression="gzip")
                grp.create_dataset("indptr", data=mat.indptr, compression="gzip")

            elif isinstance(mat, sparse.coo_matrix):
                grp = f.create_group(key)
                fmt = mat.format

                grp.attrs["format"] = fmt
                grp.attrs["shape"] = mat.shape

                grp.create_dataset("data", data=mat.data, compression="gzip")
                grp.create_dataset("row", data=mat.row, compression="gzip")
                grp.create_dataset("col", data=mat.col, compression="gzip")

            elif isinstance(mat, sparse.csc_matrix):
                grp = f.create_group(key)
                fmt = mat.format

                grp.attrs["format"] = fmt
                grp.attrs["shape"] = mat.shape

                grp.create_dataset("data", data=mat.data, compression="gzip")
                grp.create_dataset("indices", data=mat.indices, compression="gzip")
                grp.create_dataset("indptr", data=mat.indptr, compression="gzip")

            elif isinstance(mat, np.ndarray):
                dset = f.create_dataset(key, data=mat, compression="gzip")
                dset.attrs["format"] = "ndarray"

            else:
                print(
                    f"Warning: Unsupported type {type(mat)} for key '{key}'. Skipping."
                )
