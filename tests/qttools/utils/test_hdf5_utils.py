# Copyright (c) 2024-2026 ETH Zurich and the authors of the qttools package.

from pathlib import Path

import scipy.sparse as sps

from qttools import xp
from qttools.utils.hdf5_utils import load_hdf5_dict, save_hdf5_dict


def test_save_load_h5(tmp_path: Path):
    """Test the hdf5 save and load functions."""

    dict = {
        "[0,0,0]": xp.random.rand(10, 10),
        "[1,0,0]": sps.random(10, 10, density=0.5, format="csr"),
        "[0,1,0]": sps.random(10, 10, density=0.5, format="coo"),
        "[0,0,1]": sps.random(10, 10, density=0.5, format="csc"),
    }
    save_hdf5_dict(tmp_path / "dict.h5", dict)

    loaded_dict = load_hdf5_dict(tmp_path / "dict.h5")

    for key in dict.keys():
        if isinstance(dict[key], sps.spmatrix):
            assert xp.allclose(dict[key].toarray(), loaded_dict[key].toarray())
        else:
            assert xp.allclose(dict[key], loaded_dict[key])
