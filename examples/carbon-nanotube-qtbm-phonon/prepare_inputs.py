# Copyright (c) 2024 ETH Zurich and the authors of the quatrex package.


import sys
from pathlib import Path

import numpy as np

from gpaw import GPAW

from ase.phonons import Phonons 
import ase.units as units
from ase.io import read, write
from ase.build import make_supercell
from ase.build.tools import sort

import numpy as np
from pathlib import Path
import scipy.sparse as sps


def create_phonon_device_dynmat(ph:Phonons, num_cells: int, outfile_path: Path):    
    
    # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV / m^2    
    s = (units._hbar * 1e10 / np.sqrt(units._e * units._amu))**2

    D_N = ph.D_N * s

    num_atom = D_N.shape[-1] // 3

    D = np.zeros((num_atom*3*num_cells, num_atom*3*num_cells), dtype=np.complex128)
    for i in range(num_cells):
        for j in range(num_cells):
            if abs(i-j) > D_N.shape[0]//2:
                continue
            D[i*num_atom*3:(i+1)*num_atom*3, j*num_atom*3:(j+1)*num_atom*3] = (D_N[i-j, :, :] + D_N[j-i, :, :].T.conj())/2.0


    H_coo = sps.coo_array(D)
    sps.save_npz(outfile_path / "hamiltonian_0_0_0.npz", H_coo)

def create_device_lattice(atoms, num_cells, outfile_path: Path):

    P = [[1,0,0],[0,1,0],[0,0,num_cells]]
    device = make_supercell(atoms, P)
    device = sort(device,tags=device.positions[:,2])

    write(outfile_path / "lattice.xyz", device)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    num_cells = int(sys.argv[2])
    supercell = tuple([int(nk) for nk in sys.argv[3:6]])

    calculator_params = {
        "xc": "PBE",
        "basis": "dzp",
        "mode": {"name": "lcao"},
        }

    atoms = read('eq.traj')

    calc = GPAW(**calculator_params)
    calc.set(symmetry='off')

    ph = Phonons(atoms, calc, supercell=supercell)

    ph.read()

    create_phonon_device_dynmat(ph, num_cells, path)
    create_device_lattice(atoms, num_cells, path)
