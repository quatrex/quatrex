# Electronic Structure Data

To compute *non-equilibrium* electronic transport properties with
`quatrex`, we need information about the underlying electronic structure
of the system at *equilibrium*, i.e., primarily the Hamiltonian and
orbital overlap matrices. For a given system, these matrices can be
obtained via empirical tight-binding models or from first-principles
calculations, such as density functional theory (DFT).

When running a simulation, `quatrex` will then read the provided input
Hamiltonian and overlap matrices in a tight-binding-like format
from [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files. This data
is essentially a dictionary of hopping terms, where each term is a
matrix containing the elements connecting orbitals in the respective
image cell with the home cell.

<!-- Include a picture here -->

```python
hamiltonian = {
    (0, 0, 0): h_000,  # Matrix elements within the home cell
    (1, 0, 0): h_100,  # Connections to the neighboring cell in +x direction
    (0, 1, 0): h_010,  # Connections to the neighboring cell in +y direction
    # ...
}
```

The individual hopping matrices (`h_000`, `h_100`, `h_010`, ...) are
square `numpy` arrays or `scipy.sparse` matrices, of size $N \times N$,
where $N$ is the number of orbitals in the home cell.

!!! note "Units of Electronic Structure Data"
    The Hamiltonian matrix elements are expected to be given in eV by
    `quatrex`. All distances are measured in Å.

Depending on the simulation configuration, the hopping terms are then
used to construct device Hamiltonians (see
[`construct_from_unit_cell`](parameters/device/#construct_from_unit_cell))
and/or to sample the Brillouin zone in transverse directions (see
[`kpoint_grid`](parameters/device/#kpoint_grid)).

## Interfacing with DFT Codes

!!! note "Unified Electronic Structure Interface"
    We are working on a unified interface for different DFT codes, which
    will allow users to more easily extract the necessary Hamiltonian
    and overlap matrices from their DFT calculations and convert them
    into the required format for `quatrex`.

Since `quatrex` employs a "frozen" Hamiltonian approximation and does
not update exchange-correlation potentials self-consistently with the
non-equilibrium charge carrier distribution, the interface with DFT
codes reduces to just providing the initial Kohn-Sham Hamiltonian and
orbital overlap matrices.

Here we provide some information on how to extract the necessary data
from a few different DFT codes. Once you have transformed the DFT
Hamiltonian and overlap matrices into the format described above, you
can use the
[`save_hdf5_dict`](../api/qttools/utils/hdf5_utils/#qttools.utils.hdf5_utils.save_hdf5_dict)
function to save them in HDF5 format for use with `quatrex`.

!!! info "Localized Orbital Basis Requirement"
    Because any simulated system's periodicity will broken along
    transport direction, `quatrex` simulations require electronic
    structure data to be represented in localized orbital basis.

### Plane-Wave DFT & Wannier90

To extract the necessary Hamiltonian and overlap matrices from
plane-wave DFT codes, one can use the [Wannier90](https://wannier.org/)
package to project the electronic structure onto a basis of
maximally-localized Wannier functions. Besides the enforced
orthonormality, this gives one the added benefit of being able to select
only a submanifold of the full electronic structure (e.g. around the
Fermi level), which can significantly reduce the size of the Hamiltonian
and overlap matrices, and thus the computational cost of the transport
simulation.

Wannier90 outputs the Hamiltonian when the [`write_hr`
option](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/parameters/#logical-write_hr)
is enabled in the `<seedname>.win` input file. Information about the
format of Wannier90's Hamiltonian output (`seedname_hr.dat`) can be
found in the [Wannier90
documentation](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/#seedname_hrdat).

??? example "Reading Wannier90 Hamiltonians in Python"
    Wannier90 Hamiltonians can be converted into the `quatrex` HDF5
    format using the following code snippet

    ```python
    from pathlib import Path

    import numpy as np

    from qttools.utils.hdf5_utils import save_hdf5_dict


    def read_wannier_hr(path: Path, dtype=np.complex128) -> dict:
        """Read the contents of a Wannier90 `hr.dat` file.

        Parameters
        ----------
        path : Path
            Path to a `hr.dat` file.
        dtype : optional
            The data type for the Hamiltonian matrix elements. Defaults to
            `numpy.complex128`.

        Returns
        -------
        hamiltonian : dict
            A dictionary containing the Hamiltonian hopping matrices. The
            keys are the hopping cells, and the values are the Hamiltonian
            hopping matrices.

        """

        # Strip info from header.
        num_wannier_functions, num_cells = np.loadtxt(
            path, skiprows=1, max_rows=2, dtype=int
        )

        # Read wannier data (skipping degeneracy info). Wannier90 uses some
        # Fortran formatting. Thus, the degeneracy information is arranged
        # into 15 values per line.
        degenerate_rows = int(np.ceil(num_cells / 15.0))
        wannier_data = np.loadtxt(path, skiprows=3 + degenerate_rows)

        cells = wannier_data[:, :3].astype(int)

        # Initialize Hamiltonian dictionary. The keys are the hopping cells,
        # and the values are the Hamiltonian hopping matrices.
        hamiltonian = {
            f"[{rx},{ry},{rz}]": np.zeros(
                (num_wannier_functions, num_wannier_functions), dtype=dtype
            )
            for rx, ry, rz in np.unique(cells, axis=0)
        }

        # Read the Hamiltonian matrix elements.
        for line in wannier_data:
            rx, ry, rz = line[:3].astype(int)
            key = f"[{rx},{ry},{rz}]"
            i, j = line[3:5].astype(int)
            h_ij_real, h_ij_imag = line[5:]
            hamiltonian[key][i - 1, j - 1] = h_ij_real + 1j * h_ij_imag

        return hamiltonian


    hamiltonian = read_wannier_hr(
        path=Path("wannier90_hr.dat"),
        dtype=np.complex128,
    )

    save_hdf5_dict("hamiltonian.h5", hamiltonian)
    ```

### CP2K

In [CP2K](https://cp2k.org/) input files, you can enable the output of
the Hamiltonian and overlap matrices in CSR format by adding the
following block to the `&PRINT` subsection of your `&FORCE_EVAL/&DFT`
settings:

```toml {title="Printing Hamiltonian and Overlap Matrices with CP2K"}
&DFT
    # ...
    &PRINT
        # ...
        &KS_CSR_WRITE
            # ...
            # Do not write matrix elements smaller than 1e-8.
            THRESHOLD 1e-8
            # Only write the upper triangular part of the matrix.
            UPPER_TRIANGULAR
            # Write the matrix in binary format.
            BINARY
            # Write in real-space (tight-binding-like) representation.
            REAL_SPACE
        &END KS_CSR_WRITE
        &S_CSR_WRITE
            # ...
            THRESHOLD 1e-8
            UPPER_TRIANGULAR
            BINARY
            REAL_SPACE
        &END S_CSR_WRITE
    &END
&END DFT
```

!!! warning "Structure geometry and `REAL_SPACE` option"
    While in $\Gamma$-only calculations, the `REAL_SPACE` option can be
    omitted, it is required in runs with a non-trivial k-point grid.
    When using this option, make sure that the atomic structure is
    entirely contained within the unit cell, i.e., by CP2K's convention,
    coordinates of all atoms must be in the fractional range $[-0.5,
    0.5)$ in each direction.

??? example "Reading CP2K Hamiltonian and Overlap Matrices in Python"
    CP2K will output the Hamiltonian and overlap matrices in `.csr`
    binary files. Despite what the name would suggest, these files are
    actually in COO format. They can be read using `numpy` and converted
    to `scipy.sparse` matrices using the following code snippet:

    ```python
    import numpy as np
    from scipy import sparse

    dtype = np.dtype(
        [
            ("leading_padding", np.int32),
            ("rows", np.int32),
            ("cols", np.int32),
            ("data", np.float64),
            ("trailing_padding", np.int32),
        ]
    )
    with open("<file>.csr", "rb") as f:
        matrix_data = np.frombuffer(f.read(), dtype=dtype)

    matrix = sparse.coo_matrix(
        # Subtract one to convert from 1-based to 0-based indexing.
        (matrix_data["data"], (matrix_data["rows"] - 1, matrix_data["cols"] - 1))
    )
    ```

You can check CP2K's documentation for more information about the
[`KS_CSR_WRITE`](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/PRINT/KS_CSR_WRITE.html#ks-csr-write)
section.

### GPAW

We can also extract the Hamiltonian and overlap matrices from
[GPAW](https://gpaw.readthedocs.io/) calculations performed in the
[`lcao`
mode](https://gpaw.readthedocs.io/documentation/lcao/lcao.html#lcao).
For this, one can use the `get_lcao_hamiltonian` method and the
`TightBinding` class from the `gpaw.lcao` package. Some aspects of this
can be found in GPAW's documentation about [electron
transport](https://gpaw.readthedocs.io/tutorialsexercises/electronic/transport/transport.html#electron-transport).

??? example "Extracting GPAW Hamiltonian and Overlap Matrices"
    After running a GPAW calculation in `lcao` mode and saving the
    result to `device.gpw`, the Hamiltonian, overlap matrix, and the
    cell indices can be extracted from the calculator using the
    following code snippet:

    ```python
    import numpy as np
    from gpaw import GPAW, lcao

    device_calc = GPAW("device.gpw")
    hamiltonian_sk, overlap_k = lcao.tools.get_lcao_hamiltonian(calc=device_calc)
    # Assuming we do not care about spin.
    hamiltonian_k = hamiltonian_sk[0]

    tb = lcao.tightbinding.TightBinding(device_calc.atoms, device_calc)

    hamiltonian_r = tb.bloch_to_real_space(hamiltonian_k)
    overlap_r = tb.bloch_to_real_space(overlap_k)
    rs = tb.lattice_vectors()
    ```

    They can then be converted to the `quatrex` format as follows:

    ```python
    from qttools.utils.hdf5_utils import save_hdf5_dict

    hamiltonian = {f"[{rx},{ry},{rz}]": h_r for (rx, ry, rz), h_r in zip(rs, hamiltonian_r)}
    overlap = {f"[{rx},{ry},{rz}]": s_r for (rx, ry, rz), s_r in zip(rs, overlap_r)}

    save_hdf5_dict("hamiltonian.h5", hamiltonian)
    save_hdf5_dict("overlap.h5", overlap)
    ```

### Siesta

The [Siesta](https://siesta-project.org/) code also allows extracting
Hamiltonian and overlap matrices conveniently via
[`sisl`](https://sisl.readthedocs.io/en/latest/api/io/siesta.html),
which can read the `.HSX` output files from Siesta calculations.

??? example "Extracting Siesta Hamiltonian and Overlap Matrices"
    After running a Siesta calculation, the Hamiltonian, overlap matrix,
    and cell vectors can be extracted from the `.HSX` output file using
    the following code snippet:

    ```python
    import sisl

    from qttools.utils.hdf5_utils import save_hdf5_dict

    sile = sisl.get_sile("siesta.HSX")
    hamiltonian = sile.read_hamiltonian()

    rs = hamiltonian.lattice.sc_off

    hamiltonian_csr = hamiltonian.tocsr()
    num_orbitals = hamiltonian_csr.shape[0]

    # The csr matrices are just stacked together along axis 1, so we can
    # split them into a dictionary of hopping matrices.
    hamiltonian_r = {
        f"[{rx},{ry},{rz}]": hamiltonian_csr[:, i * num_orbitals : (i + 1) * num_orbitals]
        for i, (rx, ry, rz) in enumerate(rs)
    }

    overlap_csr = sile.read_overlap().tocsr()
    overlap_r = {
        f"[{rx},{ry},{rz}]": overlap_csr[:, i * num_orbitals : (i + 1) * num_orbitals]
        for i, r in enumerate(rs)
    }

    save_hdf5_dict("hamiltonian.h5", hamiltonian_r)
    save_hdf5_dict("overlap.h5", overlap_r)
    ```
