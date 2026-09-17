# Example Simulation Setups

`quatrex` comes with a set of example simulation setups that can be used
to test the installation and to get familiar with the input parameters.
The examples are located in the `examples` directory of `quatrex`. The
examples are also used as integration tests in the CI/CD pipeline.

The available example simulation setups are listed below. They are
grouped by the type of input files used to create the system Hamiltonian
and overlap matrix. The `cp2k` examples use [CP2K input
files](../input_data/#cp2k), while the examples listed under `w90` use
[Wannier90 input files](../input_data/#plane-wave-dft-wannier90). The
most notable difference between the two is the presence/absence of an
orbital overlap matrix, due to the orthonormality of maximally localized
Wannier functions.

!!! tip "Example `README.md` Files"
    We aim to provide more detailed provenance information for all
    example data sets in `README.md` files that sit at the top level of
    the example directories. These can also help you better understand
    how to acquire input data for `quatrex`.

```bash {title="Available Example Simulation Setups"}
quatrex/examples/
├── cp2k/
│   ├── carbon-chain/  # A simple chain of carbon atoms
│   │   ├── README.md
│   │   ├── inputs/
│   │   ├── phonon/
│   │   ├── qtbm/
│   │   └── qtbm-low-rank/
│   └── graphene/  # A layer of graphene
│       ├── README.md
│       ├── inputs/
│       ├── qtbm/
│       └── qtbm-low-rank/
└── w90/
    ├── carbon-nanotube/  # An (8,0) carbon nanotube
    │   ├── README.md
    │   ├── gw/
    │   ├── gw-dist/
    │   ├── gw-unit-cell/
    │   ├── inputs/
    │   └── qtbm/
    ├── mos2/  # A monolayer of MoS2
    │   ├── README.md
    │   ├── gw-kpoints/
    │   ├── gw-kpoints-symmetric/
    │   └── inputs/
    └── si-bulk/  # Bulk crystalline silicon
        ├── README.md
        ├── inputs/
        ├── qtbm/
        └── qtbm-low-rank/
```
