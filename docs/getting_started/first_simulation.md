# Setting up and running a first simulation

After [installing `quatrex`](installation.md), you can run a simple
simulation using one of the provided example setups. The examples are
located in the `examples` directory of the `quatrex` source code.

<!-- TODO: Include an image of the example setup -->
![image](https://dummyimage.com/600x400/222/eee&text=placeholder)

As a first toy example, you can consider a QTBM calculation for a ~25 nm
long chain of carbon atoms, which you can find under

```bash
./examples/cp2k/carbon-chain/qtbm
```

The electronic structure was computed using
[CP2K](https://www.cp2k.org/) with the DZVP-MOLOPT basis set and the PBE
exchange-correlation functional. The resulting device structure
(`structure.xyz`), Hamiltonian (`hamiltonian.h5`), and overlap matrix
(`overlap.h5`) are stored in the `./examples/cp2k/carbon-chain/inputs`
directory.

!!! info "Input electronic structure data"
    The input files for `quatrex` can be constructed from different
    electronic structure codes, such as CP2K, VASP, Siesta, and GPAW.
    Some preliminary procedures for constructing these input files are
    described in the [input data section](../user_guide/input_data.md).

Besides electronic structure data, `quatrex` requires a configuration
file in [TOML format](https://toml.io/en/) that specifies the simulation
parameters. The configuration file for this example looks as follows:

```toml {title="./examples/cp2k/carbon-chain/qtbm/quatrex_config.toml"}
--8<-- "examples/cp2k/carbon-chain/qtbm/quatrex_config.toml"
```

More information about each individual configuration parameter can be
found in the [simulation parameters
section](../user_guide/simulation_parameters.md) of the user guide.

You can run this example simulation using the following command:

```bash
quatrex run ./examples/cp2k/carbon-chain/qtbm/quatrex_config.toml
```

or even parallelize over energies using multiple MPI processes:

```bash
mpirun -n 4 quatrex run ./examples/cp2k/carbon-chain/qtbm/quatrex_config.toml
```

After the simulation completes, the results are stored as `numpy` arrays
in the `./examples/cp2k/carbon-chain/qtbm/outputs` directory.

```bash
./examples/cp2k/carbon-chain/qtbm
├── quatrex_config.toml
├── quatrex_times.out  # Timing information for the simulation
└── outputs
    ├── current_lr.npy  # Current from left to right lead
    ├── current_rl.npy  # Current from right to left lead
    ├── dos_l.npy  # Orbital-resolved DOS from the left lead
    ├── dos_r.npy  # Orbital-resolved DOS from the right lead
    ├── transmission_lr.npy  # Transmission from left to right lead
    └── transmission_rl.npy  # Transmission from right to left lead

```
