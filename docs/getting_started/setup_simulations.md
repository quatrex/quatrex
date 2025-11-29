# Setting up Simluations 

Quatrex requires a few inputs to run.

## 1. `quatrex_config.toml`
This a configuration file that specifies simulation parameters. It is already included in the repo and ready to use.


## 2. pre-computed and assembled Hamiltonian, Coulomb, and potential matrices
To obtain the second set of inputs, you can do:
```bash
quatrex fetch-example carbon-nanotube:
```
This downloads the Hamiltonian matrix, Coulomb matrix, and potential matrix and puts them into the `examples/carbon-nanotube/inputs` folder. These are are pre-computed using a combination of DFT simluations from [VASP](https://vasp.at/) and MLWF calculations from [Wannier90](https://wannier.org/) (see Fig 1 from [^1]). It also downloads some supplementary files used to define the shared/distributed memory usage.

[^1]: L. Deuschle et al., “Electron-electron interactions in device simulation via nonequilibrium Green’s functions and the GW approximation,” Phys. Rev. B, vol. 111, no. 19, p. 195421, May 2025, doi: 10.1103/PhysRevB.111.195421.

The currently available examples are `carbon-nanotube:` and `carbon-nanotube:dist`. For most people, the regular `carbon-nanotube:` example is sufficient. Note the inclusion of the colon `:` in the name. 

## 3. (optional for advanced users) `compute_config.toml`
The 3rd input `compute_config.toml` is only needed when running Quatrex on many distributed machines. Contact the authors on GitHub directly for help.