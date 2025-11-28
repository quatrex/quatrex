After [installing](installation.md) the package, you can run a simple
simulation using a provided example setups.

!!! warning "Page under construction"

    These pages are under construction and will be updated soon.

To run a simulation, navigate to the `examples/carbon-nanotube/` folder. Quatrex looks for a `quatrex_config.toml` to use as the input settings. For advanced users, you may include an optional `compute_config.toml` which controls low-level details of the distributed computing.

To launch a single-threaded simulation, simply run:
```bash
quatrex run
```

It will produce outputs in the `outputs/` folder. The stdout is redirected to `out.txt` which is required by the post-processing scripts. 

To run a multi-threaded simluation, use the `mpiexec` command: 
```bash
mpiexec -n <NUM_OF_THREADS> quatrex run`
```
