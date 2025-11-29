# Running Simluations

After [installing](installation.md) the package and [setting up simluations](setup_simulations.md), you can run a simple first simulation.

Navigate to the `examples/carbon-nanotube/` folder. Quatrex looks for a `quatrex_config.toml` to use as the input settings. For advanced users, you may include an optional `compute_config.toml` which controls low-level details of the distributed computing. For a first run, the default configuration file can be used. This will compute 1 iteration of the self-consistent Born approximation and exit. 

!!! Note
    There is currently no convergence criteria for stopping. The simulation stops only when it reaches the maximum number of iterations, as user-defined in the configuration file `quatrex_config.toml`. The last line of the output will read `SCBA did not converge after 1 iterations`. This is expected behaviour.

To launch a single-threaded simulation, use:
```bash
quatrex run > out.txt
```

It will save output files in the `outputs/` folder. The stdout is redirected to `out.txt`. This is necessary since the post-processing scripts parse the `out.txt` when visualizing the outputs and profiling the run.

To run a multi-threaded simluation, use the `mpiexec` command. For example, to run an 8-threaded simluation:
```bash
mpiexec -n 8 quatrex run > out.txt`
```

If you simply want to run with all of your machine's available resources, you can do:

````bash
mpiexec -n `nproc` quatrex run > out.txt
````
The `nproc` utility prints the number of threads available to use. If you're using all threads, it's a good idea to have `htop` open to monitor memory usage and ensure the simulation is not running out of RAM.