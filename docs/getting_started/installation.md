# Installation

`quatrex` can be installed using any Python package manager, but we can
generally recommend using project-based installations via
[`pixi`](https://pixi.sh/) or [`uv`](https://docs.astral.sh/uv/).


## Obtaining the source code

We currently do not ship any pre-built binaries for `quatrex`, so the
first step in installing `quatrex` is always to clone the repository.

```bash
git clone git@github.com:quatrex/quatrex.git
cd quatrex
```

We provide several example configurations, input files, and reference
outputs that we use for testing and development. You can find them in
the `examples/` directory.

We track this data using [Git LFS](https://git-lfs.com/), so make sure
to install Git LFS and pull the files after cloning the repository:

```bash
git lfs install
git lfs pull
```

## Installation using `pixi`

With [`pixi`](https://pixi.sh/), you can easily manage multiple
environments for a project, e.g. for testing, building documentation or
general development in a reproducible environment. Besides installation
for just running `quatrex`, it is also the recommended approach for
setting up local development environments.

After obtaining the source code you can get a default environment and
install `quatrex` (in editable mode) and its dependencies with:

```bash
pixi install --environment=default
```

Other available environments are:

- **`dev`**: Includes tools for development, testing, and linting, such as
  `pytest`, `ruff`, and `pre-commit`.
- **`docs`**: Includes tools for building documentation, namely `zensical`,
  `mkdocstrings-python`, `griffe`, and `tabulate`.
- **`gpu`**: Includes the conda-forge version of `cupy` for basic GPU
  support.
- **`hpc`**: Includes the default dependencies for running `quatrex` on HPC
  systems. Specifically, `cupy` and `mpi4py` are installed from source
  via PyPI to leverage the system's MPI and GPU backends. For more
  information, see the [HPC installation
  section](#installation-on-hpc-systems).

## Installation using `uv`

[`uv`](https://docs.astral.sh/uv/) is a general-purpose Python
environment and dependency manager.

After obtaining the source code you can create a virtual environment and
install `quatrex` (in editable mode) and its dependencies with:

```bash
uv venv --python >=3.13
source .venv/bin/activate
uv pip install --editable .
```

The following optional dependencies for `quatrex` can be installed
with `uv`:

```bash
uv pip install --editable .[<dev|docs|gpu>]
```

- **`dev`**: Includes tools for development, testing, and linting, such
  as `pytest`, `ruff`, and `pre-commit`.
- **`docs`**: Includes tools for building documentation, namely
  `zensical`, `mkdocstrings-python`, `griffe`, and `tabulate`.
- **`gpu`**: Includes `cupy` package for basic GPU support.

## Installation on HPC systems

Installing `quatrex` on HPC systems is also quite straightforward using
`pixi` or `uv`. However, there are some additional considerations to
keep in mind when running on HPC systems, such as the need to build
certain dependencies from source to ensure compatibility with the
system's MPI and GPU backends/features.

Below, as an example, we provide instructions for installing `quatrex`
using `pixi` and `uv` on the Alps supercomputer at the Swiss National
Supercomputing Centre (CSCS).

The steps for installing `quatrex` on other HPC systems should be
similar, i.e.:

1. Load the appropriate compiler, MPI environment, and GPU modules.
2. Install `quatrex` and its basic dependencies.
3. Make sure you [install `mpi4py` from
   source](https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources)
   to ensure compatibility with the system's MPI backend.
4. Determine whether you need to [install `cupy` from
   source](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-source)
   or if a pre-built binary is available for your system (should be the
   case for NVIDIA GPUs). If you need to build from source, make sure to
   set the appropriate environment variables for your GPU backend.

!!! note "Building `cupy` from source"
    To build `cupy` with NCCL support, you may need to set the following
    environment variables to point to your NCCL
    ```bash
    export NCCL_ROOT_DIR=/path/to/nccl
    export CPATH=$NCCL_ROOT_DIR/include:$CPATH
    export LIBRARY_PATH=$NCCL_ROOT_DIR/lib:$LIBRARY_PATH
    ```

!!! info "Pre-built containers on Alps"
    We plan to provide pre-built containers for `quatrex` on Alps in the
    future, which will simplify the installation process.

### Installing `quatrex` on Alps using `pixi`

After setting up pixi and cloning the source code, set up and start an
up-to-date programming environment with the appropriate compiler, MPI,
and GPU modules. For example, on Alps:

```bash
uenv start --view=default prgenv-gnu/26.3:v1
```

Now you can install `quatrex` and its dependencies for running on HPC
systems with:

```bash
pixi install --environment=hpc
```

The installation will take a while, as `mpi4py` and `cupy` will be built
from source. After the installation is complete, you can run `quatrex`
on multiple nodes using a batch script similar to the following:

```bash
#!/bin/bash

#SBATCH --job-name=quatrex
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default

export CUPY_CACHE_DIR=${SCRATCH}/.cupy/kernel_cache
export NUMBA_CACHE_DIR=${SCRATCH}/.numba/kernel_cache

srun pixi run --environment=hpc quatrex run <path/to/config.toml>

```

!!! note "Kernel Caches"
    When running on multiple ranks for the first time, kernel caches can
    become incoherent across nodes. It is better to warm up the caches
    or to explicitly set cache directories to a shared location, as
    shown in the example above.

### Installing `quatrex` on Alps using `uv`

The steps for installing `quatrex` using `uv` on Alps are similar to the
steps for `pixi`. After setting up `uv` and cloning the source code,
again set up and start an up-to-date programming environment.

After this, following the instructions for [installing Python software
on Alps](https://docs.cscs.ch/build-install/python/), you can create a
virtual environment and install `quatrex` and its dependencies:

```bash
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy
source .venv/bin/activate

uv pip install ".[gpu]" --no-binary=mpi4py
```

!!! note
    The `--no-binary=mpi4py` option is necessary to ensure that `mpi4py`
    is built from source and linked against the system's MPI library.
    This is automatically handled in the `pixi` installation above.

Instead of the `[gpu]` specifier, on Alps you can also use the basic
installation without optional dependencies and install a `cupy` binary
that will link against the system's runtime libraries. For example, for
CUDA 13.x, you can run:

```bash
uv pip install . --no-binary=mpi4py
uv pip install cupy-cuda13x
```

```bash
#!/bin/bash

#SBATCH --job-name=quatrex
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default

export CUPY_CACHE_DIR=${SCRATCH}/.cupy/kernel_cache
export NUMBA_CACHE_DIR=${SCRATCH}/.numba/kernel_cache

srun uv run quatrex run <path/to/config.toml>

```
