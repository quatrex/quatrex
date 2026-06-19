# Installation

Here we provide instructions for setting up a basic environment for
running quantum transport simulations with `quatrex`.

!!! note

    We currently only provide instructions for installing `quatrex` from
    source and with [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html).


First, clone the repository

```bash
git clone https://github.com/quatrex/quatrex.git
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

You can then create a conda environment from the provided
`environment-dev.yml` file

```bash
conda env create -f environment-dev.yml
conda activate quatrex
```

This basic environment includes `numpy` and `scipy` as dependencies. It
does not include `mpi4py` and `cupy` by default. The reason for this is
that you may want to leverage your system's MPI and GPU backend. This is
especially the case in HPC environments.

To leverage your system's backends, you will have to install `mpi4py`
and `cupy` from source (e.g. via [PyPI](https://pypi.org/)):


```bash
pip install mpi4py
pip install cupy  # (for GPU support)
```


If you do not care about using your system's backend, you can install
`mpi4py` and `cupy` from the `conda-forge` channel:


```bash
conda install -c conda-forge mpi4py mpich
conda install -c conda-forge cupy cuda-version=XX.X  # (for GPU support)
```

Finally, you can install `quatrex` from source:

```bash
pip install .
```
