The `quatrex` package is an *ab initio* quantum transport simulator
written in Python. We leverages Python's core CPU and GPU array
frameworks ([`numpy`](https://numpy.org/) and
[`cupy`](https://cupy.dev/)) as well as associated frameworks and
libraries ([`scipy`](https://scipy.org/),
[`mpi4py`](https://mpi4py.readthedocs.io/en/stable/),
[`numba`](https://numba.pydata.org/)). The `quatrex` codebase is
designed to be extensible, portable, and highly performant. It shows
excellent scaling and sustained exascale performance on different
supercomputers.[^1]

[^1]: N. Vetsch, et al., *Ab-initio Quantum Transport with the GW
    Approximation, 42,240 Atoms, and Sustained Exascale Performance*,
    [arXiv:2508.19138](https://doi.org/10.48550/arXiv.2508.19138)
    (2025).

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

Then create a conda environment from the provided `environment-dev.yml`
file

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
