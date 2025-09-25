!!! warning

    We currently only provide instructions for installing `qttools` from
    source and with `conda`.

First, clone the repository

```bash
git clone git@github.com:vincent-maillou/qttools.git
cd qttools
```

Create a conda environment from the provided `environment.yml` file

```bash
conda env create -f environment.yml
conda activate quatrex
```

This basic environment only includes `numpy` and `scipy` as
dependencies. You will need to install `mpi4py` (and optionally `cupy`)
manually.

The reason for this is that you may want to leverage your system's MPI
and GPU backend. For this you will have to install `mpi4py` and `cupy`
from source (e.g. via PyPI):

=== "`mpi4py`"

    ```bash
    pip install mpi4py
    ```

=== "`cupy`"

    ```bash
    pip install cupy
    ```

If you don't care about using your system's backend, you can just
install `mpi4py` and `cupy` from the `conda-forge` channel:

=== "`mpi4py`"

    ```bash
    conda install -c conda-forge mpi4py mpich
    ```

=== "`cupy`"

    ```bash
    conda install -c conda-forge cupy cuda-version=XX.X
    ```

    !!! tip

        To determine which version of cuda is running on your machine use
        `nvcc --version` or `nvidia-smi`.


Finally, install qttools:

```bash
pip install .
```
