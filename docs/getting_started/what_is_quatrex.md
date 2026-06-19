# What is `quatrex`?

The `quatrex` package is an _ab initio_ quantum transport simulator
developed at ETH Zürich. Starting from a description of a nanosystem's
geometry, its electronic structure (Kohn-Sham Hamiltonian and overlap
matrix), and a set of relevant configuration parameters, `quatrex`
computes transport properties, such as transmission and current spectra,
non-equilibrium charge carrier densities, and current-voltage
characteristics.

<!-- Good spot for a diagram/illustration -->

The underlying theory is the non-equilibrium Green's function (NEGF)
formalism, which is a powerful and widely used framework for describing
quantum transport in nanoscale systems. Besides a powerful method for
simulating coherent quantum transport based on the quantum transmitting
boundary method (QTBM), `quatrex` implements NEGF with scattering
effects, like screened Coulomb interactions at the level of the GW
approximation and electron-phonon interactions in a pseudo-scattering
potential approach. You can find more details about the theoretical
framework and the implemented methods in the
[methodology section](../user_guide/methodology/index.md) of the user
guide.

!!! note "Development status"
    `quatrex` is a research code, and its development is ongoing. The
    current version is a first release, and we are actively working on
    adding new features, further improving performance, and enhancing
    usability. We welcome feedback and contributions from the community
    to help shape the future of `quatrex`. This is best done by opening
    an issue or a pull request on the [GitHub
    repository](https://github.com/quatrex/quatrex).

In terms of implementation and performance, we leverages Python's core
CPU and GPU array frameworks ([`numpy`](https://numpy.org/) and
[`cupy`](https://cupy.dev/)) as well as associated frameworks and
libraries ([`scipy`](https://scipy.org/),
[`mpi4py`](https://mpi4py.readthedocs.io/en/stable/),
[`numba`](https://numba.pydata.org/) among others). The `quatrex`
codebase is designed to be extensible, portable, and highly performant.
It shows excellent scaling and sustained exascale performance on
different supercomputers.
