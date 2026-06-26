# What is `quatrex`?

The `quatrex` package is an _ab initio_ quantum transport simulator
developed in the [computational nanoelectronics
group](https://nano-tcad.ee.ethz.ch/) at ETH Zürich.

Starting from a description of a nanosystem's geometry, its electronic
structure (Kohn-Sham Hamiltonian and overlap matrix), and a set of
relevant configuration parameters, `quatrex` computes transport
properties, such as transmission and current spectra, non-equilibrium
charge carrier densities, and current-voltage characteristics.

<!-- Good spot for a diagram/illustration -->

The underlying theory is the non-equilibrium Green's function (NEGF)
formalism, which is a powerful and widely used framework for describing
quantum transport in nanoscale systems[^negf]. Besides a powerful method
for simulating coherent quantum transport based on the quantum
transmitting boundary method (QTBM), `quatrex` implements NEGF with
scattering effects, like screened Coulomb interactions at the level of
the GW approximation[^gw] and electron-phonon interactions in a
pseudo-scattering potential approach. You can find more details about
the theoretical framework and the implemented methods in the
[methodology section](../user_guide/methodology/index.md) of the user
guide.

!!! info "Development status"
    `quatrex` is a research code, and its development is ongoing. The
    current version is a first release, and we are actively working on
    adding new features, further improving performance, and enhancing
    usability. We welcome feedback and contributions from anyone
    interested in using or developing `quatrex`. If you are interested
    in contributing, please refer to the [contributing
    guide](../contributing.md) for more information on how to get
    involved.

In terms of implementation and performance, we leverage Python's core
CPU and GPU array frameworks, [`numpy`](https://numpy.org/) and
[`cupy`](https://cupy.dev/), as well as associated frameworks and
libraries, such as [`scipy`](https://scipy.org/),
[`mpi4py`](https://mpi4py.readthedocs.io/en/stable/), and
[`numba`](https://numba.pydata.org/). The `quatrex` codebase is designed
to be extensible, portable, and highly performant. It shows excellent
scaling and sustained exascale performance on different
supercomputers.[^sc-25]

[^negf]: S. Datta, *Electronic Transport in Mesoscopic Systems*,
    Cambridge University Press, 1995.
[^gw]: L. Deuschle et al., *Electron-electron interactions in device
    simulation via non-equilibrium Green's functions and the GW
    approximation*, Phys. Rev. B, 2025.
    https://doi.org/10.1103/PhysRevB.111.195421
[^sc-25]: N. Vetsch et al., *Ab-initio Quantum Transport with the GW
    Approximation, 42,240 Atoms, and Sustained Exascale Performance*, SC
    '25. https://doi.org/10.1145/3712285.3771784
