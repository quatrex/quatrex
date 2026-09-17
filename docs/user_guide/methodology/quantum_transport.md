# Atomistic Quantum Transport

## Driven Quantum Systems

In electronic structure calculations one is usually concerned with
spectral properties (i.e. the eigenvalues and eigenvectors) of a quantum
system in thermodynamic equilibrium. In ***quantum transport
simulations***, on the other hand, we are interested in properties of a
system under non-equilibrium conditions.

In other words, instead of solving the eigenvalue problem

$$
\mathbf{H} \boldsymbol{\psi} = E \mathbf{S} \boldsymbol{\psi},
$$

where $\mathbf{H}$ is the Hamiltonian and $\mathbf{S}$ is the orbital
overlap matrix, we are now interested in the *system matrix*

$$
\mathbf{M}(E) = E\mathbf{S} - \mathbf{H}
$$

and how the system responds to external perturbations. The energy $E$ is
now an input parameter, entering the system matrix $\mathbf{M}(E)$.

To drive the system away from equilibrium, we further couple it to
external reservoirs that allow us to exert control over the particle
flow through the system. Introducing these reservoirs leads to a change
("renormalization") of the original system's dynamics, which is
typically expressed in terms of self-energies $\mathbf{\Sigma}(E)$ that
are subtracted from the system matrix:

$$
\mathbf{M}(E) = E\mathbf{S} - \mathbf{H} - \mathbf{\Sigma}(E)
$$

A detailed description of how the self-energies $\mathbf{\Sigma}(E)$ are
computed for a given reservoir is provided in the section on
[open boundary conditions](obc.md).

Besides boundary self-energies, one can also include other types of
self-energies, such as those arising from scattering with different
systems of particles, e.g. electron-electron or electron-phonon
interactions. How these self-energies can be computed with `quatrex`, is
discussed in the section on
[non-equilibrium Green's functions](negf.md).

The self-energies $\mathbf{\Sigma}(E)$ are usually non-Hermitian, which
reflects the fact that the reservoirs and the interactions with other
systems effectively act as sinks and sources of particles. The
anti-Hermitian part of the self-energy is related to the lifetime of the
states in the system, while the Hermitian part leads to a shift of the
system's energy levels.

## Atomistic Material Descriptions

The matrices $\mathbf{H}$ and $\mathbf{S}$ are built from a set of
localized basis functions $\{\psi_i\}$, rather than from a plane-wave or
real-space grid discretization. The matrix elements are given by the
following integrals over the basis functions:

$$
\begin{aligned}
    H_{ij} &= \bra{\psi_i} \hat{H} \ket{\psi_j} \\
    S_{ij} &= \braket{\psi_i \vert \psi_j}
\end{aligned}
$$

Depending on the level of theory, the basis functions $\psi_i$ and the
matrix elements $H_{ij}$ and $S_{ij}$ could be obtained from
semi-empirical tight-binding models or from first-principles electronic
structure calculations, e.g., using density functional theory (DFT).
More information on how to provide this data to `quatrex` is provided in
the section on [electronic structure input data](../input_data/).

In any case, because the basis states are localized, $\mathbf{H}$ and
$\mathbf{S}$ will be sparse, indicating that an orbital only has
non-negligible overlap and only interacts with orbitals that are
centered close by. This operator sparsity is what makes atomistic
quantum transport computationally tractable for realistically-sized
systems at all, and it is heavily exploited in `quatrex`, both in how
all tensors are stored and in how the transport equations are solved.

In [QTBM](qtbm.md) calculations, we use optimized sparse linear solvers
to take advantage of the system's sparsity. In [NEGF](negf.md)
calculations, we can define a block-tridiagonal tiling of the matrices
(*if the atoms are ordered along the transport direction*), which allows
us to use the recursive Green's function (RGF) or related
Schur-complement-based selected-inversion algorithms to compute the
Green's functions without ever having to invert the full dense system
matrix $\mathbf{M}(E)$ explicitly.

!!! info "Ordering of atoms and orbitals in NEGF calculations"
    The ordering of the atoms and orbitals in the input data is crucial
    for the performance of the transport calculations. In particular,
    the atoms should be ordered along the transport direction. This
    ensures that the system matrix $\mathbf{M}(E)$ has a
    block-tridiagonal structure, which is necessary for the efficient
    use of the RGF algorithm in NEGF calculations.

While periodicity is always broken along the transport direction, it is
often preserved in the transverse directions. This allows us to use
Bloch's theorem to reduce the simulation domain size by solving the
transport equations for a set of transverse wavevectors
$\mathbf{k}_\perp$. The Hamiltonian and overlap matrices for every
transverse wavevector $\mathbf{k}_\perp$ are obtained by Bloch-summation
of the real-space matrices over the lattice vectors $\mathbf{R}_\perp$
in the transverse directions:

$$
\begin{aligned}
    H_{ij}(\mathbf{k}_\perp) &= \sum_{\mathbf{R}_\perp} e^{i
    \mathbf{k}_\perp \cdot \mathbf{R}_\perp} H_{ij}(\mathbf{R}_\perp) \\
    S_{ij}(\mathbf{k}_\perp) &= \sum_{\mathbf{R}_\perp}
    e^{i \mathbf{k}_\perp \cdot \mathbf{R}_\perp} S_{ij}(\mathbf{R}_\perp)
\end{aligned}
$$

Nanowire and nanoribbon structures have no periodicity in the transverse
directions, so only the $\Gamma$-point ($\mathbf{k}_\perp = 0$) needs to
be considered. For 2D materials, only the in-plane transverse direction
is periodic, so a 1D grid of $\mathbf{k}_\perp$-points can be used. For
3D bulk materials, a 2D grid of $\mathbf{k}_\perp$-points would normally
be used.
