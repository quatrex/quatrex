# Atomistic Quantum Transport

## Driven Quantum Systems

In electronic structure calculations one is usually concerned with
spectral properties (i.e. the eigenvalues and eigenvectors) of a quantum
system in thermodynamic equilibrium. In quantum transport simulations,
on the other hand, we are interested in properties of the
resolvent under non-equilibrium conditions.

In other words, instead of solving the eigenvalue problem

$$
\mathbf{H} \boldsymbol{\psi} = E \mathbf{S} \boldsymbol{\psi},
$$

where $\mathbf{H}$ is the Hamiltonian and $\mathbf{S}$ is the orbital
overlap matrix, we are now interested in the system matrix

$$
\mathbf{M}(E) = E\mathbf{S} - \mathbf{H}
$$

and how the system responds to external perturbations. The energy $E$ is
now an input parameter, entering our system matrix $\mathbf{M}(E)$.

To drive the system away from equilibrium, we further couple it to
external reservoirs that allow us to exert control over the particle
flow through the system. Introducing these reservoirs leads to a change
("renormalization") of the original system's dynamics, which is
typically expressed in terms of self-energies $\mathbf{\Sigma}(E)$ that
are added to the system matrix:

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
reflects the fact that the reservoirs act as sinks and sources of
particles. The anti-Hermitian part of the self-energy is related to the
lifetime of the states in the system, while the Hermitian part leads to
a shift of the system's energy levels.

## Atomistic Material Descriptions
