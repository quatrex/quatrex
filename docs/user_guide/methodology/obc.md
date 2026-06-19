# Open Boundary Conditions

In transport simulations, we are modeling driven quantum systems. This
means that charge carriers are injected into the considered simulation
domain from one *contact*, and they are extracted from the domain at
another *contact*. Since we have to restrict the part of the system that
we model explicitly, these contacts are approximated as semi-infinite
reservoirs in thermal equilibrium that are connected to the simulation
domain and can provide or absorb charge carriers. As long as the
contacts are sufficiently far from the active region of the device, this
is usually a good approximation.

<!-- This needs a visualization -->

In `quatrex`, the electronic structure of the contacts is extracted
directly from the provided Kohn-Sham Hamiltonian and overlap matrix. The
contact matrix elements are selected based on the geometry of the system
and the provided configuration.

<!-- This also needs a visualization -->

Since these open contacts can be understood as "re-normalizing" the
dynamics of the system, they enter the Dyson and Keldysh equations.


## Retarded Open Boundary Self-Energy

$$
\mathbf{g}^R = \left[\mathbf{m}_{0} - \mathbf{m}_{-1} \mathbf{g}^R
\mathbf{m}_{+1} \right]^{-1}
$$

$$
\sum \limits_{n=-b}^{+b} \lambda^{n} \hat{\mathbf{m}}_{n} \mathbf{v} = 0
$$


## Lesser/Greater Open Boundary Self-Energy
