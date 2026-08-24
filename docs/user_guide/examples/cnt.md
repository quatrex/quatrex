# Carbon Nanotube

This example demonstrates how to obtain electronic structure information
as inputs for `quatrex` and how to perform different kinds of transport
simulations for a roughly 10 nm long (8, 0) single-wall carbon nanotube
(CNT).

## Geometry and electronic structure

We can create the geometry of the CNT using the
[`ase`](https://wiki.fysik.dtu.dk/ase/) Python package, which has a
built-in function to generate nanotubes
([`ase.build.nanotube`](https://docs.ase-lib.org/ase/build/build.html#ase.build.nanotube)).
After exporting the geometry to a [POSCAR
file](https://vasp.at/wiki/POSCAR), one can set up a DFT calculation
using VASP to obtain the electronic structure in plane-wave basis. VASP
is then run with the [`LWANNIER90 =
.TRUE.`](https://vasp.at/wiki/LWANNIER90) flag to generate the necessary
input files for Wannier90. Projecting onto initial guesses of one $p_z$
orbital per carbon atom, we then run Wannier90 and get the Hamiltonian
in the Wannier basis, which is used as input for `quatrex`.

A procedure for generating `quatrex` input data from Wannier90 outputs
is described in the [electronic structure data section](../input_data).
The resulting device structure (`structure.xyz`) and Hamiltonian
(`hamiltonian.h5`) are stored in the
`./examples/w90/carbon-nanotube/inputs` directory.

You can find more detailed information on the input data provenance in
the `README.md` file in the example directory
`./examples/w90/carbon-nanotube/`

Here you can see the resulting *Wannier centers* of the CNT structure.
Note that the orbital centers are not necessarily coincident with the
atomic positions.

{{ mol3d("../../assets/structures/carbon-nanotube.xyz", style={"stick": {"radius":
-1}, "sphere": {"scale": 0.25}}) }}

The actual orbital coordinates are stored in an `.xyz` file that looks
like
```xyz
768
Lattice="102.84851504839999 0.0 0.0 0.0 50.0 0.0 0.0 0.0 50.0" Properties=species:S:1:pos:R:3 pbc="F F F"
X        0.72770969      21.80861750      25.06696100
X        2.01508769      22.02025450      25.89201500
X        0.15518369      22.76798050      27.19112700
X        4.05541969      25.00283250      27.35868900
X        4.34182130      25.04734650      28.15860700
X        2.81802769      26.25013450      27.98060700
X        4.24380469      27.06711950      27.31367300
X        2.22266969      27.93073350      26.16808300
X        1.18530469      28.19138250      25.05384000
X        2.06710169      27.73589650      23.58711800
X        0.45901869      27.26246150      22.75888700
X        2.09800069      26.07841450      22.18612000
X        4.21870469      24.88363450      21.84139300
X        1.95921469      24.37907150      22.02629800
...
```

The band structure of the device around its equilibrium Fermi level is
captured very well by this Wannierization.

<!-- TODO: Include proof for this -->

## Computing the transmission function

Say we are interested in seeing the transmission spectrum through this
CNT. This kind of coherent transport is most efficiently treated in the
[wavefunction formalism](../methodology/qtbm.md), so we set
`#!toml formalism = "wf"`. We will not be employing a self-consistent
solution of the Hartree potential for this purpose, so we do not include
a [`[scsp]`](../parameters/scsp.md) section in the config here.

Next we need to define the contact regions of the device. The whole
carbon nanotube is made up of 24 repeated unit cells along the transport
direction `a`. From the DFT simulation, we know that each of these
cells has a length of 4.27615261 Å and from the structure file above, we
can see that the orbital center with the smallest x-coordinate sits at
(0.15518369, 22.76798050, 27.19112700). Together, this allows us to
get a contact definition as follows:

```toml
[[device.contacts]]
name = "left"
origin = [0.1551, 0.0, 0.0]
lattice_vectors = [[4.27615261, 0, 0], [0, 50, 0], [0, 0, 50]]
direction = "a"
fermi_level = -3.6
```

The Fermi level is the one from DFT. Similarly for the right contact, we
set

```toml
[[device.contacts]]
name = "right"
origin = [102.694, 0.0, 0.0]
lattice_vectors = [[-4.27615261, 0, 0], [0, 50, 0], [0, 0, 50]]
direction = "a"
fermi_level = -3.601
```

Here we added a small chemical potential difference of 1 meV to get a
small current flowing across the device. We will compute the
transmission function at 1000 energy points between -6.5 eV and -1.0 eV,
which will yield an energy grid of 5.5 meV. Note that this may still be
too coarse to resolve sharp resonances in the transmission function, but
it is sufficient for this example. We set

```toml
energy_window_min = -6.5
energy_window_max = -1.0
energy_window_num = 1000
```

Finally, for QTBM we need to employ the [spectral OBC
algorithm](../methodology/obc/#spectral-method). For robustness, and
since this is a very small system, we choose the `#!toml obc.nevp_solver
= "full"`.

??? example "Full configuration for coherent transport simulation"
    This is what the full configuration file for this simulation run
    will look like

    ```toml
    formalism = "wf"

    [device]
    transport_direction = "a"

        [[device.contacts]]
        name = "left"
        origin = [0.1551, 0.0, 0.0]
        lattice_vectors = [[4.27615261, 0, 0], [0, 50, 0], [0, 0, 50]]
        direction = "a"
        fermi_level = -3.6

        [[device.contacts]]
        name = "right"
        origin = [102.694, 0.0, 0.0]
        lattice_vectors = [[-4.27615261, 0, 0], [0, 50, 0], [0, 0, 50]]
        direction = "a"
        fermi_level = -3.601

    [electron]

    energy_window_min = -6.5
    energy_window_max = -1.0
    energy_window_num = 1000


    obc.algorithm = "spectral"
    obc.nevp_solver = "full"
    ```

You can run this simulation by invoking

```bash
quatrex run <path/to/config.toml>
```

After `quatrex` completes, you will find the [file
`transmission_lr.npy`](../simulation_output/#transmission-function) in
the output folder.

<!-- TODO: Include picture of transmission -->

## Including phonon pseudo-scattering

Next we may want to introduce a small amount of scattering with a phonon
model. To accomplish this, we need to move beyond the wavefunction
picture and set `#!toml formalism = "negf"`.

The self-consistent Born approximation (SCBA) is used to ensure
consistency between the scattering self-energy and the Green's function.
The SCBA loop, including only phonon pseudo-scattering, is typically
very stable, so we do not need any under-relaxation here (i.e. `#!toml
mixing_factor = 1.0`). The `[scba]` section of the config will look like
this:

```toml
[scba]
max_iterations = 15
mixing_factor = 1.0
phonon = true
```

!!! warning "Differences between `"wf"` and `"negf"` inputs"
    Currently, an important difference between simulations employing the
    `"wf"` formalism and those using `"negf"` is the way the contacts
    are defined in the config:

    - In `"wf"` simulations they are inferred from real-space contact cell
    definitions ([`[[device.contacts]]`](../parameters/contact.md)) and more
    than two contacts are supported
    - In `"negf"` calculations, contact matrix elements are taken from a user-prescribed
    block-tiling ([`block_size`](../parameters/device/#block_size)) and only
    two-terminal devices can be treated.

    The reason for this is that the two formalisms were implemented more
    or less independently of one another. We are actively working on
    further consolidating input files.

As stated above, the full structure, encompassing 768 Wannier orbitals,
is made up of 24 transport cells that contain 32 orbitals each. The
`#!toml [device]` section of the config therefore now looks like this:

```toml
[device]
transport_direction = "a"
block_size = 32
```

Lastly, besides the `#!toml phonon = true` flag in the SCBA section, to
configure the actual phonon interaction, we can set the following
parameters:

```toml
[phonon]
interaction_cutoff = 5.0 # Angstrom

model = "pseudo-scattering"
phonon_energy = 40e-3         # eV
deformation_potential = 15e-3 # eV
```

??? example "Full configuration for phonon pseudo-scattering"
    This is what the full configuration file for this simulation run
    will look like.

    ```toml
    formalism = "negf"

    [scba]
    max_iterations = 15
    mixing_factor = 1.0
    phonon = true

    [device]
    transport_direction = "a"
    block_size = 32

    [electron]
    left_contact.name = "left"
    left_contact.fermi_level = -3.6

    right_contact.name = "right"
    right_contact.fermi_level = -3.601

    energy_window_min = -6.5
    energy_window_max = -1.0
    energy_window_num = 1000

    obc.algorithm = "spectral"
    obc.nevp_solver = "full"


    [phonon]
    interaction_cutoff = 5.0 # Angstrom

    model = "pseudo-scattering"
    phonon_energy = 40e-3         # eV
    deformation_potential = 15e-3 # eV
    ```

After running the simulation (`quatrex run <path/to/config.toml>`),
among other quantities, you will find the [spectral device
current](../simulation_output#spectral-current) in the outputs, taking
into account thermal scattering.
