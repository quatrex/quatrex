!!! warning "Page under construction"

    These pages are under construction and will be updated soon.


Quatrex requires a few inputs to run.

1. `quatrex_config.toml`
2. pre-saved Hamiltonian and other stuff
3. (optional) `compute_config.toml`

You need to fetch the data with `quatrex fetch <NAME_OF_EXAMPLE>`. This is downloads the pre-processed input files required for simulation. For most people, the regular carbon nanotube example is sufficient. Note the inclusion of the colon `:` in the name. 


# Full List of Examples
```python
ALLOWED_EXAMPLES = {
    "carbon-nanotube:": [
        "hamiltonian",
        "coulomb-matrix",
        "potential",
        "grid",
        "block-sizes",
    ],
    "carbon-nanotube:dist": [
        "hamiltonian",
        "coulomb-matrix",
        "potential",
        "grid",
        "block-sizes",
    ],
}
```
