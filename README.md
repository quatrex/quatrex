<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/logo/logo_text_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/logo/logo_text_light.svg">
    <img alt="quatrex logo" src="./docs/assets/logo/logo_text_light.svg" width="33%">
  </picture>
  <br><br>
</div>

---

[![Tests](https://img.shields.io/github/actions/workflow/status/quatrex/quatrex/tests.yaml?branch=main&label=tests)](https://github.com/quatrex/quatrex/actions/workflows/tests.yaml?query=branch%3Amain+branch%3Adev)
[![License](https://img.shields.io/badge/license-BSD_3--Clause-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-zensical-orange)](https://quatrex.github.io/quatrex/)
[![Code Style:
Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

The `quatrex` package is an *ab initio* quantum transport simulator
developed at ETH Zürich for predictive simulations of nanoscale
electronic devices. Starting from first-principles electronic structure
data and a description of the device geometry, `quatrex` can compute
transport observables including transmission spectra, currents,
non-equilibrium carrier densities, and current-voltage characteristics.
The code is designed for distributed-memory supercomputers and enables
simulations of realistic nanostructures.

## Key features

- **First-principles quantum transport at scale** using the
  non-equilibrium Green's function formalism and the quantum
  transmitting boundary method for realistic nanosystems.
- **Large-scale distributed execution** designed from the ground up for
  GPU-accelerated supercomputers and distributed-memory architectures.
- **Flexible electronic structure input** compatible with
  localized-basis Hamiltonians from multiple DFT workflows and
  electronic structure packages.
- **Self-consistent electrostatics and transport**, solving
  open-boundary Schrödinger and Poisson equations iteratively.
- **Hardware portability** through hardware-agnostic Python
  orchestration, enabling seamless execution across CPUs and GPUs.
- **Specialized numerical algorithms** for open boundaries, sparse
  linear systems, and Dyson-Keldysh equations.

## Requirements

- **Python 3.13+**
- An **MPI-capable environment**
- Optional **GPU support via CuPy**
- System-level HPC dependencies such as a working MPI stack and, for
  GPU-enabled systems, the appropriate GPU and communication libraries

## Installation

`quatrex` is installed from source using `pixi` (recommended) or `uv`.

```bash
git clone git@github.com:quatrex/quatrex.git
cd quatrex
pixi install --frozen
```

Detailed installation instructions, including optional GPU support and
deployment on HPC systems, are available in [the
documentation](https://quatrex.github.io/quatrex/getting_started/installation).

## Documentation

Detailed user documentation is available at:
[***quatrex.github.io/quatrex***](https://quatrex.github.io/quatrex/)

## Contributing

Contributions are welcome. Please see
[contributing.md](docs/contributing.md) for guidelines, and use the
[issue tracker](https://github.com/quatrex/quatrex/issues) for bug
reports and feature requests.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).

## Acknowledgments

The work on `quatrex` is funded by the Swiss National Science Foundation
(SNSF) under [grant number 209358
(QuaTrEx)](https://data.snf.ch/grants/grant/209358) and [grant number
205602 (NCCR
MARVEL)](https://www.snf.ch/en/teeBfD4ffpkMsP53/page/nccr/marvel), and
by the Platform for Advanced Scientific Computing in Switzerland
([BoostQT](https://pasc-ch.org/projects/2025-2028/boosting-large-scale-quantum-transport-simulations-through-gpubased-dedicated-libraries-boostqt/index.html)).
