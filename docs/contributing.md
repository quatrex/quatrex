---
hide:
  - navigation
---

## Setting up a development environment

To set up a development environment for `quatrex`, we recommend using
[`conda`](https://docs.conda.io/). You can follow similar steps as in
the [installation instructions](getting_started/installation.md), with
one key difference being that you should install the `quatrex` package
in editable mode.

```bash
pip install --editable .
```

## Development flow

- [Open an issue on
  GitHub](https://github.com/quatrex/quatrex/issues/new/choose)
  describing the feature or bug you want to address.
- Create a new branch for your feature or bugfix.
- Make your changes and commit them with clear commit messages.
- Push your branch to GitHub and open a pull request.
- Wait for code review and address any feedback.

## Example configurations
As described in the [installation instructions](getting_started/installation.md),
we provide several example configurations, input files, and reference
outputs that we use for testing and development.

This data is tracked using [Git LFS](https://git-lfs.com/). We do not
use GitHub's built-in large file storage, as it has some bandwidth
limitations. Instead, we use ETH Zürich's GitLab instance to host the
LFS files in [this project](https://gitlab.ethz.ch/quatrex/quatrex).

To update the LFS files, you will need to have write access to the
GitLab project. The files can be read without authentication.

## Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines for
  Python code. We recommend using
  [black](https://black.readthedocs.io/en/stable/) for automatic code
  formatting.
- Write clear and concise docstrings for all functions and classes in
  accordance with the [NumPy documentation
  style](https://numpydoc.readthedocs.io/en/latest/format.html). You can
  also make references to other parts of the API documentation using
  standard markdown syntax, that [`mkapi` will automatically convert for
  you](https://daizutabi.github.io/mkapi/usage/writing/#unique-features-of-mkapi).
- Try to include unit tests for new features and bug fixes. We use
  [pytest](https://docs.pytest.org/) as our testing framework, including
  [pytest-mpi](https://pytest-mpi.readthedocs.io/) for MPI-based tests.
- Try to ensure that all tests pass locally, including style checks. We
  use [pre-commit](https://pre-commit.com/) to manage our pre-commit
  hooks.

## Documentation

The documentation is built using the [Material for
mkdocs](https://squidfunk.github.io/mkdocs-material/) framework with the
[`mkapi`](https://github.com/daizutabi/mkapi) plugin to automatically
generate API reference documentation. The documentation is automatically
built and hosted on GitHub Pages. To build the documentation locally,
run:
```bash
mkdocs serve
```
