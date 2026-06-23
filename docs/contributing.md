# Contributing to `quatrex`

This guide provides some information for contributing to `quatrex`. It
covers setting up a development environment, coding standards, testing,
and the contribution workflow.

## Setting up a development environment

It is recommended to use [`pixi`](https://pixi.sh/) to set up a
development environment for `quatrex`, as described in the [installation
instructions](getting_started/installation.md#installation-using-pixi).

```bash
pixi install --environment=dev
```

The `dev` environment includes tools for development, testing, and
linting:

- `pytest` with the `coverage`,
  [`pytest-mpi`](https://pytest-mpi.readthedocs.io/en/latest/), and
  [`pytest-xdist`](https://pytest-xdist.readthedocs.io/en/stable/)
  plugins to run the tests,
- [`ruff`](https://docs.astral.sh/ruff/) for linting,
- [`black`](https://black.readthedocs.io/en/stable/) for code
  formatting,
- [`isort`](https://isort.readthedocs.io/en/latest/) for sorting imports
  according to [PEP 8](https://peps.python.org/pep-0008/) guidelines.

Additionally, [`pre-commit`](https://pre-commit.com/) is used to manage
pre-commit hooks for linting and formatting code. Install the pre-commit
hooks with:

```bash
pixi run --environment=dev pre-commit install
```

`pixi` also allows the definition of a few custom commands to simplify
common development tasks. You can list the available tasks with:

```bash
pixi task list
```

Besides other common Python development tools available in IDEs,
astral's [`ty`](https://docs.astral.sh/ty/) language server is worth
giving a try, as it may offer better performance on larger codebases.

## General guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines for
  all Python code. We are using
  [black](https://black.readthedocs.io/en/stable/) for automatic code
  formatting.
- Write clear and concise docstrings for all functions and classes in
  accordance with the [NumPy documentation
  style](https://numpydoc.readthedocs.io/en/latest/format.html). You can
  use standard markdown syntax for formatting and use `$...$` delimiters
  for in-line mathematical expressions and `$$\n...\n$$` for block-level
  equations. These are rendered using [mathjax](https://mathjax.org/).
- It is always a good idea to include new unit tests for additional
  features and bug fixes.
- Ensure that all tests pass locally, including style checks. We use
  [pre-commit](https://pre-commit.com/) to manage pre-commit hooks.

## Development flow

- [Open an issue on
  GitHub](https://github.com/quatrex/quatrex/issues/new/choose)
  describing the feature or bug you want to address.
- Create a new branch for your feature or bugfix.
- Make your changes and commit them with clear commit messages.
- Push your branch to GitHub and open a pull request, explaining your
  changes.
- To have at least a second pair of eyes on your changes, request a
  review.

## Example configurations

As mentioned in the [installation
instructions](getting_started/installation/#obtaining-the-source-code),
we provide several example configurations, input files, and reference
outputs that we use for testing and development.

This data is tracked using [Git LFS](https://git-lfs.com/). We do not
use GitHub's built-in large file storage, as it has some bandwidth
limitations. Instead, we use ETH Zürich's GitLab instance to host the
LFS files in [this project](https://gitlab.ethz.ch/quatrex/quatrex).

To update the LFS files, you will need to have write access to the
GitLab project. Contact current project maintainers for access. The
files can be read without authentication.

## Automated linting and testing

We use GitHub actions for most of the automated testing and linting. We
run linting and formatting with `ruff`, `black`, and `isort` and
single-rank and distributed tests with `pytest` and `pytest-mpi`. Since
the default GitHub runners do not have GPUs, we only run the CPU tests
on GitHub. In addition, these runners have only 4 CPU cores, so we run
the distributed tests with just 3 ranks.

We also run the test suite on Alps. See the CSCS [CI/CD
documentation](https://docs.cscs.ch/services/cicd/) for reference. We
set up a Docker image with the required dependencies and use it to run
the GPU tests on Alps.

If you have the necessary permissions, you can trigger the Alps pipeline
manually on pull requests by posting a comment starting with `cscs-ci
run`.

## Documentation

The documentation is built using the
[Zensical](https://zensical.org/docs/get-started/) framework. The
`mkdocstrings-python` plugin is used to automatically generate [API
reference documentation](api) from the docstrings in the code. The
`griffe` plugin is used to automatically generate the [simulation
parameter reference pages](user_guide/parameters) from the `pydantic`
model definitions.

All documentation is built and deployed automatically on GitHub pages
whenever changes are pushed to the default branch.

To build and view the documentation locally, you can use the `pixi`
task:

```bash
pixi run docs serve
```

For more information on writing documentation, see the documentation for
[Zensical](https://zensical.org/docs/authoring/markdown/).
