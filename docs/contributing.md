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

- [`pytest`](https://docs.pytest.org/en/stable/) with the
  [`pytest-cov`](https://pytest-cov.readthedocs.io/en/latest/readme.html),
  [`pytest-mpi`](https://pytest-mpi.readthedocs.io/en/latest/), and
  [`pytest-xdist`](https://pytest-xdist.readthedocs.io/en/stable/)
  plugins to run the tests
- [`ruff`](https://docs.astral.sh/ruff/) for linting
- [`black`](https://black.readthedocs.io/en/stable/) for code formatting
- [`isort`](https://isort.readthedocs.io/en/latest/) for sorting imports
  according to [PEP 8](https://peps.python.org/pep-0008/) guidelines
- `ipykernel` / `ipympl` to run Jupyter notebooks with matplotlib
  support enabled.

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
giving a try, as it may offer better performance on somewhat larger
Python codebases.

## General guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines for
  all Python code. We are using
  [`black`](https://black.readthedocs.io/en/stable/) for automatic code
  formatting and [`ruff`](https://docs.astral.sh/ruff/) for linting.
  Note that docstrings are not formatted by `black`, so please ensure
  that they are properly formatted and readable.
- Write clear and concise docstrings for all functions and classes in
  accordance with the [NumPy documentation
  style](https://numpydoc.readthedocs.io/en/latest/format.html). You can
  use standard markdown syntax for formatting and `$...$`/`$$...$$`
  delimiters for in-line/block-level mathematical expressions. These are
  rendered using [mathjax](https://mathjax.org/).
- It is always a good idea to include new unit tests for additional
  features and bug fixes.
- Ensure that all tests pass locally before submitting a pull request
  and that the code is properly linted and formatted. We use
  [pre-commit](https://pre-commit.com/) to manage pre-commit hooks.

## Development flow

- [Open an issue on
  GitHub](https://github.com/quatrex/quatrex/issues/new/choose)
  describing the feature or bug you want to address.
- Create a new branch for your feature or bugfix.
- Make your changes and commit them with clear commit messages.
- New features should be accompanied by new unit tests and documentation
  updates and (if applicable) a few words on the methodology behind the
  feature.
- Push your branch to GitHub and open a pull request, explaining your
  changes.
- To have at least a second pair of eyes on your changes, request a
  review.

## Example configurations

As mentioned in the [installation
instructions](getting_started/installation/#obtaining-the-source-code),
we provide several example configurations, input files, and reference
outputs that we use for testing and development. This data is tracked
using [Git LFS](https://git-lfs.com/). We do not use GitHub's built-in
large file storage, as it has some bandwidth limitations. Instead, we
use ETH Zürich's GitLab instance to host the LFS files in [this
project](https://gitlab.ethz.ch/quatrex/quatrex).

To update the LFS files, you will need to have write access to the
GitLab project. Contact current project maintainers for access. The
files can be read without authentication.

## Automated linting and testing

We use [GitHub actions](https://docs.github.com/en/actions) for most of
the automated testing and linting. We run linting and formatting with
`ruff`, `black`, and `isort` and single-rank and distributed tests with
`pytest` and `pytest-mpi`. Since the default GitHub runners do not have
GPUs, we only run the CPU tests on GitHub. These runners have only 4 CPU
cores, so we run the distributed tests with just 3 ranks.

On pull requests, the GitHub actions are triggered automatically. The
results of the tests and linting are displayed in the pull request
making use of the
[`python-coverage-comment-action`](https://github.com/py-cov-action/python-coverage-comment-action/).

!!! note "Coverage report for CPU tests only"
    Note that the coverage report is only generated for the CPU tests,
    as the coverage report for the GPU tests is not generated directly
    on GitHub.

We also run the full test suite on Alps' GPUs. See the CSCS [CI/CD
documentation](https://docs.cscs.ch/services/cicd/) for reference. If
you have the necessary permissions, you can trigger the Alps pipeline
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

To track different versions of `quatrex`'s documentation, we use
[`mike`](https://zensical.org/docs/compatibility/mkdocs/mike/). Every
merge into `dev` and every tagged release on `main` triggers a new
documentation build and deployment to the `gh-pages` branch. The
documentation is automatically made available at
[https://quatrex.github.io/quatrex/](https://quatrex.github.io/quatrex/).

To build and view the documentation locally, you can use the `pixi`
task:

```bash
pixi run docs serve
```

For more information on writing documentation, see, e.g., the [section
on authoring](https://zensical.org/docs/authoring/markdown/) in the
Zensical documentation.

## Publishing a `quatrex` release

!!! note "Semantic versioning"
    We follow [semantic versioning](https://semver.org/) for `quatrex`
    releases. The version number is in the format `X.Y.Z`, where `X` is
    the major version, `Y` is the minor version, and `Z` is the patch
    version.

!!! danger "Early development"
    We are currently in the `0.Y.Z` development phase, so we do not
    guarantee backward compatibility for `quatrex` releases. (See
    https://semver.org/#spec-item-4)

The following steps are a guideline for publishing a new release of
`quatrex`:

1. Make sure that all tests pass on the `dev` branch and that the code
   is properly linted and formatted.
2. Update the version number in `src/quatrex/__about__.py`.
3. Open a pull request to merge the changes into the `main` branch. The
   title of the pull request should be `Release vX.Y.Z`, where `X.Y.Z`
   is the new version number.
4. After all tests pass and the pull request is approved, merge the pull
   request into the `main` branch. Do not squash the commits, as we want
   to keep the commit history for the release.
5. Tag the commit with the new version number, e.g., `vX.Y.Z` and create
   a release on GitHub. The release notes should include a summary of
   the changes in the new version.

