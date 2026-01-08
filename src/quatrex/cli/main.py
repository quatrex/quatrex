# Copyright (c) 2024-2025 ETH Zurich and the authors of the quatrex package.

"""Main CLI entrypoint and command dispatch for quatrex."""

from threadpoolctl import threadpool_limits, threadpool_info  # isort: skip
import time
from pathlib import Path
from typing import Optional

import typer
from click import BadArgumentUsage
from mpi4py.MPI import COMM_WORLD as comm
from rich import print as pprint
from typing_extensions import Annotated

import quatrex
from quatrex.examples import ALLOWED_EXAMPLES, get_example_dir
from quatrex.examples import load as load_example

HEADER = rf"""
                   _                 
  __ _ _   _  __ _| |_ _ __ _____  __
 / _` | | | |/ _` | __| '__/ _ \ \/ /
| (_| | |_| | (_| | |_| | |  __/>  < 
 \__, |\__,_|\__,_|\__|_|  \___/_/\_\
    |_|                              
                        version {quatrex.__version__}
"""

quatrex_cli = typer.Typer(
    pretty_exceptions_show_locals=False,
    add_completion=False,
    rich_markup_mode="markdown",
)


def secho_header():
    """Prints the header to the console."""
    if comm.rank == 0:
        typer.secho(HEADER, fg="bright_white", bold=True)


def version_callback(value: bool):
    """Prints the version/header and exits."""
    if value:
        secho_header()
        raise typer.Exit()


def _run_qtbm(quatrex_config, compute_config):
    """Runs quatrex with the given configuration.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The main quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.

    """
    from quatrex.core.device import Device
    from quatrex.core.qtbm import QTBM

    with threadpool_limits(
        limits=compute_config.blas_num_threads, user_api=compute_config.threadpool_api
    ):
        pprint(threadpool_info()) if comm.rank == 0 else None

        device = Device(quatrex_config, compute_config)
        qtbm = QTBM(device, quatrex_config, compute_config)

        tic = time.perf_counter()
        qtbm.run()
        toc = time.perf_counter()

        if comm.rank == 0:
            typer.secho(f"Leaving QTBM after: {(toc - tic):.2f} s")


def _run_negf(quatrex_config, compute_config):
    """Runs quatrex with the given configuration using SCBA.

    Parameters
    ----------
    quatrex_config : QuatrexConfig
        The main quatrex configuration.
    compute_config : ComputeConfig
        The compute configuration.

    """

    from quatrex.core.scba import SCBA

    with threadpool_limits(
        limits=compute_config.blas_num_threads, user_api=compute_config.threadpool_api
    ):
        pprint(threadpool_info()) if comm.rank == 0 else None

        scba = SCBA(quatrex_config, compute_config)

        tic = time.perf_counter()
        scba.run()
        toc = time.perf_counter()

        if comm.rank == 0:
            typer.secho(f"Leaving SCBA after: {(toc - tic):.2f} s")


@quatrex_cli.command()
def run(
    quatrex_config: Annotated[
        Optional[Path],
        typer.Argument(
            ...,
            help="Path to the quatrex TOML configuration file, "
            "or a directory containing the configuration file(s).",
            dir_okay=True,
            resolve_path=True,
            exists=True,
        ),
    ] = None,
    compute_config: Annotated[
        Optional[Path],
        typer.Argument(
            ...,
            help="Path to the compute TOML configuration file.",
            dir_okay=False,
            resolve_path=True,
            exists=True,
        ),
    ] = None,
):
    """Runs quatrex with the provided configuration."""
    # No arguments provided, use default paths.
    if quatrex_config is None:
        quatrex_config = Path("./quatrex_config.toml")
        if not quatrex_config.exists():
            raise BadArgumentUsage(
                "No quatrex configuration file provided and default "
                "'./quatrex_config.toml' does not exist."
            )
        compute_config = Path("./compute_config.toml")
        if not compute_config.exists():
            compute_config = None

    # If a directory is provided, look for the config files inside.
    if quatrex_config.is_dir():
        if compute_config is not None:
            raise BadArgumentUsage(
                "If a directory is provided as quatrex_config, "
                "compute_config must not be provided."
            )
        quatrex_config = quatrex_config / "quatrex_config.toml"
        if not quatrex_config.exists():
            raise BadArgumentUsage(
                f"No quatrex configuration file found in directory: {quatrex_config.parent}"
            )
        compute_config = quatrex_config.parent / "compute_config.toml"
        if not compute_config.exists():
            compute_config = None

    from quatrex.core.compute_config import ComputeConfig
    from quatrex.core.compute_config import parse_config as parse_compute_config
    from quatrex.core.quatrex_config import parse_config as parse_quatrex_config

    quatrex_config = parse_quatrex_config(quatrex_config)

    if compute_config is None:
        compute_config = ComputeConfig()
    else:
        compute_config = parse_compute_config(compute_config)

    secho_header()

    # Dispatch to the appropriate runner based on the formalism.
    if quatrex_config.formalism == "qtbm":
        _run_qtbm(quatrex_config, compute_config)
    elif quatrex_config.formalism == "negf":
        _run_negf(quatrex_config, compute_config)
    else:
        raise NotImplementedError(
            f"Formalism '{quatrex_config.formalism}' is not implemented."
        )


@quatrex_cli.command()
def fetch_example(
    name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the example to fetch. Allowed examples are:\n"
            f"{'\n'.join([f'- `{ex}`' for ex in ALLOWED_EXAMPLES.keys()])}",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Forces re-download even if the dataset already exists.",
        ),
    ] = False,
):
    """Fetches a preconfigured example by name."""
    if comm.rank == 0:
        if name not in ALLOWED_EXAMPLES.keys():
            raise ValueError(
                f"Unknown example: {name}. Current examples are: {list(ALLOWED_EXAMPLES.keys())}"
            )

        typer.secho(f"Fetching example: {name}")
        device_key, __, target_dir = get_example_dir(name)

        for subname in ALLOWED_EXAMPLES[name]:
            load_example(
                device_key + "-" + subname,
                target_dir=target_dir / "inputs",
                force=force,
            )


@quatrex_cli.callback(no_args_is_help=True)
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
):
    """Quantum Transport at the Exascale and Beyond."""
    ...


def run_cli():
    """Runs the quatrex CLI app."""
    quatrex_cli()
