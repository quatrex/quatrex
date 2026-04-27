# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

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


def _run_wf(config):
    """Runs quatrex with the given configuration.

    Parameters
    ----------
    config : QuatrexConfig
        The main quatrex configuration.

    """
    from quatrex.core.qtbm import QTBM
    from quatrex.device import Device

    with threadpool_limits(
        limits=config.compute.blas_num_threads,
        user_api=config.compute.threadpool_api,
    ):
        pprint(threadpool_info()) if comm.rank == 0 else None

        device = Device(config)
        qtbm = QTBM(device, config)

        tic = time.perf_counter()
        qtbm.run()
        toc = time.perf_counter()

        if comm.rank == 0:
            typer.secho(f"Leaving QTBM after: {(toc - tic):.2f} s")


def _run_negf(config):
    """Runs quatrex with the given configuration using SCBA.

    Parameters
    ----------
    config : QuatrexConfig
        The main quatrex configuration.

    """

    from quatrex.core.scba import SCBA

    with threadpool_limits(
        limits=config.compute.blas_num_threads,
        user_api=config.compute.threadpool_api,
    ):
        pprint(threadpool_info()) if comm.rank == 0 else None

        scba = SCBA(config)

        tic = time.perf_counter()
        scba.run()
        toc = time.perf_counter()

        if comm.rank == 0:
            typer.secho(f"Leaving SCBA after: {(toc - tic):.2f} s")


@quatrex_cli.command()
def run(
    config: Annotated[
        Optional[Path],
        typer.Argument(
            ...,
            help="Path to the quatrex TOML configuration file.",
            dir_okay=True,
            resolve_path=True,
            exists=True,
        ),
    ] = None,
):
    """Runs quatrex with the provided configuration."""
    # No arguments provided, check for the default config file in the
    # working directory.
    if config is None:
        config = Path("./quatrex_config.toml")
        if not config.exists():
            raise BadArgumentUsage(
                "No quatrex configuration file provided and default "
                "'./quatrex_config.toml' does not exist."
            )

    # If a directory is provided, look for the config file inside.
    if config.is_dir():
        config = config / "quatrex_config.toml"
        if not config.exists():
            raise BadArgumentUsage(
                f"No quatrex configuration file found in directory: {config.parent}"
            )

    from qttools.profiling import Profiler
    from quatrex.core.config import parse_config

    profiler = Profiler()

    config = parse_config(config)

    secho_header()

    # Dispatch to the appropriate runner based on the formalism.
    if config.formalism == "wf":
        _run_wf(config)
    elif config.formalism == "negf":
        _run_negf(config)
    else:
        raise NotImplementedError(f"Formalism '{config.formalism}' is not implemented.")

    if config.outputs.save_profiling_results:
        profiler.dump_stats()


@quatrex_cli.command("export-environment")
def export_environment(
    config: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to a Quatrex TOML configuration with an [environment] section.",
            dir_okay=False,
            resolve_path=True,
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            help="Directory for exported environment arrays.",
            resolve_path=True,
        ),
    ] = None,
):
    """Exports environment polarization and dielectric-input arrays."""

    from quatrex.core.config import parse_config
    from quatrex.coulomb_screening.environment_export import (
        export_environment_screening,
    )

    config = parse_config(config)
    secho_header()

    result = export_environment_screening(config, output_dir=output_dir)

    if comm.rank == 0:
        typer.secho(f"Wrote environment screening export to {result.output_dir}")


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
