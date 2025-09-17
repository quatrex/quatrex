# Copyright (c) 2024-2025 ETH Zurich and the authors of the quatrex package.

"""Main CLI entrypoint and command dispatch for quatrex."""

from threadpoolctl import threadpool_limits, threadpool_info  # isort: skip
import time
from pathlib import Path
from pprint import pprint
from typing import Optional

import typer
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
    typer.secho(HEADER, fg="bright_white", bold=True)


def version_callback(value: bool):
    """Prints the version/header and exits."""
    if value:
        secho_header()
        raise typer.Exit()


def run_quatrex(
    quatrex_config_path: Path,
    compute_config_path: Path | None = None,
):
    # remove import overhead from cli startup time
    from mpi4py.MPI import COMM_WORLD as comm

    from quatrex.core.compute_config import ComputeConfig
    from quatrex.core.compute_config import parse_config as parse_compute_config
    from quatrex.core.quatrex_config import parse_config as parse_quatrex_config
    from quatrex.core.scba import SCBA

    if compute_config_path is not None:
        compute_config = parse_compute_config(compute_config_path)
    else:
        compute_config = ComputeConfig()

    quatrex_config = parse_quatrex_config(quatrex_config_path)

    with threadpool_limits(
        limits=compute_config.blas_num_threads, user_api=compute_config.threadpool_api
    ):
        pprint(threadpool_info()) if comm.rank == 0 else None

        # TODO: decide SCBA or QTBM based on config
        scba = SCBA(quatrex_config, compute_config)

        tic = time.perf_counter()
        scba.run()
        toc = time.perf_counter()

        if comm.rank == 0:
            print(f"Leaving SCBA after: {(toc - tic):.2f} s")


@quatrex_cli.command("fetch-example")
def fetch_example(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Name of the example to fetch. Allowed examples are: "
        + ", ".join(ALLOWED_EXAMPLES.keys()),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Forces re-download even if the dataset already exists.",
    ),
):
    """
    Fetch a preconfigured example by name.
    """
    if name not in ALLOWED_EXAMPLES.keys():
        raise ValueError(
            f"Unknown example: {name}. Allowed examples are: {list(ALLOWED_EXAMPLES.keys())}"
        )

    typer.echo(f"Fetching example: {name}")
    device_key, target_dir = get_example_dir(name)

    for subname in ALLOWED_EXAMPLES[name]:
        load_example(
            device_key + "-" + subname, target_dir=target_dir / "inputs", force=force
        )


@quatrex_cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = None,
    quatrex_config: Path = typer.Option(
        Path.cwd() / "quatrex_config.toml",
        "--quatrex-config",
        "-qc",
        help="Path to the Quatrex configuration file",
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
    compute_config: Path | None = typer.Option(
        None,
        "--compute-config",
        "-cc",
        help="Path to the compute configuration file",
        show_default="None",
        file_okay=True,
        dir_okay=False,
        writable=False,
        resolve_path=True,
    ),
):
    """Main entrypoint for Quatrex CLI."""

    typer.echo(HEADER)

    if ctx.invoked_subcommand is None:
        typer.echo(f"Quatrex config: {quatrex_config}")
        if compute_config is not None:
            typer.echo(f"Compute config: {compute_config}")

        run_quatrex(
            quatrex_config,
            compute_config,
        )


def run():
    """Runs the quatrex CLI app."""
    quatrex_cli()
