# Command Line Interface

The usual way of launching `quatrex` is via its command line interface
(CLI). There is one main command, `quatrex run`, which is used to run a
simulation. It takes as an argument the path to a
[TOML](https://toml.io/) file containing the [simulation parameters](parameters/index.md).

## :octicons-command-palette-24: `quatrex`

```bash
quatrex [OPTIONS] COMMAND [ARGS]...
```

| Option      | Description                 |
| ----------- | --------------------------- |
| `--version` | Print the version and exit. |
| `--help`    | Show a help message.        |

## :octicons-command-palette-24: `quatrex run`

```bash
quatrex run [OPTIONS] [CONFIG]
```

| Argument | Description                                  |
| -------- | -------------------------------------------- |
| `config` | Path to the quatrex TOML configuration file. |

| Option                                            | Description                                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `--abort-on-exception`/ `--no-abort-on-exception` | Force abort the entire MPI environment on an unhandled exception to prevent hanging processes. |
| `--help`                                          | Show a help message.                                                                           |
