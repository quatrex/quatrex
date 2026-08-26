# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Functions for merging TOML configurations."""

from typing import Union

from tomlkit.container import Container
from tomlkit.items import AoT, Table

TomlTable = Union[Table, Container]


def merge_toml(
    base: TomlTable,
    patch: TomlTable,
    id_keys: dict[str, str] | None = None,
    allow_new_aot: bool = False,
) -> None:
    """Recursively merge `patch` into `base`, in place.

    Parameters
    ----------
    base : TomlTable
        The base TOML to be modified in place.
    patch : TomlTable
        The patch TOML to merge into the base.
    id_keys : dict[str, str], optional
        A mapping of array-of-tables keys to the field used to identify
        their elements. For example, if the base and patch both have a
        `[[device.contacts]]` array-of-tables, and each contact has a
        unique `name` field, then `id_keys` should be `{"contacts":
        "name"}`.
    allow_new_aot : bool, optional
        If True, allows new entries in the patch that do not match any
        existing entries in the base. If False (default), such entries
        will raise a ValueError. This is only relevant for
        array-of-tables keys. Thus, a patch can still add new keys to
        the base, but it cannot add new entries to an existing
        array-of-tables.

    """
    id_keys = id_keys or {}
    for key, patch_val in patch.items():
        # If the key exists in both base and patch, and both values are
        # tables or containers, recursively merge them.
        if (
            key in base
            and isinstance(base[key], (Table, Container))
            and isinstance(patch_val, (Table, Container))
        ):
            merge_toml(base[key], patch_val, id_keys, allow_new_aot)

        # If the key exists in both base and patch, and both values are
        # arrays of tables, merge them based on the specified id_key.
        elif key in base and isinstance(base[key], AoT) and isinstance(patch_val, AoT):
            if key not in id_keys:
                raise ValueError(f"Missing id_key for array of tables '{key}'")
            id_key = id_keys[key]

            for patch_item in patch_val:
                if id_key not in patch_item:
                    raise ValueError(
                        f"Patch entry for '{key}' is missing identity field '{id_key}':"
                        f"{dict(patch_item)}"
                    )
                ident = patch_item[id_key]

                matches = [t for t in base[key] if t.get(id_key) == ident]
                if len(matches) > 1:
                    raise ValueError(
                        f"Ambiguous match: {len(matches)} entries in base '{key}' "
                        f"have {id_key}={ident!r}"
                    )

                if len(matches) == 1:
                    merge_toml(matches[0], patch_item, id_keys, allow_new_aot)
                elif allow_new_aot:
                    base[key].append(patch_item)
                else:
                    raise ValueError(
                        f"No entry in base '{key}' with {id_key}={ident!r}; "
                        f"available: {[t.get(id_key) for t in base[key]]}. "
                        f"Pass allow_new_aot=True to add new entries."
                    )

        else:
            base[key] = patch_val
