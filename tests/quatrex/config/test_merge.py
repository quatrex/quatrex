# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Tests for the Quatrex configuration merging functionality."""

import pytest
from tomlkit import parse as parse_toml

from quatrex.config import merge_toml


def test_simple_key_merge():
    """Test merging basic key-value pairs at the root level."""
    base = parse_toml("a = 1\nb = 2")
    patch = parse_toml("b = 3\nc = 4")

    merge_toml(base, patch)

    assert base["a"] == 1
    assert base["b"] == 3
    assert base["c"] == 4


def test_nested_table_merge():
    """Test recursive merging of nested tables."""
    base = parse_toml(
        """
        [table]
        a = 1
        b = 2
    """
    )
    patch = parse_toml(
        """
        [table]
        b = 3
        c = 4

        [table_new]
        d = 5
    """
    )

    merge_toml(base, patch)

    assert base["table"]["a"] == 1
    assert base["table"]["b"] == 3
    assert base["table"]["c"] == 4
    assert base["table_new"]["d"] == 5


def test_overwrite_table_with_value():
    """Test that a table can be overwritten by a scalar value."""
    base = parse_toml(
        """
        [my_key]
        nested = 1
    """
    )
    patch = parse_toml(
        """
        my_key = "overwritten"
    """
    )

    merge_toml(base, patch)

    assert base["my_key"] == "overwritten"


def test_aot_merge_success():
    """Test successfully merging an array of tables using an id_key."""
    base = parse_toml(
        """
        [[device.contacts]]
        name = "left"
        fermi_level = 0.0

        [[device.contacts]]
        name = "right"
        fermi_level = 1.0
    """
    )
    patch = parse_toml(
        """
        [[device.contacts]]
        name = "right"
        fermi_level = 2.0
    """
    )

    merge_toml(base, patch, id_keys={"contacts": "name"})

    contacts = base["device"]["contacts"]
    assert len(contacts) == 2
    assert contacts[0]["name"] == "left"
    assert contacts[0]["fermi_level"] == 0.0
    assert contacts[1]["name"] == "right"
    assert contacts[1]["fermi_level"] == 2.0


def test_nested_aot_merge():
    """Test that AoT items themselves are merged recursively."""
    base = parse_toml(
        """
        [[items]]
        id = 1

        [items.settings]
        enabled = true
        mode = "fast"
    """
    )
    patch = parse_toml(
        """
        [[items]]
        id = 1

        [items.settings]
        mode = "slow"
        new_key = "added"
    """
    )

    merge_toml(base, patch, id_keys={"items": "id"})

    settings = base["items"][0]["settings"]
    assert settings["enabled"] is True
    assert settings["mode"] == "slow"
    assert settings["new_key"] == "added"


def test_aot_missing_id_keys_definition():
    """Test ValueError is raised when encountering an AoT without an id_keys entry."""
    base = parse_toml(
        """
        [[items]]
        id = 1
        val = "a"
    """
    )
    patch = parse_toml(
        """
        [[items]]
        id = 1
        val = "b"
    """
    )

    with pytest.raises(ValueError, match="Missing id_key for array of tables 'items'"):
        merge_toml(base, patch)


def test_aot_patch_item_missing_identity_field():
    """Test ValueError is raised when a patch AoT item is missing the identity field."""
    base = parse_toml(
        """
        [[items]]
        id = 1
        val = "a"
    """
    )
    patch = parse_toml(
        """
        [[items]]
        val = "b"
    """
    )

    with pytest.raises(ValueError, match="missing identity field 'id'"):
        merge_toml(base, patch, id_keys={"items": "id"})


def test_aot_ambiguous_match():
    """Test ValueError is raised when multiple items in the base AoT share the same identity."""
    base = parse_toml(
        """
        [[items]]
        id = 1
        val = "a"

        [[items]]
        id = 1
        val = "b"
    """
    )
    patch = parse_toml(
        """
        [[items]]
        id = 1
        val = "c"
    """
    )

    with pytest.raises(ValueError, match="Ambiguous match: 2 entries in base 'items'"):
        merge_toml(base, patch, id_keys={"items": "id"})


def test_aot_no_match_not_allowed():
    """Test ValueError is raised when an AoT patch item isn"t in base and allow_new_aot=False."""
    base = parse_toml(
        """
        [[items]]
        id = 1
        val = "a"
    """
    )
    patch = parse_toml(
        """
        [[items]]
        id = 2
        val = "b"
    """
    )

    with pytest.raises(ValueError, match="No entry in base 'items' with id=2"):
        merge_toml(base, patch, id_keys={"items": "id"}, allow_new_aot=False)


def test_aot_no_match_allowed():
    """Test a new item is appended to the base AoT when allow_new_aot=True."""
    base = parse_toml(
        """
        [[items]]
        id = 1
        val = "a"
    """
    )
    patch = parse_toml(
        """
        [[items]]
        id = 2
        val = "b"
    """
    )

    merge_toml(base, patch, id_keys={"items": "id"}, allow_new_aot=True)

    assert len(base["items"]) == 2
    assert base["items"][0]["id"] == 1
    assert base["items"][1]["id"] == 2
    assert base["items"][1]["val"] == "b"
