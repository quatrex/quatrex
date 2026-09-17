# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.


import inspect
import re
from pathlib import Path
from typing import get_args, get_origin

import griffe
from pydantic import BaseModel
from tabulate import tabulate

frontmatter_tag = """---
tags:
    - Parameters
---
"""

class_template = """{description}"""

entry_template = """
## :octicons-sliders-24: {name}
<!-- blacken-docs:off -->
```python
{name}: {annotation}{default_value}
```

{description}
"""

# Regex to convert CamelCase class names to snake_case for the config
# section names
camel_case_pattern = re.compile("((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")

# Set up output paths.
docs_dir = Path(__file__).parent.parent
parameters_dir = docs_dir / "user_guide" / "parameters"
index_path = parameters_dir / "index.md"

# Make the line slightly higher than normal to make it easier to read
# the hierarchy in the docs.
hierarchy_style = """
<style>
pre.toml-overview {
    line-height: 1.75;
}
</style>
"""


def _extract_basemodel(annotation):
    """Recursively extract the BaseModel class from a type annotation.

    Parameters
    ----------
    annotation : type
        The type annotation to inspect.

    Returns
    -------
    type[BaseModel] or None
        The BaseModel class if found, otherwise None.

    """
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            found = _extract_basemodel(arg)
            if found:
                return found
    return None


def get_nested_models(model: type[BaseModel]) -> dict:
    """Get a dictionary of nested Pydantic models for the given model.

    Parameters
    ----------
    model : type[BaseModel]
        The Pydantic model class to inspect.

    Returns
    -------
    dict
        A dictionary where the keys are the field names and the values
        are the nested Pydantic model classes.

    """
    nested = {}
    for name, field in model.model_fields.items():
        found = _extract_basemodel(field.annotation)
        if found:
            nested[name] = found
    return nested


def generate_model_hierarchy(
    model,
    field_name=None,
    max_depth=None,
    _depth=0,
    _seen=None,
):
    """Generate a markdown representation of the hierarchy of Pydantic
    models starting from the given model.

    Parameters
    ----------
    model : type[BaseModel]
        The Pydantic model class to start from.
    field_name : str, optional
        The name of the field in the parent model that refers to this
        model.
    max_depth : int, optional
        The maximum depth to traverse in the hierarchy. If None,
        traverse the entire hierarchy.
    _depth : int, optional
        The current depth in the hierarchy. Used for recursion.
    _seen : set, optional
        A set of models that have already been seen in the hierarchy.
        Used to avoid infinite recursion in case of circular references.

    """

    _seen = _seen if _seen is not None else set()

    if model in _seen:
        return ""

    lines = []
    if field_name is not None:
        label = f"[{field_name}]"
        config_section = (
            camel_case_pattern.sub(r"_\1", model.__name__)
            .lower()
            .removesuffix("_config")
        )
        indentation = "    " * _depth
        lines.append(f"{indentation}<a href='{config_section}/'>{label}</a>")

    if max_depth is not None and _depth >= max_depth:
        return "\n".join(lines)

    _seen = _seen | {model}
    for fname, sub_model in get_nested_models(model).items():
        child = generate_model_hierarchy(
            sub_model,
            field_name=f"{field_name}.{fname}" if field_name else fname,
            max_depth=max_depth,
            _depth=_depth + 1,
            _seen=_seen,
        )
        if child:
            lines.append(child)

    return "\n".join(lines)


def generate_parameter_docs():
    """Generate markdown documentation for all config classes in the
    quatrex.core.config module.

    """

    # Get all classes in the config module using griffe
    quatrex_module = griffe.load(
        "quatrex", extensions=griffe.load_extensions("griffe_pydantic")
    )
    quatrex_config_module = quatrex_module.get_member("core.config")
    class_members = quatrex_config_module.filter_members(
        lambda m: isinstance(m, griffe.Class)
    )

    geometry_config_module = griffe.load("quatrex.electrostatics.geometry_config")
    class_members.update(
        geometry_config_module.filter_members(
            lambda m: isinstance(m, griffe.Class) and not m.name.startswith("_")
        )
    )

    # Generate a markdown file for each config class, and add an entry to
    # the index page.
    for class_member in class_members.values():

        config_section = (
            camel_case_pattern.sub(r"_\1", class_member.name)
            .lower()
            .removesuffix("_config")
        )

        doc_entries = []
        doc_info = [["Name", "Type", "Default"]]

        attributes = class_member.filter_members(
            lambda m: isinstance(m, griffe.Attribute)
        )
        for attribute in attributes.values():
            if "model_config" in attribute.name:
                continue
            name = attribute.name
            annotation = attribute.annotation
            description = (
                "" if attribute.docstring is None else attribute.docstring.value
            )
            doc_entries.append(
                entry_template.format(
                    name=name,
                    annotation=annotation,
                    default_value=f" = {attribute.value}" if attribute.value else "",
                    description=description,
                )
            )
            default = f"`{attribute.value}`" if attribute.value else "-"
            name_with_link = f"[`{name}`]({config_section}.md#{name.lower()})"

            annotation = f"`{annotation}`".replace(" | ", "` or `").strip()
            doc_info.append([name_with_link, annotation, default])

        # Append an entry to the index page for this config class, with
        # links and info for each parameter.
        with open(index_path, "a") as f:
            print(f"### [`{class_member.name}`](./{config_section}/)\n", file=f)
            short_description = (
                ""
                if class_member.docstring is None
                else class_member.docstring.value.splitlines()[0]
            )
            print(f"{short_description}\n", file=f)
            print(tabulate(doc_info, headers="firstrow", tablefmt="github"), file=f)
            print("\n\n", file=f)

        description = (
            "" if class_member.docstring is None else class_member.docstring.value
        )
        class_doc = class_template.format(description=description)
        class_doc += "\n".join(doc_entries)

        # Write a markdown file for each config class.
        with open(parameters_dir / f"{config_section}.md", "w") as f:
            print(frontmatter_tag, file=f)
            f.write(class_doc)


if __name__ == "__main__":
    # Set up the index page for the parameters section.
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        print(frontmatter_tag, file=f)
        print("# Simulation Parameters\n", file=f)
        print("\n\n", file=f)

    from quatrex.core.config import QuatrexConfig

    hierarchy = generate_model_hierarchy(model=QuatrexConfig, _depth=-1)

    with open(index_path, "a") as f:
        print("## Configuration Hierarchy\n", file=f)
        print(hierarchy_style, file=f)
        print(f'<pre class="toml-overview"><code>{hierarchy}\n</code></pre>', file=f)
        print("---\n\n", file=f)

        print("## Parameter Overview\n", file=f)

    generate_parameter_docs()
