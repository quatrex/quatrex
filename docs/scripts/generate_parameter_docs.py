# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import re
from pathlib import Path

import griffe
from tabulate import tabulate

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


# Get all classes in the config module using griffe
quatrex_module = griffe.load(
    "quatrex", extensions=griffe.load_extensions("griffe_pydantic")
)
quatrex_config_module = quatrex_module.get_member("core.config")
class_members = quatrex_config_module.filter_members(
    lambda m: isinstance(m, griffe.Class)
)

# Set up the index page for the parameters section.
index_path.parent.mkdir(parents=True, exist_ok=True)
with open(index_path, "w") as f:
    print("# Simulation Parameters\n", file=f)
    print("\n\n", file=f)

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

    attributes = class_member.filter_members(lambda m: isinstance(m, griffe.Attribute))
    for attribute in attributes.values():
        if "model_config" in attribute.name:
            continue
        name = attribute.name
        annotation = attribute.annotation
        description = "" if attribute.docstring is None else attribute.docstring.value
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
        print(f"### `{class_member.name}`\n", file=f)
        print(tabulate(doc_info, headers="firstrow", tablefmt="github"), file=f)
        print("\n\n", file=f)

    description = "" if class_member.docstring is None else class_member.docstring.value
    class_doc = class_template.format(description=description)
    class_doc += "\n".join(doc_entries)

    # Write a markdown file for each config class.
    with open(parameters_dir / f"{config_section}.md", "w") as f:
        f.write(class_doc)
