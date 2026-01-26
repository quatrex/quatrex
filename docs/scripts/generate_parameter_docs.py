import os
import re

import griffe
import mkdocs_gen_files
from griffe import Attribute, Class
from tabulate import tabulate

OUTPUT_PATH = "user_guide/parameters"


class_template = """{description}"""

entry_template = """
## :octicons-sliders-24: {name}
```python
{name}: {annotation}{default_value}
```

{description}
"""


camel_case_pattern = re.compile("((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")

quatrex_module = griffe.load(
    "quatrex", extensions=griffe.load_extensions("griffe_pydantic")
)
quatrex_config_module = quatrex_module.get_member("core.quatrex_config")


class_members = quatrex_config_module.filter_members(lambda m: isinstance(m, Class))

for class_member in class_members.values():

    # config_section = class_member.name.lower().removesuffix("config")
    config_section = (
        camel_case_pattern.sub(r"_\1", class_member.name)
        .lower()
        .removesuffix("_config")
    )
    attributes = class_member.filter_members(lambda m: isinstance(m, Attribute))
    doc_entries = []
    doc_info = [["Name", "Type", "Default"]]
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

    description = "" if class_member.docstring is None else class_member.docstring.value
    class_doc = class_template.format(description=description)
    class_doc += "\n".join(doc_entries)

    # Write a markdown file for each config class.
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with mkdocs_gen_files.open(f"{OUTPUT_PATH}/{config_section}.md", "w") as f:
        f.write(class_doc)

    with mkdocs_gen_files.open(f"{OUTPUT_PATH}/index.md", "a") as f:
        print(f"### `{class_member.name}`\n", file=f)
        print(tabulate(doc_info, headers="firstrow", tablefmt="github"), file=f)
        print("\n\n", file=f)

    mkdocs_gen_files.set_edit_path(
        f"{OUTPUT_PATH}/{config_section}.md",
        os.path.relpath(__file__, start=mkdocs_gen_files.config.docs_dir),
    )
    mkdocs_gen_files.set_edit_path(
        f"{OUTPUT_PATH}/index.md",
        os.path.relpath(__file__, start=mkdocs_gen_files.config.docs_dir),
    )
