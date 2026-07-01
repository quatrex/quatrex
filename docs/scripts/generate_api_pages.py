# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
src_dir = root_dir / "src"
api_dir = root_dir / "docs" / "api"


for path in sorted(src_dir.rglob("*.py")):
    module_path = path.relative_to(src_dir).with_suffix("")
    if "config" in module_path.parts or "cli" in module_path.parts:
        continue
    doc_path = path.relative_to(src_dir).with_suffix(".md")
    full_doc_path = api_dir / doc_path

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    elif parts[-1] == "__main__":
        continue

    full_doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_doc_path, "w") as fd:
        print("# " + parts[-1], file=fd)
        print("\n\n", file=fd)
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)
