import ast
from importlib.util import resolve_name
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig

ROOT = Path(__file__).resolve().parents[1]


def _api_definition(module: str, name: str) -> tuple[str, ast.FunctionDef | ast.ClassDef]:
    module_path = ROOT.joinpath(*module.split("."))
    source = module_path.with_suffix(".py")
    if not source.is_file():
        source = module_path / "__init__.py"
    package = module if source.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return f"{module}.{name}", node
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    target = resolve_name("." * node.level + (node.module or ""), package)
                    return _api_definition(target, alias.name)
    raise ValueError(f"No source definition found for {module}.{name}")


def on_pre_build(config: MkDocsConfig) -> None:
    tree = ast.parse((ROOT / "oops" / "__init__.py").read_text(encoding="utf-8"))
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    references = ["::: oops\n    options:\n        members: false\n        show_root_heading: false"]
    for name in exports:
        if name.startswith("_"):
            continue
        path, definition = _api_definition("oops", name)
        reference = f"::: {path}"
        if (
            isinstance(definition, ast.ClassDef)
            and definition.bases
            and not any(isinstance(member, ast.FunctionDef) and member.name == "__init__" for member in definition.body)
        ):
            reference += (
                "\n    options:\n        docstring_options:\n"
                "            returns_multiple_items: false\n            warn_unknown_params: false"
            )
        references.append(reference)
    api = "# API\n\n" + "\n\n".join(references)
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pages = {
        "index.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "index.zh.md": (ROOT / "README.zh.md").read_text(encoding="utf-8"),
        "LICENSE": license_text,
        "license.md": f"# License\n\n{license_text}",
        "api.md": api + "\n",
    }
    for name, content in pages.items():
        destination = Path(config.docs_dir) / name
        if not destination.exists() or destination.read_text(encoding="utf-8") != content:
            destination.write_text(content, encoding="utf-8")
