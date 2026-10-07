"""Inventory authored source, parse every component and check documented release identities."""

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for directory in ["src", "scripts", "tests"]:
        for path in sorted((ROOT / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            rows.append(
                dict(
                    path=str(path.relative_to(ROOT)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    lines=len(path.read_text().splitlines()),
                    module_docstring=ast.get_docstring(tree),
                    public_definitions=[
                        node.name
                        for node in tree.body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                        and not node.name.startswith("_")
                    ],
                )
            )
    init = (ROOT / "src/learning_lab/__init__.py").read_text()
    version = re.search(r'__version__ = "([^"]+)"', init).group(1)
    assert f'version = "{version}"' in (ROOT / "pyproject.toml").read_text()
    assert f"version: {version}\n" in (ROOT / "CITATION.cff").read_text()
    report = dict(
        status="passed",
        scope="AST inventory and release-identity checks; mathematical and empirical checks are separate in validation.json and implementation audit",
        version=version,
        authored_python_components=len(rows),
        source_modules=sum(r["path"].startswith("src/") for r in rows),
        components=rows,
    )
    (ROOT / "docs/component_inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Parsed", len(rows), "authored components; matched package/citation version", version)


if __name__ == "__main__":
    main()
