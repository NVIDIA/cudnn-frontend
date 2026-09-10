# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check __all__ exports from allowlisted modules in a built cudnn package."""

import argparse
import difflib
import importlib
import importlib.abc
import importlib.machinery
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def public(name):
    return isinstance(name, str) and name.isidentifier() and not name.startswith("_")


class PackageFinder(importlib.abc.MetaPathFinder):
    """Keep editable installs from substituting another cudnn checkout."""

    def __init__(self, package_root):
        self.package_root = package_root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "cudnn" and not fullname.startswith("cudnn."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path if path is not None else [str(self.package_root.parent)])
        if spec is None:
            raise ModuleNotFoundError(f"{fullname} is absent from {self.package_root}", name=fullname)
        return spec


def scan(package_root, modules_file):
    package_root = package_root.resolve()
    if package_root.name != "cudnn" or not (package_root / "__init__.py").is_file():
        raise ValueError(f"Expected a built cudnn package at {package_root}")
    modules = modules_file.read_text(encoding="utf-8").splitlines()
    if (
        not modules
        or modules != sorted(set(modules))
        or any(name.split(".")[0] != "cudnn" or not all(public(part) for part in name.split(".")) for name in modules)
    ):
        raise ValueError(f"{modules_file}: expected sorted, unique public cudnn module names")
    if "cudnn" in sys.modules:
        raise RuntimeError("Run the scanner in a fresh process before importing cudnn")
    finder = PackageFinder(package_root)
    sys.meta_path.insert(0, finder)
    names = set()
    errors = []

    def record(api, value, ancestors=frozenset()):
        names.add(api)
        if not inspect.isclass(value) or value in ancestors or value.__module__.split(".")[0] != "cudnn":
            return
        for member in dir(value):
            if public(member):
                record(f"{api}.{member}", inspect.getattr_static(value, member), ancestors | {value})

    try:
        for name in modules:
            try:
                module = importlib.import_module(name)
                if not Path(module.__file__).resolve().is_relative_to(package_root):
                    raise ValueError(f"Imported {module.__file__}, outside {package_root}")
                exports = vars(module).get("__all__")
                if not isinstance(exports, (list, tuple)) or not all(isinstance(export, str) and export.isidentifier() for export in exports):
                    raise ValueError(f"{name} must declare __all__ as a list or tuple of names")
                names.add(name)
                for export in exports:
                    if public(export):
                        try:
                            record(f"{name}.{export}", getattr(module, export))
                        except (Exception, SystemExit) as error:
                            errors.append(f"{name}.{export}: {type(error).__name__}: {error}")
            except (Exception, SystemExit) as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
    finally:
        sys.meta_path.remove(finder)
    if errors:
        raise RuntimeError("API inspection failed; no index was written:\n" + "\n".join(errors))
    return sorted(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=ROOT / "build" / "cudnn")
    parser.add_argument("--modules", type=Path, default=Path(__file__).with_name("api_index_modules.txt"))
    parser.add_argument("--index", type=Path, default=Path(__file__).with_suffix(".txt"))
    parser.add_argument("--write", action="store_true", help="Regenerate api_index.txt for review")
    args = parser.parse_args(argv)
    try:
        actual = "\n".join(scan(args.package_root, args.modules)) + "\n"
    except (Exception, SystemExit) as error:
        print(error, file=sys.stderr)
        return 2
    if args.write:
        args.index.write_text(actual, encoding="utf-8")
        print(f"Wrote {len(actual.splitlines())} names to {args.index}")
        return 0
    expected = args.index.read_text(encoding="utf-8") if args.index.exists() else ""
    if expected != actual:
        print("".join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True), fromfile=str(args.index), tofile="runtime API")), end="")
        print("API index mismatch. Review the change, then run test/python/api_index/api_index.py --write with the same built package.")
        return 1
    print(f"API index matches ({len(actual.splitlines())} names).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
