# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory Python API declarations without importing the package. See README.md."""

import argparse
import ast
import difflib
import re
import sys
import types
from pathlib import Path


def public(name):
    return bool(name) and all(part.isidentifier() and not part.startswith("_") for part in name.split("."))


def literal(node, values):
    """Evaluate export containers only; conditions contribute every possible name."""
    if node is None:
        raise ValueError("No export expression")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(f"Unresolved export expression: {ast.unparse(node)}")
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        result = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                result.extend(literal(item.value, values))
            else:
                result.append(literal(item, values))
        return tuple(result) if isinstance(node, ast.Tuple) else result
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                result.update(literal(value, values))
            else:
                result[literal(key, values)] = literal(value, values)
        return result
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left, values) + literal(node.right, values)
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)) and len(node.generators) == 1:
        generator = node.generators[0]
        if isinstance(generator.target, ast.Name):
            return [literal(node.elt, {**values, generator.target.id: item}) for item in literal(generator.iter, values)]
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        generator = node.generators[0]
        if isinstance(generator.target, ast.Name):
            return {
                literal(node.key, {**values, generator.target.id: item}): literal(node.value, {**values, generator.target.id: item})
                for item in literal(generator.iter, values)
            }
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"list", "tuple", "set", "sorted"} and len(node.args) == 1:
            return list(literal(node.args[0], values))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "keys" and not node.args:
            return list(literal(node.func.value, values))
    raise ValueError(f"Unsupported export expression: {ast.unparse(node)}")


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def scope_nodes(body):
    for node in body:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) in {"TYPE_CHECKING", "typing.TYPE_CHECKING", "__name__ == '__main__'"}:
            yield from scope_nodes(node.orelse)
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from scope_nodes(getattr(node, field, []))
        for handler in getattr(node, "handlers", []):
            yield from scope_nodes(handler.body)


class Inventory:
    def __init__(self):
        self.members = {}
        self.aliases = {}
        self.bases = {}
        self.exports = {}
        self.stars = []
        self.modules = set()
        self.instances = []

    def add(self, scope, name, target=None):
        self.members.setdefault(scope, set()).add(name)
        if target and target != f"{scope}.{name}":
            self.aliases.setdefault(f"{scope}.{name}", set()).add(target)

    def resolve(self, name, seen=frozenset()):
        if name in seen:
            return set()
        result = {name}
        for target in self.aliases.get(name, ()):
            result.update(self.resolve(target, seen | {name}))
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            for target in self.resolve(parent, seen | {name}) if parent not in self.modules else ():
                if target != parent:
                    result.update(self.resolve(f"{target}.{leaf}", seen | {name}))
        return result

    def reference(self, node, scope):
        name = dotted(node)
        if name:
            return name if name.startswith("cudnn.") or name == "cudnn" else f"{scope}.{name}"
        if isinstance(node, ast.Call) and dotted(node.func) in {"importlib.import_module", "import_module"}:
            if node.args and isinstance(node.args[0], ast.Constant):
                return self.absolute(node.args[0].value, scope)
        if isinstance(node, ast.Call) and dotted(node.func) == "getattr" and len(node.args) >= 2:
            if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                parent = self.reference(node.args[0], scope)
                return f"{parent}.{node.args[1].value}" if parent else None
        return None

    def absolute(self, name, package):
        level = len(name) - len(name.lstrip("."))
        if not level:
            return name
        return ".".join(package.split(".")[: len(package.split(".")) - level + 1] + [name.lstrip(".")]).rstrip(".")

    def python_scope(self, body, scope, package, values=None):
        values = dict(values or {})
        for node in scope_nodes(body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.add(scope, node.name)
                if node.name == "__getattr__":
                    self.lazy_getattr(node, scope, package, values)
                continue
            if isinstance(node, ast.ClassDef):
                self.add(scope, node.name)
                name = f"{scope}.{node.name}"
                self.bases[name] = [self.reference(base, scope) for base in node.bases if self.reference(base, scope)]
                self.python_scope(node.body, name, package, values)
                continue
            if isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                module = self.absolute("." * node.level + (node.module or ""), package)
                for item in node.names:
                    if item.name == "*":
                        self.stars.append((scope, module))
                    else:
                        self.add(scope, item.asname or item.name, f"{module}.{item.name}")
            elif isinstance(node, ast.Import):
                for item in node.names:
                    self.add(scope, item.asname or item.name.split(".")[0], item.name if item.asname else item.name.split(".")[0])
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        self.add(scope, target.id)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.add(scope, target.id, self.reference(node.value, scope))
                        if isinstance(node.value, ast.Call):
                            self.instances.append((scope, target.id, self.reference(node.value.func, scope)))
                        try:
                            values[target.id] = literal(node.value, values)
                        except ValueError:
                            if target.id == "__all__":
                                raise
                        if target.id == "__all__":
                            self.exports[scope] = set(values[target.id])
                    elif isinstance(target, ast.Attribute):
                        parent = self.reference(target.value, scope)
                        if parent:
                            self.add(parent, target.attr, self.reference(node.value, scope))
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for item in ast.walk(target):
                            if isinstance(item, ast.Name):
                                self.add(scope, item.id)
                    elif isinstance(target, ast.Subscript) and ast.unparse(target.value) == "globals()":
                        try:
                            name = literal(target.slice, values)
                        except ValueError:
                            continue
                        if isinstance(name, str):
                            self.add(scope, name, self.reference(node.value, scope))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
                values["__all__"] += literal(node.value, values)
                self.exports[scope] = set(values["__all__"])
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if dotted(call.func) in {"__all__.append", "__all__.extend"}:
                    names = literal(call.args[0], values)
                    values["__all__"].extend([names] if call.func.attr == "append" else names)
                    self.exports[scope] = set(values["__all__"])

        for name in self.exports.get(scope, ()):
            self.add(scope, name)
        for variable in ("_LAZY_OPTIONAL_IMPORTS", "_LAZY_EXPORTS", "_SYMBOLS") if scope in self.modules else ():
            for name, target in values.get(variable, {}).items():
                if isinstance(target, (list, tuple)) and len(target) == 2:
                    module, attr = target
                    self.add(scope, name, self.absolute(module, package) + (f".{attr}" if attr else ""))
        if scope == "cudnn":
            self.aliases["cudnn._pybind_module"] = {"cudnn._compiled_module"}
            for name in values.get("symbols_to_import", []):
                self.add(scope, name, f"cudnn._compiled_module.{name}")
            for node in scope_nodes(body):
                if isinstance(node, ast.For) and any(isinstance(child, ast.Subscript) and ast.unparse(child.value) == "globals()" for child in ast.walk(node)):
                    for name in literal(node.iter, values):
                        self.add(scope, name, f"cudnn._compiled_module.{name}")

    def lazy_getattr(self, node, scope, package, values):
        parameter = node.args.args[-1].arg
        for branch in ast.walk(node):
            if not isinstance(branch, ast.If) or not isinstance(branch.test, ast.Compare):
                continue
            test = branch.test
            if dotted(test.left) != parameter or not isinstance(test.ops[0], (ast.Eq, ast.In)):
                continue
            try:
                names = literal(test.comparators[0], values)
            except ValueError:
                continue
            names = [names] if isinstance(names, str) else names
            imports = {}
            published = {}
            for child in scope_nodes(branch.body):
                if isinstance(child, ast.ImportFrom):
                    module = self.absolute("." * child.level + (child.module or ""), package)
                    imports.update({item.asname or item.name: f"{module}.{item.name}" for item in child.names})
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            imports[target.id] = self.reference(child.value, package)
                        elif isinstance(target, ast.Subscript) and ast.unparse(target.value) == "globals()":
                            name = literal(target.slice, values)
                            value = dotted(child.value)
                            if value:
                                first, *rest = value.split(".")
                                resolved = imports.get(first) or self.reference(ast.Name(id=first), scope)
                                published[name] = resolved + ("." + ".".join(rest) if rest else "")
            returns = [child.value for child in scope_nodes(branch.body) if isinstance(child, ast.Return)]
            for name in names:
                if not isinstance(name, str):
                    continue
                target = published.get(name) or imports.get(name)
                if len(names) == 1 and returns:
                    value = dotted(returns[0])
                    if value:
                        first, *rest = value.split(".")
                        target = imports.get(first) or self.reference(ast.Name(id=first), scope)
                        if target and rest:
                            target += "." + ".".join(rest)
                self.add(scope, name, target)

    def generated_methods(self, body, module):
        tables = {}

        def keys(node):
            if isinstance(node, ast.Dict):
                result = []
                for key, value in zip(node.keys, node.values):
                    result.extend(keys(value) if key is None else [literal(key, {})])
                return result
            return list(literal(node, {}))

        def dictionaries(nodes, prefix=""):
            for node in scope_nodes(nodes):
                if isinstance(node, ast.ClassDef):
                    dictionaries(node.body, prefix + node.name + ".")
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            tables[prefix + target.id] = node.value

        dictionaries(body)
        called = {dotted(node.value.func) for node in scope_nodes(body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)}
        for function in body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or function.name not in called:
                continue
            for loop in scope_nodes(function.body):
                if not isinstance(loop, ast.For):
                    continue
                iterator = loop.iter
                if isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Attribute) and iterator.func.attr in {"items", "keys"}:
                    iterator = iterator.func.value
                key = loop.target.elts[0] if isinstance(loop.target, ast.Tuple) else loop.target
                for node in scope_nodes(loop.body):
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                        call = node.value
                        if dotted(call.func) == "setattr" and len(call.args) == 3 and dotted(call.args[1]) == dotted(key):
                            table = dotted(iterator)
                            if table not in tables:
                                raise ValueError(f"Unresolved generated API table: {table}")
                            parent = self.reference(call.args[0], module)
                            for name in keys(tables[table]):
                                self.add(parent, name)

    def generated_exports(self, root):
        """Resolve the engine manifest and the CPU-only geometry catalog."""
        manifest = root / "python/cudnn/engines/manifest.py"
        if manifest.exists():
            for node in ast.walk(ast.parse(manifest.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and dotted(node.func) == "EngineFamily":
                    fields = {keyword.arg: keyword.value for keyword in node.keywords}
                    module = literal(fields.get("module", node.args[2] if len(node.args) > 2 else None), {})
                    factory = literal(fields.get("factory", node.args[3] if len(node.args) > 3 else None), {})
                    self.add("cudnn.engines", factory, f"{module}.{factory}")

        catalog = root / "python/cudnn/gemm/frost/tile_config.py"
        if not catalog.exists():
            return
        tree = ast.parse(catalog.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "cudnn.frost.occupancy":
                constants = {}
                source = root / "python/cudnn/frost/occupancy.py"
                for assignment in ast.parse(source.read_text(encoding="utf-8")).body:
                    if isinstance(assignment, ast.Assign) and isinstance(assignment.value, ast.Constant):
                        constants.update({target.id: assignment.value.value for target in assignment.targets if isinstance(target, ast.Name)})
                index = tree.body.index(node)
                tree.body[index : index + 1] = [
                    ast.copy_location(
                        ast.Assign(targets=[ast.Name(id=name.asname or name.name, ctx=ast.Store())], value=ast.Constant(constants[name.name])), node
                    )
                    for name in node.names
                ]
        module = types.ModuleType("api_index_geometry_catalog")
        sys.modules[module.__name__] = module
        try:
            exec(compile(ast.fix_missing_locations(tree), str(catalog), "exec"), module.__dict__)
            for name in vars(module):
                if public(name):
                    self.add("cudnn.gemm.frost.tile_config", name)
        finally:
            del sys.modules[module.__name__]

    def cpp(self, paths):
        """Read literal pybind declarations, including chained class/enum members."""
        token_pattern = r'R"(?P<delimiter>[^ (\\\t\r\n]*)\(.*?\)(?P=delimiter)"|"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*|[^\s]'
        files = []
        types = {}
        for path in paths:
            tokens = [m.group() for m in re.finditer(token_pattern, path.read_text(encoding="utf-8"), re.S) if not m.group().startswith(("//", "/*"))]
            pairs, stack = {}, []
            for i, token in enumerate(tokens):
                if token == "(":
                    stack.append(i)
                elif token == ")" and stack:
                    start = stack.pop()
                    pairs[i] = start
            constructors, receivers = {}, {"m": "cudnn._compiled_module"}
            for i, token in enumerate(tokens):
                if token not in {"class_", "enum_", "register_exception"} or tokens[i + 1 : i + 2] != ["<"]:
                    continue
                end, depth = i + 2, 1
                while end < len(tokens) and depth:
                    depth += (tokens[end] == "<") - (tokens[end] == ">")
                    end += 1
                variable = None
                if tokens[end : end + 1] != ["("]:
                    variable = tokens[end] if end < len(tokens) else None
                    end += 1
                if tokens[end : end + 3] != ["(", "m", ","] or not tokens[end + 3].startswith('"'):
                    continue
                name = ast.literal_eval(tokens[end + 3])
                self.add("cudnn._compiled_module", name)
                qualified = f"cudnn._compiled_module.{name}"
                constructors[end] = qualified
                types[tokens[i + 2]] = qualified
                if variable:
                    receivers[variable] = qualified
            files.append((tokens, pairs, constructors, receivers))
        for tokens, pairs, constructors, receivers in files:
            for i, token in enumerate(tokens):
                if token == "class_" and tokens[i + 1 : i + 2] == ["<"]:
                    if tokens[i + 3 : i + 5] == [">", "&"] and tokens[i + 2] in types:
                        receivers[tokens[i + 5]] = types[tokens[i + 2]]
            for i, token in enumerate(tokens):
                if not (token == "value" or token == "def" or token.startswith("def_")):
                    continue
                if tokens[i - 1] != "." or tokens[i + 1 : i + 2] != ["("] or not tokens[i + 2].startswith('"'):
                    continue
                receiver = i - 2
                while tokens[receiver] == ")" and receiver in pairs:
                    start = pairs[receiver]
                    if start in constructors:
                        break
                    receiver = start - 3
                parent = constructors.get(pairs.get(receiver)) or receivers.get(tokens[receiver])
                if parent:
                    self.add(parent, ast.literal_eval(tokens[i + 2]))

    def collect(self):
        for scope, name, target in self.instances:
            if target and any(resolved in self.bases for resolved in self.resolve(target)):
                self.add(scope, name, target)
        for scope, module in self.stars:
            if module not in self.modules:
                raise ValueError(f"Cannot statically resolve star import from {module} in {scope}")
            for name in self.exports.get(module, self.members.get(module, ())):
                if public(name):
                    self.add(scope, name, f"{module}.{name}")
        for scope, names in list(self.members.items()):
            if scope not in self.modules:
                for target in self.resolve(scope):
                    self.members.setdefault(target, set()).update(names)

        def children(target, seen=frozenset()):
            if target in seen:
                return set()
            result = set(self.members.get(target, ()))
            for base in self.bases.get(target, ()):
                for resolved in self.resolve(base):
                    result.update(children(resolved, seen | {target}))
            return result

        result = set()

        def visit(api, target, seen):
            for resolved in self.resolve(target):
                if resolved in seen:
                    continue
                for name in children(resolved):
                    if public(name):
                        child = f"{api}.{name}"
                        result.add(child)
                        next_target = f"{resolved}.{name}"
                        if not any(t in self.modules for t in self.resolve(next_target)):
                            visit(child, next_target, seen | {resolved})

        for module in sorted(self.modules):
            if public(module):
                result.add(module)
                visit(module, module, set())
        return sorted(result)


def scan(root):
    inventory = Inventory()
    package = root / "python" / "cudnn"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"No cudnn package at {package}")
    sources = []
    for path in sorted(package.rglob("*.py")):
        parts = path.relative_to(package.parent).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if is_package else parts)
        inventory.modules.add(module)
        parent = module if is_package else module.rsplit(".", 1)[0]
        sources.append((path, module, parent))
        for i in range(1, len(module.split("."))):
            inventory.modules.add(".".join(module.split(".")[:i]))
    for module in sorted(inventory.modules):
        if "." in module:
            parent, leaf = module.rsplit(".", 1)
            inventory.add(parent, leaf)
    inventory.cpp(sorted((root / "python").glob("*.cpp")) + sorted((root / "python" / "pygraph").glob("*.cpp")))
    for path, module, parent in sources:
        try:
            body = ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
            inventory.python_scope(body, module, parent)
            inventory.generated_methods(body, module)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
    inventory.generated_exports(root)
    return inventory.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--write", action="store_true", help="Regenerate api_index.txt for review")
    args = parser.parse_args(argv)
    index = args.root / "api_index.txt"
    actual = "\n".join(scan(args.root)) + "\n"
    if args.write:
        index.write_text(actual, encoding="utf-8")
        print(f"Wrote {len(actual.splitlines())} names to {index}")
        return 0
    expected = index.read_text(encoding="utf-8") if index.exists() else ""
    if expected != actual:
        print("".join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True), fromfile="api_index.txt", tofile="scanned API")), end="")
        print("API index mismatch. Review the API change, then run python3 tests/analysis/api_index.py --write.")
        return 1
    print(f"API index matches ({len(actual.splitlines())} names).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
