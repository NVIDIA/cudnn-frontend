# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("api_index.py")
SPEC = importlib.util.spec_from_file_location("api_index", SCRIPT)
api_index = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api_index)


class ApiIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("python/cudnn/__init__.py", "")
        self.modules("cudnn")

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")

    def modules(self, *names):
        self.write("tests/analysis/api_index_modules.txt", "\n".join(sorted(names)) + "\n")

    def test_gate_uses_declared_exports_not_incidental_names(self):
        self.modules("cudnn", "cudnn.api")
        self.write("python/cudnn/__init__.py", "import os\nfrom .api import run\n__all__ = ['run']")
        self.write(
            "python/cudnn/api.py",
            "import missing_dependency\n__all__ = ['run']\ndef run(): pass\ndef helper(): pass",
        )
        self.write("python/cudnn/kernels/template.py", "TILE_SIZE = INJECTED_TILE_SIZE\ndef kernel(): pass")
        expected = {"cudnn", "cudnn.run", "cudnn.api", "cudnn.api.run"}
        self.assertEqual(set(api_index.scan(self.root)), expected)
        self.write("python/cudnn/kernels/new.py", "def helper(): pass")
        self.assertEqual(set(api_index.scan(self.root)), expected)
        report = set(api_index.scan(self.root, report=True))
        self.assertTrue({"cudnn.os", "cudnn.api.helper", "cudnn.kernels.template.kernel", "cudnn.kernels.new.helper"} <= report)
        self.write("python/cudnn/unreachable/deep.py", "__all__ = ['entry']\ndef entry(): pass")
        self.assertEqual(set(api_index.scan(self.root)), expected)
        self.modules("cudnn", "cudnn.api", "cudnn.unreachable.deep")
        self.assertEqual(set(api_index.scan(self.root)) - expected, {"cudnn.unreachable.deep", "cudnn.unreachable.deep.entry"})

    def test_gate_follows_exported_classes_and_top_level_lazy_names(self):
        self.write(
            "python/cudnn/__init__.py",
            """
            from ._impl import Builder
            __all__ = ['Builder']
            _LAZY_OPTIONAL_IMPORTS = {'LazyBuilder': ('._impl', 'Builder')}
            def __getattr__(name):
                if name == 'Deferred':
                    from ._impl import Builder as Deferred
                    return Deferred
            """,
        )
        self.write(
            "python/cudnn/_impl.py",
            """
            class Base:
                def inherited(self): pass
            class Builder(Base):
                @property
                def value(self): pass
                def build(self): pass
                def _helper(self): pass
            class Internal:
                def work(self): pass
            """,
        )
        expected = {"cudnn"}
        for name in ("Builder", "LazyBuilder", "Deferred"):
            expected.update(f"cudnn.{name}{suffix}" for suffix in ("", ".build", ".value", ".inherited"))
        self.assertEqual(set(api_index.scan(self.root)), expected)

    def test_empty_all_hides_helpers_and_geometry_is_not_executed(self):
        self.modules("cudnn", "cudnn.empty")
        self.write("python/cudnn/empty.py", "__all__ = []\ndef helper(): pass")
        self.write("python/cudnn/gemm/frost/tile_config.py", "raise RuntimeError('must not execute')")
        self.assertEqual(api_index.scan(self.root), ["cudnn", "cudnn.empty"])

    def test_report_does_not_check_or_write_baseline(self):
        self.write("python/cudnn/internal.py", "def helper(): pass")
        self.write("tests/analysis/api_index.txt", "outdated baseline\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(api_index.main(["--root", str(self.root), "--report"]), 0)
        self.assertIn("cudnn.internal.helper", output.getvalue())
        self.assertEqual((self.root / "tests/analysis/api_index.txt").read_text(), "outdated baseline\n")

    def test_listed_module_without_all_uses_definitions_and_cudnn_reexports(self):
        self.modules("cudnn.integration")
        self.write(
            "python/cudnn/integration.py",
            """
            import os
            from typing import Any
            from ._impl import run
            CACHE = {}
            def install(): pass
            """,
        )
        self.write("python/cudnn/_impl.py", "def run(): pass")
        self.assertEqual(api_index.scan(self.root), ["cudnn.integration", "cudnn.integration.install", "cudnn.integration.run"])

    def test_missing_or_invalid_module_scope_fails(self):
        scope = self.root / "tests/analysis/api_index_modules.txt"
        scope.unlink()
        with self.assertRaises(FileNotFoundError):
            api_index.scan(self.root)
        for content in ("", "cudnn\ncudnn\n", "cudnn.missing\n", "cudnn._private\n"):
            with self.subTest(content=content):
                scope.write_text(content)
                with self.assertRaises(ValueError):
                    api_index.scan(self.root)
        self.assertEqual(api_index.scan(self.root, report=True), ["cudnn"])

    def test_unimportable_modules_and_private_boundaries(self):
        self.write("python/cudnn/__init__.py", "raise RuntimeError('must never import')")
        self.write(
            "python/cudnn/unreachable/nested.py",
            """
            import missing_gpu_library as dependency
            __all__ = ['advertised']
            def advertised(): pass
            async def unlisted(): pass
            def _private(): pass
            CONSTANT = 3
            class Example:
                field: int
                @property
                def value(self): pass
                def method(self):
                    def local(): pass
                def _hidden(self): pass
            """,
        )
        self.write("python/cudnn/_internal/leak.py", "def visible(): pass")
        self.assertEqual(
            set(api_index.scan(self.root, report=True)),
            {
                "cudnn",
                "cudnn.unreachable",
                "cudnn.unreachable.nested",
                "cudnn.unreachable.nested.dependency",
                "cudnn.unreachable.nested.advertised",
                "cudnn.unreachable.nested.unlisted",
                "cudnn.unreachable.nested.CONSTANT",
                "cudnn.unreachable.nested.Example",
                "cudnn.unreachable.nested.Example.field",
                "cudnn.unreachable.nested.Example.value",
                "cudnn.unreachable.nested.Example.method",
            },
        )

    def test_parameterized_templates_are_scanned_without_execution(self):
        self.write(
            "python/cudnn/gemm/frost/kernel_templates/example.py",
            "TILE_SIZE = INJECTED_TILE_SIZE\ndef kernel(): pass\n",
        )
        names = set(api_index.scan(self.root, report=True))
        self.assertTrue(
            {
                "cudnn.gemm.frost.kernel_templates.example",
                "cudnn.gemm.frost.kernel_templates.example.TILE_SIZE",
                "cudnn.gemm.frost.kernel_templates.example.kernel",
            }
            <= names
        )

    def test_lazy_exports_and_private_class_reexports(self):
        self.write(
            "python/cudnn/__init__.py",
            """
            from ._implementation import Builder as PublicBuilder
            Alias = PublicBuilder
            _LAZY_OPTIONAL_IMPORTS = {'LazyBuilder': ('._implementation', 'Builder')}
            __all__ = ['PublicBuilder']
            __all__.append('declared_only')
            __all__ += ['another_declared_name']
            def __getattr__(name):
                if name == 'Deferred':
                    from ._implementation import Builder as Deferred
                    return Deferred
                raise AttributeError(name)
            """,
        )
        self.write(
            "python/cudnn/_implementation.py",
            """
            class Base:
                def inherited(self): pass
            class Builder(Base):
                def build(self): pass
            """,
        )
        names = set(api_index.scan(self.root, report=True))
        for alias in ("PublicBuilder", "Alias", "LazyBuilder", "Deferred"):
            self.assertTrue({f"cudnn.{alias}", f"cudnn.{alias}.build", f"cudnn.{alias}.inherited"} <= names)
        self.assertTrue({"cudnn.declared_only", "cudnn.another_declared_name"} <= names)
        self.assertFalse(any("_implementation" in name for name in names))

    def test_optional_and_conditional_exports(self):
        self.write(
            "python/cudnn/__init__.py",
            """
            symbols_to_import = ['tensor']
            for symbol in symbols_to_import:
                globals()[symbol] = getattr(_pybind_module, symbol)
            for optional in ['optional_kernel']:
                if hasattr(_pybind_module, optional):
                    globals()[optional] = getattr(_pybind_module, optional)
            __all__ = [*symbols_to_import, *(n for n in ('conditional',) if n in globals())]
            """,
        )
        self.write(
            "python/cudnn/conditional.py",
            """
            if platform_supported:
                def platform_a(): pass
            else:
                def platform_b(): pass
            try:
                from missing import optional
            except ImportError:
                pass
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                from missing import annotation_only
            if __name__ == '__main__':
                script_local = 1
            """,
        )
        names = set(api_index.scan(self.root, report=True))
        self.assertTrue(
            {"cudnn.tensor", "cudnn.optional_kernel", "cudnn.conditional.platform_a", "cudnn.conditional.platform_b", "cudnn.conditional.optional"} <= names
        )
        self.assertNotIn("cudnn.conditional.annotation_only", names)
        self.assertNotIn("cudnn.conditional.script_local", names)

    def test_pybind_classes_enums_and_python_extensions(self):
        self.write(
            "python/cudnn/__init__.py",
            """
            symbols_to_import = ['tensor', 'data_type', 'BindingError']
            _pybind_module.tensor.set_dtype = implementation
            """,
        )
        self.write(
            "python/properties.cpp",
            """
            // m.def("commented_out", ignored);
            py::class_<Tensor, std::shared_ptr<Tensor>>(m, "tensor")
                .def("get_dim", &Tensor::get_dim)
                .def("set_dim", [](Tensor& self) { return self; })
                .def_property_readonly("shape", &Tensor::shape)
                .def("_internal", &Tensor::internal);
            py::enum_<Dtype>(m, "data_type")
                .value("FLOAT", Dtype::FLOAT).value("HALF", Dtype::HALF);
            py::register_exception<Error>(m, "BindingError");
            auto doc = R"doc(m.def("fake", ignored))doc";
            """,
        )
        names = set(api_index.scan(self.root, report=True))
        self.assertTrue(
            {
                "cudnn.tensor.get_dim",
                "cudnn.tensor.set_dim",
                "cudnn.tensor.shape",
                "cudnn.tensor.set_dtype",
                "cudnn.data_type.FLOAT",
                "cudnn.data_type.HALF",
                "cudnn.BindingError",
            }
            <= names
        )
        self.assertFalse(any(name.endswith(("._internal", ".fake", ".commented_out")) for name in names))

    def test_export_tables_and_star_imports(self):
        self.write(
            "python/cudnn/lazy.py",
            """
            _LAZY_EXPORTS = {'run': ('cudnn._impl', 'run')}
            __all__ = list(_LAZY_EXPORTS)
            """,
        )
        self.write("python/cudnn/_impl.py", "def run(): pass")
        self.write("python/cudnn/reexport.py", "from .lazy import *")
        self.assertTrue({"cudnn.lazy.run", "cudnn.reexport.run"} <= set(api_index.scan(self.root, report=True)))

    def test_unknown_exports_and_syntax_errors_fail(self):
        self.write("python/cudnn/new.py", "__all__ = build_exports()")
        with self.assertRaisesRegex(ValueError, "Unsupported export expression"):
            api_index.scan(self.root, report=True)
        self.write("python/cudnn/new.py", "def broken(:")
        with self.assertRaises(SyntaxError):
            api_index.scan(self.root, report=True)

    def test_utf8_source_and_index(self):
        self.modules("cudnn", "cudnn.example")
        self.write("python/cudnn/example.py", '"""A naïve example — source is UTF-8."""\ndef café(): pass\n__all__ = ["café"]\n')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(api_index.main(["--root", str(self.root), "--write"]), 0)
            self.assertEqual(api_index.main(["--root", str(self.root)]), 0)
        self.assertIn("cudnn.example.café".encode("utf-8"), (self.root / "tests" / "analysis" / "api_index.txt").read_bytes())

    def test_generated_methods_and_module_name_collision(self):
        self.write("python/cudnn/__init__.py", "from ._builder import Builder\nfrom .graph import graph")
        self.write("python/cudnn/graph.py", "def graph(): pass")
        self.write(
            "python/cudnn/_builder.py",
            """
            class Builder:
                _OPS = {**{op: ('input',) for op in ('add', 'sub')}, 'mul': unknown_value}
            def install():
                def make(op): pass
                for op, spec in Builder._OPS.items():
                    setattr(Builder, op, make(op))
            install()
            """,
        )
        names = set(api_index.scan(self.root, report=True))
        self.assertTrue({"cudnn.Builder.add", "cudnn.Builder.sub", "cudnn.Builder.mul", "cudnn.graph", "cudnn.graph.graph"} <= names)
        self.assertNotIn("cudnn.graph.graph.graph", names)

    def test_lazy_namespace_instances_and_function_locals(self):
        self.write(
            "python/cudnn/__init__.py",
            """
            _SYMBOLS = {'run': ('._impl', 'run')}
            class Namespace:
                def __getattr__(self, name):
                    if name in _SYMBOLS:
                        return load(name)
            API = Namespace()
            def __getattr__(name):
                if name in ('Graph', 'wrapper'):
                    wrapper = importlib.import_module('._impl', __name__)
                    globals()['Graph'] = wrapper.Example
                    globals()['wrapper'] = wrapper
                    return globals()[name]
                if name == 'Deferred':
                    import importlib
                    local = importlib.import_module('._impl', __name__)
                    return local.Example
            """,
        )
        self.write("python/cudnn/_impl.py", "def run(): pass\nclass Example:\n    def build(self): pass")
        names = set(api_index.scan(self.root, report=True))
        self.assertTrue({"cudnn.API.run", "cudnn.Deferred.build", "cudnn.Graph.build"} <= names)
        self.assertNotIn("cudnn.local", names)
        self.assertNotIn("cudnn.importlib", names)

    def test_generated_catalog_is_cpu_only_and_errors_are_not_hidden(self):
        self.write("python/cudnn/frost/occupancy.py", "MAX_CLUSTER_SIZE = 3\nraise RuntimeError('must not import occupancy')")
        self.write(
            "python/cudnn/gemm/frost/tile_config.py",
            """
            from cudnn.frost.occupancy import MAX_CLUSTER_SIZE
            for i in range(MAX_CLUSTER_SIZE):
                globals()[f'CONFIG_{i}'] = i
            """,
        )
        names = set(api_index.scan(self.root, report=True))
        self.assertTrue({f"cudnn.gemm.frost.tile_config.CONFIG_{i}" for i in range(3)} <= names)
        self.write("python/cudnn/gemm/frost/tile_config.py", "raise RuntimeError('catalog failed')")
        with self.assertRaisesRegex(RuntimeError, "catalog failed"):
            api_index.scan(self.root, report=True)

    def test_check_detects_additions_removals_and_bad_baselines(self):
        self.modules("cudnn", "cudnn.new")
        self.write("python/cudnn/new.py", "")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(api_index.main(["--root", str(self.root)]), 1)
            self.assertEqual(api_index.main(["--root", str(self.root), "--write"]), 0)
            self.assertEqual(api_index.main(["--root", str(self.root)]), 0)
            self.write("python/cudnn/new.py", "__all__ = ['new_api']\ndef new_api(): pass")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(api_index.main(["--root", str(self.root)]), 1)
            self.assertIn("+cudnn.new.new_api", output.getvalue())
            api_index.main(["--root", str(self.root), "--write"])
            self.write("python/cudnn/new.py", "")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(api_index.main(["--root", str(self.root)]), 1)
            self.assertIn("-cudnn.new.new_api", output.getvalue())
            api_index.main(["--root", str(self.root), "--write"])
            with (self.root / "tests" / "analysis" / "api_index.txt").open("a", encoding="utf-8") as stream:
                stream.write("cudnn\n")
            self.assertEqual(api_index.main(["--root", str(self.root)]), 1)

    def test_repository_api_index(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--root", str(ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        names = set((ROOT / "tests" / "analysis" / "api_index.txt").read_text(encoding="utf-8").splitlines())
        self.assertTrue(
            {
                "cudnn.pygraph.add",
                "cudnn.pygraph.sdpa",
                "cudnn.pygraph.layernorm",
                "cudnn.tensor.get_dim",
                "cudnn.data_type.FLOAT",
                "cudnn.experimental.ops.swiglu_mlp",
                "cudnn.engines.BaseEngine",
                "cudnn.torch.install",
            }
            <= names
        )
        self.assertNotIn("cudnn.Any", names)
        self.assertFalse(any(".kernel_templates." in name or ".tile_config." in name for name in names))


if __name__ == "__main__":
    unittest.main()
