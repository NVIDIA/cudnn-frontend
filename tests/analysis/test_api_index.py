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

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")

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
            set(api_index.scan(self.root)),
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
        names = set(api_index.scan(self.root))
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
        names = set(api_index.scan(self.root))
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
        names = set(api_index.scan(self.root))
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
        self.assertTrue({"cudnn.lazy.run", "cudnn.reexport.run"} <= set(api_index.scan(self.root)))

    def test_unknown_exports_and_syntax_errors_fail(self):
        self.write("python/cudnn/new.py", "__all__ = build_exports()")
        with self.assertRaisesRegex(ValueError, "Unsupported export expression"):
            api_index.scan(self.root)
        self.write("python/cudnn/new.py", "def broken(:")
        with self.assertRaises(SyntaxError):
            api_index.scan(self.root)

    def test_utf8_source_and_index(self):
        self.write("python/cudnn/example.py", '"""A naïve example — source is UTF-8."""\ndef café(): pass\n')
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
        names = set(api_index.scan(self.root))
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
        names = set(api_index.scan(self.root))
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
        names = set(api_index.scan(self.root))
        self.assertTrue({f"cudnn.gemm.frost.tile_config.CONFIG_{i}" for i in range(3)} <= names)
        self.write("python/cudnn/gemm/frost/tile_config.py", "raise RuntimeError('catalog failed')")
        with self.assertRaisesRegex(RuntimeError, "catalog failed"):
            api_index.scan(self.root)

    def test_check_detects_additions_removals_and_bad_baselines(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(api_index.main(["--root", str(self.root)]), 1)
            self.assertEqual(api_index.main(["--root", str(self.root), "--write"]), 0)
            self.assertEqual(api_index.main(["--root", str(self.root)]), 0)
            self.write("python/cudnn/new.py", "def new_api(): pass")
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
                "cudnn.engines.FrostGemmEngines",
            }
            <= names
        )


if __name__ == "__main__":
    unittest.main()
