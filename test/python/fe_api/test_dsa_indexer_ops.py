# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the DSA indexer logical operations."""

import ast
from enum import Enum, auto
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0

_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_DSA_ROOT = _CUDNN_ROOT / "deepseek_sparse_attention"
_PACKAGE = "cudnn_dsa_indexer_op_test"


class _DataType(Enum):
    NOT_SET = auto()
    HALF = auto()
    BFLOAT16 = auto()
    FLOAT = auto()
    INT32 = auto()
    INT64 = auto()


class _DType:
    def __init__(self, name: str, itemsize: int):
        self.name = name
        self.itemsize = itemsize

    def __repr__(self):
        return self.name


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[:-1] == (Ellipsis,) and isinstance(key[-1], slice):
            return _Array((*self.shape[:-1], key[-1].stop), self.dtype)
        if isinstance(key, tuple) and len(key) == self.ndim and all(isinstance(value, slice) for value in key):
            shape = []
            for extent, value in zip(self.shape, key):
                start = 0 if value.start is None else value.start
                stop = extent if value.stop is None else min(value.stop, extent)
                step = 1 if value.step is None else value.step
                shape.append(max(0, (stop - start + step - 1) // step))
            return _Array(shape, self.dtype)
        raise AssertionError(f"Unexpected array slice {key!r}")


class _TensorSpec:
    def __init__(self, *, layout=None, mode=None, divisibility=None, **_kwargs):
        self.layout = layout
        self.mode = mode
        self.divisibility = divisibility


class DsaIndexerOpContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        dsa_name = f"{_PACKAGE}.deepseek_sparse_attention"
        dsa = types.ModuleType(dsa_name)
        dsa.__path__ = [str(_DSA_ROOT)]
        dsa.__package__ = dsa_name
        dsa.__spec__ = ModuleSpec(dsa_name, loader=None, is_package=True)
        sys.modules[dsa_name] = dsa

        for operation in ("indexer_forward", "indexer_top_k"):
            name = f"{dsa_name}.{operation}"
            module = types.ModuleType(name)
            module.__path__ = [str(_DSA_ROOT / operation)]
            module.__package__ = name
            module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = module

        cls.tensor = importlib.import_module(f"{_PACKAGE}._tensor_desc")
        importlib.import_module(f"{_PACKAGE}._op")
        cls.forward = importlib.import_module(f"{dsa_name}.indexer_forward.op")
        cls.topk = importlib.import_module(f"{dsa_name}.indexer_top_k.op")

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)

    def desc(self, shape, dtype, name="", stride_order=None):
        return self.tensor.make_compact_tensor_desc(
            dtype=dtype,
            shape=tuple(shape),
            stride_order=stride_order,
            name=name,
        )

    def test_fixed_forward_complete_signature(self):
        operation = self.forward.IndexerForwardOp(
            q=self.desc((2, 16, 32, 128), _DataType.BFLOAT16),
            k=self.desc((2, 17, 1, 128), _DataType.BFLOAT16),
            weight=self.desc((2, 16, 32), _DataType.BFLOAT16),
            output=self.desc((2, 16, 20), _DataType.FLOAT),
            q_causal_offsets=self.desc((2,), _DataType.INT32),
            ratio=4,
            target_compute_capability=100,
        )
        self.assertTrue(operation.check_support())
        self.assertFalse(operation.is_varlen)
        self.assertEqual((operation.s_q, operation.s_k), (16, 17))
        self.assertEqual(operation.qhead_per_kv_head, 32)

    def test_varlen_forward_complete_signature(self):
        operation = self.forward.IndexerForwardOp(
            q=self.desc((31, 64, 128), _DataType.BFLOAT16),
            k=self.desc((47, 1, 128), _DataType.BFLOAT16),
            weight=self.desc((31, 64), _DataType.BFLOAT16),
            output=self.desc((31, 20), _DataType.FLOAT),
            cu_seqlens_q=self.desc((3,), _DataType.INT32),
            cu_seqlens_k=self.desc((3,), _DataType.INT32),
            q_causal_offsets=self.desc((2,), _DataType.INT32),
            max_seqlen_q=16,
            max_seqlen_k=17,
            target_compute_capability=90,
        )
        self.assertTrue(operation.check_support())
        self.assertTrue(operation.is_varlen)
        self.assertEqual(operation.batch_size, 2)

    def test_forward_rejects_invalid_output_and_optional_signature(self):
        with self.assertRaisesRegex(ValueError, "leading shape"):
            self.forward.IndexerForwardOp(
                q=self.desc((1, 8, 32, 128), _DataType.BFLOAT16),
                k=self.desc((1, 5, 1, 128), _DataType.BFLOAT16),
                weight=self.desc((1, 8, 32), _DataType.BFLOAT16),
                output=self.desc((1, 8, 5), _DataType.FLOAT),
            ).check_support()

        oversized_output = self.forward.IndexerForwardOp(
            q=self.desc((1, 8, 32, 128), _DataType.BFLOAT16),
            k=self.desc((1, 5, 1, 128), _DataType.BFLOAT16),
            weight=self.desc((1, 8, 32), _DataType.BFLOAT16),
            output=self.desc((1, 8, 12), _DataType.FLOAT),
        )
        self.assertTrue(oversized_output.check_support())

        with self.assertRaisesRegex(ValueError, "requires both"):
            self.forward.IndexerForwardOp(
                q=self.desc((8, 32, 128), _DataType.BFLOAT16),
                k=self.desc((8, 1, 128), _DataType.BFLOAT16),
                weight=self.desc((8, 32), _DataType.BFLOAT16),
                output=self.desc((8, 8), _DataType.FLOAT),
                cu_seqlens_q=self.desc((2,), _DataType.INT32),
                max_seqlen_q=8,
                max_seqlen_k=8,
            ).check_support()

    def test_forward_requires_the_kernel_vector_axis_to_be_contiguous(self):
        operation = self.forward.IndexerForwardOp(
            q=self.desc(
                (1, 8, 32, 128),
                _DataType.BFLOAT16,
                stride_order=(2, 3, 1, 0),
            ),
            k=self.desc((1, 5, 1, 128), _DataType.BFLOAT16),
            weight=self.desc((1, 8, 32), _DataType.BFLOAT16),
            output=self.desc((1, 8, 8), _DataType.FLOAT),
            ratio=1,
        )
        with self.assertRaisesRegex(ValueError, "canonical D axis contiguous"):
            operation.check_support()

    def make_topk(self, *, return_val=True, **overrides):
        input_values = self.desc((4, 64), _DataType.FLOAT)
        arguments = {
            "input_values": input_values,
            "seq_lens": self.desc((4,), _DataType.INT32),
            "output_indices": self.desc((4, 8), _DataType.INT32),
            "output_values": self.desc((4, 8), _DataType.FLOAT) if return_val else None,
            "workspace": self.desc((4, 2, 64), _DataType.INT32),
            "top_k": 8,
            "return_val": return_val,
        }
        arguments.update(overrides)
        return self.topk.IndexerTopKOp(**arguments)

    def test_topk_complete_signature_with_and_without_values(self):
        for return_val in (True, False):
            with self.subTest(return_val=return_val):
                operation = self.make_topk(return_val=return_val)
                self.assertTrue(operation.check_support())
                self.assertEqual(operation.buffer_count, 2)
                self.assertEqual(operation.max_num_cols, 64)

    def test_topk_rejects_invalid_rows_workspace_and_copy_width(self):
        cases = (
            ({"seq_lens": self.desc((2,), _DataType.INT32)}, "must equal"),
            ({"workspace": self.desc((4, 1, 64), _DataType.INT32)}, "workspace must"),
            ({"num_copy_bits": 24}, "power-of-two"),
            ({"top_k": 65}, "num_cols=64"),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_topk(**overrides).check_support()


class DsaIndexerImportContractTest(unittest.TestCase):
    @staticmethod
    def top_level_imports(path: Path) -> tuple[str, ...]:
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append("." * node.level + (node.module or ""))
        return tuple(imports)

    def test_operation_packages_and_ops_are_framework_lazy(self):
        for operation in ("indexer_forward", "indexer_top_k"):
            root = _DSA_ROOT / operation
            for filename in ("__init__.py", "op.py"):
                imports = self.top_level_imports(root / filename)
                self.assertFalse(
                    any(name == framework or name.startswith(f"{framework}.") for name in imports for framework in ("torch", "jax", "cutlass", "cuda")),
                    f"{operation}/{filename}: {imports}",
                )

    def test_jax_reachable_kernels_do_not_require_torch(self):
        files = (
            _DSA_ROOT / "indexer_forward" / "jax.py",
            _DSA_ROOT / "indexer_forward" / "indexer_fwd_sm90.py",
            _DSA_ROOT / "indexer_forward" / "indexer_fwd_sm100.py",
            _DSA_ROOT / "indexer_top_k" / "jax.py",
            _DSA_ROOT / "indexer_top_k" / "indexer_top_k_decode_varlen.py",
            _DSA_ROOT / "indexer_top_k" / "indexer_top_k_varlen_util.py",
            _DSA_ROOT / "indexer_top_k" / "local_to_global_dsl.py",
            _DSA_ROOT / "indexer_top_k" / "compactify.py",
        )
        for path in files:
            imports = self.top_level_imports(path)
            self.assertNotIn("torch", imports, str(path))

    def test_torch_indexer_forward_passes_all_optional_kernel_arguments(self):
        path = _DSA_ROOT / "indexer_forward" / "api.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        compile_method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compile" and node.lineno < 220)
        compile_call = next(
            node for node in ast.walk(compile_method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
        )
        self.assertEqual(sum(isinstance(argument, ast.Constant) and argument.value is None for argument in compile_call.args), 3)

        tensor_api = next(node for node in ast.walk(compile_method) if isinstance(node, ast.FunctionDef) and node.name == "tensor_api")
        launch_call = next(node for node in ast.walk(tensor_api) if isinstance(node, ast.Call))
        self.assertEqual(sum(isinstance(argument, ast.Constant) and argument.value is None for argument in launch_call.args), 3)


class DsaIndexerJaxAdapterContractTest(unittest.TestCase):
    _PACKAGE = "cudnn_dsa_indexer_jax_test"

    @classmethod
    def setUpClass(cls) -> None:
        cls.float16 = _DType("float16", 2)
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float32 = _DType("float32", 4)
        cls.int32 = _DType("int32", 4)
        cls.int64 = _DType("int64", 8)

        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.float16 = cls.float16
        fake_jnp.bfloat16 = cls.bfloat16
        fake_jnp.float32 = cls.float32
        fake_jnp.int32 = cls.int32
        fake_jnp.int64 = cls.int64
        fake_jnp.dtype = lambda value: value.dtype if hasattr(value, "dtype") else value
        fake_jnp.full = lambda shape, _value, dtype: _Array(shape, dtype)
        fake_jnp.reshape = lambda value, shape: _Array(shape, value.dtype)

        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        fake_jax.numpy = fake_jnp
        fake_jax.ShapeDtypeStruct = _Array

        def fake_jit(function=None, **options):
            def decorate(resolved):
                resolved._test_static_argnames = tuple(options.get("static_argnames", ()))
                return resolved

            return decorate if function is None else decorate(function)

        fake_jax.jit = fake_jit
        fake_jax.tree_util = types.SimpleNamespace(
            DictKey=lambda key: key,
            register_pytree_with_keys=lambda *_args: None,
        )

        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.__path__ = []
        fake_cutlass_jax = types.ModuleType("cutlass.jax")
        fake_cutlass_jax.TensorSpec = _TensorSpec
        fake_cutlass_jax.cutlass_call = None
        fake_cutlass.jax = fake_cutlass_jax

        cls.fake_modules = {
            "jax": fake_jax,
            "jax.numpy": fake_jnp,
            "cutlass": fake_cutlass,
            "cutlass.jax": fake_cutlass_jax,
        }
        cls.module_patch = mock.patch.dict(sys.modules, cls.fake_modules)
        cls.module_patch.start()

        root = types.ModuleType(cls._PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = cls._PACKAGE
        root.__spec__ = ModuleSpec(cls._PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[cls._PACKAGE] = root

        dsa_name = f"{cls._PACKAGE}.deepseek_sparse_attention"
        dsa = types.ModuleType(dsa_name)
        dsa.__path__ = [str(_DSA_ROOT)]
        dsa.__package__ = dsa_name
        dsa.__spec__ = ModuleSpec(dsa_name, loader=None, is_package=True)
        sys.modules[dsa_name] = dsa
        for operation in ("indexer_forward", "indexer_top_k"):
            name = f"{dsa_name}.{operation}"
            package = types.ModuleType(name)
            package.__path__ = [str(_DSA_ROOT / operation)]
            package.__package__ = name
            package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = package

        cls.jax_base = importlib.import_module(f"{cls._PACKAGE}._jax")

        def resolve_compute_capability(
            target_compute_capability,
            supported_compute_capabilities,
            operation_name,
        ):
            del operation_name
            resolved = 100 if target_compute_capability is None else target_compute_capability
            if resolved not in supported_compute_capabilities:
                raise ValueError(f"unsupported synthetic target SM{resolved}")
            return resolved

        cls.target_resolution_patch = mock.patch.object(
            cls.jax_base.JaxApiBase,
            "_resolve_compute_capability",
            staticmethod(resolve_compute_capability),
        )
        cls.target_resolution_patch.start()
        cls.forward = importlib.import_module(f"{dsa_name}.indexer_forward.jax")
        cls.topk = importlib.import_module(f"{dsa_name}.indexer_top_k.jax")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.target_resolution_patch.stop()
        for name in tuple(sys.modules):
            if name == cls._PACKAGE or name.startswith(f"{cls._PACKAGE}."):
                sys.modules.pop(name, None)
        cls.module_patch.stop()

    @staticmethod
    def fake_call(captured):
        def call(_inputs, *, output_descs, **options):
            captured.update(output_descs=output_descs, **options)
            output_specs = options.get("output_spec")
            if output_specs is None:
                output_specs = (None,) * len(output_descs)

            def public_shape(desc, spec):
                if spec is None or spec.mode is None:
                    return desc.shape
                shape = [None] * desc.ndim
                for canonical_axis, public_axis in enumerate(spec.mode):
                    shape[public_axis] = desc.shape[canonical_axis]
                return tuple(shape)

            return tuple(_Array(public_shape(desc, spec), desc.dtype) for desc, spec in zip(output_descs, output_specs))

        return call

    def test_forward_declares_initialized_padded_output_for_fixed_and_thd(self):
        fixed = self.forward.IndexerForward(
            _Array((1, 8, 32, 128), self.bfloat16),
            _Array((1, 5, 1, 128), self.bfloat16),
            _Array((1, 8, 32), self.bfloat16),
            ratio=1,
            target_compute_capability=100,
        )
        captured = {}
        fixed._call_kernel = self.fake_call(captured)
        result = fixed(
            _Array((1, 8, 32, 128), self.bfloat16),
            _Array((1, 5, 1, 128), self.bfloat16),
            _Array((1, 8, 32), self.bfloat16),
        )
        self.assertEqual(result["scores"].shape, (1, 8, 5))
        output = captured["output_descs"][0]
        self.assertEqual(output.shape, (1, 8, 8))
        self.assertEqual(output.init_value, float("-inf"))

        thd = self.forward.IndexerForward(
            _Array((8, 32, 128), self.bfloat16),
            _Array((9, 1, 128), self.bfloat16),
            _Array((8, 32), self.bfloat16),
            sample_cu_seqlens_q=_Array((2,), self.int32),
            sample_cu_seqlens_k=_Array((2,), self.int32),
            max_seqlen_q=8,
            max_seqlen_k=5,
            target_compute_capability=90,
        )
        captured = {}
        thd._call_kernel = self.fake_call(captured)
        result = thd(
            _Array((8, 32, 128), self.bfloat16),
            _Array((9, 1, 128), self.bfloat16),
            _Array((8, 32), self.bfloat16),
            cu_seqlens_q=_Array((2,), self.int32),
            cu_seqlens_k=_Array((2,), self.int32),
        )
        self.assertEqual(result["scores"].shape, (8, 5))
        self.assertEqual(captured["output_descs"][0].shape, (8, 8))

    def test_forward_maps_independent_public_layouts_to_canonical_axes(self):
        operation = self.forward.IndexerForward(
            _Array((8, 2, 32, 128), self.bfloat16),
            _Array((2, 5, 1, 128), self.bfloat16),
            _Array((8, 2, 32), self.bfloat16),
            ratio=1,
            q_layout="SBHD",
            k_layout="BSHD",
            w_layout="SBH",
            output_layout="SBK",
            target_compute_capability=100,
        )
        captured = {}
        operation._call_kernel = self.fake_call(captured)

        result = operation(
            _Array((8, 2, 32, 128), self.bfloat16),
            _Array((2, 5, 1, 128), self.bfloat16),
            _Array((8, 2, 32), self.bfloat16),
        )

        self.assertEqual(operation.q_desc.shape, (2, 8, 32, 128))
        self.assertEqual(operation.q_desc.stride_order, (3, 2, 0, 1))
        self.assertEqual(result["scores"].shape, (8, 2, 5))
        self.assertEqual(
            tuple(spec.mode for spec in captured["input_spec"][:3]),
            ((1, 0, 2, 3), (0, 1, 2, 3), (1, 0, 2)),
        )
        self.assertEqual(
            tuple(spec.layout for spec in captured["input_spec"][:3]),
            ((3, 2, 1, 0), (3, 2, 1, 0), (2, 1, 0)),
        )
        self.assertEqual(captured["output_spec"][0].mode, (1, 0, 2))
        self.assertEqual(captured["output_spec"][0].layout, (2, 1, 0))

        explicit_output = self.forward.IndexerForward(
            _Array((8, 2, 32, 128), self.bfloat16),
            _Array((2, 5, 1, 128), self.bfloat16),
            _Array((8, 2, 32), self.bfloat16),
            sample_out=_Array((8, 2, 8), self.float32),
            ratio=1,
            q_layout="SBHD",
            k_layout="BSHD",
            w_layout="SBH",
            output_layout="SBK",
            target_compute_capability=100,
        )
        self.assertEqual(explicit_output.o_desc.shape, (2, 8, 8))
        self.assertEqual(explicit_output.o_desc.stride_order, (2, 0, 1))

    def test_forward_rejects_unimplemented_feature_axis_permutations(self):
        with self.assertRaisesRegex(ValueError, "q_layout must be one of"):
            self.forward.IndexerForward(
                _Array((1, 32, 8, 128), self.bfloat16),
                _Array((1, 5, 1, 128), self.bfloat16),
                _Array((1, 8, 32), self.bfloat16),
                q_layout="BHSD",
                target_compute_capability=100,
            )

    def test_forward_layout_arguments_are_static(self):
        static_argnames = set(self.forward.indexer_forward_wrapper._test_static_argnames)
        self.assertTrue({"q_layout", "k_layout", "w_layout", "output_layout"}.issubset(static_argnames))

    def test_topk_declares_outputs_and_hidden_workspace_for_both_modes(self):
        for return_val in (True, False):
            with self.subTest(return_val=return_val):
                api = self.topk.IndexerTopK(
                    _Array((4, 64), self.float32),
                    _Array((4,), self.int32),
                    8,
                    return_val=return_val,
                    target_compute_capability=100,
                )
                captured = {}
                api._call_kernel = self.fake_call(captured)
                result = api(_Array((4, 64), self.float32), _Array((4,), self.int32))
                self.assertEqual(result["indices"].shape, (4, 8))
                self.assertEqual(result["values"] is not None, return_val)
                self.assertEqual(captured["workspace_descs"][0].shape, (4, 2, 64))
                self.assertEqual(len(captured["output_descs"]), 2 if return_val else 1)

    def test_topk_constructor_reports_invalid_inputs_before_inference(self):
        with self.assertRaisesRegex(ValueError, "input_values must have rank 2"):
            self.topk.IndexerTopK(
                _Array((64,), self.float32),
                _Array((1,), self.int32),
                8,
                target_compute_capability=100,
            )

        with self.assertRaisesRegex(ValueError, "input_values must have dtype"):
            self.topk.IndexerTopK(
                _Array((4, 64), self.int64),
                _Array((4,), self.int32),
                8,
                target_compute_capability=100,
            )


if __name__ == "__main__":
    unittest.main()
