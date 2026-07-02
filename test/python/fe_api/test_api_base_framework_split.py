# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the framework-specific API base split."""

from __future__ import annotations

import ast
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"


class _DType:
    def __init__(self, name, itemsize=None):
        self.name = name
        self.itemsize = itemsize


def _install_test_package(name):
    package = types.ModuleType(name)
    package.__path__ = [str(_CUDNN_ROOT)]
    package.__package__ = name
    package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = package


def _remove_test_package(name):
    for module_name in tuple(sys.modules):
        if module_name == name or module_name.startswith(f"{name}."):
            sys.modules.pop(module_name, None)


class ApiBaseFrameworkSplitTest(unittest.TestCase):
    def test_torch_wrappers_use_explicit_base_name(self):
        for path in _CUDNN_ROOT.rglob("api.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    legacy_bases = [base for base in node.bases if isinstance(base, ast.Name) and base.id == "APIBase"]
                    self.assertFalse(legacy_bases, f"{path} must inherit ApiBaseTorch explicitly")

    def test_neutral_base_does_not_load_optional_frameworks(self):
        package_name = "cudnn_frontend_neutral_api_base_test"
        _install_test_package(package_name)
        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "torch": None,
                    "jax": None,
                    "jax.numpy": None,
                    "cutlass": None,
                    "cuda": None,
                },
            ):
                module = importlib.import_module(f"{package_name}.api_base")
                self.assertNotIn(f"{package_name}.api_base_torch", sys.modules)

                desc = module.TensorDesc(
                    dtype=_DType("bfloat16", 2),
                    shape=(4, 8),
                    stride_order=(1, 0),
                    name="x",
                )
                self.assertEqual(desc.shape, (4, 8))
                self.assertEqual(desc.stride, (8, 1))
                self.assertEqual(desc.dtype_name, "bfloat16")
                self.assertEqual(desc.element_bits, 16)

                base = module.ApiBase()
                self.assertEqual(base.check_tensor_shape(desc, (4, 8), "x"), (4, 8))
                self.assertEqual(base.check_dtype(desc, (_DType("bfloat16"),), "x"), "bfloat16")

                fp4 = module.TensorDesc(
                    dtype=_DType("float4_e2m1fn_x2", 1),
                    shape=(8,),
                )
                self.assertEqual(fp4.dtype_name, "float4_e2m1fn")
                self.assertEqual(fp4.element_bits, 4)
                self.assertTrue(fp4.is_fp4)
                self.assertEqual(fp4.storage_dtype_name, "float4_e2m1fn_x2")
                self.assertEqual(fp4.packing, "fp4x2")

                packed_fp4 = module.TensorDesc(
                    dtype=_DType("uint8", 1),
                    shape=(8,),
                    packing="fp4x2",
                )
                self.assertEqual(packed_fp4.storage_dtype_name, "uint8")
                self.assertEqual(packed_fp4.dtype_name, "float4_e2m1fn")
                self.assertEqual(packed_fp4.element_bits, 4)
        finally:
            _remove_test_package(package_name)

    def test_jax_descriptor_and_callable_are_abstract_and_stable(self):
        package_name = "cudnn_frontend_jax_api_base_test"
        _install_test_package(package_name)

        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.dtype = lambda value: value
        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        fake_jax.numpy = fake_jnp

        def identity_jit(fn=None, **_kwargs):
            return (lambda decorated_fn: decorated_fn) if fn is None else fn

        fake_cutlass_jax = types.ModuleType("cutlass.jax")
        fake_cutlass_jax.TensorSpec = type("TensorSpec", (), {})
        fake_cute = types.ModuleType("cutlass.cute")
        fake_cute.jit = identity_jit
        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.__path__ = []
        fake_cutlass.Constexpr = object
        fake_cutlass.cute = fake_cute
        fake_cutlass.jax = fake_cutlass_jax

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "jax": fake_jax,
                    "jax.numpy": fake_jnp,
                    "cutlass": fake_cutlass,
                    "cutlass.cute": fake_cute,
                    "cutlass.jax": fake_cutlass_jax,
                    "torch": None,
                },
            ):
                module = importlib.import_module(f"{package_name}._jax.api_base")

                value = types.SimpleNamespace(
                    shape=(2, 3, 5, 7, 11, 13),
                    dtype=_DType("float8_e4m3fn", 1),
                )
                desc = module.JaxTensorDesc.from_value(
                    value,
                    layout=(2, 1, 4, 0, 3, 5),
                    name="scale",
                )
                self.assertEqual(desc.layout, (2, 1, 4, 0, 3, 5))
                self.assertEqual(desc.stride_order, (3, 1, 0, 4, 2, 5))
                self.assertEqual(desc.stride, (21, 7, 462, 1, 42, 2310))
                self.assertEqual(desc.dtype_name, "float8_e4m3fn")

                transposed = module.JaxTensorDesc.from_value(
                    types.SimpleNamespace(shape=(2, 3, 5), dtype=_DType("bfloat16", 2)),
                    layout=(2, 1, 0),
                    mode=(1, 2, 0),
                )
                self.assertEqual(transposed.shape, (3, 5, 2))
                self.assertEqual(transposed.stride, (5, 1, 15))
                self.assertEqual(transposed.stride_order, (1, 0, 2))
                self.assertEqual(transposed.mode, (1, 2, 0))

                with self.assertRaisesRegex(ValueError, "stride_order must agree"):
                    module.JaxTensorDesc(
                        dtype=_DType("bfloat16", 2),
                        shape=(2, 3),
                        stride_order=(0, 1),
                        jax_layout=(1, 0),
                    )

                with self.assertRaisesRegex(ValueError, "stride must describe the compact layout"):
                    module.JaxTensorDesc(
                        dtype=_DType("bfloat16", 2),
                        shape=(2, 3),
                        stride=(1, 2),
                        jax_layout=(1, 0),
                    )

                with self.assertRaisesRegex(ValueError, "mode must be a permutation"):
                    module.JaxTensorDesc.from_value(
                        types.SimpleNamespace(shape=(2, 3, 5), dtype=_DType("bfloat16", 2)),
                        mode=(0, 1),
                    )

                class _JaxApi(module.ApiBaseJax):
                    def __init__(self, sample):
                        super().__init__()
                        self.sample_desc = self.make_tensor_desc(sample, name="sample")
                        self.static_option = 1
                        self.check_count = 0

                    def _check_support(self):
                        self.check_count += 1
                        return True

                    def _call_impl(self, x):
                        self.check_tensor_signature(x, self.sample_desc, name="input")
                        return x

                api = _JaxApi(value)
                self.assertTrue(all(stored is not value for stored in vars(api).values()))
                self.assertIs(api.get_jax_callable(), api)
                self.assertIs(api.as_dtype(value), value.dtype)
                self.assertIs(api.as_optional_dtype(value), value.dtype)
                self.assertIsNone(api.as_optional_dtype(None))
                self.assertIs(
                    api.require_dtype("sample.dtype", value, (value.dtype,)),
                    value.dtype,
                )
                self.assertIs(
                    api.require_dtype("output_dtype", None, (value.dtype,), default=value.dtype),
                    value.dtype,
                )
                with self.assertRaisesRegex(ValueError, "output_dtype must not be None"):
                    api.require_dtype("output_dtype", None, (value.dtype,))
                with self.assertRaisesRegex(ValueError, "sample.dtype must be one of"):
                    api.require_dtype(
                        "sample.dtype",
                        value,
                        (_DType("bfloat16", 2),),
                    )
                self.assertEqual(api.check_tensor_signature(value, desc, name="scale"), desc)
                with self.assertRaisesRegex(ValueError, "scale tensor shape mismatch"):
                    api.check_tensor_signature(
                        types.SimpleNamespace(shape=(2, 3), dtype=_DType("float8_e4m3fn", 1)),
                        desc,
                        name="scale",
                    )
                with self.assertRaisesRegex(ValueError, "scale dtype mismatch"):
                    api.check_tensor_signature(
                        types.SimpleNamespace(shape=value.shape, dtype=_DType("bfloat16", 2)),
                        desc,
                        name="scale",
                    )
                self.assertIsNone(api.make_optional_tensor_desc(None, name="bias"))
                self.assertIsNone(api.check_optional_tensor_signature(None, None, name="bias"))
                with self.assertRaisesRegex(ValueError, "bias presence mismatch"):
                    api.check_optional_tensor_signature(value, None, name="bias")
                self.assertIs(api(value), value)
                self.assertIs(api(value), value)
                self.assertEqual(api.check_count, 1)
                self.assertTrue(api.check_support())
                self.assertEqual(api.check_count, 1)
                with self.assertRaisesRegex(AttributeError, "immutable after its first call"):
                    api.static_option = 2

                mutable_before_call = _JaxApi(value)
                self.assertTrue(mutable_before_call.check_support())
                mutable_before_call.static_option = 2
                self.assertFalse(mutable_before_call._is_supported)
                self.assertIs(mutable_before_call(value), value)
                self.assertEqual(mutable_before_call.check_count, 2)
                self.assertFalse(hasattr(fake_jax, "jit"))
        finally:
            _remove_test_package(package_name)

    def test_legacy_api_base_lazily_resolves_to_torch_alias(self):
        package_name = "cudnn_frontend_torch_api_base_test"
        _install_test_package(package_name)

        fake_torch = types.ModuleType("torch")

        class _Device:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return isinstance(other, _Device) and self.value == other.value

        fake_torch.device = _Device
        fake_torch.Tensor = type("Tensor", (), {})
        fake_torch.dtype = _DType
        fake_torch.Size = tuple
        fake_torch.memory_format = type("memory_format", (), {})
        fake_torch.contiguous_format = object()
        fake_torch.preserve_format = object()
        fake_torch.channels_last = object()
        fake_torch.channels_last_3d = object()
        fake_torch.uint8 = _DType("uint8", 1)
        fake_torch.float4_e2m1fn_x2 = _DType("float4_e2m1fn_x2", 1)

        fake_cuda = types.ModuleType("cuda")
        fake_cuda.__path__ = []
        fake_bindings = types.ModuleType("cuda.bindings")
        fake_bindings.__path__ = []
        fake_driver = types.ModuleType("cuda.bindings.driver")
        fake_cuda.bindings = fake_bindings
        fake_bindings.driver = fake_driver

        fake_cutlass = types.ModuleType("cutlass")
        fake_cutlass.__path__ = []
        fake_cute = types.ModuleType("cutlass.cute")
        fake_cutlass.cute = fake_cute

        fake_datatypes = types.ModuleType(f"{package_name}.datatypes")
        fake_datatypes._convert_to_cutlass_data_type = lambda dtype, **_kwargs: dtype

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "cuda": fake_cuda,
                    "cuda.bindings": fake_bindings,
                    "cuda.bindings.driver": fake_driver,
                    "cutlass": fake_cutlass,
                    "cutlass.cute": fake_cute,
                    f"{package_name}.datatypes": fake_datatypes,
                },
            ):
                neutral = importlib.import_module(f"{package_name}.api_base")
                self.assertNotIn(f"{package_name}.api_base_torch", sys.modules)

                legacy = neutral.APIBase
                self.assertIn(f"{package_name}.api_base_torch", sys.modules)
                self.assertIs(legacy, neutral.ApiBaseTorch)
                self.assertTrue(issubclass(neutral.ApiBaseTorch, neutral.ApiBase))
                self.assertEqual(legacy.__name__, "ApiBaseTorch")

                desc = neutral.TorchTensorDesc(
                    dtype=_DType("float32", 4),
                    shape=(2, 3),
                    stride=(3, 1),
                    stride_order=(1, 0),
                    device="cuda:0",
                )
                self.assertIsInstance(desc, neutral.TensorDesc)
                self.assertEqual(desc.device, _Device("cuda:0"))
                self.assertEqual(desc.size(), (2, 3))
                transposed = desc.transpose(0, 1)
                self.assertIsInstance(transposed, neutral.TorchTensorDesc)
                self.assertEqual(transposed.shape, (3, 2))
                self.assertEqual(transposed.device, desc.device)

                class _TorchApi(neutral.ApiBaseTorch):
                    def check_support(self):
                        return True

                    def compile(self):
                        return None

                    def execute(self, *_args, **_kwargs):
                        return None

                api = _TorchApi()
                self.assertEqual(api._check_tensor_shape(desc, (2, 3), "x"), (2, 3))
                self.assertEqual(
                    api._check_tensor_stride(desc, stride=(3, 1), name="x"),
                    ((3, 1), (1, 0)),
                )
                self.assertEqual(
                    api._check_tensor_stride(None, stride=[(3, 1)], name="optional"),
                    (None, None),
                )
                self.assertIs(api._check_dtype(desc, _DType("float32", 4), "x"), desc.dtype)
                with self.assertRaisesRegex(ValueError, "must match another operand"):
                    api._check_dtype(desc, _DType("float16", 2), "x", "must match another operand")

                packed_tensor = fake_torch.Tensor()
                packed_tensor.dtype = fake_torch.uint8
                packed_tensor.shape = (2, 4)
                packed_tensor.device = _Device("cuda:0")
                packed_tensor.stride = lambda: (4, 1)
                packed_desc = neutral.TorchTensorDesc.from_tensor(
                    packed_tensor,
                    interpret_uint8_as_fp4x2=True,
                )
                self.assertEqual(packed_desc.shape, (2, 8))
                self.assertEqual(packed_desc.stride, (8, 1))
                self.assertEqual(packed_desc.packing, "fp4x2")
                self.assertEqual(packed_desc.element_bits, 4)
        finally:
            _remove_test_package(package_name)


if __name__ == "__main__":
    unittest.main()
