# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the JAX SDPA wrappers."""

from __future__ import annotations

from importlib import import_module
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


def _identity_jit(fn=None, **_kwargs):
    return (lambda decorated_fn: decorated_fn) if fn is None else fn


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_sdpa_contract_test"


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


class _TensorSpec:
    def __init__(
        self,
        *,
        layout=None,
        mode=None,
        static=None,
        ptr_assumed_align=256,
        divisibility=None,
    ):
        self.layout = layout
        self.mode = mode
        self.static = static
        self.ptr_assumed_align = ptr_assumed_align
        self.divisibility = divisibility


class _Float32:
    width = 32


class _BackwardKernel:
    @staticmethod
    def get_workspace_size(seqlen_q, head_dim, num_query_heads, batch, acc_dtype):
        assert acc_dtype is _Float32
        return (
            batch,
            ((seqlen_q + 7) // 8) * 8,
            num_query_heads,
            2 * (acc_dtype.width // 8) + head_dim * (acc_dtype.width // 8),
        )


class JaxSdpaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.float16 = _DType("float16", 2)
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float32 = _DType("float32", 4)
        cls.uint8 = _DType("uint8", 1)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        cls.fake_jnp.float16 = cls.float16
        cls.fake_jnp.bfloat16 = cls.bfloat16
        cls.fake_jnp.float32 = cls.float32
        cls.fake_jnp.uint8 = cls.uint8
        cls.fake_jnp.dtype = lambda value: value

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp
        cls.fake_jax.ShapeDtypeStruct = lambda shape, dtype: (shape, dtype)

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass.Constexpr = object
        cls.fake_cutlass.Float32 = _Float32
        cls.fake_cutlass.Int32 = int
        cls.fake_cutlass_cute = types.ModuleType("cutlass.cute")
        cls.fake_cutlass_cute.jit = _identity_jit
        cls.fake_cutlass.cute = cls.fake_cutlass_cute
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: f"cutlass.{dtype.name}"
        cls.fake_cutlass_jax.cutlass_call = None
        cls.fake_cutlass.jax = cls.fake_cutlass_jax

        cls.backward_kernel_module_name = f"{_TEST_PACKAGE}.sdpa.bwd.fmha_backward_sm100_2kernel"
        cls.backward_kernel_module = types.ModuleType(cls.backward_kernel_module_name)
        cls.backward_kernel_module.BlackwellFusedMultiHeadAttentionBackward = _BackwardKernel

        package_paths = {
            _TEST_PACKAGE: _CUDNN_ROOT,
            f"{_TEST_PACKAGE}.sdpa": _CUDNN_ROOT / "sdpa",
            f"{_TEST_PACKAGE}.sdpa.fwd": _CUDNN_ROOT / "sdpa" / "fwd",
            f"{_TEST_PACKAGE}.sdpa.bwd": _CUDNN_ROOT / "sdpa" / "bwd",
        }
        for package_name, package_path in package_paths.items():
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            package.__package__ = package_name
            sys.modules[package_name] = package

        with cls._optional_modules():
            cls.forward = import_module(f"{_TEST_PACKAGE}.sdpa.fwd.jax")
            cls.backward = import_module(f"{_TEST_PACKAGE}.sdpa.bwd.jax")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    @classmethod
    def _optional_modules(cls, *, include_backward_kernel=False):
        modules = {
            "jax": cls.fake_jax,
            "jax.numpy": cls.fake_jnp,
            "cutlass": cls.fake_cutlass,
            "cutlass.cute": cls.fake_cutlass_cute,
            "cutlass.jax": cls.fake_cutlass_jax,
        }
        if include_backward_kernel:
            modules[cls.backward_kernel_module_name] = cls.backward_kernel_module
        return mock.patch.dict(sys.modules, modules)

    @staticmethod
    def _fake_call(captured):
        def call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        return call

    def test_kernel_implementations_are_deferred_until_wrapper_call(self):
        self.assertNotIn(
            f"{_TEST_PACKAGE}.sdpa.fwd.fmha_forward_sm100_d256",
            sys.modules,
        )
        self.assertNotIn(self.backward_kernel_module_name, sys.modules)

    def test_forward_declares_bhsd_layout_and_functional_outputs(self):
        captured = {}
        launcher = object()
        q = _Array((2, 8, 64, 256), self.float16)
        k = _Array((2, 2, 96, 256), self.float16)
        v = _Array((2, 2, 96, 256), self.float16)

        with (
            self._optional_modules(),
            mock.patch.object(self.forward, "_make_launcher", return_value=launcher),
            mock.patch.object(
                self.forward,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.forward.sdpa_fwd_wrapper_sm100_d256(q, k, v)

        self.assertEqual(result.o_tensor.shape, q.shape)
        self.assertEqual(result.lse_tensor.shape, (2, 8, 64))
        self.assertEqual(captured["inputs"], (q, k, v))
        self.assertTrue(captured["use_static_tensors"])
        self.assertNotIn("workspaces", captured)
        self.assertIs(captured["launcher"], launcher)

        output, lse = captured["outputs"]
        self.assertEqual(
            (output.name, output.shape, output.dtype),
            ("o_tensor", q.shape, self.float16),
        )
        self.assertEqual(
            (lse.name, lse.shape, lse.dtype),
            ("lse_tensor", (2, 8, 64), self.float32),
        )
        self.assertEqual(output.tensor_spec.layout, (3, 1, 2, 0))
        self.assertEqual(output.tensor_spec.mode, (0, 2, 1, 3))
        self.assertTrue(all(spec is output.tensor_spec for spec in captured["input_specs"]))

    def test_backward_declares_zero_initialized_hidden_workspace(self):
        captured = {}
        launcher = object()
        q = _Array((2, 8, 65, 256), self.bfloat16)
        k = _Array((2, 2, 96, 256), self.bfloat16)
        v = _Array((2, 2, 96, 256), self.bfloat16)
        output = _Array(q.shape, self.bfloat16)
        doutput = _Array(q.shape, self.bfloat16)
        lse = _Array((2, 8, 65), self.float32)

        with (
            self._optional_modules(include_backward_kernel=True),
            mock.patch.object(self.backward, "_make_launcher", return_value=launcher) as make_launcher,
            mock.patch.object(
                self.backward,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.backward.sdpa_bwd_wrapper_sm100_d256(
                q,
                k,
                v,
                output,
                doutput,
                lse,
            )

        self.assertEqual(result.dq_tensor.shape, q.shape)
        self.assertEqual(result.dk_tensor.shape, k.shape)
        self.assertEqual(result.dv_tensor.shape, v.shape)
        self.assertEqual(captured["inputs"], (q, k, v, output, doutput, lse))
        self.assertTrue(captured["use_static_tensors"])
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype) for spec in captured["outputs"]],
            [
                ("dq_tensor", q.shape, self.bfloat16),
                ("dk_tensor", k.shape, self.bfloat16),
                ("dv_tensor", v.shape, self.bfloat16),
            ],
        )
        (workspace,) = captured["workspaces"]
        self.assertEqual(workspace.name, "workspace")
        self.assertEqual(workspace.shape, (2, 72, 8, 1032))
        self.assertEqual(workspace.dtype, self.uint8)
        self.assertEqual(workspace.fill_value, 0)
        self.assertEqual(make_launcher.call_args.kwargs["element_dtype"], "cutlass.bfloat16")
        self.assertEqual(make_launcher.call_args.kwargs["mask_kind"], "residual")

    def test_noncausal_sliding_window_is_rejected(self):
        q = _Array((1, 4, 64, 256), self.float16)
        k = _Array((1, 1, 64, 256), self.float16)
        v = _Array((1, 1, 64, 256), self.float16)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(NotImplementedError, "non-causal"),
        ):
            self.forward.sdpa_fwd_wrapper_sm100_d256(
                q,
                k,
                v,
                window_size=(32, 0),
            )

    def test_backward_validates_lse_contract_before_loading_kernel(self):
        q = _Array((1, 4, 64, 256), self.float16)
        k = _Array((1, 1, 64, 256), self.float16)
        v = _Array((1, 1, 64, 256), self.float16)
        output = _Array(q.shape, self.float16)
        doutput = _Array(q.shape, self.float16)
        bad_lse = _Array((1, 64, 4), self.float32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "lse_tensor must have shape"),
        ):
            self.backward.sdpa_bwd_wrapper_sm100_d256(
                q,
                k,
                v,
                output,
                doutput,
                bad_lse,
            )


if __name__ == "__main__":
    unittest.main()
