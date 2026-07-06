# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for JAX DSA sparse-attention backward."""

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
_TEST_PACKAGE = "cudnn_frontend_jax_dsa_sparse_bwd_contract_test"


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

    def __init__(self, value=0.0):
        self.value = value


class _Kernel:
    instances = []

    def __init__(self, *, head_dim, head_dim_v, block_tile):
        self.configuration = (head_dim, head_dim_v, block_tile)
        self.calls = []
        self.instances.append(self)

    @staticmethod
    def get_workspace_size_lse_odo(q, d, h, b, acc_dtype):
        assert acc_dtype is _Float32
        return (b, h, ((q + 7) // 8) * 8, 2 * acc_dtype.width // 8)

    @staticmethod
    def get_workspace_size_dkv(k, d, b, acc_dtype):
        assert acc_dtype is _Float32
        return (
            b,
            1,
            ((k + 7) // 8) * 8,
            ((d + 7) // 8) * 8 * acc_dtype.width // 8,
        )

    def __call__(self, *args):
        self.calls.append(args)


class JaxDsaSparseAttentionBackwardContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float32 = _DType("float32", 4)
        cls.int32 = _DType("int32", 4)
        cls.uint8 = _DType("uint8", 1)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        cls.fake_jnp.bfloat16 = cls.bfloat16
        cls.fake_jnp.float32 = cls.float32
        cls.fake_jnp.int32 = cls.int32
        cls.fake_jnp.uint8 = cls.uint8
        cls.fake_jnp.dtype = lambda value: value

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp
        cls.fake_jax.tree_util = types.SimpleNamespace(
            DictKey=lambda key: key,
            register_pytree_with_keys=lambda *_args: None,
        )
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
        cls.fake_cutlass_jax.cutlass_call = None
        cls.fake_cutlass.jax = cls.fake_cutlass_jax

        cls.kernel_module_name = f"{_TEST_PACKAGE}.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_sm100"
        cls.kernel_module = types.ModuleType(cls.kernel_module_name)
        cls.kernel_module.FlashAttentionDSABackwardSm100 = _Kernel

        package_paths = {
            _TEST_PACKAGE: _CUDNN_ROOT,
            f"{_TEST_PACKAGE}.deepseek_sparse_attention": _CUDNN_ROOT / "deepseek_sparse_attention",
            f"{_TEST_PACKAGE}.deepseek_sparse_attention.sparse_attention_backward": _CUDNN_ROOT / "deepseek_sparse_attention" / "sparse_attention_backward",
        }
        for package_name, package_path in package_paths.items():
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            package.__package__ = package_name
            sys.modules[package_name] = package

        with cls._optional_modules():
            cls.module = import_module(f"{_TEST_PACKAGE}.deepseek_sparse_attention.sparse_attention_backward.jax")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    @classmethod
    def _optional_modules(cls, *, include_kernel=False):
        modules = {
            "jax": cls.fake_jax,
            "jax.numpy": cls.fake_jnp,
            "cutlass": cls.fake_cutlass,
            "cutlass.cute": cls.fake_cutlass_cute,
            "cutlass.jax": cls.fake_cutlass_jax,
        }
        if include_kernel:
            modules[cls.kernel_module_name] = cls.kernel_module
        return mock.patch.dict(sys.modules, modules)

    @staticmethod
    def _inputs(bfloat16, float32, int32, *, with_length):
        q = _Array((65, 64, 512), bfloat16)
        kv = _Array((130, 512), bfloat16)
        out = _Array(q.shape, bfloat16)
        dout = _Array(q.shape, bfloat16)
        lse = _Array((65, 64), float32)
        attn_sink = _Array((64,), float32)
        topk_idxs = _Array((65, 32), int32)
        topk_length = _Array((65,), int32) if with_length else None
        return q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length

    @staticmethod
    def _fake_call(captured):
        def call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        return call

    def test_kernel_module_is_lazy(self):
        self.assertNotIn(self.kernel_module_name, sys.modules)

    def test_functional_outputs_and_zero_initialized_workspaces(self):
        captured = {}
        launcher = object()
        inputs = self._inputs(
            self.bfloat16,
            self.float32,
            self.int32,
            with_length=False,
        )
        q, kv, out, dout, lse, attn_sink, topk_idxs, _ = inputs

        with (
            self._optional_modules(include_kernel=True),
            mock.patch.object(self.module, "_launch_without_topk_length", new=launcher),
            mock.patch.object(
                self.module,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.module.sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
            )

        self.assertEqual(result["dq"].shape, q.shape)
        self.assertEqual(result["dkv"].shape, kv.shape)
        self.assertEqual(result["d_sink"].shape, attn_sink.shape)
        self.assertEqual(
            captured["inputs"],
            (q, kv, out, dout, lse, attn_sink, topk_idxs),
        )
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype, spec.fill_value) for spec in captured["outputs"]],
            [
                ("dq", q.shape, self.bfloat16, None),
                ("dkv", kv.shape, self.bfloat16, 0),
                ("d_sink", attn_sink.shape, self.float32, 0.0),
            ],
        )
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype, spec.fill_value) for spec in captured["workspaces"]],
            [
                ("workspace_lse_odo", (1, 64, 72, 8), self.uint8, 0),
                ("workspace_dkv", (1, 1, 136, 2048), self.uint8, 0),
            ],
        )
        self.assertEqual(captured["outputs"][0].tensor_spec.divisibility, 512)
        self.assertEqual(len(captured["input_specs"]), 7)
        self.assertEqual(
            captured["static_args"],
            {
                "total_seqlen_q": 65,
                "total_seqlen_kv": 130,
                "num_heads": 64,
                "head_dim": 512,
                "block_tile": 64,
                "softmax_scale": 1.0 / (512**0.5),
            },
        )

    def test_optional_topk_length_is_a_real_kernel_input(self):
        captured = {}
        launcher = object()
        inputs = self._inputs(
            self.bfloat16,
            self.float32,
            self.int32,
            with_length=True,
        )

        with (
            self._optional_modules(include_kernel=True),
            mock.patch.object(self.module, "_launch_with_topk_length", new=launcher),
            mock.patch.object(
                self.module,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            self.module.sparse_attention_backward_wrapper(
                *inputs[:-1],
                topk_length=inputs[-1],
            )

        self.assertIs(captured["inputs"][-1], inputs[-1])
        self.assertEqual(len(captured["input_specs"]), 8)
        self.assertEqual(captured["static_args"]["softmax_scale"], 1.0 / (512**0.5))

    def test_launchers_preserve_kernel_argument_order(self):
        for launcher_name, num_args, topk_length_idx, dq_idx, workspace_idx in (
            ("_launch_with_topk_length", 14, 8, 9, 13),
            ("_launch_without_topk_length", 13, None, 8, 12),
        ):
            with self.subTest(launcher=launcher_name):
                _Kernel.instances.clear()
                placeholders = [object() for _ in range(num_args)]
                with self._optional_modules(include_kernel=True):
                    getattr(self.module, launcher_name)(
                        *placeholders,
                        total_seqlen_q=64,
                        total_seqlen_kv=128,
                        num_heads=64,
                        head_dim=512,
                        block_tile=64,
                        softmax_scale=0.125,
                    )

                kernel = _Kernel.instances[-1]
                self.assertEqual(kernel.configuration, (512, 512, 64))
                args = kernel.calls[-1]
                self.assertEqual(args[0], (64, 128, 512, (64, 1)))
                if topk_length_idx is None:
                    self.assertIsNone(args[8])
                else:
                    self.assertIs(args[8], placeholders[topk_length_idx])
                self.assertIs(args[9], placeholders[dq_idx])
                self.assertIs(args[13], placeholders[workspace_idx])
                self.assertEqual(args[14].value, 0.125)
                self.assertIs(args[15], placeholders[0])

    def test_rejects_non_bf16_or_non_multiple_of_64_heads(self):
        inputs = self._inputs(
            self.bfloat16,
            self.float32,
            self.int32,
            with_length=False,
        )
        q, kv, out, dout, lse, attn_sink, topk_idxs, _ = inputs
        bad_q = _Array(q.shape, self.float32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "q.dtype"),
        ):
            self.module.sparse_attention_backward_wrapper(
                bad_q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
            )

        bad_q = _Array((65, 32, 512), self.bfloat16)
        bad_out = _Array(bad_q.shape, self.bfloat16)
        bad_lse = _Array((65, 32), self.float32)
        bad_sink = _Array((32,), self.float32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "divisible by 64"),
        ):
            self.module.sparse_attention_backward_wrapper(
                bad_q,
                kv,
                bad_out,
                bad_out,
                bad_lse,
                bad_sink,
                topk_idxs,
            )


if __name__ == "__main__":
    unittest.main()
