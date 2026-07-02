# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contract tests for the JAX DSA wrappers."""

from __future__ import annotations

import ast
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


def _identity_jit(fn=None, **_kwargs):
    return (lambda decorated_fn: decorated_fn) if fn is None else fn


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDNN_ROOT = _REPO_ROOT / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_dsa_contract_test"


class _DType:
    def __init__(self, name, itemsize):
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
        if not (isinstance(key, tuple) and key[:-1] == (Ellipsis,) and isinstance(key[-1], slice) and key[-1].start is None and key[-1].step is None):
            raise AssertionError(f"Unexpected test slice {key!r}")
        return _Array((*self.shape[:-1], key[-1].stop), self.dtype)


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


class JaxDsaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float16 = _DType("float16", 2)
        cls.float32 = _DType("float32", 4)
        cls.int32 = _DType("int32", 4)
        cls.int64 = _DType("int64", 8)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        cls.fake_jnp.bfloat16 = cls.bfloat16
        cls.fake_jnp.float16 = cls.float16
        cls.fake_jnp.float32 = cls.float32
        cls.fake_jnp.int32 = cls.int32
        cls.fake_jnp.int64 = cls.int64
        cls.fake_jnp.dtype = lambda value: value
        cls.fake_jnp.reshape = lambda value, shape: _Array(shape, value.dtype)

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass.Constexpr = object
        cls.fake_cutlass_cute = types.ModuleType("cutlass.cute")
        cls.fake_cutlass_cute.jit = _identity_jit
        cls.fake_cutlass.cute = cls.fake_cutlass_cute
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.is_available = lambda: True
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: f"cutlass.{dtype.name}"
        cls.fake_cutlass.jax = cls.fake_cutlass_jax

        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent
        with mock.patch.dict(
            sys.modules,
            {
                "jax": cls.fake_jax,
                "jax.numpy": cls.fake_jnp,
                "cutlass": cls.fake_cutlass,
                "cutlass.cute": cls.fake_cutlass_cute,
                "cutlass.jax": cls.fake_cutlass_jax,
            },
        ):
            importlib.import_module(f"{_TEST_PACKAGE}.jax")
            cls.forward = importlib.import_module(f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_forward.jax")
            cls.top_k = importlib.import_module(f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_top_k.jax")
            cls.score = importlib.import_module(f"{_TEST_PACKAGE}.deepseek_sparse_attention.score_recompute.jax")
            cls.score_config = importlib.import_module(f"{_TEST_PACKAGE}.deepseek_sparse_attention.score_recompute.config")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    def _optional_modules(self):
        return mock.patch.dict(
            sys.modules,
            {
                "jax": self.fake_jax,
                "jax.numpy": self.fake_jnp,
                "cutlass": self.fake_cutlass,
                "cutlass.cute": self.fake_cutlass_cute,
                "cutlass.jax": self.fake_cutlass_jax,
            },
        )

    def test_shared_index_helper_kernels_do_not_import_torch_eagerly(self):
        helper_root = _CUDNN_ROOT / "deepseek_sparse_attention" / "indexer_top_k"
        for filename in ("local_to_global_dsl.py", "compactify.py"):
            tree = ast.parse((helper_root / filename).read_text())
            eager_imports = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    eager_imports.extend(alias.name for alias in node.names if alias.name == "torch")
                elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                    eager_imports.append(node.module)
            self.assertEqual(eager_imports, [], filename)

    def test_indexer_forward_declares_initialized_padded_result(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            result = options["outputs"][0]
            return (_Array(result.shape, result.dtype),)

        q = _Array((1, 4, 32, 128), self.bfloat16)
        k = _Array((1, 5, 1, 128), self.bfloat16)
        w = _Array((1, 4, 32), self.bfloat16)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.forward,
                "_make_launcher",
                return_value=launcher,
            ),
            mock.patch.object(self.forward, "call_cutedsl", side_effect=fake_call),
        ):
            result = self.forward.indexer_forward_wrapper(q, k, w, ratio=1)

        self.assertEqual(result.scores.shape, (1, 4, 5))
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(captured["inputs"], (q, k, w))
        self.assertTrue(captured["use_static_tensors"])

        output = captured["outputs"][0]
        self.assertEqual(output.name, "scores")
        self.assertEqual(output.shape, (1, 4, 8))
        self.assertEqual(output.dtype, self.float32)
        self.assertEqual(output.fill_value, float("-inf"))
        self.assertEqual(output.tensor_spec.layout, (2, 1, 0))
        self.assertEqual(output.tensor_spec.mode, (0, 1, 2))
        self.assertEqual(output.tensor_spec.divisibility, (None, None, 4))

    def test_indexer_top_k_declares_hidden_workspace(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        input_values = _Array((2, 64), self.float32)
        seq_lens = _Array((2,), self.int32)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.top_k,
                "_make_launcher",
                return_value=launcher,
            ) as make_launcher,
            mock.patch.object(
                self.top_k,
                "call_cutedsl",
                side_effect=fake_call,
            ),
        ):
            result = self.top_k.indexer_top_k_wrapper(
                input_values,
                seq_lens,
                top_k=8,
            )

        self.assertEqual(result.indices.shape, (2, 8))
        self.assertEqual(result.values.shape, (2, 8))
        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(captured["inputs"], (input_values, seq_lens))
        self.assertTrue(captured["use_static_tensors"])
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype) for spec in captured["outputs"]],
            [
                ("indices", (2, 8), self.int32),
                ("values", (2, 8), self.float32),
            ],
        )
        self.assertEqual(len(captured["workspaces"]), 1)
        workspace = captured["workspaces"][0]
        self.assertEqual(workspace.name, "extra_buffer")
        self.assertEqual(workspace.shape, (2, 2, 64))
        self.assertEqual(workspace.dtype, self.int32)
        self.assertIsNone(workspace.fill_value)
        self.assertEqual(make_launcher.call_args.args[:2], ("cutlass.float32", 32))

    def test_indexer_top_k_rejects_unsupported_result_mode(self):
        input_values = _Array((2, 64), self.float16)
        seq_lens = _Array((2,), self.int32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(
                NotImplementedError,
                "return_val=True",
            ),
        ):
            self.top_k.indexer_top_k_wrapper(
                input_values,
                seq_lens,
                top_k=8,
                return_val=False,
            )

    def test_indexer_top_k_validates_selected_output_vector_width(self):
        with self.assertRaisesRegex(ValueError, "selected output vector width"):
            self.top_k.require_supported_top_k_output(
                top_k=513,
                num_threads_per_cta=256,
                num_copy_bits=256,
                dtype_bits=32,
            )

        # Small odd top-K values stay on the scalar path and remain valid.
        self.top_k.require_supported_top_k_output(
            top_k=3,
            num_threads_per_cta=256,
            num_copy_bits=256,
            dtype_bits=32,
        )

    def test_index_helpers_declare_functional_results(self):
        captured = []

        def fake_call(launcher, inputs, **options):
            captured.append(dict(launcher=launcher, inputs=inputs, **options))
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        local_indices = _Array((2, 3, 8), self.int64)
        compact_input = _Array((2, 3, 8), self.int32)
        global_launcher = object()
        compact_launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.top_k,
                "_make_local_to_global_launcher",
                return_value=global_launcher,
            ) as make_global_launcher,
            mock.patch.object(
                self.top_k,
                "_make_compactify_launcher",
                return_value=compact_launcher,
            ) as make_compact_launcher,
            mock.patch.object(self.top_k, "call_cutedsl", side_effect=fake_call),
        ):
            global_result = self.top_k.local_to_global_wrapper(
                local_indices,
                seqlen_k=1024,
            )
            compact_result = self.top_k.compactify_wrapper(compact_input)

        self.assertEqual(global_result.indices.shape, (2, 3, 8))
        self.assertEqual(global_result.indices.dtype, self.int32)
        self.assertEqual(compact_result.indices.shape, (6, 8))
        self.assertEqual(compact_result.topk_length.shape, (6,))

        global_call, compact_call = captured
        self.assertIs(global_call["launcher"], global_launcher)
        self.assertEqual(global_call["inputs"], (local_indices,))
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype) for spec in global_call["outputs"]],
            [("indices", (2, 3, 8), self.int32)],
        )
        make_global_launcher.assert_called_once_with(
            is_varlen=False,
            seqlen_k=1024,
        )

        self.assertIs(compact_call["launcher"], compact_launcher)
        self.assertEqual(compact_call["inputs"][0].shape, (6, 8))
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype) for spec in compact_call["outputs"]],
            [
                ("indices", (6, 8), self.int32),
                ("topk_length", (6,), self.int32),
            ],
        )
        make_compact_launcher.assert_called_once_with(rows=6, cols=8)
        self.assertTrue(global_call["use_static_tensors"])
        self.assertTrue(compact_call["use_static_tensors"])

    def test_local_to_global_packed_inputs_remain_device_values(self):
        captured = {}

        def fake_call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            spec = options["outputs"][0]
            return (_Array(spec.shape, spec.dtype),)

        local_indices = _Array((7, 8), self.int32)
        cu_seqlens_q = _Array((3,), self.int32)
        cu_seqlens_k = _Array((3,), self.int32)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(
                self.top_k,
                "_make_local_to_global_launcher",
                return_value=launcher,
            ) as make_launcher,
            mock.patch.object(self.top_k, "call_cutedsl", side_effect=fake_call),
        ):
            result = self.top_k.local_to_global_wrapper(
                local_indices,
                seqlen_k=16,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
            )

        self.assertEqual(result.indices.shape, local_indices.shape)
        self.assertEqual(
            captured["inputs"],
            (local_indices, cu_seqlens_q, cu_seqlens_k),
        )
        make_launcher.assert_called_once_with(is_varlen=True, seqlen_k=16)

    def test_index_helper_launchers_preserve_native_argument_order(self):
        calls = []

        class FakeLocalToGlobalKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args):
                calls.append(("global", args))

        class FakeCompactifyKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args):
                calls.append(("compact", args))

        package = f"{_TEST_PACKAGE}.deepseek_sparse_attention.indexer_top_k"
        global_module = types.ModuleType(f"{package}.local_to_global_dsl")
        global_module.LocalToGlobalTopK = FakeLocalToGlobalKernel
        compact_module = types.ModuleType(f"{package}.compactify")
        compact_module.CompactifyKernel = FakeCompactifyKernel
        int32 = lambda value: ("Int32", value)

        with (
            self._optional_modules(),
            mock.patch.dict(
                sys.modules,
                {
                    global_module.__name__: global_module,
                    compact_module.__name__: compact_module,
                },
            ),
            mock.patch.object(self.fake_cutlass, "Int32", int32, create=True),
        ):
            self.top_k._make_local_to_global_launcher.cache_clear()
            self.top_k._make_compactify_launcher.cache_clear()
            fixed = self.top_k._make_local_to_global_launcher(
                is_varlen=False,
                seqlen_k=32,
            )
            packed = self.top_k._make_local_to_global_launcher(
                is_varlen=True,
                seqlen_k=32,
            )
            compact = self.top_k._make_compactify_launcher(rows=6, cols=8)
            fixed("stream", "local", "global")
            packed("stream", "local", "cuq", "cuk", "global")
            compact("stream", "indices", "out", "length")
            self.top_k._make_local_to_global_launcher.cache_clear()
            self.top_k._make_compactify_launcher.cache_clear()

        self.assertEqual(
            calls,
            [
                (
                    "global",
                    (
                        "local",
                        "global",
                        ("Int32", 32),
                        None,
                        None,
                        "stream",
                    ),
                ),
                (
                    "global",
                    (
                        "local",
                        "global",
                        ("Int32", 32),
                        "cuq",
                        "cuk",
                        "stream",
                    ),
                ),
                (
                    "compact",
                    (
                        "indices",
                        "out",
                        "length",
                        ("Int32", 6),
                        "stream",
                    ),
                ),
            ],
        )

    def test_dense_score_wrappers_declare_initialized_results(self):
        captured = []

        def fake_call(launcher, inputs, **options):
            captured.append(dict(launcher=launcher, inputs=inputs, **options))
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        q = _Array((2, 4, 32, 128), self.bfloat16)
        k = _Array((2, 8, 1, 128), self.bfloat16)
        weights = _Array((2, 4, 32), self.bfloat16)
        lse = _Array((2, 4, 32), self.float32)
        q_causal_offsets = _Array((2,), self.int32)
        indexer_launcher = object()
        attention_launcher = object()

        with (
            self._optional_modules(),
            mock.patch.object(
                self.score,
                "_make_dense_launcher",
                side_effect=(indexer_launcher, attention_launcher),
            ) as make_launcher,
            mock.patch.object(self.score, "call_cutedsl", side_effect=fake_call),
        ):
            indexer = self.score.dense_indexer_score_recompute_wrapper(
                q,
                k,
                weights,
                ratio=2,
                q_causal_offsets=q_causal_offsets,
            )
            attention = self.score.dense_attn_score_recompute_wrapper(
                q,
                k,
                lse,
                softmax_scale=0.125,
            )

        for result in (indexer, attention):
            self.assertEqual((result.out.shape, result.out.dtype), ((2, 4, 8), self.float32))
            self.assertEqual((result.denom.shape, result.denom.dtype), ((2, 4), self.float32))

        indexer_call, attention_call = captured
        self.assertIs(indexer_call["launcher"], indexer_launcher)
        self.assertEqual(
            indexer_call["inputs"],
            (q, k, weights, q_causal_offsets),
        )
        self.assertEqual(indexer_call["outputs"][0].name, "out")
        self.assertEqual(indexer_call["outputs"][0].fill_value, float("-inf"))
        self.assertEqual(indexer_call["outputs"][1].name, "denom")
        self.assertIsNone(indexer_call["outputs"][1].fill_value)
        self.assertTrue(indexer_call["use_static_tensors"])

        self.assertIs(attention_call["launcher"], attention_launcher)
        self.assertEqual(attention_call["inputs"], (q, k, lse))
        self.assertEqual(attention_call["outputs"][0].fill_value, float("-inf"))
        self.assertTrue(attention_call["use_static_tensors"])

        indexer_config = make_launcher.call_args_list[0].kwargs
        self.assertEqual(indexer_config["score_type"], "indexer")
        self.assertEqual(indexer_config["ratio"], 2)
        self.assertTrue(indexer_config["have_q_causal_offsets"])
        attention_config = make_launcher.call_args_list[1].kwargs
        self.assertEqual(attention_config["score_type"], "attention")
        self.assertEqual(attention_config["scale"], 0.125)
        self.assertFalse(attention_config["have_q_causal_offsets"])

    def test_sparse_score_wrappers_declare_functional_results(self):
        captured = []

        def fake_call(launcher, inputs, **options):
            captured.append(dict(launcher=launcher, inputs=inputs, **options))
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        q = _Array((2, 4, 32, 128), self.bfloat16)
        k = _Array((2, 256, 128), self.bfloat16)
        weights = _Array((2, 4, 32), self.bfloat16)
        lse = _Array((2, 4, 32), self.float32)
        indices = _Array((2, 4, 128), self.int32)
        lengths = _Array((2, 4), self.int32)
        indexer_launcher = object()
        attention_launcher = object()

        with (
            self._optional_modules(),
            mock.patch.object(
                self.score,
                "_make_launcher",
                side_effect=(indexer_launcher, attention_launcher),
            ) as make_launcher,
            mock.patch.object(self.score, "call_cutedsl", side_effect=fake_call),
        ):
            predict = self.score.sparse_indexer_score_recompute_wrapper(
                q,
                k,
                weights,
                indices,
            ).predict
            target = self.score.sparse_attn_score_recompute_wrapper(
                q,
                k,
                lse,
                indices,
                0.125,
                topk_length=lengths,
            ).target

        self.assertEqual((predict.shape, predict.dtype), ((2, 4, 128), self.float32))
        self.assertEqual((target.shape, target.dtype), ((2, 4, 128), self.float32))

        indexer_call, attention_call = captured
        self.assertIs(indexer_call["launcher"], indexer_launcher)
        self.assertEqual(indexer_call["inputs"], (q, k, weights, indices))
        self.assertEqual(len(indexer_call["outputs"]), 1)
        self.assertEqual(indexer_call["outputs"][0].name, "predict")
        self.assertEqual(len(indexer_call["workspaces"]), 1)
        self.assertEqual(
            (
                indexer_call["workspaces"][0].name,
                indexer_call["workspaces"][0].shape,
                indexer_call["workspaces"][0].dtype,
            ),
            ("topk_length_workspace", (1, 1), self.int32),
        )

        self.assertIs(attention_call["launcher"], attention_launcher)
        self.assertEqual(attention_call["inputs"], (q, k, lse, indices, lengths))
        self.assertEqual(attention_call["outputs"][0].name, "target")
        self.assertEqual(attention_call["workspaces"], ())
        self.assertTrue(indexer_call["use_static_tensors"])
        self.assertTrue(attention_call["use_static_tensors"])

        indexer_config = make_launcher.call_args_list[0].kwargs
        self.assertEqual(indexer_config["score_type"], "indexer")
        self.assertEqual(indexer_config["n_block_size"], 128)
        self.assertFalse(indexer_config["have_topk_length"])
        attention_config = make_launcher.call_args_list[1].kwargs
        self.assertEqual(attention_config["score_type"], "attention")
        self.assertEqual(attention_config["n_block_size"], 64)
        self.assertTrue(attention_config["have_topk_length"])
        self.assertEqual(attention_config["softmax_scale"], 0.125)

    def test_sparse_score_launcher_preserves_native_argument_order(self):
        calls = []

        class FakeKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args):
                calls.append(args)

        kernel_module_name = f"{_TEST_PACKAGE}.deepseek_sparse_attention.score_recompute." "sparse_score_recompute_sm100"
        kernel_module = types.ModuleType(kernel_module_name)
        kernel_module.SparseScoreRecomputeSm100 = FakeKernel
        float32 = lambda value: ("Float32", value)

        common = dict(
            score_type="indexer",
            head_dim=128,
            qhead_per_kv_head=32,
            topk=128,
            m_block_size=32,
            n_block_size=128,
            k_block_size=None,
            kv_stage=4,
            topk_in_smem=True,
            topk_indices_global=False,
            softmax_scale=1.0,
        )
        with (
            self._optional_modules(),
            mock.patch.dict(sys.modules, {kernel_module_name: kernel_module}),
            mock.patch.object(self.fake_cutlass, "Float32", float32, create=True),
        ):
            self.score._make_launcher.cache_clear()
            with_length = self.score._make_launcher(
                **common,
                have_topk_length=True,
            )
            without_length = self.score._make_launcher(
                **common,
                have_topk_length=False,
            )
            with_length("stream", "q", "k", "aux", "indices", "length", "out")
            without_length(
                "stream",
                "q",
                "k",
                "aux",
                "indices",
                "out",
                "dummy_length",
            )
            self.score._make_launcher.cache_clear()

        self.assertEqual(
            calls,
            [
                (
                    "q",
                    "k",
                    "aux",
                    "indices",
                    "out",
                    "length",
                    ("Float32", 1.0),
                    "stream",
                ),
                (
                    "q",
                    "k",
                    "aux",
                    "indices",
                    "out",
                    "dummy_length",
                    ("Float32", 1.0),
                    "stream",
                ),
            ],
        )

    def test_dense_score_launcher_preserves_native_argument_order(self):
        calls = []

        class FakeKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args):
                calls.append(args)

        kernel_module_name = f"{_TEST_PACKAGE}.deepseek_sparse_attention.score_recompute." "dense_score_recompute_sm100"
        kernel_module = types.ModuleType(kernel_module_name)
        kernel_module.DenseScoreRecomputeSm100 = FakeKernel
        float32 = lambda value: ("Float32", value)
        int32 = lambda value: ("Int32", value)

        with (
            self._optional_modules(),
            mock.patch.dict(sys.modules, {kernel_module_name: kernel_module}),
            mock.patch.object(self.fake_cutlass, "Float32", float32, create=True),
            mock.patch.object(self.fake_cutlass, "Int32", int32, create=True),
        ):
            self.score._make_dense_launcher.cache_clear()
            fixed = self.score._make_dense_launcher(
                score_type="indexer",
                head_dim=128,
                qhead_per_kv_head=32,
                ratio=1,
                max_seqlen_q=4,
                max_seqlen_k=8,
                scale=0.5,
                have_q_causal_offsets=False,
            )
            with_offsets = self.score._make_dense_launcher(
                score_type="attention",
                head_dim=128,
                qhead_per_kv_head=32,
                ratio=2,
                max_seqlen_q=4,
                max_seqlen_k=8,
                scale=0.125,
                have_q_causal_offsets=True,
            )
            fixed("stream", "q", "k", "weights", "out", "denom")
            with_offsets(
                "stream",
                "q",
                "k",
                "lse",
                "offsets",
                "out",
                "denom",
            )
            self.score._make_dense_launcher.cache_clear()

        self.assertEqual(
            calls,
            [
                (
                    "q",
                    "k",
                    "weights",
                    "out",
                    "denom",
                    ("Float32", 0.5),
                    ("Int32", 4),
                    ("Int32", 8),
                    None,
                    None,
                    None,
                    "stream",
                ),
                (
                    "q",
                    "k",
                    "lse",
                    "out",
                    "denom",
                    ("Float32", 0.125),
                    ("Int32", 4),
                    ("Int32", 8),
                    None,
                    None,
                    "offsets",
                    "stream",
                ),
            ],
        )

    def test_sparse_score_config_matches_sm100_dispatch(self):
        indexer = self.score_config.resolve_sparse_score_kernel_config(
            score_type="indexer",
            head_dim=128,
            qhead_per_kv_head=32,
            topk=128,
            have_topk_length=False,
        )
        self.assertEqual(
            (indexer.m_block_size, indexer.n_block_size, indexer.k_block_size),
            (32, 128, None),
        )
        self.assertEqual(indexer.kv_stage, 4)
        self.assertTrue(indexer.topk_in_smem)

        large_topk = self.score_config.resolve_sparse_score_kernel_config(
            score_type="indexer",
            head_dim=128,
            qhead_per_kv_head=32,
            topk=32768,
            have_topk_length=False,
        )
        self.assertFalse(large_topk.topk_in_smem)

        compact_attention = self.score_config.resolve_sparse_score_kernel_config(
            score_type="attention",
            head_dim=512,
            qhead_per_kv_head=64,
            topk=512,
            have_topk_length=True,
        )
        full_attention = self.score_config.resolve_sparse_score_kernel_config(
            score_type="attention",
            head_dim=512,
            qhead_per_kv_head=64,
            topk=512,
            have_topk_length=False,
        )
        self.assertEqual(
            (
                compact_attention.m_block_size,
                compact_attention.n_block_size,
                compact_attention.k_block_size,
            ),
            (64, 64, None),
        )
        self.assertEqual(
            (
                full_attention.m_block_size,
                full_attention.n_block_size,
                full_attention.k_block_size,
            ),
            (64, 128, 256),
        )

        with self.assertRaisesRegex(ValueError, "multiple of the selected n_block_size"):
            self.score_config.resolve_sparse_score_kernel_config(
                score_type="indexer",
                head_dim=128,
                qhead_per_kv_head=32,
                topk=64,
                have_topk_length=False,
            )

        dense_indexer = self.score_config.resolve_dense_score_kernel_config(
            score_type="indexer",
            head_dim=128,
            qhead_per_kv_head=32,
        )
        self.assertEqual(
            (
                dense_indexer.m_block_size,
                dense_indexer.n_block_size,
                dense_indexer.k_block_size,
                dense_indexer.kv_stage,
            ),
            (64, 128, None, 4),
        )

        dense_attention = self.score_config.resolve_dense_score_kernel_config(
            score_type="attention",
            head_dim=512,
            qhead_per_kv_head=64,
        )
        self.assertEqual(
            (
                dense_attention.m_block_size,
                dense_attention.n_block_size,
                dense_attention.k_block_size,
                dense_attention.kv_stage,
            ),
            (128, 128, 64, 4),
        )

    def test_wrapper_validation_fails_before_launch(self):
        bad_q = _Array((1, 4, 32, 128), self.float16)
        k = _Array((1, 5, 1, 128), self.bfloat16)
        w = _Array((1, 4, 32), self.bfloat16)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(
                ValueError,
                "dtype bfloat16",
            ),
            mock.patch.object(self.forward, "_make_launcher") as make_forward,
        ):
            self.forward.indexer_forward_wrapper(bad_q, k, w, ratio=1)
        make_forward.assert_not_called()

        input_values = _Array((2, 64), self.float32)
        wrong_batch = _Array((1,), self.int32)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(
                ValueError,
                "num_rows.*must equal",
            ),
            mock.patch.object(self.top_k, "_make_launcher") as make_top_k,
        ):
            self.top_k.indexer_top_k_wrapper(
                input_values,
                wrong_batch,
                top_k=8,
            )
        make_top_k.assert_not_called()


if __name__ == "__main__":
    unittest.main()
