# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Source contracts shared by the JAX NSA and d=256 SDPA adapters."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.L0


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_ADAPTERS = {
    "selection": _CUDNN_ROOT / "native_sparse_attention" / "selection" / "jax.py",
    "compression": _CUDNN_ROOT / "native_sparse_attention" / "compression" / "jax.py",
    "sliding": _CUDNN_ROOT
    / "native_sparse_attention"
    / "sliding_window_attention"
    / "jax.py",
    "top_k": _CUDNN_ROOT / "native_sparse_attention" / "top_k" / "jax.py",
    "sdpa_fwd": _CUDNN_ROOT / "sdpa" / "fwd" / "jax.py",
    "sdpa_bwd": _CUDNN_ROOT / "sdpa" / "bwd" / "jax.py",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class JaxNsaSdpaSourceContractTest(unittest.TestCase):
    def test_layout_helpers_separate_logical_modes_from_physical_storage(self):
        nsa_helpers = (
            _CUDNN_ROOT / "native_sparse_attention" / "jax_utils.py"
        ).read_text()
        sdpa_helpers = (_CUDNN_ROOT / "sdpa" / "jax_utils.py").read_text()

        self.assertNotIn("BHSD_STORAGE_ORDER", nsa_helpers)
        self.assertNotIn("BHSD_STORAGE_ORDER", sdpa_helpers)
        self.assertIn("mode_from_layout(layout, kernel_axes=kernel_axes)", nsa_helpers)
        self.assertIn("kernel_stride_order is None", nsa_helpers)
        self.assertIn('KERNEL_AXES = "BSHD"', sdpa_helpers)
        self.assertIn("KERNEL_STRIDE_ORDER = (3, 2, 1, 0)", sdpa_helpers)

    def test_all_adapters_are_torch_independent_jax_api_classes(self):
        expected_classes = {
            "selection": "SelectionAttention",
            "compression": "CompressionAttention",
            "sliding": "SlidingWindowAttention",
            "top_k": "TopKReduction",
            "sdpa_fwd": "SdpafwdSm100D256",
            "sdpa_bwd": "SdpabwdSm100D256",
        }
        for operation, path in _ADAPTERS.items():
            with self.subTest(operation=operation):
                tree = _tree(path)
                imports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]
                self.assertFalse(
                    any(
                        (
                            isinstance(node, ast.Import)
                            and any(alias.name == "torch" for alias in node.names)
                        )
                        or (isinstance(node, ast.ImportFrom) and node.module == "torch")
                        for node in imports
                    )
                )
                adapter = _class(tree, expected_classes[operation])
                self.assertEqual(
                    [base.id for base in adapter.bases if isinstance(base, ast.Name)],
                    ["JaxApiBase"],
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.FunctionDef)
                        and node.name == "check_support"
                        for node in adapter.body
                    )
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.FunctionDef) and node.name == "__call__"
                        for node in adapter.body
                    )
                )

    def test_cute_launchers_follow_stream_inputs_outputs_workspaces_order(self):
        expected = {
            "selection": (
                "stream",
                "q",
                "k",
                "v",
                "block_indices",
                "block_counts",
                "cum_seqlen",
                "output",
                "lse_sum",
                "row_max",
            ),
            "compression": ("stream", "q", "k", "v", "output"),
            "top_k": ("stream", "q", "k", "lse", "topk_scores", "topk_indices"),
            "sdpa_fwd": ("stream", "q", "k", "v", "output", "lse"),
            "sdpa_bwd": (
                "stream",
                "q",
                "k",
                "v",
                "output",
                "doutput",
                "lse",
                "dq",
                "dk",
                "dv",
                "workspace",
            ),
        }
        class_names = {
            "selection": "SelectionAttention",
            "compression": "CompressionAttention",
            "top_k": "TopKReduction",
            "sdpa_fwd": "SdpafwdSm100D256",
            "sdpa_bwd": "SdpabwdSm100D256",
        }
        for operation, argument_names in expected.items():
            with self.subTest(operation=operation):
                launcher = _method(
                    _class(_tree(_ADAPTERS[operation]), class_names[operation]),
                    "_launch_kernel",
                )
                actual = tuple(argument.arg for argument in launcher.args.args[1:])
                self.assertEqual(actual, argument_names)

    def test_functional_buffers_preserve_required_initial_values(self):
        expected_literals = {
            "selection": {"0", "0.0", "float('-inf')"},
            "top_k": {"float('-inf')", "-1"},
            "sdpa_bwd": {"0"},
        }
        for operation, expected in expected_literals.items():
            with self.subTest(operation=operation):
                values = set()
                for call in (
                    node
                    for node in ast.walk(_tree(_ADAPTERS[operation]))
                    if isinstance(node, ast.Call)
                ):
                    for keyword in call.keywords:
                        if keyword.arg == "init_value":
                            values.add(ast.unparse(keyword.value))
                self.assertTrue(
                    expected.issubset(values),
                    f"{operation} initialized buffers changed: {values}",
                )

    def test_selection_uses_one_self_attention_offset_operand(self):
        tree = _tree(_ADAPTERS["selection"])
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "selection_attention_wrapper"
        )
        argument_names = tuple(argument.arg for argument in wrapper.args.args)

        self.assertIn("cum_seqlen_tensor", argument_names)
        self.assertNotIn("cum_seqlen_q_tensor", argument_names)
        self.assertNotIn("cum_seqlen_k_tensor", argument_names)
        self.assertIn("(1, 2, 4, 8, 16)", _ADAPTERS["selection"].read_text())

    def test_persistent_compression_resolves_hardware_before_lowering(self):
        tree = _tree(_ADAPTERS["compression"])
        adapter = _class(tree, "CompressionAttention")
        check_support = _method(adapter, "check_support")
        runner = _method(adapter, "_run_kernel")

        self.assertIn(
            "self._get_device_multiprocessor_count()",
            ast.unparse(check_support),
        )
        self.assertIn("persistent_sm_count", ast.unparse(runner))
        self.assertNotIn("HardwareInfo", _ADAPTERS["compression"].read_text())

    def test_compression_supports_fixed_bhsd_and_packed_thd(self):
        tree = _tree(_ADAPTERS["compression"])
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "compression_attention_wrapper"
        )
        argument_names = {
            argument.arg for argument in (*wrapper.args.args, *wrapper.args.kwonlyargs)
        }
        varlen_launcher = _method(
            _class(tree, "CompressionAttention"), "_launch_varlen_kernel"
        )

        self.assertTrue(
            {
                "cum_seqlen_q_tensor",
                "cum_seqlen_k_tensor",
                "max_s_q",
                "max_s_k",
            }.issubset(argument_names)
        )
        self.assertEqual(
            tuple(argument.arg for argument in varlen_launcher.args.args[1:]),
            (
                "stream",
                "q",
                "k",
                "v",
                "cum_seqlen_q",
                "cum_seqlen_k",
                "output",
            ),
        )
        source = _ADAPTERS["compression"].read_text()
        self.assertIn("normalize_attention_layout(layout, ranks[0])", source)
        self.assertIn("self.input_layout in FIXED_LAYOUTS", source)
        self.assertIn("jnp.float8_e4m3fn", source)
        self.assertIn("self.lse_extent = total_q", source)

    def test_fixed_adapters_expose_named_bhsd_and_bshd_layouts(self):
        for operation in ("compression", "sliding", "top_k", "sdpa_fwd", "sdpa_bwd"):
            source = _ADAPTERS[operation].read_text()
            with self.subTest(operation=operation):
                self.assertIn("layout: str | None", source)
                self.assertIn('"layout",', source)

    def test_compression_packed_views_duplicate_the_token_stride(self):
        tree = _tree(_ADAPTERS["compression"])
        packed_orders = [
            ast.literal_eval(keyword.value)
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            for keyword in call.keywords
            if keyword.arg == "public_stride_order"
            and ast.unparse(keyword.value) == "(3, 2, 0, 1)"
        ]

        self.assertEqual(len(packed_orders), 4)
        token_count, heads, head_dim = 20, 8, 128
        shape = (1, token_count, heads, head_dim)
        running = 1
        strides = [0] * len(shape)
        for axis in packed_orders[0]:
            strides[axis] = running
            running *= max(shape[axis], 1)
        self.assertEqual(
            tuple(strides),
            (heads * head_dim, heads * head_dim, head_dim, 1),
        )

    def test_sliding_window_supports_stats_two_sided_windows_and_packed_thd(self):
        source = _ADAPTERS["sliding"].read_text()

        self.assertIn("return_residual=not self.is_infer", source)
        self.assertIn("(self.left_bound - 1, self.right_bound)", source)
        self.assertIn("query_seq_lengths=query_lengths", source)
        self.assertIn("key_value_seq_lengths=key_value_lengths", source)
        self.assertIn("jnp.diff(cum_seqlen_q_tensor)", source)
        self.assertIn("self._pad_packed(", source)
        self.assertIn("self._unpad_packed(", source)
        self.assertNotIn("currently supports inference only", source)
        self.assertNotIn("currently requires S_q == S_k", source)

    def test_cutlass_kernels_are_deferred_until_adapter_calls(self):
        kernel_modules = {
            "selection": "NSA_select_attn_fwd_hmma",
            "compression": "fmha",
            "top_k": "nsa_top_k_reduction_fwd",
            "sdpa_fwd": "fmha_forward_sm100_d256",
            "sdpa_bwd": "fmha_backward_sm100_2kernel",
        }
        for operation, module_name in kernel_modules.items():
            with self.subTest(operation=operation):
                tree = _tree(_ADAPTERS[operation])
                top_level_imports = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]
                self.assertFalse(
                    any(
                        isinstance(node, ast.ImportFrom)
                        and (node.module or "").endswith(module_name)
                        for node in top_level_imports
                    )
                )
                self.assertTrue(
                    any(
                        isinstance(node, ast.ImportFrom)
                        and (node.module or "").endswith(module_name)
                        for node in ast.walk(tree)
                    )
                )

    def test_package_initializers_keep_torch_apis_lazy(self):
        initializers = [
            _CUDNN_ROOT / "native_sparse_attention" / "__init__.py",
            *(
                _CUDNN_ROOT / "native_sparse_attention" / name / "__init__.py"
                for name in (
                    "selection",
                    "compression",
                    "sliding_window_attention",
                    "top_k",
                )
            ),
            _CUDNN_ROOT / "sdpa" / "__init__.py",
            _CUDNN_ROOT / "sdpa" / "fwd" / "__init__.py",
            _CUDNN_ROOT / "sdpa" / "bwd" / "__init__.py",
        ]
        for path in initializers:
            with self.subTest(path=path.relative_to(_CUDNN_ROOT)):
                direct_api_imports = [
                    node
                    for node in _tree(path).body
                    if isinstance(node, ast.ImportFrom)
                    and node.module in {"api", ".api"}
                ]
                self.assertEqual(direct_api_imports, [])


if __name__ == "__main__":
    unittest.main()
