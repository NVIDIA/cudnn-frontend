# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Source contracts for packed-THD JAX SDPA."""

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
_FORWARD = _CUDNN_ROOT / "sdpa" / "fwd" / "jax.py"
_BACKWARD = _CUDNN_ROOT / "sdpa" / "bwd" / "jax.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _function(nodes: list[ast.stmt], name: str) -> ast.FunctionDef:
    return next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _argument_names(function: ast.FunctionDef) -> tuple[str, ...]:
    return tuple(argument.arg for argument in function.args.args)


def _jit_static_argnames(function: ast.FunctionDef) -> tuple[str, ...]:
    decorator = next(
        node
        for node in function.decorator_list
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "partial"
    )
    keyword = next(
        keyword for keyword in decorator.keywords if keyword.arg == "static_argnames"
    )
    return ast.literal_eval(keyword.value)


class JaxSdpaThdContractTest(unittest.TestCase):
    def test_constructors_bind_cumulative_metadata_and_static_bounds(self):
        for path, class_name in (
            (_FORWARD, "SdpafwdSm100D256"),
            (_BACKWARD, "SdpabwdSm100D256"),
        ):
            constructor = _function(_class(_tree(path), class_name).body, "__init__")
            arguments = _argument_names(constructor)
            with self.subTest(path=path.relative_to(_CUDNN_ROOT)):
                self.assertIn("sample_cum_seqlen_q", arguments)
                self.assertIn("sample_cum_seqlen_k", arguments)
                self.assertIn("max_s_q", arguments)
                self.assertIn("max_s_k", arguments)

    def test_functional_wrappers_keep_offsets_dynamic_and_bounds_static(self):
        for path, wrapper_name in (
            (_FORWARD, "sdpa_fwd_wrapper_sm100_d256"),
            (_BACKWARD, "sdpa_bwd_wrapper_sm100_d256"),
        ):
            wrapper = _function(_tree(path).body, wrapper_name)
            arguments = _argument_names(wrapper)
            static_arguments = _jit_static_argnames(wrapper)
            with self.subTest(path=path.relative_to(_CUDNN_ROOT)):
                self.assertIn("cum_seqlen_q_tensor", arguments)
                self.assertIn("cum_seqlen_k_tensor", arguments)
                self.assertNotIn("cum_seqlen_q_tensor", static_arguments)
                self.assertNotIn("cum_seqlen_k_tensor", static_arguments)
                self.assertIn("max_s_q", static_arguments)
                self.assertIn("max_s_k", static_arguments)

    def test_varlen_launchers_follow_stream_inputs_outputs_workspaces_order(self):
        expected = {
            _FORWARD: (
                "stream",
                "q",
                "k",
                "v",
                "cum_seqlen_q",
                "cum_seqlen_k",
                "output",
                "lse",
            ),
            _BACKWARD: (
                "stream",
                "q",
                "k",
                "v",
                "output",
                "doutput",
                "lse",
                "cum_seqlen_q",
                "cum_seqlen_k",
                "dq",
                "dk",
                "dv",
                "workspace",
            ),
        }
        class_names = {
            _FORWARD: "SdpafwdSm100D256",
            _BACKWARD: "SdpabwdSm100D256",
        }
        for path, arguments in expected.items():
            launcher = _function(
                _class(_tree(path), class_names[path]).body,
                "_launch_varlen_kernel",
            )
            with self.subTest(path=path.relative_to(_CUDNN_ROOT)):
                self.assertEqual(_argument_names(launcher)[1:], arguments)

    def test_thd_is_promoted_without_changing_public_result_ranks(self):
        forward = _FORWARD.read_text()
        backward = _BACKWARD.read_text()

        for source in (forward, backward):
            self.assertIn("JaxTensorDesc.from_shape(", source)
            self.assertIn("(1, *sample_q.shape)", source)
            self.assertIn("jnp.reshape(q_tensor, self.q_kernel_desc.shape)", source)
            self.assertIn("max_s_q and max_s_k are both required for THD", source)

        self.assertIn("jnp.reshape(output, self.o_desc.shape)", forward)
        self.assertIn("(self.total_q_tokens, self.num_query_heads)", forward)
        self.assertIn("public_stride_order=(0, 1)", forward)
        for gradient in ("dq", "dk", "dv"):
            self.assertIn(
                f"{gradient} = jnp.reshape({gradient}, self.{gradient}_desc.shape)",
                backward,
            )
        self.assertIn("(1, self.num_query_heads, self.total_q_tokens)", backward)
        self.assertNotIn("public_stride_order=(1, 2, 0)", backward)

    def test_backward_workspace_is_initialized_and_adapters_do_not_import_torch(self):
        backward_tree = _tree(_BACKWARD)
        init_values = {
            ast.unparse(keyword.value)
            for call in ast.walk(backward_tree)
            if isinstance(call, ast.Call)
            for keyword in call.keywords
            if keyword.arg == "init_value"
        }
        self.assertIn("0", init_values)

        for path in (_FORWARD, _BACKWARD):
            imports = [
                node
                for node in ast.walk(_tree(path))
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            with self.subTest(path=path.relative_to(_CUDNN_ROOT)):
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


if __name__ == "__main__":
    unittest.main()
