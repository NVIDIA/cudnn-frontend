# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU-only contract tests for the CuTe DSL JAX tracing adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

try:
    import pytest
except ImportError:
    # Keep this contract test runnable with the standard library alone.
    pass
else:
    pytestmark = pytest.mark.L0


def _identity_jit(fn=None, **kwargs):
    def decorate(decorated_fn):
        decorated_fn._cute_jit_options = kwargs
        return decorated_fn

    return decorate if fn is None else decorate(fn)


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_TEST_PACKAGE = "cudnn_frontend_jax_api_base_adapter_test"
_PARENT = types.ModuleType(_TEST_PACKAGE)
_PARENT.__path__ = [str(_CUDNN_ROOT)]
_JAX_PACKAGE = types.ModuleType(f"{_TEST_PACKAGE}._jax")
_JAX_PACKAGE.__path__ = [str(_CUDNN_ROOT / "_jax")]

_MODULE_PATH = _CUDNN_ROOT / "_jax" / "api_base.py"
_SPEC = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}._jax.api_base", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE

_bootstrap_jnp = types.ModuleType("jax.numpy")
_bootstrap_jnp.dtype = lambda value: value
_bootstrap_jax = types.ModuleType("jax")
_bootstrap_jax.__path__ = []
_bootstrap_jax.numpy = _bootstrap_jnp
_bootstrap_jax.tree_util = types.SimpleNamespace(
    DictKey=lambda key: key,
    register_pytree_with_keys=lambda *_args: None,
)
_bootstrap_cutlass_jax = types.ModuleType("cutlass.jax")
_bootstrap_cutlass_jax.TensorSpec = type("TensorSpec", (), {})
_bootstrap_cute = types.ModuleType("cutlass.cute")
_bootstrap_cute.jit = _identity_jit
_bootstrap_cutlass = types.ModuleType("cutlass")
_bootstrap_cutlass.__path__ = []
_bootstrap_cutlass.Constexpr = object
_bootstrap_cutlass.cute = _bootstrap_cute
_bootstrap_cutlass.jax = _bootstrap_cutlass_jax
with mock.patch.dict(
    sys.modules,
    {
        "jax": _bootstrap_jax,
        "jax.numpy": _bootstrap_jnp,
        "cutlass": _bootstrap_cutlass,
        "cutlass.cute": _bootstrap_cute,
        "cutlass.jax": _bootstrap_cutlass_jax,
        _TEST_PACKAGE: _PARENT,
        f"{_TEST_PACKAGE}._jax": _JAX_PACKAGE,
    },
):
    _SPEC.loader.exec_module(_MODULE)

BufferSpec = _MODULE.BufferSpec
call_cutedsl = _MODULE.call_cutedsl


class _Array:
    def __init__(self, shape, dtype, label):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.label = label

    def __repr__(self):
        return f"_Array({self.label!r}, {self.shape!r}, {self.dtype!r})"


class _ShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _FakeJax:
    ShapeDtypeStruct = _ShapeDtypeStruct


class _FakeJaxNumpy:
    def __init__(self):
        self.allocations = []

    def full(self, shape, value, dtype):
        result = _Array(shape, dtype, f"full({value})")
        self.allocations.append(("full", result, value))
        return result


@dataclass(frozen=True)
class _FakeCutlassTensorSpec:
    """Dependency-free stand-in with CUTLASS 4.5's TensorSpec fields."""

    layout: object = None
    mode: object = None
    static: object = None
    ptr_assumed_align: object = 256
    divisibility: object = None


class _FakeCutlassJax:
    TensorSpec = _FakeCutlassTensorSpec

    def __init__(self):
        self.calls = []

    def cutlass_call(self, fn, **options):
        self.calls.append((fn, options))
        known_options = {
            "output_shape_dtype",
            "input_spec",
            "output_spec",
            "input_output_aliases",
            "allow_cuda_graph",
            "compile_options",
            "use_static_tensors",
        }
        static_args = {k: v for k, v in options.items() if k not in known_options}

        def invoke(*inputs):
            aliases = options["input_output_aliases"]
            alias_by_output = {output_idx: input_idx for input_idx, output_idx in aliases.items()}
            results = []
            fresh_results = []
            for result_idx, metadata in enumerate(options["output_shape_dtype"]):
                if result_idx in alias_by_output:
                    result = inputs[alias_by_output[result_idx]]
                else:
                    result = _Array(
                        metadata.shape,
                        metadata.dtype,
                        f"result-{result_idx}",
                    )
                    fresh_results.append(result)
                results.append(result)

            fn("xla-stream", *inputs, *fresh_results, **static_args)
            return tuple(results)

        return invoke


class CallCutedslAdapterTest(unittest.TestCase):
    def _call(self, kernel, inputs, **kwargs):
        fake_jnp = _FakeJaxNumpy()
        fake_cutlass_jax = _FakeCutlassJax()
        with (
            mock.patch.object(_MODULE, "jax", _FakeJax),
            mock.patch.object(_MODULE, "jnp", fake_jnp),
            mock.patch.object(_MODULE, "cutlass_jax", fake_cutlass_jax),
        ):
            result = call_cutedsl(kernel, inputs, **kwargs)
        return result, fake_jnp, fake_cutlass_jax

    def test_launch_adapter_uses_compile_time_tracing(self):
        self.assertEqual(_MODULE._launch_adapter._cute_jit_options, {"preprocess": False})

    def test_workspace_is_passed_to_launcher_but_hidden_from_results(self):
        seen = []

        def kernel(stream, x, output, workspace, *, scale):
            seen.append((stream, x, output, workspace, scale))

        x = _Array((8,), "f32", "x")
        result, _, bridge = self._call(
            kernel,
            (x,),
            outputs=(BufferSpec("output", (8,), "f32"),),
            workspaces=(BufferSpec("workspace", (128,), "u8"),),
            static_args={"scale": 2.0},
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "result-0")
        self.assertEqual(seen[0][0], "xla-stream")
        self.assertIs(seen[0][1], x)
        self.assertEqual(seen[0][2].label, "result-0")
        self.assertEqual(seen[0][3].label, "result-1")
        self.assertEqual(seen[0][4], 2.0)
        self.assertEqual(len(bridge.calls[0][1]["output_shape_dtype"]), 2)

    def test_initialized_buffers_are_inputs_aliased_to_results(self):
        seen = []

        def kernel(stream, x, output, workspace):
            seen.append((output, workspace))

        x = _Array((4,), "bf16", "x")
        output_tensor_spec = _FakeCutlassTensorSpec(divisibility=(4,))
        workspace_tensor_spec = _FakeCutlassTensorSpec(ptr_assumed_align=128)
        result, fake_jnp, bridge = self._call(
            kernel,
            (x,),
            outputs=(
                BufferSpec(
                    "output",
                    (4,),
                    "bf16",
                    tensor_spec=output_tensor_spec,
                    fill_value=float("-inf"),
                ),
            ),
            workspaces=(
                BufferSpec(
                    "workspace",
                    (32,),
                    "u8",
                    tensor_spec=workspace_tensor_spec,
                    fill_value=0,
                ),
            ),
        )

        aliases = bridge.calls[0][1]["input_output_aliases"]
        self.assertEqual(aliases, {1: 0, 2: 1})
        input_specs = bridge.calls[0][1]["input_spec"]
        output_specs = bridge.calls[0][1]["output_spec"]
        self.assertIsNone(input_specs[0])
        self.assertIs(input_specs[1], output_tensor_spec)
        self.assertIs(input_specs[2], workspace_tensor_spec)
        self.assertIs(output_specs[0], output_tensor_spec)
        self.assertIs(output_specs[1], workspace_tensor_spec)
        self.assertIs(result[0], fake_jnp.allocations[0][1])
        self.assertIs(seen[0][0], fake_jnp.allocations[0][1])
        self.assertIs(seen[0][1], fake_jnp.allocations[1][1])
        self.assertEqual(fake_jnp.allocations[0][0], "full")
        self.assertEqual(fake_jnp.allocations[1][0], "full")
        self.assertEqual(fake_jnp.allocations[1][2], 0)

    def test_native_tensor_specs_are_forwarded_without_conversion(self):
        input_spec = _FakeCutlassTensorSpec(
            layout=(1, 0),
            mode=(1, 0),
            static=True,
            ptr_assumed_align=128,
            # CUTLASS accepts -1 divisibility sentinels. The adapter must not
            # reinterpret or narrow the native TensorSpec contract.
            divisibility=(-1, -1),
        )
        output_spec = _FakeCutlassTensorSpec(
            layout=(1, 0),
            mode=(1, 0),
            static=True,
            ptr_assumed_align=128,
            divisibility=(16, 8),
        )

        result, _, bridge = self._call(
            lambda stream, x, output: None,
            (_Array((8, 16), "f32", "x"),),
            outputs=(
                BufferSpec(
                    "output",
                    (8, 16),
                    "f32",
                    tensor_spec=output_spec,
                ),
            ),
            input_specs=(input_spec,),
            allow_cuda_graph=False,
            compile_options="--example-option",
        )

        self.assertEqual(len(result), 1)
        options = bridge.calls[0][1]
        self.assertIs(options["input_spec"][0], input_spec)
        self.assertIs(options["output_spec"][0], output_spec)
        self.assertFalse(options["allow_cuda_graph"])
        self.assertEqual(options["compile_options"], "--example-option")

    def test_default_result_spec_uses_cutlass_inference(self):
        _, _, bridge = self._call(
            lambda stream, x, output: None,
            (_Array((8,), "f32", "x"),),
            outputs=(BufferSpec("output", (8,), "f32"),),
        )

        self.assertIsNone(bridge.calls[0][1]["output_spec"][0])

    def test_static_tensors_default_to_true(self):
        _, _, bridge = self._call(
            lambda stream, x, output: None,
            (_Array((8,), "f32", "x"),),
            outputs=(BufferSpec("output", (8,), "f32"),),
        )

        self.assertTrue(bridge.calls[0][1]["use_static_tensors"])

    def test_static_tensors_can_be_disabled(self):
        _, _, bridge = self._call(
            lambda stream, x, output: None,
            (_Array((8,), "f32", "x"),),
            outputs=(BufferSpec("output", (8,), "f32"),),
            use_static_tensors=False,
        )

        self.assertFalse(bridge.calls[0][1]["use_static_tensors"])

    def test_rejects_invalid_call_plans(self):
        with self.assertRaisesRegex(ValueError, "at least one public output"):
            self._call(lambda stream, x: None, (_Array((1,), "f32", "x"),), outputs=())

        with self.assertRaisesRegex(ValueError, "must be unique"):
            self._call(
                lambda stream, x, y, z: None,
                (_Array((1,), "f32", "x"),),
                outputs=(BufferSpec("same", (1,), "f32"),),
                workspaces=(BufferSpec("same", (1,), "u8"),),
            )

        with self.assertRaisesRegex(ValueError, "Expected 1 input tensor spec"):
            self._call(
                lambda stream, x, y: None,
                (_Array((1,), "f32", "x"),),
                outputs=(BufferSpec("output", (1,), "f32"),),
                input_specs=(),
            )

        with self.assertRaisesRegex(TypeError, "flat sequence"):
            self._call(
                lambda stream, x, y: None,
                ((_Array((1,), "f32", "nested"),),),
                outputs=(BufferSpec("output", (1,), "f32"),),
            )

    def test_kernel_static_args_do_not_collide_with_cutlass_call_options(self):
        seen = []

        def kernel(stream, x, output, *, compile_options):
            seen.append((stream, x, output, compile_options))

        x = _Array((1,), "f32", "x")
        result, _, bridge = self._call(
            kernel,
            (x,),
            outputs=(BufferSpec("output", (1,), "f32"),),
            static_args={"compile_options": "kernel option"},
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(seen[0][3], "kernel option")
        self.assertIsNone(bridge.calls[0][1]["compile_options"])
        self.assertIs(bridge.calls[0][0], _MODULE._launch_adapter)
        self.assertIsInstance(hash(bridge.calls[0][1]["config"]), int)

    def test_static_launcher_config_has_a_value_stable_cache_key(self):
        def kernel(stream, x, output, *, scale):
            del stream, x, output, scale

        x = _Array((1,), "f32", "x")
        options = {
            "outputs": (BufferSpec("output", (1,), "f32"),),
            "static_args": {"scale": 2.0},
        }
        _, _, first_bridge = self._call(kernel, (x,), **options)
        _, _, second_bridge = self._call(kernel, (x,), **options)
        _, _, different_bridge = self._call(
            kernel,
            (x,),
            outputs=options["outputs"],
            static_args={"scale": 3.0},
        )

        first_config = first_bridge.calls[0][1]["config"]
        second_config = second_bridge.calls[0][1]["config"]
        different_config = different_bridge.calls[0][1]["config"]
        self.assertEqual(first_config, second_config)
        self.assertEqual(hash(first_config), hash(second_config))
        self.assertNotEqual(first_config, different_config)


if __name__ == "__main__":
    unittest.main()
