# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU-only contract tests for the CuTe DSL JAX tracing adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import pytest
except ImportError:
    # Keep this contract test runnable with the standard library alone.
    pass
else:
    pytestmark = pytest.mark.L0


_MODULE_PATH = Path(__file__).resolve().parents[3] / "python" / "cudnn" / "jax" / "cutedsl.py"
_SPEC = importlib.util.spec_from_file_location("cudnn_jax_cutedsl_poc", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BufferInitialization = _MODULE.BufferInitialization
BufferSpec = _MODULE.BufferSpec
_call_cutedsl_with_modules = _MODULE._call_cutedsl_with_modules


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

    def zeros(self, shape, dtype):
        result = _Array(shape, dtype, "zeros")
        self.allocations.append(("zeros", result, None))
        return result

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
        result = _call_cutedsl_with_modules(
            kernel,
            inputs,
            jax_module=_FakeJax,
            jax_numpy_module=fake_jnp,
            cutlass_jax_module=fake_cutlass_jax,
            **kwargs,
        )
        return result, fake_jnp, fake_cutlass_jax

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
                    initialization=BufferInitialization.VALUE,
                    fill_value=float("-inf"),
                ),
            ),
            workspaces=(
                BufferSpec(
                    "workspace",
                    (32,),
                    "u8",
                    tensor_spec=workspace_tensor_spec,
                    initialization=BufferInitialization.ZERO,
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
        self.assertEqual(fake_jnp.allocations[1][0], "zeros")

    def test_user_alias_is_reconstructed_in_canonical_output_position(self):
        seen = []

        def kernel(stream, x, output, workspace):
            seen.append((x, output, workspace))

        x = _Array((16,), "f16", "x")
        result, _, bridge = self._call(
            kernel,
            (x,),
            outputs=(BufferSpec("output", (16,), "f16"),),
            workspaces=(BufferSpec("workspace", (8,), "u8"),),
            input_output_aliases={0: 0},
        )

        self.assertIs(result[0], x)
        self.assertIs(seen[0][0], x)
        self.assertIs(seen[0][1], x)
        self.assertEqual(seen[0][2].label, "result-1")
        self.assertEqual(bridge.calls[0][1]["input_output_aliases"], {0: 0})

    def test_middle_alias_preserves_fresh_result_order(self):
        seen = []

        def kernel(stream, x, first, aliased, third, workspace):
            seen.append((first, aliased, third, workspace))

        x = _Array((4,), "f32", "x")
        result, _, _ = self._call(
            kernel,
            (x,),
            outputs=(
                BufferSpec("first", (4,), "f32"),
                BufferSpec("aliased", (4,), "f32"),
                BufferSpec("third", (4,), "f32"),
            ),
            workspaces=(BufferSpec("workspace", (16,), "u8"),),
            input_output_aliases={0: 1},
        )

        self.assertEqual([value.label for value in result], ["result-0", "x", "result-2"])
        self.assertEqual(
            [value.label for value in seen[0]],
            ["result-0", "x", "result-2", "result-3"],
        )

    def test_native_tensor_specs_are_forwarded_without_conversion(self):
        input_spec = _FakeCutlassTensorSpec(
            layout=(1, 0),
            mode=(1, 0),
            static=True,
            ptr_assumed_align=128,
            divisibility=(16, 8),
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
            use_static_tensors=True,
        )

        self.assertEqual(len(result), 1)
        options = bridge.calls[0][1]
        self.assertIs(options["input_spec"][0], input_spec)
        self.assertIs(options["output_spec"][0], output_spec)
        self.assertFalse(options["allow_cuda_graph"])
        self.assertEqual(options["compile_options"], "--example-option")
        self.assertTrue(options["use_static_tensors"])

    def test_default_result_spec_uses_cutlass_inference(self):
        _, _, bridge = self._call(
            lambda stream, x, output: None,
            (_Array((8,), "f32", "x"),),
            outputs=(BufferSpec("output", (8,), "f32"),),
        )

        self.assertIsNone(bridge.calls[0][1]["output_spec"][0])

    def test_native_tensor_spec_metadata_is_validated(self):
        invalid_specs = (
            (
                "layout permutation",
                _FakeCutlassTensorSpec(layout=(0, 0)),
                ValueError,
                "layout.*permutation",
            ),
            (
                "mode rank",
                _FakeCutlassTensorSpec(mode=(0,)),
                ValueError,
                "mode.*permutation",
            ),
            (
                "mode element type",
                _FakeCutlassTensorSpec(mode=(False, True)),
                TypeError,
                "mode entries.*integers",
            ),
            (
                "static type",
                _FakeCutlassTensorSpec(static="yes"),
                TypeError,
                "static.*bool",
            ),
            (
                "pointer alignment",
                _FakeCutlassTensorSpec(ptr_assumed_align=3),
                ValueError,
                "ptr_assumed_align.*power of two",
            ),
            (
                "divisibility value",
                _FakeCutlassTensorSpec(divisibility=(16, 0)),
                ValueError,
                "divisibility.*positive",
            ),
            (
                "divisibility rank",
                _FakeCutlassTensorSpec(divisibility=(16,)),
                ValueError,
                "divisibility.*rank 2",
            ),
        )

        for label, tensor_spec, error_type, message in invalid_specs:
            with self.subTest(label=label):
                with self.assertRaisesRegex(error_type, message):
                    self._call(
                        lambda stream, x, output: None,
                        (_Array((8, 16), "f32", "x"),),
                        outputs=(BufferSpec("output", (8, 16), "f32"),),
                        input_specs=(tensor_spec,),
                    )

        with self.assertRaisesRegex(TypeError, "TensorSpec or None"):
            self._call(
                lambda stream, x, output: None,
                (_Array((8, 16), "f32", "x"),),
                outputs=(BufferSpec("output", (8, 16), "f32"),),
                input_specs=(object(),),
            )

        with self.assertRaisesRegex(ValueError, "output.*layout.*permutation"):
            self._call(
                lambda stream, x, output: None,
                (_Array((8, 16), "f32", "x"),),
                outputs=(
                    BufferSpec(
                        "output",
                        (8, 16),
                        "f32",
                        tensor_spec=_FakeCutlassTensorSpec(layout=(0, 0)),
                    ),
                ),
            )

    def test_rejects_invalid_specs_and_aliases(self):
        with self.assertRaisesRegex(ValueError, "at least one public output"):
            self._call(lambda stream, x: None, (_Array((1,), "f32", "x"),), outputs=())

        with self.assertRaisesRegex(ValueError, "must be unique"):
            self._call(
                lambda stream, x, y, z: None,
                (_Array((1,), "f32", "x"),),
                outputs=(BufferSpec("same", (1,), "f32"),),
                workspaces=(BufferSpec("same", (1,), "u8"),),
            )

        with self.assertRaisesRegex(ValueError, "only public outputs"):
            self._call(
                lambda stream, x, y, z: None,
                (_Array((1,), "f32", "x"),),
                outputs=(BufferSpec("output", (1,), "f32"),),
                workspaces=(BufferSpec("workspace", (1,), "u8"),),
                input_output_aliases={0: 1},
            )

        with self.assertRaisesRegex(ValueError, "Expected 1 input tensor spec"):
            self._call(
                lambda stream, x, y: None,
                (_Array((1,), "f32", "x"),),
                outputs=(BufferSpec("output", (1,), "f32"),),
                input_specs=(),
            )

        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self._call(
                lambda stream, x, y: None,
                (_Array((2,), "f32", "x"),),
                outputs=(BufferSpec("output", (1,), "f32"),),
                input_output_aliases={0: 0},
            )

        with self.assertRaisesRegex(ValueError, "identical TensorSpec"):
            self._call(
                lambda stream, x, y: None,
                (_Array((2, 2), "f32", "x"),),
                outputs=(BufferSpec("output", (2, 2), "f32"),),
                input_specs=(_FakeCutlassTensorSpec(mode=(1, 0)),),
                input_output_aliases={0: 0},
            )

        with self.assertRaisesRegex(TypeError, "flat sequence"):
            self._call(
                lambda stream, x, y: None,
                ((_Array((1,), "f32", "nested"),),),
                outputs=(BufferSpec("output", (1,), "f32"),),
            )

        with self.assertRaisesRegex(ValueError, "reserved by cutlass_call"):
            self._call(
                lambda stream, x, y: None,
                (_Array((1,), "f32", "x"),),
                outputs=(BufferSpec("output", (1,), "f32"),),
                static_args={"compile_options": "not a kernel argument"},
            )

        with self.assertRaisesRegex(ValueError, "permutation"):
            self._call(
                lambda stream, x, y: None,
                (_Array((1, 1), "f32", "x"),),
                outputs=(BufferSpec("output", (1, 1), "f32"),),
                input_specs=(_FakeCutlassTensorSpec(layout=(0, 0)),),
            )


if __name__ == "__main__":
    unittest.main()
