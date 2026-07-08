# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contracts for JAX DSA score-recompute adapters."""

from enum import Enum, auto
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


_CUDNN_ROOT = Path(__file__).resolve().parents[3] / "python" / "cudnn"
_DSA_ROOT = _CUDNN_ROOT / "deepseek_sparse_attention"
_SCORE_ROOT = _DSA_ROOT / "score_recompute"
_PACKAGE = "cudnn_jax_dsa_score_contract_test"


class _DataType(Enum):
    NOT_SET = auto()
    FLOAT = auto()
    BFLOAT16 = auto()
    INT32 = auto()


class _Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)


class _TupleDict(dict):
    pass


class _TensorSpec:
    def __init__(self, *, mode=None, divisibility=None):
        self.mode = mode
        self.divisibility = divisibility


def _identity_jit(fn=None, **_kwargs):
    return (lambda decorated: decorated) if fn is None else fn


class JaxDsaScoreRecomputeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = types.ModuleType(_PACKAGE)
        root.__path__ = [str(_CUDNN_ROOT)]
        root.__package__ = _PACKAGE
        root.__spec__ = ModuleSpec(_PACKAGE, loader=None, is_package=True)
        root.data_type = _DataType
        sys.modules[_PACKAGE] = root

        cls.tensor = importlib.import_module(f"{_PACKAGE}.common.tensor_desc")

        fake_jnp = types.ModuleType("jax.numpy")
        fake_jnp.bfloat16 = _DataType.BFLOAT16
        fake_jnp.float32 = _DataType.FLOAT
        fake_jnp.int32 = _DataType.INT32
        fake_jnp.expand_dims = lambda value, axis: _Array(
            value.shape[:axis] + (1,) + value.shape[axis:], value.dtype
        )
        fake_jnp.transpose = lambda value, axes: _Array(
            tuple(value.shape[axis] for axis in axes), value.dtype
        )

        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        fake_jax.numpy = fake_jnp
        fake_jax.jit = _identity_jit
        fake_jax.ShapeDtypeStruct = _Array

        tensor_module = cls.tensor

        class JaxTensorDesc(tensor_module.TensorDesc):
            @classmethod
            def from_shape(
                cls,
                shape,
                dtype,
                *,
                name="",
                mode=None,
                public_stride_order=None,
                init_value=None,
            ):
                return FakeJaxApiBase._to_tensor_desc(
                    _Array(shape, dtype),
                    name,
                    mode=mode,
                    init_value=init_value,
                )

            @property
            def cudnn_dtype(self):
                return self.dtype

        class FakeJaxApiBase:
            @staticmethod
            def _to_tensor_desc(value, name, *, mode=None, init_value=None):
                public_shape = tuple(value.shape)
                mode = tuple(range(len(public_shape))) if mode is None else tuple(mode)
                public_stride = [0] * len(public_shape)
                running = 1
                for dimension in reversed(range(len(public_shape))):
                    public_stride[dimension] = running
                    running *= max(public_shape[dimension], 1)
                canonical_axis_by_public = [0] * len(mode)
                for canonical_axis, public_axis in enumerate(mode):
                    canonical_axis_by_public[public_axis] = canonical_axis
                desc = JaxTensorDesc(
                    dtype=value.dtype,
                    shape=tuple(public_shape[axis] for axis in mode),
                    stride=tuple(public_stride[axis] for axis in mode),
                    stride_order=tuple(
                        canonical_axis_by_public[axis]
                        for axis in reversed(range(len(public_shape)))
                    ),
                    name=name,
                    init_value=init_value,
                )
                object.__setattr__(desc, "mode", mode)
                return desc

            @staticmethod
            def _resolve_compute_capability(target, supported, operation_name):
                del operation_name
                resolved = 100 if target is None else target
                if resolved not in supported:
                    raise ValueError(f"unsupported target {resolved}")
                return resolved

            @staticmethod
            def _check_tensor_signature(value, expected, *, mode=None):
                mode = expected.mode if mode is None else tuple(mode)
                if tuple(value.shape[axis] for axis in mode) != expected.shape:
                    raise ValueError(f"{expected.name} tensor shape mismatch")
                if value.dtype != expected.cudnn_dtype:
                    raise ValueError(f"{expected.name} tensor dtype mismatch")

            @staticmethod
            def _to_tensor_spec(desc, *, mode=None, divisibility=None):
                if mode is None:
                    mode = desc.mode
                return _TensorSpec(mode=mode, divisibility=divisibility)

            def _call_kernel(
                self, inputs, *, launch, output_descs, workspace_descs=(), **options
            ):
                input_descs = options.get("input_descs")
                if input_descs is not None:
                    for value, desc in zip(inputs, input_descs):
                        self._check_tensor_signature(value, desc)
                    options["input_spec"] = tuple(
                        self._to_tensor_spec(desc) for desc in input_descs
                    )
                options["output_spec"] = tuple(
                    self._to_tensor_spec(desc) for desc in output_descs
                )
                self.captured_call = {
                    "inputs": tuple(inputs),
                    "outputs": tuple(output_descs),
                    "workspaces": tuple(workspace_descs),
                    "launch": launch,
                    **options,
                }
                specs = options.get("output_spec") or (None,) * len(output_descs)
                outputs = []
                for desc, spec in zip(output_descs, specs):
                    mode = (
                        desc.mode
                        if spec is None or spec.mode is None
                        else tuple(spec.mode)
                    )
                    canonical_axis_by_public = [0] * len(mode)
                    for canonical_axis, public_axis in enumerate(mode):
                        canonical_axis_by_public[public_axis] = canonical_axis
                    outputs.append(
                        _Array(
                            tuple(
                                desc.shape[canonical_axis_by_public[public_axis]]
                                for public_axis in range(desc.ndim)
                            ),
                            desc.cudnn_dtype,
                        )
                    )
                return tuple(outputs)

        fake_internal_jax = types.ModuleType(f"{_PACKAGE}._jax")
        fake_internal_jax.__path__ = [str(_CUDNN_ROOT / "_jax")]
        fake_internal_jax.__package__ = fake_internal_jax.__name__
        fake_internal_jax.__spec__ = ModuleSpec(
            fake_internal_jax.__name__, loader=None, is_package=True
        )
        fake_internal_jax.JaxApiBase = FakeJaxApiBase
        fake_internal_jax.JaxTensorDesc = JaxTensorDesc
        fake_internal_jax.TupleDict = _TupleDict

        dsa_name = f"{_PACKAGE}.deepseek_sparse_attention"
        dsa = types.ModuleType(dsa_name)
        dsa.__path__ = [str(_DSA_ROOT)]
        dsa.__package__ = dsa_name
        dsa.__spec__ = ModuleSpec(dsa_name, loader=None, is_package=True)

        score_name = f"{dsa_name}.score_recompute"
        score = types.ModuleType(score_name)
        score.__path__ = [str(_SCORE_ROOT)]
        score.__package__ = score_name
        score.__spec__ = ModuleSpec(score_name, loader=None, is_package=True)

        utils_name = f"{dsa_name}.utils"
        utils = types.ModuleType(utils_name)
        utils.__path__ = []
        cls.optional_modules = {
            "jax": fake_jax,
            "jax.numpy": fake_jnp,
            f"{_PACKAGE}._jax": fake_internal_jax,
            dsa_name: dsa,
            score_name: score,
            utils_name: utils,
        }
        sys.modules.update(cls.optional_modules)
        try:
            cls.adapter = importlib.import_module(f"{score_name}.jax")
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
                sys.modules.pop(name, None)
        for name in ("jax.numpy", "jax"):
            module = sys.modules.get(name)
            if module is cls.optional_modules.get(name):
                sys.modules.pop(name, None)

    def array(self, shape, dtype=_DataType.BFLOAT16):
        return _Array(shape, dtype)

    def sparse_samples(self):
        return (
            self.array((2, 4, 32, 128)),
            self.array((2, 256, 128)),
            self.array((2, 4, 32)),
            self.array((2, 4, 128), _DataType.INT32),
        )

    def sequence_major_sparse_samples(self):
        return (
            self.array((4, 2, 32, 128)),
            self.array((256, 2, 128)),
            self.array((4, 2, 32)),
            self.array((4, 2, 128), _DataType.INT32),
        )

    def dense_samples(self, *, attention=False):
        return (
            self.array((2, 4, 32, 128)),
            self.array((2, 8, 1, 128)),
            self.array(
                (2, 4, 32), _DataType.FLOAT if attention else _DataType.BFLOAT16
            ),
        )

    def sequence_major_dense_samples(self):
        return (
            self.array((4, 2, 32, 128)),
            self.array((8, 2, 1, 128)),
            self.array((4, 2, 32)),
        )

    def test_sm100_sparse_infers_output_and_hidden_length_workspace(self):
        q, k, weights, indices = self.sparse_samples()
        api = self.adapter.SparseIndexerScoreRecompute(
            q, k, weights, indices, target_compute_capability=100
        )
        result = api(q, k, weights, indices)

        self.assertIsInstance(result, _TupleDict)
        self.assertEqual(
            (result["predict"].shape, result["predict"].dtype),
            ((2, 4, 128), _DataType.FLOAT),
        )
        self.assertEqual(len(api.captured_call["workspaces"]), 1)
        workspace = api.captured_call["workspaces"][0]
        self.assertEqual(
            (workspace.shape, workspace.cudnn_dtype), ((1, 1), _DataType.INT32)
        )
        self.assertEqual(api.captured_call["compile_options"], "--gpu-arch sm_100a")

    def test_explicit_topk_length_remains_a_real_input(self):
        q, k, _, indices = self.sparse_samples()
        lse = self.array((2, 4, 32), _DataType.FLOAT)
        lengths = self.array((2, 4), _DataType.INT32)
        api = self.adapter.SparseAttnScoreRecompute(
            q,
            k,
            lse,
            indices,
            0.125,
            sample_topk_length=lengths,
            target_compute_capability=100,
        )
        result = api(q, k, lse, indices, lengths)

        self.assertEqual(result["target"].shape, (2, 4, 128))
        self.assertEqual(api.captured_call["workspaces"], ())
        self.assertIs(api.captured_call["inputs"][-1], lengths)
        self.assertTrue(api._op.config.have_topk_length)

    def test_sm90_sparse_adapts_k_and_per_head_layouts(self):
        q, k, weights, indices = self.sparse_samples()
        api = self.adapter.SparseIndexerScoreRecompute(
            q, k, weights, indices, target_compute_capability=90
        )
        api(q, k, weights, indices)

        _, kernel_k, kernel_weights, _ = api.captured_call["inputs"]
        self.assertEqual(kernel_k.shape, (2, 256, 1, 128))
        self.assertEqual(kernel_weights.shape, (2, 32, 4))
        self.assertEqual(api.captured_call["workspaces"], ())

    def test_sparse_sequence_major_layouts_preserve_public_shapes(self):
        q, k, weights, indices = self.sequence_major_sparse_samples()
        api = self.adapter.SparseIndexerScoreRecompute(
            q,
            k,
            weights,
            indices,
            q_layout="SBHD",
            k_layout="SBD",
            per_head_layout="SBH",
            score_layout="SBK",
            output_layout="SBK",
            target_compute_capability=90,
        )
        result = api(q, k, weights, indices)

        self.assertEqual(result["predict"].shape, indices.shape)
        _, kernel_k, kernel_weights, _ = api.captured_call["inputs"]
        self.assertEqual(kernel_k.shape, (256, 2, 1, 128))
        self.assertEqual(kernel_weights.shape, (2, 32, 4))
        self.assertEqual(
            tuple(spec.mode for spec in api.captured_call["input_spec"]),
            ((1, 0, 2, 3), (1, 0, 2, 3), (0, 1, 2), (1, 0, 2)),
        )
        self.assertEqual(
            api.captured_call["output_spec"][0].mode,
            (1, 0, 2),
        )

    def test_dense_bshd_outputs_are_functional_and_initialized(self):
        q, k, weights = self.dense_samples()
        offsets = self.array((2,), _DataType.INT32)
        api = self.adapter.DenseIndexerScoreRecompute(
            q,
            k,
            weights,
            ratio=2,
            sample_q_causal_offsets=offsets,
            target_compute_capability=100,
        )
        result = api(q, k, weights, q_causal_offsets=offsets)

        self.assertEqual(result["out"].shape, (2, 4, 8))
        self.assertEqual(result["denom"].shape, (2, 4))
        self.assertEqual(api.out_desc.init_value, float("-inf"))
        self.assertIsNone(api.denom_desc.init_value)
        self.assertIs(api.captured_call["inputs"][-1], offsets)

        explicit = self.adapter.DenseIndexerScoreRecompute(
            q,
            k,
            weights,
            sample_out=self.array((2, 4, 8), _DataType.FLOAT),
            sample_denom_out=self.array((2, 4), _DataType.FLOAT),
            target_compute_capability=100,
        )
        self.assertEqual(explicit.out_desc.init_value, float("-inf"))
        self.assertIsNone(explicit.denom_desc.init_value)

    def test_dense_sequence_major_layouts_preserve_public_shapes(self):
        q, k, weights = self.sequence_major_dense_samples()
        api = self.adapter.DenseIndexerScoreRecompute(
            q,
            k,
            weights,
            q_layout="SBHD",
            k_layout="SBHD",
            per_head_layout="SBH",
            output_layout="SBK",
            denom_layout="SB",
            target_compute_capability=100,
        )
        result = api(q, k, weights)

        self.assertEqual(result["out"].shape, (4, 2, 8))
        self.assertEqual(result["denom"].shape, (4, 2))
        self.assertEqual(
            tuple(spec.mode for spec in api.captured_call["input_spec"]),
            ((1, 0, 2, 3), (1, 0, 2, 3), (1, 0, 2)),
        )
        self.assertEqual(
            tuple(spec.mode for spec in api.captured_call["output_spec"]),
            ((1, 0, 2), (1, 0)),
        )

    def test_dense_thd_sm100_and_sm90_rejection(self):
        q = self.array((7, 32, 128))
        k = self.array((11, 1, 128))
        lse = self.array((7, 32), _DataType.FLOAT)
        cu_q = self.array((3,), _DataType.INT32)
        cu_k = self.array((3,), _DataType.INT32)
        api = self.adapter.DenseAttnScoreRecompute(
            q,
            k,
            lse,
            0.125,
            sample_cu_seqlens_q=cu_q,
            sample_cu_seqlens_k=cu_k,
            max_seqlen_q=4,
            max_seqlen_k=8,
            target_compute_capability=100,
        )
        result = api(q, k, lse, cu_q, cu_k)
        self.assertEqual((result["out"].shape, result["denom"].shape), ((7, 8), (7,)))
        self.assertEqual(api.captured_call["inputs"][-2:], (cu_q, cu_k))

        sm90 = self.adapter.DenseAttnScoreRecompute(
            q,
            k,
            lse,
            0.125,
            sample_cu_seqlens_q=cu_q,
            sample_cu_seqlens_k=cu_k,
            max_seqlen_q=4,
            max_seqlen_k=8,
            target_compute_capability=90,
        )
        with self.assertRaisesRegex(NotImplementedError, "cannot be traced by JAX"):
            sm90(q, k, lse, cu_q, cu_k)

    def test_explicit_output_samples_are_cross_checked(self):
        q, k, weights = self.dense_samples()
        bad_output = self.array((2, 4, 4), _DataType.FLOAT)
        denominator = self.array((2, 4), _DataType.FLOAT)
        api = self.adapter.DenseIndexerScoreRecompute(
            q,
            k,
            weights,
            sample_out=bad_output,
            sample_denom_out=denominator,
            target_compute_capability=100,
        )
        with self.assertRaisesRegex(ValueError, "output must have shape"):
            api(q, k, weights)

    def test_high_level_wrapper_infers_shapes(self):
        q, k, weights, indices = self.sparse_samples()
        with mock.patch.dict(sys.modules, self.optional_modules):
            result = self.adapter.sparse_indexer_score_recompute_wrapper(
                q,
                k,
                weights,
                indices,
                target_compute_capability=100,
            )
        self.assertEqual(result["predict"].shape, indices.shape)

    def test_sparse_launchers_preserve_native_argument_order(self):
        calls = []

        class FakeSm90Kernel:
            def __init__(self, dtype, **options):
                calls.append(("sm90_init", dtype, options))

            def __call__(self, *args):
                calls.append(("sm90_call", args))

        class FakeSm100Kernel:
            def __init__(self, **options):
                calls.append(("sm100_init", options))

            def __call__(self, *args):
                calls.append(("sm100_call", args))

        cutlass = types.ModuleType("cutlass")
        cutlass.BFloat16 = "BFloat16"
        cutlass.Float32 = lambda value: ("Float32", value)
        package = self.adapter.__package__
        sm90_module = types.ModuleType(f"{package}.sparse_score_recompute_sm90")
        sm90_module.SparseScoreRecomputeSm90 = FakeSm90Kernel
        sm100_module = types.ModuleType(f"{package}.sparse_score_recompute_sm100")
        sm100_module.SparseScoreRecomputeSm100 = FakeSm100Kernel

        q, k, weights, indices = self.sparse_samples()
        sm90 = self.adapter.SparseIndexerScoreRecompute(
            q, k, weights, indices, target_compute_capability=90
        )
        sm90.check_support()
        sm100 = self.adapter.SparseIndexerScoreRecompute(
            q, k, weights, indices, target_compute_capability=100
        )
        sm100.check_support()
        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                sm90_module.__name__: sm90_module,
                sm100_module.__name__: sm100_module,
            },
        ):
            sm90._launch_kernel("stream90", "q90", "k90", "aux90", "indices90", "out90")
            sm100._launch_kernel(
                "stream100", "q100", "k100", "aux100", "indices100", "out100", "dummy"
            )

        self.assertEqual(
            [entry for entry in calls if entry[0].endswith("call")],
            [
                (
                    "sm90_call",
                    (
                        "q90",
                        "k90",
                        "indices90",
                        "stream90",
                        "out90",
                        "aux90",
                        None,
                        None,
                    ),
                ),
                (
                    "sm100_call",
                    (
                        "q100",
                        "k100",
                        "aux100",
                        "indices100",
                        "out100",
                        "dummy",
                        ("Float32", 1.0),
                        "stream100",
                    ),
                ),
            ],
        )

    def test_dense_launchers_preserve_native_argument_order(self):
        calls = []

        class FakeSm90Kernel:
            def __init__(self, dtype, **options):
                calls.append(("sm90_init", dtype, options))

            def __call__(self, *args):
                calls.append(("sm90_call", args))

        class FakeSm100Kernel:
            def __init__(self, **options):
                calls.append(("sm100_init", options))

            def __call__(self, *args):
                calls.append(("sm100_call", args))

        cutlass = types.ModuleType("cutlass")
        cutlass.BFloat16 = "BFloat16"
        cutlass.Float32 = lambda value: ("Float32", value)
        cutlass.Int32 = lambda value: ("Int32", value)
        package = self.adapter.__package__
        sm90_module = types.ModuleType(f"{package}.dense_score_recompute_sm90")
        sm90_module.DenseScoreRecomputeSm90 = FakeSm90Kernel
        sm100_module = types.ModuleType(f"{package}.dense_score_recompute_sm100")
        sm100_module.DenseScoreRecomputeSm100 = FakeSm100Kernel

        q, k, weights = self.dense_samples()
        offsets = self.array((2,), _DataType.INT32)
        sm90 = self.adapter.DenseIndexerScoreRecompute(
            q,
            k,
            weights,
            sample_q_causal_offsets=offsets,
            target_compute_capability=90,
        )
        sm90.check_support()
        sm100 = self.adapter.DenseIndexerScoreRecompute(
            q,
            k,
            weights,
            sample_q_causal_offsets=offsets,
            target_compute_capability=100,
        )
        sm100.check_support()
        with mock.patch.dict(
            sys.modules,
            {
                "cutlass": cutlass,
                sm90_module.__name__: sm90_module,
                sm100_module.__name__: sm100_module,
            },
        ):
            sm90._launch_kernel(
                "stream90", "q90", "k90", "aux90", "offset90", "out90", "denom90"
            )
            sm100._launch_kernel(
                "stream100", "q100", "k100", "aux100", "offset100", "out100", "denom100"
            )

        self.assertEqual(
            [entry for entry in calls if entry[0].endswith("call")],
            [
                (
                    "sm90_call",
                    (
                        "q90",
                        "k90",
                        None,
                        "stream90",
                        "out90",
                        "aux90",
                        None,
                        "denom90",
                        "offset90",
                    ),
                ),
                (
                    "sm100_call",
                    (
                        "q100",
                        "k100",
                        "aux100",
                        "out100",
                        "denom100",
                        ("Float32", 1.0),
                        ("Int32", 4),
                        ("Int32", 8),
                        None,
                        None,
                        "offset100",
                        "stream100",
                    ),
                ),
            ],
        )

    def test_common_and_jax_modules_keep_framework_import_boundaries(self):
        def imports_framework(path, framework):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(
                    alias.name == framework or alias.name.startswith(f"{framework}.")
                    for alias in node.names
                ):
                    return True
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and (
                        node.module == framework
                        or (node.module or "").startswith(f"{framework}.")
                    )
                ):
                    return True
            return False

        for filename in ("config.py", "op.py"):
            path = _SCORE_ROOT / filename
            for framework in ("torch", "jax", "cutlass", "cuda"):
                self.assertFalse(
                    imports_framework(path, framework),
                    f"{filename} imports {framework}",
                )
        self.assertFalse(imports_framework(_SCORE_ROOT / "jax.py", "torch"))

        init_tree = ast.parse((_SCORE_ROOT / "__init__.py").read_text())
        imported_modules = {
            node.module
            for node in ast.walk(init_tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("api", imported_modules)


if __name__ == "__main__":
    unittest.main()
