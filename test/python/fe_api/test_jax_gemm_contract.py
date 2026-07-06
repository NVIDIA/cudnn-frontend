# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the JAX dense GEMM wrappers."""

from __future__ import annotations

import ast
import importlib
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
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
_TEST_PACKAGE = "cudnn_frontend_jax_gemm_contract_test"
_COMMON_KERNEL_CAPABILITIES = {
    "MMA_TILER_M",
    "MMA_TILER_N",
    "TWO_CTA_MMA_TILER_M",
    "MAX_CLUSTER_CTAS",
    "MAX_CLUSTER_DIMENSION",
}
_COMMON_KERNEL_METHODS = {"require_mma_tiler", "require_cluster_shape"}


def _kernel_capabilities(
    path: Path,
    class_name: str,
    field_names: set[str],
    method_names: set[str],
) -> type:
    tree = ast.parse(path.read_text(), filename=str(path))
    kernel_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    field_nodes = []
    found_fields = set()
    method_nodes = []
    found_methods = set()
    for node in kernel_class.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            target = None
        if isinstance(target, ast.Name) and target.id in field_names:
            field_nodes.append(node)
            found_fields.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in method_names:
            method_nodes.append(node)
            found_methods.add(node.name)

    missing = field_names - found_fields
    if missing:
        raise AssertionError(f"{class_name} is missing public kernel capabilities: {sorted(missing)}")
    missing = method_names - found_methods
    if missing:
        raise AssertionError(f"{class_name} is missing public kernel methods: {sorted(missing)}")

    def validation_helper(name):
        def call(*args, **kwargs):
            validation = importlib.import_module(f"{_TEST_PACKAGE}.gemm_validation")
            return getattr(validation, name)(*args, **kwargs)

        return call

    surface_class = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        body=[*field_nodes, *method_nodes],
        decorator_list=[],
    )
    surface_module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                surface_class,
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "__name__": __name__,
        "_require_mma_tiler": validation_helper("require_mma_tiler"),
        "_require_cluster_shape": validation_helper("require_cluster_shape"),
    }
    exec(compile(surface_module, str(path), "exec"), namespace)
    return namespace[class_name]


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


class JaxGemmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float16 = _DType("float16", 2)
        cls.float32 = _DType("float32", 4)
        cls.float8_e4m3fn = _DType("float8_e4m3fn", 1)
        cls.float8_e5m2 = _DType("float8_e5m2", 1)
        cls.float8_e8m0fnu = _DType("float8_e8m0fnu", 1)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        for name in (
            "bfloat16",
            "float16",
            "float32",
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e8m0fnu",
        ):
            setattr(cls.fake_jnp, name, getattr(cls, name))
        cls.fake_jnp.dtype = lambda value: value

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp
        cls.fake_jax.tree_util = types.SimpleNamespace(
            DictKey=lambda key: key,
            register_pytree_with_keys=lambda *_args: None,
        )

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass.Constexpr = object
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.is_available = lambda: True
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: f"cutlass.{dtype.name}"
        cls.fake_cutlass.jax = cls.fake_cutlass_jax
        cls.fake_cutlass_cute = types.ModuleType("cutlass.cute")
        cls.fake_cutlass_cute.jit = _identity_jit
        cls.fake_cutlass_cute.where = lambda condition, yes, no: (yes if condition else no)
        cls.fake_cutlass_cute.full_like = lambda value, fill: fill
        cls.fake_cutlass.cute = cls.fake_cutlass_cute

        cls.swiglu_capabilities = _kernel_capabilities(
            _CUDNN_ROOT / "gemm_swiglu" / "dense_gemm_persistent_swiglu.py",
            "PersistentDenseGemmKernel",
            _COMMON_KERNEL_CAPABILITIES
            | {
                "SINGLE_CTA_CLUSTER_SHAPE",
                "SWIGLU_BLOCK_COLUMNS",
                "SWIGLU_BLOCKS_PER_PAIR",
            },
            _COMMON_KERNEL_METHODS | {"get_output_n"},
        )
        cls.quantized_swiglu_capabilities = _kernel_capabilities(
            _CUDNN_ROOT / "gemm_swiglu" / "dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.py",
            "Sm100BlockScaledPersistentDenseGemmKernel",
            _COMMON_KERNEL_CAPABILITIES | {"SWIGLU_BLOCK_COLUMNS", "SWIGLU_BLOCKS_PER_PAIR"},
            _COMMON_KERNEL_METHODS | {"get_output_n"},
        )
        cls.srelu_capabilities = _kernel_capabilities(
            _CUDNN_ROOT / "gemm_srelu" / "dense_blockscaled_gemm_persistent_srelu_quant.py",
            "Sm100BlockScaledPersistentDenseGemmKernel",
            _COMMON_KERNEL_CAPABILITIES,
            _COMMON_KERNEL_METHODS,
        )
        cls.dsrelu_capabilities = _kernel_capabilities(
            _CUDNN_ROOT / "gemm_dsrelu" / "dense_blockscaled_gemm_persistent_dsrelu_quant.py",
            "Sm100BlockScaledPersistentDenseGemmKernel",
            _COMMON_KERNEL_CAPABILITIES,
            _COMMON_KERNEL_METHODS,
        )
        cls.amax_capabilities = _kernel_capabilities(
            _CUDNN_ROOT / "gemm_amax" / "dense_blockscaled_gemm_persistent_amax.py",
            "Sm100BlockScaledPersistentDenseGemmKernel",
            _COMMON_KERNEL_CAPABILITIES | {"KNOWN_HANG_MMA_TILER_M", "SF_VEC_SIZES"},
            _COMMON_KERNEL_METHODS,
        )
        cls.fake_kernel_modules = {}
        for module_suffix, class_name, capabilities in (
            (
                "gemm_swiglu.dense_gemm_persistent_swiglu",
                "PersistentDenseGemmKernel",
                cls.swiglu_capabilities,
            ),
            (
                "gemm_amax.dense_blockscaled_gemm_persistent_amax",
                "Sm100BlockScaledPersistentDenseGemmKernel",
                cls.amax_capabilities,
            ),
            (
                "gemm_srelu.dense_blockscaled_gemm_persistent_srelu_quant",
                "Sm100BlockScaledPersistentDenseGemmKernel",
                cls.srelu_capabilities,
            ),
            (
                "gemm_dsrelu.dense_blockscaled_gemm_persistent_dsrelu_quant",
                "Sm100BlockScaledPersistentDenseGemmKernel",
                cls.dsrelu_capabilities,
            ),
        ):
            module_name = f"{_TEST_PACKAGE}.{module_suffix}"
            module = types.ModuleType(module_name)
            setattr(module, class_name, capabilities)
            cls.fake_kernel_modules[module_name] = module

        parent = types.ModuleType(_TEST_PACKAGE)
        parent.__path__ = [str(_CUDNN_ROOT)]
        parent.__package__ = _TEST_PACKAGE
        sys.modules[_TEST_PACKAGE] = parent
        with cls._optional_modules():
            importlib.import_module(f"{_TEST_PACKAGE}.jax")
            cls.gemm = importlib.import_module(f"{_TEST_PACKAGE}._jax.gemm")
            cls.swiglu = importlib.import_module(f"{_TEST_PACKAGE}.gemm_swiglu.jax")
            cls.amax = importlib.import_module(f"{_TEST_PACKAGE}.gemm_amax.jax")
            cls.srelu = importlib.import_module(f"{_TEST_PACKAGE}.gemm_srelu.jax")
            cls.dsrelu = importlib.import_module(f"{_TEST_PACKAGE}.gemm_dsrelu.jax")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    @classmethod
    def _optional_modules(cls):
        modules = {
            "jax": cls.fake_jax,
            "jax.numpy": cls.fake_jnp,
            "cutlass": cls.fake_cutlass,
            "cutlass.cute": cls.fake_cutlass_cute,
            "cutlass.jax": cls.fake_cutlass_jax,
        }
        modules.update(cls.fake_kernel_modules)
        return mock.patch.dict(
            sys.modules,
            modules,
        )

    @staticmethod
    def _fake_call(captured):
        def call(launcher, inputs, **options):
            captured.update(launcher=launcher, inputs=inputs, **options)
            return tuple(_Array(spec.shape, spec.dtype) for spec in options["outputs"])

        return call

    def test_swiglu_declares_logical_outputs_and_physical_layouts(self):
        captured = {}
        a = _Array((2, 128, 256), self.bfloat16)
        b = _Array((2, 128, 128), self.bfloat16)
        with (
            self._optional_modules(),
            mock.patch.object(
                self.swiglu,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.swiglu.gemm_swiglu_wrapper_sm100(
                a,
                b,
                c_layout="LNM",
                ab12_dtype=_Array((), self.float32),
                c_dtype=_Array((), self.bfloat16),
                acc_dtype=_Array((), self.float32),
                a_layout="LKM",
                b_layout="LKN",
            )

        self.assertIs(captured["launcher"], self.swiglu._launch)
        self.assertEqual(captured["inputs"], (a, b))
        self.assertEqual(
            captured["static_args"],
            {
                "alpha": 1.0,
                "acc_dtype": self.float32,
                "mma_tiler_mn": (128, 128),
                "cluster_shape_mn": (1, 1),
                "cluster_overlap_margin": 0,
            },
        )
        self.assertNotIn("workspaces", captured)
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype, spec.fill_value) for spec in captured["outputs"]],
            [
                ("ab12_tensor", (2, 128, 256), self.float32, None),
                ("c_tensor", (2, 64, 256), self.bfloat16, None),
            ],
        )
        self.assertEqual([spec.layout for spec in captured["input_specs"]], [(2, 1, 0), (2, 1, 0)])
        self.assertEqual([spec.mode for spec in captured["input_specs"]], [(2, 1, 0), (2, 1, 0)])
        self.assertEqual(
            [spec.tensor_spec.layout for spec in captured["outputs"]],
            [(2, 1, 0), (2, 1, 0)],
        )
        self.assertEqual([spec.tensor_spec.mode for spec in captured["outputs"]], [(2, 1, 0), (2, 1, 0)])
        self.assertEqual(tuple(result.keys()), ("ab12_tensor", "c_tensor", "sfc_tensor", "amax_tensor"))
        self.assertIsNone(result["sfc_tensor"])
        self.assertIsNone(result["amax_tensor"])

    def test_layout_strings_map_public_shapes_to_canonical_kernel_metadata(self):
        cases = (
            (self.swiglu.gemm_a_tensor_spec, "lmk", (2, 3, 5), (3, 5, 2), (5, 1, 15), (1, 2, 0)),
            (self.swiglu.gemm_a_tensor_spec, "LKM", (2, 5, 3), (3, 5, 2), (1, 3, 15), (2, 1, 0)),
            (self.swiglu.gemm_b_tensor_spec, "LNK", (2, 7, 5), (7, 5, 2), (5, 1, 35), (1, 2, 0)),
            (self.swiglu.gemm_b_tensor_spec, "LKN", (2, 5, 7), (7, 5, 2), (1, 7, 35), (2, 1, 0)),
            (self.swiglu.gemm_c_tensor_spec, "LMN", (2, 3, 7), (3, 7, 2), (7, 1, 21), (1, 2, 0)),
            (self.swiglu.gemm_c_tensor_spec, "LNM", (2, 7, 3), (3, 7, 2), (1, 3, 21), (2, 1, 0)),
        )

        with self._optional_modules():
            for factory, layout, array_shape, kernel_shape, stride, mode in cases:
                with self.subTest(layout=layout):
                    spec = factory(layout)
                    desc = self.swiglu.JaxTensorDesc.from_value(
                        _Array(array_shape, self.bfloat16),
                        tensor_spec=spec,
                    )
                    self.assertEqual(spec.layout, (2, 1, 0))
                    self.assertEqual(spec.mode, mode)
                    self.assertEqual(desc.array_shape, array_shape)
                    self.assertEqual(desc.shape, kernel_shape)
                    self.assertEqual(desc.stride, stride)

            for factory, layout in (
                (self.swiglu.gemm_a_tensor_spec, "MKL"),
                (self.swiglu.gemm_b_tensor_spec, "NKL"),
                (self.swiglu.gemm_c_tensor_spec, "MNL"),
            ):
                with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, "must be one of"):
                    factory(layout)

            a_desc = self.swiglu.JaxTensorDesc.from_value(
                _Array((2, 3, 5), self.bfloat16),
                tensor_spec=self.gemm.gemm_a_tensor_spec("LMK"),
            )
            with self.assertRaisesRegex(ValueError, "descriptor layout does not match"):
                self.gemm.as_gemm_tensor_desc(
                    "a_tensor",
                    a_desc,
                    self.gemm.gemm_a_tensor_spec("LKM"),
                )

    def test_kernel_capability_fields_are_public(self):
        for capabilities in (
            self.swiglu_capabilities,
            self.quantized_swiglu_capabilities,
            self.amax_capabilities,
            self.srelu_capabilities,
            self.dsrelu_capabilities,
        ):
            with self.subTest(capabilities=capabilities):
                self.assertLessEqual(_COMMON_KERNEL_CAPABILITIES, vars(capabilities).keys())
                for method_name in _COMMON_KERNEL_METHODS:
                    self.assertTrue(callable(getattr(capabilities, method_name)))

    def test_swiglu_kernel_validates_output_n(self):
        for kernel in (self.swiglu_capabilities, self.quantized_swiglu_capabilities):
            with self.subTest(kernel=kernel):
                self.assertEqual(kernel.get_output_n(128), 64)
                with self.assertRaisesRegex(ValueError, "32-column SwiGLU block pairs"):
                    kernel.get_output_n(96)

    def test_kernel_surface_applies_kernel_specific_configuration_rules(self):
        with self.assertRaisesRegex(NotImplementedError, "currently hangs"):
            self.amax_capabilities.require_mma_tiler((256, 128))
        with self.assertRaisesRegex(ValueError, "single-CTA MMA tile"):
            self.swiglu_capabilities.require_cluster_shape((2, 2), mma_tiler_mn=(128, 128))

    def test_amax_declares_scale_layout_and_initialized_reduction(self):
        captured = {}
        a = _Array((1, 128, 128), self.float8_e4m3fn)
        b = _Array((1, 128, 128), self.float8_e4m3fn)
        sfa = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        sfb = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        with (
            self._optional_modules(),
            mock.patch.object(
                self.amax,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.amax.gemm_amax_wrapper_sm100(
                a,
                b,
                sfa,
                sfb,
                c_layout="LMN",
                c_dtype=self.float32,
            )

        self.assertEqual(captured["inputs"], (a, b, sfa, sfb))
        self.assertIs(captured["launcher"], self.amax._launch)
        self.assertEqual(
            captured["static_args"],
            {
                "sf_vec_size": 32,
                "mma_tiler_mn": (128, 128),
                "cluster_shape_mn": (1, 1),
                "cluster_overlap_margin": 0,
            },
        )
        self.assertEqual(
            [spec.layout for spec in captured["input_specs"]],
            [(2, 1, 0), (2, 1, 0), (2, 1, 4, 0, 3, 5), (2, 1, 4, 0, 3, 5)],
        )
        c_spec, amax_spec = captured["outputs"]
        self.assertEqual(
            (c_spec.name, c_spec.shape, c_spec.dtype),
            ("c_tensor", (1, 128, 128), self.float32),
        )
        self.assertEqual(c_spec.tensor_spec.layout, (2, 1, 0))
        self.assertEqual(c_spec.tensor_spec.mode, (1, 2, 0))
        self.assertEqual(
            (amax_spec.name, amax_spec.shape, amax_spec.dtype, amax_spec.fill_value),
            ("amax_tensor", (1, 1, 1), self.float32, float("-inf")),
        )
        self.assertEqual(tuple(result.keys()), ("c_tensor", "amax_tensor"))

    def test_swiglu_rejects_quantized_arguments(self):
        a = _Array((1, 128, 128), self.bfloat16)
        b = _Array((1, 128, 128), self.bfloat16)
        scale = _Array((1,), self.float8_e8m0fnu)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(NotImplementedError, "unquantized path"),
            mock.patch.object(self.swiglu, "call_cutedsl") as lower,
        ):
            self.swiglu.gemm_swiglu_wrapper_sm100(a, b, sfa_tensor=scale, sfb_tensor=scale)
        lower.assert_not_called()

    def test_swiglu_rejects_unused_nondefault_configuration(self):
        a = _Array((1, 128, 128), self.bfloat16)
        b = _Array((1, 128, 128), self.bfloat16)

        with self._optional_modules():
            for option, value, expected in (
                ("sf_vec_size", 32, "sf_vec_size=16"),
                ("vector_f32", True, "vector_f32=False"),
                ("ab12_stages", 3, "ab12_stages=4"),
            ):
                with (
                    self.subTest(option=option),
                    self.assertRaisesRegex(NotImplementedError, expected),
                    mock.patch.object(self.swiglu, "call_cutedsl") as lower,
                ):
                    self.swiglu.gemm_swiglu_wrapper_sm100(a, b, **{option: value})
                lower.assert_not_called()

    def test_swiglu_rejects_unsupported_epilogue_and_incomplete_2cta_tiles(self):
        a = _Array((1, 128, 128), self.bfloat16)
        b = _Array((1, 64, 128), self.bfloat16)

        with self._optional_modules():
            for mma_tiler_mn, message in (
                ((128, 32), r"N in \{64, 128, 192, 256\}"),
                ((256, 64), "M must be divisible by 256"),
            ):
                with (
                    self.subTest(mma_tiler_mn=mma_tiler_mn),
                    self.assertRaisesRegex(ValueError, message),
                    mock.patch.object(self.swiglu, "call_cutedsl") as lower,
                ):
                    self.swiglu.gemm_swiglu_wrapper_sm100(a, b, mma_tiler_mn=mma_tiler_mn)
                lower.assert_not_called()

    def test_amax_validates_scale_shapes_before_lowering(self):
        a = _Array((1, 128, 128), self.float8_e5m2)
        b = _Array((1, 128, 128), self.float8_e5m2)
        bad_sfa = _Array((32, 4, 1, 4, 2, 1), self.float8_e8m0fnu)
        sfb = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "sfa_tensor must have shape"),
            mock.patch.object(self.amax, "call_cutedsl") as lower,
        ):
            self.amax.gemm_amax_wrapper_sm100(a, b, bad_sfa, sfb)
        lower.assert_not_called()

    def test_srelu_and_dsrelu_declare_functional_outputs(self):
        a = _Array((2, 384, 512), self.float8_e4m3fn)
        b = _Array((2, 256, 512), self.float8_e4m3fn)
        c = _Array((2, 384, 256), self.bfloat16)
        sfa = _Array((32, 4, 3, 4, 4, 2), self.float8_e8m0fnu)
        sfb = _Array((32, 4, 2, 4, 4, 2), self.float8_e8m0fnu)
        prob = _Array((384, 1, 2), self.float32)
        srelu_captured = {}
        dsrelu_captured = {}

        with (
            self._optional_modules(),
            mock.patch.object(
                self.srelu,
                "call_cutedsl",
                side_effect=self._fake_call(srelu_captured),
            ),
            mock.patch.object(
                self.dsrelu,
                "call_cutedsl",
                side_effect=self._fake_call(dsrelu_captured),
            ),
        ):
            srelu_result = self.srelu.gemm_srelu_wrapper_sm100(
                a,
                b,
                sfa,
                sfb,
                prob,
            )
            dsrelu_result = self.dsrelu.gemm_dsrelu_wrapper_sm100(
                a,
                b,
                c,
                sfa,
                sfb,
                prob,
            )

        self.assertEqual(srelu_captured["inputs"], (a, b, sfa, sfb, prob))
        self.assertIs(srelu_captured["launcher"], self.srelu._launch)
        self.assertEqual(
            set(srelu_captured["static_args"]),
            {"alpha", "sf_vec_size", "mma_tiler_mn", "cluster_shape_mn", "vector_f32", "cluster_overlap_margin"},
        )
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype, spec.fill_value) for spec in srelu_captured["outputs"]],
            [
                ("c_tensor", (2, 384, 256), self.bfloat16, None),
                ("d_tensor", (2, 384, 256), self.bfloat16, None),
            ],
        )
        self.assertEqual(
            [spec.layout for spec in srelu_captured["input_specs"]],
            [
                (2, 1, 0),
                (2, 1, 0),
                (2, 1, 4, 0, 3, 5),
                (2, 1, 4, 0, 3, 5),
                (0, 1, 2),
            ],
        )
        self.assertIsNone(srelu_result["amax_tensor"])
        self.assertIsNone(srelu_result["sfd_tensor"])

        self.assertEqual(dsrelu_captured["inputs"], (a, b, c, sfa, sfb, prob))
        self.assertIs(dsrelu_captured["launcher"], self.dsrelu._launch)
        self.assertEqual(
            set(dsrelu_captured["static_args"]),
            {"alpha", "sf_vec_size", "mma_tiler_mn", "cluster_shape_mn", "vector_f32", "cluster_overlap_margin"},
        )
        d_spec, dprob_spec = dsrelu_captured["outputs"]
        self.assertEqual(
            (d_spec.name, d_spec.shape, d_spec.dtype),
            ("d_tensor", (2, 384, 256), self.bfloat16),
        )
        self.assertEqual(
            (
                dprob_spec.name,
                dprob_spec.shape,
                dprob_spec.dtype,
                dprob_spec.fill_value,
            ),
            ("dprob_tensor", (384, 1, 2), self.float32, 0.0),
        )
        self.assertEqual(dprob_spec.tensor_spec.layout, (0, 1, 2))
        self.assertEqual(dprob_spec.tensor_spec.mode, (0, 1, 2))
        self.assertIsNone(dsrelu_result["amax_tensor"])
        self.assertIsNone(dsrelu_result["sfd_tensor"])

    def test_relu_wrappers_reject_unpredicated_partial_mma_tile(self):
        with self._optional_modules():
            for m, mma_tiler_mn in ((129, (128, 128)), (130, (256, 256))):
                a = _Array((1, m, 512), self.float8_e4m3fn)
                b = _Array((1, 256, 512), self.float8_e4m3fn)
                c = _Array((1, m, 256), self.bfloat16)
                scale_m = (m + 127) // 128
                sfa = _Array((32, 4, scale_m, 4, 4, 1), self.float8_e8m0fnu)
                sfb = _Array((32, 4, 2, 4, 4, 1), self.float8_e8m0fnu)
                prob = _Array((m, 1, 1), self.float32)

                for module, call_args in (
                    (self.srelu, (a, b, sfa, sfb, prob)),
                    (self.dsrelu, (a, b, c, sfa, sfb, prob)),
                ):
                    wrapper_name = "gemm_srelu_wrapper_sm100" if module is self.srelu else "gemm_dsrelu_wrapper_sm100"
                    with (
                        self.subTest(module=module.__name__, mma_tiler_mn=mma_tiler_mn),
                        self.assertRaisesRegex(ValueError, f"TILE_M={mma_tiler_mn[0]}"),
                        mock.patch.object(module, "call_cutedsl") as lower,
                    ):
                        getattr(module, wrapper_name)(*call_args, mma_tiler_mn=mma_tiler_mn)
                    lower.assert_not_called()

    def test_dense_class_uses_sample_metadata_without_retaining_arrays(self):
        sample_a = _Array((1, 128, 128), self.float8_e4m3fn)
        sample_b = _Array((1, 128, 128), self.float8_e4m3fn)
        sample_sfa = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        sample_sfb = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        samples = (sample_a, sample_b, sample_sfa, sample_sfb)
        dtype_carrier = _Array((1,), self.float32)

        with self._optional_modules():
            operation = self.amax.GemmAmaxSm100(*samples, c_dtype=dtype_carrier)
            self.assertTrue(operation.check_support())

        self.assertIs(operation.get_jax_callable(), operation)
        self.assertEqual(operation.a_desc.shape, (128, 128, 1))
        self.assertEqual(operation.a_desc.array_shape, sample_a.shape)
        self.assertEqual(operation.sfa_desc.shape, sample_sfa.shape)
        self.assertFalse(any(name.endswith("_spec") for name in vars(operation)))
        self.assertTrue(all(stored is not sample for stored in vars(operation).values() for sample in samples))
        self.assertTrue(all(stored is not dtype_carrier for stored in vars(operation).values()))

        actual_a = _Array(sample_a.shape, sample_a.dtype)
        actual_b = _Array(sample_b.shape, sample_b.dtype)
        actual_sfa = _Array(sample_sfa.shape, sample_sfa.dtype)
        actual_sfb = _Array(sample_sfb.shape, sample_sfb.dtype)
        captured = {}
        with (
            self._optional_modules(),
            mock.patch.object(
                self.amax,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            operation(actual_a, actual_b, actual_sfa, actual_sfb)
        self.assertEqual(captured["inputs"], (actual_a, actual_b, actual_sfa, actual_sfb))
        self.assertEqual(
            captured["input_specs"],
            (
                operation.a_desc.tensor_spec,
                operation.b_desc.tensor_spec,
                operation.sfa_desc.tensor_spec,
                operation.sfb_desc.tensor_spec,
            ),
        )
        self.assertIs(captured["outputs"][0].tensor_spec, operation.c_desc.tensor_spec)

        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "A tensor shape mismatch"),
            mock.patch.object(self.amax, "call_cutedsl") as lower,
        ):
            operation(
                _Array((1, 64, 128), self.float8_e4m3fn),
                actual_b,
                actual_sfa,
                actual_sfb,
            )
        lower.assert_not_called()

    def test_launchers_preserve_native_argument_order(self):
        swiglu_calls = []
        amax_calls = []

        class FakeHardwareInfo:
            def get_max_active_clusters(self, cluster_size):
                self.cluster_size = cluster_size
                return 11

        class FakeSwigluKernel:
            TWO_CTA_MMA_TILER_M = 256

            def __init__(self, **options):
                self.options = options

            def __call__(self, *args):
                swiglu_calls.append(args)

        class FakeAmaxKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args):
                amax_calls.append(args)

        swiglu_module_name = f"{_TEST_PACKAGE}.gemm_swiglu.dense_gemm_persistent_swiglu"
        swiglu_module = types.ModuleType(swiglu_module_name)
        swiglu_module.PersistentDenseGemmKernel = FakeSwigluKernel
        amax_module_name = f"{_TEST_PACKAGE}.gemm_amax.dense_blockscaled_gemm_persistent_amax"
        amax_module = types.ModuleType(amax_module_name)
        amax_module.Sm100BlockScaledPersistentDenseGemmKernel = FakeAmaxKernel

        with (
            self._optional_modules(),
            mock.patch.dict(
                sys.modules,
                {
                    swiglu_module_name: swiglu_module,
                    amax_module_name: amax_module,
                },
            ),
            mock.patch.object(
                self.fake_cutlass,
                "utils",
                types.SimpleNamespace(HardwareInfo=FakeHardwareInfo),
                create=True,
            ),
            mock.patch.object(
                self.fake_cutlass,
                "Float32",
                lambda value: ("Float32", value),
                create=True,
            ),
        ):
            self.swiglu._launch(
                "stream",
                "a",
                "b",
                "ab12",
                "c",
                alpha=0.5,
                acc_dtype=self.float32,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                cluster_overlap_margin=1,
            )
            self.amax._launch(
                "stream",
                "a",
                "b",
                "sfa",
                "sfb",
                "c",
                "amax",
                sf_vec_size=32,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(2, 2),
                cluster_overlap_margin=2,
            )

        self.assertEqual(
            swiglu_calls,
            [("a", "b", "ab12", "c", ("Float32", 0.5), 10, "stream")],
        )
        self.assertEqual(
            amax_calls,
            [("a", "b", "sfa", "sfb", "c", "amax", 9, "stream")],
        )

    def test_relu_launchers_preserve_optional_native_slots(self):
        srelu_calls = []
        dsrelu_calls = []

        class FakeHardwareInfo:
            def get_max_active_clusters(self, cluster_size):
                return 8 + cluster_size

        class FakeSreluKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args, **kwargs):
                srelu_calls.append((args, kwargs))

        class FakeDsreluKernel:
            def __init__(self, **options):
                self.options = options

            def __call__(self, *args, **kwargs):
                dsrelu_calls.append((args, kwargs))

        srelu_module_name = f"{_TEST_PACKAGE}.gemm_srelu.dense_blockscaled_gemm_persistent_srelu_quant"
        srelu_module = types.ModuleType(srelu_module_name)
        srelu_module.Sm100BlockScaledPersistentDenseGemmKernel = FakeSreluKernel
        dsrelu_module_name = f"{_TEST_PACKAGE}.gemm_dsrelu.dense_blockscaled_gemm_persistent_dsrelu_quant"
        dsrelu_module = types.ModuleType(dsrelu_module_name)
        dsrelu_module.Sm100BlockScaledPersistentDenseGemmKernel = FakeDsreluKernel

        with (
            self._optional_modules(),
            mock.patch.dict(
                sys.modules,
                {
                    srelu_module_name: srelu_module,
                    dsrelu_module_name: dsrelu_module,
                },
            ),
            mock.patch.object(
                self.fake_cutlass,
                "utils",
                types.SimpleNamespace(HardwareInfo=FakeHardwareInfo),
                create=True,
            ),
            mock.patch.object(
                self.fake_cutlass,
                "Float32",
                lambda value: ("Float32", value),
                create=True,
            ),
        ):
            self.srelu._launch(
                "stream",
                "a",
                "b",
                "sfa",
                "sfb",
                "prob",
                "c",
                "d",
                alpha=0.25,
                sf_vec_size=32,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                vector_f32=False,
                cluster_overlap_margin=1,
            )
            self.dsrelu._launch(
                "stream",
                "a",
                "b",
                "c",
                "sfa",
                "sfb",
                "prob",
                "d",
                "dprob",
                alpha=0.5,
                sf_vec_size=32,
                mma_tiler_mn=(256, 256),
                cluster_shape_mn=(2, 1),
                vector_f32=True,
                cluster_overlap_margin=2,
            )

        srelu_args, srelu_kwargs = srelu_calls[0]
        self.assertEqual(
            srelu_args,
            (
                "a",
                "b",
                "sfa",
                "sfb",
                "c",
                "d",
                "prob",
                None,
                None,
                None,
                ("Float32", 0.25),
                8,
                "stream",
            ),
        )
        self.assertEqual(set(srelu_kwargs), {"epilogue_op"})
        dsrelu_args, dsrelu_kwargs = dsrelu_calls[0]
        self.assertEqual(
            dsrelu_args,
            (
                "a",
                "b",
                "sfa",
                "sfb",
                "c",
                "d",
                "prob",
                "dprob",
                None,
                None,
                None,
                ("Float32", 0.5),
                8,
                "stream",
            ),
        )
        self.assertEqual(set(dsrelu_kwargs), {"epilogue_op"})


if __name__ == "__main__":
    unittest.main()
