# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the JAX dense GEMM wrappers."""

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
            _COMMON_KERNEL_CAPABILITIES | {"SINGLE_CTA_CLUSTER_SHAPE", "SWIGLU_BLOCK_COLUMNS", "SWIGLU_BLOCKS_PER_PAIR"},
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
            _COMMON_KERNEL_CAPABILITIES | {"KNOWN_HANG_MMA_TILER_M"},
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
        a = _Array((128, 128, 1), self.bfloat16)
        b = _Array((128, 128, 1), self.bfloat16)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(self.swiglu, "_make_launcher", return_value=launcher),
            mock.patch.object(
                self.swiglu,
                "call_cutedsl",
                side_effect=self._fake_call(captured),
            ),
        ):
            result = self.swiglu.gemm_swiglu_wrapper_sm100(
                a,
                b,
                c_major="m",
                ab12_dtype=_Array((), self.float32),
                c_dtype=_Array((), self.bfloat16),
                acc_dtype=_Array((), self.float32),
                a_major="m",
                b_major="n",
            )

        self.assertIs(captured["launcher"], launcher)
        self.assertEqual(captured["inputs"], (a, b))
        self.assertTrue(captured["use_static_tensors"])
        self.assertNotIn("workspaces", captured)
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype, spec.fill_value) for spec in captured["outputs"]],
            [
                ("ab12_tensor", (128, 128, 1), self.float32, None),
                ("c_tensor", (128, 64, 1), self.bfloat16, None),
            ],
        )
        self.assertEqual([spec.layout for spec in captured["input_specs"]], [(0, 1, 2), (0, 1, 2)])
        self.assertEqual([spec.mode for spec in captured["input_specs"]], [(0, 1, 2), (0, 1, 2)])
        self.assertEqual(
            [spec.tensor_spec.layout for spec in captured["outputs"]],
            [(0, 1, 2), (0, 1, 2)],
        )
        self.assertEqual(result._fields, ("ab12_tensor", "c_tensor", "sfc_tensor", "amax_tensor"))
        self.assertIsNone(result.sfc_tensor)
        self.assertIsNone(result.amax_tensor)

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
        a = _Array((128, 128, 1), self.float8_e4m3fn)
        b = _Array((128, 128, 1), self.float8_e4m3fn)
        sfa = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        sfb = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        launcher = object()
        with (
            self._optional_modules(),
            mock.patch.object(self.amax, "_make_launcher", return_value=launcher),
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
                c_major="n",
                c_dtype=self.float32,
            )

        self.assertEqual(captured["inputs"], (a, b, sfa, sfb))
        self.assertTrue(captured["use_static_tensors"])
        self.assertEqual(
            [spec.layout for spec in captured["input_specs"]],
            [(1, 0, 2), (1, 0, 2), (2, 1, 4, 0, 3, 5), (2, 1, 4, 0, 3, 5)],
        )
        c_spec, amax_spec = captured["outputs"]
        self.assertEqual(
            (c_spec.name, c_spec.shape, c_spec.dtype),
            ("c_tensor", (128, 128, 1), self.float32),
        )
        self.assertEqual(c_spec.tensor_spec.layout, (1, 0, 2))
        self.assertEqual(
            (amax_spec.name, amax_spec.shape, amax_spec.dtype, amax_spec.fill_value),
            ("amax_tensor", (1, 1, 1), self.float32, float("-inf")),
        )
        self.assertEqual(result._fields, ("c_tensor", "amax_tensor"))

    def test_swiglu_rejects_quantized_arguments(self):
        a = _Array((128, 128, 1), self.bfloat16)
        b = _Array((128, 128, 1), self.bfloat16)
        scale = _Array((1,), self.float8_e8m0fnu)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(NotImplementedError, "unquantized path"),
            mock.patch.object(self.swiglu, "_make_launcher") as make_launcher,
        ):
            self.swiglu.gemm_swiglu_wrapper_sm100(a, b, sfa_tensor=scale, sfb_tensor=scale)
        make_launcher.assert_not_called()

    def test_swiglu_rejects_unsupported_epilogue_and_incomplete_2cta_tiles(self):
        a = _Array((128, 128, 1), self.bfloat16)
        b = _Array((64, 128, 1), self.bfloat16)

        with self._optional_modules():
            for mma_tiler_mn, message in (
                ((128, 32), r"N in \{64, 128, 192, 256\}"),
                ((256, 64), "M must be divisible by 256"),
            ):
                with (
                    self.subTest(mma_tiler_mn=mma_tiler_mn),
                    self.assertRaisesRegex(ValueError, message),
                    mock.patch.object(self.swiglu, "_make_launcher") as make_launcher,
                ):
                    self.swiglu.gemm_swiglu_wrapper_sm100(a, b, mma_tiler_mn=mma_tiler_mn)
                make_launcher.assert_not_called()

    def test_amax_validates_scale_shapes_before_lowering(self):
        a = _Array((128, 128, 1), self.float8_e5m2)
        b = _Array((128, 128, 1), self.float8_e5m2)
        bad_sfa = _Array((32, 4, 1, 4, 2, 1), self.float8_e8m0fnu)
        sfb = _Array((32, 4, 1, 4, 1, 1), self.float8_e8m0fnu)
        with (
            self._optional_modules(),
            self.assertRaisesRegex(ValueError, "sfa_tensor must have shape"),
            mock.patch.object(self.amax, "_make_launcher") as make_launcher,
        ):
            self.amax.gemm_amax_wrapper_sm100(a, b, bad_sfa, sfb)
        make_launcher.assert_not_called()

    def test_srelu_and_dsrelu_declare_functional_outputs(self):
        a = _Array((384, 512, 2), self.float8_e4m3fn)
        b = _Array((256, 512, 2), self.float8_e4m3fn)
        c = _Array((384, 256, 2), self.bfloat16)
        sfa = _Array((32, 4, 3, 4, 4, 2), self.float8_e8m0fnu)
        sfb = _Array((32, 4, 2, 4, 4, 2), self.float8_e8m0fnu)
        prob = _Array((384, 1, 2), self.float32)
        srelu_captured = {}
        dsrelu_captured = {}

        with (
            self._optional_modules(),
            mock.patch.object(self.srelu, "_make_launcher", return_value="srelu"),
            mock.patch.object(
                self.srelu,
                "call_cutedsl",
                side_effect=self._fake_call(srelu_captured),
            ),
            mock.patch.object(self.dsrelu, "_make_launcher", return_value="dsrelu"),
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
        self.assertEqual(
            [(spec.name, spec.shape, spec.dtype, spec.fill_value) for spec in srelu_captured["outputs"]],
            [
                ("c_tensor", (384, 256, 2), self.bfloat16, None),
                ("d_tensor", (384, 256, 2), self.bfloat16, None),
            ],
        )
        self.assertEqual(
            [spec.layout for spec in srelu_captured["input_specs"]],
            [
                (1, 0, 2),
                (1, 0, 2),
                (2, 1, 4, 0, 3, 5),
                (2, 1, 4, 0, 3, 5),
                (0, 1, 2),
            ],
        )
        self.assertIsNone(srelu_result.amax_tensor)
        self.assertIsNone(srelu_result.sfd_tensor)

        self.assertEqual(dsrelu_captured["inputs"], (a, b, c, sfa, sfb, prob))
        d_spec, dprob_spec = dsrelu_captured["outputs"]
        self.assertEqual(
            (d_spec.name, d_spec.shape, d_spec.dtype),
            ("d_tensor", (384, 256, 2), self.bfloat16),
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
        self.assertIsNone(dsrelu_result.amax_tensor)
        self.assertIsNone(dsrelu_result.sfd_tensor)
        self.assertTrue(srelu_captured["use_static_tensors"])
        self.assertTrue(dsrelu_captured["use_static_tensors"])

    def test_relu_wrappers_reject_unpredicated_partial_mma_tile(self):
        with self._optional_modules():
            for m, mma_tiler_mn in ((129, (128, 128)), (130, (256, 256))):
                a = _Array((m, 512, 1), self.float8_e4m3fn)
                b = _Array((256, 512, 1), self.float8_e4m3fn)
                c = _Array((m, 256, 1), self.bfloat16)
                scale_m = (m + 127) // 128
                sfa = _Array((32, 4, scale_m, 4, 4, 1), self.float8_e8m0fnu)
                sfb = _Array((32, 4, 2, 4, 4, 1), self.float8_e8m0fnu)
                prob = _Array((m, 1, 1), self.float32)

                for module, call_args in (
                    (self.srelu, (a, b, sfa, sfb, prob)),
                    (self.dsrelu, (a, b, c, sfa, sfb, prob)),
                ):
                    with (
                        self.subTest(module=module.__name__, mma_tiler_mn=mma_tiler_mn),
                        self.assertRaisesRegex(ValueError, f"TILE_M={mma_tiler_mn[0]}"),
                        mock.patch.object(module, "_make_launcher") as make_launcher,
                    ):
                        getattr(module, module.__all__[1])(*call_args, mma_tiler_mn=mma_tiler_mn)
                    make_launcher.assert_not_called()

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

        self.swiglu._make_launcher.cache_clear()
        self.amax._make_launcher.cache_clear()
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
            swiglu_launch = self.swiglu._make_launcher(
                alpha=0.5,
                acc_dtype=self.float32,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                cluster_overlap_margin=1,
            )
            swiglu_launch("stream", "a", "b", "ab12", "c")
            amax_launch = self.amax._make_launcher(
                sf_vec_size=32,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(2, 2),
                cluster_overlap_margin=2,
            )
            amax_launch("stream", "a", "b", "sfa", "sfb", "c", "amax")

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

        self.srelu._make_launcher.cache_clear()
        self.dsrelu._make_launcher.cache_clear()
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
            srelu_launch = self.srelu._make_launcher(
                alpha=0.25,
                sf_vec_size=32,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                vector_f32=False,
                cluster_overlap_margin=1,
            )
            srelu_launch("stream", "a", "b", "sfa", "sfb", "prob", "c", "d")
            dsrelu_launch = self.dsrelu._make_launcher(
                alpha=0.5,
                sf_vec_size=32,
                mma_tiler_mn=(256, 256),
                cluster_shape_mn=(2, 1),
                vector_f32=True,
                cluster_overlap_margin=2,
            )
            dsrelu_launch(
                "stream",
                "a",
                "b",
                "c",
                "sfa",
                "sfb",
                "prob",
                "d",
                "dprob",
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
