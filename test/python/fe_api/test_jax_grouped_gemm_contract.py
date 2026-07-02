# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free contracts for the JAX contiguous grouped GEMM wrappers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ast
import importlib
import importlib.util
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
_TEST_PACKAGE = "cudnn_frontend_jax_grouped_gemm_contract_test"


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
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass(frozen=True)
class _BufferSpec:
    name: str
    shape: tuple
    dtype: object
    tensor_spec: object = None
    fill_value: object = None


class _KernelSurface:
    MMA_TILER_M = (64, 128, 256)
    MMA_TILER_N = (128, 256)
    TWO_CTA_MMA_TILER_M = 256
    MAX_CLUSTER_CTAS = 16
    MAX_CLUSTER_DIMENSION = 4
    CLUSTER_TILER_M = (128, 256)
    FP8_SF_VEC_SIZE = 32
    SF_VEC_SIZES = (16, 32)
    HADAMARD_SIZE = 16
    FIX_PAD_SIZE = 256
    MAX_EXPERTS = 1024
    DYNAMIC_SCHED_WORKSPACE_BYTES = 4
    calls = []

    @classmethod
    def require_mma_tiler(cls, value):
        value = tuple(value)
        if value[0] not in cls.MMA_TILER_M or value[1] not in cls.MMA_TILER_N:
            raise ValueError(value)
        return value

    @classmethod
    def require_cluster_shape(cls, value, *, mma_tiler_mn):
        del mma_tiler_mn
        return tuple(value)

    @classmethod
    def get_dense_workspace_bytes(cls, use_dynamic_sched):
        return cls.DYNAMIC_SCHED_WORKSPACE_BYTES if use_dynamic_sched else 0

    @classmethod
    def can_implement(cls, *args):
        del args
        return True

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        type(self).calls.append((self.kwargs, args, kwargs))


class JaxGroupedGemmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfloat16 = _DType("bfloat16", 2)
        cls.float16 = _DType("float16", 2)
        cls.float32 = _DType("float32", 4)
        cls.float8_e4m3fn = _DType("float8_e4m3fn", 1)
        cls.float8_e5m2 = _DType("float8_e5m2", 1)
        cls.float8_e8m0fnu = _DType("float8_e8m0fnu", 1)
        cls.float4_e2m1fn = _DType("float4_e2m1fn", 1)
        cls.int32 = _DType("int32", 4)
        cls.uint8 = _DType("uint8", 1)

        cls.fake_jnp = types.ModuleType("jax.numpy")
        for name in (
            "bfloat16",
            "float16",
            "float32",
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e8m0fnu",
            "float4_e2m1fn",
            "int32",
            "uint8",
        ):
            setattr(cls.fake_jnp, name, getattr(cls, name))
        cls.fake_jnp.dtype = lambda value: value
        cls.fake_jnp.ones = lambda shape, dtype: _Array(shape, dtype)
        cls.fake_jnp.asarray = lambda value, dtype: _Array((len(value), len(value[0])), dtype)

        cls.fake_jax = types.ModuleType("jax")
        cls.fake_jax.__path__ = []
        cls.fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
        cls.fake_jax.numpy = cls.fake_jnp

        cls.fake_cutlass = types.ModuleType("cutlass")
        cls.fake_cutlass.__path__ = []
        cls.fake_cutlass.Constexpr = object
        cls.fake_cutlass.utils = types.SimpleNamespace(HardwareInfo=lambda: types.SimpleNamespace(get_max_active_clusters=lambda _: 8))
        cls.fake_cutlass.Int32 = int
        cls.fake_cutlass.Int64 = int
        cls.fake_cutlass.Float32 = float
        cls.fake_cutlass_cute = types.ModuleType("cutlass.cute")
        cls.fake_cutlass_cute.jit = _identity_jit
        cls.fake_cutlass_cute.where = lambda condition, yes, no: (yes if condition else no)
        cls.fake_cutlass_cute.full_like = lambda _, fill: fill
        cls.fake_cutlass.cute = cls.fake_cutlass_cute
        cls.fake_cutlass_jax = types.ModuleType("cutlass.jax")
        cls.fake_cutlass_jax.TensorSpec = _TensorSpec
        cls.fake_cutlass_jax.jax_to_cutlass_dtype = lambda dtype: (f"cutlass.{dtype.name}")
        cls.fake_cutlass.jax = cls.fake_cutlass_jax
        cls.fake_cutlass_nvgpu = types.ModuleType("cutlass.cute.nvgpu")
        cls.fake_cutlass_nvgpu.OperandMajorMode = types.SimpleNamespace(K="K")

        cls.calls = []

        def call_cutedsl(fn, inputs, *, outputs, input_specs, **kwargs):
            del input_specs
            result_buffers = tuple(types.SimpleNamespace(name=spec.name) for spec in outputs)
            workspace_buffers = tuple(types.SimpleNamespace(name=spec.name, iterator=f"{spec.name}_ptr") for spec in kwargs.get("workspaces", ()))
            fn("stream", *inputs, *result_buffers, *workspace_buffers)
            cls.calls.append((tuple(outputs), kwargs))
            return result_buffers

        cls.fake_cutedsl = types.ModuleType(f"{_TEST_PACKAGE}._jax.cutedsl")
        cls.fake_cutedsl.BufferSpec = _BufferSpec
        cls.fake_cutedsl.call_cutedsl = call_cutedsl

        cls.fake_gemm = types.ModuleType(f"{_TEST_PACKAGE}._jax.gemm")
        cls.fake_gemm.require_array = cls._require_array
        cls.fake_gemm.require_16_byte_extent = lambda *_: None
        cls.fake_gemm.block_scale_tensor_spec = lambda: "scale_spec"
        cls.fake_gemm.gemm_a_tensor_spec = lambda major: f"a_{major}"
        cls.fake_gemm.gemm_b_tensor_spec = lambda major: f"b_{major}"
        cls.fake_gemm.gemm_c_tensor_spec = lambda major: f"c_{major}"
        cls.fake_gemm.probability_tensor_spec = lambda: "prob_spec"

        cls.fake_kernel_modules = {}
        for suffix, class_name in (
            (
                "grouped_gemm.grouped_gemm_swiglu.grouped_gemm_swiglu_quant",
                "BlockScaledContiguousGroupedGemmKernel",
            ),
            (
                "grouped_gemm.grouped_gemm_dswiglu.grouped_gemm_dswiglu_quant",
                "BlockScaledContiguousGroupedGemmKernel",
            ),
            (
                "grouped_gemm.grouped_gemm_quant.grouped_gemm_quant",
                "BlockScaledMoEGroupedGemmQuantKernel",
            ),
            (
                "grouped_gemm.grouped_gemm_srelu.moe_blockscaled_grouped_gemm_srelu_quant",
                "BlockScaledMoEGroupedGemmQuantKernel",
            ),
            (
                "grouped_gemm.grouped_gemm_glu.moe_blockscaled_grouped_gemm_glu_bias",
                "BlockScaledMoEGroupedGemmGluBiasKernel",
            ),
            (
                "grouped_gemm.grouped_gemm_dglu.moe_blockscaled_grouped_gemm_dglu_dbias",
                "BlockScaledMoEGroupedGemmDgluDbiasKernel",
            ),
            (
                "grouped_gemm.grouped_gemm_glu_hadamard.moe_blockscaled_grouped_gemm_glu_hadamard",
                "BlockScaledMoEGroupedGemmGluHadamardKernel",
            ),
            (
                "grouped_gemm.grouped_gemm_dsrelu.moe_blockscaled_grouped_gemm_dsrelu_quant",
                "BlockScaledMoEGroupedGemmQuantBwdKernel",
            ),
        ):
            module = types.ModuleType(f"{_TEST_PACKAGE}.{suffix}")
            setattr(module, class_name, _KernelSurface)
            if "grouped_gemm_srelu" in suffix:
                module.EpilogueType = types.SimpleNamespace(SRELU=types.SimpleNamespace(value=1))
            if "grouped_gemm_dsrelu" in suffix:
                module.EpilogueType = types.SimpleNamespace(DSRELU=types.SimpleNamespace(value=1))
            cls.fake_kernel_modules[module.__name__] = module

        cls.fake_moe_utils = types.ModuleType(f"{_TEST_PACKAGE}.grouped_gemm.moe_utils")
        cls.fake_moe_utils.MoEWeightMode = types.SimpleNamespace(DENSE="DENSE")

        with cls._modules():
            cls._make_package(_TEST_PACKAGE, _CUDNN_ROOT)
            cls._make_package(f"{_TEST_PACKAGE}._jax", _CUDNN_ROOT / "_jax")
            cls._make_package(f"{_TEST_PACKAGE}.grouped_gemm", _CUDNN_ROOT / "grouped_gemm")
            for operation in (
                "grouped_gemm_swiglu",
                "grouped_gemm_dswiglu",
                "grouped_gemm_quant",
                "grouped_gemm_srelu",
                "grouped_gemm_glu",
                "grouped_gemm_dglu",
                "grouped_gemm_glu_hadamard",
                "grouped_gemm_dsrelu",
            ):
                cls._make_package(
                    f"{_TEST_PACKAGE}.grouped_gemm.{operation}",
                    _CUDNN_ROOT / "grouped_gemm" / operation,
                )
            cls._load_source(f"{_TEST_PACKAGE}.gemm_validation", _CUDNN_ROOT / "gemm_validation.py")
            cls._load_source(
                f"{_TEST_PACKAGE}._jax.validation",
                _CUDNN_ROOT / "_jax" / "validation.py",
            )
            cls._load_source(
                f"{_TEST_PACKAGE}._jax.grouped_gemm",
                _CUDNN_ROOT / "_jax" / "grouped_gemm.py",
            )
            cls.swiglu = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_swiglu.jax")
            cls.dswiglu = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_dswiglu.jax")
            cls.quant = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_quant.jax")
            cls.srelu = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_srelu.jax")
            cls.glu = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_glu.jax")
            cls.dglu = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_dglu.jax")
            cls.glu_hadamard = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_glu_hadamard.jax")
            cls.dsrelu = importlib.import_module(f"{_TEST_PACKAGE}.grouped_gemm.grouped_gemm_dsrelu.jax")

    @classmethod
    def tearDownClass(cls):
        for module_name in tuple(sys.modules):
            if module_name == _TEST_PACKAGE or module_name.startswith(f"{_TEST_PACKAGE}."):
                sys.modules.pop(module_name, None)

    @staticmethod
    def _require_array(name, value, rank):
        if len(value.shape) != rank:
            raise ValueError(f"{name} must have rank {rank}")
        return tuple(value.shape)

    @staticmethod
    def _make_package(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module

    @staticmethod
    def _load_source(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    @contextmanager
    def _modules(cls):
        modules = {
            "jax": cls.fake_jax,
            "jax.numpy": cls.fake_jnp,
            "cutlass": cls.fake_cutlass,
            "cutlass.cute": cls.fake_cutlass_cute,
            "cutlass.cute.nvgpu": cls.fake_cutlass_nvgpu,
            "cutlass.jax": cls.fake_cutlass_jax,
            "torch": None,
            f"{_TEST_PACKAGE}._jax.cutedsl": cls.fake_cutedsl,
            f"{_TEST_PACKAGE}._jax.gemm": cls.fake_gemm,
            f"{_TEST_PACKAGE}.grouped_gemm.moe_utils": cls.fake_moe_utils,
            **cls.fake_kernel_modules,
        }
        with mock.patch.dict(sys.modules, modules):
            yield

    def setUp(self):
        self.calls.clear()
        _KernelSurface.calls.clear()

    def _inputs(self):
        m, n, k, experts = 256, 256, 64, 4
        return {
            "a": _Array((m, k, 1), self.float8_e4m3fn),
            "b": _Array((n, k, experts), self.float8_e4m3fn),
            "sfa": _Array((32, 4, 2, 4, 1, 1), self.float8_e8m0fnu),
            "sfb": _Array((32, 4, 2, 4, 1, experts), self.float8_e8m0fnu),
            "offsets": _Array((experts,), self.int32),
            "alpha": _Array((experts,), self.float32),
            "norm": _Array((1,), self.float32),
            "prob": _Array((m, 1, 1), self.float32),
        }

    def test_forward_functional_outputs(self):
        values = self._inputs()
        with self._modules():
            result = self.swiglu.grouped_gemm_swiglu_wrapper_sm100(
                values["a"],
                values["b"],
                values["sfa"],
                values["sfb"],
                values["offsets"],
                values["alpha"],
                values["norm"],
                values["prob"],
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            [
                "c_tensor",
                "d_tensor",
                "d_col_tensor",
                "amax_tensor",
                "sfd_row_tensor",
                "sfd_col_tensor",
            ],
        )
        specs, options = self.calls[-1]
        self.assertEqual(
            next(spec.fill_value for spec in specs if spec.name == "amax_tensor"),
            -float("inf"),
        )
        self.assertEqual(
            next(spec.shape for spec in specs if spec.name == "sfd_row_tensor"),
            (32, 4, 2, 4, 1, 1),
        )
        self.assertTrue(options["use_static_tensors"])
        self.assertEqual(_KernelSurface.calls[-1][0]["expert_cnt"], 4)

    def test_backward_zero_initializes_dprob(self):
        values = self._inputs()
        c_tensor = _Array((256, 512, 1), self.bfloat16)
        with self._modules():
            result = self.dswiglu.grouped_gemm_dswiglu_wrapper_sm100(
                values["a"],
                values["b"],
                c_tensor,
                values["sfa"],
                values["sfb"],
                values["offsets"],
                values["alpha"],
                None,
                values["prob"],
                values["norm"],
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            [
                "d_row_tensor",
                "d_col_tensor",
                "dprob_tensor",
                None,
                "sfd_row_tensor",
                "sfd_col_tensor",
            ],
        )
        specs, _ = self.calls[-1]
        self.assertEqual(next(spec.fill_value for spec in specs if spec.name == "dprob_tensor"), 0.0)
        self.assertEqual(
            next(spec.dtype for spec in specs if spec.name == "d_row_tensor"),
            self.float8_e4m3fn,
        )

    def test_quantized_output_uses_hidden_dynamic_workspace(self):
        values = self._inputs()
        bias = _Array((256, 4), self.float32)
        row_scale = _Array((256,), self.float32)
        with self._modules():
            result = self.quant.grouped_gemm_quant_wrapper_sm100(
                values["a"],
                values["sfa"],
                values["offsets"],
                values["alpha"],
                values["b"],
                values["sfb"],
                bias_tensor=bias,
                norm_const_tensor=values["norm"],
                prob_tensor=values["prob"],
                row_scale_tensor=row_scale,
                d_dtype=self.float8_e4m3fn,
                use_dynamic_sched=True,
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            [
                "d_tensor",
                "d_col_tensor",
                None,
                "sfd_row_tensor",
                "sfd_col_tensor",
            ],
        )
        specs, options = self.calls[-1]
        self.assertNotIn("amax_tensor", {spec.name for spec in specs})
        workspace = options["workspaces"][0]
        self.assertEqual(workspace.shape, (4,))
        self.assertEqual(workspace.dtype, self.uint8)
        self.assertEqual(workspace.tensor_spec.ptr_assumed_align, 128)
        self.assertTrue(_KernelSurface.calls[-1][0]["use_dynamic_sched"])

    def test_srelu_nonquantized_output_initializes_amax(self):
        values = self._inputs()
        with self._modules():
            result = self.srelu.grouped_gemm_srelu_wrapper_sm100(
                values["a"],
                values["b"],
                values["sfa"],
                values["sfb"],
                values["offsets"],
                values["alpha"],
                prob_tensor=values["prob"],
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            ["c_tensor", "d_tensor", None, "amax_tensor", None, None],
        )
        specs, options = self.calls[-1]
        self.assertEqual(
            next(spec.fill_value for spec in specs if spec.name == "amax_tensor"),
            -float("inf"),
        )
        self.assertEqual(options["workspaces"][0].shape, (1,))
        self.assertEqual(_KernelSurface.calls[-1][0]["epilogue_type"], 1)

    def test_glu_nonquantized_output_uses_hidden_d_col_and_amax(self):
        values = self._inputs()
        with self._modules():
            result = self.glu.grouped_gemm_glu_wrapper_sm100(
                values["a"],
                values["sfa"],
                values["offsets"],
                values["alpha"],
                values["b"],
                values["sfb"],
                prob_tensor=values["prob"],
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            ["c_tensor", "d_tensor", None, "amax_tensor", None, None],
        )
        specs, options = self.calls[-1]
        self.assertEqual(
            next(spec.fill_value for spec in specs if spec.name == "amax_tensor"),
            -float("inf"),
        )
        self.assertEqual(
            [spec.name for spec in options["workspaces"]],
            ["d_col_scratch", "workspace"],
        )
        kernel_config, kernel_args, _ = _KernelSurface.calls[-1]
        self.assertEqual(kernel_config["weight_mode"], "DENSE")
        self.assertEqual(kernel_args[10].name, "d_col_scratch")

    def test_glu_quantized_output_supports_bias_and_dynamic_workspace(self):
        values = self._inputs()
        bias = _Array((256, 4), self.float32)
        with self._modules():
            result = self.glu.grouped_gemm_glu_wrapper_sm100(
                values["a"],
                values["sfa"],
                values["offsets"],
                values["alpha"],
                values["b"],
                values["sfb"],
                bias_tensor=bias,
                norm_const_tensor=values["norm"],
                prob_tensor=values["prob"],
                d_dtype=self.float8_e4m3fn,
                act_func="geglu",
                use_dynamic_sched=True,
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            [
                "c_tensor",
                "d_tensor",
                "d_col_tensor",
                None,
                "sfd_row_tensor",
                "sfd_col_tensor",
            ],
        )
        _, options = self.calls[-1]
        self.assertEqual([spec.name for spec in options["workspaces"]], ["workspace"])
        self.assertEqual(options["workspaces"][0].shape, (4,))
        kernel_config, _, launch_kwargs = _KernelSurface.calls[-1]
        self.assertTrue(kernel_config["enable_bias"])
        self.assertTrue(kernel_config["generate_sfd"])
        self.assertEqual(launch_kwargs["linear_offset"], 1.0)

    def test_dglu_zero_initializes_functional_reductions(self):
        values = self._inputs()
        c_tensor = _Array((256, 512, 1), self.bfloat16)
        beta = _Array((4,), self.float32)
        with self._modules():
            result = self.dglu.grouped_gemm_dglu_wrapper_sm100(
                values["a"],
                c_tensor,
                values["sfa"],
                values["offsets"],
                values["alpha"],
                beta,
                values["prob"],
                values["b"],
                values["sfb"],
                generate_dbias=True,
                norm_const_tensor=values["norm"],
                use_dynamic_sched=True,
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            [
                "d_row_tensor",
                "d_col_tensor",
                "dprob_tensor",
                "dbias_tensor",
                None,
                "sfd_row_tensor",
                "sfd_col_tensor",
            ],
        )
        specs, options = self.calls[-1]
        fills = {spec.name: spec.fill_value for spec in specs}
        self.assertEqual(fills["dprob_tensor"], 0.0)
        self.assertEqual(fills["dbias_tensor"], 0.0)
        self.assertEqual(options["workspaces"][0].shape, (4,))
        kernel_config, _, _ = _KernelSurface.calls[-1]
        self.assertEqual(kernel_config["weight_mode"], "DENSE")
        self.assertEqual(kernel_config["act_func"], "dswiglu")

    def test_glu_hadamard_native_fp4_outputs_and_workspace_are_functional(self):
        values = self._inputs()
        values["a"] = _Array(values["a"].shape, self.float4_e2m1fn)
        values["b"] = _Array(values["b"].shape, self.float4_e2m1fn)
        bias = _Array((256, 4), self.float32)
        with self._modules():
            result = self.glu_hadamard.grouped_gemm_glu_hadamard_wrapper_sm100(
                values["a"],
                values["b"],
                values["sfa"],
                values["sfb"],
                values["offsets"],
                values["alpha"],
                values["prob"],
                bias_tensor=bias,
                act_func="geglu",
                use_dynamic_sched=True,
                use_tmem_post_rht_amax=True,
            )

        self.assertEqual(
            [item.name for item in result],
            ["c_tensor", "d_tensor", "amax_tensor", "post_rht_amax_tensor"],
        )
        specs, options = self.calls[-1]
        fills = {spec.name: spec.fill_value for spec in specs}
        self.assertEqual(
            fills,
            {
                "c_tensor": 0,
                "d_tensor": 0,
                "amax_tensor": 0.0,
                "post_rht_amax_tensor": 0.0,
            },
        )
        workspace = options["workspaces"][0]
        self.assertEqual(workspace.shape, (4,))
        self.assertEqual(workspace.fill_value, 0)
        self.assertEqual(workspace.tensor_spec.ptr_assumed_align, 128)

        kernel_config, kernel_args, launch_kwargs = _KernelSurface.calls[-1]
        self.assertEqual(kernel_config["weight_mode"], "DENSE")
        self.assertTrue(kernel_config["enable_bias"])
        self.assertTrue(kernel_config["use_tmem_post_rht_amax"])
        self.assertEqual(kernel_args[8], "workspace_ptr")
        self.assertEqual(kernel_args[16].shape, (16, 16))
        self.assertIs(kernel_args[17], bias)
        self.assertEqual(launch_kwargs["linear_offset"], 1.0)

    def test_glu_hadamard_rejects_raw_uint8_fp4(self):
        values = self._inputs()
        values["a"] = _Array(values["a"].shape, self.uint8)
        values["b"] = _Array(values["b"].shape, self.uint8)
        with self._modules(), self.assertRaisesRegex(ValueError, "a_tensor.dtype"):
            self.glu_hadamard.grouped_gemm_glu_hadamard_wrapper_sm100(
                values["a"],
                values["b"],
                values["sfa"],
                values["sfb"],
                values["offsets"],
                values["alpha"],
                values["prob"],
            )

    def test_glu_hadamard_srelu_supports_e4m3_scales_and_static_workspace(self):
        values = self._inputs()
        values["a"] = _Array(values["a"].shape, self.float4_e2m1fn)
        values["b"] = _Array(values["b"].shape, self.float4_e2m1fn)
        values["sfa"] = _Array(values["sfa"].shape, self.float8_e4m3fn)
        values["sfb"] = _Array(values["sfb"].shape, self.float8_e4m3fn)
        with self._modules():
            self.glu_hadamard.grouped_gemm_glu_hadamard_wrapper_sm100(
                values["a"],
                values["b"],
                values["sfa"],
                values["sfb"],
                values["offsets"],
                values["alpha"],
                values["prob"],
                act_func="srelu",
                sf_vec_size=16,
            )

        specs, options = self.calls[-1]
        self.assertEqual(
            next(spec.shape for spec in specs if spec.name == "d_tensor"),
            (256, 256, 1),
        )
        self.assertEqual(options["workspaces"][0].shape, (1,))
        self.assertEqual(options["workspaces"][0].fill_value, 0)

    def test_glu_hadamard_lazy_surface_and_kernel_imports_are_torch_optional(self):
        init_tree = ast.parse((_CUDNN_ROOT / "grouped_gemm/grouped_gemm_glu_hadamard/__init__.py").read_text())
        exports = next(
            node for node in init_tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "_API_EXPORTS"
        )
        self.assertIsInstance(exports.value, ast.Tuple)

        for relative_path in (
            "grouped_gemm/grouped_gemm_glu_hadamard/hadamard_utils.py",
            "grouped_gemm/moe_kernel_helpers.py",
        ):
            tree = ast.parse((_CUDNN_ROOT / relative_path).read_text())
            required_torch_imports = [
                node
                for node in tree.body
                if (isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
                or (isinstance(node, ast.ImportFrom) and node.module == "torch")
            ]
            self.assertEqual(required_torch_imports, [])

    def test_dsrelu_exposes_functional_outputs_and_hidden_workspace(self):
        values = self._inputs()
        c_tensor = _Array((256, 256, 1), self.bfloat16)
        with self._modules():
            result = self.dsrelu.grouped_gemm_dsrelu_wrapper_sm100(
                values["a"],
                values["b"],
                c_tensor,
                values["sfa"],
                values["sfb"],
                values["offsets"],
                values["alpha"],
                values["prob"],
                generate_dbias=True,
                norm_const_tensor=values["norm"],
                use_dynamic_sched=True,
                use_dsrelu_reuse=True,
            )

        self.assertEqual(
            [item.name if item is not None else None for item in result],
            [
                "d_row_tensor",
                "d_col_tensor",
                "d_srelu_tensor",
                "dprob_tensor",
                "dbias_tensor",
                None,
                "sfd_row_tensor",
                "sfd_col_tensor",
                "sfd_col_d_srelu_tensor",
            ],
        )
        specs, options = self.calls[-1]
        fills = {spec.name: spec.fill_value for spec in specs}
        self.assertEqual(fills["dprob_tensor"], 0.0)
        self.assertEqual(fills["dbias_tensor"], 0.0)
        self.assertEqual(options["workspaces"][0].shape, (4,))
        kernel_config, _, _ = _KernelSurface.calls[-1]
        self.assertEqual(kernel_config["weight_mode"], "DENSE")
        self.assertEqual(kernel_config["epilogue_type"], 1)
        self.assertTrue(kernel_config["generate_d_srelu"])
        self.assertTrue(kernel_config["use_dsrelu_reuse"])

    def test_kernel_capabilities_are_public(self):
        required_fields = {
            "MMA_TILER_M",
            "MMA_TILER_N",
            "TWO_CTA_MMA_TILER_M",
            "MAX_CLUSTER_CTAS",
            "MAX_CLUSTER_DIMENSION",
            "CLUSTER_TILER_M",
            "FP8_SF_VEC_SIZE",
            "FIX_PAD_SIZE",
            "MAX_EXPERTS",
        }
        for relative_path in (
            "grouped_gemm/grouped_gemm_swiglu/grouped_gemm_swiglu_quant.py",
            "grouped_gemm/grouped_gemm_dswiglu/grouped_gemm_dswiglu_quant.py",
        ):
            tree = ast.parse((_CUDNN_ROOT / relative_path).read_text())
            kernel = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlockScaledContiguousGroupedGemmKernel")
            fields = {
                node.targets[0].id for node in kernel.body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            }
            methods = {node.name for node in kernel.body if isinstance(node, ast.FunctionDef)}
            self.assertTrue(required_fields <= fields)
            self.assertTrue({"require_mma_tiler", "require_cluster_shape"} <= methods)

        workspace_fields = required_fields | {"DYNAMIC_SCHED_WORKSPACE_BYTES"}
        for relative_path, class_name in (
            (
                "grouped_gemm/grouped_gemm_quant/grouped_gemm_quant.py",
                "BlockScaledMoEGroupedGemmQuantKernel",
            ),
            (
                "grouped_gemm/grouped_gemm_srelu/moe_blockscaled_grouped_gemm_srelu_quant.py",
                "BlockScaledMoEGroupedGemmQuantKernel",
            ),
            (
                "grouped_gemm/grouped_gemm_glu/moe_blockscaled_grouped_gemm_glu_bias.py",
                "BlockScaledMoEGroupedGemmGluBiasKernel",
            ),
            (
                "grouped_gemm/grouped_gemm_dglu/moe_blockscaled_grouped_gemm_dglu_dbias.py",
                "BlockScaledMoEGroupedGemmDgluDbiasKernel",
            ),
            (
                "grouped_gemm/grouped_gemm_dsrelu/moe_blockscaled_grouped_gemm_dsrelu_quant.py",
                "BlockScaledMoEGroupedGemmQuantBwdKernel",
            ),
        ):
            tree = ast.parse((_CUDNN_ROOT / relative_path).read_text())
            kernel = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
            fields = {
                node.targets[0].id for node in kernel.body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            }
            methods = {node.name for node in kernel.body if isinstance(node, ast.FunctionDef)}
            self.assertTrue(workspace_fields <= fields)
            self.assertTrue(
                {
                    "require_mma_tiler",
                    "require_cluster_shape",
                    "get_dense_workspace_bytes",
                }
                <= methods
            )

        tree = ast.parse((_CUDNN_ROOT / "grouped_gemm/grouped_gemm_glu_hadamard/moe_blockscaled_grouped_gemm_glu_hadamard.py").read_text())
        kernel = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlockScaledMoEGroupedGemmGluHadamardKernel")
        fields = {
            node.targets[0].id for node in kernel.body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        }
        methods = {node.name for node in kernel.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue(
            {
                "MMA_TILER_M",
                "MMA_TILER_N",
                "TWO_CTA_MMA_TILER_M",
                "MAX_CLUSTER_CTAS",
                "MAX_CLUSTER_DIMENSION",
                "CLUSTER_TILER_M",
                "SF_VEC_SIZES",
                "MAX_EXPERTS",
                "DYNAMIC_SCHED_WORKSPACE_BYTES",
                "HADAMARD_SIZE",
                "FIX_PAD_SIZE",
            }
            <= fields
        )
        self.assertTrue(
            {
                "require_mma_tiler",
                "require_cluster_shape",
                "get_dense_workspace_bytes",
            }
            <= methods
        )


if __name__ == "__main__":
    unittest.main()
