# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Source contracts for JAX grouped GEMM adapters."""

from __future__ import annotations

import ast
from enum import Enum, auto
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types
from unittest import mock

import pytest

pytestmark = pytest.mark.L0

_ROOT = Path(__file__).resolve().parents[3]
_GROUPED_ROOT = _ROOT / "python" / "cudnn" / "grouped_gemm"
_FAMILIES = (
    "swiglu",
    "dswiglu",
    "quant",
    "srelu",
    "dsrelu",
    "glu",
    "glu_hadamard",
    "dglu",
    "wgrad",
)

_NATIVE_FP4_FAMILIES = (
    "swiglu",
    "dswiglu",
    "quant",
    "srelu",
    "dsrelu",
    "glu",
    "dglu",
)


def _class_stem(family: str) -> str:
    return "".join(part.capitalize() for part in family.split("_"))


def _jax_path(family: str) -> Path:
    return _GROUPED_ROOT / f"grouped_gemm_{family}" / "jax.py"


@pytest.mark.parametrize("family", _FAMILIES)
def test_each_torch_family_has_a_jax_class_and_wrapper(family: str):
    tree = ast.parse(_jax_path(family).read_text())
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }

    assert f"GroupedGemm{_class_stem(family)}Sm100" in definitions
    wrapper = definitions[f"grouped_gemm_{family}_wrapper_sm100"]
    assert isinstance(wrapper, ast.FunctionDef)
    assert wrapper.decorator_list
    assert "jax.jit" in ast.unparse(wrapper.decorator_list[0])


@pytest.mark.parametrize("family", _FAMILIES)
def test_launchers_receive_precomputed_occupancy_and_preserve_stream_first(family: str):
    source = _jax_path(family).read_text()
    tree = ast.parse(source)
    launcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_launch"
    )

    assert launcher.args.args[0].arg == "stream"
    assert "max_active_clusters" in {
        argument.arg for argument in launcher.args.kwonlyargs
    }
    assert "HardwareInfo" not in source
    assert "resolve_max_active_clusters" not in source


@pytest.mark.parametrize("family", _FAMILIES)
def test_jax_module_has_no_top_level_torch_or_kernel_import(family: str):
    tree = ast.parse(_jax_path(family).read_text())
    imported_modules = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not any(
        module == "torch" or module.startswith("torch.") for module in imported_modules
    )
    assert not any(
        "grouped_gemm_" in module and module != "_jax_api"
        for module in imported_modules
    )


def test_shared_adapter_uses_current_jax_api_base_and_call_kernel():
    source = (_GROUPED_ROOT / "_jax_api.py").read_text()
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    assert ast.unparse(classes["ApiBaseJax"].bases[0]) == "JaxApiBase"
    assert "_CALLER._call_kernel(" in source
    assert "_CALLER._get_max_active_clusters(" in source
    assert source.index("_CALLER._get_max_active_clusters(") < source.index(
        "def launch(stream:"
    )


def test_grouped_layouts_are_explicit_row_major_public_mappings():
    source = (_GROUPED_ROOT / "_jax_api.py").read_text()

    assert "PROBABILITY_MODE = (2, 1, 0)" in source
    assert "GROUPED_BIAS_MODE = (1, 0)" in source
    assert "GROUPED_WORKSPACE_ALIGNMENT = 128" in source
    assert "_canonical_block_scale_shape(rows, k, batch, sf_vec_size)" in source
    assert "BLOCK_SCALE_MODE" in source


@pytest.mark.parametrize("family", ("dswiglu", "dglu", "dsrelu"))
def test_dprob_descriptors_use_public_probability_shape(family: str):
    tree = ast.parse(_jax_path(family).read_text())
    dprob_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_buffer_desc"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "dprob_tensor"
    ]

    assert dprob_calls
    assert all(ast.unparse(call.args[1]) == "(1, 1, m)" for call in dprob_calls)


def test_grouped_buffers_use_tensor_descriptors_directly():
    source = (_GROUPED_ROOT / "_jax_api.py").read_text()

    assert "class BufferSpec" not in source
    assert "class _SampleDesc" not in source
    assert "def make_buffer_desc(" in source
    assert "JaxTensorDesc.from_shape(" in source
    assert "JaxTensorDesc.from_array(" in source
    assert "init_value=init_value" in source


@pytest.mark.parametrize("family", _FAMILIES)
def test_grouped_lowerings_pass_descriptors_without_parallel_specs(family: str):
    tree = ast.parse(_jax_path(family).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "call_cutedsl"
    ]

    assert calls
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "input_descs" in keywords
        assert "outputs" in keywords
        assert not {
            "input_specs",
            "output_specs",
            "workspace_specs",
        } & keywords


@pytest.mark.parametrize("family", _NATIVE_FP4_FAMILIES)
def test_dense_grouped_adapters_use_native_fp4_recipe_validation(family: str):
    source = _jax_path(family).read_text()

    assert "require_grouped_input_scales(" in source
    assert "jnp.float4_e2m1fn" in source or "is_fp4_dtype(ab_dtype)" in source
    assert "torch.uint8" not in source


@pytest.mark.parametrize("family", ("swiglu", "srelu", "quant"))
def test_native_fp4_float32_output_preserves_torch_recipe_guard(family: str):
    source = _jax_path(family).read_text()

    assert "sf_vec_size == 16" in source
    assert "d_dtype == jnp.dtype(jnp.float32)" in source


def test_native_fp4_glu_float32_output_requires_bias_for_sf_vec_16():
    source = _jax_path("glu").read_text()

    assert "sf_vec_size == 16" in source
    assert "d_dtype == jnp.dtype(jnp.float32)" in source
    assert "bias_tensor is None" in source


def test_glu_preserves_torch_sfd_generation_for_all_fp8_inputs():
    source = _jax_path("glu").read_text()

    assert "generate_sfd = is_fp8_dtype(ab_dtype)" in source


def test_glu_accepts_native_fp4_intermediate_output():
    tree = ast.parse(_jax_path("glu").read_text())
    c_dtype_call = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "c_dtype"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "require_dtype"
    )

    assert "jnp.float4_e2m1fn" in ast.unparse(c_dtype_call.args[1])


@pytest.mark.parametrize("family", ("dswiglu", "dglu", "dsrelu"))
def test_native_fp4_backward_paths_return_amax_without_sfd(family: str):
    source = _jax_path(family).read_text()

    assert "generate_sfd = is_fp8_dtype(ab_dtype)" in source
    assert '"has_amax": has_amax' in source
    assert '"amax_tensor"' in source


@pytest.mark.parametrize("family", ("dswiglu", "dglu", "dsrelu"))
def test_backward_inputs_validate_canonical_descriptor_shapes(family: str):
    tree = ast.parse(_jax_path(family).read_text())
    invalid_checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_array"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "c_desc"
    ]

    assert not invalid_checks
    assert "if c_desc.shape != expected_c_shape:" in _jax_path(family).read_text()


def test_grouped_packages_keep_torch_exports_lazy():
    root_source = (_GROUPED_ROOT / "__init__.py").read_text()
    assert "make_operation_api" in root_source
    assert ".api import" not in root_source

    for family in _FAMILIES:
        source = (_GROUPED_ROOT / f"grouped_gemm_{family}" / "__init__.py").read_text()
        assert "make_operation_api" in source
        assert 'submodules=("api", "jax")' in source
        assert ".api import" not in source


def test_grouped_package_import_does_not_load_torch_apis():
    package_name = "cudnn_frontend_grouped_lazy_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_ROOT / "python" / "cudnn")]
    package.__package__ = package_name

    with mock.patch.dict(sys.modules, {package_name: package}):
        grouped = importlib.import_module(f"{package_name}.grouped_gemm")
        expected = {
            f"GroupedGemm{_class_stem(family)}Sm100" for family in _FAMILIES
        } | {f"grouped_gemm_{family}_wrapper_sm100" for family in _FAMILIES}
        assert set(grouped.__all__) == expected
        assert not any(
            module_name.startswith(f"{package_name}.grouped_gemm.")
            for module_name in sys.modules
        )

        for family in _FAMILIES:
            subpackage_name = f"{package_name}.grouped_gemm.grouped_gemm_{family}"
            subpackage = importlib.import_module(subpackage_name)
            assert "jax" in dir(subpackage)
            assert f"{subpackage_name}.api" not in sys.modules

    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)


def test_shared_layout_helpers_map_row_major_arrays_to_kernel_axes():
    class DataType(Enum):
        NOT_SET = auto()

    class TensorSpec:
        def __init__(self, *, layout=None, mode=None, **kwargs):
            self.layout = layout
            self.mode = mode
            self.__dict__.update(kwargs)

    class ShapeDtypeStruct:
        def __init__(self, shape, dtype, fill_value=None):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.fill_value = fill_value

    fake_jax = types.ModuleType("jax")
    fake_jax.__path__ = []
    fake_jax.__spec__ = ModuleSpec("jax", loader=None, is_package=True)
    fake_jax.ShapeDtypeStruct = ShapeDtypeStruct
    fake_jax.local_devices = lambda **_kwargs: ()
    fake_jax.tree_util = types.SimpleNamespace(
        DictKey=lambda key: key,
        register_pytree_with_keys=lambda *_args: None,
    )

    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.dtype = lambda value: value
    for dtype_name in (
        "float4_e2m1fn",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e8m0fnu",
    ):
        setattr(fake_jnp, dtype_name, dtype_name)
    fake_jnp.empty = lambda shape, dtype: ShapeDtypeStruct(shape, dtype)
    fake_jnp.full = lambda shape, fill_value, dtype: ShapeDtypeStruct(
        shape, dtype, fill_value
    )
    fake_jax.numpy = fake_jnp

    fake_cutlass = types.ModuleType("cutlass")
    fake_cutlass.__path__ = []
    fake_cutlass.__spec__ = ModuleSpec("cutlass", loader=None, is_package=True)
    fake_cutlass_jax = types.ModuleType("cutlass.jax")
    fake_cutlass_jax.TensorSpec = TensorSpec
    fake_cutlass.jax = fake_cutlass_jax

    package_name = "cudnn_frontend_grouped_layout_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_ROOT / "python" / "cudnn")]
    package.__package__ = package_name
    package.data_type = DataType

    grouped_package = types.ModuleType(f"{package_name}.grouped_gemm")
    grouped_package.__path__ = [str(_GROUPED_ROOT)]
    grouped_package.__package__ = grouped_package.__name__

    modules = {
        "jax": fake_jax,
        "jax.numpy": fake_jnp,
        "cutlass": fake_cutlass,
        "cutlass.jax": fake_cutlass_jax,
        package_name: package,
        grouped_package.__name__: grouped_package,
    }
    with mock.patch.dict(sys.modules, modules):
        adapter = importlib.import_module(f"{package_name}.grouped_gemm._jax_api")

        assert adapter.gemm_a_mode("LMK") == (1, 2, 0)
        assert adapter.gemm_b_mode("LKN") == (2, 1, 0)
        assert adapter.PROBABILITY_MODE == (2, 1, 0)
        assert adapter.GROUPED_BIAS_MODE == (1, 0)
        assert adapter.block_scale_shape(256, 64, 4, 32) == (4, 2, 1, 32, 4, 4)

        sample = ShapeDtypeStruct((1, 256, 64), "float8")
        desc = adapter.ApiBaseJax().make_tensor_desc(
            sample,
            mode=adapter.gemm_a_mode("LMK"),
            name="sample_a",
        )
        assert type(desc) is adapter.JaxTensorDesc
        assert desc.shape == (256, 64, 1)
        assert desc.stride == (64, 1, 16384)
        assert desc.mode == (1, 2, 0)
        assert desc.array_shape == sample.shape

        fp4_sfa = ShapeDtypeStruct(
            adapter.block_scale_shape(256, 256, 1, 16),
            "float8_e4m3fn",
        )
        fp4_sfb = ShapeDtypeStruct(
            adapter.block_scale_shape(128, 256, 2, 16),
            "float8_e4m3fn",
        )
        assert (
            adapter.require_grouped_input_scales(
                fp4_sfa,
                fp4_sfb,
                m=256,
                n=128,
                k=256,
                experts=2,
                sf_vec_size=16,
                ab_dtype="float4_e2m1fn",
            )
            == "float8_e4m3fn"
        )

        invalid_fp4_sfa = ShapeDtypeStruct(
            adapter.block_scale_shape(256, 256, 1, 32),
            "float8_e4m3fn",
        )
        invalid_fp4_sfb = ShapeDtypeStruct(
            adapter.block_scale_shape(128, 256, 2, 32),
            "float8_e4m3fn",
        )
        with pytest.raises(ValueError, match="sf_vec_size=16"):
            adapter.require_grouped_input_scales(
                invalid_fp4_sfa,
                invalid_fp4_sfb,
                m=256,
                n=128,
                k=256,
                experts=2,
                sf_vec_size=32,
                ab_dtype="float4_e2m1fn",
            )

        mxfp4_sfa = ShapeDtypeStruct(
            adapter.block_scale_shape(256, 256, 1, 32),
            "float8_e8m0fnu",
        )
        mxfp4_sfb = ShapeDtypeStruct(
            adapter.block_scale_shape(128, 256, 2, 32),
            "float8_e8m0fnu",
        )
        assert (
            adapter.require_grouped_input_scales(
                mxfp4_sfa,
                mxfp4_sfb,
                m=256,
                n=128,
                k=256,
                experts=2,
                sf_vec_size=32,
                ab_dtype="float4_e2m1fn",
            )
            == "float8_e8m0fnu"
        )
        with pytest.raises(ValueError, match="Unsupported grouped GEMM input dtype"):
            adapter.require_grouped_input_scales(
                mxfp4_sfa,
                mxfp4_sfb,
                m=256,
                n=128,
                k=256,
                experts=2,
                sf_vec_size=32,
                ab_dtype="uint8",
            )

        fp8_sfa = ShapeDtypeStruct(
            adapter.block_scale_shape(256, 256, 1, 32),
            "float8_e8m0fnu",
        )
        fp8_sfb = ShapeDtypeStruct(
            adapter.block_scale_shape(128, 256, 2, 32),
            "float8_e8m0fnu",
        )
        assert (
            adapter.require_grouped_input_scales(
                fp8_sfa,
                fp8_sfb,
                m=256,
                n=128,
                k=256,
                experts=2,
                sf_vec_size=32,
                ab_dtype="float8_e4m3fn",
            )
            == "float8_e8m0fnu"
        )

        launched = []
        lowered = {}

        def launch(stream, input_value, output_value, workspace_value, **static):
            launched.append(
                (stream, input_value, output_value, workspace_value, static)
            )

        def fake_call_kernel(inputs, **options):
            lowered.update(inputs=inputs, **options)
            options["launch"]("stream", *inputs, "output", "workspace")
            return ("result",)

        input_desc = adapter.as_gemm_tensor_desc(
            "input",
            sample,
            mode=adapter.gemm_a_mode("LMK"),
        )
        output_mode = adapter.gemm_output_mode("LMN", name="output_layout")
        with (
            mock.patch.object(
                adapter._CALLER, "_get_max_active_clusters", return_value=7
            ) as occupancy,
            mock.patch.object(
                adapter._CALLER, "_resolve_compute_capability", return_value=100
            ),
            mock.patch.object(
                adapter._CALLER, "_call_kernel", side_effect=fake_call_kernel
            ),
        ):
            result = adapter.call_cutedsl(
                launch,
                (sample,),
                input_descs=(input_desc,),
                outputs=(
                    adapter.make_buffer_desc(
                        "output",
                        (1, 256, 128),
                        "bfloat16",
                        mode=output_mode,
                    ),
                ),
                workspaces=(
                    adapter.make_buffer_desc(
                        "workspace",
                        (4,),
                        "uint8",
                        ptr_assumed_align=adapter.GROUPED_WORKSPACE_ALIGNMENT,
                    ),
                ),
                static_args={
                    "cluster_shape_mn": (2, 1),
                    "cluster_overlap_margin": 1,
                    "expert_cnt": 4,
                },
            )

        assert result == ("result",)
        occupancy.assert_called_once_with(2, overlap_margin=1)
        assert launched == [
            (
                "stream",
                sample,
                "output",
                "workspace",
                {
                    "cluster_shape_mn": (2, 1),
                    "expert_cnt": 4,
                    "max_active_clusters": 7,
                },
            )
        ]
        assert lowered["output_descs"][0].shape == (256, 128, 1)
        assert type(lowered["output_descs"][0]) is adapter.JaxTensorDesc
        assert lowered["output_descs"][0].mode == output_mode
        assert lowered["input_descs"] == (input_desc,)
        assert lowered["workspace_descs"][0].shape == (4,)
        assert (
            lowered["workspace_descs"][0].ptr_assumed_align
            == adapter.GROUPED_WORKSPACE_ALIGNMENT
        )

        with pytest.raises(ValueError, match="Expected 1 input descriptors, got 0"):
            adapter.call_cutedsl(
                launch,
                (sample,),
                input_descs=(),
                outputs=(adapter.make_buffer_desc("output", (1, 0, 128), "bfloat16"),),
            )

        def materialize(desc):
            return ShapeDtypeStruct(
                desc.array_shape,
                desc.dtype,
            )

        with (
            mock.patch.object(
                adapter._CALLER,
                "_get_max_active_clusters",
                side_effect=AssertionError(
                    "empty grouped GEMM must not query occupancy"
                ),
            ),
            mock.patch.object(
                adapter._CALLER,
                "_call_kernel",
                side_effect=AssertionError("empty grouped GEMM must not lower"),
            ),
            mock.patch.object(
                adapter._CALLER,
                "_to_shape_dtype_struct",
                side_effect=materialize,
            ),
        ):
            empty, initialized = adapter.call_cutedsl(
                launch,
                (sample,),
                input_descs=(input_desc,),
                outputs=(
                    adapter.make_buffer_desc(
                        "empty",
                        (1, 0, 128),
                        "bfloat16",
                        mode=output_mode,
                    ),
                    adapter.make_buffer_desc(
                        "amax",
                        (4, 1),
                        "float32",
                        init_value=float("-inf"),
                    ),
                ),
                static_args={"cluster_shape_mn": (2, 1)},
            )

        assert empty.shape == (1, 0, 128)
        assert initialized.shape == (4, 1)
        assert initialized.fill_value == float("-inf")

    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)
