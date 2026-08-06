# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Rubin (sm107) architecture dispatch in grouped GEMM FE APIs.

These tests verify device gating and kernel-module selection without requiring
Rubin hardware. Full numerical coverage remains in the existing grouped GEMM
FE API tests, which should pass unchanged on sm107 when cutedsl is available.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("cutlass")

RUBIN_DISPATCH_CASES = [
    pytest.param(
        "cudnn.gemm.cutedsl.grouped.quant.api",
        "cudnn.gemm.cutedsl.grouped.quant.grouped_gemm_quant",
        "BlockScaledMoEGroupedGemmQuantKernel",
        "moe_blockscaled_grouped_gemm_quant_rubin.py",
        id="grouped_gemm_quant",
    ),
    pytest.param(
        "cudnn.gemm.cutedsl.grouped.glu.api",
        "cudnn.gemm.cutedsl.grouped.glu.moe_blockscaled_grouped_gemm_glu_bias",
        "BlockScaledMoEGroupedGemmGluBiasKernel",
        "moe_blockscaled_grouped_gemm_glu_rubin.py",
        id="grouped_gemm_glu",
    ),
    pytest.param(
        "cudnn.gemm.cutedsl.grouped.dglu.api",
        "cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias",
        "BlockScaledMoEGroupedGemmDgluDbiasKernel",
        "moe_blockscaled_grouped_gemm_dglu_rubin.py",
        id="grouped_gemm_dglu",
    ),
    pytest.param(
        "cudnn.gemm.cutedsl.grouped.wgrad.api",
        "cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad",
        "BlockScaledMoEGroupedGemmWgradKernel",
        "moe_blockscaled_grouped_gemm_wgrad_rubin.py",
        id="grouped_gemm_wgrad",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GROUPED_GEMM_ROOT = _REPO_ROOT / "python" / "cudnn" / "gemm" / "cutedsl" / "grouped"


def _import_api_module(module_path: str):
    return importlib.import_module(module_path)


@pytest.mark.parametrize(
    "api_module_path,default_module_path,default_kernel_name,rubin_filename",
    RUBIN_DISPATCH_CASES,
)
def test_rubin_kernel_module_is_present(
    api_module_path,
    default_module_path,
    default_kernel_name,
    rubin_filename,
):
    family = api_module_path.split(".")[-2]
    rubin_path = _GROUPED_GEMM_ROOT / family / rubin_filename
    assert rubin_path.is_file(), f"Missing Rubin kernel module: {rubin_path}"


@pytest.mark.parametrize(
    "api_module_path,default_module_path,default_kernel_name,rubin_filename",
    RUBIN_DISPATCH_CASES,
)
def test_is_sm107_device_gating(
    api_module_path,
    default_module_path,
    default_kernel_name,
    rubin_filename,
):
    import cudnn.api_base as api_base

    with mock.patch("torch.cuda.is_available", return_value=False):
        assert api_base.is_sm107_device() is False
        assert api_base.get_device_type() == "blackwell"

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch(
            "torch.cuda.get_device_capability",
            return_value=(10, 0),
        ),
    ):
        assert api_base.is_sm107_device() is False
        assert api_base.get_device_type() == "blackwell"

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch(
            "torch.cuda.get_device_capability",
            return_value=(10, 7),
        ),
    ):
        assert api_base.is_sm107_device() is True
        assert api_base.get_device_type() == "rubin"


@pytest.mark.parametrize(
    "api_module_path,default_module_path,default_kernel_name,rubin_filename",
    RUBIN_DISPATCH_CASES,
)
def test_get_rubin_kernel_lazy_import(
    api_module_path,
    default_module_path,
    default_kernel_name,
    rubin_filename,
):
    api_mod = _import_api_module(api_module_path)
    default_mod = importlib.import_module(default_module_path)
    default_kernel = getattr(default_mod, default_kernel_name)

    rubin_kernel = api_mod._get_rubin_kernel()

    assert rubin_kernel is not default_kernel
    assert rubin_kernel.__name__ != default_kernel.__name__ or rubin_kernel.__module__ != default_kernel.__module__
    if hasattr(default_kernel, "FIX_PAD_SIZE"):
        assert rubin_kernel.FIX_PAD_SIZE == default_kernel.FIX_PAD_SIZE == 256
    else:
        assert rubin_kernel.FIX_PAD_SIZE == 256


@pytest.mark.parametrize(
    "api_module_path,default_module_path,default_kernel_name,rubin_filename",
    RUBIN_DISPATCH_CASES,
)
def test_kernel_selection_uses_rubin_on_sm107(
    api_module_path,
    default_module_path,
    default_kernel_name,
    rubin_filename,
):
    api_mod = _import_api_module(api_module_path)
    default_mod = importlib.import_module(default_module_path)
    default_kernel = getattr(default_mod, default_kernel_name)
    rubin_kernel = api_mod._get_rubin_kernel()
    import cudnn.api_base as api_base

    with (
        mock.patch("cudnn.api_base.is_sm107_device", return_value=True),
        mock.patch.object(
            api_mod,
            "_get_rubin_kernel",
            return_value=rubin_kernel,
        ),
    ):
        selected = api_mod._get_rubin_kernel() if api_base.is_sm107_device() else default_kernel

    assert selected is rubin_kernel

    with mock.patch("cudnn.api_base.is_sm107_device", return_value=False):
        selected = api_mod._get_rubin_kernel() if api_base.is_sm107_device() else default_kernel

    assert selected is default_kernel


@pytest.mark.L0
def test_grouped_gemm_quant_has_rubin_compile_branches():
    """Quant adapts compile/execute kwargs on Rubin; keep this contract covered."""
    api_mod = _import_api_module("cudnn.gemm.cutedsl.grouped.quant.api")
    source = Path(api_mod.__file__).read_text(encoding="utf-8")

    assert "self._is_rubin_kernel" in source
    assert 'kernel_kwargs["generate_c"] = False' in source
    assert 'compile_kwargs["c"]' in source
    assert "if self._is_rubin_kernel:" in source


@pytest.mark.L0
@pytest.mark.parametrize(
    "kernel_path",
    [
        "grouped_gemm_quant/grouped_gemm_quant.py",
        "grouped_gemm_quant/moe_blockscaled_grouped_gemm_quant_rubin.py",
    ],
)
def test_grouped_gemm_quant_kernels_support_optional_prob(kernel_path):
    """Blackwell and Rubin quant kernels compile out an omitted probability."""
    source = (_GROUPED_GEMM_ROOT / kernel_path).read_text(encoding="utf-8")

    assert "self.has_prob = prob is not None" in source
    assert "if cutlass.const_expr(self.has_prob):" in source


@pytest.mark.L0
@pytest.mark.parametrize(
    "ab_dtype_name,sf_dtype_name,sf_vec_size,expected",
    [
        ("float4_e2m1fn_x2", "float8_e4m3fn", 16, True),
        ("float4_e2m1fn_x2", "float8_e8m0fnu", 32, True),
        ("float8_e4m3fn", "float8_e8m0fnu", 32, True),
        ("float8_e5m2", "float8_e8m0fnu", 32, True),
        ("float4_e2m1fn_x2", "float8_e4m3fn", 32, False),
        ("float8_e4m3fn", "float8_e4m3fn", 16, False),
        ("bfloat16", "float8_e8m0fnu", 32, False),
    ],
)
def test_grouped_gemm_wgrad_rubin_quantization_validation(
    ab_dtype_name,
    sf_dtype_name,
    sf_vec_size,
    expected,
):
    api_mod = _import_api_module("cudnn.gemm.cutedsl.grouped.wgrad.api")
    torch = api_mod.torch

    assert (
        api_mod._is_supported_rubin_quantization(
            getattr(torch, ab_dtype_name),
            getattr(torch, sf_dtype_name),
            sf_vec_size,
        )
        is expected
    )


@pytest.mark.L0
def test_grouped_gemm_wgrad_rubin_tmem_plan_rejects_invalid_sf_vector():
    pytest.importorskip(
        "cutlass.utils.rubin_helpers",
        reason="Rubin helpers are unavailable in this CUTLASS DSL wheel",
    )
    rubin_mod = importlib.import_module("cudnn.gemm.cutedsl.grouped.wgrad.moe_blockscaled_grouped_gemm_wgrad_rubin")

    with pytest.raises(ValueError, match="divisible by sf_vec_size"):
        rubin_mod._make_tmem_plan(
            tile_n=256,
            tile_k=256,
            sf_vec_size=24,
            architecture="sm_107",
        )
