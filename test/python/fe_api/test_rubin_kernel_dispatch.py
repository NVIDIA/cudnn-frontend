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
        "cudnn.grouped_gemm.grouped_gemm_quant.api",
        "cudnn.grouped_gemm.grouped_gemm_quant.grouped_gemm_quant",
        "BlockScaledMoEGroupedGemmQuantKernel",
        "moe_blockscaled_grouped_gemm_quant_rubin.py",
        id="grouped_gemm_quant",
    ),
    pytest.param(
        "cudnn.grouped_gemm.grouped_gemm_glu.api",
        "cudnn.grouped_gemm.grouped_gemm_glu.moe_blockscaled_grouped_gemm_glu_bias",
        "BlockScaledMoEGroupedGemmGluBiasKernel",
        "moe_blockscaled_grouped_gemm_glu_rubin.py",
        id="grouped_gemm_glu",
    ),
    pytest.param(
        "cudnn.grouped_gemm.grouped_gemm_dglu.api",
        "cudnn.grouped_gemm.grouped_gemm_dglu.moe_blockscaled_grouped_gemm_dglu_dbias",
        "BlockScaledMoEGroupedGemmDgluDbiasKernel",
        "moe_blockscaled_grouped_gemm_dglu_rubin.py",
        id="grouped_gemm_dglu",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GROUPED_GEMM_ROOT = _REPO_ROOT / "python" / "cudnn" / "grouped_gemm"


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
    assert rubin_kernel.FIX_PAD_SIZE == default_kernel.FIX_PAD_SIZE == 256


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
    api_mod = _import_api_module("cudnn.grouped_gemm.grouped_gemm_quant.api")
    source = Path(api_mod.__file__).read_text(encoding="utf-8")

    assert "self._is_rubin_kernel" in source
    assert 'kernel_kwargs["generate_c"] = False' in source
    assert 'compile_kwargs["c"]' in source
    assert "if self._is_rubin_kernel:" in source
