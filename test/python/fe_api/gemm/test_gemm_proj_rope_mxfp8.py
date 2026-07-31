# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest

from test_utils import torch_fork_set_rng
from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import with_gemm_proj_rope_mxfp8_params


# ======================================================================================
# BF16-input sibling: class compile/execute + wrapper
# ======================================================================================
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_proj_rope_mxfp8_params
def test_gemm_proj_rope_mxfp8_bf16in_compile_execute(tokens, w_out_in, request):
    _test_bf16in_compile_execute(tokens=tokens, w_out_in=w_out_in, request=request)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_gemm_proj_rope_mxfp8_params
def test_gemm_proj_rope_mxfp8_bf16in_wrapper(tokens, w_out_in, request):
    _test_bf16in_wrapper(tokens=tokens, w_out_in=w_out_in, request=request)


# ======================================================================================
# MXFP8-input sibling: class compile/execute + wrapper
# ======================================================================================
# Keep the small case at L0 (fast); the large 4096-token case compiles a fresh SM100 kernel, so
# park it at L1 to keep L0 quick.
_MXFP8_TOKEN_PARAMS = [pytest.param(2048, marks=pytest.mark.L0), pytest.param(4096, marks=pytest.mark.L1)]


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("tokens", _MXFP8_TOKEN_PARAMS)
def test_gemm_proj_rope_mxfp8_mxfp8in_compile_execute(tokens, request):
    _test_mxfp8in_compile_execute(tokens=tokens, request=request)


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("tokens", _MXFP8_TOKEN_PARAMS)
def test_gemm_proj_rope_mxfp8_mxfp8in_wrapper(tokens, request):
    _test_mxfp8in_wrapper(tokens=tokens, request=request)


# ======================================================================================
# check_support() failure cases
# ======================================================================================
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bf16in_check_support_rejects_unaligned_tokens(request):
    """bf16in check_support() must reject a token count that is not a multiple of TILE_M."""
    try:
        from cudnn import GemmProjRopeMxfp8Bf16InSm100
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    gemm_proj_rope_mxfp8_init(request, tokens=2048, w_out_in=False)  # arch skip if needed
    tokens = 2048 + 32  # not a multiple of TILE_M (128)
    x, w, cos, sin = allocate_input_tensors(tokens, w_out_in=False)
    outs = allocate_output_tensors(tokens - 32)
    obj = GemmProjRopeMxfp8Bf16InSm100(
        sample_x=x,
        sample_w=w,
        sample_cos=cos,
        sample_sin=sin,
        sample_out_fp8_row=outs[0],
        sample_out_scales_row=outs[1],
        sample_out_fp8_col=outs[2],
        sample_out_scales_col=outs[3],
        w_out_in=False,
    )
    with pytest.raises(ValueError):
        obj.check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_bf16in_check_support_rejects_kdim_mismatch(request):
    """bf16in check_support() must reject x whose contraction dim does not match the weight's."""
    try:
        from cudnn import GemmProjRopeMxfp8Bf16InSm100
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            Q_LORA,
            allocate_input_tensors,
            allocate_output_tensors,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    gemm_proj_rope_mxfp8_init(request, tokens=2048, w_out_in=False)  # arch skip if needed
    tokens = 2048
    x, w, cos, sin = allocate_input_tensors(tokens, w_out_in=False)
    x = torch.randn(tokens, Q_LORA + 8, dtype=x.dtype, device=x.device)  # inner dim != w's K
    outs = allocate_output_tensors(tokens)
    obj = GemmProjRopeMxfp8Bf16InSm100(
        sample_x=x,
        sample_w=w,
        sample_cos=cos,
        sample_sin=sin,
        sample_out_fp8_row=outs[0],
        sample_out_scales_row=outs[1],
        sample_out_fp8_col=outs[2],
        sample_out_scales_col=outs[3],
        w_out_in=False,
    )
    with pytest.raises(ValueError):
        obj.check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_mxfp8in_check_support_rejects_bf16_code(request):
    """mxfp8in check_support() must reject a bf16 (non-fp8) x_code."""
    try:
        from cudnn import GemmProjRopeMxfp8Mxfp8InSm100
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            allocate_mxfp8_input_tensors,
            allocate_output_tensors,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    gemm_proj_rope_mxfp8_init(request, tokens=2048, w_out_in=True)  # arch skip if needed
    tokens = 2048
    x_code, x_scale, w_code, w_scale, cos, sin, _, _ = allocate_mxfp8_input_tensors(tokens)
    x_code = x_code.to(torch.bfloat16)  # wrong dtype for the MXFP8-input contract
    outs = allocate_output_tensors(tokens)
    obj = GemmProjRopeMxfp8Mxfp8InSm100(
        sample_x_code=x_code,
        sample_x_scale=x_scale,
        sample_w_code=w_code,
        sample_w_scale=w_scale,
        sample_cos=cos,
        sample_sin=sin,
        sample_out_fp8_row=outs[0],
        sample_out_scales_row=outs[1],
        sample_out_fp8_col=outs[2],
        sample_out_scales_col=outs[3],
    )
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        obj.check_support()


@pytest.mark.L0
def test_wrapper_rejects_dtype_mismatch_and_missing_scales():
    """The dispatch wrapper's dtype-agreement and scale-presence contracts (no SM100 needed --
    these asserts fire before any device work)."""
    try:
        from cudnn import gemm_proj_rope_mxfp8_wrapper_sm100
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import QK_ROPE, Q_LORA, Q_OUT
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    tokens = 128
    cos = torch.empty(tokens, QK_ROPE, dtype=torch.bfloat16)
    sin = torch.empty(tokens, QK_ROPE, dtype=torch.bfloat16)

    # x and w disagree on dtype
    x_bf16 = torch.empty(tokens, Q_LORA, dtype=torch.bfloat16)
    w_fp8 = torch.empty(Q_OUT, Q_LORA, dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="must share a dtype"):
        gemm_proj_rope_mxfp8_wrapper_sm100(x_bf16, w_fp8, cos, sin, w_out_in=True)

    # fp8 inputs without scales
    x_fp8 = torch.empty(tokens, Q_LORA, dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="require x_scale and w_scale"):
        gemm_proj_rope_mxfp8_wrapper_sm100(x_fp8, w_fp8, cos, sin, w_out_in=True)

    # bf16 inputs given scales
    w_bf16 = torch.empty(Q_OUT, Q_LORA, dtype=torch.bfloat16)
    bogus_scale = torch.empty(tokens, Q_LORA // 32, dtype=torch.uint8)
    with pytest.raises(AssertionError, match="must not be given MXFP8 scales"):
        gemm_proj_rope_mxfp8_wrapper_sm100(x_bf16, w_bf16, cos, sin, x_scale=bogus_scale, w_scale=bogus_scale, w_out_in=True)


# ======================================================================================
# PyTorch reference oracle (no SM100 kernel required)
# ======================================================================================
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("tokens", [128, 256])
@pytest.mark.parametrize("w_out_in", [False, True])
def test_gemm_proj_rope_mxfp8_reference_contract(tokens, w_out_in):
    """The reference oracle's four outputs must have the documented shapes/dtypes and be finite."""
    try:
        from cudnn.gemm.cutedsl.dense.proj_rope_mxfp8 import gemm_proj_rope_mxfp8_reference
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            BLOCK,
            HEAD_DIM,
            NUM_HEADS,
            QK_ROPE,
            Q_LORA,
            Q_OUT,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(tokens, Q_LORA, dtype=torch.bfloat16, device=dev) * 0.5
    w_shape = (Q_OUT, Q_LORA) if w_out_in else (Q_LORA, Q_OUT)
    w = torch.randn(*w_shape, dtype=torch.bfloat16, device=dev) * 0.02
    cos = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)
    sin = torch.randn(tokens, QK_ROPE, dtype=torch.bfloat16, device=dev)

    qr, sr, qc, sc = gemm_proj_rope_mxfp8_reference(x, w, cos, sin, w_out_in=w_out_in)

    assert qr.shape == (tokens, NUM_HEADS, HEAD_DIM)
    assert sr.shape == (tokens, NUM_HEADS, HEAD_DIM // BLOCK)
    assert qc.shape == (tokens, NUM_HEADS, HEAD_DIM)
    assert sc.shape == (tokens // BLOCK, NUM_HEADS, HEAD_DIM)
    assert qr.dtype == torch.float8_e4m3fn and qc.dtype == torch.float8_e4m3fn
    assert sr.dtype == torch.uint8 and sc.dtype == torch.uint8
    assert torch.isfinite(qr.float()).all() and torch.isfinite(qc.float()).all()


# ======================================================================================
# implementations
# ======================================================================================
def _test_bf16in_compile_execute(tokens, w_out_in, request):
    try:
        from cudnn import GemmProjRopeMxfp8Bf16InSm100
        from cuda.bindings import driver as cuda
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_proj_rope_mxfp8,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = gemm_proj_rope_mxfp8_init(request, tokens, w_out_in)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    x, w, cos, sin = allocate_input_tensors(cfg["tokens"], cfg["w_out_in"])
    outputs = allocate_output_tensors(cfg["tokens"])

    gemm = GemmProjRopeMxfp8Bf16InSm100(
        sample_x=x,
        sample_w=w,
        sample_cos=cos,
        sample_sin=sin,
        sample_out_fp8_row=outputs[0],
        sample_out_scales_row=outputs[1],
        sample_out_fp8_col=outputs[2],
        sample_out_scales_col=outputs[3],
        w_out_in=cfg["w_out_in"],
    )
    try:
        assert gemm.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm.compile()
    gemm.execute(x, w, cos, sin, *outputs, current_stream=stream)
    check_ref_gemm_proj_rope_mxfp8(x, w, cos, sin, outputs, cfg["w_out_in"], skip_ref=cfg["skip_ref"])


def _test_bf16in_wrapper(tokens, w_out_in, request):
    try:
        from cudnn import gemm_proj_rope_mxfp8_wrapper_sm100
        from cuda.bindings import driver as cuda
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            allocate_input_tensors,
            check_ref_gemm_proj_rope_mxfp8,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = gemm_proj_rope_mxfp8_init(request, tokens, w_out_in)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    x, w, cos, sin = allocate_input_tensors(cfg["tokens"], cfg["w_out_in"])

    try:
        for _ in range(2):  # run twice to exercise the object cache
            out = gemm_proj_rope_mxfp8_wrapper_sm100(x, w, cos, sin, w_out_in=cfg["w_out_in"], stream=stream)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    outputs = (out["out_fp8_row"], out["out_scales_row"], out["out_fp8_col"], out["out_scales_col"])
    check_ref_gemm_proj_rope_mxfp8(x, w, cos, sin, outputs, cfg["w_out_in"], skip_ref=cfg["skip_ref"])


def _test_mxfp8in_compile_execute(tokens, request):
    try:
        from cudnn import GemmProjRopeMxfp8Mxfp8InSm100
        from cuda.bindings import driver as cuda
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            allocate_mxfp8_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_proj_rope_mxfp8,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = gemm_proj_rope_mxfp8_init(request, tokens, w_out_in=True)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    x_code, x_scale, w_code, w_scale, cos, sin, x_deq, w_deq = allocate_mxfp8_input_tensors(cfg["tokens"])
    outputs = allocate_output_tensors(cfg["tokens"])

    gemm = GemmProjRopeMxfp8Mxfp8InSm100(
        sample_x_code=x_code,
        sample_x_scale=x_scale,
        sample_w_code=w_code,
        sample_w_scale=w_scale,
        sample_cos=cos,
        sample_sin=sin,
        sample_out_fp8_row=outputs[0],
        sample_out_scales_row=outputs[1],
        sample_out_fp8_col=outputs[2],
        sample_out_scales_col=outputs[3],
    )
    try:
        assert gemm.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")
    gemm.compile()
    gemm.execute(x_code, x_scale, w_code, w_scale, cos, sin, *outputs, current_stream=stream)
    # reference from the dequantized fp8 operands (the fp8 GEMM ~ bf16 GEMM of those values)
    check_ref_gemm_proj_rope_mxfp8(x_deq, w_deq, cos, sin, outputs, w_out_in=True, skip_ref=cfg["skip_ref"])


def _test_mxfp8in_wrapper(tokens, request):
    try:
        from cudnn import gemm_proj_rope_mxfp8_wrapper_sm100
        from cuda.bindings import driver as cuda
        from fe_api.gemm.test_gemm_proj_rope_mxfp8_utils import (
            allocate_mxfp8_input_tensors,
            check_ref_gemm_proj_rope_mxfp8,
            gemm_proj_rope_mxfp8_init,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = gemm_proj_rope_mxfp8_init(request, tokens, w_out_in=True)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    x_code, x_scale, w_code, w_scale, cos, sin, x_deq, w_deq = allocate_mxfp8_input_tensors(cfg["tokens"])

    try:
        for _ in range(2):  # run twice to exercise the object cache
            out = gemm_proj_rope_mxfp8_wrapper_sm100(x_code, w_code, cos, sin, x_scale=x_scale, w_scale=w_scale, w_out_in=True, stream=stream)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    outputs = (out["out_fp8_row"], out["out_scales_row"], out["out_fp8_col"], out["out_scales_col"])
    check_ref_gemm_proj_rope_mxfp8(x_deq, w_deq, cos, sin, outputs, w_out_in=True, skip_ref=cfg["skip_ref"])
