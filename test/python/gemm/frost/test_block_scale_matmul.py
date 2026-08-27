# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-scaled matmul (FP4/FP8 + per-block scale factors): analyzer
pattern-match unit tests (no GPU) + end-to-end numerics vs a torch reference.

Covers both template families: sm100 (K=32B MMA) and sm103 (fp4 K=48B UTCOMMA,
K-tile 384 B) — the sm103 sections are at the end of the file."""

from __future__ import annotations

import re

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs recorder)
import dataclasses

import pytest
import torch

from gemm_test_utils import (
    _SM,
    requires_sm100,
    requires_sm107,
    Plan as _plan,
    vp_bs as _vp_bs,
    kw as _kw,
    E2M1 as _E2M1,
    ceil_div as _ceil_div,
    to_blocked as _to_blocked,
    unpack_fp4 as _unpack_fp4,
    rand_e8m0 as _rand_e8m0,
    rand_e5m3 as _rand_e5m3,
    e5m3_to_float as _e5m3_to_float,
    block_quant_ref as _block_quant_ref,
    reduction_ref as _reduction_ref,
    assert_block_scale_reduction_close as _assert_block_scale_reduction_close,
)

from cudnn.gemm.frost import compiler as C
from cudnn.gemm.frost.dtypes import DTYPE_FROM_CUDNN as _DTYPE_FROM_CUDNN
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import GraphType, TEMPLATES, select_template
from cudnn.gemm.frost.tile_config import (
    CATALOG,
    ConfigSm100,
    ConfigSm103,
    ConfigSm107,
    TileConfig,
    by_name,
    validate_block_scale_config,
)

pytestmark = pytest.mark.L0


# Helpers


def _build_nvfp4_graph(
    M,
    N,
    K,
    block_size=16,
    sf_dt=cudnn.data_type.FP8_E4M3,
    a_dt=cudnn.data_type.FP4_E2M1,
    b_dt=None,
    a_major="k",
    b_major="k",
    reorder=True,
    out_major="n",
):
    sf_k = K // block_size
    b_dt = b_dt if b_dt is not None else a_dt
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # A: K-major (stride[-1]=1) or M-major (stride[-2]=1).
    a_stride = [M * K, K, 1] if a_major == "k" else [M * K, 1, M]
    # B logical (K, N): K-major (stride[-2]=1) or N-major.
    b_stride = [K * N, 1, K] if b_major == "k" else [K * N, N, 1]
    A = g.tensor(name="A", dim=[1, M, K], stride=a_stride, data_type=a_dt)
    B = g.tensor(name="B", dim=[1, K, N], stride=b_stride, data_type=b_dt)
    sf_kw = dict(reordering_type=cudnn.tensor_reordering.F8_128x4) if reorder else {}
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        **sf_kw,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        **sf_kw,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])
    C.set_output(True).set_data_type(cudnn.data_type.HALF)
    return g


def _build_block_scale_reduction_graph(
    M,
    N,
    K,
    mode,
    red_dims,
    block_size=16,
    sf_dt=cudnn.data_type.FP8_E4M3,
    a_dt=cudnn.data_type.FP4_E2M1,
    red_stride=None,
    red_dtype=cudnn.data_type.FLOAT,
    red_compute_dtype=None,
):
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=a_dt,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=a_dt,
    )
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    red_kwargs = {}
    if red_compute_dtype is not None:
        red_kwargs["compute_data_type"] = red_compute_dtype
    R = g.reduction(input=C, mode=mode, name="red", **red_kwargs)
    if red_stride is None:
        red_stride = [red_dims[1] * red_dims[2], red_dims[2], 1]
    R.set_dim(red_dims).set_stride(red_stride)
    R.set_output(True).set_data_type(red_dtype)
    return g


def _build_block_scale_quant_graph(
    M,
    N,
    K,
    dequant_block_size=16,
    quant_block_size=32,
    sf_dt=cudnn.data_type.FP8_E8M0,
    a_dt=cudnn.data_type.FP8_E4M3,
    out_dt=cudnn.data_type.FP8_E4M3,
    scale_dt=cudnn.data_type.FP8_E8M0,
    scale_reorder=False,
    scale_dim=None,
    global_scale=False,
):
    sf_k = K // dequant_block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=a_dt,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=a_dt,
    )
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, dequant_block_size])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[dequant_block_size, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    if global_scale:
        scale = g.tensor(
            name="global_scale",
            dim=[1, 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        C = g.mul(a=C, b=scale, name="global_scale_mul")
    Q, QS = g.block_scale_quantize(input=C, block_size=quant_block_size, name="q")
    Q.set_output(True).set_data_type(out_dt)
    if scale_dim is not None:
        QS.set_dim(list(scale_dim)).set_stride([scale_dim[1] * scale_dim[2], scale_dim[2], 1])
    QS.set_output(True).set_data_type(scale_dt)
    if scale_reorder:
        QS.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
    return g


# Compile-stage support gate: sm100_block_scale_matmul exact per-side cases

_DT_FP4, _DT_E4M3, _DT_E5M2, _DT_E8M0 = (
    cudnn.data_type.FP4_E2M1,
    cudnn.data_type.FP8_E4M3,
    cudnn.data_type.FP8_E5M2,
    cudnn.data_type.FP8_E8M0,
)
# (a_dt, sf_dt, b_dt, block_size) for the 6 supported cases.
_SUPPORTED_BS_CASES = [
    (_DT_FP4, _DT_E4M3, _DT_FP4, 16),  # 1 nvfp4
    (_DT_FP4, _DT_E8M0, _DT_FP4, 32),  # 2 mxfp4
    (_DT_E4M3, _DT_E8M0, _DT_E4M3, 32),  # 3 mxfp8 e4m3×e4m3
    (_DT_E4M3, _DT_E8M0, _DT_E5M2, 32),  # 4 mxfp8 e4m3×e5m2
    (_DT_E5M2, _DT_E8M0, _DT_E4M3, 32),  # 5 mxfp8 e5m2×e4m3
    (_DT_E5M2, _DT_E8M0, _DT_E5M2, 32),  # 6 mxfp8 e5m2×e5m2
]


@pytest.mark.parametrize("a_dt,sf_dt,b_dt,bs", _SUPPORTED_BS_CASES)
def test_block_scale_matmul_gate_accepts_supported(a_dt, sf_dt, b_dt, bs):
    from cudnn.gemm.frost.compiler import _check_block_scale_supported

    chain = analyze(_build_nvfp4_graph(256, 256, 512, block_size=bs, sf_dt=sf_dt, a_dt=a_dt, b_dt=b_dt))
    _check_block_scale_supported(chain, "sm100")


def test_block_scale_matmul_gate_rejects_mismatches():
    from cudnn.gemm.frost.compiler import _check_block_scale_supported

    # Missing F8_128x4 SF reorder layout.
    with pytest.raises(NotImplementedError, match="does not support"):
        _check_block_scale_supported(analyze(_build_nvfp4_graph(256, 256, 512, block_size=16, sf_dt=_DT_E4M3, reorder=False)), "sm100")
    # FP8 data at block 16 — the fp8 rows are block-32 only, on every pipeline.
    with pytest.raises(NotImplementedError, match="does not support"):
        _check_block_scale_supported(analyze(_build_nvfp4_graph(256, 256, 512, block_size=16, sf_dt=_DT_E8M0, a_dt=_DT_E4M3)), "sm100")
    # mixed FP4 A / FP8 B (cross-family) — unsupported.
    with pytest.raises(NotImplementedError, match="does not support"):
        _check_block_scale_supported(
            analyze(
                _build_nvfp4_graph(
                    256,
                    256,
                    512,
                    block_size=32,
                    sf_dt=_DT_E8M0,
                    a_dt=_DT_FP4,
                    b_dt=_DT_E4M3,
                )
            ),
            "sm100",
        )


def test_block_scale_matmul_gate_rejects_wrong_arch(monkeypatch):
    """The nvfp4 COMBO is arch-free (it exists on the whole sm100 family and
    is not an MMA_GPU_ARCH_SPECIAL_CASES entry); a wrong-family GPU is
    rejected by the template's PIPELINE_ARCH_RANGES gate in the jit path,
    before any compile."""
    import cudnn.gemm.frost.compiler as compiler

    g = _build_nvfp4_graph(256, 256, 512, block_size=16)
    monkeypatch.setattr(compiler, "_current_arch", lambda: 90)
    with pytest.raises(NotImplementedError, match="100 <= SM < 120.*sm_90"):
        jit_from_cudnn_graph(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))


# Analyzer pattern matching (no GPU)


def test_analyze_detects_nvfp4_block_scale_matmul():
    chain = analyze(_build_nvfp4_graph(128, 256, 256, block_size=16))
    assert chain.has_block_scale
    bs = chain.block_scale
    assert bs.a_dtype == "fp4_e2m1"
    assert bs.block_size == 16
    assert bs.sf_dtype == "fp8_e4m3"
    assert bs.mma_block_scale_kind == "MXF4NVF4"
    assert bs.scale_vec_size == "BLOCK16"
    assert bs.sf_scale_format == 0
    # Operands redirected to the packed FP4 data tensors.
    assert chain.matmul.a_dtype == "fp4_e2m1"
    assert chain.matmul.b_dtype == "fp4_e2m1"
    assert chain.matmul.M == 128 and chain.matmul.N == 256 and chain.matmul.K == 256
    # Per-side info: SF tensors are runtime-positional, not stored as TensorRefs.
    assert bs.both_sided
    assert bs.block_size_a == (1, 16) and bs.block_size_b == (16, 1)
    assert bs.sf_dtype_a == "fp8_e4m3" and bs.sf_dtype_b == "fp8_e4m3"


def test_analyze_detects_mxfp8_block_scale_matmul():
    chain = analyze(
        _build_nvfp4_graph(
            128,
            256,
            256,
            block_size=32,
            sf_dt=cudnn.data_type.FP8_E8M0,
            a_dt=cudnn.data_type.FP8_E4M3,
        )
    )
    bs = chain.block_scale
    assert bs.a_dtype == "fp8_e4m3"
    assert bs.block_size == 32 and bs.sf_dtype == "fp8_e8m0"
    assert bs.mma_block_scale_kind == "MXF8F6F4"
    assert bs.scale_vec_size == "BLOCK32"
    assert bs.sf_scale_format == 1


def test_analyze_detects_mxfp4_block_scale_matmul():
    chain = analyze(_build_nvfp4_graph(128, 256, 256, block_size=32, sf_dt=cudnn.data_type.FP8_E8M0))
    bs = chain.block_scale
    assert bs.a_dtype == "fp4_e2m1"
    assert bs.block_size == 32 and bs.sf_dtype == "fp8_e8m0"
    assert bs.mma_block_scale_kind == "MXF4NVF4"


def test_plain_matmul_has_no_block_scale_matmul():
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, 128, 128], stride=[128 * 128, 128, 1])
    B = g.tensor(name="B", dim=[1, 128, 128], stride=[128 * 128, 1, 128])
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    assert not analyze(g).has_block_scale


# End-to-end numerics (GPU)

_GPU = requires_sm100


def _make_block_scale_inputs(combo, M, N, K, dev="cuda"):
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
        b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
        a_rt = a_u8.view(torch.float4_e2m1fn_x2)
        b_rt = b_u8.view(torch.float4_e2m1fn_x2)
        a_deq = _unpack_fp4(a_u8, lut).view(M, K)
        b_deq = _unpack_fp4(b_u8, lut).view(N, K)
    else:
        a_rt = (torch.randn(1, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_rt = (torch.randn(1, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_deq = a_rt.float().view(M, K)
        b_deq = b_rt.float().view(N, K)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((M, sf_k), dev)
        sfb_log = _rand_e8m0((N, sf_k), dev)

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = a_s @ b_s.t()
    return a_rt, b_rt, sfa_log, sfb_log, ref, bs, sf_dt, a_dt


def _run_bs_numeric(combo, config_name, M, N, K, out_major="n"):
    """Block-scale matmul vs a torch dequant-matmul reference."""
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0

    if is_fp4:
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
        b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
        a_rt = a_u8.view(torch.float4_e2m1fn_x2)
        b_rt = b_u8.view(torch.float4_e2m1fn_x2)
        a_deq = _unpack_fp4(a_u8, lut).view(M, K)
        b_deq = _unpack_fp4(b_u8, lut).view(N, K)
    else:
        a_rt = (torch.randn(1, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_rt = (torch.randn(1, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_deq = a_rt.float().view(M, K)
        b_deq = b_rt.float().view(N, K)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((M, sf_k), dev)
        sfb_log = _rand_e8m0((N, sf_k), dev)

    g = _build_nvfp4_graph(M, N, K, block_size=bs, sf_dt=sf_dt, a_dt=a_dt, out_major=out_major)
    compiled = _plan(g, **_kw(config_name))
    assert compiled.block_scale
    assert (compiled.chain.block_scale.sf_dtype, compiled.chain.block_scale.block_size) == (_DTYPE_FROM_CUDNN[sf_dt], bs)

    if out_major == "m":
        c = torch.zeros(1, N, M, dtype=torch.float16, device=dev).transpose(1, 2)
    else:
        c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            c,
            _to_blocked(sfa_log).view(1, M, sf_k),
            _to_blocked(sfb_log).view(1, N, sf_k),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_s @ b_s.t()).to(torch.float16)
    # nvfp4 is bit-exact; mx paths carry fp16 rounding.
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


def _run_bs_nonpacked_numeric(combo, config_name, M, N, K, mode):
    dev = "cuda"
    torch.manual_seed(0)
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0
    c_pad = 16

    if is_fp4:
        assert mode == "padded"
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
        pad = 16
        a_storage = torch.randint(0, 256, (1, M, K // 2 + pad), dtype=torch.uint8, device=dev)
        b_storage = torch.randint(0, 256, (1, N, K // 2 + pad), dtype=torch.uint8, device=dev)
        a_u8 = a_storage[:, :, : K // 2]
        b_u8 = b_storage[:, :, : K // 2]
        a_rt = a_u8.view(torch.float4_e2m1fn_x2)
        b_rt = b_u8.view(torch.float4_e2m1fn_x2)
        a_deq = _unpack_fp4(a_u8, lut).view(M, K)
        b_deq = _unpack_fp4(b_u8, lut).view(N, K)
    elif mode == "zero_stride":
        a_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_base = (torch.randn(K, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_rt = torch.as_strided(a_base, (1, M, K), (0, 0, 1))
        b_rt = torch.as_strided(b_base, (1, N, K), (0, 0, 1))
        a_deq = a_rt.float()[0]
        b_deq = b_rt.float()[0]
    else:
        pad = 16
        a_storage = (torch.randn(1, M, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b_storage = (torch.randn(1, N, K + pad, device=dev) * 0.5).to(torch.float8_e4m3fn)
        a_rt = a_storage[:, :, :K]
        b_rt = b_storage[:, :, :K]
        a_deq = a_rt.float()[0]
        b_deq = b_rt.float()[0]

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((M, sf_k), dev)
        sfb_log = _rand_e8m0((N, sf_k), dev)

    g = _build_nvfp4_graph(M, N, K, block_size=bs, sf_dt=sf_dt, a_dt=a_dt)
    compiled = _plan(g, **_kw(config_name))
    c_storage = torch.zeros(1, M, N + c_pad, dtype=torch.float16, device=dev)
    c = c_storage[:, :, :N]
    assert not a_rt.is_contiguous() or not b_rt.is_contiguous()
    assert not c.is_contiguous()

    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            c,
            _to_blocked(sfa_log).view(1, M, sf_k),
            _to_blocked(sfb_log).view(1, N, sf_k),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_s @ b_s.t()).to(torch.float16)
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        # nvfp4 (fp4 + e4m3 scale, block16) — bit-exact (integer-valued operands).
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            256,
            512,
        ),  # acc_stages=1
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            128,
            128,
            256,
        ),  # acc_stages=2
        # CTA tile split across two MMA instructions along M (num_mma_m=2). The SF
        # words are one per 128-row block, so an M sub-block is exactly one block.
        (
            "nvfp4",
            "CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma",
            256,
            256,
            512,
        ),
        (
            "mxfp4",
            "CONFIG_sm100_256x128x128_128x128x32_cluster1x1_1ctamma",
            256,
            256,
            512,
        ),
        # ... and on the pair, where each CTA drains its own half of every M block.
        (
            "nvfp4",
            "CONFIG_sm100_256x128x128_128x128x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),
        # mxfp4 (fp4 + e8m0 scale, block32).
        (
            "mxfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            256,
            512,
        ),
        # mxfp8 (fp8 e4m3 + e8m0 scale, block32) — multi N-block + single N-block.
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            512,
            512,
        ),
        (
            "mxfp8",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            128,
            128,
            256,
        ),
        # Multicast cluster configs — validate SF multicast. 512³ keeps every CTA active.
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma",
            512,
            512,
            512,
        ),  # A-multicast x4
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x2_1ctamma",
            512,
            512,
            512,
        ),  # A+B multicast
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x1_1ctamma",
            512,
            512,
            512,
        ),  # B-multicast x4
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x2_1ctamma",
            512,
            512,
            512,
        ),  # mx + A+B multicast
        # 2-CTA MMA pair: cta_n=128 → non-overlap acc (acc_stages=2); cta_n=256 → acc-overlap.
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
            256,
            128,
            512,
        ),  # non-overlap
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),  # acc-overlap
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),  # mx + overlap
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x1_2ctamma",
            512,
            512,
            512,
        ),  # B-mcast pair
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x2_2ctamma",
            512,
            512,
            512,
        ),  # A+B-mcast pair
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x2_2ctamma",
            512,
            512,
            512,
        ),  # mx + A+B pair
    ],
)
def test_block_scale_matmul_numerics(combo, config_name, M, N, K):
    _run_bs_numeric(combo, config_name, M, N, K)


def _run_bs_quant_numeric(
    config_name,
    M,
    N,
    K,
    out_dt,
    out_torch_dt,
    scale_dt,
    scale_torch_dt,
    scale_reorder=False,
    combo="nvfp4",
    scale_dim=None,
    global_scale=None,
):
    dev = "cuda"
    torch.manual_seed(0)
    a_rt, b_rt, sfa_log, sfb_log, ref, bs, sf_dt, a_dt = _make_block_scale_inputs(combo, M, N, K, dev)
    global_scale_tensor = None
    if global_scale is not None:
        global_scale_tensor = torch.tensor([[[global_scale]]], dtype=torch.float32, device=dev)
        ref = ref * global_scale
    g = _build_block_scale_quant_graph(
        M,
        N,
        K,
        dequant_block_size=bs,
        quant_block_size=32,
        sf_dt=sf_dt,
        a_dt=a_dt,
        out_dt=out_dt,
        scale_dt=scale_dt,
        scale_reorder=scale_reorder,
        scale_dim=scale_dim,
        global_scale=global_scale is not None,
    )
    compiled = _plan(g, **_kw(config_name))
    assert compiled.block_scale and compiled.chain.quants

    q = torch.empty(1, M, N, dtype=out_torch_dt, device=dev)
    q_scale_shape = scale_dim if scale_dim is not None else (1, M, N // 32)
    # torch has no E5M3 dtype; the kernel writes the scale through an int8 byte
    # carrier (a raw_ptr store to a uint8 tensor is rejected by the DSL).
    scale_buf_dt = torch.int8 if scale_torch_dt == "e5m3" else scale_torch_dt
    if scale_reorder:
        q_scale = torch.zeros(*q_scale_shape, dtype=scale_buf_dt, device=dev)
    else:
        q_scale = torch.empty(*q_scale_shape, dtype=scale_buf_dt, device=dev)
    aux = () if global_scale_tensor is None else (global_scale_tensor,)
    sf_k_padded = _ceil_div(K // bs, 4) * 4
    sfa_rows_padded = _ceil_div(M, 128) * 128
    sfb_rows_padded = _ceil_div(N, 128) * 128
    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            [q, q_scale],
            _to_blocked(sfa_log).view(1, sfa_rows_padded, sf_k_padded),
            _to_blocked(sfb_log).view(1, sfb_rows_padded, sf_k_padded),
            *aux,
        )
    )
    torch.cuda.synchronize()

    q_ref, scale_ref = _block_quant_ref(ref, 32, out_torch_dt, scale_torch_dt)
    if scale_reorder:
        scale_ref = _to_blocked(scale_ref[0]).view_as(q_scale.view(scale_ref.dtype))
    # E5M3 scales are compared as raw BYTES — the strictest form, and the only
    # one available since torch cannot interpret the format.
    got_scale = q_scale.view(torch.uint8) if scale_torch_dt == "e5m3" else q_scale
    torch.testing.assert_close(got_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=0, rtol=0)


@_GPU
@pytest.mark.parametrize(
    "out_dt,out_torch_dt,scale_dt,scale_torch_dt,scale_reorder",
    [
        (
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            cudnn.data_type.FP8_E8M0,
            torch.float8_e8m0fnu,
            False,
        ),
        (
            cudnn.data_type.FP8_E5M2,
            torch.float8_e5m2,
            cudnn.data_type.FP8_E8M0,
            torch.float8_e8m0fnu,
            False,
        ),
        (
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            False,
        ),
        (
            cudnn.data_type.FP8_E4M3,
            torch.float8_e4m3fn,
            cudnn.data_type.FP8_E8M0,
            torch.float8_e8m0fnu,
            True,
        ),
    ],
    ids=(
        "e4m3_out_e8m0_scale",
        "e5m2_out_e8m0_scale",
        "e4m3_out_e4m3_scale",
        "e4m3_out_e8m0_scale_f8_128x4",
    ),
)
def test_block_scale_matmul_quant_epilogue_1cta(out_dt, out_torch_dt, scale_dt, scale_torch_dt, scale_reorder):
    _run_bs_quant_numeric(
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        128,
        128,
        256,
        out_dt,
        out_torch_dt,
        scale_dt,
        scale_torch_dt,
        scale_reorder=scale_reorder,
    )


@_GPU
def test_block_scale_matmul_quant_epilogue_2cta():
    _run_bs_quant_numeric(
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
        256,
        256,
        512,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
    )


@_GPU
def test_block_scale_matmul_quant_epilogue_fp4_input_global_scale_padded_f8_scale():
    _run_bs_quant_numeric(
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        144,
        160,
        256,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
        scale_reorder=True,
        combo="nvfp4",
        scale_dim=(1, 256, 8),
        global_scale=0.5,
    )


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            256,
            256,
            256,
        ),
        (
            "mxfp8",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),
    ],
)
def test_block_scale_matmul_m_major(combo, config_name, M, N, K):
    """M-major block-scale output across 1-CTA and 2-CTA."""
    _run_bs_numeric(combo, config_name, M, N, K, out_major="m")


@_GPU
@pytest.mark.parametrize(
    "combo,config_name,mode",
    [
        ("nvfp4", "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma", "padded"),
        ("nvfp4", "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma", "padded"),
    ],
)
def test_block_scale_matmul_nonpacked_tensors(combo, config_name, mode):
    _run_bs_nonpacked_numeric(combo, config_name, 256, 256, 512, mode)


def _run_bs_reduction_numeric(combo, config_name, M, N, K, mode, red_dims, red_stride, ref_dims):
    dev = "cuda"
    torch.manual_seed(0)
    a_rt, b_rt, sfa_log, sfb_log, ref, bs, sf_dt, a_dt = _make_block_scale_inputs(combo, M, N, K, dev)
    g = _build_block_scale_reduction_graph(
        M,
        N,
        K,
        mode,
        red_dims,
        block_size=bs,
        sf_dt=sf_dt,
        a_dt=a_dt,
        red_stride=red_stride,
    )
    compiled = _plan(g, **_kw(config_name))
    assert compiled.block_scale and compiled.chain.reductions

    c_term = torch.empty(1, M, N, dtype=torch.float32, device=dev)
    if red_stride is None:
        c_red = torch.empty(*red_dims, dtype=torch.float32, device=dev)
    else:
        c_red = torch.empty_strided(tuple(red_dims), tuple(red_stride), dtype=torch.float32, device=dev)
        assert not c_red.is_contiguous()
    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            [c_term, c_red],
            _to_blocked(sfa_log).view(1, M, K // bs),
            _to_blocked(sfb_log).view(1, N, K // bs),
        )
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(c_term[0], ref, atol=2e-1, rtol=2e-2)
    red_ref = _reduction_ref(c_term, mode, ref_dims)
    _assert_block_scale_reduction_close(c_red, red_ref, mode)


@_GPU
@pytest.mark.parametrize(
    "mode",
    [
        cudnn.reduction_mode.ADD,
        cudnn.reduction_mode.AMAX,
        cudnn.reduction_mode.MAX,
        cudnn.reduction_mode.MIN,
    ],
    ids=("add", "amax", "max", "min"),
)
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        (
            "nvfp4",
            "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
            128,
            128,
            256,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),
        (
            "nvfp4",
            "CONFIG_sm100_128x256x128_128x256x32_cluster4x2_2ctamma",
            512,
            512,
            512,
        ),
    ],
)
def test_block_scale_matmul_reduction_scalar(mode, combo, config_name, M, N, K):
    _run_bs_reduction_numeric(
        combo,
        config_name,
        M,
        N,
        K,
        mode,
        red_dims=[1, 1, 1],
        red_stride=None,
        ref_dims=(0, 1, 2),
    )


@_GPU
@pytest.mark.parametrize(
    "mode,red_dims,red_stride,ref_dims",
    [
        (cudnn.reduction_mode.ADD, [1, 1, 256], [0, 0, 2], (0, 1)),
        (cudnn.reduction_mode.AMAX, [1, 256, 1], [0, 2, 1], (0, 2)),
    ],
    ids=("add_per_col_strided_n", "amax_per_row_strided_m"),
)
def test_block_scale_matmul_reduction_strided_output(mode, red_dims, red_stride, ref_dims):
    _run_bs_reduction_numeric(
        "nvfp4",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
        256,
        256,
        512,
        mode,
        red_dims,
        red_stride,
        ref_dims,
    )


def test_block_scale_matmul_reduction_rejects_int32():
    g = _build_block_scale_reduction_graph(
        128,
        128,
        256,
        cudnn.reduction_mode.ADD,
        [1, 1, 1],
        red_dtype=cudnn.data_type.INT32,
        red_compute_dtype=cudnn.data_type.INT32,
    )
    with pytest.raises(NotImplementedError, match="fp32 compute/output"):
        jit_from_cudnn_graph(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))


@_GPU
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_1ctamma",  # M-OOB with cgrp_m > 1
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x2_1ctamma",  # N tile/cluster > N
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",  # 2-CTA pair + acc-overlap
    ],
)
def test_nvfp4_oob_shape(config_name):
    """nvfp4 -> bf16 on an awkward shape (M=23, N=56, K=736): exercises
    ceil-padded SF descriptors + M/N/K OOB."""
    dev = "cuda"
    M, N, K, bs = 23, 56, 736, 16
    sf_k = K // bs
    Kp = ((sf_k + 3) // 4) * 4
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    torch.manual_seed(0)
    a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
    sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(
        name="A",
        dim=[1, M, K],
        stride=[M * K, K, 1],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    B = g.tensor(
        name="B",
        dim=[1, K, N],
        stride=[K * N, 1, K],
        data_type=cudnn.data_type.FP4_E2M1,
    )
    SFA = g.tensor(
        name="SFA",
        dim=[1, M, sf_k],
        stride=[M * sf_k, sf_k, 1],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[1, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=cudnn.data_type.FP8_E4M3,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    Ad = g.block_scale_dequantize(input=A, descale=SFA, block_size=[1, bs])
    Bd = g.block_scale_dequantize(input=B, descale=SFB, block_size=[bs, 1])
    C = g.matmul(A=Ad, B=Bd, name="mm")
    C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    compiled = _plan(g, **_kw(config_name))

    c = torch.zeros(1, M, N, dtype=torch.bfloat16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_u8.view(torch.float4_e2m1fn_x2),
            b_u8.view(torch.float4_e2m1fn_x2),
            c,
            _to_blocked(sfa_log).view(1, 128, Kp),
            _to_blocked(sfb_log).view(1, 128, Kp),
        )
    )
    torch.cuda.synchronize()

    a_deq = _unpack_fp4(a_u8, lut).view(M, K) * sfa_log.float().repeat_interleave(bs, 1)
    b_deq = _unpack_fp4(b_u8, lut).view(N, K) * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_deq @ b_deq.t()).to(torch.bfloat16)
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


# mxfp8 M-major A / N-major B (operand-major layouts). FP4 stays K-major only;
# the SF layout is unchanged — only the packed-data descriptor flips.
@_GPU
@pytest.mark.parametrize(
    "config_name,M,N,K",
    [
        ("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma", 128, 128, 256),
        (
            "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
            256,
            512,
            512,
        ),  # B has 2 N-groups
        (
            "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
            256,
            256,
            512,
        ),  # 2-CTA + overlap
    ],
)
def test_mxfp8_m_major_a_n_major_b(config_name, M, N, K):
    dev = "cuda"
    torch.manual_seed(0)
    bs = 32
    sf_k = K // bs

    # K-major data re-laid-out so A is M-contiguous, B N-contiguous (same values → same ref).
    a_rt = (torch.randn(1, M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    b_rt = (torch.randn(1, N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    a_deq = a_rt.float()[0]
    b_deq = b_rt.float()[0]
    a_rt_m = a_rt.transpose(1, 2).contiguous().transpose(1, 2)  # M contiguous
    b_rt_n = b_rt.transpose(1, 2).contiguous().transpose(1, 2)  # N contiguous
    assert a_rt_m.stride() == (M * K, 1, M) and b_rt_n.stride() == (N * K, 1, N)

    sfa_log = _rand_e8m0((M, sf_k), dev)
    sfb_log = _rand_e8m0((N, sf_k), dev)

    g = _build_nvfp4_graph(
        M,
        N,
        K,
        block_size=bs,
        sf_dt=cudnn.data_type.FP8_E8M0,
        a_dt=cudnn.data_type.FP8_E4M3,
        a_major="m",
        b_major="n",
    )
    chain = analyze(g)
    assert chain.matmul.a_major == "m" and chain.matmul.b_major == "n"
    compiled = _plan(g, **_kw(config_name))

    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_rt_m,
            b_rt_n,
            c,
            _to_blocked(sfa_log).view(1, M, sf_k),
            _to_blocked(sfb_log).view(1, N, sf_k),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_s @ b_s.t()).to(torch.float16)
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


@requires_sm100
def test_fp4_rejects_non_k_major():
    """FP4 must be K-major — sub-byte packing mis-strides an M/N-major
    descriptor, so the compiler rejects it at JIT time."""
    M = N = K = 256
    for a_major, b_major in (("m", "k"), ("k", "n")):
        g = _build_nvfp4_graph(
            M,
            N,
            K,
            block_size=16,
            sf_dt=cudnn.data_type.FP8_E4M3,
            a_dt=cudnn.data_type.FP4_E2M1,
            a_major=a_major,
            b_major=b_major,
        )
        with pytest.raises(ValueError, match="must be K-major"):
            jit_from_cudnn_graph(g, **_kw("CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"))


# sm103 (fp4 K=48B UTCOMMA, K-tile 384 B): config / registry / renderer unit
# tests (run anywhere), cute.compile smokes (any Blackwell GPU), and runtime
# numerics vs a torch reference (sm103 GPU only).

_CFG_128 = "CONFIG_sm103_128x128x384_128x128x48_cluster1x1"
_CFG_256 = "CONFIG_sm103_128x256x384_128x256x48_cluster1x1"

# The kernel compiles on any Blackwell-family GPU (the K=96 mode is an idesc
requires_sm103 = pytest.mark.skipif(
    _SM is None or not (103 <= _SM < 110),
    reason="sm103 block-scale kernels run only on 103 <= SM < 110, have " + ("none" if _SM is None else f"sm_{_SM}"),
)


def _sm103_kw(config_name, cta_group=1):
    return dict(config=by_name(config_name), cta_group=cta_group)


@pytest.fixture
def _pretend_sm103(monkeypatch):
    monkeypatch.setattr(C, "_current_arch", lambda: 103)


# Config catalog / geometry guards


@requires_sm100
def test_block_scale_sf_rule_is_on_the_instruction_tile() -> None:
    """The SF 128x4 swizzle rule is "each MMA instruction covers whole SF blocks",
    so it reads off mma_inst_m/n. The old cta_tile form let mma_inst_m=64 through."""
    from cudnn.gemm.frost.tile_config import validate_block_scale_config

    validate_block_scale_config(by_name("CONFIG_sm100_256x128x128_128x128x32_cluster1x1"), 16, 256)
    with pytest.raises(NotImplementedError) as e:
        # cta_tile_m % 128 == 0 but each instruction covers only half an SF block
        validate_block_scale_config(by_name("CONFIG_sm100_128x128x128_64x128x32_cluster1x1"), 16, 256)
    assert "mma_inst_m % 128" in str(e.value)


def test_catalog_has_sm103_geometries():
    sm103 = [c for c in CATALOG if c.pipeline == "sm103"]
    # 2 cta_n × the shared 15-cluster enumeration.
    assert len(sm103) == 30
    pat = re.compile(r"^CONFIG_sm103_128x(128|256)x384_128x(128|256)x48_cluster\d+x\d+$")
    for c in sm103:
        assert pat.match(c.name), c.name
        assert c.cta_tile_m == 128 and c.cta_tile_k_bytes == 384
        assert c.mma_inst_k_bytes == 48
    assert by_name(_CFG_128).geometry_name == "128x128x384_128x128x48_cluster1x1"


def test_config_families():
    kw = dict(cta_tile_m=128, cta_tile_n=128, cgrp_size_m=1, cgrp_size_n=1, epi_tile_mn=(128, 32), threads_per_cta=256)
    # 384-B K-tile is sm103-only; sm100 keeps the 128-B SWIZZLE_128B cap.
    with pytest.raises(NotImplementedError, match="cta_tile_k_bytes=384"):
        ConfigSm100(cta_tile_k_bytes=384, mma_inst_k_bytes=48, pipeline="sm100", **kw)
    ConfigSm103(cta_tile_k_bytes=384, mma_inst_k_bytes=48, pipeline="sm103", **kw)
    with pytest.raises(NotImplementedError, match="cta_tile_k_bytes=512"):
        ConfigSm103(cta_tile_k_bytes=512, mma_inst_k_bytes=48, pipeline="sm103", **kw)
    # The sm103 K axes are the FAMILY's, not free geometry.
    with pytest.raises(NotImplementedError, match="fixes cta_tile_k_bytes=384"):
        ConfigSm103(cta_tile_k_bytes=128, mma_inst_k_bytes=32, pipeline="sm103", **kw)
    # A raw-base construction can't bypass the family invariant either.
    with pytest.raises(NotImplementedError, match="fixes cta_tile_k_bytes=384"):
        TileConfig(cta_tile_k_bytes=128, pipeline="sm103", **kw)
    # The K-tile walks the MMA instruction, so it is a multiple of the PIPELINE's
    # K width: 96 B is three sm100 instructions but not a whole number of sm107's.
    ConfigSm100(cta_tile_k_bytes=96, mma_inst_k_bytes=32, pipeline="sm100", **kw)
    with pytest.raises(NotImplementedError, match="multiple of sm107's mma_inst_k_bytes=64"):
        ConfigSm107(cta_tile_k_bytes=96, mma_inst_k_bytes=64, pipeline="sm107", **kw)
    # Catalog entries carry their family class (the template-pairing key).
    assert all(isinstance(c, ConfigSm103) for c in CATALOG if c.pipeline == "sm103")
    assert all(isinstance(c, ConfigSm100) for c in CATALOG if c.pipeline == "sm100")


def test_validate_block_scale_config_arch_fork():
    validate_block_scale_config(by_name(_CFG_128), 16, 768)
    validate_block_scale_config(by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1"), 16, 256)
    # A wrong-K sm103 config cannot even be CONSTRUCTED (family invariant),
    # so validate's arch fork only ever sees kb=384 for sm103.
    with pytest.raises(NotImplementedError, match="fixes cta_tile_k_bytes=384"):
        ConfigSm103(
            cta_tile_m=128,
            cta_tile_n=128,
            cta_tile_k_bytes=128,
            cgrp_size_m=1,
            cgrp_size_n=1,
            epi_tile_mn=(128, 32),
            threads_per_cta=256,
            pipeline="sm103",
        )


# Registry / template selection


def _bs_chain(combo="nvfp4", M=256, N=256, K=1536):
    is_fp4 = combo in ("nvfp4", "mxfp4")
    bs = 16 if combo == "nvfp4" else 32
    a_dt = cudnn.data_type.FP4_E2M1 if is_fp4 else cudnn.data_type.FP8_E4M3
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0
    return _build_nvfp4_graph(M, N, K, block_size=bs, sf_dt=sf_dt, a_dt=a_dt)


def test_select_template_dispatches_on_config_arch():
    chain = analyze(_bs_chain())
    t103 = select_template(chain, by_name(_CFG_128), cta_group=1)
    assert t103.file == "sm103_block_scale_matmul_1ctamma.py"
    from cudnn.gemm.frost.kernel_registry import PIPELINE_ARCH_RANGES

    assert PIPELINE_ARCH_RANGES[t103.pipeline] == ((103, 110),)
    t100 = select_template(chain, by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1"), cta_group=1)
    assert t100.file == "sm100_block_scale_matmul_1ctamma.py"
    t103_2 = select_template(chain, by_name(_CFG_128), cta_group=2)
    assert t103_2.file == "sm103_block_scale_matmul_2ctamma.py"
    # Pairing is by config CLASS (from the template's filename arch token):
    # a base TileConfig posing as sm103 matches no template.
    imposter = TileConfig(
        cta_tile_m=128,
        cta_tile_n=128,
        cta_tile_k_bytes=384,
        cgrp_size_m=1,
        cgrp_size_n=1,
        epi_tile_mn=(128, 32),
        threads_per_cta=256,
        pipeline="sm103",
        mma_inst_k_bytes=48,
    )
    with pytest.raises(ValueError, match="no kernel template"):
        select_template(chain, imposter, cta_group=1)


def test_sm103_template_rejects_mxfp8(_pretend_sm103):
    """mxfp8 is absent from the sm103 family's MMA_TYPE_SUPPORT existence set
    (the K=48B UTCOMMA has no fp8 variant) — expressed as data, not code."""
    chain = analyze(_bs_chain(combo="mxfp8", K=512))
    tmpl = next(t for t in TEMPLATES if t.pipeline == "sm103" and t.graph_type is GraphType.BLOCK_SCALE_MATMUL)
    reason = tmpl.accepts(chain, by_name(_CFG_128))
    assert reason is not None and "sm103" in reason and "does not support" in reason


def test_sm103_template_accepts_fp4(_pretend_sm103):
    tmpl = next(t for t in TEMPLATES if t.pipeline == "sm103" and t.graph_type is GraphType.BLOCK_SCALE_MATMUL)
    for combo in ("nvfp4", "mxfp4"):
        chain = analyze(_bs_chain(combo=combo))
        assert tmpl.accepts(chain, by_name(_CFG_128)) is None
        assert tmpl.accepts(chain, by_name(_CFG_256)) is None


def test_mma_gpu_arch_special_cases(monkeypatch):
    """Most combos are arch-free (template family gate decides); the rare
    instruction-level exceptions live in MMA_GPU_ARCH_SPECIAL_CASES — int8
    UTCIMMA exists only on SM 100 / SM 110, NOT on other family members."""
    from types import SimpleNamespace

    from cudnn.gemm.frost import kernel_registry as kr

    int8_chain = SimpleNamespace(matmul=SimpleNamespace(a_dtype="int8", b_dtype="int8", accum_dtype="int32"))
    monkeypatch.setattr(C, "_current_arch", lambda: 103)
    assert "exists only on" in kr.mma_arch_reject(int8_chain, kr.GraphType.MATMUL, "sm100")
    for ok_sm in (100, 110):
        monkeypatch.setattr(C, "_current_arch", lambda v=ok_sm: v)
        assert kr.mma_arch_reject(int8_chain, kr.GraphType.MATMUL, "sm100") is None
    # A family-portable combo is arch-free at this gate (stage 0 handles GPUs).
    bf16_chain = SimpleNamespace(matmul=SimpleNamespace(a_dtype="bf16", b_dtype="bf16", accum_dtype="fp32"))
    monkeypatch.setattr(C, "_current_arch", lambda: 90)
    assert kr.mma_arch_reject(bf16_chain, kr.GraphType.MATMUL, "sm100") is None


def test_jit_rejects_wrong_active_arch(monkeypatch):
    monkeypatch.setattr(C, "_current_arch", lambda: 100)
    with pytest.raises(NotImplementedError, match=r"103 <= SM < 110.*sm_100"):
        jit_from_cudnn_graph(_bs_chain(), **_sm103_kw(_CFG_128))


def test_sm103_rejects_multi_mma_m():
    """The sm103 chunk pipeline has NOT been adapted to a CTA tile spanning
    several MMA instructions along M: it miscomputes (A reads unwritten SMEM in
    K) and its ab_stages budget under-counts, so cta_tile_m=256 also overruns the
    SMEM cap. Both are silent-wrong / launch-fail, so the template declines the
    geometry outright. Drop `supports_multi_mma_m=False` when it is fixed."""
    wide = by_name("CONFIG_sm103_256x128x384_128x128x48_cluster1x1")
    assert wide.num_mma_m == 2
    for t in (t for t in TEMPLATES if t.pipeline == "sm103"):
        assert not t.supports_multi_mma_m
        assert "num_mma_m=2" in t.multi_mma_m_reject(wide)
        assert t.multi_mma_m_reject(by_name(_CFG_128)) is None
    # The other pipelines DO implement it — the gate is sm103-specific.
    for f in ("sm100_block_scale_matmul_1ctamma.py", "sm107_block_scale_matmul_1ctamma.py"):
        t = next(t for t in TEMPLATES if t.file == f)
        assert t.supports_multi_mma_m and t.multi_mma_m_reject(wide) is None


@requires_sm103
def test_sm103_multi_mma_m_is_declined_not_miscomputed():
    """The gate reaches the JIT path, so the geometry raises instead of running."""
    with pytest.raises(NotImplementedError, match="several MMA instructions along M"):
        jit_from_cudnn_graph(_bs_chain(), **_sm103_kw("CONFIG_sm103_256x128x384_128x128x48_cluster1x1"))


def test_tma_alignment_unified():
    """ONE alignment rule for every pipeline: contiguous extent × elem bits
    must be a multiple of 128 (the TMA 16-byte stride encode)."""
    rej = C._tma_alignment_reject
    assert rej("fp4_e2m1", "fp4_e2m1", "k", "k", 128, 128, 1024) is None
    assert "K % 32" in rej("fp4_e2m1", "fp4_e2m1", "k", "k", 128, 128, 1040)
    assert rej("fp8_e4m3", "fp8_e4m3", "k", "k", 128, 128, 1040) is None  # 1040 % 16 == 0
    assert "K % 16" in rej("fp8_e4m3", "fp8_e4m3", "k", "k", 128, 128, 1032)
    assert "K % 8" in rej("bf16", "bf16", "k", "k", 128, 128, 1004)
    # MN-major operands are gated on their OWN contiguous extent, not K.
    assert "M % 16" in rej("fp8_e4m3", "fp8_e4m3", "m", "k", 200, 128, 512)
    assert rej("fp8_e4m3", "fp8_e4m3", "m", "k", 208, 128, 512) is None


@requires_sm100
def test_sm103_rejects_misaligned_runtime_k(_pretend_sm103):
    """The kernel is shape-agnostic — the graph may be aligned while the CALL
    carries a misaligned K. The unified runtime gate must reject before any
    launch. (Partial K-tiles — K % 768 != 0 — are legal; see the K=4096
    numerics case.)"""
    M, N = 128, 128
    g = _bs_chain(M=M, N=N, K=1536)  # aligned graph → JIT succeeds
    compiled = _plan(g, **_sm103_kw(_CFG_128))
    K = 1040  # runtime K % 32 == 16
    dev = "cuda"
    sf_k = (K // 16 + 3) // 4 * 4
    a = torch.zeros(1, M, K // 2, dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    b = torch.zeros(1, N, K // 2, dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    sfa = torch.zeros(1, M, sf_k, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    sfb = torch.zeros(1, N, sf_k, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="K % 32"):
        compiled(_vp_bs(compiled, a, b, c, sfa, sfb))


# End-to-end numerics (sm103 GPU only)


def _run_sm103_numeric(combo, config_name, M, N, K, cta_group=1):
    dev = "cuda"
    torch.manual_seed(0)
    bs = 16 if combo == "nvfp4" else 32
    sf_k = K // bs
    a_dt = cudnn.data_type.FP4_E2M1
    sf_dt = cudnn.data_type.FP8_E4M3 if combo == "nvfp4" else cudnn.data_type.FP8_E8M0

    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
    a_rt = a_u8.view(torch.float4_e2m1fn_x2)
    b_rt = b_u8.view(torch.float4_e2m1fn_x2)
    a_deq = _unpack_fp4(a_u8, lut).view(M, K)
    b_deq = _unpack_fp4(b_u8, lut).view(N, K)

    if combo == "nvfp4":
        sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    else:
        sfa_log = _rand_e8m0((M, sf_k), dev)
        sfb_log = _rand_e8m0((N, sf_k), dev)

    g = _build_nvfp4_graph(M, N, K, block_size=bs, sf_dt=sf_dt, a_dt=a_dt)
    compiled = _plan(g, **_sm103_kw(config_name, cta_group))
    assert compiled.block_scale
    assert (compiled.chain.block_scale.sf_dtype, compiled.chain.block_scale.block_size) == (_DTYPE_FROM_CUDNN[sf_dt], bs)

    # The F8_128x4 reorder pads to 128-row × 4-SF blocks; view with the
    # padded dims (matters for M/N not multiples of 128).
    mp = (M + 127) // 128 * 128
    np_ = (N + 127) // 128 * 128
    kp = (sf_k + 3) // 4 * 4
    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_rt,
            b_rt,
            c,
            _to_blocked(sfa_log).view(1, mp, kp),
            _to_blocked(sfb_log).view(1, np_, kp),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * sfa_log.float().repeat_interleave(bs, 1)
    b_s = b_deq * sfb_log.float().repeat_interleave(bs, 1)
    ref = (a_s @ b_s.t()).to(torch.float16)
    torch.testing.assert_close(c[0], ref, atol=2e-1, rtol=2e-2)


@requires_sm103
@pytest.mark.parametrize(
    "combo,config_name,shape",
    [
        ("nvfp4", _CFG_128, (256, 256, 1536)),
        ("nvfp4", _CFG_128, (255, 384, 768)),  # M-OOB + multi-tile N
        ("nvfp4", _CFG_128, (256, 256, 4096)),  # partial K-tile (4096 % 768 != 0)
        ("nvfp4", _CFG_256, (256, 512, 1536)),  # N=256 tile + acc overlap
        ("mxfp4", _CFG_128, (256, 256, 1536)),
        ("mxfp4", _CFG_128, (384, 256, 2304)),
        ("mxfp4", _CFG_128, (256, 256, 4096)),  # partial K-tile
    ],
    ids=lambda v: v if isinstance(v, str) else "x".join(map(str, v)),
)
def test_sm103_block_scale_matmul_numerics(combo, config_name, shape):
    _run_sm103_numeric(combo, config_name, *shape)


@requires_sm103
@pytest.mark.parametrize(
    "combo,config_name,shape",
    [
        ("nvfp4", "CONFIG_sm103_128x128x384_128x128x48_cluster2x1", (512, 512, 1536)),
        ("nvfp4", "CONFIG_sm103_128x128x384_128x128x48_cluster2x2", (512, 512, 1536)),
        ("nvfp4", "CONFIG_sm103_128x256x384_128x256x48_cluster2x1", (512, 512, 1536)),
        ("nvfp4", "CONFIG_sm103_128x128x384_128x128x48_cluster2x1", (255, 384, 4096)),  # M-OOB + partial K
        ("mxfp4", "CONFIG_sm103_128x128x384_128x128x48_cluster2x1", (512, 512, 2304)),
    ],
    ids=lambda v: v if isinstance(v, str) else "x".join(map(str, v)),
)
def test_sm103_block_scale_matmul_numerics_2ctamma(combo, config_name, shape):
    _run_sm103_numeric(combo, config_name, *shape, cta_group=2)


@requires_sm103
@pytest.mark.parametrize(
    "cluster,M,N",
    [
        ("cluster2x1", 512, 512),
        ("cluster1x2", 512, 512),
        ("cluster2x2", 512, 512),
        # Large clusters (shared sm100 enumeration); shapes sized so the grid
        # holds at least one full cluster per axis.
        ("cluster1x4", 512, 1024),
        ("cluster4x1", 1024, 512),
        ("cluster4x4", 1024, 1024),
        ("cluster8x1", 2048, 512),
        ("cluster16x1", 2048, 512),
    ],
)
def test_sm103_block_scale_matmul_clusters(cluster, M, N):
    _run_sm103_numeric("nvfp4", f"CONFIG_sm103_128x128x384_128x128x48_{cluster}", M, N, 1536)


@requires_sm100
@pytest.mark.parametrize("M,N", [(64, 4096), (4096, 64), (128, 128), (4096, 4096)])
def test_auto_config_is_accepted_by_the_registry(M, N):
    """``select_config`` is a second decision path that does not consult the
    registry funnel, so its pick can be one the funnel rejects — the graph then
    fails to build and falls back to native cuDNN with only an INFO log. Block
    scale is the narrow case: the F8_128x4 SF swizzle needs 128-multiple tiles,
    so the plain 32/64 ladder is illegal. Pin the invariant the funnel owns:
    whatever the heuristic picks must be in ``candidates(chain)``."""
    from cudnn.gemm.frost.kernel_registry import candidates, preferred_pipeline
    from cudnn.gemm.frost.tile_config import as_pipeline, select_config

    chain = analyze(_build_nvfp4_graph(M, N, 512))
    assert chain.has_block_scale
    cfg, _cta_group = select_config(chain.matmul.M, chain.matmul.N, chain.num_gemms, block_scale=chain.has_block_scale)
    cfg = as_pipeline(cfg, preferred_pipeline(chain))  # the config build_gemm_plan actually builds
    accepted = {c.name for _t, c in candidates(chain)}
    assert accepted, "the registry accepts no geometry at all for this chain"
    assert cfg.name in accepted, f"select_config picked {cfg.name!r}, which the registry rejects for this graph"


# --- SF blob packing guard -------------------------------------------------
# The templates rebuild the F8_128x4 layout from the SF BASE POINTER alone (a
# packed run of 512-B atoms, 128 rows x 4 SF-K), so a blob that is not one dense
# byte run of that size is read out of bounds and silently miscomputes. The
# graph declares the LOGICAL scale factors, whose shape legitimately differs from
# the reordered blob, so only the call site can check this.
_SF_GUARD_CFG = "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma"


def _sf_guard_case(M, N, K, kind):
    """Returns (compiled, variant-pack args, expect_reject)."""
    dev = "cuda"
    torch.manual_seed(0)
    bs, sf_k = 16, K // 16
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
    a_rt, b_rt = a_u8.view(torch.float4_e2m1fn_x2), b_u8.view(torch.float4_e2m1fn_x2)
    sfa_log = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfb_log = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)
    sfa_ok = _to_blocked(sfa_log).view(1, -1, sf_k)
    sfb_ok = _to_blocked(sfb_log).view(1, -1, sf_k)

    if kind == "packed":
        sfa = sfa_ok
    elif kind == "padded_slice":
        arena = torch.zeros(1, sfa_ok.shape[1], sf_k + 4, dtype=torch.uint8, device=dev)
        arena[:, :, :sf_k] = sfa_ok.reshape(1, -1, sf_k).view(torch.uint8)
        sfa = arena[:, :, :sf_k].view(torch.float8_e4m3fn)
    elif kind == "undersized":
        sfa = sfa_log.reshape(1, M, sf_k).clone()
    elif kind == "arena_row_slice":
        rows = sfa_ok.shape[1]
        arena = torch.zeros(1, 2 * rows, sf_k, dtype=torch.uint8, device=dev)
        arena[:, rows:, :] = sfa_ok.reshape(1, -1, sf_k).view(torch.uint8)
        sfa = arena[:, rows:, :].view(torch.float8_e4m3fn)
    else:
        raise AssertionError(kind)

    g = _build_nvfp4_graph(M, N, K, block_size=bs)
    compiled = _plan(g, **_kw(_SF_GUARD_CFG))
    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    return compiled, _vp_bs(compiled, a_rt, b_rt, c, sfa, sfb_ok)


@requires_sm100
@pytest.mark.parametrize(
    "M,K,kind,rejected",
    [
        (256, 512, "packed", False),
        (256, 512, "arena_row_slice", False),
        (256, 512, "padded_slice", True),
        (192, 512, "undersized", True),
    ],
)
def test_sf_blob_must_be_packed(M, K, kind, rejected):
    compiled, vp = _sf_guard_case(M, 256, K, kind)
    if rejected:
        with pytest.raises(ValueError, match="F8_128x4 scale factors must be a packed blob"):
            compiled(vp)
    else:
        compiled(vp)
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# sm107 block-scale pipeline (the sm100 pipeline on the 64-byte-K MMA)
# ---------------------------------------------------------------------------

_SM107_128 = "CONFIG_sm107_128x128x128_128x128x64_cluster1x1"
_SM107_256 = "CONFIG_sm107_128x256x128_128x256x64_cluster1x1"


@pytest.fixture
def _pretend_sm107(monkeypatch):
    monkeypatch.setattr(C, "_current_arch", lambda: 107)


def _sm107_kw(config_name, cta_group=1):
    return dict(config=by_name(config_name), cta_group=cta_group)


def test_catalog_has_sm107_geometries():
    sm107 = [c for c in CATALOG if c.pipeline == "sm107"]
    # num_mma_m {1,2} × 2 cta_n × the shared 15-cluster enumeration.
    assert len(sm107) == 60
    pat = re.compile(r"^CONFIG_sm107_(128|256)x(128|256)x128_128x(128|256)x64_cluster\d+x\d+$")
    for c in sm107:
        assert pat.match(c.name), c.name
        assert isinstance(c, ConfigSm107)
        # mma_inst_m is pinned: the block-scale F8_128x4 SF swizzle needs % 128 == 0.
        assert c.mma_inst_m == 128 and c.cta_tile_k_bytes == 128
        assert c.cta_tile_m == 128 * c.num_mma_m
        assert c.mma_inst_k_bytes == 64
    assert by_name(_SM107_128).geometry_name == "128x128x128_128x128x64_cluster1x1"


def test_sm107_family_fixes_the_mma_k_width():
    """The 64-byte MMA K is the FAMILY's, not free geometry — while the rest of
    the geometry axes stay exactly sm100's."""
    kw = dict(
        cta_tile_m=128,
        cta_tile_n=128,
        cta_tile_k_bytes=128,
        cgrp_size_m=1,
        cgrp_size_n=1,
        epi_tile_mn=(128, 32),
        threads_per_cta=256,
        pipeline="sm107",
    )
    ConfigSm107(mma_inst_k_bytes=64, **kw)
    with pytest.raises(NotImplementedError, match="sm107 fixes mma_inst_k_bytes=64"):
        ConfigSm107(mma_inst_k_bytes=32, **kw)
    # A raw-base construction can't bypass the family invariant either.
    with pytest.raises(NotImplementedError, match="sm107 fixes mma_inst_k_bytes=64"):
        TileConfig(**kw)


def test_sm107_template_selection_and_arch_gate(_pretend_sm107):
    chain = analyze(_bs_chain())
    for cta_group, want in ((1, "sm107_block_scale_matmul_1ctamma.py"), (2, "sm107_block_scale_matmul_2ctamma.py")):
        cfg = by_name(_SM107_128 if cta_group == 1 else "CONFIG_sm107_128x128x128_128x128x64_cluster2x1")
        tmpl = select_template(chain, cfg, cta_group=cta_group)
        assert tmpl.file == want
        assert tmpl.accepts(chain, cfg) is None
    # An sm100 config still pairs with the sm100 templates on the same GPU.
    sm100_cfg = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    assert select_template(chain, sm100_cfg, cta_group=1).file == "sm100_block_scale_matmul_1ctamma.py"


def test_sm107_templates_reject_older_blackwell(monkeypatch):
    monkeypatch.setattr(C, "_current_arch", lambda: 100)
    chain = analyze(_bs_chain())
    tmpl = next(t for t in TEMPLATES if t.file == "sm107_block_scale_matmul_1ctamma.py")
    assert "107 <= SM < 110" in tmpl.accepts(chain, by_name(_SM107_128))


@requires_sm107
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize(
    "config_name,cta_group",
    [
        (_SM107_128 + "_1ctamma", 1),
        (_SM107_256 + "_1ctamma", 1),
        ("CONFIG_sm107_128x128x128_128x128x64_cluster1x2_1ctamma", 1),
        ("CONFIG_sm107_128x128x128_128x128x64_cluster2x1_2ctamma", 2),
        ("CONFIG_sm107_128x256x128_128x256x64_cluster2x1_2ctamma", 2),
        ("CONFIG_sm107_128x256x128_128x256x64_cluster2x2_2ctamma", 2),
    ],
    ids=lambda v: v if isinstance(v, str) else f"cta{v}",
)
def test_sm107_block_scale_matmul_numerics(combo, config_name, cta_group):
    _run_bs_numeric(combo, config_name, 256, 256, 512)


@requires_sm107
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize("cta_m,cta_n", [(128, 256), (256, 128), (256, 256)])
def test_sm107_block_scale_matmul_multi_mma_m(combo, cta_group, cta_m, cta_n):
    """The CTA tile spanning several MMA instructions along M, on the 64-byte-K
    pipeline. This is where the two SF regions stop agreeing: at nvfp4 a scale
    word spans word_atoms=2 atoms, and SFA is indexed per M block (one MMA
    instruction covers one 128-row block, so its word must be contiguous) while
    SFB is walked across all N blocks by one instruction. Both layouts collapse
    to the same addresses at a single block, so only cta_m/cta_n = 256 tells
    them apart -- 256x256 is the case where both regions split at once."""
    cluster = "cluster1x1" if cta_group == 1 else "cluster2x1"
    suffix = "1ctamma" if cta_group == 1 else "2ctamma"
    geometry = f"CONFIG_sm107_{cta_m}x{cta_n}x128_128x{cta_n}x64_{cluster}"
    assert by_name(geometry).num_mma_m == cta_m // 128
    _run_bs_numeric(combo, f"{geometry}_{suffix}", 256, 256, 512)


@requires_sm107
@pytest.mark.parametrize(
    "combo,config_name,M,N,K",
    [
        ("nvfp4", _SM107_128 + "_1ctamma", 256, 384, 768),  # multi-tile N
        ("mxfp4", _SM107_256 + "_1ctamma", 256, 256, 4096),  # many K-tiles
        ("mxfp8", "CONFIG_sm107_128x128x128_128x128x64_cluster2x1_2ctamma", 384, 512, 256),
        ("nvfp4", "CONFIG_sm107_128x128x128_128x128x64_cluster4x1_2ctamma", 1024, 512, 512),
        ("nvfp4", "CONFIG_sm107_128x128x128_128x128x64_cluster1x4_1ctamma", 512, 1024, 512),
    ],
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_sm107_block_scale_matmul_shapes_and_clusters(combo, config_name, M, N, K):
    _run_bs_numeric(combo, config_name, M, N, K)


@requires_sm107
@pytest.mark.parametrize("combo", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm107_128x128x128_128x128x64_cluster4x1_2ctamma",
        "CONFIG_sm107_128x128x128_128x128x64_cluster4x2_2ctamma",
        "CONFIG_sm107_256x256x128_128x256x64_cluster2x4_2ctamma",
        "CONFIG_sm107_128x128x128_128x128x64_cluster4x1_1ctamma",
        "CONFIG_sm107_128x128x128_128x128x64_cluster1x4_1ctamma",
        "CONFIG_sm107_128x128x128_128x128x64_cluster2x2_1ctamma",
    ],
)
def test_sm107_block_scale_mixed_cga(combo, config_name):
    """Mixed CGA rides along with no caller change: any config whose cluster is
    wider than the MMA mode's minimum launches it as the PREFERRED shape plus that
    minimum as the fallback. The tile decomposition is the identity map for either
    cluster shape, so both kinds cover the problem exactly once; only the multicast
    masks, mbarrier arrival counts and rank math follow the shape the CTA actually
    landed in. M is large enough that the grid outruns what the preferred clusters
    hold resident, which is when the device substitutes the fallback shape."""
    cta_group = 2 if config_name.endswith("_2ctamma") else 1
    cfg = by_name(config_name.rsplit("_", 1)[0])
    assert C._mixed_cga_fallback(cfg, cta_group, f"sm107_block_scale_matmul_{cta_group}ctamma.py") == (cta_group, 1)
    _run_bs_numeric(combo, config_name, 1920, 1920, 512)


@requires_sm107
@pytest.mark.parametrize("cta_group", [1, 2])
def test_mixed_cga_fallback_is_the_mma_mode_minimum(cta_group):
    """The fallback shape is derived, never passed: one CTA for a 1-CTA MMA, the
    pair for a 2-CTA one — and a config already AT that minimum has nothing to
    fall back to, so it launches as a plain fixed cluster."""
    tmpl = f"sm107_block_scale_matmul_{cta_group}ctamma.py"
    assert C.min_fallback_cluster(cta_group) == (cta_group, 1)
    wide = by_name("CONFIG_sm107_128x128x128_128x128x64_cluster4x2")
    assert C._mixed_cga_fallback(wide, cta_group, tmpl) == (cta_group, 1)
    minimal = by_name(f"CONFIG_sm107_128x128x128_128x128x64_cluster{cta_group}x1")
    assert C._mixed_cga_fallback(minimal, cta_group, tmpl) is None


@requires_sm107
def test_mixed_cga_is_off_where_it_cannot_be_honored(monkeypatch):
    """Every gate is a fact, not a knob: the GPU's ability to substitute clusters,
    whether the template consumes the fallback constant at all (an unported one
    would hang — its cluster constants are baked to the preferred shape), and
    whether the config pins the N-super-block walk (not invariant across the two
    cluster shapes)."""
    wide = by_name("CONFIG_sm107_128x128x128_128x128x64_cluster4x2")
    sm107_tmpl = "sm107_block_scale_matmul_2ctamma.py"
    assert C._mixed_cga_fallback(wide, 2, sm107_tmpl) == (2, 1)

    # Template that never reads the constant -> no fallback attached. The MoE
    # ones stay that way: their fixed-grid persistent scheduler strides by a
    # host-computed cluster count, which mixed clusters invalidate.
    moe_tmpl = "sm100_moe_grouped_block_scale_matmul_fwd_2ctamma.py"
    assert not C._template_reads_fallback_cluster(moe_tmpl)
    assert C._mixed_cga_fallback(wide, 2, moe_tmpl) is None

    # Substitution is a floor, not a range: every part from SM 10.0 up can do it.
    assert C._mixed_cga_supported(100) and C._mixed_cga_supported(110)
    # A pre-Blackwell part -> plain fixed cluster, as before.
    monkeypatch.setattr(C, "_current_arch", lambda: 90)
    assert not C._mixed_cga_supported()
    assert C._mixed_cga_fallback(wide, 2, sm107_tmpl) is None
    monkeypatch.undo()

    # A pinned N-super-block walk -> skipped rather than silently mis-tiled.
    pinned = dataclasses.replace(wide, tile_swizzle_n=8)
    assert C._mixed_cga_fallback(pinned, 2, sm107_tmpl) is None

    # The escape hatch for A/B measurement.
    monkeypatch.setenv("CUDNN_FROST_DISABLE_MIXED_CGA", "1")
    assert C._mixed_cga_fallback(wide, 2, sm107_tmpl) is None


@requires_sm107
def test_mixed_cga_ported_templates_attach_a_fallback():
    """A ported template on a wide-cluster config launches with both shapes and
    still computes the same result; an unported one renders exactly as it did
    before mixed CGA existed."""
    sm100_cfg = "CONFIG_sm100_128x128x128_128x128x32_cluster4x2_2ctamma"
    _run_bs_numeric("nvfp4", sm100_cfg, 512, 512, 512)
    g = _build_nvfp4_graph(256, 256, 512)
    src = _plan(g, **_kw(sm100_cfg)).generated_path.read_text()
    assert "fallback_cluster=fallback_cluster_shape_mnk" in src
    assert "fallback_cluster_shape_mnk = (2, 1, 1)" in src
    # What makes the tile walk the identity map for BOTH shapes: the renderer
    # pins the N-super-block width, so _auto_swizzle_w const-folds to 1.
    assert "tile_swizzle_n = 1" in src

    # Already-minimal cluster -> nothing to fall back to, plain fixed launch.
    minimal_cfg = "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma"
    src = _plan(_build_nvfp4_graph(256, 256, 512), **_kw(minimal_cfg)).generated_path.read_text()
    assert "fallback_cluster_shape_mnk = None" in src


@requires_sm107
@pytest.mark.parametrize(
    "config_name",
    [
        _SM107_128 + "_1ctamma",
        "CONFIG_sm107_128x256x128_128x256x64_cluster2x1_1ctamma",  # M-OOB with cgrp_m > 1
        "CONFIG_sm107_128x128x128_128x128x64_cluster1x2_1ctamma",  # N tile/cluster > N
        "CONFIG_sm107_128x256x128_128x256x64_cluster2x1_2ctamma",  # 2-CTA pair
    ],
)
def test_sm107_nvfp4_oob_shape(config_name):
    """M=23, N=56, K=736 — ceil-padded SF descriptors + M/N/K OOB, on the
    64-byte-K MMA (the last K-tile is only 736 % 256 = 224 elements)."""
    test_nvfp4_oob_shape(config_name)


@requires_sm107
@pytest.mark.parametrize(
    "combo,config_name",
    [
        ("nvfp4", _SM107_128 + "_1ctamma"),
        ("mxfp8", "CONFIG_sm107_128x256x128_128x256x64_cluster2x1_2ctamma"),
    ],
)
def test_sm107_block_scale_matmul_m_major(combo, config_name):
    _run_bs_numeric(combo, config_name, 256, 256, 512, out_major="m")


@requires_sm107
@pytest.mark.parametrize("scale_reorder", [False, True])
@pytest.mark.parametrize("config_name", [_SM107_128 + "_1ctamma", "CONFIG_sm107_128x256x128_128x256x64_cluster2x1_2ctamma"])
def test_e5m3_quant_epilogue(config_name, scale_reorder):
    """The epilogue can PRODUCE E5M3 scales, bit-exact against the torch
    reference. The `cvt ... ue5m3x2.f32` this needs exists only on sm_107 —
    strictly narrower than CONSUMING E5M3 scales, where the format is a
    descriptor field every pipeline emits."""
    _run_bs_quant_numeric(
        config_name,
        256,
        256,
        512,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E5M3,
        "e5m3",
        scale_reorder=scale_reorder,
    )


def test_e5m3_quant_rejects_off_sm107(monkeypatch):
    """...and off sm_107 it declines cleanly rather than emitting PTX ptxas
    would reject."""
    g = _build_block_scale_quant_graph(256, 256, 512, dequant_block_size=32, scale_dt=cudnn.data_type.FP8_E5M3)
    for arch in (100, 103, 110, 120):
        monkeypatch.setattr(C, "_current_arch", lambda a=arch: a)
        with pytest.raises(NotImplementedError, match=f"sm_{arch}"):
            jit_from_cudnn_graph(g, **_kw(_SM107_128 + "_1ctamma"))


@requires_sm107
@pytest.mark.parametrize("config_name", [_SM107_128 + "_1ctamma", "CONFIG_sm107_128x256x128_128x256x64_cluster2x1_2ctamma"])
def test_sm107_block_scale_matmul_quant_epilogue(config_name):
    _run_bs_quant_numeric(
        config_name,
        256,
        256,
        512,
        cudnn.data_type.FP8_E4M3,
        torch.float8_e4m3fn,
        cudnn.data_type.FP8_E8M0,
        torch.float8_e8m0fnu,
    )


@requires_sm107
@pytest.mark.parametrize("mode", [cudnn.reduction_mode.ADD, cudnn.reduction_mode.AMAX], ids=("add", "amax"))
def test_sm107_block_scale_matmul_reduction_scalar(mode):
    _run_bs_reduction_numeric(
        "nvfp4",
        _SM107_128 + "_1ctamma",
        128,
        128,
        256,
        mode,
        red_dims=[1, 1, 1],
        red_stride=None,
        ref_dims=(0, 1, 2),
    )


@requires_sm107
@pytest.mark.parametrize("config_name", [_SM107_128 + "_1ctamma", "CONFIG_sm107_128x256x128_128x256x64_cluster2x1_2ctamma"])
def test_sm107_mxfp8_m_major_a_n_major_b(config_name):
    test_mxfp8_m_major_a_n_major_b(config_name, 256, 256, 512)


def test_auto_path_prefers_sm107_where_it_has_a_template(monkeypatch):
    """On SM 10.7 the auto path builds block-scale graphs with the sm107
    pipeline; graph types sm107 has no template for (plain matmul, MoE) fall
    back to sm100 on their own, and so does older Blackwell."""
    from cudnn.gemm.frost.kernel_registry import preferred_pipeline
    from cudnn.gemm.frost.tile_config import as_pipeline, select_config

    pg = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    pa = pg.tensor(name="A", dim=[1, 128, 128], stride=[128 * 128, 128, 1])
    pb = pg.tensor(name="B", dim=[1, 128, 128], stride=[128 * 128, 1, 128])
    pg.matmul(A=pa, B=pb, name="mm").set_output(True)

    bs_chain = analyze(_bs_chain())
    plain_chain = analyze(pg)

    monkeypatch.setattr(C, "_current_arch", lambda: 107)
    assert preferred_pipeline(bs_chain) == "sm107"
    assert preferred_pipeline(plain_chain) == "sm100"

    monkeypatch.setattr(C, "_current_arch", lambda: 100)
    assert preferred_pipeline(bs_chain) == "sm100"

    # select_config scores pure geometry (an sm100 config); moving it to another
    # family touches only that family's fixed MMA-inst K.
    geo = select_config(4096, 4096, 1, block_scale=True)[0]
    assert as_pipeline(geo, "sm100") is geo
    geo107 = as_pipeline(geo, "sm107")
    assert geo107.pipeline == "sm107" and geo107.mma_inst_k_bytes == 64
    assert (geo107.cta_tile_mn, geo107.cgrp_size_mn) == (geo.cta_tile_mn, geo.cgrp_size_mn)
    assert geo107.cta_tile_k_bytes == geo.cta_tile_k_bytes
    # sm103 fixes a 384-byte K-tile, so a scored geometry cannot become one — the
    # family invariant lives on the config, not in a second whitelist.
    with pytest.raises(NotImplementedError, match="cta_tile_k_bytes=384"):
        as_pipeline(geo, "sm103")


# ---------------------------------------------------------------------------
# FP4 with E5M3 scale factors (SM 10.7+)
#
# UTCOMMA's instruction descriptor picks the SF format (0=E4M3, 1=E8M0,
# 2=E5M3); only SM 10.7 decodes 2, and unlike nvfp4/mxfp4 -- which each fix one
# K-block -- E5M3 is legal with both 16 and 32. Nothing else moves: the SF is
# still a byte in the F8_128x4 layout, so the templates are untouched.
# ---------------------------------------------------------------------------


def _e5m3_graph(M, N, K, block_size, **kw):
    return _build_nvfp4_graph(
        M,
        N,
        K,
        block_size=block_size,
        sf_dt=cudnn.data_type.FP8_E5M3,
        a_dt=cudnn.data_type.FP4_E2M1,
        **kw,
    )


@pytest.mark.parametrize("block_size", [16, 32])
def test_e5m3_analyzer_reads_the_scale_dtype(block_size):
    bs = analyze(_e5m3_graph(256, 256, 512, block_size)).block_scale
    assert bs.sf_dtype_a == "fp8_e5m3" and bs.sf_dtype_b == "fp8_e5m3"
    assert bs.block_size == block_size
    assert bs.a_dtype == "fp4_e2m1"
    assert bs.sf_scale_format == 2


@pytest.mark.parametrize(
    "sf_dt,sf_name", [(cudnn.data_type.FP8_E4M3, "fp8_e4m3"), (cudnn.data_type.FP8_E8M0, "fp8_e8m0"), (cudnn.data_type.FP8_E5M3, "fp8_e5m3")]
)
@pytest.mark.parametrize("block_size", [16, 32])
def test_fp4_scale_dtype_and_block_are_orthogonal(sf_dt, sf_name, block_size):
    """All three FP4 scale-factor dtypes are legal at BOTH K-blocks — nvfp4
    (e4m3/16) and mxfp4 (e8m0/32) are just the best-known corners, not a
    coupling. The registry must carry the full 3x2 matrix on every pipeline that
    has fp4 at all."""
    from cudnn.gemm.frost.kernel_registry import GraphType as _GT, MMA_TYPE_SUPPORT, _bs_key

    key = _bs_key("fp4_e2m1", sf_name, "fp4_e2m1", sf_name, block_size)
    for pipeline in ("sm100", "sm103", "sm107"):
        assert key in MMA_TYPE_SUPPORT[pipeline][_GT.BLOCK_SCALE_MATMUL], f"{pipeline} is missing fp4+{sf_name}/{block_size}"
    # and the analyzer reads the pair back off a graph built with it
    bs = analyze(_build_nvfp4_graph(256, 256, 512, block_size=block_size, sf_dt=sf_dt, a_dt=cudnn.data_type.FP4_E2M1)).block_scale
    assert (bs.a_dtype, bs.sf_dtype, bs.block_size) == ("fp4_e2m1", sf_name, block_size)


# Block-scale cases that only some GPUs decode: SM 10.7 added the E5M3 scale
# format (either K-block) and E4M3 at block 32. Keyed by (SF dtype, K-block).
_GPU_GATED_FP4_CASES = {("fp8_e5m3", 16), ("fp8_e5m3", 32), ("fp8_e4m3", 32)}
_DTYPE_GATED_SF_DTYPES = {"fp8_e5m3"}
_GPU_GATED_RANGES = ((107, 110),)


@_GPU
@pytest.mark.parametrize("sf_dt,sf_name", [(cudnn.data_type.FP8_E4M3, "fp8_e4m3"), (cudnn.data_type.FP8_E8M0, "fp8_e8m0")])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("config_name", ["CONFIG_sm100_128x128x128_128x128x32_cluster1x1", pytest.param(_SM107_128, marks=requires_sm107)])
def test_fp4_all_scale_block_corners_numerics(config_name, sf_dt, sf_name, block_size):
    """Numerics for the whole non-E5M3 fp4 matrix, including the two corners the
    nvfp4 / mxfp4 pair leaves out: e4m3 at block 32 and e8m0 at block 16.

    e4m3 at block 32 is one of the GPU-gated cases — it is a 10.7 addition on
    EVERY pipeline, so it runs here only on a 10.7 part."""
    if (sf_name, block_size) in _GPU_GATED_FP4_CASES and not any(lo <= _SM < hi for lo, hi in _GPU_GATED_RANGES):
        spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in _GPU_GATED_RANGES)
        pytest.skip(f"fp4+{sf_name} at block {block_size} decodes only on {spans}, have sm_{_SM}")
    dev = "cuda"
    torch.manual_seed(0)
    M, N, K = 256, 256, 512
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
    if sf_name == "fp8_e8m0":
        sfa, sfb = _rand_e8m0((M, sf_k), dev), _rand_e8m0((N, sf_k), dev)
    else:
        sfa = torch.randint(1, 4, (M, sf_k), device=dev).to(torch.float8_e4m3fn)
        sfb = torch.randint(1, 4, (N, sf_k), device=dev).to(torch.float8_e4m3fn)

    g = _build_nvfp4_graph(M, N, K, block_size=block_size, sf_dt=sf_dt, a_dt=cudnn.data_type.FP4_E2M1)
    compiled = _plan(g, config=by_name(config_name), cta_group=1)
    assert (compiled.chain.block_scale.sf_dtype, compiled.chain.block_scale.block_size) == (sf_name, block_size)

    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled(
        _vp_bs(
            compiled, a_u8.view(torch.float4_e2m1fn_x2), b_u8.view(torch.float4_e2m1fn_x2), c, _to_blocked(sfa).view(1, 1, -1), _to_blocked(sfb).view(1, 1, -1)
        )
    )
    torch.cuda.synchronize()

    a_s = _unpack_fp4(a_u8, lut).view(M, K) * sfa.float().repeat_interleave(block_size, 1)
    b_s = _unpack_fp4(b_u8, lut).view(N, K) * sfb.float().repeat_interleave(block_size, 1)
    torch.testing.assert_close(c[0], (a_s @ b_s.t()).to(torch.float16), atol=2e-1, rtol=2e-2)


def test_gpu_gated_cases_are_narrowed_everywhere():
    """The load-bearing invariant behind putting the GPU-gated fp4 cases in the
    ordinary case sets: EVERY one of them, on EVERY pipeline that carries it,
    needs its own MMA_GPU_ARCH_SPECIAL_CASES entry. Miss one — a new K-block, a
    new family inheriting _BLOCK_SCALE_CASES — and that combo is accepted on a
    part whose descriptor cannot encode it: silently wrong scales, not a clean
    rejection. The registry cannot derive this, so it is pinned here."""
    from cudnn.gemm.frost.kernel_registry import GraphType as _GT, MMA_GPU_ARCH_SPECIAL_CASES, MMA_TYPE_SUPPORT, _bs_key

    gated = [
        (pipeline, _bs_key("fp4_e2m1", sf, "fp4_e2m1", sf, blk))
        for pipeline, by_type in MMA_TYPE_SUPPORT.items()
        for sf, blk in _GPU_GATED_FP4_CASES
        if _bs_key("fp4_e2m1", sf, "fp4_e2m1", sf, blk) in by_type.get(_GT.BLOCK_SCALE_MATMUL, ())
    ]
    assert len(gated) == 3 * len(_GPU_GATED_FP4_CASES), f"expected every pipeline to carry every gated case, got {len(gated)}"
    bad = [pk for pk in gated if MMA_GPU_ARCH_SPECIAL_CASES.get(pk) != _GPU_GATED_RANGES]
    assert not bad, f"GPU-gated cases missing their {_GPU_GATED_RANGES} narrowing: {bad}"


def test_dtype_and_mma_arch_gates_are_independent():
    """A narrow DTYPE and a narrow MMA INSTRUCTION are separate facts that happen
    to share a range today. Keep both: the dtype's range can widen on a later
    part (E5M3 elsewhere than a block-scale MMA operand), while this fp4+E5M3
    instruction's cannot. Collapsing either into the other would let one widen
    the other silently."""
    from cudnn.gemm.frost.dtypes import DTYPE_GPU_ARCH_RANGES
    from cudnn.gemm.frost.kernel_registry import MMA_GPU_ARCH_SPECIAL_CASES, _bs_key

    for sf in _DTYPE_GATED_SF_DTYPES:
        assert DTYPE_GPU_ARCH_RANGES.get(sf) == _GPU_GATED_RANGES, f"{sf} is not narrowed by the dtype table"
        for pipeline in ("sm100", "sm103", "sm107"):
            for blk in (16, 32):
                key = (pipeline, _bs_key("fp4_e2m1", sf, "fp4_e2m1", sf, blk))
                assert MMA_GPU_ARCH_SPECIAL_CASES.get(key) == _GPU_GATED_RANGES, f"{key} lost its independent MMA-instruction narrowing"


@pytest.mark.parametrize("sf_name,block_size", sorted(_GPU_GATED_FP4_CASES))
@pytest.mark.parametrize("pipeline", ["sm100", "sm103", "sm107"])
def test_gpu_gated_cases_reject_off_sm107(pipeline, sf_name, block_size, monkeypatch):
    """...and the narrowing actually bites: accepted on 10.7/10.9, turned away
    by the ARCH gate everywhere else, on every pipeline."""
    from cudnn.gemm.frost.kernel_registry import GraphType as _GT, mma_arch_reject

    sf_dt = {"fp8_e5m3": cudnn.data_type.FP8_E5M3, "fp8_e4m3": cudnn.data_type.FP8_E4M3}[sf_name]
    chain = analyze(_build_nvfp4_graph(256, 256, 512, block_size=block_size, sf_dt=sf_dt, a_dt=cudnn.data_type.FP4_E2M1))
    for arch in (107, 109):
        monkeypatch.setattr(C, "_current_arch", lambda a=arch: a)
        assert mma_arch_reject(chain, _GT.BLOCK_SCALE_MATMUL, pipeline) is None
    for arch in (100, 103, 120):
        monkeypatch.setattr(C, "_current_arch", lambda a=arch: a)
        reason = mma_arch_reject(chain, _GT.BLOCK_SCALE_MATMUL, pipeline)
        assert reason is not None, f"{pipeline} accepted fp4+{sf_name}/{block_size} on sm_{arch}"
        assert f"sm_{arch}" in reason and "107 <= SM < 110" in reason, reason


@pytest.mark.parametrize("block_size", [16, 32])
def test_dtype_gated_scales_reject_off_sm107(block_size, monkeypatch):
    """The dtype gate bites wherever the dtype is NAMED — it reads the chain, so
    it does not care which pipeline, graph shape or code path would have run."""
    from cudnn.gemm.frost.dtypes import dtype_arch_reject

    chain = analyze(_build_nvfp4_graph(256, 256, 512, block_size=block_size, sf_dt=cudnn.data_type.FP8_E5M3, a_dt=cudnn.data_type.FP4_E2M1))
    assert "fp8_e5m3" in chain.dtypes_used()
    for arch in (107, 109):
        assert dtype_arch_reject(chain, arch) is None
    for arch in (100, 103, 110, 120):
        reason = dtype_arch_reject(chain, arch)
        assert reason is not None, f"accepted an E5M3 scale on sm_{arch}"
        assert f"sm_{arch}" in reason and "107 <= SM < 110" in reason, reason
    monkeypatch.setattr(C, "_current_arch", lambda: 100)
    with pytest.raises(NotImplementedError, match="sm_100"):
        jit_from_cudnn_graph(
            _build_nvfp4_graph(256, 256, 512, block_size=block_size, sf_dt=cudnn.data_type.FP8_E5M3, a_dt=cudnn.data_type.FP4_E2M1),
            **_kw(_SM107_128 + "_1ctamma"),
        )


@requires_sm107
@pytest.mark.parametrize("block_size", [16, 32])
def test_e5m3_runs_on_the_sm100_pipeline(block_size):
    """The sm100 templates reach scale_format=2 through the MX descriptor rather
    than the OMMA one, and SM 10.7 decodes it there too — so an sm100-pipeline
    config is a legitimate way to run E5M3 on this part."""
    _run_e5m3_numeric("CONFIG_sm100_128x128x128_128x128x32_cluster1x1", 1, block_size)


def _run_e5m3_numeric(config_name, cta_group, block_size, M=256, N=256, K=512):
    dev = "cuda"
    torch.manual_seed(0)
    sf_k = K // block_size
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    a_u8 = torch.randint(0, 256, (1, M, K // 2), dtype=torch.uint8, device=dev)
    b_u8 = torch.randint(0, 256, (1, N, K // 2), dtype=torch.uint8, device=dev)
    a_deq = _unpack_fp4(a_u8, lut).view(M, K)
    b_deq = _unpack_fp4(b_u8, lut).view(N, K)
    sfa, sfb = _rand_e5m3((M, sf_k), dev), _rand_e5m3((N, sf_k), dev)

    g = _e5m3_graph(M, N, K, block_size)
    compiled = _plan(g, config=by_name(config_name), cta_group=cta_group)
    assert (compiled.chain.block_scale.sf_dtype, compiled.chain.block_scale.block_size) == ("fp8_e5m3", block_size)

    # The SF blob is read by base pointer and to_blocked() ceil-pads to whole
    # 128x4 atoms, so its element count is not M*sf_k for a ragged shape — pass
    # the byte run itself rather than a logical view of it.
    c = torch.zeros(1, M, N, dtype=torch.float16, device=dev)
    compiled(
        _vp_bs(
            compiled,
            a_u8.view(torch.float4_e2m1fn_x2),
            b_u8.view(torch.float4_e2m1fn_x2),
            c,
            _to_blocked(sfa).view(1, 1, -1),
            _to_blocked(sfb).view(1, 1, -1),
        )
    )
    torch.cuda.synchronize()

    a_s = a_deq * _e5m3_to_float(sfa).repeat_interleave(block_size, 1)
    b_s = b_deq * _e5m3_to_float(sfb).repeat_interleave(block_size, 1)
    torch.testing.assert_close(c[0], (a_s @ b_s.t()).to(torch.float16), atol=2e-1, rtol=2e-2)


@requires_sm107
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize(
    "config_name,cta_group",
    [
        (_SM107_128, 1),
        (_SM107_256, 1),
        ("CONFIG_sm107_128x128x128_128x128x64_cluster1x2", 1),
        ("CONFIG_sm107_128x128x128_128x128x64_cluster2x1", 2),
        ("CONFIG_sm107_128x256x128_128x256x64_cluster2x1", 2),
    ],
    ids=lambda v: v if isinstance(v, str) else f"cta{v}",
)
def test_e5m3_block_scale_matmul_numerics(config_name, cta_group, block_size):
    _run_e5m3_numeric(config_name, cta_group, block_size)


@requires_sm107
@pytest.mark.parametrize("block_size", [16, 32])
def test_e5m3_block_scale_matmul_oob_shape(block_size):
    """M-OOB / K past a tile boundary is TMA zero-fill, same as every other
    block-scale combo."""
    _run_e5m3_numeric(_SM107_128, 1, block_size, M=255, N=256, K=512 + 4 * block_size)
