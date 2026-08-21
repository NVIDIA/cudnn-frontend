# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused SwiGLU (cuDNN DualMatmulSiluMulDequant): out = silu(a@b0) * (a@b1) * scale.

A is shared by both GEMMs; b0/b1 distinct. Runs the multi-GEMM path, checked vs
a torch reference.
"""

from __future__ import annotations

import cudnn
import cudnn.gemm.frost  # noqa: F401  (installs hook)
import pytest
import torch

from gemm_test_utils import requires_sm100, Plan as _plan, vp_mg as _vp_mg, block_quant_ref as _block_quant_ref

from cudnn.gemm.frost.tile_config import CATALOG

pytestmark = [pytest.mark.L0, requires_sm100]


_CUDNN_DT = {"bf16": cudnn.data_type.BFLOAT16, "fp16": cudnn.data_type.HALF}
_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}

# N=128 cta_group=1 geometry — small enough that dual-GEMM (2 acc) and triple fit.
_CFG_N128 = next(c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 128 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 1 and c.cgrp_size_n == 1)
_CFG_N256 = next(c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 256 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 1 and c.cgrp_size_n == 1)
# cluster2x1 N=256 geometry for the 2-CTA-MMA templates.
_CFG_N256_C2 = next(
    c for c in CATALOG if c.cta_tile_m == 128 and c.cta_tile_n == 256 and c.cta_tile_k_bytes == 128 and c.cgrp_size_m == 2 and c.cgrp_size_n == 1
)

# Every multi-GEMM-capable plain-matmul strategy: (label, cta_group, config).
_STRATEGIES = [
    ("1ctamma", 1, _CFG_N256),
    ("2ctamma", 2, _CFG_N256_C2),
]


def _build_swiglu(B, M, N, K, in_dt, out_dt, quant_block=None):
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DT[in_dt],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    aT = g.tensor(name="aTensor", dim=[B, M, K], stride=[M * K, K, 1])
    b0T = g.tensor(name="b0Tensor", dim=[B, K, N], stride=[K * N, 1, K])
    b1T = g.tensor(name="b1Tensor", dim=[B, K, N], stride=[K * N, 1, K])
    sf = g.tensor(
        name="scaleFactor",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    c0 = g.matmul(A=aT, B=b0T, name="mm0")
    c1 = g.matmul(A=aT, B=b1T, name="mm1")
    c0silu = g.swish(input=c0, name="silu0")
    mul = g.mul(a=c0silu, b=c1, name="mul0")
    dq = g.mul(a=mul, b=sf, name="dequant0")
    if quant_block is not None:
        Q, QS = g.block_scale_quantize(input=dq, block_size=quant_block, name="q")
        Q.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        QS.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
    else:
        dq.set_output(True).set_data_type(_CUDNN_DT[out_dt])
    return g


def _reference(a, b0, b1, scale, out_dt):
    c0 = torch.einsum("bmk,bnk->bmn", a.float(), b0.float())
    c1 = torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    out = torch.nn.functional.silu(c0) * c1 * scale.flatten()[0]
    return out.to(_TORCH_DT[out_dt])


def _run(B, M, N, K, in_dt, out_dt, cfg, *, seed=0, cta_group=1):
    torch.manual_seed(seed)
    a = torch.randn(B, M, K, device="cuda", dtype=_TORCH_DT[in_dt]) * 0.4
    b0 = torch.randn(B, N, K, device="cuda", dtype=_TORCH_DT[in_dt]) * 0.4
    b1 = torch.randn(B, N, K, device="cuda", dtype=_TORCH_DT[in_dt]) * 0.4
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    out = torch.zeros(B, M, N, device="cuda", dtype=_TORCH_DT[out_dt])
    compiled = _plan(
        _build_swiglu(B, M, N, K, in_dt, out_dt),
        config=cfg,
        cta_group=cta_group,
    )
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], out, scale))
    torch.cuda.synchronize()
    return out, _reference(a, b0, b1, scale, out_dt)


def _nonpacked_inputs(B, M, N, K, in_dt, out_dt, mode):
    td_in, td_out = _TORCH_DT[in_dt], _TORCH_DT[out_dt]
    torch.manual_seed(11)
    if mode == "zero_stride":
        a_base = torch.randn(K, device="cuda", dtype=td_in) * 0.4
        b0_base = torch.randn(K, device="cuda", dtype=td_in) * 0.4
        b1_base = torch.randn(K, device="cuda", dtype=td_in) * 0.4
        a = torch.as_strided(a_base, (B, M, K), (0, 0, 1))
        b0 = torch.as_strided(b0_base, (B, N, K), (0, 0, 1))
        b1 = torch.as_strided(b1_base, (B, N, K), (0, 0, 1))
    else:
        a_store = torch.randn(B, M, K + 16, device="cuda", dtype=td_in) * 0.4
        b0_store = torch.randn(B, N, K + 16, device="cuda", dtype=td_in) * 0.4
        b1_store = torch.randn(B, N, K + 32, device="cuda", dtype=td_in) * 0.4
        a = a_store[:, :, :K]
        b0 = b0_store[:, :, :K]
        b1 = b1_store[:, :, :K]
    out_store = torch.zeros(B, M, N + 16, device="cuda", dtype=td_out)
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)
    return a, b0, b1, out_store[:, :, :N], scale


@pytest.mark.parametrize("label,cta_group,cfg", _STRATEGIES, ids=[s[0] for s in _STRATEGIES])
def test_swiglu_all_templates(label, cta_group, cfg) -> None:
    """SwiGLU on every multi-GEMM template: 1ctamma / 2ctamma."""
    out, ref = _run(
        512,
        256,
        256,
        128,
        "bf16",
        "bf16",
        cfg,
        cta_group=cta_group,
    )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "B,M,N,K",
    [
        (1, 256, 256, 128),
        (1, 512, 256, 256),
        (1, 384, 128, 128),  # M not a tile multiple
    ],
)
def test_swiglu_bf16(B, M, N, K) -> None:
    cfg = _CFG_N256 if N >= 256 else _CFG_N128
    out, ref = _run(B, M, N, K, "bf16", "bf16", cfg)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_swiglu_fp16() -> None:
    out, ref = _run(1, 256, 256, 128, "fp16", "fp16", _CFG_N256)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_swiglu_batched() -> None:
    """B>1: independent same-shape SwiGLU blocks."""
    out, ref = _run(2, 256, 256, 128, "bf16", "bf16", _CFG_N256)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "label,cta_group,cfg,mode",
    [
        ("1ctamma-padded", 1, _CFG_N256, "padded"),
        ("2ctamma-padded", 2, _CFG_N256_C2, "padded"),
        ("1ctamma-zero", 1, _CFG_N256, "zero_stride"),
        ("2ctamma-zero", 2, _CFG_N256_C2, "zero_stride"),
    ],
)
def test_swiglu_nonpacked_tensors(label, cta_group, cfg, mode) -> None:
    B, M, N, K = 1, 256, 256, 128
    a, b0, b1, out, scale = _nonpacked_inputs(B, M, N, K, "bf16", "bf16", mode)
    assert not a.is_contiguous() or not b0.is_contiguous() or not b1.is_contiguous()
    assert not out.is_contiguous()
    compiled = _plan(
        _build_swiglu(B, M, N, K, "bf16", "bf16"),
        config=cfg,
        cta_group=cta_group,
    )
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], out, scale))
    torch.cuda.synchronize()
    torch.testing.assert_close(out, _reference(a, b0, b1, scale, "bf16"), rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("label,cta_group,cfg", _STRATEGIES, ids=[s[0] for s in _STRATEGIES])
def test_swiglu_quant_epilogue(label, cta_group, cfg):
    """Terminal block_scale_quantize on the dual-GEMM SwiGLU chain: the fused
    result is re-quantized to FP8 E4M3 + per-32-block E8M0 scale (two outputs)."""
    B, M, N, K = 1, 256, 256, 512
    qblock = 32
    torch.manual_seed(0)
    # Small-integer inputs keep the matmuls exact; only swish fast-math wiggles.
    a = torch.empty(B, M, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b0 = torch.empty(B, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    b1 = torch.empty(B, N, K, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([[[0.5]]], device="cuda", dtype=torch.float32)

    compiled = _plan(
        _build_swiglu(B, M, N, K, "bf16", "bf16", quant_block=qblock),
        config=cfg,
        cta_group=cta_group,
    )
    assert compiled.chain.quants
    assert compiled.chain.output_dtype == "fp8_e4m3"

    q = torch.zeros(B, M, N, dtype=torch.float8_e4m3fn, device="cuda")
    q_scale = torch.zeros(B, M, N // qblock, dtype=torch.float8_e8m0fnu, device="cuda")
    compiled(_vp_mg(compiled, [(a, b0), (a, b1)], [q, q_scale], scale))
    torch.cuda.synchronize()

    c0 = torch.einsum("bmk,bnk->bmn", a.float(), b0.float())
    c1 = torch.einsum("bmk,bnk->bmn", a.float(), b1.float())
    ref = (torch.nn.functional.silu(c0) * c1 * 0.5)[0]
    q_ref, scale_ref = _block_quant_ref(ref, qblock, torch.float8_e4m3fn, torch.float8_e8m0fnu)
    # The kernel's swish uses fast __expf → pre-quant values sit within ~1e-3
    # rel of torch; allow one E4M3 mantissa step where that crosses a boundary.
    torch.testing.assert_close(q_scale.float(), scale_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(q.float(), q_ref.float(), atol=2**-8, rtol=1 / 8)
