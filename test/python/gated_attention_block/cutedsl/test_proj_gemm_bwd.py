# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four backward projection GEMMs of the gated attention block, on the shipped FROST GEMM.

``nn.Linear`` orientation (``api_bwd.py``): ``dW = dY^T @ X`` (A M-major, B N-major) and
``dX = dY @ W`` (B N-major).  Nothing here writes a kernel; what is under test is the
DRIVER's claims (``kernels/proj_gemm.py``):

* the appended ``a_major`` / ``b_major`` kwargs declare the graph strides the FROST GEMM
  engine renders as its M-major-A / N-major-B template arms, and the tile the block
  FORCES for every ``n % 256 == 0`` (``_forced_tile_config``) renders and computes
  correctly on cc 10.7 under those majors -- the riskiest assumption behind the block's
  backward, settled by :func:`test_forced_tile_renders_mn_major_on_cc107`;
* ``run_wgrad_gemm`` / ``run_dgrad_gemm`` bind transposed VIEWS (zero-copy) against a plan
  DECLARED with the matching majors, and refuse a view whose strides are not EXACTLY the
  declared ones (a wrong major, or a column slice of a wider slab) BEFORE any launch, on
  both routes -- the graph fallback would read the declared strides with no check (a
  silent reinterpretation) and the JIT re-labels B into kernel order only on
  an exact match;
* ``split_k`` semantics: ``0`` the driver's pick, ``1`` a pinned JIT at one slice that
  refuses the graph fallback, ``S >= 2`` the fixed-order two-kernel split whose fp32
  partials ride ``plan.workspace_bytes``;
* the QUANTIZED backward's drivers: the per-tensor e4m3 MN-major renderings the driver
  admits (``FP8_MN_MAJOR_VALIDATED``, a table) match the fp64 reference of the dequantized
  products at the forced tile in BOTH MMA K forms, at full and RAGGED token counts (the bf16
  twins' ``T = 4104``) and under ``split_k=2`` (:func:`test_fp8_split_k_matches_fp64`) --
  :func:`test_fp8_mn_major_matches_fp64_on_cc107` is the validation that lifted the refusal,
  everything outside the table stays a typed decline, and the fp8 ``K % 16`` rule is the
  K-contiguous operands' (a wgrad's ragged ``K = T`` is admitted); ``mma_tile_k_bytes=64`` is an
  EXPLICIT request of a dense e4m3 plan (never keyed on the dtype) and the forward's fp8 plans
  stay at K32; the ``alpha`` epilogue reads a device slot the caller owns (:func:`device_alpha`);
  the block-scale (MXFP8) drivers bind K-major transposed operands with their F8_128x4 scale
  factors, name a missing or wrong-sized blob by THEIR keyword, and decline a ragged token
  axis (``T % 32``), ``split_k`` and every non-K major, typed.

The accept tests need a Rubin device (the block targets SM107 only); the reject / host
tests run anywhere -- those that bind CUDA tensors to hand-built plans need a CUDA device of
any arch (``requires_cuda``), the rest none.
"""

import functools
import os
import re
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    FP8_MN_MAJOR_VALIDATED,
    ProjGemmPlan,
    SplitKPinRefused,
    _forced_tile_config,
    _frost_plan_index,
    build_proj_gemm,
    device_alpha,
    run_dgrad_gemm,
    run_dgrad_gemm_block_scale,
    run_wgrad_gemm,
    run_wgrad_gemm_block_scale,
    sf_blob_bytes,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_stream_probe import park_the_default_stream  # noqa: E402

_FORCED_TILE = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
_FORCED_TILE_K64 = "CONFIG_sm100_128x256x128_128x256x64_cluster2x1_2ctamma"  # the same geometry at the 64-byte MMA K (tile_config.as_mma_tile_k)
_FP8 = getattr(torch, "float8_e4m3fn", None)
_FP8_MAX = 448.0
_SENTINEL = 1.5e30

# The REGISTERED marker of cutedsl/conftest.py (the skip is applied at collection) -- switched from the per-module
# skipif copy when this module was next touched.
requires_rubin = pytest.mark.requires_rubin
# The reject tests that bind CUDA tensors to HAND-BUILT plans (nothing launched, no Rubin needed) still need a device:
# on a CUDA-less host they skip here instead of failing at `torch.zeros(..., device="cuda")`.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


# ---------------------------------------------------------------------------
# Tolerance: bf16 / f16 inputs, fp32 accumulate, bf16 / f16 output, vs an fp64 oracle
# ---------------------------------------------------------------------------
#
# The oracle multiplies the SAME quantized inputs in fp64, so the only differences are
# (i) the output rounding -- half an ulp = 2^-9 relative for bf16 (8 significand bits),
# 2^-12 for f16 -- and (ii) the fp32 accumulation order over K terms, whose error is
# ABSOLUTE and about sqrt(K) * 2^-24 * |typical partial sum| (K = 8192: ~1e-5 of the
# tensor's scale, three orders below either bound).  So rtol = 4x the half-ulp of the
# OUTPUT dtype -- 2^-7 for bf16, 2^-10 for f16, derived the same way;
# one bound per dtype, never one for both (a shared 2^-7 would let an f16 cell 16x its
# half-ulp through, so the f16 arms would assert less than they appear to) -- and
# atol = rtol * max|ref| (tensor-scaled, test/AGENTS.md: a fixed absolute bound turns
# wrong when the magnitudes grow).  Measured worst cells on Rubin (cc 10.7, 204 SMs), 2026-09-29: bf16
# 0.16-0.23 of its bound, f16 0.02-0.03 of the 2^-7 bound = 0.16-0.24 of 2^-10 -- the
# same 4-6x margin at both dtypes.  Never widened; a violation is reported with its magnitude.
_RTOL_BY_DTYPE = {torch.bfloat16: 2.0**-7, torch.float16: 2.0**-10}


def _assert_close_vs_fp64(out: torch.Tensor, ref64: torch.Tensor, what: str) -> None:
    rtol = _RTOL_BY_DTYPE[out.dtype]
    ref_max = ref64.abs().max().item()
    atol = rtol * ref_max
    diff = (out.double() - ref64).abs()
    scale = atol + rtol * ref64.abs()
    worst = (diff / scale).max().item()
    msg = f"{what}: max|diff| = {diff.max().item():.4g} (max|ref| = {ref_max:.4g}); worst cell at {worst:.3f} of the bound (rtol = atol/max|ref| = {rtol}, {out.dtype})"
    print(msg)
    torch.testing.assert_close(out.double(), ref64, rtol=rtol, atol=atol, msg=msg)


def _wgrad_operands(rows: int, t: int, cols: int, dtype: torch.dtype, seed: int = 0):
    """``dy_like [T, rows]`` and ``x [T, cols]`` (both row-major -- what the block holds), plus ``dw [rows, cols]``."""
    torch.manual_seed(seed)
    dy = (torch.randn(t, rows, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
    x = (torch.randn(t, cols, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
    dw = torch.zeros(rows, cols, device="cuda", dtype=dtype)
    return dy, x, dw


def _dgrad_operands(t: int, k: int, n: int, dtype: torch.dtype, seed: int = 0):
    """``dy_like [T, K]`` and the UN-transposed row-major weight ``w [K, N]``, plus ``dx [T, N]``."""
    torch.manual_seed(seed)
    dy = (torch.randn(t, k, device="cuda", dtype=torch.float32) * 0.5).to(dtype)
    w = (torch.randn(k, n, device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    dx = torch.zeros(t, n, device="cuda", dtype=dtype)
    return dy, w, dx


def _ws(plan: ProjGemmPlan) -> torch.Tensor:
    return torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")


# The block's two geometries: the test geometry every cutedsl/ suite runs, and Qwen3.5-397B's
# full-attention layer.  `(dm, HD, N)` = (d_model, h_q * d_head, n_qkvg); every backward GEMM's
# n is one of dm / HD (both % 256 == 0 at both geometries), so all four take the FORCED tile.
_GEOMS = {
    "test": GatedAttentionBlockGeometry(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64),
    "397B": GatedAttentionBlockGeometry(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64),
}


def _dims(geom_id: str) -> tuple[int, int, int]:
    g = _GEOMS[geom_id]
    return g.d_model, g.h_q * g.d_head, g.n_qkvg


def _stage_mkn(stage: str, geom_id: str, t: int) -> tuple[int, int, int]:
    """``(m, k, n)`` of one backward projection plan at the block's geometry."""
    dm, hd, n_qkvg = _dims(geom_id)
    return {
        "B1_dw_o": (dm, t, hd),  # dW_o [dm, HD] = dY[T, dm]^T @ O_gated[T, HD]
        "B7_dw_qkvg": (n_qkvg, t, dm),  # dW_qkvg [N, dm] = dQKVG[T, N]^T @ h[T, dm]
        "B2_do_gated": (t, dm, hd),  # dO_gated [T, HD] = dY[T, dm] @ W_o[dm, HD]
        "B8_dh": (t, n_qkvg, dm),  # dh [T, dm] = dQKVG[T, N] @ W_qkvg[N, dm]
    }[stage]


_WGRAD = ("B1_dw_o", "B7_dw_qkvg")
_DGRAD = ("B2_do_gated", "B8_dh")


@functools.lru_cache(maxsize=None)
def _plan(kind: str, m: int, k: int, n: int, dtype: torch.dtype, split_k: int = 0) -> ProjGemmPlan:
    """One plan per distinct declaration for the whole process: the rendered kernel is symbolic in
    (M, N, K) and cached by source digest, but every build still asks the backend for its ranked
    list, so the tests below share plans rather than rebuild them."""
    majors = dict(a_major="m", b_major="n") if kind == "wgrad" else dict(a_major="k", b_major="n")
    return build_proj_gemm(m=m, k=k, n=n, dtype=dtype, label=f"{kind}_{m}x{k}x{n}_{str(dtype).replace('torch.', '')}_sk{split_k}", split_k=split_k, **majors)


# ---------------------------------------------------------------------------
# THE probe: the forced tile renders and computes M-major A / N-major B on cc 10.7
# ---------------------------------------------------------------------------


@requires_rubin
def test_forced_tile_renders_mn_major_on_cc107():
    """The riskiest assumption behind the block's backward: the bf16 M-major-A /
    N-major-B renderings of the tile the block FORCES (``..._cluster2x1_2ctamma``)
    have never run on cc 10.7 -- the FROST GEMM suite covers the layouts on the sm100
    pipeline, and the block's own forward only ever rendered K-major operands.

    B1's geometry at the 397B column shapes: ``dW_o [dm=4096, HD=8192] = dY[T=8192,
    dm]^T @ O_gated[T, HD]``.  Three assertions, in the order a failure would show:
    the plan IS the forced JIT (a fallback to the heuristic is a FAILURE here, not a
    skip -- it changes route, config and possibly split-K), the output is not silently
    zero (the >256 KiB tcgen05 descriptor landmine's signature, ``test_proj_gemm.py::
    test_output_is_not_silently_zero``), and the numbers match the fp64 oracle."""
    m, k, n = 4096, 8192, 8192
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="mn_major_probe_dw_o", a_major="m", b_major="n")
    assert plan.tile_config_name == _FORCED_TILE, f"not the forced tile: {plan.tile_config_name!r} (route {plan.route!r})"
    assert plan.jit is not None, f"no JIT artifact -- the forced compile fell back to the graph heuristic (route {plan.route!r})"
    assert (plan.a_major, plan.b_major) == ("m", "n")
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    run_wgrad_gemm(plan, dy, x, dw, _ws(plan))
    torch.cuda.synchronize()
    nonzero = (dw != 0).float().mean().item()
    assert nonzero > 0.99, f"only {100 * nonzero:.1f}% of dW is non-zero -- suspect the MN-major SMEM descriptor / the >256 KiB descriptor version"
    assert torch.isfinite(dw.float()).all()
    ref = dy.double().T @ x.double()
    _assert_close_vs_fp64(dw, ref, "dW_o = dY^T @ O_gated (m=4096, k=8192, n=8192, bf16)")


# ---------------------------------------------------------------------------
# Numerics vs the fp64 oracle -- the four plans at both geometries
# ---------------------------------------------------------------------------
#
# T = 4104 is not a multiple of any tile: a ragged K tail for the wgrads (K = T; the CTA K tile
# is 64 bf16 elements, 4104 = 64 * 64 + 8) and a ragged M tail for the dgrads (M = T, tile 128).
# f16 rides the small geometry only (one more rendering per driver; the same bound).

_WGRAD_CASES = [(st, g, t, torch.bfloat16) for g in ("test", "397B") for st in _WGRAD for t in (2048, 4104)] + [
    (st, "test", 2048, torch.float16) for st in _WGRAD
]
_DGRAD_CASES = [(st, g, t, torch.bfloat16) for g in ("test", "397B") for st in _DGRAD for t in (2048, 4104)] + [
    (st, "test", 2048, torch.float16) for st in _DGRAD
]


def _case_id(c) -> str:
    return f"{c[0]}-{c[1]}-T{c[2]}-{str(c[3]).replace('torch.', '')}"


@requires_rubin
@pytest.mark.parametrize("stage,geom_id,t,dtype", _WGRAD_CASES, ids=[_case_id(c) for c in _WGRAD_CASES])
def test_wgrad_matches_fp64(stage, geom_id, t, dtype):
    """``dW = dY^T @ X`` through ``run_wgrad_gemm`` on the plan each stage builds: both weight
    gradients (B1, B7) at the test geometry and at the 397B column shapes, vs ``dy.double().T @
    x.double()`` within the bf16-output bound above."""
    m, k, n = _stage_mkn(stage, geom_id, t)
    plan = _plan("wgrad", m, k, n, dtype)
    assert plan.jit is not None and plan.tile_config_name == _FORCED_TILE, (plan.tile_config_name, plan.route)
    dy, x, dw = _wgrad_operands(m, k, n, dtype)
    run_wgrad_gemm(plan, dy, x, dw, _ws(plan))
    torch.cuda.synchronize()
    assert (dw != 0).float().mean().item() > 0.99, "dW is (nearly) all zero"
    _assert_close_vs_fp64(dw, dy.double().T @ x.double(), f"{stage} @ {geom_id}, T={t}, {dtype}")


@requires_rubin
@pytest.mark.parametrize("stage,geom_id,t,dtype", _DGRAD_CASES, ids=[_case_id(c) for c in _DGRAD_CASES])
def test_dgrad_matches_fp64(stage, geom_id, t, dtype):
    """``dX = dY @ W`` through ``run_dgrad_gemm`` with the UN-transposed row-major weight: both
    input gradients (B2, B8), same geometries, same bound."""
    m, k, n = _stage_mkn(stage, geom_id, t)
    plan = _plan("dgrad", m, k, n, dtype)
    assert plan.jit is not None and plan.tile_config_name == _FORCED_TILE, (plan.tile_config_name, plan.route)
    dy, w, dx = _dgrad_operands(m, k, n, dtype)
    run_dgrad_gemm(plan, dy, w, dx, _ws(plan))
    torch.cuda.synchronize()
    assert (dx != 0).float().mean().item() > 0.99, "dX is (nearly) all zero"
    _assert_close_vs_fp64(dx, dy.double() @ w.double(), f"{stage} @ {geom_id}, T={t}, {dtype}")


# ---------------------------------------------------------------------------
# The QUANTIZED backward's per-tensor fp8 GEMMs: MN-major renderings at the forced tile, K32 and K64, device alpha
# ---------------------------------------------------------------------------
#
# Oracle and bound: the e4m3 operands are EXACT in fp64, so `(dequant(A) @ dequant(B))` in fp64 is the exact
# value of the dequantized products and the kernel's only departures are the fp32 accumulation order, the
# fp32 alpha multiply and the single bf16 output rounding -- the SAME derivation as `_RTOL_BY_DTYPE`, so the
# bf16 bound (rtol 2^-7, atol rtol * max|ref|) applies unchanged.  Never a new bound.


def _quant_e4m3(x32: torch.Tensor):
    """Per-tensor amax/448 quantization; the descale is a 1-element fp32 DEVICE tensor (no ``.item()``)."""
    descale = (x32.abs().amax().clamp_min(1e-8) / _FP8_MAX).float().reshape(1)
    return (x32 / descale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8), descale


def _fp8_wgrad_operands(rows: int, t: int, cols: int, seed: int = 0):
    """e4m3 ``dy_like [T, rows]`` and ``x [T, cols]`` with their device descales, a sentinel-filled bf16 ``dw``."""
    torch.manual_seed(seed)
    dy8, d_dy = _quant_e4m3(torch.randn(t, rows, device="cuda") * 0.5)
    x8, d_x = _quant_e4m3(torch.randn(t, cols, device="cuda") * 0.5)
    dw = torch.full((rows, cols), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    return dy8, d_dy, x8, d_x, dw


def _fp8_dgrad_operands(t: int, k: int, n: int, seed: int = 0):
    """e4m3 ``dy_like [T, K]`` and the UN-transposed weight ``w [K, N]`` with their device descales, a sentinel-filled ``dx``."""
    torch.manual_seed(seed)
    dy8, d_dy = _quant_e4m3(torch.randn(t, k, device="cuda") * 0.5)
    w8, d_w = _quant_e4m3(torch.randn(k, n, device="cuda") * 0.05)
    dx = torch.full((t, n), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    return dy8, d_dy, w8, d_w, dx


@functools.lru_cache(maxsize=None)
def _fp8_plan(kind: str, m: int, k: int, n: int, k_bytes: int, alpha: bool = True, split_k: int = 0) -> ProjGemmPlan:
    """:func:`_plan`'s e4m3 twin: one plan per (kind, shape, MMA K form, alpha, split_k) for the whole process, the K form
    requested EXPLICITLY through ``mma_tile_k_bytes`` (the only spelling that reaches K64)."""
    majors = dict(a_major="m", b_major="n") if kind == "wgrad" else dict(a_major="k", b_major="n")
    return build_proj_gemm(
        m=m, k=k, n=n, dtype=_FP8, label=f"fp8_{kind}_{m}x{k}x{n}_k{k_bytes}_sk{split_k}", alpha=alpha, mma_tile_k_bytes=k_bytes, split_k=split_k, **majors
    )


def _assert_fp8_plan_is_the_forced_tile(plan: ProjGemmPlan, k_bytes: int, split_k: int = 0) -> None:
    """The plan IS the forced JIT at the requested K form -- and, for ``split_k >= 2``, the catalog's ``_splitK<S>``
    spelling of it running S slices; a fallback to the heuristic is a FAILURE here, not a skip."""
    slices = split_k if split_k >= 2 else 1
    expect = (_FORCED_TILE if k_bytes == 32 else _FORCED_TILE_K64) + (f"_splitK{slices}" if slices > 1 else "")
    assert plan.jit is not None, f"no JIT artifact -- the forced compile fell back to the graph heuristic (route {plan.route!r})"
    assert plan.tile_config_name == expect, f"not the forced tile at K{k_bytes}, split_k={split_k}: {plan.tile_config_name!r} (route {plan.route!r})"
    assert plan.mma_tile_k_bytes == k_bytes and plan.jit.config.mma_tile_k_bytes == k_bytes, (plan.mma_tile_k_bytes, plan.jit.config.mma_tile_k_bytes)
    assert plan.jit.config.split_k_slices == slices and plan.split_k == split_k, (plan.jit.config.split_k_slices, plan.split_k)
    assert plan.has_alpha and plan.dtype == _FP8 and plan.out_dtype == torch.bfloat16 and plan.route == "graph+jit"


def _check_fp8_cell(out1: torch.Tensor, out2: torch.Tensor, ref64: torch.Tensor, what: str) -> None:
    """Sentinel-free (every cell written), finite, not silently zero, two launches BITWISE equal, and within the bound."""
    surv = int((out1.float() == _SENTINEL).sum().item())
    assert surv == 0, f"{what}: {surv} sentinel survivors (cells the kernel never wrote)"
    assert torch.isfinite(out1.float()).all(), f"{what}: non-finite output"
    nonzero = (out1 != 0).float().mean().item()
    assert (
        nonzero > 0.99
    ), f"{what}: only {100 * nonzero:.1f}% of the output is non-zero -- suspect the MN-major SMEM descriptor / the >256 KiB descriptor version"
    assert torch.equal(out1, out2), f"{what}: two launches differ: max|diff| = {(out1.float() - out2.float()).abs().max().item()}"
    _assert_close_vs_fp64(out1, ref64, what)


# The 397B column shapes at T = 2048 and (B1 / B2, the out_proj class) T = 8192, and the test geometry at T = 2048;
# every (kind, K form) pairs with its own rendering, so the four renderings each see several shapes.  Then the bf16
# twins' RAGGED T = 4104 at both geometries: the wgrads contract over K = T -- 32 CTA K tiles of 128 e4m3 elements plus
# an 8-element tail, TMA zero-fill under an M-major A / N-major B (no operand has K as its contiguous axis, so the fp8
# K % 16 rule does not apply to them; `test_fp8_k_rule_is_the_k_contiguous_operands_rule`) -- and the dgrads' M = T is
# a partial 128-row M tile.
_FP8_SHAPES = (
    [
        ("B1_dw_o", "397B", 2048),
        ("B1_dw_o", "397B", 8192),
        ("B7_dw_qkvg", "397B", 2048),
        ("B2_do_gated", "397B", 2048),
        ("B2_do_gated", "397B", 8192),
        ("B8_dh", "397B", 2048),
    ]
    + [(st, "test", 2048) for st in _WGRAD + _DGRAD]
    + [(st, g, 4104) for g in ("test", "397B") for st in _WGRAD + _DGRAD]
)
_FP8_CASES = [(st, g, t, kb) for (st, g, t) in _FP8_SHAPES for kb in (32, 64)]


@requires_rubin
@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
@pytest.mark.parametrize("stage,geom_id,t,k_bytes", _FP8_CASES, ids=[f"{c[0]}-{c[1]}-T{c[2]}-K{c[3]}" for c in _FP8_CASES])
def test_fp8_mn_major_matches_fp64_on_cc107(stage, geom_id, t, k_bytes):
    """THE validation behind the fp8 MN-major lift (``FP8_MN_MAJOR_VALIDATED``): the e4m3 M-major-A / N-major-B
    (wgrad) and N-major-B (dgrad) renderings of the forced tile, in both MMA K forms (``mma_tile_k_bytes`` 32 and
    64), with the descale product bound as the device ``alpha`` epilogue, against the fp64 reference of the
    dequantized products under the bf16-output bound.  Each cell also pins: the plan IS the forced JIT at the
    requested K form (a fallback is a FAILURE), no sentinel survivor, not silently zero, two launches bitwise
    equal.  The 397B column shapes are the block's real GEMMs; the test geometry rides along; ``T = 4104`` is the
    bf16 twins' ragged token count -- a partial CTA K tile (the wgrads, K = T) or a partial M tile (the dgrads)."""
    m, k, n = _stage_mkn(stage, geom_id, t)
    kind = "wgrad" if stage in _WGRAD else "dgrad"
    plan = _fp8_plan(kind, m, k, n, k_bytes)
    _assert_fp8_plan_is_the_forced_tile(plan, k_bytes)
    ws = _ws(plan)
    alpha_slot = torch.zeros(1, dtype=torch.float32, device="cuda")  # the caller's workspace slot
    if kind == "wgrad":
        dy8, d_dy, x8, d_x, out1 = _fp8_wgrad_operands(m, k, n)
        out2 = out1.clone()
        alpha = device_alpha(alpha_slot, d_dy, d_x)
        run_wgrad_gemm(plan, dy8, x8, out1, ws, alpha=alpha)
        run_wgrad_gemm(plan, dy8, x8, out2, ws, alpha=alpha)
        torch.cuda.synchronize()
        ref64 = (dy8.double().T @ x8.double()) * (d_dy.double() * d_x.double())
    else:
        dy8, d_dy, w8, d_w, out1 = _fp8_dgrad_operands(m, k, n)
        out2 = out1.clone()
        alpha = device_alpha(alpha_slot, d_dy, d_w)
        run_dgrad_gemm(plan, dy8, w8, out1, ws, alpha=alpha)
        run_dgrad_gemm(plan, dy8, w8, out2, ws, alpha=alpha)
        torch.cuda.synchronize()
        ref64 = (dy8.double() @ w8.double()) * (d_dy.double() * d_w.double())
    _check_fp8_cell(out1, out2, ref64, f"fp8 {stage} @ {geom_id}, T={t}, K{k_bytes}, {plan.tile_config_name}")


@requires_rubin
@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
def test_fp8_k32_and_k64_are_two_renderings_of_one_function():
    """K32 and K64 are PERF knobs of one function: the same operands through both forms meet the same bound,
    and their difference is fp32 reassociation of exact e4m3 products -- below the bound by a wide margin
    (reported).  Also the two plans are distinct JIT configs (``_splitK``-style name pinning, the K form in
    the name), not one plan re-labelled."""
    m, k, n = _stage_mkn("B2_do_gated", "test", 2048)
    p32, p64 = _fp8_plan("dgrad", m, k, n, 32), _fp8_plan("dgrad", m, k, n, 64)
    assert p32.jit is not p64.jit and p32.tile_config_name != p64.tile_config_name
    dy8, d_dy, w8, d_w, dx32 = _fp8_dgrad_operands(m, k, n)
    dx64 = dx32.clone()
    alpha = device_alpha(torch.zeros(1, dtype=torch.float32, device="cuda"), d_dy, d_w)
    run_dgrad_gemm(p32, dy8, w8, dx32, _ws(p32), alpha=alpha)
    run_dgrad_gemm(p64, dy8, w8, dx64, _ws(p64), alpha=alpha)
    torch.cuda.synchronize()
    ref64 = (dy8.double() @ w8.double()) * (d_dy.double() * d_w.double())
    _assert_close_vs_fp64(dx32, ref64, "B2 fp8 K32")
    _assert_close_vs_fp64(dx64, ref64, "B2 fp8 K64")
    d = (dx32.float() - dx64.float()).abs().max().item()
    print(
        f"K32 vs K64 max|diff| = {d:.4g} (max|ref| = {ref64.abs().max().item():.4g}; the bound is {_RTOL_BY_DTYPE[torch.bfloat16] * ref64.abs().max().item():.4g})"
    )
    assert d <= 2.0 * _RTOL_BY_DTYPE[torch.bfloat16] * ref64.abs().max().item()


@requires_rubin
@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
@pytest.mark.parametrize("k_bytes,t", [(32, 2048), (64, 2048), (32, 4104), (64, 4104)], ids=["K32-T2048", "K64-T2048", "K32-T4104", "K64-T4104"])
def test_fp8_split_k_matches_fp64(k_bytes, t):
    """``split_k=2`` on an admitted fp8 MN-major plan.  Under the two-kernel split, kernel 1 stores fp32 partials and
    kernel 2 (the fixed-order reducer) renders the graph's epilogue, so the ``alpha`` descale product must be applied
    there exactly once -- a dropped or doubled alpha is off by ~1/alpha or alpha and fails the bound outright.  B1's
    weight gradient at the test geometry (K = T), both MMA K forms, at T = 2048 and the ragged T = 4104 (33 CTA K
    tiles of 128 over two slices, the last an 8-element tail -- the slice bookkeeping the even case never exercises).
    Pins: the plan IS the forced tile's ``_splitK2`` spelling at the requested K form with two slices, its fp32
    partials ride ``plan.workspace_bytes``, no sentinel survivor, two launches bitwise equal, the fp64 bound, and the
    one-slice plan's output within fp32 reassociation of it (reported)."""
    m, k, n = _stage_mkn("B1_dw_o", "test", t)
    plan = _fp8_plan("wgrad", m, k, n, k_bytes, split_k=2)
    _assert_fp8_plan_is_the_forced_tile(plan, k_bytes, split_k=2)
    assert plan.jit.workspace_bytes > 0 and plan.workspace_bytes >= plan.jit.workspace_bytes >= 2 * m * n * 4, (plan.workspace_bytes, plan.jit.workspace_bytes)
    one = _fp8_plan("wgrad", m, k, n, k_bytes)
    _assert_fp8_plan_is_the_forced_tile(one, k_bytes)
    dy8, d_dy, x8, d_x, out1 = _fp8_wgrad_operands(m, k, n)
    out2, out_one = out1.clone(), out1.clone()
    alpha = device_alpha(torch.zeros(1, dtype=torch.float32, device="cuda"), d_dy, d_x)
    ws = _ws(plan)
    run_wgrad_gemm(plan, dy8, x8, out1, ws, alpha=alpha)
    run_wgrad_gemm(plan, dy8, x8, out2, ws, alpha=alpha)
    run_wgrad_gemm(one, dy8, x8, out_one, _ws(one), alpha=alpha)
    torch.cuda.synchronize()
    ref64 = (dy8.double().T @ x8.double()) * (d_dy.double() * d_x.double())
    _check_fp8_cell(out1, out2, ref64, f"fp8 B1 wgrad split_k=2 @ test, T={t}, K{k_bytes}, {plan.tile_config_name}")
    bound = _RTOL_BY_DTYPE[torch.bfloat16] * ref64.abs().max().item()
    d = (out1.float() - out_one.float()).abs().max().item()
    print(f"split_k=2 vs one slice: max|diff| = {d:.4g} (max|ref| = {ref64.abs().max().item():.4g}; the bound is {bound:.4g})")
    assert d <= 2.0 * bound, f"split_k=2 and the one-slice plan differ by {d:.4g} (bound {bound:.4g})"


@requires_rubin
@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
def test_forward_fp8_gemm_plans_stay_k32():
    """The forward's per-tensor fp8 projection GEMMs are PINNED at the K=32 MMA form -- the plans the block was
    measured and shipped at.  The 64-byte form reaches a plan only through an explicit ``mma_tile_k_bytes=64``,
    which no forward stage passes; keying the forced-tile pick on the dtype would flip these silently.  Both
    the fp8 block (two per-tensor GEMMs) and the MXFP8 block (its out_proj is per-tensor fp8 with alpha; its
    qkv projection is BLOCK-SCALE and keeps the engine's measured K64 preference -- pinned as such)."""
    from cudnn.gated_attention_block import GatedAttentionBlockFwd, MxQuantSpec, QuantSpec

    geom = _GEOMS["test"]
    dev = "cuda"
    bf = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    h8 = torch.zeros(1, 256, geom.d_model, dtype=_FP8, device=dev)
    w8 = torch.zeros(geom.n_qkvg, geom.d_model, dtype=_FP8, device=dev)
    wo8 = torch.zeros(geom.d_model, geom.h_q * geom.d_head, dtype=_FP8, device=dev)
    args = (h8, w8, bf(geom.d_head), bf(geom.d_head), bf(1, 256, geom.rope_dim), bf(1, 256, geom.rope_dim), wo8, bf(1, 256, geom.d_model), geom)
    fp8 = GatedAttentionBlockFwd(
        *args, quant=QuantSpec(descale_h=0.01, descale_w_qkvg=0.02, descale_w_o=0.03, scale_q=1.0, scale_k=2.0, scale_v=3.0, scale_o=4.0)
    )
    fp8.compile()
    for st in (fp8._proj, fp8._out_proj):
        plan = st._plan
        assert plan.dtype == _FP8 and plan.has_alpha and not plan.block_scale, st.name
        assert plan.tile_config_name == _FORCED_TILE and plan.mma_tile_k_bytes == 32 and plan.jit.config.mma_tile_k_bytes == 32, (
            st.name,
            plan.tile_config_name,
            plan.mma_tile_k_bytes,
        )
    hsf = torch.zeros(sf_blob_bytes(256, geom.d_model), dtype=torch.uint8, device=dev)
    wsf = torch.zeros(sf_blob_bytes(geom.n_qkvg, geom.d_model), dtype=torch.uint8, device=dev)
    mx = GatedAttentionBlockFwd(*args, quant=MxQuantSpec(descale_w_o=0.03), sample_h_sf=hsf, sample_w_qkvg_sf=wsf)
    mx.compile()
    out_plan = mx._out_proj._plan
    assert out_plan.dtype == _FP8 and out_plan.has_alpha and not out_plan.block_scale
    assert out_plan.tile_config_name == _FORCED_TILE and out_plan.mma_tile_k_bytes == 32, (out_plan.tile_config_name, out_plan.mma_tile_k_bytes)
    qkv_plan = mx._proj._plan
    assert qkv_plan.block_scale and qkv_plan.mma_tile_k_bytes == 64 and qkv_plan.tile_config_name == _FORCED_TILE_K64, (
        qkv_plan.tile_config_name,
        qkv_plan.mma_tile_k_bytes,
    )


def test_backward_gemm_stage_mma_tile_k_bytes_is_appended_and_explicit(monkeypatch):
    """``_GemmStage(mma_tile_k_bytes=)`` is appended with default ``None`` -- the plan request is byte-identical
    to before (the kwarg reaches ``build_proj_gemm`` as ``None``) -- and a value on a bf16 / fp16 stage is a
    typed decline at ``check_support`` naming the field: the 64-byte MMA K is an EXPLICIT request of an e4m3
    stage, never derived from the dtype."""
    import inspect

    import cudnn.gated_attention_block.api_bwd as ab

    sig = inspect.signature(ab._GemmStage.__init__)
    params = list(sig.parameters)
    # Append-only: ``mma_tile_k_bytes`` was appended first, the e4m3 stage's ``out_dtype`` / ``alpha`` after it, the block-scale (MXFP8)
    # stage's ``block_scale`` / ``w_dtype`` / ``block_size`` / ``sf_dtype`` after those -- the suffix ORDER is the pin (every append extends it).
    assert params[-7:] == ["mma_tile_k_bytes", "out_dtype", "alpha", "block_scale", "w_dtype", "block_size", "sf_dtype"]
    assert sig.parameters["mma_tile_k_bytes"].default is None and sig.parameters["out_dtype"].default is None and sig.parameters["alpha"].default is False
    assert sig.parameters["block_scale"].default is False and sig.parameters["w_dtype"].default is None
    assert sig.parameters["block_size"].default == 32 and sig.parameters["sf_dtype"].default is None
    exe = list(inspect.signature(ab._GemmStage.execute).parameters)
    assert exe[-3:] == ["alpha", "sf_a", "sf_b"], exe  # the blobs of a block-scale plan, appended after ``alpha``
    seen = {}

    def spy(**kw):
        """Stands in for ``build_proj_gemm``: records the kwargs the stage hands it, returns a placeholder plan."""
        seen.update(kw)
        return "plan"

    import cudnn.gated_attention_block.kernels.proj_gemm as pg

    monkeypatch.setattr(pg, "build_proj_gemm", spy)
    st = ab._OutProjWgrad(m=512, k=2048, n=2048, dtype=torch.bfloat16, label="b1")
    st.check_support()
    st.compile()
    assert seen["mma_tile_k_bytes"] is None and (seen["a_major"], seen["b_major"]) == ("m", "n") and st.plan == "plan"
    # the block-scale declaration reaches the driver at ITS defaults: the plan request of a per-tensor stage is byte-identical
    assert (seen["block_scale"], seen["w_dtype"], seen["block_size"], seen["sf_dtype"]) == (False, None, 32, None), seen
    for dt in (torch.bfloat16, torch.float16):
        st64 = ab._QkvGateDgrad(m=2048, k=5120, n=512, dtype=dt, label="b8", mma_tile_k_bytes=64)
        with pytest.raises(NotImplementedError, match=r"mma_tile_k_bytes=64 is a knob of an 8-bit \(e4m3\) GEMM stage"):
            st64.check_support()


# ---------------------------------------------------------------------------
# The MXFP8 backward's block-scale GEMMs: K-major TRANSPOSED operands + F8_128x4 scale factors (B7 / B8)
# ---------------------------------------------------------------------------
#
# The TE "columnwise" convention: every block-scale operand is K-major, so the wgrad binds dQKVG^T [N, T] against
# h^T [dm, T] (scale factors along T) and the dgrad binds dQKVG [T, N] against W_qkvg^T [dm, N] (along N) -- the
# FORWARD's own K-major declaration at the backward's shapes.  Oracle: fp64 of the dequantized operands
# (code * 2^(e - 127)), exact; bound: the bf16-output bound, unchanged.


def _mx():
    """``blocked_sf`` / ``mxfp8_quant`` of the forward MXFP8 GEMM test (same directory)."""
    import test_proj_gemm_mxfp8 as t

    return t


def _mx_operand(rows: int, k: int, seed: int, lo: int = 120, hi: int = 134):
    """e4m3 codes ``[rows, k]`` + DISTINCTIVE per-32-block E8M0 exponents, their padded F8_128x4 blob, and the
    fp64 dequantized matrix -- the forward MXFP8 test's generator (``distinctive_case``) on one operand."""
    t = _mx()
    mq = t.mxfp8_quant()
    torch.manual_seed(seed)
    codes = (torch.randn(rows, k, device="cuda") * 0.5).to(_FP8)
    e = torch.randint(lo, hi + 1, (rows, k // 32), dtype=torch.uint8, device="cuda")
    deq64 = codes.double() * mq.e8m0_to_float(e).double().repeat_interleave(32, 1)
    return codes, t.blocked_sf(e), deq64


@functools.lru_cache(maxsize=None)
def _bs_plan(m: int, k: int, n: int) -> ProjGemmPlan:
    """A block-scale plan at the K-major defaults -- the forward's declaration at a backward shape."""
    return build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, label=f"bs_{m}x{k}x{n}", block_scale=True)


# (stage, geom, T): both drivers at the test geometry (T = 2048 and the ragged 32-multiple 2016 = 63 x 32, a partial
# CTA K tile for the wgrad / M tile for the dgrad) and at the 397B column shapes at T = 2048.
_BS_CASES = [
    ("B7_dw_qkvg", "test", 2048),
    ("B7_dw_qkvg", "test", 2016),
    ("B8_dh", "test", 2048),
    ("B8_dh", "test", 2016),
    ("B7_dw_qkvg", "397B", 2048),
    ("B8_dh", "397B", 2048),
]


@requires_rubin
@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
@pytest.mark.parametrize("stage,geom_id,t", _BS_CASES, ids=[f"{c[0]}-{c[1]}-T{c[2]}" for c in _BS_CASES])
def test_block_scale_transposed_drivers_match_fp64(stage, geom_id, t):
    """``run_wgrad_gemm_block_scale`` / ``run_dgrad_gemm_block_scale`` on K-major transposed MXFP8 operands vs the
    fp64 reference of the dequantized operands, within the bf16-output bound; the plan IS the forced tile at
    the engine's block-scale K preference (K64 on cc 10.7); no sentinel survivor; not silently zero; two
    launches bitwise equal (no split-K, no atomics)."""
    m, k, n = _stage_mkn(stage, geom_id, t)
    plan = _bs_plan(m, k, n)
    assert plan.jit is not None and plan.block_scale and (plan.a_major, plan.b_major) == ("k", "k"), (plan.tile_config_name, plan.route)
    assert plan.tile_config_name == _FORCED_TILE_K64 and plan.mma_tile_k_bytes == 64, (plan.tile_config_name, plan.mma_tile_k_bytes)
    ws = _ws(plan)
    a8, sf_a, a64 = _mx_operand(m, k, seed=1)  # A [m, k] K-major: dQKVG^T [N, T] (wgrad) or dQKVG [T, N] (dgrad)
    w8, sf_w, w64 = _mx_operand(n, k, seed=2)  # W [n, k] K-major: h^T [dm, T] (wgrad) or W_qkvg^T [dm, N] (dgrad)
    out1 = torch.full((m, n), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    out2 = out1.clone()
    runner = run_wgrad_gemm_block_scale if stage in _WGRAD else run_dgrad_gemm_block_scale
    kw = dict(sf_dy_t=sf_a, sf_x_t=sf_w) if stage in _WGRAD else dict(sf_dy=sf_a, sf_w_t=sf_w)
    runner(plan, a8, w8, out1, ws, **kw)
    runner(plan, a8, w8, out2, ws, **kw)
    torch.cuda.synchronize()
    _check_fp8_cell(out1, out2, a64 @ w64.T, f"block-scale {stage} @ {geom_id}, T={t}, {plan.tile_config_name}")


@requires_cuda
@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
def test_block_scale_backward_declines_are_typed():
    """Plan time: a token-axis wgrad whose ``T % 32 != 0`` (one E8M0 scale per 32-element K block) is a
    ``ValueError`` naming K, BEFORE any graph; so are ``split_k`` with block scale and ANY non-K major with block
    scale (the scale-factor blobs run along K-major rows).  Run time: the drivers refuse a plan that is not
    block-scale, a plan declared MN-major, an operand that is a ``.t()`` view (stride-1 axis on the wrong side)
    or a slice of a wider slab (row stride != K), a wrong-shaped output, and a missing or wrong-sized scale-factor
    blob -- named by the DRIVER's keyword (``sf_dy_t`` / ``sf_x_t``, ``sf_dy`` / ``sf_w_t``), with the operand it
    scales and the byte count, never by ``run_proj_gemm``'s ``sf_a`` / ``sf_w`` -- each a typed ``ValueError`` naming
    the operand, before any launch (a spy stands in for the JIT)."""
    dm, n_qkvg = 512, 5120
    with pytest.raises(ValueError, match=r"block_scale=True needs K % 32 == 0.*got K=1000"):
        build_proj_gemm(m=n_qkvg, k=1000, n=dm, dtype=_FP8, label="b7_ragged_t", block_scale=True)
    with pytest.raises(ValueError, match=r"split_k=2 on the block-scale GEMM is not served"):
        build_proj_gemm(m=n_qkvg, k=2048, n=dm, dtype=_FP8, label="b7_split", block_scale=True, split_k=2)
    with pytest.raises(ValueError, match=r"M-major A / N-major B are served on the dense path only"):
        build_proj_gemm(m=n_qkvg, k=2048, n=dm, dtype=_FP8, label="b7_mn", block_scale=True, a_major="m", b_major="n")
    with pytest.raises(ValueError, match=r"served on the dense path only"):
        build_proj_gemm(m=2048, k=n_qkvg, n=dm, dtype=_FP8, label="b8_n", block_scale=True, a_major="k", b_major="n")

    class _SpyJit:
        """Stands in for the plan's JIT: records the variant pack of every launch it is handed, launches nothing."""

        def __init__(self):
            """No launches recorded yet."""
            self.calls = []

        def __call__(self, vp, **kw):
            """Record the variant pack; the launch kwargs (stream, workspace) are accepted and ignored."""
            self.calls.append(vp)

    t = 2048
    dev = "cuda"
    a8 = torch.zeros(n_qkvg, t, dtype=_FP8, device=dev)  # dQKVG^T [N, T]
    w8 = torch.zeros(dm, t, dtype=_FP8, device=dev)  # h^T [dm, T]
    dw = torch.zeros(n_qkvg, dm, dtype=torch.bfloat16, device=dev)
    sf_a = torch.zeros(sf_blob_bytes(n_qkvg, t), dtype=torch.uint8, device=dev)
    sf_w = torch.zeros(sf_blob_bytes(dm, t), dtype=torch.uint8, device=dev)
    ws = torch.empty(1, dtype=torch.uint8, device=dev)
    dense = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=n_qkvg, k=t, n=dm, label="dense", dtype=_FP8, out_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"run_wgrad_gemm_block_scale serves block-scale plans"):
        run_wgrad_gemm_block_scale(dense, a8, w8, dw, ws, sf_dy_t=sf_a, sf_x_t=sf_w)
    mn = ProjGemmPlan(
        graph=None, a=None, b=None, c=None, m=n_qkvg, k=t, n=dm, label="mn", dtype=_FP8, out_dtype=torch.bfloat16, block_scale=True, a_major="m", b_major="n"
    )
    with pytest.raises(ValueError, match=r"binds K-major TRANSPOSED operands.*a_major='m', b_major='n'"):
        run_wgrad_gemm_block_scale(mn, a8, w8, dw, ws, sf_dy_t=sf_a, sf_x_t=sf_w)
    bs = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=n_qkvg, k=t, n=dm, label="bs", dtype=_FP8, out_dtype=torch.bfloat16, block_scale=True)
    bs.jit = _SpyJit()
    # the UN-transposed dQKVG [T, N] handed as its .t() view: right shape, wrong stride-1 axis
    with pytest.raises(
        ValueError, match=r"dy_t \(dy_like\^T, \[rows, T\]\) has strides \(1, 5120\) but this plan declared a contiguous row-major \[5120, 2048\]"
    ):
        run_wgrad_gemm_block_scale(bs, torch.zeros(t, n_qkvg, dtype=_FP8, device=dev).t(), w8, dw, ws, sf_dy_t=sf_a, sf_x_t=sf_w)
    # a column slice of a wider slab: right stride-1 axis, row stride != K
    with pytest.raises(ValueError, match=r"x_t \(x\^T, \[cols, T\]\) has strides \(2112, 1\)"):
        run_wgrad_gemm_block_scale(bs, a8, torch.zeros(dm, t + 64, dtype=_FP8, device=dev)[:, :t], dw, ws, sf_dy_t=sf_a, sf_x_t=sf_w)
    with pytest.raises(ValueError, match=r"has shape \(512, 2047\); this plan declared it as the K-major \[512, 2048\]"):
        run_wgrad_gemm_block_scale(bs, a8, torch.zeros(dm, t - 1, dtype=_FP8, device=dev), dw, ws, sf_dy_t=sf_a, sf_x_t=sf_w)
    with pytest.raises(ValueError, match=r"dw view has shape \(1, 5120, 256\); the plan declared C as dims \(1, 5120, 512\)"):
        run_wgrad_gemm_block_scale(bs, a8, w8, torch.zeros(n_qkvg, dm // 2, dtype=torch.bfloat16, device=dev), ws, sf_dy_t=sf_a, sf_x_t=sf_w)

    # the scale-factor blobs, by the DRIVER's keywords: a missing one names the keyword, the operand it scales and the
    # byte count it needs; a wrong-sized one names the keyword and the F8_128x4 arithmetic -- run_proj_gemm's own
    # `sf_a` / `sf_w` (the names its message would use) appear in neither.
    def _no_inner_name(exc) -> bool:
        """True when the message names neither ``sf_a`` nor ``sf_w`` -- ``run_proj_gemm``'s keywords, which the drivers' gate
        must not leak."""
        return re.search(r"\bsf_[aw]\b", str(exc)) is None

    with pytest.raises(ValueError, match=r"pass sf_dy_t= \(the padded F8_128x4 scale-factor blob of dy_t over its rows x T, \d+ bytes.*No silent 1.0") as ei:
        run_wgrad_gemm_block_scale(bs, a8, w8, dw, ws, sf_dy_t=None, sf_x_t=sf_w)
    assert _no_inner_name(ei.value) and f"{sf_blob_bytes(n_qkvg, t)} bytes for {n_qkvg} rows x K={t}" in str(ei.value), str(ei.value)
    with pytest.raises(ValueError, match=r"pass sf_x_t= \(the padded F8_128x4 scale-factor blob of x_t over its cols x T, \d+ bytes") as ei:
        run_wgrad_gemm_block_scale(bs, a8, w8, dw, ws, sf_dy_t=sf_a, sf_x_t=None)
    assert _no_inner_name(ei.value) and f"{sf_blob_bytes(dm, t)} bytes for {dm} rows x K={t}" in str(ei.value), str(ei.value)
    with pytest.raises(ValueError, match=r"sf_x_t has \d+ bytes; the F8_128x4 blob over 512 rows x K=2048") as ei:
        run_wgrad_gemm_block_scale(bs, a8, w8, dw, ws, sf_dy_t=sf_a, sf_x_t=sf_w[:-512])
    assert _no_inner_name(ei.value), str(ei.value)
    with pytest.raises(ValueError, match=r"sf_dy_t has \d+ bytes; the F8_128x4 blob over 5120 rows x K=2048") as ei:
        run_wgrad_gemm_block_scale(bs, a8, w8, dw, ws, sf_dy_t=sf_a[:-512], sf_x_t=sf_w)
    assert _no_inner_name(ei.value), str(ei.value)
    assert not bs.jit.calls, "a refused operand reached the launch"
    # the declared operands reach the (spy) launch once, with the blobs bound as the (1, rows_pad, 4*k4) views
    run_wgrad_gemm_block_scale(bs, a8, w8, dw, ws, sf_dy_t=sf_a, sf_x_t=sf_w)
    assert len(bs.jit.calls) == 1
    # dgrad: dQKVG [T, N] against W_qkvg^T [dm, N]
    dg = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=n_qkvg, n=dm, label="dg", dtype=_FP8, out_dtype=torch.bfloat16, block_scale=True)
    dg.jit = _SpyJit()
    dy8 = torch.zeros(t, n_qkvg, dtype=_FP8, device=dev)
    wt8 = torch.zeros(dm, n_qkvg, dtype=_FP8, device=dev)
    sf_dy = torch.zeros(sf_blob_bytes(t, n_qkvg), dtype=torch.uint8, device=dev)
    sf_wt = torch.zeros(sf_blob_bytes(dm, n_qkvg), dtype=torch.uint8, device=dev)
    with pytest.raises(ValueError, match=r"w_t \(w\^T, \[N, K\]\) has strides \(1, 512\)"):
        run_dgrad_gemm_block_scale(
            dg, dy8, torch.zeros(n_qkvg, dm, dtype=_FP8, device=dev).t(), torch.zeros(t, dm, dtype=torch.bfloat16, device=dev), ws, sf_dy=sf_dy, sf_w_t=sf_wt
        )
    dx = torch.zeros(t, dm, dtype=torch.bfloat16, device=dev)
    with pytest.raises(ValueError, match=r"pass sf_dy= \(the padded F8_128x4 scale-factor blob of dy_like over its T rows x K, \d+ bytes.*No silent 1.0") as ei:
        run_dgrad_gemm_block_scale(dg, dy8, wt8, dx, ws, sf_dy=None, sf_w_t=sf_wt)
    assert _no_inner_name(ei.value) and f"{sf_blob_bytes(t, n_qkvg)} bytes for {t} rows x K={n_qkvg}" in str(ei.value), str(ei.value)
    with pytest.raises(ValueError, match=r"pass sf_w_t= \(the padded F8_128x4 scale-factor blob of w_t over its N rows x K") as ei:
        run_dgrad_gemm_block_scale(dg, dy8, wt8, dx, ws, sf_dy=sf_dy, sf_w_t=None)
    assert _no_inner_name(ei.value), str(ei.value)
    with pytest.raises(ValueError, match=r"sf_w_t has \d+ bytes; the F8_128x4 blob over 512 rows x K=5120") as ei:
        run_dgrad_gemm_block_scale(dg, dy8, wt8, dx, ws, sf_dy=sf_dy, sf_w_t=sf_wt[:-512])
    assert _no_inner_name(ei.value), str(ei.value)
    assert not dg.jit.calls
    run_dgrad_gemm_block_scale(dg, dy8, wt8, dx, ws, sf_dy=sf_dy, sf_w_t=sf_wt)
    assert len(dg.jit.calls) == 1


# ---------------------------------------------------------------------------
# The fp4 weight modes' dgrads on the block-scale drivers: the MIXED e4m3 x e2m1 row and the NVFP4 x NVFP4 row
# ---------------------------------------------------------------------------
#
# Under an MXFP4 `W_qkvg` B8 reads the caller's e2m1 `W_qkvg^T` -- stored PACKED [dm, N // 2], two codes per byte along K -- against
# the rowwise e4m3 dQKVG (the catalog's MIXED row, E8M0 per 32 on both sides); under an MXFP4 `W_o` B2 reads an MX-rowwise e4m3 dY
# against the e2m1 `W_o^T` [HD, dm // 2] (the same row); under an NVFP4 `W_o` B2 reads dY itself cast to NVFP4 ([T, dm // 2] + e4m3
# scales per 16) against the e2m1 `W_o^T` with its e4m3/16 blob (the NVFP4 x NVFP4 row): BOTH operands packed.  The renderings are the
# forward's own (its stage-1 mixed row, its fp4 out projection) at the dgrad's (m, k, n) on the forced tile's K64 twin; the oracle is
# fp64 of every operand dequantized THROUGH its blob (`fp4_dequant_rowwise_2d` for an e2m1 side: a code x its scale is exact in fp64);
# the bound is the bf16-output bound, unchanged.  The weight gradients stay 8-bit under every fp4 mode (the wgrad driver's e2m1 rule is
# pinned on the host only).

_E2M1 = getattr(torch, "float4_e2m1fn_x2", None)
needs_fp4 = pytest.mark.skipif(_FP8 is None or _E2M1 is None, reason="this torch has no float8_e4m3fn / float4_e2m1fn_x2")


def _fp4_operand(rows: int, k: int, fmt: str, seed: int):
    """e2m1 codes ``[rows, k // 2]`` viewed ``float4_e2m1fn_x2`` + the format's PADDED F8_128x4 blob + the fp64 matrix dequantized
    THROUGH that blob -- the oracle's ``fp4_quantize_rowwise_2d`` over a bf16 random matrix whose rows carry a power-of-two spread
    (distinct block scales across rows; the block's own artifacts come from the same quantizer)."""
    from gated_block_reference import fp4_dequant_rowwise_2d, fp4_format, fp4_quantize_rowwise_2d, mx_swizzle_sf_rowwise_padded

    _, block, _ = fp4_format(fmt)
    torch.manual_seed(seed)
    spread = torch.pow(2.0, torch.randint(-4, 4, (rows, 1), device="cuda").float())
    x = (torch.randn(rows, k, device="cuda") * 0.5 * spread).to(torch.bfloat16)
    packed, e = fp4_quantize_rowwise_2d(x.float(), fmt)
    blob = mx_swizzle_sf_rowwise_padded(e, block=block)
    assert blob.numel() == sf_blob_bytes(rows, k, block), (blob.numel(), sf_blob_bytes(rows, k, block))
    return packed.view(_E2M1), blob, fp4_dequant_rowwise_2d(packed, blob, fmt, out_dtype=torch.float64)


@functools.lru_cache(maxsize=None)
def _fp4_plan(m: int, k: int, n: int, row: str) -> ProjGemmPlan:
    """A block-scale plan at the K-major defaults of the MIXED row (``dtype=e4m3, w_dtype=e2m1``, E8M0 per 32 -- the scale dtype and
    block resolved by ``block_scale_pairing``) or of the NVFP4 row (``dtype=w_dtype=e2m1``, ``block_size=16``, e4m3 scales spelled)."""
    import cudnn

    if row == "mixed":
        return build_proj_gemm(m=m, k=k, n=n, dtype=_FP8, w_dtype=_E2M1, label=f"mixed_{m}x{k}x{n}", block_scale=True)
    return build_proj_gemm(
        m=m, k=k, n=n, dtype=_E2M1, w_dtype=_E2M1, label=f"nvfp4_{m}x{k}x{n}", block_scale=True, block_size=16, sf_dtype=cudnn.data_type.FP8_E4M3
    )


# (stage, geom, T, row): B8 on the mixed row at (T, N, dm) and B2 on the mixed and the NVFP4 row at (T, dm, HD), at the test geometry
# (T = 2048 and the ragged 32-multiple 2016 = 63 x 32: a partial 128-row M tile) and at the 397B column shapes.
_FP4_DGRAD_CASES = [
    ("B8_dh", "test", 2048, "mixed"),
    ("B8_dh", "test", 2016, "mixed"),
    ("B8_dh", "397B", 2048, "mixed"),
    ("B2_do_gated", "test", 2048, "mixed"),
    ("B2_do_gated", "397B", 2048, "mixed"),
    ("B2_do_gated", "test", 2048, "nvfp4"),
    ("B2_do_gated", "test", 2016, "nvfp4"),
    ("B2_do_gated", "397B", 2048, "nvfp4"),
]


@requires_rubin
@needs_fp4
@pytest.mark.parametrize("stage,geom_id,t,row", _FP4_DGRAD_CASES, ids=[f"{c[0]}-{c[1]}-T{c[2]}-{c[3]}" for c in _FP4_DGRAD_CASES])
def test_fp4_block_scale_dgrad_matches_fp64_through_the_blobs(stage, geom_id, t, row):
    """``run_dgrad_gemm_block_scale`` on the fp4 weight modes' operands -- the mixed row's e4m3 ``dy_like`` x PACKED e2m1 ``w_t``, the
    NVFP4 row's packed e2m1 ``dy_like`` x packed e2m1 ``w_t`` -- vs the fp64 product of the operands dequantized THROUGH their blobs,
    within the bf16-output bound; its plan IS the forced tile's K64 twin with the row's scale dtype and block; no sentinel survivor;
    not silently zero; two launches bitwise (no split-K, no atomics)."""
    import cudnn

    m, k, n = _stage_mkn(stage, geom_id, t)
    plan = _fp4_plan(m, k, n, row)
    assert plan.jit is not None and plan.block_scale and (plan.a_major, plan.b_major) == ("k", "k"), (plan.tile_config_name, plan.route)
    assert plan.tile_config_name == _FORCED_TILE_K64 and plan.mma_tile_k_bytes == 64, (plan.tile_config_name, plan.mma_tile_k_bytes)
    if row == "mixed":
        assert (plan.dtype, plan.w_dtype, plan.block_size, plan.sf_dtype) == (_FP8, _E2M1, 32, cudnn.data_type.FP8_E8M0)
        a, sf_a, a64 = _mx_operand(m, k, seed=1)  # the rowwise e4m3 gradient [T, K] with its canonical blob
        w, sf_w, w64 = _fp4_operand(n, k, "mxfp4", seed=2)  # the caller's e2m1 W^T, packed [N, K // 2], E8M0 per 32
    else:
        assert (plan.dtype, plan.w_dtype, plan.block_size, plan.sf_dtype) == (_E2M1, _E2M1, 16, cudnn.data_type.FP8_E4M3)
        a, sf_a, a64 = _fp4_operand(m, k, "nvfp4", seed=1)  # dY cast to NVFP4: packed [T, K // 2], e4m3 per 16
        w, sf_w, w64 = _fp4_operand(n, k, "nvfp4", seed=2)  # the caller's e2m1 W_o^T, packed [N, K // 2], e4m3 per 16
    assert tuple(w.shape) == (n, k // 2) and (tuple(a.shape) == (m, k // 2) if a.dtype == _E2M1 else tuple(a.shape) == (m, k))
    ws = _ws(plan)
    out1 = torch.full((m, n), _SENTINEL, device="cuda", dtype=torch.bfloat16)
    out2 = out1.clone()
    run_dgrad_gemm_block_scale(plan, a, w, out1, ws, sf_dy=sf_a, sf_w_t=sf_w)
    run_dgrad_gemm_block_scale(plan, a, w, out2, ws, sf_dy=sf_a, sf_w_t=sf_w)
    torch.cuda.synchronize()
    _check_fp8_cell(out1, out2, a64 @ w64.T, f"fp4 {row} {stage} @ {geom_id}, T={t}, {plan.tile_config_name} ({plan.route})")


@requires_cuda
@needs_fp4
def test_fp4_block_scale_driver_declines_are_typed():
    """Plan time, before any graph (``block_scale_pairing`` / ``build_proj_gemm``): an e2m1 side without ``block_scale``, the mixed pair at
    block 16, two e2m1 sides at E8M0 per 16, the mixed pair with e4m3 scales (no catalog row for each), a non-K major with an e2m1 side.
    Run time, on hand-built plans with a spy JIT (nothing launches), each a typed ``ValueError`` naming the operand: uint8 packed codes
    handed to an e2m1 side (the ``.view(torch.float4_e2m1fn_x2)`` hint), a LOGICAL ``[N, K]`` fp4 tensor (twice the data: the packed
    extent ``K // 2``), a non-K-major fp4 side (a ``.t()`` view: the stride message at the PACKED extent), the wrong row count, e4m3
    codes where e2m1 was declared and the reverse, the OTHER format's blob under the NVFP4 row (block 16 vs 32: exactly 2x the
    bytes) and a right-sized blob of the wrong scale dtype; the declared packed operands reach the launch exactly once.  The wgrad
    driver takes the same e2m1 rule (not a row the block's backward declares: its weight gradients stay 8-bit)."""
    import cudnn

    dm, hd, n_qkvg, t = 512, 2048, 5120, 2048
    with pytest.raises(ValueError, match="ride the block-scale GEMM only"):
        build_proj_gemm(m=t, k=n_qkvg, n=dm, dtype=_FP8, w_dtype=_E2M1, label="fp4_dense")
    with pytest.raises(ValueError, match="no block-scale GEMM row"):
        build_proj_gemm(m=t, k=n_qkvg, n=dm, dtype=_FP8, w_dtype=_E2M1, label="mixed16", block_scale=True, block_size=16)
    with pytest.raises(ValueError, match="no block-scale GEMM row"):
        build_proj_gemm(m=t, k=dm, n=hd, dtype=_E2M1, w_dtype=_E2M1, label="nv_e8m0_16", block_scale=True, block_size=16, sf_dtype=cudnn.data_type.FP8_E8M0)
    with pytest.raises(ValueError, match="no block-scale GEMM row"):
        build_proj_gemm(m=t, k=n_qkvg, n=dm, dtype=_FP8, w_dtype=_E2M1, label="mixed_e4m3", block_scale=True, sf_dtype=cudnn.data_type.FP8_E4M3)
    with pytest.raises(ValueError, match="served on the dense path only"):
        build_proj_gemm(m=t, k=n_qkvg, n=dm, dtype=_FP8, w_dtype=_E2M1, label="mixed_n", block_scale=True, b_major="n")

    class _SpyJit:
        """Stands in for a hand-built plan's JIT: records the variant pack of every launch it is handed, launches nothing."""

        def __init__(self):
            """No launches recorded yet."""
            self.calls = []

        def __call__(self, vp, **kw):
            """Record the variant pack; the launch kwargs (stream, workspace) are accepted and ignored."""
            self.calls.append(vp)

    def _plan(m, k, n, label, dtype, w_dtype, block_size, sf_dtype):
        """A hand-built block-scale plan of one row (no graph) with a spy JIT: the drivers' checks run, nothing launches."""
        p = ProjGemmPlan(
            graph=None,
            a=None,
            b=None,
            c=None,
            m=m,
            k=k,
            n=n,
            label=label,
            dtype=dtype,
            out_dtype=torch.bfloat16,
            block_scale=True,
            w_dtype=w_dtype,
            block_size=block_size,
        )
        p.sf_dtype = sf_dtype
        p.jit = _SpyJit()
        return p

    dev = "cuda"
    ws = torch.empty(1, dtype=torch.uint8, device=dev)
    u8 = lambda *shape: torch.zeros(*shape, dtype=torch.uint8, device=dev)  # noqa: E731
    # --- the mixed row on B8: e4m3 dQKVG [T, N] x the e2m1 W_qkvg^T packed [dm, N // 2] ---
    mixed = _plan(t, n_qkvg, dm, "mixed", _FP8, _E2M1, 32, cudnn.data_type.FP8_E8M0)
    a8 = torch.zeros(t, n_qkvg, dtype=_FP8, device=dev)
    dx = torch.zeros(t, dm, dtype=torch.bfloat16, device=dev)
    sf_a, sf_w = u8(sf_blob_bytes(t, n_qkvg)), u8(sf_blob_bytes(dm, n_qkvg))
    with pytest.raises(
        ValueError, match=r"w_t \(w\^T, \[N, K\]\) is torch.uint8 but this plan was built for torch.float4_e2m1fn_x2.*\.view\(torch\.float4_e2m1fn_x2\)"
    ):
        run_dgrad_gemm_block_scale(mixed, a8, u8(dm, n_qkvg // 2), dx, ws, sf_dy=sf_a, sf_w_t=sf_w)  # packed bytes, no view
    with pytest.raises(ValueError, match=r"w_t \(w\^T, \[N, K\]\) is fp4 storage \(512, 5120\).*last extent must be K/2 = 2560, not 5120"):
        run_dgrad_gemm_block_scale(mixed, a8, u8(dm, n_qkvg).view(_E2M1), dx, ws, sf_dy=sf_a, sf_w_t=sf_w)  # a LOGICAL [dm, N] fp4 tensor
    with pytest.raises(
        ValueError, match=r"has strides \(1, 512\) but this plan declared a contiguous row-major \[512, 2560\] \(strides \(2560, 1\) -- the PACKED e2m1 storage"
    ):
        run_dgrad_gemm_block_scale(mixed, a8, u8(n_qkvg // 2, dm).view(_E2M1).t(), dx, ws, sf_dy=sf_a, sf_w_t=sf_w)  # a non-K-major fp4 side
    with pytest.raises(ValueError, match=r"has shape \(511, 2560\); this plan declared it as the K-major \[512, 2560\] \(rows x K, K contiguous -- the PACKED"):
        run_dgrad_gemm_block_scale(mixed, a8, u8(dm - 1, n_qkvg // 2).view(_E2M1), dx, ws, sf_dy=sf_a, sf_w_t=sf_w)
    with pytest.raises(ValueError, match=r"w_t \(w\^T, \[N, K\]\) is torch.float8_e4m3fn but this plan was built for torch.float4_e2m1fn_x2"):
        run_dgrad_gemm_block_scale(mixed, a8, torch.zeros(dm, n_qkvg, dtype=_FP8, device=dev), dx, ws, sf_dy=sf_a, sf_w_t=sf_w)  # e4m3 where e2m1 was declared
    with pytest.raises(ValueError, match=r"dy_like \(\[T, K\]\) is torch.float4_e2m1fn_x2 but this plan was built for torch.float8_e4m3fn"):
        run_dgrad_gemm_block_scale(
            mixed, u8(t, n_qkvg // 2).view(_E2M1), u8(dm, n_qkvg // 2).view(_E2M1), dx, ws, sf_dy=sf_a, sf_w_t=sf_w
        )  # e2m1 where e4m3 was
    assert not mixed.jit.calls, "a refused operand reached the launch"
    run_dgrad_gemm_block_scale(mixed, a8, u8(dm, n_qkvg // 2).view(_E2M1), dx, ws, sf_dy=sf_a, sf_w_t=sf_w)
    assert len(mixed.jit.calls) == 1
    # --- the NVFP4 row on B2: dY4 packed [T, dm // 2] x the e2m1 W_o^T packed [HD, dm // 2], e4m3 scales per 16 ---
    nv = _plan(t, dm, hd, "nvfp4", _E2M1, _E2M1, 16, cudnn.data_type.FP8_E4M3)
    dy4, wo4 = u8(t, dm // 2).view(_E2M1), u8(hd, dm // 2).view(_E2M1)
    dxo = torch.zeros(t, hd, dtype=torch.bfloat16, device=dev)
    sf_dy, sf_wo = u8(sf_blob_bytes(t, dm, 16)), u8(sf_blob_bytes(hd, dm, 16))
    assert sf_blob_bytes(hd, dm, 16) == 2 * sf_blob_bytes(hd, dm, 32), "the two formats' blobs differ by exactly the block ratio at this geometry"
    with pytest.raises(ValueError, match=r"sf_w_t has \d+ bytes; the F8_128x4 blob over 2048 rows x K=512 at block 16"):
        run_dgrad_gemm_block_scale(nv, dy4, wo4, dxo, ws, sf_dy=sf_dy, sf_w_t=u8(sf_blob_bytes(hd, dm, 32)))  # the OTHER format's blob
    with pytest.raises(ValueError, match=r"sf_dy must be uint8 or float8_e4m3fn"):
        run_dgrad_gemm_block_scale(nv, dy4, wo4, dxo, ws, sf_dy=sf_dy.view(torch.float8_e8m0fnu), sf_w_t=sf_wo)  # right size, the other scale dtype
    with pytest.raises(ValueError, match=r"dy_like \(\[T, K\]\) is torch.float8_e4m3fn but this plan was built for torch.float4_e2m1fn_x2"):
        run_dgrad_gemm_block_scale(nv, torch.zeros(t, dm, dtype=_FP8, device=dev), wo4, dxo, ws, sf_dy=sf_dy, sf_w_t=sf_wo)
    with pytest.raises(ValueError, match=r"dy_like \(\[T, K\]\) is fp4 storage \(2048, 512\).*K/2 = 256"):
        run_dgrad_gemm_block_scale(nv, u8(t, dm).view(_E2M1), wo4, dxo, ws, sf_dy=sf_dy, sf_w_t=sf_wo)  # a LOGICAL [T, dm] fp4 A
    with pytest.raises(ValueError, match=r"pass sf_dy= \(the padded F8_128x4 scale-factor blob of dy_like over its T rows x K, \d+ bytes"):
        run_dgrad_gemm_block_scale(nv, dy4, wo4, dxo, ws, sf_dy=None, sf_w_t=sf_wo)
    assert not nv.jit.calls, "a refused operand reached the launch"
    run_dgrad_gemm_block_scale(nv, dy4, wo4, dxo, ws, sf_dy=sf_dy, sf_w_t=sf_wo)
    assert len(nv.jit.calls) == 1
    # --- the wgrad driver on an e2m1 side: the same packed rule (the block's weight gradients stay 8-bit; the driver is row-agnostic) ---
    wg = _plan(n_qkvg, t, dm, "wg_mixed", _FP8, _E2M1, 32, cudnn.data_type.FP8_E8M0)
    a_t8 = torch.zeros(n_qkvg, t, dtype=_FP8, device=dev)
    dw = torch.zeros(n_qkvg, dm, dtype=torch.bfloat16, device=dev)
    sf_at, sf_xt = u8(sf_blob_bytes(n_qkvg, t)), u8(sf_blob_bytes(dm, t))
    with pytest.raises(ValueError, match=r"x_t \(x\^T, \[cols, T\]\) is fp4 storage \(512, 2048\).*K/2 = 1024"):
        run_wgrad_gemm_block_scale(wg, a_t8, u8(dm, t).view(_E2M1), dw, ws, sf_dy_t=sf_at, sf_x_t=sf_xt)
    with pytest.raises(ValueError, match=r"x_t \(x\^T, \[cols, T\]\) is torch.uint8 but this plan was built for torch.float4_e2m1fn_x2"):
        run_wgrad_gemm_block_scale(wg, a_t8, u8(dm, t // 2), dw, ws, sf_dy_t=sf_at, sf_x_t=sf_xt)
    assert not wg.jit.calls
    run_wgrad_gemm_block_scale(wg, a_t8, u8(dm, t // 2).view(_E2M1), dw, ws, sf_dy_t=sf_at, sf_x_t=sf_xt)
    assert len(wg.jit.calls) == 1


# ---------------------------------------------------------------------------
# The device alpha slot (the per-tensor fp8 GEMMs' descale product)
# ---------------------------------------------------------------------------


def _fp32_product(*xs: float) -> float:
    """The correctly-rounded fp32 product of ``xs`` taken left to right -- what ``device_alpha`` documents (one
    fp32 multiply per factor, round-to-nearest); the Python double ``0.25 * 0.03`` is NOT it."""
    acc = torch.tensor(xs[0], dtype=torch.float32)
    for x in xs[1:]:
        acc = (acc.double() * torch.tensor(x, dtype=torch.float32).double()).float()  # exact in double, one fp32 rounding
    return acc.item()


@requires_cuda
def test_device_alpha_contract():
    """``device_alpha`` writes ``descale_a * descale_b (* scale_out)`` into the CALLER's fp32 slot ON THE DEVICE and
    returns the ``[1, 1, 1]`` view the GEMM binds: the view ALIASES the slot (same ``data_ptr``), nothing is
    allocated per call, no host sync happens (the inputs are device tensors and never ``.item()``-ed), the value
    is the correctly-rounded fp32 product (``_fp32_product``, bit-exact -- not the Python double), and the
    multiplies run on the launch stream (deterministic probe: the default stream
    is parked and the INPUTS are overwritten on the side stream right after the call -- a product issued on the
    default stream would read the overwritten scales).  Rejects, typed and before any write: a slot or input that
    is not a 1-element fp32 CUDA tensor, an input on another device where one is visible."""
    dev = torch.device("cuda")
    region = torch.zeros(8, dtype=torch.float32, device=dev)  # a workspace REGION; slot 3 is this GEMM's alpha
    slot = region[3:4]
    a = torch.tensor([0.25], dtype=torch.float32, device=dev)
    b = torch.tensor([0.03], dtype=torch.float32, device=dev)
    c = torch.tensor([64.0], dtype=torch.float32, device=dev)
    v = device_alpha(slot, a, b)
    torch.cuda.synchronize()
    assert tuple(v.shape) == (1, 1, 1) and v.dtype == torch.float32 and v.data_ptr() == slot.data_ptr() == region[3:4].data_ptr()
    assert region[3].item() == _fp32_product(0.25, 0.03) and region[[0, 1, 2, 4, 5, 6, 7]].abs().sum().item() == 0
    v2 = device_alpha(slot, a, b, c)
    torch.cuda.synchronize()
    assert v2.data_ptr() == slot.data_ptr() and region[3].item() == _fp32_product(0.25, 0.03, 64.0)
    before = torch.cuda.memory_allocated()
    device_alpha(slot, a, b)
    device_alpha(slot, a, b, c)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "a call allocated"
    # stream threading: the product lands on the side stream, ahead of the overwrite issued there
    side = torch.cuda.Stream()
    expect = _fp32_product(0.25, 0.03)
    for how in ("ambient", "explicit"):
        slot.zero_()
        a.fill_(0.25)
        b.fill_(0.03)
        torch.cuda.synchronize()
        park_the_default_stream()
        if how == "ambient":
            with torch.cuda.stream(side):
                device_alpha(slot, a, b)
        else:
            device_alpha(slot, a, b, stream=side.cuda_stream)
        with torch.cuda.stream(side):
            a.fill_(7.0)  # ordered AFTER the product on the side stream; a product parked on the default stream reads 7.0
            b.fill_(7.0)
        torch.cuda.synchronize()
        assert slot.item() == expect, f"device_alpha ran off the launch stream ({how}): slot = {slot.item()}"
    # rejects
    with pytest.raises(ValueError, match=r"alpha_out must be a 1-element fp32 CUDA tensor"):
        device_alpha(torch.zeros(2, dtype=torch.float32, device=dev), a, b)
    with pytest.raises(ValueError, match=r"alpha_out must be a 1-element fp32 CUDA tensor"):
        device_alpha(torch.zeros(1, dtype=torch.float64, device=dev), a, b)
    with pytest.raises(ValueError, match=r"alpha_out must be a 1-element fp32 CUDA tensor"):
        device_alpha(torch.zeros(1, dtype=torch.float32), a, b)
    with pytest.raises(ValueError, match=r"descale_b must be a 1-element fp32 CUDA tensor"):
        device_alpha(slot, a, torch.zeros(1, dtype=torch.bfloat16, device=dev))
    with pytest.raises(ValueError, match=r"descale_a must be a 1-element fp32 CUDA tensor"):
        device_alpha(slot, 0.25, b)
    with pytest.raises(ValueError, match=r"scale_out must be a 1-element fp32 CUDA tensor"):
        device_alpha(slot, a, b, torch.zeros(1, 2, dtype=torch.float32, device=dev))
    if torch.cuda.device_count() > 1:
        other = (dev.index if dev.index is not None else torch.cuda.current_device()) + 1
        with pytest.raises(ValueError, match=r"descale_b is on cuda:\d+ but the alpha slot is on"):
            device_alpha(slot, a, torch.tensor([0.03], dtype=torch.float32, device=f"cuda:{other % torch.cuda.device_count()}"))
    # the MN-major drivers thread it: a plan without alpha refuses one, a plan with alpha requires one (run_proj_gemm's gate)
    t, dm, hd = 256, 512, 1024
    with_alpha = ProjGemmPlan(
        graph=None,
        a=None,
        b=None,
        c=None,
        m=dm,
        k=t,
        n=hd,
        label="wa",
        dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        alpha=object(),
        a_major="m",
        b_major="n",
    )
    without = ProjGemmPlan(
        graph=None, a=None, b=None, c=None, m=t, k=dm, n=hd, label="wo", dtype=torch.bfloat16, out_dtype=torch.bfloat16, a_major="k", b_major="n"
    )
    dy = torch.zeros(t, dm, device=dev, dtype=torch.bfloat16)
    x = torch.zeros(t, hd, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(1, dtype=torch.uint8, device=dev)
    with pytest.raises(ValueError, match=r"built with alpha=True; pass alpha="):
        run_wgrad_gemm(with_alpha, dy, x, torch.zeros(dm, hd, device=dev, dtype=torch.bfloat16), ws)
    with pytest.raises(ValueError, match=r"has no alpha epilogue.*refusing to drop the value"):
        run_dgrad_gemm(without, dy, torch.zeros(dm, hd, device=dev, dtype=torch.bfloat16), torch.zeros(t, hd, device=dev, dtype=torch.bfloat16), ws, alpha=v)


# ---------------------------------------------------------------------------
# The driver's contract: zero-copy binds, the plan pins, split-K, workspace, stream
# ---------------------------------------------------------------------------


@requires_rubin
def test_bind_is_zero_copy(monkeypatch):
    """The drivers bind VIEWS: the A / B / C buffers ``run_proj_gemm`` receives share their
    ``data_ptr()`` with the caller's tensors and carry the declared rank-3 dims and strides, and
    two further executes allocate nothing (``torch.cuda.memory_allocated`` delta 0 -- Rule 1: no
    per-execute allocation; a ``reshape`` that copied would show here)."""
    import cudnn.gated_attention_block.kernels.proj_gemm as pg

    seen = {}
    orig = pg.run_proj_gemm

    def spy(plan_, a, w, out, workspace, handle=None, **kw):
        seen.update(a=a, w=w, out=out)
        return orig(plan_, a, w, out, workspace, handle, **kw)

    monkeypatch.setattr(pg, "run_proj_gemm", spy)
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16)
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    ws = _ws(plan)
    run_wgrad_gemm(plan, dy, x, dw, ws)
    torch.cuda.synchronize()
    assert seen["a"].data_ptr() == dy.data_ptr() and seen["w"].data_ptr() == x.data_ptr() and seen["out"].data_ptr() == dw.data_ptr()
    assert tuple(seen["a"].shape) == (1, m, k) and tuple(seen["a"].stride()) == (k * m, 1, m), (seen["a"].shape, seen["a"].stride())
    assert tuple(seen["w"].shape) == (1, k, n) and tuple(seen["w"].stride()) == (k * n, n, 1)
    before = torch.cuda.memory_allocated()
    run_wgrad_gemm(plan, dy, x, dw, ws)
    run_wgrad_gemm(plan, dy, x, dw, ws)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "an execute allocated"
    # dgrad: the un-transposed weight is bound as the declared [1, K, N] row-major view.
    m, k, n = _stage_mkn("B2_do_gated", "test", 2048)
    plan = _plan("dgrad", m, k, n, torch.bfloat16)
    dy, w, dx = _dgrad_operands(m, k, n, torch.bfloat16)
    ws = _ws(plan)
    run_dgrad_gemm(plan, dy, w, dx, ws)
    torch.cuda.synchronize()
    assert seen["a"].data_ptr() == dy.data_ptr() and seen["w"].data_ptr() == w.data_ptr() and seen["out"].data_ptr() == dx.data_ptr()
    assert tuple(seen["w"].shape) == (1, k, n) and tuple(seen["w"].stride()) == (k * n, n, 1)
    before = torch.cuda.memory_allocated()
    run_dgrad_gemm(plan, dy, w, dx, ws)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "an execute allocated"


@requires_rubin
@pytest.mark.parametrize("stage", _WGRAD + _DGRAD)
def test_plan_is_frost_and_named(stage):
    """Every backward plan at the test geometry: ``frost_gemm`` is pinned in the graph's ranked
    list (the name, exact or with its knob bracket), the plan IS the forced JIT at the named tile
    (a fallback to the heuristic is a FAILURE here, not a skip: it changes route, config and
    possibly split-K), one slice, and the majors are stamped for the perf table."""
    m, k, n = _stage_mkn(stage, "test", 2048)
    kind = "wgrad" if stage in _WGRAD else "dgrad"
    plan = _plan(kind, m, k, n, torch.bfloat16)
    names = [plan.graph.get_plan_name_at_index(i) for i in range(len(plan.graph.plans))]
    assert _frost_plan_index(names) is not None, names
    assert plan.tile_config_name == _FORCED_TILE, (plan.tile_config_name, plan.route)
    assert plan.jit is not None and plan.route == "graph+jit"
    assert plan.jit.config.split_k_slices == 1 and plan.split_k == 0
    assert (plan.a_major, plan.b_major) == (("m", "n") if kind == "wgrad" else ("k", "n"))
    assert (plan.m, plan.k, plan.n) == (m, k, n)


@requires_rubin
@pytest.mark.parametrize("split_k,t", [(0, 2048), (2, 2048), (2, 4104)], ids=["auto", "split_k=2", "split_k=2-T4104"])
def test_two_runs_bitwise(split_k, t):
    """Determinism by construction: two executes of the same plan agree bit for bit -- the
    split-K reducer is a fixed-order tree, no atomics anywhere.  The ``split_k=2`` plan is a JIT
    CONFIG replace (``plan.jit.config.split_k_slices == 2``), not a graph-route knob replay, and its
    fp32 partials show in ``plan.workspace_bytes``; its result meets the same fp64 bound.  The
    block's wgrads have K = T, so ``T = 4104`` splits an UNEVEN K (65 CTA-K tiles of 64 over two
    slices, the last tile ragged) -- the slice bookkeeping the even case never exercises."""
    m, k, n = _stage_mkn("B1_dw_o", "test", t)
    plan = _plan("wgrad", m, k, n, torch.bfloat16, split_k)
    # The catalog spells a split config by suffix (`TileConfig.name`, `by_name` round-trips it), so the
    # stamped name says which slice count runs -- what the perf table wants to see.
    assert plan.tile_config_name == _FORCED_TILE + ("_splitK2" if split_k == 2 else "") and plan.jit is not None
    assert plan.jit.config.split_k_slices == (2 if split_k == 2 else 1) and plan.split_k == split_k
    if split_k == 2:
        assert plan.jit.workspace_bytes > 0 and plan.workspace_bytes >= plan.jit.workspace_bytes >= 2 * m * n * 4
    dy, x, dw1 = _wgrad_operands(m, k, n, torch.bfloat16)
    dw2 = torch.zeros_like(dw1)
    ws = _ws(plan)
    run_wgrad_gemm(plan, dy, x, dw1, ws)
    run_wgrad_gemm(plan, dy, x, dw2, ws)
    torch.cuda.synchronize()
    assert torch.equal(dw1, dw2), f"two executes differ: max|diff| = {(dw1.float() - dw2.float()).abs().max().item()}"
    _assert_close_vs_fp64(dw1, dy.double().T @ x.double(), f"B1 wgrad split_k={split_k}, T={t}")


def _refuse_the_forced_compile(monkeypatch) -> None:
    """Stub ``compiler.jit_from_cudnn_graph`` to decline (``NotImplementedError``) ONLY the compile ``build_proj_gemm``
    issues ITSELF (its caller frame), so the arms of its compile-exception handler are exercised without a shape the
    forced tile cannot take.  The backend's own ``frost_gemm`` plan -- the heuristic's pick, built inside ``_backend_pin``
    -> ``g.build_plans()`` -> the engine -> ``graph_analyzer`` -- goes through the same entry point with the same kwargs
    and must still build, or the graph fallback under test could never be reached.  Keyed on the caller, not on the
    config NAME: a heuristic that one day picks the forced tile's family would otherwise turn the test into an unrelated
    ``_backend_pin`` error.  ``monkeypatch`` restores the attribute at teardown."""
    import cudnn.gemm.frost.compiler as compiler

    orig = compiler.jit_from_cudnn_graph

    def decline(graph, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == "build_proj_gemm":
            config = kwargs.get("config", args[0] if args else None)
            raise NotImplementedError(f"probe: {getattr(config, 'name', config)} declined")
        return orig(graph, *args, **kwargs)

    monkeypatch.setattr(compiler, "jit_from_cudnn_graph", decline)


@requires_rubin
def test_split_k_1_pins_one_slice_and_refuses_the_fallback(monkeypatch):
    """``split_k=1`` is a PIN, not a no-op: the plan is the forced JIT at one slice; and when the
    forced compile is refused, the graph-heuristic fallback ``split_k=0`` takes silently is REFUSED
    (typed ``SplitKPinRefused`` carrying the compiler's decline) -- what the follow-up recompute
    plans need for their bit-identical claim.  The compiler is stubbed to decline so the
    fallback arm is exercised without a shape the tile cannot take."""
    m, k, n = _stage_mkn("B7_dw_qkvg", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16, 1)
    assert plan.jit is not None and plan.jit.config.split_k_slices == 1 and plan.tile_config_name == _FORCED_TILE and plan.route == "graph+jit"
    assert plan.split_k == 1 and plan.jit.workspace_bytes == 0
    # The recompute plans' premise, pinned: the pinned one-slice plan and the driver's pick run the SAME config,
    # so their outputs are bitwise equal -- a tripwire against a future catalog `split_k_slices`
    # change (or an auto-split) on the forced tile.
    dy, x, dw_pinned = _wgrad_operands(m, k, n, torch.bfloat16)
    dw_auto = torch.zeros_like(dw_pinned)
    plan0 = _plan("wgrad", m, k, n, torch.bfloat16, 0)
    run_wgrad_gemm(plan, dy, x, dw_pinned, _ws(plan))
    run_wgrad_gemm(plan0, dy, x, dw_auto, _ws(plan0))
    torch.cuda.synchronize()
    assert dw_pinned.abs().max().item() > 0 and torch.equal(
        dw_pinned, dw_auto
    ), "split_k=1 (pinned) and split_k=0 (the driver's pick) differ on the forced tile"

    _refuse_the_forced_compile(monkeypatch)
    with pytest.raises(SplitKPinRefused, match=r"split_k=1 pins the JIT at .*cluster2x1_2ctamma.*fallback is refused") as ei:
        build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="pinned", a_major="m", b_major="n", split_k=1)
    assert isinstance(ei.value.__cause__, NotImplementedError)
    with pytest.raises(SplitKPinRefused, match="split_k=2"):
        build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="pinned2", a_major="m", b_major="n", split_k=2)
    # split_k=0 under the same decline: the documented fallback -- a GRAPH-route plan, the heuristic's config.
    fell = build_proj_gemm(m=m, k=k, n=n, dtype=torch.bfloat16, label="fallback", a_major="m", b_major="n", split_k=0)
    assert fell.jit is None and fell.route == "graph" and fell.tile_config_name == "heuristic (graph engine)"


@requires_rubin
def test_split_k_workspace_must_be_a_cuda_buffer_on_the_launch_device():
    """A split-K plan carves its fp32 partials out of the CALLER's workspace.  A host (CPU) buffer of the right
    size and alignment passes the presence / size / alignment checks, so without a device pin it reached the
    launch boundary as a bogus device pointer.  ``run_proj_gemm`` binds the workspace with the launch device
    (``out.device``): a CPU buffer -- and a buffer on another CUDA device, where one is visible -- is a typed
    ``ValueError`` BEFORE any launch (the output is untouched); the CUDA twin still runs (positive control).
    Shared by both backward drivers and the forward."""
    m, k, n = _stage_mkn("B7_dw_qkvg", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16, 2)
    assert plan.jit is not None and plan.jit.workspace_bytes > 0, "split_k=2 must carry fp32 partials"
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    before = dw.clone()
    ws_cpu = torch.empty(plan.workspace_bytes, dtype=torch.uint8)  # host memory: right size, aligned, wrong place
    with pytest.raises(ValueError, match="workspace must be a CUDA device buffer"):
        run_wgrad_gemm(plan, dy, x, dw, ws_cpu)
    torch.cuda.synchronize()
    assert torch.equal(dw, before), "the refusal must fire before any launch"
    if torch.cuda.device_count() > 1:
        other = (dw.device.index + 1) % torch.cuda.device_count()  # an ordinal that is NOT the launch device, whichever that is
        ws_other = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=f"cuda:{other}")
        with pytest.raises(ValueError, match=rf"workspace must be on {dw.device}, got {ws_other.device}"):
            run_wgrad_gemm(plan, dy, x, dw, ws_other)
        torch.cuda.synchronize()
        assert torch.equal(dw, before)
    run_wgrad_gemm(plan, dy, x, dw, _ws(plan))  # positive control: the CUDA workspace on the launch device
    torch.cuda.synchronize()
    assert dw.abs().max().item() > 0


@requires_rubin
def test_workspace_is_honest():
    """The block's engine scratch is ``max(plan.workspace_bytes)`` over every backward plan
    REGARDLESS of route: a split-K plan's fp32 partials are in its number, a too-small or
    missing workspace is a typed refusal BEFORE the launch (an overflow would not fail it), and the
    max covers each plan.

    The plan under test must make ``workspace_bytes = max(graph, jit)`` LOAD-BEARING: at the test
    geometry the backend's own frost_gemm plan auto-splits 3-way (``SPLIT_K_SLC=3`` in its name,
    ``graph.get_workspace_size()`` = 12 MiB, measured on Rubin cc 10.7, 2026-09-29), which already exceeds
    a JIT ``split_k=2``'s 8 MiB -- a ``workspace_bytes`` that forgot the JIT term would still pass
    there.  ``split_k=8`` needs 32 MiB, and the precondition below asserts the ordering, so the
    test tells you to move the shape if the heuristic ever splits deeper instead of going quiet."""
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan8 = _plan("wgrad", m, k, n, torch.bfloat16, 8)
    assert plan8.jit is not None and plan8.jit.config.split_k_slices == 8 and plan8.tile_config_name == _FORCED_TILE + "_splitK8"
    graph_ws, jit_ws = int(plan8.graph.get_workspace_size()), int(plan8.jit.workspace_bytes)
    assert jit_ws >= 8 * m * n * 4
    assert (
        graph_ws < jit_ws
    ), f"precondition: the graph's number ({graph_ws}) must be BELOW the JIT's ({jit_ws}) for max() to be load-bearing -- raise split_k or move the shape"
    assert plan8.workspace_bytes >= jit_ws, f"plan.workspace_bytes = {plan8.workspace_bytes} forgot the JIT's split-K partials ({jit_ws}; graph {graph_ws})"
    dy, x, dw = _wgrad_operands(m, k, n, torch.bfloat16)
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan8, dy, x, dw, torch.empty(16, dtype=torch.uint8, device="cuda"))
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan8, dy, x, dw, torch.empty(graph_ws, dtype=torch.uint8, device="cuda"))  # the graph-only size is NOT enough
    with pytest.raises(ValueError, match="workspace"):
        run_wgrad_gemm(plan8, dy, x, dw, None)
    plans = [_plan("wgrad" if st in _WGRAD else "dgrad", *_stage_mkn(st, "test", 2048), torch.bfloat16) for st in _WGRAD + _DGRAD] + [plan8]
    scratch = max(p.workspace_bytes for p in plans)
    assert scratch >= jit_ws and all(scratch >= p.workspace_bytes for p in plans)
    ws = torch.empty(scratch, dtype=torch.uint8, device="cuda")
    run_wgrad_gemm(plan8, dy, x, dw, ws)  # the shared scratch serves the split-K plan
    dw2 = torch.zeros_like(dw)
    run_wgrad_gemm(plan8, dy, x, dw2, ws)
    torch.cuda.synchronize()
    assert torch.equal(dw, dw2), "two executes of the split_k=8 plan differ"
    _assert_close_vs_fp64(dw, dy.double().T @ x.double(), "B1 wgrad split_k=8 on the shared scratch")


@requires_rubin
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_stream_threading(how):
    """The GEMM launches on the CALLER's stream -- ambient (``with torch.cuda.stream(s):``) or
    explicit (``stream=``) -- through ``run_proj_gemm``'s JIT ``stream=`` (Rule 5).  Deterministic
    probe (``test_a_caller_stream_orders_every_stage``): the default stream is parked behind a
    long spin and the INPUTS are zeroed on the side stream right after the GEMM, so a launch that
    landed on the default stream would run late and read zeros; correct threading gives a result
    BIT-IDENTICAL to the default-stream run."""
    m, k, n = _stage_mkn("B1_dw_o", "test", 2048)
    plan = _plan("wgrad", m, k, n, torch.bfloat16)
    dy, x, dw_ref = _wgrad_operands(m, k, n, torch.bfloat16)
    ws = _ws(plan)
    run_wgrad_gemm(plan, dy, x, dw_ref, ws)
    torch.cuda.synchronize()
    assert dw_ref.abs().max().item() > 0
    dw = torch.zeros_like(dw_ref)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    park_the_default_stream()
    if how == "ambient":
        with torch.cuda.stream(side):
            run_wgrad_gemm(plan, dy, x, dw, ws)
    else:
        run_wgrad_gemm(plan, dy, x, dw, ws, stream=side.cuda_stream)
    with torch.cuda.stream(side):
        dy.zero_()  # ordered AFTER the GEMM on the side stream; a GEMM parked on the default stream reads these zeros instead
        x.zero_()
    torch.cuda.synchronize()
    assert torch.equal(dw, dw_ref), f"the GEMM ran off the caller's stream ({how}): zeros = {(dw == 0).float().mean().item():.0%}"


# ---------------------------------------------------------------------------
# Reject / host tests -- run anywhere
# ---------------------------------------------------------------------------


def test_build_refuses_an_unknown_major_or_split_k_before_any_graph():
    """The kwargs are validated FIRST: a typo in a major or a negative / bool split_k is a
    ``ValueError`` naming the kwarg, before ``cudnn`` builds anything."""
    with pytest.raises(ValueError, match="a_major"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", a_major="n")
    with pytest.raises(ValueError, match="b_major"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", b_major="m")
    with pytest.raises(ValueError, match="split_k"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", split_k=-1)
    with pytest.raises(ValueError, match="split_k"):
        build_proj_gemm(m=256, k=256, n=256, dtype=torch.bfloat16, label="bad", split_k=True)


def test_split_k_needs_pin_frost():
    """A pinned split is a JIT plan by definition; ``pin_frost=False`` deselects the FROST engine, so
    there is nothing to pin -- a typed ``ValueError`` naming both kwargs, not a silently dropped knob
    (before this, the plan came back ``route='graph'``, ``jit=None`` with ``plan.split_k`` still
    stamped with the request)."""
    for split_k in (1, 2):
        with pytest.raises(ValueError, match=rf"split_k={split_k}.*pin_frost=False"):
            build_proj_gemm(m=512, k=2048, n=2048, dtype=torch.bfloat16, label="x", a_major="m", b_major="n", split_k=split_k, pin_frost=False)


@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
def test_fp8_mn_major_outside_the_table_is_a_typed_decline():
    """The fp8 MN-major lift is a TABLE (``FP8_MN_MAJOR_VALIDATED``): exactly the wgrad ``("m", "n")`` and
    dgrad ``("k", "n")`` e4m3 triples :func:`test_fp8_mn_major_matches_fp64_on_cc107` validated.  Anything else
    with an fp8 side and a non-K major -- the ``("m", "k")`` triple nobody's GEMM needs, and a MIXED dtype pair
    (a bf16 A against an e4m3 W, or the reverse) -- is a typed ``NotImplementedError`` BEFORE any graph exists,
    naming the table, rather than an admitted but never-run rendering.  The forward's K-major fp8 plans are
    untouched (``test_proj_gemm.py``), and the table holds only e4m3 (the driver's one fp8 dtype).

    The table also holds at the FORCED tile only -- what the device cells validated: an N that 256 does not divide
    (the heuristic's tile), the K64 twin named EXPLICITLY (K64 is reached through ``mma_tile_k_bytes=64``, how it was
    validated) and ``pin_frost=False`` (the graph route, no JIT) are the same typed decline, naming the tile."""
    assert FP8_MN_MAJOR_VALIDATED == frozenset({(_FP8, "m", "n"), (_FP8, "k", "n")})
    with pytest.raises(NotImplementedError, match=r"fp8 \(e4m3\).*a_major='m', b_major='k'.*validated \(dtype, a_major, b_major\) triples") as ei:
        build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, label="fp8_mk", a_major="m", b_major="k")
    assert "a_major='m', b_major='n'" in str(ei.value) and "a_major='k', b_major='n'" in str(ei.value), str(ei.value)
    with pytest.raises(NotImplementedError, match=r"fp8 \(e4m3\).*w_dtype=torch.float8_e4m3fn"):
        build_proj_gemm(m=512, k=2048, n=2048, dtype=torch.bfloat16, w_dtype=_FP8, label="fp8_w", a_major="k", b_major="n")
    with pytest.raises(NotImplementedError, match=r"fp8 \(e4m3\).*dtype=torch.float8_e4m3fn, w_dtype=torch.bfloat16"):
        build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, w_dtype=torch.bfloat16, label="fp8_a", a_major="m", b_major="n")
    assert _forced_tile_config(4112) is None and 4112 % 16 == 0  # past the TMA rule, short of the forced tile
    with pytest.raises(NotImplementedError, match=r"validated at the forced tile '" + _FORCED_TILE + r"'.*n=4112.*the graph heuristic's tile"):
        build_proj_gemm(m=512, k=2048, n=4112, dtype=_FP8, label="fp8_n4112", a_major="k", b_major="n")
    with pytest.raises(NotImplementedError, match=r"validated at the forced tile .*resolves to the tile '" + _FORCED_TILE_K64 + "'"):
        build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, label="fp8_k64_by_name", a_major="m", b_major="n", tile_config=_FORCED_TILE_K64)
    with pytest.raises(
        NotImplementedError, match=r"validated at the forced tile .*pin_frost=False resolves to the tile '" + _FORCED_TILE + r"' on the graph route \(no JIT\)"
    ):
        build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, label="fp8_unpinned", a_major="k", b_major="n", pin_frost=False)


# (dtype, a_major, b_major, mma_tile_k_bytes, what the refused forced compile must become)
_REFUSED_FORCED_COMPILE_CASES = [
    pytest.param(_FP8, "m", "n", None, "typed decline", id="e4m3-mn-K-default"),
    pytest.param(_FP8, "k", "n", None, "typed decline", id="e4m3-kn-K-default"),
    pytest.param(_FP8, "m", "n", 32, "compiler's decline", id="e4m3-mn-K32-explicit"),
    pytest.param(torch.bfloat16, "m", "n", None, "graph fallback", id="bf16-mn-control"),
]


@requires_rubin
@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
@pytest.mark.parametrize("dtype,a_major,b_major,k_bytes,expect", _REFUSED_FORCED_COMPILE_CASES)
def test_fp8_mn_major_refuses_the_graph_fallback_when_the_forced_compile_declines(monkeypatch, dtype, a_major, b_major, k_bytes, expect):
    """The table holds at the forced tile on the FROST JIT -- and a forced compile the compiler REFUSES must not escape
    that promise through the graph-heuristic fallback the bf16 plans take under the DEFAULT knobs (``tile_config="auto"``,
    ``split_k=0``, ``mma_tile_k_bytes=None``): the heuristic's tile (on cc 10.7 also its K64 MMA form) is not the one
    :func:`test_fp8_mn_major_matches_fp64_on_cc107` validated.  With the forced compile stubbed to decline (the graph
    engine's own build intact, :func:`_refuse_the_forced_compile`) both admitted e4m3 triples are the admission's typed
    ``NotImplementedError`` -- naming the tile, the refused compile and the unvalidated fallback, the compiler's decline
    chained -- and NO plan with route ``"graph"`` comes back; an explicit ``mma_tile_k_bytes=32`` still surfaces the
    compiler's own decline (unchanged); the bf16 control at the same shape keeps its documented fallback (route
    ``"graph"``, no JIT, the heuristic's config)."""
    _refuse_the_forced_compile(monkeypatch)
    m = k = n = 256  # the forced tile's N; the TMA rule on M / N and the fp8 K rule are met
    assert _forced_tile_config(n) == _FORCED_TILE
    kw = dict(m=m, k=k, n=n, dtype=dtype, label=f"refused_{a_major}{b_major}", a_major=a_major, b_major=b_major, tile_config="auto", split_k=0)
    kw.update(mma_tile_k_bytes=k_bytes, alpha=dtype is _FP8)  # the backward's e4m3 plans carry the descale epilogue
    if expect == "typed decline":
        with pytest.raises(
            NotImplementedError,
            match=r"validated at the forced tile '"
            + _FORCED_TILE
            + r"' on the FROST JIT only .*that compile was refused \(NotImplementedError: probe: "
            + _FORCED_TILE
            + r" declined\); the graph heuristic's tile is not validated for e4m3 MN-major operands, so the fallback .*is refused too",
        ) as ei:
            build_proj_gemm(**kw)
        assert isinstance(ei.value.__cause__, NotImplementedError) and str(ei.value.__cause__).startswith("probe: "), repr(ei.value.__cause__)
    elif expect == "compiler's decline":
        with pytest.raises(NotImplementedError, match=r"^probe: " + _FORCED_TILE + r" declined$") as ei:
            build_proj_gemm(**kw)
        assert ei.value.__cause__ is None  # re-raised as is, not re-wrapped
    else:
        plan = build_proj_gemm(**kw)
        assert plan.jit is None and plan.route == "graph" and plan.tile_config_name == "heuristic (graph engine)", (plan.route, plan.tile_config_name)


@pytest.mark.skipif(_FP8 is None, reason="this torch has no float8_e4m3fn")
def test_fp8_k_rule_is_the_k_contiguous_operands_rule(monkeypatch):
    """The fp8 ``K % 16`` decline is TMA's 16-byte rule on an operand whose CONTIGUOUS axis is K -- the forward's two
    K-major operands, a dgrad's K-major A -- raised by name, naming those operands, before any graph exists.  A wgrad
    (M-major A, N-major B) has no K-contiguous operand: its 16-byte rule fell on M (and N) and still fires first, and a
    ragged ``K = T`` such as the bf16 twins' 4104 passes every pre-graph check -- the graph constructor is the first
    thing it reaches (a probe stands in for it: building the plan needs the device the accept cells run on, and
    :func:`test_fp8_mn_major_matches_fp64_on_cc107` at ``T = 4104`` is that validation, in both MMA K forms)."""
    import cudnn

    with pytest.raises(ValueError, match=r"FP8 operands need K % 16 == 0 .*on A \(k-major\) and B \(k-major\): K is its contiguous axis.*got K=4104"):
        build_proj_gemm(m=512, k=4104, n=2048, dtype=_FP8, label="fwd_k4104")
    with pytest.raises(ValueError, match=r"FP8 operands need K % 16 == 0 .*on A \(k-major\): K is its contiguous axis.*got K=4104"):
        build_proj_gemm(m=2048, k=4104, n=2048, dtype=_FP8, label="dgrad_k4104", a_major="k", b_major="n")
    with pytest.raises(ValueError, match=r"A is m-major.*M % 16 == 0.*got M=520"):
        build_proj_gemm(m=520, k=4104, n=2048, dtype=_FP8, label="wgrad_m520", a_major="m", b_major="n")

    class _ReachedTheGraph(Exception):
        """Raised by the ``cudnn.pygraph`` stand-in: the declaration passed every pre-graph check."""

    def probe(*args, **kwargs):
        """Stands in for ``cudnn.pygraph``: raises ``_ReachedTheGraph`` instead of building a graph (the device-side half is
        the accept cells')."""
        raise _ReachedTheGraph()

    monkeypatch.setattr(cudnn, "pygraph", probe)
    with pytest.raises(_ReachedTheGraph):
        build_proj_gemm(m=512, k=4104, n=2048, dtype=_FP8, label="wgrad_k4104", a_major="m", b_major="n", alpha=True, mma_tile_k_bytes=64)


def test_dense_mma_tile_k_bytes_is_an_8bit_knob():
    """``mma_tile_k_bytes`` on a DENSE plan: refused on bf16 / f16 (one MMA K width exists -- a typed
    ``ValueError``, never a silently kept default), refused outside {32, 64} on e4m3, refused with
    ``pin_frost=False`` (no JIT to re-target: the knob would be dropped), all BEFORE any graph exists.
    The admitted e4m3 values are exercised on the device by :func:`test_fp8_mn_major_matches_fp64_on_cc107`."""
    for dt in (torch.bfloat16, torch.float16):
        with pytest.raises(ValueError, match=r"mma_tile_k_bytes is a knob of the 8-bit MMA paths.*one MMA K width"):
            build_proj_gemm(m=512, k=2048, n=2048, dtype=dt, label="k64_bf16", a_major="m", b_major="n", mma_tile_k_bytes=64)
    if _FP8 is not None:
        with pytest.raises(ValueError, match=r"mma_tile_k_bytes must be None, 32 or 64"):
            build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, label="k48", mma_tile_k_bytes=48)
        with pytest.raises(ValueError, match=r"mma_tile_k_bytes=64.*pin_frost=False"):
            build_proj_gemm(m=512, k=2048, n=2048, dtype=_FP8, label="k64_unpinned", mma_tile_k_bytes=64, pin_frost=False)


def test_forced_tile_pick_is_dtype_agnostic():
    """The forced-tile pick takes the shape only -- ``_forced_tile_config(n)`` -- and names the K=32 form;
    the 64-byte MMA K reaches a plan ONLY through an explicit ``mma_tile_k_bytes=64`` (the backward GEMM
    stage's request).  Keying the pick on the dtype would silently flip the FORWARD's pinned fp8 plans."""
    import inspect

    assert list(inspect.signature(_forced_tile_config).parameters) == ["n"]
    assert _forced_tile_config(4096) == _FORCED_TILE and _forced_tile_config(8192) == _FORCED_TILE and _forced_tile_config(17408) == _FORCED_TILE
    assert _forced_tile_config(4100) is None


def test_workspace_bytes_is_the_max_of_graph_and_jit():
    """``ProjGemmPlan.workspace_bytes`` on fakes, so the ``max(graph, jit)`` rule is pinned
    independently of what the backend heuristic happens to report: the JIT's split-K
    partials count whenever a JIT exists (``run_proj_gemm`` launches it), the graph's number when
    it is the larger, the JIT's alone on the JIT-only route, and never 0."""

    class _Graph:
        def __init__(self, ws):
            self.ws = ws

        def get_workspace_size(self):
            return self.ws

    class _Jit:
        def __init__(self, ws):
            self.workspace_bytes = ws

    plan = ProjGemmPlan(graph=_Graph(12 << 20), a=None, b=None, c=None, m=512, k=2048, n=2048, label="ws", dtype=torch.bfloat16)
    plan.route = "graph+jit"
    plan.jit = _Jit(32 << 20)
    assert plan.workspace_bytes == 32 << 20  # the JIT's split-K partials win
    plan.jit = _Jit(8 << 20)
    assert plan.workspace_bytes == 12 << 20  # the graph's number wins (the test-geometry case above)
    plan.jit = _Jit(0)
    assert plan.workspace_bytes == 12 << 20
    plan.jit = None
    plan.route = "graph"
    assert plan.workspace_bytes == 12 << 20
    plan.graph = _Graph(0)
    assert plan.workspace_bytes == 1  # never 0: an empty buffer would fail Workspace's presence check
    plan.route = "jit-only"
    plan.jit = _Jit(8 << 20)
    assert plan.workspace_bytes == 8 << 20  # no backend plan to ask
    plan.jit = None
    assert plan.workspace_bytes == 1


def test_tma_rule_is_typed():
    """B1 at ``dm = 4100``: A is M-major, so TMA reads M contiguously and needs ``M % 8 == 0``
    at bf16 (16-byte contiguous-extent rule).  A typed ``ValueError`` naming the operand
    and the rule, raised by the DRIVER -- the engine's own decline would surface only as
    'no frost_gemm plan' after the graph was built."""
    with pytest.raises(ValueError, match=r"A is m-major.*M % 8 == 0.*16-byte"):
        build_proj_gemm(m=4100, k=8192, n=8192, dtype=torch.bfloat16, label="dw_o", a_major="m", b_major="n")
    with pytest.raises(ValueError, match=r"B is n-major.*N % 8 == 0.*16-byte"):
        build_proj_gemm(m=4096, k=8192, n=8196, dtype=torch.bfloat16, label="dw_o", a_major="m", b_major="n")


@requires_cuda
def test_wrong_major_view_is_a_typed_refusal():
    """A K-major plan handed an M-major view (``dy.view(T, dm).t()``) -> ``ValueError`` naming
    the operand and BOTH strides, raised by the DRIVER before any launch.  Why the driver
    checks: the JIT route re-reads runtime strides and refuses a mismatch, but the graph
    fallback binds pointers against the DECLARED strides with no check -- a wrong view there
    is a silent reinterpretation.  Hand-built plans, so this runs on any device
    (nothing is launched; the check precedes every route)."""
    t, dm, hd = 256, 512, 1024
    dy = torch.zeros(t, dm, device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16)
    dw = torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    # (1) the plan's majors are the first gate: a wgrad through a K-major plan is refused by name.
    kmajor = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="kmajor", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"a_major='k'.*b_major='k'"):
        run_wgrad_gemm(kmajor, dy, x, dw, ws)
    # (2) the right plan, the wrong VIEW: dy_like arrives already transposed, so the A view's
    #     stride-1 axis is K, not the declared M -- refused naming A and both strides.
    mn = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="mn", dtype=torch.bfloat16, a_major="m", b_major="n")
    with pytest.raises(ValueError, match=r"A.*stride-1 axis.*declared") as ei:
        run_wgrad_gemm(mn, dy.t().contiguous().t(), x, dw, ws)  # a [T, rows] view whose storage is [rows, T]
    assert "stride" in str(ei.value)
    # (3) B through a plan that declared it N-major, handed a K-major storage: same refusal for B.
    with pytest.raises(ValueError, match=r"B.*stride-1 axis.*declared"):
        run_wgrad_gemm(mn, dy, x.t().contiguous().t(), dw, ws)
    # (4) a dgrad through a plan whose B is K-major (the FORWARD's declaration): refused by name.
    fwd = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=dm, n=hd, label="fwd", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"b_major='k'"):
        run_dgrad_gemm(fwd, dy, torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16), torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16), ws)
    # (5) the right major, a PADDED view: x as a column slice of a wider slab has the declared stride-1
    #     axis but a row stride != n -- refused naming both strides, on the graph route (jit=None) ...
    x_wide = torch.zeros(t, hd + 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"B.*strides \(\d+, 1088, 1\) but the plan declared \(\d+, 1024, 1\).*exactly the declared layout"):
        run_wgrad_gemm(mn, dy, x_wide[:, :hd], dw, ws)
    # (6) ... and on the JIT route alike (the JIT re-labels B into kernel order only on an exact match,
    #     so its own refusal would be the generic 'input layout' one; the driver's typed message comes first).
    mn_jit = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="mn_jit", dtype=torch.bfloat16, a_major="m", b_major="n")
    mn_jit.jit = object()
    with pytest.raises(ValueError, match=r"B.*strides \(\d+, 1088, 1\) but the plan declared"):
        run_wgrad_gemm(mn_jit, dy, x_wide[:, :hd], dw, ws)
    # (7) a padded A: dy_like as a column slice of a wider slab -> the transposed view's K stride != m.
    dy_wide = torch.zeros(t, dm + 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"A.*strides \(\d+, 1, 576\) but the plan declared \(\d+, 1, 512\)"):
        run_wgrad_gemm(mn_jit, dy_wide[:, :dm], x, dw, ws)
    # (8) a dgrad's un-transposed weight as a column slice: same refusal for B.
    dg = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=dm, n=hd, label="dg", dtype=torch.bfloat16, a_major="k", b_major="n")
    w_wide = torch.zeros(dm, hd + 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"B \(w\).*strides \(\d+, 1088, 1\) but the plan declared"):
        run_dgrad_gemm(dg, dy, w_wide[:, :hd], torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16), ws)


@requires_cuda
def test_wrong_output_dtype_is_a_typed_refusal():
    """The OUTPUT's dtype is checked like A's and W's: a ``dw`` / ``dx`` / ``out`` whose dtype is
    not the plan's ``out_dtype`` is a ``ValueError`` naming the plan and both dtypes, raised by
    ``run_proj_gemm`` -- so the forward and both backward drivers share the gate -- before any
    route.  Why: the graph carries C's dtype and the JIT binds a pointer, so a same-size f16
    buffer on a bf16 plan would be filled with bf16 bit patterns (wrong values, no error), and a
    NARROWER buffer (an e4m3 ``[T, N]`` on a bf16 plan, half the bytes) would have 2-byte stores
    run past its allocation.  A spy stands in for the JIT to prove nothing launched.  Hand-built
    plans, so this runs on any device; a plan with no ``out_dtype`` (the stream-routing probes)
    checks nothing, exactly as for A / W."""
    from cudnn.gated_attention_block.kernels.proj_gemm import run_proj_gemm

    class _SpyJit:
        def __init__(self):
            self.calls = []

        def __call__(self, vp, **kw):
            self.calls.append(vp)

    t, dm, hd = 256, 512, 1024
    narrow = _FP8 if _FP8 is not None else torch.uint8  # a 1-byte output buffer on a 2-byte plan
    dy = torch.zeros(t, dm, device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    # (1) wgrad: a same-size f16 dw on a bf16 plan -- it passes the shape / stride gates and is refused by dtype.
    mn = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=dm, k=t, n=hd, label="mn", dtype=torch.bfloat16, out_dtype=torch.bfloat16, a_major="m", b_major="n")
    mn.jit = _SpyJit()
    with pytest.raises(ValueError, match=r"^mn: out is torch\.float16 but this plan was built for torch\.bfloat16; refusing to reinterpret the bytes$"):
        run_wgrad_gemm(mn, dy, x, torch.zeros(dm, hd, device="cuda", dtype=torch.float16), ws)
    # (2) dgrad: a NARROWER dx (1 B/elem) on the bf16 plan -- the case whose stores would overrun the buffer.
    dg = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=dm, n=hd, label="dg", dtype=torch.bfloat16, out_dtype=torch.bfloat16, a_major="k", b_major="n")
    dg.jit = _SpyJit()
    with pytest.raises(ValueError) as ei:
        run_dgrad_gemm(dg, dy, w, torch.zeros(t, hd, device="cuda", dtype=narrow), ws)
    assert str(ei.value) == f"dg: out is {narrow} but this plan was built for torch.bfloat16; refusing to reinterpret the bytes"
    # (3) the forward's entry point shares the gate: a K-major plan through run_proj_gemm directly.
    fwd = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=hd, n=dm, label="fwd", dtype=torch.bfloat16, out_dtype=torch.bfloat16)
    fwd.jit = _SpyJit()
    with pytest.raises(ValueError, match=r"^fwd: out is torch\.float16 but this plan was built for torch\.bfloat16"):
        run_proj_gemm(fwd, x, w, torch.zeros(t, dm, device="cuda", dtype=torch.float16), ws)
    assert not (mn.jit.calls or dg.jit.calls or fwd.jit.calls), "a mismatched output reached the launch"
    # (4) the declared dtype passes the gate and reaches the (spy) launch, once per call.
    run_wgrad_gemm(mn, dy, x, torch.zeros(dm, hd, device="cuda", dtype=torch.bfloat16), ws)
    run_dgrad_gemm(dg, dy, w, torch.zeros(t, hd, device="cuda", dtype=torch.bfloat16), ws)
    run_proj_gemm(fwd, x, w, torch.zeros(t, dm, device="cuda", dtype=torch.bfloat16), ws)
    assert [len(p.jit.calls) for p in (mn, dg, fwd)] == [1, 1, 1]
    # (5) a hand-built plan with no out_dtype (the stream-routing probes) checks nothing -- as for A / W.
    probe = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=t, k=hd, n=dm, label="probe")
    probe.jit = _SpyJit()
    run_proj_gemm(probe, x, w, torch.zeros(t, dm, device="cuda", dtype=torch.float16), ws)
    assert len(probe.jit.calls) == 1
