# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the unfused subchannel-scaled grouped GEMM tests.

Two-level NVFP4 input generation (per-(1, 16) e4m3 first-level scales plus
power-of-two f32 second-level scales at ``block2_shape`` granularity) and the
byte-exact torch reference of the kernel's op:

    d = (sum over sgk-tiles of sfa2 * sfb2 * (A @ B^T)) * alpha * prob
        (with bias: d = gemm * alpha + prob * bias)

The second-level scales are POWERS OF TWO (exp2-ceil of block_amax /
max_representable), so dividing by them is exact in f32 and the kernel-vs-
reference comparison is bit-exact (with ``vector_f32=True``, the kernels'
byte-exact-verified configuration). Ported from the bs_ggemm_harness
``tensors/fc2_dgrad.py`` generator and ``references/fc2_dgrad.py::fc2_dgrad_gemm``.
"""

import math
from typing import List, Optional, Tuple

import torch
import pytest

from fe_api.test_fe_api_utils import (
    ceil_div,
    create_sf_layout_tensor,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)
from test_low_precision_matmul import _bfloat16_to_float4_e2m1fn_x2

_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# NVFP4 encode range: format_max(e2m1) * format_max(e4m3)
NVFP4_MAX_REPRESENTABLE = _E2M1_MAX * _E4M3_MAX  # 2688.0


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round float32 to the nearest e2m1 value (dequantized FP4), same shape."""
    r = torch.zeros_like(x, dtype=torch.float32)
    r[(x > 0.25) & (x < 0.75)] = 0.5
    r[(x >= 0.75) & (x <= 1.25)] = 1.0
    r[(x > 1.25) & (x < 1.75)] = 1.5
    r[(x >= 1.75) & (x <= 2.5)] = 2.0
    r[(x > 2.5) & (x < 3.5)] = 3.0
    r[(x >= 3.5) & (x <= 5.0)] = 4.0
    r[x > 5.0] = 6.0
    r[(x < -0.25) & (x > -0.75)] = -0.5
    r[(x <= -0.75) & (x >= -1.25)] = -1.0
    r[(x < -1.25) & (x > -1.75)] = -1.5
    r[(x <= -1.75) & (x >= -2.5)] = -2.0
    r[(x < -2.5) & (x > -3.5)] = -3.0
    r[(x <= -3.5) & (x >= -5.0)] = -4.0
    r[x < -5.0] = -6.0
    return r


def _quantize_nvfp4_first_level(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """First-level NVFP4 quantization: f32 (rows, cols) in, (decoded fp4
    values f32 (rows, cols), decoded e4m3 block scales f32 (rows, cols/16)) out.
    Op sequence ported from the harness pure-torch quantizer (global encode
    scale 1)."""
    x = x.to(torch.float32)
    rows, cols = x.shape
    assert cols % 16 == 0, f"cols ({cols}) must be a multiple of the sf vec size (16)"
    xr = x.view(rows, cols // 16, 16)
    vec_max = torch.amax(torch.abs(xr), dim=-1, keepdim=True)
    fmax = torch.tensor(torch.finfo(torch.float32).max, device=x.device)

    dec = torch.min(vec_max * (1.0 / _E2M1_MAX), fmax)
    dec_f = torch.clamp(dec, -_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    enc = torch.min(torch.div(1.0, dec_f), fmax)

    scaled = xr * enc
    vals = _round_e2m1(torch.clamp(scaled, -_E2M1_MAX, _E2M1_MAX).reshape(rows, cols))
    return vals, dec_f.squeeze(-1)


def _second_level_block_amax(x: torch.Tensor, row_g: int, col_g: int) -> torch.Tensor:
    """Per-(row_g x col_g)-block max(|x|) of a 2-D tensor, f32
    (ceil(rows/row_g), ceil(cols/col_g))."""
    r, c = x.shape
    rb, cb = ceil_div(r, row_g), ceil_div(c, col_g)
    pad_r, pad_c = rb * row_g - r, cb * col_g - c
    xp = torch.nn.functional.pad(x.float().abs(), (0, pad_c, 0, pad_r))
    return xp.reshape(rb, row_g, cb, col_g).amax(dim=(1, 3))


def _quantize_two_level(x: torch.Tensor, row_g: int, col_g: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-level blockwise quantization: f32 power-of-two second-level scales at
    (row_g x col_g) granularity, then the first-level quantizer on x / sf2.
    Returns (decoded fp4 values f32, decoded first-level scales f32, sf2 f32)."""
    rows, cols = x.shape
    amax = _second_level_block_amax(x.float(), row_g, col_g)
    # Smallest power of two making block_amax / sf2 fit the one-level encode
    # range; amax-0 blocks get scale 1 (nothing to represent). exp2/log2 run on
    # CPU (mirrors the harness generator).
    amax_cpu = amax.cpu()
    sf2 = torch.exp2(torch.ceil(torch.log2(amax_cpu / NVFP4_MAX_REPRESENTABLE)))
    sf2 = torch.where(amax_cpu == 0.0, torch.ones_like(sf2), sf2).to(amax.device)
    inv = (1.0 / sf2).repeat_interleave(row_g, dim=0)[:rows].repeat_interleave(col_g, dim=1)[:, :cols]
    vals, scales = _quantize_nvfp4_first_level((x.float() * inv).to(torch.bfloat16).float())
    return vals, scales, sf2


def _pack_fp4x2(vals: torch.Tensor) -> torch.Tensor:
    """Encode decoded e2m1 values f32 (rows, cols) into a packed
    torch.float4_e2m1fn_x2 tensor (rows, cols/2). Exact: the values are on the
    e2m1 grid."""
    return _bfloat16_to_float4_e2m1fn_x2(vals.to(torch.bfloat16))


def _sf_to_mma(sf: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """Scatter decoded first-level scales f32 (l, mn, sf_k) into the MMA-tiled
    (32, 4, ceil(mn/128), 4, ceil(sf_k/4), l) cuda tensor of the given dtype."""
    from cutlass.cute.runtime import from_dlpack

    l, mn, sf_k = sf.shape
    mma_f32_cpu, _ = create_sf_layout_tensor(l, mn, sf_k * 16, 16)
    ref_cpu = sf.float().cpu().permute(1, 2, 0).contiguous().permute(0, 1, 2)  # (mn, sf_k, l)
    if ref_cpu.numel() > 0:
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_cpu),
            from_dlpack(mma_f32_cpu),
        )
    return mma_f32_cpu.to(dtype).cuda()


def _sf2_strided(sf2: torch.Tensor) -> torch.Tensor:
    """Lay out second-level scales f32 (l, rows, cols) as the kernels'
    rows-contiguous (rows, cols, l) cuda view with stride (1, rows, rows*cols)."""
    l, rows, cols = sf2.shape
    t = torch.empty((l, cols, rows), dtype=torch.float32, device="cuda").permute(2, 1, 0)
    t[:] = sf2.permute(1, 2, 0).to(t.device)
    return t


def _group_ranges(group_sizes: List[int]) -> List[Tuple[int, int]]:
    ranges, start = [], 0
    for rows in group_sizes:
        ranges.append((start, start + rows))
        start += rows
    return ranges


def _grouped_gemm_second_level_scaled(
    a: torch.Tensor,  # (mt, k) f32, first-level dequantized
    b: torch.Tensor,  # (l, n, k) f32, first-level dequantized
    sfa2: torch.Tensor,  # (mt/sgm, k/sgk) f32
    sfb2: torch.Tensor,  # (l, n/sgn, k/sgk) f32
    group_sizes: List[int],
    sgm: int,
    sgn: int,
    sgk: int,
) -> torch.Tensor:
    """c (mt, n) f32: per-expert, per-k-tile matmul with second-level scaling
    (byte-exact op order of the harness reference)."""
    mt, k = a.shape
    n = b.shape[1]
    num_k_tiles = math.ceil(k / sgk)

    a = a.float()
    b = b.float()
    sfa2 = sfa2.float().to(a.device)
    sfb2 = sfb2.float().to(a.device)

    col_scale_idx = torch.arange(n, device=a.device) // sgn
    c = torch.zeros((mt, n), dtype=torch.float32, device=a.device)
    for e, (r0, r1) in enumerate(_group_ranges(group_sizes)):
        row_scale_idx = torch.arange(r0, r1, device=a.device) // sgm
        for t in range(num_k_tiles):
            k0, k1 = t * sgk, min((t + 1) * sgk, k)
            partial = a[r0:r1, k0:k1] @ b[e, :, k0:k1].t()
            row_scale = sfa2[row_scale_idx, t]  # (rows,)
            col_scale = sfb2[e, col_scale_idx, t]  # (n,)
            c[r0:r1] += (partial * row_scale.unsqueeze(1)) * col_scale.unsqueeze(0)
    return c


def reference_unfused_subchannel_scaled(
    a_vals: torch.Tensor,  # (mt, k) f32 decoded fp4
    a_sf: torch.Tensor,  # (mt, k/16) f32 decoded first-level scales
    a_sf2: torch.Tensor,  # (mt/sgm, k/sgk) f32
    b_vals: torch.Tensor,  # (l, n, k) f32 decoded fp4
    b_sf: torch.Tensor,  # (l, n, k/16) f32
    b_sf2: torch.Tensor,  # (l, n/sgn, k/sgk) f32
    prob: torch.Tensor,  # (mt,) or (mt, 1, 1) f32
    alpha: torch.Tensor,  # (l,) f32
    bias: Optional[torch.Tensor],  # (n, l) bf16 or None
    group_sizes: List[int],
    block2_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """The kernel's op in the byte-exact reference order. Returns (mt, n) bf16."""
    sgm, sgn, sgk = block2_shape
    mt = a_vals.shape[0]
    a = a_vals * a_sf.repeat_interleave(16, dim=-1)
    b = b_vals * b_sf.repeat_interleave(16, dim=-1)
    g = _grouped_gemm_second_level_scaled(a, b, a_sf2, b_sf2, group_sizes, sgm, sgn, sgk)
    # alpha applies once, per expert
    for e, (r0, r1) in enumerate(_group_ranges(group_sizes)):
        g[r0:r1] *= float(alpha[e])
    prob_col = prob.reshape(mt, 1).to(g.device)
    if bias is None:
        d = g * prob_col
    else:
        bias_term = torch.zeros_like(g)
        for e, (r0, r1) in enumerate(_group_ranges(group_sizes)):
            bias_term[r0:r1] += bias[:, e].float()
        d = g + bias_term * prob_col
    return d.to(torch.bfloat16)


def make_unfused_subchannel_scaled_problem(
    m_per_expert: int,
    n: int,
    k: int,
    l: int,
    block2_shape: Tuple[int, int, int] = (1, 256, 256),
    with_bias: bool = False,
    seed: int = 0,
):
    """Build FE-format inputs plus the byte-exact reference output.

    Returns a dict with the wrapper-ready tensors (a/sfa/sfa2, b/sfb/sfb2,
    padded_offsets/alpha/prob[/bias]) and ``ref_d`` (valid_m, n) bf16.
    """
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("Current torch version does not support float4_e2m1fn_x2")
    torch.manual_seed(seed)
    sgm, sgn, sgk = block2_shape
    assert m_per_expert % 256 == 0 and m_per_expert % sgm == 0
    assert k % sgk == 0 and k % 32 == 0 and n % 64 == 0
    mt = l * m_per_expert
    group_sizes = [m_per_expert] * l

    a_master = torch.randn((mt, k), dtype=torch.bfloat16, device="cuda")
    b_master = torch.randn((l * n, k), dtype=torch.bfloat16, device="cuda")

    a_vals, a_sf, a_sf2 = _quantize_two_level(a_master.float(), sgm, sgk)
    # B second-level blocks must not straddle experts: quantize per expert.
    b_parts = [_quantize_two_level(b_master[e * n : (e + 1) * n].float(), sgn, sgk) for e in range(l)]
    b_vals = torch.stack([p[0] for p in b_parts])  # (l, n, k)
    b_sf = torch.stack([p[1] for p in b_parts])  # (l, n, k/16)
    b_sf2 = torch.stack([p[2] for p in b_parts])  # (l, ceil(n/sgn), k/sgk)

    # ---- FE-format tensors ----
    # A: (valid_m, k/2, 1) fp4x2, k-major, l-outermost physical layout
    a_packed = _pack_fp4x2(a_vals)  # (mt, k/2)
    a_tensor = a_packed.reshape(1, mt, k // 2).permute(1, 2, 0)
    # B: (n, k/2, l) fp4x2
    b_packed = torch.stack([_pack_fp4x2(p[0]) for p in b_parts])  # (l, n, k/2)
    b_tensor = b_packed.permute(1, 2, 0)

    sfa_tensor = _sf_to_mma(a_sf.reshape(1, mt, k // 16))
    sfb_tensor = _sf_to_mma(b_sf)

    sfa2_tensor = _sf2_strided(a_sf2.reshape(1, mt // sgm, k // sgk))
    sfb2_tensor = _sf2_strided(b_sf2)

    padded_offsets = torch.tensor(
        [sum(group_sizes[: e + 1]) for e in range(l)],
        dtype=torch.int32,
        device="cuda",
    )
    alpha = (0.75 + 0.25 * (torch.arange(l) % 4)).float().cuda()
    prob = torch.randint(-2, 2, (mt, 1, 1), dtype=torch.float32, device="cuda").float()

    bias = None
    if with_bias:
        bias_ln = (torch.randn((l, n), device="cuda") * 0.1).to(torch.bfloat16)
        bias = torch.empty((l, n), dtype=torch.bfloat16, device="cuda").t()  # (n, l) stride (1, n)
        bias.copy_(bias_ln.t())

    ref_d = reference_unfused_subchannel_scaled(
        a_vals,
        a_sf,
        a_sf2,
        b_vals,
        b_sf,
        b_sf2,
        prob,
        alpha,
        bias,
        group_sizes,
        block2_shape,
    )

    return {
        "a_tensor": a_tensor,
        "sfa_tensor": sfa_tensor,
        "sfa2_tensor": sfa2_tensor,
        "b_tensor": b_tensor,
        "b_packed": b_packed,  # (l, n, k/2) contiguous backing for discrete pointers
        "sfb_tensor": sfb_tensor,
        "sfb2_tensor": sfb2_tensor,
        "padded_offsets": padded_offsets,
        "alpha_tensor": alpha,
        "prob_tensor": prob,
        "bias_tensor": bias,
        "valid_m": mt,
        "n": n,
        "k": k,
        "l": l,
        "block2_shape": block2_shape,
        "ref_d": ref_d,
    }


def build_discrete_pointers(problem) -> dict:
    """Per-expert int64 pointer arrays into the dense backings (uniform
    per-expert layout, back-to-back blocks)."""
    b_packed = problem["b_packed"]  # (l, n, k/2) contiguous
    sfb_tensor = problem["sfb_tensor"]  # MMA-tiled, l outermost in memory
    sfb2_tensor = problem["sfb2_tensor"]  # (rows, cols, l) stride (1, rows, rows*cols)
    l = problem["l"]

    b_ptrs = torch.tensor([b_packed[e].data_ptr() for e in range(l)], dtype=torch.int64, device="cuda")
    sfb_ptrs = torch.tensor([sfb_tensor[..., e].data_ptr() for e in range(l)], dtype=torch.int64, device="cuda")
    sfb2_ptrs = torch.tensor([sfb2_tensor[..., e].data_ptr() for e in range(l)], dtype=torch.int64, device="cuda")
    return {"b_ptrs": b_ptrs, "sfb_ptrs": sfb_ptrs, "sfb2_ptrs": sfb2_ptrs}
