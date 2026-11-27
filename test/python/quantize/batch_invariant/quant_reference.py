# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent torch implementations of the block-scale quantization recipes, and shared test inputs."""

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
F32_MAX = torch.finfo(torch.float32).max


def make_input(device, dtype, m, k, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    scale = torch.exp2(torch.randint(-6, 7, (m, k // 16, 1), device=device, generator=g).float())
    x = (torch.randn(m, k // 16, 16, device=device, generator=g) * scale).reshape(m, k)
    return x.to(dtype)


def _block_pattern(kind, dtype, g):
    """One 32-element block of a boundary-value family, in float32 (the caller converts to ``dtype``)."""
    n = 32
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    info = torch.finfo(dtype)
    if kind == "zeros":
        return torch.zeros(n)
    if kind == "signed_zeros":
        return torch.where(torch.arange(n) % 2 == 0, 0.0, -0.0)
    if kind == "subnormal":  # every element below the smallest normal number of the input dtype
        return sign * info.smallest_normal * torch.exp2(-torch.randint(1, 4, (n,), generator=g).float())
    if kind == "huge":
        return sign * info.max * torch.exp2(-torch.randint(0, 3, (n,), generator=g).float())
    if kind == "fp4_midpoints":  # amax 6 fixes a unit scale; the rest sit exactly between two E2M1 codes
        mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        v = torch.cat([torch.tensor([6.0]), mids, -mids]).repeat(3)[:n]
        return v
    if kind == "fp8_midpoints":  # amax 448 fixes a unit scale; the rest sit between two E4M3 codes
        mids = torch.tensor([1.0625, 1.1875, 2.125, 17.0, 0.0029296875, 0.001953125 + 0.0009765625 / 2, 240.0])
        return torch.cat([torch.tensor([448.0]), mids, -mids]).repeat(3)[:n]
    if kind == "wide_range":
        return sign * torch.exp2(torch.linspace(-20, 20, n))
    if kind == "outlier":
        v = torch.randn(n, generator=g)
        v[int(torch.randint(0, n, (1,), generator=g))] *= 4096.0
        return v
    raise ValueError(kind)


FINITE_KINDS = ("zeros", "signed_zeros", "subnormal", "huge", "fp4_midpoints", "fp8_midpoints", "wide_range", "outlier")


def adversarial_input(device, dtype, m, k, seed, nonfinite=False):
    """Boundary values per 32-element block, cycling through the families row by row and block by block.

    With ``nonfinite`` the families also include blocks holding +-inf and NaN, which have no independent reference."""
    kinds = FINITE_KINDS + (("inf", "nan", "mixed_nonfinite") if nonfinite else ())
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(m, k // 32, 32)
    for r in range(m):
        for b in range(k // 32):
            kind = kinds[(r + 3 * b) % len(kinds)]
            if kind == "inf":
                blk = torch.randn(32, generator=g)
                blk[::7] = float("inf")
                blk[3::11] = float("-inf")
            elif kind == "nan":
                blk = torch.randn(32, generator=g)
                blk[int(torch.randint(0, 32, (1,), generator=g))] = float("nan")
            elif kind == "mixed_nonfinite":
                blk = torch.randn(32, generator=g)
                blk[0], blk[1], blk[2] = float("nan"), float("inf"), float("-inf")
            else:
                blk = _block_pattern(kind, dtype, g)
            out[r, b] = blk
    return out.reshape(m, k).to(device=device, dtype=dtype)


def e2m1_pack(scaled):
    """Round-to-nearest-even onto the E2M1 grid (saturating at +-6, sign kept), two codes per byte, even element low."""
    m = scaled.shape[0]
    scaled = scaled.reshape(m, -1).clamp(-6.0, 6.0)
    grid = torch.tensor(E2M1, device=scaled.device)
    dist = (scaled.abs().unsqueeze(-1) - grid).abs()
    cand = dist == dist.min(-1, keepdim=True).values
    # A tie between two codes goes to the even code.
    idx = torch.arange(8, device=scaled.device)
    low = torch.where(cand, idx, torch.full_like(idx, 99)).amin(-1)
    high = torch.where(cand, idx, torch.full_like(idx, -1)).amax(-1)
    code = torch.where(low % 2 == 0, low, high)
    sign = ((scaled < 0) | ((scaled == 0) & torch.signbit(scaled))).to(torch.int64) * 8
    code = (code | sign).to(torch.uint8)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()


def nvfp4_reference(x, global_scale, per_token=False):
    """TE's NVFP4 recipe with every fp32 operation correctly rounded.

    Returns the packed codes, the logical e4m3 scale bytes ``[M, K / 16]`` and, per token, the fp32 per-row scale."""
    m, k = x.shape
    f = lambda v: torch.tensor(v, dtype=torch.float32, device=x.device)
    one, sixth = f(1.0), f(1.0 / 6.0)
    blocks = x.float().reshape(m, k // 16, 16)
    amax = blocks.abs().amax(-1)
    gs = f(global_scale)
    if per_token:
        gs0 = one / gs  # the argument is the inverse base scale
        res = torch.clamp(gs0 / blocks.abs().amax((-2, -1)), max=F32_MAX)
        ges = torch.where(res != 0, res, one)
        pts = one / ges
        gs = ges.unsqueeze(-1)
    sf_f = amax * (gs * sixth)
    sf = sf_f.clamp(max=448.0).to(torch.float8_e4m3fn)  # cvt.rn.satfinite
    sf_d = sf.float()
    inv = torch.clamp(one / (sf_d * (one / gs)), max=F32_MAX)
    codes = e2m1_pack(blocks * inv.unsqueeze(-1))
    scales = sf.view(torch.uint8)
    return (codes, scales, pts) if per_token else (codes, scales)
