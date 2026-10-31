# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SM107 half-precision inputs and references, independent of test entries."""

import pytest


def _rubin_only():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the fused epilogue gate is served by the sm107 d256 kernels (cc10.7) only")


def _half_prefold_tensors(b, hq, hkv, s_q, s_kv, d_qk, d_v, dtype, prefold_scale, *, seed):
    """Half Q / K / V in BSHD storage (BHSD views) the way the engine consumes them.  ``prefold_scale`` multiplies the f32 Q
    BEFORE the half cast (the softmax_scale_prefolded contract; 1.0 on the unfolded leg)."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, device="cuda", generator=g) * 0.5

    q = (rnd(b, s_q, hq, d_qk) * prefold_scale).to(dtype).transpose(1, 2)
    k = rnd(b, s_kv, hkv, d_qk).to(dtype).transpose(1, 2)
    v = rnd(b, s_kv, hkv, d_v).to(dtype).transpose(1, 2)
    return q, k, v


def _dense_mask(s_q, s_kv, *, causal=False, bottom_right=False, window_left=None):
    """The (s_q, s_kv) bool mask of the dense band the adapter kwargs describe, or None for no mask."""
    import torch

    if not causal and window_left is None:
        return None
    i = torch.arange(s_q, device="cuda")[:, None] + ((s_kv - s_q) if bottom_right else 0)
    j = torch.arange(s_kv, device="cuda")[None, :]
    masked = torch.zeros(s_q, s_kv, dtype=torch.bool, device="cuda")
    if causal:
        masked |= j > i
    if window_left is not None:
        masked |= j < i - window_left
    return masked


def _half_oracle(q, k, v, *, logit_scale, masked=None, sinks=None):
    """float64 softmax(logit_scale * Q K^T) V on the half inputs the kernel saw (BHSD, GQA expanded) -> (O, LSE natural).
    ``masked`` broadcasts over (..., s_q, s_kv).  A keyless row gives O = 0 and LSE = -inf, or LSE = the head's sink logit
    when a sink joins the row as one extra column (the sink's mass is the row's whole denominator then)."""
    import torch

    rep = q.shape[1] // k.shape[1]
    logits = (q.double() @ k.double().repeat_interleave(rep, 1).transpose(-1, -2)) * logit_scale
    if masked is not None:
        logits = logits.masked_fill(masked, float("-inf"))
    full = logits if sinks is None else torch.cat([logits, sinks.double().view(1, -1, 1, 1).expand(*logits.shape[:3], 1)], dim=-1)
    probs = torch.softmax(full, dim=-1).nan_to_num(0.0)[..., : logits.shape[-1]]
    return probs @ v.double().repeat_interleave(rep, 1), torch.logsumexp(full, dim=-1)


def _half_prefold_api(
    d_qk,
    d_v,
    dtype,
    *,
    prefolded,
    with_stats,
    split_kv,
    b,
    hq,
    hkv,
    s_q,
    s_kv,
    causal=False,
    bottom_right=False,
    window_left=None,
    sink=False,
    seed=0,
    attn_scale=None,
    pack_gqa=False,
    tile_n=None,
):
    """The adapter of one leg (folded or unfolded) of a dense problem on the cc 10.7 half twin, plus its tensors.
    ``attn_scale`` defaults to d_qk ** -0.5; the leading-dead-tile cells pass 1.0 (scale_log2 > 1)."""
    import math

    import torch
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    attn_scale = d_qk**-0.5 if attn_scale is None else attn_scale
    q, k, v = _half_prefold_tensors(b, hq, hkv, s_q, s_kv, d_qk, d_v, dtype, attn_scale * math.log2(math.e) if prefolded else 1.0, seed=seed)
    out = torch.full((b, s_q, hq, d_v), float("nan"), device="cuda", dtype=dtype).transpose(1, 2)  # sentinel: an unclaimed tile stays visible
    lse = torch.full((b, hq, s_q), float("nan"), device="cuda", dtype=torch.float32) if with_stats else None
    sinks = torch.linspace(-2.0, 6.0, hq, device="cuda", dtype=torch.float32) if sink else None
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_lse=lse,
        scale_softmax=None if prefolded else attn_scale,
        is_causal=causal,
        causal_bottom_right=bottom_right,
        window_size_left=window_left,
        has_sink=sink,
        cga=2,
        split_kv=split_kv,
        pack_gqa=pack_gqa,
        softmax_scale_prefolded=prefolded,
        tile_n=tile_n,
    )
    return api, dict(q=q, k=k, v=v, out=out, lse=lse, sinks=sinks, attn_scale=attn_scale)
