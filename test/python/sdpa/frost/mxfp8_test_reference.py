# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""MXFP8 input quantization and reference math shared by SDPA tests."""

from typing import NamedTuple
import torch

_BLOCK = 32


class _ReferenceWithStats(NamedTuple):
    output: torch.Tensor
    stats: torch.Tensor


def _cdiv(a, b):
    return (a + b - 1) // b


def _quantize(t, b, h, s, d, fp8, *, columnwise):
    """MXFP8-quantize [b,h,s,d] fp32 → (fp8_data[b,h,s,d], swizzled SF, per-elem dequant
    scale[b,h,s,d], (scale_padded, dblock_padded)). columnwise=True scales along S (V),
    else along D (Q/K). Torch-only (sdpa.mxfp8_quant), TE-equivalent semantics."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d_scale_pad = _cdiv(_cdiv(d, _BLOCK), 4) * 4
    d_pad = d_scale_pad * _BLOCK
    s_scale_pad = _cdiv(_cdiv(s, _BLOCK), 4) * 4
    s_pad = s_scale_pad * _BLOCK
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t, b, h, s, d, _BLOCK, fp8, with_ref=True)
    if columnwise:
        dq = dq_s.reshape(b, h, s, d)
        return data_s, swz_s, dq, (s_scale_pad, d_pad)
    dq = dq_d.reshape(b, h, s, d)
    return data_d, swz_d, dq, (s_pad, d_scale_pad)


def _ref(
    qd,
    kd,
    vd,
    *,
    scale,
    is_causal=False,
    bottom_right=False,
    swa_window=None,
    right_bound=0,
    sinks=None,
    seq_lens_q=None,
    seq_lens_kv=None,
    return_stats=False,
):
    """fp32 reference matching the kernel's mask + sink semantics (BHSD; GQA-aware)."""
    b, h_q, s_q, _ = qd.shape
    _, h_kv, s_kv, _ = vd.shape
    dev = qd.device
    g = h_q // h_kv
    k_e = kd.repeat_interleave(g, dim=1)
    v_e = vd.repeat_interleave(g, dim=1)
    scores = torch.matmul(qd, k_e.transpose(-1, -2)) * scale
    i = torch.arange(s_q, device=dev).view(1, 1, s_q, 1)
    j = torch.arange(s_kv, device=dev).view(1, 1, 1, s_kv)
    masked = torch.zeros(1, 1, s_q, s_kv, dtype=torch.bool, device=dev)
    # Bottom-right anchors the diagonal at each batch's corner (seq_len_q[b], seq_len_kv[b]),
    # not at the padded (S_q, S_kv): a shorter Q keeps its rows aligned with the END of its KV.
    diag = 0
    if bottom_right:
        slq_d = torch.as_tensor(seq_lens_q if seq_lens_q is not None else [s_q] * b, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        slk_d = torch.as_tensor(seq_lens_kv if seq_lens_kv is not None else [s_kv] * b, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        diag = slk_d - slq_d
    if is_causal:
        lim = i + diag
        masked = masked | (j > lim + right_bound)
    if swa_window is not None:
        swa_base = i + diag
        masked = masked | (j < swa_base - swa_window)
    if seq_lens_kv is not None:
        # Per-batch KV padding: columns j >= seq_len_kv[b] are padding -> masked.
        slk = torch.as_tensor(seq_lens_kv, device=dev, dtype=torch.long).view(b, 1, 1, 1)
        masked = masked | (j >= slk)
    scores = scores.masked_fill(masked, float("-inf"))

    def finalize(o, lse=None):
        if seq_lens_q is not None:
            slq = torch.as_tensor(seq_lens_q, device=dev, dtype=torch.long).view(b, 1, 1, 1)
            valid_q = i < slq
            o = torch.where(valid_q, o, torch.zeros_like(o))
            if lse is not None:
                lse = torch.where(valid_q.squeeze(-1), lse, torch.full_like(lse, float("-inf")))
        return _ReferenceWithStats(o, lse) if lse is not None else o

    if sinks is not None:
        col = sinks.view(1, h_q, 1, 1).float().expand(b, h_q, s_q, 1).to(dev)
        ext = torch.cat([scores, col], dim=-1)
        probs = torch.softmax(ext, dim=-1)
        o = torch.matmul(probs[..., :s_kv], v_e)
        return finalize(o, torch.logsumexp(ext, dim=-1) if return_stats else None)
    row_has_kv = torch.isfinite(scores).any(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(row_has_kv, probs, torch.zeros_like(probs))
    o = torch.matmul(probs, v_e)
    return finalize(o, torch.logsumexp(scores, dim=-1) if return_stats else None)


def _quantize_seq(t_1hsd, h, s, d, fp8, *, columnwise):
    """Per-sequence MXFP8 quantization for the THD packing.

    Returns (fp8_data [1,h,s,d], per-elem dequant scale [1,h,s,d], SF tiles
    [h, n_tiles, SF_SMEM] uint8) where n_tiles = ceil(s/128): the quantizer's
    F8_128x4 atom padding rounds the S extent up to a multiple of 128, i.e.
    exactly the kernel's per-sequence-TILE-padded SF layout."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    if s == 0:
        # Zero-length sequence: no tokens, no SF tiles.
        empty = t_1hsd.new_zeros((1, h, 0, d))
        d_scale = _cdiv(_cdiv(d, _BLOCK), 4) * 4
        return empty.to(fp8), empty.float(), torch.zeros((h, 0, 128 * d_scale), dtype=torch.uint8, device=t_1hsd.device)
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t_1hsd, 1, h, s, d, _BLOCK, fp8, with_ref=True)
    n_tiles = _cdiv(s, 128)
    if columnwise:
        # Columnwise F8_128x4 is plane-major for D > 128. THD's packed
        # contract keeps every sequence tile contiguous, so transpose
        # [ceil(D/128), H, tile, 512] into [H, tile, ceil(D/128), 512] before
        # sequences are concatenated in cu_seqlens order.
        planes = _cdiv(d, 128)
        sf = swz_s.view(torch.uint8).reshape(planes, h, n_tiles, 512).permute(1, 2, 0, 3).contiguous().reshape(h, n_tiles, -1)
        return data_s, dq_s.reshape(1, h, s, d), sf
    return data_d, dq_d.reshape(1, h, s, d), swz_d.view(torch.uint8).reshape(h, n_tiles, -1)
