# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent FP32 QSA oracle; selected blocks are expanded in bounded chunks.

Mask contract: Miles 0cc6516, ops/attention.py:_publish_block_form.
The synthetic selector draws uniform scores; it is not a captured model trace.
"""

import torch


def make_inputs(axes, device, seed):
    total, batch = int(axes["T"]), int(axes["B"])
    hq, hkv, dim = int(axes["Hq"]), int(axes["Hkv"]), int(axes["D"])
    generator = torch.Generator(device=device).manual_seed(int(seed or 0))
    lengths = [total // batch + (i < total % batch) for i in range(batch)]
    first, last, token_base, block_base = [], [], [], []
    offset, base = 0, 0
    for length in lengths:
        spans = list(range(offset, offset + length, 4))
        first.extend(spans)
        last.extend(min(x + 3, offset + length - 1) for x in spans)
        token_base.extend([offset] * length)
        block_base.extend([base] * length)
        offset += length
        base += len(spans)
    first, last, lo, blk_base = [torch.tensor(x, dtype=torch.int32, device=device) for x in (first, last, token_base, block_base)]
    hi = torch.arange(total, device=device, dtype=torch.int32)
    sel = torch.zeros(total, base, dtype=torch.uint8, device=device)
    block_ids = torch.arange(base, device=device, dtype=torch.int64)
    for start in range(0, total, 256):
        stop = min(start + 256, total)
        scores = torch.rand(stop - start, base, generator=generator, device=device)
        # Positive float32 bit patterns preserve the full score order.
        ranks = scores.view(torch.int32).to(torch.int64) * base + block_ids
        complete = (first[None, :] >= lo[start:stop, None]) & (first[None, :] + 3 <= hi[start:stop, None]) & (last[None, :] - first[None, :] == 3)
        ranks.masked_fill_(~complete, -1)
        values, ids = ranks.topk(min(512, base), dim=1)
        sel[start:stop].scatter_(1, ids, (values >= 0).byte())
    partial = (hi - lo + 1) % 4 != 0
    rows = hi[partial].long()
    sel[rows, (blk_base[partial] + (hi[partial] - lo[partial]) // 4).long()] = 1
    if int(axes.get("Empty", 0)):
        sel[::17] = 0

    def normal(heads):
        return torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16, generator=generator)

    return dict(
        q=normal(hq),
        k=normal(hkv),
        v=normal(hkv),
        sel_blocks=sel,
        lo=lo,
        hi=hi,
        blk_base=blk_base,
        tok_base=lo.clone(),
        block_first=first,
        block_last=last,
        scale=dim**-0.5,
        block_size=4,
    )


def _gather(q, k, v, sel, lo, hi, first, last, begin, end, block_size):
    count = max(1, int(sel[begin:end].count_nonzero(dim=1).max().item()))
    ranks = (sel[begin:end] != 0).to(torch.int64) * sel.shape[1] - torch.arange(sel.shape[1], device=q.device)
    values, blocks = ranks.topk(count, dim=1)
    flags = values > 0
    j = first[blocks.long(), None] + torch.arange(block_size, device=q.device)
    valid = (flags[:, :, None] != 0) & (j <= last[blocks.long(), None]) & (j >= lo[begin:end, None, None]) & (j <= hi[begin:end, None, None])
    j = j.reshape(end - begin, -1).long()
    valid = valid.reshape(end - begin, -1)
    safe = j.clamp(0, k.shape[0] - 1)
    kk = k[safe].float().permute(0, 2, 1, 3)
    vv = v[safe].float().permute(0, 2, 1, 3)
    qq = q[begin:end].float().reshape(end - begin, k.shape[1], -1, q.shape[-1])
    return qq, kk, vv, safe, valid


@torch.no_grad()
def forward_reference(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size):
    total, hq, dim = q.shape
    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(hq, total, device=q.device, dtype=torch.float32)
    for begin in range(0, total, 128):
        end = min(begin + 128, total)
        qq, kk, vv, _, valid = _gather(q, k, v, sel_blocks, lo, hi, block_first, block_last, begin, end, block_size)
        scores = (qq @ kk.transpose(-1, -2)) * scale
        scores.masked_fill_(~valid[:, None, None, :], -torch.inf)
        normalizer = torch.logsumexp(scores, dim=-1)
        live = valid.any(dim=1)[:, None, None]
        p = torch.exp(scores - torch.where(live, normalizer, 0)[..., None])
        p = torch.where(valid[:, None, None, :], p, 0)
        out[begin:end] = (p @ vv).reshape(end - begin, hq, dim)
        lse[:, begin:end] = normalizer.reshape(end - begin, hq).T
    return out.to(q.dtype), out, lse


@torch.no_grad()
def backward_reference(grad_out, q, k, v, out_fp32, lse, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size):
    total, hq, dim = q.shape
    hkv = k.shape[1]
    dq = torch.empty_like(q, dtype=torch.float32)
    dk, dv = torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32)
    for begin in range(0, total, 128):
        end = min(begin + 128, total)
        qq, kk, vv, indices, valid = _gather(q, k, v, sel_blocks, lo, hi, block_first, block_last, begin, end, block_size)
        do = grad_out[begin:end].float().reshape(end - begin, hkv, -1, dim)
        saved_lse = lse[:, begin:end].T.reshape(end - begin, hkv, -1)
        scores = (qq @ kk.transpose(-1, -2)) * scale
        p = torch.exp(scores - torch.where(saved_lse.isfinite(), saved_lse, 0)[..., None])
        p = torch.where(valid[:, None, None, :] & saved_lse.isfinite()[..., None], p, 0)
        delta = (grad_out[begin:end].float() * out_fp32[begin:end]).sum(-1)
        ds = p * ((do @ vv.transpose(-1, -2)) - delta.reshape(end - begin, hkv, -1, 1))
        dq[begin:end] = ((ds @ kk) * scale).reshape(end - begin, hq, dim)
        dki = ((ds.transpose(-1, -2) @ qq) * scale).permute(0, 2, 1, 3)
        dvi = (p.transpose(-1, -2) @ do).permute(0, 2, 1, 3)
        dk.index_add_(0, indices.flatten(), dki.reshape(-1, hkv, dim))
        dv.index_add_(0, indices.flatten(), dvi.reshape(-1, hkv, dim))
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def compare(ref_outputs, user_outputs, **kwargs):
    result = {}
    for name, expected in ref_outputs.items():
        actual = user_outputs[name]
        finite = expected.isfinite()
        special_ok = torch.equal(torch.isneginf(actual), torch.isneginf(expected))
        special_ok = special_ok and not bool(torch.isnan(actual).any())
        if bool(finite.any()):
            error = actual[finite].float() - expected[finite].float()
            norm = expected[finite].float().square().sum().sqrt().clamp_min(1e-12)
            relative = float(error.square().sum().sqrt() / norm)
            pointwise = torch.allclose(actual[finite].float(), expected[finite].float(), atol=0.04, rtol=0.04)
            passed = special_ok and pointwise and relative <= 0.025
        else:
            passed = special_ok
        result[name] = {
            "passed": bool(passed),
            "relative_l2": relative if bool(finite.any()) else 0.0,
            "max_absolute_error": float(error.abs().max()) if bool(finite.any()) else 0.0,
        }
    return result
