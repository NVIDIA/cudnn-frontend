# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the extended SDPA torch custom ops (cudnn.sdpa.fwd.torch_op).

``torch.ops.cudnn.sdpa_fwd`` / ``sdpa_bwd`` expose the cuDNN feature
surface that ``torch.nn.functional.scaled_dot_product_attention``'s aten
contract cannot: attention sinks, sliding windows, bottom-right causal
diagonals, padded batches, and THD/varlen packing (FlashAttention-style
``(T, H, D)`` + ``cu_seqlens``). Each case checks numerics against a pure
fp32 PyTorch reference. The engine Router picks the serving plan (FROST OSS
kernels or cuDNN-backend engines) per configuration — these tests pass on
either route.
"""

import math

import pytest
import torch

import cudnn

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] < 8:
    pytest.skip("cuDNN SDPA requires sm80+", allow_module_level=True)
if cudnn.backend_version() < 90600:
    pytest.skip("requires cuDNN >= 9.6 (THD token-major stats)", allow_module_level=True)

from cudnn.sdpa.fwd import torch_op  # noqa: E402  (registers torch.ops.cudnn.sdpa_fwd / sdpa_bwd)

# SDPA rejects sink_token below 9.13 (scaled_dot_product_flash_attention.h:
# "SDPA with sink_token is not supported before 9.13."), while the module
# gate above only needs 9.6 for token-major THD stats.
_SINKS_UNSUPPORTED = pytest.mark.skipif(cudnn.backend_version() < 91300, reason="sink_token requires cuDNN >= 9.13")

TOL = 2.5e-2  # bf16 rounding at these magnitudes


def ref_attention(q, k, v, scale, is_causal=False, bottom_right=False, window_left=-1, sinks=None, return_lse=False):
    """fp32 reference in BHSD. window_left counts VISIBLE tokens including self
    (the cuDNN diagonal_band_left_bound convention). sinks: (H,) extra softmax
    logit per query head, contributing no value (but part of the softmax
    denominator, so part of the LSE too)."""
    q, k, v = q.float(), k.float(), v.float()
    B, Hq, Sq, _ = q.shape
    Hkv, Skv = k.shape[1], k.shape[2]
    if Hq != Hkv:
        k = k.repeat_interleave(Hq // Hkv, dim=1)
        v = v.repeat_interleave(Hq // Hkv, dim=1)
    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale

    i = torch.arange(Sq, device=q.device).view(-1, 1)
    j = torch.arange(Skv, device=q.device).view(1, -1)
    off = (Skv - Sq) if bottom_right else 0
    mask = torch.zeros(Sq, Skv, dtype=torch.bool, device=q.device)
    if is_causal:
        mask |= j > (i + off)
    if window_left >= 0:
        mask |= j <= (i + off - window_left)
    s = s.masked_fill(mask, float("-inf"))

    if sinks is not None:
        sink_col = sinks.float().view(1, Hq, 1, 1).expand(B, Hq, Sq, 1)
        s = torch.cat([s, sink_col], dim=-1)
        p = torch.softmax(s, dim=-1)[..., :-1]
    else:
        p = torch.softmax(s, dim=-1)
    o = torch.einsum("bhqk,bhkd->bhqd", p, v)
    if return_lse:
        return o, torch.logsumexp(s, dim=-1, keepdim=True)  # (B, Hq, Sq, 1)
    return o


def bshd(B, H, S, D, dtype=torch.bfloat16, requires_grad=False):
    t = torch.randn(B, S, H, D, dtype=dtype, device="cuda").transpose(1, 2)
    return t.requires_grad_(True) if requires_grad else t


class TestSdpaFwdDense:
    @pytest.mark.L0
    @_SINKS_UNSUPPORTED
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_sinks(self, is_causal):
        torch.manual_seed(0)
        B, H, S, D = 2, 8, 512, 128
        q, k, v = bshd(B, H, S, D), bshd(B, H, S, D), bshd(B, H, S, D)
        sinks = torch.randn(H, device="cuda", dtype=torch.float32)
        scale = D**-0.5
        o, lse = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=is_causal, sinks=sinks, return_lse=True)
        ref, ref_lse = ref_attention(q, k, v, scale, is_causal=is_causal, sinks=sinks, return_lse=True)
        assert (o.float() - ref).abs().max().item() < TOL
        # LSE values (not just metadata): the sink logit is part of the denominator.
        assert lse.shape == (B, H, S, 1) and lse.dtype == torch.float32
        assert (lse - ref_lse).abs().max().item() < TOL

    @pytest.mark.L0
    def test_padded_seq_lens(self):
        """Dense padded batches: per-batch actual lengths via seq_len_q/kv.
        Only rows/cols inside each batch's actual lengths are compared (rows
        past seq_len_q are dead by contract)."""
        torch.manual_seed(0)
        B, H, S, D = 2, 8, 256, 128
        q, k, v = bshd(B, H, S, D), bshd(B, H, S, D), bshd(B, H, S, D)
        len_q = torch.tensor([200, 96], device="cuda", dtype=torch.int32)
        len_kv = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
        scale = D**-0.5
        o, _ = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, seq_len_q=len_q, seq_len_kv=len_kv, return_lse=False)
        for b in range(B):
            lq, lkv = int(len_q[b]), int(len_kv[b])
            ref = ref_attention(q[b : b + 1, :, :lq], k[b : b + 1, :, :lkv], v[b : b + 1, :, :lkv], scale)
            assert (o[b, :, :lq].float() - ref[0]).abs().max().item() < TOL, f"batch {b}"

    @pytest.mark.L0
    @pytest.mark.parametrize("window_left", [64, 128])
    def test_sliding_window_causal(self, window_left):
        torch.manual_seed(0)
        B, H, S, D = 2, 8, 512, 128
        q, k, v = bshd(B, H, S, D), bshd(B, H, S, D), bshd(B, H, S, D)
        scale = D**-0.5
        o, _ = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=True, window_left=window_left, return_lse=False)
        ref = ref_attention(q, k, v, scale, is_causal=True, window_left=window_left)
        assert (o.float() - ref).abs().max().item() < TOL

    @pytest.mark.L0
    def test_bottom_right_causal_cross_seqlen(self):
        torch.manual_seed(0)
        B, H, Sq, Skv, D = 2, 8, 128, 512, 128
        q, k, v = bshd(B, H, Sq, D), bshd(B, H, Skv, D), bshd(B, H, Skv, D)
        scale = D**-0.5
        o, _ = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=True, causal_bottom_right=True, return_lse=False)
        ref = ref_attention(q, k, v, scale, is_causal=True, bottom_right=True)
        assert (o.float() - ref).abs().max().item() < TOL

    @pytest.mark.L0
    def test_gqa_hk_ne_hv(self):
        """cuDNN supports h_k != h_v (each dividing h_q) — K and V carry
        independent head counts."""
        torch.manual_seed(0)
        B, S, D = 2, 256, 128
        q, k, v = bshd(B, 32, S, D), bshd(B, 8, S, D), bshd(B, 4, S, D)
        scale = D**-0.5
        o, _ = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=True, return_lse=False)
        kx = k.repeat_interleave(4, dim=1)  # expand both to H_q for the reference
        vx = v.repeat_interleave(8, dim=1)
        ref = ref_attention(q, kx, vx, scale, is_causal=True)
        assert (o.float() - ref).abs().max().item() < TOL

    @pytest.mark.L0
    @pytest.mark.parametrize("permute", [(0, 1, 2, 3), (0, 2, 1, 3), (1, 2, 0, 3), (2, 1, 0, 3)])
    def test_output_adopts_query_layout(self, permute):
        """O is allocated in Q's dim-permutation (any B/H/S order, D innermost)
        — the aten contract test_cudnn_attention_preserves_query_layout relies
        on this."""
        torch.manual_seed(0)
        BHSD = (2, 8, 256, 64)
        shape = tuple(BHSD[i] for i in permute)
        reverse = [permute.index(i) for i in range(4)]
        q = torch.randn(*shape, dtype=torch.bfloat16, device="cuda").permute(reverse)
        k = torch.randn(*shape, dtype=torch.bfloat16, device="cuda").permute(reverse)
        v = torch.randn(*shape, dtype=torch.bfloat16, device="cuda").permute(reverse)
        scale = BHSD[3] ** -0.5
        o, _ = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=True, return_lse=False)
        assert o.permute(permute).is_contiguous(), f"O stride {o.stride()} does not follow Q layout {q.stride()}"
        ref = ref_attention(q, k, v, scale, is_causal=True)
        assert (o.float() - ref).abs().max().item() < TOL

    @pytest.mark.L0
    @_SINKS_UNSUPPORTED
    def test_sinks_with_window(self):
        torch.manual_seed(0)
        B, H, S, D = 2, 8, 512, 128
        q, k, v = bshd(B, H, S, D), bshd(B, H, S, D), bshd(B, H, S, D)
        sinks = torch.randn(H, device="cuda", dtype=torch.float32)
        scale = D**-0.5
        o, _ = torch.ops.cudnn.sdpa_fwd(q, k, v, scale, is_causal=True, window_left=256, sinks=sinks, return_lse=False)
        ref = ref_attention(q, k, v, scale, is_causal=True, window_left=256, sinks=sinks)
        assert (o.float() - ref).abs().max().item() < TOL


class TestOpContract:
    @pytest.mark.L0
    def test_opcheck(self):
        """torch.library.opcheck: fake-vs-real metadata agreement, schema
        round-trip, and autograd registration — including dynamic-shape AOT
        dispatch (the torch.compile contract)."""
        torch.manual_seed(0)
        q = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        torch.library.opcheck(torch.ops.cudnn.sdpa_fwd, (q, q.clone(), q.clone(), 0.125), dict(is_causal=True, return_lse=True))

        lens = torch.tensor([100, 156], device="cuda")
        cu = torch.nn.functional.pad(lens.cumsum(0), (1, 0)).to(torch.int32)
        T, mx = int(cu[-1]), int(lens.max())
        q, k, v = (torch.randn(T, 8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True) for _ in range(3))
        torch.library.opcheck(
            torch.ops.cudnn.sdpa_fwd,
            (q, k, v, 0.125),
            dict(is_causal=True, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=mx, max_seqlen_kv=mx, return_lse=True),
        )


class TestSdpaVarlen:
    """THD (packed varlen) forward + backward through the ops directly."""

    def _make(self, lens, Hq, Hkv, D, dtype=torch.bfloat16):
        lens_t = torch.tensor(lens, device="cuda")
        cu = torch.nn.functional.pad(lens_t.cumsum(0), (1, 0)).to(torch.int32)
        T, mx = int(cu[-1]), int(lens_t.max())
        q = torch.randn(T, Hq, D, dtype=dtype, device="cuda")
        k = torch.randn(T, Hkv, D, dtype=dtype, device="cuda")
        v = torch.randn(T, Hkv, D, dtype=dtype, device="cuda")
        return q, k, v, cu, T, mx

    def _ref(self, q, k, v, cu, is_causal, grad=None):
        """Per-sequence fp32 reference; returns (out, lse, dq, dk, dv) — the
        packed (T, Hq) log-sum-exp always, grads None unless grad given."""
        qr, kr, vr = (t.detach().float().requires_grad_(grad is not None) for t in (q, k, v))
        Hq, Hkv = q.shape[1], k.shape[1]
        outs, lses = [], []
        for i in range(cu.numel() - 1):
            a, b = int(cu[i]), int(cu[i + 1])
            qi, ki, vi = (t[a:b].transpose(0, 1).unsqueeze(0) for t in (qr, kr, vr))
            if Hq != Hkv:
                ki = ki.repeat_interleave(Hq // Hkv, dim=1)
                vi = vi.repeat_interleave(Hq // Hkv, dim=1)
            s = torch.einsum("bhqd,bhkd->bhqk", qi, ki) * q.shape[-1] ** -0.5
            if is_causal:
                Sq = qi.shape[2]
                m = torch.ones(Sq, Sq, dtype=torch.bool, device=q.device).triu(1)
                s = s.masked_fill(m, float("-inf"))
            outs.append(torch.einsum("bhqk,bhkd->bhqd", torch.softmax(s, dim=-1), vi)[0].transpose(0, 1))
            lses.append(torch.logsumexp(s.detach(), dim=-1)[0].transpose(0, 1))  # (b-a, Hq)
        out = torch.cat(outs)
        lse = torch.cat(lses)
        if grad is None:
            return out, lse, None, None, None
        out.backward(grad.float())
        return out, lse, qr.grad, kr.grad, vr.grad

    @pytest.mark.L0
    @pytest.mark.parametrize("lens", [[333, 128, 512, 47], [256, 384]])
    def test_thd_forward(self, lens):
        torch.manual_seed(0)
        H, D = 8, 128
        q, k, v, cu, T, mx = self._make(lens, H, H, D)
        o, lse = torch.ops.cudnn.sdpa_fwd(
            q, k, v, D**-0.5, is_causal=True,
            cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=mx, max_seqlen_kv=mx, return_lse=True,
        )  # fmt: skip
        ref, ref_lse, _, _, _ = self._ref(q, k, v, cu, is_causal=True)
        assert (o.float() - ref).abs().max().item() < TOL
        assert lse.shape == (T, H, 1) and lse.dtype == torch.float32
        assert (lse[:, :, 0] - ref_lse).abs().max().item() < TOL

    @pytest.mark.L0
    @pytest.mark.parametrize("gqa", [False, True])
    def test_thd_forward_backward(self, gqa):
        torch.manual_seed(0)
        lens, D = [200, 312, 96], 128
        Hq, Hkv = (16, 4) if gqa else (8, 8)
        q, k, v, cu, _, mx = self._make(lens, Hq, Hkv, D)
        scale = D**-0.5
        o, lse = torch.ops.cudnn.sdpa_fwd(
            q, k, v, scale, is_causal=True,
            cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=mx, max_seqlen_kv=mx, return_lse=True,
        )  # fmt: skip
        grad = torch.randn_like(o)

        # bwd takes PADDED (B, H, maxS, 1) LSE (backend restriction: bprop THD
        # rejects ragged LSE on SM8X/SM12X) — scatter the packed TH1 stats.
        B = cu.numel() - 1
        lse_padded = torch.zeros(B, Hq, mx, 1, dtype=torch.float32, device="cuda")
        for i in range(B):
            a, b = int(cu[i]), int(cu[i + 1])
            lse_padded[i, :, : b - a, 0] = lse[a:b, :, 0].transpose(0, 1)

        dq, dk, dv = torch.ops.cudnn.sdpa_bwd(
            grad, q, k, v, o, lse_padded, scale, is_causal=True,
            cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=mx, max_seqlen_kv=mx,
        )  # fmt: skip

        ref, _, rdq, rdk, rdv = self._ref(q, k, v, cu, is_causal=True, grad=grad)
        group = Hq // Hkv
        assert (o.float() - ref).abs().max().item() < TOL
        assert (dq.float() - rdq).abs().max().item() < TOL
        # dk/dv accumulate GQA groups in bf16 — error grows ~sqrt(group)
        assert (dk.float() - rdk).abs().max().item() < TOL * group**0.5
        assert (dv.float() - rdv).abs().max().item() < TOL * group**0.5

    @pytest.mark.L0
    def test_thd_autograd(self):
        """sdpa_fwd is differentiable end to end on the varlen path: the
        registered autograd glue converts the packed TH1 stats to the padded
        LSE layout and routes grads through cudnn::sdpa_bwd."""
        torch.manual_seed(0)
        lens, H, D = [200, 312, 96], 8, 128
        q, k, v, cu, _, mx = self._make(lens, H, H, D)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, _ = torch.ops.cudnn.sdpa_fwd(
            q, k, v, D**-0.5, is_causal=True,
            cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=mx, max_seqlen_kv=mx, return_lse=True,
        )  # fmt: skip
        assert o.grad_fn is not None, "sdpa_fwd output is detached from autograd"
        grad = torch.randn_like(o)
        o.backward(grad)

        ref, _, rdq, rdk, rdv = self._ref(q, k, v, cu, is_causal=True, grad=grad)
        assert (o.float() - ref).abs().max().item() < TOL
        assert (q.grad.float() - rdq).abs().max().item() < TOL
        assert (k.grad.float() - rdk).abs().max().item() < TOL
        assert (v.grad.float() - rdv).abs().max().item() < TOL

    @pytest.mark.L0
    def test_thd_kv_packed_views(self):
        """K/V as views of a kv-interleaved [T, 2, H, D] buffer (token stride
        2*H*D) — the layout torch.nn.attention.varlen users produce. Must be
        served correctly (by whichever engine the router picks) or declined."""
        torch.manual_seed(0)
        H, D = 8, 128
        q, _, _, cu, T, mx = self._make([333, 128, 512, 47], H, H, D)
        kv = torch.randn(T, 2, H, D, dtype=torch.bfloat16, device="cuda")
        k, v = kv[:, 0], kv[:, 1]
        o, _ = torch.ops.cudnn.sdpa_fwd(
            q, k, v, D**-0.5, is_causal=True,
            cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=mx, max_seqlen_kv=mx, return_lse=False,
        )  # fmt: skip
        ref, _, _, _, _ = self._ref(q, k, v, cu, is_causal=True)
        assert (o.float() - ref).abs().max().item() < TOL
