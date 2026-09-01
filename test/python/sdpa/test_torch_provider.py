# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the "CUDNN" torch.nn.attention provider (cudnn.torch).

Two surfaces, both served by the cuDNN *Python* API after activation:

- vanilla ``F.scaled_dot_product_attention`` under
  ``sdpa_kernel([SDPBackend.CUDNN_ATTENTION])`` — fwd + autograd bwd, checked
  against the fp32 math backend with the stock flash backend's error on the
  same inputs as the rounding yardstick (same-precision kernels differ only
  in accumulation order, so cuDNN passes within 3x of flash's error);
- ``torch.nn.attention.varlen.varlen_attn`` — fwd + bwd against a
  per-sequence fp32 dense reference, including GQA, causal sliding windows
  (which the in-tree cuDNN varlen branch rejects), and non-contiguous
  kv-interleaved K/V views (the layout users produce by slicing a fused KV
  projection).

The engine Router picks the serving plan (FROST OSS kernels or cuDNN-backend
engines) per configuration — these tests pass on either route.
"""

import math

import pytest
import torch

import cudnn  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

try:
    from torch.nn.attention import SDPBackend, activate_flash_attention_impl, restore_flash_attention_impl, sdpa_kernel
    from torch.nn.attention.varlen import AuxRequest, varlen_attn
except ImportError:
    pytest.skip("torch.nn.attention flash-impl registry required (torch >= 2.13)", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

import cudnn.torch as provider  # noqa: E402  (registers the "CUDNN" provider)


@pytest.fixture(autouse=True)
def _activate_provider():
    activate_flash_attention_impl("CUDNN")
    yield
    restore_flash_attention_impl()


def math_ref(q, k, v, is_causal, scale, enable_gqa=False):
    """fp32 math-backend reference, differentiable."""
    q_, k_, v_ = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    with sdpa_kernel([SDPBackend.MATH]):
        o = F.scaled_dot_product_attention(q_, k_, v_, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    return o, q_, k_, v_


DENSE_CASES = [
    # B, Hq, Hkv, Sq, Skv, D, dtype, is_causal, scale, enable_gqa, bshd
    pytest.param(2, 8, 8, 512, 512, 128, torch.bfloat16, False, None, False, False, id="bf16-plain"),
    pytest.param(2, 8, 8, 512, 512, 128, torch.bfloat16, True, None, False, False, id="bf16-causal"),
    pytest.param(2, 8, 8, 512, 512, 128, torch.float16, True, None, False, False, id="fp16-causal"),
    pytest.param(2, 16, 4, 512, 512, 128, torch.bfloat16, True, None, True, False, id="gqa"),
    pytest.param(1, 8, 8, 1024, 2048, 64, torch.bfloat16, True, None, False, False, id="cross-seqlen-d64"),
    pytest.param(2, 8, 8, 512, 512, 128, torch.float16, True, 0.05, False, False, id="custom-scale"),
    pytest.param(2, 8, 8, 512, 512, 128, torch.bfloat16, True, None, False, True, id="bshd-projection"),
    pytest.param(2, 16, 4, 1024, 1024, 128, torch.bfloat16, True, None, True, True, id="bshd-gqa"),
]


@pytest.mark.L0
@pytest.mark.parametrize("B,Hq,Hkv,Sq,Skv,D,dtype,is_causal,scale,enable_gqa,bshd", DENSE_CASES)
def test_sdpa_dense_parity(B, Hq, Hkv, Sq, Skv, D, dtype, is_causal, scale, enable_gqa, bshd):
    torch.manual_seed(0)
    if bshd:  # realistic transformer layout: (B,S,H,D) projections viewed as BHSD
        q = torch.randn(B, Sq, Hq, D, dtype=dtype, device="cuda").transpose(1, 2).requires_grad_(True)
        k = torch.randn(B, Skv, Hkv, D, dtype=dtype, device="cuda").transpose(1, 2).requires_grad_(True)
        v = torch.randn(B, Skv, Hkv, D, dtype=dtype, device="cuda").transpose(1, 2).requires_grad_(True)
    else:
        q = torch.randn(B, Hq, Sq, D, dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn(B, Hkv, Skv, D, dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn(B, Hkv, Skv, D, dtype=dtype, device="cuda", requires_grad=True)

    # Dense: fwd runs on the python API; bwd routes to the C++ worker op
    # (bit-exact hybrid) until dense backward lands in cudnn::sdpa_bwd.
    fwd_before, bwd_before = provider.calls["fwd"], provider.calls["bwd_cpp"]
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    grad_o = torch.randn_like(o)
    o.backward(grad_o)
    assert provider.calls["fwd"] == fwd_before + 1, "provider fwd did not intercept"
    assert provider.calls["bwd_cpp"] == bwd_before + 1, "provider bwd did not intercept"
    assert o.grad_fn.__class__.__name__.startswith("ScaledDotProductCudnnAttention"), o.grad_fn

    o_ref, q_ref, k_ref, v_ref = math_ref(q, k, v, is_causal, scale, enable_gqa)
    o_ref.backward(grad_o.float())

    # Rounding yardstick: the stock flash backend's error on identical inputs.
    qf, kf, vf = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        o_fa = F.scaled_dot_product_attention(qf, kf, vf, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    o_fa.backward(grad_o)

    def err(a, b):
        return (a.float() - b).abs().max().item()

    floor = {torch.bfloat16: 1e-2, torch.float16: 2e-3}[dtype]
    for name, ours, flash in (
        ("o", err(o, o_ref), err(o_fa, o_ref)),
        ("dq", err(q.grad, q_ref.grad), err(qf.grad, q_ref.grad)),
        ("dk", err(k.grad, k_ref.grad), err(kf.grad, k_ref.grad)),
        ("dv", err(v.grad, v_ref.grad), err(vf.grad, v_ref.grad)),
    ):
        assert ours <= max(3 * flash, floor), f"{name}: cudnn err {ours:.4f} vs flash {flash:.4f}"


def ref_varlen(q, k, v, cu_q, cu_kv, is_causal, window_left=-1):
    """Per-sequence fp32 dense reference; returns (out, q_ref, k_ref, v_ref)."""
    qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    Hq, Hkv = q.shape[1], k.shape[1]
    outs = []
    for i in range(cu_q.numel() - 1):
        aq, bq = int(cu_q[i]), int(cu_q[i + 1])
        ak, bk = int(cu_kv[i]), int(cu_kv[i + 1])
        qi = qr[aq:bq].transpose(0, 1).unsqueeze(0)
        ki = kr[ak:bk].transpose(0, 1).unsqueeze(0)
        vi = vr[ak:bk].transpose(0, 1).unsqueeze(0)
        if Hq != Hkv:
            ki = ki.repeat_interleave(Hq // Hkv, dim=1)
            vi = vi.repeat_interleave(Hq // Hkv, dim=1)
        s = torch.einsum("bhqd,bhkd->bhqk", qi, ki) * q.shape[-1] ** -0.5
        Sq, Skv = qi.shape[2], ki.shape[2]
        ii = torch.arange(Sq, device=q.device).view(-1, 1)
        jj = torch.arange(Skv, device=q.device).view(1, -1)
        mask = torch.zeros(Sq, Skv, dtype=torch.bool, device=q.device)
        if is_causal:
            mask |= jj > ii
        if window_left >= 0:
            mask |= jj < (ii - window_left)  # FA2: window (w, 0) attends [i-w, i]
        s = s.masked_fill(mask, float("-inf"))
        outs.append(torch.einsum("bhqk,bhkd->bhqd", torch.softmax(s, dim=-1), vi)[0].transpose(0, 1))
    out = torch.cat(outs)
    return out, qr, kr, vr


VARLEN_CASES = [
    # Hq, Hkv, D, lens, window, enable_gqa, kv_packed
    pytest.param(8, 8, 128, [333, 128, 512, 47], (-1, 0), False, False, id="causal"),
    pytest.param(8, 8, 128, [256, 384], (-1, -1), False, False, id="non-causal"),
    pytest.param(16, 4, 128, [200, 312, 96], (-1, 0), True, False, id="gqa"),
    pytest.param(8, 8, 128, [400, 288], (128, 0), False, False, id="window-128"),
    pytest.param(8, 8, 128, [400, 288], (4, 0), False, False, id="window-4-tight"),
    pytest.param(8, 8, 64, [512, 512], (-1, 0), False, False, id="d64"),
    pytest.param(
        8,
        8,
        128,
        [333, 128, 512, 47],
        (-1, 0),
        False,
        True,
        id="kv-interleaved",
    ),
]


@pytest.mark.L0
@pytest.mark.parametrize("Hq,Hkv,D,lens,window,enable_gqa,kv_packed", VARLEN_CASES)
def test_varlen_attn(Hq, Hkv, D, lens, window, enable_gqa, kv_packed):
    torch.manual_seed(0)
    lens_t = torch.tensor(lens, device="cuda")
    cu = torch.nn.functional.pad(lens_t.cumsum(0), (1, 0)).to(torch.int32)
    T, mx = int(cu[-1]), int(lens_t.max())
    q = torch.randn(T, Hq, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    if kv_packed:  # non-contiguous k/v views of one buffer (token stride 2*H*D)
        kv = torch.randn(T, 2, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        k, v = kv[:, 0], kv[:, 1]
    else:
        k = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        v = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    fwd0, bwd0 = provider.calls["fwd"], provider.calls["bwd"]
    out, lse = varlen_attn(q, k, v, cu, cu, mx, mx, window_size=window, enable_gqa=enable_gqa, return_aux=AuxRequest(lse=True))
    grad = torch.randn_like(out)
    out.backward(grad)
    assert provider.calls["fwd"] == fwd0 + 1 and provider.calls["bwd"] == bwd0 + 1, "provider did not intercept"

    is_causal = window[1] == 0
    ref, qr, kr, vr = ref_varlen(q, k, v, cu, cu, is_causal, window[0])
    ref.backward(grad.float())
    if kv_packed:
        kv_grad = torch.stack([kr.grad, vr.grad], dim=1)
        dk_err = dv_err = (kv.grad.float() - kv_grad).abs().max().item()
    else:
        dk_err = (k.grad.float() - kr.grad).abs().max().item()
        dv_err = (v.grad.float() - vr.grad).abs().max().item()

    # dk/dv accumulate Hq/Hkv gradient groups in bf16 — error grows ~sqrt(group)
    # (stock flash shows the same inflation on GQA).
    group = Hq // Hkv
    tol = {"o": 2.5e-2, "dq": 2.5e-2, "dk": 2.5e-2 * group**0.5, "dv": 2.5e-2 * group**0.5}
    errs = {
        "o": (out.float() - ref).abs().max().item(),
        "dq": (q.grad.float() - qr.grad).abs().max().item(),
        "dk": dk_err,
        "dv": dv_err,
    }
    for name, e in errs.items():
        assert e < tol[name], f"{name}: err {e:.4f} tol {tol[name]:.4f}"


@pytest.mark.L0
def test_varlen_backward_does_not_sync():
    """The varlen backward must not read cu_seqlens to host.

    It used to repad the packed LSE with `for i in range(B): int(cu_seq_q[i])`
    — 2*B blocking D2H copies before the kernel even launched, turning an
    async-launch API synchronous and blocking stream capture
    (python/cudnn/AGENTS.md Rule 3). The conversion is device-side now;
    CUDA's sync-debug mode turns any regression into an error.
    """
    torch.manual_seed(0)
    H, D = 8, 128
    lens_t = torch.tensor([128, 96, 200], device="cuda")
    cu = torch.nn.functional.pad(lens_t.cumsum(0), (1, 0)).to(torch.int32)
    T, mx = int(cu[-1]), int(lens_t.max())
    q, k, v = (torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True) for _ in range(3))

    bwd0 = provider.calls["bwd"]
    out = varlen_attn(q, k, v, cu, cu, mx, mx, window_size=(-1, 0))
    grad = torch.randn_like(out)

    # Any blocking D2H inside the backward raises here.
    torch.cuda.set_sync_debug_mode("error")
    try:
        out.backward(grad)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert provider.calls["bwd"] == bwd0 + 1, "provider did not serve the backward"
    assert q.grad is not None and k.grad is not None and v.grad is not None


@pytest.mark.L0
def test_d256_direct_aten_op():
    """d=256: torch's C++ fused_sdp_choice still gates cuDNN to head_dim<=128,
    so F.sdpa cannot reach it — but the python path serves it through the aten
    op directly (what a fixed selection gate would dispatch to)."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 384, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k, v = torch.randn_like(q, requires_grad=True), torch.randn_like(q, requires_grad=True)
    try:
        out = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, False)
    except RuntimeError as e:
        pytest.skip(f"no engine serves d=256 on this arch: {str(e).splitlines()[0][:80]}")
    o = out[0]
    o.backward(torch.ones_like(o))
    o_ref, q_ref, _, _ = math_ref(q, k, v, False, None)
    o_ref.backward(torch.ones_like(o_ref))
    assert (o.float() - o_ref).abs().max().item() < 0.05
    assert (q.grad.float() - q_ref.grad).abs().max().item() < 0.5
