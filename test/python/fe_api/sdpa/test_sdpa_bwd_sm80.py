# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SM80 (A100) SDPA backward API.

Skipped automatically on non-SM80 devices or when the optional CuTe-DSL
dependency (``nvidia-cutlass-dsl``) is missing.  Runs the SM80 forward to
produce O/LSE, then compares dQ/dK/dV against a fp32 torch autograd
reference.
"""

import math
import pytest
import torch

from test_utils import torch_fork_set_rng


def _is_sm80() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) == (8, 0)


def _dsl_available() -> bool:
    # The kernels need the CuTe DSL *with* cutlass.experimental (cutlass-dsl
    # >= 4.7). The package imports lazily (PEP 562), so a missing/old DSL
    # only surfaces at kernel-load time — probe it here so the suite SKIPS
    # instead of erroring mid-test.
    try:
        import cutlass.experimental  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_is_sm80() and _dsl_available()),
    reason="SM80 SDPA API requires an SM80 (A100) device and nvidia-cutlass-dsl >= 4.7.",
)


def _bshd_randn(b, h, s, d, **kw):
    """BHSD-logical tensor with the BSHD-physical stride order the SM80
    adapters require."""
    return torch.randn((b, s, h, d), **kw).permute(0, 2, 1, 3)


def _ref_grads(q, k, v, do, *, is_causal, window_left, scale):
    """fp32 autograd reference; returns (o, dq, dk, dv)."""
    _, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = k.shape
    g = h_q // h_kv
    q_ref = q.detach().to(torch.float32).requires_grad_()
    k_ref = k.detach().to(torch.float32).requires_grad_()
    v_ref = v.detach().to(torch.float32).requires_grad_()

    k_exp = k_ref.repeat_interleave(g, dim=1)
    v_exp = v_ref.repeat_interleave(g, dim=1)
    scores = torch.matmul(q_ref, k_exp.transpose(-1, -2)) * scale
    if is_causal:
        # Top-left diagonal, matching the wrapper's default mask (the tests
        # never pass causal_bottom_right); with a bottom-right-anchored
        # reference the two would agree only while s_q == s_kv.
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        keep = j <= i
        if window_left >= 0:
            keep &= (i - j) <= window_left
        scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs, v_exp)
    o.backward(do.to(torch.float32))
    return o.detach(), q_ref.grad, k_ref.grad, v_ref.grad


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_bwd_sm80_smoke():
    """One representative case at L0 (llama flavor, fp16, causal, MHA);
    the full flavor x mask x GQA x dtype sweep runs at L2."""
    test_sdpa_bwd_sm80_wrapper(torch.float16, 128, 128, "causal", (8, 8))


@pytest.mark.L2
@pytest.mark.parametrize("d_qk,d_v", [(64, 64), (128, 128), (192, 128), (256, 256)], ids=["gptoss", "llama", "dsv3", "qwen"])
@pytest.mark.parametrize("mask", ["none", "causal", "causal_swa"])
@pytest.mark.parametrize("gqa", [(8, 8), (16, 4)], ids=["mha", "gqa4x"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_sdpa_bwd_sm80_wrapper(dtype, d_qk, d_v, mask, gqa):
    try:
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, s_q, s_kv = 2, 512, 512
    h_q, h_kv = gqa
    device = "cuda"

    q = _bshd_randn(b, h_q, s_q, d_qk, dtype=dtype, device=device)
    k = _bshd_randn(b, h_kv, s_kv, d_qk, dtype=dtype, device=device)
    v = _bshd_randn(b, h_kv, s_kv, d_v, dtype=dtype, device=device)
    do = _bshd_randn(b, h_q, s_q, d_v, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(d_qk)

    is_causal = mask in ("causal", "causal_swa")
    window = (128, 0) if mask == "causal_swa" else (-1, -1)

    try:
        fwd = sdpa_fwd_wrapper_sm80(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            is_causal=is_causal,
            window_size=window,
            scale_softmax=scale,
        )
        out = sdpa_bwd_wrapper_sm80(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=fwd["o_tensor"],
            do_tensor=do,
            lse_tensor=fwd["lse_tensor"],
            is_causal=is_causal,
            window_size=window,
            scale_softmax=scale,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    _, dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do, is_causal=is_causal, window_left=window[0], scale=scale)

    # fp16 backward accumulates over S; scale tolerance accordingly.
    torch.testing.assert_close(out["dq_tensor"].to(torch.float32), dq_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out["dk_tensor"].to(torch.float32), dk_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out["dv_tensor"].to(torch.float32), dv_ref, rtol=3e-2, atol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_bwd_sm80_deterministic_repeatable():
    """deterministic=True must produce bitwise-identical dQ across runs."""
    try:
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, s, d = 1, 4, 1024, 128
    dtype = torch.float16
    q = _bshd_randn(b, h, s, d, dtype=dtype, device="cuda")
    k = _bshd_randn(b, h, s, d, dtype=dtype, device="cuda")
    v = _bshd_randn(b, h, s, d, dtype=dtype, device="cuda")
    do = _bshd_randn(b, h, s, d, dtype=dtype, device="cuda")
    scale = 1.0 / math.sqrt(d)

    fwd = sdpa_fwd_wrapper_sm80(q_tensor=q, k_tensor=k, v_tensor=v, is_causal=True, scale_softmax=scale)

    def _run():
        return sdpa_bwd_wrapper_sm80(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=fwd["o_tensor"],
            do_tensor=do,
            lse_tensor=fwd["lse_tensor"],
            is_causal=True,
            scale_softmax=scale,
            deterministic=True,
        )

    a = _run()
    bwd = _run()
    assert torch.equal(a["dq_tensor"], bwd["dq_tensor"])
    assert torch.equal(a["dk_tensor"], bwd["dk_tensor"])
    assert torch.equal(a["dv_tensor"], bwd["dv_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_bwd_sm80_d64_fast_path():
    """The dedicated d=64 kernel routes only for plain dense MHA and agrees
    with the generic kernel on the same inputs."""
    try:
        from cudnn.sdpa.bwd import api as api_sm80
        from cudnn.sdpa.bwd.kernels import bprop_d64_f16_sm80 as d64, bprop_f16_sm80 as gen
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    common = dict(d_qk=64, d_v=64, h_q=8, h_kv=8, s_q=512, s_kv=512, mask_token="none", right_bound=0, causal_bottom_right=False, bw_kwargs={})
    assert api_sm80._d64_fast_path_eligible(**common)
    # every gated condition individually disqualifies
    for override in (
        dict(d_qk=48, d_v=48),  # padded flavor
        dict(h_kv=4),  # GQA
        dict(s_q=500),  # not M_BLOCK-aligned
        dict(mask_token="causal"),
        dict(right_bound=2),
        dict(causal_bottom_right=True),
        dict(bw_kwargs={"bias": object()}),
        dict(bw_kwargs={"deterministic": True}),
    ):
        assert not api_sm80._d64_fast_path_eligible(**{**common, **override}), override

    b, h, s, d = 2, 8, 512, 64
    q = torch.randn(b, s, h, d, dtype=torch.float16, device="cuda")  # BSHD (kernel layout)
    k, v, do, o = (torch.randn_like(q) for _ in range(4))
    lse = torch.randn(b, h, s, dtype=torch.float32, device="cuda").abs() + 5
    scale = 1.0 / math.sqrt(d)
    dq_g, dk_g, dv_g = gen.backward(q, k, v, do, o, lse, scale=scale, mask="none")
    dq_d, dk_d, dv_d = d64.backward(q, k, v, do, o, lse, scale=scale)
    torch.testing.assert_close(dq_d.float(), dq_g.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dk_d.float(), dk_g.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dv_d.float(), dv_g.float(), rtol=2e-2, atol=2e-2)
