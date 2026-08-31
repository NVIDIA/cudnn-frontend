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
def test_sdpa_bwd_sm80_d64_fast_path(monkeypatch):
    """The dedicated d=64 kernel routes only for plain dense MHA and agrees
    with the generic kernel on the same inputs (the generic side runs through
    the adapter with the d64 gate forced off — the generic module is a
    TemplateParams template with no standalone entry point)."""
    try:
        from cudnn.sdpa.bwd import api_dsl as api_sm80
        from cudnn.sdpa.bwd.kernels import bprop_d64_f16_sm80 as d64
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    common = dict(d_qk=64, d_v=64, h_q=8, h_kv=8, s_q=512, s_kv=512, mask_token="none", right_bound=0, causal_bottom_right=False, bw_kwargs={})
    assert api_sm80._sm80_d64_fast_path_eligible(**common)
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
        assert not api_sm80._sm80_d64_fast_path_eligible(**{**common, **override}), override

    b, h, s, d = 2, 8, 512, 64
    q = torch.randn(b, s, h, d, dtype=torch.float16, device="cuda")  # BSHD (kernel layout)
    k, v, do, o = (torch.randn_like(q) for _ in range(4))
    lse = torch.randn(b, h, s, dtype=torch.float32, device="cuda").abs() + 5
    scale = 1.0 / math.sqrt(d)
    # Generic path: build the adapter directly (BHSD-logical views of the same
    # BSHD storage) with the d64 gate forced off, so it compiles + launches
    # the generic TemplateParams module.
    monkeypatch.setattr(api_sm80, "_sm80_d64_fast_path_eligible", lambda **kw: False)
    qb, kb, vb, ob, dob = (t.transpose(1, 2) for t in (q, k, v, o, do))
    dq_g = torch.empty(b, s, h, d, dtype=q.dtype, device="cuda").transpose(1, 2)
    dk_g = torch.empty_like(dq_g)
    dv_g = torch.empty_like(dq_g)
    eng = api_sm80.SdpaBwdDslSm80(
        sample_q=qb,
        sample_k=kb,
        sample_v=vb,
        sample_o=ob,
        sample_do=dob,
        sample_stats=lse,
        sample_dq=dq_g,
        sample_dk=dk_g,
        sample_dv=dv_g,
        is_causal=False,
        scale_softmax=scale,
    )
    assert eng.check_support()
    eng.compile()
    assert not eng._use_d64
    eng.execute(
        q_tensor=qb, k_tensor=kb, v_tensor=vb, o_tensor=ob, do_tensor=dob, stats_tensor=lse, dq_tensor=dq_g, dk_tensor=dk_g, dv_tensor=dv_g, scale_softmax=scale
    )
    dq_g, dk_g, dv_g = (t.transpose(1, 2) for t in (dq_g, dk_g, dv_g))  # back to BSHD
    dq_d, dk_d, dv_d = d64.backward(q, k, v, do, o, lse, scale=scale)
    torch.testing.assert_close(dq_d.float(), dq_g.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dk_d.float(), dk_g.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dv_d.float(), dv_g.float(), rtol=2e-2, atol=2e-2)


def _thd_run_and_check(lens, h, d_qk, d_v, *, check_grads=True):
    """Run the THD fwd+bwd wrappers on packed random inputs; when
    ``check_grads``, compare each sequence's dQ/dK/dV slice against the dense
    fp32 autograd reference (the existing ``_ref_grads`` pattern)."""
    import itertools

    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_sm80
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80

    t = int(sum(lens))
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device="cuda")
    q = torch.randn(1, t, h, d_qk, dtype=torch.float16, device="cuda")
    k = torch.randn(1, t, h, d_qk, dtype=torch.float16, device="cuda")
    v = torch.randn(1, t, h, d_v, dtype=torch.float16, device="cuda")
    do = torch.randn(1, t, h, d_v, dtype=torch.float16, device="cuda")
    fwd = sdpa_fwd_wrapper_sm80(q, k, v, is_causal=True, cum_seqlen_q_tensor=cu, cum_seqlen_k_tensor=cu, max_s_q=int(max(lens)))
    out = sdpa_bwd_wrapper_sm80(q, k, v, fwd["o_tensor"], do, fwd["lse_tensor"], is_causal=True, cum_seqlen_q_tensor=cu, cum_seqlen_k_tensor=cu)
    if check_grads:
        for i in range(len(lens)):
            lo, hi = int(cu[i]), int(cu[i + 1])

            # packed [T, H, D] slice -> BHSD [1, H, S, D]
            def _bhsd(x, _lo=lo, _hi=hi):
                return x[0, _lo:_hi].permute(1, 0, 2).unsqueeze(0)

            _, dq_ref, dk_ref, dv_ref = _ref_grads(_bhsd(q), _bhsd(k), _bhsd(v), _bhsd(do), is_causal=True, window_left=-1, scale=1.0 / math.sqrt(d_qk))
            torch.testing.assert_close(_bhsd(out["dq_tensor"]).to(torch.float32), dq_ref, rtol=3e-2, atol=3e-2)
            torch.testing.assert_close(_bhsd(out["dk_tensor"]).to(torch.float32), dk_ref, rtol=3e-2, atol=3e-2)
            torch.testing.assert_close(_bhsd(out["dv_tensor"]).to(torch.float32), dv_ref, rtol=3e-2, atol=3e-2)
    return out


@pytest.mark.L0
@pytest.mark.parametrize("d_qk,d_v", [(64, 64), (192, 128)], ids=["gptoss_env", "dsv3_env"])
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_thd_flavor_envelope_dims(d_qk, d_v):
    """Regression: THD must compile the kernel at the FLAVOR ENVELOPE dims,
    not the template's 128/128 defaults — a flavor-name-only params build
    wrote dQ/dK/dV out of bounds at d=64 and returned wrong gradients at
    192/128 (CodeRabbit critical on the TemplateParams port)."""
    try:
        _thd_run_and_check([96, 160], h=4, d_qk=d_qk, d_v=d_v)
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_thd_compile_key_plan_time_only():
    """Issue #604 regression (backward): the packed THD token totals are
    RUNTIME values, so two varlen backward calls with different totals must
    re-bind ONE compiled artifact (the bprop template's per-shape lru sees a
    single miss) — never mint a compile per step, the continuous-batching
    pathology no correctness test catches."""
    from cudnn.frost import template_loader
    from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_sm80
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80

    H, D = 4, 128

    def varlen(lens):
        import itertools

        t = int(sum(lens))
        cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device="cuda")
        q = torch.randn(1, t, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        do = torch.randn_like(q)
        fwd = sdpa_fwd_wrapper_sm80(q, k, v, is_causal=True, cum_seqlen_q_tensor=cu, cum_seqlen_k_tensor=cu, max_s_q=int(max(lens)))
        return sdpa_bwd_wrapper_sm80(
            q,
            k,
            v,
            fwd["o_tensor"],
            do,
            fwd["lse_tensor"],
            is_causal=True,
            cum_seqlen_q_tensor=cu,
            cum_seqlen_k_tensor=cu,
        )

    def cache_totals():
        # Count ONLY the bprop template's per-shape lru (the fwd wrapper runs
        # too, and its counters are covered by the forward's twin test); the
        # counters are session-global, so assert on DELTAS across our calls.
        mods = [m for (path, _params), m in template_loader._MODULES.items() if "bprop" in str(path)]
        infos = [m.compile.cache_info() for m in mods if hasattr(m.compile, "cache_info")]
        return sum(i.misses for i in infos), sum(i.hits for i in infos)

    varlen([96, 160])  # first call: one compile
    n_modules_before = len(template_loader._MODULES)
    misses_0, hits_0 = cache_totals()
    varlen([128, 64, 320])  # different totals AND batch count
    # Different logical batch counts legitimately re-specialize (the cu fake
    # length is plan-time); different TOKEN TOTALS at the same batch count
    # must not — and the re-bound artifact must still be CORRECT, so this
    # call's gradients are validated against the dense per-sequence reference.
    # The counter window brackets THIS call alone: a pure cache hit, zero
    # misses (netting against the n_seqs=3 call could mask a leak).
    misses_pre_rebind, hits_pre_rebind = cache_totals()
    _thd_run_and_check([64, 256], h=H, d_qk=D, d_v=D)  # same n_seqs as call 1, different totals
    misses_post_rebind, hits_post_rebind = cache_totals()
    assert misses_post_rebind == misses_pre_rebind, "the same-batch-count re-bind minted a compile (token totals leaked into the key)"
    assert hits_post_rebind > hits_pre_rebind, "expected the same-batch-count re-bind to cache-hit"
    assert len(template_loader._MODULES) == n_modules_before, "a new template specialization was minted by runtime data"
    misses_1, hits_1 = cache_totals()
    # Call 2 (n_seqs=3) may legitimately re-specialize once; call 3 shares
    # call 1's key (n_seqs=2, different token totals) and MUST cache-hit.
    assert misses_1 - misses_0 <= 1, f"THD bprop compile key leaked runtime data: {misses_1 - misses_0} new misses"
    assert hits_1 - hits_0 >= 1, "expected a cache hit on the same-batch-count re-call"
