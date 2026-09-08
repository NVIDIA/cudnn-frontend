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


def _ref_grads(q, k, v, do, *, is_causal, window_left, scale, causal_bottom_right=False, sink=None):
    """fp32 autograd reference; returns (o, dq, dk, dv) — or (o, dq, dk, dv,
    dsink) when ``sink`` ((H,) fp32 natural-log logits) is given: one extra
    softmax column per head that absorbs probability mass and contributes
    nothing to ``o``.

    Mask semantics mirror the kernel's ``_mask_p``: the diagonal anchor is
    ``q`` (top-left) or ``q + (s_kv - s_q)`` (bottom-right); causal keeps
    ``kv <= anchor``; a left window keeps ``kv >= anchor - W`` (also without
    causal — the non-causal "swa" mask)."""
    _, h_q, s_q, _ = q.shape
    _, h_kv, s_kv, _ = k.shape
    g = h_q // h_kv
    q_ref = q.detach().to(torch.float32).requires_grad_()
    k_ref = k.detach().to(torch.float32).requires_grad_()
    v_ref = v.detach().to(torch.float32).requires_grad_()
    sink_ref = sink.detach().to(torch.float32).requires_grad_() if sink is not None else None

    k_exp = k_ref.repeat_interleave(g, dim=1)
    v_exp = v_ref.repeat_interleave(g, dim=1)
    scores = torch.matmul(q_ref, k_exp.transpose(-1, -2)) * scale
    if is_causal or window_left >= 0:
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        anchor = i + (s_kv - s_q) if causal_bottom_right else i
        keep = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device)
        if is_causal:
            keep &= j <= anchor
        if window_left >= 0:
            keep &= (anchor - j) <= window_left
        scores = scores.masked_fill(~keep, float("-inf"))
    if sink_ref is not None:
        col = sink_ref.view(1, h_q, 1, 1).expand(scores.shape[0], h_q, s_q, 1)
        probs = torch.softmax(torch.cat([scores, col], dim=-1), dim=-1)[..., :s_kv]
    else:
        probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs, v_exp)
    o.backward(do.to(torch.float32))
    if sink_ref is not None:
        return o.detach(), q_ref.grad, k_ref.grad, v_ref.grad, sink_ref.grad
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


def _thd_run_and_check(lens, h, d_qk, d_v, *, check_grads=True, window_left=-1, sinks=None, deterministic=False, max_s_q=None):
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
    window = (window_left, 0)
    fwd = sdpa_fwd_wrapper_sm80(
        q, k, v, is_causal=True, window_size=window, sinks=sinks, cum_seqlen_q_tensor=cu, cum_seqlen_k_tensor=cu, max_s_q=int(max(lens))
    )
    extra = {"max_s_q": max_s_q} if max_s_q is not None else {}
    out = sdpa_bwd_wrapper_sm80(
        q,
        k,
        v,
        fwd["o_tensor"],
        do,
        fwd["lse_tensor"],
        is_causal=True,
        window_size=window,
        sinks=sinks,
        deterministic=deterministic,
        cum_seqlen_q_tensor=cu,
        cum_seqlen_k_tensor=cu,
        **extra,
    )
    if check_grads:
        dsink_sum = torch.zeros(h, dtype=torch.float32, device="cuda") if sinks is not None else None
        for i in range(len(lens)):
            lo, hi = int(cu[i]), int(cu[i + 1])

            # packed [T, H, D] slice -> BHSD [1, H, S, D]
            def _bhsd(x, _lo=lo, _hi=hi):
                return x[0, _lo:_hi].permute(1, 0, 2).unsqueeze(0)

            refs = _ref_grads(_bhsd(q), _bhsd(k), _bhsd(v), _bhsd(do), is_causal=True, window_left=window_left, scale=1.0 / math.sqrt(d_qk), sink=sinks)
            _, dq_ref, dk_ref, dv_ref = refs[:4]
            torch.testing.assert_close(_bhsd(out["dq_tensor"]).to(torch.float32), dq_ref, rtol=3e-2, atol=3e-2)
            torch.testing.assert_close(_bhsd(out["dk_tensor"]).to(torch.float32), dk_ref, rtol=3e-2, atol=3e-2)
            torch.testing.assert_close(_bhsd(out["dv_tensor"]).to(torch.float32), dv_ref, rtol=3e-2, atol=3e-2)
            if sinks is not None:
                dsink_sum += refs[4]
        if sinks is not None:
            # dSink sums over every sequence's tokens; a NONZERO reference is
            # required so an all-zero kernel result cannot pass.
            assert dsink_sum.abs().max().item() > 0
            torch.testing.assert_close(out["dsink_tensor"], dsink_sum, rtol=3e-2, atol=3e-2 * max(1.0, dsink_sum.abs().max().item()))
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
        modules = [m for (path, _params), m in template_loader._MODULES.items() if "bprop" in str(path)]
        infos = [m.compile.cache_info() for m in modules if hasattr(m.compile, "cache_info")]
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


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_strided_stats_native_reads():
    """The kernels read a STRIDED Stats/LSE input natively (the #712 analogue
    for the backward's loads): an interleaved-stride stats declaration must
    produce bitwise the same grads as the contiguous run — with no gather
    staging (the workspace stays exactly the packed plan's size)."""
    try:
        from cudnn.sdpa.bwd import api_dsl as api_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, s, d = 2, 4, 512, 128
    scale = 1.0 / math.sqrt(d)
    q = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    k = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    v = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    do = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    fwd = sdpa_fwd_wrapper_sm80(q_tensor=q, k_tensor=k, v_tensor=v, is_causal=True, scale_softmax=scale)
    lse_c = fwd["lse_tensor"].contiguous()

    # Interleave the (B, H, S) LSE into a doubled-S buffer: stride (2*H*S, 2*S, 2).
    buf = torch.full((b, h, s, 2), float("nan"), dtype=torch.float32, device="cuda")
    lse_strided = buf.as_strided((b, h, s), (2 * h * s, 2 * s, 2))
    lse_strided.copy_(lse_c)

    def run(lse):
        dq = torch.empty(b, s, h, d, dtype=torch.float16, device="cuda").transpose(1, 2)
        dk, dv = torch.empty_like(dq), torch.empty_like(dq)
        eng = api_sm80.SdpaBwdDslSm80(
            sample_q=q,
            sample_k=k,
            sample_v=v,
            sample_o=fwd["o_tensor"],
            sample_do=do,
            sample_stats=lse,
            sample_dq=dq,
            sample_dk=dk,
            sample_dv=dv,
            is_causal=True,
            scale_softmax=scale,
        )
        assert eng.check_support()
        eng.compile()
        ws = torch.empty(eng.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")
        eng.execute(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=fwd["o_tensor"],
            do_tensor=do,
            stats_tensor=lse,
            dq_tensor=dq,
            dk_tensor=dk,
            dv_tensor=dv,
            scale_softmax=scale,
            workspace=ws,
        )
        return eng, dq, dk, dv

    eng_c, dq_c, dk_c, dv_c = run(lse_c)
    eng_s, dq_s, dk_s, dv_s = run(lse_strided)
    assert eng_s._lse_stride == (2 * h * s, 2 * s, 2)
    # Native reads: the strided plan needs NO staging — same scratch as packed.
    assert eng_s.scratch_workspace_bytes() == eng_c.scratch_workspace_bytes()
    assert torch.equal(dq_s, dq_c)
    assert torch.equal(dk_s, dk_c)
    assert torch.equal(dv_s, dv_c)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_thd_max_s_kv_hint():
    """The THD wrapper's max_s_kv grid hint must (a) produce bitwise the same
    grads as the hint-less host-read path, including when over-provisioned
    (short kv-tiles early-out), and (b) be rejected on the dense path."""
    import itertools

    try:
        from cudnn.sdpa.bwd.api_dsl import sdpa_bwd_wrapper_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    h, d = 4, 128
    lens = [96, 320, 160]
    t = int(sum(lens))
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device="cuda")
    q = torch.randn(1, t, h, d, dtype=torch.float16, device="cuda")
    k, v, do = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    fwd = sdpa_fwd_wrapper_sm80(q, k, v, is_causal=True, cum_seqlen_q_tensor=cu, cum_seqlen_k_tensor=cu, max_s_q=max(lens))

    def run(**kw):
        return sdpa_bwd_wrapper_sm80(q, k, v, fwd["o_tensor"], do, fwd["lse_tensor"], is_causal=True, cum_seqlen_q_tensor=cu, cum_seqlen_k_tensor=cu, **kw)

    base = run()
    exact = run(max_s_kv=max(lens))
    over = run(max_s_kv=t)  # upper bound: extra kv-tiles early-out
    for key in ("dq_tensor", "dk_tensor", "dv_tensor"):
        assert torch.equal(exact[key], base[key]), key
        assert torch.equal(over[key], base[key]), key

    with pytest.raises(ValueError, match="max_s_kv"):
        sdpa_bwd_wrapper_sm80(
            _bshd_randn(1, h, 128, d, dtype=torch.float16, device="cuda"),
            _bshd_randn(1, h, 128, d, dtype=torch.float16, device="cuda"),
            _bshd_randn(1, h, 128, d, dtype=torch.float16, device="cuda"),
            _bshd_randn(1, h, 128, d, dtype=torch.float16, device="cuda"),
            _bshd_randn(1, h, 128, d, dtype=torch.float16, device="cuda"),
            torch.randn(1, h, 128, dtype=torch.float32, device="cuda"),
            max_s_kv=128,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_wrapper_lse_stride_in_cache_key():
    """The wrapper's plan cache must key on the Stats/LSE geometry: the
    compiled kernel is SPECIALIZED on the declared LSE strides (native
    strided reads), so a cache hit across differently-laid-out LSE views
    would misread the stats — same q/k/v, contiguous-then-strided LSE must
    produce bitwise-identical gradients."""
    try:
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, s, d = 2, 4, 512, 128
    scale = 1.0 / math.sqrt(d)
    q = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    k = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    v = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    do = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    fwd = sdpa_fwd_wrapper_sm80(q_tensor=q, k_tensor=k, v_tensor=v, is_causal=True, scale_softmax=scale)
    lse_c = fwd["lse_tensor"].contiguous()

    # Interleaved-stride view of the SAME values: stride (2*h*s, 2*s, 2).
    buf = torch.full((b, h, s, 2), float("nan"), dtype=torch.float32, device="cuda")
    lse_strided = buf.as_strided((b, h, s), (2 * h * s, 2 * s, 2))
    lse_strided.copy_(lse_c)

    def run(lse):
        return sdpa_bwd_wrapper_sm80(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=fwd["o_tensor"],
            do_tensor=do,
            lse_tensor=lse,
            is_causal=True,
            scale_softmax=scale,
        )

    base = run(lse_c)  # plan cached with the packed LSE layout
    strided = run(lse_strided)  # must NOT reuse the packed plan
    for key in ("dq_tensor", "dk_tensor", "dv_tensor"):
        assert torch.equal(strided[key], base[key]), key

    # Device guard: a host-side LSE must be rejected loudly, never bound as a
    # kernel pointer (the plan's cached adapter would otherwise launch on it).
    with pytest.raises(ValueError, match="device"):
        run(lse_c.cpu())

    # And at PLAN time: a CPU Stats DECLARATION must fail check_support —
    # otherwise the execute-time guard (which validates against
    # stats_desc.device) would anchor to the host device and pass.
    from cudnn.sdpa.bwd import api_dsl as _api

    dq = torch.empty_like(q)
    eng = _api.SdpaBwdDslSm80(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=fwd["o_tensor"],
        sample_do=do,
        sample_stats=lse_c.cpu(),
        sample_dq=dq,
        sample_dk=dq,
        sample_dv=dq,
        is_causal=True,
        scale_softmax=scale,
    )
    with pytest.raises(ValueError, match="device"):
        eng.check_support()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_swa_long_seq_smoke():
    """One long-sequence sliding-window case at L0 (the q-loop bound is a kernel
    change, so a single S >> W check runs by default); the three window
    geometries sweep at L1."""
    test_sm80_bwd_swa_long_seq(True, False, 2048, 2048)


@pytest.mark.L1
@pytest.mark.parametrize(
    "is_causal,bottom_right,s_q,s_kv",
    [(True, False, 2048, 2048), (False, False, 2048, 2048), (True, True, 1536, 2048)],
    ids=["causal_swa_tl", "swa_only", "causal_swa_br"],
)
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_swa_long_seq(is_causal, bottom_right, s_q, s_kv):
    """Sliding window at S >> W: the backward bounds each kv-tile's q-loop to
    the window (q <= kv + W - br_diag), the mirror of the forward's kv_left
    trim, instead of sweeping every q-tile and masking. Correctness against the
    dense reference across the three window geometries the kernel serves:
    top-left causal+window, window-only (non-causal), bottom-right causal+window
    with s_q != s_kv (the anchor shifts by s_kv - s_q)."""
    try:
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, d, w = 1, 4, 64, 128
    scale = 1.0 / math.sqrt(d)
    q = _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda")
    k = _bshd_randn(b, h, s_kv, d, dtype=torch.float16, device="cuda")
    v = _bshd_randn(b, h, s_kv, d, dtype=torch.float16, device="cuda")
    do = _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda")
    # window_size_right is a causal-band attribute: the wrappers take -1
    # (unset) for a non-causal left window and 0 (no widening) under causal.
    window = (w, 0 if is_causal else -1)
    fwd = sdpa_fwd_wrapper_sm80(
        q_tensor=q, k_tensor=k, v_tensor=v, is_causal=is_causal, window_size=window, causal_bottom_right=bottom_right, scale_softmax=scale
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
        causal_bottom_right=bottom_right,
        scale_softmax=scale,
    )
    _, dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, do, is_causal=is_causal, window_left=w, scale=scale, causal_bottom_right=bottom_right)
    torch.testing.assert_close(out["dq_tensor"].to(torch.float32), dq_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out["dk_tensor"].to(torch.float32), dk_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out["dv_tensor"].to(torch.float32), dv_ref, rtol=3e-2, atol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_thd_swa():
    """THD + sliding window: the q-loop bound is per packed sequence (in-seq
    q/kv indices); sequences longer and shorter than the window mix."""
    try:
        _thd_run_and_check([96, 1024, 640], h=4, d_qk=64, d_v=64, window_left=128)
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_swa_deterministic_smoke():
    """Top-left deterministic + window at L0 (the relay's first-visitor turn is
    the hang-risk path, so one case runs by default); bottom-right at L1."""
    test_sm80_bwd_swa_deterministic(False, 2048, 2048)


@pytest.mark.L1
@pytest.mark.parametrize("bottom_right,s_q,s_kv", [(False, 2048, 2048), (True, 1536, 2048)], ids=["tl", "br"])
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_swa_deterministic(bottom_right, s_q, s_kv):
    """Deterministic dQ + sliding window at S >> W: the window-bounded q-loop
    means a q-tile's kv-tile visitors start at the window floor, not at 0, so
    the ordered relay counts turns from that first visitor (the SM120 kernel's
    `relay_turn`). Must be bitwise repeatable AND agree with the
    non-deterministic path — a wrong first-visitor would hang or mis-order."""
    try:
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, d, w = 1, 4, 64, 128
    scale = 1.0 / math.sqrt(d)
    q = _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda")
    k = _bshd_randn(b, h, s_kv, d, dtype=torch.float16, device="cuda")
    v = _bshd_randn(b, h, s_kv, d, dtype=torch.float16, device="cuda")
    do = _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda")
    window = (w, 0)
    fwd = sdpa_fwd_wrapper_sm80(q_tensor=q, k_tensor=k, v_tensor=v, is_causal=True, window_size=window, causal_bottom_right=bottom_right, scale_softmax=scale)

    def run(det):
        return sdpa_bwd_wrapper_sm80(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=fwd["o_tensor"],
            do_tensor=do,
            lse_tensor=fwd["lse_tensor"],
            is_causal=True,
            window_size=window,
            causal_bottom_right=bottom_right,
            scale_softmax=scale,
            deterministic=det,
        )

    a, bb = run(True), run(True)
    for key in ("dq_tensor", "dk_tensor", "dv_tensor"):
        assert torch.equal(a[key], bb[key]), key
    nd = run(False)
    _, dq_ref, _, _ = _ref_grads(q, k, v, do, is_causal=True, window_left=w, scale=scale, causal_bottom_right=bottom_right)
    torch.testing.assert_close(a["dq_tensor"].to(torch.float32), dq_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(a["dq_tensor"].to(torch.float32), nd["dq_tensor"].to(torch.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_thd_sinks():
    """THD + attention sinks: dQ/dK/dV per sequence against the sink-aware
    reference, and dSink (H,) against the sum of the per-sequence sink grads
    (the reduction runs over each packed sequence's own tokens)."""
    sinks = torch.randn(4, dtype=torch.float32, device="cuda")
    try:
        _thd_run_and_check([96, 320, 160], h=4, d_qk=128, d_v=128, sinks=sinks)
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_thd_deterministic():
    """THD + deterministic dQ: correct against the reference, bitwise
    repeatable, and identical whether the relay semaphore is sized from an
    exact max_s_q hint, an over-provisioned one, or (no hint) the packed
    total."""
    lens = [96, 640, 160]
    try:
        a = _thd_run_and_check(lens, h=4, d_qk=128, d_v=128, deterministic=True)
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")
    torch.manual_seed(0)
    b = _thd_run_and_check(lens, h=4, d_qk=128, d_v=128, deterministic=True, check_grads=False, max_s_q=max(lens))
    torch.manual_seed(0)
    c = _thd_run_and_check(lens, h=4, d_qk=128, d_v=128, deterministic=True, check_grads=False, max_s_q=sum(lens))
    for key in ("dq_tensor", "dk_tensor", "dv_tensor"):
        assert torch.equal(a[key], b[key]), key
        assert torch.equal(a[key], c[key]), key


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_dense_sinks():
    """Dense attention sinks through the wrapper -> adapter -> dSink kernel:
    dQ/dK/dV and dSink (H,) against the sink-aware reference (dSink sums over
    the batch, as the reference's shared sink leaf does)."""
    try:
        from cudnn.sdpa.bwd import sdpa_bwd_wrapper_sm80
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, s, d = 2, 4, 512, 128
    scale = 1.0 / math.sqrt(d)
    q = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    k = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    v = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    do = _bshd_randn(b, h, s, d, dtype=torch.float16, device="cuda")
    sinks = torch.randn(h, dtype=torch.float32, device="cuda")
    fwd = sdpa_fwd_wrapper_sm80(q_tensor=q, k_tensor=k, v_tensor=v, is_causal=True, sinks=sinks, scale_softmax=scale)
    out = sdpa_bwd_wrapper_sm80(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        o_tensor=fwd["o_tensor"],
        do_tensor=do,
        lse_tensor=fwd["lse_tensor"],
        is_causal=True,
        sinks=sinks,
        scale_softmax=scale,
    )
    _, dq_ref, dk_ref, dv_ref, dsink_ref = _ref_grads(q, k, v, do, is_causal=True, window_left=-1, scale=scale, sink=sinks)
    torch.testing.assert_close(out["dq_tensor"].to(torch.float32), dq_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out["dk_tensor"].to(torch.float32), dk_ref, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out["dv_tensor"].to(torch.float32), dv_ref, rtol=3e-2, atol=3e-2)
    assert dsink_ref.abs().max().item() > 0
    torch.testing.assert_close(out["dsink_tensor"], dsink_ref, rtol=3e-2, atol=3e-2 * max(1.0, dsink_ref.abs().max().item()))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sm80_bwd_thd_sinks_deterministic_compile_key():
    """Rule 4 for the two new THD specializations: sinks and deterministic each
    mint exactly one bprop template compile (a new parameter set), and then
    re-bind across different token totals — and, for deterministic, a
    different max_s_q — without a new compile: neither T_q nor the relay
    counter's size is part of the key."""
    from cudnn.frost import template_loader

    def cache_totals():
        """(misses, hits) summed over the loaded bprop template modules' compile caches."""
        modules = [m for (path, _params), m in template_loader._MODULES.items() if "bprop" in str(path)]
        infos = [m.compile.cache_info() for m in modules if hasattr(m.compile, "cache_info")]
        return sum(i.misses for i in infos), sum(i.hits for i in infos)

    sinks = torch.randn(4, dtype=torch.float32, device="cuda")
    try:
        _thd_run_and_check([96, 160], h=4, d_qk=128, d_v=128, sinks=sinks, check_grads=False)
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")
    m0, h0 = cache_totals()
    _thd_run_and_check([64, 256], h=4, d_qk=128, d_v=128, sinks=sinks, check_grads=False)  # same n_seq, other totals
    m1, h1 = cache_totals()
    assert m1 == m0 and h1 > h0, "THD+sinks re-bind must be a pure cache hit"

    _thd_run_and_check([96, 160], h=4, d_qk=128, d_v=128, deterministic=True, check_grads=False, max_s_q=160)
    m2, h2 = cache_totals()
    _thd_run_and_check([64, 256], h=4, d_qk=128, d_v=128, deterministic=True, check_grads=False, max_s_q=512)  # other totals AND max_s_q
    m3, h3 = cache_totals()
    assert m3 == m2 and h3 > h2, "THD+deterministic re-bind (different totals and max_s_q) must be a pure cache hit"
