# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical layouts and device-read-free execute for the BF16 grouped GEMM backends.

Canonical operands -- A/C/D (m, x), prob/dprob (m,), dense B (experts, n, k) -- must give
bit-identical results to the legacy kernel-facing views, and a warm execute with fresh
offset / pointer tensors must never read device memory (python/cudnn/AGENTS.md Rule 3).
"""

import pytest
import torch

pytestmark = pytest.mark.L0

# 256-aligned per-expert token counts, including an empty expert.
SPLITS = (512, 0, 256, 768)


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def _offsets(splits=SPLITS):
    return torch.tensor(splits, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)


def _rand(generator, *shape, scale=0.125):
    return torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).mul_(scale).to(torch.bfloat16)


def _legacy_mx(tensor):
    """Canonical (m, x) -> the legacy kernel-facing (m, x, 1) view."""
    return tensor.unsqueeze(0).permute(1, 2, 0)


def _ptrs(storage):
    return torch.tensor([storage[e].data_ptr() for e in range(storage.shape[0])], dtype=torch.int64, device="cuda")


def _problem(m, k, n, seed, b_major="k"):
    """A (m, k), per-expert B storage, prob (m,) and alpha. ``b_major="k"`` stores each expert as
    (n, k) row-major; ``"n"`` stores it as (k, n) row-major (TE's dgrad orientation)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    experts = len(SPLITS)
    storage = _rand(generator, experts, n, k) if b_major == "k" else _rand(generator, experts, k, n)
    return {
        "a": _rand(generator, m, k),
        "storage": storage,
        "b_ptrs": _ptrs(storage),
        "prob": torch.linspace(0.25, 0.875, m, dtype=torch.float32, device="cuda"),
        "alpha": torch.tensor([0.75, -1.25, 1.0, 0.5], dtype=torch.float32, device="cuda"),
        "offsets": _offsets(),
    }


@pytest.mark.parametrize("act_func", ["swiglu", "geglu"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_glu_bf16_canonical_matches_legacy(discrete, act_func):
    from cudnn import grouped_gemm_glu_wrapper_sm100 as glu

    m, k, n = sum(SPLITS), 256, 512
    p = _problem(m, k, n, seed=11)
    common = dict(
        sfa_tensor=None,
        padded_offsets=p["offsets"],
        alpha_tensor=p["alpha"],
        acc_dtype=torch.float32,
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        act_func=act_func,
        generate_c=True,
        use_dynamic_sched=True,
    )
    if discrete:
        weights = lambda canonical: dict(b_ptrs=p["b_ptrs"], n=n, b_dtype=torch.bfloat16, b_major="k")
    else:
        weights = lambda canonical: dict(b_tensor=p["storage"] if canonical else p["storage"].permute(1, 2, 0))
    legacy = glu(a_tensor=_legacy_mx(p["a"]), prob_tensor=p["prob"].view(m, 1, 1), **common, **weights(False))
    cold = glu(a_tensor=p["a"], prob_tensor=p["prob"], **common, **weights(True))
    warm = glu(a_tensor=p["a"], prob_tensor=p["prob"], **common, **weights(True))
    for out in (cold, warm):
        assert out["c_tensor"].shape == (m, n) and out["d_tensor"].shape == (m, n // 2)
        assert torch.equal(out["c_tensor"], legacy["c_tensor"].squeeze(-1))
        assert torch.equal(out["d_tensor"], legacy["d_tensor"].squeeze(-1))


@pytest.mark.parametrize("weights", ["dense-k", "discrete-k", "discrete-n"])
@pytest.mark.parametrize("supplied_d", [False, True], ids=["alloc-d", "caller-d"])
def test_gemm_bf16_canonical_matches_legacy_and_reference(weights, supplied_d):
    from cudnn import grouped_gemm_wrapper_sm100 as gemm

    m, k, n = sum(SPLITS), 512, 256
    b_major = weights[-1]
    p = _problem(m, k, n, seed=12, b_major=b_major)
    if weights == "dense-k":
        b = lambda canonical: dict(b_tensor=p["storage"] if canonical else p["storage"].permute(1, 2, 0))
    else:
        b = lambda canonical: dict(b_ptrs=p["b_ptrs"], n=n, b_dtype=torch.bfloat16, b_major=b_major)
    common = dict(padded_offsets=p["offsets"], alpha_tensor=p["alpha"], acc_dtype=torch.float32, use_dynamic_sched=True)
    legacy = gemm(a_tensor=_legacy_mx(p["a"]), prob_tensor=p["prob"].view(m, 1, 1), **common, **b(False))["d_tensor"]
    for _ in range(2):  # cold, then the memoized wrapper path
        d_out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda") if supplied_d else None
        out = gemm(a_tensor=p["a"], prob_tensor=p["prob"], d_tensor=d_out, **common, **b(True))["d_tensor"]
        assert out.shape == (m, n)
        if supplied_d:
            assert out.data_ptr() == d_out.data_ptr()
        assert torch.equal(out, legacy.squeeze(-1))

    # Independent reference: N-major storage holds each expert as (k, n), K-major as (n, k).
    ref = torch.empty(m, n, dtype=torch.float32, device="cuda")
    begin = 0
    for e, end in enumerate(p["offsets"].tolist()):
        w = p["storage"][e].float()
        w_kn = w if b_major == "n" else w.t()
        ref[begin:end] = (p["a"][begin:end].float() @ w_kn) * p["alpha"][e] * p["prob"][begin:end, None]
        begin = end
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_dglu_bf16_canonical_matches_legacy():
    from cudnn import grouped_gemm_dglu_wrapper_sm100 as dglu

    # TE's FC2 dgrad: A = dY (m, k=hidden), weights (hidden, ffn) read N-major, C = (m, 2*ffn).
    m, k, n = sum(SPLITS), 256, 256
    p = _problem(m, k, n, seed=13, b_major="n")
    generator = torch.Generator(device="cuda").manual_seed(14)
    c = _rand(generator, m, 2 * n, scale=0.5)
    common = dict(
        sfa_tensor=None,
        padded_offsets=p["offsets"],
        alpha_tensor=p["alpha"],
        beta_tensor=torch.ones(len(SPLITS), dtype=torch.float32, device="cuda"),
        b_ptrs=p["b_ptrs"],
        n=n,
        b_dtype=torch.bfloat16,
        b_major="n",
        acc_dtype=torch.float32,
        d_dtype=torch.bfloat16,
        act_func="dswiglu",
        use_dynamic_sched=True,
    )
    legacy = dglu(
        a_tensor=_legacy_mx(p["a"]),
        c_tensor=_legacy_mx(c),
        prob_tensor=p["prob"].view(m, 1, 1),
        dprob_tensor=torch.zeros(m, 1, 1, dtype=torch.float32, device="cuda"),
        **common,
    )
    for _ in range(2):
        out = dglu(a_tensor=p["a"], c_tensor=c, prob_tensor=p["prob"], dprob_tensor=torch.zeros(m, dtype=torch.float32, device="cuda"), **common)
        assert out["d_row_tensor"].shape == (m, 2 * n) and out["dprob_tensor"].shape == (m,)
        assert torch.equal(out["d_row_tensor"], legacy["d_row_tensor"].squeeze(-1))
        torch.testing.assert_close(out["dprob_tensor"], legacy["dprob_tensor"].view(-1), rtol=1e-5, atol=1e-6)


def _fresh_calls(canonical):
    """Wrapper calls as TE issues them, each taking freshly allocated offsets / pointer tables.

    ``canonical=False`` uses the legacy kernel-facing views, which develop also accepts, so the
    same check runs against FE versions without canonical BF16 support."""
    from cudnn import grouped_gemm_dglu_wrapper_sm100, grouped_gemm_glu_wrapper_sm100, grouped_gemm_wgrad_wrapper_sm100, grouped_gemm_wrapper_sm100

    m, k, n = sum(SPLITS), 256, 256
    p = _problem(m, k, n, seed=15, b_major="k")
    alpha = p["alpha"]
    c = torch.zeros(m, 2 * n, dtype=torch.bfloat16, device="cuda")
    dy = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    wgrad = torch.empty(len(SPLITS), n, k, dtype=torch.bfloat16, device="cuda")
    mx = (lambda t: t) if canonical else _legacy_mx
    a, cc, prob = mx(p["a"]), mx(c), p["prob"] if canonical else p["prob"].view(m, 1, 1)
    dprob_shape = (m,) if canonical else (m, 1, 1)
    calls = {
        "glu": lambda offsets, ptrs: grouped_gemm_glu_wrapper_sm100(
            a_tensor=a,
            sfa_tensor=None,
            padded_offsets=offsets,
            alpha_tensor=alpha,
            b_ptrs=ptrs,
            n=2 * n,
            b_dtype=torch.bfloat16,
            prob_tensor=prob,
            acc_dtype=torch.float32,
            c_dtype=torch.bfloat16,
            d_dtype=torch.bfloat16,
            generate_c=True,
            use_dynamic_sched=True,
        ),
        "gemm": lambda offsets, ptrs: grouped_gemm_wrapper_sm100(
            a_tensor=a,
            padded_offsets=offsets,
            alpha_tensor=alpha,
            b_ptrs=ptrs,
            n=n,
            b_dtype=torch.bfloat16,
            prob_tensor=prob,
            acc_dtype=torch.float32,
            use_dynamic_sched=True,
        ),
        "dglu": lambda offsets, ptrs: grouped_gemm_dglu_wrapper_sm100(
            a_tensor=a,
            c_tensor=cc,
            sfa_tensor=None,
            padded_offsets=offsets,
            alpha_tensor=alpha,
            beta_tensor=alpha,
            prob_tensor=prob,
            dprob_tensor=torch.zeros(dprob_shape, dtype=torch.float32, device="cuda"),
            b_ptrs=ptrs,
            n=n,
            b_dtype=torch.bfloat16,
            b_major="k",
            acc_dtype=torch.float32,
            d_dtype=torch.bfloat16,
            act_func="dswiglu",
            use_dynamic_sched=True,
        ),
        "wgrad": lambda offsets, ptrs: grouped_gemm_wgrad_wrapper_sm100(
            a_tensor=dy.T,
            b_tensor=x,
            sfa_tensor=None,
            sfb_tensor=None,
            offsets_tensor=offsets,
            output_mode="dense",
            wgrad_tensor=wgrad,
            acc_dtype=torch.float32,
        ),
    }
    # GLU reads 2n-row weights; the others n-row ones. Pointer values only need to be valid.
    glu_storage = torch.zeros(len(SPLITS), 2 * n, k, dtype=torch.bfloat16, device="cuda")
    tables = {"glu": _ptrs(glu_storage), "gemm": p["b_ptrs"], "dglu": p["b_ptrs"], "wgrad": None}
    return calls, tables, p["offsets"], glu_storage


@pytest.mark.parametrize("layout", ["canonical", "legacy"])
@pytest.mark.parametrize("op", ["glu", "gemm", "dglu", "wgrad"])
def test_bf16_warm_execute_never_reads_device_values(op, layout):
    """Fresh offset / pointer tensors on every call (as TE passes them) must not trigger a
    blocking device-to-host read; the identity-keyed validation caches used to miss here."""
    calls, tables, offsets, _keepalive = _fresh_calls(layout == "canonical")
    call, table = calls[op], tables[op]
    call(offsets.clone(), None if table is None else table.clone())  # compile + warm the memo
    torch.cuda.synchronize()
    fresh = [(offsets.clone(), None if table is None else table.clone()) for _ in range(3)]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for args in fresh:
            call(*args)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_bf16_api_rejects_layout_other_than_compiled():
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import GroupedGemmBf16API

    m, k, n = sum(SPLITS), 256, 256
    p = _problem(m, k, n, seed=16)
    c = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    d = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    api = GroupedGemmBf16API(
        sample_a=p["a"],
        sample_c=c,
        sample_d=d,
        sample_padded_offsets=p["offsets"],
        sample_alpha=p["alpha"],
        sample_b=p["storage"],
        sample_prob=p["prob"],
        use_dynamic_sched=True,
    )
    assert api.check_support()
    api.compile()
    args = dict(padded_offsets=p["offsets"], alpha_tensor=p["alpha"], b_tensor=p["storage"])
    api.execute(a_tensor=p["a"], c_tensor=c, d_tensor=d, prob_tensor=p["prob"], **args)
    with pytest.raises(ValueError, match="canonical"):
        api.execute(a_tensor=_legacy_mx(p["a"]), c_tensor=c, d_tensor=d, prob_tensor=p["prob"], **args)
    with pytest.raises(ValueError, match="canonical"):
        api.execute(a_tensor=p["a"], c_tensor=c, d_tensor=d, prob_tensor=p["prob"].view(m, 1, 1), **args)
