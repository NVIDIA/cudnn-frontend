# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SM80 (A100) SDPA forward API.

Skipped automatically on non-SM80 devices or when the optional CuTe-DSL
dependency (``nvidia-cutlass-dsl``) is missing.  Compares the kernel's
output against a fp32 torch reference at a per-element fp16 tolerance.
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


def _has_supported_cutedsl() -> bool:
    """Rule 7 (python/cudnn/AGENTS.md): skip below CUTEDSL_MIN_VERSION instead of
    failing inside the DSL when the kernel module loads."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    return bool(installed) and not cutedsl_too_old(version)


pytestmark = [
    pytest.mark.skipif(not _is_sm80(), reason="SM80 SDPA API requires an SM80 (A100) device"),
    pytest.mark.skipif(not _has_supported_cutedsl(), reason="requires nvidia-cutlass-dsl at or above cudnn.frost.buffers.CUTEDSL_MIN_VERSION"),
]


def _bshd_randn(b, h, s, d, **kw):
    """BHSD-logical tensor with the BSHD-physical stride order the SM80
    adapters require."""
    return torch.randn((b, s, h, d), **kw).permute(0, 2, 1, 3)


def _ref_sdpa(q, k, v, *, is_causal, window_size, scale):
    """Reference SDPA in fp32 with the same masking semantics the kernel
    promises (causal / SWA window described by (left, right))."""
    _b, h_q, s_q, _d_qk = q.shape
    _, h_kv, s_kv, _ = k.shape
    _, _, _, _d_v = v.shape
    g = h_q // h_kv
    # Expand K/V across GQA groups for ref.
    k_ref = k.repeat_interleave(g, dim=1).to(torch.float32)
    v_ref = v.repeat_interleave(g, dim=1).to(torch.float32)
    q_ref = q.to(torch.float32)

    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale  # [B,H,Sq,Skv]

    if is_causal:
        # Causal aligns the right edge of Q to the right edge of K.
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        keep = j <= (i + (s_kv - s_q))
        if window_size[0] >= 0:
            keep &= ((i + (s_kv - s_q)) - j) <= window_size[0]
        scores = scores.masked_fill(~keep, float("-inf"))
    elif window_size[0] >= 0:
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        keep = (i - j).abs() <= window_size[0]
        scores = scores.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs, v_ref).to(q.dtype)
    return o


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_smoke():
    """One representative case at L0 (llama flavor, fp16, causal, MHA);
    the full flavor x mask x GQA x dtype sweep runs at L2."""
    test_sdpa_fwd_sm80_wrapper(torch.float16, 128, 128, "causal", (8, 8))


@pytest.mark.L2
@pytest.mark.parametrize("d_qk,d_v", [(64, 64), (128, 128), (192, 128), (256, 256)], ids=["gptoss", "llama", "dsv3", "qwen"])
@pytest.mark.parametrize("mask", ["none", "causal", "swa"])
@pytest.mark.parametrize("gqa", [(8, 8), (16, 4)], ids=["mha", "gqa4x"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_wrapper(dtype, d_qk, d_v, mask, gqa):
    """End-to-end check against torch reference SDPA (full sweep, L2)."""
    try:
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, s_q, s_kv = 2, 1024, 1024
    h_q, h_kv = gqa
    device = "cuda"

    q = _bshd_randn(b, h_q, s_q, d_qk, dtype=dtype, device=device)
    k = _bshd_randn(b, h_kv, s_kv, d_qk, dtype=dtype, device=device)
    v = _bshd_randn(b, h_kv, s_kv, d_v, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(d_qk)

    is_causal = mask == "causal"
    window = (128, 0) if mask == "swa" else (-1, -1)
    if mask == "swa":
        is_causal = True

    try:
        out = sdpa_fwd_wrapper_sm80(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            is_causal=is_causal,
            window_size=window,
            scale_softmax=scale,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    o = out["o_tensor"]
    lse = out["lse_tensor"]

    o_ref = _ref_sdpa(q, k, v, is_causal=is_causal, window_size=window, scale=scale)

    # ~4x FP16 ULP — matches the cudnn-FE upstream SDPA test tolerance.
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=4e-3)
    assert lse.shape == (b, h_q, s_q)
    assert lse.dtype == torch.float32


def _run_sm80_fwd(q, k, v, *, scheduler, is_causal=False, causal_bottom_right=False, window=-1, strided_lse=False, seq_kv_lens=None, sinks=None, bias=None):
    """One SdpaFwdDslSm80 plan + execute into NaN-prefilled O and LSE, so a
    tile no CTA writes stays NaN instead of passing as stale memory."""
    from cudnn.sdpa.fwd import SdpaFwdDslSm80

    b, h_q, s_q, _ = q.shape
    o = torch.full((b, s_q, h_q, v.shape[-1]), float("nan"), dtype=q.dtype, device=q.device).permute(0, 2, 1, 3)
    if strided_lse:
        # One unused float after every row: exercises the stride-aware LSE store.
        lse = torch.full((2 * b * h_q * s_q,), float("nan"), dtype=torch.float32, device=q.device).as_strided((b, h_q, s_q), (2 * h_q * s_q, 2 * s_q, 2))
    else:
        lse = torch.full((b, h_q, s_q), float("nan"), dtype=torch.float32, device=q.device)
    api = SdpaFwdDslSm80(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        is_causal=is_causal,
        causal_bottom_right=causal_bottom_right,
        window_size_left=(window if window >= 0 else None),
        window_size_right=(0 if window >= 0 else None),
        scale_softmax=1.0 / math.sqrt(q.shape[-1]),
        seq_kv_lens_present=seq_kv_lens is not None,
        has_sink=sinks is not None,
        scheduler=scheduler,
        bias_present=bias is not None,
    )
    api.check_support()
    api.compile()
    api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, lse_tensor=lse, sinks=sinks, seq_kv_lens=seq_kv_lens, bias_tensor=bias)
    torch.cuda.synchronize()
    return o, lse


# (id, b, h_q, h_kv, s_q, s_kv, features).  G = 4 with h_kv >= 2 wherever
# possible: MQA maps every Q head to KV head 0, so it cannot catch a wrong
# head mapping on its own.
_GQA_BITWISE_CASES = [
    ("sq1", 2, 8, 2, 1, 1024, {}),
    ("causal_tl_sq2", 2, 8, 2, 2, 1024, {"is_causal": True}),
    ("causal_br_sq256", 2, 8, 2, 256, 1024, {"is_causal": True, "causal_bottom_right": True}),
    ("causal_sq1024", 2, 8, 2, 1024, 1024, {"is_causal": True}),
    ("mqa_causal", 2, 8, 1, 1024, 1024, {"is_causal": True}),
    ("swa", 2, 8, 2, 1024, 1024, {"is_causal": True, "window": 128}),
    ("padded_kv", 2, 8, 2, 256, 1024, {"seq_kv_lens": (1024, 700)}),
    ("sink", 2, 8, 2, 1024, 1024, {"is_causal": True, "sink": True}),
    ("bias", 2, 8, 2, 256, 1024, {"bias": True}),
    ("strided_lse", 2, 8, 2, 1024, 1024, {"is_causal": True, "strided_lse": True}),
    # LPT_L2 short last block (S from _L2_SHORT_LAST_BLOCK_S): 3 KV groups,
    # 2 per block, so the last block holds 1.
    ("l2_short_last_block", 1, 12, 3, None, None, {"is_causal": True}),
]

# SKV at which one K+V group (SKV * 2 * d * 2 bytes) is half the flavor's
# causal L2 budget: llama 16 MiB at d=128, qwen 8 MiB at d=256.
_L2_SHORT_LAST_BLOCK_S = {128: 16384, 256: 4096}


@pytest.mark.L1
@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
@pytest.mark.parametrize("scheduler", ["natural", "lpt", "lpt_l2"])
@pytest.mark.parametrize("case", _GQA_BITWISE_CASES, ids=[c[0] for c in _GQA_BITWISE_CASES])
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_native_gqa_matches_expanded_bitwise(case, scheduler, d):
    """Native dense GQA (K/V bound at H_kv heads; the kernel maps Q head h to
    KV head h // (H_q // H_kv)) must be BITWISE equal to the same inputs with
    K/V pre-expanded to H_q heads and run as MHA: every tile runs the same
    MMA/softmax sequence, only which CTA serves which tile changes.  Every
    scheduler is pinned, since LPT_L2 packs H_q // H_kv Q heads into each
    (batch, KV head) group; d=128 and d=256 cover both kernel files
    (prefill_f16 / prefill_d256_f16)."""
    _, b, h_q, h_kv, s_q, s_kv, feat = case
    if s_q is None:
        s_q = s_kv = _L2_SHORT_LAST_BLOCK_S[d]
    rep = h_q // h_kv
    dev = "cuda"
    q = _bshd_randn(b, h_q, s_q, d, dtype=torch.float16, device=dev)
    k = _bshd_randn(b, h_kv, s_kv, d, dtype=torch.float16, device=dev)
    v = _bshd_randn(b, h_kv, s_kv, d, dtype=torch.float16, device=dev)
    # The MHA twin, same BSHD-physical layout: its head j is KV head j // rep.
    k_mha = k.transpose(1, 2).repeat_interleave(rep, dim=2).transpose(1, 2)
    v_mha = v.transpose(1, 2).repeat_interleave(rep, dim=2).transpose(1, 2)
    kw = dict(
        is_causal=feat.get("is_causal", False),
        causal_bottom_right=feat.get("causal_bottom_right", False),
        window=feat.get("window", -1),
        strided_lse=feat.get("strided_lse", False),
        seq_kv_lens=(torch.tensor(feat["seq_kv_lens"], dtype=torch.int32, device=dev) if "seq_kv_lens" in feat else None),
        sinks=(torch.randn(h_q, dtype=torch.float32, device=dev) if feat.get("sink") else None),
        bias=(torch.randn(1, h_q, s_q, s_kv, dtype=torch.float16, device=dev) if feat.get("bias") else None),
    )
    o_gqa, lse_gqa = _run_sm80_fwd(q, k, v, scheduler=scheduler, **kw)
    o_mha, lse_mha = _run_sm80_fwd(q, k_mha, v_mha, scheduler=scheduler, **kw)

    assert not torch.isnan(o_gqa).any() and not torch.isnan(lse_gqa).any(), "native GQA left O/LSE entries unwritten"
    assert torch.equal(o_gqa, o_mha), f"O differs from the pre-expanded MHA run (max |diff| {(o_gqa.float() - o_mha.float()).abs().max().item()})"
    assert torch.equal(lse_gqa, lse_mha), f"LSE differs from the pre-expanded MHA run (max |diff| {(lse_gqa - lse_mha).abs().max().item()})"


@pytest.mark.L0
@pytest.mark.parametrize("d", [64, 256], ids=["generic-kernel", "d256-kernel"])
@pytest.mark.parametrize("s_q", [128, 100], ids=["tile-aligned", "unaligned"])
def test_sdpa_fwd_sm80_padded_row_lse_trim(d, s_q):
    """Padded query rows (q >= seq_len_q[b]) must read LSE = -inf in BOTH
    store paths: the is_even_mn fast path (tile-aligned S_q) used to skip the
    trim, leaving finite LSE on padded rows — caught by
    test_mhas_v2::test_sdpa_random_bwd_L0 once a random config combined a
    tile-aligned S_q with bottom-right padding."""
    try:
        from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h = 4, 2
    torch.manual_seed(0)
    seq_len_q = torch.tensor([s_q, 3 * s_q // 4, s_q // 2, 5], dtype=torch.int32, device="cuda")
    seq_kv_lens = torch.tensor([s_q // 4, s_q, s_q // 5, 3 * s_q // 5], dtype=torch.int32, device="cuda")

    out = sdpa_fwd_wrapper_sm80(
        _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda"),
        _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda"),
        _bshd_randn(b, h, s_q, d, dtype=torch.float16, device="cuda"),
        scale_softmax=1.0 / math.sqrt(d),
        is_causal=True,
        causal_bottom_right=True,
        seq_kv_lens=seq_kv_lens,
        seq_len_q=seq_len_q,
    )
    lse = out["lse_tensor"]
    for bi in range(b):
        tail = lse[bi, :, seq_len_q[bi] :]
        assert torch.isinf(tail).all() and (tail < 0).all(), f"batch {bi}: padded rows lack the -inf LSE trim"
        head = lse[bi, :, : seq_len_q[bi]]
        assert torch.isfinite(head).any(), f"batch {bi}: valid rows unexpectedly all -inf"


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_check_support_rejections():
    """check_support rejects out-of-envelope heads and bad layouts eagerly."""
    try:
        from cudnn.sdpa.fwd import SdpaFwdDslSm80
    except ImportError as e:
        pytest.skip(f"SM80 SDPA API not available: {e}")

    b, h, s = 1, 4, 128
    q = torch.zeros((b, s, h, 512), dtype=torch.float16, device="cuda").permute(0, 2, 1, 3)
    lse = torch.zeros((b, h, s), dtype=torch.float32, device="cuda")
    api = SdpaFwdDslSm80(sample_q=q, sample_k=q, sample_v=q, sample_o=q, sample_lse=lse)
    with pytest.raises(ValueError, match="exceeds"):
        api.check_support()

    q64 = torch.zeros((b, s, h, 64), dtype=torch.float32, device="cuda").permute(0, 2, 1, 3)
    lse64 = torch.zeros((b, h, s), dtype=torch.float32, device="cuda")
    api = SdpaFwdDslSm80(sample_q=q64, sample_k=q64, sample_v=q64, sample_o=q64, sample_lse=lse64)
    with pytest.raises(ValueError, match="dtype"):
        api.check_support()


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_sm80_thd_off_flavor_head_dim():
    """Off-flavor THD head dim (d=96 rides the llama d=128 envelope): Q/K are
    host-padded to the flavor width, and the kernel derives its Q/K row
    strides from the runtime ``d`` — so the launch must pass the PADDED
    width, or every row past the first reads the wrong address."""
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80

    H, D = 4, 96
    lens = [80, 33]
    t = sum(lens)
    cu = torch.tensor([0, lens[0], t], dtype=torch.int32, device="cuda")
    q = torch.randn(1, t, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(1, t, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(1, t, H, D, dtype=torch.float16, device="cuda")
    scale = 1.0 / math.sqrt(D)
    out = sdpa_fwd_wrapper_sm80(
        q,
        k,
        v,
        is_causal=True,
        scale_softmax=scale,
        cum_seqlen_q_tensor=cu,
        cum_seqlen_k_tensor=cu,
        max_s_q=max(lens),
    )
    o = out["o_tensor"]
    for b0, b1 in zip(cu[:-1].tolist(), cu[1:].tolist()):
        o_ref = _ref_sdpa(
            q[0, b0:b1].permute(1, 0, 2)[None],
            k[0, b0:b1].permute(1, 0, 2)[None],
            v[0, b0:b1].permute(1, 0, 2)[None],
            is_causal=True,
            window_size=(-1, -1),
            scale=scale,
        )
        torch.testing.assert_close(o[0, b0:b1].permute(1, 0, 2)[None], o_ref, rtol=1e-2, atol=4e-3)


@pytest.mark.L1
@torch_fork_set_rng(seed=0)
def test_sm80_thd_compile_key_plan_time_only():
    """Issue #604 regression: the packed THD token totals are RUNTIME values,
    so two varlen calls with different totals must re-bind ONE compiled
    artifact (the template module's per-shape lru sees a single miss) —
    never mint a compile per step, which is the continuous-batching
    pathology no correctness test catches."""
    from cudnn.frost import template_loader
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_sm80

    H, D = 4, 128

    def varlen(lens):
        import itertools

        t = int(sum(lens))
        cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device="cuda")
        q = torch.randn(1, t, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, t, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, t, H, D, dtype=torch.float16, device="cuda")
        return sdpa_fwd_wrapper_sm80(
            q,
            k,
            v,
            is_causal=True,
            cum_seqlen_q_tensor=cu,
            cum_seqlen_k_tensor=cu,
            max_s_q=int(max(lens)),
        )

    def cache_totals():
        # The lru counters are session-global (earlier tests in a full run
        # accumulate misses), so assert on DELTAS across our calls only.
        modules = [m for (path, _params), m in template_loader._MODULES.items() if "sm80" in str(path)]
        infos = [m.compile.cache_info() for m in modules if hasattr(m.compile, "cache_info")]
        return sum(i.misses for i in infos), sum(i.hits for i in infos)

    varlen([96, 160])  # first call: one compile
    n_modules_before = len(template_loader._MODULES)
    misses_0, hits_0 = cache_totals()
    varlen([128, 64, 320])  # different totals AND batch count... same artifact?
    # Different logical batch counts legitimately re-specialize (the cu fake
    # length is plan-time); different TOKEN TOTALS at the same batch count
    # must not.
    varlen([64, 192])  # same n_seqs as call 1, different totals
    assert len(template_loader._MODULES) == n_modules_before, "a new template specialization was minted by runtime data"
    misses_1, hits_1 = cache_totals()
    # Call 2 (n_seqs=3) may legitimately re-specialize once; call 3 shares
    # call 1's key (n_seqs=2, different token totals) and MUST cache-hit.
    assert misses_1 - misses_0 <= 1, f"THD compile key leaked runtime data: {misses_1 - misses_0} new misses"
    assert hits_1 - hits_0 >= 1, "expected a cache hit on the same-batch-count re-call"
