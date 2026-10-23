# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 dense indexer scores (``out_dtype=torch.bfloat16`` / a BF16 ``out``).

The SM100-family scorer keeps the FP32 accumulation and the FP32 weighted head reduction and
rounds each finished score once as it stores it. So a BF16 score must be bit-identical to the
FP32 score of the same call cast with ``.to(torch.bfloat16)`` (round-to-nearest-even), for both
scorer precisions, both layouts and every epilogue variant. The other tests pin the per-dtype
output contract (rows of whole 16 bytes: ``ceil8(S_k)`` for BF16), the declines, and what the
rounding means for a following top-k: the BF16 selection differs from the FP32 selection only
by keys whose BF16 score equals the BF16 cutoff value.
"""

import inspect

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import (
    expand_mxfp8_scale,
    make_random_mxfp8_scale,
    pack_mxfp8_scales_thd,
    quantize_mxfp8,
)
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    check_ref_indexer_forward,
    check_ref_indexer_top_k,
)

# |round_bf16(x) - x| <= 2**-9 * |x| for round-to-nearest-even (8 significant bits); the bound
# below is one full BF16 ulp so the reference's own FP32/FP64 difference never decides it.
_BF16_RTOL = 2.0**-8


def _import_dsa():
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    return DSA


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("BF16 indexer scores are an SM100-family (SM100/SM103) feature")


def _allocation_count() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocation.all.allocated"]


def _assert_bits_equal(bf16_scores: torch.Tensor, fp32_scores: torch.Tensor) -> None:
    """BF16 output == RNE cast of the FP32 output, compared as bit patterns (signed zeros, -inf)."""
    assert bf16_scores.dtype == torch.bfloat16 and fp32_scores.dtype == torch.float32
    assert bf16_scores.shape == fp32_scores.shape
    expected = fp32_scores.to(torch.bfloat16)
    mismatch = bf16_scores.view(torch.int16) != expected.view(torch.int16)
    assert not bool(mismatch.any()), (
        f"{int(mismatch.sum())} of {mismatch.numel()} BF16 scores are not the round-to-nearest-even "
        f"cast of the FP32 scores, e.g. at {mismatch.nonzero()[0].tolist()}"
    )


def _cu(lengths, device):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)


def _last_row_offset(s_q, s_k, ratio):
    """Causal offset that puts a segment's last query token on its last compressed key."""
    return max(s_k * ratio - s_q, 0)


@pytest.mark.L0
def test_out_dtype_is_appended_last():
    """``out_dtype`` is a new trailing parameter (append-only public signatures); the fused top-k path does not take it."""
    DSA = _import_dsa()
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface

    for fn in (DSA.indexer_forward_wrapper, _interface.indexer_fwd):
        params = inspect.signature(fn).parameters
        assert tuple(params)[-1] == "out_dtype", f"{fn.__qualname__}: out_dtype must be the last parameter"
        assert params["out_dtype"].default is None
    assert "out_dtype" not in inspect.signature(DSA.indexer_forward_top_k_wrapper).parameters


@pytest.mark.L0
def test_sm90_declines_bf16_scores_before_launch(monkeypatch):
    """Hopper writes FP32 scores only: BF16 is a NotImplementedError raised before the SM90 interface runs."""
    _import_dsa()
    from cudnn.deepseek_sparse_attention.indexer_forward import api

    def must_not_run(*args, **kwargs):
        raise AssertionError("the SM90 interface must not be reached for BF16 scores")

    monkeypatch.setattr(api, "device_major", lambda *args, **kwargs: 9)
    monkeypatch.setattr(api, "indexer_fwd_sm90", must_not_run)
    q = torch.zeros(1, 8, 32, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16)
    w = torch.zeros(1, 8, 32, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="float32 scores only"):
        api.indexer_forward_wrapper(q, k, w, ratio=1, out_dtype=torch.bfloat16)
    out = torch.empty(1, 8, 8, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="float32 scores only"):
        api.indexer_forward_wrapper(q, k, w, ratio=1, out=out)


def _bshd_inputs(precision, weight_dtype, b, s_q, s_k, h_q, device):
    d = 128
    if precision == "bf16":
        q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
        k = torch.randn(b, s_k, 1, d, dtype=torch.bfloat16, device=device)
        w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)
        if weight_dtype == torch.float32:
            w = (w.abs() + 1).float() + 2**-10
        return dict(q=q, k=k, w=w), (q, k, w), {}

    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    q_ref = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(b, s_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale((b, s_q, h_q, d // 32), device=device, seed=131, exponent_min=-2, exponent_max=3)
    k_scale_logical = make_random_mxfp8_scale((b, s_k, 1, d // 32), device=device, seed=137, exponent_min=-2, exponent_max=3)
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    extra = dict(
        precision="mxfp8",
        q_scale=scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q),
        k_scale=scale_utils.pack_k_scale_bshd(k_scale_logical),
    )
    return dict(q=q, k=k, w=w), (q_deq, k_deq, w), extra


def _thd_inputs(precision, weight_dtype, shapes, h_q, device):
    d = 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q, cu_k = _cu(q_lengths, device), _cu(k_lengths, device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    thd = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths))
    if precision == "bf16":
        q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
        k = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device)
        w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device)
        if weight_dtype == torch.float32:
            w = (w.abs() + 1).float() + 2**-10
        return dict(q=q, k=k, w=w), (q, k, w), thd

    q_ref = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale((total_q, h_q, d // 32), device=device, seed=139, exponent_min=-2, exponent_max=3)
    k_scale_logical = make_random_mxfp8_scale((total_k, 1, d // 32), device=device, seed=149, exponent_min=-2, exponent_max=3)
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale, k_scale, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(q_scale_logical, k_scale_logical, cu_q, cu_k, h_q)
    thd.update(
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
    )
    return dict(q=q, k=k, w=w), (q_deq, k_deq, w), thd


# One case per (scorer, layout) plus every epilogue variant: the BF16 kernel has one dense epilogue for all head
# groups; the MXFP8 kernel has three (64 heads: one token per warpgroup, 32 heads: token pairs, 4/8/16: grouped).
# S_k is chosen so that the BF16 rows need padding the FP32 rows do not (1004 = 4 * 251) or both need it.
_ROUNDING_CASES = [
    pytest.param("bshd", "bf16", torch.bfloat16, 32, 4, id="bshd-bf16-w_bf16-h32-r4"),
    pytest.param("bshd", "bf16", torch.float32, 64, 1, id="bshd-bf16-w_fp32-h64-r1"),
    pytest.param("bshd", "bf16", torch.bfloat16, 16, 4, id="bshd-bf16-w_bf16-h16-r4"),
    pytest.param("thd", "bf16", torch.bfloat16, 32, 1, id="thd-bf16-w_bf16-h32-r1"),
    pytest.param("thd", "bf16", torch.float32, 64, 4, id="thd-bf16-w_fp32-h64-r4"),
    pytest.param("bshd", "mxfp8", torch.bfloat16, 64, 4, id="bshd-mxfp8-h64-r4"),
    pytest.param("bshd", "mxfp8", torch.bfloat16, 8, 1, id="bshd-mxfp8-h8-r1"),
    pytest.param("thd", "mxfp8", torch.bfloat16, 32, 1, id="thd-mxfp8-h32-r1"),
]


@pytest.mark.L0
@torch_fork_set_rng(seed=41)
@pytest.mark.parametrize("layout,precision,weight_dtype,h_q,ratio", _ROUNDING_CASES)
def test_bf16_scores_are_the_rounded_fp32_scores(layout, precision, weight_dtype, h_q, ratio):
    """Same call twice, FP32 then BF16: the BF16 scores are bit for bit ``fp32_scores.to(bfloat16)``,
    keep the ``-inf`` mask, and sit within one BF16 ulp of an FP64 reference."""
    DSA = _import_dsa()
    _require_sm100()
    device = torch.device("cuda")
    atol, rtol = (5e-3, 5e-3) if precision == "mxfp8" else (1e-4, 1e-4)

    if layout == "bshd":
        b, s_q, s_k = 2, 300, 1004 if ratio == 1 else 251
        offsets = torch.tensor([0, s_k * ratio - s_q], dtype=torch.int32, device=device)
        args, ref_args, extra = _bshd_inputs(precision, weight_dtype, b, s_q, s_k, h_q, device)
        common = dict(ratio=ratio, qhead_per_kv_head=h_q, q_causal_offsets=offsets, **extra)
        scores32 = DSA.indexer_forward_wrapper(**args, **common)["scores"]
        scores16 = DSA.indexer_forward_wrapper(**args, **common, out_dtype=torch.bfloat16)["scores"]
        torch.cuda.synchronize()
        assert scores32.dtype == torch.float32 and scores16.dtype == torch.bfloat16
        assert scores16.shape == (b, s_q, s_k) and scores16.is_contiguous()
        _assert_bits_equal(scores16, scores32)
        check_ref_indexer_forward(*ref_args, scores16.float(), ratio, q_causal_offsets=offsets, atol=atol, rtol=rtol + _BF16_RTOL, compute_dtype=torch.float64)
        return

    # THD: ragged segments whose K tails end short of max_seqlen_k; ratio 1 needs BF16-only row padding
    # (300 = 4 * 75), ratio 4 has a segment whose rows see no complete compressed block.
    shapes = [(3, 7), (130, 135), (257, 300)] if ratio == 1 else [(3, 1), (130, 35), (257, 70)]
    offsets = torch.tensor([_last_row_offset(s_q, s_k, ratio) for s_q, s_k in shapes], dtype=torch.int32, device=device)
    args, ref_args, extra = _thd_inputs(precision, weight_dtype, shapes, h_q, device)
    common = dict(ratio=ratio, qhead_per_kv_head=h_q, q_causal_offsets=offsets, **extra)
    scores32 = DSA.indexer_forward_wrapper(**args, **common)["scores"]
    scores16 = DSA.indexer_forward_wrapper(**args, **common, out_dtype=torch.bfloat16)["scores"]
    torch.cuda.synchronize()
    max_k = max(s_k for _, s_k in shapes)
    assert scores16.shape == (int(extra["cu_seqlens_q"][-1]), max_k) and scores16.is_contiguous()
    _assert_bits_equal(scores16, scores32)
    q_ref, k_ref, w_ref = ref_args
    cu_q, cu_k = extra["cu_seqlens_q"].tolist(), extra["cu_seqlens_k"].tolist()
    for seg, (_, s_k) in enumerate(shapes):
        q0, q1, k0, k1 = cu_q[seg], cu_q[seg + 1], cu_k[seg], cu_k[seg + 1]
        check_ref_indexer_forward(
            q_ref[q0:q1].unsqueeze(0),
            k_ref[k0:k1].unsqueeze(0),
            w_ref[q0:q1].unsqueeze(0),
            scores16[q0:q1, :s_k].float().unsqueeze(0),
            ratio,
            q_causal_offsets=offsets[seg : seg + 1],
            atol=atol,
            rtol=rtol + _BF16_RTOL,
            compute_dtype=torch.float64,
        )
        if s_k < max_k:
            assert bool(torch.isneginf(scores16[q0:q1, s_k:]).all())


@pytest.mark.L0
@torch_fork_set_rng(seed=43)
def test_indexer_forward_class_bf16_out(compile_allocates_nothing):
    """``IndexerForward`` with a BF16 ``sample_out``: rows padded to ``ceil8(S_k)`` (``ceil4`` is declined), the
    ``[..., :S_k]`` view bound directly (pad columns never written), no allocation across executes, the output dtype
    in the compile key, a mismatched ``out`` rejected, CUDA-graph replay, and scores equal to the FP32 plan's, rounded."""
    DSA = _import_dsa()
    _require_sm100()
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface as dense_impl

    b, s_q, s_k, ratio, h_q = 2, 1000, 1001, 1, 32
    q = torch.randn(b, s_q, h_q, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s_k, 1, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device="cuda")

    out32 = torch.empty(b, s_q, 1004, dtype=torch.float32, device="cuda")
    api32 = DSA.IndexerForward(q, k, w, out32, ratio=ratio)
    assert api32.check_support()
    api32.compile()
    api32.execute(q, k, w, out32)
    keys_before = set(dense_impl._compile_cache)

    # FP32 padding (1004 columns) leaves 2008-byte BF16 rows: declined.
    with pytest.raises(ValueError, match="multiple of 8"):
        DSA.IndexerForward(q, k, w, torch.empty(b, s_q, 1004, dtype=torch.bfloat16, device="cuda"), ratio=ratio).check_support()

    sentinel = 4242.0  # exact in BF16
    out16 = torch.full((b, s_q, 1008), sentinel, dtype=torch.bfloat16, device="cuda")
    api16 = DSA.IndexerForward(q, k, w, out16, ratio=ratio)
    assert api16.check_support()
    compile_allocates_nothing(api16)
    api16.execute(q, k, w, out16)  # compiles on the first execute
    new_keys = set(dense_impl._compile_cache) - keys_before
    assert len(new_keys) == 1, f"the BF16 output must add exactly one compile-cache entry, got {new_keys}"
    (key16,) = new_keys
    fp32_twins = [
        key
        for key in dense_impl._compile_cache
        if len(key) == len(key16) and [(a, b) for a, b in zip(key, key16) if a != b] == [(torch.float32, torch.bfloat16)]
    ]
    assert len(fp32_twins) == 1, f"the BF16 key must differ from the FP32 plan's key in the output dtype only: {key16}"

    before = _allocation_count()
    for _ in range(3):
        api16.execute(q, k, w, out16)
    assert _allocation_count() == before
    torch.cuda.synchronize()
    _assert_bits_equal(out16[..., :s_k], out32[..., :s_k])
    assert torch.equal(out16[..., s_k:], torch.full_like(out16[..., s_k:], sentinel)), "pad columns must never be written"

    with pytest.raises(ValueError):
        api16.execute(q, k, w, out32)  # validate, never convert (Rule 1)

    # Graph capture and replay with new inputs (the -inf fill and the kernel are both captured).
    q_static, w_static = q.clone(), w.clone()
    api16.execute(q_static, k, w_static, out16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api16.execute(q_static, k, w_static, out16)
    q_static.copy_(torch.randn_like(q))
    w_static.copy_(torch.randn_like(w))
    graph.replay()
    api32.execute(q_static, k, w_static, out32)
    torch.cuda.synchronize()
    _assert_bits_equal(out16[..., :s_k], out32[..., :s_k])


@pytest.mark.L0
@torch_fork_set_rng(seed=47)
def test_wrapper_bf16_out_layouts_and_declines():
    """Torch-op contract for BF16: a padded-stride ``out`` view is bound directly, any other ``out`` layout is staged and
    copied back (identity kept), and the strict entry refuses BF16 rows that are not whole 16 bytes. Mismatched or
    unsupported dtypes and BF16 on the compressed-logits path raise ``ValueError``; the default stays FP32."""
    DSA = _import_dsa()
    _require_sm100()
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface as dense_impl

    b, s_q, s_k, ratio, h_q = 1, 64, 1001, 1, 32
    q = torch.randn(b, s_q, h_q, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s_k, 1, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device="cuda")

    default = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio)["scores"]
    assert default.dtype == torch.float32
    reference = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out_dtype=torch.bfloat16)["scores"]
    assert reference.dtype == torch.bfloat16 and reference.is_contiguous()
    _assert_bits_equal(reference, default)

    # A padded-stride BF16 view (rows of 1008 = ceil8(1001)) is bound directly: same storage, pad untouched.
    sentinel = -7.0
    buf = torch.full((b, s_q, 1008), sentinel, dtype=torch.bfloat16, device="cuda")
    view = buf[..., :s_k]
    got = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=view)["scores"]
    assert got.data_ptr() == view.data_ptr()
    assert torch.equal(buf[..., :s_k].view(torch.int16), reference.view(torch.int16))
    assert torch.equal(buf[..., s_k:], torch.full_like(buf[..., s_k:], sentinel))

    # A contiguous BF16 out of the logical shape (2002-byte rows) is staged and copied back; out_dtype may be omitted.
    plain = torch.empty(b, s_q, s_k, dtype=torch.bfloat16, device="cuda")
    got = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=plain)["scores"]
    assert got is plain and torch.equal(plain.view(torch.int16), reference.view(torch.int16))
    with pytest.raises(ValueError, match="16-byte-aligned"):
        dense_impl._indexer_fwd_bound(q, k, w, ratio=ratio, out=torch.empty(b, s_q, 1004, dtype=torch.bfloat16, device="cuda")[..., :s_k])

    with pytest.raises(ValueError, match="out_dtype"):
        DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=torch.empty(b, s_q, s_k, device="cuda"), out_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="float32 or torch.bfloat16"):
        DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out_dtype=torch.float16)
    with pytest.raises(ValueError, match="compressed|candidate"):
        dense_impl.indexer_fwd(q, k, w, ratio=ratio, is_compressed_logits=True, topk=64, out_dtype=torch.bfloat16)


def _causal_row_lengths(cu_q, offsets, ratio, k_lengths):
    """Per-row ratio-causal key count, clamped to the segment's key length (the top-k row length)."""
    lengths = []
    for seg in range(len(k_lengths)):
        rows = torch.arange(cu_q[seg + 1] - cu_q[seg], dtype=torch.int64)
        lengths.append(torch.div(offsets[seg] + rows + 1, ratio, rounding_mode="floor").clamp(0, k_lengths[seg]))
    return torch.cat(lengths).to(torch.int32)


@pytest.mark.L0
@torch_fork_set_rng(seed=53)
@pytest.mark.parametrize("top_k", [512, 2048])
def test_bf16_top_k_differs_from_fp32_only_by_cutoff_ties(top_k):
    """GLM-5.2-like THD geometry (32 heads, ratio 1, causal offsets, rows longer and shorter than ``top_k``).
    FE ``indexer_top_k`` on the BF16 scores is exact for its input, and every key on which the BF16 and FP32
    selections disagree has a BF16 score equal to the row's BF16 cutoff: RNE rounding is monotonic, so the
    only selections it can change are among keys it rounds onto the cutoff value."""
    DSA = _import_dsa()
    _require_sm100()
    device = torch.device("cuda")
    h_q, ratio = 32, 1
    # (query rows, key length): a long-context tail of 256 rows over 32K keys, a short segment, a mid segment.
    shapes = [(256, 32768), (96, 1500), (160, 9000)]
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    offsets_host = [s_k - s_q for s_q, s_k in shapes]
    cu_q, cu_k = _cu(q_lengths, device), _cu(k_lengths, device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, h_q, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, 1, 128, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device)
    common = dict(
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        q_causal_offsets=torch.tensor(offsets_host, dtype=torch.int32, device=device),
    )
    scores32 = DSA.indexer_forward_wrapper(q, k, w, **common)["scores"]
    scores16 = DSA.indexer_forward_wrapper(q, k, w, **common, out_dtype=torch.bfloat16)["scores"]
    _assert_bits_equal(scores16, scores32)

    row_lens = _causal_row_lengths(cu_q.tolist(), offsets_host, ratio, k_lengths).to(device)
    sel32 = DSA.indexer_top_k_wrapper(scores32, row_lens, top_k, tie_break=1)
    sel16 = DSA.indexer_top_k_wrapper(scores16, row_lens, top_k, tie_break=1)
    assert sel16["values"].dtype == torch.bfloat16
    check_ref_indexer_top_k(scores16, row_lens, top_k, 1, sel16["indices"], sel16["values"], return_val=True, tie_break=1)
    check_ref_indexer_top_k(scores32, row_lens, top_k, 1, sel32["indices"], sel32["values"], return_val=True, tie_break=1)

    n_rows, n_cols = scores16.shape
    valid = torch.arange(top_k, device=device)[None, :] < row_lens[:, None].clamp(max=top_k)

    def membership(indices):
        mask = torch.zeros(n_rows, n_cols + 1, dtype=torch.bool, device=device)
        mask.scatter_(1, torch.where(valid, indices.long(), n_cols), True)
        return mask[:, :n_cols]

    in32, in16 = membership(sel32["indices"]), membership(sel16["indices"])
    cutoff16 = torch.where(in16, scores16.float(), float("inf")).amin(dim=1, keepdim=True)
    disagree = in32 ^ in16
    assert torch.equal(scores16.float()[disagree], cutoff16.expand_as(disagree)[disagree]), "a selection change that is not a cutoff tie"
    short = row_lens <= top_k
    assert not bool(disagree[short].any()), "rows with at most top_k keys select every key in both dtypes"

    overlap = (in32 & in16).sum(dim=1).float() / row_lens.clamp(max=top_k).clamp(min=1).float()
    long_rows = ~short
    print(
        f"recall@{top_k} BF16 vs FP32 on {int(long_rows.sum())} rows longer than top_k: "
        f"mean {overlap[long_rows].mean():.6f} min {overlap[long_rows].min():.6f}"
    )
