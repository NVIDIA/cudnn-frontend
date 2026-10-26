# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import (
    _require_mxfp4_indexer,
    _require_sm90,
    dsa_init,
    expand_mxfp8_scale,
    make_random_mxfp4,
    make_random_mxfp8_scale,
    pack_mxfp8_scales_thd,
    quantize_e2m1,
    quantize_fp8_1x128,
    quantize_mxfp8,
    with_dsa_indexer_forward_params,
)
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    check_ref_compressed_topk,
    check_ref_indexer_forward,
    ref_indexer_forward,
)


@pytest.mark.L0
def test_DSA_indexer_forward_api_is_split_from_top_k():
    try:
        from cudnn import DSA
        import cudnn.deepseek_sparse_attention as dsa
        import cudnn.deepseek_sparse_attention.indexer_forward as indexer_forward
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    dense_parameters = inspect.signature(DSA.indexer_forward_wrapper).parameters
    fused_parameters = inspect.signature(DSA.indexer_forward_top_k_wrapper).parameters
    compressed_only = {
        "top_k",
        "return_softmax",
        "softmax_out",
        "microbatch_rows",
        "topk_indices_global",
        "cand_buffer",
        "out_indices",
        "out_logits",
        "cand_batch_offsets",
        "deterministic",
    }

    assert compressed_only.isdisjoint(dense_parameters)
    assert "is_compressed_logits" not in dense_parameters
    assert "is_compressed_logits" not in fused_parameters
    assert tuple(fused_parameters)[:4] == ("q", "k", "w", "top_k")
    assert compressed_only <= fused_parameters.keys()
    assert "indexer_forward_top_k_wrapper" in dsa.__all__
    assert "indexer_forward_top_k_wrapper" in indexer_forward.__all__
    assert DSA.indexer_forward_top_k_wrapper is indexer_forward.indexer_forward_top_k_wrapper


def _allocate_inputs(cfg, weight_dtype=torch.bfloat16):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv

    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device="cuda")
    if weight_dtype == torch.float32:
        # A coherent sub-BF16 offset makes silent downcasting observable in scores.
        w = (w.abs() + 1).float() + 2**-10
    return q, k, w


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ GPU required")


def _sm100_mqa_inputs(b, s_q, s_k, h_q=64, d=128):
    device = torch.device("cuda")
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)
    return q, k, w


def _allocation_count() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocation.all.allocated"]


_PLACEHOLDER_POOL_NAMES = (
    "_denom_placeholder_cache",
    "_denom_placeholder_cache_lock",
    "_get_fwd_denom_placeholder",
    "_get_fwd_unified_denom_placeholder",
    "_return_output",
)


@pytest.mark.L0
@torch_fork_set_rng(seed=21)
def test_denom_slot_compiled_out(compile_allocates_nothing):
    """Forward compiles compute_lse=False with denom=None (Rule 8, R3): no module-level placeholder pool, no per-execute allocation."""
    _require_sm100()
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_forward import _compressed_top_k_sm100 as compressed_impl
        from cudnn.deepseek_sparse_attention.indexer_forward import _interface as dense_impl
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    for module in (dense_impl, compressed_impl):
        for name in _PLACEHOLDER_POOL_NAMES:
            assert not hasattr(module, name), f"{module.__name__}.{name} still exists"

    b, s_q, s_k, ratio = 1, 128, 512, 4
    q, k, w = _sm100_mqa_inputs(b, s_q, s_k)
    out = torch.empty(b, s_q, s_k, dtype=torch.float32, device=q.device)
    api = DSA.IndexerForward(q, k, w, out, ratio=ratio)
    assert api.check_support()
    compile_allocates_nothing(api)
    api.execute(q, k, w, out)  # the kernel compiles lazily on the first execute
    before = _allocation_count()
    for _ in range(3):
        api.execute(q, k, w, out)
    assert _allocation_count() == before
    check_ref_indexer_forward(q, k, w, out, ratio)


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
def test_padded_out_is_bound_directly(compile_allocates_nothing):
    """S_k % 4 != 0: the (B, S_q, ceil4(S_k)) allocation is bound through its [..., :S_k] view; no staging, no copy-back, tail untouched."""
    _require_sm100()
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_forward import _interface as dense_impl
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    b, s_q, s_k, ratio = 2, 1000, 1001, 1
    s_k_padded = 1004
    q, k, w = _sm100_mqa_inputs(b, s_q, s_k)
    sentinel = 12345.0
    out = torch.full((b, s_q, s_k_padded), sentinel, dtype=torch.float32, device=q.device)
    api = DSA.IndexerForward(q, k, w, out, ratio=ratio)
    assert api.check_support()
    compile_allocates_nothing(api)
    api.execute(q, k, w, out)
    before = _allocation_count()
    for _ in range(3):
        api.execute(q, k, w, out)
    assert _allocation_count() == before
    check_ref_indexer_forward(q, k, w, out[..., :s_k], ratio)
    assert torch.equal(out[..., s_k:], torch.full_like(out[..., s_k:], sentinel))

    # The strict entry the plan binds: a contiguous (B, S_q, 1001) buffer has 4004-byte rows, not bindable; out is required.
    plain = torch.empty(b, s_q, s_k, dtype=torch.float32, device=q.device)
    with pytest.raises(ValueError, match="16-byte-aligned"):
        dense_impl._indexer_fwd_bound(q, k, w, ratio=ratio, out=plain)
    with pytest.raises(ValueError, match="requires out"):
        dense_impl._indexer_fwd_bound(q, k, w, ratio=ratio)

    # The convenience entry (imported by Megatron-LM without out=) keeps develop's torch-op contract.
    scores = dense_impl.indexer_fwd(q, k, w, ratio=ratio)
    assert scores.shape == (b, s_q, s_k) and scores.is_contiguous()
    check_ref_indexer_forward(q, k, w, scores, ratio)
    got = dense_impl.indexer_fwd(q, k, w, ratio=ratio, out=plain)
    assert got is plain and torch.equal(plain, scores), "an unbindable caller out is staged and copied back"

    scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio)["scores"]
    assert scores.shape == (b, s_q, s_k) and scores.is_contiguous(), "the wrapper's default result keeps the torch-op layout"
    check_ref_indexer_forward(q, k, w, scores, ratio)


@pytest.mark.L0
@torch_fork_set_rng(seed=29)
def test_wrapper_ragged_k_out_layouts():
    """S_k % 4 != 0 at the torch-op layer (the plan binds only a padded-stride view): the default result is
    contiguous; a caller's contiguous ``out`` is filled and returned (identity kept, staged + copied back);
    a caller's padded-stride view is bound directly; a wrong shape is refused; a non-default stream works."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    b, s_q, s_k, ratio = 2, 64, 1001, 1
    q, k, w = _sm100_mqa_inputs(b, s_q, s_k)
    dev = q.device
    reference = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio)["scores"]
    assert reference.is_contiguous()
    check_ref_indexer_forward(q, k, w, reference, ratio)

    plain = torch.full((b, s_q, s_k), -7.0, dtype=torch.float32, device=dev)
    got = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=plain)["scores"]
    assert got is plain and torch.equal(plain, reference)

    padded = torch.full((b, s_q, 1004), -7.0, dtype=torch.float32, device=dev)
    view = padded[..., :s_k]
    got = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=view)["scores"]
    assert got is view and torch.equal(view, reference)
    assert torch.equal(padded[..., s_k:], torch.full_like(padded[..., s_k:], -7.0)), "the pad columns are never written"

    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        strided_out = torch.empty(b, s_q, 2 * s_k, dtype=torch.float32, device=dev)[..., ::2]
        got = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=strided_out, stream=side.cuda_stream)["scores"]
    torch.cuda.synchronize()
    assert got is strided_out and torch.equal(strided_out, reference)

    with pytest.raises(ValueError, match="out must have shape"):
        DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=torch.empty(b, s_q, 1004, dtype=torch.float32, device=dev))


@pytest.mark.L0
@torch_fork_set_rng(seed=27)
def test_check_support_declines_non_unit_innermost_stride():
    """check_support() declines strided Q/K/W and a non-contiguous padded Out (Rule 2 / R5); the wrapper keeps its documented .contiguous() convenience."""
    _require_sm100()
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_forward import _interface as dense_impl
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    b, s_q, s_k, h_q, d, ratio = 1, 64, 256, 64, 128, 4
    q, k, w = _sm100_mqa_inputs(b, s_q, s_k, h_q=h_q, d=d)
    device = q.device
    out = torch.empty(b, s_q, s_k, dtype=torch.float32, device=device)

    q_t = torch.randn(b, s_q, d, h_q, dtype=torch.bfloat16, device=device).transpose(2, 3)
    k_t = torch.randn(b, d, s_k, 1, dtype=torch.bfloat16, device=device).permute(0, 2, 3, 1)
    w_t = torch.randn(b, h_q, s_q, dtype=torch.bfloat16, device=device).transpose(1, 2)
    for tensor, expected_shape in ((q_t, q.shape), (k_t, k.shape), (w_t, w.shape)):
        assert tensor.shape == expected_shape and tensor.stride(-1) != 1

    for args in ((q_t, k, w), (q, k_t, w), (q, k, w_t)):
        with pytest.raises(NotImplementedError, match="unit innermost stride"):
            DSA.IndexerForward(*args, out, ratio=ratio).check_support()
    out_strided = torch.empty(b, s_q, s_k + 4, dtype=torch.float32, device=device)[..., :s_k]
    with pytest.raises(NotImplementedError, match="contiguous"):
        DSA.IndexerForward(q, k, w, out_strided, ratio=ratio).check_support()

    with pytest.raises(ValueError, match="unit innermost stride"):
        dense_impl._indexer_fwd_bound(q_t, k, w, ratio=ratio, out=out)
    check_ref_indexer_forward(q_t, k, w, dense_impl.indexer_fwd(q_t, k, w, ratio=ratio), ratio)

    scores = DSA.indexer_forward_wrapper(q_t, k, w, ratio=ratio)["scores"]
    check_ref_indexer_forward(q_t, k, w, scores, ratio)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_forward_params
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
def test_DSA_indexer_forward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    ratio,
    request,
    weight_dtype,
    monkeypatch,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        s_q_default=256,
        s_kv_default=512,
        min_compute_capability=90,
    )
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Indexer forward requires Hopper or Blackwell")
    if torch.cuda.get_device_capability()[0] == 9 and qhead_per_kv_head < 16:
        pytest.skip("the SM90 indexer forward serves qhead_per_kv_head 16 / 32 / 64; the 4 / 8 groups are SM100-family cells")
    q, k, w = _allocate_inputs(cfg, weight_dtype)
    q_causal_offsets = torch.full((cfg["b"],), 4, dtype=torch.int32, device=q.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    from cudnn.deepseek_sparse_attention.indexer_forward import _interface, _interface_sm90

    interface = _interface_sm90 if torch.cuda.get_device_capability()[0] == 9 else _interface
    monkeypatch.setattr(interface, "_compile_cache", {})
    other_dtype = torch.float32 if weight_dtype == torch.bfloat16 else torch.bfloat16
    # Exercise both cache-population orders at fixed Q/K shape, then reuse the first.
    for weights in (w, w.to(other_dtype), w):
        result = DSA.indexer_forward_wrapper(
            q,
            k,
            weights,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            q_causal_offsets=q_causal_offsets,
            stream=stream,
        )
        if not cfg["skip_ref"]:
            check_ref_indexer_forward(q, k, weights, result["scores"], ratio, q_causal_offsets=q_causal_offsets, compute_dtype=torch.float64)
    if weight_dtype == torch.float32 and not cfg["skip_ref"]:
        rounded = ref_indexer_forward(q, k, w.bfloat16(), ratio, q_causal_offsets=q_causal_offsets, compute_dtype=torch.float64)
        assert not torch.allclose(result["scores"], rounded, atol=1e-4, rtol=1e-4)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
def test_DSA_indexer_forward_wrapper_qh16_causal_block_boundary():
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()  # return_lse=True: the SM100 dense path does not expose LSE
    device = torch.device("cuda")
    b, s_q, s_k, h_q, h_kv, d = 1, 8, 128, 16, 1, 128
    ratio = 4
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)
    q_causal_offsets = torch.tensor([252], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        q_causal_offsets=q_causal_offsets,
        return_lse=True,
    )
    scores = result["scores"]
    torch.cuda.synchronize()

    # The causal limit moves from 63 to 64 inside the CTA. In particular,
    # k=63 is masked for q=0..2, then becomes valid at q=3.
    assert bool(torch.isneginf(scores[0, :3, 63]).all())
    assert bool(torch.isfinite(scores[0, 3:, 63]).all())
    assert bool(torch.isneginf(scores[0, :7, 64]).all())
    assert bool(torch.isfinite(scores[0, 7, 64]))
    check_ref_indexer_forward(
        q,
        k,
        w,
        scores,
        ratio,
        q_causal_offsets=q_causal_offsets,
    )
    scores_ref = ref_indexer_forward(
        q,
        k,
        w,
        ratio,
        q_causal_offsets=q_causal_offsets,
    )
    torch.testing.assert_close(
        result["lse"],
        torch.logsumexp(scores_ref, dim=-1),
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=5)
def test_DSA_indexer_forward_wrapper_zero_key_query_tail_returns_empty_scores():
    """The geometry the trailing-query rule admits at its edge: a query shorter than one compressed block (S_q 3 at ratio 4) has NO
    compressed key yet (S_k 0).  The dense wrapper returns the empty (B, S_q, 0) scores -- nothing to score, nothing compiled or launched,
    the empty buffer has no row stride to hold to the 16-byte rule -- the compressed top-k wrapper returns its padded result (-1 indices,
    -inf logits) for the same input, and one query token more (S_q 4 = a complete block with no key) is still refused by name."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the SM100-family dense indexer forward")
    device = torch.device("cuda")
    b, s_q, s_k, h_q, h_kv, d, ratio = 1, 3, 0, 4, 1, 128, 4
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.empty(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)

    scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, qhead_per_kv_head=h_q)["scores"]
    torch.cuda.synchronize()
    assert tuple(scores.shape) == (b, s_q, 0), tuple(scores.shape)
    assert scores.dtype == torch.float32 and scores.is_cuda

    topk = DSA.indexer_forward_top_k_wrapper(q, k, w, top_k=8, ratio=ratio, qhead_per_kv_head=h_q, return_softmax=False)
    torch.cuda.synchronize()
    assert tuple(topk["indices"].shape) == (b, s_q, 8), tuple(topk["indices"].shape)
    assert bool((topk["indices"] == -1).all()) and bool(torch.isneginf(topk["logits"]).all())

    q4 = torch.randn(b, ratio, h_q, d, dtype=torch.bfloat16, device=device)
    w4 = torch.randn(b, ratio, h_q, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match=r"must be <= seqlen_k \* ratio \+ \(ratio - 1\)"):
        DSA.indexer_forward_wrapper(q4, k, w4, ratio=ratio, qhead_per_kv_head=h_q)


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
@pytest.mark.parametrize(
    "qhead_per_kv_head,s_q_default,ratio",
    [
        pytest.param(64, 128, 4, id="qh64"),
        pytest.param(32, 5, 4, id="qh32_query_tail"),
        pytest.param(32, 5, 1, id="qh32_ratio1_query_tail"),
    ],
)
def test_DSA_indexer_forward_wrapper_mxfp8_matches_dequant_reference(
    request,
    qhead_per_kv_head,
    s_q_default,
    ratio,
):
    """Compare MXFP8 dense and top-k indexer outputs with references."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    cfg = dsa_init(
        request=request,
        head_dim=128,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        min_compute_capability=100,
        b_default=1,
        s_q_default=s_q_default,
        s_kv_default=128,
    )

    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv
    sf_groups = (d + 31) // 32
    device = "cuda"

    q_ref = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale(
        (b, s_q, h_q, sf_groups),
        device=device,
        seed=101,
        exponent_min=-2,
        exponent_max=3,
    )
    k_scale_logical = make_random_mxfp8_scale(
        (b, s_k, h_kv, sf_groups),
        device=device,
        seed=103,
        exponent_min=-2,
        exponent_max=3,
    )
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=qhpkv)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    q_causal_offsets = torch.full((b,), 16, dtype=torch.int32, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=cfg["ratio"],
        qhead_per_kv_head=qhpkv,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        stream=stream,
    )
    torch.cuda.synchronize()

    check_ref_indexer_forward(
        q_deq,
        k_deq,
        w,
        result["scores"],
        cfg["ratio"],
        q_causal_offsets=q_causal_offsets,
        atol=5e-3,
        rtol=5e-3,
    )

    compressed = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=min(32, s_k),
        ratio=cfg["ratio"],
        qhead_per_kv_head=qhpkv,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        topk_indices_global=False,
        stream=stream,
    )
    torch.cuda.synchronize()
    check_ref_compressed_topk(
        result["scores"],
        compressed["indices"],
        compressed["logits"],
        min(32, s_k),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
@pytest.mark.parametrize("h_q", [16, 32, 64])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize(
    "recompute,weight_dtype",
    [(False, torch.bfloat16), (False, torch.float32), (True, torch.bfloat16)],
    ids=["forward-w-bf16", "forward-w-fp32", "recompute-w-bf16"],
)
def test_DSA_indexer_forward_wrapper_thd_varlen_tails(h_q, ratio, recompute, weight_dtype):
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Requires Hopper or Blackwell")
    if recompute and torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Dense score recompute declines THD on SM90 (BSHD-native kernel); see test_DSA_dense_score_recompute.py")
    device = torch.device("cuda")
    shapes = [(1, 1), (3, 7), (7, 67), (11, 70)]
    h_kv, d = 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    total_q, total_k = int(cu_seqlens_q[-1]), int(cu_seqlens_k[-1])
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device)
    if weight_dtype == torch.float32:
        w = (w.abs() + 1).float() + 2**-10
    q_causal_offsets = torch.tensor(
        [0, 0, *[s_k * ratio - s_q for s_q, s_k in shapes[2:]]],
        dtype=torch.int32,
        device=device,
    )

    wrapper = DSA.dense_indexer_score_recompute_wrapper if recompute else DSA.indexer_forward_wrapper
    forward_lse = not recompute and torch.cuda.get_device_capability()[0] == 9
    options = {} if recompute else {"return_lse": forward_lse}
    result = wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        q_causal_offsets=q_causal_offsets,
        **options,
    )
    scores = result["out"] if recompute else result["scores"]
    lse = result["denom"] if recompute else result.get("lse")
    torch.cuda.synchronize()

    cu_q_host = cu_seqlens_q.tolist()
    cu_k_host = cu_seqlens_k.tolist()
    max_seqlen_k = max(k_lengths)
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        check_ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            scores[q0:q1, :s_k].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
            compute_dtype=torch.float64,
        )
        scores_ref = ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
        ).squeeze(0)
        if lse is not None:
            torch.testing.assert_close(
                lse[q0:q1],
                torch.logsumexp(scores_ref, dim=-1),
                atol=5e-3,
                rtol=5e-3,
            )
        if s_k < max_seqlen_k:
            assert bool(torch.isneginf(scores[q0:q1, s_k:]).all())


@pytest.mark.L0
@torch_fork_set_rng(seed=73)
@pytest.mark.parametrize("h_q", [4, 8])
def test_DSA_indexer_forward_wrapper_mxfp8_small_head_groups_match_dequant_reference(h_q):
    """The DENSE MXFP8 scorer at 4 / 8 heads per KV head (a decode selector's shape: a few query rows, the 128-row tile
    mostly padding): random per-32 E8M0 scales, per-batch causal offsets, a key length that is not a multiple of the
    key tile, against the fp64 reference on the dequantized values."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 indexer forward requires SM100+")
    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    device = torch.device("cuda")
    b, s_q, s_k, d = 3, 37, 200, 128
    ratio = 4
    q_ref = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(b, s_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale((b, s_q, h_q, d // 32), device=device, seed=79 + h_q, exponent_min=-2, exponent_max=3)
    k_scale_logical = make_random_mxfp8_scale((b, s_k, 1, d // 32), device=device, seed=83 + h_q, exponent_min=-2, exponent_max=3)
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_causal_offsets = torch.tensor([0, 300, 4 * s_k - s_q], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q),
        k_scale=scale_utils.pack_k_scale_bshd(k_scale_logical),
    )
    torch.cuda.synchronize()

    ref = ref_indexer_forward(q_deq, k_deq, w, ratio, q_causal_offsets=q_causal_offsets, compute_dtype=torch.float64)  # fp32 of an fp64 value
    scores = result["scores"]
    assert scores.shape == (b, s_q, s_k)
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isneginf(scores), ~finite)
    torch.testing.assert_close(scores[finite], ref[finite], atol=2e-3, rtol=2e-3)
    # the third sequence's last row sees every key; the first sequence's first three rows see none
    assert bool(finite[2, -1].all()) and bool((~finite[0, :3]).all())


@pytest.mark.L0
def test_DSA_indexer_forward_wrapper_mxfp8_requires_mqa():
    """Verify that MXFP8 indexer forward rejects non-MQA inputs."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 indexer forward requires SM100+")

    from cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils import make_scale_cu_seqlens_padded
    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    device = torch.device("cuda")
    b, s_q, s_k, h_q, h_kv, d = 1, 128, 32, 128, 2, 128
    qhead_per_kv_head = h_q // h_kv
    q = torch.zeros((b, s_q, h_q, d), dtype=torch.float8_e4m3fn, device=device)
    k = torch.zeros((b, s_k, h_kv, d), dtype=torch.float8_e4m3fn, device=device)
    w = torch.ones((b, s_q, h_q), dtype=torch.bfloat16, device=device)
    q_scale_logical = torch.ones((b, s_q, h_q, d // 32), device=device).to(torch.float8_e8m0fnu)
    k_scale_logical = torch.ones((b, s_k, h_kv, d // 32), device=device).to(torch.float8_e8m0fnu)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    common = {
        "qhead_per_kv_head": qhead_per_kv_head,
        "precision": "mxfp8",
        "q_scale": q_scale,
        "k_scale": k_scale,
    }

    with pytest.raises(ValueError, match="requires n_heads_kv=1"):
        DSA.indexer_forward_wrapper(q, k, w, **common)

    cu_q = torch.tensor([0, s_q], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, s_k], dtype=torch.int32, device=device)
    cu_q_scale = make_scale_cu_seqlens_padded(cu_q, token_alignment=2)
    cu_k_scale = make_scale_cu_seqlens_padded(cu_k, token_alignment=128)
    with pytest.raises(ValueError, match="requires n_heads_kv=1"):
        DSA.indexer_forward_wrapper(
            q[0],
            k[0],
            w[0],
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=s_q,
            max_seqlen_k=s_k,
            cu_seqlens_q_scale_padded=cu_q_scale,
            cu_seqlens_k_scale_padded=cu_k_scale,
            **common,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
@pytest.mark.parametrize("h_q", [64, 32], ids=["qh64", "qh32"])
def test_DSA_indexer_forward_wrapper_thd_mxfp8_compact_padded_scales(h_q):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("THD MXFP8 indexer forward requires SM100+")

    device = torch.device("cuda")
    shapes = [(127, 32), (129, 64)]
    ratio, h_kv, d = 4, 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    max_q, max_k = max(q_lengths), max(k_lengths)

    q_ref = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale(
        (total_q, h_q, d // 32),
        device=device,
        seed=107,
        exponent_min=-2,
        exponent_max=3,
    )
    k_scale_logical = make_random_mxfp8_scale(
        (total_k, h_kv, d // 32),
        device=device,
        seed=109,
        exponent_min=-2,
        exponent_max=3,
    )
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale, k_scale, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(
        q_scale_logical,
        k_scale_logical,
        cu_q,
        cu_k,
        h_q // h_kv,
        q_alignment=256,
        k_alignment=256,
    )
    expected_scale_prefix = torch.tensor(
        [0, 256, 512],
        dtype=torch.int32,
        device=device,
    )
    assert torch.equal(cu_q_scale, expected_scale_prefix)
    assert torch.equal(cu_k_scale, expected_scale_prefix)
    q_causal_offsets = torch.tensor([1, 17], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q // h_kv,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    assert result["scores"].shape == (total_q, max_k)
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        check_ref_indexer_forward(
            q_deq[q0:q1].unsqueeze(0),
            k_deq[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            result["scores"][q0:q1, :s_k].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
            atol=5e-3,
            rtol=5e-3,
        )


@torch_fork_set_rng(seed=21)
@pytest.mark.parametrize(
    "qhead_per_kv_head,b_default,s_q_default,s_kv_default,ratio,offsets",
    [
        pytest.param(64, 1, 128, 128, 4, (16,), marks=pytest.mark.L0, id="qh64"),
        pytest.param(32, 1, 5, 128, 4, (16,), marks=pytest.mark.L0, id="qh32_query_tail"),
        pytest.param(32, 1, 5, 128, 1, (16,), marks=pytest.mark.L0, id="qh32_ratio1_query_tail"),
        # S_q = ratio * S_k + ratio - 1: the last ratio - 1 query rows trail the last complete block and see all S_k keys.
        pytest.param(64, 1, 131, 32, 4, (0,), marks=pytest.mark.L0, id="qh64_ratio4_query_past_last_block"),
        # Two batches exercise the packed-scale L index; batch 0 reaches the S_k=200 tail (one full, one partial KV block).
        pytest.param(64, 2, 64, 200, 1, (136, 100), marks=pytest.mark.L1, id="qh64_ratio1_b2_key_tail"),
        # Three KV blocks per tile rotate every accumulator and K-scale TMEM slot: three slots at ratio 4 (direct PTX),
        # two at QH32 ratio 1 (CuTe GEMM).
        pytest.param(64, 1, 128, 384, 4, (1408,), marks=pytest.mark.L0, id="qh64_ratio4_three_kv_blocks"),
        pytest.param(32, 1, 40, 320, 1, (280,), marks=pytest.mark.L1, id="qh32_ratio1_three_kv_blocks"),
        # More query tiles than SMs: persistent CTAs carry TMEM slot phases across tiles, and 2..4 KV blocks per tile
        # wrap the three-slot ring unevenly.
        pytest.param(64, 1, 1024, 512, 4, (1024,), marks=pytest.mark.L1, id="qh64_ratio4_persistent_mixed_blocks"),
        # Small head groups pack 128 // group query tokens per tile. Per-batch offsets: the first sequence's first rows
        # see no key and the last sequence's last row sees every key, across a key tail.
        pytest.param(4, 3, 37, 200, 4, (0, 300, 763), marks=pytest.mark.L0, id="qh4_b3_key_tail"),
        pytest.param(8, 3, 37, 200, 4, (0, 300, 763), marks=pytest.mark.L0, id="qh8_b3_key_tail"),
        pytest.param(16, 1, 64, 96, 2, (128,), marks=pytest.mark.L1, id="qh16_ratio2"),
    ],
)
def test_DSA_indexer_forward_wrapper_mxfp4_matches_dequant_reference(
    request,
    qhead_per_kv_head,
    b_default,
    s_q_default,
    s_kv_default,
    ratio,
    offsets,
):
    """Dense and compressed MXFP4 scores match an FP32 reference on the dequantized inputs."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    cuda = pytest.importorskip("cuda.bindings.driver", reason="Environment not supported: cudnn[cutedsl] not installed")
    _require_mxfp4_indexer()

    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    cfg = dsa_init(
        request=request,
        head_dim=128,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        min_compute_capability=100,
        b_default=b_default,
        s_q_default=s_q_default,
        s_kv_default=s_kv_default,
    )

    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv
    device = "cuda"

    q, q_deq, q_scale_logical = make_random_mxfp4((b, s_q, h_q, d), device=device, seed=201)
    k, k_deq, k_scale_logical = make_random_mxfp4((b, s_k, h_kv, d), device=device, seed=203)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    assert q.shape == (b, s_q, h_q, d // 2)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=qhpkv)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    q_causal_offsets = torch.tensor([offsets[i % len(offsets)] for i in range(b)], dtype=torch.int32, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    common = dict(
        ratio=cfg["ratio"],
        qhead_per_kv_head=qhpkv,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp4",
        q_scale=q_scale,
        k_scale=k_scale,
        stream=stream,
    )

    result = DSA.indexer_forward_wrapper(q, k, w, **common)
    torch.cuda.synchronize()

    check_ref_indexer_forward(
        q_deq,
        k_deq,
        w,
        result["scores"],
        cfg["ratio"],
        q_causal_offsets=q_causal_offsets,
        atol=1e-3,
        rtol=1e-3,
    )

    compressed = DSA.indexer_forward_top_k_wrapper(q, k, w, top_k=min(32, s_k), topk_indices_global=False, **common)
    torch.cuda.synchronize()
    check_ref_compressed_topk(
        result["scores"],
        compressed["indices"],
        compressed["logits"],
        min(32, s_k),
        atol=1e-4,
        rtol=1e-4,
    )


@torch_fork_set_rng(seed=23)
@pytest.mark.parametrize(
    "h_q",
    [
        pytest.param(64, marks=pytest.mark.L0, id="qh64"),
        pytest.param(32, marks=pytest.mark.L1, id="qh32"),
        pytest.param(4, marks=pytest.mark.L1, id="qh4"),
    ],
)
def test_DSA_indexer_forward_wrapper_thd_mxfp4_compact_padded_scales(h_q):
    """THD MXFP4 scores with 256-token padded scale spans match an FP32 reference on the dequantized inputs."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    cuda = pytest.importorskip("cuda.bindings.driver", reason="Environment not supported: cudnn[cutedsl] not installed")
    _require_mxfp4_indexer()

    device = torch.device("cuda")
    shapes = [(127, 32), (129, 64)]
    ratio, h_kv, d = 4, 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    max_q, max_k = max(q_lengths), max(k_lengths)

    q, q_deq, q_scale_logical = make_random_mxfp4((total_q, h_q, d), device=device, seed=207)
    k, k_deq, k_scale_logical = make_random_mxfp4((total_k, h_kv, d), device=device, seed=209)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale, k_scale, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(
        q_scale_logical,
        k_scale_logical,
        cu_q,
        cu_k,
        h_q // h_kv,
        q_alignment=256,
        k_alignment=256,
    )
    q_causal_offsets = torch.tensor([1, 17], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q // h_kv,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp4",
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    assert result["scores"].shape == (total_q, max_k)
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        check_ref_indexer_forward(
            q_deq[q0:q1].unsqueeze(0),
            k_deq[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            result["scores"][q0:q1, :s_k].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=25)
@pytest.mark.parametrize("ratio", [4, 1], ids=["direct_ptx", "cute_gemm"])
def test_DSA_indexer_forward_wrapper_mxfp4_scale_group_routing(ratio):
    """Q/K data confined to one 32-element group is scaled by that group's E8M0 scales, for each of the four groups."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_mxfp4_indexer()
    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    device = "cuda"
    h_q, d, s_q, s_k = 64, 128, 128, 128
    # Every (Q group, K group) pair has a distinct exponent sum, so a scale read from the wrong group moves the score by
    # a power of two, and data read from the wrong group is zero and zeroes it.
    q_exponents = torch.tensor([-6.0, -2.0, 2.0, 6.0], device=device)
    k_exponents = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=device)
    q_scale_logical = torch.ldexp(torch.ones_like(q_exponents), q_exponents).expand(1, s_q, h_q, d // 32).contiguous().to(torch.float8_e8m0fnu)
    k_scale_logical = torch.ldexp(torch.ones_like(k_exponents), k_exponents).expand(1, s_k, 1, d // 32).contiguous().to(torch.float8_e8m0fnu)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    w = torch.rand(1, s_q, h_q, dtype=torch.bfloat16, device=device) + 0.5
    q_causal_offsets = torch.tensor([ratio * s_k - s_q], dtype=torch.int32, device=device)

    for group in range(d // 32):
        # Non-negative data: no head's dot product is hidden by the ReLU.
        in_group = (torch.arange(d, device=device) // 32 == group).float()
        q, q_deq = quantize_e2m1(torch.rand(1, s_q, h_q, d, device=device) * 6 * in_group * expand_mxfp8_scale(q_scale_logical, d), q_scale_logical)
        k, k_deq = quantize_e2m1(torch.rand(1, s_k, 1, d, device=device) * 6 * in_group * expand_mxfp8_scale(k_scale_logical, d), k_scale_logical)

        result = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, q_causal_offsets=q_causal_offsets, precision="mxfp4", q_scale=q_scale, k_scale=k_scale)
        torch.cuda.synchronize()

        scores = result["scores"]
        assert bool((scores[torch.isfinite(scores)] > 0).all()), f"group {group}: a visible score is zero"
        check_ref_indexer_forward(q_deq, k_deq, w, scores, ratio, q_causal_offsets=q_causal_offsets, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=27)
@pytest.mark.parametrize("ratio", [4, 1], ids=["direct_ptx", "cute_gemm"])
def test_DSA_indexer_forward_mxfp4_two_mmas_per_k_chunk(monkeypatch, ratio):
    """128-wide K chunks issue two K=64 MXFP4 MMAs per chunk, each reading its own pair of E8M0 scale groups."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    _require_mxfp4_indexer()
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface
    from cudnn.deepseek_sparse_attention.score_recompute import _block_scaled
    from cudnn.deepseek_sparse_attention.score_recompute.indexer_score_unified_sm100_mxfp4 import IndexerScoreUnifiedSm100Mxfp4
    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    constructed = []

    class WideKChunk(IndexerScoreUnifiedSm100Mxfp4):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **{**kwargs, "k_block_size": 128, "kv_stage": 12})

    monkeypatch.setitem(_block_scaled.BLOCK_SCALED_FORMATS, "mxfp4", dataclasses.replace(_block_scaled.BLOCK_SCALED_FORMATS["mxfp4"], kernel_cls=WideKChunk))
    monkeypatch.setattr(_interface, "_compile_cache", {})

    device = "cuda"
    h_q, d = 64, 128
    s_q, s_k = (128, 384) if ratio == 4 else (64, 256)
    q, q_ref, q_scale_logical = make_random_mxfp4((1, s_q, h_q, d), device=device, seed=211)
    k, k_ref, k_scale_logical = make_random_mxfp4((1, s_k, 1, d), device=device, seed=213)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    w = torch.randn(1, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_causal_offsets = torch.tensor([ratio * s_k - s_q], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, q_causal_offsets=q_causal_offsets, precision="mxfp4", q_scale=q_scale, k_scale=k_scale)
    torch.cuda.synchronize()
    assert constructed, "the MXFP4 indexer did not build the 128-wide K-chunk kernel"
    check_ref_indexer_forward(q_ref, k_ref, w, result["scores"], ratio, q_causal_offsets=q_causal_offsets, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
def test_DSA_indexer_forward_imports_without_block_scaled_torch_dtypes():
    """The indexer still imports, and names the missing dtypes, on a torch without E8M0 or packed FP4."""
    import os
    import subprocess
    import sys

    cudnn = pytest.importorskip("cudnn", reason="Environment not supported: cudnn[cutedsl] not installed")
    _block_scaled = pytest.importorskip(
        "cudnn.deepseek_sparse_attention.score_recompute._block_scaled", reason="Environment not supported: cudnn[cutedsl] not installed"
    )
    package_root = os.path.dirname(os.path.dirname(cudnn.__file__))
    # The cudnn directory the indexer was imported from; a conftest may have put a source tree ahead of the installed
    # package on cudnn.__path__.
    source_cudnn = os.path.dirname(os.path.dirname(os.path.dirname(_block_scaled.__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))}
    script = (
        "import torch\n"
        "for name in ('float8_e8m0fnu', 'float4_e2m1fn_x2'):\n"
        "    if hasattr(torch, name):\n"
        "        delattr(torch, name)\n"
        "import cudnn\n"
        f"if {source_cudnn!r} not in cudnn.__path__:\n"
        f"    cudnn.__path__.insert(0, {source_cudnn!r})\n"
        "from cudnn.deepseek_sparse_attention.score_recompute import _block_scaled\n"
        f"assert _block_scaled.__file__ == {_block_scaled.__file__!r}, _block_scaled.__file__\n"
        "import cudnn.deepseek_sparse_attention.indexer_forward\n"
        "assert _block_scaled.validate_indexer_precision('bf16') == 'bf16'\n"
        "for precision, dtypes in (('mxfp8', ('float8_e8m0fnu',)), ('mxfp4', ('float4_e2m1fn_x2', 'float8_e8m0fnu'))):\n"
        "    try:\n"
        "        _block_scaled.validate_indexer_precision(precision)\n"
        "    except NotImplementedError as exc:\n"
        "        assert all(dtype in str(exc) for dtype in dtypes), exc\n"
        "    else:\n"
        "        raise AssertionError(precision)\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0 and "No module named 'cutlass'" in proc.stderr:
        pytest.skip("requires CuTe DSL")
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-2000:]


@pytest.mark.L0
@pytest.mark.parametrize("block_scale", [1.0, 0.125, 4.0])
def test_quantize_e2m1_rounds_ties_to_even(block_scale):
    """``quantize_e2m1`` rounds signed E2M1 midpoints to the even code and saturates at 6, one E8M0 scale per 32 elements."""
    if getattr(torch, "float4_e2m1fn_x2", None) is None:
        pytest.skip("requires torch.float4_e2m1fn_x2")
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.75, -5.0, 0.3, -2.9, 0.0, 6.0, 0.1, -0.1])
    rounded = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -1.0, -4.0, 0.5, -3.0, 0.0, 6.0, 0.0, -0.0])
    # Two 32-element groups, the second with twice the scale: each is quantized against its own scale.
    group_scales = torch.tensor([block_scale, 2 * block_scale])
    x = torch.cat([values, -values]).repeat(2) * group_scales.repeat_interleave(32)
    packed, dequantized = quantize_e2m1(x.unsqueeze(0), group_scales.unsqueeze(0).to(torch.float8_e8m0fnu))
    expected = torch.cat([rounded, -rounded]).repeat(2) * group_scales.repeat_interleave(32)
    assert torch.equal(dequantized, expected.unsqueeze(0))
    codes = packed.view(torch.uint8)
    assert codes.shape == (1, 32)
    # Element 2i in the low nibble: 0.25 -> code 0 (low), 0.75 -> 1.0 = code 2 (high).
    assert int(codes[0, 0]) == 0x20


@pytest.mark.L0
def test_DSA_indexer_forward_wrapper_mxfp4_rejects_mismatched_inputs():
    """MXFP4 inputs that break the contract are refused on the host."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    _require_mxfp4_indexer()

    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    device = "cuda"
    b, s_q, s_k, h_q, d = 1, 8, 128, 64, 128
    q, _, q_scale_logical = make_random_mxfp4((b, s_q, h_q, d), device=device, seed=1)
    k, _, k_scale_logical = make_random_mxfp4((b, s_k, 1, d), device=device, seed=2)
    w = torch.rand(b, s_q, h_q, dtype=torch.bfloat16, device=device)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    common = dict(precision="mxfp4", q_scale=q_scale, k_scale=k_scale)

    q_e4m3 = torch.zeros(b, s_q, h_q, d, dtype=torch.float8_e4m3fn, device=device)
    k_e4m3 = torch.zeros(b, s_k, 1, d, dtype=torch.float8_e4m3fn, device=device)
    for wrapper, options in ((DSA.indexer_forward_wrapper, {}), (DSA.indexer_forward_top_k_wrapper, {"top_k": 16})):
        with pytest.raises(TypeError, match="requires q and k to be torch.float4_e2m1fn_x2"):
            wrapper(q_e4m3, k_e4m3, w, **options, **common)
        with pytest.raises(TypeError, match="requires q_scale and k_scale to be torch.float8_e8m0fnu"):
            wrapper(q, k, w, **options, **{**common, "k_scale": k_scale.view(torch.float8_e4m3fn)})
        with pytest.raises(ValueError, match="requires sf_vec_size=32"):
            wrapper(q, k, w, sf_vec_size=16, **options, **common)
        with pytest.raises(ValueError, match="requires n_heads_kv=1"):
            wrapper(q, torch.cat([k, k], dim=2), w, **options, **common)
    with pytest.raises(TypeError, match="w to be torch.bfloat16"):
        DSA.indexer_forward_wrapper(q, k, w.float(), **common)
    with pytest.raises(ValueError, match="q_scale packed shape"):
        DSA.indexer_forward_wrapper(q, k, w, **{**common, "q_scale": k_scale})
    with pytest.raises(NotImplementedError, match="microbatch_rows"):
        DSA.indexer_forward_top_k_wrapper(q, k, w, top_k=16, microbatch_rows=8, **common)
    # S_q = 8 trails ratio * S_k = 4 by more than ratio - 1 tokens.
    with pytest.raises(ValueError, match=r"must be <= seqlen_k \* ratio \+ \(ratio - 1\)"):
        DSA.indexer_forward_wrapper(q, k[:, :1], w, ratio=4, **common)
    with pytest.raises(ValueError, match=r"must be <= seqlen_k \* ratio \+ \(ratio - 1\)"):
        DSA.indexer_forward_top_k_wrapper(q, k[:, :1], w, top_k=16, ratio=4, **common)
    # An unpacked last dim (head_dim, not head_dim // 2) is reported by its packed width, not as a scale or head_dim mismatch.
    q_wide, k_wide = (torch.cat([t.view(torch.uint8)] * 2, dim=-1).view(t.dtype) for t in (q, k))
    with pytest.raises(ValueError, match="stored as q/k last dim 64"):
        DSA.indexer_forward_wrapper(q_wide, k_wide, w, **common)
    with pytest.raises(ValueError, match="stored as q/k last dim 64"):
        DSA.indexer_forward_top_k_wrapper(q_wide, k_wide, w, top_k=16, **common)
    with pytest.raises(ValueError, match=r"q last dim \(64\) != k last dim \(128\)"):
        DSA.indexer_forward_wrapper(q, k_wide, w, **common)


@pytest.mark.L0
def test_DSA_indexer_forward_wrapper_rejects_mismatched_qk_last_dim():
    """BF16 Q/K whose last dims differ are refused on the host, in BSHD and THD."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")

    device = "cuda"
    q = torch.randn(1, 8, 64, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 32, 1, 64, dtype=torch.bfloat16, device=device)
    w = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match=r"q last dim \(128\) != k last dim \(64\)"):
        DSA.indexer_forward_wrapper(q, k, w, ratio=4)
    cu_q = torch.tensor([0, 8], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 32], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match=r"q last dim \(128\) != k last dim \(64\)"):
        DSA.indexer_forward_wrapper(q[0], k[0], w[0], ratio=4, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=8, max_seqlen_k=32)


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
@pytest.mark.parametrize("h_q", [64, 32], ids=["qh64", "qh32"])
def test_DSA_indexer_forward_wrapper_mxfp4_zero_key_query_tail(h_q):
    """With no compressed key (S_q 3 at ratio 4, S_k 0), MXFP4 dense scores are empty and top-k is all padding
    (-1 indices, -inf logits); S_q 4 is refused."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    _require_mxfp4_indexer()

    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    device = "cuda"
    b, s_q, s_k, d, ratio, top_k = 1, 3, 0, 128, 4, 8
    q, _, q_scale_logical = make_random_mxfp4((b, s_q, h_q, d), device=device, seed=11)
    k = torch.empty((b, s_k, 1, d // 2), dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2)
    w = torch.rand(b, s_q, h_q, dtype=torch.bfloat16, device=device)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q)
    k_scale = torch.empty((b, s_k, d // 32), dtype=torch.float8_e8m0fnu, device=device)
    common = dict(ratio=ratio, precision="mxfp4", q_scale=q_scale, k_scale=k_scale)

    scores = DSA.indexer_forward_wrapper(q, k, w, **common)["scores"]
    torch.cuda.synchronize()
    assert tuple(scores.shape) == (b, s_q, 0)
    assert scores.dtype == torch.float32 and scores.is_cuda

    topk = DSA.indexer_forward_top_k_wrapper(q, k, w, top_k=top_k, return_softmax=False, **common)
    torch.cuda.synchronize()
    assert tuple(topk["indices"].shape) == (b, s_q, top_k)
    assert bool((topk["indices"] == -1).all()) and bool(torch.isneginf(topk["logits"]).all())

    q4, _, q4_scale_logical = make_random_mxfp4((b, ratio, h_q, d), device=device, seed=13)
    w4 = torch.rand(b, ratio, h_q, dtype=torch.bfloat16, device=device)
    q4_scale = scale_utils.pack_q_scale_bshd(q4_scale_logical, qhead_per_kv_head=h_q)
    with pytest.raises(ValueError, match=r"must be <= seqlen_k \* ratio \+ \(ratio - 1\)"):
        DSA.indexer_forward_wrapper(q4, k, w4, **{**common, "q_scale": q4_scale})


@pytest.mark.L0
@pytest.mark.parametrize("precision", ["mxfp8", "mxfp4"])
def test_DSA_indexer_forward_wrapper_sm90_rejects_block_scaled_precisions(monkeypatch, precision):
    """On SM90 a block-scaled precision, in either case, fails the SM90 precision check, and its scale prefixes are SM100-only."""
    api = pytest.importorskip("cudnn.deepseek_sparse_attention.indexer_forward.api", reason="Environment not supported: cudnn[cutedsl] not installed")

    monkeypatch.setattr(api, "device_major", lambda: 9)
    q = torch.zeros(1, 8, 64, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16)
    w = torch.zeros(1, 8, 64, dtype=torch.bfloat16)
    for name in (precision, precision.upper()):
        with pytest.raises(ValueError, match="precision must be 'bf16' or 'fp8'"):
            api.indexer_forward_wrapper(q, k, w, precision=name)
    prefix = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="Block-scale padded cu_seqlens are only supported on SM100"):
        api.indexer_forward_wrapper(q, k, w, precision=precision, cu_seqlens_q_scale_padded=prefix, cu_seqlens_k_scale_padded=prefix)


@pytest.mark.L0
@pytest.mark.parametrize("gate", ["dsl_floor", "dsl_target", "arch_map"])
def test_require_mxfp4_indexer_skips_on_an_unsupported_dsl(monkeypatch, gate):
    """``_require_mxfp4_indexer`` skips, not errors, below the CuTe DSL floor, on a DSL without the GPU target, or off the DSA arch map."""
    if getattr(torch, "float4_e2m1fn_x2", None) is None:
        pytest.skip("requires torch.float4_e2m1fn_x2")
    from cudnn.frost import buffers
    from cudnn.deepseek_sparse_attention.utils import compiler

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    if gate == "dsl_floor":
        monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.6.2")))
        expected = "requires nvidia-cutlass-dsl >= .*; found 4.6.2"
    elif gate == "dsl_target":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
        monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.7.1")))
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
        expected = "SM107 requires a CuTe DSL build supporting sm_107a; found nvidia-cutlass-dsl 4.7.1"
    else:
        monkeypatch.setattr(buffers, "_DSL_STATE", (True, ("nvidia-cutlass-dsl", "4.8.0")))

        def unmapped_gpu():
            raise RuntimeError("Unsupported GPU compute capability (10, 9) for DSA CuTe kernels.")

        monkeypatch.setattr(compiler, "gpu_arch_flag", unmapped_gpu)
        expected = "Unsupported GPU compute capability"
    with pytest.raises(pytest.skip.Exception, match=expected):
        _require_mxfp4_indexer()


@pytest.mark.L0
@torch_fork_set_rng(seed=12)
@pytest.mark.parametrize("qhead_per_kv_head", [32, 64])
def test_DSA_indexer_forward_wrapper_fp8_sm90_matches_dequant_reference(request, qhead_per_kv_head):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    cfg = dsa_init(
        request=request,
        head_dim=128,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=4,
        min_compute_capability=90,
        b_default=1,
        s_q_default=128,
        s_kv_default=128,
    )

    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    h_q = h_kv * qhpkv
    device = "cuda"

    q_ref = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q, q_scale = quantize_fp8_1x128(q_ref)
    k, k_scale = quantize_fp8_1x128(k_ref)
    q_deq = q.float() * q_scale.unsqueeze(-1)
    k_deq = k.float() * k_scale.unsqueeze(-1)
    q_causal_offsets = torch.full((b,), 16, dtype=torch.int32, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=cfg["ratio"],
        qhead_per_kv_head=qhpkv,
        q_causal_offsets=q_causal_offsets,
        precision="fp8",
        q_scale=q_scale,
        k_scale=k_scale,
        return_lse=True,
        stream=stream,
    )
    torch.cuda.synchronize()

    scores_ref = ref_indexer_forward(
        q_deq,
        k_deq,
        w,
        cfg["ratio"],
        q_causal_offsets=q_causal_offsets,
    )
    check_ref_indexer_forward(
        q_deq,
        k_deq,
        w,
        result["scores"],
        cfg["ratio"],
        q_causal_offsets=q_causal_offsets,
        atol=2e-3,
        rtol=2e-3,
    )
    lse_ref = torch.logsumexp(scores_ref, dim=-1)
    assert torch.equal(torch.isfinite(result["lse"]), torch.isfinite(lse_ref))
    finite_lse = torch.isfinite(lse_ref)
    torch.testing.assert_close(
        result["lse"][finite_lse],
        lse_ref[finite_lse],
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_DSA_indexer_forward_wrapper_fp8_sm90_thd_matches_dequant_reference():
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    _require_sm90()
    device = torch.device("cuda")
    shapes = [(5, 67), (7, 70)]
    ratio, h_q, h_kv, d = 4, 32, 1, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])

    q_ref = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q, q_scale = quantize_fp8_1x128(q_ref)
    k, k_scale = quantize_fp8_1x128(k_ref)
    q_deq = q.float() * q_scale.unsqueeze(-1)
    k_deq = k.float() * k_scale.unsqueeze(-1)
    q_causal_offsets = torch.tensor(
        [s_k * ratio - s_q for s_q, s_k in shapes],
        dtype=torch.int32,
        device=device,
    )

    result = DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        q_causal_offsets=q_causal_offsets,
        precision="fp8",
        q_scale=q_scale,
        k_scale=k_scale,
        return_lse=True,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    assert result["scores"].shape == (total_q, max(k_lengths))
    assert result["lse"].shape == (total_q,)
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        scores_ref = ref_indexer_forward(
            q_deq[q0:q1].unsqueeze(0),
            k_deq[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
        ).squeeze(0)
        torch.testing.assert_close(
            result["scores"][q0:q1, :s_k],
            scores_ref,
            atol=2e-3,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            result["lse"][q0:q1],
            torch.logsumexp(scores_ref, dim=-1),
            atol=5e-3,
            rtol=5e-3,
        )
        if s_k < max(k_lengths):
            assert bool(torch.isneginf(result["scores"][q0:q1, s_k:]).all())


@pytest.mark.L0
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
def test_DSA_indexer_forward_class_head_weight_contract(monkeypatch, weight_dtype):
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.indexer_forward import api

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("IndexerForward class API requires Blackwell")
    q = torch.empty(1, 1, 64, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(1, 4, 1, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(1, 1, 64, dtype=weight_dtype, device="cuda")
    out = torch.empty(1, 1, 4, dtype=torch.float32, device="cuda")
    plan = DSA.IndexerForward(q, k, w, out, ratio=1, qhead_per_kv_head=64)
    plan.compile()

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("Mismatched head-weight dtype must be rejected before launch")

    with monkeypatch.context() as patched:
        patched.setattr(api, "indexer_fwd_sm100", unexpected_launch)
        wrong_dtype = torch.float32 if weight_dtype == torch.bfloat16 else torch.bfloat16
        with pytest.raises(ValueError, match="W dtype mismatch"):
            plan.execute(q, k, w.to(wrong_dtype), out)

    if weight_dtype == torch.float32:
        from cudnn.deepseek_sparse_attention.indexer_forward import _interface

        strided_w = torch.empty(1, 1, 128, dtype=weight_dtype, device="cuda")[..., ::2]
        with pytest.raises(NotImplementedError, match="stride"):
            DSA.IndexerForward(q, k, strided_w, out, ratio=1).check_support()

        def unexpected_normalization(*args, **kwargs):
            raise AssertionError("Unsupported FP32 W stride must be rejected before normalization")

        monkeypatch.setattr(_interface, "_maybe_contiguous", unexpected_normalization)
        with pytest.raises(NotImplementedError, match="stride"):
            DSA.indexer_forward_wrapper(q, k, strided_w, ratio=1)
        with pytest.raises(NotImplementedError, match="stride"):
            DSA.indexer_forward_top_k_wrapper(q, k, strided_w, top_k=1, ratio=1)
