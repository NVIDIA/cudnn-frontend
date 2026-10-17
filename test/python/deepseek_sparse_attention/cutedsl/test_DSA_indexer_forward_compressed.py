# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_reference import (
    check_ref_compressed_topk,
    ref_indexer_forward,
)
from deepseek_sparse_attention.cutedsl.dsa_utils import (
    expand_mxfp8_scale,
    make_random_mxfp8_scale,
    pack_mxfp8_scales_thd,
    quantize_mxfp8,
)


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ GPU required")


def _bshd_global_to_local(indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    batch_base = torch.arange(indices.shape[0], device=indices.device, dtype=torch.int64).view(-1, 1, 1).mul(seqlen_k)
    return torch.where(
        indices >= 0,
        indices.to(torch.int64) - batch_base,
        indices.to(torch.int64),
    ).to(torch.int32)


def _check_fused_softmax(
    indices: torch.Tensor,
    logits: torch.Tensor,
    softmax: torch.Tensor,
) -> None:
    """Validate the stage-2 softmax, including all-padding rows."""
    invalid = indices < 0
    expected = torch.softmax(logits.masked_fill(invalid, float("-inf")), dim=-1).masked_fill(invalid, 0.0)
    torch.testing.assert_close(softmax, expected, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(softmax).all()


@pytest.mark.L0
@pytest.mark.parametrize("qhead_per_kv_head", [2, 12, 128])
def test_compressed_indexer_rejects_unsupported_qhead_group_before_launch(qhead_per_kv_head):
    """A head group the packed tile does not serve declines with a ValueError naming the set (8 was the
    probe here until the 4 / 8 / 16 groups landed; 12 is not a power of two and 128 exceeds the tile)."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    q = torch.randn((1, 8, qhead_per_kv_head, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((1, 2, 1, 128), dtype=torch.bfloat16, device=device)
    w = torch.randn((1, 8, qhead_per_kv_head), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match=r"qhead_per_kv_head in \(4, 8, 16, 32, 64\)"):
        DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=1,
            qhead_per_kv_head=qhead_per_kv_head,
            return_softmax=False,
        )


@pytest.mark.L0
@pytest.mark.parametrize("qhead_per_kv_head", [2, 12, 128])
def test_compressed_indexer_mxfp8_rejects_unsupported_qhead_group_before_launch(qhead_per_kv_head):
    """The MXFP8 kernel serves the same head groups as BF16 at its fixed 128-row tile (4 was the decline sample
    here until the small groups landed); a group outside the set declines with a ValueError naming it."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    q = torch.zeros((1, 8, qhead_per_kv_head, 128), dtype=torch.float8_e4m3fn, device=device)
    k = torch.zeros((1, 2, 1, 128), dtype=torch.float8_e4m3fn, device=device)
    w = torch.ones((1, 8, qhead_per_kv_head), dtype=torch.bfloat16, device=device)
    q_rows = (8 * qhead_per_kv_head + 127) // 128 * 128
    q_scale = torch.ones((1, q_rows, 4), device=device).to(torch.float8_e8m0fnu)
    k_scale = torch.ones((1, 128, 4), device=device).to(torch.float8_e8m0fnu)

    with pytest.raises(ValueError, match=r"precision='mxfp8' indexer requires qhead_per_kv_head in \(4, 8, 16, 32, 64\)"):
        DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=1,
            qhead_per_kv_head=qhead_per_kv_head,
            precision="mxfp8",
            q_scale=q_scale,
            k_scale=k_scale,
            return_softmax=False,
        )


def _assert_topk_sets_match(dense_ref: torch.Tensor, indices: torch.Tensor, top_k: int, tie_tol: float, logits: torch.Tensor | None = None) -> None:
    """The selected SET equals torch.topk on the dense reference up to flips at the k-th boundary.

    ``dense_ref`` is ``(rows, n_k)`` with ``-inf`` on the masked positions, ``indices`` ``(rows, top_k)`` local ids,
    ``-1`` padded (``top_k`` may exceed ``n_k``: the row is then padded past its visible count). A row's valid
    count must be ``min(top_k, visible)``, its ids unique and in range, and every id that is in exactly one of the
    two sets must score within ``tie_tol`` of the row's k-th reference score. With ``logits`` the selected values
    must be the reference scores of the selected ids and ``-inf`` on the padding.
    """
    assert indices.shape == (dense_ref.shape[0], top_k)
    valid = indices >= 0
    visible = torch.isfinite(dense_ref).sum(dim=-1)
    assert torch.equal(valid.sum(dim=-1), visible.clamp(max=top_k))
    assert bool(((indices < dense_ref.shape[-1]) | ~valid).all())
    in_actual = torch.zeros_like(dense_ref, dtype=torch.bool)
    rows = torch.arange(dense_ref.shape[0], device=dense_ref.device).unsqueeze(1).expand_as(indices)
    in_actual[rows[valid], indices[valid].long()] = True
    assert torch.equal(in_actual.sum(dim=-1), valid.sum(dim=-1)), "a row lists a block id twice"
    if logits is not None:
        assert logits.shape == indices.shape
        gathered = dense_ref[rows[valid], indices[valid].long()]
        torch.testing.assert_close(logits[valid].to(dense_ref.dtype), gathered, atol=tie_tol, rtol=tie_tol)
        assert bool(torch.isneginf(logits[~valid]).all())
    k_eff = min(top_k, dense_ref.shape[-1])
    ref_topk = torch.topk(dense_ref, k_eff, dim=-1)
    in_expected = torch.zeros_like(in_actual)
    exp_valid = torch.isfinite(ref_topk.values)
    rows_k = torch.arange(dense_ref.shape[0], device=dense_ref.device).unsqueeze(1).expand(-1, k_eff)
    in_expected[rows_k[exp_valid], ref_topk.indices[exp_valid]] = True
    kth = torch.where(exp_valid, ref_topk.values, torch.full_like(ref_topk.values, float("inf"))).min(dim=-1).values
    differs = in_actual ^ in_expected
    if bool(differs.any()):
        gap = (dense_ref - kth.unsqueeze(1)).abs()[differs]
        assert bool((gap <= tie_tol).all()), f"{int(differs.sum())} selected ids differ from torch.topk by more than a tie: max gap {float(gap.max()):.3e}"


@pytest.mark.L0
@torch_fork_set_rng(seed=41)
@pytest.mark.parametrize("h_q", [4, 8, 16])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
def test_DSA_compressed_indexer_forward_bshd_small_head_groups(h_q, weight_dtype):
    """4 / 8 / 16 heads per KV head pack 8 query tokens into a 32- / 64- / 128-column tile; the per-token head
    reduce, the compact store and the top-k are checked against the dense fp64 reference, with a query tail
    past the last complete block (seqlen_q = 4 * 33 + 1) and a per-batch causal offset."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    b, s_q, s_k, d = 2, 133, 64, 128
    ratio, top_k, sm_scale = 4, 24, d**-0.5
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    if weight_dtype == torch.float32:
        w = (w.abs() + 1).float() + 2**-10
    q_causal_offsets = torch.tensor([0, 96], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        sm_scale=sm_scale,
        q_causal_offsets=q_causal_offsets,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
    )
    torch.cuda.synchronize()

    dense_ref = ref_indexer_forward(q, k, w, ratio, q_causal_offsets=q_causal_offsets, compute_dtype=torch.float64) * sm_scale
    tol = 1e-4 if weight_dtype == torch.float32 else 2e-3
    check_ref_compressed_topk(dense_ref, result["indices"], result["logits"], top_k, atol=tol, rtol=tol)
    _assert_topk_sets_match(dense_ref.view(b * s_q, s_k), result["indices"].view(b * s_q, top_k), top_k, tie_tol=tol)


@pytest.mark.L0
@torch_fork_set_rng(seed=43)
def test_DSA_compressed_indexer_forward_thd_small_head_group_tails():
    """THD at 4 heads per KV head: sequences whose length is not a multiple of the ratio (the trailing tokens
    see the complete blocks only), a one-token sequence that sees nothing (all -1), a sequence that sees fewer
    blocks than top_k, and a chunked one whose causal offset makes every row see every block."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    # (seqlen_q, seqlen_k, q_causal_offset): seqlen_q <= seqlen_k * 4 + 3 under offset 0
    shapes = [(67, 16, 0), (1, 1, 0), (30, 8, 0), (40, 40, 120)]
    ratio, top_k, h_q, d = 4, 12, 4, 128
    cu_q = torch.tensor([0, *torch.tensor([s[0] for s in shapes]).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor([s[1] for s in shapes]).cumsum(0).tolist()], dtype=torch.int32, device=device)
    q_causal_offsets = torch.tensor([s[2] for s in shapes], dtype=torch.int32, device=device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.ones(total_q, h_q, dtype=torch.bfloat16, device=device)
    sm_scale = d**-0.5

    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(s[0] for s in shapes),
        max_seqlen_k=max(s[1] for s in shapes),
        q_causal_offsets=q_causal_offsets,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
    )
    torch.cuda.synchronize()
    assert result["indices"].shape == (total_q, top_k)

    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (s_q, s_k, offset) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        dense_ref = (
            ref_indexer_forward(
                q[q0:q1].unsqueeze(0),
                k[k0:k1].unsqueeze(0),
                w[q0:q1].unsqueeze(0),
                ratio,
                q_causal_offsets=q_causal_offsets[batch : batch + 1],
                compute_dtype=torch.float64,
            )
            * sm_scale
        )
        indices = result["indices"][q0:q1]
        if top_k <= s_k:
            check_ref_compressed_topk(dense_ref, indices.unsqueeze(0), result["logits"][q0:q1].unsqueeze(0), top_k, atol=1e-4, rtol=1e-4)
        _assert_topk_sets_match(dense_ref.squeeze(0), indices, top_k, tie_tol=1e-4, logits=result["logits"][q0:q1])
        if offset == 0:
            # rows 0..2 have no complete block yet: nothing selected
            assert bool((indices[: min(3, s_q)] == -1).all())
    # the one-token sequence selects nothing; the chunked sequence (offset 120, 40 keys) selects every key for every row
    one_token = cu_q_host[1]
    assert bool((result["indices"][one_token] == -1).all())
    last_q0 = cu_q_host[3]
    assert bool((result["indices"][last_q0:] >= 0).all())


@pytest.mark.L0
@torch_fork_set_rng(seed=47)
@pytest.mark.parametrize("layout", ["bshd", "thd"])
def test_DSA_compressed_indexer_forward_q_tail_past_the_last_complete_block(layout):
    """A query may be up to ratio - 1 tokens longer than ratio * seqlen_k (its trailing tokens have not completed
    a compressed block); one more token is a geometry mismatch and is refused before any launch."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    s_k, ratio, h_q, d, top_k = 32, 4, 32, 128, 16
    s_q_ok, s_q_bad = ratio * s_k + ratio - 1, ratio * s_k + ratio
    for s_q, expect_ok in ((s_q_ok, True), (s_q_bad, False)):
        q = torch.randn(1, s_q, h_q, d, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, s_k, 1, d, dtype=torch.bfloat16, device=device)
        w = torch.ones(1, s_q, h_q, dtype=torch.bfloat16, device=device)
        kwargs = dict(top_k=top_k, ratio=ratio, topk_indices_global=False, return_softmax=False, deterministic=True)
        if layout == "thd":
            cu = lambda n: torch.tensor([0, n], dtype=torch.int32, device=device)  # noqa: E731
            call = lambda: DSA.indexer_forward_top_k_wrapper(  # noqa: E731
                q[0], k[0], w[0], cu_seqlens_q=cu(s_q), cu_seqlens_k=cu(s_k), max_seqlen_q=s_q, max_seqlen_k=s_k, **kwargs
            )
        else:
            call = lambda: DSA.indexer_forward_top_k_wrapper(q, k, w, **kwargs)  # noqa: E731
        if not expect_ok:
            with pytest.raises(ValueError, match=r"must be <= .*ratio \+ \(ratio - 1\)"):
                call()
            continue
        result = call()
        torch.cuda.synchronize()
        dense_ref = ref_indexer_forward(q, k, w, ratio, compute_dtype=torch.float64)
        indices = result["indices"].view(1, s_q, top_k)
        check_ref_compressed_topk(dense_ref, indices, result["logits"].view(1, s_q, top_k), top_k, atol=1e-4, rtol=1e-4)
        # the last ratio - 1 rows all see exactly s_k blocks
        assert bool((indices[0, -(ratio - 1) :] >= 0).all())


@pytest.mark.L0
@torch_fork_set_rng(seed=53)
@pytest.mark.parametrize("n_blocks", [1, 7, 512, 513, 8192, 65536])
def test_DSA_compressed_indexer_forward_four_heads_top512_over_n_blocks(n_blocks):
    """The 4-head block selector at top_k = 512 over n_blocks compressed keys: the selected set equals
    torch.topk on the fp64 reference up to ties at the 512th score, -1 padded below 512 visible blocks. Short
    key lengths run from position 0 (the first three rows see no block, the identity region selects every
    visible block); the long ones score the last rows of the sequence through a causal offset so the dense
    reference stays small while every row sees up to n_blocks keys."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    ratio, top_k, h_q, d = 4, 512, 4, 128
    if n_blocks <= 512:
        s_q, offset = ratio * n_blocks + ratio - 1, 0
    else:
        s_q = 128 if n_blocks <= 8192 else 64
        offset = ratio * n_blocks - s_q
    q = torch.randn(1, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, n_blocks, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.ones(1, s_q, h_q, dtype=torch.bfloat16, device=device)
    sm_scale = d**-0.5
    q_causal_offsets = torch.tensor([offset], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        sm_scale=sm_scale,
        q_causal_offsets=q_causal_offsets,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
    )
    torch.cuda.synchronize()

    dense_ref = ref_indexer_forward(q, k, w, ratio, q_causal_offsets=q_causal_offsets, compute_dtype=torch.float64) * sm_scale
    indices = result["indices"].view(s_q, top_k)
    if top_k <= n_blocks:
        check_ref_compressed_topk(dense_ref, indices.unsqueeze(0), result["logits"].view(1, s_q, top_k), top_k, atol=1e-4, rtol=1e-4)
    _assert_topk_sets_match(dense_ref.view(s_q, n_blocks), indices, top_k, tie_tol=1e-4, logits=result["logits"].view(s_q, top_k))
    visible = torch.isfinite(dense_ref.view(s_q, n_blocks)).sum(dim=-1)
    assert int(visible[-1]) == n_blocks
    if n_blocks <= 512:
        # identity region: every visible block is selected, the rest of the row is -1
        assert torch.equal((indices >= 0).sum(dim=-1), visible)
    else:
        # a row that sees >= 512 blocks fills every slot; the earlier rows of the window are padded
        full = visible >= top_k
        assert bool(full[-1]) and bool((indices[full] >= 0).all())
        assert torch.equal((indices[~full] >= 0).sum(dim=-1), visible[~full])


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
@pytest.mark.parametrize("h_q", [4, 8, 16])
def test_DSA_compressed_indexer_forward_bshd_mxfp8_small_head_groups(h_q):
    """MXFP8 at 4 / 8 / 16 heads per KV head: the kernel's 128-row tile packs 32 / 16 / 8 query tokens and each epilogue
    warpgroup reduces its 64-column half; random per-32 E8M0 scales (the scale packing is per 128 packed rows,
    ``token * group + head``, for every group), a query tail past the last complete block and a per-batch causal
    offset, against the fp64 reference on the dequantized values. The online LSE per token rides along (``return_lse``):
    its running max / sum are per token of the epilogue warpgroup's half, checked against ``logsumexp`` of the reference
    row, ``-inf`` exactly where a token sees no block."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    scale_utils = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils")

    device = torch.device("cuda")
    b, s_q, s_k, d = 2, 133, 64, 128
    ratio, top_k, sm_scale = 4, 24, d**-0.5
    q_ref = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(b, s_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale((b, s_q, h_q, d // 32), device=device, seed=61 + h_q, exponent_min=-2, exponent_max=3)
    k_scale_logical = make_random_mxfp8_scale((b, s_k, 1, d // 32), device=device, seed=67 + h_q, exponent_min=-2, exponent_max=3)
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_causal_offsets = torch.tensor([0, 96], dtype=torch.int32, device=device)

    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        sm_scale=sm_scale,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q),
        k_scale=scale_utils.pack_k_scale_bshd(k_scale_logical),
        topk_indices_global=False,
        return_softmax=False,
        return_lse=True,
        deterministic=True,
    )
    torch.cuda.synchronize()

    dense_ref = ref_indexer_forward(q_deq, k_deq, w, ratio, q_causal_offsets=q_causal_offsets, compute_dtype=torch.float64) * sm_scale
    check_ref_compressed_topk(dense_ref, result["indices"], result["logits"], top_k, atol=2e-3, rtol=2e-3)
    _assert_topk_sets_match(dense_ref.view(b * s_q, s_k), result["indices"].view(b * s_q, top_k), top_k, tie_tol=2e-3)
    lse_ref = torch.logsumexp(dense_ref, dim=-1)  # fp32 of the fp64 reference rows
    assert result["lse"].shape == (b, s_q)
    assert torch.equal(torch.isfinite(result["lse"]), torch.isfinite(lse_ref))
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(result["lse"][finite], lse_ref[finite], atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=71)
def test_DSA_compressed_indexer_forward_thd_mxfp8_unit_scales_is_the_plain_e4m3_scorer():
    """THD at 4 heads per KV head with ALL-ONES E8M0 scales (the fp8 indexer cache: plain e4m3 inputs, no scales): the
    Q scale prefix rounds every sequence up to 32 tokens (128 packed rows), the K prefix to 128 keys; sequences with a
    tail past the last complete block, a one-token sequence, a chunked one. The selected sets equal the fp64 top-k of
    the e4m3 VALUES up to ties, and the BF16 kernel on the (exactly) upcast values selects the same sets: the MXFP8 arm
    with unit scales computes the same function as bf16 on the upcast inputs."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    # (seqlen_q, seqlen_k, q_causal_offset): seqlen_q <= seqlen_k * 4 + 3 under offset 0
    shapes = [(67, 16, 0), (1, 1, 0), (30, 8, 0), (40, 40, 120), (135, 160, 0)]
    ratio, top_k, h_q, d = 4, 12, 4, 128
    cu_q = torch.tensor([0, *torch.tensor([s[0] for s in shapes]).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor([s[1] for s in shapes]).cumsum(0).tolist()], dtype=torch.int32, device=device)
    q_causal_offsets = torch.tensor([s[2] for s in shapes], dtype=torch.int32, device=device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    w = torch.ones(total_q, h_q, dtype=torch.bfloat16, device=device)
    sm_scale = d**-0.5
    unit_q = torch.ones(total_q, h_q, d // 32, device=device).to(torch.float8_e8m0fnu)
    unit_k = torch.ones(total_k, 1, d // 32, device=device).to(torch.float8_e8m0fnu)
    q_scale, k_scale, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(unit_q, unit_k, cu_q, cu_k, h_q, q_alignment=32, k_alignment=128)
    assert q_scale.shape[1] % 128 == 0 and int(cu_q_scale[-1]) * h_q == q_scale.shape[1]
    # the packer zero-fills the PADDED rows of each sequence's span; every logical scale byte is 127 (= 2^0)
    for blob, logical_rows in ((q_scale, total_q * h_q), (k_scale, total_k)):
        bytes_ = blob.view(torch.uint8)
        assert bool(((bytes_ == 127) | (bytes_ == 0)).all()) and int((bytes_ == 127).sum()) == logical_rows * (d // 32)
    common = dict(
        top_k=top_k,
        ratio=ratio,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(s[0] for s in shapes),
        max_seqlen_k=max(s[1] for s in shapes),
        q_causal_offsets=q_causal_offsets,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
    )

    fp8 = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
        **common,
    )
    upcast = DSA.indexer_forward_top_k_wrapper(q.to(torch.bfloat16), k.to(torch.bfloat16), w, **common)
    torch.cuda.synchronize()
    assert fp8["indices"].shape == upcast["indices"].shape == (total_q, top_k)

    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (s_q, s_k, offset) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        dense_ref = (
            ref_indexer_forward(
                q[q0:q1].unsqueeze(0),
                k[k0:k1].unsqueeze(0),
                w[q0:q1].unsqueeze(0),
                ratio,
                q_causal_offsets=q_causal_offsets[batch : batch + 1],
                compute_dtype=torch.float64,
            )
            * sm_scale
        )
        for result in (fp8, upcast):
            indices = result["indices"][q0:q1]
            if top_k <= s_k:
                check_ref_compressed_topk(dense_ref, indices.unsqueeze(0), result["logits"][q0:q1].unsqueeze(0), top_k, atol=1e-4, rtol=1e-4)
            _assert_topk_sets_match(dense_ref.squeeze(0), indices, top_k, tie_tol=1e-4, logits=result["logits"][q0:q1])
        if offset == 0:
            assert bool((fp8["indices"][q0 : q0 + min(3, s_q)] == -1).all())
    # the two kernels select the same sets: an id in exactly one of them scores within an fp32 rounding of the k-th
    valid = fp8["indices"] >= 0
    assert torch.equal(valid, upcast["indices"] >= 0)
    rows = torch.arange(total_q, device=device)[:, None].expand_as(valid)
    for a, b_ in ((fp8, upcast), (upcast, fp8)):
        in_a = torch.zeros(total_q, max(s[1] for s in shapes) + 1, dtype=torch.bool, device=device)
        in_a[rows[valid], a["indices"][valid].long()] = True
        in_b = torch.zeros_like(in_a)
        in_b[rows[valid], b_["indices"][valid].long()] = True
        only_a = in_a & ~in_b
        if bool(only_a.any()):
            kth = torch.where(valid, a["logits"], torch.full_like(a["logits"], float("inf"))).min(dim=-1).values
            gathered = torch.zeros_like(in_a, dtype=torch.float32)
            gathered[rows[valid], a["indices"][valid].long()] = a["logits"][valid]
            assert bool(
                ((gathered - kth[:, None]).abs()[only_a] <= 1e-4).all()
            ), "the MXFP8 arm with unit scales and the bf16 kernel on the upcast values differ beyond a tie"


@pytest.mark.L0
def test_indexer_forward_support_tables_are_the_shared_contract():
    """Host-only: the head-group set, the per-group tile width and the query-tail geometry bound every entry
    point of indexer_forward applies come from one module, and read as documented."""
    try:
        from cudnn.deepseek_sparse_attention.indexer_forward import _support
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    assert _support.supported_qhead_per_kv_head("bf16") == (4, 8, 16, 32, 64)
    assert _support.supported_qhead_per_kv_head("mxfp8") == (4, 8, 16, 32, 64)
    with pytest.raises(ValueError, match=r"qhead_per_kv_head in \(4, 8, 16, 32, 64\), got 12"):
        _support.validate_qhead_per_kv_head(12, "bf16")
    with pytest.raises(ValueError, match=r"precision='mxfp8' indexer requires qhead_per_kv_head in \(4, 8, 16, 32, 64\), got 12"):
        _support.validate_qhead_per_kv_head(12, "mxfp8")
    # the default 128-wide tile resolves to (qhead_per_kv_head x tokens) per path; 32 / 64 keep their measured picks
    dense = {g: _support.resolve_m_block_size(128, g, 128, compressed=False, path="t") for g in (4, 8, 16, 32, 64)}
    compressed = {g: _support.resolve_m_block_size(128, g, 128, compressed=True, path="t") for g in (4, 8, 16, 32, 64)}
    assert dense == {4: 32, 8: 64, 16: 128, 32: 128, 64: 128}
    assert compressed == {4: 32, 8: 64, 16: 128, 32: 64, 64: 128}
    assert all(m // g <= 8 for g, m in dense.items()) and all(m // g <= 8 for g, m in compressed.items())
    # an explicit width past the cap is refused, a smaller one kept
    with pytest.raises(ValueError, match="at most 8 q tokens per tile"):
        _support.resolve_m_block_size(64, 4, 128, compressed=True, path="t")
    assert _support.resolve_m_block_size(16, 4, 128, compressed=True, path="t") == 16
    # a query may trail the last complete block by ratio - 1 tokens, no more
    _support.check_q_covered_by_k(4 * 32 + 3, 32, 4)
    with pytest.raises(ValueError, match=r"seqlen_q \(132\) must be <= seqlen_k \* ratio \+ \(ratio - 1\) \(131\)"):
        _support.check_q_covered_by_k(4 * 32 + 4, 32, 4)
    _support.check_q_covered_by_k(5, 5, 1)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        _support.check_q_covered_by_k(6, 5, 1, what_q="max_seqlen_q", what_k="max_seqlen_k")


@pytest.mark.L0
@torch_fork_set_rng(seed=35)
def test_compressed_denom_slot_compiled_out():
    """Without LSE the stage-1 kernel takes denom=None (Rule 8, R3): no placeholder pool, and a fully pre-allocated call allocates nothing."""
    _require_sm100()
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_forward import _compressed_top_k_sm100 as compressed_impl
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    for name in ("_denom_placeholder_cache", "_denom_placeholder_cache_lock", "_get_fwd_unified_denom_placeholder"):
        assert not hasattr(compressed_impl, name), f"{name} still exists"

    device = torch.device("cuda")
    b, s_q, s_k, h_q, d = 2, 128, 64, 64, 128
    ratio, top_k = 4, 16
    q = torch.randn((b, s_q, h_q, d), dtype=torch.bfloat16, device=device)
    k = torch.randn((b, s_k, 1, d), dtype=torch.bfloat16, device=device)
    w = torch.randn((b, s_q, h_q), dtype=torch.bfloat16, device=device).abs() * 0.1
    cand = torch.empty(DSA.compress_topk_cand_buffer_size(b, s_q, s_k, ratio, microbatch_rows=0), dtype=torch.float32, device=device)
    out_indices = torch.empty((b, s_q, top_k), dtype=torch.int32, device=device)
    out_logits = torch.empty((b, s_q, top_k), dtype=torch.float32, device=device)

    def run():
        return DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=top_k,
            ratio=ratio,
            microbatch_rows=0,
            topk_indices_global=False,
            return_softmax=False,
            cand_buffer=cand,
            out_indices=out_indices,
            out_logits=out_logits,
        )

    run()  # lazy compile with denom=None
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before

    result = run()
    torch.cuda.synchronize()
    check_ref_compressed_topk(ref_indexer_forward(q, k, w, ratio), result["indices"], result["logits"], top_k, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=29)
@pytest.mark.parametrize("deterministic", [False, True])
def test_DSA_compressed_indexer_forward_bshd_cand_2d(deterministic):
    _require_sm100()
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    b, s_q, s_k, h_q, d = 2, 128, 32, 64, 128
    ratio, top_k = 4, 16
    q = torch.randn((b, s_q, h_q, d), dtype=torch.bfloat16, device=device)
    k = torch.randn((b, s_k, 1, d), dtype=torch.bfloat16, device=device)
    w = torch.randn((b, s_q, h_q), dtype=torch.bfloat16, device=device).abs() * 0.1
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        topk_indices_global=False,
        deterministic=deterministic,
        stream=cuda.CUstream(side.cuda_stream),
    )
    side.synchronize()

    dense_ref = ref_indexer_forward(q, k, w, ratio)
    check_ref_compressed_topk(
        dense_ref,
        result["indices"],
        result["logits"],
        top_k,
        atol=2e-3,
        rtol=2e-3,
    )
    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])


@pytest.mark.L0
@torch_fork_set_rng(seed=30)
def test_DSA_compressed_indexer_forward_single_launch_lse():
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    b, s_q, s_k, h_q, d = 1, 1, 512, 64, 128
    ratio, top_k = 4, 32
    q = torch.randn((b, s_q, h_q, d), dtype=torch.bfloat16, device=device)
    k = torch.randn((b, s_k, 1, d), dtype=torch.bfloat16, device=device)
    w = torch.randn((b, s_q, h_q), dtype=torch.bfloat16, device=device).abs() * 0.1
    q_causal_offsets = torch.tensor([s_k * ratio - s_q], dtype=torch.int32, device=device)
    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        q_causal_offsets=q_causal_offsets,
        topk_indices_global=False,
        return_lse=True,
    )
    torch.cuda.synchronize()

    dense_ref = ref_indexer_forward(q, k, w, ratio, q_causal_offsets=q_causal_offsets)
    check_ref_compressed_topk(
        dense_ref,
        result["indices"],
        result["logits"],
        top_k,
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        result["lse"],
        torch.logsumexp(dense_ref, dim=-1),
        atol=1e-2,
        rtol=1e-2,
    )
    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])


@pytest.mark.L0
@torch_fork_set_rng(seed=33)
def test_DSA_compressed_indexer_forward_microbatch_strided_outputs():
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    # bs > 1 makes each per-window output slice non-contiguous across batches.
    b, s_q, s_k, h_q, d = 2, 256, 64, 64, 128
    ratio, top_k, microbatch_rows = 4, 16, 128
    q = torch.randn((b, s_q, h_q, d), dtype=torch.bfloat16, device=device)
    k = torch.randn((b, s_k, 1, d), dtype=torch.bfloat16, device=device)
    w = torch.randn((b, s_q, h_q), dtype=torch.bfloat16, device=device).abs() * 0.1
    cand_floats = DSA.compress_topk_cand_buffer_size(
        b,
        s_q,
        s_k,
        ratio,
        microbatch_rows=microbatch_rows,
    )
    cand = torch.empty(cand_floats, dtype=torch.float32, device=device)
    out_indices = torch.empty((b, s_q, top_k), dtype=torch.int32, device=device)
    out_logits = torch.empty((b, s_q, top_k), dtype=torch.float32, device=device)
    out_softmax = torch.empty((b, s_q, top_k), dtype=torch.float32, device=device)
    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        topk_indices_global=False,
        microbatch_rows=microbatch_rows,
        cand_buffer=cand,
        out_indices=out_indices,
        out_logits=out_logits,
        softmax_out=out_softmax,
    )
    torch.cuda.synchronize()
    assert result["indices"].data_ptr() == out_indices.data_ptr()
    assert result["logits"].data_ptr() == out_logits.data_ptr()
    assert result["softmax"].data_ptr() == out_softmax.data_ptr()
    check_ref_compressed_topk(
        ref_indexer_forward(q, k, w, ratio),
        result["indices"],
        result["logits"],
        top_k,
        atol=2e-3,
        rtol=2e-3,
    )
    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])


@pytest.mark.L0
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,batch,ratio,top_k,num_buckets",
    [
        (64, 64, 1, 1, 8, 3),
        (128, 128, 2, 1, 16, 4),
        (256, 64, 1, 4, 32, 5),
        (96, 96, 1, 1, 8, 2),
    ],
)
def test_DSA_compressed_stage2_deterministic_ties(
    seqlen_q,
    seqlen_k,
    batch,
    ratio,
    top_k,
    num_buckets,
):
    """Deterministic stage-2 selects the stable smallest-index tie set."""
    _require_sm100()
    try:
        from cudnn.deepseek_sparse_attention.indexer_top_k import compress_top_k_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1234)
    dense = torch.randint(
        0,
        num_buckets,
        (batch, seqlen_q, seqlen_k),
        generator=generator,
        device=device,
    ).float()
    candidates = compress_top_k_sm100.build_compact_buffer(dense, ratio, q_causal_offset=0)

    default_indices, default_logits = compress_top_k_sm100.compress_stage2_topk(
        candidates,
        batch,
        seqlen_q,
        seqlen_k,
        top_k,
        ratio,
        deterministic=False,
    )
    indices, logits = compress_top_k_sm100.compress_stage2_topk(
        candidates,
        batch,
        seqlen_q,
        seqlen_k,
        top_k,
        ratio,
        deterministic=True,
    )
    cache_size = len(compress_top_k_sm100._compile_cache)
    indices_again, logits_again = compress_top_k_sm100.compress_stage2_topk(
        candidates,
        batch,
        seqlen_q,
        seqlen_k,
        top_k,
        ratio,
        deterministic=True,
    )
    assert len(compress_top_k_sm100._compile_cache) == cache_size
    torch.cuda.synchronize()

    matching_cache_keys = [key for key in compress_top_k_sm100._compile_cache if key[:9] == (batch, seqlen_q, seqlen_k, top_k, ratio, 512, False, False, False)]
    assert {key[-1] for key in matching_cache_keys} == {False, True}

    rows = torch.arange(seqlen_q, device=device)
    row_end = ((rows + 1) // ratio).clamp(min=0, max=seqlen_k)
    effective_k = row_end.clamp(max=top_k)
    columns = torch.arange(seqlen_k, device=device)
    masked = dense.masked_fill(columns[None, None, :] >= row_end[None, :, None], float("-inf"))
    valid_slot = torch.arange(top_k, device=device)[None, None, :] < effective_k[None, :, None]
    stable_indices = torch.argsort(masked, dim=-1, descending=True, stable=True)[..., :top_k].to(torch.int32)
    reference_indices = torch.where(valid_slot, stable_indices, torch.full_like(stable_indices, -1))
    reference_logits = masked.topk(top_k, dim=-1).values

    assert torch.equal(indices.sort(dim=-1).values, reference_indices.sort(dim=-1).values)
    assert torch.equal(indices.sort(dim=-1).values, indices_again.sort(dim=-1).values)
    assert torch.equal(
        logits.sort(dim=-1, descending=True).values,
        logits_again.sort(dim=-1, descending=True).values,
    )
    assert torch.equal(logits.sort(dim=-1, descending=True).values, reference_logits)
    assert torch.equal(default_logits.sort(dim=-1, descending=True).values, reference_logits)

    boundary_slot = (effective_k - 1).clamp(min=0)
    boundary = reference_logits.gather(-1, boundary_slot.view(1, -1, 1).expand(batch, -1, 1))
    equal_count = (masked == boundary).sum(dim=-1)
    assert int(((row_end[None, :] > top_k) & (equal_count > 1)).sum()) > 0
    assert default_indices.shape == indices.shape


@pytest.mark.L0
@pytest.mark.parametrize("pattern", ["zero", "late", "mixed"])
@pytest.mark.parametrize(
    "seqlen_k,top_k,block_threads",
    [(4096, 64, 512), (2047, 64, 128), (2048, 64, 256), (2049, 2048, 512), (8193, 2048, 32), (8193, 2048, 1024), (8193, 1, 512), (31, 64, 64)],
)
def test_DSA_compressed_stage2_deterministic_shrink_fallback(seqlen_k, top_k, block_threads, pattern):
    """Preserve the stable tie set across shrink overflow, CTA tiles, and padding."""
    _require_sm100()
    try:
        from cudnn.deepseek_sparse_attention.indexer_top_k.compress_top_k_sm100 import compress_stage2_topk
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    candidates = torch.zeros(seqlen_k, dtype=torch.float32, device=device)
    if pattern == "late":
        candidates[: seqlen_k // 2] = -1
    elif pattern == "mixed":
        positions = torch.arange(seqlen_k, device=device)
        candidates = torch.where(positions % 8 < 4, 1.0, torch.where(positions % 8 == 4, 3.0 + positions / 32768, -positions.float()))
    cand_batch_offsets = torch.tensor([0, seqlen_k], dtype=torch.int64, device=device)
    q_causal_offsets = torch.tensor([seqlen_k - 1], dtype=torch.int32, device=device)

    def run(**kwargs):
        return compress_stage2_topk(
            candidates,
            1,
            1,
            seqlen_k,
            top_k,
            1,
            block_threads=block_threads,
            cand_batch_offsets=cand_batch_offsets,
            q_causal_offsets=q_causal_offsets,
            **kwargs,
        )

    selected = candidates.argsort(descending=True, stable=True)[:top_k]
    expected = torch.full((1, 1, top_k), -1, dtype=torch.int32, device=device)
    expected[..., : selected.numel()] = selected.int()
    expected_logits = torch.full(expected.shape, -torch.inf, device=device)
    expected_logits[..., : selected.numel()] = candidates[selected]
    _, default_logits = run(deterministic=False)
    torch.testing.assert_close(default_logits.sort(-1).values, expected_logits.sort(-1).values, atol=0, rtol=0)
    indices, logits = run(deterministic=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run(deterministic=True, out_indices=indices, out_logits=logits)
        for _ in range(3):
            indices.fill_(-99)
            logits.fill_(99)
            graph.replay()
            torch.testing.assert_close(indices.sort(-1).values, expected.sort(-1).values, atol=0, rtol=0)
            valid = indices >= 0
            torch.testing.assert_close(logits[valid], candidates[indices[valid].long()], atol=0, rtol=0)
            assert torch.isneginf(logits[~valid]).all()
    finally:
        graph.reset()


@pytest.mark.L0
def test_DSA_compressed_stage2_deterministic_thd_ties(monkeypatch):
    """Varlen stage-2 applies the same stable local-index tie policy per batch."""
    _require_sm100()
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_top_k import compress_top_k_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    top_k = 8
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, 64, 160], dtype=torch.int32, device=device)
    cand_batch_offsets = torch.tensor([0, 64, 160], dtype=torch.int64, device=device)
    q_causal_offsets = torch.tensor([63, 95], dtype=torch.int32, device=device)
    candidates = torch.zeros(160, dtype=torch.float32, device=device)

    compress_top_k_sm100.compress_stage2_topk_varlen(
        candidates,
        cu_seqlens_q,
        cu_seqlens_k,
        cand_batch_offsets,
        total_q=2,
        max_seqlen_q=1,
        max_seqlen_k=96,
        topk=top_k,
        ratio=1,
        q_causal_offsets=q_causal_offsets,
        deterministic=False,
    )
    indices, logits = compress_top_k_sm100.compress_stage2_topk_varlen(
        candidates,
        cu_seqlens_q,
        cu_seqlens_k,
        cand_batch_offsets,
        total_q=2,
        max_seqlen_q=1,
        max_seqlen_k=96,
        topk=top_k,
        ratio=1,
        q_causal_offsets=q_causal_offsets,
        deterministic=True,
    )
    cache_size = len(compress_top_k_sm100._compile_cache)
    indices_again, _ = compress_top_k_sm100.compress_stage2_topk_varlen(
        candidates,
        cu_seqlens_q,
        cu_seqlens_k,
        cand_batch_offsets,
        total_q=2,
        max_seqlen_q=1,
        max_seqlen_k=96,
        topk=top_k,
        ratio=1,
        q_causal_offsets=q_causal_offsets,
        deterministic=True,
    )
    assert len(compress_top_k_sm100._compile_cache) == cache_size
    torch.cuda.synchronize()

    matching_cache_keys = [key for key in compress_top_k_sm100._compile_cache if key[:-1] == ("varlen", top_k, 1, 512, True, False)]
    assert {key[-1] for key in matching_cache_keys} == {False, True}

    expected = torch.arange(top_k, dtype=torch.int32, device=device).expand(2, -1)
    assert torch.equal(indices.sort(dim=-1).values, expected)
    assert torch.equal(indices.sort(dim=-1).values, indices_again.sort(dim=-1).values)
    assert torch.equal(logits, torch.zeros_like(logits))

    # Exercise the complete public THD propagation chain while spying on the
    # stage-2 keyword so a dropped deterministic flag cannot pass by chance.
    observed_deterministic = []
    original_stage2 = compress_top_k_sm100.compress_stage2_topk_varlen

    def stage2_spy(*args, **kwargs):
        observed_deterministic.append(kwargs.get("deterministic"))
        return original_stage2(*args, **kwargs)

    monkeypatch.setattr(compress_top_k_sm100, "compress_stage2_topk_varlen", stage2_spy)
    h_q, head_dim = 64, 128
    q = torch.zeros((2, h_q, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.zeros((160, 1, head_dim), dtype=torch.bfloat16, device=device)
    w = torch.ones((2, h_q), dtype=torch.bfloat16, device=device)
    public_result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=1,
        max_seqlen_k=96,
        q_causal_offsets=q_causal_offsets,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
    )
    torch.cuda.synchronize()

    assert observed_deterministic == [True]
    assert torch.equal(public_result["indices"].sort(dim=-1).values, expected)
    assert torch.equal(public_result["logits"], torch.zeros_like(public_result["logits"]))


@pytest.mark.L0
@torch_fork_set_rng(seed=59)
def test_DSA_compressed_indexer_forward_deterministic():
    """The public combined path preserves the deterministic tie policy."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    batch, seqlen_q, seqlen_k, h_q, head_dim = 2, 512, 128, 64, 128
    ratio, top_k = 4, 64
    q = torch.randn((batch, seqlen_q, h_q, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((batch, seqlen_k, 1, head_dim), dtype=torch.bfloat16, device=device)
    w = torch.randn((batch, seqlen_q, h_q), dtype=torch.bfloat16, device=device).abs() * 0.1

    dense = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, sm_scale=head_dim**-0.5)["scores"]

    def run(deterministic):
        return DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=top_k,
            ratio=ratio,
            sm_scale=head_dim**-0.5,
            topk_indices_global=False,
            return_softmax=deterministic,
            deterministic=deterministic,
        )

    result = run(True)
    result_again = run(True)
    default_result = run(False)
    torch.cuda.synchronize()

    check_ref_compressed_topk(dense, result["indices"], result["logits"], top_k, atol=2e-3, rtol=2e-3)
    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])
    assert torch.equal(result["indices"].sort(dim=-1).values, result_again["indices"].sort(dim=-1).values)
    assert torch.equal(
        result["logits"].sort(dim=-1, descending=True).values,
        result_again["logits"].sort(dim=-1, descending=True).values,
    )
    check_ref_compressed_topk(
        dense,
        default_result["indices"],
        default_result["logits"],
        top_k,
        atol=2e-3,
        rtol=2e-3,
    )


@pytest.mark.L0
def test_DSA_compressed_indexer_forward_deterministic_microbatch():
    """Windowed BSHD forwards the deterministic policy into each stage-2 launch."""
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(19)
    batch, seqlen_q, seqlen_k, h_q, head_dim = 2, 1024, 256, 64, 128
    ratio, top_k, microbatch_rows = 4, 64, 256
    codebook = torch.randn(8, head_dim, generator=generator, device=device, dtype=torch.bfloat16)
    code_ids = torch.randint(0, 8, (batch, seqlen_k), generator=generator, device=device)
    q = torch.randn(batch, seqlen_q, h_q, head_dim, generator=generator, device=device, dtype=torch.bfloat16)
    k = codebook[code_ids].unsqueeze(2).contiguous()
    w = torch.randn(batch, seqlen_q, h_q, generator=generator, device=device, dtype=torch.bfloat16).abs() * 0.1

    dense = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, sm_scale=head_dim**-0.5)["scores"]

    def run(rows):
        return DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=top_k,
            ratio=ratio,
            sm_scale=head_dim**-0.5,
            microbatch_rows=rows,
            topk_indices_global=False,
            return_softmax=False,
            deterministic=True,
        )

    windowed = run(microbatch_rows)
    windowed_again = run(microbatch_rows)
    single_launch = run(0)
    torch.cuda.synchronize()

    rows = torch.arange(seqlen_q, device=device)
    row_end = ((rows + 1) // ratio).clamp(min=0, max=seqlen_k)
    effective_k = row_end.clamp(max=top_k)
    valid_slot = torch.arange(top_k, device=device)[None, None, :] < effective_k[None, :, None]
    stable_indices = torch.argsort(dense, dim=-1, descending=True, stable=True)[..., :top_k].to(torch.int32)
    reference_indices = torch.where(valid_slot, stable_indices, torch.full_like(stable_indices, -1))

    assert torch.equal(windowed["indices"].sort(dim=-1).values, single_launch["indices"].sort(dim=-1).values)
    assert torch.equal(windowed["indices"].sort(dim=-1).values, reference_indices.sort(dim=-1).values)
    assert torch.equal(windowed["indices"].sort(dim=-1).values, windowed_again["indices"].sort(dim=-1).values)

    reference_logits = dense.topk(top_k, dim=-1).values
    boundary_slot = (effective_k - 1).clamp(min=0)
    boundary = reference_logits.gather(-1, boundary_slot.view(1, -1, 1).expand(batch, -1, 1))
    equal_count = (dense == boundary).sum(dim=-1)
    assert int(((row_end[None, :] > top_k) & (equal_count > 1)).sum()) > 0


@pytest.mark.L0
@torch_fork_set_rng(seed=31)
@pytest.mark.parametrize("h_q", [32, 64])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
def test_DSA_compressed_indexer_forward_bshd_preallocated_lse(h_q, weight_dtype):
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    b, s_q, s_k, h_kv, d = 2, 128, 64, 1, 128
    ratio, top_k, sm_scale = 4, 32, d**-0.5
    q = torch.randn(b, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    if weight_dtype == torch.float32:
        w = (w.abs() + 1).float() + 2**-10
    # Different per-batch offsets exercise tight, non-uniform candidate slabs.
    q_causal_offsets = torch.tensor([0, 128], dtype=torch.int32, device=device)

    cand_floats = DSA.compress_topk_cand_buffer_size(
        b,
        s_q,
        s_k,
        ratio,
        microbatch_rows=0,
        return_lse=True,
        q_causal_offsets=q_causal_offsets,
    )
    cand = torch.empty(cand_floats, dtype=torch.float32, device=device)
    out_indices = torch.empty((b, s_q, top_k), dtype=torch.int32, device=device)
    out_logits = torch.empty((b, s_q, top_k), dtype=torch.float32, device=device)
    out_softmax = torch.empty((b, s_q, top_k), dtype=torch.float32, device=device)
    lse_out = torch.empty((b, s_q), dtype=torch.float32, device=device)

    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        sm_scale=sm_scale,
        q_causal_offsets=q_causal_offsets,
        microbatch_rows=0,
        cand_buffer=cand,
        out_indices=out_indices,
        out_logits=out_logits,
        softmax_out=out_softmax,
        return_lse=True,
        lse_out=lse_out,
    )
    torch.cuda.synchronize()

    assert result["indices"].data_ptr() == out_indices.data_ptr()
    assert result["logits"].data_ptr() == out_logits.data_ptr()
    assert result["softmax"].data_ptr() == out_softmax.data_ptr()
    assert result["lse"].data_ptr() == lse_out.data_ptr()

    dense_ref = (
        ref_indexer_forward(
            q,
            k,
            w,
            ratio,
            q_causal_offsets=q_causal_offsets,
            compute_dtype=torch.float64,
        )
        * sm_scale
    )
    local_indices = _bshd_global_to_local(result["indices"], s_k)
    check_ref_compressed_topk(
        dense_ref,
        local_indices,
        result["logits"],
        top_k,
        atol=1e-4 if weight_dtype == torch.float32 else 2e-3,
        rtol=1e-4 if weight_dtype == torch.float32 else 2e-3,
    )
    lse_ref = torch.logsumexp(dense_ref, dim=-1)
    assert torch.equal(torch.isfinite(result["lse"]), torch.isfinite(lse_ref))
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(
        result["lse"][finite],
        lse_ref[finite],
        atol=1e-2,
        rtol=1e-2,
    )
    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])


@pytest.mark.L0
@torch_fork_set_rng(seed=37)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
def test_DSA_compressed_indexer_forward_thd_preallocated_global_indices(weight_dtype):
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    shapes = [(64, 32), (96, 32)]
    ratio, top_k, h_q, h_kv, d = 4, 16, 64, 1, 128
    q_lengths = [shape[0] for shape in shapes]
    k_lengths = [shape[1] for shape in shapes]
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
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    if weight_dtype == torch.float32:
        w = (w.abs() + 1).float() + 2**-10

    cand_offsets, cand_floats = DSA.compress_topk_cand_buffer_size_thd(
        cu_q,
        cu_k,
        ratio,
    )
    cand = torch.empty(cand_floats, dtype=torch.float32, device=device)
    out_indices = torch.empty((total_q, top_k), dtype=torch.int32, device=device)
    out_logits = torch.empty((total_q, top_k), dtype=torch.float32, device=device)
    out_softmax = torch.empty((total_q, top_k), dtype=torch.float32, device=device)
    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        topk_indices_global=True,
        cand_buffer=cand,
        cand_batch_offsets=cand_offsets,
        out_indices=out_indices,
        out_logits=out_logits,
        softmax_out=out_softmax,
    )
    torch.cuda.synchronize()

    assert result["indices"].data_ptr() == out_indices.data_ptr()
    assert result["logits"].data_ptr() == out_logits.data_ptr()
    assert result["softmax"].data_ptr() == out_softmax.data_ptr()
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (s_q, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        indices = result["indices"][q0:q1]
        valid = indices >= 0
        assert bool((((indices >= k0) & (indices < k1)) | ~valid).all())
        local_indices = torch.where(
            valid,
            indices.to(torch.int64) - k0,
            indices.to(torch.int64),
        ).to(torch.int32)
        dense_ref = ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            compute_dtype=torch.float64,
        )
        check_ref_compressed_topk(
            dense_ref,
            local_indices.unsqueeze(0),
            result["logits"][q0:q1].unsqueeze(0),
            top_k,
            atol=1e-4 if weight_dtype == torch.float32 else 2e-3,
            rtol=1e-4 if weight_dtype == torch.float32 else 2e-3,
        )
    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])


@pytest.mark.L0
@torch_fork_set_rng(seed=43)
@pytest.mark.parametrize("h_q", [4, 64], ids=["h4", "h64"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_DSA_compressed_indexer_forward_thd_mxfp8_lse(deterministic, h_q):
    """THD MXFP8 with the online LSE per token, at 64 heads (one token per epilogue warpgroup) and at 4 heads (the small-group
    epilogue: 16 tokens per warpgroup, each carrying its own running max / sum -- the heaviest rendering of that epilogue).
    The 256-token scale alignment is a multiple of both groups' minimum (``128 // gcd(128, h_q)`` = 2, resp. 32), so the
    padded prefixes are the same for both; the LSE is checked against ``logsumexp`` of the dense MXFP8 scorer's rows."""
    _require_sm100()
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils import (
            pack_k_scale_bshd,
            pack_q_scale_bshd,
        )
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    shapes = [(127, 32), (129, 64)]
    ratio, top_k, h_kv, d = 4, 16, 1, 128
    q_lengths = [shape[0] for shape in shapes]
    k_lengths = [shape[1] for shape in shapes]
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
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale((total_q, h_q, d // 32), device=device, seed=47)
    k_scale_logical = make_random_mxfp8_scale((total_k, h_kv, d // 32), device=device, seed=53)
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
    lse_out = torch.empty(total_q, dtype=torch.float32, device=device)

    result = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        sm_scale=d**-0.5,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
        topk_indices_global=False,
        return_lse=True,
        lse_out=lse_out,
        deterministic=deterministic,
    )
    assert result["lse"].data_ptr() == lse_out.data_ptr()

    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (s_q, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        # .clone(): a per-sequence slice of the packed slab is 16-byte aligned at 64 heads but not at 4 (127 tokens x
        # 4 heads x 2 bytes), and the dense reference kernel takes 16-byte-aligned tensors
        dense = DSA.indexer_forward_wrapper(
            q[q0:q1].unsqueeze(0).clone(),
            k[k0:k1].unsqueeze(0).clone(),
            w[q0:q1].unsqueeze(0).clone(),
            ratio=ratio,
            sm_scale=d**-0.5,
            precision="mxfp8",
            q_scale=pack_q_scale_bshd(
                q_scale_logical[q0:q1].unsqueeze(0).clone(),
                qhead_per_kv_head=h_q,
            ),
            k_scale=pack_k_scale_bshd(k_scale_logical[k0:k1].unsqueeze(0)),
        )["scores"]
        check_ref_compressed_topk(
            dense,
            result["indices"][q0:q1].unsqueeze(0),
            result["logits"][q0:q1].unsqueeze(0),
            top_k,
            atol=2e-3,
            rtol=2e-3,
        )
        lse_ref = torch.logsumexp(dense, dim=-1).squeeze(0)
        finite = torch.isfinite(lse_ref)
        assert torch.equal(torch.isfinite(result["lse"][q0:q1]), finite)
        torch.testing.assert_close(
            result["lse"][q0:q1][finite],
            lse_ref[finite],
            atol=1e-2,
            rtol=1e-2,
        )

    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])
