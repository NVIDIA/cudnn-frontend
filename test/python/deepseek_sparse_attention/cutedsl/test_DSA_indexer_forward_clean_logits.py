# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense indexer forward ``clean_logits``: opt-out of the ``-inf`` prefill.

``clean_logits=False`` leaves every element outside each row's ratio-causal
prefix unspecified. The tests NaN-poison the output so that any consumer read
outside that prefix is visible, then require:

* the causal prefix to match the reference and the ``clean_logits=True`` run
  bit for bit;
* ``indexer_top_k`` with ``next_n=1`` and per-row lengths equal to the causal
  limit to return identical indices and values for both settings;
* the default (``clean_logits=True``) to keep overwriting every masked
  position with ``-inf``;
* ``clean_logits=False`` to be rejected where a caller ``out`` would be staged
  and copied back, and on the compressed-logits path.
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
    ref_indexer_forward,
)


def _require_dense_indexer_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("Dense indexer forward requires Hopper or Blackwell")


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 dense indexer forward required")


def _dense_interface():
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface, _interface_sm90

    return _interface_sm90 if torch.cuda.get_device_capability()[0] == 9 else _interface


def _causal_col_limits(q_lengths, k_lengths, offsets, ratio, n_rows, device):
    """Per-row written-column count ``clamp((offset + q_local + 1) // ratio, 0, s_k)``; uncovered rows get 0."""
    limits = torch.zeros(n_rows, dtype=torch.int64)
    row = 0
    for s_q, s_k, offset in zip(q_lengths, k_lengths, offsets):
        q_local = torch.arange(s_q, dtype=torch.int64)
        limits[row : row + s_q] = torch.div(offset + q_local + 1, ratio, rounding_mode="floor").clamp(0, s_k)
        row += s_q
    return limits.to(device=device, dtype=torch.int32)


def _valid_columns(limits, num_cols):
    return torch.arange(num_cols, device=limits.device)[None, :] < limits[:, None]


def _canonical_top_k(result):
    """Top-K slot order is unspecified: order each row by index (padding ``-1`` first) and carry the values along."""
    indices = result["indices"]
    order = torch.argsort(indices, dim=1, stable=True)
    return indices.gather(1, order), result["values"].gather(1, order)


def _assert_top_k_unchanged(DSA, scores_clean, scores_skip, limits, top_k):
    n_rows = scores_clean.shape[0]
    clean = DSA.indexer_top_k_wrapper(scores_clean, limits, top_k, next_n=1, return_val=True, tie_break=1)
    skip = DSA.indexer_top_k_wrapper(scores_skip, limits, top_k, next_n=1, return_val=True, tie_break=1)
    torch.cuda.synchronize()
    idx_clean, val_clean = _canonical_top_k(clean)
    idx_skip, val_skip = _canonical_top_k(skip)
    assert idx_skip.shape == (n_rows, top_k)
    assert torch.equal(idx_skip, idx_clean)
    assert torch.equal(val_skip, val_clean), "a top-k value differs (a NaN here means a read outside the causal prefix)"
    check_ref_indexer_top_k(scores_clean, limits, top_k, 1, skip["indices"], skip["values"], True, tie_break=1)


def _assert_prefix_equal_and_tail_skipped(out_clean, out_skip, valid):
    assert torch.equal(out_skip[valid], out_clean[valid])
    assert bool(torch.isneginf(out_clean[~valid]).all()), "clean_logits=True must still overwrite every masked position with -inf"
    assert bool(torch.isnan(out_skip[~valid]).any()), "clean_logits=False must not prefill the masked region"


@pytest.mark.L0
def test_DSA_indexer_forward_clean_logits_is_keyword_only():
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_forward import _interface, _interface_sm90
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    for fn in (DSA.indexer_forward_wrapper, DSA.IndexerForward.__init__, _interface.indexer_fwd, _interface_sm90.indexer_fwd):
        parameter = inspect.signature(fn).parameters["clean_logits"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, fn
        assert parameter.default is True, fn
    # The compressed path writes no dense score tensor, so it has nothing to clean.
    assert "clean_logits" not in inspect.signature(DSA.indexer_forward_top_k_wrapper).parameters


@pytest.mark.L0
@torch_fork_set_rng(seed=31)
@pytest.mark.parametrize("h_q", [4, 16, 32, 64])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("top_k", [64, 2048])
def test_DSA_indexer_forward_wrapper_clean_logits_thd_top_k(h_q, ratio, top_k):
    """THD with per-segment causal offsets, ragged K tails and rows not covered by ``cu_seqlens_q``.

    The head groups cover each SM100 tile packing (8 / 8 / 4 / 2 query tokens per tile)."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_dense_indexer_gpu()
    if torch.cuda.get_device_capability()[0] == 9 and h_q not in (16, 32, 64):
        pytest.skip("SM90 dense indexer forward serves qhead_per_kv_head in (16, 32, 64)")

    device = torch.device("cuda")
    shapes = [(1, 1), (7, 67), (64, 300), (256, 2600)]
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    offsets = [0, 0, *[s_k * ratio - s_q for s_q, s_k in shapes[2:]]]
    uncovered_rows = 3
    total_q, total_k = sum(q_lengths), sum(k_lengths)
    n_rows = total_q + uncovered_rows
    max_seqlen_k = max(k_lengths)
    assert max_seqlen_k % 4 == 0, "a caller out with a ragged row would be staged; covered by the staging test"

    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    q_causal_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    q = torch.randn(n_rows, h_q, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, 1, 128, dtype=torch.bfloat16, device=device)
    w = torch.randn(n_rows, h_q, dtype=torch.bfloat16, device=device)

    common = dict(
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
    )
    out_clean = torch.full((n_rows, max_seqlen_k), float("nan"), dtype=torch.float32, device=device)
    out_skip = torch.full_like(out_clean, float("nan"))
    assert DSA.indexer_forward_wrapper(q, k, w, out=out_clean, **common)["scores"] is out_clean
    assert DSA.indexer_forward_wrapper(q, k, w, out=out_skip, clean_logits=False, **common)["scores"] is out_skip
    torch.cuda.synchronize()

    limits = _causal_col_limits(q_lengths, k_lengths, offsets, ratio, n_rows, device)
    valid = _valid_columns(limits, max_seqlen_k)
    _assert_prefix_equal_and_tail_skipped(out_clean, out_skip, valid)

    cu_q_host, cu_k_host = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for batch, s_k in enumerate(k_lengths):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        reference = ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
            compute_dtype=torch.float64,
        ).squeeze(0)
        finite = torch.isfinite(reference)
        assert torch.equal(finite, valid[q0:q1, :s_k]), "the documented causal limit must match the reference mask"
        torch.testing.assert_close(out_skip[q0:q1, :s_k][finite], reference[finite], atol=1e-4, rtol=1e-4)

    _assert_top_k_unchanged(DSA, out_clean, out_skip, limits, top_k)


@pytest.mark.L0
@torch_fork_set_rng(seed=37)
@pytest.mark.parametrize("ratio", [1, 4])
def test_DSA_indexer_forward_wrapper_clean_logits_bshd_default_unchanged(ratio, monkeypatch):
    """BSHD without offsets (top-left causal). The default equals ``clean_logits=True`` bit for bit, and the flag is not a compile key."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_dense_indexer_gpu()

    device = torch.device("cuda")
    b, s_q, s_k, h_q = 2, 128, 512, 64
    q = torch.randn(b, s_q, h_q, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, 1, 128, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)

    interface = _dense_interface()
    monkeypatch.setattr(interface, "_compile_cache", {})
    outs = {}
    for name, options in (("default", {}), ("clean", {"clean_logits": True}), ("skip", {"clean_logits": False})):
        outs[name] = torch.full((b, s_q, s_k), float("nan"), dtype=torch.float32, device=device)
        result = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, qhead_per_kv_head=h_q, out=outs[name], **options)
        assert result["scores"] is outs[name]
    torch.cuda.synchronize()
    assert len(interface._compile_cache) == 1, "clean_logits must not be part of the compile key"

    assert torch.equal(outs["default"], outs["clean"])
    check_ref_indexer_forward(q, k, w, outs["default"], ratio)

    limits = _causal_col_limits([s_q] * b, [s_k] * b, [0] * b, ratio, b * s_q, device)
    valid = _valid_columns(limits, s_k)
    clean, skip = outs["clean"].view(b * s_q, s_k), outs["skip"].view(b * s_q, s_k)
    _assert_prefix_equal_and_tail_skipped(clean, skip, valid)
    _assert_top_k_unchanged(DSA, clean, skip, limits, top_k=64)


@pytest.mark.L0
@torch_fork_set_rng(seed=41)
@pytest.mark.parametrize("h_q", [64, 32, 8], ids=["qh64", "qh32", "qh8"])
def test_DSA_indexer_forward_wrapper_clean_logits_thd_mxfp8(h_q):
    """One head group per MXFP8 dense epilogue: single token (64), token pair (32), small group (8)."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100()

    device = torch.device("cuda")
    shapes = [(127, 32), (129, 64)]
    ratio, d = 4, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    offsets = [1, 17]
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    max_k = max(k_lengths)

    q_ref = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device)
    k_ref = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale((total_q, h_q, d // 32), device=device, seed=113, exponent_min=-2, exponent_max=3)
    k_scale_logical = make_random_mxfp8_scale((total_k, 1, d // 32), device=device, seed=127, exponent_min=-2, exponent_max=3)
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale, k_scale, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(q_scale_logical, k_scale_logical, cu_q, cu_k, h_q, q_alignment=256, k_alignment=256)
    q_causal_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

    common = dict(
        ratio=ratio,
        qhead_per_kv_head=h_q,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max_k,
        q_causal_offsets=q_causal_offsets,
        precision="mxfp8",
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
    )
    out_clean = torch.full((total_q, max_k), float("nan"), dtype=torch.float32, device=device)
    out_skip = torch.full_like(out_clean, float("nan"))
    DSA.indexer_forward_wrapper(q, k, w, out=out_clean, **common)
    DSA.indexer_forward_wrapper(q, k, w, out=out_skip, clean_logits=False, **common)
    torch.cuda.synchronize()

    limits = _causal_col_limits(q_lengths, k_lengths, offsets, ratio, total_q, device)
    valid = _valid_columns(limits, max_k)
    _assert_prefix_equal_and_tail_skipped(out_clean, out_skip, valid)

    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, s_k in enumerate(k_lengths):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        reference = ref_indexer_forward(
            q_deq[q0:q1].unsqueeze(0),
            k_deq[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
        ).squeeze(0)
        finite = torch.isfinite(reference)
        assert torch.equal(finite, valid[q0:q1, :s_k])
        torch.testing.assert_close(out_skip[q0:q1, :s_k][finite], reference[finite], atol=5e-3, rtol=5e-3)

    _assert_top_k_unchanged(DSA, out_clean, out_skip, limits, top_k=16)


@pytest.mark.L0
@torch_fork_set_rng(seed=43)
def test_DSA_indexer_forward_clean_logits_rejects_staged_out_and_compressed_path():
    """A caller ``out`` that must be staged and copied back cannot honour ``clean_logits=False``; neither can the compressed path."""
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.indexer_forward import _interface
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100()

    device = torch.device("cuda")
    b, s_q, s_k, h_q, ratio = 1, 64, 67, 64, 1
    q = torch.randn(b, s_q, h_q, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, 1, 128, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)

    # A contiguous (B, S_q, 67) out has 268-byte rows: the kernel cannot bind it, so it would be staged.
    staged = torch.full((b, s_q, s_k), float("nan"), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="clean_logits=False"):
        DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=staged, clean_logits=False)
    torch.cuda.synchronize()
    assert bool(torch.isnan(staged).all()), "the rejection must happen before any launch"

    # The default still stages and copies back; the result is the documented -inf-masked score.
    assert DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=staged)["scores"] is staged
    check_ref_indexer_forward(q, k, w, staged, ratio)

    # Without a caller out nothing is copied back into caller storage: the allocated result is returned.
    scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, clean_logits=False)["scores"]
    valid = _valid_columns(_causal_col_limits([s_q], [s_k], [0], ratio, s_q, device), s_k)
    assert scores.shape == (b, s_q, s_k)
    assert torch.equal(scores[0][valid], staged[0][valid])

    with pytest.raises(ValueError, match="is_compressed_logits"):
        _interface.indexer_fwd(q, k, w, ratio=ratio, is_compressed_logits=True, topk=16, clean_logits=False)


@pytest.mark.L0
@torch_fork_set_rng(seed=47)
def test_DSA_indexer_forward_clean_logits_padded_view_bound_directly():
    """S_k % 4 != 0 through a padded-stride view (wrapper ``out=`` and the ``IndexerForward`` plan): no staging, so ``clean_logits=False`` holds."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100()

    device = torch.device("cuda")
    b, s_q, s_k, s_k_padded, h_q, ratio = 2, 256, 1001, 1004, 64, 1
    q = torch.randn(b, s_q, h_q, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, 1, 128, dtype=torch.bfloat16, device=device)
    w = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device)
    offsets = torch.full((b,), s_k - s_q, dtype=torch.int32, device=device)
    limits = _causal_col_limits([s_q] * b, [s_k] * b, [s_k - s_q] * b, ratio, b * s_q, device)
    valid = _valid_columns(limits, s_k)

    padded_clean = torch.full((b, s_q, s_k_padded), float("nan"), dtype=torch.float32, device=device)
    padded_skip = torch.full_like(padded_clean, float("nan"))
    view_clean, view_skip = padded_clean[..., :s_k], padded_skip[..., :s_k]
    assert DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, q_causal_offsets=offsets, out=view_clean)["scores"] is view_clean
    got = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, q_causal_offsets=offsets, out=view_skip, clean_logits=False)["scores"]
    assert got is view_skip
    torch.cuda.synchronize()
    clean_rows, skip_rows = view_clean.reshape(b * s_q, s_k), view_skip.reshape(b * s_q, s_k)
    _assert_prefix_equal_and_tail_skipped(clean_rows, skip_rows, valid)
    assert bool(torch.isnan(padded_skip[..., s_k:]).all()), "the pad columns are never written"
    _assert_top_k_unchanged(DSA, clean_rows.contiguous(), skip_rows.contiguous(), limits, top_k=64)

    # The plan binds the same padded-stride view; with clean_logits=False execute() launches the score kernel only.
    plans = {}
    for name, options in (("clean", {}), ("skip", {"clean_logits": False})):
        plans[name] = torch.full((b, s_q, s_k_padded), float("nan"), dtype=torch.float32, device=device)
        plan = DSA.IndexerForward(q, k, w, plans[name], ratio=ratio, **options)
        assert plan.check_support()
        plan.compile()
        plan.execute(q, k, w, plans[name])
    torch.cuda.synchronize()
    plan_limits = _causal_col_limits([s_q] * b, [s_k] * b, [0] * b, ratio, b * s_q, device)
    plan_valid = _valid_columns(plan_limits, s_k)
    plan_clean = plans["clean"][..., :s_k].reshape(b * s_q, s_k)
    plan_skip = plans["skip"][..., :s_k].reshape(b * s_q, s_k)
    check_ref_indexer_forward(q, k, w, plans["clean"][..., :s_k], ratio)
    _assert_prefix_equal_and_tail_skipped(plan_clean, plan_skip, plan_valid)
    assert bool(torch.isnan(plans["skip"][..., s_k:]).all())
