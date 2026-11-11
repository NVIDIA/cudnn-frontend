# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import dsa_init, with_dsa_sparse_score_recompute_params
from deepseek_sparse_attention.cutedsl.dsa_reference import check_ref_sparse_score_recompute


def _allocate(cfg, score_type: str, has_topk_length: bool):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    topk = cfg["topk"]
    device = "cuda"

    q = torch.randn(b, s_q, qhpkv, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, d, dtype=torch.bfloat16, device=device)

    # Random top-K indices in [0, s_k). Use a guaranteed-valid range.
    topk_k = min(topk, s_k)
    topk_indices = torch.stack([torch.stack([torch.randperm(s_k, device=device)[:topk_k] for _ in range(s_q)]) for _ in range(b)]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((b, s_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    topk_length = None
    if has_topk_length:
        topk_length = torch.randint(1, topk_k + 1, (b, s_q), dtype=torch.int32, device=device)
        positions = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, topk)
        topk_indices = topk_indices.masked_fill(positions >= topk_length.unsqueeze(-1), -1)

    if score_type == "indexer":
        weights = torch.randn(b, s_q, qhpkv, dtype=torch.bfloat16, device=device)
        return q, k, weights, topk_indices, topk_length
    else:
        lse = torch.randn(b, s_q, qhpkv, dtype=torch.float32, device=device)
        return q, k, lse, topk_indices, topk_length


def _local_to_global_topk_indices(topk_indices: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    batch_offsets = torch.arange(topk_indices.shape[0], device=topk_indices.device, dtype=torch.int32).view(-1, 1, 1).mul_(int(seqlen_k))
    return torch.where(topk_indices >= 0, topk_indices + batch_offsets, topk_indices)


@pytest.mark.L1
@torch_fork_set_rng(seed=1234)
@pytest.mark.parametrize("num_heads,head_dim,topk", [(32, 128, 128), (64, 512, 512), (128, 512, 1024), (64, 576, 1024), (128, 576, 512)])
@pytest.mark.parametrize("has_topk_length", [False, True])
@pytest.mark.parametrize("use_global", [False, True])
def test_sparse_attn_sm100_tiles_match_reference(num_heads, head_dim, topk, has_topk_length, use_global):
    from deepseek_sparse_attention.cutedsl.dsa_utils import _require_sm100_family

    _require_sm100_family()
    from cudnn import DSA

    cfg = {"b": 2, "s_q": 32, "s_kv": 2048, "head_dim": head_dim, "qhead_per_kv_head": num_heads, "topk": topk}
    q, k, lse, local_ids, lengths = _allocate(cfg, "attention", has_topk_length)
    # Exercise partial final tiles across n128/n64 and K-split specializations,
    # preserving the invalid sentinel when converting to global IDs.
    valid_topk = topk * 3 // 4 + 5
    local_ids[..., valid_topk:] = -1
    if lengths is not None:
        lengths.clamp_(max=valid_topk)
    ids = _local_to_global_topk_indices(local_ids, cfg["s_kv"]) if use_global else local_ids
    scale = 1.0 / math.sqrt(cfg["head_dim"])
    actual = DSA.sparse_attn_score_recompute_wrapper(q, k, lse, ids, scale, topk_length=lengths, topk_indices_global=use_global)["target"]
    check_ref_sparse_score_recompute("attention", q, lse, local_ids, actual, aux=k, softmax_scale=scale, topk_length=lengths)


@pytest.mark.L0
@torch_fork_set_rng(seed=96)
@pytest.mark.parametrize("has_topk_length", [False, True])
def test_sparse_attn_h96_native_matches_reference(has_topk_length):
    """H96 uses native head tiles on both SM90 and the SM100 architecture family."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("native H96 sparse attention score recompute requires SM90 or SM100 family")
    from cudnn import DSA

    cfg = {"b": 1, "s_q": 4, "s_kv": 256, "head_dim": 576, "qhead_per_kv_head": 96, "topk": 128}
    q, k, lse, local_ids, _ = _allocate(cfg, "attention", has_topk_length=False)
    lengths = None
    if has_topk_length:
        lengths = torch.tensor([[0, 1, 127, 128]], dtype=torch.int32, device="cuda")
        positions = torch.arange(cfg["topk"], device="cuda", dtype=torch.int32).view(1, 1, -1)
        local_ids.masked_fill_(positions >= lengths.unsqueeze(-1), -1)
    else:
        local_ids[..., -37:] = -1

    scale = 1.0 / math.sqrt(cfg["head_dim"])
    actual = DSA.sparse_attn_score_recompute_wrapper(
        q,
        k,
        lse,
        local_ids,
        scale,
        qhead_per_kv_head=cfg["qhead_per_kv_head"],
        topk_length=lengths,
        topk_indices_global=False,
    )["target"]
    check_ref_sparse_score_recompute("attention", q, lse, local_ids, actual, aux=k, softmax_scale=scale, topk_length=lengths)


@pytest.mark.L1
@torch_fork_set_rng(seed=2051)
@pytest.mark.parametrize("logical_topk", [2048, 2049, 2050, 2051])
def test_sparse_attn_h96_native_production_topk(logical_topk):
    """Production top-k boundaries retain empty-row and invalid-suffix semantics."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("native H96 sparse attention score recompute requires SM90 or SM100 family")
    from cudnn import DSA

    heads, head_dim = 96, 576
    physical_topk = math.ceil(logical_topk / 128) * 128
    q = torch.zeros((1, 2, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((1, physical_topk, head_dim), dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros((1, 2, heads), dtype=torch.float32, device="cuda")
    indices = torch.arange(physical_topk, dtype=torch.int32, device="cuda").view(1, 1, -1).expand(1, 2, -1).clone()
    lengths = torch.tensor([[0, logical_topk]], dtype=torch.int32, device="cuda")
    positions = torch.arange(physical_topk, dtype=torch.int32, device="cuda").view(1, 1, -1)
    indices.masked_fill_(positions >= lengths.unsqueeze(-1), -1)

    actual = DSA.sparse_attn_score_recompute_wrapper(
        q,
        k,
        lse,
        indices,
        1.0 / math.sqrt(head_dim),
        qhead_per_kv_head=heads,
        topk_length=lengths,
        topk_indices_global=False,
    )["target"]
    expected = torch.zeros_like(actual)
    expected[0, 1, :logical_topk] = 1.0 / logical_topk
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-4)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_sparse_score_recompute_params
def test_DSA_sparse_score_recompute_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    score_type,
    has_topk_length,
    request,
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
        score_type=score_type,
        has_topk_length=has_topk_length,
        min_compute_capability=90,
        s_q_default=256,
        s_kv_default=2048,
    )
    q, k, aux, topk_indices, topk_length = _allocate(
        cfg,
        score_type,
        has_topk_length=has_topk_length,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        if score_type == "indexer":
            result = DSA.sparse_indexer_score_recompute_wrapper(
                q,
                k,
                aux,
                topk_indices,
                qhead_per_kv_head=qhead_per_kv_head,
                topk_length=topk_length,
                stream=stream,
            )
            actual = result["predict"]
        else:
            softmax_scale = 1.0 / math.sqrt(head_dim)
            result = DSA.sparse_attn_score_recompute_wrapper(
                q,
                k,
                aux,
                topk_indices,
                softmax_scale,
                qhead_per_kv_head=qhead_per_kv_head,
                topk_length=topk_length,
                stream=stream,
            )
            actual = result["target"]
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    if not cfg["skip_ref"]:
        if score_type == "indexer":
            check_ref_sparse_score_recompute(
                "indexer",
                q,
                k,
                topk_indices,
                actual,
                aux=aux,
                topk_length=topk_length,
            )
        else:
            check_ref_sparse_score_recompute(
                "attention",
                q,
                aux,
                topk_indices,
                actual,
                aux=k,
                softmax_scale=softmax_scale,
                topk_length=topk_length,
            )


@pytest.mark.L0
@torch_fork_set_rng(seed=1)
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@pytest.mark.parametrize("has_topk_length", [False, True])
def test_DSA_sparse_score_recompute_wrapper_batch_gt_one(score_type, has_topk_length, request):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=torch.bfloat16,
        acc_dtype=torch.float32,
        head_dim=128,
        qhead_per_kv_head=32,
        topk=128,
        score_type=score_type,
        has_topk_length=has_topk_length,
        min_compute_capability=90,
        b_default=2,
        s_q_default=32,
        s_kv_default=256,
    )
    q, k, aux, topk_indices, topk_length = _allocate(
        cfg,
        score_type,
        has_topk_length=has_topk_length,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    for use_global in (False, True):
        topk_for_kernel = _local_to_global_topk_indices(topk_indices, cfg["s_kv"]) if use_global else topk_indices
        try:
            if score_type == "indexer":
                result = DSA.sparse_indexer_score_recompute_wrapper(
                    q,
                    k,
                    aux,
                    topk_for_kernel,
                    qhead_per_kv_head=cfg["qhead_per_kv_head"],
                    topk_length=topk_length,
                    topk_indices_global=use_global,
                    stream=stream,
                )
                actual = result["predict"]
            else:
                softmax_scale = 1.0 / math.sqrt(cfg["head_dim"])
                result = DSA.sparse_attn_score_recompute_wrapper(
                    q,
                    k,
                    aux,
                    topk_for_kernel,
                    softmax_scale,
                    qhead_per_kv_head=cfg["qhead_per_kv_head"],
                    topk_length=topk_length,
                    topk_indices_global=use_global,
                    stream=stream,
                )
                actual = result["target"]
        except (ValueError, NotImplementedError, RuntimeError) as e:
            pytest.skip(f"Unsupported testcase: {e}")

        if not cfg["skip_ref"]:
            if score_type == "indexer":
                check_ref_sparse_score_recompute(
                    "indexer",
                    q,
                    k,
                    topk_indices,
                    actual,
                    aux=aux,
                    topk_length=topk_length,
                )
            else:
                check_ref_sparse_score_recompute(
                    "attention",
                    q,
                    aux,
                    topk_indices,
                    actual,
                    aux=k,
                    softmax_scale=softmax_scale,
                    topk_length=topk_length,
                )


# ---------------------------------------------------------------------------
# Rule 8 detectors on the classes (recipes R5 / R3 / R9)
# ---------------------------------------------------------------------------

_SMALL_CFG = {"b": 2, "s_q": 8, "s_kv": 256, "head_dim": 128, "qhead_per_kv_head": 32, "topk": 128}


def _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out, topk_length=None):
    if score_type == "indexer":
        return DSA.SparseIndexerScoreRecompute(
            sample_q_indexer=q,
            sample_k_indexer=k,
            sample_weights=aux,
            sample_topk_indices=topk_indices,
            sample_out=out,
            sample_topk_length=topk_length,
            qhead_per_kv_head=q.shape[2],
        )
    return DSA.SparseAttnScoreRecompute(
        sample_q_attn=q,
        sample_k_attn=k,
        sample_lse=aux,
        sample_topk_indices=topk_indices,
        sample_out=out,
        softmax_scale=q.shape[-1] ** -0.5,
        sample_topk_length=topk_length,
        qhead_per_kv_head=q.shape[2],
    )


def _sparse_case(score_type, has_topk_length=False):
    from cudnn import DSA

    from cudnn.tensor_adapter import has_tcgen05

    if torch.cuda.get_device_capability()[0] != 9 and not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Sparse score recompute requires SM90+")
    q, k, aux, topk_indices, topk_length = _allocate(_SMALL_CFG, score_type, has_topk_length)
    out = torch.empty(topk_indices.shape, dtype=torch.float32, device="cuda")
    return DSA, q, k, aux, topk_indices, topk_length, out


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_sparse_score_recompute_output_contiguity_declined(score_type):
    """R5: a layout the kernel cannot address natively is declined in check_support(), never copied."""
    DSA, q, k, aux, topk_indices, _, out = _sparse_case(score_type)
    b, s_q, topk = topk_indices.shape
    out_t = torch.empty(b, topk, s_q, dtype=torch.float32, device="cuda").transpose(1, 2)  # (b, s_q, topk) view, non-contiguous
    with pytest.raises(NotImplementedError, match="out must be contiguous"):
        _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out_t).check_support()
    idx_t = torch.empty(b, topk, s_q, dtype=torch.int32, device="cuda").transpose(1, 2)
    with pytest.raises(NotImplementedError, match="topk_indices must be contiguous"):
        _sparse_plan(DSA, score_type, q, k, aux, idx_t, out).check_support()
    q_strided = torch.empty(*q.shape[:-1], 2 * q.shape[-1], dtype=q.dtype, device="cuda")[..., ::2]  # innermost stride 2
    with pytest.raises(NotImplementedError, match="Q must have a unit innermost stride"):
        _sparse_plan(DSA, score_type, q_strided, k, aux, topk_indices, out).check_support()
    assert _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_sparse_score_recompute_wrapper_stages_noncontiguous_out(score_type):
    """The eager wrapper fills a non-contiguous caller ``out`` (staged + copied back, identity kept); the class declines it."""
    DSA, q, k, aux, topk_indices, _, _ = _sparse_case(score_type)
    b, s_q, topk = topk_indices.shape
    out_t = torch.empty(b, topk, s_q, dtype=torch.float32, device="cuda").transpose(1, 2)
    if score_type == "indexer":
        reference = DSA.sparse_indexer_score_recompute_wrapper(q, k, aux, topk_indices, qhead_per_kv_head=q.shape[2])["predict"]
        got = DSA.sparse_indexer_score_recompute_wrapper(q, k, aux, topk_indices, qhead_per_kv_head=q.shape[2], out=out_t)["predict"]
    else:
        scale = q.shape[-1] ** -0.5
        reference = DSA.sparse_attn_score_recompute_wrapper(q, k, aux, topk_indices, scale, qhead_per_kv_head=q.shape[2])["target"]
        got = DSA.sparse_attn_score_recompute_wrapper(q, k, aux, topk_indices, scale, qhead_per_kv_head=q.shape[2], out=out_t)["target"]
    torch.cuda.synchronize()
    assert got is out_t
    torch.testing.assert_close(out_t, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_sparse_score_recompute_execute_requires_outputs(score_type, compile_allocates_nothing):
    """out is a required execute argument, re-validated live (Rule 1): missing raises, a strided view raises."""
    DSA, q, k, aux, topk_indices, _, out = _sparse_case(score_type)
    plan = _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    with pytest.raises(TypeError):
        plan.execute(q, k, aux, topk_indices)
    b, s_q, topk = topk_indices.shape
    out_t = torch.empty(b, topk, s_q, dtype=torch.float32, device="cuda").transpose(1, 2)
    with pytest.raises(ValueError, match="out must be contiguous"):
        plan.execute(q, k, aux, topk_indices, out_t)
    with pytest.raises(ValueError, match="topk_length was not declared"):
        plan.execute(q, k, aux, topk_indices, out, topk_length=torch.ones(b, s_q, dtype=torch.int32, device="cuda"))


@pytest.mark.L0
def test_DSA_sparse_score_recompute_disjoint_check_ignores_empty_views():
    """Empty and adjacent views remain legal even when they share the output's storage."""
    from cudnn.deepseek_sparse_attention.score_recompute.api import _check_sparse_score_disjoint

    storage = torch.empty(8, dtype=torch.float32, device="cuda")
    _check_sparse_score_disjoint(storage[:4], (("empty", storage[:0]), ("adjacent", storage[4:])))


@pytest.mark.L0
def test_DSA_sparse_score_recompute_disjoint_binding_cache_tracks_metadata_without_retaining_storage():
    """The hot-path cache hits only for the same live binding and owns no caller storage."""
    from cudnn.deepseek_sparse_attention.score_recompute.api import _ScoreRecomputeBase
    from cudnn.tensor_adapter import get_version

    class ConcreteScoreRecompute(_ScoreRecomputeBase):
        def check_support(self):
            return True

        def execute(self):
            return None

    out = torch.empty(8)
    reader = torch.empty(8)
    tensors = (("out", out), ("reader", reader))
    binding = _ScoreRecomputeBase._make_sparse_score_disjoint_binding(tensors)
    assert _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, tensors)

    # A value-only mutation changes the version but preserves every byte-span
    # field, so the second-level cache can refresh without a full overlap walk.
    reader.fill_(0)
    assert not _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, tensors)
    refreshed = _ScoreRecomputeBase._refresh_sparse_score_disjoint_binding(binding, tensors)
    assert refreshed is not None
    assert _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(refreshed, tensors)
    binding = refreshed

    # A distinct Tensor over the same address/storage (the allocator-reuse
    # worst case) misses on tensor identity.
    replacement = reader.view_as(reader)
    assert replacement.data_ptr() == reader.data_ptr()
    assert not _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, (("out", out), ("reader", replacement)))

    # set_ to a different storage, and same-storage offset / stride changes,
    # each bump the mutation counter and invalidate the cached relation.
    other = torch.empty(8)
    reader.set_(other)
    assert not _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, tensors)

    reader = torch.empty(8)
    tensors = (("out", out), ("reader", reader))
    binding = _ScoreRecomputeBase._make_sparse_score_disjoint_binding(tensors)
    storage = reader.untyped_storage()
    reader.set_(storage, 1, (7,), (1,))
    assert not _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, tensors)

    reader.set_(storage, 0, (2, 4), (1, 2))
    assert not _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, tensors)

    # Assigning .data can rebind storage without incrementing the mutation
    # counter. The cache must still miss and run the exact overlap check.
    reader = torch.empty(8)
    tensors = (("out", out), ("reader", reader))
    binding = _ScoreRecomputeBase._make_sparse_score_disjoint_binding(tensors)
    cached_version = binding[1][1]
    reader.data = out
    assert get_version(reader) == cached_version
    assert not _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, tensors)
    plan = ConcreteScoreRecompute()
    plan._sparse_score_disjoint_binding = binding
    with pytest.raises(ValueError, match=r"out must not alias reader"):
        plan._check_sparse_score_disjoint_binding(out, (("reader", reader),))

    reader.set_(storage, 0, (8,), (1,))
    binding = _ScoreRecomputeBase._make_sparse_score_disjoint_binding(tensors)
    storage_ref = weakref.ref(storage)
    storage.resize_(storage.nbytes() + 16)
    plan = ConcreteScoreRecompute()
    plan._sparse_score_disjoint_binding = binding
    plan._check_sparse_score_disjoint_binding(out, (("reader", reader),))
    binding = plan._sparse_score_disjoint_binding
    assert _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(binding, tensors)

    tensor_ref = binding[1][0]
    del reader, storage, tensors, binding
    gc.collect()
    assert tensor_ref() is None
    assert storage_ref() is None

    with torch.inference_mode():
        inference_tensors = (("out", torch.empty(8)), ("reader", torch.empty(8)))
        assert _ScoreRecomputeBase._make_sparse_score_disjoint_binding(inference_tensors) is None

    # Restoring one storage to the same extent may move its base pointer, but
    # cannot change the relative relation of two views of that same storage.
    shared = torch.empty(16)
    out, reader = shared[:8], shared[8:]
    tensors = (("out", out), ("reader", reader))
    binding = _ScoreRecomputeBase._make_sparse_score_disjoint_binding(tensors)
    storage = shared.untyped_storage()
    original_nbytes = storage.nbytes()
    storage.resize_(original_nbytes // 2)
    storage.resize_(original_nbytes)
    plan = ConcreteScoreRecompute()
    plan._sparse_score_disjoint_binding = binding
    plan._check_sparse_score_disjoint_binding(out, (("reader", reader),))
    assert _ScoreRecomputeBase._sparse_score_disjoint_binding_matches(plan._sparse_score_disjoint_binding, tensors)


@pytest.mark.L0
def test_DSA_sparse_score_recompute_disjoint_binding_cache_is_thread_safe():
    """Concurrent new bindings each validate before one complete cache entry wins under the GIL."""
    from cudnn.deepseek_sparse_attention.score_recompute.api import _ScoreRecomputeBase

    class ConcreteScoreRecompute(_ScoreRecomputeBase):
        def check_support(self):
            return True

        def execute(self):
            return None

    plan = ConcreteScoreRecompute()

    def validate(_):
        out = torch.empty(8)
        reader = torch.empty(8)
        readers = (("reader", reader),)
        plan._check_sparse_score_disjoint_binding(out, readers)
        return out, reader

    with ThreadPoolExecutor(max_workers=4) as pool:
        bindings = list(pool.map(validate, range(16)))

    assert (
        sum(plan._sparse_score_disjoint_binding_matches(plan._sparse_score_disjoint_binding, (("out", out), ("reader", reader))) for out, reader in bindings)
        == 1
    )


@pytest.mark.L0
def test_DSA_sparse_score_recompute_wrapper_does_not_stage_invalid_aliases():
    """An aliased output with an invalid dtype/shape still reaches the original class contract error."""
    DSA, q, k, weights, indices, _, _ = _sparse_case("indexer", False)
    with pytest.raises(ValueError, match="out dtype"):
        DSA.sparse_indexer_score_recompute_wrapper(q, k, weights, indices, out=indices)

    wrong_shape = indices[..., :-1].view(torch.float32)
    with pytest.raises(ValueError, match="out shape"):
        DSA.sparse_indexer_score_recompute_wrapper(q, k, weights, indices, out=wrong_shape)


@pytest.mark.L1
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=81)
def test_DSA_sparse_score_recompute_sm90_prepared_rejects_alias_but_wrapper_stages(score_type, compile_allocates_nothing):
    """Prepared SM90 execution rejects input/output overlap; the allocating wrapper preserves convenience."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 sparse score kernel")
    from cudnn import DSA

    cfg = {"b": 1, "s_q": 2, "s_kv": 64, "head_dim": 128, "qhead_per_kv_head": 32, "topk": 32}
    q, k, aux, indices, _ = _allocate(cfg, score_type, False)
    if score_type == "indexer":
        out = indices.view(torch.float32)
        expected_aux = aux
        expected_indices = indices.clone()
    else:
        out = aux
        expected_aux = aux.clone()
        expected_indices = indices

    valid_out = torch.empty_like(indices, dtype=torch.float32)
    plan = _sparse_plan(DSA, score_type, q, k, aux, indices, valid_out)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    plan.execute(q, k, aux, indices, valid_out)
    with pytest.raises(ValueError, match=r"out must not alias (topk_indices|lse)"):
        plan.execute(q, k, aux, indices, out)

    if score_type == "indexer":
        actual = DSA.sparse_indexer_score_recompute_wrapper(q, k, aux, indices, out=out)["predict"]
    else:
        actual = DSA.sparse_attn_score_recompute_wrapper(q, k, aux, indices, q.shape[-1] ** -0.5, out=out)["target"]
    torch.cuda.synchronize()
    assert actual is out
    check_ref_sparse_score_recompute(
        score_type,
        q,
        expected_aux if score_type == "attention" else k,
        expected_indices,
        actual,
        aux=k if score_type == "attention" else expected_aux,
        softmax_scale=q.shape[-1] ** -0.5,
    )


def _executes_allocate_nothing(plan, run, repeats=3):
    """R9: warm once, then no torch allocation and no host sync across ``repeats`` executes."""
    run()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(repeats):
            run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return torch.cuda.memory_stats()["allocation.all.allocated"] - before


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@pytest.mark.parametrize("has_topk_length", [False, True])
@torch_fork_set_rng(seed=7)
def test_DSA_sparse_score_recompute_execute_allocates_nothing_and_never_synchronizes(score_type, has_topk_length, compile_allocates_nothing):
    from cuda.bindings import driver as cuda

    DSA, q, k, aux, topk_indices, topk_length, out = _sparse_case(score_type, has_topk_length)
    plan = _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out, topk_length)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    delta = _executes_allocate_nothing(plan, lambda: plan.execute(q, k, aux, topk_indices, out, topk_length=topk_length, current_stream=stream))
    assert delta == 0, f"{type(plan).__name__}.execute() made {delta} torch allocation(s) across 3 warm executes"
    check_ref_sparse_score_recompute(
        score_type,
        q,
        k if score_type == "indexer" else aux,
        topk_indices,
        out,
        aux=aux if score_type == "indexer" else k,
        softmax_scale=q.shape[-1] ** -0.5,
        topk_length=topk_length,
    )


@pytest.mark.L1
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_sparse_score_recompute_sm90_symbolic_compile_cache_is_bounded(score_type, monkeypatch, compile_allocates_nothing):
    """SM90 compile artifacts ignore B/Sq/Sk/exact-K and split only at the 128-slot capacity boundary."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 sparse score kernel")
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.score_recompute import _interface_sm90

    monkeypatch.setattr(_interface_sm90.compile_sparse_score_recompute, "compile_cache", {})

    def make_plan(b, s_q, s_k, topk):
        cfg = {"b": b, "s_q": s_q, "s_kv": s_k, "head_dim": 128, "qhead_per_kv_head": 32, "topk": topk}
        q, k, aux, indices, _ = _allocate(cfg, score_type, False)
        out = torch.empty_like(indices, dtype=torch.float32)
        return _sparse_plan(DSA, score_type, q, k, aux, indices, out)

    same_bucket_a = make_plan(1, 3, 384, 129)
    same_bucket_b = make_plan(2, 5, 512, 255)
    next_bucket = make_plan(1, 7, 640, 257)
    for plan in (same_bucket_a, same_bucket_b, next_bucket):
        assert plan.check_support()
        compile_allocates_nothing(plan)

    assert same_bucket_a._compiled_kernel is same_bucket_b._compiled_kernel
    assert next_bucket._compiled_kernel is not same_bucket_a._compiled_kernel
    assert len(_interface_sm90.compile_sparse_score_recompute.compile_cache) == 2


@pytest.mark.L0
def test_DSA_sparse_score_recompute_sm90_topk_capacity_boundaries():
    from cudnn.deepseek_sparse_attention.score_recompute import _interface_sm90

    assert _interface_sm90._sparse_topk_capacity(1) == 128
    assert _interface_sm90._sparse_topk_capacity(128) == 128
    assert _interface_sm90._sparse_topk_capacity(129) == 256
    assert _interface_sm90._sparse_topk_capacity(2176) == 2176
    for invalid in (0, 2177):
        with pytest.raises(ValueError, match=r"must be in \[1, 2176\]"):
            _interface_sm90._sparse_topk_capacity(invalid)


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@pytest.mark.parametrize("physical_topk", [0, 2177])
def test_DSA_sparse_score_recompute_public_support_rejects_sm90_topk_boundaries(score_type, physical_topk, monkeypatch):
    """The public support query rejects widths that the prepared SM90 compiler cannot serve."""
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.score_recompute import api as score_api

    monkeypatch.setattr(score_api, "device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (9, 0))
    q = torch.empty((1, 1, 32, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.empty((1, 1, 128), dtype=torch.bfloat16, device="cuda")
    aux_dtype = torch.bfloat16 if score_type == "indexer" else torch.float32
    aux = torch.empty((1, 1, 32), dtype=aux_dtype, device="cuda")
    topk_indices = torch.empty((1, 1, physical_topk), dtype=torch.int32, device="cuda")
    out = torch.empty((1, 1, physical_topk), dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match=r"SM90 sparse score topk must be in \[1, 2176\]"):
        _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@pytest.mark.parametrize("physical_topk", [0, 2177])
def test_DSA_sparse_score_recompute_public_support_preserves_sm100_topk_contract(score_type, physical_topk, monkeypatch):
    """The SM90 capacity gate does not narrow the existing SM100 public support query."""
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.score_recompute import api as score_api

    monkeypatch.setattr(score_api, "device_capability", lambda _device: (10, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (10, 0))
    q = torch.empty((1, 1, 32, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.empty((1, 1, 128), dtype=torch.bfloat16, device="cuda")
    aux_dtype = torch.bfloat16 if score_type == "indexer" else torch.float32
    aux = torch.empty((1, 1, 32), dtype=aux_dtype, device="cuda")
    topk_indices = torch.empty((1, 1, physical_topk), dtype=torch.int32, device="cuda")
    out = torch.empty((1, 1, physical_topk), dtype=torch.float32, device="cuda")

    assert _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out).check_support()


@pytest.mark.L1
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=9001)
def test_DSA_sparse_score_recompute_sm90_prepared_graph_replay_is_bitwise(score_type, compile_allocates_nothing):
    """A prepared SM90 plan captures on an explicit non-default stream and poisoned replays stay bitwise stable."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 sparse score kernel")
    from cuda.bindings import driver as cuda

    DSA, q, k, aux, topk_indices, _, out = _sparse_case(score_type)
    # Exercise a non-bucket-aligned physical width and short/partial/full rows.
    q = q[:, :4]
    aux = aux[:, :4]
    topk_indices = torch.cat((topk_indices[:, :4], topk_indices[:, :4, :1]), dim=-1).contiguous()
    out = torch.empty_like(topk_indices, dtype=torch.float32)
    lengths = torch.tensor([[1, 2, 128, 129], [129, 65, 17, 3]], dtype=torch.int32, device="cuda")
    positions = torch.arange(129, dtype=torch.int32, device="cuda").view(1, 1, -1)
    topk_indices.masked_fill_(positions >= lengths.unsqueeze(-1), -1)
    plan = _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out, lengths)
    assert plan.check_support()
    compile_allocates_nothing(plan)

    capture_stream = torch.cuda.Stream()
    cu_stream = cuda.CUstream(capture_stream.cuda_stream)

    def launch():
        plan.execute(q, k, aux, topk_indices, out, topk_length=lengths, current_stream=cu_stream)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(capture_stream):
            launch()
            capture_stream.synchronize()
            with torch.cuda.graph(graph, stream=capture_stream):
                launch()
            capture_stream.synchronize()

        with torch.cuda.stream(capture_stream):
            # Change live inputs after capture and poison the caller-owned output;
            # replay must read the captured pointers' new contents and overwrite every slot.
            q.copy_(torch.randn_like(q))
            k.copy_(torch.randn_like(k))
            aux.copy_(torch.randn_like(aux))
            out.fill_(float("nan"))
            capture_stream.synchronize()
            graph.replay()
            capture_stream.synchronize()
            expected = out.clone()
            check_ref_sparse_score_recompute(
                score_type,
                q,
                k if score_type == "indexer" else aux,
                topk_indices,
                expected,
                aux=aux if score_type == "indexer" else k,
                softmax_scale=q.shape[-1] ** -0.5,
                topk_length=lengths,
            )

            for repeat in range(10):
                out.fill_(float("nan"))
                capture_stream.synchronize()
                graph.replay()
                capture_stream.synchronize()
                assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8)), f"graph replay changed output bytes at repeat {repeat}"
    finally:
        graph.reset()


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=8)
def test_DSA_sparse_score_recompute_sm100_topk_length_none_compiles_out(score_type, monkeypatch):
    """R3: without topk_length the SM100 mTopkLength slot is None at compile and launch (no (1,1) placeholder), keyed have_topk_length=False."""
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM100 sparse score kernel")
    from cuda.bindings import driver as cuda
    from cudnn.deepseek_sparse_attention.score_recompute import _interface_sm100

    fn = _interface_sm100._sparse_indexer_score_recompute if score_type == "indexer" else _interface_sm100._sparse_attn_score_recompute
    monkeypatch.setattr(fn, "compile_cache", {})
    DSA, q, k, aux, topk_indices, _, out = _sparse_case(score_type)
    plan = _sparse_plan(DSA, score_type, q, k, aux, topk_indices, out)
    assert plan.check_support()
    plan.compile()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    delta = _executes_allocate_nothing(plan, lambda: plan.execute(q, k, aux, topk_indices, out, current_stream=stream))
    assert delta == 0, f"a dead topk_length slot must be compiled out, not allocated per execute ({delta} allocations)"
    (key,) = fn.compile_cache.keys()
    assert key[-3] is False, f"compile key must carry have_topk_length=False: {key}"


@pytest.mark.L0
@pytest.mark.parametrize("topk", [384, 1152])
def test_DSA_sparse_indexer_score_recompute_resets_partial_tmem_ring(topk):
    """A second launch must not consume stale scores from a partial TMEM ring."""

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")

    torch.manual_seed(20260907)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim = 1, topk, 32, 128
    q = torch.randn((1, s_q, heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((1, s_kv, head_dim), dtype=torch.bfloat16, device=device)
    weights = torch.ones((1, s_q, heads), dtype=torch.bfloat16, device=device)
    indices = torch.arange(topk, dtype=torch.int32, device=device).view(1, 1, topk)

    DSA.sparse_indexer_score_recompute_wrapper(q, k, weights, indices)
    torch.cuda.synchronize()
    q.zero_()
    k.zero_()

    result = DSA.sparse_indexer_score_recompute_wrapper(q, k, weights, indices)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        result["predict"],
        torch.full_like(result["predict"], 1.0 / topk),
        atol=1e-6,
        rtol=1e-4,
    )


@pytest.mark.L0
def test_DSA_sparse_attention_score_recompute_tracks_partial_ring_across_tiles():
    """Keep per-slot TMEM phases across persistent tiles, including skipped blocks."""

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")

    torch.manual_seed(20260907)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim, topk = 512, 1152, 128, 512, 1152
    q = torch.randn((1, s_q, heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((1, s_kv, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.zeros((1, s_q, heads), dtype=torch.float32, device=device)
    indices = torch.arange(topk, dtype=torch.int32, device=device).view(1, 1, topk).expand(1, s_q, topk).contiguous()
    topk_length = torch.full((1, s_q), topk, dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # More query tiles than can be resident at once force persistent CTAs to
    # carry ring state into later work.  The first launch leaves non-uniform
    # values in a 9-block / 4-slot ring.
    DSA.sparse_attn_score_recompute_wrapper(
        q,
        k,
        lse,
        indices,
        softmax_scale,
        qhead_per_kv_head=heads,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    # A compact second traversal exercises synthetic full-barrier arrivals for
    # the skipped blocks.  Zero Q/K makes every active slot exactly uniform.
    q.zero_()
    k.zero_()
    active_topk = 257
    topk_length.fill_(active_topk)
    result = DSA.sparse_attn_score_recompute_wrapper(
        q,
        k,
        lse,
        indices,
        softmax_scale,
        qhead_per_kv_head=heads,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    target = result["target"]
    torch.testing.assert_close(
        target[..., :active_topk],
        torch.full_like(target[..., :active_topk], 1.0 / active_topk),
        atol=1e-6,
        rtol=1e-4,
    )
    assert torch.count_nonzero(target[..., active_topk:]) == 0


@pytest.mark.L0
def test_DSA_sparse_attention_score_recompute_uses_launchable_smem_budget():
    """A legal Top-K boundary must not plan more than usable dynamic SMEM."""
    from cudnn import DSA

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    heads, head_dim, topk = 128, 512, 4224
    q = torch.zeros((1, 1, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((1, topk, head_dim), dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros((1, 1, heads), dtype=torch.float32, device="cuda")
    indices = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, 1, topk)
    result = DSA.sparse_attn_score_recompute_wrapper(q, k, lse, indices, 1.0 / math.sqrt(head_dim), qhead_per_kv_head=heads)
    torch.cuda.synchronize()
    torch.testing.assert_close(result["target"], torch.full_like(result["target"], 1.0 / topk), atol=1e-6, rtol=1e-4)
