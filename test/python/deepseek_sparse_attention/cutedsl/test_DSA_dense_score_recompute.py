# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import (
    _require_sm100_indexer,
    dsa_init,
    make_random_indexer_input,
    make_random_mxfp4,
    pack_mxfp8_scales_thd,
    with_dsa_score_recompute_params,
)
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    _batched_ratio_causal_mask,
    _ratio_causal_mask,
    check_ref_dense_score_recompute,
    ref_dense_attn_score_recompute,
    ref_indexer_forward,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("singleton_first", [True, False])
def test_DSA_dense_indexer_sm90_singleton_and_mask_cache(heads, ratio, singleton_first, monkeypatch):
    from cudnn import DSA

    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This regression exercises the SM90 dense score kernel")
    from cudnn.deepseek_sparse_attention.score_recompute import _interface_sm90

    # Exercise both compile orders: singleton layouts must also work on a cache hit.
    monkeypatch.setattr(_interface_sm90._dense_score_recompute, "compile_cache", {})
    lengths = [(1, 1), (3, 7), (7, 67)]
    if not singleton_first:
        lengths.reverse()
    for seqlen_q, seqlen_k in lengths:
        q = torch.randn(1, seqlen_q, heads, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, seqlen_k, 1, 128, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(1, seqlen_q, heads, dtype=torch.bfloat16, device="cuda")
        offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
        result = DSA.dense_indexer_score_recompute_wrapper(q, k, weights, ratio=ratio, qhead_per_kv_head=heads, q_causal_offsets=offsets)
        expected = ref_indexer_forward(q, k, weights, ratio, q_causal_offsets=offsets)
        torch.testing.assert_close(result["out"], expected, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(result["denom"], torch.logsumexp(expected, dim=-1), atol=5e-3, rtol=5e-3)
    assert len(_interface_sm90._dense_score_recompute.compile_cache) == 1


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
@pytest.mark.parametrize("heads", [32, 128])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("is_thd", [False, True])
def test_DSA_dense_attn_sm90_masked_output(heads, ratio, is_thd):
    """Masked scores stay -inf across KV tiles while contributing zero to L1; THD goes through the SM90 wrapper adapter."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This regression exercises the SM90 dense score kernel")
    from cudnn import DSA

    q_lengths, k_lengths = ([1, 8], [1, 128]) if is_thd else ([8, 8], [128, 128])
    q_shape = (sum(q_lengths), heads, 128) if is_thd else (2, 8, heads, 128)
    k_shape = (sum(k_lengths), 1, 128) if is_thd else (2, 128, 1, 128)
    q = torch.randn(q_shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(k_shape, dtype=torch.bfloat16, device="cuda")
    lse = torch.randn(q_shape[:-1], dtype=torch.float32, device="cuda")
    out_shape = (sum(q_lengths), 128) if is_thd else (2, 8, 128)
    out = torch.full(out_shape, 123.0, dtype=torch.float32, device="cuda")
    offsets = torch.tensor([0, 0 if is_thd else 64 * ratio - 1], dtype=torch.int32, device="cuda")
    options = {}
    if is_thd:
        options = dict(
            cu_seqlens_q=torch.tensor([0, 1, 9], dtype=torch.int32, device="cuda"),
            cu_seqlens_k=torch.tensor([0, 1, 129], dtype=torch.int32, device="cuda"),
            max_seqlen_q=8,
            max_seqlen_k=128,
        )
    scale = 128**-0.5
    # THD on SM90 is served by the wrapper's per-batch adapter (the plan class declines it).
    result = DSA.dense_attn_score_recompute_wrapper(q, k, lse, scale, qhead_per_kv_head=heads, out=out, ratio=ratio, q_causal_offsets=offsets, **options)
    assert result["out"].data_ptr() == out.data_ptr()
    q0, k0 = 0, 0
    for b, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
        q_b = q[q0 : q0 + sq].unsqueeze(0) if is_thd else q[b : b + 1]
        k_b = k[k0 : k0 + sk].unsqueeze(0) if is_thd else k[b : b + 1]
        lse_b = lse[q0 : q0 + sq].unsqueeze(0) if is_thd else lse[b : b + 1]
        expected, denom = ref_dense_attn_score_recompute(q_b, k_b, lse_b, scale, ratio, offsets[b : b + 1])
        valid = _batched_ratio_causal_mask(sq, sk, ratio, q.device, 1, offsets[b : b + 1])
        expected = expected.masked_fill(~valid, float("-inf"))
        actual = result["out"][q0 : q0 + sq, :sk].unsqueeze(0) if is_thd else result["out"][b : b + 1]
        actual_denom = result["denom"][q0 : q0 + sq].unsqueeze(0) if is_thd else result["denom"][b : b + 1]
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(actual_denom, denom, atol=5e-3, rtol=5e-3)
        if is_thd:
            assert torch.isneginf(result["out"][q0 : q0 + sq, sk:]).all()
        q0 += sq
        k0 += sk


@pytest.mark.L0
def test_DSA_ratio_causal_mask_offsets_reference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_default = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=torch.bool,
        device=device,
    )
    expected_cp = torch.tensor(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=torch.bool,
        device=device,
    )
    expected_batched = torch.stack(
        [
            expected_default[:5],
            expected_cp,
            torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                dtype=torch.bool,
                device=device,
            ),
        ]
    )

    assert torch.equal(_ratio_causal_mask(10, 3, 4, device), expected_default)
    assert torch.equal(_ratio_causal_mask(5, 3, 4, device, q_causal_offset=4), expected_cp)
    offsets = torch.tensor([0, 4, 9], dtype=torch.int32, device=device)
    assert torch.equal(_batched_ratio_causal_mask(5, 3, 4, device, 3, offsets), expected_batched)


def _allocate(cfg, score_type: str):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    qhpkv = cfg["qhead_per_kv_head"]
    h_kv = cfg["h_kv"]
    device = "cuda"

    q = torch.randn(b, s_q, h_kv * qhpkv, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, h_kv, d, dtype=torch.bfloat16, device=device)
    if score_type == "indexer":
        weights = torch.randn(b, s_q, h_kv * qhpkv, dtype=torch.bfloat16, device=device)
        return q, k, weights
    lse = torch.randn(b, s_q, h_kv * qhpkv, dtype=torch.float32, device=device)
    return q, k, lse


def _dense_attn_lse(q, k, softmax_scale):
    h_q = q.shape[2]
    h_kv = k.shape[2]
    qhpkv = h_q // h_kv
    k_exp = k.repeat_interleave(qhpkv, dim=2)
    qk = torch.einsum("bqhd,bkhd->bqhk", q.float(), k_exp.float()) * softmax_scale
    return torch.logsumexp(qk, dim=-1)


def _thd_plan_samples(score_type: str, total_q: int = 16, total_k: int = 32, heads: int = 32, d: int = 128):
    """Packed (T, H, D) samples for a THD dense score plan; ``out`` is (total_q, max_seqlen_k=total_k)."""
    device = torch.device("cuda")
    q = torch.randn(total_q, heads, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device)
    aux_dtype = torch.bfloat16 if score_type == "indexer" else torch.float32
    aux = torch.randn(total_q, heads, dtype=aux_dtype, device=device)
    out = torch.empty(total_q, total_k, dtype=torch.float32, device=device)
    denom = torch.empty(total_q, dtype=torch.float32, device=device)
    return q, k, aux, out, denom


def _make_dense_plan(DSA, score_type: str, q, k, aux, out, denom, **kwargs):
    if score_type == "indexer":
        return DSA.DenseIndexerScoreRecompute(
            sample_q=q, sample_k=k, sample_weights=aux, sample_out=out, sample_denom_out=denom, qhead_per_kv_head=q.shape[-2], **kwargs
        )
    return DSA.DenseAttnScoreRecompute(
        sample_q=q,
        sample_k=k,
        sample_lse=aux,
        sample_out=out,
        sample_denom_out=denom,
        softmax_scale=q.shape[-1] ** -0.5,
        qhead_per_kv_head=q.shape[-2],
        **kwargs,
    )


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_dense_score_recompute_thd_plan_requires_max_seqlen(score_type):
    """The THD launch envelope is a plan-time host int; it is never read back from cu_seqlens (Rule 3/8)."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from cudnn.tensor_adapter import has_tcgen05

    if torch.cuda.get_device_capability()[0] != 9 and not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Dense score recompute requires SM90+")

    q, k, aux, out, denom = _thd_plan_samples(score_type)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        _make_dense_plan(DSA, score_type, q, k, aux, out, denom, is_thd=True).check_support()
    with pytest.raises(ValueError, match="max_seqlen_k"):
        _make_dense_plan(DSA, score_type, q, k, aux, out, denom, is_thd=True, max_seqlen_q=8, max_seqlen_k=out.shape[1] + 1).check_support()
    plan = _make_dense_plan(DSA, score_type, q, k, aux, out, denom, is_thd=True, max_seqlen_q=8)
    assert (plan.max_seqlen_q, plan.max_seqlen_k) == (8, out.shape[1])
    # A BSHD plan takes its envelope from the sample shapes and rejects the THD-only ints.
    bshd = [t.unsqueeze(0) for t in (q, k, aux, out, denom)]
    with pytest.raises(ValueError, match="THD-only"):
        _make_dense_plan(DSA, score_type, *bshd, max_seqlen_q=8).check_support()
    assert _make_dense_plan(DSA, score_type, *bshd).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_dense_score_recompute_thd_sm90_declines_in_check_support(score_type):
    """SM90 dense score is BSHD-native: THD is declined at plan time (the class), BSHD stays served.
    The eager wrapper serves SM90 THD per batch instead (test_DSA_dense_score_recompute_wrapper_thd_sm90_per_batch_adapter)."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This decline is specific to SM90")

    q, k, aux, out, denom = _thd_plan_samples(score_type)
    plan = _make_dense_plan(DSA, score_type, q, k, aux, out, denom, is_thd=True, max_seqlen_q=8, max_seqlen_k=out.shape[1])
    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        plan.check_support()

    bshd = [t.unsqueeze(0) for t in (q, k, aux, out, denom)]
    assert _make_dense_plan(DSA, score_type, *bshd).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=19)
def test_DSA_dense_score_recompute_wrapper_thd_sm90_per_batch_adapter(score_type):
    """SM90 THD at the wrapper level: per-batch BSHD launches from a host copy of cu_seqlens; a caller ``out`` keeps its
    identity, padding columns are -inf, and a non-default stream works."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This adapter is specific to SM90")

    device = torch.device("cuda")
    shapes = [(5, 40), (16, 32), (0, 8), (3, 64)]  # (s_q, s_k) per segment, one empty
    ratio, heads, d = 1, 32, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_q, total_k = sum(q_lengths), sum(k_lengths)
    max_q, max_k = max(q_lengths), max(k_lengths)
    q = torch.randn(total_q, heads, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device)
    softmax_scale = None
    if score_type == "indexer":
        aux = torch.randn(total_q, heads, dtype=torch.bfloat16, device=device).abs() * 0.1
    else:
        softmax_scale = 1.0 / math.sqrt(d)
        lse_parts = []
        q0 = k0 = 0
        for s_q, s_k in shapes:
            if s_q:
                lse_parts.append(_dense_attn_lse(q[q0 : q0 + s_q].unsqueeze(0), k[k0 : k0 + s_k].unsqueeze(0), softmax_scale).squeeze(0))
            q0 += s_q
            k0 += s_k
        aux = torch.cat(lse_parts, dim=0).contiguous()
    out = torch.full((total_q, max_k), 123.0, dtype=torch.float32, device=device)
    side = torch.cuda.Stream()
    stream = cuda.CUstream(side.cuda_stream)
    thd = dict(qhead_per_kv_head=heads, ratio=ratio, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k, out=out, stream=stream)

    with torch.cuda.stream(side):
        if score_type == "indexer":
            result = DSA.dense_indexer_score_recompute_wrapper(q, k, aux, **thd)
        else:
            result = DSA.dense_attn_score_recompute_wrapper(q, k, aux, softmax_scale, **thd)
    torch.cuda.synchronize()

    assert result["out"] is out
    assert result["denom"].shape == (total_q,)
    q0 = k0 = 0
    for s_q, s_k in shapes:
        if s_q:
            check_ref_dense_score_recompute(
                score_type,
                q[q0 : q0 + s_q].unsqueeze(0),
                k[k0 : k0 + s_k].unsqueeze(0),
                aux[q0 : q0 + s_q].unsqueeze(0),
                out[q0 : q0 + s_q, :s_k].unsqueeze(0),
                result["denom"][q0 : q0 + s_q].unsqueeze(0),
                softmax_scale=softmax_scale,
                ratio=ratio,
            )
            assert torch.isneginf(out[q0 : q0 + s_q, s_k:]).all(), "padding columns are -inf, never the caller's stale values"
        q0 += s_q
        k0 += s_k

    # The plan class still declines SM90 THD; the adapter is a wrapper-only path.
    denom = torch.empty(total_q, dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        _make_dense_plan(DSA, score_type, q, k, aux, out, denom, is_thd=True, max_seqlen_q=max_q).check_support()


def _bshd_plan_samples(score_type: str, b: int = 2, s_q: int = 8, s_k: int = 128, heads: int = 32, d: int = 128):
    device = torch.device("cuda")
    q = torch.randn(b, s_q, heads, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, s_k, 1, d, dtype=torch.bfloat16, device=device)
    aux = torch.randn(b, s_q, heads, dtype=torch.bfloat16 if score_type == "indexer" else torch.float32, device=device)
    out = torch.empty(b, s_q, s_k, dtype=torch.float32, device=device)
    denom = torch.empty(b, s_q, dtype=torch.float32, device=device)
    return q, k, aux, out, denom


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_dense_score_recompute_output_contiguity_declined(score_type):
    """R5: a layout the kernel cannot address natively is declined in check_support(), never copied."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from cudnn.tensor_adapter import has_tcgen05

    if torch.cuda.get_device_capability()[0] != 9 and not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Dense score recompute requires SM90+")

    q, k, aux, out, denom = _bshd_plan_samples(score_type)
    b, s_q, s_k = out.shape
    out_t = torch.empty(b, s_k, s_q, dtype=torch.float32, device="cuda").transpose(1, 2)
    with pytest.raises(NotImplementedError, match="out must be contiguous"):
        _make_dense_plan(DSA, score_type, q, k, aux, out_t, denom).check_support()
    denom_t = torch.empty(s_q, b, dtype=torch.float32, device="cuda").transpose(0, 1)
    with pytest.raises(NotImplementedError, match="denom_out must be contiguous"):
        _make_dense_plan(DSA, score_type, q, k, aux, out, denom_t).check_support()
    k_strided = torch.empty(*k.shape[:-1], 2 * k.shape[-1], dtype=k.dtype, device="cuda")[..., ::2]
    with pytest.raises(NotImplementedError, match="K must have a unit innermost stride"):
        _make_dense_plan(DSA, score_type, q, k_strided, aux, out, denom).check_support()
    assert _make_dense_plan(DSA, score_type, q, k, aux, out, denom).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=17)
def test_DSA_dense_score_recompute_wrapper_stages_noncontiguous_outputs(score_type):
    """The eager wrapper fills a non-contiguous caller ``out`` / ``denom_out`` (staged + copied back, identity kept); the class declines them."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from cudnn.tensor_adapter import has_tcgen05

    if torch.cuda.get_device_capability()[0] != 9 and not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Dense score recompute requires SM90+")

    q, k, aux, out, denom = _bshd_plan_samples(score_type)
    softmax_scale = q.shape[-1] ** -0.5
    if score_type == "attention":
        aux = _dense_attn_lse(q, k, softmax_scale)
    b, s_q, s_k = out.shape
    out_t = torch.full((b, s_k, s_q), 7.0, dtype=torch.float32, device="cuda").transpose(1, 2)
    denom_t = torch.full((s_q, b), 7.0, dtype=torch.float32, device="cuda").transpose(0, 1)
    common = dict(qhead_per_kv_head=q.shape[-2], out=out_t, denom_out=denom_t)
    if score_type == "indexer":
        result = DSA.dense_indexer_score_recompute_wrapper(q, k, aux, **common)
    else:
        result = DSA.dense_attn_score_recompute_wrapper(q, k, aux, softmax_scale, **common)
    torch.cuda.synchronize()
    assert result["out"] is out_t and result["denom"] is denom_t
    check_ref_dense_score_recompute(score_type, q, k, aux, out_t, denom_t, softmax_scale=softmax_scale if score_type == "attention" else None)


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
def test_DSA_dense_score_recompute_execute_requires_outputs(score_type, compile_allocates_nothing):
    """out/denom_out are required execute arguments, re-validated live (Rule 1)."""
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from cudnn.tensor_adapter import has_tcgen05

    if torch.cuda.get_device_capability()[0] != 9 and not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Dense score recompute requires SM90+")

    q, k, aux, out, denom = _bshd_plan_samples(score_type)
    plan = _make_dense_plan(DSA, score_type, q, k, aux, out, denom)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    with pytest.raises(TypeError):
        plan.execute(q, k, aux)
    with pytest.raises(TypeError):
        plan.execute(q, k, aux, out)
    b, s_q, s_k = out.shape
    out_t = torch.empty(b, s_k, s_q, dtype=torch.float32, device="cuda").transpose(1, 2)
    with pytest.raises(ValueError, match="out must be contiguous"):
        plan.execute(q, k, aux, out_t, denom)
    with pytest.raises(ValueError, match="denom_out shape"):
        plan.execute(q, k, aux, out, denom[:, : s_q - 1])
    with pytest.raises(ValueError, match="q_causal_offsets must be a contiguous 1-D int32"):
        plan.execute(q, k, aux, out, denom, q_causal_offsets=torch.zeros(2 * b, dtype=torch.int32, device="cuda")[::2])


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=9)
def test_DSA_dense_score_recompute_execute_allocates_nothing_and_never_synchronizes(score_type, compile_allocates_nothing):
    """R9: warm once, then three executes with caller-provided outputs allocate nothing and never block the host."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from cudnn.tensor_adapter import has_tcgen05

    if torch.cuda.get_device_capability()[0] != 9 and not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("Dense score recompute requires SM90+")

    q, k, aux, out, denom = _bshd_plan_samples(score_type)
    softmax_scale = q.shape[-1] ** -0.5
    if score_type == "attention":
        aux = _dense_attn_lse(q, k, softmax_scale)
    plan = _make_dense_plan(DSA, score_type, q, k, aux, out, denom)
    assert plan.check_support()
    compile_allocates_nothing(plan)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    offsets = torch.full((q.shape[0],), 8, dtype=torch.int32, device="cuda")

    def run():
        plan.execute(q, k, aux, out, denom, q_causal_offsets=offsets, current_stream=stream)

    run()  # warm: the score backend compiles on the first execute
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    delta = torch.cuda.memory_stats()["allocation.all.allocated"] - before
    if torch.cuda.get_device_capability()[0] == 9 and delta != 0:
        pytest.xfail("SM90 keeps the per-head (B,S,H)->(B,H,S) transpose copy (Rule 8 batch item D4, deferred)")
    assert delta == 0, f"{type(plan).__name__}.execute() made {delta} torch allocation(s) across 3 warm executes"
    check_ref_dense_score_recompute(
        score_type, q, k, aux, out, denom, softmax_scale=softmax_scale if score_type == "attention" else None, q_causal_offsets=offsets
    )


@pytest.mark.L0
@pytest.mark.parametrize("score_type", ["indexer", "attention"])
@torch_fork_set_rng(seed=0)
def test_DSA_dense_score_recompute_thd_execute_uses_plan_envelope(score_type):
    """execute() takes max_seqlen_q/k from the plan (no device read of cu_seqlens); mismatched values raise."""
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    from cudnn.tensor_adapter import has_tcgen05

    if not has_tcgen05(torch.cuda.get_device_capability()[0]):
        pytest.skip("THD dense score recompute is SM100+")

    device = torch.device("cuda")
    shapes = [(5, 40), (16, 32)]  # (s_q, s_k) per segment
    ratio, heads, d = 1, 32, 128
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_q, total_k = sum(q_lengths), sum(k_lengths)
    max_q, max_k = max(q_lengths), max(k_lengths)
    q = torch.randn(total_q, heads, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, 1, d, dtype=torch.bfloat16, device=device)
    softmax_scale = None
    if score_type == "indexer":
        aux = torch.randn(total_q, heads, dtype=torch.bfloat16, device=device).abs() * 0.1
    else:
        softmax_scale = 1.0 / math.sqrt(d)
        lse_parts = []
        q0 = k0 = 0
        for s_q, s_k in shapes:
            lse_parts.append(_dense_attn_lse(q[q0 : q0 + s_q].unsqueeze(0), k[k0 : k0 + s_k].unsqueeze(0), softmax_scale).squeeze(0))
            q0 += s_q
            k0 += s_k
        aux = torch.cat(lse_parts, dim=0).contiguous()
    out = torch.empty(total_q, max_k, dtype=torch.float32, device=device)
    denom = torch.empty(total_q, dtype=torch.float32, device=device)

    plan = _make_dense_plan(DSA, score_type, q, k, aux, out, denom, is_thd=True, max_seqlen_q=max_q)
    assert plan.check_support()
    plan.compile()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    common = dict(out=out, denom_out=denom, ratio=ratio, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, current_stream=stream)

    plan.execute(q, k, aux, **common)  # warm: the score backend compiles on first execute
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any blocking D2H read now raises
    try:
        plan.execute(q, k, aux, **common)  # no max_seqlen passed: the plan's envelope is used
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

    q0 = k0 = 0
    for s_q, s_k in shapes:
        check_ref_dense_score_recompute(
            score_type,
            q[q0 : q0 + s_q].unsqueeze(0),
            k[k0 : k0 + s_k].unsqueeze(0),
            aux[q0 : q0 + s_q].unsqueeze(0),
            out[q0 : q0 + s_q, :s_k].unsqueeze(0),
            denom[q0 : q0 + s_q].unsqueeze(0),
            softmax_scale=softmax_scale,
            ratio=ratio,
        )
        q0 += s_q
        k0 += s_k

    with pytest.raises(ValueError, match="must match the plan"):
        plan.execute(q, k, aux, max_seqlen_q=max_q + 1, **common)
    with pytest.raises(ValueError, match="must match the plan"):
        plan.execute(q, k, aux, max_seqlen_k=max_k + 1, **common)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_score_recompute_params
def test_DSA_dense_score_recompute_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    score_type,
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
        min_compute_capability=90,
        s_q_default=256,
        s_kv_default=1024,
    )
    q, k, aux = _allocate(cfg, score_type)
    q_causal_offsets = torch.full((cfg["b"],), 8, dtype=torch.int32, device=q.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        if score_type == "indexer":
            result = DSA.dense_indexer_score_recompute_wrapper(
                q,
                k,
                aux,
                qhead_per_kv_head=qhead_per_kv_head,
                q_causal_offsets=q_causal_offsets,
                stream=stream,
            )
        else:
            softmax_scale = 1.0 / math.sqrt(head_dim)
            result = DSA.dense_attn_score_recompute_wrapper(
                q,
                k,
                aux,
                softmax_scale,
                qhead_per_kv_head=qhead_per_kv_head,
                q_causal_offsets=q_causal_offsets,
                stream=stream,
            )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    out = result["out"]
    denom = result["denom"]

    assert out.shape == (cfg["b"], cfg["s_q"], cfg["s_kv"])
    assert denom.shape == (cfg["b"], cfg["s_q"])
    assert torch.isfinite(out).any()
    assert (torch.isfinite(out) | torch.isneginf(out)).all()
    assert torch.isfinite(denom).all()

    if not cfg["skip_ref"]:
        if score_type == "indexer":
            check_ref_dense_score_recompute(
                "indexer",
                q,
                k,
                aux,
                out,
                denom,
                q_causal_offsets=q_causal_offsets,
            )
        else:
            check_ref_dense_score_recompute(
                "attention",
                q,
                k,
                aux,
                out,
                denom,
                softmax_scale=softmax_scale,
                q_causal_offsets=q_causal_offsets,
            )


def _with_precisions(precisions, cases):
    """Cross ``precisions`` with ``cases`` (each a ``pytest.param`` whose first value is the score type); the dense attention
    score has no MXFP4 kernel."""
    return [
        pytest.param(precision, *case.values, marks=case.marks, id=f"{case.id}-{precision}")
        for case in cases
        for precision in precisions
        if not (case.values[0] == "attention" and precision == "mxfp4")
    ]


@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize(
    "precision,score_type,qhead_per_kv_head,b,s_q,s_k,head_dim,ratio,offsets",
    _with_precisions(
        ("bf16", "mxfp8", "mxfp4"),
        [
            pytest.param("indexer", 64, 1, 128, 128, 128, 4, (16,), marks=pytest.mark.L0, id="indexer_qh64"),
            pytest.param("indexer", 32, 1, 5, 128, 128, 4, (16,), marks=pytest.mark.L0, id="indexer_qh32_query_tail"),
            # Two batches exercise the packed-scale L index; batch 0 reaches the S_k=200 key tail (one full and one partial KV block).
            pytest.param("indexer", 64, 2, 64, 200, 128, 1, (136, 100), marks=pytest.mark.L1, id="indexer_qh64_ratio1_b2_key_tail"),
            # Batch 0's ratio-causal limit (up to 264) exceeds S_k=200, so the KV length bounds its rows.
            pytest.param("indexer", 64, 2, 64, 200, 128, 1, (200, 136), marks=pytest.mark.L1, id="indexer_qh64_ratio1_b2_kv_clamp"),
            # Three KV blocks per tile rotate the accumulator and K-scale TMEM slots.
            pytest.param("indexer", 32, 1, 40, 384, 128, 4, (1496,), marks=pytest.mark.L1, id="indexer_qh32_ratio4_three_kv_blocks"),
            pytest.param("indexer", 32, 1, 40, 320, 128, 1, (280,), marks=pytest.mark.L0, id="indexer_qh32_ratio1_three_kv_blocks"),
            # More query tiles than SMs, each with 6..8 KV blocks: persistent CTAs carry the ring phases across tiles.
            pytest.param("indexer", 64, 1, 1024, 1024, 128, 4, (3072,), marks=pytest.mark.L0, id="indexer_qh64_ratio4_persistent"),
            pytest.param("attention", 64, 1, 128, 128, 128, 4, (16,), marks=pytest.mark.L0, id="attention_qh64"),
            # One query token per 128-row tile (H128), at both head dims; D512 streams four 128-wide K chunks.
            pytest.param("attention", 128, 1, 37, 200, 128, 4, (16,), marks=pytest.mark.L0, id="attention_qh128_key_tail"),
            # Two queries per tile, more tiles than SMs, and a ratio-causal limit past S_k=300.
            pytest.param("attention", 64, 1, 1200, 300, 128, 4, (16,), marks=pytest.mark.L1, id="attention_qh64_persistent_kv_clamp"),
            pytest.param("attention", 128, 1, 64, 300, 512, 4, (16,), marks=pytest.mark.L0, id="attention_d512_qh128"),
            # More tiles than SMs: persistent CTAs carry the accumulator / scale / LSE ring phases across tiles.
            pytest.param("attention", 128, 1, 600, 160, 512, 4, (16,), marks=pytest.mark.L1, id="attention_d512_qh128_persistent"),
        ],
    ),
)
def test_DSA_dense_score_recompute_wrapper_matches_dequant_reference(precision, score_type, qhead_per_kv_head, b, s_q, s_k, head_dim, ratio, offsets):
    """SM100 BF16 / MXFP8 / MXFP4 dense indexer and attention scores and denominators match an FP32 reference on the
    (dequantized) inputs; signed indexer weights pin the ReLU before weighting."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    cuda = pytest.importorskip("cuda.bindings.driver", reason="Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_indexer(precision)

    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    device = "cuda"
    h_q = qhead_per_kv_head
    q, q_deq, q_scale_logical = make_random_indexer_input(precision, (b, s_q, h_q, head_dim), device=device, seed=301)
    k, k_deq, k_scale_logical = make_random_indexer_input(precision, (b, s_k, 1, head_dim), device=device, seed=303)
    q_causal_offsets = torch.tensor([offsets[i % len(offsets)] for i in range(b)], dtype=torch.int32, device=device)
    common = dict(
        qhead_per_kv_head=h_q,
        ratio=ratio,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    if precision != "bf16":
        common.update(
            q_scale=scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q),
            k_scale=scale_utils.pack_k_scale_bshd(k_scale_logical),
        )

    if score_type == "indexer":
        softmax_scale = None
        aux = torch.randn(b, s_q, h_q, dtype=torch.bfloat16, device=device) * 0.1
        result = DSA.dense_indexer_score_recompute_wrapper(q, k, aux, **common)
    else:
        softmax_scale = 1.0 / math.sqrt(head_dim)
        aux = _dense_attn_lse(q_deq, k_deq, softmax_scale)
        result = DSA.dense_attn_score_recompute_wrapper(q, k, aux, softmax_scale, **common)
    torch.cuda.synchronize()

    assert result["out"].shape == (b, s_q, s_k)
    tol = 1e-3 if score_type == "indexer" else 5e-3
    check_ref_dense_score_recompute(
        score_type,
        q_deq,
        k_deq,
        aux,
        result["out"],
        result["denom"],
        softmax_scale=softmax_scale,
        ratio=ratio,
        q_causal_offsets=q_causal_offsets,
        atol_scores=tol,
        rtol_scores=tol,
        atol_denom=tol,
        rtol_denom=tol,
    )


@torch_fork_set_rng(seed=37)
@pytest.mark.parametrize(
    "precision,score_type,head_dim,h_q,shapes,offsets,q_alignment",
    _with_precisions(
        ("mxfp8", "mxfp4"),
        [
            # Segment 0's ratio-causal limit (up to 47) exceeds its 32 keys.
            pytest.param("indexer", 128, 64, [(127, 32), (129, 64)], (64, 17), 256, marks=pytest.mark.L0, id="indexer_qh64_kv_clamp"),
            pytest.param("indexer", 128, 32, [(127, 32), (129, 64)], (64, 17), 256, marks=pytest.mark.L1, id="indexer_qh32_kv_clamp"),
            # The packer's minimal Q alignment puts segment 1's scale prefix at a token offset that is not a multiple of 128.
            pytest.param("indexer", 128, 64, [(63, 32), (129, 64)], (1, 17), None, marks=pytest.mark.L1, id="indexer_qh64_tight_q_alignment"),
            pytest.param("indexer", 128, 32, [(61, 32), (129, 64)], (1, 17), None, marks=pytest.mark.L0, id="indexer_qh32_tight_q_alignment"),
            pytest.param("attention", 128, 64, [(127, 32), (129, 64)], (1, 17), 256, marks=pytest.mark.L0, id="attention_d128"),
            pytest.param("attention", 512, 64, [(127, 32), (129, 64)], (1, 17), 256, marks=pytest.mark.L0, id="attention_d512"),
            pytest.param("attention", 512, 128, [(127, 32), (129, 64)], (1, 17), 256, marks=pytest.mark.L0, id="attention_d512_qh128"),
        ],
    ),
)
def test_DSA_dense_score_recompute_wrapper_thd_block_scaled_compact_padded_scales(precision, score_type, head_dim, h_q, shapes, offsets, q_alignment):
    """THD MXFP8 / MXFP4 dense indexer and attention scores and denominators match the per-segment reference through
    compact padded scale prefixes; columns past a segment's own keys stay -inf."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    cuda = pytest.importorskip("cuda.bindings.driver", reason="Environment not supported: cudnn[cutedsl] not installed")
    _require_sm100_indexer(precision)

    device = torch.device("cuda")
    ratio = 4
    q_lengths = [s_q for s_q, _ in shapes]
    k_lengths = [s_k for _, s_k in shapes]
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    max_q, max_k = max(q_lengths), max(k_lengths)
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()

    q, q_deq, q_scale_logical = make_random_indexer_input(precision, (total_q, h_q, head_dim), device=device, seed=311)
    k, k_deq, k_scale_logical = make_random_indexer_input(precision, (total_k, 1, head_dim), device=device, seed=313)
    q_scale, k_scale, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(
        q_scale_logical,
        k_scale_logical,
        cu_q,
        cu_k,
        h_q,
        q_alignment=q_alignment,
        k_alignment=256,
    )
    q_causal_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    common = dict(
        qhead_per_kv_head=h_q,
        ratio=ratio,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    if score_type == "indexer":
        softmax_scale = None
        aux = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device) * 0.1
        result = DSA.dense_indexer_score_recompute_wrapper(q, k, aux, **common)
    else:
        softmax_scale = 1.0 / math.sqrt(head_dim)
        segments = zip(zip(cu_q_host, cu_q_host[1:]), zip(cu_k_host, cu_k_host[1:]))
        aux = torch.cat([_dense_attn_lse(q_deq[q0:q1].unsqueeze(0), k_deq[k0:k1].unsqueeze(0), softmax_scale).squeeze(0) for (q0, q1), (k0, k1) in segments])
        result = DSA.dense_attn_score_recompute_wrapper(q, k, aux, softmax_scale, **common)
    torch.cuda.synchronize()

    assert result["out"].shape == (total_q, max_k)
    assert result["denom"].shape == (total_q,)
    tol = 1e-3 if score_type == "indexer" else 5e-3
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
        # Columns past a segment's own keys are padding and stay -inf.
        assert torch.isneginf(result["out"][q0:q1, s_k:]).all()
        check_ref_dense_score_recompute(
            score_type,
            q_deq[q0:q1].unsqueeze(0),
            k_deq[k0:k1].unsqueeze(0),
            aux[q0:q1].unsqueeze(0),
            result["out"][q0:q1, :s_k].unsqueeze(0),
            result["denom"][q0:q1].unsqueeze(0),
            softmax_scale=softmax_scale,
            ratio=ratio,
            q_causal_offsets=q_causal_offsets[batch : batch + 1],
            atol_scores=tol,
            rtol_scores=tol,
            atol_denom=tol,
            rtol_denom=tol,
        )


@pytest.mark.L0
@pytest.mark.parametrize(
    "score_type,precision",
    [("indexer", "mxfp8"), ("indexer", "mxfp4"), ("attention", "mxfp8")],
    ids=["indexer-mxfp8", "indexer-mxfp4", "attention-mxfp8"],
)
def test_DSA_dense_score_recompute_wrapper_block_scaled_rejects_mismatched_inputs(score_type, precision):
    """Block-scaled dense indexer / attention score inputs that break the contract are refused on the host."""
    DSA = pytest.importorskip("cudnn.deepseek_sparse_attention", reason="Environment not supported: cudnn[cutedsl] not installed").DSA
    _require_sm100_indexer(precision)

    from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference as scale_utils

    device = "cuda"
    b, s_q, s_k, h_q, d = 1, 8, 128, 64, 128
    q, _, q_scale_logical = make_random_indexer_input(precision, (b, s_q, h_q, d), device=device, seed=1)
    k, _, k_scale_logical = make_random_indexer_input(precision, (b, s_k, 1, d), device=device, seed=2)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=h_q)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    common = dict(qhead_per_kv_head=h_q, precision=precision, q_scale=q_scale, k_scale=k_scale)
    if score_type == "indexer":
        aux = torch.rand(b, s_q, h_q, dtype=torch.bfloat16, device=device)
        wrong_aux, aux_error = aux.float(), "weights to be torch.bfloat16"

        def score(q, k, aux, **kwargs):
            return DSA.dense_indexer_score_recompute_wrapper(q, k, aux, **kwargs)

    else:
        aux = torch.zeros(b, s_q, h_q, dtype=torch.float32, device=device)
        wrong_aux, aux_error = aux.bfloat16(), "lse to be torch.float32"

        def score(q, k, aux, **kwargs):
            return DSA.dense_attn_score_recompute_wrapper(q, k, aux, 1.0, **kwargs)

    with pytest.raises(ValueError, match="requires q_scale and k_scale"):
        score(q, k, aux, qhead_per_kv_head=h_q, precision=precision)
    with pytest.raises(TypeError, match="q_scale and k_scale"):
        score(q, k, aux, **{**common, "q_scale": q_scale.view(torch.uint8)})
    with pytest.raises(TypeError, match="q and k"):
        score(q.view(torch.uint8), k.view(torch.uint8), aux, **common)
    with pytest.raises(TypeError, match=aux_error):
        score(q, k, wrong_aux, **common)
    with pytest.raises(ValueError, match="sf_vec_size=32"):
        score(q, k, aux, sf_vec_size=16, **common)
    with pytest.raises(ValueError, match="q_scale packed shape"):
        score(q, k, aux, **{**common, "q_scale": k_scale})
    if precision != "mxfp4":
        return

    # Unpacked E4M3 Q/K under precision='mxfp4' are named as a dtype error, not as a doubled head_dim.
    q8, k8 = (torch.randn(*t.shape[:-1], d, device=device).to(torch.float8_e4m3fn) for t in (q, k))
    with pytest.raises(TypeError, match="q and k"):
        score(q8, k8, aux, **common)
    # Packed E2M1 Q/K under precision='mxfp8' are a dtype error as well.
    with pytest.raises(TypeError, match="q and k to be torch.float8_e4m3fn"):
        score(q, k, aux, **{**common, "precision": "mxfp8"})
    k_wide = torch.cat([k.view(torch.uint8)] * 2, dim=-1).view(k.dtype)
    with pytest.raises(ValueError, match=r"q last dim \(\d+\) != k last dim"):
        score(q, k_wide, aux, **common)
    # THD reports the same mismatch.
    cu_q = torch.tensor([0, s_q], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, s_k], dtype=torch.int32, device=device)
    q_scale_thd, k_scale_thd, cu_q_scale, cu_k_scale = pack_mxfp8_scales_thd(q_scale_logical[0], k_scale_logical[0], cu_q, cu_k, h_q)
    thd = dict(
        qhead_per_kv_head=h_q,
        precision=precision,
        q_scale=q_scale_thd,
        k_scale=k_scale_thd,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        cu_seqlens_q_scale_padded=cu_q_scale,
        cu_seqlens_k_scale_padded=cu_k_scale,
        max_seqlen_q=s_q,
        max_seqlen_k=s_k,
    )
    with pytest.raises(ValueError, match=r"q last dim \(64\) != k last dim \(128\)"):
        score(q[0], k_wide[0], aux[0], **thd)
    # A last dim of head_dim instead of head_dim // 2 names the packed width.
    q_wide = torch.cat([q.view(torch.uint8)] * 2, dim=-1).view(q.dtype)
    with pytest.raises(ValueError, match="stored as q/k last dim 64"):
        score(q_wide, k_wide, aux, **common)
    # The block-scaled dense indexer score packs 32 or 64 heads per KV head.
    q16, _, q16_scale_logical = make_random_mxfp4((b, s_q, 16, d), device=device, seed=3)
    q16_scale = scale_utils.pack_q_scale_bshd(q16_scale_logical, qhead_per_kv_head=16)
    with pytest.raises(ValueError, match="precision='mxfp4' dense indexer requires qhead_per_kv_head=32 or 64"):
        score(q16, k, aux[..., :16], **{**common, "qhead_per_kv_head": 16, "q_scale": q16_scale})
    # The dense attention score has no MXFP4 kernel.
    lse = torch.zeros(b, s_q, h_q, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="precision must be 'bf16' or 'mxfp8', got 'mxfp4'"):
        DSA.dense_attn_score_recompute_wrapper(q, k, lse, 1.0, **common)


@pytest.mark.L0
def test_DSA_dense_indexer_score_recompute_wrapper_sm90_declines_block_scales(monkeypatch):
    """On SM90 the dense indexer score declines the block-scaled precisions and stray block scales as SM100-only."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    api = pytest.importorskip("cudnn.deepseek_sparse_attention.score_recompute.api", reason="Environment not supported: cudnn[cutedsl] not installed")

    monkeypatch.setattr(api, "device_capability", lambda device=None: (9, 0))
    q = torch.zeros(1, 8, 64, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(1, 8, 64, dtype=torch.bfloat16, device="cuda")
    declined = r"Dense indexer score with block scales \(precision='mxfp8' or 'mxfp4'\) is SM100-only"
    for precision in ("mxfp8", "MXFP4"):
        with pytest.raises(NotImplementedError, match=declined):
            api.dense_indexer_score_recompute_wrapper(q, k, w, precision=precision)
    with pytest.raises(NotImplementedError, match=declined):
        api.dense_indexer_score_recompute_wrapper(q, k, w, q_scale=torch.zeros(1, dtype=torch.uint8, device="cuda"))
