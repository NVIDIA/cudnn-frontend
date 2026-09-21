# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import (
    dsa_init,
    expand_mxfp8_scale,
    make_random_mxfp8_scale,
    pack_mxfp8_scales_thd,
    quantize_mxfp8,
    with_dsa_score_recompute_params,
)
from fe_api.dsa.dsa_reference import (
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
    """Masked scores stay -inf across KV tiles while contributing zero to L1; THD is declined on SM90."""
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
    if is_thd:
        # The SM90 kernel is BSHD-native; THD is declined at plan time instead of
        # being adapted by a host loop over cu_seqlens (Rule 3/8).
        with pytest.raises(NotImplementedError, match="cu_seqlens"):
            DSA.dense_attn_score_recompute_wrapper(q, k, lse, scale, qhead_per_kv_head=heads, out=out, ratio=ratio, q_causal_offsets=offsets, **options)
        return
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


def _allocate_mxfp8_qk(cfg, scale_utils):
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
    q_scale_logical = make_random_mxfp8_scale(
        (b, s_q, h_q, sf_groups),
        device=device,
        seed=201,
        exponent_min=-2,
        exponent_max=3,
    )
    k_scale_logical = make_random_mxfp8_scale(
        (b, s_k, h_kv, sf_groups),
        device=device,
        seed=203,
        exponent_min=-2,
        exponent_max=3,
    )
    q = quantize_mxfp8(q_ref, q_scale_logical)
    k = quantize_mxfp8(k_ref, k_scale_logical)
    q_deq = q.float() * expand_mxfp8_scale(q_scale_logical, d)
    k_deq = k.float() * expand_mxfp8_scale(k_scale_logical, d)
    q_scale = scale_utils.pack_q_scale_bshd(q_scale_logical, qhead_per_kv_head=qhpkv)
    k_scale = scale_utils.pack_k_scale_bshd(k_scale_logical)
    return q, k, q_deq, k_deq, q_scale, k_scale


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
    if torch.cuda.get_device_capability()[0] < 9:
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
    """SM90 dense score is BSHD-native: THD is declined at plan time (class and wrapper), BSHD stays served."""
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

    cu_q = torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda")
    thd = dict(qhead_per_kv_head=q.shape[-2], cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=8, max_seqlen_k=out.shape[1])
    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        if score_type == "indexer":
            DSA.dense_indexer_score_recompute_wrapper(q, k, aux, **thd)
        else:
            DSA.dense_attn_score_recompute_wrapper(q, k, aux, q.shape[-1] ** -0.5, **thd)

    bshd = [t.unsqueeze(0) for t in (q, k, aux, out, denom)]
    assert _make_dense_plan(DSA, score_type, *bshd).check_support()


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
    if torch.cuda.get_device_capability()[0] < 10:
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


@pytest.mark.L0
@pytest.mark.parametrize(
    "score_type,qhead_per_kv_head,s_q_default",
    [
        pytest.param("indexer", 64, 128, id="indexer_qh64"),
        pytest.param("attention", 64, 128, id="attention_qh64"),
        pytest.param("indexer", 32, 5, id="indexer_qh32_query_tail"),
    ],
)
@torch_fork_set_rng(seed=13)
def test_DSA_dense_score_recompute_wrapper_mxfp8_matches_dequant_reference(
    score_type,
    qhead_per_kv_head,
    s_q_default,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    scale_utils = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils")
    cfg = dsa_init(
        request=request,
        head_dim=128,
        qhead_per_kv_head=qhead_per_kv_head,
        score_type=score_type,
        min_compute_capability=100,
        b_default=1,
        s_q_default=s_q_default,
        s_kv_default=128,
    )
    ratio = 4
    q, k, q_deq, k_deq, q_scale, k_scale = _allocate_mxfp8_qk(cfg, scale_utils)
    q_causal_offsets = torch.full((cfg["b"],), 16, dtype=torch.int32, device=q.device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if score_type == "indexer":
        aux = (
            torch.randn(
                cfg["b"],
                cfg["s_q"],
                cfg["h_kv"] * cfg["qhead_per_kv_head"],
                dtype=torch.bfloat16,
                device="cuda",
            ).abs()
            * 0.1
        )
        result = DSA.dense_indexer_score_recompute_wrapper(
            q,
            k,
            aux,
            qhead_per_kv_head=cfg["qhead_per_kv_head"],
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            precision="mxfp8",
            q_scale=q_scale,
            k_scale=k_scale,
            stream=stream,
        )
        check_ref_dense_score_recompute(
            "indexer",
            q_deq,
            k_deq,
            aux,
            result["out"],
            result["denom"],
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            atol_scores=5e-3,
            rtol_scores=5e-3,
            atol_denom=5e-3,
            rtol_denom=5e-3,
        )
    else:
        softmax_scale = 1.0 / math.sqrt(cfg["head_dim"])
        aux = _dense_attn_lse(q_deq, k_deq, softmax_scale)
        result = DSA.dense_attn_score_recompute_wrapper(
            q,
            k,
            aux,
            softmax_scale,
            qhead_per_kv_head=cfg["qhead_per_kv_head"],
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            precision="mxfp8",
            q_scale=q_scale,
            k_scale=k_scale,
            stream=stream,
        )
        check_ref_dense_score_recompute(
            "attention",
            q_deq,
            k_deq,
            aux,
            result["out"],
            result["denom"],
            softmax_scale=softmax_scale,
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            atol_scores=5e-3,
            rtol_scores=5e-3,
            atol_denom=5e-3,
            rtol_denom=5e-3,
        )


@pytest.mark.L0
@pytest.mark.parametrize(
    "score_type,head_dim",
    [
        pytest.param("indexer", 128, id="indexer_d128"),
        pytest.param("attention", 128, id="attention_d128"),
        pytest.param("attention", 512, id="attention_d512"),
    ],
)
@torch_fork_set_rng(seed=23)
def test_DSA_dense_score_recompute_wrapper_thd_mxfp8_compact_padded_scales(
    score_type,
    head_dim,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("THD MXFP8 dense score recompute requires SM100+")

    device = torch.device("cuda")
    shapes = [(127, 32), (129, 64)]
    ratio, h_q, h_kv, d = 4, 64, 1, head_dim
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
    q_scale_logical = make_random_mxfp8_scale(
        (total_q, h_q, d // 32),
        device=device,
        seed=211,
        exponent_min=-2,
        exponent_max=3,
    )
    k_scale_logical = make_random_mxfp8_scale(
        (total_k, h_kv, d // 32),
        device=device,
        seed=223,
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
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if score_type == "indexer":
        aux = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
        result = DSA.dense_indexer_score_recompute_wrapper(
            q,
            k,
            aux,
            qhead_per_kv_head=h_q // h_kv,
            ratio=ratio,
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
            stream=stream,
        )
        softmax_scale = None
    else:
        softmax_scale = 1.0 / math.sqrt(d)
        lse_parts = []
        cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
        for batch in range(len(shapes)):
            q0, q1 = cu_q_host[batch : batch + 2]
            k0, k1 = cu_k_host[batch : batch + 2]
            lse_parts.append(
                _dense_attn_lse(
                    q_deq[q0:q1].unsqueeze(0),
                    k_deq[k0:k1].unsqueeze(0),
                    softmax_scale,
                ).squeeze(0)
            )
        aux = torch.cat(lse_parts, dim=0).contiguous()
        result = DSA.dense_attn_score_recompute_wrapper(
            q,
            k,
            aux,
            softmax_scale,
            qhead_per_kv_head=h_q // h_kv,
            ratio=ratio,
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
            stream=stream,
        )
    torch.cuda.synchronize()

    assert result["out"].shape == (total_q, max_k)
    assert result["denom"].shape == (total_q,)
    cu_q_host, cu_k_host = cu_q.tolist(), cu_k.tolist()
    for batch, (_, s_k) in enumerate(shapes):
        q0, q1 = cu_q_host[batch : batch + 2]
        k0, k1 = cu_k_host[batch : batch + 2]
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
            atol_scores=5e-3,
            rtol_scores=5e-3,
            atol_denom=5e-3,
            rtol_denom=5e-3,
        )
