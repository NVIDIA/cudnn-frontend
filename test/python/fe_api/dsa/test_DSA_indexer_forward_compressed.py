# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_reference import (
    check_ref_compressed_topk,
    ref_indexer_forward,
)
from fe_api.dsa.dsa_utils import (
    make_random_mxfp8_scale,
    pack_mxfp8_scales_thd,
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
def test_compressed_indexer_rejects_unsupported_qhead_group_before_launch():
    _require_sm100()
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    q = torch.randn((1, 8, 8, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((1, 2, 1, 128), dtype=torch.bfloat16, device=device)
    w = torch.randn((1, 8, 8), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="qhead_per_kv_head=32 or 64"):
        DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=1,
            qhead_per_kv_head=8,
            return_softmax=False,
        )


@pytest.mark.L0
def test_indexer_denom_placeholders_use_stable_power_of_two_buckets():
    _require_sm100()
    try:
        from cudnn.deepseek_sparse_attention.indexer_forward import _compressed_top_k_sm100 as compressed_impl
        from cudnn.deepseek_sparse_attention.indexer_forward import _interface as dense_impl
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    for getter in (
        compressed_impl._get_fwd_unified_denom_placeholder,
        dense_impl._get_fwd_denom_placeholder,
    ):
        bucket_8_view_5 = getter((5,), device)
        bucket_8_view_8 = getter((8,), device)
        bucket_16_view_9 = getter((9,), device)
        bucket_8_view_5_again = getter((5,), device)
        rank_2_bucket_8 = getter((1, 5), device)

        assert bucket_8_view_5.shape == (5,)
        assert bucket_8_view_5.untyped_storage().nbytes() == 8 * bucket_8_view_5.element_size()
        assert bucket_8_view_5.data_ptr() == bucket_8_view_8.data_ptr()
        assert bucket_16_view_9.untyped_storage().nbytes() == 16 * bucket_16_view_9.element_size()
        assert bucket_16_view_9.data_ptr() != bucket_8_view_5.data_ptr()
        assert bucket_8_view_5_again.data_ptr() == bucket_8_view_5.data_ptr()
        assert rank_2_bucket_8.data_ptr() != bucket_8_view_5.data_ptr()


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
def test_DSA_compressed_stage2_deterministic_shrink_fallback():
    """A 4096-way exact tie forces the full-row fallback and selects ids 0..K-1."""
    _require_sm100()
    try:
        from cudnn.deepseek_sparse_attention.indexer_top_k.compress_top_k_sm100 import compress_stage2_topk
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    seqlen_k, top_k = 4096, 64
    candidates = torch.zeros(seqlen_k, dtype=torch.float32, device=device)
    cand_batch_offsets = torch.tensor([0, seqlen_k], dtype=torch.int64, device=device)
    q_causal_offsets = torch.tensor([seqlen_k - 1], dtype=torch.int32, device=device)

    compress_stage2_topk(
        candidates,
        1,
        1,
        seqlen_k,
        top_k,
        1,
        cand_batch_offsets=cand_batch_offsets,
        q_causal_offsets=q_causal_offsets,
        deterministic=False,
    )
    indices, logits = compress_stage2_topk(
        candidates,
        1,
        1,
        seqlen_k,
        top_k,
        1,
        cand_batch_offsets=cand_batch_offsets,
        q_causal_offsets=q_causal_offsets,
        deterministic=True,
    )
    indices_again, _ = compress_stage2_topk(
        candidates,
        1,
        1,
        seqlen_k,
        top_k,
        1,
        cand_batch_offsets=cand_batch_offsets,
        q_causal_offsets=q_causal_offsets,
        deterministic=True,
    )
    torch.cuda.synchronize()

    expected = torch.arange(top_k, dtype=torch.int32, device=device).view(1, 1, top_k)
    assert torch.equal(indices.sort(dim=-1).values, expected)
    assert torch.equal(indices.sort(dim=-1).values, indices_again.sort(dim=-1).values)
    assert torch.equal(logits, torch.zeros_like(logits))


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
def test_DSA_compressed_indexer_forward_bshd_preallocated_lse(h_q):
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
        )
        * sm_scale
    )
    local_indices = _bshd_global_to_local(result["indices"], s_k)
    check_ref_compressed_topk(
        dense_ref,
        local_indices,
        result["logits"],
        top_k,
        atol=2e-3,
        rtol=2e-3,
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
def test_DSA_compressed_indexer_forward_thd_preallocated_global_indices():
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
        )
        check_ref_compressed_topk(
            dense_ref,
            local_indices.unsqueeze(0),
            result["logits"][q0:q1].unsqueeze(0),
            top_k,
            atol=2e-3,
            rtol=2e-3,
        )
    _check_fused_softmax(result["indices"], result["logits"], result["softmax"])


@pytest.mark.L0
@torch_fork_set_rng(seed=43)
@pytest.mark.parametrize("deterministic", [False, True])
def test_DSA_compressed_indexer_forward_thd_mxfp8_lse(deterministic):
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
        dense = DSA.indexer_forward_wrapper(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio=ratio,
            sm_scale=d**-0.5,
            precision="mxfp8",
            q_scale=pack_q_scale_bshd(
                q_scale_logical[q0:q1].unsqueeze(0),
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
