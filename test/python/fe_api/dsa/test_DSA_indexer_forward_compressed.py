import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_reference import (
    check_ref_compressed_topk,
    ref_indexer_forward,
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compressed_buffer_contract_rejects_unsafe_offsets():
    from cudnn import DSA
    from cudnn.deepseek_sparse_attention.indexer_forward._compressed_top_k_sm100 import (
        _indexer_fwd_compress_topk_thd,
    )

    device = torch.device("cuda")
    with pytest.raises(ValueError, match="0 <= q_causal_offsets"):
        DSA.compress_topk_cand_buffer_size(
            1,
            8,
            2,
            4,
            q_causal_offsets=torch.tensor([-1], dtype=torch.int32, device=device),
        )
    # A positive offset may move the query segment beyond the KV prefix; valid
    # row lengths clamp to S_k, matching the dense indexer mask semantics.
    shifted_bshd_floats = DSA.compress_topk_cand_buffer_size(
        1,
        8,
        2,
        4,
        q_causal_offsets=torch.tensor([1], dtype=torch.int32, device=device),
    )
    assert shifted_bshd_floats > 0

    cu_q = torch.tensor([0, 8], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 2], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="0 <= q_causal_offsets"):
        DSA.compress_topk_cand_buffer_size_thd(
            cu_q,
            cu_k,
            4,
            q_causal_offsets=torch.tensor([-1], dtype=torch.int32, device=device),
        )
    shifted_offsets, shifted_thd_floats = DSA.compress_topk_cand_buffer_size_thd(
        cu_q,
        cu_k,
        4,
        q_causal_offsets=torch.tensor([1], dtype=torch.int32, device=device),
    )
    assert int(shifted_offsets[-1]) == shifted_thd_floats > 0
    offsets, cand_floats = DSA.compress_topk_cand_buffer_size_thd(cu_q, cu_k, 4)
    q = torch.randn((8, 64, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((2, 1, 128), dtype=torch.bfloat16, device=device)
    w = torch.randn((8, 64), dtype=torch.bfloat16, device=device)
    out_indices = torch.empty((8, 1), dtype=torch.int32, device=device)
    out_logits = torch.empty((8, 1), dtype=torch.float32, device=device)

    bad_offsets = offsets.clone()
    bad_offsets[-1] += 1
    with pytest.raises(ValueError, match="cand_batch_offsets do not match"):
        _indexer_fwd_compress_topk_thd(
            q,
            k,
            w,
            1,
            ratio=4,
            qhead_per_kv_head=64,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=8,
            max_seqlen_k=2,
            cand_buffer=torch.empty(cand_floats + 1, device=device),
            cand_batch_offsets=bad_offsets,
            out_indices=out_indices,
            out_logits=out_logits,
        )
    with pytest.raises(ValueError, match="cand_buffer too small"):
        _indexer_fwd_compress_topk_thd(
            q,
            k,
            w,
            1,
            ratio=4,
            qhead_per_kv_head=64,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=8,
            max_seqlen_k=2,
            cand_buffer=torch.empty(cand_floats - 1, device=device),
            cand_batch_offsets=offsets,
            out_indices=out_indices,
            out_logits=out_logits,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=29)
def test_DSA_compressed_indexer_forward_bshd_cand_2d():
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
def test_DSA_compressed_indexer_forward_thd_mxfp8_lse():
    _require_sm100()
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.utils.sm100.mxfp8_scale_utils import (
            pack_k_scale_bshd,
            pack_k_scale_thd,
            pack_q_scale_bshd,
            pack_q_scale_thd,
        )
        from fe_api.dsa.dsa_utils import make_random_mxfp8_scale
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
    max_q, max_k = max(q_lengths), max(k_lengths)
    q = torch.randn(total_q, h_q, d, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k = torch.randn(total_k, h_kv, d, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    w = torch.randn(total_q, h_q, dtype=torch.bfloat16, device=device).abs() * 0.1
    q_scale_logical = make_random_mxfp8_scale((total_q, h_q, d // 32), device=device, seed=47)
    k_scale_logical = make_random_mxfp8_scale((total_k, h_kv, d // 32), device=device, seed=53)
    q_scale = pack_q_scale_thd(
        q_scale_logical,
        cu_q,
        qhead_per_kv_head=h_q,
        max_seqlen_q=max_q,
    )
    k_scale = pack_k_scale_thd(
        k_scale_logical,
        cu_k,
        max_seqlen_k=max_k,
    )
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
        topk_indices_global=False,
        return_lse=True,
        lse_out=lse_out,
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
