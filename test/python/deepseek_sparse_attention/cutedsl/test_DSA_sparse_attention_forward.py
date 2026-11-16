# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contract and numerical tests for SM100 DSA sparse Prefill forward.

These tests are GPU-exclusive because the H128 specialization is a 2-CTA
cluster kernel using Blackwell TMEM. Running it concurrently with kernels from
another pytest-xdist CUDA context can produce nondeterministic NaNs.
"""

import math
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from deepseek_sparse_attention.cutedsl.dsa_reference import ref_sparse_attention_forward

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

_SUPPORTED_FORWARD_INSTANCES = (
    (64, 512, 0),
    (64, 512, 512),
    (64, 512, 1024),
    (64, 512, 2048),
    (64, 576, 0),
    (64, 576, 512),
    (64, 576, 1024),
    (64, 576, 2048),
    (96, 512, 0),
    (96, 512, 512),
    (96, 512, 1024),
    (96, 512, 2048),
    (96, 576, 0),
    (96, 576, 512),
    (96, 576, 1024),
    (96, 576, 2048),
    (128, 512, 0),
    (128, 512, 512),
    (128, 512, 1024),
    (128, 512, 2048),
    (128, 576, 0),
    (128, 576, 512),
    (128, 576, 1024),
    (128, 576, 2048),
)


@pytest.mark.L0
def test_DSA_sparse_attention_forward_check_support_rejects_cpu_inputs():
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    q = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    kv = torch.empty((1, 512), dtype=torch.bfloat16)
    topk_idxs = torch.zeros((1, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="must live on CUDA"):
        DSA.SparseAttentionForward(q, kv, topk_idxs).check_support()


@pytest.mark.L0
def test_DSA_sparse_attention_forward_check_support_rejects_pre_sm90(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    q = torch.empty((1, 64, 512), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((1, 512), dtype=torch.bfloat16, device="cuda")
    topk_idxs = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    with pytest.raises(RuntimeError, match="requires an SM90 or SM100-family GPU"):
        DSA.SparseAttentionForward(q, kv, topk_idxs).check_support()


@pytest.mark.L0
def test_DSA_sparse_attention_forward_rejects_invalid_public_contracts():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    q64 = torch.empty((1, 64, 512), dtype=torch.bfloat16, device=device)
    q128 = torch.empty((1, 128, 512), dtype=torch.bfloat16, device=device)
    kv = torch.empty((2, 512), dtype=torch.bfloat16, device=device)
    topk64 = torch.zeros((1, 64), dtype=torch.int32, device=device)

    assert DSA.SparseAttentionForward(q64.to(torch.float16), kv.to(torch.float16), topk64).check_support()
    with pytest.raises(ValueError, match="Q must be 3-D"):
        DSA.SparseAttentionForward(q64.squeeze(0), kv, topk64).check_support()
    with pytest.raises(ValueError, match="Q dtype mismatch"):
        DSA.SparseAttentionForward(q64.to(torch.float32), kv, topk64).check_support()
    with pytest.raises(ValueError, match="KV dtype mismatch"):
        DSA.SparseAttentionForward(q64.to(torch.float16), kv, topk64).check_support()
    with pytest.raises(ValueError, match="topk_idxs dtype mismatch"):
        DSA.SparseAttentionForward(q64, kv, topk64.to(torch.int64)).check_support()
    with pytest.raises(ValueError, match="KV head dimension"):
        DSA.SparseAttentionForward(q64, kv[:, :-1], topk64).check_support()
    with pytest.raises(ValueError, match="topk_idxs first dimension"):
        DSA.SparseAttentionForward(q64, kv, topk64.expand(2, -1)).check_support()
    with pytest.raises(ValueError, match="All inputs must share"):
        DSA.SparseAttentionForward(q64, kv.cpu(), topk64).check_support()

    with pytest.raises(ValueError, match="indexer_topk=256 is unsupported"):
        DSA.SparseAttentionForward(q64, kv, torch.zeros((1, 256), dtype=torch.int32, device=device), indexer_topk=256).check_support()
    with pytest.raises(ValueError, match="must not exceed logical K"):
        DSA.SparseAttentionForward(q64, kv, topk64, indexer_topk=512).check_support()
    # H128 D512 serves indexer_topk=2048 on two H64 head windows.
    assert DSA.SparseAttentionForward(
        q128,
        kv,
        torch.zeros((1, 2048), dtype=torch.int32, device=device),
        indexer_topk=2048,
    ).check_support()
    with pytest.raises(ValueError, match="indexer_topk=256 is unsupported"):
        DSA.SparseAttentionForward(q128, kv, torch.zeros((1, 256), dtype=torch.int32, device=device), indexer_topk=256).check_support()


@pytest.mark.L0
def test_DSA_sparse_attention_forward_rejects_stream_from_another_device(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100 as interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda", torch.cuda.current_device())
    q = torch.empty((1, 64, 512), dtype=torch.bfloat16, device=device)
    kv = torch.empty((1, 512), dtype=torch.bfloat16, device=device)
    topk_idxs = torch.empty((1, 0), dtype=torch.int32, device=device)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    other_device = device.index + 1
    monkeypatch.setattr(interface_sm100.cuda, "cuStreamGetDevice", lambda _: (cuda.CUresult.CUDA_SUCCESS, other_device))

    with pytest.raises(ValueError, match=f"stream belongs to cuda:{other_device}"):
        interface_sm100.sparse_attention_forward_sm100(q, kv, topk_idxs, current_stream=stream)


@pytest.mark.L0
def test_DSA_sparse_attention_forward_reference_preserves_duplicate_slots_and_sentinels():
    q = torch.tensor([[[1.0, 0.0], [0.5, -0.25]]], dtype=torch.float32)
    kv = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32)
    topk_idxs = torch.tensor([[0, 0, 1, -7, 2]], dtype=torch.int32)

    out, max_logits, lse, lse_indexer = ref_sparse_attention_forward(
        q,
        kv,
        None,
        topk_idxs,
        softmax_scale=1.0,
        return_full=True,
    )
    gathered = kv[torch.tensor([0, 0, 1])]
    scores = torch.einsum("hd,kd->hk", q[0], gathered)
    expected_weights = torch.softmax(scores, dim=-1)
    expected_out = torch.einsum("hk,kd->hd", expected_weights, gathered)

    torch.testing.assert_close(out[0], expected_out)
    torch.testing.assert_close(max_logits[0], scores.amax(dim=-1))
    torch.testing.assert_close(lse[0], torch.logsumexp(scores, dim=-1))
    assert lse_indexer is None

    empty_length = torch.zeros(1, dtype=torch.int32)
    empty_out, empty_max, empty_lse, _ = ref_sparse_attention_forward(
        q,
        kv,
        torch.zeros(2, dtype=torch.float32),
        topk_idxs,
        topk_length=empty_length,
        return_full=True,
    )
    assert torch.equal(empty_out, torch.zeros_like(empty_out))
    assert torch.isneginf(empty_max).all()
    assert torch.isposinf(empty_lse).all()


@pytest.mark.L0
def test_DSA_sparse_attention_forward_reference_sink_changes_only_out():
    torch.manual_seed(7)
    q = torch.randn(2, 3, 8, dtype=torch.float32)
    kv = torch.randn(5, 8, dtype=torch.float32)
    topk_idxs = torch.tensor([[4, 1, 1, -1], [0, 7, 2, 3]], dtype=torch.int32)

    without_sink = ref_sparse_attention_forward(q, kv, None, topk_idxs, return_full=True)
    with_sink = ref_sparse_attention_forward(q, kv, torch.tensor([0.5, -1.0, float("inf")]), topk_idxs, return_full=True)
    assert not torch.equal(without_sink[0], with_sink[0])
    torch.testing.assert_close(without_sink[1], with_sink[1])
    torch.testing.assert_close(without_sink[2], with_sink[2])
    assert torch.equal(with_sink[0][:, 2], torch.zeros_like(with_sink[0][:, 2]))


@pytest.mark.L0
def test_DSA_sparse_attention_forward_empty_wrapper_and_support_contract():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    q = torch.empty((2, 64, 512), dtype=torch.bfloat16, device=device)
    kv = torch.empty((0, 512), dtype=torch.bfloat16, device=device)
    topk_idxs = torch.empty((2, 0), dtype=torch.int32, device=device)

    op = DSA.SparseAttentionForward(q, kv, topk_idxs)
    assert op.check_support()
    with pytest.raises(ValueError, match="kernel not compiled"):
        op.execute(q, kv, topk_idxs)
    op.compile()
    result = DSA.sparse_attention_forward_wrapper(q, kv, topk_idxs)
    assert list(result.keys()) == ["out", "max_logits", "lse", "lse_indexer"]
    assert result["out"].shape == (2, 64, 512)
    assert torch.equal(result["out"], torch.zeros_like(result["out"]))
    assert torch.isneginf(result["max_logits"]).all()
    assert torch.isposinf(result["lse"]).all()
    assert result["lse_indexer"] is None

    out_storage = torch.empty(result["out"].numel() + 1, dtype=torch.bfloat16, device=device)
    misaligned_out = out_storage[1:].view_as(result["out"])
    assert misaligned_out.is_contiguous() and misaligned_out.data_ptr() % 16 == 2
    with pytest.raises(ValueError, match="base pointer must be 16-byte aligned"):
        op.execute(q, kv, topk_idxs, out=misaligned_out)

    bad_q = torch.empty((2, 48, 576), dtype=torch.bfloat16, device=device)
    bad_kv = torch.empty((1, 576), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="supports only"):
        DSA.SparseAttentionForward(bad_q, bad_kv, topk_idxs).check_support()


@pytest.mark.L0
@pytest.mark.parametrize("total_s_q,total_s_kv,logical_topk", [(0, 3, 5), (2, 3, 0), (2, 0, 5)])
def test_DSA_sparse_attention_forward_zero_size_paths(total_s_q, total_s_kv, logical_topk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    q = torch.empty((total_s_q, 64, 512), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((total_s_kv, 512), dtype=torch.bfloat16, device="cuda")
    topk_idxs = torch.zeros((total_s_q, logical_topk), dtype=torch.int32, device="cuda")
    result = DSA.sparse_attention_forward_wrapper(q, kv, topk_idxs)

    assert result["out"].shape == (total_s_q, 64, 512)
    assert torch.equal(result["out"], torch.zeros_like(result["out"]))
    assert torch.isneginf(result["max_logits"]).all()
    assert torch.isposinf(result["lse"]).all()
    assert result["lse_indexer"] is None


@pytest.mark.L0
def test_DSA_sparse_attention_forward_records_originals_and_zero_size_outputs(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cuda.bindings import driver as cuda
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100 as interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda", torch.cuda.current_device())
    q_storage = torch.empty(2 * 64 * 512 + 1, dtype=torch.bfloat16, device=device)
    q = q_storage[1:].view(2, 64, 512)  # contiguous but under-aligned
    kv_storage = torch.empty((3, 1024), dtype=torch.bfloat16, device=device)
    kv = kv_storage[:, ::2]
    topk_idxs = torch.empty((2, 0), dtype=torch.int32, device=device)
    sink_storage = torch.empty(128, dtype=torch.float32, device=device)
    attn_sink = sink_storage[::2]
    length_storage = torch.zeros(4, dtype=torch.int32, device=device)
    topk_length = length_storage[::2]
    out = torch.empty((2, 64, 512), dtype=torch.bfloat16, device=device)
    max_logits = torch.empty((2, 64), dtype=torch.float32, device=device)
    lse = torch.empty_like(max_logits)

    recorded_ids = []
    original_record_stream = torch.Tensor.record_stream

    def record_stream_spy(tensor, stream):
        recorded_ids.append(id(tensor))
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream_spy)
    side_stream = torch.cuda.Stream(device=device)
    stream = cuda.CUstream(side_stream.cuda_stream)
    interface_sm100.sparse_attention_forward_sm100(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
        max_logits=max_logits,
        lse=lse,
        current_stream=stream,
    )
    side_stream.synchronize()

    expected = (q, kv, topk_idxs, attn_sink, topk_length, out, max_logits, lse)
    assert {id(tensor) for tensor in expected}.issubset(set(recorded_ids))


@pytest.mark.L0
@pytest.mark.parametrize(
    "num_heads,head_dim,indexer_topk",
    _SUPPORTED_FORWARD_INSTANCES,
    ids=lambda value: str(value),
)
def test_DSA_sparse_attention_forward_supports_all_frozen_instances(num_heads, head_dim, indexer_topk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    logical_topk = max(indexer_topk, 1)
    q = torch.empty((1, num_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((1, head_dim), dtype=torch.bfloat16, device="cuda")
    topk_idxs = torch.zeros((1, logical_topk), dtype=torch.int32, device="cuda")

    op = DSA.SparseAttentionForward(q, kv, topk_idxs, indexer_topk=indexer_topk)
    assert op.check_support()
    assert (op.num_heads, op.head_dim, op.head_dim_v, op.logical_topk) == (num_heads, head_dim, 512, logical_topk)


@pytest.mark.L2
@pytest.mark.parametrize(
    "num_heads,head_dim,indexer_topk",
    _SUPPORTED_FORWARD_INSTANCES,
    ids=[f"H{h}-D{d}-indexer{k}" for h, d, k in _SUPPORTED_FORWARD_INSTANCES],
)
def test_DSA_sparse_attention_forward_executes_all_frozen_instances(num_heads, head_dim, indexer_topk):
    """Compile and numerically execute every supported specialization."""

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    torch.manual_seed(911 + num_heads + head_dim + indexer_topk)
    total_s_kv = 73
    logical_topk = max(indexer_topk, 65)
    q = (torch.randn(1, num_heads, head_dim, device="cuda") / math.sqrt(head_dim)).to(torch.bfloat16)
    kv = (torch.randn(total_s_kv, head_dim, device="cuda") / math.sqrt(head_dim)).to(torch.bfloat16)
    # Repeated modulo indices deliberately exercise slot multiplicity for the
    # long indexer-prefix instances without requiring a large KV allocation.
    topk_idxs = (torch.arange(logical_topk, device="cuda", dtype=torch.int32) % total_s_kv).unsqueeze(0)
    attn_sink = torch.linspace(-0.5, 0.5, num_heads, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(head_dim)

    result = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        softmax_scale=scale,
        indexer_topk=indexer_topk,
    )
    ref_out, ref_max, ref_lse, ref_lse_indexer = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=scale,
        indexer_topk=indexer_topk,
        return_full=True,
    )
    torch.cuda.synchronize()

    assert result["out"].dtype == q.dtype
    torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], ref_max, atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-6, rtol=2.01 / 65536)
    if indexer_topk:
        torch.testing.assert_close(result["lse_indexer"], ref_lse_indexer, atol=1e-6, rtol=2.01 / 65536)
    else:
        assert result["lse_indexer"] is None


@pytest.mark.L2
def test_DSA_sparse_attention_forward_head128_persistent_clc_reuse_and_drain():
    """Carry CLC jobs and O-completion phases across queries before the final drain."""

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    torch.manual_seed(173)
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    total_s_q = sm_count + 17
    total_s_kv, num_heads, head_dim, logical_topk = 73, 128, 512, 65
    q = (torch.randn(total_s_q, num_heads, head_dim, device=device) / math.sqrt(head_dim)).to(torch.bfloat16)
    kv = (torch.randn(total_s_kv, head_dim, device=device) / math.sqrt(head_dim)).to(torch.bfloat16)
    slots = torch.arange(logical_topk, dtype=torch.int32, device=device)
    query_offsets = torch.arange(total_s_q, dtype=torch.int32, device=device).unsqueeze(1)
    topk_idxs = (slots.unsqueeze(0) + query_offsets).remainder(total_s_kv)
    length_pattern = torch.tensor([0, 1, 63, 64, 65], dtype=torch.int32, device=device)
    topk_length = length_pattern.repeat((total_s_q + length_pattern.numel() - 1) // length_pattern.numel())[:total_s_q]
    attn_sink = torch.linspace(-0.5, 0.5, num_heads, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    result = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=scale,
    )
    ref_out, ref_max, ref_lse, _ = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=scale,
        return_full=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], ref_max, atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-6, rtol=2.01 / 65536)
    assert result["lse_indexer"] is None


@pytest.mark.L2
def test_DSA_sparse_attention_forward_head128_softmax_exchange_scratch_reuse():
    """Keep both softmax warp pairs synchronized while li scratch aliases score exchange."""

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    torch.manual_seed(716)
    device = torch.device("cuda")
    total_s_q, total_s_kv, num_heads, head_dim = 16, 73, 128, 512
    logical_topk, indexer_topk = 1152, 1024
    q = (torch.randn(total_s_q, num_heads, head_dim, device=device) / math.sqrt(head_dim)).to(torch.bfloat16)
    kv = (torch.randn(total_s_kv, head_dim, device=device) / math.sqrt(head_dim)).to(torch.bfloat16)
    slots = torch.arange(logical_topk, dtype=torch.int32, device=device)
    query_offsets = torch.arange(total_s_q, dtype=torch.int32, device=device).unsqueeze(1)
    topk_idxs = (slots.unsqueeze(0) + query_offsets).remainder(total_s_kv)
    attn_sink = torch.linspace(-0.5, 0.5, num_heads, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    op = DSA.SparseAttentionForward(
        q,
        kv,
        topk_idxs,
        sample_attn_sink=attn_sink,
        softmax_scale=scale,
        indexer_topk=indexer_topk,
    )
    assert op.check_support()
    op.compile()
    out = torch.empty_like(q)
    max_logits = torch.empty((total_s_q, num_heads), dtype=torch.float32, device=device)
    lse = torch.empty_like(max_logits)
    lse_indexer = torch.empty_like(max_logits)

    def execute():
        return op.execute(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            softmax_scale=scale,
            out=out,
            max_logits=max_logits,
            lse=lse,
            lse_indexer=lse_indexer,
        )

    result = execute()
    baseline = tuple(tensor.clone() for tensor in result)
    ref = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale=scale,
        indexer_topk=indexer_topk,
        return_full=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(result[0].float(), ref[0].float(), atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(result[1], ref[1], atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result[2], ref[2], atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result[3], ref[3], atol=1e-6, rtol=2.01 / 65536)

    for _ in range(20):
        result = execute()
        for actual, expected in zip(result, baseline):
            assert torch.equal(actual, expected)


@pytest.mark.L2
@pytest.mark.parametrize(
    "num_heads,head_dim",
    [(64, 512), (96, 512), (128, 512), (128, 576)],
    ids=["H64-D512", "H96-D512", "H128-D512", "H128-D576"],
)
def test_DSA_sparse_attention_forward_threshold_rescale_is_warp_group_uniform(num_heads, head_dim):
    """Lock down the threshold=6 vote-based rescale semantics."""

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    q = torch.zeros((1, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((128, head_dim), dtype=torch.bfloat16, device=device)
    q[0, 0, 0] = 1  # Head 0 jumps by 7 log2 units and triggers group 0.
    q[0, 1, 1] = 1  # Head 1 has a sub-threshold jump in the same group.
    q[0, 33, 1] = 1  # Head 33 has the same jump in non-triggered group 1.
    kv[:64, 2] = 1
    kv[64:, 0] = 7
    kv[64:, 1] = 3 / 128
    kv[64:, 2] = 2
    topk_idxs = torch.arange(128, dtype=torch.int32, device=device).unsqueeze(0)

    result = DSA.sparse_attention_forward_wrapper(q, kv, topk_idxs, softmax_scale=math.log(2.0))
    torch.cuda.synchronize()

    # Group 0 advances head 1's mi and quantizes the new P tile around 1;
    # group 1 keeps head 33's old mi and quantizes it around exp2(3/128).
    # The resulting BF16 outputs differ, so a per-head threshold decision
    # fails this check even though exact real-number softmax is invariant.
    expected = torch.tensor([1.5078125, 1.5], dtype=torch.bfloat16, device=device)
    assert torch.equal(result["out"][0, torch.tensor([1, 33], device=device), 2], expected)
    torch.testing.assert_close(result["lse"][0, 1], result["lse"][0, 33], atol=1e-6, rtol=0)


@pytest.mark.L1
@pytest.mark.parametrize(
    "num_heads,head_dim",
    [(64, 512), (64, 576), (96, 512), (96, 576), (128, 512), (128, 576)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_DSA_sparse_attention_forward_matches_gather_reference(num_heads, head_dim, dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    torch.manual_seed(17)
    device = torch.device("cuda")
    total_s_q, total_s_kv, logical_topk = 3, 96, 65
    q = (torch.randn(total_s_q, num_heads, head_dim, device=device) / math.sqrt(head_dim)).to(dtype)
    kv = (torch.randn(total_s_kv, head_dim, device=device) / math.sqrt(head_dim)).to(dtype)
    topk_idxs = torch.randint(0, total_s_kv, (total_s_q, logical_topk), dtype=torch.int32, device=device)
    nan_row = total_s_kv - 1
    topk_idxs[topk_idxs == nan_row] = 0
    topk_idxs[:, 1] = topk_idxs[:, 0]  # duplicate slot
    topk_idxs[0, -3:] = torch.tensor([total_s_kv, -1, nan_row], dtype=torch.int32, device=device)
    kv[nan_row].fill_(float("nan"))  # ignored by topk_length; predicated gather must not read it
    topk_length = torch.tensor([64, 63, 0], dtype=torch.int32, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    result = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    ref_out, ref_max, ref_lse, _ = ref_sparse_attention_forward(
        q,
        kv,
        attn_sink,
        topk_idxs,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_full=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], ref_max, atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-6, rtol=2.01 / 65536)
    assert torch.isfinite(result["out"]).all()
    assert torch.equal(result["out"][2], torch.zeros_like(result["out"][2]))
    assert torch.isneginf(result["max_logits"][2]).all()
    assert torch.isposinf(result["lse"][2]).all()
    assert result["lse_indexer"] is None

    window_starts = {(96, 512): (0, 32), (96, 576): (0, 32), (128, 576): (0, 64)}.get((num_heads, head_dim), ())
    if window_starts:
        # H96 runs the H64 kernel on heads [0, 64) and [32, 96), and H128 D576
        # on [0, 64) and [64, 128). Both H96 launches must write the same bytes
        # to the overlap, so each window equals an H64 run on a contiguous copy
        # of its heads.
        for start in window_starts:
            heads = slice(start, start + 64)
            window = DSA.sparse_attention_forward_wrapper(
                q[:, heads].contiguous(),
                kv,
                topk_idxs,
                attn_sink=attn_sink[heads].contiguous(),
                topk_length=topk_length,
                softmax_scale=softmax_scale,
            )
            for name in ("out", "max_logits", "lse"):
                assert torch.equal(result[name][:, heads], window[name]), f"{name}, heads [{start}, {start + 64})"

    no_sink_result = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        topk_idxs,
        attn_sink=torch.full_like(attn_sink, float("-inf")),
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(no_sink_result["max_logits"], result["max_logits"], atol=0, rtol=0)
    torch.testing.assert_close(no_sink_result["lse"], result["lse"], atol=0, rtol=0)

    positive_infinity_sink = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        topk_idxs,
        attn_sink=torch.full_like(attn_sink, float("inf")),
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )
    assert torch.equal(positive_infinity_sink["out"], torch.zeros_like(positive_infinity_sink["out"]))
    torch.testing.assert_close(positive_infinity_sink["max_logits"], result["max_logits"], atol=0, rtol=0)
    torch.testing.assert_close(positive_infinity_sink["lse"], result["lse"], atol=0, rtol=0)


@pytest.mark.L2
@pytest.mark.parametrize(
    "num_heads,head_dim",
    [(64, 512), (64, 576), (96, 576), (128, 512), (128, 576)],
    ids=["H64-D512", "H64-D576", "H96-D576", "H128-D512", "H128-D576"],
)
def test_DSA_sparse_attention_forward_reuses_compilation_across_dynamic_extents(num_heads, head_dim, monkeypatch):
    """One specialization must handle simultaneous Q, KV, and padded-K growth."""

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100 as interface_sm100
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import api as forward_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    scale = 1.0 / math.sqrt(head_dim)
    original_compile_kernel = interface_sm100._compile_kernel
    compile_calls = []

    def counted_compile_kernel(*args, **kwargs):
        compiled = original_compile_kernel(*args, **kwargs)
        compile_calls.append(compiled)
        return compiled

    # Isolate this behavioral check from kernels and API objects populated by
    # earlier tests in the same process.
    monkeypatch.setattr(interface_sm100, "_compile_cache", {})
    monkeypatch.setattr(forward_api, "_cache_of_sparse_attention_forward_objects", {})
    monkeypatch.setattr(interface_sm100, "_compile_kernel", counted_compile_kernel)

    def make_inputs(total_s_q, total_s_kv, logical_topk, seed, old_total_s_kv=None, singleton_broadcast_strides=False):
        torch.manual_seed(seed)
        q = (torch.randn(total_s_q, num_heads, head_dim, device=device) / math.sqrt(head_dim)).to(torch.bfloat16)
        kv = (torch.randn(total_s_kv, head_dim, device=device) / math.sqrt(head_dim)).to(torch.bfloat16)
        slots = torch.arange(logical_topk, dtype=torch.int32, device=device)
        row_offsets = 7 * torch.arange(total_s_q, dtype=torch.int32, device=device).unsqueeze(1)
        topk_idxs = (slots.unsqueeze(0) + row_offsets).remainder(total_s_kv)
        if old_total_s_kv is not None:
            # The first slot in the newly added physical tile reads KV tokens
            # that did not exist in the shape used to compile the kernel.
            topk_idxs[:, 64] = old_total_s_kv + torch.arange(total_s_q, dtype=torch.int32, device=device)
        attn_sink = torch.linspace(-0.25, 0.25, num_heads, dtype=torch.float32, device=device)
        topk_length = torch.full((total_s_q,), logical_topk, dtype=torch.int32, device=device)
        if singleton_broadcast_strides:
            # PyTorch considers these singleton broadcast views contiguous.
            # They must compile to the same canonical dynamic-layout ABI as
            # the larger tensors used below, independent of call order.
            q = q.as_strided(q.shape, (0, head_dim, 1))
            topk_idxs = topk_idxs.as_strided(topk_idxs.shape, (0, 1))
            topk_length = topk_length.as_strided(topk_length.shape, (0,))
            assert q.is_contiguous() and topk_idxs.is_contiguous() and topk_length.is_contiguous()
        return q, kv, topk_idxs, attn_sink, topk_length

    shape_a = make_inputs(total_s_q=1, total_s_kv=67, logical_topk=64, seed=1901, singleton_broadcast_strides=True)
    shape_b = make_inputs(total_s_q=3, total_s_kv=97, logical_topk=65, seed=1902, old_total_s_kv=shape_a[1].shape[0])

    assert shape_b[0].shape[0] > shape_a[0].shape[0]
    assert shape_b[1].shape[0] > shape_a[1].shape[0]
    assert (shape_a[2].shape[1] + 63) // 64 == 1
    assert (shape_b[2].shape[1] + 63) // 64 == 2
    assert torch.all(shape_b[2][:, 64] >= shape_a[1].shape[0])

    def run_and_check(inputs):
        q, kv, topk_idxs, attn_sink, topk_length = inputs
        result = DSA.sparse_attention_forward_wrapper(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            softmax_scale=scale,
        )
        ref_out, ref_max, ref_lse, _ = ref_sparse_attention_forward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            topk_length=topk_length,
            softmax_scale=scale,
            return_full=True,
        )
        torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-4, rtol=3.01 / 128)
        torch.testing.assert_close(result["max_logits"], ref_max, atol=1e-6, rtol=2.01 / 65536)
        torch.testing.assert_close(result["lse"], ref_lse, atol=1e-6, rtol=2.01 / 65536)
        assert result["lse_indexer"] is None
        return result

    result_a = run_and_check(shape_a)
    assert len(compile_calls) == len(interface_sm100._compile_cache) == 1
    assert len(forward_api._cache_of_sparse_attention_forward_objects) == 1
    compiled = next(iter(interface_sm100._compile_cache.values()))
    api_object = next(iter(forward_api._cache_of_sparse_attention_forward_objects.values()))

    result_b = run_and_check(shape_b)
    assert len(compile_calls) == len(interface_sm100._compile_cache) == 1
    assert next(iter(interface_sm100._compile_cache.values())) is compiled
    assert len(forward_api._cache_of_sparse_attention_forward_objects) == 1
    assert next(iter(forward_api._cache_of_sparse_attention_forward_objects.values())) is api_object

    result_a_again = run_and_check(shape_a)
    torch.cuda.synchronize()
    assert len(compile_calls) == len(interface_sm100._compile_cache) == 1
    assert next(iter(interface_sm100._compile_cache.values())) is compiled
    assert len(forward_api._cache_of_sparse_attention_forward_objects) == 1
    assert next(iter(forward_api._cache_of_sparse_attention_forward_objects.values())) is api_object
    torch.testing.assert_close(result_a_again["out"], result_a["out"], atol=0, rtol=0)
    assert not torch.equal(result_b["out"][1], torch.zeros_like(result_b["out"][1]))


@pytest.mark.L1
def test_DSA_sparse_attention_forward_compile_cache_separates_structural_specializations(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100 as interface_sm100
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import api as forward_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    compiled_callables = []

    def fake_compile_kernel(*_args, **_kwargs):
        def launch(*args):
            args[3].zero_()
            args[4].fill_(float("-inf"))
            args[5].fill_(float("inf"))
            if args[6] is not None:
                args[6].fill_(float("inf"))

        compiled_callables.append(launch)
        return launch

    monkeypatch.setattr(interface_sm100, "_compile_cache", {})
    monkeypatch.setattr(forward_api, "_cache_of_sparse_attention_forward_objects", {})
    monkeypatch.setattr(interface_sm100, "_compile_kernel", fake_compile_kernel)

    def run(head_dim, *, dtype=torch.bfloat16, has_sink=False, has_length=False, indexer_topk=0):
        logical_topk = max(indexer_topk, 1)
        q = torch.zeros((1, 64, head_dim), dtype=dtype, device="cuda")
        kv = torch.zeros((1, head_dim), dtype=dtype, device="cuda")
        topk_idxs = torch.zeros((1, logical_topk), dtype=torch.int32, device="cuda")
        attn_sink = torch.zeros(64, dtype=torch.float32, device="cuda") if has_sink else None
        topk_length = torch.full((1,), logical_topk, dtype=torch.int32, device="cuda") if has_length else None
        return DSA.sparse_attention_forward_wrapper(
            q,
            kv,
            topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=indexer_topk,
        )

    specializations = (
        (512, torch.bfloat16, False, False, 0),
        (576, torch.bfloat16, False, False, 0),
        (512, torch.bfloat16, True, False, 0),
        (512, torch.bfloat16, False, True, 0),
        (512, torch.bfloat16, False, False, 512),
        (512, torch.float16, False, False, 0),
    )
    for expected_entries, (head_dim, dtype, has_sink, has_length, indexer_topk) in enumerate(specializations, start=1):
        run(head_dim, dtype=dtype, has_sink=has_sink, has_length=has_length, indexer_topk=indexer_topk)
        assert len(compiled_callables) == len(interface_sm100._compile_cache) == expected_entries
        assert len(forward_api._cache_of_sparse_attention_forward_objects) == expected_entries

    assert len({id(compiled) for compiled in interface_sm100._compile_cache.values()}) == len(specializations)
    first_compiled = next(iter(interface_sm100._compile_cache.values()))
    run(512)
    assert len(compiled_callables) == len(interface_sm100._compile_cache) == len(specializations)
    assert next(iter(interface_sm100._compile_cache.values())) is first_compiled


@pytest.mark.L0
def test_DSA_sparse_attention_forward_packed_kv_declarations(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_forward._kv_formats import KV_FORMATS
    except ImportError:
        return pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    queries = {d: torch.empty((1, 64, d), dtype=torch.bfloat16, device=device) for d in (512, 576)}
    q = queries[512]
    kv = torch.empty((2, 512), dtype=torch.bfloat16, device=device)
    topk = torch.zeros((1, 640), dtype=torch.int32, device=device)

    # Every format: its byte rows, also as the view of a padded buffer, are validated, then declined until a kernel serves them.
    for fmt in KV_FORMATS.values():
        q_fmt = queries[fmt.head_dim_qk]
        rows = torch.empty((2, fmt.record_bytes), dtype=torch.uint8, device=device)
        padded = torch.empty((2, fmt.record_bytes + 32), dtype=torch.uint8, device=device)[:, : fmt.record_bytes]
        for pool in (rows, padded):
            with pytest.raises(NotImplementedError, match=f"does not serve kv_format='{fmt.name}' on SM10"):
                DSA.SparseAttentionForward(q_fmt, pool, topk, kv_format=fmt.name).check_support()
        with pytest.raises(NotImplementedError, match=f"does not serve kv_format='{fmt.name}'"):
            DSA.sparse_attention_forward_wrapper(q_fmt, rows, topk, kv_format=fmt.name)
        with pytest.raises(ValueError, match=f"must be the \\(total_S_kv, {fmt.record_bytes}\\) byte rows"):
            DSA.SparseAttentionForward(q_fmt, rows[:, :-1], topk, kv_format=fmt.name).check_support()
        other_d = 576 if fmt.head_dim_qk == 512 else 512
        with pytest.raises(ValueError, match=f"{fmt.name} rows hold D_qk={fmt.head_dim_qk} channels, but Q has D_qk={other_d}"):
            DSA.SparseAttentionForward(queries[other_d], rows, topk, kv_format=fmt.name).check_support()
        with pytest.raises(ValueError, match="KV dtype mismatch: expected torch.uint8"):
            DSA.SparseAttentionForward(q_fmt, rows.view(torch.float8_e4m3fn), topk, kv_format=fmt.name).check_support()

    # Unknown names, scattered row bytes, and a uint8 cache without its format.
    rows = torch.empty((2, 528), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="unknown kv_format 'mxfp8'; the packed KV cache formats are"):
        DSA.SparseAttentionForward(q, rows, topk, kv_format="mxfp8").check_support()
    with pytest.raises(ValueError, match="each row's bytes contiguous"):
        DSA.SparseAttentionForward(q, torch.empty((528, 2), dtype=torch.uint8, device=device).t(), topk, kv_format="mxfp8_528").check_support()
    for pool, pitch in (
        (torch.empty((528,), dtype=torch.uint8, device=device).expand(2, 528), 0),  # rows overlap
        (torch.empty((2, 529), dtype=torch.uint8, device=device)[:, :528], 529),  # rows not 16-byte apart
    ):
        with pytest.raises(ValueError, match=f"KV row pitch stride\\(0\\) must be a multiple of 16 bytes and at least 528, got {pitch}"):
            DSA.SparseAttentionForward(q, pool, topk, kv_format="mxfp8_528").check_support()
    with pytest.raises(ValueError, match="unknown kv_format \\['mxfp8_528'\\]"):
        DSA.sparse_attention_forward_wrapper(q, rows, topk, kv_format=["mxfp8_528"])
    with pytest.raises(ValueError, match="KV must have same dtype as Q; a packed torch.uint8 cache also needs kv_format"):
        DSA.SparseAttentionForward(q, rows, topk).check_support()

    # A per-tensor-scaled FP8 pool: float8_e4m3fn channels with an FP32 (1,) descale, declined until a kernel serves it.
    kv_fp8 = torch.empty((2, 512), dtype=torch.float8_e4m3fn, device=device)
    descale = torch.ones((1,), dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="does not serve a per-tensor-scaled FP8 KV pool \\(descale_kv\\) on SM10"):
        DSA.SparseAttentionForward(q, kv_fp8, topk, sample_descale_kv=descale).check_support()
    with pytest.raises(NotImplementedError, match="per-tensor-scaled FP8 KV pool"):
        DSA.sparse_attention_forward_wrapper(q, kv_fp8, topk, descale_kv=descale)
    for kv_case, descale_case, message in (
        (kv_fp8, None, "a float8_e4m3fn KV pool needs descale_kv"),
        (kv_fp8, descale.double(), "descale_kv dtype mismatch: expected torch.float32"),
        (kv_fp8, descale.new_ones((2,)), "descale_kv must have shape \\(1,\\)"),
        (kv, descale, "descale_kv is the descale of a float8_e4m3fn KV pool, but KV is torch.bfloat16"),
        (kv_fp8[:, :500], descale, "KV head dimension \\(500\\) must match Q \\(512\\)"),
    ):
        with pytest.raises(ValueError, match=message):
            DSA.SparseAttentionForward(q, kv_case, topk, sample_descale_kv=descale_case).check_support()
    with pytest.raises(ValueError, match="descale_kv does not apply to a mxfp8_528 cache, which carries its own scales"):
        DSA.SparseAttentionForward(q, rows, topk, kv_format="mxfp8_528", sample_descale_kv=descale).check_support()

    # A second pool with its own token ids: validated like kv, then declined until a kernel serves it.
    extra = torch.empty((3, 288), dtype=torch.uint8, device=device)
    extra_bf16 = torch.empty((3, 512), dtype=torch.bfloat16, device=device)
    extra_idxs = torch.zeros((1, 512), dtype=torch.int32, device=device)
    extra_len = torch.zeros((1,), dtype=torch.int32, device=device)
    packed_extra = dict(sample_extra_kv=extra, extra_kv_format="nvfp4_288", sample_extra_topk_idxs=extra_idxs)
    with pytest.raises(NotImplementedError, match="does not serve a second KV pool \\(extra_kv\\) on SM10"):
        DSA.SparseAttentionForward(q, kv, topk, **packed_extra, sample_extra_topk_length=extra_len).check_support()
    with pytest.raises(NotImplementedError, match="does not serve a second KV pool"):
        DSA.SparseAttentionForward(q, kv, topk, sample_extra_kv=extra_bf16, sample_extra_topk_idxs=extra_idxs).check_support()
    with pytest.raises(NotImplementedError, match="does not serve a second KV pool"):
        DSA.sparse_attention_forward_wrapper(q, kv, topk, extra_kv=extra, extra_kv_format="nvfp4_288", extra_topk_idxs=extra_idxs, extra_topk_length=extra_len)
    with pytest.raises(NotImplementedError, match="does not serve kv_format='mxfp8_528'"):
        DSA.sparse_attention_forward_wrapper(q, rows, topk, kv_format="mxfp8_528", extra_kv=extra, extra_kv_format="nvfp4_288", extra_topk_idxs=extra_idxs)
    with pytest.raises(NotImplementedError, match="indexer_topk > 0 \\(lse_indexer\\) with a second KV pool"):
        DSA.SparseAttentionForward(q, kv, topk, indexer_topk=512, **packed_extra).check_support()

    # The pool's token ids and format travel with it.
    with pytest.raises(ValueError, match="extra_kv needs extra_topk_idxs"):
        DSA.SparseAttentionForward(q, kv, topk, sample_extra_kv=extra, extra_kv_format="nvfp4_288").check_support()
    for orphan in (
        dict(extra_kv_format="nvfp4_288"),
        dict(sample_extra_topk_idxs=extra_idxs),
        dict(sample_extra_topk_length=extra_len),
        dict(sample_descale_extra_kv=descale),
    ):
        with pytest.raises(ValueError, match="describe a second KV pool, but extra_kv is not given"):
            DSA.SparseAttentionForward(q, kv, topk, **orphan).check_support()
    with pytest.raises(ValueError, match="a float8_e4m3fn extra_kv pool needs descale_extra_kv"):
        DSA.SparseAttentionForward(q, kv, topk, sample_extra_kv=kv_fp8, sample_extra_topk_idxs=extra_idxs).check_support()
    with pytest.raises(NotImplementedError, match="does not serve a second KV pool"):
        DSA.SparseAttentionForward(q, kv, topk, sample_extra_kv=kv_fp8, sample_extra_topk_idxs=extra_idxs, sample_descale_extra_kv=descale).check_support()
    with pytest.raises(ValueError, match="unknown extra_kv_format 'nvfp4'"):
        DSA.SparseAttentionForward(q, kv, topk, **{**packed_extra, "extra_kv_format": "nvfp4"}).check_support()
    with pytest.raises(ValueError, match="extra_kv in format nvfp4_288 must be the \\(total_S_extra_kv, 288\\) byte rows"):
        DSA.SparseAttentionForward(q, kv, topk, **{**packed_extra, "sample_extra_kv": extra[:, :-1]}).check_support()
    with pytest.raises(ValueError, match="extra_kv must have same dtype as Q; a packed torch.uint8 cache also needs extra_kv_format"):
        DSA.SparseAttentionForward(q, kv, topk, sample_extra_kv=extra, sample_extra_topk_idxs=extra_idxs).check_support()
    with pytest.raises(ValueError, match="extra_kv head dimension \\(576\\) must match Q \\(512\\)"):
        DSA.SparseAttentionForward(q, kv, topk, sample_extra_kv=extra_bf16.new_empty((3, 576)), sample_extra_topk_idxs=extra_idxs).check_support()
    for bad_idxs, message in (
        (extra_idxs.long(), "extra_topk_idxs dtype mismatch: expected torch.int32"),
        (extra_idxs[0], "extra_topk_idxs must be 2-D"),
        (extra_idxs.new_zeros((2, 512)), "extra_topk_idxs first dimension \\(2\\) must match Q \\(1\\)"),
    ):
        with pytest.raises(ValueError, match=message):
            DSA.SparseAttentionForward(q, kv, topk, **{**packed_extra, "sample_extra_topk_idxs": bad_idxs}).check_support()
    with pytest.raises(ValueError, match="extra_topk_length must have shape \\(1,\\)"):
        DSA.SparseAttentionForward(q, kv, topk, **packed_extra, sample_extra_topk_length=extra_len.new_zeros((2,))).check_support()

    # A plan declared without a second pool or a descale refuses them at execute rather than ignoring them.
    plan = DSA.SparseAttentionForward(q, kv, topk)
    assert plan.check_support()
    plan.compile()
    for name, value in (
        ("extra_kv", extra_bf16),
        ("extra_topk_idxs", extra_idxs),
        ("extra_topk_length", extra_len),
        ("descale_kv", descale),
        ("descale_extra_kv", descale),
    ):
        with pytest.raises(ValueError, match=f"{name} must be given exactly when"):
            plan.execute(q, kv, topk, **{name: value})

    # The new parameters are keyword-only, so positional callers keep their meaning.
    with pytest.raises(TypeError):
        DSA.SparseAttentionForward(q, rows, topk, None, None, None, 0, "mxfp8_528")

    # The architecture gate runs before the packed-cache decline.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    with pytest.raises(RuntimeError, match="requires an SM90 or SM100-family GPU"):
        DSA.SparseAttentionForward(q, rows, topk, kv_format="mxfp8_528").check_support()


@dataclass(frozen=True)
class _Pool:
    """One KV pool of a model's attention call: how it is stored and how wide its index table is."""

    form: str  # "bf16" (Q's dtype), "fp8" (float8_e4m3fn with a per-tensor descale), or a kv_format name
    slots: int


@dataclass(frozen=True)
class _ModelCall:
    """One attention call of a model, as the forward's arguments express it."""

    model: str
    layer: str
    heads: int  # published query heads on one GPU
    d_qk: int
    kv: _Pool
    extra: Optional[_Pool] = None
    indexer_topk: int = 0
    sink: bool = False
    served: bool = False  # a kernel serves this call today


# How sparse-attention models express their attention calls, by layer kind and by how their caches are stored. One-pool
# calls are today's form (a window folded into the index row after the indexer prefix); two-pool calls keep the window
# cache (kv) and the compressed cache (extra_kv) apart, each with its own token ids.
_MODEL_CALLS = (
    # DeepSeek-V3.2 and GLM-5.3: a 512-channel latent plus 64 RoPE channels, indexer top-2048 tokens, one pool.
    *(
        _ModelCall(model, "sparse", heads, 576, _Pool(form, 2048), served=form == "bf16")
        for model, heads in (("deepseek-v3.2", 128), ("glm-5.3", 64))
        for form in ("bf16", "fp8_bf16rope_656", "nvfp4_fp8rope_352", "fp8")
    ),
    # GLM-5.3-Flash: a 512-channel latent without RoPE channels, 512 indexer-selected pools of 4 tokens. Its FP8 cache is
    # fp8_528, or fp8_bf16rope_656 with a zero RoPE slot read by a Q zero-padded to 576 channels.
    _ModelCall("glm-5.3-flash", "sparse", 64, 512, _Pool("bf16", 2048), served=True),
    _ModelCall("glm-5.3-flash", "sparse", 64, 512, _Pool("fp8_528", 2048)),
    _ModelCall("glm-5.3-flash", "sparse", 64, 576, _Pool("fp8_bf16rope_656", 2048)),
    # DeepSeek-V4-Flash and V4-Pro: a 512-channel K = V record, a 128-token window and a per-head sink. Compressed layers
    # attend to indexer-selected entries (CSA: top-512 / top-1024) or to every causal compressed block (HCA).
    _ModelCall("deepseek-v4-flash", "window", 64, 512, _Pool("bf16", 128), sink=True, served=True),
    _ModelCall("deepseek-v4-flash", "window", 64, 512, _Pool("fp8", 128), sink=True),
    _ModelCall("deepseek-v4-flash", "csa", 64, 512, _Pool("bf16", 640), indexer_topk=512, sink=True, served=True),
    _ModelCall("deepseek-v4-flash", "csa", 64, 512, _Pool("bf16", 128), _Pool("bf16", 512), sink=True),
    _ModelCall("deepseek-v4-flash", "csa", 64, 512, _Pool("fp8", 128), _Pool("fp8", 512), sink=True),
    _ModelCall("deepseek-v4-flash", "hca", 64, 512, _Pool("bf16", 128), _Pool("bf16", 128), sink=True),
    _ModelCall("deepseek-v4-pro", "csa", 128, 512, _Pool("bf16", 1152), indexer_topk=1024, sink=True, served=True),
    _ModelCall("deepseek-v4-pro", "csa", 128, 512, _Pool("bf16", 128), _Pool("bf16", 1024), sink=True),
    _ModelCall("deepseek-v4-pro", "csa", 128, 512, _Pool("fp8", 128), _Pool("fp8", 1024), sink=True),
    # DeepSeek-V4.1-Flash: as V4-Flash, with window-only layers and compressed layers (ratio 1 or 2) whose indexer picks
    # 512 entries. Its window cache is mxfp8_528 or per-tensor FP8, and its compressed cache nvfp4_288.
    _ModelCall("deepseek-v4.1-flash", "window", 64, 512, _Pool("mxfp8_528", 128), sink=True),
    _ModelCall("deepseek-v4.1-flash", "window", 64, 512, _Pool("fp8", 128), sink=True),
    _ModelCall("deepseek-v4.1-flash", "compressed", 64, 512, _Pool("bf16", 640), indexer_topk=512, sink=True, served=True),
    _ModelCall("deepseek-v4.1-flash", "compressed", 64, 512, _Pool("bf16", 128), _Pool("bf16", 512), sink=True),
    _ModelCall("deepseek-v4.1-flash", "compressed", 64, 512, _Pool("mxfp8_528", 128), _Pool("nvfp4_288", 512), sink=True),
    _ModelCall("deepseek-v4.1-flash", "compressed", 64, 512, _Pool("fp8", 128), _Pool("nvfp4_288", 512), sink=True),
)


def _model_call_id(call: _ModelCall) -> str:
    pools = call.kv.form if call.extra is None else f"{call.kv.form}+{call.extra.form}"
    return f"{call.model}-{call.layer}-{pools}"


def _model_pool(pool: _Pool, d_qk: int, device: torch.device):
    """``(tensor, kv_format, descale)`` of one pool; 64 rows stand for a cache of any size."""
    if pool.form == "bf16":
        return torch.empty((64, d_qk), dtype=torch.bfloat16, device=device), None, None
    if pool.form == "fp8":
        return torch.empty((64, d_qk), dtype=torch.float8_e4m3fn, device=device), None, torch.ones((1,), dtype=torch.float32, device=device)
    from cudnn.deepseek_sparse_attention.sparse_attention_forward._kv_formats import KV_FORMATS

    return torch.empty((64, KV_FORMATS[pool.form].record_bytes), dtype=torch.uint8, device=device), pool.form, None


@pytest.mark.L0
@pytest.mark.parametrize("call", _MODEL_CALLS, ids=_model_call_id)
def test_DSA_sparse_attention_forward_model_calls(call):
    """Each model's attention call is a valid declaration: a served call passes ``check_support``, and every other call
    passes all validation and reaches the decline of the part no kernel serves yet."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        return pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    total_s_q = 2

    def token_ids(slots):
        return torch.zeros((total_s_q, slots), dtype=torch.int32, device=device)

    def lengths(slots):
        return torch.full((total_s_q,), slots, dtype=torch.int32, device=device)

    q = torch.empty((total_s_q, call.heads, call.d_qk), dtype=torch.bfloat16, device=device)
    kv, kv_format, descale_kv = _model_pool(call.kv, call.d_qk, device)
    pool_args = dict(kv_format=kv_format, sample_descale_kv=descale_kv)
    if call.extra is not None:
        extra_kv, extra_kv_format, descale_extra_kv = _model_pool(call.extra, call.d_qk, device)
        pool_args.update(
            sample_extra_kv=extra_kv,
            extra_kv_format=extra_kv_format,
            sample_descale_extra_kv=descale_extra_kv,
            sample_extra_topk_idxs=token_ids(call.extra.slots),
            sample_extra_topk_length=lengths(call.extra.slots),
        )
    plan = DSA.SparseAttentionForward(
        q,
        kv,
        token_ids(call.kv.slots),
        sample_attn_sink=torch.zeros((call.heads,), dtype=torch.float32, device=device) if call.sink else None,
        sample_topk_length=lengths(call.kv.slots),
        indexer_topk=call.indexer_topk,
        **pool_args,
    )
    if call.served:
        assert plan.check_support()
    else:
        with pytest.raises(NotImplementedError, match="SparseAttentionForward does not serve"):
            plan.check_support()


@pytest.mark.L1
def test_DSA_sparse_attention_forward_nondefault_stream_layout_normalization_runtime_scale_cache():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cuda.bindings import driver as cuda
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100 as interface_sm100
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import api as forward_api
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    torch.manual_seed(29)
    device = torch.device("cuda")
    total_s_q, total_s_kv, num_heads, head_dim, logical_topk = 2, 73, 64, 512, 67

    q_storage = torch.randn(total_s_q * num_heads * head_dim + 1, dtype=torch.float32, device=device).to(torch.bfloat16)
    kv_storage = torch.randn(total_s_kv * head_dim + 1, dtype=torch.float32, device=device).to(torch.bfloat16)
    topk_storage = torch.randint(0, total_s_kv, (total_s_q, 2 * logical_topk), dtype=torch.int32, device=device)
    sink_storage = torch.randn(2 * num_heads, dtype=torch.float32, device=device)
    length_storage = torch.tensor([logical_topk, -1, logical_topk - 6, -1], dtype=torch.int32, device=device)
    # These storage-offset views are contiguous but only 2-byte aligned.  The
    # interface must materialize them before promising 16-byte alignment to
    # CuTe's DLPack wrapper.
    q = q_storage[1:].view(total_s_q, num_heads, head_dim)
    kv = kv_storage[1:].view(total_s_kv, head_dim)
    topk_idxs = topk_storage[:, ::2]
    attn_sink = sink_storage[::2]
    topk_length = length_storage[::2]
    assert q.is_contiguous() and kv.is_contiguous()
    assert q.data_ptr() % 16 == 2 and kv.data_ptr() % 16 == 2
    assert all(not tensor.is_contiguous() for tensor in (topk_idxs, attn_sink, topk_length))

    default_stream = torch.cuda.current_stream(device)
    side_stream = torch.cuda.Stream(device=device)
    assert side_stream.cuda_stream != default_stream.cuda_stream
    inputs_ready = torch.cuda.Event()
    inputs_ready.record(default_stream)
    side_stream.wait_event(inputs_ready)
    stream = cuda.CUstream(side_stream.cuda_stream)

    scale_a = 1.0 / math.sqrt(head_dim)
    scale_b = 0.5 / math.sqrt(head_dim)
    result_a = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=scale_a,
        stream=stream,
    )
    api_cache_after_first = {key: id(value) for key, value in forward_api._cache_of_sparse_attention_forward_objects.items()}
    compile_cache_after_first = {key: id(value) for key, value in interface_sm100._compile_cache.items()}

    result_b = DSA.sparse_attention_forward_wrapper(
        q,
        kv,
        topk_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length,
        softmax_scale=scale_b,
        stream=stream,
    )
    assert {key: id(value) for key, value in forward_api._cache_of_sparse_attention_forward_objects.items()} == api_cache_after_first
    assert {key: id(value) for key, value in interface_sm100._compile_cache.items()} == compile_cache_after_first

    outputs_ready = torch.cuda.Event()
    outputs_ready.record(side_stream)
    default_stream.wait_event(outputs_ready)

    for result, scale in ((result_a, scale_a), (result_b, scale_b)):
        ref_out, ref_max, ref_lse, _ = ref_sparse_attention_forward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            topk_length=topk_length,
            softmax_scale=scale,
            return_full=True,
        )
        torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-3, rtol=3.01 / 128)
        torch.testing.assert_close(result["max_logits"], ref_max, atol=2e-5, rtol=2.01 / 65536)
        torch.testing.assert_close(result["lse"], ref_lse, atol=2e-5, rtol=2.01 / 65536)
        assert result["lse_indexer"] is None

    assert not torch.equal(result_a["max_logits"], result_b["max_logits"])


@pytest.mark.L1
def test_DSA_sparse_attention_forward_graph_replay_uses_capture_stream():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    from cudnn import DSA

    torch.manual_seed(1932)
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(73, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    indices = torch.arange(64, device="cuda", dtype=torch.int32).expand(2, -1).contiguous()
    DSA.sparse_attention_forward_wrapper(q, kv, indices)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        DSA.sparse_attention_forward_wrapper(q, kv, indices)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = DSA.sparse_attention_forward_wrapper(q, kv, indices)

    # A cached default-stream handle can silently capture an empty graph.
    # Change the data and poison every output so replay must do real work.
    kv.mul_(2)
    for name in ("out", "max_logits", "lse"):
        result[name].fill_(float("nan"))
    graph.replay()
    ref_out, ref_max, ref_lse, _ = ref_sparse_attention_forward(q, kv, None, indices, return_full=True)
    torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], ref_max, atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-6, rtol=2.01 / 65536)


@pytest.mark.L1
@pytest.mark.parametrize(
    "num_heads,indexer_topk,logical_topk",
    [(64, 512, 576), (128, 1024, 1152), (128, 2048, 2112)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_DSA_sparse_attention_forward_indexer_lse(num_heads, indexer_topk, logical_topk, dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    device = torch.device("cuda")
    total_s_kv = logical_topk + 64
    q = torch.randn(1, num_heads, 512, dtype=dtype, device=device)
    kv = torch.randn(total_s_kv, 512, dtype=dtype, device=device)
    topk_idxs = torch.randperm(total_s_kv, device=device)[:logical_topk].to(torch.int32).unsqueeze(0)
    topk_length = torch.tensor([logical_topk], dtype=torch.int32, device=device)

    result = DSA.sparse_attention_forward_wrapper(q, kv, topk_idxs, topk_length=topk_length, indexer_topk=indexer_topk)
    _, _, _, ref_lse_indexer = ref_sparse_attention_forward(
        q,
        kv,
        None,
        topk_idxs,
        topk_length=topk_length,
        indexer_topk=indexer_topk,
        return_full=True,
    )
    torch.testing.assert_close(result["lse_indexer"], ref_lse_indexer, atol=1e-6, rtol=2.01 / 65536)


@pytest.mark.L0
def test_DSA_sparse_attention_forward_head128_routing():
    """H128 keeps its dedicated D512 kernel up to indexer_topk=1024; other H128 shapes run H64 windows."""
    try:
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100 as interface_sm100
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    for indexer_topk in (0, 512, 1024):
        assert interface_sm100._kernel_variant(128, 512, indexer_topk) == "head128_small_topk_prefill"
    assert interface_sm100._kernel_variant(128, 512, 2048) == "head64_regular"
    for indexer_topk in (0, 512, 1024, 2048):
        assert interface_sm100._kernel_variant(128, 576, indexer_topk) == "head64_regular"
    assert interface_sm100._HEAD64_WINDOW_STARTS[128] == (0, 64)
    with pytest.raises(ValueError, match="Unsupported SparseAttentionForward variant"):
        interface_sm100._kernel_variant(48, 576)


@pytest.mark.L1
@pytest.mark.parametrize("head_dim,indexer_topk", [(576, 0), (576, 2048), (512, 2048)], ids=["D576", "D576-indexer2048", "D512-indexer2048"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_DSA_sparse_attention_forward_head128_windows_match_head64(head_dim, indexer_topk, dtype):
    """Windowed H128 writes, for each 64-head half, the bytes an H64 run on that half writes."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    torch.manual_seed(23)
    device = torch.device("cuda")
    total_s_q, logical_topk = 5, max(indexer_topk, 64) + 65
    total_s_kv = logical_topk + 128
    q = (torch.randn(total_s_q, 128, head_dim, device=device) / math.sqrt(head_dim)).to(dtype)
    kv = (torch.randn(total_s_kv, head_dim, device=device) / math.sqrt(head_dim)).to(dtype)
    topk_idxs = torch.stack([torch.randperm(total_s_kv, device=device)[:logical_topk] for _ in range(total_s_q)]).to(torch.int32)
    topk_idxs[0, -2:] = -1
    topk_length = torch.tensor([logical_topk, logical_topk - 1, logical_topk, max(indexer_topk, 1), 0], dtype=torch.int32, device=device)
    attn_sink = torch.randn(128, dtype=torch.float32, device=device)
    kwargs = dict(topk_length=topk_length, softmax_scale=1.0 / math.sqrt(head_dim), indexer_topk=indexer_topk)

    result = DSA.sparse_attention_forward_wrapper(q, kv, topk_idxs, attn_sink=attn_sink, **kwargs)
    names = ("out", "max_logits", "lse") + (("lse_indexer",) if indexer_topk else ())
    for start in (0, 64):
        heads = slice(start, start + 64)
        window = DSA.sparse_attention_forward_wrapper(q[:, heads].contiguous(), kv, topk_idxs, attn_sink=attn_sink[heads].contiguous(), **kwargs)
        for name in names:
            assert torch.equal(result[name][:, heads], window[name]), f"{name}, heads [{start}, {start + 64})"
    ref_out, ref_max, ref_lse, ref_lse_indexer = ref_sparse_attention_forward(
        q, kv, attn_sink, topk_idxs, topk_length=topk_length, softmax_scale=kwargs["softmax_scale"], indexer_topk=indexer_topk, return_full=True
    )
    torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-6, rtol=2.01 / 65536)
    if indexer_topk:
        live = topk_length >= indexer_topk
        torch.testing.assert_close(result["lse_indexer"][live], ref_lse_indexer[live], atol=1e-6, rtol=2.01 / 65536)
