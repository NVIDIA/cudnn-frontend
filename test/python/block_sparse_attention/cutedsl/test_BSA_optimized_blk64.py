# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and layout regressions for the SM100/SM103 blk64 path."""

import importlib

import pytest
import torch

from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _bsa():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    from cudnn import BSA

    return BSA


def _force_paired_kv(monkeypatch):
    """Route eligible variable-count calls through the paired-KV kernel regardless of the size gate.

    Returns a list that records each paired kernel the dispatcher builds, so a test can prove the route
    without reading compile-cache key positions.
    """
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface, "_SM100_BLK64_PAIRED_KV_MIN_KV_TASKS", 0)
    builds = []
    paired_cls = interface.MixedKvBackwardSm100

    class RecordingMixedKvBackward(paired_cls):
        def __init__(self, *args, **kwargs):
            builds.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(interface, "MixedKvBackwardSm100", RecordingMixedKvBackward)
    return builds


@pytest.mark.L0
@pytest.mark.parametrize("num_q_blocks", [1, 2999, 3000, 4097, 8192, 65536])
def test_large_bucket_policy_is_a_valid_choice(num_q_blocks):
    """The automatic bucket is a performance heuristic, so only its contract is asserted: the choice is either
    the default (``None``) or a usable bucket size."""
    from cudnn.block_sparse_attention.csrc.bwd.sm100_blk64.bsa_bwd_sm100 import sm100_bwd_auto_bucketed_k2q_size_blocks

    choice = sm100_bwd_auto_bucketed_k2q_size_blocks(num_q_blocks)
    assert choice is None or (isinstance(choice, int) and choice >= 1)


@pytest.mark.L0
@pytest.mark.parametrize("misalignment", ["stride", "address", "aligned_stride"])
@pytest.mark.parametrize("tensor_index", [0, 1, 2], ids=["q", "k", "v"])
@pytest.mark.parametrize("splits", [1, 2])
def test_bshd_sliced_inputs(misalignment, tensor_index, splits):
    bsa = _bsa()
    torch.manual_seed(20260911)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = [torch.randn((2, 64, 3, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    padding = 1 if misalignment == "stride" else 8
    start = 1 if misalignment == "address" else 0
    storage = torch.empty((2, 64, 3, 128 + padding), device="cuda", dtype=torch.bfloat16)
    sliced = storage[..., start : start + 128]
    sliced.copy_(inputs[tensor_index])
    inputs[tensor_index] = sliced
    indices = torch.zeros((2, 3, 1, 1), device="cuda", dtype=torch.int32)
    sizes = torch.full((1,), 64, device="cuda", dtype=torch.int32)
    output, lse = bsa.block_sparse_attention_forward(*inputs, indices, 1, sizes, sparse_block_size=64, layout="bshd", kv_splits=splits)
    canonical = [t.transpose(1, 2) for t in inputs]
    mask = block_sparse_mask(indices, 1, sizes, 64, 64, 64)
    ref_output, ref_lse = attention_reference(*canonical, mask)
    assert output.is_contiguous()
    torch.testing.assert_close(output.transpose(1, 2).float(), ref_output, atol=0.03, rtol=0.03)
    torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("clc,splits", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("seqlen", [193, 1089])
def test_forward_empty_rows_and_partial_tiles(clc, splits, seqlen):
    bsa = _bsa()
    torch.manual_seed(19000 + seqlen)
    torch.backends.cuda.matmul.allow_tf32 = False
    q, k, v = [torch.randn((1, 2, seqlen, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    blocks = (seqlen + 63) // 64
    indices = torch.arange(blocks - 1, -1, -1, device="cuda", dtype=torch.int32).view(1, 1, 1, -1).expand(1, 2, blocks, blocks).contiguous()
    counts = (torch.arange(2 * blocks, device="cuda", dtype=torch.int32) % (blocks + 1)).view(1, 2, blocks)
    sizes = torch.full((blocks,), 64, device="cuda", dtype=torch.int32)
    sizes[-1] = seqlen - (blocks - 1) * 64
    mask = block_sparse_mask(indices, 0, sizes, seqlen, seqlen, 64, counts)
    expected, expected_lse = attention_reference(q, k, v, mask)
    output, lse = bsa.block_sparse_attention_forward(
        q,
        k,
        v,
        indices,
        block_sizes=sizes,
        q2k_block_nums=counts,
        sparse_block_size=64,
        use_clc=clc,
        kv_splits=splits,
        allow_empty_block_nums=True,
    )
    torch.testing.assert_close(output.float(), expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(lse, expected_lse, atol=0.03, rtol=0.03)


@pytest.mark.L1
@pytest.mark.parametrize("clc,splits", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("variable", [False, True])
def test_signed_scale_forward_backward_and_cache(monkeypatch, clc, splits, variable):
    bsa = _bsa()
    if variable:
        _force_paired_kv(monkeypatch)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface.bsa_attn_fwd_blk64_cutedsl, "compile_cache", {})
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    torch.manual_seed(27191)
    q = torch.randn((1, 2, 193, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 129, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    indices = torch.arange(3, device="cuda", dtype=torch.int32).view(1, 1, 1, 3).expand(1, 2, 4, 3).contiguous()
    counts = torch.tensor([3, 0, 1, 2], device="cuda", dtype=torch.int32).view(1, 1, 4).expand(1, 2, 4).contiguous() if variable else None
    sizes = torch.tensor([64, 37, 1], device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(indices, 3, sizes, 193, 129, 64, counts)
    nonempty = torch.isfinite(mask).any(-1, keepdim=True)
    kwargs = dict(block_sparse_num=3, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=64)
    cache_sizes = []
    # Changing the sign never recompiles the backward; the forward compiles one extra, separately specialized
    # nonpositive-scale kernel so the positive-scale softmax loop stays free of the masking-order branch.
    for scale in (128**-0.5, 0.0, -(128**-0.5), 128**-0.5):
        qr, kr, vr = [t.double().detach().requires_grad_() for t in (q, k, v)]
        scores = (qr @ kr.transpose(-1, -2)) * scale + mask
        probabilities = torch.where(nonempty, scores, 0).softmax(-1).masked_fill(~torch.isfinite(mask), 0)
        expected = probabilities @ vr
        expected.backward(do.double())
        output, lse = bsa.block_sparse_attention_forward(
            q, k, v, indices, **kwargs, softmax_scale=scale, use_clc=clc, kv_splits=splits, allow_empty_block_nums=True
        )
        gradients = bsa.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, softmax_scale=scale, bucket_size_blocks=3)
        torch.testing.assert_close(lse.double(), scores.logsumexp(-1), atol=0.03, rtol=0.03)
        for got, ref in zip((output, *gradients), (expected, qr.grad, kr.grad, vr.grad)):
            torch.testing.assert_close(got.double(), ref, atol=0.03, rtol=0.03)
        cache_sizes.append((len(interface.bsa_attn_fwd_blk64_cutedsl.compile_cache), len(interface._bsa_attn_bwd_bucketed_k2q_csr.compile_cache)))
    forward_sizes, backward_sizes = zip(*cache_sizes)
    assert set(backward_sizes) == {backward_sizes[0]}
    assert forward_sizes == (forward_sizes[0], forward_sizes[0] + 1, forward_sizes[0] + 1, forward_sizes[0] + 1)


@pytest.mark.L1
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("sq,sk", [(193, 129), (248, 2177), (320, 1088), (513, 1025)])
@pytest.mark.parametrize("variable", [False, True])
def test_backward_partial_tiles_and_buckets(layout, sq, sk, variable, monkeypatch):
    bsa = _bsa()
    if variable:
        _force_paired_kv(monkeypatch)
    torch.manual_seed(27000 + sq)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn((2, 3, sq, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 3, sk, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    nq, nk = (sq + 63) // 64, (sk + 63) // 64
    width = min(4, nk)
    indices = torch.rand((2, 3, nq, nk), device="cuda").argsort(-1)[..., :width].int().contiguous()
    counts = torch.randint(0, width + 1, (2, 3, nq), device="cuda", dtype=torch.int32) if variable else None
    sizes = torch.randint(1, 65, (nk,), device="cuda", dtype=torch.int32)
    sizes[-1].clamp_(max=sk - (nk - 1) * 64)
    mask = block_sparse_mask(indices, width, sizes, sq, sk, 64, counts)
    qr, kr, vr = [t.float().detach().requires_grad_() for t in (q, k, v)]
    scores = qr @ kr.transpose(-1, -2) / 128**0.5 + mask
    nonempty = torch.isfinite(mask).any(-1, keepdim=True)
    probabilities = torch.where(nonempty, scores, 0).softmax(-1).masked_fill(~torch.isfinite(mask), 0)
    expected = probabilities @ vr
    expected.backward(do.float())
    tensors = (q, k, v, do)
    if layout == "bshd":
        tensors = tuple(t.transpose(1, 2).contiguous() for t in tensors)
    q, k, v, do = tensors
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=64, layout=layout)
    output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    canonical_output = output.transpose(1, 2) if layout == "bshd" else output
    torch.testing.assert_close(canonical_output.float(), expected, atol=0.03, rtol=0.03)
    buffers = tuple(torch.empty_like(t) for t in (q, k, v))
    for bucket_size in (1, 3, nq + 1):
        for buffer in buffers:
            buffer.fill_(float("nan"))
        gradients = bsa.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            output,
            lse,
            indices,
            **kwargs,
            bucket_size_blocks=bucket_size,
            dq_tensor=buffers[0],
            dk_tensor=buffers[1],
            dv_tensor=buffers[2],
        )
        for got, buffer, ref in zip(gradients, buffers, (qr.grad, kr.grad, vr.grad)):
            assert got.data_ptr() == buffer.data_ptr()
            got = got.transpose(1, 2) if layout == "bshd" else got
            torch.testing.assert_close(got.float(), ref, atol=0.03, rtol=0.03)


@pytest.mark.L0
@pytest.mark.parametrize("scale", [0.0, -(128**-0.5), float("nan")], ids=["zero", "negative", "nan"])
def test_sage_fp8_rejects_nonpositive_softmax_scale(scale):
    """Sage FP8 folds the softmax scale into its Q*K dequantization scales; a nonpositive scale would turn fully
    masked groups into NaN/+inf, so the public entry point rejects it instead of returning non-finite output."""
    bsa = _bsa()
    q, k, v = [torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    indices = torch.zeros((1, 2, 2, 1), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="softmax_scale must be positive"):
        bsa.block_sparse_attention_fp8_forward(q, k, v, indices, 1, softmax_scale=scale)


@pytest.mark.L0
@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.parametrize("kv_blocks", [17, 35])
def test_paired_multiset_and_graph_replay(monkeypatch, duplicates, kv_blocks):
    bsa = _bsa()
    paired_builds = _force_paired_kv(monkeypatch)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    torch.manual_seed(27123)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn((1, 2, 320, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, kv_blocks * 64, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    indices = torch.tensor([0, 2, 8, kv_blocks - 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 4).expand(1, 2, 5, 4).contiguous()
    counts = torch.full((1, 2, 5), 4, device="cuda", dtype=torch.int32)
    if duplicates:
        indices[:, :, 1, 1] = indices[:, :, 1, 0]
    kwargs = dict(q2k_block_nums=counts, sparse_block_size=64)

    def step():
        output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs)
        gradients = bsa.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, bucket_size_blocks=3)
        return output, *gradients

    eager = step()
    # Duplicate slots contribute repeatedly; a boolean mask would hide a lost edge.
    multiplicity = torch.zeros((1, 2, 320, kv_blocks * 64), device="cuda", dtype=torch.float32)
    for h in range(2):
        for row in range(5):
            for block in indices[0, h, row].tolist():
                multiplicity[0, h, row * 64 : (row + 1) * 64, block * 64 : (block + 1) * 64] += 1
    qr, kr, vr = [t.float().detach().requires_grad_() for t in (q, k, v)]
    ref_output = (qr @ kr.transpose(-1, -2) / 128**0.5 + multiplicity.log()).softmax(-1) @ vr
    ref_output.backward(do.float())
    for got, expected in zip(eager, (ref_output, qr.grad, kr.grad, vr.grad)):
        torch.testing.assert_close(got.float(), expected, atol=0.03, rtol=0.03)
    assert paired_builds, "the dispatcher did not build the paired-KV kernel"
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = step()
        q.mul_(0.8)
        counts.fill_(2)
        graph.replay()
        for got, expected in zip(captured, step()):
            torch.testing.assert_close(got, expected, atol=0.03, rtol=0.03)
    finally:
        graph.reset()


@pytest.mark.L1
def test_paired_cache_rebinds_bitset_lengths(monkeypatch):
    bsa = _bsa()
    paired_builds = _force_paired_kv(monkeypatch)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    monkeypatch.setattr(interface._bsa_attn_bwd_bucketed_k2q_csr, "compile_cache", {})
    cache = interface._bsa_attn_bwd_bucketed_k2q_csr.compile_cache
    for sq in (8192, 8256, 262144, 262208, 8192):
        q = torch.zeros((1, 1, sq, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((1, 1, 128, 128), device="cuda", dtype=q.dtype)
        v, do = torch.full_like(k, 0.25), torch.ones_like(q)
        indices = torch.arange(2, device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 1, sq // 64, 2).contiguous()
        counts = torch.full((1, 1, sq // 64), 2, device="cuda", dtype=torch.int32)
        kwargs = dict(q2k_block_nums=counts, sparse_block_size=64)
        output, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs)
        gradients = bsa.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, bucket_size_blocks=256)
        torch.testing.assert_close(output, torch.full_like(output, 0.25), atol=0, rtol=0)
        for i, gradient in enumerate(gradients):
            expected = torch.full_like(gradient, sq / 128 if i == 2 else 0)
            torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
        assert len(cache) == 1
    assert len(paired_builds) == 1
