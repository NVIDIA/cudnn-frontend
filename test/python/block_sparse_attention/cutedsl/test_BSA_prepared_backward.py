# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``block_sparse_attention_prepare_backward`` / ``block_sparse_attention_backward(metadata=...)`` on SM90 blk64.

A prepared object must give the gradients of the default backward (within the FP32 reduce-add order noise), skip every
metadata kernel, work on both the paired-KV and the regular SM90 path, in BHSD and BSHD, under CUDA graph capture and
without synchronizing, be rejected with a clear error when it does not describe the call's metadata, and survive its release
right after a backward on another stream (the backward records the tensors it reads on its own stream).
"""

import importlib
import re

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask
from block_sparse_attention.cutedsl.bsa_utils import cuda_graph, import_bsa, no_host_sync

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

BLOCK = 64
# Kernel-name fragments of the backward's metadata launches: the K2Q CSR build and the paired-KV pairing.
METADATA_KERNELS = ("count_edges", "offsets_kernel", "scatter_q_indices", "local_pair", "PairedKvNeighborhoods", "LocalPairMatch8", "LocalPairPartition8")


def _sm90(reason="the prepared backward serves the SM90 blk64 backward"):
    bsa = import_bsa(families=(9,), reason=reason)
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    return bsa, interface


def _force_paired_kv(monkeypatch, interface):
    """Route eligible variable-count calls through the paired-KV kernel regardless of the task-count gate."""
    monkeypatch.setattr(interface, "_BLK64_PAIRED_KV_MIN_KV_TASKS", 0)


def _problem(batch, heads, nq, nk, width, seed, *, kv_tail=0):
    """BF16 inputs and a local-window mask with variable counts, partial KV blocks and empty rows; a KV tail when asked."""
    torch.manual_seed(seed)
    sq, sk = nq * BLOCK, nk * BLOCK - kv_tail
    q = torch.randn((batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16)
    v, do = torch.randn_like(k), torch.randn_like(q)
    centers = (torch.arange(nq, device="cuda") * nk // nq).clamp(max=nk - width)
    window = centers[:, None] + torch.arange(width, device="cuda")
    indices = window.expand(batch, heads, nq, width).int().contiguous()
    counts = torch.randint(max(1, width - 2), width + 1, (batch, heads, nq), device="cuda", dtype=torch.int32)
    counts[:, :, ::7] = 0
    sizes = torch.randint(1, BLOCK + 1, (nk,), device="cuda", dtype=torch.int32)
    sizes[::3] = BLOCK
    sizes[-1].clamp_(max=sk - (nk - 1) * BLOCK)
    return q, k, v, do, indices, counts, sizes


def _reference_grads(q, k, v, do, indices, width, sizes, counts):
    mask = block_sparse_mask(indices, width, sizes, q.shape[2], k.shape[2], BLOCK, counts)
    _, _, dq, dk, dv = attention_reference(q, k, v, mask, do=do)
    return dq, dk, dv


def _kernel_names(fn):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def _metadata_kernels(names):
    return [n for n in names if any(fragment in n for fragment in METADATA_KERNELS)]


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("paired", [True, False], ids=["paired", "regular"])
def test_prepared_backward_matches_default_and_skips_metadata_kernels(monkeypatch, layout, paired):
    bsa, interface = _sm90()
    if paired:
        _force_paired_kv(monkeypatch, interface)
    batch, heads, nq, nk, width = 2, 2, 17, 34, 6
    q, k, v, do, indices, counts, sizes = _problem(batch, heads, nq, nk, width, seed=31 + paired)
    dq_ref, dk_ref, dv_ref = _reference_grads(q, k, v, do, indices, width, sizes, counts)
    tensors = (q, k, v, do)
    if layout == "bshd":
        tensors = tuple(t.transpose(1, 2).contiguous() for t in tensors)
    q, k, v, do = tensors
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=BLOCK, layout=layout)
    o, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)

    # Two groups of 9 Q blocks in the paired case; one group (direct dK/dV stores) in the regular case.
    bucket = 9 if paired else None
    prepared = bsa.block_sparse_attention_prepare_backward(
        indices, width, sizes, counts, seqlen_q=q.shape[1 if layout == "bshd" else 2], seqlen_k=sizes.numel() * BLOCK, bucket_size_blocks=bucket
    )
    assert prepared.use_paired_kv == paired
    assert prepared.direct_kv == (not paired)
    assert prepared.bucket_size_blocks == (bucket or 384)
    default = bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, bucket_size_blocks=bucket)
    with_meta = bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, metadata=prepared)
    # A second call over the same object: the metadata is reusable, not consumed.
    again = bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, bucket_size_blocks=bucket, metadata=prepared)
    for got_default, got_meta, got_again, ref in zip(default, with_meta, again, (dq_ref, dk_ref, dv_ref)):
        if layout == "bshd":
            got_default, got_meta, got_again = (t.transpose(1, 2) for t in (got_default, got_meta, got_again))
        torch.testing.assert_close(got_meta.float(), ref, atol=0.03, rtol=0.03)
        # Same kernels over the same metadata: only the FP32 reduce-add order may differ.
        torch.testing.assert_close(got_meta.float(), got_default.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(got_again.float(), got_default.float(), atol=1e-2, rtol=1e-2)

    default_names = _kernel_names(lambda: bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, bucket_size_blocks=bucket))
    prepared_names = _kernel_names(lambda: bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, metadata=prepared))
    default_metadata = _metadata_kernels(default_names)
    assert len(default_metadata) >= (8 if paired else 3), default_names
    assert not _metadata_kernels(prepared_names), prepared_names
    # The compute kernels are the same ones in the same order.
    compute = lambda names: [n for n in names if "bsa_bwd_sm90" in n]
    assert compute(prepared_names) == compute(default_names), (prepared_names, default_names)
    assert len(compute(prepared_names)) == (5 if paired else 3)  # sum_OdO, main, converts (dQ, dK, dV) / sum_OdO, main, convert dQ (direct dK/dV)


@pytest.mark.L0
def test_prepared_backward_rejects_stale_or_foreign_metadata(monkeypatch):
    bsa, interface = _sm90()
    _force_paired_kv(monkeypatch, interface)
    batch, heads, nq, nk, width = 1, 2, 10, 20, 5
    q, k, v, do, indices, counts, sizes = _problem(batch, heads, nq, nk, width, seed=41)
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=BLOCK)
    o, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    prepare = lambda **kw: bsa.block_sparse_attention_prepare_backward(indices, width, sizes, counts, seqlen_q=q.shape[2], seqlen_k=k.shape[2], **kw)
    backward = lambda meta, **kw: bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **{**kwargs, **kw}, metadata=meta)
    prepared = prepare()
    backward(prepared)

    # Other tensors with the same contents and shapes are other metadata.
    with pytest.raises(ValueError, match="different q2k_block_index"):
        bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices.clone(), **kwargs, metadata=prepared)
    with pytest.raises(ValueError, match="different q2k_block_nums"):
        backward(prepared, q2k_block_nums=counts.clone())
    with pytest.raises(ValueError, match="different block_sizes"):
        backward(prepared, block_sizes=sizes.clone())
    # Fixed counts are a different pattern from the same index tensor; block_sparse_num is then part of the identity.
    fixed = bsa.block_sparse_attention_prepare_backward(indices, width, sizes, None, seqlen_q=q.shape[2], seqlen_k=k.shape[2])
    with pytest.raises(ValueError, match="different block_sparse_num"):
        backward(fixed)
    with pytest.raises(ValueError, match="different block_sparse_num"):
        bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, width - 1, sizes, None, sparse_block_size=BLOCK, metadata=fixed)
    # An in-place change of a metadata tensor after the build is stale metadata.
    counts[0, 0, 0] = 1
    with pytest.raises(ValueError, match="q2k_block_nums modified in place"):
        backward(prepared)
    prepared = prepare()
    sizes[1] = 7
    with pytest.raises(ValueError, match="block_sizes modified in place"):
        backward(prepared)
    prepared = prepare()
    indices[0, 0, 0, 0] = indices[0, 0, 0, 1]
    with pytest.raises(ValueError, match="q2k_block_index modified in place"):
        backward(prepared)
    prepared = prepare(bucket_size_blocks=4)
    backward(prepared)  # None defers to the object's bucket
    backward(prepared, bucket_size_blocks=4)
    with pytest.raises(ValueError, match="bucket_size_blocks=4"):
        backward(prepared, bucket_size_blocks=5)
    # Another problem shape.
    with pytest.raises(ValueError, match="query / .* key tokens"):
        bsa.block_sparse_attention_backward(
            do[:, :, :BLOCK],
            q[:, :, :BLOCK],
            k,
            v,
            o[:, :, :BLOCK],
            lse[:, :, :BLOCK],
            indices[:, :, :1],
            width,
            sizes,
            counts[:, :, :1],
            sparse_block_size=BLOCK,
            metadata=prepared,
        )
    with pytest.raises(TypeError, match="BsaBackwardMetadata"):
        backward(object())
    # Exclusive with kpage and reserved for the SM90 blk64 path.
    with pytest.raises(NotImplementedError, match="exclusive"):
        bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, width, sizes, counts, sparse_block_size=128, kpage=True, metadata=prepared)
    with pytest.raises(NotImplementedError, match="SM90 blk64"):
        bsa.block_sparse_attention_prepare_backward(indices, width, sizes, counts, seqlen_q=q.shape[2], seqlen_k=k.shape[2], sparse_block_size=128)
    with pytest.raises(ValueError, match="shape prefix"):
        bsa.block_sparse_attention_prepare_backward(indices, width, sizes, counts, seqlen_q=q.shape[2] + BLOCK, seqlen_k=k.shape[2])
    with pytest.raises(ValueError, match="block_sparse_num must be in"):
        bsa.block_sparse_attention_prepare_backward(indices, width + 1, sizes, None, seqlen_q=q.shape[2], seqlen_k=k.shape[2])
    assert re.match(
        r"BsaBackwardMetadata\(batch_size=1, num_heads=2, seqlen_q=640, seqlen_k=1280, bucket_size_blocks=4, num_q_groups=3, path=paired", repr(prepared)
    )


@pytest.mark.L0
@pytest.mark.parametrize("paired", [True, False], ids=["paired", "regular"])
def test_prepared_backward_does_not_sync_and_replays_in_a_graph(monkeypatch, paired):
    bsa, interface = _sm90()
    if paired:
        _force_paired_kv(monkeypatch, interface)
    batch, heads, nq, nk, width = 1, 2, 12, 23, 5
    # A KV tail keeps the regular path on the FP32 dK/dV workspace too (direct_kv needs whole KV blocks).
    q, k, v, do, indices, counts, sizes = _problem(batch, heads, nq, nk, width, seed=51 + paired, kv_tail=0 if paired else 24)
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=BLOCK)
    o, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    step = lambda meta: bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, bucket_size_blocks=5, metadata=meta)
    prepared = bsa.block_sparse_attention_prepare_backward(indices, width, sizes, counts, seqlen_q=q.shape[2], seqlen_k=k.shape[2], bucket_size_blocks=5)
    assert prepared.use_paired_kv == paired and not prepared.direct_kv
    eager = step(prepared)
    torch.cuda.synchronize()
    with no_host_sync():
        # Warm (compiled) prepare and backward never read the device or block the host.
        warm = bsa.block_sparse_attention_prepare_backward(indices, width, sizes, counts, seqlen_q=q.shape[2], seqlen_k=k.shape[2], bucket_size_blocks=5)
        step(warm)
        step(None)
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            captured = step(prepared)
        do.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        for got, expected in zip(captured, step(prepared)):
            torch.testing.assert_close(got.float(), expected.float(), atol=1e-2, rtol=1e-2)
        for got, expected in zip(captured, eager):
            torch.testing.assert_close(got.float(), (expected.float() * 0.5), atol=1e-2, rtol=1e-2)


@pytest.mark.L0
def test_prepare_on_a_side_stream_orders_before_the_backward(monkeypatch):
    """The prepare runs on the current stream; a side stream plus the usual event ordering is enough for the backward."""
    bsa, interface = _sm90()
    _force_paired_kv(monkeypatch, interface)
    q, k, v, do, indices, counts, sizes = _problem(1, 2, 16, 32, 6, seed=61)
    kwargs = dict(block_sparse_num=6, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=BLOCK)
    o, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    reference = bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        prepared = bsa.block_sparse_attention_prepare_backward(indices, 6, sizes, counts, seqlen_q=q.shape[2], seqlen_k=k.shape[2])
    torch.cuda.current_stream().wait_stream(side)
    grads = bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, metadata=prepared)
    for got, expected in zip(grads, reference):
        torch.testing.assert_close(got.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@pytest.mark.parametrize("paired", [True, False], ids=["paired", "regular"])
def test_prepared_metadata_built_on_a_side_stream_survives_its_release_after_the_backward(monkeypatch, paired):
    """R1 across streams: the backward records the prepared tensors it reads on its own stream, so an object built on a side
    stream and released right after the backward call keeps its blocks out of that side stream's pool until the kernels ran."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep is unavailable")
    bsa, interface = _sm90()
    if paired:
        _force_paired_kv(monkeypatch, interface)
    q, k, v, do, indices, counts, sizes = _problem(1, 2, 16, 32, 6, seed=71 + paired)
    kwargs = dict(block_sparse_num=6, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=BLOCK)
    o, lse = bsa.block_sparse_attention_forward(q, k, v, indices, **kwargs, allow_empty_block_nums=True)
    reference = bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs)
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    prepare = lambda: bsa.block_sparse_attention_prepare_backward(indices, 6, sizes, counts, seqlen_q=q.shape[2], seqlen_k=k.shape[2])
    backward = lambda meta: bsa.block_sparse_attention_backward(do, q, k, v, o, lse, indices, **kwargs, metadata=meta)
    # Every kernel compiled and launched once outside the timed sequence (a first launch waits for the device to drain).
    with torch.cuda.stream(side):
        warm = prepare()
    main.wait_stream(side)
    backward(warm)
    del warm
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    recorded = []
    original_record_stream = torch.Tensor.record_stream

    def record_stream_spy(tensor, stream):
        recorded.append((tensor.data_ptr(), stream))
        return original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream_spy)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        prepared = prepare()
    assert prepared.use_paired_kv == paired
    assert not recorded  # the prepare itself records nothing
    # What the compute entry reads: the CSR, the (B, nk) block_sizes copy and, when paired, the task records and the
    # arrival-bookkeeping template (the backward copies it on its own stream); the pair bitsets, duplicates, flags and
    # residuals feed the metadata kernels only.
    consumed = (prepared._offsets, prepared._indices, prepared._block_sizes, *((*prepared._pair_tensors[4:], prepared._pair_sync) if paired else ()))
    released = [(t.data_ptr(), t.numel() * t.element_size()) for t in consumed]
    probe_specs = [(tuple(t.shape), t.dtype) for t in consumed]
    main.wait_stream(side)
    torch.cuda._sleep(1_000_000_000)  # park the backward's stream: its kernels queue behind this and read the metadata later
    grads = backward(prepared)
    recorded_on_main = {ptr for ptr, stream in recorded if stream == main}
    assert {ptr for ptr, _ in released} <= recorded_on_main, (released, recorded)
    del prepared, consumed  # the caller releases the object while the backward is still pending
    with torch.cuda.stream(side):
        probes = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in probe_specs]
        for probe in probes:
            begin, end = probe.data_ptr(), probe.data_ptr() + probe.numel() * probe.element_size()
            overlap = [ptr for ptr, nbytes in released if ptr < end and begin < ptr + nbytes]
            assert not overlap, "a released metadata block was handed to the side stream while the backward still reads it"
        for probe in probes:
            probe.fill_(-1)  # harmless only because of the assertion above: these are other blocks
    torch.cuda.synchronize()
    for got, expected in zip(grads, reference):
        torch.testing.assert_close(got.float(), expected.float(), atol=1e-2, rtol=1e-2)
    # The default path allocates its metadata on the backward's own stream and records nothing (its host sequence is unchanged).
    recorded.clear()
    backward(None)
    torch.cuda.synchronize()
    assert not recorded
