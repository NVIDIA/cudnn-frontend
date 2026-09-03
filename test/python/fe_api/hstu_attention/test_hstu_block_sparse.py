# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FP16/BF16 HSTU arbitrary-mask block metadata and consumption."""

from __future__ import annotations

import pytest
import torch

try:
    import cutlass  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f"CuTe DSL is unavailable: {exc}", allow_module_level=True)

from cudnn.hstu_attention import (
    HSTUBwdSm100,
    HSTUFwdSm100,
    hstu_attention_backward,
    hstu_attention_forward,
)
from cudnn.hstu_attention import _interface
from cudnn.hstu_attention._kernels import (
    hstu_bwd_256_cute as hstu_bwd_256_cute_module,
)
from cudnn.hstu_attention._kernels.block_sparse_builder import (
    build_hstu_d256_bwd_block_sparse,
    build_hstu_k2q_block_sparse,
    build_hstu_q2k_block_sparse,
)

from .block_sparse_ref import (
    arbitrary_forward_reference,
    k2q_block_sparse_reference,
    make_arbitrary_func,
    packed_cu_seqlens,
    q2k_block_sparse_reference,
)

pytestmark = [
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
]

_HAS_CUDA = torch.cuda.is_available()
_IS_SM10X = _HAS_CUDA and torch.cuda.get_device_capability()[0] == 10
_DYNAMIC_METADATA_SHAPE_CASES = (
    ((129,), (257,)),
    ((257, 65), (385, 129)),
)


def _assert_q2k_metadata_equal(actual, expected) -> None:
    for field in (
        "mask_block_cnt",
        "mask_block_offset",
        "full_block_cnt",
        "full_block_offset",
    ):
        torch.testing.assert_close(
            getattr(actual, field).cpu(),
            getattr(expected, field),
            rtol=0,
            atol=0,
        )

    for field in ("mask_block_idx", "full_block_idx"):
        expected_indices = getattr(expected, field)
        actual_indices = getattr(actual, field)
        assert actual_indices.numel() >= expected_indices.numel()
        torch.testing.assert_close(
            actual_indices[: expected_indices.numel()].cpu(),
            expected_indices,
            rtol=0,
            atol=0,
        )


def _assert_csr_rows_are_strictly_ascending(metadata) -> None:
    for prefix in ("mask", "full"):
        counts = getattr(metadata, f"{prefix}_block_cnt").flatten().cpu()
        offsets = getattr(metadata, f"{prefix}_block_offset").cpu()
        indices = getattr(metadata, f"{prefix}_block_idx").cpu()
        torch.testing.assert_close(
            offsets[1:] - offsets[:-1],
            counts,
            rtol=0,
            atol=0,
        )
        assert int(offsets[0]) == 0
        assert bool(torch.all(offsets[1:] >= offsets[:-1]))
        for row in range(counts.numel()):
            row_indices = indices[offsets[row] : offsets[row + 1]]
            if row_indices.numel() > 1:
                assert bool(torch.all(row_indices[1:] > row_indices[:-1]))


def _packed_qkv(
    q_lengths,
    k_lengths,
    *,
    head_dim: int,
    dtype: torch.dtype,
):
    torch.manual_seed(20260723 + head_dim)
    q = (
        torch.randn(
            (sum(q_lengths), 1, head_dim),
            dtype=dtype,
            device="cuda",
        )
        * 0.2
    )
    k = (
        torch.randn(
            (sum(k_lengths), 1, head_dim),
            dtype=dtype,
            device="cuda",
        )
        * 0.2
    )
    v = torch.randn_like(k) * 0.2
    cu_q = packed_cu_seqlens(q_lengths, device=q.device)
    cu_k = packed_cu_seqlens(k_lengths, device=q.device)
    return q, k, v, cu_q, cu_k


def _arbitrary_backward_reference(
    q,
    k,
    v,
    do,
    cu_q,
    cu_k,
    func,
    *,
    alpha: float,
    scaling_seqlen: float,
):
    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    out_ref = arbitrary_forward_reference(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    return torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do.float(),
    )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize("pattern", ["empty", "full", "mask", "mixed"])
def test_q2k_builder_matches_reference_for_packed_tails(tile_m, pattern):
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern=pattern,
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
        "block_size": (tile_m, 128),
    }

    actual = build_hstu_q2k_block_sparse(func, cu_q, cu_k, **kwargs)
    expected = q2k_block_sparse_reference(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(actual, expected)
    assert actual.block_size == (tile_m, 128)
    assert actual.orientation == "q2k"


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_q2k_k2q_builders_classify_endpoint_boundaries_exactly():
    endpoints = (0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256)
    tile_q = 128
    q_length = len(endpoints) * tile_q
    k_length = 384
    cu_q = packed_cu_seqlens((q_length,), device="cuda")
    cu_k = packed_cu_seqlens((k_length,), device="cuda")
    func = torch.zeros(
        (1, 1, q_length + 256),
        dtype=torch.int32,
        device="cuda",
    )
    for q_block, endpoint in enumerate(endpoints):
        q_begin = q_block * tile_q
        func[0, 0, q_begin : q_begin + tile_q] = endpoint

    kwargs = {
        "max_seqlen_q": q_length,
        "max_seqlen_k": k_length,
        "block_size": (tile_q, 128),
    }
    q2k = build_hstu_q2k_block_sparse(func, cu_q, cu_k, **kwargs)
    k2q = build_hstu_k2q_block_sparse(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(
        q2k,
        q2k_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )
    _assert_q2k_metadata_equal(
        k2q,
        k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )

    assert q2k.mask_block_cnt.cpu().flatten().tolist() == [
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
    ]
    assert q2k.full_block_cnt.cpu().flatten().tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
    ]
    assert q2k.mask_block_idx[:9].cpu().tolist() == [0] * 8 + [1]
    assert q2k.full_block_idx[:4].cpu().tolist() == [0, 0, 0, 1]

    assert k2q.mask_block_cnt.cpu().flatten().tolist() == [8, 1, 0]
    assert k2q.full_block_cnt.cpu().flatten().tolist() == [3, 1, 0]
    assert k2q.mask_block_idx[:9].cpu().tolist() == list(range(1, 9)) + [10]
    assert k2q.full_block_idx[:4].cpu().tolist() == [9, 10, 11, 11]


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_q2k_builder_rebuilds_mutated_func_on_current_stream():
    q_lengths = (257, 65)
    k_lengths = (385, 129)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="full",
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
        "block_size": (256, 128),
    }
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        full_metadata = build_hstu_q2k_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    stream.synchronize()
    _assert_q2k_metadata_equal(
        full_metadata,
        q2k_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )

    with torch.cuda.stream(stream):
        func.zero_()
        empty_metadata = build_hstu_q2k_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    stream.synchronize()
    expected_empty = q2k_block_sparse_reference(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(empty_metadata, expected_empty)
    assert int(empty_metadata.mask_block_cnt.sum()) == 0
    assert int(empty_metadata.full_block_cnt.sum()) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_q2k_builder_is_cuda_graph_replayable_after_func_mutation():
    q_lengths = (129,)
    k_lengths = (257,)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="full",
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
        "block_size": (128, 128),
    }

    # Compile and finish all lazy setup before capture.
    build_hstu_q2k_block_sparse(func, cu_q, cu_k, **kwargs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = build_hstu_q2k_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    torch.cuda.synchronize()

    func.zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected_empty = q2k_block_sparse_reference(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(captured_metadata, expected_empty)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("tile_q", [128, 256])
@pytest.mark.parametrize("pattern", ["empty", "full", "mask", "mixed"])
def test_k2q_builder_matches_reference_for_packed_tails(tile_q, pattern):
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern=pattern,
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
        "block_size": (tile_q, 128),
    }

    actual = build_hstu_k2q_block_sparse(func, cu_q, cu_k, **kwargs)
    expected = k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(actual, expected)
    _assert_csr_rows_are_strictly_ascending(actual)

    num_k_blocks = (max(k_lengths) + 127) // 128
    num_q_blocks = (max(q_lengths) + tile_q - 1) // tile_q
    assert actual.mask_block_cnt.shape == (
        len(q_lengths),
        1,
        num_k_blocks,
    )
    assert actual.full_block_cnt.shape == (
        len(q_lengths),
        1,
        num_k_blocks,
    )
    assert actual.mask_block_idx.numel() == (len(q_lengths) * num_k_blocks * num_q_blocks)
    assert actual.full_block_idx.numel() == (len(q_lengths) * num_k_blocks * num_q_blocks)
    assert actual.block_size == (tile_q, 128)
    assert actual.orientation == "k2q"


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_k2q_builder_rebuilds_mutated_func_on_current_stream():
    q_lengths = (257, 65)
    k_lengths = (385, 129)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="full",
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
        "block_size": (128, 128),
    }
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        full_metadata = build_hstu_k2q_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    stream.synchronize()
    _assert_q2k_metadata_equal(
        full_metadata,
        k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )

    with torch.cuda.stream(stream):
        func.zero_()
        empty_metadata = build_hstu_k2q_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    stream.synchronize()
    expected_empty = k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(empty_metadata, expected_empty)
    assert int(empty_metadata.mask_block_cnt.sum()) == 0
    assert int(empty_metadata.full_block_cnt.sum()) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_k2q_builder_is_cuda_graph_replayable_after_func_mutation():
    q_lengths = (257,)
    k_lengths = (385,)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="full",
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
        "block_size": (128, 128),
    }

    # Compile and finish all lazy setup before capture.
    build_hstu_k2q_block_sparse(func, cu_q, cu_k, **kwargs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = build_hstu_k2q_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    torch.cuda.synchronize()

    func.zero_()
    graph.replay()
    torch.cuda.synchronize()
    expected_empty = k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(captured_metadata, expected_empty)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("func_num", [5, 7])
def test_k2q_builder_handles_many_intervals_and_logical_tails(func_num):
    q_lengths = (513,)
    k_lengths = (385,)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = torch.zeros(
        (1, func_num, q_lengths[0] + 256),
        dtype=torch.int32,
        device="cuda",
    )
    # Adjacent intervals jointly cover K block 0.  The suffix covers K block
    # 2 and the one-element logical K tail, while block 1 stays empty.
    endpoints = (64, 64, 128, 256, 385, 385, 385)
    for endpoint_idx in range(func_num):
        func[0, endpoint_idx, : q_lengths[0]] = endpoints[endpoint_idx]
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
        "block_size": (128, 128),
    }

    actual = build_hstu_k2q_block_sparse(func, cu_q, cu_k, **kwargs)
    expected = k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs)
    _assert_q2k_metadata_equal(actual, expected)
    _assert_csr_rows_are_strictly_ascending(actual)

    assert actual.full_block_cnt.cpu().tolist() == [[[5, 0, 5, 5]]]
    assert int(actual.mask_block_cnt.sum()) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_k2q_builder_scan_spans_multiple_scan_blocks():
    # 1025 K2Q rows exercise both the per-block scans and the device-side
    # scan of block sums without allocating a large dense reference mask.
    batch_size = 1025
    cu_q = torch.arange(
        batch_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = cu_q.clone()
    func = torch.ones(
        (1, 1, batch_size + 256),
        dtype=torch.int32,
        device="cuda",
    )
    actual = build_hstu_k2q_block_sparse(
        func,
        cu_q,
        cu_k,
        max_seqlen_q=1,
        max_seqlen_k=1,
        block_size=(128, 128),
    )

    assert int(actual.mask_block_cnt.sum()) == 0
    torch.testing.assert_close(
        actual.full_block_cnt,
        torch.ones_like(actual.full_block_cnt),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.full_block_offset,
        torch.arange(
            batch_size + 1,
            dtype=torch.int32,
            device="cuda",
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.full_block_idx,
        torch.zeros(
            batch_size,
            dtype=torch.int32,
            device="cuda",
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_builders_support_batch_size_above_cuda_grid_y_limit():
    # CUDA grid.y is limited to 65535.  Keep one real token in the final
    # sequence so this also verifies coordinate decoding above that boundary.
    # 65537 rows produce 257 first-level scan blocks, exercising prefix
    # propagation across more than one second-level hierarchy block.
    batch_size = 65537
    cu = torch.zeros(
        batch_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    cu[-1] = 1
    func = torch.ones(
        (1, 1, 1 + 256),
        dtype=torch.int32,
        device="cuda",
    )

    q2k = build_hstu_q2k_block_sparse(
        func,
        cu,
        cu,
        max_seqlen_q=1,
        max_seqlen_k=1,
        block_size=(256, 128),
    )
    k2q = build_hstu_k2q_block_sparse(
        func,
        cu,
        cu,
        max_seqlen_q=1,
        max_seqlen_k=1,
        block_size=(128, 128),
    )
    d256_q2k, d256_k2q = build_hstu_d256_bwd_block_sparse(
        func,
        cu,
        cu,
        max_seqlen_q=1,
        max_seqlen_k=1,
    )

    for tensors in (q2k, k2q, d256_q2k, d256_k2q):
        assert int(tensors.mask_block_cnt.sum()) == 0
        assert int(tensors.full_block_cnt.sum()) == 1
        assert int(tensors.full_block_offset[-1]) == 1


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("pattern", ["empty", "full", "mask", "mixed"])
def test_d256_bwd_paired_builder_matches_packed_tail_oracles(pattern):
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern=pattern,
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
    }

    q2k, k2q = build_hstu_d256_bwd_block_sparse(
        func,
        cu_q,
        cu_k,
        **kwargs,
    )
    block_kwargs = {**kwargs, "block_size": (256, 128)}
    expected_q2k = q2k_block_sparse_reference(
        func,
        cu_q,
        cu_k,
        **block_kwargs,
    )
    expected_k2q = k2q_block_sparse_reference(
        func,
        cu_q,
        cu_k,
        **block_kwargs,
    )
    _assert_q2k_metadata_equal(q2k, expected_q2k)
    _assert_q2k_metadata_equal(k2q, expected_k2q)
    _assert_csr_rows_are_strictly_ascending(q2k)
    _assert_csr_rows_are_strictly_ascending(k2q)

    assert q2k.block_size == (256, 128)
    assert q2k.orientation == "q2k"
    assert k2q.block_size == (256, 128)
    assert k2q.orientation == "k2q"


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d256_paired_builder_preserves_transposed_edges_with_different_row_distributions():
    q_length = 4 * 256
    k_length = 5 * 128
    cu_q = packed_cu_seqlens((q_length,), device="cuda")
    cu_k = packed_cu_seqlens((k_length,), device="cuda")
    func = torch.zeros(
        (1, 1, q_length + 256),
        dtype=torch.int32,
        device="cuda",
    )
    for q_block, endpoint in enumerate((0, 128, 255, 512)):
        q_begin = q_block * 256
        func[0, 0, q_begin : q_begin + 256] = endpoint

    q2k, k2q = build_hstu_d256_bwd_block_sparse(
        func,
        cu_q,
        cu_k,
        max_seqlen_q=q_length,
        max_seqlen_k=k_length,
    )
    kwargs = {
        "max_seqlen_q": q_length,
        "max_seqlen_k": k_length,
        "block_size": (256, 128),
    }
    _assert_q2k_metadata_equal(
        q2k,
        q2k_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )
    _assert_q2k_metadata_equal(
        k2q,
        k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )

    q2k_mask_counts = q2k.mask_block_cnt.cpu().flatten().tolist()
    q2k_full_counts = q2k.full_block_cnt.cpu().flatten().tolist()
    k2q_mask_counts = k2q.mask_block_cnt.cpu().flatten().tolist()
    k2q_full_counts = k2q.full_block_cnt.cpu().flatten().tolist()
    assert q2k_mask_counts == [0, 0, 1, 0]
    assert q2k_full_counts == [0, 1, 1, 4]
    assert k2q_mask_counts == [0, 1, 0, 0, 0]
    assert k2q_full_counts == [3, 1, 1, 1, 0]

    assert any(mask + full == 0 for mask, full in zip(q2k_mask_counts, q2k_full_counts))
    assert any(mask + full > 0 for mask, full in zip(q2k_mask_counts, q2k_full_counts))
    assert any(mask + full == 0 for mask, full in zip(k2q_mask_counts, k2q_full_counts))
    assert any(mask + full > 0 for mask, full in zip(k2q_mask_counts, k2q_full_counts))
    assert sum(q2k_mask_counts) == sum(k2q_mask_counts)
    assert sum(q2k_full_counts) == sum(k2q_full_counts)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d256_bwd_q256_supertile_with_different_q128_states_is_mask():
    q_lengths = (256,)
    k_lengths = (128,)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = torch.zeros(
        (1, 1, q_lengths[0] + 256),
        dtype=torch.int32,
        device="cuda",
    )
    # The first Q128 subtile is FULL and the second is EMPTY.  Neither state
    # may be selected independently: the shared Q256 supertile must be MASK.
    func[0, 0, :128] = 128

    q2k, k2q = build_hstu_d256_bwd_block_sparse(
        func,
        cu_q,
        cu_k,
        max_seqlen_q=256,
        max_seqlen_k=128,
    )
    kwargs = {
        "max_seqlen_q": 256,
        "max_seqlen_k": 128,
        "block_size": (256, 128),
    }
    _assert_q2k_metadata_equal(
        q2k,
        q2k_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )
    _assert_q2k_metadata_equal(
        k2q,
        k2q_block_sparse_reference(func, cu_q, cu_k, **kwargs),
    )
    assert q2k.mask_block_cnt.cpu().tolist() == [[[1]]]
    assert q2k.full_block_cnt.cpu().tolist() == [[[0]]]
    assert k2q.mask_block_cnt.cpu().tolist() == [[[1]]]
    assert k2q.full_block_cnt.cpu().tolist() == [[[0]]]
    assert int(q2k.mask_block_idx[0]) == 0
    assert int(k2q.mask_block_idx[0]) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d256_bwd_paired_builder_rebuilds_mutated_func_on_current_stream():
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="full",
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
    }
    stream = torch.cuda.Stream()
    producer_stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        stream.wait_stream(producer_stream)
        full_q2k, full_k2q = build_hstu_d256_bwd_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    stream.synchronize()
    block_kwargs = {**kwargs, "block_size": (256, 128)}
    _assert_q2k_metadata_equal(
        full_q2k,
        q2k_block_sparse_reference(func, cu_q, cu_k, **block_kwargs),
    )
    _assert_q2k_metadata_equal(
        full_k2q,
        k2q_block_sparse_reference(func, cu_q, cu_k, **block_kwargs),
    )

    with torch.cuda.stream(stream):
        func.zero_()
        empty_q2k, empty_k2q = build_hstu_d256_bwd_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    stream.synchronize()
    expected_q2k = q2k_block_sparse_reference(
        func,
        cu_q,
        cu_k,
        **block_kwargs,
    )
    expected_k2q = k2q_block_sparse_reference(
        func,
        cu_q,
        cu_k,
        **block_kwargs,
    )
    _assert_q2k_metadata_equal(empty_q2k, expected_q2k)
    _assert_q2k_metadata_equal(empty_k2q, expected_k2q)
    assert int(empty_q2k.mask_block_cnt.sum()) == 0
    assert int(empty_q2k.full_block_cnt.sum()) == 0
    assert int(empty_k2q.mask_block_cnt.sum()) == 0
    assert int(empty_k2q.full_block_cnt.sum()) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_d256_bwd_paired_builder_is_graph_replayable_after_func_mutation():
    q_lengths = (257,)
    k_lengths = (385,)
    cu_q = packed_cu_seqlens(q_lengths, device="cuda")
    cu_k = packed_cu_seqlens(k_lengths, device="cuda")
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="full",
        device="cuda",
    )
    kwargs = {
        "max_seqlen_q": max(q_lengths),
        "max_seqlen_k": max(k_lengths),
    }

    # Finish JIT setup before capture; every replay still reclassifies the
    # shared state matrix and compacts both orientations.
    build_hstu_d256_bwd_block_sparse(func, cu_q, cu_k, **kwargs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_q2k, captured_k2q = build_hstu_d256_bwd_block_sparse(
            func,
            cu_q,
            cu_k,
            **kwargs,
        )
    torch.cuda.synchronize()

    func.zero_()
    graph.replay()
    torch.cuda.synchronize()
    block_kwargs = {**kwargs, "block_size": (256, 128)}
    _assert_q2k_metadata_equal(
        captured_q2k,
        q2k_block_sparse_reference(func, cu_q, cu_k, **block_kwargs),
    )
    _assert_q2k_metadata_equal(
        captured_k2q,
        k2q_block_sparse_reference(func, cu_q, cu_k, **block_kwargs),
    )


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("pattern", ["empty", "full", "mask", "mixed"])
def test_arbitrary_forward_packed_tails_match_pytorch(dtype, head_dim, pattern):
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=dtype,
    )
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern=pattern,
        device=q.device,
    )
    alpha = 0.7
    scaling_seqlen = 128.0

    expected = arbitrary_forward_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    actual = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )["o_tensor"]

    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=4e-2,
        atol=4e-2,
    )
    if pattern == "empty":
        assert int(torch.count_nonzero(actual)) == 0


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    "dtype,func_num",
    [
        (torch.bfloat16, 5),
        (torch.bfloat16, 7),
        (torch.bfloat16, 9),
        (torch.float16, 9),
    ],
)
def test_arbitrary_forward_handles_many_intervals(dtype, head_dim, func_num):
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=dtype,
    )
    func = torch.zeros(
        (1, func_num, sum(q_lengths) + 256),
        dtype=torch.int32,
        device=q.device,
    )
    q_offset = 0
    for q_length, k_length in zip(q_lengths, k_lengths):
        endpoints = tuple((endpoint + 1) * k_length // func_num for endpoint in range(func_num))
        for endpoint in range(func_num):
            func[0, endpoint, q_offset : q_offset + q_length] = endpoints[endpoint]
        q_offset += q_length

    alpha = 0.7
    scaling_seqlen = 128.0
    expected = arbitrary_forward_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    actual = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )["o_tensor"]

    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=4e-2,
        atol=4e-2,
    )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize(
    "head_dim,expected_block_size",
    [(64, (256, 128)), (256, (128, 128))],
)
def test_fp16_arbitrary_forward_uses_auto_metadata(
    monkeypatch,
    head_dim,
    expected_block_size,
):
    q_lengths = (193, 65)
    k_lengths = (257, 129)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=torch.float16,
    )
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="mixed",
        device=q.device,
    )
    alpha = 0.7
    scaling_seqlen = 96.0
    expected = arbitrary_forward_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )

    original_builder = _interface.build_hstu_q2k_block_sparse
    builder_calls = []

    def record_block_builder(*args, **kwargs):
        builder_calls.append(kwargs["block_size"])
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        _interface,
        "build_hstu_q2k_block_sparse",
        record_block_builder,
    )
    actual = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )["o_tensor"]

    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=5e-2,
        atol=5e-2,
    )
    assert builder_calls
    assert all(block_size == expected_block_size for block_size in builder_calls)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_bf16_forward_mixes_empty_and_nonempty_work_rows(head_dim):
    q_lengths = (300,)
    k_lengths = (385,)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=torch.bfloat16,
    )
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="empty",
        device=q.device,
    )
    q_work_tile = 128 if head_dim == 256 else 256
    func[0, 0, :q_work_tile] = k_lengths[0]

    expected = arbitrary_forward_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        func,
        alpha=0.7,
        scaling_seqlen=128.0,
    )
    actual = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=0.7,
        scaling_seqlen=128.0,
        func_tensor=func,
    )["o_tensor"]

    torch.testing.assert_close(actual.float(), expected, rtol=4e-2, atol=4e-2)
    assert int(torch.count_nonzero(actual[q_work_tile:])) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_graph_rebuilds_metadata_after_func_mutation(dtype):
    q, k, v, cu_q, cu_k = _packed_qkv(
        (256,),
        (256,),
        head_dim=64,
        dtype=dtype,
    )
    func = make_arbitrary_func(
        (256,),
        (256,),
        pattern="full",
        device=q.device,
    )
    out = torch.empty_like(q)
    api = HSTUFwdSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=256,
        max_seqlen_k=256,
        sample_func=func,
        scaling_seqlen=128.0,
    )
    api.check_support()
    api.compile()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.execute(q, k, v, out, cu_q, cu_k, func)
    graph.replay()
    torch.cuda.synchronize()
    full_output = out.clone()
    assert int(torch.count_nonzero(full_output)) > 0

    func.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(out)) == 0

    func.fill_(256)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, full_output, rtol=0, atol=0)


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("pattern", ["full", "mask", "mixed"])
def test_arbitrary_backward_packed_tails_match_pytorch(
    dtype,
    head_dim,
    pattern,
):
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=dtype,
    )
    do = torch.randn_like(q) * 0.2
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern=pattern,
        device=q.device,
    )
    alpha = 0.7
    scaling_seqlen = 128.0
    expected = _arbitrary_backward_reference(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )

    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )

    for name, expected_grad in zip(
        ("dq_tensor", "dk_tensor", "dv_tensor"),
        expected,
    ):
        torch.testing.assert_close(
            actual[name].float(),
            expected_grad,
            rtol=8e-2,
            atol=8e-2,
        )


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize(
    "head_dim,func_num",
    [
        (64, 3),
        (128, 9),
        (256, 3),
        (256, 9),
    ],
)
def test_arbitrary_overlapping_intervals_match_union_semantics(
    head_dim,
    func_num,
):
    q_lengths = (128,)
    k_lengths = (128,)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=torch.bfloat16,
    )
    do = torch.randn_like(q) * 0.2
    func = torch.zeros(
        (1, func_num, sum(q_lengths) + 256),
        dtype=torch.int32,
        device=q.device,
    )
    # Every later interval is nested in the first interval.  In particular,
    # F=(96, 32, 64) keeps [0, 96), while the old complement-of-holes
    # interpretation incorrectly kept only [0, 64).
    func[0, 0, : q_lengths[0]] = 96
    for interval in range(func_num // 2):
        func[0, 2 * interval + 1, : q_lengths[0]] = 32
        func[0, 2 * interval + 2, : q_lengths[0]] = 64

    alpha = 0.7
    # Do not attenuate the backward discrepancy: with 128 here the old
    # complement implementation could fall below the BF16 absolute tolerance.
    scaling_seqlen = 1.0
    expected_out = arbitrary_forward_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    actual_out = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )["o_tensor"]
    torch.testing.assert_close(
        actual_out.float(),
        expected_out,
        rtol=4e-2,
        atol=4e-2,
    )

    expected_grads = _arbitrary_backward_reference(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    actual_grads = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )
    for name, expected_grad in zip(
        ("dq_tensor", "dk_tensor", "dv_tensor"),
        expected_grads,
    ):
        torch.testing.assert_close(
            actual_grads[name].float(),
            expected_grad,
            rtol=8e-2,
            atol=8e-2,
        )


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    "dtype,func_num",
    [
        (torch.bfloat16, 1),
        (torch.bfloat16, 3),
        (torch.bfloat16, 5),
        (torch.bfloat16, 7),
        (torch.bfloat16, 9),
        (torch.float16, 9),
    ],
)
def test_arbitrary_backward_endpoint_prefetch_tracks_each_query_row(
    dtype,
    head_dim,
    func_num,
):
    # One complete Q128 fragment followed by a residual row exercises every
    # fused split-WG and D256 score coordinates; the second packed sequence
    # makes an accidental residual-tile func-row overread observably different.
    q_lengths = (129, 33)
    k_lengths = (193, 129)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=dtype,
    )
    do = torch.randn_like(q) * 0.2
    func = torch.zeros(
        (1, func_num, sum(q_lengths) + 256),
        dtype=torch.int32,
        device=q.device,
    )

    q_offset = 0
    for batch_idx, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        query_row = torch.arange(
            q_length,
            dtype=torch.int32,
            device=q.device,
        )
        for endpoint_idx in range(func_num):
            # Put each endpoint in a disjoint K band.  The vectors therefore
            # stay strictly increasing while changing at every important
            # fragment/chunk boundary (7/8, 15/16, 31/32, ..., 111/112).
            band_begin = endpoint_idx * k_length // func_num
            band_end = (endpoint_idx + 1) * k_length // func_num
            band_span = band_end - band_begin - 1
            assert band_span > 0
            func[
                0,
                endpoint_idx,
                q_offset : q_offset + q_length,
            ] = (
                band_begin + 1 + (query_row * (2 * endpoint_idx + 1) + 3 * endpoint_idx + 7 * batch_idx) % band_span
            )
        q_offset += q_length

    alpha = 0.7
    # Keep gradients large enough that an endpoint-row indexing error cannot
    # hide under the 16-bit absolute tolerance.
    scaling_seqlen = 1.0
    expected = _arbitrary_backward_reference(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        func,
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
    )
    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )

    for name, expected_grad in zip(
        ("dq_tensor", "dk_tensor", "dv_tensor"),
        expected,
    ):
        torch.testing.assert_close(
            actual[name].float(),
            expected_grad,
            rtol=8e-2,
            atol=5e-2,
        )


@pytest.mark.L1
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_arbitrary_backward_all_empty_is_exact_zero(dtype, head_dim):
    q_lengths = (257, 129)
    k_lengths = (385, 257)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=dtype,
    )
    do = torch.randn_like(q) * 0.2
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="empty",
        device=q.device,
    )
    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=0.7,
        scaling_seqlen=128.0,
        func_tensor=func,
    )

    for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
        assert int(torch.count_nonzero(actual[name])) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_d256_empty_preallocated_noncontiguous_outputs_are_zero(dtype):
    q_lengths = (129,)
    k_lengths = (257,)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=256,
        dtype=dtype,
    )
    do = torch.randn_like(q) * 0.2
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="empty",
        device=q.device,
    )

    output_storage = [
        torch.full(
            (reference.shape[0], 2, *reference.shape[1:]),
            7.0,
            dtype=reference.dtype,
            device=reference.device,
        )
        for reference in (q, k, v)
    ]
    dq, dk, dv = (storage[:, 0] for storage in output_storage)
    assert not dq.is_contiguous()
    assert not dk.is_contiguous()
    assert not dv.is_contiguous()

    api = HSTUBwdSm100(
        sample_do=do,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_dq=dq,
        sample_dk=dk,
        sample_dv=dv,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        sample_func=func,
        alpha=0.7,
        scaling_seqlen=128.0,
    )
    api.check_support()
    api.compile()
    torch.cuda.synchronize()

    # compile_only must not initialize user-provided outputs.
    for output in (dq, dk, dv):
        assert bool(torch.all(output == 7.0))

    api.execute(
        do,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        cu_q,
        cu_k,
        func,
    )
    torch.cuda.synchronize()

    for output, storage in zip((dq, dk, dv), output_storage):
        assert int(torch.count_nonzero(output)) == 0
        assert bool(torch.all(storage[:, 1] == 7.0))


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_fp16_backward_exceeds_legacy_q_limit():
    q_length = 32769
    k_length = 32769
    active = 128
    alpha = 0.7
    scaling_seqlen = 128.0
    q, k, v, cu_q, cu_k = _packed_qkv(
        (q_length,),
        (k_length,),
        head_dim=64,
        dtype=torch.float16,
    )
    do = torch.randn_like(q) * 0.2
    func = torch.zeros(
        (1, 1, q_length + 256),
        dtype=torch.int32,
        device=q.device,
    )
    func[0, 0, :active] = active

    q_ref = q[:active].float().detach().requires_grad_(True)
    k_ref = k[:active].float().detach().requires_grad_(True)
    v_ref = v[:active].float().detach().requires_grad_(True)
    scores = alpha * torch.einsum("qhd,khd->hqk", q_ref, k_ref)
    out_ref = (
        torch.einsum(
            "hqk,khd->qhd",
            torch.nn.functional.silu(scores),
            v_ref,
        )
        / scaling_seqlen
    )
    expected = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        do[:active].float(),
    )

    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=q_length,
        max_seqlen_k=k_length,
        window_size=(-1, -1),
        alpha=alpha,
        scaling_seqlen=scaling_seqlen,
        func_tensor=func,
    )
    for name, expected_grad in zip(
        ("dq_tensor", "dk_tensor", "dv_tensor"),
        expected,
    ):
        torch.testing.assert_close(
            actual[name][:active].float(),
            expected_grad,
            rtol=8e-2,
            atol=8e-2,
        )
        assert int(torch.count_nonzero(actual[name][active:])) == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 256])
def test_backward_graph_rebuilds_metadata_after_func_mutation(
    dtype,
    head_dim,
):
    q, k, v, cu_q, cu_k = _packed_qkv(
        (129,),
        (257,),
        head_dim=head_dim,
        dtype=dtype,
    )
    do = torch.randn_like(q) * 0.2
    dq, dk, dv = (torch.empty_like(tensor) for tensor in (q, k, v))
    func = make_arbitrary_func(
        (129,),
        (257,),
        pattern="full",
        device=q.device,
    )
    full_func = func.clone()
    api = HSTUBwdSm100(
        sample_do=do,
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_dq=dq,
        sample_dk=dk,
        sample_dv=dv,
        sample_cu_seqlens_q=cu_q,
        sample_cu_seqlens_k=cu_k,
        max_seqlen_q=129,
        max_seqlen_k=257,
        sample_func=func,
        alpha=0.7,
        scaling_seqlen=128.0,
    )
    api.check_support()
    api.compile()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.execute(
            do,
            q,
            k,
            v,
            dq,
            dk,
            dv,
            cu_q,
            cu_k,
            func,
        )
    graph.replay()
    torch.cuda.synchronize()
    full_grads = tuple(tensor.clone() for tensor in (dq, dk, dv))
    assert all(int(torch.count_nonzero(tensor)) > 0 for tensor in full_grads)

    func.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert all(int(torch.count_nonzero(tensor)) == 0 for tensor in (dq, dk, dv))

    func.copy_(full_func)
    graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip((dq, dk, dv), full_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [64, 256])
def test_fp16_arbitrary_backward_uses_auto_metadata(monkeypatch, head_dim):
    q_lengths = (193, 65)
    k_lengths = (257, 129)
    q, k, v, cu_q, cu_k = _packed_qkv(
        q_lengths,
        k_lengths,
        head_dim=head_dim,
        dtype=torch.float16,
    )
    do = torch.randn_like(q) * 0.2
    func = make_arbitrary_func(
        q_lengths,
        k_lengths,
        pattern="mixed",
        device=q.device,
    )
    expected = _arbitrary_backward_reference(
        q,
        k,
        v,
        do,
        cu_q,
        cu_k,
        func,
        alpha=0.7,
        scaling_seqlen=96.0,
    )

    original_k2q_builder = _interface.build_hstu_k2q_block_sparse
    original_paired_builder = hstu_bwd_256_cute_module.build_hstu_d256_bwd_block_sparse
    k2q_builder_calls = []
    paired_builder_calls = 0

    def record_k2q_builder(*args, **kwargs):
        k2q_builder_calls.append(kwargs["block_size"])
        return original_k2q_builder(*args, **kwargs)

    def record_paired_builder(*args, **kwargs):
        nonlocal paired_builder_calls
        paired_builder_calls += 1
        return original_paired_builder(*args, **kwargs)

    monkeypatch.setattr(
        _interface,
        "build_hstu_k2q_block_sparse",
        record_k2q_builder,
    )
    monkeypatch.setattr(
        hstu_bwd_256_cute_module,
        "build_hstu_d256_bwd_block_sparse",
        record_paired_builder,
    )
    actual = hstu_attention_backward(
        do,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        window_size=(-1, -1),
        alpha=0.7,
        scaling_seqlen=96.0,
        func_tensor=func,
    )

    for name, expected_grad in zip(
        ("dq_tensor", "dk_tensor", "dv_tensor"),
        expected,
    ):
        torch.testing.assert_close(
            actual[name].float(),
            expected_grad,
            rtol=8e-2,
            atol=8e-2,
        )
    if head_dim == 256:
        assert paired_builder_calls > 0
        assert not k2q_builder_calls
    else:
        assert k2q_builder_calls
        assert all(block_size == (128, 128) for block_size in k2q_builder_calls)
        assert paired_builder_calls == 0


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_auto_block_metadata_builder_cache_reuses_dynamic_shapes():
    build_hstu_q2k_block_sparse.compile_cache.clear()
    build_hstu_k2q_block_sparse.compile_cache.clear()
    build_hstu_d256_bwd_block_sparse.compile_cache.clear()

    for q_lengths, k_lengths in _DYNAMIC_METADATA_SHAPE_CASES:
        cu_q = packed_cu_seqlens(q_lengths, device="cuda")
        cu_k = packed_cu_seqlens(k_lengths, device="cuda")
        func = make_arbitrary_func(
            q_lengths,
            k_lengths,
            pattern="mixed",
            device="cuda",
        )
        max_seqlen_q = max(q_lengths)
        max_seqlen_k = max(k_lengths)

        q2k = build_hstu_q2k_block_sparse(
            func,
            cu_q,
            cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size=(256, 128),
        )
        _assert_q2k_metadata_equal(
            q2k,
            q2k_block_sparse_reference(
                func,
                cu_q,
                cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size=(256, 128),
            ),
        )

        k2q = build_hstu_k2q_block_sparse(
            func,
            cu_q,
            cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size=(128, 128),
        )
        _assert_q2k_metadata_equal(
            k2q,
            k2q_block_sparse_reference(
                func,
                cu_q,
                cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_size=(128, 128),
            ),
        )

        paired_q2k, paired_k2q = build_hstu_d256_bwd_block_sparse(
            func,
            cu_q,
            cu_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        paired_kwargs = {
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "block_size": (256, 128),
        }
        _assert_q2k_metadata_equal(
            paired_q2k,
            q2k_block_sparse_reference(
                func,
                cu_q,
                cu_k,
                **paired_kwargs,
            ),
        )
        _assert_q2k_metadata_equal(
            paired_k2q,
            k2q_block_sparse_reference(
                func,
                cu_q,
                cu_k,
                **paired_kwargs,
            ),
        )

    assert len(build_hstu_q2k_block_sparse.compile_cache) == 1
    assert len(build_hstu_k2q_block_sparse.compile_cache) == 1
    assert len(build_hstu_d256_bwd_block_sparse.compile_cache) == 1


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
@pytest.mark.parametrize("head_dim", [64, 256])
def test_auto_block_metadata_consumer_cache_reuses_dynamic_shapes(head_dim):
    _interface.hstu_varlen_fwd_100.compile_cache.clear()
    _interface.hstu_varlen_bwd_100.compile_cache.clear()
    hstu_bwd_256_cute_module.hstu_varlen_bwd_256_cute.compile_cache.clear()
    build_hstu_q2k_block_sparse.compile_cache.clear()
    build_hstu_k2q_block_sparse.compile_cache.clear()
    build_hstu_d256_bwd_block_sparse.compile_cache.clear()

    alpha = 0.7
    scaling_seqlen = 96.0
    for q_lengths, k_lengths in _DYNAMIC_METADATA_SHAPE_CASES:
        q, k, v, cu_q, cu_k = _packed_qkv(
            q_lengths,
            k_lengths,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        )
        do = torch.randn_like(q) * 0.2
        func = make_arbitrary_func(
            q_lengths,
            k_lengths,
            pattern="mixed",
            device=q.device,
        )
        max_seqlen_q = max(q_lengths)
        max_seqlen_k = max(k_lengths)

        actual_out, _ = _interface.hstu_varlen_fwd_100(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_k,
            -1,
            -1,
            alpha,
            func,
            scaling_seqlen=scaling_seqlen,
        )
        expected_out = arbitrary_forward_reference(
            q,
            k,
            v,
            cu_q,
            cu_k,
            func,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )
        torch.testing.assert_close(
            actual_out.float(),
            expected_out,
            rtol=4e-2,
            atol=4e-2,
        )

        expected_grads = _arbitrary_backward_reference(
            q,
            k,
            v,
            do,
            cu_q,
            cu_k,
            func,
            alpha=alpha,
            scaling_seqlen=scaling_seqlen,
        )
        actual_grads = _interface.hstu_varlen_bwd_100(
            do,
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_k,
            None,
            None,
            None,
            -1,
            -1,
            alpha,
            func,
            False,
            scaling_seqlen,
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad.float(),
                expected_grad,
                rtol=8e-2,
                atol=8e-2,
            )

    assert len(_interface.hstu_varlen_fwd_100.compile_cache) == 1
    assert len(build_hstu_q2k_block_sparse.compile_cache) == 1
    if head_dim == 256:
        assert not _interface.hstu_varlen_bwd_100.compile_cache
        assert len(hstu_bwd_256_cute_module.hstu_varlen_bwd_256_cute.compile_cache) == 1
        assert not build_hstu_k2q_block_sparse.compile_cache
        assert len(build_hstu_d256_bwd_block_sparse.compile_cache) == 1
    else:
        assert len(_interface.hstu_varlen_bwd_100.compile_cache) == 1
        assert not hstu_bwd_256_cute_module.hstu_varlen_bwd_256_cute.compile_cache
        assert len(build_hstu_k2q_block_sparse.compile_cache) == 1
        assert not build_hstu_d256_bwd_block_sparse.compile_cache


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_bf16_builder_cache_hit_does_not_read_device_metadata_on_host(
    monkeypatch,
):
    q, k, v, cu_q, cu_k = _packed_qkv(
        (128,),
        (257,),
        head_dim=64,
        dtype=torch.bfloat16,
    )
    func = make_arbitrary_func(
        (128,),
        (257,),
        pattern="mixed",
        device=q.device,
    )
    kwargs = {
        "max_seqlen_q": 128,
        "max_seqlen_k": 257,
        "func_tensor": func,
        "scaling_seqlen": 64.0,
    }
    expected = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        **kwargs,
    )["o_tensor"]
    torch.cuda.synchronize()

    def fail_host_inspection(*_args, **_kwargs):
        raise AssertionError("unexpected CUDA metadata value inspection")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", fail_host_inspection)
        patch.setattr(torch.Tensor, "item", fail_host_inspection)
        patch.setattr(torch.Tensor, "tolist", fail_host_inspection)
        actual = hstu_attention_forward(
            q,
            k,
            v,
            cu_q,
            cu_k,
            **kwargs,
        )["o_tensor"]

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.L0
@pytest.mark.skipif(not _IS_SM10X, reason="requires an SM10x Blackwell GPU")
def test_fp16_forward_exceeds_legacy_k_limit():
    q_length = 128
    k_length = 65537
    q, k, v, cu_q, cu_k = _packed_qkv(
        (q_length,),
        (k_length,),
        head_dim=64,
        dtype=torch.float16,
    )
    func = make_arbitrary_func(
        (q_length,),
        (k_length,),
        pattern="empty",
        device=q.device,
    )
    func[0, 0, :q_length] = 128

    actual = hstu_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen_q=q_length,
        max_seqlen_k=k_length,
        func_tensor=func,
        scaling_seqlen=64.0,
    )["o_tensor"]
    expected = arbitrary_forward_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        func,
        alpha=1.0,
        scaling_seqlen=64.0,
    )
    torch.testing.assert_close(actual.float(), expected, rtol=4e-2, atol=4e-2)
