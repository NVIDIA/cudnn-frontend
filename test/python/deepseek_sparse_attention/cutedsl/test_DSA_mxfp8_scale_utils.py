# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SM100 MXFP8 scale-layout contract."""

import pytest
import torch

from cudnn.deepseek_sparse_attention.utils.sm100 import mxfp8_scale_utils

pytestmark = pytest.mark.L0


@pytest.fixture
def cuda_scale_packer():
    """Keep host coverage on lanes below the CUDA packer's requirements."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state
    from cudnn.deepseek_sparse_attention.utils.compiler import gpu_arch_flag

    if not cutedsl_state()[0]:
        pytest.skip("requires CuTe DSL")
    requirement_error = cutedsl_requirement_error("SM100 DSA MXFP8 scale packing")
    if requirement_error:
        pytest.skip(requirement_error)
    try:
        gpu_arch_flag()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_mxfp8_scale_pack_roundtrip_host_only():
    """Pack and unpack a non-tile-aligned logical scale tensor."""
    logical = torch.arange(130 * 5 * 3, dtype=torch.int64).reshape(130, 5, 3)

    packed = mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical)

    assert packed.shape == (3, 256, 8)
    assert packed.is_contiguous()
    unpacked = mxfp8_scale_utils.unpack_blockscaled_scale_mkl(packed, mn=130, sf_groups=5)
    assert torch.equal(unpacked, logical)


def test_mxfp8_scale_pack_reuses_caller_owned_output_host_only():
    """Reuse exact caller-owned storage in the reference packer."""
    logical = torch.arange(130 * 5 * 3, dtype=torch.int64).reshape(130, 5, 3)
    out = torch.full((3, 256, 8), -1, dtype=torch.int64)

    packed = mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical, out=out)

    assert packed is out
    unpacked = mxfp8_scale_utils.unpack_blockscaled_scale_mkl(packed, mn=130, sf_groups=5)
    assert torch.equal(unpacked, logical)
    assert torch.count_nonzero(packed == -1) == 0


def test_mxfp8_scale_pack_rejects_invalid_caller_owned_output_host_only():
    """Reject wrong shapes, dtypes, and aliased caller storage."""
    logical = torch.arange(128 * 4, dtype=torch.int64).reshape(128, 4, 1)
    with pytest.raises(ValueError, match="out must have shape"):
        mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical, out=torch.empty((1, 64, 4), dtype=torch.int64))
    with pytest.raises(ValueError, match="out must have dtype"):
        mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical, out=torch.empty((1, 128, 4), dtype=torch.int32))
    with pytest.raises(ValueError, match="must not alias"):
        mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical, out=logical.reshape(1, 128, 4))


def test_mxfp8_scale_pack_uses_blackwell_datapath_order_host_only():
    """Match the Blackwell blockscaled scale-factor datapath order."""
    logical = torch.arange(128 * 4, dtype=torch.int64).reshape(128, 4, 1)

    packed = mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical).reshape(-1)

    expected_first_datapath_words = logical[[0, 32, 64, 96], :, 0].reshape(-1)
    expected_second_datapath_words = logical[[1, 33, 65, 97], :, 0].reshape(-1)
    assert torch.equal(packed[:16], expected_first_datapath_words)
    assert torch.equal(packed[16:32], expected_second_datapath_words)


@pytest.mark.parametrize(
    "qhead_per_kv_head,seqlen_q",
    [
        pytest.param(64, 2, id="qh64"),
        pytest.param(32, 4, id="qh32"),
    ],
)
def test_mxfp8_scale_pack_order_bshd_q_host_only(qhead_per_kv_head, seqlen_q):
    """Map BSHD Q scales to logical MKL order for supported GQA ratios."""
    bs, n_heads_q, sf_groups = 2, qhead_per_kv_head, 4
    batch = torch.arange(bs, dtype=torch.int64).view(bs, 1, 1, 1) * 100_000
    token = torch.arange(seqlen_q, dtype=torch.int64).view(1, seqlen_q, 1, 1) * 1_000
    head = torch.arange(n_heads_q, dtype=torch.int64).view(1, 1, n_heads_q, 1) * 10
    group = torch.arange(sf_groups, dtype=torch.int64).view(1, 1, 1, sf_groups)
    q_scale = batch + token + head + group

    q_mkl = mxfp8_scale_utils.logical_q_scale_to_mkl(
        q_scale,
        qhead_per_kv_head=qhead_per_kv_head,
    )

    assert q_mkl.shape == (128, 4, 2)
    assert torch.equal(q_mkl[0, :, 0], q_scale[0, 0, 0, :])
    assert torch.equal(
        q_mkl[qhead_per_kv_head - 1, :, 0],
        q_scale[0, 0, qhead_per_kv_head - 1, :],
    )
    assert torch.equal(q_mkl[qhead_per_kv_head, :, 0], q_scale[0, 1, 0, :])
    assert torch.equal(q_mkl[127, :, 0], q_scale[0, seqlen_q - 1, qhead_per_kv_head - 1, :])
    assert torch.equal(q_mkl[0, :, 1], q_scale[1, 0, 0, :])

    q_packed = mxfp8_scale_utils.pack_q_scale_bshd(
        q_scale,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    q_unpacked = mxfp8_scale_utils.unpack_blockscaled_scale_mkl(q_packed, mn=128, sf_groups=4)
    assert torch.equal(q_unpacked, q_mkl)


def test_mxfp8_scale_pack_order_bshd_k_l_dimension_host_only():
    """Flatten BSHD K batch and KV heads into the packed L dimension."""
    bs, seqlen_k, n_heads_kv, sf_groups = 2, 3, 2, 4
    batch = torch.arange(bs, dtype=torch.int64).view(bs, 1, 1, 1) * 100_000
    token = torch.arange(seqlen_k, dtype=torch.int64).view(1, seqlen_k, 1, 1) * 1_000
    head = torch.arange(n_heads_kv, dtype=torch.int64).view(1, 1, n_heads_kv, 1) * 10
    group = torch.arange(sf_groups, dtype=torch.int64).view(1, 1, 1, sf_groups)
    k_scale = batch + token + head + group

    k_mkl = mxfp8_scale_utils.logical_k_scale_to_mkl(k_scale)

    assert k_mkl.shape == (3, 4, 4)
    assert torch.equal(k_mkl[0, :, 0], k_scale[0, 0, 0, :])
    assert torch.equal(k_mkl[0, :, 1], k_scale[0, 0, 1, :])
    assert torch.equal(k_mkl[0, :, 2], k_scale[1, 0, 0, :])
    assert torch.equal(k_mkl[2, :, 3], k_scale[1, 2, 1, :])

    k_packed = mxfp8_scale_utils.pack_k_scale_bshd(k_scale)
    k_unpacked = mxfp8_scale_utils.unpack_blockscaled_scale_mkl(k_packed, mn=3, sf_groups=4)
    assert torch.equal(k_unpacked, k_mkl)


def test_mxfp8_scale_pack_qk_bshd_matches_individual_packers_host_only():
    """Match fused reference results to the individual BSHD packers."""
    bs, sq, sk, hq, hkv, groups, qhpkv = 2, 3, 5, 8, 2, 5, 4
    q = torch.arange(bs * sq * hq * groups, dtype=torch.int64).reshape(bs, sq, hq, groups)
    k = torch.arange(bs * sk * hkv * groups, dtype=torch.int64).reshape(bs, sk, hkv, groups)

    q_packed, k_packed = mxfp8_scale_utils.pack_qk_scale_bshd(
        q,
        k,
        qhead_per_kv_head=qhpkv,
    )

    assert torch.equal(q_packed, mxfp8_scale_utils.pack_q_scale_bshd(q, qhpkv))
    assert torch.equal(k_packed, mxfp8_scale_utils.pack_k_scale_bshd(k))


@pytest.mark.parametrize(
    "batch,seqlen_q,seqlen_k",
    [
        pytest.param(1, 0, 0, id="both-sequences-empty"),
        pytest.param(1, 0, 5, id="q-sequence-empty"),
        pytest.param(1, 3, 0, id="k-sequence-empty"),
        pytest.param(0, 3, 5, id="batch-empty"),
    ],
)
def test_mxfp8_scale_pack_qk_bshd_accepts_empty_extents_host_only(batch, seqlen_q, seqlen_k):
    """Do not mistake the null storage pointer of empty tensors for aliasing."""
    qhpkv, hkv, groups = 64, 1, 5
    q = torch.empty((batch, seqlen_q, hkv * qhpkv, groups), dtype=torch.int64)
    k = torch.empty((batch, seqlen_k, hkv, groups), dtype=torch.int64)

    q_packed, k_packed = mxfp8_scale_utils.pack_qk_scale_bshd(q, k, qhpkv)

    assert torch.equal(q_packed, mxfp8_scale_utils.pack_q_scale_bshd(q, qhpkv))
    assert torch.equal(k_packed, mxfp8_scale_utils.pack_k_scale_bshd(k))


def test_mxfp8_scale_pack_qk_bshd_rejects_shared_nonempty_storage_host_only():
    """Keep source/output and output/output alias rejection for live storage."""
    q = torch.empty((1, 2, 64, 4), dtype=torch.int64)
    k = torch.empty((1, 128, 1, 4), dtype=torch.int64)
    packed_shape = (1, 128, 4)

    with pytest.raises(ValueError, match="q_out must not alias either source"):
        mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 64, q_out=k.view(packed_shape))

    shared_out = torch.empty(packed_shape, dtype=q.dtype)
    with pytest.raises(ValueError, match="q_out and k_out must not alias"):
        mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 64, q_out=shared_out, k_out=shared_out)


def test_mxfp8_scale_pack_order_thd_with_padding_host_only():
    """Preserve logical THD rows while zero-filling sequence padding."""
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)
    qhead_per_kv_head = 4
    n_heads_kv, sf_groups = 2, 4
    n_heads_q = n_heads_kv * qhead_per_kv_head
    cu_q_scale = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_q, token_alignment=32)
    cu_k_scale = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_k, token_alignment=128)
    assert torch.equal(cu_q_scale, torch.tensor([0, 32, 64], dtype=torch.int32))
    assert torch.equal(cu_k_scale, torch.tensor([0, 128, 256], dtype=torch.int32))

    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())
    q_token = torch.arange(total_q, dtype=torch.int64).view(total_q, 1, 1) * 1_000
    q_head = torch.arange(n_heads_q, dtype=torch.int64).view(1, n_heads_q, 1) * 10
    group = torch.arange(sf_groups, dtype=torch.int64).view(1, 1, sf_groups)
    q_scale = q_token + q_head + group

    k_token = torch.arange(total_k, dtype=torch.int64).view(total_k, 1, 1) * 1_000
    k_head = torch.arange(n_heads_kv, dtype=torch.int64).view(1, n_heads_kv, 1) * 10
    k_scale = k_token + k_head + group

    q_mkl = mxfp8_scale_utils.logical_q_scale_to_mkl_thd(
        q_scale,
        cu_q,
        cu_q_scale,
        qhead_per_kv_head,
    )
    assert q_mkl.shape == (256, sf_groups, n_heads_kv)
    assert torch.equal(q_mkl[0, :, 0], q_scale[0, 0, :])
    assert torch.equal(q_mkl[3, :, 0], q_scale[0, 3, :])
    assert torch.equal(q_mkl[4, :, 0], q_scale[1, 0, :])
    assert torch.equal(q_mkl[0, :, 1], q_scale[0, 4, :])
    assert torch.equal(q_mkl[128, :, 0], q_scale[2, 0, :])
    assert torch.equal(q_mkl[128, :, 1], q_scale[2, 4, :])
    assert torch.equal(q_mkl[8:128], torch.zeros(120, sf_groups, n_heads_kv, dtype=torch.int64))

    k_mkl = mxfp8_scale_utils.logical_k_scale_to_mkl_thd(
        k_scale,
        cu_k,
        cu_k_scale,
    )
    assert k_mkl.shape == (256, sf_groups, n_heads_kv)
    assert torch.equal(k_mkl[0, :, 0], k_scale[0, 0, :])
    assert torch.equal(k_mkl[2, :, 0], k_scale[2, 0, :])
    assert torch.equal(k_mkl[0, :, 1], k_scale[0, 1, :])
    assert torch.equal(k_mkl[128, :, 0], k_scale[3, 0, :])
    assert torch.equal(k_mkl[129, :, 1], k_scale[4, 1, :])
    assert torch.equal(k_mkl[3:128], torch.zeros(125, sf_groups, n_heads_kv, dtype=torch.int64))

    q_packed = mxfp8_scale_utils.pack_q_scale_thd(
        q_scale,
        cu_q,
        cu_q_scale,
        qhead_per_kv_head,
    )
    k_packed = mxfp8_scale_utils.pack_k_scale_thd(
        k_scale,
        cu_k,
        cu_k_scale,
    )
    q_unpacked = mxfp8_scale_utils.unpack_blockscaled_scale_mkl(
        q_packed,
        mn=256,
        sf_groups=sf_groups,
    )
    k_unpacked = mxfp8_scale_utils.unpack_blockscaled_scale_mkl(
        k_packed,
        mn=256,
        sf_groups=sf_groups,
    )
    assert torch.equal(q_unpacked, q_mkl)
    assert torch.equal(k_unpacked, k_mkl)


def test_mxfp8_scale_pack_thd_reference_fallback_accepts_capacity_output_host_only():
    """Copy exact reference results into larger outputs and clear their tails."""
    qhpkv, hkv, groups = 4, 2, 5
    cu_q = torch.tensor([0, 2, 3], dtype=torch.int64)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int64)
    cu_q_scale = torch.tensor([0, 32, 64], dtype=torch.int32)
    cu_k_scale = torch.tensor([0, 128, 256], dtype=torch.int32)
    q = torch.arange(3 * hkv * qhpkv * groups, dtype=torch.int64).reshape(3, hkv * qhpkv, groups)
    k = torch.arange(5 * hkv * groups, dtype=torch.int64).reshape(5, hkv, groups)
    q_ref = mxfp8_scale_utils.pack_q_scale_thd(q, cu_q, cu_q_scale, qhpkv)
    k_ref = mxfp8_scale_utils.pack_k_scale_thd(k, cu_k, cu_k_scale)
    q_out = torch.full((hkv, q_ref.shape[1] + 128, q_ref.shape[2]), -1, dtype=q.dtype)
    k_out = torch.full((hkv, k_ref.shape[1] + 128, k_ref.shape[2]), -1, dtype=k.dtype)

    actual_q = mxfp8_scale_utils.pack_q_scale_thd(q, cu_q, cu_q_scale, qhpkv, out=q_out)
    actual_k = mxfp8_scale_utils.pack_k_scale_thd(k, cu_k, cu_k_scale, out=k_out)

    assert actual_q is q_out
    assert actual_k is k_out
    assert torch.equal(q_out[:, : q_ref.shape[1]], q_ref)
    assert torch.equal(k_out[:, : k_ref.shape[1]], k_ref)
    assert torch.count_nonzero(q_out[:, q_ref.shape[1] :]) == 0
    assert torch.count_nonzero(k_out[:, k_ref.shape[1] :]) == 0


def test_mxfp8_scale_pack_thd_reference_fallback_preserves_singleton_prefix_host_only():
    """Keep the legacy empty-THD singleton-prefix case with capacity output."""
    hkv, qhpkv, groups = 2, 4, 5
    cu = torch.tensor([0], dtype=torch.int64)
    cu_scale = torch.tensor([0], dtype=torch.int32)
    q = torch.empty((0, hkv * qhpkv, groups), dtype=torch.int64)
    k = torch.empty((0, hkv, groups), dtype=torch.int64)
    q_out = torch.empty((hkv, 0, 8), dtype=q.dtype)
    k_out = torch.empty((hkv, 0, 8), dtype=k.dtype)

    actual_q = mxfp8_scale_utils.pack_q_scale_thd(q, cu, cu_scale, qhpkv, out=q_out)
    actual_k = mxfp8_scale_utils.pack_k_scale_thd(k, cu, cu_scale, out=k_out)

    assert actual_q is q_out
    assert actual_k is k_out


def test_mxfp8_scale_pack_thd_reference_fallback_rejects_small_capacity_host_only():
    """Reject caller-owned THD storage smaller than the reference result."""
    qhpkv, hkv, groups = 4, 2, 5
    cu = torch.tensor([0, 2, 3], dtype=torch.int64)
    cu_q_scale = torch.tensor([0, 32, 64], dtype=torch.int32)
    q = torch.arange(3 * hkv * qhpkv * groups, dtype=torch.int64).reshape(3, hkv * qhpkv, groups)
    out = torch.empty((hkv, 128, 8), dtype=q.dtype)

    with pytest.raises(ValueError, match="cannot hold packed extent"):
        mxfp8_scale_utils.pack_q_scale_thd(q, cu, cu_q_scale, qhpkv, out=out)


def test_mxfp8_scale_prefix_padding_127_128_host_only():
    """Round each THD scale-prefix span independently to alignment."""
    cu = torch.tensor([0, 127, 255], dtype=torch.int32)
    q_padded = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu, token_alignment=2)
    k_padded = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu, token_alignment=128)
    expected = torch.tensor([0, 128, 256], dtype=torch.int32)
    assert torch.equal(q_padded, expected)
    assert torch.equal(k_padded, expected)


@pytest.mark.parametrize(
    "total_tokens,num_sequences,alignment,expected",
    [
        (0, 4, 128, 0),
        (1, 4, 128, 128),
        (255, 2, 128, 384),
        (3, 2, 32, 64),
        (8192, 1, 2, 8192),
    ],
)
def test_minimal_padded_token_capacity_host_only(total_tokens, num_sequences, alignment, expected):
    """Compute a safe reusable capacity without reading device prefixes."""
    assert mxfp8_scale_utils.minimal_padded_token_capacity(total_tokens, num_sequences, alignment) == expected


def test_minimal_padded_token_capacity_rejects_tokens_without_sequences_host_only():
    """Reject a nonempty token extent with no declared sequences."""
    with pytest.raises(ValueError, match="num_sequences must be positive"):
        mxfp8_scale_utils.minimal_padded_token_capacity(1, 0, 128)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_cuda_matches_reference_and_reuses_output():
    """Match individual CUDA BSHD packers to the host reference."""
    device = torch.device("cuda")
    bs, sq, sk, hq, hkv, groups, qhpkv = 2, 131, 259, 64, 1, 5, 64
    q = torch.arange(bs * sq * hq * groups, dtype=torch.int64).to(torch.uint8).reshape(bs, sq, hq, groups)
    k = torch.arange(bs * sk * hkv * groups, dtype=torch.int64).to(torch.uint8).reshape(bs, sk, hkv, groups)
    q_ref = mxfp8_scale_utils.pack_q_scale_bshd(q, qhpkv)
    k_ref = mxfp8_scale_utils.pack_k_scale_bshd(k)
    q_cuda = q.to(device)
    k_cuda = k.to(device)
    q_out = torch.empty_like(q_ref, device=device)
    k_out = torch.empty_like(k_ref, device=device)

    assert mxfp8_scale_utils.pack_q_scale_bshd(q_cuda, qhpkv, out=q_out) is q_out
    assert mxfp8_scale_utils.pack_k_scale_bshd(k_cuda, out=k_out) is k_out
    torch.cuda.synchronize()

    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_qk_bshd_cuda_matches_reference_and_reuses_outputs():
    """Match the fused CUDA BSHD packer and preserve output identities."""
    device = torch.device("cuda")
    bs, sq, sk, hq, hkv, groups, qhpkv = 2, 131, 259, 64, 1, 5, 64
    q_cpu = torch.arange(bs * sq * hq * groups, dtype=torch.int64).to(torch.uint8).reshape(bs, sq, hq, groups)
    k_cpu = torch.arange(bs * sk * hkv * groups, dtype=torch.int64).to(torch.uint8).reshape(bs, sk, hkv, groups)
    q_ref = mxfp8_scale_utils.pack_q_scale_bshd(q_cpu, qhpkv)
    k_ref = mxfp8_scale_utils.pack_k_scale_bshd(k_cpu)
    q = q_cpu.to(device)
    k = k_cpu.to(device)
    q_out = torch.empty_like(q_ref, device=device)
    k_out = torch.empty_like(k_ref, device=device)

    actual_q, actual_k = mxfp8_scale_utils.pack_qk_scale_bshd(
        q,
        k,
        qhead_per_kv_head=qhpkv,
        q_out=q_out,
        k_out=k_out,
    )
    torch.cuda.synchronize()

    assert actual_q is q_out
    assert actual_k is k_out
    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        pytest.param(0, 0, id="both-sequences-empty"),
        pytest.param(0, 5, id="q-sequence-empty"),
        pytest.param(3, 0, id="k-sequence-empty"),
    ],
)
@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_qk_bshd_cuda_accepts_empty_extents(seqlen_q, seqlen_k):
    """Pack a live Q or K operand when the paired CUDA operand has null storage."""
    qhpkv, hkv, groups = 64, 1, 5
    q_cpu = torch.arange(seqlen_q * qhpkv * groups, dtype=torch.int64).to(torch.uint8).reshape(1, seqlen_q, qhpkv, groups)
    k_cpu = torch.arange(seqlen_k * hkv * groups, dtype=torch.int64).to(torch.uint8).reshape(1, seqlen_k, hkv, groups)
    q_ref = mxfp8_scale_utils.pack_q_scale_bshd(q_cpu, qhpkv)
    k_ref = mxfp8_scale_utils.pack_k_scale_bshd(k_cpu)
    q = q_cpu.cuda()
    k = k_cpu.cuda()
    q_out = torch.empty_like(q_ref, device="cuda")
    k_out = torch.empty_like(k_ref, device="cuda")

    actual_q, actual_k = mxfp8_scale_utils.pack_qk_scale_bshd(q, k, qhpkv, q_out=q_out, k_out=k_out)
    torch.cuda.synchronize()

    assert actual_q is q_out
    assert actual_k is k_out
    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_qk_bshd_cuda_is_async_graph_capturable_and_honors_stream():
    """Keep the fused fast path asynchronous, stream-safe, and capturable."""
    cuda = pytest.importorskip("cuda.bindings.driver")
    q = torch.arange(129 * 64 * 5, dtype=torch.int64, device="cuda").to(torch.uint8).reshape(1, 129, 64, 5)
    k = torch.arange(259 * 5, dtype=torch.int64, device="cuda").to(torch.uint8).reshape(1, 259, 1, 5)
    q_ref = mxfp8_scale_utils.pack_q_scale_bshd(q.cpu(), 64)
    k_ref = mxfp8_scale_utils.pack_k_scale_bshd(k.cpu())
    q_out = torch.empty_like(q_ref, device="cuda")
    k_out = torch.empty_like(k_ref, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    torch.cuda.set_sync_debug_mode("error")
    try:
        mxfp8_scale_utils.pack_qk_scale_bshd(
            q,
            k,
            64,
            q_out=q_out,
            k_out=k_out,
            stream=cuda.CUstream(side.cuda_stream),
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 64, q_out=q_out, k_out=k_out)
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(q_out.cpu(), q_ref)
        assert torch.equal(k_out.cpu(), k_ref)
    finally:
        graph.reset()


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_cuda_thd_caller_owned_output_matches_reference():
    """Match direct CUDA THD Q/K packing to the reference implementation."""
    device = torch.device("cuda")
    qhpkv, hkv, groups = 4, 2, 4
    cu_q_cpu = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k_cpu = torch.tensor([0, 3, 5], dtype=torch.int32)
    cu_q_scale_cpu = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_q_cpu, 32)
    cu_k_scale_cpu = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_k_cpu, 128)
    q_cpu = torch.arange(3 * (hkv * qhpkv) * groups, dtype=torch.int64).to(torch.uint8).reshape(3, hkv * qhpkv, groups)
    k_cpu = torch.arange(5 * hkv * groups, dtype=torch.int64).to(torch.uint8).reshape(5, hkv, groups)
    q_ref = mxfp8_scale_utils.pack_q_scale_thd(q_cpu, cu_q_cpu, cu_q_scale_cpu, qhpkv)
    k_ref = mxfp8_scale_utils.pack_k_scale_thd(k_cpu, cu_k_cpu, cu_k_scale_cpu)

    q = q_cpu.to(device)
    k = k_cpu.to(device)
    cu_q = cu_q_cpu.to(device)
    cu_k = cu_k_cpu.to(device)
    cu_q_scale = cu_q_scale_cpu.to(device)
    cu_k_scale = cu_k_scale_cpu.to(device)
    q_out = torch.empty_like(q_ref, device=device)
    k_out = torch.empty_like(k_ref, device=device)
    mxfp8_scale_utils.pack_q_scale_thd(q, cu_q, cu_q_scale, qhpkv, out=q_out)
    mxfp8_scale_utils.pack_k_scale_thd(k, cu_k, cu_k_scale, out=k_out)
    torch.cuda.synchronize()

    assert torch.equal(q_out.cpu(), q_ref)
    assert torch.equal(k_out.cpu(), k_ref)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_cuda_thd_strided_fallback_accepts_capacity_output():
    """Support larger caller outputs on the CUDA strided-input fallback."""
    qhpkv, hkv, groups = 4, 2, 5
    cu_q_cpu = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_k_cpu = torch.tensor([0, 3, 5], dtype=torch.int32)
    cu_q_scale_cpu = torch.tensor([0, 32, 64], dtype=torch.int32)
    cu_k_scale_cpu = torch.tensor([0, 128, 256], dtype=torch.int32)
    q_storage = torch.arange(3 * hkv * qhpkv * groups * 2, dtype=torch.int64).to(torch.uint8).reshape(3, hkv * qhpkv, groups * 2)
    k_storage = torch.arange(5 * hkv * groups * 2, dtype=torch.int64).to(torch.uint8).reshape(5, hkv, groups * 2)
    q_cpu = q_storage[..., ::2]
    k_cpu = k_storage[..., ::2]
    q_ref = mxfp8_scale_utils.pack_q_scale_thd(q_cpu.contiguous(), cu_q_cpu, cu_q_scale_cpu, qhpkv)
    k_ref = mxfp8_scale_utils.pack_k_scale_thd(k_cpu.contiguous(), cu_k_cpu, cu_k_scale_cpu)
    q = q_storage.cuda()[..., ::2]
    k = k_storage.cuda()[..., ::2]
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    q_out = torch.full((hkv, q_ref.shape[1] + 128, q_ref.shape[2]), 0xA5, dtype=q.dtype, device=q.device)
    k_out = torch.full((hkv, k_ref.shape[1] + 128, k_ref.shape[2]), 0xA5, dtype=k.dtype, device=k.device)

    mxfp8_scale_utils.pack_q_scale_thd(q, cu_q_cpu.cuda(), cu_q_scale_cpu.cuda(), qhpkv, out=q_out)
    mxfp8_scale_utils.pack_k_scale_thd(k, cu_k_cpu.cuda(), cu_k_scale_cpu.cuda(), out=k_out)
    torch.cuda.synchronize()

    assert torch.equal(q_out[:, : q_ref.shape[1]].cpu(), q_ref)
    assert torch.equal(k_out[:, : k_ref.shape[1]].cpu(), k_ref)
    assert torch.count_nonzero(q_out[:, q_ref.shape[1] :]) == 0
    assert torch.count_nonzero(k_out[:, k_ref.shape[1] :]) == 0


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_cuda_hot_path_does_not_synchronize():
    """Avoid host synchronization on the direct CUDA packing path."""
    device = torch.device("cuda")
    q = torch.zeros((1, 128, 64, 4), dtype=torch.uint8, device=device)
    out = torch.empty((1, 8192, 4), dtype=torch.uint8, device=device)
    mxfp8_scale_utils.pack_q_scale_bshd(q, out=out)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        mxfp8_scale_utils.pack_q_scale_bshd(q, out=out)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_cuda_caller_owned_path_is_graph_capturable():
    """Capture and replay direct caller-owned BSHD packing."""
    q = torch.arange(129 * 64 * 5, dtype=torch.int64, device="cuda").to(torch.uint8).reshape(1, 129, 64, 5)
    reference = mxfp8_scale_utils.pack_q_scale_bshd(q.cpu(), 64)
    out = torch.empty(reference.shape, dtype=torch.uint8, device="cuda")
    mxfp8_scale_utils.pack_q_scale_bshd(q, 64, out=out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            mxfp8_scale_utils.pack_q_scale_bshd(q, 64, out=out)
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(out.cpu(), reference)
    finally:
        graph.reset()


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_cuda_honors_explicit_stream():
    """Enqueue direct BSHD packing on the requested driver stream."""
    cuda = pytest.importorskip("cuda.bindings.driver")
    q = torch.arange(129 * 64 * 5, dtype=torch.int64, device="cuda").to(torch.uint8).reshape(1, 129, 64, 5)
    reference = mxfp8_scale_utils.pack_q_scale_bshd(q.cpu(), 64)
    out = torch.empty(reference.shape, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    mxfp8_scale_utils.pack_q_scale_bshd(q, 64, out=out, stream=cuda.CUstream(side.cuda_stream))
    torch.cuda.current_stream().wait_stream(side)

    assert torch.equal(out.cpu(), reference)


@pytest.mark.usefixtures("cuda_scale_packer")
def test_mxfp8_scale_pack_cuda_reference_fallback_honors_explicit_stream():
    """Enqueue the strided-input reference fallback on the requested stream."""
    cuda = pytest.importorskip("cuda.bindings.driver")
    storage = torch.arange(129 * 64 * 5 * 2, dtype=torch.int64, device="cuda").to(torch.uint8).reshape(1, 129, 64, 10)
    q = storage[..., ::2]
    assert not q.is_contiguous()
    reference = mxfp8_scale_utils.pack_q_scale_bshd(q.cpu().contiguous(), 64)
    out = torch.empty(reference.shape, dtype=torch.uint8, device="cuda")
    out.fill_(0xA5)
    side = torch.cuda.Stream()
    verifier = torch.cuda.Stream()
    observed = torch.empty_like(reference, pin_memory=True)
    mxfp8_scale_utils.pack_q_scale_bshd(q, 64, out=torch.empty_like(out))
    torch.cuda.synchronize()

    # Delay the ambient stream. A fallback that ignores the explicit stream
    # leaves ``out`` poisoned when the verifier consumes it after ``side``.
    torch.cuda._sleep(50_000_000)
    mxfp8_scale_utils.pack_q_scale_bshd(q, 64, out=out, stream=cuda.CUstream(side.cuda_stream))
    verifier.wait_stream(side)
    with torch.cuda.stream(verifier):
        observed.copy_(out, non_blocking=True)
    verifier.synchronize()

    assert torch.equal(observed, reference)
    torch.cuda.synchronize()


def test_mxfp8_scale_pack_thd_custom_256_padding_host_only():
    """Preserve caller-declared THD padding larger than the minimum."""
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    cu_scale = torch.tensor([0, 256, 512], dtype=torch.int32)
    q_scale = torch.arange(2 * 64 * 4, dtype=torch.int64).reshape(2, 64, 4)
    k_scale = torch.arange(2 * 4, dtype=torch.int64).reshape(2, 1, 4)

    q_mkl = mxfp8_scale_utils.logical_q_scale_to_mkl_thd(q_scale, cu, cu_scale, 64)
    k_mkl = mxfp8_scale_utils.logical_k_scale_to_mkl_thd(k_scale, cu, cu_scale)
    assert q_mkl.shape == (512 * 64, 4, 1)
    assert k_mkl.shape == (512, 4, 1)
    q_packed = mxfp8_scale_utils.pack_q_scale_thd(q_scale, cu, cu_scale, 64)
    k_packed = mxfp8_scale_utils.pack_k_scale_thd(k_scale, cu, cu_scale)
    assert q_packed.shape == (1, 512 * 64, 4)
    assert k_packed.shape == (1, 512, 4)
    assert torch.equal(q_mkl[256 * 64, :, 0], q_scale[1, 0])
    assert torch.equal(k_mkl[256, :, 0], k_scale[1, 0])


def test_blockscaled_umma_descriptor_bitfields_host_only():
    """Encode every blockscaled UMMA descriptor field in its ABI bit."""
    cutlass = pytest.importorskip("cutlass")
    mma_desc = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mma_desc")

    desc = mma_desc.make_blockscaled_instr_desc(
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
        M=256,
        N=64,
        a_major=mma_desc.Major.MN,
        b_major=mma_desc.Major.K,
        a_sf_id=3,
        b_sf_id=2,
        scale_format=1,
        is_sparse=True,
    )

    assert (desc >> 2) & 0x1 == 1  # sparse
    assert (desc >> 4) & 0x3 == 2  # B scale-factor ID
    assert (desc >> 7) & 0x7 == mma_desc.MXF8F6F4Format.E4M3
    assert (desc >> 10) & 0x7 == mma_desc.MXF8F6F4Format.E5M2
    assert (desc >> 15) & 0x1 == mma_desc.Major.MN
    assert (desc >> 16) & 0x1 == mma_desc.Major.K
    assert (desc >> 17) & 0x3F == 64 // 8
    assert (desc >> 23) & 0x1 == 1  # E8M0 scale format
    assert (desc >> 24) & 0x1F == 256 // 16
    assert (desc >> 29) & 0x3 == 3  # A scale-factor ID

    reserved_mask = 0x3 | (1 << 3) | (1 << 6) | (0x3 << 13) | (1 << 31)
    assert desc & reserved_mask == 0


def test_blockscaled_mma_op_adapter_maps_scale_and_major_modes_host_only():
    """Map CuTe MMA operand modes and scale IDs to the descriptor ABI."""
    from types import SimpleNamespace

    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    mma_desc = pytest.importorskip("cudnn.deepseek_sparse_attention.utils.sm100.mma_desc")

    op = SimpleNamespace(
        a_dtype=cutlass.Float8E4M3FN,
        b_dtype=cutlass.Float8E5M2,
        shape_mnk=(128, 256, 64),
        a_major_mode=cute.nvgpu.tcgen05.mma.OperandMajorMode.K,
        b_major_mode=cute.nvgpu.tcgen05.mma.OperandMajorMode.MN,
    )
    desc = mma_desc.blockscaled_mma_op_to_idesc(op, sf_id=2)

    assert (desc >> 4) & 0x3 == 2
    assert (desc >> 15) & 0x1 == mma_desc.Major.K
    assert (desc >> 16) & 0x1 == mma_desc.Major.MN
    assert (desc >> 29) & 0x3 == 2
    assert desc == mma_desc.make_blockscaled_instr_desc(
        op.a_dtype,
        op.b_dtype,
        M=128,
        N=256,
        a_major=mma_desc.Major.K,
        b_major=mma_desc.Major.MN,
        a_sf_id=2,
        b_sf_id=2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("unavailable", ["old_dsl", "no_dsl", "unsupported_arch"])
def test_mxfp8_scale_pack_cuda_preserves_reference_without_fast_path(monkeypatch, unavailable):
    """Existing CUDA callers still pack correctly when the optimized route is unavailable."""
    from cudnn.frost import buffers
    from cudnn.deepseek_sparse_attention.utils import runtime

    q_cpu = torch.arange(3 * 8 * 5, dtype=torch.int64).to(torch.uint8).reshape(1, 3, 8, 5)
    k_cpu = torch.arange(3 * 2 * 5, dtype=torch.int64).to(torch.uint8).reshape(1, 3, 2, 5)
    q_ref, k_ref = mxfp8_scale_utils.pack_qk_scale_bshd(q_cpu, k_cpu, 4)
    cu_cpu = torch.tensor([0, 0, 2, 3], dtype=torch.int32)
    pq_cpu = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_cpu, 32)
    pk_cpu = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu_cpu, 128)
    qt_ref = mxfp8_scale_utils.pack_q_scale_thd(q_cpu[0], cu_cpu, pq_cpu, 4)
    kt_ref = mxfp8_scale_utils.pack_k_scale_thd(k_cpu[0], cu_cpu, pk_cpu)
    q, k = q_cpu.cuda(), k_cpu.cuda()
    cu, pq, pk = cu_cpu.cuda(), pq_cpu.cuda(), pk_cpu.cuda()
    q_out, k_out = torch.empty_like(q_ref, device="cuda"), torch.empty_like(k_ref, device="cuda")
    qt_out = torch.full((2, qt_ref.shape[1] + 128, 8), 0xA5, dtype=torch.uint8, device="cuda")
    kt_out = torch.full((2, kt_ref.shape[1] + 128, 8), 0xA5, dtype=torch.uint8, device="cuda")

    state = (False, None) if unavailable == "no_dsl" else (True, ("nvidia-cutlass-dsl", "4.6.2" if unavailable == "old_dsl" else "4.7.1"))
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: state)
    if unavailable == "unsupported_arch":
        monkeypatch.setattr(runtime, "device_capability", lambda device: (8, 0))

    def unexpected_fast_path():
        raise AssertionError("unsupported environment entered the CUDA packer")

    monkeypatch.setattr(mxfp8_scale_utils, "_cuda_pack_launchers", unexpected_fast_path)
    supports_device = getattr(mxfp8_scale_utils, "_cuda_packer_supports_device", None)
    if supports_device is not None:
        supports_device.cache_clear()
    try:
        actual_q, actual_k = mxfp8_scale_utils.pack_qk_scale_bshd(q, k, 4, q_out=q_out, k_out=k_out)
        assert actual_q is q_out and actual_k is k_out
        assert torch.equal(actual_q.cpu(), q_ref)
        assert torch.equal(actual_k.cpu(), k_ref)
        assert mxfp8_scale_utils.pack_q_scale_thd(q[0], cu, pq, 4, out=qt_out) is qt_out
        assert mxfp8_scale_utils.pack_k_scale_thd(k[0], cu, pk, out=kt_out) is kt_out
        assert torch.equal(qt_out[:, : qt_ref.shape[1]].cpu(), qt_ref)
        assert torch.equal(kt_out[:, : kt_ref.shape[1]].cpu(), kt_ref)
        assert torch.count_nonzero(qt_out[:, qt_ref.shape[1] :]) == 0
        assert torch.count_nonzero(kt_out[:, kt_ref.shape[1] :]) == 0
    finally:
        if supports_device is not None:
            supports_device.cache_clear()
