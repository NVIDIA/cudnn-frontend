# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the SM100 MXFP8 scale-layout contract."""

import pytest
import torch

from cudnn.deepseek_sparse_attention.utils.sm100 import mxfp8_scale_utils

pytestmark = pytest.mark.L0


def test_mxfp8_scale_pack_roundtrip_host_only():
    logical = torch.arange(130 * 5 * 3, dtype=torch.int64).reshape(130, 5, 3)

    packed = mxfp8_scale_utils.pack_blockscaled_scale_mkl(logical)

    assert packed.shape == (3, 256, 8)
    assert packed.is_contiguous()
    unpacked = mxfp8_scale_utils.unpack_blockscaled_scale_mkl(packed, mn=130, sf_groups=5)
    assert torch.equal(unpacked, logical)


def test_mxfp8_scale_pack_uses_blackwell_datapath_order_host_only():
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


def test_mxfp8_scale_pack_order_thd_with_padding_host_only():
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


def test_mxfp8_scale_prefix_padding_127_128_host_only():
    cu = torch.tensor([0, 127, 255], dtype=torch.int32)
    q_padded = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu, token_alignment=2)
    k_padded = mxfp8_scale_utils.make_scale_cu_seqlens_padded(cu, token_alignment=128)
    expected = torch.tensor([0, 128, 256], dtype=torch.int32)
    assert torch.equal(q_padded, expected)
    assert torch.equal(k_padded, expected)


def test_mxfp8_scale_pack_thd_custom_256_padding_host_only():
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
