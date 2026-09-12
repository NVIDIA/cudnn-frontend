# SPDX-FileCopyrightText: Copyright (c) 2026 tiffany940107. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.bsa.bsa_reference import attention_reference, block_sparse_mask
from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.metadata import lower_sm120_blk128_metadata

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _import_bsa():
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")

        return BSA
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")


def _require_sm120() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("logical blk128 forward is specific to SM120")


@pytest.mark.L0
def test_sm120_blk128_metadata_lowers_odd_count_and_q_tail():
    q2k = torch.tensor([[[[0, 2, 3], [3, 1, 0]]]], dtype=torch.int32)

    lowered = lower_sm120_blk128_metadata(
        q2k,
        block_sparse_num=3,
        block_sizes=None,
        q2k_block_nums=None,
        seqlen_q=193,
    )

    expected_first_row = torch.tensor([0, 1, 4, 5, 6, 7], dtype=torch.int32)
    expected_last_row = torch.tensor([6, 7, 2, 3, 0, 1], dtype=torch.int32)
    assert lowered.q2k_block_index.shape == (1, 1, 4, 6)
    torch.testing.assert_close(lowered.q2k_block_index[0, 0, 0], expected_first_row)
    torch.testing.assert_close(lowered.q2k_block_index[0, 0, 1], expected_first_row)
    torch.testing.assert_close(lowered.q2k_block_index[0, 0, 2], expected_last_row)
    torch.testing.assert_close(lowered.q2k_block_index[0, 0, 3], expected_last_row)
    assert lowered.block_sparse_num == 6
    assert lowered.block_sizes is None
    assert lowered.q2k_block_nums is None


@pytest.mark.L0
@pytest.mark.parametrize(
    "block_sizes",
    (
        torch.tensor([128, 96, 64, 1], dtype=torch.int32),
        torch.tensor([[128, 96, 64, 1], [1, 64, 96, 128]], dtype=torch.int32),
        torch.tensor(
            [
                [[128, 96, 64, 1], [1, 64, 96, 128]],
                [[65, 80, 127, 128], [128, 127, 80, 65]],
            ],
            dtype=torch.int32,
        ),
    ),
)
def test_sm120_blk128_metadata_lowers_block_sizes(block_sizes):
    q2k = torch.zeros((2, 2, 1, 1), dtype=torch.int32)

    lowered = lower_sm120_blk128_metadata(
        q2k,
        block_sparse_num=1,
        block_sizes=block_sizes,
        q2k_block_nums=None,
        seqlen_q=128,
    )

    expected_first = block_sizes.clamp(min=0, max=64)
    expected_second = (block_sizes - 64).clamp(min=0, max=64)
    expected = torch.stack((expected_first, expected_second), dim=-1).flatten(-2)
    assert lowered.block_sizes.shape == (*block_sizes.shape[:-1], 2 * block_sizes.shape[-1])
    torch.testing.assert_close(lowered.block_sizes, expected)


@pytest.mark.L0
def test_sm120_blk128_metadata_lowers_variable_counts():
    q2k = torch.tensor(
        [[[[3, 2, 1], [0, 1, 3]], [[1, 3, 2], [2, 0, 1]]]],
        dtype=torch.int32,
    )
    block_nums = torch.tensor([[[1, 3], [2, 1]]], dtype=torch.int32)

    lowered = lower_sm120_blk128_metadata(
        q2k,
        block_sparse_num=99,
        block_sizes=None,
        q2k_block_nums=block_nums,
        seqlen_q=191,
    )

    expected_nums = torch.tensor([[[2, 2, 6], [4, 4, 2]]], dtype=torch.int32)
    assert lowered.q2k_block_index.shape == (1, 2, 3, 6)
    assert lowered.block_sparse_num == 0
    torch.testing.assert_close(lowered.q2k_block_nums, expected_nums)


@pytest.mark.L0
@torch_fork_set_rng(seed=120128)
def test_bsa_attention_forward_sm120_blk128_fixed_odd_count():
    _require_sm120()
    BSA = _import_bsa()
    block_size = 128
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 193, 512, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = torch.tensor(
        [[[[0, 1, 3], [3, 2, 0]], [[2, 1, 0], [1, 3, 2]]]],
        device="cuda",
        dtype=torch.int32,
    )
    block_sizes = torch.tensor([128, 96, 64, 1], device="cuda", dtype=torch.int32)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=3,
        block_sizes=block_sizes,
        sparse_block_size=block_size,
    )
    mask = block_sparse_mask(q2k, 3, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)

    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    result_bshd = BSA.block_sparse_attention_forward(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        q2k,
        block_sparse_num=3,
        block_sizes=block_sizes,
        sparse_block_size=block_size,
        layout="bshd",
    )
    torch.testing.assert_close(result_bshd["o_tensor"].transpose(1, 2), result["o_tensor"], atol=0, rtol=0)
    torch.testing.assert_close(result_bshd["lse_tensor"], result["lse_tensor"], atol=0, rtol=0)


@pytest.mark.L0
@torch_fork_set_rng(seed=120129)
def test_bsa_attention_forward_sm120_blk128_variable_count_gqa():
    _require_sm120()
    BSA = _import_bsa()
    block_size = 128
    batch, q_heads, kv_heads, seqlen_q, seqlen_k, dim = 1, 4, 2, 191, 512, 128
    q = torch.randn((batch, q_heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, kv_heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = torch.tensor(
        [
            [
                [[3, 1, 0], [0, 2, 3]],
                [[2, 0, 3], [3, 1, 2]],
                [[1, 2, 3], [2, 0, 1]],
                [[0, 3, 2], [1, 2, 0]],
            ]
        ],
        device="cuda",
        dtype=torch.int32,
    )
    block_nums = torch.tensor([[[1, 3], [2, 1], [3, 2], [1, 2]]], device="cuda", dtype=torch.int32)
    reference_block_sizes = torch.full((seqlen_k // block_size,), block_size, device="cuda", dtype=torch.int32)

    result = BSA.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sizes=None,
        q2k_block_nums=block_nums,
        sparse_block_size=block_size,
        pack_gqa=False,
    )
    mask = block_sparse_mask(q2k, 0, reference_block_sizes, seqlen_q, seqlen_k, block_size, block_nums)
    o_ref, lse_ref = attention_reference(q, k, v, mask)

    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
def test_bsa_attention_forward_sm120_blk128_rejects_unsupported_inputs():
    _require_sm120()
    BSA = _import_bsa()
    q = torch.empty((1, 1, 128, 128), dtype=torch.float16, device="cuda")
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    q2k = torch.zeros((1, 1, 1, 1), dtype=torch.int32, device="cuda")

    with pytest.raises(NotImplementedError, match="requires BF16"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, sparse_block_size=128)

    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    with pytest.raises(NotImplementedError, match="pack_gqa"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, sparse_block_size=128, pack_gqa=True)

    partial_k = torch.empty((1, 1, 129, 128), dtype=torch.bfloat16, device="cuda")
    partial_q2k = torch.zeros((1, 1, 1, 1), dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="seqlen_k.*multiple of 128"):
        BSA.block_sparse_attention_forward(q, partial_k, partial_k, partial_q2k, sparse_block_size=128)
