# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPage forward over sparse_block_size=64 metadata (kpage=True / kpage=KPageLayout) against the base blk64 kernel and the fp32 reference.

The contract is the BSA one: ``block_sizes`` masks keys, every query row is valid.  The path is forward only.
"""

import importlib

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

BLOCK = 64


def _import_bsa():
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("kpage is specific to SM100/SM103")
    try:
        from cudnn import BSA

        importlib.import_module("cudnn.block_sparse_attention._interface")
    except (ImportError, OSError) as error:
        pytest.skip(f"block sparse attention optional dependencies are unavailable: {error}")
    from cudnn.block_sparse_attention._kpage_api import KPAGE_MIN_CUTEDSL
    from cudnn.block_sparse_attention._interface import _cutlass_dsl_version

    if _cutlass_dsl_version() < KPAGE_MIN_CUTEDSL:
        pytest.skip(f"kpage requires nvidia-cutlass-dsl>={'.'.join(map(str, KPAGE_MIN_CUTEDSL))}")
    return BSA


def _valid_rows(block_sizes: torch.Tensor) -> torch.Tensor:
    return (torch.arange(BLOCK, device=block_sizes.device).view(1, BLOCK) < block_sizes.view(-1, 1)).reshape(-1)


def _inputs(heads: int, num_q_blocks: int, block_sizes: torch.Tensor):
    """BF16 q/k/v; masked key rows hold a large finite value (any finite value is allowed)."""
    seqlen_q, seqlen_k = num_q_blocks * BLOCK, block_sizes.numel() * BLOCK
    q = torch.randn((1, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    padding = ~_valid_rows(block_sizes)
    k[0, :, padding] = v[0, :, padding] = 1000.0
    return q, k, v


def _random_metadata(heads: int, num_q_blocks: int, num_kv_blocks: int, kmax: int):
    """Unique random KV block lists with variable counts."""
    kmax = min(kmax, num_kv_blocks)
    q2k = (
        torch.stack([torch.randperm(num_kv_blocks, device="cuda")[:kmax] for _ in range(heads * num_q_blocks)])
        .view(1, heads, num_q_blocks, kmax)
        .to(torch.int32)
    )
    nums = torch.randint(1, kmax + 1, (1, heads, num_q_blocks), device="cuda", dtype=torch.int32)
    return q2k, nums


def _reference(q, k, v, q2k, nums, bs):
    mask = block_sparse_mask(q2k, 0, bs, q.shape[2], k.shape[2], BLOCK, q2k_block_nums=nums)
    return attention_reference(q, k, v, mask)


def _run(BSA, q, k, v, q2k, nums, bs, **kwargs):
    return BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, sparse_block_size=BLOCK, **kwargs)


@pytest.mark.L0
@torch_fork_set_rng(seed=21)
@pytest.mark.parametrize(
    "num_q_blocks, kv_block_sizes",
    [
        (6, [64, 17, 64, 1, 40, 64]),  # partial blocks, short and long tails
        (3, [5, 64, 9, 16, 48, 64, 64, 63]),  # more KV blocks than Q blocks, several tiles of partial pages
        (8, [64, 64, 64]),  # no padding: same work as the base kernel
        (130, [64, 33, 64, 64, 7, 64, 64, 64, 50, 64]),  # long walks: many N256 iterations per Q block
    ],
)
def test_kpage_blk64_matches_base_kernel_and_reference(num_q_blocks, kv_block_sizes):
    BSA = _import_bsa()
    heads = 2
    bs = torch.tensor(kv_block_sizes, dtype=torch.int32, device="cuda")
    q, k, v = _inputs(heads, num_q_blocks, bs)
    q2k, nums = _random_metadata(heads, num_q_blocks, bs.numel(), kmax=max(3, bs.numel() - 1))
    o_ref, lse_ref = _reference(q, k, v, q2k, nums, bs)

    paged = _run(BSA, q, k, v, q2k, nums, bs, kpage=True)
    base = _run(BSA, q, k, v, q2k, nums, bs)
    torch.testing.assert_close(paged["o_tensor"].float(), base["o_tensor"].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(paged["lse_tensor"], base["lse_tensor"], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(paged["o_tensor"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(paged["lse_tensor"], lse_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=22)
def test_kpage_blk64_fixed_count_matches_base_kernel():
    BSA = _import_bsa()
    heads, num_blocks = 2, 6
    bs = torch.tensor([64, 40, 64, 7, 64, 64], dtype=torch.int32, device="cuda")
    q, k, v = _inputs(heads, num_blocks, bs)
    q2k, _ = _random_metadata(heads, num_blocks, num_blocks, kmax=3)
    paged = BSA.block_sparse_attention_forward(q, k, v, q2k, 3, bs, sparse_block_size=BLOCK, kpage=True, softmax_scale=0.2)
    base = BSA.block_sparse_attention_forward(q, k, v, q2k, 3, bs, sparse_block_size=BLOCK, softmax_scale=0.2)
    torch.testing.assert_close(paged["o_tensor"].float(), base["o_tensor"].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(paged["lse_tensor"], base["lse_tensor"], atol=1e-3, rtol=1e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=23)
def test_kpage_blk64_layout_reuse():
    BSA = _import_bsa()
    heads, num_blocks = 3, 8
    bs = torch.tensor([64, 40, 64, 7, 25, 64, 64, 50], dtype=torch.int32, device="cuda")
    q2k, nums = _random_metadata(heads, num_blocks, num_blocks, kmax=4)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, sparse_block_size=BLOCK)
    assert isinstance(layout, BSA.KPageLayout)
    assert layout.sparse_block_size == BLOCK
    for _ in range(2):  # same layout, new activations
        q, k, v = _inputs(heads, num_blocks, bs)
        o_ref, _ = _reference(q, k, v, q2k, nums, bs)
        reused = _run(BSA, q, k, v, q2k, nums, bs, kpage=layout)
        built = _run(BSA, q, k, v, q2k, nums, bs, kpage=True)
        torch.testing.assert_close(reused["o_tensor"].float(), o_ref, atol=2e-2, rtol=2e-2)
        assert torch.equal(reused["o_tensor"], built["o_tensor"])
        assert torch.equal(reused["lse_tensor"], built["lse_tensor"])


@pytest.mark.L0
@torch_fork_set_rng(seed=24)
def test_kpage_blk64_bshd_layout_matches_bhsd():
    BSA = _import_bsa()
    heads, num_blocks = 2, 6
    bs = torch.tensor([64, 40, 64, 7, 64, 50], dtype=torch.int32, device="cuda")
    q, k, v = _inputs(heads, num_blocks, bs)
    q2k, nums = _random_metadata(heads, num_blocks, num_blocks, kmax=3)
    bhsd = _run(BSA, q, k, v, q2k, nums, bs, kpage=True)
    bshd = _run(BSA, q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), q2k, nums, bs, kpage=True, layout="bshd")
    torch.testing.assert_close(bshd["o_tensor"].transpose(1, 2).float(), bhsd["o_tensor"].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(bshd["lse_tensor"], bhsd["lse_tensor"], atol=1e-3, rtol=1e-3)


@pytest.mark.L0
def test_kpage_blk64_rejects_unsupported_configurations():
    BSA = _import_bsa()
    heads, num_blocks = 2, 4
    seqlen = num_blocks * BLOCK
    q = torch.randn((1, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q2k = torch.tensor([0, 2], dtype=torch.int32, device="cuda").view(1, 1, 1, 2).expand(1, heads, num_blocks, 2).contiguous()
    bs = torch.full((num_blocks,), BLOCK, dtype=torch.int32, device="cuda")

    # forward only
    o, lse = _run(BSA, q, k, v, q2k, None, bs, kpage=True).values()
    with pytest.raises(NotImplementedError, match="forward only"):
        BSA.block_sparse_attention_backward(torch.randn_like(q), q, k, v, o, lse, q2k, 2, bs, sparse_block_size=BLOCK, kpage=True)
    with pytest.raises(NotImplementedError, match="forward only"):
        BSA.kpage_prepare_layout(q2k, 2, bs, sparse_block_size=BLOCK, with_backward=True)
    with pytest.raises(NotImplementedError, match="forward only"):
        BSA.kpage_prepare_layout(q2k, 2, bs, q_block_sizes=bs, sparse_block_size=BLOCK)
    # MHA only
    q_gqa = torch.randn((1, 2 * heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    q2k_gqa = q2k.repeat_interleave(2, dim=1).contiguous()
    with pytest.raises(NotImplementedError, match="MHA"):
        BSA.block_sparse_attention_forward(q_gqa, k, v, q2k_gqa, 2, bs, sparse_block_size=BLOCK, kpage=True)
    # BF16 head_dim 128 (the base blk64 contract)
    with pytest.raises(NotImplementedError, match="BF16"):
        BSA.block_sparse_attention_forward(q.half(), k.half(), v.half(), q2k, 2, bs, sparse_block_size=BLOCK, kpage=True)
    # seqlen_q and seqlen_k must be multiples of 64
    q_tail = q[:, :, : seqlen - 16].contiguous()
    with pytest.raises(NotImplementedError, match="multiples of 64"):
        BSA.block_sparse_attention_forward(q_tail, k, v, q2k, 2, bs, sparse_block_size=BLOCK, kpage=True)
    # metadata contract, with the blk64 block size
    with pytest.raises(ValueError, match="block_sizes"):
        BSA.kpage_prepare_layout(q2k, 2, torch.full((num_blocks,), 100, dtype=torch.int32, device="cuda"), sparse_block_size=BLOCK)
    # a layout is bound to its sparse_block_size
    q2k_128 = torch.tensor([0, 1], dtype=torch.int32, device="cuda").view(1, 1, 1, 2).expand(1, heads, 2, 2).contiguous()
    layout_128 = BSA.kpage_prepare_layout(q2k_128, 2, torch.full((2,), 128, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="sparse_block_size"):
        _run(BSA, q, k, v, q2k, None, bs, kpage=layout_128)
