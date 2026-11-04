# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16/FP16 block sparse attention forward against the torch reference.

Tests without an architecture prefix run on every GPU the forward supports (SM90, SM100/SM103/SM110, SM120), at both
sparse block sizes where parametrized.  ``test_sm<arch>_blk<size>_*`` tests cover one kernel's scheduling, pipelining
or tail handling and skip on other GPUs.  Argument checks and dispatch policies are in ``test_BSA_api.py``.
"""

import importlib
import math

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask
from block_sparse_attention.cutedsl.bsa_utils import (
    blk128_case,
    blk128_reference,
    check_blk128,
    cuda_graph,
    import_bsa,
    make_fixed_metadata,
    make_variable_metadata,
    require_arch,
    supported_block_size,
)

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

with_block_size = pytest.mark.parametrize("block_size", (64, 128), ids=("blk64", "blk128"))
FORWARD_FAMILIES = (9, 10, 11, 12)


def _has_kv_splits(block_size: int) -> bool:
    """kv_splits is exposed by the SM90 forward and the SM100/SM110 blk64 forward."""
    major = torch.cuda.get_device_capability()[0]
    return major == 9 or (major in (10, 11) and block_size == 64)


# ---- every architecture


@pytest.mark.L0
@with_block_size
@torch_fork_set_rng(seed=0)
def test_fixed_blocks(block_size):
    """A fixed count over full KV blocks; with kv_splits=2 where the kernel exposes it, in the same output layout."""
    ops = import_bsa(families=FORWARD_FAMILIES)
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)

    result = ops.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, block_sizes, sparse_block_size=block_size)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)
    if not _has_kv_splits(block_size):
        return
    split = ops.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, block_sizes, sparse_block_size=block_size, kv_splits=2)
    torch.testing.assert_close(split["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(split["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)
    for name in ("o_tensor", "lse_tensor"):
        assert split[name].is_contiguous() and split[name].stride() == result[name].stride()


@pytest.mark.L0
@with_block_size
@torch_fork_set_rng(seed=1)
def test_variable_blocks_and_layout(block_size):
    """Variable counts and a partial KV block; BSHD matches BHSD exactly; an empty row gets O = 0 and LSE = -inf."""
    ops = import_bsa(families=FORWARD_FAMILIES)
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_nums, block_sizes = make_variable_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    kwargs = dict(q2k_block_nums=block_nums, sparse_block_size=block_size)

    result = ops.block_sparse_attention_forward(q, k, v, q2k, 0, block_sizes, **kwargs)
    mask = block_sparse_mask(q2k, 0, block_sizes, seqlen_q, seqlen_k, block_size, block_nums)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    result_bshd = ops.block_sparse_attention_forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), q2k, 0, block_sizes, **kwargs, layout="bshd")
    torch.testing.assert_close(result_bshd["o_tensor"].transpose(1, 2), result["o_tensor"], atol=0, rtol=0)
    torch.testing.assert_close(result_bshd["lse_tensor"], result["lse_tensor"], atol=0, rtol=0)

    empty_block_nums = block_nums.clone()
    empty_block_nums[..., 0] = 0
    empty_result = ops.block_sparse_attention_forward(
        q, k, v, q2k, 0, block_sizes, q2k_block_nums=empty_block_nums, sparse_block_size=block_size, allow_empty_block_nums=True
    )
    empty_mask = block_sparse_mask(q2k, 0, block_sizes, seqlen_q, seqlen_k, block_size, empty_block_nums)
    empty_o_ref, empty_lse_ref = attention_reference(q, k, v, empty_mask)
    torch.testing.assert_close(empty_result["o_tensor"].float(), empty_o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(empty_result["lse_tensor"], empty_lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@with_block_size
@torch_fork_set_rng(seed=19)
def test_partial_q_and_kv_tiles(block_size):
    """A one-row final query tile and a one-token final KV block (declared through block_sizes); BSHD matches BHSD."""
    ops = import_bsa(families=FORWARD_FAMILIES)
    if block_size == 64 and torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("the SM90 blk64 forward takes whole query blocks")
    batch, heads, seqlen_q, seqlen_k, dim = 1, 1, block_size + 1, 2 * block_size + 1, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = torch.tensor([0, 2], device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 1, 2, 2).contiguous()
    block_sizes = torch.tensor([block_size, block_size, 1], device="cuda", dtype=torch.int32)

    result = ops.block_sparse_attention_forward(q, k, v, q2k, 2, block_sizes, sparse_block_size=block_size)
    mask = block_sparse_mask(q2k, 2, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    assert torch.isfinite(result["o_tensor"]).all() and torch.isfinite(result["lse_tensor"]).all()
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    result_bshd = ops.block_sparse_attention_forward(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), q2k, 2, block_sizes, sparse_block_size=block_size, layout="bshd"
    )
    torch.testing.assert_close(result_bshd["o_tensor"].transpose(1, 2), result["o_tensor"], atol=0, rtol=0)
    torch.testing.assert_close(result_bshd["lse_tensor"], result["lse_tensor"], atol=0, rtol=0)


@pytest.mark.L0
@torch_fork_set_rng(seed=3)
def test_gqa_without_block_sizes():
    """GQA with block_sizes=None; on SM100/SM110 blk128 the packed-GQA metadata gives the same result."""
    ops = import_bsa()
    block_size = supported_block_size()
    batch, q_heads, kv_heads, seqlen_q, seqlen_k, dim = 1, 4, 2, 2 * block_size, 4 * block_size, 128
    q = torch.randn((batch, q_heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, kv_heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_sparse_num, full_block_sizes = make_fixed_metadata(batch, q_heads, seqlen_q, seqlen_k, block_size)

    result = ops.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, block_sizes=None, sparse_block_size=block_size, pack_gqa=False)
    mask = block_sparse_mask(q2k, block_sparse_num, full_block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)

    if torch.cuda.get_device_capability()[0] in {10, 11}:
        gqa_ratio = q_heads // kv_heads
        packed_q2k, packed_count, _ = make_fixed_metadata(batch, kv_heads, seqlen_q * gqa_ratio, seqlen_k, block_size)
        packed_result = ops.block_sparse_attention_forward(q, k, v, packed_q2k, packed_count, block_sizes=None, sparse_block_size=block_size)
        torch.testing.assert_close(packed_result["o_tensor"], result["o_tensor"], atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(packed_result["lse_tensor"], result["lse_tensor"], atol=2e-3, rtol=2e-3)


# Large-logit regressions: the kernels must center P on the running maximum through their 16-bit conversion.  An
# exponent offset by the rounding residual of max * softmax_scale (* log2(e)) overflows or underflows there (O = NaN
# or 0).  Each row names the kernel family it pins.
SM90 = dict(families=(9,), reason="the SM90 blk64 forward is under test")
SM100 = dict(families=(10, 11), reason="the SM100/SM110 forward is under test")


@pytest.mark.L0
@pytest.mark.parametrize(
    ("gate", "sparse_block_size", "dtype", "q_value", "k_value"),
    [
        pytest.param(SM90, 64, torch.float16, 10000.0, 10008.0, id="sm90-blk64-fp16-p-underflow"),
        pytest.param(SM90, 64, torch.float16, 30000.0, 30128.0, id="sm90-blk64-fp16-exp-overflow"),
        pytest.param(SM100, 128, torch.float16, 10000.0, 10008.0, id="sm100-blk128-fp16-p-underflow"),
        pytest.param(SM100, 128, torch.float16, 30000.0, 30128.0, id="sm100-blk128-fp16-exp-overflow"),
        pytest.param(SM100, 64, torch.bfloat16, 16384.0, 16896.0, id="sm100-blk64-bf16-exp-overflow"),
        pytest.param(SM100, 64, torch.bfloat16, 16384.0, 16640.0, id="sm100-blk64-bf16-exp-underflow"),
    ],
)
def test_large_equal_logits(gate, sparse_block_size, dtype, q_value, k_value):
    """Equal ~1e9 logits with V = 1 weight every key equally, so O is exactly 1 and LSE is known in closed form."""
    ops = import_bsa(**gate)
    seqlen_q, seqlen_k, dim = 128, 256, 128
    q = torch.full((1, 1, seqlen_q, dim), q_value, device="cuda", dtype=dtype)
    k = torch.full((1, 1, seqlen_k, dim), k_value, device="cuda", dtype=dtype)
    v = torch.ones_like(k)
    num_kv_blocks = seqlen_k // sparse_block_size
    q2k = torch.arange(num_kv_blocks, device="cuda", dtype=torch.int32).expand(1, 1, seqlen_q // sparse_block_size, -1).contiguous()

    result = ops.block_sparse_attention_forward(q, k, v, q2k, num_kv_blocks, sparse_block_size=sparse_block_size)
    torch.testing.assert_close(result["o_tensor"], torch.ones_like(q), atol=0, rtol=0)
    q_value, k_value = q[0, 0, 0, 0].item(), k[0, 0, 0, 0].item()  # as rounded to dtype
    lse_ref = torch.full_like(result["lse_tensor"], q_value * k_value * math.sqrt(dim) + math.log(seqlen_k), dtype=torch.float64)
    torch.testing.assert_close(result["lse_tensor"].double(), lse_ref, atol=0, rtol=2e-6)


@pytest.mark.L0
@pytest.mark.parametrize("max_order", ("rises", "falls"))
@pytest.mark.parametrize(
    ("gate", "sparse_block_size", "dtype", "q_value", "k_low", "k_high"),
    [
        pytest.param(SM90, 64, torch.float16, 10000.0, 10000.0, 10048.0, id="sm90-blk64-fp16"),
        pytest.param(SM90, 64, torch.bfloat16, 30000.0, 30000.0, 30208.0, id="sm90-blk64-bf16"),
        pytest.param(SM100, 128, torch.float16, 30000.0, 30000.0, 30128.0, id="sm100-blk128-fp16"),
        pytest.param(SM100, 64, torch.bfloat16, 16384.0, 16640.0, 16896.0, id="sm100-blk64-bf16"),
    ],
)
def test_large_logits_max_rises_and_falls(gate, sparse_block_size, dtype, q_value, k_low, k_high, max_order):
    """Half of the KV blocks score far below the other half; V is 1 there and 2 on the high blocks, so O is exactly 2.

    Lists are walked from the end (SM90 walks the whole list; SM100 blk128 alternates slots between its softmax
    warpgroups, blk64 hands out groups of four), so "rises" lists the high blocks first and "falls" lists them last.
    """
    ops = import_bsa(**gate)
    seqlen_q, dim = 128, 128
    num_kv_blocks = 4 if sparse_block_size == 128 else 16
    seqlen_k = num_kv_blocks * sparse_block_size
    num_low_keys = num_kv_blocks // 2 * sparse_block_size
    q = torch.full((1, 1, seqlen_q, dim), q_value, device="cuda", dtype=dtype)
    k = torch.full((1, 1, seqlen_k, dim), k_high, device="cuda", dtype=dtype)
    k[:, :, :num_low_keys] = k_low
    v = torch.full_like(k, 2.0)
    v[:, :, :num_low_keys] = 1.0
    low = torch.arange(num_kv_blocks // 2, device="cuda", dtype=torch.int32)
    high = low + num_kv_blocks // 2
    order = torch.cat((high, low) if max_order == "rises" else (low, high))
    q2k = order.expand(1, 1, seqlen_q // sparse_block_size, -1).contiguous()

    result = ops.block_sparse_attention_forward(q, k, v, q2k, num_kv_blocks, sparse_block_size=sparse_block_size)
    torch.testing.assert_close(result["o_tensor"], torch.full_like(q, 2.0), atol=0, rtol=0)
    num_high_keys = seqlen_k - num_low_keys
    q_value, k_high = q[0, 0, 0, 0].item(), k[0, 0, -1, 0].item()  # as rounded to dtype
    lse_ref = torch.full_like(result["lse_tensor"], q_value * k_high * math.sqrt(dim) + math.log(num_high_keys), dtype=torch.float64)
    torch.testing.assert_close(result["lse_tensor"].double(), lse_ref, atol=0, rtol=2e-6)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("gate", "sparse_block_size", "dtype", "q_value", "k_value"),
    [
        pytest.param(SM90, 64, torch.float16, 10000.0, 10008.0, id="sm90-blk64-fp16"),
        pytest.param(SM90, 64, torch.bfloat16, 10000.0, 10008.0, id="sm90-blk64-bf16"),
        pytest.param(SM100, 128, torch.float16, 20000.0, 20016.0, id="sm100-blk128-fp16"),
        pytest.param(SM100, 64, torch.bfloat16, 32768.0, 16896.0, id="sm100-blk64-bf16"),
    ],
)
@torch_fork_set_rng(seed=5)
def test_large_logits_mixed_warp(gate, sparse_block_size, dtype, q_value, k_value):
    """Even query rows see equal ~1e9 logits on every key (O = mean V); odd rows are ordinary attention. Every warp
    holds both kinds, so centering the large rows must leave the ordinary rows as accurate as before."""
    ops = import_bsa(**gate)
    seqlen_q, seqlen_k, dim = 128, 256, 128
    q = torch.zeros((1, 1, seqlen_q, dim), device="cuda", dtype=dtype)
    q[..., 0::2, : dim // 2] = q_value
    q[..., 1::2, dim // 2 :] = torch.randn((1, 1, seqlen_q // 2, dim // 2), device="cuda", dtype=dtype)
    k = torch.randn((1, 1, seqlen_k, dim), device="cuda", dtype=dtype)
    k[..., : dim // 2] = k_value
    v = torch.randn_like(k)
    num_kv_blocks = seqlen_k // sparse_block_size
    q2k = torch.arange(num_kv_blocks, device="cuda", dtype=torch.int32).expand(1, 1, seqlen_q // sparse_block_size, -1).contiguous()

    result = ops.block_sparse_attention_forward(q, k, v, q2k, num_kv_blocks, sparse_block_size=sparse_block_size)
    mask = torch.zeros((seqlen_q, seqlen_k), device="cuda", dtype=torch.float32)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    # Even rows: LSE ~1e9, where the FP32 reference itself is off by a few hundred; bound it relatively.
    torch.testing.assert_close(result["lse_tensor"][..., 0::2].double(), lse_ref[..., 0::2].double(), atol=0, rtol=2e-6)
    torch.testing.assert_close(result["lse_tensor"][..., 1::2], lse_ref[..., 1::2], atol=2e-3, rtol=2e-3)


# ---- SM90 blk64


@pytest.mark.L0
@pytest.mark.parametrize("kv_splits", (1, 2))
@pytest.mark.parametrize("with_block_sizes", (True, False), ids=("block_sizes", "short_last_tile"))
@torch_fork_set_rng(seed=6)
def test_sm90_blk64_partial_kv_blocks(with_block_sizes, kv_splits):
    """Partial KV blocks are masked, full ones skip the mask. They come from block_sizes or, without it, from a short
    last KV tile. Each list is walked from its last slot, so row 0 masks in the prologue and row 1 in the loop."""
    ops = import_bsa(families=(9,), reason="the blk64 forward kernel under test is specific to SM90")
    seqlen_q, seqlen_k = 2 * 64, 4 * 64 - (0 if with_block_sizes else 17)
    q = torch.randn((1, 2, seqlen_q, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, seqlen_k, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = torch.tensor([[0, 1, 3], [3, 2, 0]], device="cuda", dtype=torch.int32).expand(1, 2, 2, 3).contiguous()
    block_sizes = torch.tensor([64, 40, 64, 23], device="cuda", dtype=torch.int32) if with_block_sizes else None

    result = ops.block_sparse_attention_forward(q, k, v, q2k, 3, block_sizes, sparse_block_size=64, kv_splits=kv_splits)
    reference_sizes = block_sizes if with_block_sizes else torch.full((4,), 64, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, 3, reference_sizes, seqlen_q, seqlen_k, 64)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L1
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 96, 128))
@pytest.mark.parametrize("value_dim", (96, 128))
@pytest.mark.parametrize("with_block_sizes", (False, True))
@torch_fork_set_rng(seed=28)
def test_sm90_blk64_ring_head_dims(dtype, head_dim, value_dim, with_block_sizes):
    """Exercise real K-ring dispatch, short lists, wraparound, empty rows, and both partial-block paths.

    Repeat five query tiles to cross the large-grid gate without constructing a large dense reference.
    Counts 0/1/2/3/8 cover first fill, first reuse, multiple barrier phases, and drain.
    """
    ops = import_bsa(families=(9,), reason="the blk64 forward kernel under test is specific to SM90")
    copies, seqlen_q, seqlen_k = 256, 5 * 64, 16 * 64 - 17
    q = torch.randn((1, 1, seqlen_q, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((1, 1, seqlen_k, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((1, 1, seqlen_k, value_dim), device="cuda", dtype=dtype)
    counts = torch.tensor([[[0, 1, 2, 3, 8]]], device="cuda", dtype=torch.int32)
    indices = torch.tensor([15, 1, 4, 6, 8, 10, 12, 14], device="cuda", dtype=torch.int32).repeat(1, 1, 5, 1)
    indices[:, :, 4] = indices[:, :, 4].flip(-1)  # The tail occurs in both prologue and steady state.
    indices.masked_fill_(torch.arange(8, device="cuda") >= counts[..., None], 2147483647)
    sizes = torch.full((16,), 64, device="cuda", dtype=torch.int32)
    if with_block_sizes:
        sizes[1::2] = 23
    mask = block_sparse_mask(indices, 8, sizes, seqlen_q, seqlen_k, 64, counts)
    o_ref, lse_ref = attention_reference(q, k, v, mask)

    result = ops.block_sparse_attention_forward(
        q.repeat(1, 1, copies, 1),
        k,
        v,
        indices.repeat(1, 1, copies, 1),
        8,
        sizes if with_block_sizes else None,
        q2k_block_nums=counts.repeat(1, 1, copies),
        sparse_block_size=64,
        kv_splits=1,
        allow_empty_block_nums=True,
    )
    torch.testing.assert_close(result["o_tensor"].float(), o_ref.repeat(1, 1, copies, 1), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref.repeat(1, 1, copies), atol=2e-3, rtol=2e-3)


@pytest.mark.L1
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 96, 128))
@pytest.mark.parametrize("value_dim", (64, 96, 128))
@pytest.mark.parametrize("with_block_sizes", (False, True))
@torch_fork_set_rng(seed=29)
def test_sm90_blk64_rescale_row_slots(dtype, head_dim, value_dim, with_block_sizes):
    """Cover unchanged/increasing maxima, different row slots, and mixed rows within a warp vote."""
    ops = import_bsa(families=(9,), reason="the blk64 forward kernel under test is specific to SM90")
    seqlen_q, seqlen_k = 5 * 64, 16 * 64 - 17
    q = torch.zeros((1, 1, seqlen_q, head_dim), device="cuda", dtype=dtype)
    row = torch.arange(64, device="cuda")
    q[0, 0, 64:128, 0] = 8
    q[0, 0, 128:192, 0] = torch.where(row % 16 < 8, -8, 8)
    q[0, 0, 192:256, 0] = torch.where(row % 16 < 8, 8, -8)
    q[0, 0, 256:320, 0] = torch.where(row % 2 == 0, 8, -8)
    k = torch.zeros((1, 1, seqlen_k, head_dim), device="cuda", dtype=dtype)
    k[0, 0, :, 0] = (torch.arange(seqlen_k, device="cuda") // 64).to(dtype) * 0.5
    v = torch.randn((1, 1, seqlen_k, value_dim), device="cuda", dtype=dtype)
    # Reverse traversal visits blocks in increasing logit order for positive Q rows.
    indices = torch.tensor([15, 6, 5, 4, 3, 2, 1, 0], device="cuda", dtype=torch.int32).repeat(1, 1, 5, 1)
    sizes = torch.full((16,), 64, device="cuda", dtype=torch.int32)
    if with_block_sizes:
        sizes[:2] = 0  # Two fully masked blocks before the first finite maximum.
        sizes[15] = 23
    result = ops.block_sparse_attention_forward(q, k, v, indices, 8, sizes if with_block_sizes else None, sparse_block_size=64, kv_splits=1)
    mask = block_sparse_mask(indices, 8, sizes, seqlen_q, seqlen_k, 64)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=7)
def test_sm90_blk64_auto_kv_splits_matches_unsplit(monkeypatch):
    """kv_splits="auto" resolves to a split count before the output is allocated and matches the unsplit result."""
    BSA = import_bsa(families=(9,), reason="SM90 blk64 split policy")
    interface = importlib.import_module("cudnn.block_sparse_attention._interface")
    q = torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 1024, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ids = torch.arange(0, 16, device="cuda", dtype=torch.int32).expand(1, 2, 4, 16).contiguous()
    expected = BSA.block_sparse_attention_forward(q, k, v, ids, 16, sparse_block_size=64, kv_splits=1)
    seen = []

    def choose(q_arg, k_arg, v_arg, count):
        seen.append(count)
        return 2

    monkeypatch.setattr(interface, "_sm90_blk64_auto_kv_splits", choose)
    result = BSA.block_sparse_attention_forward(q, k, v, ids, 16, sparse_block_size=64, kv_splits="auto")
    assert seen == [16]
    torch.testing.assert_close(result["o_tensor"], expected["o_tensor"], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(result["lse_tensor"], expected["lse_tensor"], atol=1e-3, rtol=1e-3)


# ---- SM90 blk128


@pytest.mark.L1
@pytest.mark.parametrize("d,dv", [(d, dv) for d in (64, 96, 128) for dv in (64, 96, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm90_blk128_dimensions(d, dv, dtype):
    require_arch(families=(9,), reason="the native SM90 blk128 kernels are under test")
    check_blk128(d=d, dv=dv, dtype=dtype, backward=False, ratio=4, layout="bshd")


@pytest.mark.L1
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("splits", [1, 2, 7, "auto"])
def test_sm90_blk128_split_partial_gqa(rank, splits):
    require_arch(families=(9,), reason="the native SM90 blk128 kernels are under test")
    check_blk128(sq=129, sk=513, variable=True, sizes_rank=rank, splits=splits, ratio=2, backward=False)


@pytest.mark.L1
@pytest.mark.parametrize("d,dv,dtype", [(64, 96, torch.float16), (96, 128, torch.float16), (128, 64, torch.bfloat16), (96, 96, torch.bfloat16)])
def test_sm90_blk128_split_mixed_dimensions(d, dv, dtype):
    require_arch(families=(9,), reason="the native SM90 blk128 kernels are under test")
    check_blk128(sq=129, sk=257, d=d, dv=dv, dtype=dtype, layout="bshd", ratio=4, variable=True, sizes_rank=3, splits=7, backward=False)


@pytest.mark.L1
def test_sm90_blk128_maximum_splits():
    require_arch(families=(9,), reason="the native SM90 blk128 kernels are under test")
    check_blk128(sq=1, sk=1, d=64, dv=96, dtype=torch.float16, layout="bshd", ratio=4, splits=256, backward=False)


@pytest.mark.L1
@pytest.mark.parametrize("active_count", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("variable", [False, True])
def test_sm90_blk128_pipeline_lengths(active_count, variable):
    """Prologue/drain, both ring phases, and empty rows alongside active rows; variable counts use two stages even
    when the maximum count is one."""
    require_arch(families=(9,), reason="the native SM90 blk128 kernels are under test")
    check_blk128(sq=129, sk=1153, active_count=active_count, variable=variable, sizes_rank=2, backward=False)


@pytest.mark.L1
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125])
@pytest.mark.parametrize("variable", [False, True])
def test_sm90_blk128_partial_prologue(scale, variable):
    """Begin with the one-token physical KV tail, then visit full blocks; variable counts include empty rows and
    exercise the scheduler's short-list paths."""
    ops = import_bsa(families=(9,), reason="the native SM90 blk128 kernels are under test")
    q, k, v, ids, counts, sizes, count = blk128_case(sq=129, sk=1153, dv=96, ratio=2, dtype=torch.float16, active_count=8, sizes_rank=2, variable=variable)
    order = torch.cat((torch.tensor([9], device=ids.device, dtype=ids.dtype), torch.arange(count - 1, device=ids.device, dtype=ids.dtype)))
    ids.copy_(order)
    sizes.fill_(128)
    sizes[:, -1] = 1
    result = ops.block_sparse_attention_forward(
        q, k, v, ids, count, q2k_block_nums=counts, allow_empty_block_nums=variable, block_sizes=sizes, sparse_block_size=128, softmax_scale=scale
    )
    ro, rl, _ = blk128_reference(q, k, v, ids, counts, sizes, count, scale, "bhsd")
    torch.testing.assert_close(result["o_tensor"].double(), ro, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(result["lse_tensor"].double(), rl, atol=1e-4, rtol=1e-5)


# ---- SM100/SM110 blk64


@pytest.mark.L1
@pytest.mark.parametrize("use_clc", (False, True))
@pytest.mark.parametrize("layout", ("bhsd", "bshd"))
@pytest.mark.parametrize("variable_counts", (False, True), ids=("fixed", "variable_with_empty"))
@pytest.mark.parametrize("with_block_sizes", (False, True), ids=("full_blocks", "partial_blocks"))
@torch_fork_set_rng(seed=4593)
def test_sm100_blk64_persistent_query_replay(use_clc, layout, variable_counts, with_block_sizes):
    """Multiple persistent waves exercise KV-ring rotation, empty queries and changed-input replay.

    Repeat distinct query tiles to bound the independent dense reference while
    forcing each SM to process several tiles. Counts cross the 8-block padding
    boundary and different ring phases. Q always has a partial final tile;
    partial KV blocks use explicit block_sizes, as required by the API.
    """
    ops = import_bsa(families=(10, 11), reason="persistent blk64 forward requires SM100/SM110")
    batch, heads, pattern_tiles, capacity = 2, 4, 9, 32
    copies = max(2, math.ceil(4 * torch.cuda.get_device_properties(0).multi_processor_count / (batch * heads * pattern_tiles)))
    length_q = pattern_tiles * 64 * copies - 17
    length_k = capacity * 64 - (17 if with_block_sizes else 0)
    q_base = torch.randn((batch, heads, pattern_tiles * 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, length_k, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # The final KV block occurs in every active list, with no duplicates.
    other_indices = torch.rand((batch, heads, pattern_tiles, capacity - 1), device="cuda").argsort(-1).to(torch.int32)
    indices_base = torch.cat((torch.full_like(other_indices[..., :1], capacity - 1), other_indices), -1)
    counts_base = torch.tensor([0, 1, 3, 7, 8, 9, 16, 17, 32], device="cuda", dtype=torch.int32).expand(batch, heads, -1).clone()
    if not variable_counts:
        counts_base.fill_(8)
    indices = indices_base.repeat(1, 1, copies, 1)
    counts = counts_base.repeat(1, 1, copies)
    q = q_base.repeat(1, 1, copies, 1)[..., :length_q, :].contiguous()
    reference_sizes = torch.full((capacity,), 64, device="cuda", dtype=torch.int32)
    if with_block_sizes:
        reference_sizes[1::2] = 23
        reference_sizes[2] = 0
        reference_sizes[-1] = 47
    inputs = [q, k, v] if layout == "bhsd" else [t.transpose(1, 2).contiguous() for t in (q, k, v)]

    def invoke():
        return ops.block_sparse_attention_forward(
            *inputs,
            indices,
            block_sparse_num=capacity if variable_counts else 8,
            block_sizes=reference_sizes if with_block_sizes else None,
            q2k_block_nums=counts if variable_counts else None,
            sparse_block_size=64,
            layout=layout,
            use_clc=use_clc,
            kv_splits=1,
            allow_empty_block_nums=variable_counts,
        )

    invoke()
    torch.cuda.synchronize()
    with cuda_graph() as graph:
        with torch.cuda.graph(graph):
            result = invoke()
        for replay in range(2):
            if replay:
                q_base.mul_(0.75)
                inputs[0].mul_(0.75)
                inputs[2].neg_()
                counts_base = counts_base.flip(-1)
            counts.copy_(counts_base.repeat(1, 1, copies))
            indices.copy_(indices_base.repeat(1, 1, copies, 1))
            indices.masked_fill_(torch.arange(capacity, device="cuda") >= counts[..., None], torch.iinfo(torch.int32).max)
            result["o_tensor"].fill_(float("nan"))
            result["lse_tensor"].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            mask = block_sparse_mask(indices_base, capacity, reference_sizes, pattern_tiles * 64, length_k, 64, counts_base)
            k_ref, v_ref = inputs[1:] if layout == "bhsd" else [t.transpose(1, 2) for t in inputs[1:]]
            o_ref, lse_ref = attention_reference(q_base, k_ref, v_ref, mask)
            expected_o = o_ref.repeat(1, 1, copies, 1)[..., :length_q, :]
            actual_o = result["o_tensor"] if layout == "bhsd" else result["o_tensor"].transpose(1, 2)
            torch.testing.assert_close(actual_o.float(), expected_o, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(result["lse_tensor"], lse_ref.repeat(1, 1, copies)[..., :length_q], atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=2029)
def test_sm100_blk64_split_kv_clc_persistent_tiles():
    """Split-KV with CLC scheduling over more tiles than two waves matches the static schedule, and repeats exactly."""
    ops = import_bsa(families=(10, 11), reason="blk64 split-KV CLC scheduling is specific to SM100/SM110")
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    batch, heads, num_kv_blocks, head_dim = 2, 3, 8, 128
    kv_splits = 3
    num_q_blocks = max(32, 2 * sm_count // (batch * heads * kv_splits) + 1)
    block_size = 64
    seqlen_q, seqlen_k = num_q_blocks * block_size, num_kv_blocks * block_size
    assert batch * heads * num_q_blocks * kv_splits > 2 * sm_count

    q = torch.randn((batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k = (
        torch.arange(num_kv_blocks, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, num_kv_blocks)
        .expand(batch, heads, num_q_blocks, num_kv_blocks)
        .contiguous()
    )
    block_sizes = torch.full((num_kv_blocks,), block_size, device="cuda", dtype=torch.int32)

    def run(use_clc):
        result = ops.block_sparse_attention_forward(
            q, k, v, q2k, num_kv_blocks, block_sizes, sparse_block_size=block_size, use_clc=use_clc, kv_splits=kv_splits
        )
        torch.cuda.synchronize()
        return result

    reference, actual, repeated = run(False), run(True), run(True)
    torch.testing.assert_close(actual["o_tensor"], reference["o_tensor"], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual["lse_tensor"], reference["lse_tensor"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(repeated["o_tensor"], actual["o_tensor"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated["lse_tensor"], actual["lse_tensor"], rtol=0.0, atol=0.0)


@pytest.mark.L0
@torch_fork_set_rng(seed=6)
@pytest.mark.parametrize("seqlen_q", [1, 63, 65])
def test_sm100_blk64_combine_partial_tail_rows(seqlen_q):
    """Exercise a partial Q tile through both the producer and split combine.

    A non-multiple-of-64 sequence makes the blk64 producer's final tile partial
    and leaves invalid rows in the split-combine tile. The former must keep every
    correction warp converged through its exchange barriers; the latter must
    initialize invalid shared-LSE rows to ``-inf``.
    """
    ops = import_bsa(families=(10, 11), reason="blk64 split combine is specific to SM100/SM110")
    block_size = 64
    batch, heads, seqlen_k, dim = 1, 1, 256, 128
    q = torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, seqlen_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q2k, block_sparse_num, block_sizes = make_fixed_metadata(batch, heads, seqlen_q, seqlen_k, block_size)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)

    for kv_splits in (1, 2):
        actual = ops.block_sparse_attention_forward(
            q, k, v, q2k, block_sparse_num, block_sizes, sparse_block_size=block_size, use_clc=False, kv_splits=kv_splits
        )
        torch.testing.assert_close(actual["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(actual["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@pytest.mark.parametrize("clc,splits", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("seqlen", [193, 1089])
def test_sm100_blk64_empty_rows_and_partial_tiles(clc, splits, seqlen):
    """Variable counts with empty rows, a partial final Q tile and a partial final KV block, per scheduler and split."""
    bsa = import_bsa(archs=(100, 103), reason="requires SM100/SM103")
    torch.manual_seed(19000 + seqlen)
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


@pytest.mark.L0
@pytest.mark.parametrize("misalignment", ["stride", "address", "aligned_stride"])
@pytest.mark.parametrize("tensor_index", [0, 1, 2], ids=["q", "k", "v"])
@pytest.mark.parametrize("splits", [1, 2])
def test_sm100_blk64_bshd_sliced_inputs(misalignment, tensor_index, splits):
    """BSHD q/k/v views with a padded row stride or a misaligned start address."""
    bsa = import_bsa(archs=(100, 103), reason="requires SM100/SM103")
    torch.manual_seed(20260911)
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


# ---- SM120 blk128


def _require_fa4():
    """Skip unless the CuTe DSL meets the FA4-style blk128 floor."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("FA4-style blk128 requires a supported CuTe DSL version")


@pytest.mark.L0
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("block_sparse_num", (1, 2, 3, 4, 5))
@torch_fork_set_rng(seed=20)
def test_sm120_blk128_fixed_topk(dtype, block_sparse_num):
    """Cover fixed top-k 1-5, GQA, partial Q, and query-dependent KV selections."""
    ops = import_bsa(families=(12,), reason="FA4-style blk128 forward is specific to SM120")
    _require_fa4()
    block_size = 128
    batch, q_heads, kv_heads, seqlen_q, seqlen_k, dim = 2, 4, 2, block_size + 1, 6 * block_size, 128
    q = torch.randn((batch, q_heads, seqlen_q, dim), device="cuda", dtype=dtype)
    k = torch.randn((batch, kv_heads, seqlen_k, dim), device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    q_block = torch.arange(2, device="cuda", dtype=torch.int32).view(1, 1, 2, 1)
    batch_head = torch.arange(batch * q_heads, device="cuda", dtype=torch.int32).view(batch, q_heads, 1, 1)
    slots = torch.tensor([0, 2, 3, 5, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 5)
    q2k = ((q_block + batch_head + slots) % 6).contiguous()

    result = ops.block_sparse_attention_forward(
        q,
        k,
        v,
        q2k,
        block_sparse_num=block_sparse_num,
        sparse_block_size=block_size,
    )
    block_sizes = torch.full((6,), block_size, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("wave_kind,block_sparse_num", (("tail_only", 3), ("unsplit", 1), ("full", 5), ("mixed", 3)))
@torch_fork_set_rng(seed=20)
def test_sm120_blk128_wave_boundaries(dtype, wave_kind, block_sparse_num):
    """Validate all query rows across full, split-tail, mixed, and unsplit waves."""
    require_arch(archs=(120,), reason="native blk128 wave scheduling requires SM120")
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    full_wave_q_blocks = sm_count // 2 if sm_count % 2 == 0 else sm_count
    q_blocks = {
        "tail_only": 2,
        "unsplit": sm_count // 4 + 1,
        "full": full_wave_q_blocks,
        "mixed": full_wave_q_blocks + 2,
    }[wave_kind]
    sequence = (q_blocks - 1) * 128 + 1
    ops = import_bsa()
    _require_fa4()
    q = torch.randn((1, 2, sequence, 128), device="cuda", dtype=dtype)
    k = torch.randn((1, 1, 768, 128), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    q_block = torch.arange(q_blocks, device="cuda", dtype=torch.int32).view(1, 1, -1, 1)
    head = torch.arange(2, device="cuda", dtype=torch.int32).view(1, 2, 1, 1)
    slots = torch.tensor([0, 2, 3, 5, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 5)
    q2k = ((3 * q_block + 7 * head + slots) % 6).contiguous()
    result = ops.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num=block_sparse_num, sparse_block_size=128)

    # Exercise every output row, including the Q64 boundary inside blk128 and
    # the final partial Q tile. Per-query metadata prevents accidental reuse
    # of the first sparse row from passing this check.
    block_sizes = torch.full((6,), 128, device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, block_sparse_num, block_sizes, sequence, 768, 128)
    o_ref, lse_ref = attention_reference(q, k, v, mask)
    torch.testing.assert_close(result["o_tensor"].float(), o_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(result["lse_tensor"], lse_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.L0
def test_sm120_blk128_wave_planning_is_not_repeated_on_cache_hits(monkeypatch):
    """Ensure cached launches reuse the SM-count decision made during tracing."""
    ops = import_bsa(archs=(120,), reason="native blk128 wave scheduling requires SM120")
    _require_fa4()
    from cudnn.block_sparse_attention import _interface
    from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128 import bsa_fwd_sm120_fa4

    query_count = 0
    original_query = bsa_fwd_sm120_fa4._device_sm_count

    def counted_query():
        """Count device-metadata queries while preserving the actual SM count."""
        nonlocal query_count
        query_count += 1
        return original_query()

    monkeypatch.setattr(bsa_fwd_sm120_fa4, "_device_sm_count", counted_query)
    monkeypatch.setattr(_interface.bsa_attn_fwd, "compile_cache", {})
    q = torch.zeros((1, 1, 129, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    indices = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)
    first = ops.block_sparse_attention_forward(q, k, v, indices, block_sparse_num=1, sparse_block_size=128)
    assert query_count > 0
    trace_query_count = query_count
    second = ops.block_sparse_attention_forward(q, k, v, indices, block_sparse_num=1, sparse_block_size=128)
    assert query_count == trace_query_count
    torch.testing.assert_close(first["o_tensor"], torch.ones_like(q), atol=0.0, rtol=0.0)
    torch.testing.assert_close(second["o_tensor"], first["o_tensor"], atol=0.0, rtol=0.0)
