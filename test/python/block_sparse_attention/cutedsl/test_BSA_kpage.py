# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPage (kpage=True / kpage=KPageLayout) forward and backward against the base kernels and the fp32 reference.

Two padding contracts are covered: the BSA one (``block_sizes`` masks keys,
every query row is valid) and the page-compacted one where the query side
carries the same per-block padding (``kpage_prepare_layout(q_block_sizes=...)``).
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import attention_backward_reference, attention_reference, block_sparse_mask
from block_sparse_attention.cutedsl.bsa_utils import import_kpage_bsa, random_metadata, valid_rows

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

BLOCK = 128


def _is_sm90() -> bool:
    return torch.cuda.get_device_capability()[0] == 9


def _inputs(heads: int, block_sizes: torch.Tensor, pad_fill: float = 0.0):
    """BF16 q/k/v/do; padding rows hold ``pad_fill`` (any finite value is allowed)."""
    seqlen = block_sizes.numel() * BLOCK
    valid = valid_rows(block_sizes, BLOCK)
    tensors = []
    for _ in range(4):
        t = torch.randn((1, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
        t[0, :, ~valid] = pad_fill
        tensors.append(t)
    return tensors


def _check(actual: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor, tol: float):
    """Valid rows within ``tol`` scaled to the reference magnitude (GQA dK/dV sum over a head group); padding rows exactly zero."""
    scale = max(reference[0, :, valid].abs().max().item(), 1.0)
    torch.testing.assert_close(actual[0, :, valid].float(), reference[0, :, valid], atol=tol * scale, rtol=tol)
    assert torch.equal(actual[0, :, ~valid], torch.zeros_like(actual[0, :, ~valid]))


@pytest.mark.L0
@torch_fork_set_rng(seed=11)
@pytest.mark.parametrize("pad_fill", [0.0, 1000.0])
@pytest.mark.parametrize(
    "block_sizes",
    [
        [128, 128, 128, 128],  # all full: same work as the base blk128 kernel
        [128, 17, 64, 1, 100, 128, 33, 96],  # partial blocks on both routes (<=64 rows -> M64 x N256)
        [5, 9, 16, 48, 64, 64, 65, 127],  # mostly small blocks, several per K/V tile
    ],
)
def test_kpage_q_padding_matches_reference(block_sizes, pad_fill):
    """q_block_sizes=block_sizes: padding rows of q/k/v/do may hold any finite value; valid rows are exact, padding rows zero."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads = 2
    bs = torch.tensor(block_sizes, dtype=torch.int32, device="cuda")
    num_blocks = bs.numel()
    seqlen = num_blocks * BLOCK
    q, k, v, do = _inputs(heads, bs, pad_fill)
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=4)
    mask = block_sparse_mask(q2k, 0, bs, seqlen, seqlen, BLOCK, q2k_block_nums=nums)
    valid = valid_rows(bs, BLOCK)
    # The reference has no notion of padding query rows: give them zero upstream
    # gradient so they contribute nothing to dK/dV, as in the kernel.
    do_ref = do.clone()
    do_ref[0, :, ~valid] = 0
    o_ref, lse_ref, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do_ref, mask)

    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs)
    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK)
    _check(forward["o_tensor"], o_ref, valid, tol=2e-2)
    torch.testing.assert_close(forward["lse_tensor"][0, :, valid], lse_ref[0, :, valid], atol=1e-3, rtol=1e-3)
    # same K masking on the valid query rows as the base kernel
    base = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, sparse_block_size=BLOCK)
    torch.testing.assert_close(forward["o_tensor"][0, :, valid].float(), base["o_tensor"][0, :, valid].float(), atol=2e-2, rtol=2e-2)

    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK
    )
    _check(backward["dq_tensor"], dq_ref, valid, tol=3e-2)
    _check(backward["dk_tensor"], dk_ref, valid, tol=3e-2)
    _check(backward["dv_tensor"], dv_ref, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=10)
@pytest.mark.parametrize("seqlen_q_blocks, kv_block_sizes", [(6, [128, 17, 64, 1, 100, 128]), (3, [5, 128, 9, 16, 48, 64, 64, 127]), (8, [128, 128, 128])])
def test_kpage_bsa_semantics_matches_base_kernel(seqlen_q_blocks, kv_block_sizes):
    """kpage=True keeps the base contract: block_sizes masks keys, every query row is valid, seqlen_q and seqlen_k are independent."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads = 2
    bs = torch.tensor(kv_block_sizes, dtype=torch.int32, device="cuda")
    num_kv_blocks = bs.numel()
    seqlen_q, seqlen_k = seqlen_q_blocks * BLOCK, num_kv_blocks * BLOCK
    kv_valid = valid_rows(bs, BLOCK)
    q = torch.randn((1, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16)
    do = torch.randn_like(q)
    k = torch.randn((1, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k[0, :, ~kv_valid] = v[0, :, ~kv_valid] = 1000.0  # masked keys may hold any finite value
    kmax = min(3, num_kv_blocks)
    q2k = (
        torch.stack([torch.randperm(num_kv_blocks, device="cuda")[:kmax] for _ in range(heads * seqlen_q_blocks)])
        .view(1, heads, seqlen_q_blocks, kmax)
        .to(torch.int32)
    )
    nums = torch.randint(1, kmax + 1, (1, heads, seqlen_q_blocks), device="cuda", dtype=torch.int32)
    mask = block_sparse_mask(q2k, 0, bs, seqlen_q, seqlen_k, BLOCK, q2k_block_nums=nums)
    o_ref, lse_ref, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)

    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=True, sparse_block_size=BLOCK)
    base = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, sparse_block_size=BLOCK)
    torch.testing.assert_close(forward["o_tensor"].float(), base["o_tensor"].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(forward["lse_tensor"], base["lse_tensor"], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(forward["o_tensor"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(forward["lse_tensor"], lse_ref, atol=1e-3, rtol=1e-3)

    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=True, sparse_block_size=BLOCK
    )
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    _check(backward["dk_tensor"], dk_ref, kv_valid, tol=3e-2)
    _check(backward["dv_tensor"], dv_ref, kv_valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize("route", ["m128", "m64"])
@pytest.mark.parametrize("softmax_scale", [0.125, 0.0, -0.125])
def test_kpage_softmax_scale_sign_with_partial_blocks(softmax_scale, route):
    """Zero and negative scales with partially valid key blocks, on both forward routes and the backward.

    max(scale * S) is not scale * max(S) for a negative scale, and a masked key's -inf times a zero or negative
    scale is NaN / +inf; the M64 route also shares its first softmax step with the later ones.  ``m64`` gives every
    query block 64 valid rows, so all of them take the M64 x N256 kernel.
    """
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 2, 4
    seqlen = num_blocks * BLOCK
    bs = torch.tensor([128, 40, 128, 7], dtype=torch.int32, device="cuda")
    q_sizes = torch.full((num_blocks,), 64 if route == "m64" else BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, do = _inputs(heads, bs)
    kv_valid = valid_rows(bs, BLOCK)
    k[0, :, ~kv_valid] = v[0, :, ~kv_valid] = 1000.0  # masked keys may hold any finite value
    q_valid = valid_rows(q_sizes, BLOCK)
    do[0, :, ~q_valid] = 0.0  # padding query rows carry dO = 0
    q2k = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(1, 1, 1, num_blocks).expand(1, heads, num_blocks, num_blocks).contiguous()
    layout = BSA.kpage_prepare_layout(q2k, num_blocks, bs, q_block_sizes=q_sizes)
    mask = block_sparse_mask(q2k, num_blocks, bs, seqlen, seqlen, BLOCK)
    o_ref, lse_ref, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask, softmax_scale=softmax_scale)

    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, num_blocks, bs, kpage=layout, softmax_scale=softmax_scale, sparse_block_size=BLOCK)
    _check(forward["o_tensor"], o_ref, q_valid, tol=2e-2)
    torch.testing.assert_close(forward["lse_tensor"][0, :, q_valid], lse_ref[0, :, q_valid], atol=1e-3, rtol=1e-3)

    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, num_blocks, bs, kpage=layout, softmax_scale=softmax_scale, sparse_block_size=BLOCK
    )
    _check(backward["dq_tensor"], dq_ref, q_valid, tol=3e-2)
    _check(backward["dk_tensor"], dk_ref, kv_valid, tol=3e-2)
    _check(backward["dv_tensor"], dv_ref, kv_valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=12)
def test_kpage_layout_reuse_and_preallocated_outputs():
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 3, 6
    bs = torch.tensor([128, 40, 128, 7, 90, 128], dtype=torch.int32, device="cuda")
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs)
    assert isinstance(layout, BSA.KPageLayout)
    valid = valid_rows(bs, BLOCK)
    for _ in range(2):  # same layout, new activations
        q, k, v, do = _inputs(heads, bs)
        mask = block_sparse_mask(q2k, 0, bs, num_blocks * BLOCK, num_blocks * BLOCK, BLOCK, q2k_block_nums=nums)
        o_ref, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)
        forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK)
        _check(forward["o_tensor"], o_ref, valid, tol=2e-2)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        backward = BSA.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            forward["o_tensor"],
            forward["lse_tensor"],
            q2k,
            None,
            bs,
            nums,
            dq_tensor=dq,
            dk_tensor=dk,
            dv_tensor=dv,
            kpage=layout,
            sparse_block_size=BLOCK,
        )
        assert backward["dq_tensor"].data_ptr() == dq.data_ptr()
        assert backward["dk_tensor"].data_ptr() == dk.data_ptr()
        assert backward["dv_tensor"].data_ptr() == dv.data_ptr()
        _check(dq, dq_ref, valid, tol=3e-2)
        _check(dk, dk_ref, valid, tol=3e-2)
        _check(dv, dv_ref, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize("softmax_scale", [None, 0.2])
@pytest.mark.parametrize("block_sparse_num", [2, 3])
def test_kpage_fixed_count_full_blocks_matches_base_kernel(softmax_scale, block_sparse_num):
    """kpage with block_sizes=None and a fixed count is the plain blk128 problem (an odd count is fine here)."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 2, 4
    seqlen = num_blocks * BLOCK
    q = torch.randn((1, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    selected = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    q2k = selected.view(1, 1, 1, 3).expand(1, heads, num_blocks, 3).contiguous()
    paged = BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, None, softmax_scale=softmax_scale, kpage=True, sparse_block_size=BLOCK)
    if block_sparse_num % 2 == 0:
        base = BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, None, softmax_scale=softmax_scale, sparse_block_size=BLOCK)
        torch.testing.assert_close(paged["o_tensor"].float(), base["o_tensor"].float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(paged["lse_tensor"], base["lse_tensor"], atol=1e-3, rtol=1e-3)
    bs = torch.full((num_blocks,), BLOCK, dtype=torch.int32, device="cuda")
    mask = block_sparse_mask(q2k, block_sparse_num, bs, seqlen, seqlen, BLOCK)
    o_ref, lse_ref = attention_reference(q, k, v, mask, softmax_scale=softmax_scale)
    torch.testing.assert_close(paged["o_tensor"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(paged["lse_tensor"], lse_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
def test_kpage_rejects_unsupported_configurations():
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 2, 4
    seqlen = num_blocks * BLOCK
    q = torch.randn((1, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q2k = torch.tensor([0, 2], dtype=torch.int32, device="cuda").view(1, 1, 1, 2).expand(1, heads, num_blocks, 2).contiguous()
    bs = torch.full((num_blocks,), BLOCK, dtype=torch.int32, device="cuda")

    with pytest.raises(NotImplementedError, match="contiguous q2k_block_index"):
        BSA.block_sparse_attention_forward(q, k, v, q2k.transpose(2, 3).contiguous().transpose(2, 3), 2, bs, kpage=True, sparse_block_size=BLOCK)
    q2, k2, v2 = (torch.randn((2, seqlen, heads, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3))
    with pytest.raises(NotImplementedError, match="BSHD batch > 1"):
        BSA.block_sparse_attention_forward(q2, k2, v2, q2k.expand(2, -1, -1, -1).contiguous(), 2, bs, layout="bshd", kpage=True, sparse_block_size=BLOCK)
    with pytest.raises(NotImplementedError, match="BF16"):
        BSA.block_sparse_attention_forward(q.half(), k.half(), v.half(), q2k, 2, bs, kpage=True, sparse_block_size=BLOCK)
    with pytest.raises(ValueError, match="q_block_sizes"):
        BSA.kpage_prepare_layout(q2k, 2, bs, q_block_sizes=bs[:2])
    with pytest.raises(ValueError, match="seqlen_k"):
        BSA.kpage_prepare_layout(q2k, 2, bs, seqlen_k=2 * BLOCK)
    k_strided = torch.randn((1, heads, seqlen, 256), device="cuda", dtype=torch.bfloat16)[..., :128]
    with pytest.raises(NotImplementedError, match="compact"):
        BSA.block_sparse_attention_forward(q, k_strided, v, q2k, 2, bs, kpage=True, sparse_block_size=BLOCK)
    with pytest.raises(NotImplementedError, match="pack_gqa"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, 2, bs, kv_splits=2, kpage=True, sparse_block_size=BLOCK)
    with pytest.raises(TypeError, match="kpage must be"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, 2, bs, kpage="yes", sparse_block_size=BLOCK)
    if _is_sm90():
        # SM90 runs blk128 KPage on 64-row halves: MHA, sequences that are multiples of 64, no bwd_kv_groups
        with pytest.raises(NotImplementedError, match="multiples of 64"):
            BSA.block_sparse_attention_forward(q[:, :, :500].contiguous(), k, v, q2k, 2, bs, kpage=True, sparse_block_size=BLOCK)
        with pytest.raises(NotImplementedError, match="multiples of 64"):
            BSA.kpage_prepare_layout(q2k, 2, bs, seqlen_k=500)
        with pytest.raises(NotImplementedError, match="MHA only"):
            BSA.block_sparse_attention_forward(q, k[:, :1].contiguous(), v[:, :1].contiguous(), q2k, 2, bs, kpage=True, sparse_block_size=BLOCK)
        with pytest.raises(NotImplementedError, match="bwd_kv_groups"):
            BSA.kpage_prepare_layout(q2k, 2, bs, bwd_kv_groups=2)

    # metadata contract: duplicate ids, out-of-range ids, empty rows, bad block_sizes
    dup = torch.tensor([1, 1], dtype=torch.int32, device="cuda").view(1, 1, 1, 2).expand(1, heads, num_blocks, 2).contiguous()
    with pytest.raises(ValueError, match="duplicate"):
        BSA.kpage_prepare_layout(dup, 2, bs)
    oob = torch.tensor([0, 9], dtype=torch.int32, device="cuda").view(1, 1, 1, 2).expand(1, heads, num_blocks, 2).contiguous()
    with pytest.raises(ValueError, match="outside"):
        BSA.kpage_prepare_layout(oob, 2, bs)
    nums = torch.full((1, heads, num_blocks), 2, dtype=torch.int32, device="cuda")
    nums[0, 0, 1] = 0
    with pytest.raises(ValueError, match="at least one KV block"):
        BSA.kpage_prepare_layout(q2k, None, bs, nums)
    with pytest.raises(ValueError, match="at least one KV block"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=True, sparse_block_size=BLOCK)
    nums[0, 0, 1] = -1
    with pytest.raises(ValueError, match="at least one KV block"):
        BSA.kpage_prepare_layout(q2k, None, bs, nums, allow_empty_block_nums=True)
    with pytest.raises(ValueError, match="block_sizes"):
        BSA.kpage_prepare_layout(q2k, 2, torch.full((num_blocks,), 200, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="zero valid tokens"):
        BSA.kpage_prepare_layout(q2k, 2, torch.tensor([0, BLOCK, 0, BLOCK], dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="exceeds"):
        BSA.kpage_prepare_layout(q2k, None, bs, torch.full((1, heads, num_blocks), 3, dtype=torch.int32, device="cuda"))
    # more walks than the device check has CTAs: a clean pattern repeated in every walk passes; a violation in a late
    # walk (ids in different bitset words) is still found
    many_blocks = 1200
    many_bs = torch.full((80,), BLOCK, dtype=torch.int32, device="cuda")
    many = torch.tensor([0, 40, 70], dtype=torch.int32, device="cuda").repeat(1, heads, many_blocks, 1)
    BSA.kpage_prepare_layout(many, 3, many_bs)
    late = many.clone()
    late[0, heads - 1, many_blocks - 7, 2] = 0
    with pytest.raises(ValueError, match="duplicate"):
        BSA.kpage_prepare_layout(late, 3, many_bs)
    late[0, heads - 1, many_blocks - 7, 2] = 80
    with pytest.raises(ValueError, match="outside"):
        BSA.kpage_prepare_layout(late, 3, many_bs)

    # a layout built for another problem, or from other metadata tensors, is rejected
    other = BSA.kpage_prepare_layout(q2k[:, :1], 2, bs)
    with pytest.raises(ValueError, match="heads"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, 2, bs, kpage=other, sparse_block_size=BLOCK)
    with pytest.raises(ValueError, match="key tokens"):
        BSA.block_sparse_attention_forward(
            q, k[:, :, :BLOCK].contiguous(), v[:, :, :BLOCK].contiguous(), q2k, 2, bs[:1], kpage=BSA.kpage_prepare_layout(q2k, 2, bs), sparse_block_size=BLOCK
        )
    layout = BSA.kpage_prepare_layout(q2k, 2, bs)
    with pytest.raises(ValueError, match="different sparse metadata"):
        BSA.block_sparse_attention_forward(q, k, v, q2k.clone(), 2, bs, kpage=layout, sparse_block_size=BLOCK)
    with pytest.raises(ValueError, match="different sparse metadata"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, 2, None, kpage=layout, sparse_block_size=BLOCK)


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
def test_kpage_bshd_layout_matches_bhsd():
    """layout='bshd' runs on views of the BSHD storage (batch 1) and returns BSHD o/dq/dk/dv."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 3, 6
    seqlen = num_blocks * BLOCK
    bs = torch.tensor([128, 40, 128, 7, 90, 64], dtype=torch.int32, device="cuda")
    q, k, v, do = _inputs(heads, bs, pad_fill=1000.0)
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs, with_backward=True)

    ref_fwd = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK)
    ref_bwd = BSA.block_sparse_attention_backward(
        do, q, k, v, ref_fwd["o_tensor"], ref_fwd["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK
    )

    q_s, k_s, v_s, do_s = (t.transpose(1, 2).contiguous() for t in (q, k, v, do))
    fwd = BSA.block_sparse_attention_forward(q_s, k_s, v_s, q2k, None, bs, nums, layout="bshd", kpage=layout, sparse_block_size=BLOCK)
    assert fwd["o_tensor"].shape == q_s.shape and fwd["o_tensor"].is_contiguous() and fwd["lse_tensor"].shape == (1, heads, seqlen)
    assert torch.equal(fwd["o_tensor"].transpose(1, 2), ref_fwd["o_tensor"]) and torch.equal(fwd["lse_tensor"], ref_fwd["lse_tensor"])
    dq_s = torch.empty_like(q_s)
    bwd = BSA.block_sparse_attention_backward(
        do_s, q_s, k_s, v_s, fwd["o_tensor"], fwd["lse_tensor"], q2k, None, bs, nums, layout="bshd", kpage=layout, dq_tensor=dq_s, sparse_block_size=BLOCK
    )
    assert bwd["dq_tensor"] is dq_s and bwd["dk_tensor"].is_contiguous()
    for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
        torch.testing.assert_close(bwd[name].transpose(1, 2), ref_bwd[name], atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=20)
@pytest.mark.parametrize("with_block_sizes", [False, True])
@pytest.mark.parametrize(
    "seqlen_q, seqlen_k",
    [
        (1000, 700),  # 7 * 128 + 104 query rows (M128 tail); 5 * 128 + 60 key rows
        (1050, 700),  # 8 * 128 + 26 query rows (the tail takes the M64 x N256 route)
        (960, 704),  # 7 * 128 + 64 / 5 * 128 + 64: the tails SM90 takes (its 64-row halves end with the sequence)
    ],
)
def test_kpage_seqlen_not_multiple_of_128(with_block_sizes, seqlen_q, seqlen_k):
    """Rows past seqlen_q / seqlen_k are padding of the final block, with or without block_sizes.

    The base path needs block_sizes for a partial final KV block (its contract); kpage derives the tail itself.
    With several heads, a tail of at most 64 rows on the query side makes the M64 work IDs (stride 2 * num_q_blocks)
    differ from ceil(seqlen_q / 64) unless the scheduler takes the stride from the metadata.  SM90 runs blk128 KPage on
    the metadata's 64-row halves and takes sequences that are multiples of 64 only.
    """
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    if _is_sm90() and (seqlen_q % 64 or seqlen_k % 64):
        pytest.skip("SM90 KPage takes sequences that are multiples of 64 (rejection checked in test_kpage_rejects_unsupported_configurations)")
    heads = 2
    nq, nk = (seqlen_q + BLOCK - 1) // BLOCK, (seqlen_k + BLOCK - 1) // BLOCK
    q, do = (torch.randn((1, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    k, v = (torch.randn((1, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    q2k = torch.stack([torch.randperm(nk, device="cuda")[:3] for _ in range(heads * nq)]).view(1, heads, nq, 3).to(torch.int32)
    nums = torch.randint(1, 4, (1, heads, nq), device="cuda", dtype=torch.int32)
    bs = torch.tensor([128, 40, 128, 128, 128, 128], dtype=torch.int32, device="cuda") if with_block_sizes else None  # last entry exceeds the tail
    bs_eff = torch.full((nk,), BLOCK, dtype=torch.int32, device="cuda") if bs is None else bs.clone()
    bs_eff[-1] = min(int(bs_eff[-1]), seqlen_k % BLOCK)
    kv_valid = valid_rows(bs_eff, BLOCK)[:seqlen_k]

    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=True, sparse_block_size=BLOCK)
    base = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs_eff, nums, sparse_block_size=BLOCK)
    assert forward["o_tensor"].shape == q.shape and forward["lse_tensor"].shape == (1, heads, seqlen_q)
    torch.testing.assert_close(forward["o_tensor"].float(), base["o_tensor"].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(forward["lse_tensor"], base["lse_tensor"], atol=1e-3, rtol=1e-3)

    mask = block_sparse_mask(q2k, 0, bs_eff, seqlen_q, seqlen_k, BLOCK, q2k_block_nums=nums)
    o_ref, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)
    torch.testing.assert_close(forward["o_tensor"].float(), o_ref, atol=2e-2, rtol=2e-2)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, seqlen_q=seqlen_q, seqlen_k=seqlen_k, with_backward=True)
    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK
    )
    assert backward["dq_tensor"].shape == q.shape and backward["dk_tensor"].shape == k.shape
    torch.testing.assert_close(backward["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    _check(backward["dk_tensor"], dk_ref, kv_valid, tol=3e-2)
    _check(backward["dv_tensor"], dv_ref, kv_valid, tol=3e-2)
    with pytest.raises(ValueError, match="key tokens"):  # a layout built for the padded length does not fit the shorter tensors
        BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=BSA.kpage_prepare_layout(q2k, None, bs, nums), sparse_block_size=BLOCK)


@pytest.mark.L0
@torch_fork_set_rng(seed=18)
def test_kpage_empty_query_blocks():
    """allow_empty_block_nums: a query block with no KV blocks gets O = 0, LSE = -inf, dQ = 0 (as on the base path)."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 2, 6
    seqlen = num_blocks * BLOCK
    bs = torch.tensor([128, 40, 128, 7, 90, 128], dtype=torch.int32, device="cuda")
    valid = valid_rows(bs, BLOCK)
    q, k, v, do = _inputs(heads, bs, pad_fill=1000.0)
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    nums[0, 0, 0] = 0  # full block, M128 route
    nums[0, 1, 3] = 0  # 7-row block, M64 route
    nums[0, 1, 4] = 0
    empty = torch.zeros(heads, seqlen, dtype=torch.bool, device="cuda")
    empty[0, :BLOCK] = empty[1, 3 * BLOCK : 5 * BLOCK] = True

    with pytest.raises(ValueError, match="at least one KV block"):
        BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs, allow_empty_block_nums=True)
    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, allow_empty_block_nums=True, sparse_block_size=BLOCK)
    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK
    )

    # The reference cannot take an all-masked row; give empty rows a dense row with dO = 0 (no dK/dV contribution).
    mask = block_sparse_mask(q2k, 0, bs, seqlen, seqlen, BLOCK, q2k_block_nums=nums)
    mask[0][empty] = 0
    do_ref = do.clone()
    do_ref[0, :, ~valid] = 0
    do_ref[0][empty] = 0
    o_ref, lse_ref, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do_ref, mask)
    o_ref[0][empty] = 0
    lse_ref[0][empty] = -float("inf")
    dq_ref[0][empty] = 0
    _check(forward["o_tensor"], o_ref, valid, tol=2e-2)
    _check(backward["dq_tensor"], dq_ref, valid, tol=3e-2)
    _check(backward["dk_tensor"], dk_ref, valid, tol=3e-2)
    _check(backward["dv_tensor"], dv_ref, valid, tol=3e-2)
    lse = forward["lse_tensor"][0]
    assert torch.equal(lse[empty & valid], lse_ref[0][empty & valid]) and torch.equal(lse[:, ~valid], torch.zeros_like(lse[:, ~valid]))
    torch.testing.assert_close(lse[~empty & valid], lse_ref[0][~empty & valid], atol=1e-2, rtol=1e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
def test_kpage_empty_query_blocks_ignore_stale_tmem():
    """An empty tile on the M64 route must not read the O accumulators, which it never wrote.

    Base forwards on NaN inputs first leave NaN in the tensor memory of every SM; 0 * NaN would then turn the
    O of the empty query blocks into NaN instead of 0.
    """
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 8, 32
    bs = torch.full((num_blocks,), 40, dtype=torch.int32, device="cuda")  # at most 64 valid rows: every block takes the M64 route
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    nums[0, :, ::2] = 0
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs, allow_empty_block_nums=True)
    q, k, v, _ = _inputs(heads, bs)
    valid = valid_rows(bs, BLOCK)
    empty = (nums[0] == 0).repeat_interleave(BLOCK, dim=1) & valid.view(1, -1)

    poison = torch.full_like(q, float("nan"))
    full_sizes, full_nums = torch.full_like(bs, BLOCK), torch.full_like(nums, 3)
    for _ in range(4):
        BSA.block_sparse_attention_forward(poison, poison, poison, q2k, None, full_sizes, full_nums, sparse_block_size=BLOCK)
    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, allow_empty_block_nums=True, sparse_block_size=BLOCK)
    o, lse = forward["o_tensor"][0], forward["lse_tensor"][0]
    assert torch.equal(o[empty], torch.zeros_like(o[empty]))
    assert torch.equal(lse[empty], torch.full_like(lse[empty], -float("inf")))
    assert not torch.isnan(o).any()


@pytest.mark.L0
@torch_fork_set_rng(seed=14)
def test_kpage_backward_accepts_base_forward_output():
    """With full blocks the base forward and the kpage backward compose."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    heads, num_blocks = 2, 4
    seqlen = num_blocks * BLOCK
    bs = torch.full((num_blocks,), BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, do = _inputs(heads, bs)
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    mask = block_sparse_mask(q2k, 0, bs, seqlen, seqlen, BLOCK, q2k_block_nums=nums)
    _, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q, k, v, do, mask)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, with_backward=True)
    base = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, sparse_block_size=BLOCK)
    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, base["o_tensor"], base["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK
    )
    valid = valid_rows(bs, BLOCK)
    _check(backward["dq_tensor"], dq_ref, valid, tol=3e-2)
    _check(backward["dk_tensor"], dk_ref, valid, tol=3e-2)
    _check(backward["dv_tensor"], dv_ref, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_kpage_compile_cache_is_shape_independent():
    """A second problem with other heads / sequence length reuses the compiled kernels."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    from cudnn.block_sparse_attention.kpage import _build_kernels, _interface

    def run(heads, block_sizes):
        bs = torch.tensor(block_sizes, dtype=torch.int32, device="cuda")
        q, k, v, do = _inputs(heads, bs)
        q2k, nums = random_metadata(heads, bs.numel(), bs.numel(), kmax=3)
        layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs)
        fwd = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK)
        BSA.block_sparse_attention_backward(do, q, k, v, fwd["o_tensor"], fwd["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK)

    def run_tail(heads, seqlen_q, seqlen_k):
        q, do = (torch.randn((1, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
        k, v = (torch.randn((1, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
        nq, nk = (seqlen_q + BLOCK - 1) // BLOCK, (seqlen_k + BLOCK - 1) // BLOCK
        q2k = torch.stack([torch.randperm(nk, device="cuda")[:2] for _ in range(heads * nq)]).view(1, heads, nq, 2).to(torch.int32)
        fwd = BSA.block_sparse_attention_forward(q, k, v, q2k, 2, None, kpage=True, sparse_block_size=BLOCK)
        BSA.block_sparse_attention_backward(do, q, k, v, fwd["o_tensor"], fwd["lse_tensor"], q2k, 2, None, kpage=True, sparse_block_size=BLOCK)

    caches = [
        _interface._fwd_m128.compile_cache,
        _interface._fwd_m64.compile_cache,
        _interface._bwd_paged_kv.compile_cache,
        _interface._fwd_sm90_pairs.compile_cache,
    ]
    caches += [getattr(_build_kernels, name).compile_cache for name in dir(_build_kernels) if hasattr(getattr(_build_kernels, name), "compile_cache")]
    run(2, [128, 40, 64, 7])
    sizes = [len(c) for c in caches]
    run(3, [128, 128, 9, 100, 128, 64, 128, 128, 50])
    assert [len(c) for c in caches] == sizes
    # A K/V length that ends inside a 16-row page selects the row-extent forward descriptors: at most one
    # more forward compile (none if an earlier test compiled it), shared by every such length; the
    # backward is not specialized on it.
    run_tail(2, 960, 704) if _is_sm90() else run_tail(2, 1000, 700)  # SM90 takes multiples of 64
    sizes_tail = [len(c) for c in caches]
    assert sizes[0] <= sizes_tail[0] <= sizes[0] + 1 and sizes_tail[1:] == sizes[1:]
    run_tail(3, 1472, 896) if _is_sm90() else run_tail(3, 1500, 900)
    assert [len(c) for c in caches] == sizes_tail


@pytest.mark.L0
@torch_fork_set_rng(seed=16)
def test_kpage_batched_matches_per_sample_reference():
    """batch > 1 runs as batch x heads; every sample has its own sparse pattern, block_sizes is shared."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    batch, heads, num_blocks = 2, 2, 6
    seqlen = num_blocks * BLOCK
    bs = torch.tensor([128, 40, 128, 7, 90, 128], dtype=torch.int32, device="cuda")
    valid = valid_rows(bs, BLOCK)
    q, k, v, do = (torch.randn((batch, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16) for _ in range(4))
    q2k = (
        torch.stack([torch.randperm(num_blocks, device="cuda")[:3] for _ in range(batch * heads * num_blocks)])
        .view(batch, heads, num_blocks, 3)
        .to(torch.int32)
    )
    nums = torch.randint(1, 4, (batch, heads, num_blocks), device="cuda", dtype=torch.int32)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs)
    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK)
    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK
    )
    assert forward["o_tensor"].shape == q.shape and forward["lse_tensor"].shape == (batch, heads, seqlen)
    do_ref = do.clone()
    do_ref[:, :, ~valid] = 0
    for b in range(batch):
        sl = slice(b, b + 1)
        mask = block_sparse_mask(q2k[sl], 0, bs, seqlen, seqlen, BLOCK, q2k_block_nums=nums[sl])
        o_ref, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q[sl], k[sl], v[sl], do_ref[sl], mask)
        _check(forward["o_tensor"][sl], o_ref, valid, tol=2e-2)
        _check(backward["dq_tensor"][sl], dq_ref, valid, tol=3e-2)
        _check(backward["dk_tensor"][sl], dk_ref, valid, tol=3e-2)
        _check(backward["dv_tensor"][sl], dv_ref, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=17)
@pytest.mark.parametrize("batch, heads_kv", [(1, 1), (1, 2), (2, 2)])
def test_kpage_gqa_forward_backward(batch, heads_kv):
    """GQA/MQA: the q heads of one kv head read the same K/V pages and sum into one dK/dV."""
    BSA = import_kpage_bsa("the blk128 KPage path runs on SM90 and SM100/SM103")
    if _is_sm90():
        pytest.skip("SM90 KPage is MHA only (test_kpage_rejects_unsupported_configurations checks the error)")
    heads, num_blocks = 4, 6
    seqlen = num_blocks * BLOCK
    bs = torch.tensor([128, 40, 128, 7, 90, 64], dtype=torch.int32, device="cuda")
    valid = valid_rows(bs, BLOCK)
    q = torch.randn((batch, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads_kv, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    for t in (q, k, v, do):
        t[:, :, ~valid] = 1000.0
    q2k = torch.cat([random_metadata(heads, num_blocks, num_blocks, kmax=3)[0] for _ in range(batch)])
    nums = torch.randint(1, 4, (batch, heads, num_blocks), device="cuda", dtype=torch.int32)

    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=bs)
    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK)
    backward = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=BLOCK
    )
    assert backward["dk_tensor"].shape == k.shape and backward["dv_tensor"].shape == v.shape
    do_ref = do.clone()
    do_ref[:, :, ~valid] = 0
    for b in range(batch):
        sl = slice(b, b + 1)
        mask = block_sparse_mask(q2k[sl], 0, bs, seqlen, seqlen, BLOCK, q2k_block_nums=nums[sl])
        o_ref, _, dq_ref, dk_ref, dv_ref = attention_backward_reference(q[sl], k[sl], v[sl], do_ref[sl], mask)
        _check(forward["o_tensor"][sl], o_ref, valid, tol=2e-2)
        _check(backward["dq_tensor"][sl], dq_ref, valid, tol=3e-2)
        _check(backward["dk_tensor"][sl], dk_ref, valid, tol=3e-2)
        _check(backward["dv_tensor"][sl], dv_ref, valid, tol=3e-2)
