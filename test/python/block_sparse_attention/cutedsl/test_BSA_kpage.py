# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPage (``kpage=True`` / ``kpage=KPageLayout``) against the base kernels and the FP32 reference.

KPage runs on SM90 and SM100/SM103 for ``sparse_block_size`` 128 (forward and backward) and 64 (forward only).  Two
padding contracts are covered: the BSA one (``block_sizes`` masks keys, every query row is valid) and, for blk128, the
page-compacted one where the query side carries the same per-block padding (``kpage_prepare_layout(q_block_sizes=...)``).
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from block_sparse_attention.cutedsl.bsa_reference import attention_reference, block_sparse_mask
from block_sparse_attention.cutedsl.bsa_utils import import_kpage_bsa, random_metadata, valid_rows

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

BLOCK = 128
KPAGE = "KPage runs on SM90 and SM100/SM103"
with_block = pytest.mark.parametrize("block", (64, 128), ids=("blk64", "blk128"))


def _is_sm90() -> bool:
    return torch.cuda.get_device_capability()[0] == 9


def _inputs(heads: int, block_sizes: torch.Tensor, pad_fill: float = 0.0, *, block: int = BLOCK, pad_q: bool = True, num_q_blocks=None):
    """BF16 q/k/v/do; padding rows of k/v (and of q/do with ``pad_q``) hold ``pad_fill`` (any finite value is allowed)."""
    seqlen_k = block_sizes.numel() * block
    seqlen_q = seqlen_k if num_q_blocks is None else num_q_blocks * block
    padding = ~valid_rows(block_sizes, block)
    q, do = (torch.randn((1, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    k, v = (torch.randn((1, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
    for t in (q, k, v, do) if pad_q else (k, v):
        t[0, :, padding] = pad_fill
    return q, k, v, do


def _check(actual: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor, tol: float):
    """Valid rows within ``tol`` scaled to the reference magnitude (GQA dK/dV sum over a head group); padding rows exactly zero."""
    scale = max(reference[0, :, valid].abs().max().item(), 1.0)
    torch.testing.assert_close(actual[0, :, valid].float(), reference[0, :, valid], atol=tol * scale, rtol=tol)
    assert torch.equal(actual[0, :, ~valid], torch.zeros_like(actual[0, :, ~valid]))


# ---- blk64 and blk128 (the blk64 path is forward only)


@pytest.mark.L0
@torch_fork_set_rng(seed=10)
@pytest.mark.parametrize(
    "block, num_q_blocks, kv_block_sizes, kmax",
    [
        pytest.param(128, 6, [128, 17, 64, 1, 100, 128], 3, id="blk128-partial"),
        pytest.param(128, 3, [5, 128, 9, 16, 48, 64, 64, 127], 3, id="blk128-more_kv_than_q"),
        pytest.param(128, 8, [128, 128, 128], 3, id="blk128-full"),
        pytest.param(64, 6, [64, 17, 64, 1, 40, 64], 5, id="blk64-partial"),
        pytest.param(64, 3, [5, 64, 9, 16, 48, 64, 64, 63], 7, id="blk64-more_kv_than_q"),
        pytest.param(64, 8, [64, 64, 64], 3, id="blk64-full"),
        pytest.param(64, 130, [64, 33, 64, 64, 7, 64, 64, 64, 50, 64], 9, id="blk64-long_walks"),
    ],
)
def test_matches_base_kernel_and_reference(block, num_q_blocks, kv_block_sizes, kmax):
    """kpage=True keeps the base contract: block_sizes masks keys, every query row is valid, seqlen_q and seqlen_k are
    independent.  On SM90 blk64 one walk serves a pair of query blocks, which the odd counts and unsorted lists exercise."""
    BSA = import_kpage_bsa(KPAGE)
    heads = 2
    bs = torch.tensor(kv_block_sizes, dtype=torch.int32, device="cuda")
    kv_valid = valid_rows(bs, block)
    q, k, v, do = _inputs(heads, bs, 1000.0, block=block, pad_q=False, num_q_blocks=num_q_blocks)  # masked keys may hold any finite value
    q2k, nums = random_metadata(heads, num_q_blocks, bs.numel(), kmax=kmax)
    mask = block_sparse_mask(q2k, 0, bs, q.shape[2], k.shape[2], block, q2k_block_nums=nums)
    backward = block == 128
    o_ref, lse_ref, *grads_ref = attention_reference(q, k, v, mask, do=do if backward else None)

    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=True, sparse_block_size=block)
    base = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, sparse_block_size=block)
    torch.testing.assert_close(forward["o_tensor"].float(), base["o_tensor"].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(forward["lse_tensor"], base["lse_tensor"], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(forward["o_tensor"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(forward["lse_tensor"], lse_ref, atol=1e-3, rtol=1e-3)
    if not backward:
        return

    gradients = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=True, sparse_block_size=block
    )
    dq_ref, dk_ref, dv_ref = grads_ref
    torch.testing.assert_close(gradients["dq_tensor"].float(), dq_ref, atol=3e-2, rtol=3e-2)
    _check(gradients["dk_tensor"], dk_ref, kv_valid, tol=3e-2)
    _check(gradients["dv_tensor"], dv_ref, kv_valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@with_block
@pytest.mark.parametrize("softmax_scale", [None, 0.2])
@pytest.mark.parametrize("block_sparse_num", [2, 3])
def test_fixed_count_matches_base_kernel(block, softmax_scale, block_sparse_num):
    """A fixed count with block_sizes=None is the plain problem; KPage takes odd counts, which the SM100/SM110 blk128
    base kernel does not."""
    BSA = import_kpage_bsa(KPAGE)
    heads, num_blocks = 2, 4
    seqlen = num_blocks * block
    q = torch.randn((1, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    selected = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    q2k = selected.view(1, 1, 1, 3).expand(1, heads, num_blocks, 3).contiguous()
    kwargs = dict(softmax_scale=softmax_scale, sparse_block_size=block)
    paged = BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, None, kpage=True, **kwargs)
    if block == 64 or block_sparse_num % 2 == 0:
        base = BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num, None, **kwargs)
        torch.testing.assert_close(paged["o_tensor"].float(), base["o_tensor"].float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(paged["lse_tensor"], base["lse_tensor"], atol=1e-3, rtol=1e-3)
    mask = block_sparse_mask(q2k, block_sparse_num, None, seqlen, seqlen, block)
    o_ref, lse_ref = attention_reference(q, k, v, mask, softmax_scale=softmax_scale)
    torch.testing.assert_close(paged["o_tensor"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(paged["lse_tensor"], lse_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.L0
@torch_fork_set_rng(seed=13)
@pytest.mark.parametrize("block, route", [(128, "m128"), (128, "m64"), (64, None)], ids=("blk128-m128", "blk128-m64", "blk64"))
@pytest.mark.parametrize("softmax_scale", [0.125, 0.0, -0.125])
def test_softmax_scale_sign_with_partial_blocks(block, route, softmax_scale):
    """Zero and negative scales with partially valid key blocks, forward (and the blk128 backward).

    max(scale * S) is not scale * max(S) for a negative scale, and a masked key's -inf times a zero or negative
    scale is NaN / +inf; the blk128 M64 route also shares its first softmax step with the later ones.  ``m64`` gives
    every query block 64 valid rows (q_block_sizes), so all of them take the M64 x N256 kernel.
    """
    BSA = import_kpage_bsa(KPAGE)
    heads, num_blocks = 2, 4
    seqlen = num_blocks * block
    bs = torch.tensor([block, 40, block, 7], dtype=torch.int32, device="cuda")
    q_sizes = None if route is None else torch.full((num_blocks,), 64 if route == "m64" else block, dtype=torch.int32, device="cuda")
    q, k, v, do = _inputs(heads, bs, block=block, pad_q=False)
    kv_valid = valid_rows(bs, block)
    k[0, :, ~kv_valid] = v[0, :, ~kv_valid] = 1000.0  # masked keys may hold any finite value
    q_valid = valid_rows(q_sizes, block) if q_sizes is not None else torch.ones(seqlen, dtype=torch.bool, device="cuda")
    do[0, :, ~q_valid] = 0.0  # padding query rows carry dO = 0
    q2k = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(1, 1, 1, num_blocks).expand(1, heads, num_blocks, num_blocks).contiguous()
    layout = BSA.kpage_prepare_layout(q2k, num_blocks, bs, q_block_sizes=q_sizes, sparse_block_size=block)
    mask = block_sparse_mask(q2k, num_blocks, bs, seqlen, seqlen, block)
    backward = block == 128
    o_ref, lse_ref, *grads_ref = attention_reference(q, k, v, mask, softmax_scale=softmax_scale, do=do if backward else None)

    kwargs = dict(kpage=layout, softmax_scale=softmax_scale, sparse_block_size=block)
    forward = BSA.block_sparse_attention_forward(q, k, v, q2k, num_blocks, bs, **kwargs)
    _check(forward["o_tensor"], o_ref, q_valid, tol=2e-2)
    torch.testing.assert_close(forward["lse_tensor"][0, :, q_valid], lse_ref[0, :, q_valid], atol=1e-3, rtol=1e-3)
    if not backward:
        return

    gradients = BSA.block_sparse_attention_backward(do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, num_blocks, bs, **kwargs)
    for name, expected, valid in zip(("dq_tensor", "dk_tensor", "dv_tensor"), grads_ref, (q_valid, kv_valid, kv_valid)):
        _check(gradients[name], expected, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=18)
@with_block
def test_empty_query_blocks(block):
    """allow_empty_block_nums: a query block with no KV blocks gets O = 0, LSE = -inf and dQ = 0, as on the base path;
    a layout built without the flag rejects the metadata.  blk128 also pads the query side (q_block_sizes)."""
    BSA = import_kpage_bsa(KPAGE)
    heads, num_blocks = 2, 6
    seqlen = num_blocks * block
    bs = torch.tensor({128: [128, 40, 128, 7, 90, 128], 64: [64, 40, 64, 7, 64, 50]}[block], dtype=torch.int32, device="cuda")
    q_sizes = bs if block == 128 else None
    valid = valid_rows(bs, block) if q_sizes is not None else torch.ones(seqlen, dtype=torch.bool, device="cuda")
    q, k, v, do = _inputs(heads, bs, 1000.0, block=block, pad_q=q_sizes is not None)
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    nums[0, 0, 0] = 0  # a full block (blk128: M128 route)
    nums[0, 1, 3] = 0  # a 7-row block (blk128: M64 route)
    nums[0, 1, 4] = 0
    empty = torch.zeros(heads, seqlen, dtype=torch.bool, device="cuda")
    empty[0, :block] = empty[1, 3 * block : 5 * block] = True

    with pytest.raises(ValueError, match="at least one KV block"):
        BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=q_sizes, sparse_block_size=block)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=q_sizes, sparse_block_size=block, allow_empty_block_nums=True)
    mask = block_sparse_mask(q2k, 0, bs, seqlen, seqlen, block, q2k_block_nums=nums)
    do_ref = do.clone()
    do_ref[0, :, ~valid] = 0
    o_ref, lse_ref, *grads_ref = attention_reference(q, k, v, mask, do=do_ref)
    # kpage=True builds the BSA-contract layout, which has no query padding.
    for kpage in (layout,) if q_sizes is not None else (layout, True):
        forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=kpage, allow_empty_block_nums=True, sparse_block_size=block)
        _check(forward["o_tensor"], o_ref, valid, tol=2e-2)
        o, lse = forward["o_tensor"][0], forward["lse_tensor"][0]
        assert torch.equal(o[empty], torch.zeros_like(o[empty]))
        assert torch.isneginf(lse[empty & valid]).all() and torch.equal(lse[:, ~valid], torch.zeros_like(lse[:, ~valid]))
        torch.testing.assert_close(lse[~empty & valid], lse_ref[0][~empty & valid], atol=1e-2, rtol=1e-2)
    if block == 64:
        return

    gradients = BSA.block_sparse_attention_backward(
        do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=block
    )
    for name, expected in zip(("dq_tensor", "dk_tensor", "dv_tensor"), grads_ref):
        _check(gradients[name], expected, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=12)
@with_block
def test_layout_reuse_and_preallocated_outputs(block):
    """One layout serves new activations; it equals building one on the call (kpage=True).  blk128 also pads the query
    side and writes its gradients into caller-provided tensors."""
    BSA = import_kpage_bsa(KPAGE)
    heads, num_blocks = 3, 6
    bs = torch.tensor({128: [128, 40, 128, 7, 90, 128], 64: [64, 40, 64, 7, 25, 64]}[block], dtype=torch.int32, device="cuda")
    q_sizes = bs if block == 128 else None
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=q_sizes, sparse_block_size=block)
    assert isinstance(layout, BSA.KPageLayout) and layout.sparse_block_size == block
    valid = valid_rows(bs, block) if q_sizes is not None else torch.ones(num_blocks * block, dtype=torch.bool, device="cuda")
    for _ in range(2):  # same layout, new activations
        q, k, v, do = _inputs(heads, bs, block=block, pad_q=q_sizes is not None)
        mask = block_sparse_mask(q2k, 0, bs, num_blocks * block, num_blocks * block, block, q2k_block_nums=nums)
        o_ref, _, *grads_ref = attention_reference(q, k, v, mask, do=do if block == 128 else None)
        forward = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=block)
        _check(forward["o_tensor"], o_ref, valid, tol=2e-2)
        if block == 64:
            built = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=True, sparse_block_size=block)
            assert torch.equal(forward["o_tensor"], built["o_tensor"]) and torch.equal(forward["lse_tensor"], built["lse_tensor"])
            continue
        outputs = dict(dq_tensor=torch.empty_like(q), dk_tensor=torch.empty_like(k), dv_tensor=torch.empty_like(v))
        gradients = BSA.block_sparse_attention_backward(
            do, q, k, v, forward["o_tensor"], forward["lse_tensor"], q2k, None, bs, nums, **outputs, kpage=layout, sparse_block_size=block
        )
        for (name, given), expected in zip(outputs.items(), grads_ref):
            assert gradients[name].data_ptr() == given.data_ptr()
            _check(given, expected, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=19)
@with_block
def test_bshd_layout_matches_bhsd(block):
    """layout='bshd' runs on views of the BSHD storage (batch 1) and returns BSHD o (and dq/dk/dv) equal to BHSD."""
    BSA = import_kpage_bsa(KPAGE)
    heads, num_blocks = 3, 6
    seqlen = num_blocks * block
    bs = torch.tensor({128: [128, 40, 128, 7, 90, 64], 64: [64, 40, 64, 7, 64, 50]}[block], dtype=torch.int32, device="cuda")
    q_sizes = bs if block == 128 else None
    q, k, v, do = _inputs(heads, bs, 1000.0, block=block, pad_q=q_sizes is not None)
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    layout = BSA.kpage_prepare_layout(q2k, None, bs, nums, q_block_sizes=q_sizes, with_backward=block == 128, sparse_block_size=block)

    ref_fwd = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, nums, kpage=layout, sparse_block_size=block)
    q_s, k_s, v_s, do_s = (t.transpose(1, 2).contiguous() for t in (q, k, v, do))
    fwd = BSA.block_sparse_attention_forward(q_s, k_s, v_s, q2k, None, bs, nums, layout="bshd", kpage=layout, sparse_block_size=block)
    assert fwd["o_tensor"].shape == q_s.shape and fwd["o_tensor"].is_contiguous() and fwd["lse_tensor"].shape == (1, heads, seqlen)
    assert torch.equal(fwd["o_tensor"].transpose(1, 2), ref_fwd["o_tensor"]) and torch.equal(fwd["lse_tensor"], ref_fwd["lse_tensor"])
    if block == 64:
        return

    ref_bwd = BSA.block_sparse_attention_backward(
        do, q, k, v, ref_fwd["o_tensor"], ref_fwd["lse_tensor"], q2k, None, bs, nums, kpage=layout, sparse_block_size=block
    )
    dq_s = torch.empty_like(q_s)
    bwd = BSA.block_sparse_attention_backward(
        do_s, q_s, k_s, v_s, fwd["o_tensor"], fwd["lse_tensor"], q2k, None, bs, nums, layout="bshd", kpage=layout, dq_tensor=dq_s, sparse_block_size=block
    )
    assert bwd["dq_tensor"] is dq_s and bwd["dk_tensor"].is_contiguous()
    for name in ("dq_tensor", "dk_tensor", "dv_tensor"):
        torch.testing.assert_close(bwd[name].transpose(1, 2), ref_bwd[name], atol=1e-2, rtol=1e-2)


@pytest.mark.L0
def test_rejects_unsupported_configurations():
    BSA = import_kpage_bsa(KPAGE)

    # -- blk128
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

    # -- blk64: forward only, MHA, BF16 head_dim 128, sequences that are multiples of 64
    heads, num_blocks = 2, 4
    seqlen = num_blocks * 64
    q = torch.randn((1, heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q2k = torch.tensor([0, 2], dtype=torch.int32, device="cuda").view(1, 1, 1, 2).expand(1, heads, num_blocks, 2).contiguous()
    bs = torch.full((num_blocks,), 64, dtype=torch.int32, device="cuda")

    # forward only
    o, lse = BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, sparse_block_size=64, kpage=True).values()
    with pytest.raises(NotImplementedError, match="forward only"):
        BSA.block_sparse_attention_backward(torch.randn_like(q), q, k, v, o, lse, q2k, 2, bs, sparse_block_size=64, kpage=True)
    with pytest.raises(NotImplementedError, match="forward only"):
        BSA.kpage_prepare_layout(q2k, 2, bs, sparse_block_size=64, with_backward=True)
    with pytest.raises(NotImplementedError, match="forward only"):
        BSA.kpage_prepare_layout(q2k, 2, bs, q_block_sizes=bs, sparse_block_size=64)
    with pytest.raises(NotImplementedError, match="bwd_kv_groups"):
        BSA.kpage_prepare_layout(q2k, 2, bs, sparse_block_size=64, bwd_kv_groups=2)
    # MHA only
    q_gqa = torch.randn((1, 2 * heads, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    q2k_gqa = q2k.repeat_interleave(2, dim=1).contiguous()
    with pytest.raises(NotImplementedError, match="MHA"):
        BSA.block_sparse_attention_forward(q_gqa, k, v, q2k_gqa, 2, bs, sparse_block_size=64, kpage=True)
    # BF16 head_dim 128 (the base blk64 contract)
    with pytest.raises(NotImplementedError, match="BF16"):
        BSA.block_sparse_attention_forward(q.half(), k.half(), v.half(), q2k, 2, bs, sparse_block_size=64, kpage=True)
    # seqlen_q and seqlen_k must be multiples of 64
    q_tail = q[:, :, : seqlen - 16].contiguous()
    with pytest.raises(NotImplementedError, match="multiples? of 64"):
        BSA.block_sparse_attention_forward(q_tail, k, v, q2k, 2, bs, sparse_block_size=64, kpage=True)
    # metadata contract, with the blk64 block size
    with pytest.raises(ValueError, match="block_sizes"):
        BSA.kpage_prepare_layout(q2k, 2, torch.full((num_blocks,), 100, dtype=torch.int32, device="cuda"), sparse_block_size=64)
    q2k_128 = torch.tensor([0, 1], dtype=torch.int32, device="cuda").view(1, 1, 1, 2).expand(1, heads, 2, 2).contiguous()
    # a layout is bound to its sparse_block_size
    layout_128 = BSA.kpage_prepare_layout(q2k_128, 2, torch.full((2,), 128, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="sparse_block_size"):
        BSA.block_sparse_attention_forward(q, k, v, q2k, None, bs, sparse_block_size=64, kpage=layout_128)


# ---- blk128 only: the query-side padding contract, tails, batch/GQA and the backward


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
def test_q_padding_matches_reference(block_sizes, pad_fill):
    """q_block_sizes=block_sizes: padding rows of q/k/v/do may hold any finite value; valid rows are exact, padding rows zero."""
    BSA = import_kpage_bsa(KPAGE)
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
    o_ref, lse_ref, dq_ref, dk_ref, dv_ref = attention_reference(q, k, v, mask, do=do_ref)

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
def test_seqlen_not_multiple_of_128(with_block_sizes, seqlen_q, seqlen_k):
    """Rows past seqlen_q / seqlen_k are padding of the final block, with or without block_sizes.

    The base path needs block_sizes for a partial final KV block (its contract); kpage derives the tail itself.
    With several heads, a tail of at most 64 rows on the query side makes the M64 work IDs (stride 2 * num_q_blocks)
    differ from ceil(seqlen_q / 64) unless the scheduler takes the stride from the metadata.  SM90 runs blk128 KPage on
    the metadata's 64-row halves and takes sequences that are multiples of 64 only.
    """
    BSA = import_kpage_bsa(KPAGE)
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
    o_ref, _, dq_ref, dk_ref, dv_ref = attention_reference(q, k, v, mask, do=do)
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
@torch_fork_set_rng(seed=19)
def test_empty_query_blocks_ignore_stale_tmem():
    """An empty tile on the M64 route must not read the O accumulators, which it never wrote.

    Base forwards on NaN inputs first leave NaN in the tensor memory of every SM; 0 * NaN would then turn the
    O of the empty query blocks into NaN instead of 0.
    """
    BSA = import_kpage_bsa(KPAGE)
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
def test_backward_accepts_base_forward_output():
    """With full blocks the base forward and the kpage backward compose."""
    BSA = import_kpage_bsa(KPAGE)
    heads, num_blocks = 2, 4
    seqlen = num_blocks * BLOCK
    bs = torch.full((num_blocks,), BLOCK, dtype=torch.int32, device="cuda")
    q, k, v, do = _inputs(heads, bs)
    q2k, nums = random_metadata(heads, num_blocks, num_blocks, kmax=3)
    mask = block_sparse_mask(q2k, 0, bs, seqlen, seqlen, BLOCK, q2k_block_nums=nums)
    _, _, dq_ref, dk_ref, dv_ref = attention_reference(q, k, v, mask, do=do)
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
@torch_fork_set_rng(seed=16)
def test_batched_matches_per_sample_reference():
    """batch > 1 runs as batch x heads; every sample has its own sparse pattern, block_sizes is shared."""
    BSA = import_kpage_bsa(KPAGE)
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
        o_ref, _, dq_ref, dk_ref, dv_ref = attention_reference(q[sl], k[sl], v[sl], mask, do=do_ref[sl])
        _check(forward["o_tensor"][sl], o_ref, valid, tol=2e-2)
        _check(backward["dq_tensor"][sl], dq_ref, valid, tol=3e-2)
        _check(backward["dk_tensor"][sl], dk_ref, valid, tol=3e-2)
        _check(backward["dv_tensor"][sl], dv_ref, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=17)
@pytest.mark.parametrize("batch, heads_kv", [(1, 1), (1, 2), (2, 2)])
def test_gqa_forward_backward(batch, heads_kv):
    """GQA/MQA: the q heads of one kv head read the same K/V pages and sum into one dK/dV."""
    BSA = import_kpage_bsa(KPAGE)
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
        o_ref, _, dq_ref, dk_ref, dv_ref = attention_reference(q[sl], k[sl], v[sl], mask, do=do_ref[sl])
        _check(forward["o_tensor"][sl], o_ref, valid, tol=2e-2)
        _check(backward["dq_tensor"][sl], dq_ref, valid, tol=3e-2)
        _check(backward["dk_tensor"][sl], dk_ref, valid, tol=3e-2)
        _check(backward["dv_tensor"][sl], dv_ref, valid, tol=3e-2)


@pytest.mark.L0
@torch_fork_set_rng(seed=15)
def test_compile_cache_is_shape_independent():
    """A second problem with other heads / sequence length reuses the compiled kernels."""
    BSA = import_kpage_bsa(KPAGE)
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
