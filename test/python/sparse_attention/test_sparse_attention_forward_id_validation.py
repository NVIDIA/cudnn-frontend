# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage-native id validation: out-of-range and cross-batch-aliasing ids.

Adapted from GitHub issue NVIDIA/cudnn-frontend#550 ("[Bug]: DSA indexer
backward: with topk_indices_global=False and B > 1, a positive out-of-range
local top-k id (id >= S_k) can alias a later batch's KV instead of being
rejected"). That issue is scoped to the *backward* indexer
(``python/cudnn/deepseek_sparse_attention/indexer_backward/indexer_backward_sm100.py``):
a positive local id was offset into the flattened ``(B * S_k, D)`` storage
*before* being bounds-checked, and the bounds check then only compared
against the *global* flat extent rather than the per-batch one, so a local
id in ``[S_k, (B-b)*S_k)`` (batch ``b``) silently aliased a later batch's
dK/dQ rows instead of being rejected. Root-cause detail (from the issue):
negative (``-1``) padding is unaffected -- it never receives the batch
offset and is always rejected by the ``>= 0`` guard -- only *positive*
overflow ids alias.

This file is the analogous regression coverage for the *forward* op's
storage-native id contract (``python/cudnn/sparse_attention/fwd/api.py``,
BSHD: within-sequence ids in ``[0, S_kv)``; THD: global flat ids in
``[0, T_kv)``). It targets two layers:

1. End-to-end, on the actual dispatched SM100 GQA-substrate device kernel
   (``cudnn.sparse_attention.fwd.sm100_gqa``, the only kernel currently
   registered for BSHD + granularity-4/64/128) -- the layer that would
   reproduce #550's failure mode if this op's addressing had the same bug.
2. Directly on the granularity/gather-address arithmetic that both the
   registered kernel (``_common_sm100.resolve_entry_window``, no host
   mirror -- exercised only indirectly via (1)) and the not-yet-integrated
   round-2 gather-address module
   (``sa_fwd_sm100_gqa.GranularityGatherAddr.gather_row_global`` /
   ``entry_token_window``, which *do* ship pure-Python host mirrors
   -- ``host_gather_row_global`` / ``host_entry_token_window`` -- for
   exactly this kind of GPU-free arithmetic check) are built on.

Empirical finding recorded by ``test_gather_row_global_clamp_computes_aliased_address_but_relies_on_validity_flag``
below: ``GranularityGatherAddr.gather_row_global``'s ``total_kv_rows``
defensive clamp does **not** prevent the *address* it computes for an
out-of-range entry from landing exactly on a later batch's KV row -- e.g.
for ``B=2``, ``S_kv=64``, batch 0, an entry one block past the per-batch
bound resolves to ``row_global == S_kv``, i.e. batch 1's row 0. The clamp's
only job is keeping that address inside the flattened tensor's *global*
extent (mirroring #550's suggested-fix framing: bounding against the global
extent is not the same as bounding against the per-batch extent). Safety
against actually *using* that aliased data depends entirely on the
``is_valid``/``token_valid`` flag returned alongside it -- callers must gate
on it, the address alone does not encode validity. This is consistent with
this codebase's oracle (``sparse_attention_reference.py``: computes
``gather_ids`` the same clamped-but-possibly-aliased way, then masks via
``token_valid`` before the softmax) but is a *different, safer* strategy
than the currently-registered device kernel's own mainloop
(``gqa_prefill_bf16_sm100.py``), which never computes the address at all
for an invalid window (Python-level ``if is_valid:`` skips the loop body
entirely) -- see that test's assertions for why both are checked.
"""

from __future__ import annotations

import pytest
import torch

from test_utils import torch_fork_set_rng

pytestmark = pytest.mark.L0


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 GPU required")


def _wrapper():
    from cudnn.sparse_attention import sparse_attention_forward_wrapper

    return sparse_attention_forward_wrapper


def _gqa_kernel_available():
    from cudnn.sparse_attention.fwd.sm100_gqa import sparse_attention_forward_wrapper  # noqa: F401

    return True


def _require_gqa_kernel():
    try:
        _gqa_kernel_available()
    except ImportError:
        pytest.skip("SM100 GQA-substrate sparse-attention kernel module not present in this tree")


from sparse_attention.sparse_attention_reference import reference_sparse_attention_forward  # noqa: E402


# ---------------------------------------------------------------------------
# Shared small BSHD/THD fixtures: B=2, H_kv=2 > 1, H_q=4, granularity=4 (the
# GQA-substrate envelope: G == H_kv, granularity in (4, 64, 128), BF16).
# ---------------------------------------------------------------------------
_B, _S_Q, _S_KV, _H_KV, _H_Q, _D, _G, _TOPK_MAX = 2, 2, 64, 2, 4, 32, 4, 8
_N_BLOCKS = _S_KV // _G  # 16: valid local entry ids are in [0, 16)


def _mk_bshd_inputs(device):
    dtype = torch.bfloat16
    q = (torch.randn(_B, _S_Q, _H_Q, _D, device=device) * 0.1).to(dtype)
    k = (torch.randn(_B, _S_KV, _H_KV, _D, device=device) * 0.1).to(dtype)
    v = (torch.randn(_B, _S_KV, _H_KV, _D, device=device) * 0.1).to(dtype)
    return q, k, v


def _mk_bshd_idxs_with_batch0_oor(device, oor_val):
    """Batch 0: 4 valid random block ids + 1 out-of-range slot (``oor_val``,
    a *local* block id relative to ``_S_KV``). Batch 1: entirely ``-1``
    (dead rows) -- so any contamination from batch 0's out-of-range slot is
    visible as a non-dead batch-1 row."""
    idxs = torch.full((_B, _S_Q, _H_KV, _TOPK_MAX), -1, dtype=torch.int32, device=device)
    for s in range(_S_Q):
        for h in range(_H_KV):
            idxs[0, s, h, :4] = torch.randperm(_N_BLOCKS, device=device)[:4].to(torch.int32)
            idxs[0, s, h, 4] = oor_val
    return idxs


# ---------------------------------------------------------------------------
# (1) BSHD: one out-of-range id exactly at the per-batch bound; batch 1
#     entirely -1-padded must stay untouched.
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=550)
def test_bshd_out_of_range_id_does_not_alias_next_batch():
    _require_sm100()
    _require_gqa_kernel()
    device = "cuda"
    q, k, v = _mk_bshd_inputs(device)
    # Smallest out-of-range value: one past the last valid local block id
    # (== #550's minimal reproducer, "raw_id == S_k", translated to block
    # units: entry == S_kv // granularity).
    idxs = _mk_bshd_idxs_with_batch0_oor(device, _N_BLOCKS)

    res = _wrapper()(q, k, v, idxs, index_granularity=_G)
    out, lse = res["out"], res["lse"]

    # Batch 1 (entirely -1-padded) must be exactly the dead-row sentinel:
    # not "safely clamped to something small" but untouched at all.
    assert torch.isneginf(lse[1]).all(), "batch 1 LSE must stay -inf (dead-row sentinel), not contaminated"
    assert (out[1] == 0).all(), "batch 1 output must stay exactly 0 (dead-row sentinel), not contaminated"

    # Batch 0's own row: the out-of-range slot must not raise, and must not
    # silently change batch 0's answer versus simply dropping that slot.
    # The oracle applies the same storage-native contract (an out-of-range
    # entry contributes nothing -- see sparse_attention_reference.py's
    # token_valid masking), so device-vs-oracle agreement here also confirms
    # batch 0's row is unaffected by its own out-of-range slot.
    ref_out, ref_lse = reference_sparse_attention_forward(q, k, v, idxs, index_granularity=_G)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# (2) BSHD: sweep the aliasing window from #550's root-cause section,
#     translated to block units -- local entry ids in
#     [S_kv/g, (B-b)*S_kv/g) for batch b=0 (i.e. [16, 32)), plus values
#     beyond the *global* flat extent for completeness. None may perturb
#     batch 1.
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=551)
@pytest.mark.parametrize(
    "oor_val",
    [
        _N_BLOCKS,  # exactly one past the per-batch bound (boundary)
        _N_BLOCKS + 1,
        2 * _N_BLOCKS - 1,  # last id in the #550-style aliasing window (would land on batch 1's last block)
        2 * _N_BLOCKS,  # exactly at the *global* flat extent (out of range even for the whole tensor)
        2 * _N_BLOCKS + 5,  # past the global extent entirely
        10_000,  # pathologically large
    ],
)
def test_bshd_aliasing_window_sweep_never_perturbs_later_batch(oor_val):
    _require_sm100()
    _require_gqa_kernel()
    device = "cuda"
    q, k, v = _mk_bshd_inputs(device)
    idxs = _mk_bshd_idxs_with_batch0_oor(device, oor_val)

    res = _wrapper()(q, k, v, idxs, index_granularity=_G)
    out, lse = res["out"], res["lse"]

    assert torch.isneginf(lse[1]).all(), f"oor_val={oor_val}: batch 1 LSE contaminated"
    assert (out[1] == 0).all(), f"oor_val={oor_val}: batch 1 output contaminated"

    ref_out, ref_lse = reference_sparse_attention_forward(q, k, v, idxs, index_granularity=_G)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# (3) THD counterpart: a single flat bound, no per-batch aliasing surface,
#     but the same degenerate "id >= bound" case must still be rejected
#     safely (no crash, no silent corruption, matches the oracle).
# ---------------------------------------------------------------------------
@torch_fork_set_rng(seed=552)
@pytest.mark.parametrize("oor_val", [_N_BLOCKS, _N_BLOCKS + 3, 10_000])
def test_thd_out_of_range_global_id_rejected_safely(oor_val):
    _require_sm100()
    _require_gqa_kernel()
    device = "cuda"
    dtype = torch.bfloat16
    t_q, t_kv, h_kv, h_q, d, g = 4, _S_KV, _H_KV, _H_Q, _D, _G
    n_blocks = t_kv // g

    q = (torch.randn(t_q, h_q, d, device=device) * 0.1).to(dtype)
    k = (torch.randn(t_kv, h_kv, d, device=device) * 0.1).to(dtype)
    v = (torch.randn(t_kv, h_kv, d, device=device) * 0.1).to(dtype)
    cu = torch.tensor([0, t_q], dtype=torch.int32, device=device)

    idxs = torch.full((t_q, h_kv, _TOPK_MAX), -1, dtype=torch.int32, device=device)
    for t in range(t_q):
        for h in range(h_kv):
            idxs[t, h, :4] = torch.randperm(n_blocks, device=device)[:4].to(torch.int32)
            idxs[t, h, 4] = oor_val

    res = _wrapper()(q, k, v, idxs, index_granularity=g, cu_seqlens_q=cu)
    ref_out, ref_lse = reference_sparse_attention_forward(q, k, v, idxs, index_granularity=g)

    torch.testing.assert_close(res["out"].float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(res["lse"], ref_lse, atol=1e-3, rtol=1e-3)
    # No row is dead-by-construction here (every row has 4 valid entries),
    # so nothing should be -inf/0 -- confirms the out-of-range slot was
    # dropped, not that the whole row got wrongly killed.
    assert not torch.isneginf(res["lse"]).any()


# ---------------------------------------------------------------------------
# (4) Host-side arithmetic: GranularityGatherAddr.gather_row_global /
#     entry_token_window (via their documented pure-Python mirrors
#     host_gather_row_global / host_entry_token_window -- exact, line-for-
#     line copies used specifically so this arithmetic is testable without a
#     GPU or the CuTe DSL toolchain). No CUDA required.
# ---------------------------------------------------------------------------
def _host_mirrors():
    from cudnn.sparse_attention.fwd.sm100_gqa.sa_fwd_sm100_gqa import (
        host_entry_token_window,
        host_gather_row_global,
    )

    return host_entry_token_window, host_gather_row_global


def test_entry_token_window_rejects_entry_at_per_batch_bound():
    """The upstream guard (``entry_token_window``'s ``is_valid``) already
    catches a local id at/after the per-batch ``kv_bound`` -- this is what
    lets the registered device kernel skip the gather entirely for such an
    entry (see module docstring)."""
    host_entry_token_window, _ = _host_mirrors()
    g, s_kv = 4, 64
    n_blocks = s_kv // g

    # In range: last valid entry.
    tile_start, num_valid, is_valid = host_entry_token_window(n_blocks - 1, g, s_kv)
    assert is_valid and num_valid == g

    # Exactly at the bound (#550's minimal case, translated to block units).
    tile_start, num_valid, is_valid = host_entry_token_window(n_blocks, g, s_kv)
    assert not is_valid
    assert num_valid == 0

    # Deep inside the #550-style aliasing window and beyond the global extent.
    for oor in (n_blocks + 1, 2 * n_blocks - 1, 2 * n_blocks, 10_000):
        _, num_valid, is_valid = host_entry_token_window(oor, g, s_kv)
        assert not is_valid, f"entry {oor} (kv_bound={s_kv}) must be rejected"
        assert num_valid == 0


def test_gather_row_global_clamp_computes_aliased_address_but_relies_on_validity_flag():
    """Empirical check of the docstring's claim: does the ``total_kv_rows``
    defensive clamp in ``gather_row_global`` prevent a BSHD out-of-range
    local id from *computing* an address that lands on a later batch's KV
    row?

    Observed answer: **no** -- it only bounds the address against the
    tensor's *global* flat extent (exactly the #550 root-cause pattern: a
    guard against the global extent is not a guard against the per-batch
    extent). For batch 0 (``kv_base = 0``) of a ``B=2``, ``S_kv=64`` layout,
    a local entry exactly at the per-batch bound resolves to
    ``row_global == S_kv``, i.e. batch 1's row 0 in the flattened storage --
    a real alias, not a clamp to some safe dummy row inside batch 0. Safety
    is provided entirely by the accompanying ``token_valid`` flag (``False``
    here), which the caller must check before using the gathered data --
    the address itself does not encode validity. This mirrors how
    ``sparse_attention_reference.py``'s oracle handles the same case
    (computes ``gather_ids`` unconditionally, masks via ``token_valid``
    before the softmax) -- unlike the currently-registered device kernel
    (``gqa_prefill_bf16_sm100.py``), which never computes the address at
    all for an invalid window (see ``test_bshd_out_of_range_id_does_not_alias_next_batch``).
    """
    _, host_gather_row_global = _host_mirrors()
    g, s_kv, b = 4, 64, 2
    n_blocks = s_kv // g
    total_kv_rows = b * s_kv

    # Batch 0, local entry exactly at the per-batch bound.
    row_global, token_valid = host_gather_row_global(
        entry_idx=n_blocks,
        granularity=g,
        row_in_tile=0,
        kv_base=0,  # batch 0
        kv_bound=s_kv,
        total_kv_rows=total_kv_rows,
    )
    assert row_global == s_kv, "expected the computed address to land exactly on batch 1's row 0 (the alias)"
    assert token_valid is False, "the alias must be marked invalid -- callers must gate on this, not the address"

    # Sweep the rest of the #550-style aliasing window for batch 0: every
    # computed address either aliases a later batch's row or is clamped to
    # the last row of the flat tensor, but token_valid is False throughout.
    for local_entry in range(n_blocks, 2 * n_blocks):
        row_global, token_valid = host_gather_row_global(
            entry_idx=local_entry,
            granularity=g,
            row_in_tile=0,
            kv_base=0,
            kv_bound=s_kv,
            total_kv_rows=total_kv_rows,
        )
        assert token_valid is False, f"local_entry={local_entry}: must be marked invalid"
        assert 0 <= row_global < total_kv_rows, f"local_entry={local_entry}: address must still be in-bounds for the whole tensor"

    # A pathologically large entry_idx: the total_kv_rows clamp keeps the
    # address in-bounds for the flat tensor (its documented purpose), still
    # gated by token_valid == False.
    row_global, token_valid = host_gather_row_global(
        entry_idx=10_000,
        granularity=g,
        row_in_tile=0,
        kv_base=0,
        kv_bound=s_kv,
        total_kv_rows=total_kv_rows,
    )
    assert row_global == total_kv_rows - 1
    assert token_valid is False


def test_gather_row_global_thd_single_flat_bound_degenerate_case():
    """THD has no per-batch aliasing surface (``kv_base == 0`` always), so
    the #550 pattern degenerates to a single flat-bound check -- included
    for completeness per the subtask. An id at/past the flat bound is
    rejected (``token_valid is False``) and the clamp keeps the computed
    address in-bounds."""
    _, host_gather_row_global = _host_mirrors()
    g, t_kv = 4, 64
    n_blocks = t_kv // g

    for oor in (n_blocks, n_blocks + 5, 10_000):
        row_global, token_valid = host_gather_row_global(
            entry_idx=oor,
            granularity=g,
            row_in_tile=0,
            kv_base=0,  # THD: always 0
            kv_bound=t_kv,
            total_kv_rows=t_kv,  # THD: total == the single flat bound
        )
        assert token_valid is False, f"entry {oor}: must be rejected"
        assert 0 <= row_global < t_kv, f"entry {oor}: clamp must keep the address in-bounds"
