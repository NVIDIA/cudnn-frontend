# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_select_decode``: the decode form of the block-sparse indexer's top-k block selection.

A decode step has one to a few query rows per sequence (the step-0 token plus speculative drafts); the selector
scores them with the dense DSA unified scorer at the smallest MMA tile and selects per row with the radix top-k
kernel, one visible-block count per row.  The reject / gate tests and the count-rule tests run anywhere (CPU or any
CUDA device: the declines fire before a kernel is compiled); the accept cells need an SM100-family GPU (the scorer
is a tcgen05 kernel) and compare

* the selected SETS against ``torch.topk`` of an fp64 re-derivation of the indexer score on the SAME bf16 inputs,
  up to ties at the k-th boundary (the sharp check), and
* the per-row visible-block COUNTS against the QSA oracle module's count rule (``full_block_ids``), exactly, and
* the shared (per-sequence) list against the step-0 row of the per-row form, and its visible set against the
  oracle's ``qsa_visible_mask`` with ``pos0`` -- the speculative-verify contract.
"""

import os
import sys

import pytest
import torch

import cudnn.gated_attention_block.qsa_select as qsa_mod
from cudnn.gated_attention_block.qsa_select import QSA_BLOCK_SIZE, QSA_TOP_K, TOP_K_KERNEL_MAX, decode_row_positions_and_counts, qsa_select, qsa_select_decode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import full_block_ids, qsa_visible_mask  # noqa: E402

pytestmark = pytest.mark.L0

_D = 128
_H = 4
_E4M3 = torch.float8_e4m3fn


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is needed for the input tensors")
needs_sm100_family = pytest.mark.skipif(_cc() is None or _cc()[0] != 10, reason=f"the scorer is a tcgen05 kernel (cc 10.x); found {_cc()}")


def _inputs(batch, rows_per_seq, n_blocks_max, device, *, seed=0, n_heads=_H):
    g = torch.Generator(device=device).manual_seed(seed)
    qi = torch.randn(batch, rows_per_seq, n_heads, _D, dtype=torch.bfloat16, device=device, generator=g)
    kbar = torch.randn(batch, n_blocks_max, _D, dtype=torch.bfloat16, device=device, generator=g)
    return qi, kbar


def _kv_lens(pos0_list, rows_per_seq, device):
    """``kv_lens = pos0 + rows_per_seq`` (the logical KV length including the step's rows)."""
    return torch.tensor([p + rows_per_seq for p in pos0_list], dtype=torch.int32, device=device)


def _decode_reference(qi, kbar, positions, block_size, scale):
    """fp64 dense scores per decode row: ``[B, rows, n_blocks_max]`` with ``-inf`` outside the row's visible blocks
    (``b < floor((pos + 1) / block_size)``) -- re-derived from the module docstring, independent of the DSA tree."""
    batch, rows, _, _ = qi.shape
    n_blocks_max = kbar.shape[1]
    scores = torch.relu(torch.einsum("brhd,bnd->brhn", qi.double(), kbar.double())).sum(dim=2) * scale
    limit = torch.div(positions.to(torch.long) + 1, block_size, rounding_mode="floor").clamp_(min=0)
    visible = torch.arange(n_blocks_max, device=qi.device).view(1, 1, -1) < limit.view(batch, rows, 1)
    return scores.masked_fill(~visible, float("-inf"))


def _assert_rows(block_ids, scores, block_lens, ref, top_k, tie_tol=1e-4):
    """Per row of ``ref [rows, n_blocks_max]``: the valid count equals ``block_lens`` = ``min(top_k, visible)``, ids are
    unique local ids below the visible count, the scores are the reference scores of the selected ids (fp32 of an
    fp64 value), and the SET equals torch.topk's up to flips within ``tie_tol`` of the k-th reference score."""
    rows, n_blocks_max = ref.shape
    assert block_ids.shape == scores.shape == (rows, top_k)
    assert block_ids.dtype == torch.int32 and scores.dtype == torch.float32 and block_lens.dtype == torch.int32
    valid = block_ids >= 0
    visible = torch.isfinite(ref).sum(dim=-1)
    assert torch.equal(valid.sum(dim=-1).to(torch.int32), block_lens), "the valid prefix differs from block_lens"
    assert torch.equal(block_lens.long(), visible.clamp(max=top_k)), "block_lens differs from min(top_k, visible)"
    prefix = torch.arange(top_k, device=ref.device)[None, :] < block_lens[:, None].long()
    assert torch.equal(valid, prefix), "the valid ids are not a prefix followed by -1"
    assert bool(((block_ids < visible[:, None]) | ~valid).all()), "an id at or past the row's visible count"
    assert bool(torch.isneginf(scores[~valid]).all())
    row_idx = torch.arange(rows, device=ref.device).unsqueeze(1).expand_as(block_ids)
    gathered = ref[row_idx[valid], block_ids[valid].long()]
    torch.testing.assert_close(scores[valid].double(), gathered, atol=tie_tol, rtol=tie_tol)
    in_actual = torch.zeros_like(ref, dtype=torch.bool)
    in_actual[row_idx[valid], block_ids[valid].long()] = True
    assert torch.equal(in_actual.sum(dim=-1), valid.sum(dim=-1)), "a row lists a block twice"
    k_eff = min(top_k, n_blocks_max)
    ref_topk = torch.topk(ref, k_eff, dim=-1)
    exp_valid = torch.isfinite(ref_topk.values)
    in_expected = torch.zeros_like(in_actual)
    rows_k = torch.arange(rows, device=ref.device).unsqueeze(1).expand(-1, k_eff)
    in_expected[rows_k[exp_valid], ref_topk.indices[exp_valid]] = True
    kth = torch.where(exp_valid, ref_topk.values, torch.full_like(ref_topk.values, float("inf"))).min(dim=-1).values
    differs = in_actual ^ in_expected
    if bool(differs.any()):
        gap = (ref - kth.unsqueeze(1)).abs()[differs]
        assert bool((gap <= tie_tol).all()), f"{int(differs.sum())} ids differ from torch.topk beyond a tie (max gap {float(gap.max()):.3e})"


def _oracle_counts(positions, top_k, block_size):
    """The QSA oracle module's count rule: ``min(top_k, floor((p + 1) / block_size))`` per row (``full_block_ids``)."""
    _, lens = full_block_ids(positions.to(torch.long), top_k, block_size)
    return lens


# --------------------------------------------------------------------------------------------- the count rule (CPU)


@pytest.mark.parametrize("rows_per_seq", [1, 2, 3, 4, 5])
def test_decode_row_counts_equal_the_oracle_count_rule_per_draft_row(rows_per_seq):
    """Every residue of the step-0 position modulo the block size, kv lengths from the identity region to past it:
    the per-row counts equal the oracle's ``min(top_k, floor((p + 1) / 4))`` and the positions are ``pos0 + j``."""
    kv = torch.tensor([rows_per_seq, 5, 6, 7, 8, 2051, 2052, 2053, 2054, 2055, 65536, 131072], dtype=torch.int32)
    pos, counts = decode_row_positions_and_counts(kv, rows_per_seq, QSA_BLOCK_SIZE, 1 << 20, list_per_sequence=False)
    assert pos.shape == counts.shape == (kv.numel(), rows_per_seq) and counts.dtype == torch.int32
    assert torch.equal(pos, (kv - rows_per_seq)[:, None] + torch.arange(rows_per_seq, dtype=torch.int32)[None, :])
    want = torch.div(pos + 1, QSA_BLOCK_SIZE, rounding_mode="floor")
    assert torch.equal(counts, want.to(torch.int32))
    assert torch.equal(torch.minimum(counts, torch.full_like(counts, QSA_TOP_K)), _oracle_counts(pos, QSA_TOP_K, QSA_BLOCK_SIZE))
    # consecutive rows share a count except across a block boundary (block-granular visibility): the per-key draft
    # stagger of the radix kernel (one more key per row) would be wrong by up to block_size - 1 blocks on the early
    # rows.  Four consecutive positions hold exactly ONE completing position (p % 4 == 3); when it is the step-0
    # row itself the counts of the four rows are all equal.
    deltas = counts[:, 1:] - counts[:, :-1]
    assert bool(((deltas == 0) | (deltas == 1)).all()) and bool((counts[:, -1] - counts[:, 0] <= 1).all())
    if rows_per_seq == QSA_BLOCK_SIZE:
        assert bool(((pos % QSA_BLOCK_SIZE == QSA_BLOCK_SIZE - 1).sum(dim=1) == 1).all())
        assert torch.equal(counts[:, -1] - counts[:, 0], (pos[:, 0] % QSA_BLOCK_SIZE != QSA_BLOCK_SIZE - 1).to(torch.int32))
    shared_pos, shared_counts = decode_row_positions_and_counts(kv, rows_per_seq, QSA_BLOCK_SIZE, 1 << 20, list_per_sequence=True)
    assert torch.equal(shared_pos, pos[:, 0]) and torch.equal(shared_counts, counts[:, 0])


def test_decode_row_counts_are_clamped_to_the_cache_and_to_zero():
    kv = torch.tensor([1, 4, 4096], dtype=torch.int32)
    _, counts = decode_row_positions_and_counts(kv, 1, QSA_BLOCK_SIZE, 64, list_per_sequence=True)
    assert counts.tolist() == [0, 1, 64]
    # a (contract-violating) kv length shorter than the step still yields a non-negative count
    _, counts = decode_row_positions_and_counts(torch.tensor([1], dtype=torch.int32), 4, QSA_BLOCK_SIZE, 64, list_per_sequence=False)
    assert counts.tolist() == [[0, 0, 0, 0]]


# --------------------------------------------------------------------------------------------- rejects / gates


@needs_cuda
def test_decode_declines_top_k_past_the_radix_kernel_bound():
    """The radix top-k kernel serves ``0 < top_k <= 2048``; the decode selector refuses beyond it before any launch."""
    device = torch.device("cuda")
    qi, kbar = _inputs(2, 1, 8, device)
    kv = _kv_lens([3, 7], 1, device)
    with pytest.raises(ValueError, match=str(TOP_K_KERNEL_MAX)):
        qsa_select_decode(qi, kbar, kv, top_k=TOP_K_KERNEL_MAX + 1)
    with pytest.raises(ValueError, match="top_k"):
        qsa_select_decode(qi, kbar, kv, top_k=0)


@needs_cuda
def test_decode_rejects_malformed_inputs():
    device = torch.device("cuda")
    qi, kbar = _inputs(2, 4, 16, device)
    kv = _kv_lens([3, 7], 4, device)
    with pytest.raises(ValueError, match=r"\[B, rows_per_seq, n_heads, head_dim\]"):
        qsa_select_decode(qi[0], kbar, kv)
    with pytest.raises(ValueError, match="head_dim 128"):
        qsa_select_decode(qi[..., :64], kbar[..., :64], kv)
    with pytest.raises(ValueError, match="bfloat16"):
        qsa_select_decode(qi.half(), kbar, kv)
    with pytest.raises(ValueError, match="B = 2 sequences but kbar holds 1"):
        qsa_select_decode(qi, kbar[:1], kv)
    with pytest.raises(ValueError, match="multiple of 8"):
        qsa_select_decode(qi, kbar[:, :12].contiguous(), kv)
    with pytest.raises(ValueError, match="contiguous"):
        qsa_select_decode(qi, kbar.transpose(1, 2).contiguous().transpose(1, 2), kv)
    with pytest.raises(ValueError, match="unit last stride"):
        qsa_select_decode(qi.transpose(2, 3).contiguous().transpose(2, 3), kbar, kv)
    with pytest.raises(ValueError, match="kv_lens"):
        qsa_select_decode(qi, kbar, kv.to(torch.int64))
    with pytest.raises(ValueError, match="kv_lens"):
        qsa_select_decode(qi, kbar, kv.cpu())
    with pytest.raises(ValueError, match="kv_lens"):
        qsa_select_decode(qi, kbar, kv[:1])
    with pytest.raises(ValueError, match="tie_break"):
        qsa_select_decode(qi, kbar, kv, tie_break=3)
    with pytest.raises(ValueError, match="m_block_size"):
        qsa_select_decode(qi, kbar, kv, m_block_size=24)
    with pytest.raises(ValueError, match="block_size"):
        qsa_select_decode(qi, kbar, kv, block_size=0)


@needs_cuda
def test_decode_host_geometry_check_names_the_sequence():
    device = torch.device("cuda")
    qi, kbar = _inputs(2, 4, 16, device)
    kv = _kv_lens([13, 60], 4, device)  # pos0 17 and 64 -> kv 17 / 64 hold 4 / 16 complete blocks
    with pytest.raises(ValueError, match="sequence 1: a KV length of 64 holds"):
        qsa_select_decode(qi, kbar, kv, n_blocks_per_seq=[4, 15])
    with pytest.raises(ValueError, match="kbar holds 16 rows"):
        qsa_select_decode(qi, kbar, kv, n_blocks_per_seq=[4, 17])
    with pytest.raises(ValueError, match="entries for B = 2"):
        qsa_select_decode(qi, kbar, kv, n_blocks_per_seq=[4])
    with pytest.raises(ValueError, match="sequence 0: kv_lens = 2 is shorter than the 4 new rows"):
        qsa_select_decode(qi, kbar, torch.tensor([2, 64], dtype=torch.int32, device=device), n_blocks_per_seq=[0, 16])


@needs_cuda
def test_decode_declines_below_the_dsl_floor_by_version(monkeypatch):
    """Rule 7: the CuTe DSL gate runs before the scorer or the top-k module is imported and names the version."""
    device = torch.device("cuda")
    qi, kbar = _inputs(1, 1, 8, device)
    monkeypatch.setattr(qsa_mod, "cutedsl_requirement_error", lambda what: f"{what} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    with pytest.raises(NotImplementedError, match="qsa_select_decode requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        qsa_select_decode(qi, kbar, _kv_lens([5], 1, device))


@needs_cuda
def test_decode_declines_a_dsl_without_the_target(monkeypatch):
    device = torch.device("cuda")
    qi, kbar = _inputs(1, 1, 8, device)
    monkeypatch.setattr(
        qsa_mod, "cutedsl_arch_requirement_error", lambda cc: "SM107 requires a CuTe DSL build supporting sm_107a; found 4.7.0 without that target"
    )
    with pytest.raises(NotImplementedError, match="sm_107a"):
        qsa_select_decode(qi, kbar, _kv_lens([5], 1, device))


@needs_cuda
def test_decode_declines_devices_below_the_sm100_family(monkeypatch):
    device = torch.device("cuda")
    qi, kbar = _inputs(1, 1, 8, device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(NotImplementedError, match=r"cc 10\.x.*found cc 9\.0"):
        qsa_select_decode(qi, kbar, _kv_lens([5], 1, device))


@needs_cuda
@pytest.mark.parametrize("n_heads", [2, 3, 12, 128])
def test_decode_declines_unsupported_head_counts(n_heads, monkeypatch):
    """The head set is named after the DSL / device gates pass (the decline needs the scorer's own table); the
    device gate is satisfied here by reporting an SM100-family part, so the test runs on any CUDA device."""
    device = torch.device("cuda")
    qi, kbar = _inputs(1, 1, 8, device, n_heads=n_heads)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    monkeypatch.setattr(qsa_mod, "cutedsl_arch_requirement_error", lambda cc: None)
    with pytest.raises(NotImplementedError, match=r"n_heads in \(4, 8, 16, 32, 64\)"):
        qsa_select_decode(qi, kbar, _kv_lens([5], 1, device))


# --------------------------------------------------------------------------------------------- accept cells


@needs_sm100_family
@pytest.mark.parametrize("n_blocks_max", [8, 520, 4096])
@pytest.mark.parametrize("rows_per_seq", [1, 2, 3, 4])
def test_decode_per_row_lists_match_the_fp64_reference_and_the_oracle_counts(rows_per_seq, n_blocks_max):
    """Four sequences whose step-0 positions cover every residue modulo the block size; the per-row form: every row's
    SET equals torch.topk of the fp64 reference up to ties, and its count equals the oracle's count rule exactly
    (block-granular: rows across a block boundary gain exactly one block)."""
    device = torch.device("cuda")
    batch = 4
    base = QSA_BLOCK_SIZE * n_blocks_max - rows_per_seq - 3  # the last row of the batch sees every block
    pos0 = [max(0, base + r) for r in range(batch)]
    qi, kbar = _inputs(batch, rows_per_seq, n_blocks_max, device, seed=n_blocks_max + rows_per_seq)
    kv = _kv_lens(pos0, rows_per_seq, device)

    result = qsa_select_decode(qi, kbar, kv, list_per_sequence=False, n_blocks_per_seq=[n_blocks_max] * batch)
    torch.cuda.synchronize()

    positions = torch.tensor(pos0, device=device)[:, None] + torch.arange(rows_per_seq, device=device)[None, :]
    assert result["block_ids"].shape == (batch, rows_per_seq, QSA_TOP_K) and result["block_lens"].shape == (batch, rows_per_seq)
    assert torch.equal(result["block_lens"], _oracle_counts(positions, QSA_TOP_K, QSA_BLOCK_SIZE).to(torch.int32))
    ref = _decode_reference(qi, kbar, positions, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_rows(
        result["block_ids"].view(-1, QSA_TOP_K),
        result["scores"].view(-1, QSA_TOP_K),
        result["block_lens"].view(-1),
        ref.view(batch * rows_per_seq, n_blocks_max),
        QSA_TOP_K,
    )
    if n_blocks_max <= QSA_TOP_K:
        # the identity region: every visible block is selected -- the set equals the oracle's full list
        want_ids, _ = full_block_ids(positions.to(torch.long), QSA_TOP_K, QSA_BLOCK_SIZE)
        assert torch.equal(torch.sort(result["block_ids"], dim=-1).values, torch.sort(want_ids.to(device), dim=-1).values)


@needs_sm100_family
def test_decode_shared_list_is_the_step0_row_and_is_reused_by_the_verify_rows():
    """``list_per_sequence=True`` returns the step-0 row's list ([B, top_k]); it equals the per-row form's row 0 and
    an explicit one-row step at the same position.  Through the oracle's ``qsa_visible_mask`` with ``pos0`` the
    verify rows see the listed blocks plus every token from the step-0 tail start to their own position -- the block
    completed between the step-0 position and a draft row stays visible although the shared list cannot hold it."""
    device = torch.device("cuda")
    rows_per_seq, n_blocks_max = 4, 1024
    pos0 = [4 * 600 + 2, 4 * 700 + 3, 4 * 800, 4 * 900 + 1]  # pos0 % 4 = 2, 3, 0, 1
    batch = len(pos0)
    qi, kbar = _inputs(batch, rows_per_seq, n_blocks_max, device, seed=23)
    kv = _kv_lens(pos0, rows_per_seq, device)

    shared = qsa_select_decode(qi, kbar, kv, list_per_sequence=True)
    per_row = qsa_select_decode(qi, kbar, kv, list_per_sequence=False)
    single = qsa_select_decode(qi[:, :1].contiguous(), kbar, torch.tensor([p + 1 for p in pos0], dtype=torch.int32, device=device), list_per_sequence=True)
    torch.cuda.synchronize()

    assert shared["block_ids"].shape == (batch, QSA_TOP_K) and shared["block_lens"].shape == (batch,)
    for other in (per_row["block_ids"][:, 0], single["block_ids"]):
        assert torch.equal(torch.sort(shared["block_ids"], dim=-1).values, torch.sort(other, dim=-1).values)
    assert torch.equal(shared["block_lens"], per_row["block_lens"][:, 0]) and torch.equal(shared["block_lens"], single["block_lens"])
    positions0 = torch.tensor(pos0, device=device)
    assert torch.equal(shared["block_lens"], _oracle_counts(positions0, QSA_TOP_K, QSA_BLOCK_SIZE).to(torch.int32))
    ref0 = _decode_reference(qi[:, :1], kbar, positions0[:, None], QSA_BLOCK_SIZE, _D**-0.5)
    _assert_rows(shared["block_ids"], shared["scores"], shared["block_lens"], ref0.view(batch, n_blocks_max), QSA_TOP_K)

    # the speculative-verify contract on the selector's output, per sequence: the oracle's visible set of row j
    # under the shared list (pos0 = the step-0 position) is a superset of row 0's, grows only by the open tail
    # tokens up to the row's own position, and contains the block completed between pos0 and pos_j
    s_kv = QSA_BLOCK_SIZE * n_blocks_max + rows_per_seq
    for b in range(batch):
        ids = shared["block_ids"][b : b + 1].expand(rows_per_seq, -1)
        lens = shared["block_lens"][b : b + 1].expand(rows_per_seq)
        positions = positions0[b] + torch.arange(rows_per_seq, device=device)
        kv_b = torch.full((rows_per_seq,), int(kv[b]), device=device)
        visible = qsa_visible_mask(ids, lens, positions, kv_b, s_kv, QSA_BLOCK_SIZE, top_k=QSA_TOP_K, pos0=positions0[b].expand(rows_per_seq))
        tail_start = QSA_BLOCK_SIZE * ((pos0[b] + 1) // QSA_BLOCK_SIZE)
        for j in range(rows_per_seq):
            assert bool((visible[j] | ~visible[0]).all()), f"sequence {b} row {j}: the step-0 set is not contained"
            grown = visible[j] & ~visible[0]
            idx = grown.nonzero().flatten()
            assert bool(((idx >= tail_start) & (idx <= pos0[b] + j)).all()), f"sequence {b} row {j}: a token outside the step-0 tail appeared"
            assert bool(visible[j, tail_start : pos0[b] + j + 1].all()), f"sequence {b} row {j}: the tail to the row's own position is not fully visible"
        # a draft row at position p with p % 4 == 3 completes block p // 4 (its own block), which the step-0 row cannot see
        completed = [(pos0[b] + j) // QSA_BLOCK_SIZE for j in range(1, rows_per_seq) if (pos0[b] + j) % QSA_BLOCK_SIZE == QSA_BLOCK_SIZE - 1]
        for blk in completed:  # completed by a draft row: not in the step-0 list, visible through the tail to the last row
            assert int(blk) not in shared["block_ids"][b].tolist()
            assert bool(visible[rows_per_seq - 1, QSA_BLOCK_SIZE * blk : QSA_BLOCK_SIZE * blk + QSA_BLOCK_SIZE].all())


@needs_sm100_family
def test_decode_ties_keep_the_smaller_block_ids():
    """Identical compressed keys tie every visible block: ``tie_break = 1`` (the default) keeps the smallest ids of
    each row, so the set is reproducible; a row in the identity region keeps every visible block."""
    device = torch.device("cuda")
    rows_per_seq, n_blocks_max = 2, 640
    pos0 = [4 * 600 + 1, 4 * 100 + 2]
    qi, kbar = _inputs(2, rows_per_seq, n_blocks_max, device, seed=29)
    kbar = kbar[:, :1].expand(-1, n_blocks_max, -1).contiguous()
    kv = _kv_lens(pos0, rows_per_seq, device)

    result = qsa_select_decode(qi, kbar, kv, list_per_sequence=False)
    torch.cuda.synchronize()

    expected0 = torch.arange(QSA_TOP_K, device=device, dtype=torch.int32)
    for j in range(rows_per_seq):
        assert torch.equal(torch.sort(result["block_ids"][0, j]).values, expected0), f"row {j}"
        n1 = (pos0[1] + j + 1) // QSA_BLOCK_SIZE
        ids1 = result["block_ids"][1, j]
        assert torch.equal(torch.sort(ids1[:n1]).values, torch.arange(n1, device=device, dtype=torch.int32)) and bool((ids1[n1:] == -1).all())
    positions = torch.tensor(pos0, device=device)[:, None] + torch.arange(rows_per_seq, device=device)[None, :]
    ref = _decode_reference(qi, kbar, positions, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_rows(
        result["block_ids"].view(-1, QSA_TOP_K), result["scores"].view(-1, QSA_TOP_K), result["block_lens"].view(-1), ref.view(-1, n_blocks_max), QSA_TOP_K
    )


@needs_sm100_family
def test_decode_agrees_with_the_prefill_selector_on_the_same_rows():
    """The prefill selector (compact-logits + fused top-k, THD) over the last rows of each sequence and the decode
    selector (dense scorer + radix top-k) over the same rows select the same sets up to ties."""
    device = torch.device("cuda")
    rows_per_seq, n_blocks_max = 4, 2048
    pos0 = [4 * 1500 + 1, 4 * 2000 + 3, 4 * 300]
    batch = len(pos0)
    qi, kbar = _inputs(batch, rows_per_seq, n_blocks_max, device, seed=31)
    kv = _kv_lens(pos0, rows_per_seq, device)

    decode = qsa_select_decode(qi, kbar, kv, list_per_sequence=False)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * rows_per_seq
    prefill = qsa_select(qi.reshape(batch * rows_per_seq, _H, _D), kbar, cu, q_pos0=torch.tensor(pos0, dtype=torch.int32, device=device))
    torch.cuda.synchronize()

    positions = torch.tensor(pos0, device=device)[:, None] + torch.arange(rows_per_seq, device=device)[None, :]
    ref = _decode_reference(qi, kbar, positions, QSA_BLOCK_SIZE, _D**-0.5).view(-1, n_blocks_max)
    dec_ids = decode["block_ids"].view(-1, QSA_TOP_K)
    _assert_rows(dec_ids, decode["scores"].view(-1, QSA_TOP_K), decode["block_lens"].view(-1), ref, QSA_TOP_K)
    _assert_rows(prefill["block_ids"], prefill["scores"], (prefill["block_ids"] >= 0).sum(dim=-1).to(torch.int32), ref, QSA_TOP_K)
    # both deterministic at the k-th boundary (the smaller id): the sets agree wherever no tie sits at the cutoff
    kth = torch.topk(ref, QSA_TOP_K, dim=-1).values[:, -1]
    for r in range(batch * rows_per_seq):
        a, b = set(dec_ids[r].tolist()) - {-1}, set(prefill["block_ids"][r].tolist()) - {-1}
        if a != b:
            diff = torch.tensor(sorted(a ^ b), device=device)
            assert bool(((ref[r, diff] - kth[r]).abs() <= 1e-4).all()), f"row {r}: {len(a ^ b)} ids differ beyond a tie"


@needs_sm100_family
def test_decode_scorer_tile_is_a_knob():
    """``m_block_size`` changes the scorer's schedule only: 16 (the default, four tokens of four heads) and 32 give
    the same sets and scores."""
    device = torch.device("cuda")
    rows_per_seq, n_blocks_max = 3, 1024
    pos0 = [4 * 900 + 2, 4 * 50]
    qi, kbar = _inputs(2, rows_per_seq, n_blocks_max, device, seed=37)
    kv = _kv_lens(pos0, rows_per_seq, device)
    a = qsa_select_decode(qi, kbar, kv, list_per_sequence=False)
    b = qsa_select_decode(qi, kbar, kv, list_per_sequence=False, m_block_size=32)
    torch.cuda.synchronize()
    assert torch.equal(torch.sort(a["block_ids"], dim=-1).values, torch.sort(b["block_ids"], dim=-1).values)
    assert torch.equal(torch.sort(a["scores"], dim=-1).values, torch.sort(b["scores"], dim=-1).values)
    assert torch.equal(a["block_lens"], b["block_lens"])


# --------------------------------------------------------------------------------------------- the fp8 indexer cache arm


@needs_cuda
def test_decode_rejects_a_mixed_cache_dtype_and_the_tile_knob_with_e4m3():
    device = torch.device("cuda")
    qi, kbar = _inputs(2, 1, 16, device)
    kv = _kv_lens([40, 41], 1, device)
    with pytest.raises(ValueError, match="both be bfloat16, or both float8_e4m3fn"):
        qsa_select_decode(qi.to(_E4M3), kbar, kv)
    with pytest.raises(ValueError, match="both be bfloat16, or both float8_e4m3fn"):
        qsa_select_decode(qi, kbar.to(_E4M3), kv)
    with pytest.raises(ValueError, match="fixed 128-row tile"):
        qsa_select_decode(qi.to(_E4M3), kbar.to(_E4M3), kv, m_block_size=16)


@needs_sm100_family
@pytest.mark.parametrize("rows_per_seq", [1, 4])
def test_decode_e4m3_cache_matches_the_fp64_reference_on_the_e4m3_values(rows_per_seq):
    """The fp8 indexer cache at decode: e4m3 ``qi`` / ``kbar`` through the dense MXFP8 scorer (its 128-row tile, all-ones
    scales) and the radix top-k; both list forms; sets equal the fp64 top-k of the e4m3 values up to ties, counts equal
    the oracle's; a cache capacity that is a multiple of 8 but not of the 128-row scale atom."""
    device = torch.device("cuda")
    batch, n_blocks_max = 3, 1032
    pos0 = [4 * n_blocks_max - rows_per_seq, 4 * 600 + 1, 4 * 100 + 2]
    qi, kbar = _inputs(batch, rows_per_seq, n_blocks_max, device, seed=41 + rows_per_seq)
    q8, k8 = qi.to(_E4M3), kbar.to(_E4M3)
    kv = _kv_lens(pos0, rows_per_seq, device)

    per_row = qsa_select_decode(q8, k8, kv, list_per_sequence=False, n_blocks_per_seq=[n_blocks_max] * batch)
    shared = qsa_select_decode(q8, k8, kv, list_per_sequence=True)
    torch.cuda.synchronize()

    positions = torch.tensor(pos0, device=device)[:, None] + torch.arange(rows_per_seq, device=device)[None, :]
    assert torch.equal(per_row["block_lens"], _oracle_counts(positions, QSA_TOP_K, QSA_BLOCK_SIZE).to(torch.int32))
    ref = _decode_reference(q8, k8, positions, QSA_BLOCK_SIZE, _D**-0.5)
    _assert_rows(
        per_row["block_ids"].view(-1, QSA_TOP_K),
        per_row["scores"].view(-1, QSA_TOP_K),
        per_row["block_lens"].view(-1),
        ref.view(batch * rows_per_seq, n_blocks_max),
        QSA_TOP_K,
    )
    _assert_rows(shared["block_ids"], shared["scores"], shared["block_lens"], ref[:, 0], QSA_TOP_K)
