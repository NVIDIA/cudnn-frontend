# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decode form of the QSA block-compression kernel (``kernels/qsa_compress.py::frost_qsa_compress_step``): a
step's new raw keys enter a position-indexed ring, and every committed row that completes a 4-token block pools it
from the ring and the step's rows through the prefill kernel's chain.

The acceptance is BITWISE: a token stream fed step by step (one, four, or a mixed number of rows per step; sequences
finishing at different steps) produces the prefill compress of the same tokens bit for bit, no row is written for
an incomplete group, and a rejected draft (a row past the accepted prefix) leaves no trace in the ring or in the
cache -- the next step with the true tokens recovers the prefill result.  Plain vectorized LDG/STG plus warp
shuffles, so these tests run on whatever CUDA device is at hand.
"""

import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.gated_attention_block.kernels.qsa_compress import (
    DEFAULT_POOL,
    SLOT_CAPACITY,
    SLOT_NO_BLOCK,
    QsaCompressStepRecipe,
    build_qsa_compress,
    build_qsa_compress_step,
    compile_qsa_compress_step,
    min_ring_rows,
    run_qsa_compress_step,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import build_rope_tables  # noqa: E402

pytestmark = pytest.mark.L0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_EPS = 1e-6
_D = 128
_ROPE = 64
_POOL = DEFAULT_POOL
_THETA = 1e7
_SENTINEL = -7777.0
_RING = 8  # serves steps of up to 5 rows (min_ring_rows(5) == 8)


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _make(batch, s_tok, dtype, *, seed=0, table_batch=1):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)

    k_raw = rnd(batch, s_tok, _D).to(dtype)
    w = (0.05 * rnd(_D)).to(dtype)
    cos, sin = build_rope_tables(s_tok + _POOL, _ROPE, base=_THETA, batch=table_batch, device="cuda", dtype=dtype)
    return k_raw, w, cos, sin


def _sentinel(*shape, dtype):
    return torch.full(shape, _SENTINEL, device="cuda", dtype=dtype)


def _prefill(k_raw, w, cos, sin, seq_lens, *, offset=1.0):
    """The prefill kernel's compressed keys (sentinel where it writes nothing) and the per-sequence block counts."""
    batch, s_tok, _ = k_raw.shape
    out = _sentinel(batch, s_tok // _POOL, _D, dtype=k_raw.dtype)
    build_qsa_compress(k_raw, out, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=offset, stream=_stream())
    torch.cuda.synchronize()
    return out, (seq_lens.long().clamp(0, s_tok) // _POOL)


def _ring_from_prefix(k_raw, pos0, ring_rows):
    """The ring state after committing positions ``[0, pos0[b])`` of every sequence: row ``p % RING`` holds the raw
    key of position ``p`` for the last ``RING`` committed positions; every other row the sentinel."""
    batch = k_raw.shape[0]
    ring = _sentinel(batch, ring_rows, _D, dtype=k_raw.dtype)
    for b in range(batch):
        for p in range(max(0, int(pos0[b]) - ring_rows), int(pos0[b])):
            ring[b, p % ring_rows] = k_raw[b, p]
    return ring


def _step(k_new, ring, out, w, cos, sin, pos0, n_commit=None, *, offset=1.0, rope_dim=_ROPE, **kw):
    """One decode step; returns the per-row slot report."""
    slot = torch.full(tuple(k_new.shape[:2]), -9, dtype=torch.int32, device="cuda")
    build_qsa_compress_step(k_new, ring, out, w, cos, sin, pos0, slot, n_commit, rope_dim=rope_dim, eps=_EPS, norm_weight_offset=offset, stream=_stream(), **kw)
    torch.cuda.synchronize()
    return slot


def _run_stream(k_raw, lens, schedule, *, w, cos, sin, offset, ring_rows=_RING, n_blocks_cap=None):
    """Feed every sequence its tokens step by step (``schedule`` = rows per step, cycled; the batch shares the step
    shape, a sequence past its length commits nothing) and return the cache, the ring and the slot reports."""
    batch, s_tok, _ = k_raw.shape
    nb = s_tok // _POOL if n_blocks_cap is None else n_blocks_cap
    out = _sentinel(batch, nb, _D, dtype=k_raw.dtype)
    ring = _sentinel(batch, ring_rows, _D, dtype=k_raw.dtype)
    pos = [0] * batch
    slots = [[] for _ in range(batch)]
    step = 0
    while any(pos[b] < lens[b] for b in range(batch)):
        rows = schedule[step % len(schedule)]
        step += 1
        k_new = torch.full((batch, rows, _D), float("nan"), device="cuda", dtype=k_raw.dtype)  # poison: uncommitted rows must never be pooled
        n_commit = []
        for b in range(batch):
            take = max(0, min(rows, lens[b] - pos[b]))
            if take:
                k_new[b, :take] = k_raw[b, pos[b] : pos[b] + take]
            n_commit.append(take)
        slot = _step(k_new, ring, out, w, cos, sin, _i32(pos), _i32(n_commit), offset=offset)
        for b in range(batch):
            slots[b].extend(slot[b].tolist())
            pos[b] += n_commit[b]
    return out, ring, slots


# ---------------------------------------------------------------------------
# Shape algebra -- no GPU
# ---------------------------------------------------------------------------


def test_min_ring_rows_is_rows_plus_pool_minus_one():
    assert min_ring_rows(1) == 4 and min_ring_rows(4) == 7 and min_ring_rows(5) == 8 and min_ring_rows(2, pool=8) == 9
    assert SLOT_NO_BLOCK == -1 and SLOT_CAPACITY == -2


# ---------------------------------------------------------------------------
# Numerics: bitwise the prefill compress of the same tokens
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("offset", [0.0, 1.0], ids=["folded_1_plus_w", "zero_centred_w"])
@pytest.mark.parametrize("schedule", [[1], [4], [1, 4, 2, 3, 5, 1, 2, 4]], ids=["one_row_steps", "four_row_steps", "mixed_steps"])
def test_step_stream_equals_the_prefill_compress_bitwise(schedule, offset, dtype):
    """Four sequences (full, a 7-token tail, 37 tokens, EMPTY) fed through steps of the given row counts: every
    complete block of every sequence is written exactly once (its slot reported by the completing row), bitwise the
    prefill kernel's row; incomplete groups, the empty sequence and the rows past a sequence's end write nothing."""
    s_tok, lens = 517, [517, 510, 37, 0]
    batch = len(lens)
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=1)
    ref, n_blocks = _prefill(k_raw, w, cos, sin, _i32(lens), offset=offset)
    out, ring, slots = _run_stream(k_raw, lens, schedule, w=w, cos=cos, sin=sin, offset=offset)
    sentinel = _sentinel(1, 1, _D, dtype=dtype)[0, 0]
    for b in range(batch):
        nb = int(n_blocks[b])
        assert torch.equal(out[b, :nb], ref[b, :nb]), f"sequence {b}: the incremental compress differs from the prefill compress"
        assert bool((out[b, nb:] == sentinel).all()), f"sequence {b}: a row past the sequence's blocks was written"
        written = [s for s in slots[b] if s >= 0]
        assert sorted(written) == list(range(nb)), f"sequence {b}: blocks written {sorted(written)} != 0..{nb - 1}"
        assert SLOT_CAPACITY not in slots[b]
        # the ring holds the raw keys of the last RING committed positions, bit for bit
        for p in range(max(0, lens[b] - _RING), lens[b]):
            assert torch.equal(ring[b, p % _RING], k_raw[b, p]), f"sequence {b}: ring row {p % _RING} is not the raw key of position {p}"
    assert bool((out[3] == sentinel).all()) and bool((ring[3] == sentinel).all())


@requires_cuda
def test_a_block_completes_at_the_fourth_token():
    """pos0 = 4k, four rows: rows 0..2 complete nothing, row 3 completes block k from the step's own rows; the
    four raw keys enter the ring (a bit copy) and nothing else moves."""
    dtype, s_tok, k = torch.bfloat16, 128, 17
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=2)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok, s_tok]))
    pos0 = [4 * k, 4 * k]
    ring = _ring_from_prefix(k_raw, pos0, _RING)
    ring_before = ring.clone()
    out = _sentinel(2, s_tok // _POOL, _D, dtype=dtype)
    sentinel = out.clone()
    slot = _step(k_raw[:, 4 * k : 4 * k + 4].contiguous(), ring, out, w, cos, sin, _i32(pos0))
    assert slot.tolist() == [[-1, -1, -1, k]] * 2
    assert torch.equal(out[:, k], ref[:, k])
    mask = torch.ones_like(out, dtype=torch.bool)
    mask[:, k] = False
    assert torch.equal(out[mask], sentinel[mask])
    for p in range(4 * k, 4 * k + 4):
        assert torch.equal(ring[:, p % _RING], k_raw[:, p])
        ring_before[:, p % _RING] = ring[:, p % _RING]
    assert torch.equal(ring, ring_before)


@requires_cuda
@pytest.mark.parametrize("residue", [0, 1, 2, 3])
def test_exactly_one_completion_per_four_row_step_whatever_the_alignment(residue):
    """Four consecutive positions hold exactly one position with ``p % 4 == 3``: one block per four-row step, at
    row ``3 - pos0 % 4`` -- the step-0 row itself when ``pos0 % 4 == 3`` (its block pools three ring keys)."""
    dtype, s_tok, k = torch.float16, 96, 11
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=3 + residue)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok]))
    pos0 = 4 * k + residue
    ring = _ring_from_prefix(k_raw, [pos0], _RING)
    out = _sentinel(1, s_tok // _POOL, _D, dtype=dtype)
    slot = _step(k_raw[:, pos0 : pos0 + 4].contiguous(), ring, out, w, cos, sin, _i32([pos0]))
    row = 3 - residue
    blk = (pos0 + row) // _POOL
    want = [-1] * 4
    want[row] = blk
    assert slot.tolist() == [want]
    assert torch.equal(out[0, blk], ref[0, blk])
    assert int((out[0] != _SENTINEL).any(dim=-1).sum()) == 1


@requires_cuda
def test_two_blocks_per_step_need_five_rows():
    """At four rows per step a sequence completes at most one block; a five-row step at ``pos0 % 4 == 3`` completes
    two (rows 0 and 4), both bitwise the prefill rows."""
    dtype, s_tok, k = torch.bfloat16, 96, 9
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=8)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok]))
    pos0 = 4 * k + 3
    ring = _ring_from_prefix(k_raw, [pos0], _RING)
    out = _sentinel(1, s_tok // _POOL, _D, dtype=dtype)
    slot = _step(k_raw[:, pos0 : pos0 + 5].contiguous(), ring, out, w, cos, sin, _i32([pos0]))
    assert slot.tolist() == [[k, -1, -1, -1, k + 1]]
    assert torch.equal(out[0, k : k + 2], ref[0, k : k + 2])


@requires_cuda
def test_rejected_drafts_are_never_committed_and_the_next_step_recovers():
    """pos0 = 4k + 1, four rows, two accepted: the completing row (position 4k + 3) is a rejected draft -- no block,
    no ring row, slot -1 -- and the ring rows of the rejected positions keep their previous content.  The next step
    at pos0 = 4k + 3 with the TRUE tokens completes block k from the ring (positions 4k .. 4k + 2, two of them
    committed by the previous step) and its own row: bitwise the prefill compress of the accepted stream."""
    dtype, s_tok, k = torch.bfloat16, 128, 13
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=5)  # the TRUE stream
    drafts = torch.randn(1, 4, _D, device="cuda").to(dtype)  # the step's rows: two accepted (= true), two rejected
    drafts[:, :2] = k_raw[:, 4 * k + 1 : 4 * k + 3]
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok]))
    pos0 = 4 * k + 1
    ring = _ring_from_prefix(k_raw, [pos0], _RING)
    ring_before = ring.clone()
    out = _sentinel(1, s_tok // _POOL, _D, dtype=dtype)
    sentinel = out.clone()

    slot = _step(drafts, ring, out, w, cos, sin, _i32([pos0]), _i32([2]))
    assert slot.tolist() == [[-1, -1, -1, -1]]
    assert torch.equal(out, sentinel), "a block completed by a rejected draft was written"
    for p in (pos0, pos0 + 1):
        assert torch.equal(ring[0, p % _RING], k_raw[0, p])
        ring_before[0, p % _RING] = ring[0, p % _RING]
    assert torch.equal(ring, ring_before), "a rejected draft's raw key entered the ring"

    # the next step: the true tokens at 4k + 3 and 4k + 4
    pos1 = 4 * k + 3
    slot = _step(k_raw[:, pos1 : pos1 + 2].contiguous(), ring, out, w, cos, sin, _i32([pos1]), _i32([2]))
    assert slot.tolist() == [[k, -1]]
    assert torch.equal(out[0, k], ref[0, k])
    mask = torch.ones_like(out, dtype=torch.bool)
    mask[0, k] = False
    assert torch.equal(out[mask], sentinel[mask])


@requires_cuda
def test_two_phase_commit_is_idempotent():
    """The serving step's two calls (n_commit = 1 at the start, 1 + accepted after verification) compose to one call
    with the final prefix: the re-committed row rewrites the same ring row and the same block row bit for bit."""
    dtype, s_tok, k = torch.float16, 128, 21
    k_raw, w, cos, sin = _make(3, s_tok, dtype, seed=6)
    pos0 = [4 * k + 3, 4 * k, 4 * k + 2]  # row 0 completes a block / row 3 does / row 1 does
    accepted = [3, 4, 2]
    k_new = torch.stack([k_raw[b, p : p + 4] for b, p in enumerate(pos0)]).contiguous()

    ring_a = _ring_from_prefix(k_raw, pos0, _RING)
    out_a = _sentinel(3, s_tok // _POOL, _D, dtype=dtype)
    _step(k_new, ring_a, out_a, w, cos, sin, _i32(pos0), _i32([1, 1, 1]))
    slot_a = _step(k_new, ring_a, out_a, w, cos, sin, _i32(pos0), _i32(accepted))

    ring_b = _ring_from_prefix(k_raw, pos0, _RING)
    out_b = _sentinel(3, s_tok // _POOL, _D, dtype=dtype)
    slot_b = _step(k_new, ring_b, out_b, w, cos, sin, _i32(pos0), _i32(accepted))

    assert torch.equal(ring_a, ring_b) and torch.equal(out_a, out_b) and torch.equal(slot_a, slot_b)
    assert slot_b.tolist() == [[k, -1, -1, -1], [-1, -1, -1, k], [-1, k, -1, -1]]


@requires_cuda
def test_capacity_overflow_reports_minus_two_and_writes_nothing():
    """A completing row whose block is past the compressed-key cache, or whose block start is past the RoPE table,
    reports -2 and writes no block; the row's raw key still enters the ring (it is committed)."""
    dtype, s_tok, k = torch.bfloat16, 128, 7
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=9)
    pos0 = 4 * k
    k_new = k_raw[:, pos0 : pos0 + 4].contiguous()
    # (a) a cache of k rows holds blocks 0 .. k - 1 only
    ring = _ring_from_prefix(k_raw, [pos0], _RING)
    out = _sentinel(1, k, _D, dtype=dtype)
    slot = _step(k_new, ring, out, w, cos, sin, _i32([pos0]))
    assert slot.tolist() == [[-1, -1, -1, SLOT_CAPACITY]]
    assert bool((out == _SENTINEL).all())
    assert torch.equal(ring[0, (pos0 + 3) % _RING], k_raw[0, pos0 + 3])
    # (b) a RoPE table that ends before the block's start position
    ring = _ring_from_prefix(k_raw, [pos0], _RING)
    out = _sentinel(1, s_tok // _POOL, _D, dtype=dtype)
    slot = _step(k_new, ring, out, w, cos[:, :pos0].contiguous(), sin[:, :pos0].contiguous(), _i32([pos0]))
    assert slot.tolist() == [[-1, -1, -1, SLOT_CAPACITY]]
    assert bool((out == _SENTINEL).all())


@requires_cuda
def test_finished_sequence_and_negative_position_are_harmless():
    """n_commit = 0 (a sequence that finished earlier in the batch) commits nothing and reports -1 for every row;
    a negative pos0 is clamped to 0 (the first step of a sequence)."""
    dtype, s_tok = torch.bfloat16, 64
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=10)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok, s_tok]))
    ring = _sentinel(2, _RING, _D, dtype=dtype)
    out = _sentinel(2, s_tok // _POOL, _D, dtype=dtype)
    slot = _step(k_raw[:, :4].contiguous(), ring, out, w, cos, sin, _i32([-5, 40]), _i32([4, 0]))
    assert slot.tolist() == [[-1, -1, -1, 0], [-1, -1, -1, -1]]
    assert torch.equal(out[0, 0], ref[0, 0]) and bool((out[1] == _SENTINEL).all()) and bool((ring[1] == _SENTINEL).all())


@requires_cuda
def test_k_new_as_a_slab_column_slice_is_bitwise_the_compact_run():
    """The step's raw keys as the INDEX band of a ``[B, S_q, 640]`` projection slab (token stride 640) and as a
    compact copy give bitwise-equal ring rows, cache rows and slots: one artifact, symbolic strides."""
    dtype, s_tok, k = torch.bfloat16, 128, 5
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=11)
    pos0 = [4 * k + 2, 4 * k + 3]
    slab = torch.randn(2, 4, 640, device="cuda").to(dtype)
    for b, p in enumerate(pos0):
        slab[b, :, 512:640] = k_raw[b, p : p + 4]
    k_slab = slab[:, :, 512:640]
    assert k_slab.stride(1) == 640
    results = []
    for k_new in (k_slab, k_slab.contiguous()):
        ring = _ring_from_prefix(k_raw, pos0, _RING)
        out = _sentinel(2, s_tok // _POOL, _D, dtype=dtype)
        slot = _step(k_new, ring, out, w, cos, sin, _i32(pos0))
        results.append((ring, out, slot))
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1]) and torch.equal(results[0][2], results[1][2])
    assert results[0][2].tolist() == [[-1, k, -1, -1], [k, -1, -1, -1]]  # row 1 at 4k + 3 completes block k; row 0 at 4k + 3 too


@requires_cuda
def test_one_artifact_serves_every_step_shape_and_ring():
    dtype = torch.bfloat16
    r = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=True)
    assert isinstance(r, QsaCompressStepRecipe) and r.has_n_commit and r.rows_per_cta == 8
    assert compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=True).compiled is r.compiled
    k_raw, w, cos, sin = _make(1, 256, dtype, seed=12)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([256]))
    for rows, ring_rows in ((1, 8), (3, 8), (5, 8), (4, 16), (9, 12)):
        pos0 = 4 * 20 + (3 - (rows - 1) % 4)  # the last row completes a block
        ring = _ring_from_prefix(k_raw, [pos0], ring_rows)
        out = _sentinel(1, 64, _D, dtype=dtype)
        slot = torch.full((1, rows), -9, dtype=torch.int32, device="cuda")
        run_qsa_compress_step(
            r,
            k_raw[:, pos0 : pos0 + rows].contiguous(),
            ring,
            out,
            w,
            cos,
            sin,
            _i32([pos0]),
            slot,
            _i32([rows]),
            eps=_EPS,
            norm_weight_offset=1.0,
            stream=_stream(),
        )
        torch.cuda.synchronize()
        blk = (pos0 + rows - 1) // _POOL
        assert slot[0, rows - 1].item() == blk and torch.equal(out[0, blk], ref[0, blk]), (rows, ring_rows)


# ---------------------------------------------------------------------------
# Host-side contract: every refusal is typed and names the operand
# ---------------------------------------------------------------------------


@requires_cuda
def test_ring_depth_rule_is_enforced_on_the_host():
    dtype, s_tok = torch.bfloat16, 64
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=13)
    out = _sentinel(1, 16, _D, dtype=dtype)
    k_new = k_raw[:, :4].contiguous()
    with pytest.raises(ValueError, match=r"needs at least 7 \(rows \+ pool - 1 = 4 \+ 4 - 1\)"):
        _step(k_new, _sentinel(1, 6, _D, dtype=dtype), out, w, cos, sin, _i32([0]))
    slot = _step(k_new, _sentinel(1, 7, _D, dtype=dtype), out, w, cos, sin, _i32([0]))
    assert slot.tolist() == [[-1, -1, -1, 0]]


@requires_cuda
def test_run_rejects_contract_violations():
    dtype, s_tok = torch.bfloat16, 64
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=14)
    k_new = k_raw[:, :4].contiguous()
    ring = _sentinel(2, _RING, _D, dtype=dtype)
    out = _sentinel(2, 16, _D, dtype=dtype)
    pos0 = _i32([0, 0])
    slot = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    with_commit = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=True)
    without = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False)

    def run(r, k=k_new, rg=ring, o=out, wt=w, c=cos, s=sin, p=pos0, sl=slot, nc=None):
        run_qsa_compress_step(r, k, rg, o, wt, c, s, p, sl, nc, eps=_EPS, stream=_stream())

    with pytest.raises(ValueError, match="WITH an accepted-prefix vector"):
        run(with_commit)
    with pytest.raises(ValueError, match="WITHOUT n_commit"):
        run(without, nc=_i32([4, 4]))
    with pytest.raises(ValueError, match="n_commit must be a contiguous int32"):
        run(with_commit, nc=torch.tensor([4, 4], dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="pos0 must be a contiguous int32"):
        run(without, p=pos0.to(torch.int64))
    with pytest.raises(ValueError, match="pos0 must be a contiguous int32"):
        run(without, p=pos0.cpu())
    with pytest.raises(ValueError, match=r"slot_out must be a contiguous int32 CUDA tensor of shape \(2, 4\)"):
        run(without, sl=slot[:, :3])
    with pytest.raises(ValueError, match="ring must be a compact"):
        run(without, rg=ring[:1])
    with pytest.raises(ValueError, match="ring must be a compact"):
        run(without, rg=_sentinel(2, _RING, 2 * _D, dtype=dtype)[:, :, :_D])
    with pytest.raises(ValueError, match=r"out must be a compact \[B=2, NB >= 1"):
        run(without, o=_sentinel(2, 0, _D, dtype=dtype))
    with pytest.raises(ValueError, match="dtype"):
        run(without, k=k_new.float())
    with pytest.raises(ValueError, match=r"k_new must be \[B, S_q, D\]"):
        run(without, k=k_new[0])
    with pytest.raises(ValueError, match="norm weight"):
        run(without, wt=w[:64])
    with pytest.raises(ValueError, match="at least one position row"):
        run(without, c=cos[:, :0], s=sin[:, :0])
    with pytest.raises(ValueError, match="share a shape"):
        run(without, s=sin[:, :8].contiguous())
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        run(without, k=torch.randn(2, 4, 132, device="cuda").to(dtype)[:, :, :_D])


def test_compile_rejects_unsupported_dtypes():
    with pytest.raises(ValueError, match="bf16/f16"):
        compile_qsa_compress_step(dtype=torch.float32, d=_D, rope_dim=_ROPE)
    with pytest.raises(ValueError, match="16-bit cos"):
        compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, table_dtype=torch.float32)
