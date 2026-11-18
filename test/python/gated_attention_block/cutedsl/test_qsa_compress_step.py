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

The PAGED form (the serving stack's ring blocks behind a state table and its compressed-key cache through the main block
table, positions from ``kv_lens``): a token stream through the paged form is bitwise the dense form and the prefill
compress at every page size, the derived ring rows are the stack's own ``block x cap + pos % cap`` slots and the explicit
slot form is bitwise the derived one, a padding sequence (``kv_lens 0``) writes nothing, a block past the table or on an
unassigned page reports ``-2``, a live sequence whose state entry is the null block or lies past the ring owns no ring (no ring
row anywhere; a block needing ring rows ``-2``, one from the step's rows alone written), one artifact serves every page size and
ring capacity, the launch is ONE kernel with no allocation, and every refusal is typed in both directions.

The RoPE table is the CEILING on positions in both forms: a committed row at or past it (completing or not) reports ``-2`` and
writes nothing -- no ring row, no block -- while the row below the ceiling is served; the cache capacity is not the ceiling (a
position past the cache inside the table is served, and only a block past the cache is ``-2``).

The e4m3 STORE arm (``index_cache_dtype=torch.float8_e4m3fn``; sm_89+ -- the cells skip below it and are UNCOVERED there): the dense
and the paged step arms on the SAME token stream write rows BITWISE each other and bitwise the bf16 arm's rows ``.to(float8_e4m3fn)``
(the e4m3 cast of the bf16 row: one rounding convention across every writer), the slot reports identical, the ring untouched in bf16, the
block completing at the 4th token and the page boundaries served; the e4m3 artifact distinct from the bf16 one; the dtype refusals both
ways (an e4m3 cache under a bf16 recipe and a bf16 cache under an e4m3 recipe name the recipe's dtype; an e4m3 RING stays the ring's rule;
f16 io has no e4m3 arm).

The CHUNK form (``ring_write=False``; 2026-10-29): a raw-key stream behind a prefix ``P`` of every residue mod 4, cut into chunks of
``T`` rows (1 .. 2048, the ``T < 4 - P % 4`` chunks that complete no block included) over the PAGED side caches at page 16 / 32 / 64 and
ring cap 3 / 4 / 8 -- the ring primed by the TEST with the stream's last ``cap`` positions (the tail arm is the cache-write kernel's,
pinned in ``test_qsa_prepare.py``) and the rows the kernel must never read poisoned with NaN: after every chunk the cache rows of every
complete block of the stream so far BITWISE the from-scratch prefill compress, the ring BITWISE untouched, the slot reports the position
rule's; the key distinct from the step form's, ``ring_slots`` refused with it, cap 3 served and cap 2 refused naming ``pool - 1``, the
dense form the paged form's twin, one launch and no allocation at 2048 rows, the e4m3 cache arm the bf16 rows' cast.

The rounding attribute (``rounding="hf"``): the dense, the paged and the e4m3 stream cells and the chunk cells run under BOTH values --
the step form, the chunk form and the prefill form stay bitwise one another under ``"hf"`` as under ``"single"`` (what "the writers
round identically" means; the arm's numerics against the HF-rounding oracle are ``test_qsa_compress.py``'s).
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
_CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
# the e4m3 store arm's cvt.rn.satfinite.e4m3x2 is an sm_89+ instruction: its accept cells skip (UNCOVERED) on an older part
requires_sm89 = pytest.mark.skipif(_CC is None or _CC < (8, 9), reason=f"the e4m3 store arm needs sm_89+ (found {_CC})")
_E4M3 = torch.float8_e4m3fn

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


def _sentinel_value(dtype) -> float:
    """The sentinel fill per dtype: -7777 for the 16-bit dtypes; -448 for e4m3 (its max magnitude: representable -- torch.full refuses a value
    past the dtype's range -- and never a written row: the e4m3 cells assert |bf16 row| < 448 first)."""
    return -448.0 if dtype == _E4M3 else _SENTINEL


def _sentinel(*shape, dtype):
    return torch.full(shape, _sentinel_value(dtype), device="cuda", dtype=dtype)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The bit patterns of a 16-bit or an 8-bit tensor (a NaN-safe bitwise comparison)."""
    return x.contiguous().view(torch.int8 if x.element_size() == 1 else torch.int16)


def _prefill(k_raw, w, cos, sin, seq_lens, *, offset=1.0, rounding="single"):
    """The prefill kernel's compressed keys (sentinel where it writes nothing) and the per-sequence block counts."""
    batch, s_tok, _ = k_raw.shape
    out = _sentinel(batch, s_tok // _POOL, _D, dtype=k_raw.dtype)
    build_qsa_compress(k_raw, out, w, cos, sin, seq_lens, rope_dim=_ROPE, eps=_EPS, norm_weight_offset=offset, stream=_stream(), rounding=rounding)
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


def _run_stream(k_raw, lens, schedule, *, w, cos, sin, offset, ring_rows=_RING, n_blocks_cap=None, cache_dtype=None, rounding="single"):
    """Feed every sequence its tokens step by step (``schedule`` = rows per step, cycled; the batch shares the step
    shape, a sequence past its length commits nothing) and return the cache, the ring and the slot reports.  ``cache_dtype``
    (``None`` = the io dtype) is the cache's dtype: ``float8_e4m3fn`` runs the e4m3 store arm (the ring stays the io dtype);
    ``rounding`` the chain's rounding points."""
    batch, s_tok, _ = k_raw.shape
    nb = s_tok // _POOL if n_blocks_cap is None else n_blocks_cap
    out = _sentinel(batch, nb, _D, dtype=k_raw.dtype if cache_dtype is None else cache_dtype)
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
        slot = _step(k_new, ring, out, w, cos, sin, _i32(pos), _i32(n_commit), offset=offset, rounding=rounding)
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
@pytest.mark.parametrize("rounding", ["single", "hf"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("offset", [0.0, 1.0], ids=["folded_1_plus_w", "zero_centred_w"])
@pytest.mark.parametrize("schedule", [[1], [4], [1, 4, 2, 3, 5, 1, 2, 4]], ids=["one_row_steps", "four_row_steps", "mixed_steps"])
def test_step_stream_equals_the_prefill_compress_bitwise(schedule, offset, dtype, rounding):
    """Four sequences (full, a 7-token tail, 37 tokens, EMPTY) fed through steps of the given row counts: every
    complete block of every sequence is written exactly once (its slot reported by the completing row), bitwise the
    prefill kernel's row; incomplete groups, the empty sequence and the rows past a sequence's end write nothing.  Under
    BOTH rounding attributes: the step form and the prefill form round identically."""
    s_tok, lens = 517, [517, 510, 37, 0]
    batch = len(lens)
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=1)
    ref, n_blocks = _prefill(k_raw, w, cos, sin, _i32(lens), offset=offset, rounding=rounding)
    out, ring, slots = _run_stream(k_raw, lens, schedule, w=w, cos=cos, sin=sin, offset=offset, rounding=rounding)
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
    """A completing row whose block is past the compressed-key cache reports -2 and writes no block while its raw key still enters
    the ring (the position itself is served); a step whose rows lie at or past the RoPE table reports -2 for EVERY one of them and
    writes nothing at all, no ring row either (the table is the ceiling on positions)."""
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
    # (b) a RoPE table that ends AT the block's start position: the table is the ceiling on positions, so every row of the step (all at
    # or past it) reports -2 and writes nothing -- no ring row either (where the cache rule above still enters the committed row's key)
    ring = _ring_from_prefix(k_raw, [pos0], _RING)
    ring_before = ring.clone()
    out = _sentinel(1, s_tok // _POOL, _D, dtype=dtype)
    slot = _step(k_new, ring, out, w, cos[:, :pos0].contiguous(), sin[:, :pos0].contiguous(), _i32([pos0]))
    assert slot.tolist() == [[SLOT_CAPACITY] * 4]
    assert bool((out == _SENTINEL).all()) and torch.equal(ring, ring_before), "a row at or past the table wrote something"


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


# ---------------------------------------------------------------------------
# The PAGED form: the stack's ring and compressed-key cache through their block tables, bitwise the dense step form
# ---------------------------------------------------------------------------


def _sent_of(dtype) -> float:
    """The sentinel AS THE DTYPE HOLDS IT (bf16 rounds -7777.0; e4m3 holds -448.0): the only value a 'row untouched' check may compare against."""
    return torch.full((), _sentinel_value(dtype), dtype=dtype).float().item()


class _Paged:
    """The serving stack's two side caches for a batch: ring blocks ``1 .. B`` (block 0 is the null block) of ``cap`` rows each,
    and a compressed-key cache of ``P / 4`` rows per attention page behind a PERMUTED block table (page 0 unused by any live
    sequence); the sequences in ``padding`` sit on block 0 / page 0 with ``kv_lens 0``."""

    def __init__(self, batch, s_tok, dtype, *, page, cap, padding=(), seed=0, table_pages=None, cache_dtype=None):
        self.rpp, self.cap, self.page, self.batch, self.dtype = page // _POOL, cap, page, batch, dtype
        self.cache_dtype = dtype if cache_dtype is None else cache_dtype  # the compressed-key cache's dtype (e4m3 = the e4m3 store arm)
        n_lpages = (s_tok // _POOL + self.rpp - 1) // self.rpp if table_pages is None else table_pages
        self.n_pages = 1 + batch * max(n_lpages, 1)
        g = torch.Generator(device="cuda").manual_seed(seed + 100)
        perm = torch.randperm(self.n_pages - 1, generator=g, device="cuda") + 1
        self.block_table = perm[: batch * n_lpages].view(batch, n_lpages).to(torch.int32).contiguous()
        self.state_table = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).view(batch, 1)
        for b in padding:
            self.block_table[b] = 0
            self.state_table[b] = 0
        self.ring = _sentinel(batch + 1, cap, 1, _D, dtype=dtype)
        self.cache = _sentinel(self.n_pages, self.rpp, 1, _D, dtype=self.cache_dtype)
        self.sent = _sent_of(self.cache_dtype)  # the CACHE's sentinel as its dtype holds it (e4m3 saturates it to -448)

    def kbar_row(self, b, blk):
        return self.cache[int(self.block_table[b, blk // self.rpp]), blk % self.rpp, 0]

    def ring_row(self, b, pos):
        return self.ring[int(self.state_table[b, 0]), pos % self.cap, 0]

    def rows_written(self) -> int:
        return int((self.cache.float() != self.sent).any(-1).sum())

    def fill_ring_from_prefix(self, k_raw, pos0):
        """The ring state after committing positions ``[0, pos0[b])``: row ``p % cap`` of the sequence's block holds the raw key of
        position ``p`` for the last ``cap`` committed positions (the paged twin of ``_ring_from_prefix``)."""
        for b in range(self.batch):
            for p in range(max(0, int(pos0[b]) - self.cap), int(pos0[b])):
                self.ring[int(self.state_table[b, 0]), p % self.cap, 0] = k_raw[b, p]

    def ring_slots(self, kv_lens, rows, dtype=torch.int64) -> torch.Tensor:
        """The stack's own ring slot mapping, ``block x cap + pos % cap`` for a live row (``pos >= 0``, a ring block ``> 0``) and
        ``-1`` otherwise -- the CircularBufferSpec formula, spelled on the host as the kernel's derived form must agree with it."""
        pos = torch.tensor([[kv - rows + j for j in range(rows)] for kv in kv_lens], device="cuda", dtype=torch.int64)
        blk = self.state_table.long()
        live = (pos >= 0) & (blk > 0)
        return torch.where(live, blk * self.cap + pos % self.cap, torch.full_like(pos, -1)).reshape(-1).to(dtype)


def _step_paged(pg: _Paged, k_new, w, cos, sin, kv_lens, n_commit=None, *, offset=1.0, ring_slots=None, **kw):
    """One paged decode step; returns the per-row slot report."""
    slot = torch.full(tuple(k_new.shape[:2]), -9, dtype=torch.int32, device="cuda")
    build_qsa_compress_step(
        k_new,
        pg.ring,
        pg.cache,
        w,
        cos,
        sin,
        None,
        slot,
        n_commit,
        rope_dim=_ROPE,
        eps=_EPS,
        norm_weight_offset=offset,
        stream=_stream(),
        kv_lens=_i32(kv_lens),
        block_table=pg.block_table,
        index_state_block_table=pg.state_table,
        index_state_slot_mapping=ring_slots,
        **kw,
    )
    torch.cuda.synchronize()
    return slot


def _run_stream_paged(k_raw, lens, schedule, *, w, cos, sin, offset, page, cap, padding=(), seed=0, use_slots=None, cache_dtype=None, rounding="single"):
    """The paged twin of ``_run_stream``: every step carries ``rows`` rows per sequence with ``kv_lens = pos + rows`` (the KV length
    after the step, the stack's ``seq_lens``) and the accepted prefix in ``n_commit``; a padding sequence keeps ``kv_lens 0``.
    ``cache_dtype`` as in ``_run_stream`` (the e4m3 store arm when ``float8_e4m3fn``)."""
    batch, s_tok, _ = k_raw.shape
    pg = _Paged(batch, s_tok, k_raw.dtype, page=page, cap=cap, padding=padding, seed=seed, cache_dtype=cache_dtype)
    pos = [0] * batch
    slots = [[] for _ in range(batch)]
    step = 0
    while any(pos[b] < lens[b] for b in range(batch)):
        rows = schedule[step % len(schedule)]
        step += 1
        k_new = torch.full((batch, rows, _D), float("nan"), device="cuda", dtype=k_raw.dtype)
        n_commit, kv_lens = [], []
        for b in range(batch):
            take = max(0, min(rows, lens[b] - pos[b]))
            if take:
                k_new[b, :take] = k_raw[b, pos[b] : pos[b] + take]
            n_commit.append(take)
            kv_lens.append(pos[b] + rows if b not in padding else 0)
        rs = pg.ring_slots(kv_lens, rows, dtype=use_slots) if use_slots is not None else None
        slot = _step_paged(pg, k_new, w, cos, sin, kv_lens, _i32(n_commit), offset=offset, ring_slots=rs, rounding=rounding)
        for b in range(batch):
            slots[b].extend(slot[b].tolist())
            pos[b] += n_commit[b]
    return pg, slots


@requires_cuda
@pytest.mark.parametrize("rounding", ["single", "hf"])
@pytest.mark.parametrize("page", [16, 32, 64], ids=["p16", "p32", "p64"])
@pytest.mark.parametrize("schedule", [[1], [4], [1, 4, 2, 3, 5, 1, 2, 4]], ids=["one_row_steps", "four_row_steps", "mixed_steps"])
def test_paged_step_stream_is_bitwise_the_dense_step_and_the_prefill_compress(schedule, page, rounding):
    """The same token stream through the dense step form and the PAGED one (ring blocks behind the state table, the cache's rows
    through a permuted block table at P 16 / 32 / 64 -- 4 / 8 / 16 compressed rows per page, so the blocks cross page boundaries):
    every compressed row the paged form writes is bitwise the dense form's and the prefill kernel's, the slot reports are
    identical step by step, exactly the live blocks are written (page 0 and the null ring block untouched -- the padding
    sequence has ``kv_lens 0``), and the ring holds the last ``cap`` raw keys of every live sequence bit for bit."""
    dtype, cap = torch.bfloat16, 8  # cap 8 serves steps of up to 5 rows (min_ring_rows(5) == 8)
    s_tok, lens = 517, [517, 510, 37, 0]
    batch = len(lens)
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=21)
    ref, n_blocks = _prefill(k_raw, w, cos, sin, _i32(lens), offset=1.0, rounding=rounding)
    out_d, ring_d, slots_d = _run_stream(k_raw, lens, schedule, w=w, cos=cos, sin=sin, offset=1.0, rounding=rounding)
    pg, slots_p = _run_stream_paged(k_raw, lens, schedule, w=w, cos=cos, sin=sin, offset=1.0, page=page, cap=cap, padding=(3,), rounding=rounding)
    assert slots_p == slots_d, "the paged form's slot reports differ from the dense form's"
    for b in range(batch):
        nb = int(n_blocks[b])
        for blk in range(nb):
            row = pg.kbar_row(b, blk)
            assert torch.equal(
                row.view(torch.int16), ref[b, blk].view(torch.int16)
            ), f"sequence {b} block {blk}: the paged compress differs from the prefill compress"
            assert torch.equal(
                row.view(torch.int16), out_d[b, blk].view(torch.int16)
            ), f"sequence {b} block {blk}: the paged compress differs from the dense step form"
        for p in range(max(0, lens[b] - cap), lens[b]):
            assert torch.equal(pg.ring_row(b, p).view(torch.int16), k_raw[b, p].view(torch.int16)), f"sequence {b}: ring row of position {p} is not its raw key"
    assert pg.rows_written() == sum(int(n) for n in n_blocks), "rows written != the live blocks"
    assert bool((pg.cache[0].float() == pg.sent).all()) and bool((pg.ring[0].float() == pg.sent).all()), "page 0 / the null ring block were written"


@requires_cuda
@pytest.mark.parametrize("k", [16, 19], ids=["first_row_of_a_page", "last_row_of_a_page"])
def test_a_block_completes_at_the_fourth_token_on_either_side_of_a_page_boundary(k):
    """P = 16 (four compressed rows per page): block ``k`` completes at its 4th token from the step's own rows and lands on row
    ``k % 4`` of page ``block_table[b, k // 4]`` -- the first row of a page at k = 16, the last at k = 19 -- bitwise the prefill row;
    the four raw keys enter the ring; nothing else moves."""
    dtype, s_tok = torch.bfloat16, 128
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=22)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok, s_tok]))
    pg = _Paged(2, s_tok, dtype, page=16, cap=8, seed=22)
    cache_before, ring_before = pg.cache.clone(), pg.ring.clone()
    pos0 = 4 * k
    slot = _step_paged(pg, k_raw[:, pos0 : pos0 + 4].contiguous(), w, cos, sin, [pos0 + 4] * 2)
    assert slot.tolist() == [[-1, -1, -1, k]] * 2
    for b in range(2):
        assert torch.equal(pg.kbar_row(b, k).view(torch.int16), ref[b, k].view(torch.int16))
        page, row = int(pg.block_table[b, k // 4]), k % 4
        assert row == (0 if k == 16 else 3)
        cache_before[page, row, 0] = pg.cache[page, row, 0]
        for p in range(pos0, pos0 + 4):
            assert torch.equal(pg.ring_row(b, p).view(torch.int16), k_raw[b, p].view(torch.int16))
            ring_before[int(pg.state_table[b, 0]), p % 8, 0] = pg.ring_row(b, p)
    assert torch.equal(pg.cache.view(torch.int16), cache_before.view(torch.int16)) and torch.equal(pg.ring.view(torch.int16), ring_before.view(torch.int16))


@requires_cuda
@pytest.mark.parametrize("cap, rows", [(4, 1), (8, 1), (8, 2), (8, 3), (8, 4)], ids=["cap4_sq1", "cap8_sq1", "cap8_sq2", "cap8_sq3", "cap8_sq4"])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32], ids=["slots_i64", "slots_i32"])
def test_derived_ring_rows_are_the_circular_buffer_slots_and_the_slot_form_is_bitwise(cap, rows, slot_dtype):
    """After one step the ring rows the kernel derived are exactly the stack's own slots ``block x cap + pos % cap`` (``_Paged.ring_slots``,
    the CircularBufferSpec formula; the ring capacities a stack allocates: 4 at one row per step, 8 with speculative rows), and
    the explicit-slot form fed those slots (int64 or int32) is bitwise the derived form -- ring, cache and slot report."""
    dtype, s_tok, batch = torch.float16, 64, 3
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=23)
    pos0 = [4 * 5 + 1, 4 * 3, 4 * 7 + 3]  # the ring rows land on different residues of cap
    k_new = torch.stack([k_raw[b, p : p + rows] for b, p in enumerate(pos0)]).contiguous()
    kv_lens = [p + rows for p in pos0]
    derived = _Paged(batch, s_tok, dtype, page=16, cap=cap, seed=23)
    slot_d = _step_paged(derived, k_new, w, cos, sin, kv_lens)
    flat = derived.ring.view(-1, _D)  # the ring's blocks are contiguous: flat row = block x cap + row
    slots = derived.ring_slots(kv_lens, rows, dtype=slot_dtype)
    for b in range(batch):
        for j in range(rows):
            s = int(slots[b * rows + j])
            assert s == int(derived.state_table[b, 0]) * cap + (pos0[b] + j) % cap
            assert torch.equal(flat[s].view(torch.int16), k_new[b, j].view(torch.int16)), f"sequence {b} row {j}: the derived ring row is not slot {s}"
    n_rows_written = int((derived.ring.float() != derived.sent).any(-1).sum())
    assert n_rows_written == batch * rows
    explicit = _Paged(batch, s_tok, dtype, page=16, cap=cap, seed=23)
    slot_e = _step_paged(explicit, k_new, w, cos, sin, kv_lens, ring_slots=slots)
    assert torch.equal(explicit.ring.view(torch.int16), derived.ring.view(torch.int16)) and torch.equal(
        explicit.cache.view(torch.int16), derived.cache.view(torch.int16)
    )
    assert torch.equal(slot_e, slot_d)


@requires_cuda
def test_kv_lens_zero_rows_write_nothing_and_report_minus_one():
    """A padding sequence (``kv_lens 0``, the null ring block, page 0 in its table row) between two live ones: its rows are dead --
    no ring row, no block, every slot ``-1`` -- and the null block and page 0 keep their sentinel bit for bit; the live sequences
    are served bitwise the dense step form.  The dense form has no such row (a negative ``pos0`` is its first step), so the
    padding sequence there is a finished one (``n_commit 0``)."""
    dtype, s_tok, batch = torch.bfloat16, 96, 3
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=24)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok] * batch))
    pos0 = [4 * 6, 0, 4 * 9]  # the padding sequence's pos0 is irrelevant: kv_lens 0 -> every position negative
    kv_lens = [pos0[0] + 4, 0, pos0[2] + 4]
    k_new = torch.stack([k_raw[b, p : p + 4] for b, p in enumerate(pos0)]).contiguous()
    k_new[1] = torch.randn(4, _D, device="cuda").to(dtype)  # whatever a padding row carries must never be read into a write
    pg = _Paged(batch, s_tok, dtype, page=16, cap=8, padding=(1,), seed=24)
    slot = _step_paged(pg, k_new, w, cos, sin, kv_lens)
    assert slot.tolist() == [[-1, -1, -1, 6], [-1, -1, -1, -1], [-1, -1, -1, 9]]
    assert bool((pg.ring[0].float() == pg.sent).all()), "the null ring block was written"
    assert bool((pg.cache[0].float() == pg.sent).all()), "page 0 was written"
    assert pg.rows_written() == 2
    for b, k in ((0, 6), (2, 9)):
        assert torch.equal(pg.kbar_row(b, k).view(torch.int16), ref[b, k].view(torch.int16))
    # the dense twin on the live sequences (the padding one as a finished sequence): the same rows, bit for bit
    ring_d = _ring_from_prefix(k_raw, pos0, _RING)
    out_d = _sentinel(batch, s_tok // _POOL, _D, dtype=dtype)
    slot_d = _step(k_new, ring_d, out_d, w, cos, sin, _i32(pos0), _i32([4, 0, 4]))
    assert slot_d.tolist() == slot.tolist()
    for b, k in ((0, 6), (2, 9)):
        assert torch.equal(pg.kbar_row(b, k).view(torch.int16), out_d[b, k].view(torch.int16))


@requires_cuda
def test_rows_below_position_zero_are_dead_and_the_rest_are_served():
    """``kv_lens < S_q`` (a sequence's very first step has fewer real tokens than the step's rows): the rows at negative positions are
    dead (``-1``), the rows at positions ``0 .. kv_lens - 1`` enter the ring, and the block completing at position 3 is pooled from
    the step's own rows -- bitwise the prefill compress of those tokens."""
    dtype, s_tok = torch.bfloat16, 64
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=25)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok]))
    rows = 5
    k_new = torch.full((1, rows, _D), float("nan"), device="cuda", dtype=dtype)
    k_new[0, 1:] = k_raw[0, :4]  # kv_lens 4 with 5 rows: row 0 sits at position -1
    pg = _Paged(1, s_tok, dtype, page=16, cap=8, seed=25)
    slot = _step_paged(pg, k_new, w, cos, sin, [4])
    assert slot.tolist() == [[-1, -1, -1, -1, 0]]
    assert torch.equal(pg.kbar_row(0, 0).view(torch.int16), ref[0, 0].view(torch.int16))
    for p in range(4):
        assert torch.equal(pg.ring_row(0, p).view(torch.int16), k_raw[0, p].view(torch.int16))
    assert int((pg.ring.float() != pg.sent).any(-1).sum()) == 4 and pg.rows_written() == 1


@requires_cuda
def test_paged_capacity_reports_minus_two_and_writes_nothing():
    """A completing row whose block lies past the block table's width, on a negative (unassigned) page, or on a page past the
    cache reports ``-2`` and writes no block; its raw key still enters the ring (it is committed).  A step at or past the RoPE table
    reports ``-2`` for every row and writes nothing at all (the table ceiling, the dense form's rule)."""
    dtype, s_tok = torch.bfloat16, 128
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=26)
    k = 7
    pos0 = 4 * k
    k_new = k_raw[:, pos0 : pos0 + 4].contiguous()
    # (a) a table of 1 page (4 compressed rows) holds blocks 0 .. 3 only
    pg = _Paged(1, s_tok, dtype, page=16, cap=8, seed=26, table_pages=1)
    assert pg.block_table.shape == (1, 1)
    slot = _step_paged(pg, k_new, w, cos, sin, [pos0 + 4])
    assert slot.tolist() == [[-1, -1, -1, SLOT_CAPACITY]] and pg.rows_written() == 0
    assert torch.equal(pg.ring_row(0, pos0 + 3).view(torch.int16), k_raw[0, pos0 + 3].view(torch.int16))
    # (b) the block's page is unassigned (-1)
    pg = _Paged(1, s_tok, dtype, page=16, cap=8, seed=26)
    pg.block_table[0, k // 4] = -1
    slot = _step_paged(pg, k_new, w, cos, sin, [pos0 + 4])
    assert slot.tolist() == [[-1, -1, -1, SLOT_CAPACITY]] and pg.rows_written() == 0
    # (c) the block's page lies past the cache
    pg = _Paged(1, s_tok, dtype, page=16, cap=8, seed=26)
    pg.block_table[0, k // 4] = pg.n_pages + 5
    slot = _step_paged(pg, k_new, w, cos, sin, [pos0 + 4])
    assert slot.tolist() == [[-1, -1, -1, SLOT_CAPACITY]] and pg.rows_written() == 0
    # (d) the RoPE table ends AT the block's start position: the table is the ceiling on positions (the dense form's rule) -- every row of
    # the step reports -2 and writes nothing, no ring row either
    pg = _Paged(1, s_tok, dtype, page=16, cap=8, seed=26)
    slot = _step_paged(pg, k_new, w, cos[:, :pos0].contiguous(), sin[:, :pos0].contiguous(), [pos0 + 4])
    assert slot.tolist() == [[SLOT_CAPACITY] * 4] and pg.rows_written() == 0
    assert bool((pg.ring.float() == pg.sent).all()), "a row at or past the table wrote its ring row"


@requires_cuda
@pytest.mark.parametrize("paged", [False, True], ids=["dense_state", "paged_state"])
def test_a_row_at_or_past_the_rope_table_reports_minus_two_and_writes_nothing(paged):
    """The RoPE table is the CEILING on positions, checked on the LIVE (device-derived) position of EVERY committed row, completing or
    not.  A 16-position compressed-key cache (dense: 4 blocks; paged: one page of P 16 behind a 1-wide block table) with a table of
    exactly those 16 rows: a one-row step at position 16 (``pos0 16``; paged ``kv_lens 17``) reports ``-2`` and writes NOTHING -- its
    ring row keeps the prefix key it held, no block -- where the previous kernel reported ``-1`` and wrote the ring row; the in-bounds
    neighbour (position 15; ``kv_lens 16``) completes block 3 from the ring bitwise the prefill, slot 3, its ring row written; a four-row
    step straddling the ceiling (positions 14 .. 17) is ``[-1, 3, -2, -2]`` with ring rows for 14 / 15 only.  The cache capacity is NOT
    the ceiling: with the full table a position past the 16-position cache is served (``-1``, its ring row written) and a block past the
    cache stays ``-2`` by the cache rule with the row's ring row written (the position itself is served)."""
    dtype, s_tok = torch.bfloat16, 64
    k_raw, w, cos, sin = _make(1, s_tok, dtype, seed=31)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok]))
    n_pos = 16
    cos16, sin16 = cos[:, :n_pos].contiguous(), sin[:, :n_pos].contiguous()
    sent = _sent_of(dtype)

    def same(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    class State:
        """One fresh 16-position state (the ring pre-filled with the positions below ``pos0``), dense or paged, behind one accessor set."""

        def __init__(self, pos0):
            if paged:
                self.pg = _Paged(1, s_tok, dtype, page=16, cap=8, seed=31, table_pages=1)
                assert self.pg.block_table.shape == (1, 1)
                self.pg.fill_ring_from_prefix(k_raw, [pos0])
            else:
                self.ring = _ring_from_prefix(k_raw, [pos0], _RING)
                self.out = _sentinel(1, n_pos // _POOL, _D, dtype=dtype)

        def step(self, pos0, rows, table):
            k_new = k_raw[:, pos0 : pos0 + rows].contiguous()
            if paged:
                return _step_paged(self.pg, k_new, w, *table, [pos0 + rows]).tolist()
            return _step(k_new, self.ring, self.out, w, *table, _i32([pos0])).tolist()

        def ring_clone(self):
            return (self.pg.ring if paged else self.ring).clone()

        def ring_equal(self, other):
            return same(self.pg.ring if paged else self.ring, other)

        def ring_row(self, p):
            return self.pg.ring_row(0, p) if paged else self.ring[0, p % _RING]

        def kbar(self, blk):
            return self.pg.kbar_row(0, blk) if paged else self.out[0, blk]

        def blocks_written(self):
            return self.pg.rows_written() if paged else int((self.out.float() != sent).any(-1).sum())

    # (a) the row AT the ceiling: -2, not one byte written (its ring row keeps the prefix key of position 8)
    st = State(16)
    before = st.ring_clone()
    assert st.step(16, 1, (cos16, sin16)) == [[SLOT_CAPACITY]]
    assert st.ring_equal(before), "the ring row of a position past the table was written"
    assert st.blocks_written() == 0
    # (b) the neighbour below it: block 3 from the ring, bitwise the prefill; its ring row written
    st = State(15)
    assert st.step(15, 1, (cos16, sin16)) == [[3]]
    assert same(st.kbar(3), ref[0, 3]) and same(st.ring_row(15), k_raw[0, 15]) and st.blocks_written() == 1
    # (c) a four-row step straddling the ceiling: rows 14 / 15 served (block 3), rows 16 / 17 -2 with their ring rows untouched
    st = State(14)
    assert st.step(14, 4, (cos16, sin16)) == [[-1, 3, SLOT_CAPACITY, SLOT_CAPACITY]]
    assert same(st.kbar(3), ref[0, 3]) and st.blocks_written() == 1
    assert same(st.ring_row(14), k_raw[0, 14]) and same(st.ring_row(15), k_raw[0, 15])
    assert same(st.ring_row(16), k_raw[0, 8]) and same(st.ring_row(17), k_raw[0, 9]), "ring rows 16 % 8 / 17 % 8 must still hold positions 8 / 9"
    # (d) the cache capacity is NOT the ceiling: the full table serves a position past the 16-position cache (-1, its ring row written);
    # a block past the cache stays -2 by the cache rule, the committed row's key still in the ring
    st = State(16)
    assert st.step(16, 1, (cos, sin)) == [[-1]]
    assert same(st.ring_row(16), k_raw[0, 16]) and st.blocks_written() == 0
    st = State(19)
    assert st.step(19, 1, (cos, sin)) == [[SLOT_CAPACITY]]
    assert same(st.ring_row(19), k_raw[0, 19]) and st.blocks_written() == 0


@requires_cuda
def test_state_entry_past_the_ring_or_null_owns_no_ring_and_a_ring_fed_block_reports_minus_two():
    """A LIVE sequence whose state-table entry is one past the ring's last block (the off-by-one of a ring allocated one block per
    request while the ids start at 1), the null block (0) or negative owns no ring: not one ring byte changes (the rows past the
    allocation included), a block pooled from this step's rows alone is still written -- bitwise the prefill -- and a block that
    needs ring rows the sequence does not own reports ``-2`` and leaves its page row at the sentinel: the null block's rows are
    never pooled into a live sequence's block.  The same step on an owned ring writes that block."""
    dtype, s_tok, batch = torch.bfloat16, 128, 3
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=30)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok] * batch))
    pg = _Paged(batch, s_tok, dtype, page=16, cap=8, seed=30)
    pg.state_table[0, 0] = int(pg.ring.shape[0])  # one past the last ring block
    pg.state_table[1, 0] = 0  # the null block, with a LIVE kv_lens
    pg.state_table[2, 0] = -3  # unassigned
    ring_before = pg.ring.clone()
    # (a) S_q 4 at a block-aligned pos0: the completing block pools this step's four rows only -> written; no ring row anywhere
    k = 5
    pos0 = 4 * k
    slot = _step_paged(pg, k_raw[:, pos0 : pos0 + 4].contiguous(), w, cos, sin, [pos0 + 4] * batch)
    assert slot.tolist() == [[-1, -1, -1, k]] * batch
    assert torch.equal(pg.ring.view(torch.int16), ring_before.view(torch.int16)), "a sequence that owns no ring wrote a ring row"
    for b in range(batch):
        assert torch.equal(pg.kbar_row(b, k).view(torch.int16), ref[b, k].view(torch.int16))
    assert pg.rows_written() == batch
    # (b) S_q 1 at pos0 = 4 k2 + 3: the completing block needs ring rows 4 k2 .. 4 k2 + 2 the sequence does not own -> -2, nothing written
    k2 = 9
    pos1 = 4 * k2 + 3
    slot = _step_paged(pg, k_raw[:, pos1 : pos1 + 1].contiguous(), w, cos, sin, [pos1 + 1] * batch)
    assert slot.tolist() == [[SLOT_CAPACITY]] * batch
    assert torch.equal(pg.ring.view(torch.int16), ring_before.view(torch.int16))
    assert pg.rows_written() == batch, "(b) wrote a block from rows the sequence does not own"
    for b in range(batch):
        assert bool((pg.kbar_row(b, k2).float() == pg.sent).all())
    # the rule is about ownership, not the block: the same step on sequences that own their ring blocks writes block k2 bitwise the prefill
    owned = _Paged(batch, s_tok, dtype, page=16, cap=8, seed=30)
    owned.fill_ring_from_prefix(k_raw, [pos1] * batch)
    slot = _step_paged(owned, k_raw[:, pos1 : pos1 + 1].contiguous(), w, cos, sin, [pos1 + 1] * batch)
    assert slot.tolist() == [[k2]] * batch
    for b in range(batch):
        assert torch.equal(owned.kbar_row(b, k2).view(torch.int16), ref[b, k2].view(torch.int16))
        assert torch.equal(owned.ring_row(b, pos1).view(torch.int16), k_raw[b, pos1].view(torch.int16))


@requires_cuda
def test_one_paged_artifact_serves_every_page_size_and_cap():
    """One compile per (dtype, d, rope_dim, pool, n_commit, paged, ring_slots): the page geometry, the ring capacity, the batch and the
    rows ride in as runtime shapes."""
    dtype = torch.bfloat16
    r = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False, paged=True)
    assert isinstance(r, QsaCompressStepRecipe) and r.paged and not r.ring_slots
    assert compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False, paged=True).compiled is r.compiled
    assert compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False).compiled is not r.compiled
    k_raw, w, cos, sin = _make(2, 256, dtype, seed=27)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([256, 256]))
    for page, cap, rows in ((16, 4, 1), (32, 8, 3), (64, 8, 5), (16, 8, 4)):
        pg = _Paged(2, 256, dtype, page=page, cap=cap, seed=27)
        pos0 = 4 * 20 + (3 - (rows - 1) % 4)  # the last row completes a block, pooling the ring's rows below pos0
        pg.fill_ring_from_prefix(k_raw, [pos0, pos0])
        slot = torch.full((2, rows), -9, dtype=torch.int32, device="cuda")
        run_qsa_compress_step(
            r,
            k_raw[:, pos0 : pos0 + rows].contiguous(),
            pg.ring,
            pg.cache,
            w,
            cos,
            sin,
            None,
            slot,
            None,
            eps=_EPS,
            norm_weight_offset=1.0,
            stream=_stream(),
            kv_lens=_i32([pos0 + rows] * 2),
            block_table=pg.block_table,
            index_state_block_table=pg.state_table,
        )
        torch.cuda.synchronize()
        blk = (pos0 + rows - 1) // _POOL
        assert slot[:, rows - 1].tolist() == [blk, blk], (page, cap, rows)
        for b in range(2):
            assert torch.equal(pg.kbar_row(b, blk).view(torch.int16), ref[b, blk].view(torch.int16)), (page, cap, rows, b)


@requires_cuda
def test_paged_step_is_one_launch_and_allocates_nothing():
    """MEASURED: the paged step is ONE kernel launch (``torch.profiler``'s census), it allocates nothing and never synchronises
    (``torch.cuda.set_sync_debug_mode("error")`` around the launch; the caching allocator's allocation count unchanged)."""
    from torch.profiler import ProfilerActivity, profile

    dtype, s_tok = torch.bfloat16, 64
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=28)
    pg = _Paged(2, s_tok, dtype, page=16, cap=8, seed=28)
    r = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=True, paged=True)
    k_new = k_raw[:, 8:12].contiguous()
    slot = torch.full((2, 4), -9, dtype=torch.int32, device="cuda")
    kv, nc = _i32([12, 12]), _i32([4, 4])

    def launch():
        run_qsa_compress_step(
            r,
            k_new,
            pg.ring,
            pg.cache,
            w,
            cos,
            sin,
            None,
            slot,
            nc,
            eps=_EPS,
            stream=_stream(),
            kv_lens=kv,
            block_table=pg.block_table,
            index_state_block_table=pg.state_table,
        )

    launch()  # warm: the first call pays nothing either, but the census below must count kernels only
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            launch()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, "the paged step allocated on the execute path"
    prof = profile(activities=[ProfilerActivity.CUDA])
    try:
        prof.start()
    except Exception:  # noqa: BLE001 -- CUPTI unavailable on this box: the bitwise cells stand
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        prof.stop()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "memset" not in e.name.lower() and "memcpy" not in e.name.lower()
    ]
    if not names:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    assert len(names) == 1 and "frost_qsa_compress_step" in names[0], names


# ---------------------------------------------------------------------------
# Host-side contract of the paged form: every refusal typed, both directions
# ---------------------------------------------------------------------------


@requires_cuda
def test_paged_run_rejects_contract_violations():
    dtype, s_tok, batch, rows = torch.bfloat16, 64, 2, 4
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=29)
    pg = _Paged(batch, s_tok, dtype, page=16, cap=8, seed=29)
    k_new = k_raw[:, :rows].contiguous()
    slot = torch.zeros(batch, rows, dtype=torch.int32, device="cuda")
    kv = _i32([rows, rows])
    paged = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False, paged=True)
    slots_r = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False, paged=True, ring_slots=True, ring_slot_dtype=torch.int64)
    dense = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False)

    def run(r, k=k_new, rg=None, o=None, p=None, sl=slot, kvl=kv, bt=None, st=None, sm=None):
        rg = pg.ring if rg is None else rg
        o = pg.cache if o is None else o
        bt = pg.block_table if bt is None else bt
        st = pg.state_table if st is None else st
        run_qsa_compress_step(
            r,
            k,
            rg,
            o,
            w,
            cos,
            sin,
            p,
            sl,
            None,
            eps=_EPS,
            stream=_stream(),
            kv_lens=kvl,
            block_table=bt,
            index_state_block_table=st,
            index_state_slot_mapping=sm,
        )

    # the position source, both directions
    with pytest.raises(ValueError, match="pos0 is refused on the paged form"):
        run(paged, p=_i32([0, 0]))
    with pytest.raises(ValueError, match="kv_lens .* block_table .* index_state_block_table .* must be bound"):
        run(paged, kvl=None)
    with pytest.raises(ValueError, match="compiled WITHOUT the paged compress arms"):
        run_qsa_compress_step(
            dense,
            k_new,
            _sentinel(batch, _RING, _D, dtype=dtype),
            _sentinel(batch, 16, _D, dtype=dtype),
            w,
            cos,
            sin,
            _i32([0, 0]),
            slot,
            None,
            eps=_EPS,
            stream=_stream(),
            kv_lens=kv,
        )
    with pytest.raises(ValueError, match="the dense form needs pos0"):
        run_qsa_compress_step(
            dense,
            k_new,
            _sentinel(batch, _RING, _D, dtype=dtype),
            _sentinel(batch, 16, _D, dtype=dtype),
            w,
            cos,
            sin,
            None,
            slot,
            None,
            eps=_EPS,
            stream=_stream(),
        )
    # the ring: 4-D, the capacity rule, the dtype, the alignment
    with pytest.raises(ValueError, match=r"ring must be a 4-D \[blocks >= 1, rows, 1, D=128\]"):
        run(paged, rg=_sentinel(batch, 8, _D, dtype=dtype))
    with pytest.raises(ValueError, match="ring holds 4 rows per block; this launch needs at least 7"):
        run(paged, rg=_sentinel(batch + 1, 4, 1, _D, dtype=dtype))
    with pytest.raises(ValueError, match="dtype"):
        run(paged, rg=_sentinel(batch + 1, 8, 1, _D, dtype=torch.float16))
    with pytest.raises(ValueError, match="16-B aligned"):
        n = (batch + 1) * 8 * (_D + 8)
        run(paged, rg=torch.empty(n + 8, device="cuda", dtype=dtype)[4 : 4 + n].view(batch + 1, 8, 1, _D + 8)[..., :_D])  # an 8-byte base offset
    # the cache: 4-D, the recipe's dtype BOTH ways (INVERTED 2026-10-21: the e4m3 store arm is served -- an e4m3 cache under the bf16 recipe
    # and a bf16 cache under the e4m3 recipe each name the recipe's dtype; an e4m3 RING stays the ring's own rule), the alignment
    with pytest.raises(ValueError, match=r"out must be a 4-D \[blocks >= 1, rows, 1, D=128\]"):
        run(paged, o=_sentinel(batch, 16, _D, dtype=dtype))
    with pytest.raises(ValueError, match=r"out is float8_e4m3fn but this artifact stores the compressed-key cache in torch.bfloat16"):
        run(paged, o=torch.empty(pg.n_pages, pg.rpp, 1, _D, device="cuda", dtype=torch.float8_e4m3fn))
    if _CC >= (8, 9):
        e4 = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False, paged=True, index_cache_dtype=torch.float8_e4m3fn)
        assert e4.index_cache_dtype == torch.float8_e4m3fn and e4.compiled is not paged.compiled
        with pytest.raises(ValueError, match=r"out must have dtype torch.float8_e4m3fn \(the recipe's\), got torch.bfloat16"):
            run(e4)
    with pytest.raises(ValueError, match="raw-key ring is a bit copy .* never quantized"):  # an e4m3 RING is a form error, not the cache's dtype
        run(paged, rg=torch.empty(batch + 1, 8, 1, _D, device="cuda", dtype=torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        run(paged, o=torch.empty(pg.n_pages, pg.rpp, 1, _D + 4, device="cuda", dtype=dtype)[..., :_D])
    # the tables and the lengths
    with pytest.raises(ValueError, match="block_table must be a 2-D int32 CUDA block table"):
        run(paged, bt=pg.block_table.to(torch.int64))
    with pytest.raises(ValueError, match=r"block_table must be \[B=2, cols >= 1\]"):
        run(paged, bt=pg.block_table[:1])
    with pytest.raises(ValueError, match="contiguous columns"):
        run(paged, bt=pg.block_table.t().contiguous().t())
    with pytest.raises(ValueError, match=r"index_state_block_table must be \[B=2, 1\]"):
        run(paged, st=torch.zeros(batch, 2, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="kv_lens must be a contiguous int32"):
        run(paged, kvl=kv.to(torch.int64))
    # the slot-mapping arm, both directions
    with pytest.raises(ValueError, match="bind index_state_slot_mapping"):
        run(slots_r)
    with pytest.raises(ValueError, match="index_state_slot_mapping would be silently ignored"):
        run(paged, sm=pg.ring_slots([rows, rows], rows))
    with pytest.raises(ValueError, match="index_state_slot_mapping must be a contiguous torch.int64"):
        run(slots_r, sm=pg.ring_slots([rows, rows], rows, dtype=torch.int32))
    with pytest.raises(ValueError, match="needs the ring's blocks contiguous"):
        gapped = _sentinel(batch + 1, 10, 1, _D, dtype=dtype)[:, :8]  # 8 of 10 rows per block: block stride != cap x row stride
        run(slots_r, rg=gapped, sm=pg.ring_slots([rows, rows], rows))
    # the row-count rule stays
    with pytest.raises(ValueError, match="k_new must hold at least one row"):
        run(paged, k=k_raw[:, :0].contiguous(), sl=torch.zeros(batch, 0, dtype=torch.int32, device="cuda"))


def test_paged_compile_rejects_unsupported_declarations():
    with pytest.raises(ValueError, match="needs paged=True"):
        compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, ring_slots=True)
    with pytest.raises(ValueError, match="ring_slot_dtype must be int32 or int64"):
        compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, paged=True, ring_slots=True, ring_slot_dtype=torch.int16)
    # INVERTED 2026-10-21: the e4m3 store arm compiles on sm_89+ (a distinct artifact, keyed; the bf16 spelling of the io dtype is the SAME
    # artifact as None) and is a typed decline naming the instruction below it; float32 stays a ValueError; f16 io has no e4m3 arm
    with pytest.raises(ValueError, match="must hold the io dtype .* or torch.float8_e4m3fn"):
        compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, paged=True, index_cache_dtype=torch.float32)
    with pytest.raises(ValueError, match="needs bf16 io"):
        compile_qsa_compress_step(dtype=torch.float16, d=_D, rope_dim=_ROPE, paged=True, index_cache_dtype=torch.float8_e4m3fn)
    if _CC is not None and _CC < (8, 9):
        with pytest.raises(NotImplementedError, match=r"cvt.rn.satfinite.e4m3x2.f32\) needs sm_89 or later"):
            compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, paged=True, index_cache_dtype=torch.float8_e4m3fn)
    elif _CC is not None:
        bf = compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, paged=True)
        assert compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, paged=True, index_cache_dtype=torch.bfloat16).compiled is bf.compiled
        e4 = compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, paged=True, index_cache_dtype=torch.float8_e4m3fn)
        e4d = compile_qsa_compress_step(dtype=torch.bfloat16, d=_D, rope_dim=_ROPE, paged=False, index_cache_dtype=torch.float8_e4m3fn)
        assert e4.compiled is not bf.compiled and e4d.compiled is not e4.compiled and e4.index_cache_dtype == e4d.index_cache_dtype == torch.float8_e4m3fn
        assert bf.index_cache_dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# The e4m3 store arm (sm_89+): the dense and the paged step arms write the e4m3 cast of the bf16 row
# ---------------------------------------------------------------------------


@requires_sm89
@pytest.mark.parametrize("rounding", ["single", "hf"])
@pytest.mark.parametrize("page", [16, 32, 64], ids=["p16", "p32", "p64"])
@pytest.mark.parametrize("schedule", [[1], [4], [1, 4, 2, 3, 5, 1, 2, 4]], ids=["one_row_steps", "four_row_steps", "mixed_steps"])
def test_e4m3_step_stream_dense_and_paged_are_bitwise_the_bf16_rows_cast(schedule, page, rounding):
    """The e4m3 store arm (``index_cache_dtype=torch.float8_e4m3fn``) on the SAME token stream as the bf16 arm, dense AND paged: every
    compressed row the two e4m3 arms write is bitwise each other and bitwise the bf16 arm's row ``.to(float8_e4m3fn)`` (the e4m3 cast of
    the bf16 row -- the serving stacks' convention and the prefill form's), with ``|bf16 row| < 448`` asserted first; the slot reports
    identical step by step; the ring (bf16, a bit copy) identical to the bf16 arm's; the padding sequence writes nothing; page 0 and the
    null ring block keep their sentinels (the e4m3 cache's sentinel is -448, the dtype's max magnitude)."""
    dtype, cap = torch.bfloat16, 8
    s_tok, lens = 517, [517, 510, 37, 0]
    batch = len(lens)
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=41)
    ref, n_blocks = _prefill(k_raw, w, cos, sin, _i32(lens), offset=1.0, rounding=rounding)
    out16, ring16, slots16 = _run_stream(k_raw, lens, schedule, w=w, cos=cos, sin=sin, offset=1.0, rounding=rounding)
    out8, ring8, slots8 = _run_stream(k_raw, lens, schedule, w=w, cos=cos, sin=sin, offset=1.0, cache_dtype=_E4M3, rounding=rounding)
    pg, slots_p = _run_stream_paged(
        k_raw, lens, schedule, w=w, cos=cos, sin=sin, offset=1.0, page=page, cap=cap, padding=(3,), seed=41, cache_dtype=_E4M3, rounding=rounding
    )
    assert out8.dtype == pg.cache.dtype == _E4M3 and ring8.dtype == pg.ring.dtype == dtype
    assert slots8 == slots16 and slots_p == slots16, "the e4m3 arms' slot reports differ from the bf16 arm's"
    assert torch.equal(_bits(ring8), _bits(ring16)), "the e4m3 arm changed the ring (a bf16 bit copy)"
    for b in range(batch):
        nb = int(n_blocks[b])
        if nb:
            assert float(ref[b, :nb].float().abs().max()) < 448.0, f"sequence {b}: a bf16 row at or past the e4m3 range"
            want = ref[b, :nb].to(_E4M3)
            assert torch.equal(_bits(out8[b, :nb]), _bits(want)), f"sequence {b}: the dense e4m3 arm is not the e4m3 cast of the bf16 rows"
            assert torch.equal(_bits(out16[b, :nb].to(_E4M3)), _bits(want))
        for blk in range(nb):
            row = pg.kbar_row(b, blk)
            assert torch.equal(_bits(row), _bits(out8[b, blk])), f"sequence {b} block {blk}: the paged e4m3 arm differs from the dense e4m3 arm"
        for p in range(max(0, lens[b] - cap), lens[b]):
            assert torch.equal(_bits(pg.ring_row(b, p)), _bits(k_raw[b, p])), f"sequence {b}: ring row of position {p} is not its raw key"
    sent8 = _sent_of(_E4M3)
    assert sent8 == -448.0 and pg.sent == sent8
    assert bool((out8[3].float() == sent8).all()) and bool((ring8[3] == _sent_of(dtype)).all())
    assert pg.rows_written() == sum(int(n) for n in n_blocks) == int((out8.float() != sent8).any(-1).sum())
    assert bool((pg.cache[0].float() == sent8).all()) and bool((pg.ring[0].float() == _sent_of(dtype)).all()), "page 0 / the null ring block were written"


@requires_sm89
@pytest.mark.parametrize("k", [16, 19], ids=["first_row_of_a_page", "last_row_of_a_page"])
def test_e4m3_block_completes_at_the_fourth_token_on_either_side_of_a_page_boundary(k):
    """The e4m3 paged arm at P 16: block ``k`` completes at its 4th token on the first (k = 16) or the last (k = 19) row of a page,
    bitwise the prefill row's e4m3 cast; the four raw keys enter the bf16 ring; nothing else moves (the 8-bit sentinel elsewhere)."""
    dtype, s_tok = torch.bfloat16, 128
    k_raw, w, cos, sin = _make(2, s_tok, dtype, seed=42)
    ref, _ = _prefill(k_raw, w, cos, sin, _i32([s_tok, s_tok]))
    pg = _Paged(2, s_tok, dtype, page=16, cap=8, seed=42, cache_dtype=_E4M3)
    cache_before, ring_before = pg.cache.clone(), pg.ring.clone()
    pos0 = 4 * k
    slot = _step_paged(pg, k_raw[:, pos0 : pos0 + 4].contiguous(), w, cos, sin, [pos0 + 4] * 2)
    assert slot.tolist() == [[-1, -1, -1, k]] * 2
    for b in range(2):
        assert float(ref[b, k].float().abs().max()) < 448.0
        assert torch.equal(_bits(pg.kbar_row(b, k)), _bits(ref[b, k].to(_E4M3)))
        page, row = int(pg.block_table[b, k // 4]), k % 4
        assert row == (0 if k == 16 else 3)
        cache_before[page, row, 0] = pg.cache[page, row, 0]
        for p in range(pos0, pos0 + 4):
            assert torch.equal(_bits(pg.ring_row(b, p)), _bits(k_raw[b, p]))
            ring_before[int(pg.state_table[b, 0]), p % 8, 0] = pg.ring_row(b, p)
    assert torch.equal(_bits(pg.cache), _bits(cache_before)) and torch.equal(_bits(pg.ring), _bits(ring_before))


# ---------------------------------------------------------------------------
# The CHUNK form (ring_write=False): a prefill chunk behind a cached prefix, the ring read-only
# ---------------------------------------------------------------------------

_CHUNK_RESIDUE_PREFIXES = [20, 21, 22, 23]  # one live sequence per residue mod 4 (every prefix >= cap and >= 4), plus a padding sequence
_CHUNK_T = [1, 2, 3, 4, 5, 17, 128, 2048]


def _run_chunk_stream(k_raw, prefixes, t, n_chunks, *, w, cos, sin, page, cap, padding=(), rounding="single", cache_dtype=None, offset=1.0):
    """The chunk form over the paged side caches: every live sequence ``b`` starts behind its prefix ``prefixes[b]`` and receives
    ``n_chunks`` chunks of ``t`` rows (the uniform-T contract); before every chunk the TEST primes the ring with the stream's last ``cap``
    positions -- the tail arm's job (``cache_write`` over the ring's slots, pinned in ``test_qsa_prepare.py``) -- with the positions below
    ``4 floor(P / 4)`` POISONED (NaN: the kernel must never read them), and the prefix's complete blocks sit in the cache from the
    from-scratch compress (the cache-parity precondition).  Returns the paged caches, the reference, the per-chunk slot reports and, per
    chunk, whether the ring came back bitwise untouched."""
    batch, s_tot, _ = k_raw.shape
    lens = [0 if b in padding else prefixes[b] + n_chunks * t for b in range(batch)]
    ref, _ = _prefill(k_raw, w, cos, sin, _i32(lens), offset=offset, rounding=rounding)
    ref_c = ref if cache_dtype is None else ref.to(cache_dtype)
    pg = _Paged(batch, s_tot, k_raw.dtype, page=page, cap=cap, padding=padding, seed=t + page + cap, cache_dtype=cache_dtype)
    nan_row = torch.full((_D,), float("nan"), device="cuda", dtype=k_raw.dtype)
    for b in range(batch):
        if b in padding:
            continue
        for blk in range(prefixes[b] // _POOL):  # the prefix's complete blocks: the cache holds them already
            pg.cache[int(pg.block_table[b, blk // pg.rpp]), blk % pg.rpp, 0] = ref_c[b, blk]
    slots, ring_untouched = [], []
    for c in range(n_chunks):
        kv_lens = []
        for b in range(batch):
            if b in padding:
                kv_lens.append(0)
                continue
            kv_before = prefixes[b] + c * t
            kv_lens.append(kv_before + t)
            entry = int(pg.state_table[b, 0])
            for p in range(kv_before - cap, kv_before):  # the ring INPUT contract: the last cap positions, the unread ones poisoned
                pg.ring[entry, p % cap, 0] = k_raw[b, p] if p >= _POOL * (kv_before // _POOL) else nan_row
        ring_before = pg.ring.clone()
        k_new = torch.stack([k_raw[b, prefixes[b] + c * t : prefixes[b] + (c + 1) * t] for b in range(batch)]).contiguous()
        slot = _step_paged(pg, k_new, w, cos, sin, kv_lens, None, offset=offset, ring_write=False, rounding=rounding)
        slots.append(slot.tolist())
        ring_untouched.append(torch.equal(_bits(pg.ring), _bits(ring_before)))
    return pg, ref_c, slots, ring_untouched, lens


def _assert_chunk_stream(pg: _Paged, ref_c, slots, ring_untouched, prefixes, t, n_chunks, padding, label):
    """After the LAST chunk: every complete block of every live sequence bitwise the from-scratch compress, nothing past it, page 0 and
    the null block untouched, the padding sequence untouched; per chunk: the ring untouched and the slot reports the position rule's."""
    batch = pg.batch
    assert all(ring_untouched), f"{label}: the ring was written by a ring_write=False launch (chunks {[i for i, ok in enumerate(ring_untouched) if not ok]})"
    for c in range(n_chunks):
        for b in range(batch):
            if b in padding:
                assert slots[c][b] == [-1] * t, f"{label}: the padding sequence reported a block"
                continue
            kv_before = prefixes[b] + c * t
            want = [((kv_before + j) // _POOL if (kv_before + j) % _POOL == _POOL - 1 else -1) for j in range(t)]
            assert slots[c][b] == want, f"{label}: chunk {c} sequence {b} (P {prefixes[b]}): slots {slots[c][b][:12]} != {want[:12]}"
            if t < _POOL - kv_before % _POOL:  # a chunk shorter than the group remainder: no block completes
                assert slots[c][b] == [-1] * t
    n_written = 0
    for b in range(batch):
        if b in padding:
            continue
        kv_end = prefixes[b] + n_chunks * t
        for blk in range(kv_end // _POOL):
            assert torch.equal(_bits(pg.kbar_row(b, blk)), _bits(ref_c[b, blk])), f"{label}: sequence {b} block {blk} differs from the from-scratch compress"
        n_written += kv_end // _POOL
        if (kv_end // _POOL) < int(ref_c.shape[1]):  # the partially filled group's row (and every later row) untouched
            assert bool((pg.kbar_row(b, kv_end // _POOL).float() == pg.sent).all()), f"{label}: sequence {b}: the incomplete group's row was written"
    assert pg.rows_written() == n_written, f"{label}: rows written {pg.rows_written()} != the complete blocks {n_written}"
    assert bool((pg.cache[0].float() == pg.sent).all()) and bool(
        (pg.ring[0].float() == _sent_of(pg.dtype)).all()
    ), f"{label}: page 0 / the null ring block were written"


@requires_cuda
@pytest.mark.parametrize("cap", [3, 4, 8], ids=["cap3", "cap4", "cap8"])
@pytest.mark.parametrize("page", [16, 32, 64], ids=["p16", "p32", "p64"])
@pytest.mark.parametrize("t", _CHUNK_T)
def test_ring_read_only_chunk_stream_is_bitwise_the_prefill_compress(t, page, cap):
    """Five sequences -- one live sequence per residue of the chunk start (prefixes 20 .. 23) and a padding one -- fed ``n`` chunks of ``t``
    rows through the chunk form over the paged caches at this page size and ring cap (cap 3 = the chunk form's floor ``pool - 1``): after
    every chunk the paged cache rows of every complete block of the stream so far are BITWISE the from-scratch prefill compress (the
    straddling block pooled from the ring + the chunk's rows), the ring BITWISE untouched, the slot reports the position rule's; the
    ``t < 4 - P % 4`` cells (t 1 / 2 at residue 1, t 1 at residue 2, t 1 / 2 / 3 at residue 0) write nothing and report ``-1``; the
    poisoned ring rows below ``4 floor(P / 4)`` are never read (the rows stay bitwise); the padding sequence writes nothing."""
    dtype = torch.bfloat16
    n_chunks = 2 if t >= 128 else 4
    prefixes = _CHUNK_RESIDUE_PREFIXES + [0]
    padding = (4,)
    s_tot = max(prefixes) + n_chunks * t
    k_raw, w, cos, sin = _make(len(prefixes), s_tot, dtype, seed=100 + t)
    pg, ref, slots, ring_ok, _ = _run_chunk_stream(k_raw, prefixes, t, n_chunks, w=w, cos=cos, sin=sin, page=page, cap=cap, padding=padding)
    _assert_chunk_stream(pg, ref, slots, ring_ok, prefixes, t, n_chunks, padding, f"T {t} page {page} cap {cap}")
    # the matrix carries the shorter-than-the-remainder cells by construction: name them
    short = [(b, prefixes[b]) for b in range(4) if t < _POOL - prefixes[b] % _POOL]
    if t <= 3:
        assert short, "the matrix lost its shorter-than-the-remainder cells"


@requires_cuda
@pytest.mark.parametrize("rounding", ["single", "hf"])
@pytest.mark.parametrize("t", [3, 17], ids=["T3", "T17"])
def test_ring_read_only_chunk_stream_under_both_roundings_is_bitwise_the_prefill_compress(t, rounding):
    """The chunk form and the prefill form round identically under ``"hf"`` as under ``"single"`` (page 16, cap 4: vLLM's defaults)."""
    prefixes = _CHUNK_RESIDUE_PREFIXES + [0]
    k_raw, w, cos, sin = _make(len(prefixes), max(prefixes) + 3 * t, torch.bfloat16, seed=300 + t)
    pg, ref, slots, ring_ok, _ = _run_chunk_stream(k_raw, prefixes, t, 3, w=w, cos=cos, sin=sin, page=16, cap=4, padding=(4,), rounding=rounding)
    _assert_chunk_stream(pg, ref, slots, ring_ok, prefixes, t, 3, (4,), f"rounding {rounding} T {t}")


@requires_cuda
def test_ring_write_false_keys_the_artifact_refuses_ring_slots_and_relaxes_the_cap_rule():
    """The recipe's appended ``ring_write`` field; a DISTINCT compile key (the default key unchanged); ``ring_slots=True`` with
    ``ring_write=False`` a ``ValueError``; ``min_ring_rows(rows, pool, ring_write=False) == pool - 1`` whatever the rows; cap 3 served at
    ``S_q = 2048`` (a step form would need 2051), cap 2 refused naming ``pool - 1``; the DENSE form takes the key too and is bitwise the
    paged form on one chunk (the existing twin pin)."""
    dtype = torch.bfloat16
    base = dict(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False)
    rw = compile_qsa_compress_step(paged=True, **base)
    ro = compile_qsa_compress_step(paged=True, ring_write=False, **base)
    assert rw.ring_write is True and ro.ring_write is False and ro.compiled is not rw.compiled
    assert compile_qsa_compress_step(paged=True, ring_write=True, **base).compiled is rw.compiled
    assert compile_qsa_compress_step(paged=True, ring_write=False, **base).compiled is ro.compiled
    with pytest.raises(ValueError, match="ring_slots=True names the ring WRITE rows, but ring_write=False"):
        compile_qsa_compress_step(paged=True, ring_slots=True, ring_write=False, **base)
    assert min_ring_rows(2048, _POOL, ring_write=False) == 3 and min_ring_rows(1, _POOL, ring_write=False) == 3 and min_ring_rows(2048, _POOL) == 2051
    # cap 3 served at 2048 rows; cap 2 refused naming the rule
    t, prefixes = 2048, [21, 22]
    k_raw, w, cos, sin = _make(2, max(prefixes) + t, dtype, seed=400)
    pg3, ref, slots, ring_ok, _ = _run_chunk_stream(k_raw, prefixes, t, 1, w=w, cos=cos, sin=sin, page=16, cap=3)
    _assert_chunk_stream(pg3, ref, slots, ring_ok, prefixes, t, 1, (), "cap 3 at T 2048")
    pg2 = _Paged(2, max(prefixes) + t, dtype, page=16, cap=2, seed=1)
    k_new = torch.stack([k_raw[b, prefixes[b] : prefixes[b] + t] for b in range(2)]).contiguous()
    with pytest.raises(ValueError, match=r"ring holds 2 rows per block; this launch needs at least 3 \(raw-key ring, read-only in the chunk form"):
        _step_paged(pg2, k_new, w, cos, sin, [p + t for p in prefixes], None, ring_write=False)
    # the dense form: the same chunk over a dense ring [B, 3, D] and a dense cache, bitwise the paged form's rows and slots
    ring_d = _sentinel(2, 3, _D, dtype=dtype)
    for b in range(2):
        for p in range(_POOL * (prefixes[b] // _POOL), prefixes[b]):
            ring_d[b, p % 3] = k_raw[b, p]
    ring_d_before = ring_d.clone()
    out_d = _sentinel(2, (max(prefixes) + t) // _POOL, _D, dtype=dtype)
    slot_d = _step(k_new, ring_d, out_d, w, cos, sin, _i32(prefixes), None, ring_write=False)
    assert torch.equal(_bits(ring_d), _bits(ring_d_before))
    assert slot_d.tolist() == slots[0]
    for b in range(2):
        for blk in range(prefixes[b] // _POOL, (prefixes[b] + t) // _POOL):
            assert torch.equal(_bits(out_d[b, blk]), _bits(pg3.kbar_row(b, blk)))
    with pytest.raises(ValueError, match=r"the chunk form needs at least 3 \(raw-key ring, read-only"):
        _step(k_new, _sentinel(2, 2, _D, dtype=dtype), out_d, w, cos, sin, _i32(prefixes), None, ring_write=False)


@requires_cuda
def test_ring_read_only_chunk_is_one_launch_and_allocates_nothing_at_2048_rows():
    """MEASURED: the chunk form at 2048 rows is ONE kernel launch (``torch.profiler``'s census), allocates nothing and never synchronises."""
    from torch.profiler import ProfilerActivity, profile

    dtype, t = torch.bfloat16, 2048
    k_raw, w, cos, sin = _make(2, 22 + t, dtype, seed=500)
    pg = _Paged(2, 22 + t, dtype, page=16, cap=4, seed=500)
    r = compile_qsa_compress_step(dtype=dtype, d=_D, rope_dim=_ROPE, has_n_commit=False, paged=True, ring_write=False)
    k_new = torch.stack([k_raw[b, 21 : 21 + t] for b in range(2)]).contiguous()
    slot = torch.full((2, t), -9, dtype=torch.int32, device="cuda")
    kv = _i32([21 + t, 21 + t])

    def launch():
        run_qsa_compress_step(
            r,
            k_new,
            pg.ring,
            pg.cache,
            w,
            cos,
            sin,
            None,
            slot,
            None,
            eps=_EPS,
            stream=_stream(),
            kv_lens=kv,
            block_table=pg.block_table,
            index_state_block_table=pg.state_table,
        )

    launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            launch()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, "the chunk form allocated on the execute path"
    assert int((slot >= 0).sum()) == 2 * ((21 + t) // _POOL - 21 // _POOL)
    prof = profile(activities=[ProfilerActivity.CUDA])
    try:
        prof.start()
    except Exception:  # noqa: BLE001 -- CUPTI unavailable on this box: the bitwise cells stand
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        prof.stop()
    names = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "memset" not in e.name.lower() and "memcpy" not in e.name.lower()
    ]
    if not names:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable)")
    assert len(names) == 1 and "frost_qsa_compress_step" in names[0], names


@requires_sm89
@pytest.mark.parametrize("t", [2, 5, 128], ids=["T2", "T5", "T128"])
def test_e4m3_ring_read_only_chunk_rows_are_the_bf16_rows_cast(t):
    """The e4m3 cache arm of the chunk form on every residue: the rows the chunk writes are bitwise the bf16 chunk's rows ``.to(e4m3)``
    (= the from-scratch e4m3 compress), the ring stays the bf16 bit copy, the slot reports equal the bf16 arm's."""
    prefixes = _CHUNK_RESIDUE_PREFIXES + [0]
    k_raw, w, cos, sin = _make(len(prefixes), max(prefixes) + 3 * t, torch.bfloat16, seed=600 + t)
    pg8, ref8, slots8, ring_ok8, _ = _run_chunk_stream(k_raw, prefixes, t, 3, w=w, cos=cos, sin=sin, page=16, cap=4, padding=(4,), cache_dtype=_E4M3)
    pg16, ref16, slots16, ring_ok16, _ = _run_chunk_stream(k_raw, prefixes, t, 3, w=w, cos=cos, sin=sin, page=16, cap=4, padding=(4,))
    assert pg8.cache.dtype == _E4M3 and pg8.ring.dtype == torch.bfloat16
    _assert_chunk_stream(pg8, ref8, slots8, ring_ok8, prefixes, t, 3, (4,), f"e4m3 chunk T {t}")
    assert slots8 == slots16
    for b in range(4):
        for blk in range((prefixes[b] + 3 * t) // _POOL):
            assert float(pg16.kbar_row(b, blk).float().abs().max()) < 448.0
            assert torch.equal(_bits(pg8.kbar_row(b, blk)), _bits(pg16.kbar_row(b, blk).to(_E4M3)))


@requires_cuda
def test_paged_kv_lens_at_the_int32_extremes_are_dead_and_never_wrap():
    """``kv_lens`` in {INT_MIN, INT_MIN + 1, -1, 0} on LIVE table rows (a stack bug; a padding entry is the ``0``) beside a ``4`` control, S_q 4:
    every ``L <= 0`` sequence is DEAD -- every slot ``-1``, no ring row, no cache row (the sentinel bit for bit) -- whatever the magnitude, because
    the position is derived from ``max(L, 0) - S_q`` (``step_position``, the kbar gather's derivation): the int32 ``L - S_q`` that wrapped to a live
    positive position at ``INT_MIN`` never enters (before the clamp the wrapped rows landed in the past-the-table path by accident: slots ``-2``,
    nothing written).  The control's row 3 completes block 0 bitwise the prefill compress."""
    dtype, s_tok, batch, rows = torch.bfloat16, 16, 5, 4
    k_raw, w, cos, sin = _make(batch, s_tok, dtype, seed=31)
    int_min = -(2**31)
    kv_lens = [int_min, int_min + 1, -1, 0, 4]
    k_new = k_raw[:, :rows].contiguous()
    pg = _Paged(batch, s_tok, dtype, page=16, cap=8, seed=31)  # every sequence owns pages and a ring block: the dead ones are dead by LENGTH alone
    slot = _step_paged(pg, k_new, w, cos, sin, kv_lens)
    assert slot.tolist() == [[-1] * rows] * 4 + [[-1, -1, -1, 0]], slot.tolist()
    assert pg.rows_written() == 1, "a dead sequence wrote a cache row"
    assert int((pg.ring.float() != pg.sent).any(-1).sum()) == rows, "a dead sequence wrote a ring row"
    for b in range(4):
        assert bool((pg.ring[int(pg.state_table[b, 0]), :, 0].float() == pg.sent).all()), f"sequence {b} (kv_lens {kv_lens[b]}) wrote the ring"
    ref, _ = _prefill(k_raw[4:5], w, cos, sin, _i32([s_tok]))
    assert torch.equal(_bits(pg.kbar_row(4, 0)), _bits(ref[0, 0])), "the control's block 0 is not the prefill compress"
    for p in range(rows):
        assert torch.equal(_bits(pg.ring_row(4, p)), _bits(k_raw[4, p]))
