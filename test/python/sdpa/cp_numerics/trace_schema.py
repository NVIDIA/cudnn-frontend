# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Machine-readable CP trace records and the scheduling invariants over them.

The record fields are the ones the #752 execution plan enumerates::

    rank, ring_step, source_rank, q_global_begin, q_global_end,
    kv_global_begin, kv_global_end, causal_mode, query_half,
    partial_slot_id, lse_update_index, out_update_index,
    partial_dtype, accumulator_dtype, lse_base

Additive fields (``segment``/``section``/``*_global_ranges``/``causal_alignment``)
carry information the minimum set cannot express; they never replace it.

**Global-range convention.** For a CP ring step the token axis handed to the
attention kernel is not always one contiguous global span: TE concatenates the
source rank's two load-balanced chunks (``s`` and ``2P-1-s``) into a single
``[B, Skv, H, D]`` tensor. ``kv_global_ranges`` is therefore the authoritative
ordered list of global spans; ``kv_global_begin`` / ``kv_global_end`` are the
envelope (first span's begin, last span's end) kept because the plan asks for
those two fields. The same convention applies to ``q_global_ranges``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# query halves
QUERY_HALF_BOTH = "both"
QUERY_HALF_FIRST = "first"
QUERY_HALF_SECOND = "second"
QUERY_HALVES = (QUERY_HALF_BOTH, QUERY_HALF_FIRST, QUERY_HALF_SECOND)

# TE section names, transcribed from cp_p2p_fwd_prepare_qkv / the ring loop
SECTION_ALL = "all"
SECTION_DIAGONAL = "diagonal"
SECTION_LOWER = "lower-triangle"
SECTION_UPPER = "upper-triangle"
SECTIONS = (SECTION_ALL, SECTION_DIAGONAL, SECTION_LOWER, SECTION_UPPER)

# causal alignment
ALIGNMENT_TOP_LEFT = "top-left"
ALIGNMENT_BOTTOM_RIGHT = "bottom-right"

Span = Tuple[int, int]


@dataclass(frozen=True)
class TraceRecord:
    """One ring-step event for one rank."""

    rank: int
    ring_step: int
    source_rank: int
    q_global_begin: int
    q_global_end: int
    kv_global_begin: int
    kv_global_end: int
    causal_mode: str
    query_half: str
    partial_slot_id: str
    lse_update_index: int
    out_update_index: int
    partial_dtype: str
    accumulator_dtype: str
    lse_base: str

    # --- additive, documented above -------------------------------------
    section: str = ""
    q_global_ranges: Tuple[Span, ...] = ()
    kv_global_ranges: Tuple[Span, ...] = ()
    causal_alignment: str = ALIGNMENT_TOP_LEFT
    lse_dtype: str = "float32"

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["q_global_ranges"] = [list(x) for x in self.q_global_ranges]
        d["kv_global_ranges"] = [list(x) for x in self.kv_global_ranges]
        return d

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "TraceRecord":
        payload = dict(d)
        payload["q_global_ranges"] = tuple(tuple(int(v) for v in pair) for pair in payload.get("q_global_ranges") or ())
        payload["kv_global_ranges"] = tuple(tuple(int(v) for v in pair) for pair in payload.get("kv_global_ranges") or ())
        return TraceRecord(**payload)  # type: ignore[arg-type]

    def kv_positions(self) -> List[int]:
        return [p for begin, end in self.kv_global_ranges for p in range(begin, end)]

    def q_positions(self) -> List[int]:
        return [p for begin, end in self.q_global_ranges for p in range(begin, end)]


def trace_to_json(records: Sequence[TraceRecord]) -> str:
    return json.dumps([r.as_dict() for r in records], indent=2, sort_keys=True)


def trace_from_json(text: str) -> List[TraceRecord]:
    return [TraceRecord.from_dict(d) for d in json.loads(text)]


@dataclass
class InvariantResult:
    name: str
    ok: bool
    detail: str = ""


def _chunk_of(ranges: Sequence[Span], chunk_len: int) -> List[int]:
    """Map global spans onto the chunk indices they cover (assumes aligned spans)."""
    chunks: List[int] = []
    for begin, end in ranges:
        if begin % chunk_len != 0 or end % chunk_len != 0:
            raise ValueError(f"span ({begin}, {end}) is not chunk-aligned to {chunk_len}")
        chunks.extend(range(begin // chunk_len, end // chunk_len))
    return chunks


def check_invariants(
    records: Sequence[TraceRecord],
    *,
    cp_size: int,
    seq_len: int,
    chunk_len: int,
    causal: bool,
    valid_len: Optional[int] = None,
) -> List[InvariantResult]:
    """Run every invariant the execution plan asks for. Returns one result per check.

    Invariants
    ----------
    ``coverage_no_duplication``
        For each rank, the union of the KV global positions over its ring steps
        is exactly the set the rank's query chunks need, and no step re-reads a
        position another step already read.
    ``unique_partial_slots``
        Every ``(rank, ring_step)`` has exactly one ``partial_slot_id`` and every
        slot id is globally unique.
    ``order_preserved``
        ``lse_update_index`` / ``out_update_index`` are the ring-step indices,
        i.e. the receive order is the compute/merge order. The KV source of step
        ``i`` is ``(rank - i) % cp_size``; no reordering is smuggled in.
    ``query_half_matches_schedule``
        ``query_half == "second"`` exactly on the causal upper-triangle steps.
    ``global_token_order_restored``
        Walking ranks ``0..P-1`` and, within a rank, its chunk pair in order,
        reconstructs global positions ``0..seq_len-1`` exactly once each.
    ``no_padding_as_valid_kv``
        No KV span reaches past ``valid_len`` (or ``seq_len``); padding is never
        handed to attention as a valid key.
    ``dtype_metadata_uniform``
        All records agree on ``partial_dtype`` / ``accumulator_dtype`` /
        ``lse_base``, so a merge cannot silently mix storage precisions.
    """
    results: List[InvariantResult] = []
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    n_chunks = 2 * cp_size
    if seq_len != n_chunks * chunk_len:
        raise ValueError(f"seq_len {seq_len} != {n_chunks} chunks * {chunk_len}")

    by_rank: Dict[int, List[TraceRecord]] = {r: [] for r in range(cp_size)}
    for rec in records:
        by_rank.setdefault(rec.rank, []).append(rec)

    # --- coverage without duplication ---------------------------------
    problems: List[str] = []
    for rank in range(cp_size):
        steps = sorted(by_rank.get(rank, []), key=lambda r: r.ring_step)
        if len(steps) != cp_size:
            problems.append(f"rank {rank}: {len(steps)} ring steps, expected {cp_size}")
            continue
        seen: Dict[int, int] = {}
        for rec in steps:
            for pos in rec.kv_positions():
                if pos in seen:
                    problems.append(f"rank {rank}: kv position {pos} covered twice (steps {seen[pos]} and {rec.ring_step})")
                seen[pos] = rec.ring_step
        first, second = rank, 2 * cp_size - 1 - rank
        if causal:
            # rank r's queries live in chunks {r, 2P-1-r}; the union of the KV they
            # can see is every chunk index <= 2P-1-r, i.e. positions [0, (2P-r)*chunk_len).
            needed = set(range(0, (2 * cp_size - rank) * chunk_len))
        else:
            needed = set(range(0, seq_len))
        missing = sorted(needed - set(seen))
        extra = sorted(set(seen) - needed)
        if missing:
            problems.append(f"rank {rank}: {len(missing)} uncovered kv positions, first {missing[:4]}")
        if extra:
            problems.append(f"rank {rank}: {len(extra)} out-of-range kv positions, first {extra[:4]}")
        if valid_len is not None and valid_len < seq_len:
            bad = sorted(p for p in seen if p >= valid_len)
            if bad:
                problems.append(f"rank {rank}: {len(bad)} kv positions beyond valid_len {valid_len}")
        del first, second
    results.append(InvariantResult("coverage_no_duplication", not problems, "; ".join(problems[:8])))

    # --- unique partial slots -----------------------------------------
    problems = []
    slots: Dict[str, Tuple[int, int]] = {}
    for rank in range(cp_size):
        per_step = [r.partial_slot_id for r in by_rank.get(rank, [])]
        if len(set(per_step)) != len(per_step):
            problems.append(f"rank {rank}: duplicate partial_slot_id within the rank")
        for rec in by_rank.get(rank, []):
            if rec.partial_slot_id in slots:
                problems.append(f"partial_slot_id {rec.partial_slot_id!r} reused by {slots[rec.partial_slot_id]} and {(rec.rank, rec.ring_step)}")
            slots[rec.partial_slot_id] = (rec.rank, rec.ring_step)
    if len(slots) != cp_size * cp_size:
        problems.append(f"{len(slots)} unique slots for cp_size={cp_size}, expected {cp_size * cp_size}")
    results.append(InvariantResult("unique_partial_slots", not problems, "; ".join(problems[:8])))

    # --- order preservation -------------------------------------------
    problems = []
    for rank in range(cp_size):
        steps = sorted(by_rank.get(rank, []), key=lambda r: r.ring_step)
        if [r.ring_step for r in steps] != list(range(len(steps))):
            problems.append(f"rank {rank}: ring_step values are not 0..{len(steps) - 1}")
        for rec in steps:
            expected_src = (rank - rec.ring_step) % cp_size
            if rec.source_rank != expected_src:
                problems.append(f"rank {rank} step {rec.ring_step}: source_rank {rec.source_rank} != (rank-step)%P = {expected_src}")
            if rec.lse_update_index != rec.ring_step:
                problems.append(f"rank {rank} step {rec.ring_step}: lse_update_index {rec.lse_update_index} != ring_step")
            if rec.out_update_index != rec.ring_step:
                problems.append(f"rank {rank} step {rec.ring_step}: out_update_index {rec.out_update_index} != ring_step")
        lse_order = [r.lse_update_index for r in steps]
        out_order = [r.out_update_index for r in steps]
        if lse_order != sorted(lse_order) or out_order != sorted(out_order):
            problems.append(f"rank {rank}: merge indices are not monotone in receive order")
    results.append(InvariantResult("order_preserved", not problems, "; ".join(problems[:8])))

    # --- query half matches the schedule ------------------------------
    problems = []
    for rank in range(cp_size):
        for rec in by_rank.get(rank, []):
            want = QUERY_HALF_SECOND if (causal and rec.ring_step > rank) else QUERY_HALF_BOTH
            if rec.query_half != want:
                problems.append(f"rank {rank} step {rec.ring_step}: query_half {rec.query_half!r} != {want!r}")
            want_section = (
                SECTION_ALL if not causal else (SECTION_DIAGONAL if rec.ring_step == 0 else (SECTION_LOWER if rec.ring_step <= rank else SECTION_UPPER))
            )
            if rec.section != want_section:
                problems.append(f"rank {rank} step {rec.ring_step}: section {rec.section!r} != {want_section!r}")
    results.append(InvariantResult("query_half_matches_schedule", not problems, "; ".join(problems[:8])))

    # --- global token order restored ----------------------------------
    #
    # The reassembly rule is: walk global chunks 0..2P-1 ascending; for each,
    # take the (rank, half) whose stored query span *is* that chunk. Check that
    # the rule is total, unambiguous, and lands exactly on the chunk's own
    # ascending token range -- i.e. concatenating in that order restores the
    # original global token order rather than a permutation of it.
    problems = []
    owner: Dict[int, Tuple[int, int]] = {}
    for rank in range(cp_size):
        steps = sorted(by_rank.get(rank, []), key=lambda r: r.ring_step)
        if not steps:
            continue
        # Rank r's query chunks, in the order the rank stores them.
        want_chunks = list(rank_chunk_indices(rank, cp_size))
        seen_chunks = _chunk_of(steps[0].q_global_ranges, chunk_len)
        if sorted(seen_chunks) != sorted(want_chunks):
            problems.append(f"rank {rank}: query chunks {sorted(seen_chunks)} != {sorted(want_chunks)}")
        if seen_chunks != want_chunks:
            problems.append(f"rank {rank}: query chunk order {seen_chunks} != storage order {want_chunks}")
        for half, (begin, end) in enumerate(steps[0].q_global_ranges):
            for chunk in range(begin // chunk_len, end // chunk_len):
                if chunk in owner:
                    problems.append(f"chunk {chunk} claimed by both {owner[chunk]} and {(rank, half)}")
                owner[chunk] = (rank, half)
    if sorted(owner) != list(range(n_chunks)):
        problems.append(f"query chunks covered {sorted(owner)}, expected 0..{n_chunks - 1}")
    for chunk in range(n_chunks):
        if chunk not in owner:
            continue
        rank, half = owner[chunk]
        spans = sorted(by_rank[rank], key=lambda r: r.ring_step)[0].q_global_ranges
        if spans[half] != (chunk * chunk_len, (chunk + 1) * chunk_len):
            problems.append(f"reassembly rule picks rank {rank} half {half} for chunk {chunk}, whose span is {spans[half]}")
    results.append(InvariantResult("global_token_order_restored", not problems, "; ".join(problems[:8])))

    # --- padding is never valid KV ------------------------------------
    problems = []
    limit = valid_len if valid_len is not None else seq_len
    for rec in records:
        for begin, end in rec.kv_global_ranges:
            if begin < 0 or end > seq_len or begin >= end:
                problems.append(f"rank {rec.rank} step {rec.ring_step}: degenerate/out-of-range kv span ({begin}, {end})")
            elif end > limit:
                problems.append(f"rank {rec.rank} step {rec.ring_step}: kv span ({begin}, {end}) crosses valid_len {limit}")
        for begin, end in rec.q_global_ranges:
            if begin < 0 or end > seq_len or begin >= end:
                problems.append(f"rank {rec.rank} step {rec.ring_step}: degenerate/out-of-range q span ({begin}, {end})")
    results.append(InvariantResult("no_padding_as_valid_kv", not problems, "; ".join(problems[:8])))

    # --- dtype metadata ------------------------------------------------
    problems = []
    for attr in ("partial_dtype", "accumulator_dtype", "lse_base", "lse_dtype"):
        values = sorted({getattr(r, attr) for r in records})
        if len(values) > 1:
            problems.append(f"{attr} is not uniform across the trace: {values}")
    results.append(InvariantResult("dtype_metadata_uniform", not problems, "; ".join(problems[:8])))

    return results


def rank_chunk_indices(rank: int, cp_size: int) -> Tuple[int, int]:
    """Chunk indices owned by ``rank``: ``(rank, 2P-1-rank)`` (TE dual-chunk swap)."""
    return rank, 2 * cp_size - 1 - rank


def assert_invariants(records: Sequence[TraceRecord], **kwargs: object) -> List[InvariantResult]:
    results = check_invariants(records, **kwargs)  # type: ignore[arg-type]
    failed = [r for r in results if not r.ok]
    if failed:
        raise AssertionError("CP trace invariants failed:\n" + "\n".join(f"  {r.name}: {r.detail}" for r in failed))
    return results


def restore_global_order(per_rank_out: Dict[int, "object"], cp_size: int, chunk_len: int) -> "object":
    """Concatenate per-rank local outputs into global token order.

    ``per_rank_out[r]`` is the rank's local output with the two query halves
    concatenated along the sequence axis (``[B, 2 * chunk_len, H, D]``). The
    result is ``[B, 2P * chunk_len, H, D]`` in ascending global token order.
    """
    import torch  # local import: keep this module importable without torch for schema-only users

    pieces = []
    for chunk in range(2 * cp_size):
        rank = chunk if chunk < cp_size else 2 * cp_size - 1 - chunk
        local = per_rank_out[rank]
        half = 0 if rank_chunk_indices(rank, cp_size)[0] == chunk else 1
        pieces.append(local[:, half * chunk_len : (half + 1) * chunk_len])
    return torch.cat(pieces, dim=1)
