# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the frozen-partial fixtures used by the R0/R1/CP2/CP4 runs.

The fixtures are generated from the **FP64 mathematical oracle only**
(``reference_math``). A fixture contains

* the global ``Q``/``K``/``V`` used to build it (in ``qkv_dtype``),
* one ``(O_i, L_i)`` pair per ``(rank, ring_step)``, in the shape TE hands to the
  attention kernel for that section, cast to ``partial_storage_dtype``,
* the monolithic FP64 expected output and LSE,
* a unique token id per global position (for the token-order invariant),
* the full trace plus the explicit ``P -> schedule`` table.

The partials are **synthetic**: they come from an FP64 chunked attention, not
from any GPU kernel. Every header therefore records ``o_i_normalized`` and the
storage dtype so a reader cannot mistake them for native FP32 SDPA partials.

Usage::

    python test/python/sdpa/cp_numerics/make_fixtures.py \
        --out-dir test/python/sdpa/cp_numerics/fixtures

``--record-te-sha`` additionally writes ``evidence/752/TE_SHA.txt`` from a local
TransformerEngine checkout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from cp_numerics import reference_math as rm  # noqa: E402
from cp_numerics import te_adapter as tea  # noqa: E402
from cp_numerics import trace_schema as ts  # noqa: E402


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def make_qkv(
    *,
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (batch, seq_len, num_heads, head_dim)
    q = torch.randn(shape, generator=generator, dtype=torch.float64) * 0.5
    k = torch.randn(shape, generator=generator, dtype=torch.float64) * 0.5
    v = torch.randn(shape, generator=generator, dtype=torch.float64) * 0.5
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _step_partial(
    *,
    q64: torch.Tensor,
    k64: torch.Tensor,
    v64: torch.Tensor,
    rec: ts.TraceRecord,
    scale: float,
    valid_len: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One step's ``(O_i, L_i)`` in FP64, using the step's own global mask.

    The mask is derived from **global positions** rather than from the local
    layout, so it stays correct for the sections whose local KV axis is a
    concatenation of two disjoint chunks.
    """
    q_positions = torch.cat([torch.arange(b, e) for b, e in rec.q_global_ranges])
    kv_positions = torch.cat([torch.arange(b, e) for b, e in rec.kv_global_ranges])
    keep = rm.causal_keep_mask_from_positions(
        q_positions,
        kv_positions,
        causal=(rec.causal_mode == "causal"),
        valid_kv_len=valid_len,
    )
    q_local = q64.index_select(1, q_positions)
    k_local = k64.index_select(1, kv_positions)
    v_local = v64.index_select(1, kv_positions)
    partial = rm.chunk_partial_with_mask_fp64(
        q_local,
        k_local,
        v_local,
        scale=scale,
        keep_mask=keep,
        chunk_index=rec.ring_step,
        kv_begin=rec.kv_global_begin,
        kv_end=rec.kv_global_end,
    )
    expected_tokens = sum(e - b for b, e in rec.q_global_ranges)
    if partial.out.shape[1] != expected_tokens:
        raise AssertionError(f"step {rec.rank}/{rec.ring_step}: {partial.out.shape[1]} tokens, expected {expected_tokens}")
    return partial.out, partial.lse


def build_fixture(
    *,
    name: str,
    cp_size: int,
    causal: bool,
    seq_len: int,
    batch: int = 1,
    num_heads: int = 4,
    head_dim: int = 64,
    seed: int = 20250918,
    qkv_dtype: torch.dtype = torch.float32,
    partial_storage_dtype: torch.dtype = torch.float32,
    accumulator_dtype: torch.dtype = torch.float32,
    o_dtype: torch.dtype = torch.float32,
    lane: str = "te_fidelity",
    note: str = "",
    empty_chunks: Sequence[Tuple[int, int]] = (),
    valid_len: Optional[int] = None,
) -> tea.FrozenFixture:
    """Build one fixture from the FP64 oracle.

    ``empty_chunks`` lists ``(rank, ring_step)`` pairs whose partial is forced to
    ``L = -inf`` / ``O = 0`` (an empty KV chunk), so the merge's empty-chunk
    handling is exercised on real fixture data.
    """
    if cp_size < 1:
        raise ValueError("cp_size must be >= 1")
    n_chunks = 2 * cp_size
    if seq_len % n_chunks != 0:
        raise ValueError(f"seq_len {seq_len} is not divisible by 2*cp_size = {n_chunks}; " "use build_ragged_reference_fixture for ragged tails")
    chunk_len = seq_len // n_chunks
    if head_dim <= 0 or num_heads <= 0 or batch <= 0:
        raise ValueError("batch/num_heads/head_dim must be positive")
    if valid_len is None:
        valid_len = seq_len
    if not 0 < valid_len <= seq_len:
        raise ValueError("valid_len must be in (0, seq_len]")

    scale = float(head_dim) ** -0.5
    q, k, v = make_qkv(batch=batch, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim, dtype=qkv_dtype, seed=seed)
    q64, k64, v64 = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)

    trace = tea.te_schedule(
        cp_size,
        chunk_len,
        causal=causal,
        partial_dtype=_dtype_name(partial_storage_dtype),
        accumulator_dtype=_dtype_name(accumulator_dtype),
        lse_dtype="float32",
    )

    empty_set = {(int(r), int(s)) for r, s in empty_chunks}

    # Expected value. Without empty-chunk probes this is the untouched global
    # attention. With them, the emptied (query, key) pairs are removed from the
    # mask first, because dropping a partial genuinely removes that key range
    # from the result -- comparing against the untouched attention would only
    # measure the deliberate removal.
    keep = rm.top_left_causal_keep_mask(seq_len, seq_len, causal=causal, valid_kv_len=valid_len)
    trace_by_slot = {(r.rank, r.ring_step): r for r in trace}
    for empty_rank, empty_step in empty_set:
        record = trace_by_slot[(empty_rank, empty_step)]
        q_positions = torch.cat([torch.arange(b, e) for b, e in record.q_global_ranges])
        kv_positions = torch.cat([torch.arange(b, e) for b, e in record.kv_global_ranges])
        drop = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        drop[q_positions.unsqueeze(1), kv_positions.unsqueeze(0)] = True
        keep &= ~drop
    expected = rm.full_attention_with_mask_fp64(q64, k64, v64, scale=scale, keep_mask=keep)

    tensors: Dict[str, torch.Tensor] = {
        "q_global": q.detach().clone(),
        "k_global": k.detach().clone(),
        "v_global": v.detach().clone(),
        "expected_out_fp64": expected.out.detach().clone(),
        "expected_lse_fp64": expected.lse.detach().clone(),
        "expected_keep_mask": keep,
        "token_ids": torch.arange(seq_len, dtype=torch.int64),
    }

    for rec in trace:
        out_i, lse_i = _step_partial(q64=q64, k64=k64, v64=v64, rec=rec, scale=scale, valid_len=valid_len)
        if (rec.rank, rec.ring_step) in empty_set:
            out_i = torch.zeros_like(out_i)
            lse_i = torch.full_like(lse_i, float("-inf"))
        tensors[f"out_r{rec.rank}_s{rec.ring_step}"] = out_i.to(partial_storage_dtype).contiguous()
        tensors[f"lse_r{rec.rank}_s{rec.ring_step}"] = lse_i.to(torch.float32).contiguous()

    header = tea.FixtureHeader(
        schema_version=1,
        name=name,
        lane=lane,
        cp_size=cp_size,
        causal=bool(causal),
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        qkv_dtype=_dtype_name(qkv_dtype),
        partial_storage_dtype=_dtype_name(partial_storage_dtype),
        accumulator_dtype=_dtype_name(accumulator_dtype),
        o_dtype=_dtype_name(o_dtype),
        lse_dtype="float32",
        lse_base=rm.LSE_BASE,
        o_i_normalized=True,
        mask="causal-top-left" if causal else "no_mask",
        layout=tea.TE_SUPPORTED_LAYOUT,
        head_mapping="full: every rank holds all H heads (no head split; cp_size_a2a == 1)",
        te_revision=tea.TE_PINNED_SHA,
        te_tag=tea.TE_PINNED_TAG,
        te_source_relpath=tea.TE_SOURCE_RELPATH,
        te_source_sha256=tea.TE_SOURCE_SHA256,
        fuser_mode=tea.TE_FUSER_MODE,
        chunk_len=chunk_len,
        valid_len=valid_len,
        te_divisible=True,
        note=note,
        generated_utc=_utc_now(),
        p_schedule_map=tea.p_schedule_map(cp_size, chunk_len, causal=causal),
        trace=[r.as_dict() for r in trace],
        empty_chunks=[[int(r), int(s)] for r, s in sorted(empty_set)],
    )
    if empty_set:
        header.note = ((note + " | ") if note else "") + (
            "empty-chunk probe: the listed (rank, step) partials are forced to L=-inf/O=0 and the " "expected value is recomputed over the retained key set"
        )
    return tea.FrozenFixture(header=header, tensors=tensors)


def build_ragged_reference_fixture(
    *,
    name: str,
    cp_size: int = 2,
    seq_len: int = 102,
    causal: bool = True,
    batch: int = 1,
    num_heads: int = 2,
    head_dim: int = 32,
    seed: int = 4242,
) -> tea.FrozenFixture:
    """A reference-planner-only stress case with a ragged tail (non-divisible).

    TE's CP path requires ``seq_len % (2 * cp_size) == 0``; this fixture exists to
    exercise the *reference planner* on an uneven split and is labelled
    ``lane="reference_only"`` so the TE lane refuses it instead of quietly
    pretending the tail does not exist.
    """
    n_chunks = 2 * cp_size
    base = seq_len // n_chunks
    remainder = seq_len - base * n_chunks
    splits = [base + (1 if i < remainder else 0) for i in range(n_chunks)]
    assert sum(splits) == seq_len
    bounds = rm.chunk_bounds_from_splits(splits)
    scale = float(head_dim) ** -0.5
    q, k, v = make_qkv(batch=batch, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim, dtype=torch.float32, seed=seed)
    q64, k64, v64 = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
    expected = rm.full_attention_fp64(q64, k64, v64, scale=scale, causal=causal)
    partials = rm.chunk_partials_fp64(q64, k64, v64, scale=scale, causal=causal, chunk_bounds=bounds)

    tensors: Dict[str, torch.Tensor] = {
        "q_global": q,
        "k_global": k,
        "v_global": v,
        "expected_out_fp64": expected.out,
        "expected_lse_fp64": expected.lse,
        "token_ids": torch.arange(seq_len, dtype=torch.int64),
        "ragged_splits": torch.tensor(splits, dtype=torch.int64),
    }
    for index, part in enumerate(partials):
        tensors[f"out_chunk_{index}"] = part.out.to(torch.float32).contiguous()
        tensors[f"lse_chunk_{index}"] = part.lse.to(torch.float32).contiguous()

    header = tea.FixtureHeader(
        schema_version=1,
        name=name,
        lane="reference_only",
        cp_size=cp_size,
        causal=bool(causal),
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        qkv_dtype="float32",
        partial_storage_dtype="float32",
        accumulator_dtype="float32",
        o_dtype="float32",
        lse_dtype="float32",
        lse_base=rm.LSE_BASE,
        o_i_normalized=True,
        mask="causal-top-left" if causal else "no_mask",
        layout=tea.TE_SUPPORTED_LAYOUT,
        head_mapping="n/a (reference planner only)",
        te_revision=tea.TE_PINNED_SHA,
        te_tag=tea.TE_PINNED_TAG,
        te_source_relpath=tea.TE_SOURCE_RELPATH,
        te_source_sha256=tea.TE_SOURCE_SHA256,
        fuser_mode=tea.TE_FUSER_MODE,
        chunk_len=base,
        valid_len=seq_len,
        te_divisible=False,
        note=f"ragged splits {splits}; TE CP requires divisibility, so this fixture is reference-planner-only",
        generated_utc=_utc_now(),
        p_schedule_map=[],
        trace=[],
        empty_chunks=[],
    )
    return tea.FrozenFixture(header=header, tensors=tensors)


#: The fixed fixture matrix. Every entry keeps the *global* problem shape
#: identical across P so a CP2-vs-CP4 timing comparison is apples to apples.
MATRIX = (
    dict(name="cp_p1_f32_s128_causal", cp_size=1, seq_len=128, causal=True),
    dict(name="cp_p2_f32_s128_causal", cp_size=2, seq_len=128, causal=True),
    dict(name="cp_p4_f32_s128_causal", cp_size=4, seq_len=128, causal=True),
    dict(name="cp_p2_f32_s128_noncausal", cp_size=2, seq_len=128, causal=False),
    dict(name="cp_p4_f32_s128_noncausal", cp_size=4, seq_len=128, causal=False),
    dict(name="cp_p2_f32_s128_causal_empty", cp_size=2, seq_len=128, causal=True, empty_chunks=[(1, 1)]),
    dict(
        name="cp_p2_f16_s128_causal",
        cp_size=2,
        seq_len=128,
        causal=True,
        qkv_dtype=torch.float16,
        partial_storage_dtype=torch.float16,
        o_dtype=torch.float16,
        note="FP16 storage for partials AND accumulator, matching TE fwd_nominal_dtype; not a native FP32 partial",
    ),
    dict(
        name="cp_p2_bf16_s128_causal",
        cp_size=2,
        seq_len=128,
        causal=True,
        qkv_dtype=torch.bfloat16,
        partial_storage_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        note="BF16 storage for partials AND accumulator; the FP64 expected value is unaffected by this cast",
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="generate #752 frozen-partial fixtures")
    parser.add_argument("--out-dir", default=os.path.join(_HERE, "fixtures"))
    parser.add_argument("--seed", type=int, default=20250918)
    parser.add_argument("--te-repo", default=None, help="TransformerEngine checkout, for --record-te-sha")
    parser.add_argument("--record-te-sha", action="store_true")
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument("--only", default=None, help="generate a single fixture by name")
    args = parser.parse_args(argv)

    if args.record_te_sha:
        if not args.te_repo or not args.evidence_dir:
            parser.error("--record-te-sha requires --te-repo and --evidence-dir")
        digest = tea.helper_source_sha256(args.te_repo)
        head = subprocess.run(["git", "-C", args.te_repo, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        tag = subprocess.run(["git", "-C", args.te_repo, "describe", "--tags"], check=True, capture_output=True, text=True).stdout.strip()
        when = subprocess.run(["git", "-C", args.te_repo, "log", "-1", "--format=%H %ci %s"], check=True, capture_output=True, text=True).stdout.strip()
        if head != tea.TE_PINNED_SHA:
            print(f"WARNING: TE checkout HEAD {head} != pinned {tea.TE_PINNED_SHA}", file=sys.stderr)
        if not tea.TE_SOURCE_SHA256:
            print(
                f"NOTE: te_adapter.TE_SOURCE_SHA256 is unset; record {digest} there so the binding is verifiable",
                file=sys.stderr,
            )
        os.makedirs(args.evidence_dir, exist_ok=True)
        with open(os.path.join(args.evidence_dir, "TE_SHA.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"{head}\n{tag}\n{when}\nsource_sha256 {digest}\n")
        print(f"recorded TE_SHA.txt: {head} {tag} source_sha256={digest}")

    written: List[str] = []
    for entry in MATRIX:
        if args.only and entry["name"] != args.only:
            continue
        kwargs = dict(entry)
        name = kwargs.pop("name")
        fixture = build_fixture(name=name, seed=args.seed, **kwargs)
        path = os.path.join(args.out_dir, f"{name}.pt")
        digest = tea.save_fixture(path, fixture)
        written.append(path)
        print(f"{path}  P={fixture.header.cp_size} causal={fixture.header.causal} digest={digest[:16]}")

    if not args.only:
        ragged = build_ragged_reference_fixture(name="ref_ragged_p2_s102_causal")
        path = os.path.join(args.out_dir, "ref_ragged_p2_s102_causal.pt")
        digest = tea.save_fixture(path, ragged)
        written.append(path)
        print(f"{path}  P={ragged.header.cp_size} lane={ragged.header.lane} digest={digest[:16]}")

    print(f"\n{len(written)} fixtures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
