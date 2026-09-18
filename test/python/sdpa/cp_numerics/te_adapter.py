# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R1: merge-only oracle bound to a FIXED NVIDIA TransformerEngine revision.

What this module is
-------------------
A faithful transcription of the TransformerEngine context-parallel *merge*
helpers -- LSE correction, output-correction init, per-step output correction and
the second-half variants -- together with the ring schedule those helpers are
driven by. It consumes **frozen partials**; it never runs a TE attention kernel
and it makes no claim about partials produced by any particular GPU kernel.

What this module is not
-----------------------
It is not a proof of bitwise parity with TransformerEngine's compiled GPU path.
The elementwise expressions, operand dtypes and evaluation order match the
pinned source, but the kernels that produce the partials, the reduction
scheduling of ``torch.compile`` and the CUDA stream interleaving are out of
scope. See ``report.md`` for the exact claim boundary.

Upstream attribution
--------------------
The five helpers below are transcribed from

    NVIDIA/TransformerEngine
    transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py
    Apache License 2.0 (see the upstream repository's LICENSE)
    pinned revision: ``TE_PINNED_SHA`` (recorded in ``evidence/752/TE_SHA.txt``)

Line numbers in the comments refer to that pinned revision. ``verify_helper_source``
re-reads the pinned checkout (when one is available) and confirms the constructs we
transcribed are still the ones in the file, so the binding cannot rot silently.

Matched execution / fuser mode
------------------------------
``@jit_fuser`` in ``transformer_engine/pytorch/jit.py`` is ``lazy_compile`` when
``torch >= 2.0`` and ``NVTE_TORCH_COMPILE`` is unset or truthy (its default), and
the identity decorator otherwise. ``lazy_compile`` wraps the function in
``torch.compile``. The correction helpers therefore run under **TorchDynamo /
Inductor** in the default configuration, on one of two alternating CUDA streams
(``flash_attn_streams``). Note ``lazy_compile`` calls the undecorated function
directly when asked to trace itself, so a dynamo-traced caller sees eager
semantics. This adapter reproduces the *eager* expression tree exactly and can
additionally run it through ``torch.compile`` for an A/B on the fuser mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import trace_schema as ts

#: TransformerEngine revision this adapter is bound to. Written by
#: ``make_fixtures.py --record-te-sha`` into ``evidence/752/TE_SHA.txt``; if the
#: pinned checkout moves, both must be updated in the same commit.
TE_PINNED_SHA = "ae34b34f18d85cc62d90ca35cf6efe84c5a45a7e"
TE_PINNED_TAG = "v0.1-2155-gae34b34f"
TE_PINNED_DATE = "2026-09-17 21:20:44 +0200"
TE_SOURCE_RELPATH = "transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py"

#: sha256 of the pinned source file. ``None`` means "not recorded", in which case
#: ``verify_helper_source`` reports ``None`` for the match rather than a false pass.
TE_SOURCE_SHA256: Optional[str] = "37b389cefd7e80060ccb0a856f4970daafe307e0705b79ddc2f3cbce30fc002a"

#: ``jit_fuser`` resolves to ``lazy_compile`` (torch.compile) by default.
TE_FUSER_MODE = "torch.compile via transformer_engine.pytorch.jit.lazy_compile (NVTE_TORCH_COMPILE default 1)"
TE_FUSER_MODE_EAGER_FALLBACK = "identity decorator when NVTE_TORCH_COMPILE=0 or torch < 2.0"

#: ``fused_attn_fwd`` returns ``softmax_lse`` as float32 (see the docstring of
#: transformer_engine/pytorch/cpp_extensions/fused_attn.py), shape
#: ``[b, h, max_seqlen_q, 1]``, squeezed to ``[b, h, max_seqlen_q]`` by the CP
#: forward before the corrections run.
TE_LSE_DTYPE_NAME = "float32"

#: The CP forward uses natural-log LSE: the helpers are ``exp``/``log1p`` with no
#: ``log2(e)`` scaling anywhere.
TE_LSE_BASE = "e"

#: Only the ``bshd`` layout is transcribed: ``seq_dim = qkv_format.index("s") = 1``.
TE_SUPPORTED_LAYOUT = "bshd"
TE_SEQ_DIM = 1

_SCHEMA_VERSION = 1


# ===========================================================================
# transcribed upstream helpers
# ===========================================================================
#
# These are line-for-line the upstream bodies. No decorator is applied: upstream
# ``@jit_fuser`` is ``lazy_compile`` (torch.compile) by default, and the compiled
# variant is selected explicitly through ``_helpers(compiled=True)`` so the eager
# and compiled lanes can be compared instead of conflated.


def flash_attn_fwd_out_correction_init(
    out_init_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_init_step: torch.Tensor,
    seq_dim: int,
) -> torch.Tensor:
    """Merge partial outputs of the first step (upstream lines 154-165)."""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_init_step - softmax_lse).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_init_step * softmax_lse_corrected_exp
    return out_corrected.to(out_init_step.dtype)


def flash_attn_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    seq_dim: int,
) -> None:
    """Merge partial outputs of each step (upstream lines 168-180). In place."""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)


def flash_attn_fwd_second_half_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    seq_dim: int,
) -> None:
    """Merge second-half partial outputs of each step (upstream lines 183-197)."""
    out_ = out.select(seq_dim, 1)
    softmax_lse_ = softmax_lse.view(*softmax_lse.shape[:-1], 2, -1)[..., 1, :]
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse_).movedim(2, seq_dim)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out_.add_(out_corrected)


def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
) -> None:
    """Merge softmax stats of each step (upstream lines 200-209). In place."""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


def flash_attn_fwd_second_half_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
) -> None:
    """Merge second-half softmax stats of each step (upstream lines 212-222)."""
    softmax_lse_ = softmax_lse[..., 1, :]
    max_scale = torch.max(softmax_lse_, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse_, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse_.copy_(new_scale)


#: substrings ``verify_helper_source`` looks for in the pinned checkout.
_HELPER_ANCHORS: Dict[str, str] = {
    "out_correction_init_expr": "out_corrected = out_init_step * softmax_lse_corrected_exp",
    "out_correction_init_cast": "return out_corrected.to(out_init_step.dtype)",
    "out_correction_add": "out.add_(out_corrected)",
    "second_half_out_correction_add": "out_.add_(out_corrected)",
    "second_half_out_select": "out_ = out.select(seq_dim, 1)",
    "lse_correction_log1p": "new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))",
    "second_half_lse_slice": "softmax_lse_ = softmax_lse[..., 1, :]",
    "second_half_lse_view": "softmax_lse.view(*softmax_lse.shape[:-1], 2, -1)[..., 1, :]",
    "ring_send_dst": "send_dst = cp_global_ranks[(rank + 1) % cp_size",
    "ring_recv_src": "recv_src = cp_global_ranks[(rank - 1) % cp_size",
    "dual_chunk_swap": "total_slices - cp_rank - 1",
    "lse_squeeze_to_3d": "softmax_lse_per_step[i - 1].squeeze_(-1)",
    "correction_loop": "for i in range(cp_size + 1):",
    "out_correction_loop": "for i in range(cp_size):",
    "cp_per_step_configs": "def cp_per_step_configs(",
}


def helper_source_sha256(te_repo: str) -> str:
    path = os.path.join(te_repo, TE_SOURCE_RELPATH)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def verify_helper_source(te_repo: str) -> Dict[str, Any]:
    """Check the pinned checkout still contains every transcribed construct.

    Returns a mapping from anchor name to ``True``/``False`` plus
    ``source_sha256`` and ``source_sha256_matches_recorded`` entries. Raises
    ``FileNotFoundError`` when the pinned file is absent, so a caller that asks
    for verification never gets a silent pass.
    """
    path = os.path.join(te_repo, TE_SOURCE_RELPATH)
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    out: Dict[str, Any] = {name: (anchor in text) for name, anchor in _HELPER_ANCHORS.items()}
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out["source_sha256"] = digest
    out["source_sha256_matches_recorded"] = (digest == TE_SOURCE_SHA256) if TE_SOURCE_SHA256 else None
    return out


# ===========================================================================
# schedule
# ===========================================================================


def te_schedule(
    cp_size: int,
    chunk_len: int,
    *,
    causal: bool,
    partial_dtype: str = "float32",
    accumulator_dtype: str = "float32",
    lse_dtype: str = TE_LSE_DTYPE_NAME,
) -> List[ts.TraceRecord]:
    """The ring schedule implemented by TE's ``AttnFuncWithCPAndKVP2P.forward``.

    Derived from the pinned source:

    * ring: ``send_dst = (rank + 1) % cp_size``, ``recv_src = (rank - 1) % cp_size``
      (lines 1650-1651), so step ``i`` consumes the KV of source ``(rank - i) % P``.
    * rank ``r`` owns chunks ``(r, 2P-1-r)`` (``get_batch_on_this_cp_rank``:
      first slice ``cp_rank``, second slice ``total_slices - cp_rank - 1`` with
      ``total_slices = 2 * cp_size``).
    * causal sections (loop at line 1941, section chosen around lines 2040-2130):
      ``step 0`` diagonal (full local Q x full local KV, causal mask);
      ``0 < step <= rank`` lower-triangle (full local Q x source's **first** half,
      non-causal mask); ``step > rank`` upper-triangle (local **second** half x
      source's full KV, non-causal mask).
    * non-causal: every step uses section ``"all"`` (full local Q x full source
      KV, non-causal mask).
    * ``cp_per_step_configs`` (line 5423) confirms the mask of the two triangle
      sections is ``padding``/``no_mask`` and that the diagonal keeps the
      caller's ``attn_mask_type``.
    """
    if cp_size < 1:
        raise ValueError("cp_size must be >= 1")
    if chunk_len < 1:
        raise ValueError("chunk_len must be >= 1")
    records: List[ts.TraceRecord] = []
    for rank in range(cp_size):
        first_chunk, second_chunk = ts.rank_chunk_indices(rank, cp_size)
        first_span = (first_chunk * chunk_len, (first_chunk + 1) * chunk_len)
        second_span = (second_chunk * chunk_len, (second_chunk + 1) * chunk_len)
        for ring_step in range(cp_size):
            source_rank = (rank - ring_step) % cp_size
            src_first, src_second = ts.rank_chunk_indices(source_rank, cp_size)
            src_first_span = (src_first * chunk_len, (src_first + 1) * chunk_len)
            src_second_span = (src_second * chunk_len, (src_second + 1) * chunk_len)
            if not causal:
                section = ts.SECTION_ALL
                query_half = ts.QUERY_HALF_BOTH
                q_ranges = (first_span, second_span)
                kv_ranges = (src_first_span, src_second_span)
                causal_mode = "no_mask"
            elif ring_step == 0:
                section = ts.SECTION_DIAGONAL
                query_half = ts.QUERY_HALF_BOTH
                q_ranges = (first_span, second_span)
                kv_ranges = (first_span, second_span)
                causal_mode = "causal"
            elif ring_step <= rank:
                section = ts.SECTION_LOWER
                query_half = ts.QUERY_HALF_BOTH
                q_ranges = (first_span, second_span)
                # cp_p2p_fwd_prepare_qkv "lower-triangle": k_part[:, 0, ...] -> first half only
                kv_ranges = (src_first_span,)
                causal_mode = "no_mask"
            else:
                section = ts.SECTION_UPPER
                query_half = ts.QUERY_HALF_SECOND
                # cp_p2p_fwd_prepare_qkv "upper-triangle": q_part[:, 1, ...] -> second half only
                q_ranges = (second_span,)
                kv_ranges = (src_first_span, src_second_span)
                causal_mode = "no_mask"
            records.append(
                ts.TraceRecord(
                    rank=rank,
                    ring_step=ring_step,
                    source_rank=source_rank,
                    q_global_begin=q_ranges[0][0],
                    q_global_end=q_ranges[-1][1],
                    kv_global_begin=kv_ranges[0][0],
                    kv_global_end=kv_ranges[-1][1],
                    causal_mode=causal_mode,
                    query_half=query_half,
                    partial_slot_id=f"r{rank}s{ring_step}",
                    lse_update_index=ring_step,
                    out_update_index=ring_step,
                    partial_dtype=partial_dtype,
                    accumulator_dtype=accumulator_dtype,
                    lse_base=TE_LSE_BASE,
                    section=section,
                    q_global_ranges=q_ranges,
                    kv_global_ranges=kv_ranges,
                    causal_alignment=ts.ALIGNMENT_TOP_LEFT,
                    lse_dtype=lse_dtype,
                )
            )
    return records


def p_schedule_map(cp_size: int, chunk_len: int, *, causal: bool) -> List[Dict[str, Any]]:
    """Compact, human-checkable P -> schedule table for fixture headers."""
    rows = []
    for rec in te_schedule(cp_size, chunk_len, causal=causal):
        rows.append(
            {
                "rank": rec.rank,
                "ring_step": rec.ring_step,
                "source_rank": rec.source_rank,
                "section": rec.section,
                "query_half": rec.query_half,
                "causal_mode": rec.causal_mode,
                "q_chunks": [b // chunk_len for b, _ in rec.q_global_ranges],
                "kv_chunks": [b // chunk_len for b, _ in rec.kv_global_ranges],
            }
        )
    return rows


# ===========================================================================
# merges
# ===========================================================================


def _helpers(compiled: bool) -> Dict[str, Any]:
    if not compiled:
        return {
            "init": flash_attn_fwd_out_correction_init,
            "out": flash_attn_fwd_out_correction,
            "out2": flash_attn_fwd_second_half_out_correction,
            "lse": flash_attn_fwd_softmax_lse_correction,
            "lse2": flash_attn_fwd_second_half_softmax_lse_correction,
            "mode": "eager",
        }
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable; cannot exercise the TE fuser mode")
    return {
        "init": torch.compile(flash_attn_fwd_out_correction_init),
        "out": torch.compile(flash_attn_fwd_out_correction),
        "out2": torch.compile(flash_attn_fwd_second_half_out_correction),
        "lse": torch.compile(flash_attn_fwd_softmax_lse_correction),
        "lse2": torch.compile(flash_attn_fwd_second_half_softmax_lse_correction),
        "mode": "torch.compile",
    }


def merge_te_fidelity(
    *,
    rank: int,
    cp_size: int,
    causal: bool,
    partial_out: Sequence[torch.Tensor],
    partial_lse: Sequence[torch.Tensor],
    o_local_shape: Sequence[int],
    seq_dim: int = TE_SEQ_DIM,
    compiled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce TE's two correction loops over frozen partials.

    ``partial_out[i]`` / ``partial_lse[i]`` are the frozen per-ring-step results
    for ``rank``. ``partial_out[i]`` has the shape TE hands to the kernel for that
    section (``[B, 2*chunk_len, H, D]`` for the diagonal/lower/all sections,
    ``[B, chunk_len, H, D]`` for the causal upper-triangle) and carries
    ``fwd_nominal_dtype``; ``partial_lse[i]`` is float32.

    Returns ``(out, lse)`` where ``out`` is ``[B, 2, chunk_len, H, D]`` (``bshd``
    with the query-half axis retained, matching TE's ``o_shape``) and ``lse`` is
    float32 ``[B, H, 2*chunk_len]``. Callers reshape ``out`` to ``[B, S_local,
    H, D]``; that is a view, exactly as in TE.
    """
    if len(partial_out) != cp_size or len(partial_lse) != cp_size:
        raise ValueError(f"expected {cp_size} per-step partials, got {len(partial_out)}/{len(partial_lse)}")
    for index, tensor in enumerate(partial_out):
        if not tensor.is_contiguous():
            raise ValueError(
                f"partial_out[{index}] must be contiguous: TE receives dense kernel outputs and calls "
                "out.view(o_shape) / broadcasts against them without a copy"
            )
    for index, tensor in enumerate(partial_lse):
        if not tensor.is_contiguous():
            raise ValueError(f"partial_lse[{index}] must be contiguous: the second-half helpers call " "softmax_lse.view(*softmax_lse.shape[:-1], 2, -1)")
    helpers = _helpers(compiled)

    # --- upstream lines 2157-2204: softmax LSE correction, in ring-step order
    softmax_lse = partial_lse[0].clone()
    for step in range(1, cp_size):
        if (not causal) or step <= rank:
            helpers["lse"](softmax_lse, partial_lse[step])
        else:
            helpers["lse2"](softmax_lse.view(*softmax_lse.shape[:-1], 2, -1), partial_lse[step])

    # --- upstream lines 2224-2272: output correction, in ring-step order
    out = helpers["init"](partial_out[0], softmax_lse, partial_lse[0], seq_dim)
    if not out.is_contiguous():
        raise AssertionError("TE calls out.view(o_shape); a non-contiguous correction result would raise there too")
    out = out.view(*o_local_shape)
    for step in range(1, cp_size):
        if (not causal) or step <= rank:
            helpers["out"](out.view(*partial_out[step].shape), partial_out[step], softmax_lse, partial_lse[step], seq_dim)
        else:
            helpers["out2"](out, partial_out[step], softmax_lse, partial_lse[step], seq_dim)
    return out, softmax_lse


PART_FIRST = "first"
PART_SECOND = "second"
PART_WHOLE = "whole"


def half_step_lists(cp_size: int, rank: int, causal: bool) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """``(ring_step, which part of that step's local query axis)`` per output half.

    TE's diagonal and lower-triangle sections carry the rank's *full* local query
    axis (both halves) while the causal upper-triangle carries only the second
    half, so a full-axis step must be sliced when it feeds one output half.
    """
    first_half: List[Tuple[int, str]] = []
    second_half: List[Tuple[int, str]] = []
    for step in range(cp_size):
        if (not causal) or step == 0 or step <= rank:
            first_half.append((step, PART_FIRST))
            second_half.append((step, PART_SECOND))
        else:
            second_half.append((step, PART_WHOLE))
    return first_half, second_half


def _half_slice(tensor: torch.Tensor, part: str, chunk_len: int, seq_axis: int) -> torch.Tensor:
    if part == PART_WHOLE:
        return tensor
    index = [slice(None)] * tensor.dim()
    index[seq_axis] = slice(0, chunk_len) if part == PART_FIRST else slice(chunk_len, 2 * chunk_len)
    return tensor[tuple(index)]


def _emulator_merge_half(
    selectors: Sequence[Tuple[int, str]],
    partial_out: Sequence[torch.Tensor],
    partial_lse: Sequence[torch.Tensor],
    out_dtype: torch.dtype,
    chunk_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One reduction over ``selectors``: logsumexp then a single weighted sum."""
    outs = [_half_slice(partial_out[step], part, chunk_len, 1) for step, part in selectors]
    lses = [_half_slice(partial_lse[step], part, chunk_len, -1).to(torch.float32) for step, part in selectors]
    stacked = torch.stack(lses, dim=0)
    finite = torch.isfinite(stacked)
    any_finite = finite.any(dim=0)
    # ``logsumexp`` sees the real -inf; substituting 0 for an empty step's L would
    # add exp(0) to the sum and corrupt every row that has one.
    global_lse = torch.logsumexp(stacked, dim=0)
    global_lse = torch.where(any_finite, global_lse, torch.full_like(global_lse, float("-inf")))
    weights = torch.exp(stacked - global_lse.unsqueeze(0))  # [N, B, H, Sq]
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    acc: Optional[torch.Tensor] = None
    for index, out in enumerate(outs):
        term = out.to(torch.float32) * weights[index].movedim(1, 2).unsqueeze(-1)
        acc = term if acc is None else acc + term
    assert acc is not None
    return acc.to(out_dtype), global_lse


def merge_emulator(
    *,
    rank: int,
    cp_size: int,
    causal: bool,
    partial_out: Sequence[torch.Tensor],
    partial_lse: Sequence[torch.Tensor],
    o_local_shape: Sequence[int],
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A **self-written** FP32 schedule emulator -- the second R1 lane.

    Deliberately *not* TE's step order: it forms one global LSE per query half
    with ``logsumexp`` over that half's steps and then a single weighted
    reduction, casting to ``out_dtype`` once at the end. Agreeing with this lane
    proves nothing about TE; it exists to separate "difference caused by the
    merge order" from "difference already present in the frozen partials".

    The step sets still follow the causal section rules (upper-triangle steps
    contribute only to the second query half); dropping that would not be the
    same schedule at all.

    Returns ``(out, lse)`` with the same shapes as :func:`merge_te_fidelity`.
    """
    if len(partial_out) != cp_size or len(partial_lse) != cp_size:
        raise ValueError(f"expected {cp_size} per-step partials, got {len(partial_out)}/{len(partial_lse)}")
    if len(o_local_shape) == 5:  # [B, 2, chunk, H, D]
        chunk_len = int(o_local_shape[2])
    else:  # non-causal [B, 2*chunk, H, D]
        chunk_len = int(o_local_shape[1]) // 2

    first_steps, second_steps = half_step_lists(cp_size, rank, causal)
    first_half, lse_first = _emulator_merge_half(first_steps, partial_out, partial_lse, out_dtype, chunk_len)
    second_half, lse_second = _emulator_merge_half(second_steps, partial_out, partial_lse, out_dtype, chunk_len)

    if len(o_local_shape) == 5:
        out = torch.stack([first_half, second_half], dim=1).reshape(*o_local_shape)
    else:
        out = torch.cat([first_half, second_half], dim=1).reshape(*o_local_shape)
    softmax_lse = torch.cat([lse_first, lse_second], dim=-1)
    if out.dtype != out_dtype:
        raise AssertionError(f"emulator produced {out.dtype}, expected {out_dtype}")
    return out, softmax_lse


# ===========================================================================
# fixture contract
# ===========================================================================


@dataclass
class FixtureHeader:
    """Everything needed to reproduce and interpret a frozen-partial fixture."""

    schema_version: int
    name: str
    lane: str  # "te_fidelity" | "reference_only"
    cp_size: int
    causal: bool
    batch: int
    seq_len: int
    num_heads: int
    head_dim: int
    scale: float
    qkv_dtype: str
    partial_storage_dtype: str
    accumulator_dtype: str
    o_dtype: str
    lse_dtype: str
    lse_base: str
    o_i_normalized: bool
    mask: str
    layout: str
    head_mapping: str
    te_revision: str
    te_tag: str
    te_source_relpath: str
    te_source_sha256: Optional[str]
    fuser_mode: str
    chunk_len: int
    valid_len: int
    te_divisible: bool
    note: str = ""
    generator: str = "test/python/sdpa/cp_numerics/make_fixtures.py"
    generated_utc: str = ""
    fixture_sha256: str = ""
    p_schedule_map: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    empty_chunks: List[List[int]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "FixtureHeader":
        payload = json.loads(text)
        known = set(FixtureHeader.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"fixture header has unknown fields: {sorted(unknown)}")
        missing = known - set(payload)
        if missing:
            raise ValueError(f"fixture header is missing fields: {sorted(missing)}")
        return FixtureHeader(**payload)


@dataclass
class FrozenFixture:
    header: FixtureHeader
    tensors: Dict[str, torch.Tensor]

    def rank_step_out(self, rank: int, step: int) -> torch.Tensor:
        return self.tensors[f"out_r{rank}_s{step}"]

    def rank_step_lse(self, rank: int, step: int) -> torch.Tensor:
        return self.tensors[f"lse_r{rank}_s{step}"]


def _tensor_digest(tensors: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_fixture(path: str, fixture: FrozenFixture) -> str:
    """Write the fixture plus a human-readable ``.header.json`` sidecar."""
    tensors = {k: v.detach().cpu().contiguous() for k, v in fixture.tensors.items()}
    digest = _tensor_digest(tensors)
    fixture.header.fixture_sha256 = digest
    payload = {"header": fixture.header.to_json(), "tensors": tensors}
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)
    with open(os.path.splitext(path)[0] + ".header.json", "w", encoding="utf-8") as handle:
        handle.write(fixture.header.to_json())
    return digest


def load_fixture(path: str, *, verify_digest: bool = True) -> FrozenFixture:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    header = FixtureHeader.from_json(payload["header"])
    tensors = payload["tensors"]
    if verify_digest:
        actual = _tensor_digest(tensors)
        if header.fixture_sha256 and actual != header.fixture_sha256:
            raise ValueError(f"fixture digest mismatch for {path}: header {header.fixture_sha256} != payload {actual}")
    return FrozenFixture(header=header, tensors=tensors)


def fixture_config_digest(header: FixtureHeader) -> str:
    """Digest of everything that must agree across ranks before a collective."""
    keys = (
        "schema_version",
        "name",
        "lane",
        "cp_size",
        "causal",
        "batch",
        "seq_len",
        "num_heads",
        "head_dim",
        "scale",
        "qkv_dtype",
        "partial_storage_dtype",
        "accumulator_dtype",
        "o_dtype",
        "lse_dtype",
        "lse_base",
        "chunk_len",
        "valid_len",
        "te_revision",
        "fixture_sha256",
    )
    blob = json.dumps({k: getattr(header, k) for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def te_requires_divisibility(header: FixtureHeader) -> Tuple[bool, str]:
    """Whether the TE merge lane can be exercised for this fixture, and why not."""
    if header.lane != "te_fidelity":
        return False, f"fixture lane is {header.lane!r}; the TE merge lane needs lane='te_fidelity'"
    if not header.te_divisible:
        return False, (
            "fixture is a ragged tail: seq_len is not divisible by 2*cp_size, and TE's CP path requires "
            "exact divisibility, so this reference-planner-only case is rejected for the TE lane"
        )
    if header.layout != TE_SUPPORTED_LAYOUT:
        return False, f"layout {header.layout!r} is not transcribed; only {TE_SUPPORTED_LAYOUT!r} is"
    if header.lse_base != TE_LSE_BASE:
        return False, f"lse_base {header.lse_base!r} != {TE_LSE_BASE!r}; natural-log and log2 LSE must not be mixed"
    if header.cp_size < 1:
        return False, "cp_size must be >= 1"
    return True, ""


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unknown dtype name {name!r}") from exc


def bytes_per_element(name: str) -> int:
    return torch.tensor([], dtype=dtype_from_name(name)).element_size()


def merge_payload_bytes(rank: int, header: FixtureHeader) -> int:
    """Bytes ``rank`` must read to run the merge: its per-step O_i and LSE_i."""
    b, h, d = header.batch, header.num_heads, header.head_dim
    o_bytes = bytes_per_element(header.partial_storage_dtype)
    l_bytes = bytes_per_element(header.lse_dtype)
    total = 0
    for rec in te_schedule(header.cp_size, header.chunk_len, causal=header.causal):
        if rec.rank != rank:
            continue
        out_tokens = sum(end - begin for begin, end in rec.q_global_ranges)
        total += b * out_tokens * h * d * o_bytes
        total += b * h * out_tokens * l_bytes
    return total


def check_schedule_matches_trace(header: FixtureHeader) -> None:
    """The fixture must carry the very schedule this adapter derives for its P."""
    derived = [
        r.as_dict()
        for r in te_schedule(
            header.cp_size,
            header.chunk_len,
            causal=header.causal,
            partial_dtype=header.partial_storage_dtype,
            accumulator_dtype=header.accumulator_dtype,
            lse_dtype=header.lse_dtype,
        )
    ]
    if len(derived) != len(header.trace):
        raise AssertionError(f"fixture trace has {len(header.trace)} records, derived {len(derived)}")
    for got, want in zip(header.trace, derived):
        if got != want:
            raise AssertionError(f"fixture trace record mismatch:\n  fixture: {got}\n  derived: {want}")
    expected_map = p_schedule_map(header.cp_size, header.chunk_len, causal=header.causal)
    if header.p_schedule_map != expected_map:
        raise AssertionError("fixture p_schedule_map does not match the derived P->schedule table")
    if header.seq_len != 2 * header.cp_size * header.chunk_len:
        raise AssertionError(f"seq_len {header.seq_len} != 2*P*chunk_len = {2 * header.cp_size * header.chunk_len}")


def describe_te_binding() -> Dict[str, Any]:
    return {
        "te_revision": TE_PINNED_SHA,
        "te_tag": TE_PINNED_TAG,
        "te_source_relpath": TE_SOURCE_RELPATH,
        "te_source_sha256": TE_SOURCE_SHA256,
        "fuser_mode": TE_FUSER_MODE,
        "fuser_mode_fallback": TE_FUSER_MODE_EAGER_FALLBACK,
        "layout": TE_SUPPORTED_LAYOUT,
        "seq_dim": TE_SEQ_DIM,
        "lse_dtype": TE_LSE_DTYPE_NAME,
        "lse_base": TE_LSE_BASE,
        "torch": torch.__version__,
    }


def read_recorded_te_sha(evidence_dir: str) -> str:
    path = os.path.join(evidence_dir, "TE_SHA.txt")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().splitlines()[0].strip()
