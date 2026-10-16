# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared THD / varlen (packed ``[T,H,D]`` + ``cu_seqlens``) device helpers.

* :data:`TENSOR_MAP_QWORDS` (from :mod:`.host`), int64 words per 128-byte TMA descriptor.
* :func:`emit_seq_descs`, device helper (one electing thread) that builds
  a per-BATCH TMA-descriptor array in GMEM for varlen loads/stores over a
  packed ``[T,H,D]`` tensor whose head axis is a load coordinate.  Each op
  calls it from a single prologue-kernel launch (one
  warp per array).
* :func:`emit_checkpoint_seq_descs`, its per-chunk-checkpoint sibling; derives the
  per-sequence checkpoint offsets from the token ``cu_seqlens``.
* The token-major (token, slot, head, dim) operands of the sub-token families (GDP's k / v, n = num_householder
  slots per token, any token stride): :func:`expanded_view`, :func:`slot_view`,
  :func:`emit_family_descs`, :func:`tma_load_family`.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda.tensor_map as tma
import cutlass.experimental.primitives as nvvm
from cutlass.base_dsl.typing import Pointer

from .host import TENSOR_MAP_QWORDS

TENSOR_MAP_BIT21 = 1 << 21  # qword 1: encoder's "tensor >= 128 KiB" flag; tensormap.replace does not update it (issue #1013)


@cute.jit
def set_tensor_map_bit21(dptr, new_bytes: cutlass.Int64) -> None:
    """Recompute bit 21 from the patched extent (as cuTensorMapEncodeTiled would)."""
    w = (dptr + 1).load() & cutlass.Int64(~TENSOR_MAP_BIT21)
    (dptr + 1).store((w | cutlass.Int64(TENSOR_MAP_BIT21)) if new_bytes >= cutlass.Int64(128 << 10) else w)


@cute.jit
def emit_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int] = 1,
    lanes: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-BATCH TMA-descriptor array for a VARLEN (THD) tensor whose base
    descriptor carries the head axis as a real dimension (``(d, head,
    token)``); the head index is a load COORDINATE, so only the sequence
    base and length are patched per slot.  GLOBAL_ADDRESS folds
    ``cu_seqlens[b] * base_ptr.stride[0]`` (Int64, the token-axis stride); GLOBAL_DIM[``seq_ord``] is
    capped to the per-sequence token count so tail loads zero-fill and tail
    stores clip in hardware.  ``expand_num`` scales every loaded ``cu``
    value onto GDP's sub-token timeline (1 = off).  GQA/GVA head grouping
    happens at the issue site (``head_idx // group`` with a static group),
    not here.  ``lanes == 1``: one electing thread emits every slot (the
    caller elects and release-fences GENERIC->TENSORMAP); ``lanes == 32``:
    the whole warp calls, lane ``l`` emits slots ``l, l + 32, ...`` and every
    lane release-fences its own."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    first = cutlass.Int32(0)
    if cutlass.const_expr(lanes > 1):
        first = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(lanes)
    for b in cutlass.range(first, n_batch, lanes, unroll=1):
        cu_b = cutlass.Int32(cu[b])
        s_b = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cu_b
        if cutlass.const_expr(expand_num > 1):
            cu_b = cu_b * cutlass.Int32(expand_num)
            s_b = s_b * cutlass.Int32(expand_num)
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(cu_b) * cutlass.Int64(base_ptr.stride[0])
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=addr.toint(cutlass.Int64),
        )
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=s_b,
            ord=seq_ord,
        )
        set_tensor_map_bit21(dptr, cutlass.Int64(s_b) * cutlass.Int64(base_ptr.stride[0]) * cutlass.Int64(base_ptr.element_type.width // 8))


@cute.jit
def emit_tile_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    seq_ord: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int] = 1,
    lanes: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-BATCH descriptor array for the chunk-inverse tile buffer ``(row, head, b_t, b_t)`` of gdn_tinv_f16.  Sequence b
    owns the tile rows from ``cu[b] // b_t + b`` (one padding row per sequence keeps that base closed-form), so
    GLOBAL_ADDRESS advances by that row times ``base_ptr.stride[0]`` and GLOBAL_DIM[``seq_ord``] is capped to the
    sequence's chunk count.  ``expand_num`` scales the loaded ``cu`` values onto GDP's sub-token timeline (1 = off);
    ``lanes`` as in :func:`emit_seq_descs`."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    first = cutlass.Int32(0)
    if cutlass.const_expr(lanes > 1):
        first = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(lanes)
    for b in cutlass.range(first, n_batch, lanes, unroll=1):
        cu_b = cutlass.Int32(cu[b])
        s_b = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cu_b
        if cutlass.const_expr(expand_num > 1):
            cu_b = cu_b * cutlass.Int32(expand_num)
            s_b = s_b * cutlass.Int32(expand_num)
        row_b = cu_b // cutlass.Int32(b_t) + b
        n_b = (s_b + cutlass.Int32(b_t - 1)) // cutlass.Int32(b_t)
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(row_b) * cutlass.Int64(base_ptr.stride[0])
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=addr.toint(cutlass.Int64),
        )
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=n_b,
            ord=seq_ord,
        )
        set_tensor_map_bit21(dptr, cutlass.Int64(n_b) * cutlass.Int64(base_ptr.stride[0]) * cutlass.Int64(base_ptr.element_type.width // 8))


@cute.jit
def emit_checkpoint_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    every_n: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int] = 1,
    lanes: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-BATCH descriptor array for the per-chunk checkpoint tensor with the head
    axis as a descriptor dimension (``(dv, dk, chunk, head)``).  Derives the
    per-sequence checkpoint offsets from the TOKEN ``cu_seqlens`` on the fly
    (``count_b = (batch_seqlen - 1) // every_n + 1``, running-prefix-summed) and
    caps GLOBAL_DIM[``seq_ord``] to ``count_b``; GLOBAL_ADDRESS advances by
    ``count_prefix * base_ptr.stride[0]`` (the checkpoint-row stride).  ``expand_num`` scales the
    loaded per-sequence token counts onto GDP's sub-token timeline (1 =
    off).  The head index is a load coordinate.  ``lanes`` as in
    :func:`emit_seq_descs`; a striped lane re-derives the prefix of its
    slot from the sequences before it."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    first = cutlass.Int32(0)
    if cutlass.const_expr(lanes > 1):
        first = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(lanes)
    run = cutlass.Int32(0)
    for b0 in cutlass.range(0, first, 1, unroll=1):
        if b0 < n_batch:
            s_tok0 = cutlass.Int32(cu[b0 + cutlass.Int32(1)]) - cutlass.Int32(cu[b0])
            if cutlass.const_expr(expand_num > 1):
                s_tok0 = s_tok0 * cutlass.Int32(expand_num)
            cnt0 = (s_tok0 - cutlass.Int32(1)) // every_n + cutlass.Int32(1)
            run = run + (cnt0 if s_tok0 > 0 else cutlass.Int32(0))
    for b in cutlass.range(first, n_batch, lanes, unroll=1):
        s_tok = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cutlass.Int32(cu[b])
        if cutlass.const_expr(expand_num > 1):
            s_tok = s_tok * cutlass.Int32(expand_num)
        cnt = (s_tok - cutlass.Int32(1)) // every_n + cutlass.Int32(1)
        cnt = cnt if s_tok > 0 else cutlass.Int32(0)
        checkpoint_base = run
        run = run + cnt
        for bn in cutlass.range(b + cutlass.Int32(1), b + cutlass.Int32(lanes), 1, unroll=1):
            if bn < n_batch:
                s_tokn = cutlass.Int32(cu[bn + cutlass.Int32(1)]) - cutlass.Int32(cu[bn])
                if cutlass.const_expr(expand_num > 1):
                    s_tokn = s_tokn * cutlass.Int32(expand_num)
                cntn = (s_tokn - cutlass.Int32(1)) // every_n + cutlass.Int32(1)
                run = run + (cntn if s_tokn > 0 else cutlass.Int32(0))
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(checkpoint_base) * cutlass.Int64(base_ptr.stride[0])
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=addr.toint(cutlass.Int64),
        )
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=cnt,
            ord=seq_ord,
        )
        set_tensor_map_bit21(dptr, cutlass.Int64(cnt) * cutlass.Int64(base_ptr.stride[0]) * cutlass.Int64(base_ptr.element_type.width // 8))


@cute.jit
def expanded_view(t: cute.Tensor, slots: cutlass.Constexpr[int], expand_num: cutlass.Constexpr[int]):
    """The (token, slot, head, dim) view of a sub-token operand for a load family of ``slots`` slots per token.
    ``slots == 1``: the operand's expanded rows have one uniform stride (expanded ``[token * n, head, dim]``, or a
    token-major ``[token, n, head, dim]`` view whose token stride is ``n`` slot strides), viewed ``(row, 1, head, dim)`` so
    every chunk is one ``b_t``-row box (the pre-family load path); at ``expand_num == 1`` the token-major slot axis is a
    singleton with an arbitrary stride, so the rows take the token stride.  ``slots == n``: the token-major operand with
    its own token stride, ``(token, n, head, dim)`` for the box family (an expanded rank-3 input is folded onto that view)."""
    if cutlass.const_expr(len(t.shape) == 4):
        if cutlass.const_expr(slots == 1):
            if cutlass.const_expr(expand_num == 1):
                return cute.make_tensor(
                    t.iterator, cute.make_layout((t.shape[0], 1, t.shape[2], t.shape[3]), stride=(t.stride[0], t.stride[0], t.stride[2], t.stride[3]))
                )
            return cute.make_tensor(
                t.iterator,
                cute.make_layout((t.shape[0] * t.shape[1], 1, t.shape[2], t.shape[3]), stride=(t.stride[1], t.stride[1], t.stride[2], t.stride[3])),
            )
        return t
    if cutlass.const_expr(slots == 1):
        return cute.make_tensor(
            t.iterator, cute.make_layout((t.shape[0], 1, t.shape[1], t.shape[2]), stride=(t.stride[0], t.stride[0], t.stride[1], t.stride[2]))
        )
    return cute.make_tensor(
        t.iterator,
        cute.make_layout((t.shape[0] // slots, slots, t.shape[1], t.shape[2]), stride=(t.stride[0] * slots, t.stride[0], t.stride[1], t.stride[2])),
    )


@cute.jit
def slot_view(t: cute.Tensor, slots: cutlass.Constexpr[int], expand_num: cutlass.Constexpr[int]):
    """The (token, slot, head) view of a per-row scalar operand (beta) for ``slots`` slots per token: ``slots == 1`` the
    uniform rows as ``(row, 1, head)`` (expanded ``[token * n, head]`` or a dense token-major ``[token, n, head]``; at
    ``expand_num == 1`` the singleton slot axis is skipped and the rows take the token stride), else the token-major
    ``(token, n, head)`` with its own strides."""
    if cutlass.const_expr(len(t.shape) == 3):
        if cutlass.const_expr(slots == 1):
            if cutlass.const_expr(expand_num == 1):
                return cute.make_tensor(t.iterator, cute.make_layout((t.shape[0], 1, t.shape[2]), stride=(t.stride[0], t.stride[0], t.stride[2])))
            return cute.make_tensor(t.iterator, cute.make_layout((t.shape[0] * t.shape[1], 1, t.shape[2]), stride=(t.stride[1], t.stride[1], t.stride[2])))
        return t
    if cutlass.const_expr(slots == 1):
        return cute.make_tensor(t.iterator, cute.make_layout((t.shape[0], 1, t.shape[1]), stride=(t.stride[0], t.stride[0], t.stride[1])))
    return cute.make_tensor(t.iterator, cute.make_layout((t.shape[0] // slots, slots, t.shape[1]), stride=(t.stride[0] * slots, t.stride[0], t.stride[1])))


@cute.jit
def emit_family_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    slots: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int] = 1,
    lanes: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-BATCH descriptor family of a sub-token operand with ``slots`` slots per token (:func:`expanded_view`).
    ``slots == 1``: the operand's rows are uniform and ``base_desc`` is the (dim, head, row) map with the
    ``b_t``-row box; one descriptor per batch over the EXPANDED rows, ``expand_num`` scaling ``cu`` as
    :func:`emit_seq_descs` does.  Otherwise ``base_desc`` is 4-D in ``(dim, head, slot, token)`` order with box
    ``(granule, 1, slots, full_tokens)``: ``slots + 1`` arrays of ``n_batch`` descriptors each (one when ``slots`` divides ``b_t``; slot ``s`` at word
    ``s * n_batch * TENSOR_MAP_QWORDS``), the box re-patched per slot (BOX_DIM ord 3 = tokens, ord 2 = slots) and every
    descriptor patched per sequence over the TOKEN axis (GLOBAL_ADDRESS by ``cu[b] * base_ptr.stride[0]``, GLOBAL_DIM[3] =
    the sequence's tokens, bit 21 recomputed).  ``lanes`` as in :func:`emit_seq_descs`."""
    base_ptr = expanded_view(base_ptr, slots, expand_num)
    if cutlass.const_expr(slots == 1):
        emit_seq_descs(base_desc, desc_words, cu_seqlens, base_ptr, n_batch, 2, expand_num, lanes)
    else:
        full_tokens = b_t // slots
        n_slots = 1 if b_t % slots == 0 else slots + 1
        desc_base = desc_words.iterator.raw_ptr()
        src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
        cu = cutlass.make_array_view(cu_seqlens)
        base = base_ptr.iterator.raw_ptr()
        first = cutlass.Int32(0)
        if cutlass.const_expr(lanes > 1):
            first = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(lanes)
        arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
        for s in cutlass.range_constexpr(n_slots):
            for b in cutlass.range(first, n_batch, lanes, unroll=1):
                cu_b = cutlass.Int32(cu[b])
                s_b = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cu_b
                dptr = desc_base + s * arr_words + b * cutlass.Int32(TENSOR_MAP_QWORDS)
                for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                    (dptr + i).store((src_words + i).load())
                if cutlass.const_expr(s == 1):
                    nvvm.tensormap_replace(nvvm.TensormapField.BOX_DIM, dptr, new_value=cutlass.Int32(full_tokens - 1), ord=3)
                if cutlass.const_expr(s >= 2):
                    nvvm.tensormap_replace(nvvm.TensormapField.BOX_DIM, dptr, new_value=cutlass.Int32(1), ord=3)
                    nvvm.tensormap_replace(nvvm.TensormapField.BOX_DIM, dptr, new_value=cutlass.Int32(s - 1), ord=2)
                addr = base + cutlass.Int64(cu_b) * cutlass.Int64(base_ptr.stride[0])
                nvvm.tensormap_replace(
                    nvvm.TensormapField.GLOBAL_ADDRESS,
                    dptr,
                    new_value=addr.toint(cutlass.Int64),
                )
                nvvm.tensormap_replace(
                    nvvm.TensormapField.GLOBAL_DIM,
                    dptr,
                    new_value=s_b,
                    ord=3,
                )
                set_tensor_map_bit21(dptr, cutlass.Int64(s_b) * cutlass.Int64(base_ptr.stride[0]) * cutlass.Int64(base_ptr.element_type.width // 8))


@cute.jit
def tma_load_family(
    smem_tile,
    desc_family,
    arr_words,
    head,
    chunk_idx,
    mbar,
    d_offset,
    b_t: cutlass.Constexpr[int],
    slots: cutlass.Constexpr[int],
) -> None:
    """Load the ``b_t``-row chunk ``chunk_idx`` of a sub-token operand with ``slots`` slots per token into ``smem_tile``
    (the row-major expanded tile), one elected lane issuing per column granule.  ``slots == 1``: one box of the (dim,
    head, row) map at expanded row ``chunk_idx * b_t``.  ``slots`` dividing ``b_t``: one box of the (dim, head, slot,
    token) map at token ``chunk_idx * (b_t // slots)``.  Otherwise the family: the chunk starts at expanded row ``u =
    chunk_idx * b_t`` = token ``u // n`` slot ``u % n``; a leading partial token (``n - p`` rows, when ``p > 0``), the
    full-token run (slot 0 or 1 descriptor), a trailing partial token (``(u + b_t) % n`` rows, when non-zero), each at its
    row offset of the tile.  Rows past the sequence's extent zero-fill and still count toward the barrier's expected
    bytes, so the caller expects the whole tile.  ``d_offset`` is the first column of the tile (the d_v split's half)."""
    full_tokens = b_t // slots
    n_slots = 1 if b_t % slots == 0 else slots + 1
    num_iters = smem_tile.tma_loads_per_tile
    granu = smem_tile.tma_granu_elems
    sub_stride = smem_tile.tma_subtile_stride_elems
    if cutlass.const_expr(slots == 1):
        row = chunk_idx * cutlass.Int32(b_t)
        desc = desc_family.tospace(cutlass.AddressSpace.generic)
        for i in cutlass.range_constexpr(num_iters):
            if nvvm.elect_sync():
                nvvm.cp_async_bulk_tensor_shared_cta_global(
                    smem_tile.base.subview(i * sub_stride), desc, [d_offset + cutlass.Int32(i * granu), head, row], mbar
                )
    else:
        if cutlass.const_expr(n_slots == 1):
            tok = chunk_idx * cutlass.Int32(b_t // slots)
            desc = desc_family.tospace(cutlass.AddressSpace.generic)
            for i in cutlass.range_constexpr(num_iters):
                if nvvm.elect_sync():
                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                        smem_tile.base.subview(i * sub_stride), desc, [d_offset + cutlass.Int32(i * granu), head, cutlass.Int32(0), tok], mbar
                    )
        else:
            n = cutlass.Int32(slots)
            u = chunk_idx * cutlass.Int32(b_t)
            t0 = u // n
            p = u - t0 * n
            lead = n - p if p > 0 else cutlass.Int32(0)
            u_end = u + cutlass.Int32(b_t)
            trail = u_end - (u_end // n) * n
            full = (cutlass.Int32(b_t) - lead - trail) // n
            full_slot = cutlass.Int32(0) if full == cutlass.Int32(full_tokens) else cutlass.Int32(1)
            t_full = t0 + (cutlass.Int32(1) if p > 0 else cutlass.Int32(0))
            t_trail = t_full + full
            desc_lead = (desc_family + (cutlass.Int32(1) + lead) * arr_words).tospace(cutlass.AddressSpace.generic)
            desc_full = (desc_family + full_slot * arr_words).tospace(cutlass.AddressSpace.generic)
            desc_trail = (desc_family + (cutlass.Int32(1) + trail) * arr_words).tospace(cutlass.AddressSpace.generic)
            for i in cutlass.range_constexpr(num_iters):
                d = d_offset + cutlass.Int32(i * granu)
                if nvvm.elect_sync():
                    if p > 0:
                        nvvm.cp_async_bulk_tensor_shared_cta_global(smem_tile.base.subview(i * sub_stride), desc_lead, [d, head, p, t0], mbar)
                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                        smem_tile.base.subview(i * sub_stride + lead * granu), desc_full, [d, head, cutlass.Int32(0), t_full], mbar
                    )
                    if trail > 0:
                        nvvm.cp_async_bulk_tensor_shared_cta_global(
                            smem_tile.base.subview(i * sub_stride + (cutlass.Int32(b_t) - trail) * granu),
                            desc_trail,
                            [d, head, cutlass.Int32(0), t_trail],
                            mbar,
                        )
