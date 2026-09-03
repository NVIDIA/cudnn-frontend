# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""THD / varlen setup launch for the SM100 SDPA backward chain.

One launch per execute, ahead of stages 1-3, building everything the three
kernels need that only the DEVICE can know (the packed totals and per-sequence
lengths live in `cu_seqlens`, and reading them on the host would be a D2H sync
per iteration -- issue #552):

* the shared THD metadata buffer (`tile_dsl.thd`), extended with the
  ``row_off(B+1)`` block offsets of the **blocked S/dS workspace**;
* the persistent scheduler's live-unit total and claim counter;
* **per-sequence output descriptors** for dQ / dK / dV.  Stage 3's last M tile
  of a sequence overshoots into the NEXT sequence's rows with a live
  accumulator behind it, so the store must be clipped at the sequence
  boundary.  Patching each descriptor's ``GLOBAL_DIM[seq]`` does that in
  hardware -- dense packing with no predicated epilogue;
* **packed-total-clamped input descriptors** for Q / K / V / dO.  A THD caller
  binds these at buffer CAPACITY, so a tile-tail read steps into caller-owned
  bytes that may never have been written (issue #624).  Zero score columns
  times a NaN operand row is NaN, so the tail must read as zeros, which is what
  clamping the extent to the packed total gets for free.

The descriptor slot map (each slot is one 128-byte tensor map)::

    [ 0     .. B-1  ]  dQ, per sequence   (base cu_q[b], extent s_q[b])
    [ B     .. 2B-1 ]  dK, per sequence   (base cu_k[b], extent s_kv[b])
    [ 2B    .. 3B-1 ]  dV, per sequence   (base cu_k[b], extent s_kv[b])
      3B               Q,  clamped to cu_q[B]
      3B+1             dO, clamped to cu_q[B]
      3B+2             K,  clamped to cu_k[B]
      3B+3             V,  clamped to cu_k[B]

``THD_BWD_DESC_SLOTS(B)`` sizes the buffer; the accessors below are the only
place the map is written down, so a reader and a writer cannot drift apart.
"""

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.thd import (
    TENSOR_MAP_QWORDS,
    THD_BWD_META_WORDS,
    THD_ROWOFF_OFF,
    THD_SETUP_THREADS,
    emit_clamped_desc,
    emit_seq_descs,
    write_thd_batch_remap,
    write_thd_live_and_ctr,
    write_thd_meta,
    write_thd_row_offsets,
)

# Descriptor slots: three per-sequence output arrays + four clamped inputs.
THD_BWD_DESC_SLOTS = lambda b: 3 * b + 4  # noqa: E731
# Bases of the three per-sequence ARRAYS (sequence b sits at base + b) ...
DQ_SLOT_BASE = lambda b: 0  # noqa: E731
DK_SLOT_BASE = lambda b: b  # noqa: E731
DV_SLOT_BASE = lambda b: 2 * b  # noqa: E731
# ... and the four single clamped whole-tensor descriptors.
Q_SLOT = lambda b: 3 * b  # noqa: E731
DO_SLOT = lambda b: 3 * b + 1  # noqa: E731
K_SLOT = lambda b: 3 * b + 2  # noqa: E731
V_SLOT = lambda b: 3 * b + 3  # noqa: E731

# The sequence axis of a packed [1, T, H, D] view. `ord` is innermost-first, so
# with D contiguous the token axis is 2 -- NOT 1.
_SEQ_ORD = 2

__all__ = [
    "DK_SLOT_BASE",
    "DO_SLOT",
    "DQ_SLOT_BASE",
    "DV_SLOT_BASE",
    "K_SLOT",
    "Q_SLOT",
    "THD_BWD_DESC_SLOTS",
    "V_SLOT",
    "build_thd_bwd_setup_kernel",
]


@cute.kernel
def build_thd_bwd_setup_kernel(
    dq_tensor: cute.Tensor,
    dk_tensor: cute.Tensor,
    dv_tensor: cute.Tensor,
    base_dq_desc: cutlass.GridConstant[tmap.TensorMap],
    base_dk_desc: cutlass.GridConstant[tmap.TensorMap],
    base_dv_desc: cutlass.GridConstant[tmap.TensorMap],
    base_q_desc: cutlass.GridConstant[tmap.TensorMap],
    base_do_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    dq_row_stride: cutlass.Int32,
    dk_row_stride: cutlass.Int32,
    dv_row_stride: cutlass.Int32,
    ws_gran: cutlass.Int32,
    cga_tile_m: cutlass.Int32,
    n_clusters: cutlass.Int32,
) -> None:
    """Build the metadata, the workspace block offsets and all seven descriptor
    groups.  ``*_row_stride`` are in ELEMENTS of each target's dtype (``H*D``
    for a packed ``[1, T, H, D]`` buffer) -- ``.raw_ptr()`` is element-
    addressed, so a byte stride here would double every offset.  ``ws_gran`` is
    stage 2's Q-row write granularity (``TILE_M * CTA_MMA``), which is what the
    S/dS blocks are padded to.  ``cga_tile_m`` is stage 2's unit height, which
    is the same number today but is passed separately so a tile change cannot
    silently redefine the workspace layout."""
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    # Warp 0's leader only: elect_sync elects one thread PER WARP, and this
    # block is THD_SETUP_THREADS wide for the parallel ranking below. Without
    # the tidx predicate, one warp's descriptor base-copy can land after
    # another's tensormap_replace and revert the patched address.
    if nvvm.elect_sync() and tidx < cutlass.Int32(32):
        meta = cutlass.make_array_view(meta_t)
        write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
        # Blocked-workspace row offsets, and then every descriptor -- all from
        # the cu_seqlens written just above, on the same thread, so plain
        # program order is the only ordering needed.
        write_thd_row_offsets(meta, n_batch, ws_gran)
        cu_q0 = n_batch
        cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
        emit_seq_descs(base_dq_desc, desc_words, meta, cu_q0, dq_tensor, n_batch, dq_row_stride, seq_ord=_SEQ_ORD, slot_base=0)
        emit_seq_descs(base_dk_desc, desc_words, meta, cu_k0, dk_tensor, n_batch, dk_row_stride, seq_ord=_SEQ_ORD, slot_base=n_batch)
        emit_seq_descs(base_dv_desc, desc_words, meta, cu_k0, dv_tensor, n_batch, dv_row_stride, seq_ord=_SEQ_ORD, slot_base=cutlass.Int32(2) * n_batch)
        t_q = cutlass.Int32(meta[cu_q0 + n_batch])  # cu_q[B]
        t_kv = cutlass.Int32(meta[cu_k0 + n_batch])  # cu_k[B]
        base3 = cutlass.Int32(3) * n_batch
        emit_clamped_desc(base_q_desc, desc_words, base3, t_q, seq_ord=_SEQ_ORD)
        emit_clamped_desc(base_do_desc, desc_words, base3 + cutlass.Int32(1), t_q, seq_ord=_SEQ_ORD)
        emit_clamped_desc(base_k_desc, desc_words, base3 + cutlass.Int32(2), t_kv, seq_ord=_SEQ_ORD)
        emit_clamped_desc(base_v_desc, desc_words, base3 + cutlass.Int32(3), t_kv, seq_ord=_SEQ_ORD)
        # ONE release fence covers every array built above.
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )
    # Outside the elect: every thread helps rank the batches. The barrier makes
    # the cu_seqlens_q written above visible to the whole block first. Stage 2's
    # decode walks this permutation, so skipping it leaves the region
    # uninitialized and units decode garbage batches.
    cute.arch.barrier()
    write_thd_batch_remap(cutlass.make_array_view(meta_t), n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    cute.arch.barrier()
    write_thd_live_and_ctr(cutlass.make_array_view(meta_t), n_batch, n_qh, cga_tile_m, n_clusters, cutlass.Int32(tidx))


build_thd_bwd_setup_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
