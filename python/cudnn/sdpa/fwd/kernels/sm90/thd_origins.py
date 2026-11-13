# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""THD setup for SM90 plans that read the graph's bound ragged offsets (caller-buffer origins).

A ragged port's sequence ``b`` starts at element ``ro[b] * M`` of the caller's buffer, which
TE-style padded THD (``cu_seqlens_padded``) puts somewhere other than ``prefix(lengths)``. This
launch writes the shared metadata (lengths and compact prefixes, which still define every
sequence's LENGTH), the per-port token origins, and one map per sequence for O, K and V, based at
that sequence's own origin and clipped to its own length. K/V tile tails past a sequence end are
then TMA-OOB zeros instead of the next sequence's rows or the gap after it, which may hold NaN
(``0 * NaN`` in P @ V would poison the row), and O tile stores never reach a gap.

Origins are runtime data, so every one is clamped into its buffer: a bad offset addresses
garbage rows of the bound buffer, never memory outside it. The SM90 work decode reads only the
lengths, prefixes, origins and maps; the batch ranking and unit prefix stay unwritten.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.tile_dsl.thd import (
    TENSOR_MAP_QWORDS,
    THD_SETUP_THREADS,
    set_tensor_map_bit21,
    thd_port_origin,
    write_thd_meta,
    write_thd_prefix_warp,
)

# Origin rows, in TemplateParams.thd_origins order.
ROW_Q, ROW_K, ROW_V, ROW_O, ROW_STATS = range(5)


def origin_row(specs, row: int) -> int:
    """The origins row a port reads, or -1 for a port without offsets (compact prefix)."""
    return row if specs is not None and specs[row] is not None else -1


@cute.jit
def write_clamped_origins(
    org, ro, row: cutlass.Constexpr[int], mult: cutlass.Constexpr[int], ts: cutlass.Int64, cap: cutlass.Int64, n_batch, tid, nthreads
) -> None:
    """``org[row, b] = clamp(ro[b] * mult // ts, 0, cap)`` for this thread's batches.

    ``ts`` is the port's runtime token stride in elements; FROST takes whole-token offsets as a
    supported-input precondition. The clamp keeps a bad offset inside the bound buffer."""
    org_p = cutlass.make_array_view(org).data_ptr() + cutlass.Int64(row) * cutlass.Int64(org.stride[0])
    ro_p = cutlass.make_array_view(ro).data_ptr()
    for b in cutlass.range(tid, n_batch, nthreads, unroll=1):
        v = (cutlass.Int64(Pointer(ro_p + b, dtype=ro.element_type).load()) * cutlass.Int64(mult)) // ts
        Pointer(org_p + b, dtype=cutlass.Int64).store(cute.math.min(cute.math.max(v, cutlass.Int64(0)), cap))


@cute.jit
def emit_origin_maps(
    base_desc,
    maps,
    meta,
    cu0: cutlass.Int32,
    base_tensor,
    n_batch: cutlass.Int32,
    row_stride: cutlass.Int64,
    cap: cutlass.Int64,
    slot_base: cutlass.Int32,
    first_batch,
    batch_step,
    lane,
    origins,
    row: cutlass.Constexpr[int],
) -> None:
    """One map per sequence over a ``(H, T, D)`` port, based at its origin and clipped.

    ``emit_seq_descs`` with the extent clipped to the room left in the buffer after the origin as
    well as to the sequence length (``meta[cu0 + b + 1] - meta[cu0 + b]``). The extent stays >= 1:
    a zero-extent map is invalid, and the scheduler hands out no tile of an empty sequence.

    Called by every lane of a warp: the warp's lanes copy a map's 16 qwords together, and lane 0
    patches it after the warp barrier (``3B + 1`` maps would otherwise serialize on one thread)."""
    desc_base = maps.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    base = base_tensor.iterator.raw_ptr()
    for b in cutlass.range(first_batch, n_batch, batch_step, unroll=1):
        cu_b = cutlass.Int32(meta[cu0 + b])
        s_b = cutlass.Int64(cutlass.Int32(meta[cu0 + b + cutlass.Int32(1)]) - cu_b)
        dptr = desc_base + (b + slot_base) * cutlass.Int32(TENSOR_MAP_QWORDS)
        if lane < cutlass.Int32(TENSOR_MAP_QWORDS):
            (dptr + lane).store((src_words + lane).load())
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        if lane == cutlass.Int32(0):
            # Compact prefixes are bounded by the binder's capacity; origins are clamped by the writer.
            origin = cute.math.min(thd_port_origin(origins, row, b, cu_b), cute.math.max(cap - cutlass.Int64(1), cutlass.Int64(0)))
            extent = cutlass.Int32(cute.math.max(cute.math.min(s_b, cap - origin), cutlass.Int64(1)))
            nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_ADDRESS, dptr, new_value=(base + origin * row_stride).toint(cutlass.Int64))
            nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_DIM, dptr, new_value=extent, ord=2)
            set_tensor_map_bit21(dptr, cutlass.Int64(extent) * row_stride * cutlass.Int64(base_tensor.element_type.width // 8))


@cute.kernel
def build_thd_origin_setup_kernel(
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    maps: cute.Tensor,
    origins: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
    ro_q: Optional[cute.Tensor],
    ro_k: Optional[cute.Tensor],
    ro_v: Optional[cute.Tensor],
    ro_o: Optional[cute.Tensor],
    ro_stats: Optional[cute.Tensor],
    token_strides,
    caps,
    specs: cutlass.Constexpr,
) -> None:
    """Metadata, clamped token origins, then the per-sequence O/K/V maps.

    ``token_strides`` and ``caps`` are each port's (Q, K, V, O, Stats) token stride in elements
    and token capacity. Two warps build the prefixes for B > 1, as in the shared setup launches;
    every thread writes origins; after the block barrier each warp builds disjoint sequences'
    maps and releases them to the tensor-map proxy. Callers launch
    ``THD_SETUP_THREADS`` threads."""
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp, lane = tid // 32, tid % 32
    meta = cutlass.make_array_view(meta_t)
    if n_batch <= cutlass.Int32(1):
        if tid == cutlass.Int32(0):
            write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
    else:
        if warp == cutlass.Int32(0):
            write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens_t), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
        if warp == cutlass.Int32(1):
            write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens_t), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
    offsets = (ro_q, ro_k, ro_v, ro_o, ro_stats)
    for row in cutlass.range_constexpr(5):
        if cutlass.const_expr(specs[row] is not None):
            write_clamped_origins(
                origins, offsets[row], row, specs[row][1], token_strides[row], cutlass.Int64(caps[row]), n_batch, tid, cutlass.Int32(THD_SETUP_THREADS)
            )
    cute.arch.barrier()
    step = cutlass.Int32(THD_SETUP_THREADS // 32)
    cu_k = 2 * n_batch + 1
    emit_origin_maps(
        base_o_desc,
        maps,
        meta,
        n_batch,
        o,
        n_batch,
        token_strides[ROW_O],
        cutlass.Int64(caps[ROW_O]),
        cutlass.Int32(0),
        warp,
        step,
        lane,
        origins,
        origin_row(specs, ROW_O),
    )
    emit_origin_maps(
        base_k_desc,
        maps,
        meta,
        cu_k,
        k,
        n_batch,
        token_strides[ROW_K],
        cutlass.Int64(caps[ROW_K]),
        n_batch + 1,
        warp,
        step,
        lane,
        origins,
        origin_row(specs, ROW_K),
    )
    emit_origin_maps(
        base_v_desc,
        maps,
        meta,
        cu_k,
        v,
        n_batch,
        token_strides[ROW_V],
        cutlass.Int64(caps[ROW_V]),
        2 * n_batch + 1,
        warp,
        step,
        lane,
        origins,
        origin_row(specs, ROW_V),
    )
    nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


build_thd_origin_setup_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
