# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.thd import THD_SETUP_THREADS, thd_port_origin
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_accum_thd_helpers import setup_thd_d128_kernel, _origin_row, _cast_group_vector
from cudnn.sdpa.bwd.kernels.sm100.bprop_d128_helpers import THREADS, ROWS_PER_CTA, LANES_PER_ROW, ELEMENTS_PER_LANE, RN
from cudnn.sdpa.bwd.config_sm100_d192_accum import CFG
from .bprop_d192_accum_f16 import _tensor_map
from .bprop_d192_accum_thd_f16 import host as accumulation_host


@cute.kernel
def delta_zero_thd_d192_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    origin_specs: cutlass.Constexpr,
):
    "Eight rows per CTA; all lanes participate in the FP32 delta dot tree."
    block, batch, head = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row = cutlass.Int32(block) * ROWS_PER_CTA + cutlass.Int32(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    data = cutlass.make_array_view(meta)
    start = cutlass.Int32(data[n_batch + batch])
    length = cutlass.Int32(data[n_batch + batch + 1]) - start
    live = row < length
    do_values = cutlass.Vector.from_elements(tuple(do.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), do.element_type)
    o_values = cutlass.Vector.from_elements(tuple(o.element_type(0.0) for _ in range(ELEMENTS_PER_LANE)), o.element_type)
    if live:
        token = cutlass.Int64(start) + cutlass.Int64(row)
        dq_offset = token * cutlass.Int64(dq_accum.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2]) + column
        zero4 = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(4)), cutlass.Float32)
        (dq_accum.iterator.raw_ptr() + dq_offset).store(zero4, alignment=16)
        (dq_accum.iterator.raw_ptr() + dq_offset + 4).store(zero4, alignment=16)
        if column < 64:
            (dq_accum.iterator.raw_ptr() + dq_offset + 128).store(zero4, alignment=16)
            (dq_accum.iterator.raw_ptr() + dq_offset + 132).store(zero4, alignment=16)
        o_token = thd_port_origin(origins, _origin_row(origin_specs, 3), batch, start) + cutlass.Int64(row)
        do_token = thd_port_origin(origins, _origin_row(origin_specs, 4), batch, start) + cutlass.Int64(row)
        o_offset = o_token * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
        do_offset = do_token * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
        do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
        o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    pairs = []
    for j in cutlass.range_constexpr(ELEMENTS_PER_LANE // 2):
        high = cute.math.mul(do_values[2 * j + 1].to(cutlass.Float32), o_values[2 * j + 1].to(cutlass.Float32), rounding=RN, ftz=True)
        pairs.append(cute.math.fma(do_values[2 * j].to(cutlass.Float32), o_values[2 * j].to(cutlass.Float32), high, rounding=RN, ftz=True))
    dot = cute.math.add(pairs[0], pairs[1], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[2], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[3], rounding=RN, ftz=True)
    # XOR distances 8,4,2,1 never cross a 16-lane row. Inactive row lanes
    # carry zeros but remain active for the full-warp shuffle mask.
    for step in cutlass.range_constexpr(4):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=1 << (3 - step), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, cutlass.Float32(1.0), rounding=RN, ftz=True)
    if live and tid % LANES_PER_ROW == 0:
        token = cutlass.Int64(start) + cutlass.Int64(row)
        delta_offset = cutlass.Int64(head) * cutlass.Int64(delta.stride[1]) + token * cutlass.Int64(delta.stride[2])
        (delta.iterator.raw_ptr() + delta_offset).store(dot)


delta_zero_thd_d192_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def cast_dq_thd_d192_kernel(
    dq_accum: cute.Tensor,
    dq_out: cute.Tensor,
    meta: cute.Tensor,
    origins: cute.Tensor,
    n_batch: cutlass.Int32,
    dq_origin: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
):
    "Cast compact FP32 dQ into each external sequence's own native rows."
    block, batch, head = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row = cutlass.Int32(block) * ROWS_PER_CTA + cutlass.Int32(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    data = cutlass.make_array_view(meta)
    start = cutlass.Int32(data[n_batch + batch])
    length = cutlass.Int32(data[n_batch + batch + 1]) - start
    if row < length:
        src_token = cutlass.Int64(start) + cutlass.Int64(row)
        dst_token = thd_port_origin(origins, dq_origin, batch, start) + cutlass.Int64(row)
        src_offset = src_token * cutlass.Int64(dq_accum.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2]) + column
        dst_offset = dst_token * cutlass.Int64(dq_out.stride[1]) + cutlass.Int64(head) * cutlass.Int64(dq_out.stride[2]) + column
        lo = (dq_accum.iterator.raw_ptr() + src_offset).load(count=4, alignment=16)
        hi = (dq_accum.iterator.raw_ptr() + src_offset + 4).load(count=4, alignment=16)
        packed = cutlass.Vector.from_elements(
            (
                fp32_to_fp16(lo[0], lo[1], dtype=io_dtype),
                fp32_to_fp16(lo[2], lo[3], dtype=io_dtype),
                fp32_to_fp16(hi[0], hi[1], dtype=io_dtype),
                fp32_to_fp16(hi[2], hi[3], dtype=io_dtype),
            ),
            cutlass.Int32,
        )
        Pointer(cutlass.make_array_view(dq_out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)
        if column < 64:
            _cast_group_vector(dq_accum, dq_out, origins, dq_origin, batch, start, row, head, column + 128, io_dtype)


cast_dq_thd_d192_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def host_thd(
    q,
    k,
    v,
    o,
    do,
    lse,
    dq,
    dk,
    dv,
    dq_accum,
    delta,
    meta,
    origins,
    maps,
    q_lens,
    kv_lens,
    ro,
    origin_specs: cutlass.Constexpr,
    lens_form,
    batch: cutlass.Constexpr[int],
    max_sq,
    max_skv,
    scale,
    causal: cutlass.Constexpr[bool],
    stream,
):
    """Four declared native launches; no host reads, tensor copies or allocation."""
    setup_thd_d128_kernel(
        _tensor_map(q, CFG.N, CFG.HEAD_BOX),
        _tensor_map(k, CFG.M, CFG.HEAD_BOX),
        _tensor_map(v, CFG.M, CFG.HEAD_BOX),
        _tensor_map(do, CFG.N, CFG.HEAD_BOX),
        _tensor_map(dq_accum, CFG.N, CFG.DQ_BOX),
        _tensor_map(dk, CFG.M, CFG.HEAD_BOX),
        _tensor_map(dv, CFG.M, CFG.HEAD_BOX),
        q,
        k,
        v,
        do,
        dq_accum,
        dk,
        dv,
        meta,
        origins,
        maps,
        q_lens,
        kv_lens,
        *ro,
        origin_specs,
        lens_form,
        cutlass.Int32(batch),
    ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    q_grid = cute.math.max((max_sq + ROWS_PER_CTA - 1) // ROWS_PER_CTA, cutlass.Int32(1))
    delta_zero_thd_d192_kernel(o, do, delta, dq_accum, meta, origins, cutlass.Int32(batch), origin_specs).launch(
        grid=(q_grid, batch, q.shape[2]),
        block=(THREADS, 1, 1),
        stream=stream,
    )
    accumulation_host(
        maps,
        lse,
        delta,
        meta,
        origins,
        cutlass.Int32(batch),
        q.shape[2],
        max_skv,
        _origin_row(origin_specs, 8),
        scale,
        causal,
        q.element_type,
        stream,
    )
    cast_dq_thd_d192_kernel(dq_accum, dq, meta, origins, cutlass.Int32(batch), _origin_row(origin_specs, 5), q.element_type).launch(
        grid=(q_grid, batch, q.shape[2]),
        block=(THREADS, 1, 1),
        stream=stream,
    )
