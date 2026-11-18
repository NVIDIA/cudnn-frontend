# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct causal convolution schedules, with no FFT or tensor-core dependency."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.utils import SmemAllocator


@cute.jit
def _activate(value, silu: cutlass.Constexpr, low_precision: cutlass.Constexpr = False):
    """Apply identity or SiLU with the arithmetic selected for the backend path."""
    if cutlass.const_expr(silu):
        if cutlass.const_expr(low_precision):
            value = value * (0.5 + 0.5 * cute.tanh(0.5 * value, fastmath=True))
        else:
            if cutlass.const_expr(value.dtype == cutlass.Float64):
                value = value / (1.0 + cute.exp(-value, fastmath=True))
            else:
                # SE compiles __expf and division with --use_fast_math.
                value = cute.math.div(value, 1.0 + cute.exp(-value, fastmath=True), approx=True, ftz=True)
    return value


def _nwh_fwd_seqpar(dtype, width, shape):
    """Match the pinned backend's forward arithmetic route, including fallback."""
    from cutlass.cutlass_dsl import CuTeDSL
    from cutlass.utils import get_smem_capacity_in_bytes

    if dtype == cutlass.Float64 or int(CuTeDSL._get_dsl().get_arch_enum().major) < 8:
        return False
    if width > (128 if dtype == cutlass.Float32 else 32):
        return False
    tile_s = 128 if width > 65 else 64
    if shape[0] > 65535 or (shape[1] + tile_s - 1) // tile_s > 65535:
        return False
    if dtype == cutlass.Float32 and width > 8 and shape[2] % 4 != 0:
        return False
    smem = 3 * (tile_s + width - 1) * 64 * (dtype.width // 8) + (width + 1) * 64 * 4
    return smem <= get_smem_capacity_in_bytes()


@cute.kernel
def causal_conv1d_fwd_ncw(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    elements: cutlass.Constexpr,
    vector: cutlass.Constexpr,
):
    """One warp per channel and sequence tile; shift the causal window in registers."""
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    blocks_s = cute.ceil_div(x.shape[2], 32 * elements)
    blocks_h = cute.ceil_div(x.shape[1], 4)
    block = cute.arch.block_idx()[0]
    if cutlass.const_expr(blocks_s == 1):
        batch = block % x.shape[0]
        channel = (block // x.shape[0]) * 4 + tid // 32
        tile = 0
    else:
        tile = block % blocks_s
        channel = (block // blocks_s % blocks_h) * 4 + tid // 32
        batch = block // (blocks_s * blocks_h)
    start = tile * (32 * elements) + lane * elements
    compute = cutlass.Float32
    if cutlass.const_expr(x.element_type == cutlass.Float64):
        compute = cutlass.Float64
    if cutlass.const_expr(vector == 1):
        smem = SmemAllocator().allocate_tensor(x.element_type, cute.make_layout((4, 32 * elements), stride=(32 * elements, 1)), byte_alignment=16)
    current = cute.make_rmem_tensor((elements,), compute)
    previous = cute.make_rmem_tensor((elements,), compute)
    result = cute.make_rmem_tensor((elements,), compute)
    current.fill(0)
    previous.fill(0)
    result.fill(0)
    if channel < x.shape[1]:
        # Coalesce wide-filter loads across the warp, then broadcast each tap.
        if cutlass.const_expr(width >= 64):
            weight_regs = cute.make_rmem_tensor((cute.ceil_div(width, 32),), compute)
            weight_regs.fill(0)
            for group in cutlass.range_constexpr(cute.ceil_div(width, 32)):
                k = group * 32 + lane
                if k < width:
                    weight_regs[group] = w[channel, k].to(compute)
        if cutlass.const_expr(vector > 1):
            values = cute.make_rmem_tensor((elements,), x.element_type)
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=128)
            base = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(channel) * x.stride[1] + start
            if start < x.shape[2]:
                view = cute.make_tensor((x.iterator + base).align(16), cute.make_layout((elements,)))
                cute.copy(atom, view, values)
                current.store(values.load().to(compute))
            if tile > 0 and lane >= 32 - cute.ceil_div(width - 1, elements):
                view = cute.make_tensor((x.iterator + base - 32 * elements).align(16), cute.make_layout((elements,)))
                cute.copy(atom, view, values)
                previous.store(values.load().to(compute))
        if cutlass.const_expr(vector == 1):
            for i in cutlass.range_constexpr(elements):
                pos = tile * 32 * elements + lane + i * 32
                value = x.element_type(0)
                if pos < x.shape[2]:
                    value = x[batch, channel, pos]
                smem[tid // 32, lane + i * 32] = value
            cute.arch.sync_warp()
            for i in cutlass.range_constexpr(elements):
                current[i] = smem[tid // 32, lane * elements + i].to(compute)
            if cutlass.const_expr(width <= elements):
                for i in cutlass.range_constexpr(elements):
                    if tile > 0 and lane == 31:
                        previous[i] = x[batch, channel, start + i - 32 * elements].to(compute)
            else:
                cute.arch.sync_warp()
                for i in cutlass.range_constexpr(elements):
                    pos = (tile - 1) * 32 * elements + lane + i * 32
                    value = x.element_type(0)
                    if pos >= 0 and pos < x.shape[2]:
                        value = x[batch, channel, pos]
                    smem[tid // 32, lane + i * 32] = value
                cute.arch.sync_warp()
                for i in cutlass.range_constexpr(elements):
                    previous[i] = smem[tid // 32, lane * elements + i].to(compute)
        for i in cutlass.range_constexpr(elements):
            result[i] = bias[channel].to(compute)
        for tap in cutlass.range_constexpr(width):
            if cutlass.const_expr(tap > 0):
                shifted = cute.arch.shuffle_sync(current[elements - 1], (lane + 31) % 32)
                halo = cute.arch.shuffle_sync(previous[elements - 1 - (tap - 1) % elements], 31 - (tap - 1) // elements)
                for i in cutlass.range_constexpr(elements - 1, 0, -1):
                    current[i] = current[i - 1]
                current[0] = shifted
                if lane == 0:
                    current[0] = halo
            if cutlass.const_expr(width >= 64):
                weight = cute.arch.shuffle_sync(weight_regs[(width - 1 - tap) // 32], (width - 1 - tap) % 32)
            else:
                weight = w[channel, width - 1 - tap].to(compute)
            for i in cutlass.range_constexpr(elements):
                result[i] = result[i] + current[i] * weight
        for i in cutlass.range_constexpr(elements):
            result[i] = _activate(result[i], silu)
        if cutlass.const_expr(vector == 1):
            cute.arch.sync_warp()
            for i in cutlass.range_constexpr(elements):
                smem[tid // 32, lane * elements + i] = result[i].to(y.element_type)
            cute.arch.sync_warp()
            for i in cutlass.range_constexpr(elements):
                pos = tile * 32 * elements + lane + i * 32
                if pos < x.shape[2]:
                    y[batch, channel, pos] = smem[tid // 32, lane + i * 32]
        if cutlass.const_expr(vector > 1):
            if start < x.shape[2]:
                values.store(result.load().to(y.element_type))
                view = cute.make_tensor((y.iterator + base).align(16), cute.make_layout((elements,)))
                cute.copy(atom, values, view)


causal_conv1d_fwd_ncw.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def causal_conv1d_fwd_nwc(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """Channel-coalesced sequence tiles with a sliding register window."""
    tid = cute.arch.thread_idx()[0]
    channel = cute.arch.block_idx()[0] * 128 + tid
    start = cute.arch.block_idx()[1] * rows
    batch = cute.arch.block_idx()[2]
    compute = cutlass.Float32
    if cutlass.const_expr(x.element_type == cutlass.Float64):
        compute = cutlass.Float64
    window = cute.make_rmem_tensor((width + rows - 1,), compute)
    weights = cute.make_rmem_tensor((width,), compute)
    window.fill(0)
    if channel < x.shape[2]:
        for i in cutlass.range_constexpr(width + rows - 1):
            t = start + i - width + 1
            if t >= 0 and t < x.shape[1]:
                window[i] = x[batch, t, channel].to(compute)
        for tap in cutlass.range_constexpr(width):
            weights[tap] = w[tap, channel].to(compute)
        for i in cutlass.range_constexpr(rows):
            value = bias[channel].to(compute)
            for tap in cutlass.range_constexpr(width):
                if cutlass.const_expr(not _nwh_fwd_seqpar(x.element_type, width, x.shape)):
                    value = value + window[i + width - 1 - tap] * weights[width - 1 - tap]
                else:
                    # The SE sequence-parallel path accumulates ascending weight taps.
                    value = value + window[i + tap] * weights[tap]
            if start + i < x.shape[1]:
                y[batch, start + i, channel] = _activate(value, silu, x.element_type.width == 16 and _nwh_fwd_seqpar(x.element_type, width, x.shape)).to(
                    y.element_type
                )


causal_conv1d_fwd_nwc.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def causal_conv1d_fwd_nwc_vec(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    vector: cutlass.Constexpr,
    packed: cutlass.Constexpr,
    threads: cutlass.Constexpr = 128,
):
    """Convolve channel vectors with a register window and optional paired FP32 FMA."""
    tid = cute.arch.thread_idx()[0]
    channel = (cute.arch.block_idx()[0] * threads + tid) * vector
    start = cute.arch.block_idx()[1] * rows
    batch = cute.arch.block_idx()[2]
    window = cute.make_rmem_tensor((vector, width + rows - 1), cutlass.Float32)
    weights = cute.make_rmem_tensor((vector, width), cutlass.Float32)
    values = cute.make_rmem_tensor((vector,), x.element_type)
    result = cute.make_rmem_tensor((vector,), cutlass.Float32)
    bias_reg = cute.make_rmem_tensor((vector,), cutlass.Float32)
    window.fill(0)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=vector * x.element_type.width)
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), x.element_type, num_bits_per_copy=vector * x.element_type.width, store_cache_mode=cute.nvgpu.StoreCacheMode.STREAMING
    )
    if channel < x.shape[2]:
        for i in cutlass.range_constexpr(width + rows - 1):
            t = start + i - width + 1
            if t >= 0 and t < x.shape[1]:
                offset = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(t) * x.stride[1] + channel
                view = cute.make_tensor((x.iterator + offset).align(vector * x.element_type.width // 8), cute.make_layout((vector,)))
                cute.copy(atom, view, values)
                window[None, i].store(values.load().to(cutlass.Float32))
        for tap in cutlass.range_constexpr(width):
            view = cute.make_tensor(
                (w.iterator + cutlass.Int64(tap) * w.stride[0] + channel).align(vector * x.element_type.width // 8), cute.make_layout((vector,))
            )
            cute.copy(atom, view, values)
            weights[None, tap].store(values.load().to(cutlass.Float32))
        view = cute.make_tensor((bias.iterator + channel).align(vector * x.element_type.width // 8), cute.make_layout((vector,)))
        cute.copy(atom, view, values)
        bias_reg.store(values.load().to(cutlass.Float32))
        for i in cutlass.range_constexpr(rows):
            result.store(bias_reg.load())
            for step in cutlass.range_constexpr(width):
                tap = step if cutlass.const_expr(_nwh_fwd_seqpar(x.element_type, width, x.shape)) else width - 1 - step
                if cutlass.const_expr(packed):
                    for v in cutlass.range_constexpr(vector // 2):
                        lo, hi = cute.arch.fma_packed_f32x2(
                            (window[2 * v, i + tap], window[2 * v + 1, i + tap]),
                            (weights[2 * v, tap], weights[2 * v + 1, tap]),
                            (result[2 * v], result[2 * v + 1]),
                        )
                        result[2 * v], result[2 * v + 1] = lo, hi
                else:
                    for v in cutlass.range_constexpr(vector):
                        result[v] = result[v] + window[v, i + tap] * weights[v, tap]
            for v in cutlass.range_constexpr(vector):
                values[v] = _activate(result[v], silu, x.element_type.width == 16 and _nwh_fwd_seqpar(x.element_type, width, x.shape)).to(x.element_type)
            if start + i < x.shape[1]:
                offset = cutlass.Int64(batch) * y.stride[0] + cutlass.Int64(start + i) * y.stride[1] + channel
                view = cute.make_tensor((y.iterator + offset).align(vector * x.element_type.width // 8), cute.make_layout((vector,)))
                cute.copy(store_atom, values, view)


causal_conv1d_fwd_nwc_vec.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def causal_conv1d_fwd_nwc_grouped(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    vector: cutlass.Constexpr,
    packed: cutlass.Constexpr,
):
    """Bound register storage independently of filter width by walking eight taps."""
    tid = cute.arch.thread_idx()[0]
    channel = (cute.arch.block_idx()[0] * 128 + tid) * vector
    start = cute.arch.block_idx()[1] * rows
    batch = cute.arch.block_idx()[2]
    compute = cutlass.Float32
    if cutlass.const_expr(x.element_type == cutlass.Float64):
        compute = cutlass.Float64
    group_size = 32 if vector == 1 else 16
    window = cute.make_rmem_tensor((vector, group_size + rows - 1), compute)
    weights = cute.make_rmem_tensor((vector, group_size), compute)
    result = cute.make_rmem_tensor((vector, rows), compute)
    values = cute.make_rmem_tensor((vector,), x.element_type)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=vector * x.element_type.width)
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyR2GOp(), x.element_type, num_bits_per_copy=vector * x.element_type.width, store_cache_mode=cute.nvgpu.StoreCacheMode.STREAMING
    )
    if channel < x.shape[2]:
        view = cute.make_tensor((bias.iterator + channel).align(vector * x.element_type.width // 8), cute.make_layout((vector,)))
        cute.copy(atom, view, values)
        for row in cutlass.range_constexpr(rows):
            result[None, row].store(values.load().to(compute))
        for group in cutlass.range(0, cute.ceil_div(width, group_size), unroll=1):
            group_start = group * group_size
            if cutlass.const_expr(not _nwh_fwd_seqpar(x.element_type, width, x.shape)):
                group_start = width - (group + 1) * group_size
            window.fill(0)
            weights.fill(0)
            for i in cutlass.range_constexpr(group_size + rows - 1):
                t = start - width + 1 + group_start + i
                if t >= 0 and t < x.shape[1]:
                    offset = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(t) * x.stride[1] + channel
                    view = cute.make_tensor((x.iterator + offset).align(vector * x.element_type.width // 8), cute.make_layout((vector,)))
                    cute.copy(atom, view, values)
                    window[None, i].store(values.load().to(compute))
            for k in cutlass.range_constexpr(group_size):
                if group_start + k >= 0 and group_start + k < width:
                    offset = cutlass.Int64(group_start + k) * w.stride[0] + channel
                    view = cute.make_tensor((w.iterator + offset).align(vector * x.element_type.width // 8), cute.make_layout((vector,)))
                    cute.copy(atom, view, values)
                    weights[None, k].store(values.load().to(compute))
            for row in cutlass.range_constexpr(rows):
                for step in cutlass.range_constexpr(group_size):
                    k = step if cutlass.const_expr(_nwh_fwd_seqpar(x.element_type, width, x.shape)) else group_size - 1 - step
                    if cutlass.const_expr(width % group_size == 0):
                        if cutlass.const_expr(packed):
                            for v in cutlass.range_constexpr(vector // 2):
                                lo, hi = cute.arch.fma_packed_f32x2(
                                    (window[2 * v, row + k], window[2 * v + 1, row + k]),
                                    (weights[2 * v, k], weights[2 * v + 1, k]),
                                    (result[2 * v, row], result[2 * v + 1, row]),
                                )
                                result[2 * v, row], result[2 * v + 1, row] = lo, hi
                        else:
                            for v in cutlass.range_constexpr(vector):
                                result[v, row] += window[v, row + k] * weights[v, k]
                    else:
                        if group_start + k >= 0 and group_start + k < width:
                            if cutlass.const_expr(packed):
                                for v in cutlass.range_constexpr(vector // 2):
                                    lo, hi = cute.arch.fma_packed_f32x2(
                                        (window[2 * v, row + k], window[2 * v + 1, row + k]),
                                        (weights[2 * v, k], weights[2 * v + 1, k]),
                                        (result[2 * v, row], result[2 * v + 1, row]),
                                    )
                                    result[2 * v, row], result[2 * v + 1, row] = lo, hi
                            else:
                                for v in cutlass.range_constexpr(vector):
                                    result[v, row] += window[v, row + k] * weights[v, k]
        for row in cutlass.range_constexpr(rows):
            for v in cutlass.range_constexpr(vector):
                values[v] = _activate(result[v, row], silu, x.element_type.width == 16 and _nwh_fwd_seqpar(x.element_type, width, x.shape)).to(x.element_type)
            if start + row < x.shape[1]:
                offset = cutlass.Int64(batch) * y.stride[0] + cutlass.Int64(start + row) * y.stride[1] + channel
                view = cute.make_tensor((y.iterator + offset).align(vector * x.element_type.width // 8), cute.make_layout((vector,)))
                cute.copy(store_atom, values, view)


causal_conv1d_fwd_nwc_grouped.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _nwh_async_vector(src, dst, offset, target, valid, vector: cutlass.Constexpr):
    """Issue one predicated 128-bit global-to-shared copy for an NWH tile."""
    safe = cutlass.Int64(0)
    if valid:
        safe = offset
    layout = cute.make_layout((vector, 1), stride=(1, vector))
    g = cute.make_tensor((src.iterator + safe).align(16), layout)
    s = cute.make_tensor((dst.iterator + target).align(16), layout)
    pred = cute.make_rmem_tensor((vector, 1), cutlass.Boolean)
    pred.fill(valid)
    atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.common.LoadCacheMode.GLOBAL), src.element_type, num_bits_per_copy=128)
    cute.copy(atom, g, s, pred=pred)


@cute.jit
def _nwh_load_vector(src, offset, vector: cutlass.Constexpr):
    """Load one aligned 128-bit vector into registers."""
    raw = cute.make_rmem_tensor((vector,), src.element_type)
    view = cute.make_tensor((src.iterator + offset).align(16), cute.make_layout((vector,)))
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=128)
    cute.copy(atom, view, raw)
    return raw


def _uses_packed_f32():
    """Return whether the compile target supports the selected paired FP32 path."""
    from cutlass.cutlass_dsl import CuTeDSL

    return CuTeDSL._get_dsl().get_arch_enum().major >= 10


@cute.jit
def _nwh_fwd_stage(x, sx, batch, h0, tile, stage: cutlass.Constexpr, width: cutlass.Constexpr, ts: cutlass.Constexpr, th: cutlass.Constexpr):
    """Stage an input tile and its causal halo, then commit its asynchronous copies."""
    vector = 128 // x.element_type.width
    tpr = th // vector
    xr = ts + width - 1
    tid = cute.arch.thread_idx()[0]
    for index in cutlass.range(tid, xr * tpr, 128, unroll=1):
        row = index // tpr
        col = index % tpr * vector
        pos = tile * ts - width + 1 + row
        valid = pos >= 0 and pos < x.shape[1] and h0 + col < x.shape[2]
        offset = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(pos) * x.stride[1] + h0 + col
        _nwh_async_vector(x, sx, offset, (stage * xr + row) * th + col, valid, vector)
    cute.arch.cp_async_commit_group()


@cute.jit
def _nwh_fwd_tile(
    sx,
    y,
    weights,
    bias,
    batch,
    h0,
    tile,
    stage: cutlass.Constexpr,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
):
    """Compute one staged NWH tile with cached weights and store valid outputs."""
    vector = 128 // y.element_type.width
    tpr = th // vector
    rows = ts // (128 // tpr)
    xr = ts + width - 1
    tid = cute.arch.thread_idx()[0]
    col = tid % tpr * vector
    rbase = tid // tpr * rows
    window = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    for j in cutlass.range_constexpr(width - 1):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j) * th + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[j, v] = raw[v].to(cutlass.Float32)
    for j in cutlass.range_constexpr(rows):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j + width - 1) * th + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[(j + width - 1) % width, v] = raw[v].to(cutlass.Float32)
        result = cute.make_rmem_tensor((vector,), cutlass.Float32)
        result.store(bias.load())
        for k in cutlass.range_constexpr(width):
            if cutlass.const_expr(_uses_packed_f32()):
                for v in cutlass.range_constexpr(0, vector, 2):
                    lo, hi = cute.arch.fma_packed_f32x2(
                        (window[(j + k) % width, v], window[(j + k) % width, v + 1]), (weights[k, v], weights[k, v + 1]), (result[v], result[v + 1])
                    )
                    result[v], result[v + 1] = lo, hi
            else:
                for v in cutlass.range_constexpr(vector):
                    result[v] = cute.math.fma(window[(j + k) % width, v], weights[k, v], result[v])
        output = cute.make_rmem_tensor((vector,), y.element_type)
        for v in cutlass.range_constexpr(vector):
            output[v] = _activate(result[v], silu, y.element_type.width == 16).to(y.element_type)
        if cutlass.const_expr(y.shape[1] % ts == 0 and y.shape[2] % th == 0) or (tile * ts + rbase + j < y.shape[1] and h0 + col < y.shape[2]):
            offset = cutlass.Int64(batch) * y.stride[0] + cutlass.Int64(tile * ts + rbase + j) * y.stride[1] + h0 + col
            view = cute.make_tensor((y.iterator + offset).align(16), cute.make_layout((vector,)))
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(), y.element_type, num_bits_per_copy=128, store_cache_mode=cute.nvgpu.common.StoreCacheMode.STREAMING
            )
            cute.copy(atom, output, view)


@cute.jit
def _nwh_fwd_payload(x, sx, batch, h0, tile, stage: cutlass.Constexpr, width: cutlass.Constexpr, ts: cutlass.Constexpr, th: cutlass.Constexpr):
    """Stage only new sequence rows; the pipeline carries the causal halo separately."""
    vector = 128 // x.element_type.width
    tpr = th // vector
    rows = ts // (128 // tpr)
    xr = ts + width - 1
    tid = cute.arch.thread_idx()[0]
    col = tid % tpr * vector
    rbase = tid // tpr * rows
    for j in cutlass.range_constexpr(rows):
        pos = tile * ts + rbase + j
        valid = True
        if cutlass.const_expr(x.shape[1] % ts != 0 or x.shape[2] % th != 0):
            valid = pos < x.shape[1] and h0 + col < x.shape[2]
        offset = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(pos) * x.stride[1] + h0 + col
        _nwh_async_vector(x, sx, offset, (stage * xr + width - 1 + rbase + j) * th + col, valid, vector)
    cute.arch.cp_async_commit_group()


@cute.jit
def _nwh_fwd_carry(sx, source: cutlass.Constexpr, target: cutlass.Constexpr, width: cutlass.Constexpr, ts: cutlass.Constexpr, th: cutlass.Constexpr):
    """Copy the previous tile tail into the next shared-memory stage halo."""
    vector = 128 // sx.element_type.width
    tpr = th // vector
    xr = ts + width - 1
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sx.element_type, num_bits_per_copy=128)
    raw = cute.make_rmem_tensor((vector,), sx.element_type)
    layout = cute.make_layout((vector,))
    for index in cutlass.range(cute.arch.thread_idx()[0], (width - 1) * tpr, 128, unroll=1):
        row, col = index // tpr, index % tpr * vector
        src = cute.make_tensor((sx.iterator + (source * xr + ts + row) * th + col).align(16), layout)
        dst = cute.make_tensor((sx.iterator + (target * xr + row) * th + col).align(16), layout)
        cute.copy(atom, src, raw)
        cute.copy(atom, raw, dst)


@cute.kernel
def nwh_fwd_pipeline3(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    nt: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
):
    """Pipeline NWH tiles through three shared-memory stages while reusing halo rows."""
    vector = 128 // x.element_type.width
    tpr = th // vector
    tid = cute.arch.thread_idx()[0]
    first = cute.arch.block_idx()[1] * nt
    batch = cute.arch.block_idx()[2]
    h0 = cute.arch.block_idx()[0] * th
    col = tid % tpr * vector
    count = min(nt, cute.ceil_div(x.shape[1], ts) - first)
    sx = SmemAllocator().allocate_tensor(x.element_type, cute.make_layout((3 * (ts + width - 1) * th,)), byte_alignment=16)
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    bias = cute.make_rmem_tensor((vector,), cutlass.Float32)
    weights.fill(0)
    bias.fill(0)
    if h0 + col < x.shape[2]:
        for k in cutlass.range_constexpr(width):
            raw = _nwh_load_vector(w, cutlass.Int64(k) * x.shape[2] + h0 + col, vector)
            for v in cutlass.range_constexpr(vector):
                weights[k, v] = raw[v].to(cutlass.Float32)
        raw = _nwh_load_vector(b, h0 + col, vector)
        for v in cutlass.range_constexpr(vector):
            bias[v] = raw[v].to(cutlass.Float32)
    _nwh_fwd_stage(x, sx, batch, h0, first, 0, width, ts, th)
    if count > 1:
        _nwh_fwd_payload(x, sx, batch, h0, first + 1, 1, width, ts, th)
    for index in cutlass.range_constexpr(nt):
        if index < count:
            if index + 1 < count:
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if cutlass.const_expr(index > 0):
                _nwh_fwd_carry(sx, (index + 2) % 3, index % 3, width, ts, th)
            cute.arch.barrier()
            if index + 2 < count:
                _nwh_fwd_payload(x, sx, batch, h0, first + index + 2, (index + 2) % 3, width, ts, th)
            _nwh_fwd_tile(sx, y, weights, bias, batch, h0, first + index, index % 3, width, silu, ts, th)
    cute.arch.cp_async_wait_group(0)


nwh_fwd_pipeline3.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def nwh_fwd_pipeline(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    nt: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
):
    """Alternate two shared-memory stages while loading each tile with its full halo."""
    vector = 128 // x.element_type.width
    tpr = th // vector
    tid = cute.arch.thread_idx()[0]
    first = cute.arch.block_idx()[1] * nt
    batch = cute.arch.block_idx()[2]
    h0 = cute.arch.block_idx()[0] * th
    col = tid % tpr * vector
    count = min(nt, cute.ceil_div(x.shape[1], ts) - first)
    sx = SmemAllocator().allocate_tensor(x.element_type, cute.make_layout((2 * (ts + width - 1) * th,)), byte_alignment=16)
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    bias = cute.make_rmem_tensor((vector,), cutlass.Float32)
    weights.fill(0)
    bias.fill(0)
    if h0 + col < x.shape[2]:
        for k in cutlass.range_constexpr(width):
            raw = _nwh_load_vector(w, cutlass.Int64(k) * x.shape[2] + h0 + col, vector)
            for v in cutlass.range_constexpr(vector):
                weights[k, v] = raw[v].to(cutlass.Float32)
        raw = _nwh_load_vector(b, h0 + col, vector)
        for v in cutlass.range_constexpr(vector):
            bias[v] = raw[v].to(cutlass.Float32)
    _nwh_fwd_stage(x, sx, batch, h0, first, 0, width, ts, th)
    for pair in cutlass.range(0, cute.ceil_div(count, 2), unroll=1):
        ia = pair * 2
        if ia + 1 < count:
            _nwh_fwd_stage(x, sx, batch, h0, first + ia + 1, 1, width, ts, th)
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()
        _nwh_fwd_tile(sx, y, weights, bias, batch, h0, first + ia, 0, width, silu, ts, th)
        cute.arch.barrier()
        if ia + 2 < count:
            _nwh_fwd_stage(x, sx, batch, h0, first + ia + 2, 0, width, ts, th)
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()
        if ia + 1 < count:
            _nwh_fwd_tile(sx, y, weights, bias, batch, h0, first + ia + 1, 1, width, silu, ts, th)
        cute.arch.barrier()
    cute.arch.cp_async_wait_group(0)


nwh_fwd_pipeline.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def nwh_wide_shared(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    # Store wide filters once per CTA; registers retain the input window and 16 outputs.
    """Reuse wide filters in shared memory and compute paired output rows in registers."""
    tid = cute.arch.thread_idx()[0]
    channel = cute.arch.block_idx()[0] * threads + tid
    start = cute.arch.block_idx()[1] * rows
    batch = cute.arch.block_idx()[2]
    weights = SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout((width, threads), stride=(threads, 1)), byte_alignment=16)
    for k in cutlass.range_constexpr(width):
        value = cutlass.Float32(0)
        if channel < x.shape[2]:
            value = w[k, channel].to(cutlass.Float32)
        weights[k, tid] = value
    cute.arch.barrier()
    if channel < x.shape[2]:
        window = cute.make_rmem_tensor((width + rows - 1,), cutlass.Float32)
        result = cute.make_rmem_tensor((rows,), cutlass.Float32)
        window.fill(0)
        result.fill(bias[channel].to(cutlass.Float32))
        for i in cutlass.range_constexpr(width + rows - 1):
            t = start + i - width + 1
            if t >= 0 and t < x.shape[1]:
                window[i] = x[batch, t, channel].to(cutlass.Float32)
        for step in cutlass.range_constexpr(width):
            k = width - 1 - step
            weight = weights[k, tid]
            for i in cutlass.range_constexpr(0, rows, 2):
                lo, hi = cute.arch.fma_packed_f32x2((window[i + k], window[i + k + 1]), (weight, weight), (result[i], result[i + 1]))
                result[i], result[i + 1] = lo, hi
        for i in cutlass.range_constexpr(rows):
            if start + i < x.shape[1]:
                y[batch, start + i, channel] = _activate(result[i], silu, False).to(y.element_type)


nwh_wide_shared.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch_fwd(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    y: cute.Tensor,
    stream: cuda.CUstream,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    channel_last: cutlass.Constexpr,
    tile: cutlass.Constexpr,
    vector: cutlass.Constexpr,
    packed: cutlass.Constexpr,
):
    """Select and launch the direct-convolution schedule for the layout, shape, and GPU."""
    from cutlass.cutlass_dsl import CuTeDSL

    if cutlass.const_expr(
        not channel_last and x.element_type.width == 16 and width <= 8 and x.shape[2] % 512 == 0 and CuTeDSL._get_dsl().get_arch_enum().major in (9, 10)
    ):
        tile = 16
    if cutlass.const_expr(
        not channel_last
        and x.element_type == cutlass.Float32
        and silu
        and width <= 8
        and x.shape[2] <= 256
        and (
            CuTeDSL._get_dsl().get_arch_enum().major == 12 or (CuTeDSL._get_dsl().get_arch_enum().major == 8 and CuTeDSL._get_dsl().get_arch_enum().minor == 6)
        )
    ):
        tile = 4
    if cutlass.const_expr(not channel_last and x.element_type == cutlass.Float64 and x.shape[2] <= 32):
        tile = 4
    arch = CuTeDSL._get_dsl().get_arch_enum()
    if cutlass.const_expr(
        channel_last
        and x.element_type.width == 16
        and width == 4
        and x.shape[2] % 8 == 0
        and not silu
        and cute.size(x) >= 100_000_000
        and arch.major == 10
        and arch.minor == 0
    ):
        nwh_fwd_pipeline3(x, w, bias, y, width, silu, 8, 32, 128).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], 256), x.shape[0]),
            block=(128, 1, 1),
            stream=stream,
        )
    elif cutlass.const_expr(
        channel_last and x.element_type.width == 16 and width == 3 and x.shape[2] % 8 == 0 and cute.size(x) >= 6_000_000 and arch.major == 12
    ):
        nwh_fwd_pipeline3(x, w, bias, y, width, silu, 8, 64, 64).launch(
            grid=(cute.ceil_div(x.shape[2], 64), cute.ceil_div(x.shape[1], 512), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(
        channel_last
        and x.element_type.width == 16
        and width == 4
        and x.shape[2] % 8 == 0
        and 32_000_000 <= cute.size(x) < 48_000_000
        and arch.major == 10
        and arch.minor == 0
    ):
        nwh_fwd_pipeline3(x, w, bias, y, width, silu, 4, 64, 128).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], 256), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(channel_last and x.element_type.width == 16 and width == 128 and arch.major == 10 and arch.minor in (0, 3)):
        nwh_wide_shared(x, w, bias, y, width, silu, 16, 64).launch(
            grid=(cute.ceil_div(x.shape[2], 64), cute.ceil_div(x.shape[1], 16), x.shape[0]), block=(64, 1, 1), stream=stream
        )
    elif cutlass.const_expr(
        channel_last
        and x.element_type == cutlass.Float32
        and width == 7
        and x.shape[2] % 4 == 0
        and cute.size(x) >= 8_000_000
        and ((arch.major == 8 and arch.minor == 6) or (arch.major == 10 and arch.minor == 3) or arch.major == 12)
    ):
        nwh_fwd_pipeline3(x, w, bias, y, width, silu, min(4, cute.ceil_div(x.shape[1], 64)), 64, 64).launch(
            grid=(cute.ceil_div(x.shape[2], 64), cute.ceil_div(x.shape[1], 256), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(
        channel_last and x.element_type.width == 16 and width == 4 and x.shape[2] % 8 == 0 and 32_000_000 <= cute.size(x) < 100_000_000 and arch.major >= 10
    ):
        nwh_fwd_pipeline3(x, w, bias, y, width, silu, min(16, cute.ceil_div(x.shape[1], 32)), 32, 128).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], 512), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(
        channel_last and x.element_type.width == 16 and width == 4 and not silu and x.shape[2] % 8 == 0 and cute.size(x) >= 100_000_000 and arch.major == 9
    ):
        nwh_fwd_pipeline(x, w, bias, y, width, silu, 4, 64, 64).launch(
            grid=(cute.ceil_div(x.shape[2], 64), cute.ceil_div(x.shape[1], 256), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(
        channel_last
        and x.element_type.width == 16
        and width == 4
        and x.shape[2] % 8 == 0
        and cute.size(x) >= 100_000_000
        and CuTeDSL._get_dsl().get_arch_enum().major >= 10
    ):
        nt = 4
        nwh_fwd_pipeline(x, w, bias, y, width, silu, nt, 64, 128).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], 64 * nt), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(channel_last):
        if cutlass.const_expr(x.element_type.width == 16 and 32 < width <= 64):
            causal_conv1d_fwd_nwc(x, w, bias, y, width, silu, 16).launch(
                grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], 16), x.shape[0]), block=(128, 1, 1), stream=stream
            )
        elif cutlass.const_expr(width >= 32):
            causal_conv1d_fwd_nwc_grouped(x, w, bias, y, width, silu, tile, vector, packed).launch(
                grid=(cute.ceil_div(x.shape[2], 128 * vector), cute.ceil_div(x.shape[1], tile), x.shape[0]), block=(128, 1, 1), stream=stream
            )
        elif cutlass.const_expr(vector > 1):
            threads = 64 if x.element_type.width == 16 and width == 3 and cute.size(x) >= 6_000_000 and CuTeDSL._get_dsl().get_arch_enum().major == 12 else 128
            causal_conv1d_fwd_nwc_vec(x, w, bias, y, width, silu, tile, vector, packed, threads).launch(
                grid=(cute.ceil_div(x.shape[2], threads * vector), cute.ceil_div(x.shape[1], tile), x.shape[0]), block=(threads, 1, 1), stream=stream
            )
        else:
            causal_conv1d_fwd_nwc(x, w, bias, y, width, silu, tile).launch(
                grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], tile), x.shape[0]),
                block=(128, 1, 1),
                stream=stream,
            )
    else:
        causal_conv1d_fwd_ncw(x, w, bias, y, width, silu, tile, vector).launch(
            grid=(cute.ceil_div(x.shape[2], 32 * tile) * cute.ceil_div(x.shape[1], 4) * x.shape[0], 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )
