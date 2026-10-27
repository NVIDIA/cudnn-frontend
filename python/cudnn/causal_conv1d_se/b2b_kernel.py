# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused direct back-to-back convolution with register-resident intermediates."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.utils import SmemAllocator


@cute.jit
def _shift_left(current, previous, tap: cutlass.Constexpr, lane):
    elements = cute.size(current)
    shifted = cute.arch.shuffle_sync(current[(elements - 1)], (lane + 31) % 32)
    halo = cute.arch.shuffle_sync(previous[(elements - 1) - (tap - 1) % elements], 31 - (tap - 1) // elements)
    for i in cutlass.range_constexpr((elements - 1), 0, -1):
        current[i] = current[i - 1]
    current[0] = shifted
    if lane == 0:
        current[0] = halo


@cute.jit
def _shift_right(current, following, tap: cutlass.Constexpr, lane):
    elements = cute.size(current)
    shifted = cute.arch.shuffle_sync(current[0], (lane + 1) % 32)
    halo = cute.arch.shuffle_sync(following[(tap - 1) % elements], (tap - 1) // elements)
    for i in cutlass.range_constexpr((elements - 1)):
        current[i] = current[i + 1]
    current[(elements - 1)] = shifted
    if lane == 31:
        current[(elements - 1)] = halo


@cute.jit
def _warp_sum(value):
    for i in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, 1 << i)
    return value


@cute.kernel
def b2b_zero_grad(dwp: cute.Tensor, dwm: cute.Tensor, ds: cute.Tensor):
    i = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
    fp = cute.make_tensor(dwp.iterator, cute.make_layout((cute.size(dwp),)))
    fm = cute.make_tensor(dwm.iterator, cute.make_layout((cute.size(dwm),)))
    if i < cute.size(dwp):
        fp[i] = dwp.element_type(0)
    if i < cute.size(dwm):
        fm[i] = dwm.element_type(0)
    if i < cute.size(ds):
        ds[i] = ds.element_type(0)


b2b_zero_grad.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _load_tile(x, batch, channel, tile, lane, compute: cutlass.Constexpr, halo: cutlass.Constexpr = 0, elements: cutlass.Constexpr = 8):
    halo = 32 * elements if cutlass.const_expr(halo == 0) else halo
    values = cute.make_rmem_tensor((elements,), compute)
    values.fill(0)
    if tile >= 0 and lane >= 32 - cute.ceil_div(halo, elements):
        start = tile * (32 * elements) + lane * elements
        if cutlass.const_expr(x.shape[2] % elements == 0):
            if start < x.shape[2]:
                atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=min(128, elements * x.element_type.width))
                raw = cute.make_rmem_tensor((elements,), x.element_type)
                offset = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(channel) * x.stride[1] + start
                view = cute.make_tensor((x.iterator + offset).align(min(16, elements * x.element_type.width // 8)), cute.make_layout((elements,)))
                cute.copy(atom, view, raw)
                from cutlass.cutlass_dsl import CuTeDSL

                if cutlass.const_expr(x.element_type == cutlass.BFloat16 and CuTeDSL._get_dsl().get_arch_enum().major == 12):
                    for i in cutlass.range_constexpr(elements):
                        values[i] = (raw[i].bitcast(cutlass.Uint16).to(cutlass.Uint32) << 16).bitcast(cutlass.Float32)
                else:
                    values.store(raw.load().to(compute))
        else:
            for i in cutlass.range_constexpr(elements):
                if start + i < x.shape[2]:
                    values[i] = x[batch, channel, start + i].to(compute)
    return values


@cute.jit
def _tail_load(x, batch, channel, tile, lane, compute: cutlass.Constexpr, scratch: cute.Tensor, halo: cutlass.Constexpr = 0, elements: cutlass.Constexpr = 8):
    halo = 32 * elements if cutlass.const_expr(halo == 0) else halo
    values = cute.make_rmem_tensor((elements,), compute)
    values.fill(0)
    warp = cute.arch.thread_idx()[0] // 32
    if tile >= 0:
        for i in cutlass.range_constexpr(elements):
            pos = lane + i * 32
            value = compute(0)
            if pos >= 32 * elements - cute.ceil_div(halo, elements) * elements and tile * (32 * elements) + pos < x.shape[2]:
                value = x[batch, channel, tile * (32 * elements) + pos].to(compute)
            scratch[warp, pos ^ (pos >> 5)] = value
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(elements):
            pos = lane * elements + i
            values[i] = scratch[warp, pos ^ (pos >> 5)]
        cute.arch.sync_warp()
    return values


@cute.jit
def _tail_store(y, batch, channel, start, values, scratch: cute.Tensor):
    elements = cute.size(values)
    warp = cute.arch.thread_idx()[0] // 32
    lane = cute.arch.thread_idx()[0] % 32
    tile_start = start - lane * elements
    for i in cutlass.range_constexpr(elements):
        pos = lane * elements + i
        scratch[warp, pos ^ (pos >> 5)] = values[i]
    cute.arch.sync_warp()
    for i in cutlass.range_constexpr(elements):
        pos = lane + i * 32
        if tile_start + pos < y.shape[2]:
            y[batch, channel, tile_start + pos] = scratch[warp, pos ^ (pos >> 5)].to(y.element_type)
    cute.arch.sync_warp()


@cute.jit
def _bwd_load(x, batch, channel, tile, lane, compute: cutlass.Constexpr, scratch, elements: cutlass.Constexpr):
    if cutlass.const_expr(x.shape[2] % elements != 0):
        return _tail_load(x, batch, channel, tile, lane, compute, scratch, elements=elements)
    else:
        return _load_tile(x, batch, channel, tile, lane, compute, elements=elements)


@cute.jit
def _bwd_store(y, batch, channel, start, values, scratch):
    elements = cute.size(values)
    if cutlass.const_expr(y.shape[2] % elements != 0):
        _tail_store(y, batch, channel, start, values, scratch)
    else:
        _store_tile(y, batch, channel, start, values)


@cute.jit
def _bwd_projection(
    x,
    w,
    batch,
    channel,
    tile,
    lane,
    width: cutlass.Constexpr,
    compute: cutlass.Constexpr,
    scratch,
    weight_channel,
    elements: cutlass.Constexpr,
    register_weights: cutlass.Constexpr,
):
    current = _bwd_load(x, batch, channel, tile, lane, compute, scratch, elements)
    previous = _bwd_load(x, batch, channel, tile - 1, lane, compute, scratch, elements)
    result = cute.make_rmem_tensor((elements,), compute)
    result.fill(0)
    for tap in cutlass.range_constexpr(width):
        if cutlass.const_expr(tap > 0):
            _shift_left(current, previous, tap, lane)
        if cutlass.const_expr(register_weights):
            value = cute.arch.shuffle_sync(w[(width - 1 - tap) // 32, weight_channel % 3], (width - 1 - tap) % 32)
        else:
            value = w[weight_channel, width - 1 - tap].to(compute)
        for i in cutlass.range_constexpr(elements):
            result[i] += current[i] * value
    return result


@cute.jit
def _store_tile(y, batch, channel, start, values):
    elements = cute.size(values)
    if cutlass.const_expr(y.shape[2] % elements == 0):
        if start < y.shape[2]:
            raw = cute.make_rmem_tensor((elements,), y.element_type)
            raw.store(values.load().to(y.element_type))
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), y.element_type, num_bits_per_copy=min(128, elements * y.element_type.width))
            offset = cutlass.Int64(batch) * y.stride[0] + cutlass.Int64(channel) * y.stride[1] + start
            view = cute.make_tensor((y.iterator + offset).align(min(16, elements * y.element_type.width // 8)), cute.make_layout((elements,)))
            cute.copy(atom, raw, view)
    else:
        for i in cutlass.range_constexpr(elements):
            if start + i < y.shape[2]:
                y[batch, channel, start + i] = values[i].to(y.element_type)


@cute.jit
def _persistent_projection(
    x,
    weights,
    previous,
    output,
    batch,
    channel,
    component,
    tile,
    lane,
    width: cutlass.Constexpr,
    compute: cutlass.Constexpr,
    elements: cutlass.Constexpr,
    scratch,
    distributed: cutlass.Constexpr,
    shared_weights: cutlass.Constexpr,
    packed: cutlass.Constexpr,
):
    if cutlass.const_expr(x.shape[2] % elements != 0):
        current = _tail_load(x, batch, 3 * channel + component, tile, lane, compute, scratch, elements=elements)
    else:
        current = _load_tile(x, batch, 3 * channel + component, tile, lane, compute, elements=elements)
    keep = cute.make_rmem_tensor((elements,), compute)
    keep.store(current.load())
    for tap in cutlass.range_constexpr(width):
        if cutlass.const_expr(tap > 0):
            _shift_left(current, previous, tap, lane)
        if cutlass.const_expr(shared_weights):
            weight = weights[cute.arch.thread_idx()[0] // 32, width - 1 - tap, component]
        elif cutlass.const_expr(distributed):
            weight = cute.arch.shuffle_sync(weights[(width - 1 - tap) // 32, component], (width - 1 - tap) % 32)
        else:
            weight = weights[width - 1 - tap, component]
        if cutlass.const_expr(packed):
            for i in cutlass.range_constexpr(0, elements, 2):
                lo, hi = cute.arch.fma_packed_f32x2((current[i], current[i + 1]), (weight, weight), (output[i], output[i + 1]))
                output[i], output[i + 1] = lo, hi
        else:
            for i in cutlass.range_constexpr(elements):
                output[i] += current[i] * weight
    previous.store(keep.load())


@cute.kernel
def b2b_fwd_persistent(
    x: cute.Tensor,
    wp: cute.Tensor,
    wm: cute.Tensor,
    skip: cute.Tensor,
    y: cute.Tensor,
    gated: cute.Tensor,
    proj_width: cutlass.Constexpr,
    mixer_width: cutlass.Constexpr,
    warps: cutlass.Constexpr = 4,
    elements: cutlass.Constexpr = 8,
):
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    low_parallel = x.element_type.width == 16 and mixer_width >= 64 and x.shape[2] >= 1024 and x.shape[0] * y.shape[1] <= 128
    shared_weights = low_parallel and (
        (arch.major == 8 and (arch.minor == 0 or (arch.minor in (6, 9) and x.element_type == cutlass.Float16))) or arch.major == 9
    )
    packed = low_parallel and arch.major == 10 and arch.minor in (0, 3)
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    channel_block = cute.arch.block_idx()[0] % cute.ceil_div(y.shape[1], warps) if cutlass.const_expr(packed) else cute.arch.block_idx()[0]
    channel = channel_block * warps + tid // 32
    batch = cute.arch.block_idx()[1]
    compute = cutlass.Float64 if x.element_type == cutlass.Float64 else cutlass.Float32
    smem = SmemAllocator()
    scratch = None
    if cutlass.const_expr(x.shape[2] % elements != 0):
        scratch = smem.allocate_tensor(compute, cute.make_layout((warps, 32 * elements), stride=(32 * elements, 1)), byte_alignment=16)
    previous = cute.make_rmem_tensor((elements, 3), compute)
    mix_previous = cute.make_rmem_tensor((elements,), compute)
    projection = cute.make_rmem_tensor((elements, 3), compute)
    distributed = proj_width > 8 or mixer_width > 16
    if cutlass.const_expr(shared_weights):
        weights = smem.allocate_tensor(compute, cute.make_layout((warps, proj_width, 3), stride=(3 * proj_width, 3, 1)), byte_alignment=16)
        mix_weights = smem.allocate_tensor(compute, cute.make_layout((warps, mixer_width), stride=(mixer_width, 1)), byte_alignment=16)
    else:
        weights = cute.make_rmem_tensor((cute.ceil_div(proj_width, 32) if distributed else proj_width, 3), compute)
        mix_weights = cute.make_rmem_tensor((cute.ceil_div(mixer_width, 32) if distributed else mixer_width,), compute)
    previous.fill(0)
    mix_previous.fill(0)
    if channel < y.shape[1]:
        if cutlass.const_expr(distributed):
            for c in cutlass.range_constexpr(3):
                for group in cutlass.range_constexpr(cute.ceil_div(proj_width, 32)):
                    k = group * 32 + lane
                    value = compute(0)
                    if k < proj_width:
                        value = wp[3 * channel + c, k].to(compute)
                    if cutlass.const_expr(shared_weights):
                        if k < proj_width:
                            weights[tid // 32, k, c] = value
                    else:
                        weights[group, c] = value
            for group in cutlass.range_constexpr(cute.ceil_div(mixer_width, 32)):
                k = group * 32 + lane
                value = compute(0)
                if k < mixer_width:
                    value = wm[channel, k].to(compute)
                if cutlass.const_expr(shared_weights):
                    if k < mixer_width:
                        mix_weights[tid // 32, k] = value
                else:
                    mix_weights[group] = value
            if cutlass.const_expr(shared_weights):
                cute.arch.sync_warp()
        else:
            for c in cutlass.range_constexpr(3):
                for k in cutlass.range_constexpr(proj_width):
                    weights[k, c] = wp[3 * channel + c, k].to(compute)
            for k in cutlass.range_constexpr(mixer_width):
                mix_weights[k] = wm[channel, k].to(compute)
        skip_value = skip[channel].to(compute)
        tile_begin = cute.arch.block_idx()[0] // cute.ceil_div(y.shape[1], warps) if cutlass.const_expr(packed) else 0
        tile_end = min(tile_begin + 1, cute.ceil_div(x.shape[2], 32 * elements)) if cutlass.const_expr(packed) else cute.ceil_div(x.shape[2], 32 * elements)
        # Split low-parallelism sequences across CTAs; rebuild only the preceding halo.
        if cutlass.const_expr(packed):
            if tile_begin > 0:
                projection.fill(0)
                for c in cutlass.range_constexpr(3):
                    if cutlass.const_expr(x.shape[2] % elements != 0):
                        raw = _tail_load(x, batch, 3 * channel + c, tile_begin - 2, lane, compute, scratch, elements=elements)
                    else:
                        raw = _load_tile(x, batch, 3 * channel + c, tile_begin - 2, lane, compute, elements=elements)
                    previous[None, c].store(raw.load())
                    _persistent_projection(
                        x,
                        weights,
                        previous[None, c],
                        projection[None, c],
                        batch,
                        channel,
                        c,
                        tile_begin - 1,
                        lane,
                        proj_width,
                        compute,
                        elements,
                        scratch,
                        distributed,
                        shared_weights,
                        packed,
                    )
                for i in cutlass.range_constexpr(elements):
                    mix_previous[i] = projection[i, 1] * projection[i, 2]
        for tile in cutlass.range(tile_begin, tile_end, unroll=1):
            start = tile * (32 * elements) + lane * elements
            projection.fill(0)
            for c in cutlass.range_constexpr(3):
                _persistent_projection(
                    x,
                    weights,
                    previous[None, c],
                    projection[None, c],
                    batch,
                    channel,
                    c,
                    tile,
                    lane,
                    proj_width,
                    compute,
                    elements,
                    scratch,
                    distributed,
                    shared_weights,
                    packed,
                )
            product = cute.make_rmem_tensor((elements,), compute)
            keep_product = cute.make_rmem_tensor((elements,), compute)
            result = cute.make_rmem_tensor((elements,), compute)
            for i in cutlass.range_constexpr(elements):
                product[i] = projection[i, 1] * projection[i, 2]
                result[i] = product[i] * skip_value
            keep_product.store(product.load())
            for tap in cutlass.range_constexpr(mixer_width):
                if cutlass.const_expr(tap > 0):
                    _shift_left(product, mix_previous, tap, lane)
                if cutlass.const_expr(shared_weights):
                    weight = mix_weights[tid // 32, mixer_width - 1 - tap]
                elif cutlass.const_expr(distributed):
                    weight = cute.arch.shuffle_sync(mix_weights[(mixer_width - 1 - tap) // 32], (mixer_width - 1 - tap) % 32)
                else:
                    weight = mix_weights[mixer_width - 1 - tap]
                if cutlass.const_expr(packed and tap > 0):
                    for i in cutlass.range_constexpr(0, elements, 2):
                        lo, hi = cute.arch.fma_packed_f32x2((product[i], product[i + 1]), (weight, weight), (result[i], result[i + 1]))
                        result[i], result[i + 1] = lo, hi
                else:
                    for i in cutlass.range_constexpr(elements):
                        result[i] += product[i] * weight
            mix_previous.store(keep_product.load())
            if cutlass.const_expr(x.shape[2] % elements != 0):
                _tail_store(y, batch, channel, start, result, scratch)
            else:
                _store_tile(y, batch, channel, start, result)
            for i in cutlass.range_constexpr(elements):
                result[i] *= projection[i, 0]
            if cutlass.const_expr(x.shape[2] % elements != 0):
                _tail_store(gated, batch, channel, start, result, scratch)
            else:
                _store_tile(gated, batch, channel, start, result)


b2b_fwd_persistent.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _projection_backward(
    grad,
    original,
    following,
    wp,
    proj_grad,
    dx,
    batch,
    channel,
    start,
    lane,
    proj_width: cutlass.Constexpr,
    distributed: cutlass.Constexpr,
    c: cutlass.Constexpr,
    compute: cutlass.Constexpr,
    scratch,
    weight_channel,
    register_weights: cutlass.Constexpr,
):
    elements = cute.size(grad)
    keep = cute.make_rmem_tensor((elements,), compute)
    keep.store(grad.load())
    result = cute.make_rmem_tensor((elements,), compute)
    result.fill(0)
    for tap in cutlass.range_constexpr(proj_width):
        if cutlass.const_expr(tap > 0):
            _shift_right(grad, following, tap, lane)
        if cutlass.const_expr(register_weights):
            weight = cute.arch.shuffle_sync(wp[(proj_width - 1 - tap) // 32, c], (proj_width - 1 - tap) % 32)
        else:
            weight = wp[3 * weight_channel + c, proj_width - 1 - tap].to(compute)
        local_grad = compute(0)
        for i in cutlass.range_constexpr(elements):
            result[i] += grad[i] * weight
            local_grad += original[i] * grad[i]
        if cutlass.const_expr(distributed):
            local_grad = _warp_sum(local_grad)
            if lane == (proj_width - 1 - tap) % 32:
                proj_grad[(proj_width - 1 - tap) // 32, c] += local_grad
        else:
            proj_grad[proj_width - 1 - tap, c] += local_grad
    following.store(keep.load())
    _bwd_store(dx, batch, 3 * channel + c, start, result, scratch)


@cute.jit
def _backward_split_halo(
    x,
    wp,
    wm,
    skip,
    y,
    dy,
    next_mix,
    next_proj,
    batch,
    channel,
    tile,
    lane,
    proj_width: cutlass.Constexpr,
    mixer_width: cutlass.Constexpr,
    compute: cutlass.Constexpr,
    scratch,
    weight_channel,
    elements: cutlass.Constexpr,
    register_weights: cutlass.Constexpr,
):
    # Recompute only the future data-gradient halo; parameter sums belong to core tiles.
    following = cute.make_rmem_tensor((elements,), compute)
    following.fill(0)
    if tile + 1 < cute.ceil_div(x.shape[2], 32 * elements):
        p1 = _bwd_projection(x, wp, batch, 3 * channel, tile + 1, lane, proj_width, compute, scratch, 3 * weight_channel, elements, register_weights)
        future_dy = _bwd_load(dy, batch, channel, tile + 1, lane, compute, scratch, elements)
        for i in cutlass.range_constexpr(elements):
            following[i] = future_dy[i] * p1[i]
    projections = cute.make_rmem_tensor((elements, 3), compute)
    for c in cutlass.range_constexpr(3):
        values = _bwd_projection(x, wp, batch, 3 * channel + c, tile, lane, proj_width, compute, scratch, 3 * weight_channel + c, elements, register_weights)
        projections[None, c].store(values.load())
    grad = _bwd_load(dy, batch, channel, tile, lane, compute, scratch, elements)
    saved_y = _bwd_load(y, batch, channel, tile, lane, compute, scratch, elements)
    current = cute.make_rmem_tensor((elements,), compute)
    grad_product = cute.make_rmem_tensor((elements,), compute)
    for i in cutlass.range_constexpr(elements):
        current[i] = grad[i] * projections[i, 0]
        next_mix[i] = current[i]
        grad_product[i] = current[i] * skip[weight_channel].to(compute)
        next_proj[i, 0] = grad[i] * saved_y[i]
    for tap in cutlass.range_constexpr(mixer_width):
        if cutlass.const_expr(tap > 0):
            _shift_right(current, following, tap, lane)
        weight = cute.arch.shuffle_sync(wm[(mixer_width - 1 - tap) // 32], (mixer_width - 1 - tap) % 32)
        for i in cutlass.range_constexpr(elements):
            if cutlass.const_expr(tap == 0):
                grad_product[i] = cute.math.fma(current[i], skip[weight_channel].to(compute), current[i] * weight)
            else:
                grad_product[i] += current[i] * weight
    for i in cutlass.range_constexpr(elements):
        next_proj[i, 1] = grad_product[i] * projections[i, 2]
        next_proj[i, 2] = grad_product[i] * projections[i, 1]


@cute.kernel
def b2b_bwd_fused(
    x: cute.Tensor,
    wp: cute.Tensor,
    wm: cute.Tensor,
    skip: cute.Tensor,
    y: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dwp: cute.Tensor,
    dwm: cute.Tensor,
    ds: cute.Tensor,
    proj_width: cutlass.Constexpr,
    mixer_width: cutlass.Constexpr,
    warps: cutlass.Constexpr,
    elements: cutlass.Constexpr = 8,
    split: cutlass.Constexpr = False,
):
    """Retain future gradients; sparse wide workloads reconstruct halos per sequence CTA."""
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    channel = cute.arch.block_idx()[1] * warps + tid // 32
    batch = cute.arch.block_idx()[0] % y.shape[0] if cutlass.const_expr(split) else cute.arch.block_idx()[0]
    first_tile = cute.arch.block_idx()[0] // y.shape[0] if cutlass.const_expr(split) else 0
    tile_count = 1 if cutlass.const_expr(split) else cute.ceil_div(x.shape[2], 32 * elements)
    compute = dwp.element_type
    smem = SmemAllocator()
    scratch = None
    if cutlass.const_expr(x.shape[2] % elements != 0):
        scratch = smem.allocate_tensor(compute, cute.make_layout((warps, 32 * elements), stride=(32 * elements, 1)), byte_alignment=16)
    register_weights = proj_width > 8 or mixer_width > 16
    cached_wp, cached_wm, cached_skip = wp, wm, skip
    weight_channel = channel
    if cutlass.const_expr(register_weights):
        cached_wp = cute.make_rmem_tensor((cute.ceil_div(proj_width, 32), 3), compute)
        cached_wm = cute.make_rmem_tensor((cute.ceil_div(mixer_width, 32),), compute)
        cached_skip = cute.make_rmem_tensor((1,), compute)
        cached_wp.fill(0)
        cached_wm.fill(0)
        cached_skip.fill(0)
        if channel < y.shape[1]:
            for c in cutlass.range_constexpr(3):
                for g in cutlass.range_constexpr(cute.ceil_div(proj_width, 32)):
                    k = g * 32 + lane
                    if k < proj_width:
                        cached_wp[g, c] = wp[3 * channel + c, k].to(compute)
            for g in cutlass.range_constexpr(cute.ceil_div(mixer_width, 32)):
                k = g * 32 + lane
                if k < mixer_width:
                    cached_wm[g] = wm[channel, k].to(compute)
            cached_skip[0] = skip[channel].to(compute)
        weight_channel = 0
    next_mix = cute.make_rmem_tensor((elements,), compute)
    next_proj = cute.make_rmem_tensor((elements, 3), compute)
    distributed = proj_width > 8 or mixer_width > 16
    proj_slots = cute.ceil_div(proj_width, 32) if distributed else proj_width
    mix_slots = cute.ceil_div(mixer_width, 32) if distributed else mixer_width
    proj_grad = cute.make_rmem_tensor((proj_slots, 3), compute)
    mix_grad = cute.make_rmem_tensor((mix_slots,), compute)
    next_mix.fill(0)
    next_proj.fill(0)
    proj_grad.fill(0)
    mix_grad.fill(0)
    skip_grad = compute(0)
    if channel < y.shape[1]:
        if cutlass.const_expr(split):
            if first_tile + tile_count < cute.ceil_div(x.shape[2], 32 * elements):
                _backward_split_halo(
                    x,
                    cached_wp,
                    cached_wm,
                    cached_skip,
                    y,
                    dy,
                    next_mix,
                    next_proj,
                    batch,
                    channel,
                    first_tile + tile_count,
                    lane,
                    proj_width,
                    mixer_width,
                    compute,
                    scratch,
                    weight_channel,
                    elements,
                    register_weights,
                )
        for reverse_tile in cutlass.range(0, tile_count, unroll=1):
            tile = first_tile + tile_count - 1 - reverse_tile
            start = tile * (32 * elements) + lane * elements
            projections = cute.make_rmem_tensor((elements, 3), compute)
            originals = cute.make_rmem_tensor((elements, 3), compute)
            for c in cutlass.range_constexpr(3):
                projection = _bwd_projection(
                    x, cached_wp, batch, 3 * channel + c, tile, lane, proj_width, compute, scratch, 3 * weight_channel + c, elements, register_weights
                )
                original = _bwd_load(x, batch, 3 * channel + c, tile, lane, compute, scratch, elements)
                projections[None, c].store(projection.load())
                originals[None, c].store(original.load())
            grad_out = _bwd_load(dy, batch, channel, tile, lane, compute, scratch, elements)
            saved_y = _bwd_load(y, batch, channel, tile, lane, compute, scratch, elements)
            current_mix = cute.make_rmem_tensor((elements,), compute)
            keep_mix = cute.make_rmem_tensor((elements,), compute)
            product = cute.make_rmem_tensor((elements,), compute)
            grad_product = cute.make_rmem_tensor((elements,), compute)
            for i in cutlass.range_constexpr(elements):
                current_mix[i] = grad_out[i] * projections[i, 0]
                keep_mix[i] = current_mix[i]
                product[i] = projections[i, 1] * projections[i, 2]
                grad_product[i] = current_mix[i] * cached_skip[weight_channel].to(compute)
                skip_grad += current_mix[i] * product[i]
            first_grad = cute.make_rmem_tensor((elements,), compute)
            for i in cutlass.range_constexpr(elements):
                first_grad[i] = grad_out[i] * saved_y[i]
            _projection_backward(
                first_grad,
                originals[None, 0],
                next_proj[None, 0],
                cached_wp,
                proj_grad,
                dx,
                batch,
                channel,
                start,
                lane,
                proj_width,
                distributed,
                0,
                compute,
                scratch,
                weight_channel,
                register_weights,
            )
            for tap in cutlass.range_constexpr(mixer_width):
                if cutlass.const_expr(tap > 0):
                    _shift_right(current_mix, next_mix, tap, lane)
                if cutlass.const_expr(register_weights):
                    weight = cute.arch.shuffle_sync(cached_wm[(mixer_width - 1 - tap) // 32], (mixer_width - 1 - tap) % 32)
                else:
                    weight = cached_wm[weight_channel, mixer_width - 1 - tap].to(compute)
                local_grad = compute(0)
                for i in cutlass.range_constexpr(elements):
                    if cutlass.const_expr(tap == 0 and distributed):
                        # Match the backend first-tap contraction after the skip product.
                        grad_product[i] = cute.math.fma(current_mix[i], cached_skip[weight_channel].to(compute), current_mix[i] * weight)
                    else:
                        grad_product[i] += current_mix[i] * weight
                    local_grad += product[i] * current_mix[i]
                if cutlass.const_expr(distributed):
                    local_grad = _warp_sum(local_grad)
                    if lane == (mixer_width - 1 - tap) % 32:
                        mix_grad[(mixer_width - 1 - tap) // 32] += local_grad
                else:
                    mix_grad[mixer_width - 1 - tap] += local_grad
            next_mix.store(keep_mix.load())
            current_proj = cute.make_rmem_tensor((elements, 3), compute)
            for i in cutlass.range_constexpr(elements):
                current_proj[i, 1] = grad_product[i] * projections[i, 2]
                current_proj[i, 2] = grad_product[i] * projections[i, 1]
            for c in cutlass.range_constexpr(1, 3):
                _projection_backward(
                    current_proj[None, c],
                    originals[None, c],
                    next_proj[None, c],
                    cached_wp,
                    proj_grad,
                    dx,
                    batch,
                    channel,
                    start,
                    lane,
                    proj_width,
                    distributed,
                    c,
                    compute,
                    scratch,
                    weight_channel,
                    register_weights,
                )
        if cutlass.const_expr(not distributed):
            # Interleave independent sums to hide shuffle latency, then distribute
            # parameter atomics across lanes instead of serializing them on lane 0.
            for offset in cutlass.range_constexpr(5):
                skip_grad += cute.arch.shuffle_sync_bfly(skip_grad, 1 << offset)
                for k in cutlass.range_constexpr(mixer_width):
                    mix_grad[k] += cute.arch.shuffle_sync_bfly(mix_grad[k], 1 << offset)
                for c in cutlass.range_constexpr(3):
                    for k in cutlass.range_constexpr(proj_width):
                        proj_grad[k, c] += cute.arch.shuffle_sync_bfly(proj_grad[k, c], 1 << offset)
            proj_value = compute(0)
            for c in cutlass.range_constexpr(3):
                for k in cutlass.range_constexpr(proj_width):
                    proj_value = compute(cutlass.select_(lane == c * proj_width + k, proj_grad[k, c], proj_value))
            mix_value = compute(0)
            for k in cutlass.range_constexpr(mixer_width):
                mix_value = compute(cutlass.select_(lane == k, mix_grad[k], mix_value))
            if lane < 3 * proj_width:
                cute.arch.atomic_add(dwp.iterator + cutlass.Int64(3 * channel) * proj_width + lane, proj_value)
            if lane < mixer_width:
                cute.arch.atomic_add(dwm.iterator + cutlass.Int64(channel) * mixer_width + lane, mix_value)
            if lane == 0:
                cute.arch.atomic_add(ds.iterator + channel, skip_grad)
        else:
            skip_grad = _warp_sum(skip_grad)
            if lane == 0:
                cute.arch.atomic_add(ds.iterator + channel, skip_grad)
            for k in cutlass.range_constexpr(mix_slots):
                if k * 32 + lane < mixer_width:
                    cute.arch.atomic_add(dwm.iterator + cutlass.Int64(channel) * mixer_width + k * 32 + lane, mix_grad[k])
            for c in cutlass.range_constexpr(3):
                for k in cutlass.range_constexpr(proj_slots):
                    if k * 32 + lane < proj_width:
                        cute.arch.atomic_add(dwp.iterator + cutlass.Int64(3 * channel + c) * proj_width + k * 32 + lane, proj_grad[k, c])


b2b_bwd_fused.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch_b2b_fwd(
    x: cute.Tensor,
    wp: cute.Tensor,
    wm: cute.Tensor,
    skip: cute.Tensor,
    y: cute.Tensor,
    gated: cute.Tensor,
    stream: cuda.CUstream,
    proj_width: cutlass.Constexpr,
    mixer_width: cutlass.Constexpr,
):
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    warps = 4
    if cutlass.const_expr(proj_width > 8 or mixer_width > 16):
        if cutlass.const_expr(arch.major == 12):
            warps = 1
        elif cutlass.const_expr(arch.major == 8 and arch.minor == 6):
            warps = 2
    elements = 8
    if cutlass.const_expr(x.element_type.width == 16 and mixer_width >= 64 and x.shape[2] >= 1024 and x.shape[0] * y.shape[1] <= 128):
        if cutlass.const_expr(arch.major == 8 and arch.minor in (6, 9)):
            warps = 1
        elif cutlass.const_expr(arch.major == 10 and arch.minor == 3):
            elements, warps = 8, 4
        elif cutlass.const_expr(arch.major == 12):
            elements, warps = 16, 1
    groups = (
        cute.ceil_div(x.shape[2], 32 * elements)
        if cutlass.const_expr(
            x.element_type.width == 16
            and mixer_width >= 64
            and x.shape[2] >= 1024
            and x.shape[0] * y.shape[1] <= 128
            and arch.major == 10
            and arch.minor in (0, 3)
        )
        else 1
    )
    b2b_fwd_persistent(x, wp, wm, skip, y, gated, proj_width, mixer_width, warps, elements).launch(
        grid=(cute.ceil_div(y.shape[1], warps) * groups, y.shape[0], 1), block=(32 * warps, 1, 1), stream=stream
    )


@cute.jit
def launch_b2b_bwd(
    x: cute.Tensor,
    wp: cute.Tensor,
    wm: cute.Tensor,
    skip: cute.Tensor,
    y: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dwp: cute.Tensor,
    dwm: cute.Tensor,
    ds: cute.Tensor,
    stream: cuda.CUstream,
    proj_width: cutlass.Constexpr,
    mixer_width: cutlass.Constexpr,
    accumulate: cutlass.Constexpr = False,
):
    if cutlass.const_expr(not accumulate):
        b2b_zero_grad(dwp, dwm, ds).launch(grid=(cute.ceil_div(max(cute.size(dwp), cute.size(dwm)), 256), 1, 1), block=(256, 1, 1), stream=stream)
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    warps = 4
    if cutlass.const_expr(
        x.element_type.width == 16
        and proj_width <= 8
        and mixer_width <= 8
        and x.shape[2] >= 1024
        and ((arch.major == 8 and arch.minor == 9) or (arch.major == 10 and arch.minor == 0))
    ):
        warps = 8
    if cutlass.const_expr(arch.major >= 10 and proj_width <= 8 and mixer_width <= 16 and x.shape[2] <= 256):
        warps = 1
    if cutlass.const_expr(proj_width > 8 or mixer_width > 16):
        warps = 1
    elements = 8
    if cutlass.const_expr(arch.major == 8 and arch.minor == 9 and x.element_type.width == 16 and proj_width <= 8 and mixer_width <= 8 and x.shape[2] >= 1024):
        elements, warps = 16, 4
    if cutlass.const_expr(
        arch.major == 10
        and arch.minor == 0
        and x.element_type.width == 16
        and (proj_width > 8 or mixer_width > 16)
        and x.shape[2] >= 1024
        and x.shape[0] * y.shape[1] <= 128
    ):
        elements, warps = 16, 4
    groups = 1
    split = False
    if cutlass.const_expr(
        (arch.major == 12 or (arch.major == 10 and arch.minor == 3))
        and x.element_type.width == 16
        and mixer_width >= 64
        and x.shape[2] >= 1024
        and x.shape[0] * y.shape[1] <= 128
    ):
        split = True
        elements, warps = 8, 1
        groups = cute.ceil_div(x.shape[2], 32 * elements)
    b2b_bwd_fused(x, wp, wm, skip, y, dy, dx, dwp, dwm, ds, proj_width, mixer_width, warps, elements, split).launch(
        grid=(y.shape[0] * groups, cute.ceil_div(y.shape[1], warps), 1), block=(32 * warps, 1, 1), stream=stream
    )
