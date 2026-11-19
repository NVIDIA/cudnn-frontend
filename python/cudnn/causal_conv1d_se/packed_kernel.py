# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed direct convolution with device-side sequence boundaries and history.

The mathematical contract follows cuhyena's packed_causal_conv1d_nwh branch
at 03151ddfe59d7ea3d94d98eb55f7a649dd559e6b. Tensors use token-major
[T, H], logical [H, K] filters, and [S, K-1, H] histories with explicit strides.
"""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from cudnn._cutlass_compat import SmemAllocator

from .kernel import _activate, _nwh_async_vector, _nwh_load_vector


@cute.jit
def _validate_offsets(cu: cute.Tensor, total_tokens):
    """Distribute metadata checks across the convolution grid without a host read."""
    thread = cutlass.Int64(cute.arch.block_idx()[0]) * cute.arch.block_dim()[0] + cute.arch.thread_idx()[0]
    stride = cutlass.Int64(cute.arch.grid_dim()[0]) * cute.arch.block_dim()[0]
    sequences = cutlass.Int64(cu.shape[0] - 1)
    if thread == 0:
        inline_ptx("trap;", predicate=cu[0] != 0)
        inline_ptx("trap;", predicate=cutlass.Int64(cu[sequences]) != total_tokens)
    boundary = thread
    while boundary < sequences:
        left = cutlass.Int64(cu[boundary])
        right = cutlass.Int64(cu[boundary + 1])
        inline_ptx("trap;", predicate=left < 0 or right < left or right > total_tokens)
        boundary = boundary + stride


@cute.kernel
def packed_validate_empty(cu: cute.Tensor):
    _validate_offsets(cu, cutlass.Int64(0))


packed_validate_empty.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _sequence(cu: cute.Tensor, token, index_type: cutlass.Constexpr = cutlass.Int64):
    lo = index_type(0)
    hi = index_type(cu.shape[0] - 1)
    while lo + 1 < hi:
        mid = lo + (hi - lo) // 2
        if cu[mid] <= token:
            lo = mid
        else:
            hi = mid
    return lo


@cute.jit
def _input(x: cute.Tensor, initial: cute.Tensor, seq, start, pos, h, width: cutlass.Constexpr):
    value = cutlass.Float32(0)
    if pos >= start and pos >= 0 and pos < x.shape[0]:
        value = x[pos, h].to(cutlass.Float32)
    elif cutlass.const_expr(initial is not None):
        row = pos - start + width - 1
        if row >= 0 and row < width - 1:
            value = initial[cutlass.Int64(seq), row, h].to(cutlass.Float32)
    return value


@cute.jit
def _preactivation(x: cute.Tensor, w: cute.Tensor, bias: cute.Tensor, initial: cute.Tensor, seq, start, pos, h, width: cutlass.Constexpr):
    acc = cutlass.Float32(0)
    if cutlass.const_expr(bias is not None):
        acc = bias[h].to(cutlass.Float32)
    for k in cutlass.range_constexpr(width):
        value = _input(x, initial, seq, start, pos + k - width + 1, h, width)
        acc = acc + value * w[h, k].to(cutlass.Float32)
    return acc


@cute.kernel
def packed_forward(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    _validate_offsets(cu, x.shape[0])
    blocks_h = cute.ceil_div(x.shape[1], 128)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    first = cutlass.Int64(cute.arch.block_idx()[0] // blocks_h) * rows
    if h < x.shape[1]:
        seq = _sequence(cu, first)
        for r in cutlass.range_constexpr(rows):
            pos = first + r
            if pos < x.shape[0]:
                while seq + 1 < cu.shape[0] - 1 and cu[seq + 1] <= pos:
                    seq = seq + 1
                start = min(max(cutlass.Int64(cu[seq]), 0), x.shape[0])
                acc = _preactivation(x, w, bias, initial, seq, start, pos, h, width)
                y[pos, h] = _activate(acc, silu, x.element_type.width == 16).to(y.element_type)


packed_forward.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def packed_final_state(x: cute.Tensor, cu: cute.Tensor, initial: cute.Tensor, final: cute.Tensor, width: cutlass.Constexpr):
    _validate_offsets(cu, x.shape[0])
    blocks_h = cute.ceil_div(x.shape[1], 128)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    seq = cute.arch.block_idx()[0] // blocks_h
    if h < x.shape[1]:
        start = min(max(cutlass.Int64(cu[seq]), 0), x.shape[0])
        end = min(max(cutlass.Int64(cu[seq + 1]), start), x.shape[0])
        for k in cutlass.range_constexpr(width - 1):
            final[cutlass.Int64(seq), k, h] = _input(x, initial, seq, start, end - width + 1 + k, h, width).to(final.element_type)


packed_final_state.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _compensated_silu_gradient_fp32(grad, z):
    """Evaluate the bounded SiLU derivative with FP32 rounding residuals."""
    one = cutlass.Float32(1.0)
    exponential = cute.exp(-cute.abs(z), fastmath=False)
    denominator = one + exponential
    denominator_low = exponential - (denominator - one)
    inverse = cute.math.div(one, denominator)
    inverse_residual = cute.math.fma(-inverse, denominator, one) - inverse * denominator_low

    shifted = one + z
    z_virtual = shifted - one
    shifted_low = (one - (shifted - z_virtual)) + (z - z_virtual)
    numerator = cutlass.Float32(0.0)
    numerator_low = cutlass.Float32(0.0)
    if z >= 0.0:
        product = exponential * shifted
        product_low = cute.math.fma(exponential, shifted, -product) + exponential * shifted_low
        numerator = one + product
        product_virtual = numerator - one
        numerator_low = (one - (numerator - product_virtual)) + (product - product_virtual) + product_low
    else:
        total = shifted + exponential
        exp_virtual = total - shifted
        total_low = (shifted - (total - exp_virtual)) + (exponential - exp_virtual) + shifted_low
        numerator = exponential * total
        numerator_low = cute.math.fma(exponential, total, -numerator) + exponential * total_low

    quotient = numerator * inverse
    quotient_low = cute.math.fma(numerator, inverse, -quotient) + numerator_low * inverse
    derivative = quotient * inverse
    derivative_low = cute.math.fma(quotient, inverse, -derivative) + quotient_low * inverse + (2.0 * inverse_residual) * derivative
    result = grad * derivative
    if cute.abs(result) <= cutlass.Float32(3.4028234663852886e38):
        result_low = cute.math.fma(grad, derivative, -result) + grad * derivative_low
        result = result + result_low
    return result


@cute.kernel
def packed_dz(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    dy: cute.Tensor,
    dz: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
):
    _validate_offsets(cu, x.shape[0])
    blocks_h = cute.ceil_div(x.shape[1], 128)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    pos = cutlass.Int64(cute.arch.block_idx()[0] // blocks_h)
    if h < x.shape[1]:
        grad = dy[pos, h].to(cutlass.Float32)
        if cutlass.const_expr(silu):
            seq = _sequence(cu, pos)
            start = min(max(cutlass.Int64(cu[seq]), 0), x.shape[0])
            z = _preactivation(x, w, bias, initial, seq, start, pos, h, width)
            if cutlass.const_expr(x.element_type == cutlass.Float32):
                grad = _compensated_silu_gradient_fp32(grad, z)
            else:
                sigmoid = cute.math.div(1.0, 1.0 + cute.exp(-z, fastmath=True), approx=True, ftz=True)
                grad = grad * sigmoid * (1.0 + z * (1.0 - sigmoid))
        dz[cutlass.Int64(pos), h] = grad


packed_dz.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def packed_dx(w: cute.Tensor, cu: cute.Tensor, dz: cute.Tensor, dfinal: cute.Tensor, dx: cute.Tensor, width: cutlass.Constexpr):
    blocks_h = cute.ceil_div(dx.shape[1], 128)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    pos = cutlass.Int64(cute.arch.block_idx()[0] // blocks_h)
    if h < dx.shape[1]:
        seq = _sequence(cu, pos)
        end = min(max(cutlass.Int64(cu[seq + 1]), 0), dx.shape[0])
        acc = cutlass.Float32(0)
        for k in cutlass.range_constexpr(width):
            q = pos + width - 1 - k
            if q < end:
                acc = acc + dz[q, h] * w[h, k].to(cutlass.Float32)
        if cutlass.const_expr(dfinal is not None):
            row = pos - end + width - 1
            if row >= 0 and row < width - 1:
                acc = acc + dfinal[cutlass.Int64(seq), row, h].to(cutlass.Float32)
        dx[pos, h] = acc.to(dx.element_type)


packed_dx.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def packed_dinitial(w: cute.Tensor, cu: cute.Tensor, dz: cute.Tensor, dfinal: cute.Tensor, dinitial: cute.Tensor, width: cutlass.Constexpr):
    blocks_h = cute.ceil_div(dz.shape[1], 128)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    seq = cute.arch.block_idx()[0] // blocks_h
    if h < dz.shape[1]:
        start = min(max(cutlass.Int64(cu[seq]), 0), dz.shape[0])
        end = min(max(cutlass.Int64(cu[seq + 1]), start), dz.shape[0])
        for j in cutlass.range_constexpr(width - 1):
            acc = cutlass.Float32(0)
            for k in cutlass.range_constexpr(j + 1):
                q = start + j - k
                if q < end:
                    acc = acc + dz[q, h] * w[h, k].to(cutlass.Float32)
            if cutlass.const_expr(dfinal is not None):
                row = j - (end - start)
                if row >= 0 and row < width - 1:
                    acc = acc + dfinal[cutlass.Int64(seq), row, h].to(cutlass.Float32)
            dinitial[cutlass.Int64(seq), j, h] = acc.to(dinitial.element_type)


packed_dinitial.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _compensated_product(a, b, high, low):
    """Preserve product and addition residuals with explicit rounding."""
    return inline_ptx(
        """{
        .reg .f32 p, pe, s, bv, av, br, ar, e, correction, next_low;
        .reg .pred finite, product_finite;
        mul.rn.f32 p, $2, $3;
        neg.f32 pe, p;
        fma.rn.f32 pe, $2, $3, pe;
        add.rn.f32 s, $4, p;
        sub.rn.f32 bv, s, $4;
        sub.rn.f32 av, s, bv;
        sub.rn.f32 br, p, bv;
        sub.rn.f32 ar, $4, av;
        add.rn.f32 e, ar, br;
        add.rn.f32 correction, e, pe;
        add.rn.f32 next_low, $5, correction;
        testp.finite.f32 finite, s;
        testp.finite.f32 product_finite, p;
        and.pred finite, finite, product_finite;
        @!finite fma.rn.f32 s, $2, $3, $4;
        @!finite mov.b32 next_low, 0;
        mov.f32 $0, s;
        mov.f32 $1, next_low;
        }""",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a, b, high, low],
    )


@cute.jit
def _compensated_sum(value, high, low):
    return inline_ptx(
        """{
        .reg .f32 s, bv, av, br, ar, e, next_low;
        .reg .pred finite;
        add.rn.f32 s, $3, $2;
        sub.rn.f32 bv, s, $3;
        sub.rn.f32 av, s, bv;
        sub.rn.f32 br, $2, bv;
        sub.rn.f32 ar, $3, av;
        add.rn.f32 e, ar, br;
        add.rn.f32 next_low, $4, e;
        testp.finite.f32 finite, s;
        @!finite mov.b32 next_low, 0;
        mov.f32 $0, s;
        mov.f32 $1, next_low;
        }""",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[value, high, low],
    )


@cute.kernel
def packed_parameter_partials(x: cute.Tensor, cu: cute.Tensor, initial: cute.Tensor, dz: cute.Tensor, partials: cute.Tensor, width: cutlass.Constexpr):
    blocks_h = cute.ceil_div(x.shape[1], 128)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    split = cute.arch.block_idx()[0] // blocks_h
    span = cute.ceil_div(x.shape[0], partials.shape[0])
    first = cutlass.Int64(split) * span
    last = min(first + span, x.shape[0])
    if h < x.shape[1]:
        accum = cute.make_rmem_tensor((width + 1,), partials.element_type)
        accum.fill(0)
        seq = _sequence(cu, first)
        for pos in cutlass.range(first, last, 1):
            while seq + 1 < cu.shape[0] - 1 and cu[seq + 1] <= pos:
                seq = seq + 1
            start = min(max(cutlass.Int64(cu[seq]), 0), x.shape[0])
            grad = dz[cutlass.Int64(pos), h]
            for k in cutlass.range_constexpr(width):
                value = _input(x, initial, seq, start, pos + k - width + 1, h, width)
                accum[k] = cute.math.fma(value.to(partials.element_type), grad.to(partials.element_type), accum[k])
            accum[width] = accum[width] + grad.to(partials.element_type)
        for k in cutlass.range_constexpr(width + 1):
            partials[cutlass.Int64(split), k, h] = accum[k]


packed_parameter_partials.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def packed_parameter_reduce(partials: cute.Tensor, dw: cute.Tensor, db: cute.Tensor, width: cutlass.Constexpr):
    h = cutlass.Int64(cute.arch.block_idx()[0]) * 128 + cute.arch.thread_idx()[0]
    if h < dw.shape[0]:
        for k in cutlass.range_constexpr(width + 1):
            acc = cutlass.Float32(0)
            for split in cutlass.range(partials.shape[0]):
                acc = acc + partials[cutlass.Int64(split), k, h]
            if cutlass.const_expr(k < width):
                dw[h, k] = acc.to(dw.element_type)
            elif cutlass.const_expr(db is not None):
                db[h] = acc.to(db.element_type)


packed_parameter_reduce.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def packed_parameter_reduce_fp32(partials: cute.Tensor, dw: cute.Tensor, db: cute.Tensor, width: cutlass.Constexpr):
    """Accumulate FP32 gradient partials in FP64, parallel across filter taps."""
    blocks_h = cute.ceil_div(dw.shape[0], 128)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    tap = cute.arch.block_idx()[0] // blocks_h
    if h < dw.shape[0]:
        acc = cutlass.Float64(0)
        for split in cutlass.range(partials.shape[0]):
            acc = acc + partials[cutlass.Int64(split), tap, h].to(cutlass.Float64)
        if tap < width:
            dw[h, tap] = acc.to(dw.element_type)
        elif cutlass.const_expr(db is not None):
            db[h] = acc.to(db.element_type)


packed_parameter_reduce_fp32.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _packed_forward_rows(
    sx: cute.Tensor,
    weights: cute.Tensor,
    biases: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    token_start,
    h0,
    col,
    rbase,
    tile_seq,
    tile_start,
    tile_end,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    th: cutlass.Constexpr,
    boundary: cutlass.Constexpr,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    vector = 128 // y.element_type.width
    window = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    for k in cutlass.range_constexpr(width - 1):
        raw = _nwh_load_vector(sx, (rbase + k) * th + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[k, v] = raw[v].to(cutlass.Float32)
    if cutlass.const_expr(boundary):
        seq = tile_seq
        start = tile_start
        end = tile_end
    if h0 + col < y.shape[1]:
        for r in cutlass.range_constexpr(rows):
            pos = token_start + rbase + r
            raw = _nwh_load_vector(sx, (rbase + r + width - 1) * th + col, vector)
            for v in cutlass.range_constexpr(vector):
                window[(r + width - 1) % width, v] = raw[v].to(cutlass.Float32)
            valid = cutlass.Boolean(True)
            if cutlass.const_expr(boundary):
                valid = pos < y.shape[0]
            if valid:
                if cutlass.const_expr(boundary):
                    while pos >= end and seq + 1 < cu.shape[0] - 1:
                        seq = seq + 1
                        start = min(max(index_type(cu[seq]), 0), y.shape[0])
                        end = min(max(index_type(cu[seq + 1]), start), y.shape[0])
                    if pos - start < width - 1:
                        for k in cutlass.range_constexpr(width):
                            state_row = pos - start + k
                            if state_row < width - 1:
                                for v in cutlass.range_constexpr(vector):
                                    value = cutlass.Float32(0)
                                    if cutlass.const_expr(initial is not None):
                                        if state_row >= 0:
                                            value = initial[cutlass.Int64(seq), cutlass.Int64(state_row), cutlass.Int64(h0) + col + v].to(cutlass.Float32)
                                    window[(r + k) % width, v] = value
                result = cute.make_rmem_tensor((vector,), cutlass.Float32)
                result.store(biases.load())
                for k in cutlass.range_constexpr(width):
                    for v in cutlass.range_constexpr(vector):
                        result[v] = cute.math.fma(window[(r + k) % width, v], weights[k, v], result[v])
                out = cute.make_rmem_tensor((vector,), y.element_type)
                for v in cutlass.range_constexpr(vector):
                    out[v] = _activate(result[v], silu, y.element_type.width == 16).to(y.element_type)
                target = cute.make_tensor((y.iterator + cutlass.Int64(pos) * y.stride[0] + h0 + col).align(16), cute.make_layout((vector,)))
                atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), y.element_type, num_bits_per_copy=128)
                cute.copy(atom, out, target)


@cute.kernel
def packed_forward_tiled(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
    threads: cutlass.Constexpr = 128,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    _validate_offsets(cu, x.shape[0])
    vector = 128 // x.element_type.width
    tpr = th // vector
    row_groups = threads // tpr
    rows = ts // row_groups
    tid = cute.arch.thread_idx()[0]
    blocks_h = cute.ceil_div(x.shape[1], th)
    token_start = index_type(cute.arch.block_idx()[0] // blocks_h) * ts
    h0 = index_type(cute.arch.block_idx()[0] % blocks_h) * th
    col = (tid % tpr) * vector
    rbase = (tid // tpr) * rows
    allocator = SmemAllocator()
    sx = allocator.allocate_tensor(x.element_type, cute.make_layout(((ts + width - 1) * th,)), byte_alignment=16)
    bounds = allocator.allocate_tensor(index_type, cute.make_layout((3,)), byte_alignment=16)
    for r in cutlass.range_constexpr(cute.ceil_div(ts + width - 1, row_groups)):
        row = (tid // tpr) + r * row_groups
        pos = token_start + row - width + 1
        if row < ts + width - 1:
            _nwh_async_vector(
                x, sx, cutlass.Int64(pos) * x.stride[0] + h0 + col, row * th + col, pos >= 0 and pos < x.shape[0] and h0 + col < x.shape[1], vector
            )
    cute.arch.cp_async_commit_group()
    if tid == 0:
        seq = _sequence(cu, token_start, index_type)
        start = min(max(index_type(cu[seq]), 0), x.shape[0])
        end = min(max(index_type(cu[seq + 1]), start), x.shape[0])
        bounds[0] = seq
        bounds[1] = start
        bounds[2] = end
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    biases = cute.make_rmem_tensor((vector,), cutlass.Float32)
    weights.fill(0)
    biases.fill(0)
    if h0 + col < x.shape[1]:
        for k in cutlass.range_constexpr(width):
            for v in cutlass.range_constexpr(vector):
                weights[k, v] = w[cutlass.Int64(h0) + col + v, k].to(cutlass.Float32)
        if cutlass.const_expr(bias is not None):
            raw_bias = _nwh_load_vector(bias, h0 + col, vector)
            biases.store(raw_bias.load().to(cutlass.Float32))
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()
    seq, start, end = bounds[0], bounds[1], bounds[2]
    # A whole-tile proof removes the sequence walk, prefix masks and token
    # predicates from the common interior path. The same staged input serves
    # boundary/partial tiles without another launch or host metadata read.
    if token_start - width + 1 >= start and token_start + ts <= end:
        _packed_forward_rows(sx, weights, biases, cu, initial, y, token_start, h0, col, rbase, seq, start, end, width, silu, rows, th, False, index_type)
    else:
        _packed_forward_rows(sx, weights, biases, cu, initial, y, token_start, h0, col, rbase, seq, start, end, width, silu, rows, th, True, index_type)


packed_forward_tiled.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def packed_forward_short(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    final: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    weight_vector: cutlass.Constexpr,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    """Keep short-sequence history in registers and fuse its final write.

    T == S selects this schedule but does not imply every sequence has L == 1;
    [2, 0] and empty sequences still execute their declared ranges correctly.
    """
    _validate_offsets(cu, x.shape[0])
    vector = 128 // x.element_type.width
    tile_h = 64
    threads = 128
    lanes_per_sequence = tile_h // vector
    sequences_per_block = threads // lanes_per_sequence
    blocks_h = cute.ceil_div(x.shape[1], tile_h)
    tile_channel = index_type(cute.arch.block_idx()[0] % blocks_h) * tile_h
    tid = cute.arch.thread_idx()[0]
    seq = cutlass.Int64(cute.arch.block_idx()[0] // blocks_h) * sequences_per_block + tid // lanes_per_sequence
    col = (tid % lanes_per_sequence) * vector
    h = tile_channel + col
    allocator = SmemAllocator()
    shared_w = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((width, tile_h), stride=(tile_h, 1)), byte_alignment=16)
    shared_b = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((tile_h,)), byte_alignment=16)
    for slot in cutlass.range_constexpr(cute.ceil_div(width * tile_h, threads)):
        linear = slot * threads + tid
        if linear < width * tile_h:
            tap = linear // tile_h
            channel = tile_channel + linear % tile_h
            if channel < x.shape[1]:
                shared_w[tap, linear % tile_h] = w[cutlass.Int64(channel), tap].to(cutlass.Float32)
    if tid < tile_h:
        channel = tile_channel + tid
        if channel < x.shape[1]:
            if cutlass.const_expr(bias is not None):
                shared_b[tid] = bias[cutlass.Int64(channel)].to(cutlass.Float32)
            else:
                shared_b[tid] = cutlass.Float32(0)
    cute.arch.sync_threads()
    if seq < cu.shape[0] - 1 and h < x.shape[1]:
        start = min(max(index_type(cu[seq]), 0), x.shape[0])
        end = min(max(index_type(cu[seq + 1]), start), x.shape[0])
        weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
        history = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), x.element_type)
        history.fill(0)
        biases = cute.make_rmem_tensor((vector,), cutlass.Float32)
        biases.fill(0)
        for k in cutlass.range_constexpr(width):
            for chunk in cutlass.range_constexpr(vector // 4):
                raw = _nwh_load_vector(shared_w, k * tile_h + col + chunk * 4, 4)
                for v in cutlass.range_constexpr(4):
                    weights[k, chunk * 4 + v] = raw[v]
        for chunk in cutlass.range_constexpr(vector // 4):
            raw = _nwh_load_vector(shared_b, col + chunk * 4, 4)
            for v in cutlass.range_constexpr(4):
                biases[chunk * 4 + v] = raw[v]
        if cutlass.const_expr(initial is not None):
            for k in cutlass.range_constexpr(width - 1):
                if cutlass.const_expr(
                    cute.is_static(initial.stride)
                    and initial.stride[2] == 1
                    and initial.stride[0] * x.element_type.width % 128 == 0
                    and initial.stride[1] * x.element_type.width % 128 == 0
                ):
                    initial_raw = _nwh_load_vector(initial, cutlass.Int64(seq) * initial.stride[0] + k * initial.stride[1] + h, vector)
                    for v in cutlass.range_constexpr(vector):
                        history[k, v] = initial_raw[v]
                else:
                    for v in cutlass.range_constexpr(vector):
                        history[k, v] = initial[cutlass.Int64(seq), k, cutlass.Int64(h) + v]
        atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), y.element_type, num_bits_per_copy=128)
        for pos in cutlass.range(start, end):
            current = _nwh_load_vector(x, cutlass.Int64(pos) * x.stride[0] + h, vector)
            for v in cutlass.range_constexpr(vector):
                history[width - 1, v] = current[v]
            result = cute.make_rmem_tensor((vector,), cutlass.Float32)
            result.store(biases.load())
            for k in cutlass.range_constexpr(width):
                for v in cutlass.range_constexpr(vector):
                    result[v] = cute.math.fma(history[k, v].to(cutlass.Float32), weights[k, v], result[v])
            out = cute.make_rmem_tensor((vector,), y.element_type)
            for v in cutlass.range_constexpr(vector):
                out[v] = _activate(result[v], silu, y.element_type.width == 16).to(y.element_type)
            target = cute.make_tensor((y.iterator + cutlass.Int64(pos) * y.stride[0] + h).align(16), cute.make_layout((vector,)))
            cute.copy(atom, out, target)
            for k in cutlass.range_constexpr(width - 1):
                for v in cutlass.range_constexpr(vector):
                    history[k, v] = history[k + 1, v]
        if cutlass.const_expr(final is not None):
            if cutlass.const_expr(cute.is_static(final.stride) and final.stride[1] == 1 and final.stride[2] == width - 1):
                # Interleave the raw register history into contiguous public SHK order.
                state_out = cute.make_rmem_tensor((vector * (width - 1),), final.element_type)
                for v in cutlass.range_constexpr(vector):
                    for k in cutlass.range_constexpr(width - 1):
                        state_out[v * (width - 1) + k] = history[k, v]
                destination = cute.make_tensor(
                    (final.iterator + cutlass.Int64(seq) * final.stride[0] + cutlass.Int64(h) * (width - 1)).align(16),
                    cute.make_layout((vector * (width - 1),)),
                )
                cute.copy(atom, state_out, destination)
            else:
                for k in cutlass.range_constexpr(width - 1):
                    state_out = cute.make_rmem_tensor((vector,), final.element_type)
                    for v in cutlass.range_constexpr(vector):
                        state_out[v] = history[k, v]
                    destination = cute.make_tensor(
                        (final.iterator + cutlass.Int64(seq) * final.stride[0] + k * final.stride[1] + h).align(16), cute.make_layout((vector,))
                    )
                    cute.copy(atom, state_out, destination)


packed_forward_short.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _use_tma_forward():
    """Use the measured TMA schedule on datacenter Blackwell compile targets."""
    from cutlass.cutlass_dsl import CuTeDSL

    return CuTeDSL._get_dsl().get_arch_enum().major == 10


@cute.kernel
def packed_forward_tma(
    atom: cute.CopyAtom,
    tx: cute.Tensor,
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
    threads: cutlass.Constexpr = 128,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    _validate_offsets(cu, x.shape[0])
    vector = 128 // x.element_type.width
    tpr = th // vector
    row_groups = threads // tpr
    rows = ts // row_groups
    tid = cute.arch.thread_idx()[0]
    blocks_h = cute.ceil_div(x.shape[1], th)
    token_start = index_type(cute.arch.block_idx()[0] // blocks_h) * ts
    h0 = index_type(cute.arch.block_idx()[0] % blocks_h) * th
    col = (tid % tpr) * vector
    rbase = (tid // tpr) * rows
    allocator = SmemAllocator()
    sx = allocator.allocate_tensor(x.element_type, cute.make_layout(((ts + width - 1) * th,)), byte_alignment=128)
    bounds = allocator.allocate_tensor(index_type, cute.make_layout((3,)), byte_alignment=16)
    mb = allocator.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=8)
    if tid == 0:
        cpasync.prefetch_descriptor(atom)
        cute.arch.mbarrier_init(mb.iterator, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    shifted = cute.domain_offset((h0, token_start - width + 1), tx)
    gt = cute.local_tile(shifted, (th, ts + width - 1), (0, 0))
    st = cute.make_tensor(sx.iterator, cute.make_layout((th, ts + width - 1), stride=(1, th)))
    sp, gp = cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(st, 0, 2), cute.group_modes(gt, 0, 2))
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(mb.iterator, (ts + width - 1) * th * (x.element_type.width // 8))
    if tid < 32:
        cute.copy(atom, gp, sp, tma_bar_ptr=mb.iterator)
    if tid == 0:
        seq = _sequence(cu, token_start, index_type)
        start = min(max(index_type(cu[seq]), 0), x.shape[0])
        end = min(max(index_type(cu[seq + 1]), start), x.shape[0])
        bounds[0] = seq
        bounds[1] = start
        bounds[2] = end
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    biases = cute.make_rmem_tensor((vector,), cutlass.Float32)
    weights.fill(0)
    biases.fill(0)
    if h0 + col < x.shape[1]:
        for k in cutlass.range_constexpr(width):
            for v in cutlass.range_constexpr(vector):
                weights[k, v] = w[cutlass.Int64(h0) + col + v, k].to(cutlass.Float32)
        if cutlass.const_expr(bias is not None):
            raw_bias = _nwh_load_vector(bias, h0 + col, vector)
            biases.store(raw_bias.load().to(cutlass.Float32))
    cute.arch.mbarrier_wait(mb.iterator, 0)
    cute.arch.barrier()
    seq, start, end = bounds[0], bounds[1], bounds[2]
    # A whole-tile proof removes the sequence walk, prefix masks and token
    # predicates from the common interior path. The same staged input serves
    # boundary/partial tiles without another launch or host metadata read.
    if token_start - width + 1 >= start and token_start + ts <= end:
        _packed_forward_rows(sx, weights, biases, cu, initial, y, token_start, h0, col, rbase, seq, start, end, width, silu, rows, th, False, index_type)
    else:
        _packed_forward_rows(sx, weights, biases, cu, initial, y, token_start, h0, col, rbase, seq, start, end, width, silu, rows, th, True, index_type)


packed_forward_tma.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _packed_forward_register_raw(x: cute.Tensor, pos, h, vector: cutlass.Constexpr, boundary: cutlass.Constexpr):
    raw = cute.make_rmem_tensor((vector,), x.element_type)
    if cutlass.const_expr(boundary):
        raw.fill(0)
        if pos >= 0 and pos < x.shape[0]:
            raw.store(_nwh_load_vector(x, cutlass.Int64(pos) * x.stride[0] + h, vector).load())
    else:
        raw.store(_nwh_load_vector(x, cutlass.Int64(pos) * x.stride[0] + h, vector).load())
    return raw


@cute.jit
def _packed_forward_register_rows(
    x: cute.Tensor,
    weights: cute.Tensor,
    biases: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    token_start,
    h0,
    col,
    rbase,
    tile_seq,
    tile_start,
    tile_end,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    th: cutlass.Constexpr,
    boundary: cutlass.Constexpr,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    vector = 128 // y.element_type.width
    window = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    for k in cutlass.range_constexpr(width - 1):
        raw = _packed_forward_register_raw(x, token_start + rbase + k - width + 1, h0 + col, vector, boundary)
        for v in cutlass.range_constexpr(vector):
            window[k, v] = raw[v].to(cutlass.Float32)
    if cutlass.const_expr(boundary):
        seq = tile_seq
        start = tile_start
        end = tile_end
    if h0 + col < y.shape[1]:
        for r in cutlass.range_constexpr(rows):
            pos = token_start + rbase + r
            raw = _packed_forward_register_raw(x, token_start + rbase + r, h0 + col, vector, boundary)
            for v in cutlass.range_constexpr(vector):
                window[(r + width - 1) % width, v] = raw[v].to(cutlass.Float32)
            valid = cutlass.Boolean(True)
            if cutlass.const_expr(boundary):
                valid = pos < y.shape[0]
            if valid:
                if cutlass.const_expr(boundary):
                    while pos >= end and seq + 1 < cu.shape[0] - 1:
                        seq = seq + 1
                        start = min(max(index_type(cu[seq]), 0), y.shape[0])
                        end = min(max(index_type(cu[seq + 1]), start), y.shape[0])
                    if pos - start < width - 1:
                        for k in cutlass.range_constexpr(width):
                            state_row = pos - start + k
                            if state_row < width - 1:
                                for v in cutlass.range_constexpr(vector):
                                    value = cutlass.Float32(0)
                                    if cutlass.const_expr(initial is not None):
                                        if state_row >= 0:
                                            value = initial[cutlass.Int64(seq), cutlass.Int64(state_row), cutlass.Int64(h0) + col + v].to(cutlass.Float32)
                                    window[(r + k) % width, v] = value
                result = cute.make_rmem_tensor((vector,), cutlass.Float32)
                result.store(biases.load())
                for k in cutlass.range_constexpr(width):
                    for v in cutlass.range_constexpr(vector):
                        result[v] = cute.math.fma(window[(r + k) % width, v], weights[k, v], result[v])
                out = cute.make_rmem_tensor((vector,), y.element_type)
                for v in cutlass.range_constexpr(vector):
                    out[v] = _activate(result[v], silu, y.element_type.width == 16).to(y.element_type)
                target = cute.make_tensor((y.iterator + cutlass.Int64(pos) * y.stride[0] + h0 + col).align(16), cute.make_layout((vector,)))
                atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), y.element_type, num_bits_per_copy=128)
                cute.copy(atom, out, target)


@cute.kernel
def packed_forward_register(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
    threads: cutlass.Constexpr = 128,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    _validate_offsets(cu, x.shape[0])
    vector = 128 // x.element_type.width
    tpr = th // vector
    groups = threads // tpr
    rows = ts // groups
    tid = cute.arch.thread_idx()[0]
    blocks_h = cute.ceil_div(x.shape[1], th)
    token_start = index_type(cute.arch.block_idx()[0] // blocks_h) * ts
    h0 = index_type(cute.arch.block_idx()[0] % blocks_h) * th
    col = (tid % tpr) * vector
    rbase = (tid // tpr) * rows
    if h0 + col < x.shape[1]:
        first = token_start + rbase
        seq = _sequence(cu, first, index_type)
        start = min(max(index_type(cu[seq]), 0), x.shape[0])
        end = min(max(index_type(cu[seq + 1]), start), x.shape[0])
        weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
        biases = cute.make_rmem_tensor((vector,), cutlass.Float32)
        biases.fill(0)
        if cutlass.const_expr(isinstance(w.stride[0], int) and isinstance(w.stride[1], int)):
            if cutlass.const_expr(w.stride[0] == 1 and w.stride[1] % vector == 0):
                for k in cutlass.range_constexpr(width):
                    raw_weight = _nwh_load_vector(w, cutlass.Int64(k) * w.stride[1] + h0 + col, vector)
                    for v in cutlass.range_constexpr(vector):
                        weights[k, v] = raw_weight[v].to(cutlass.Float32)
            elif cutlass.const_expr(w.stride[1] == 1 and w.stride[0] == width):
                for chunk in cutlass.range_constexpr(width):
                    raw_weight = _nwh_load_vector(w, (cutlass.Int64(h0) + col) * width + chunk * vector, vector)
                    for v in cutlass.range_constexpr(vector):
                        linear = chunk * vector + v
                        weights[linear % width, linear // width] = raw_weight[v].to(cutlass.Float32)
            else:
                for k in cutlass.range_constexpr(width):
                    for v in cutlass.range_constexpr(vector):
                        weights[k, v] = w[cutlass.Int64(h0) + col + v, k].to(cutlass.Float32)
        else:
            for k in cutlass.range_constexpr(width):
                for v in cutlass.range_constexpr(vector):
                    weights[k, v] = w[cutlass.Int64(h0) + col + v, k].to(cutlass.Float32)
        if cutlass.const_expr(bias is not None):
            biases.store(_nwh_load_vector(bias, h0 + col, vector).load().to(cutlass.Float32))
        if first - width + 1 >= start and first + rows <= end:
            _packed_forward_register_rows(
                x, weights, biases, cu, initial, y, token_start, h0, col, rbase, seq, start, end, width, silu, rows, th, False, index_type
            )
        else:
            _packed_forward_register_rows(
                x, weights, biases, cu, initial, y, token_start, h0, col, rbase, seq, start, end, width, silu, rows, th, True, index_type
            )


packed_forward_register.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _use_register_forward():
    """Use the qualified register schedule only on native SM120 targets."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return arch.major == 12 and arch.minor == 0


@cute.kernel
def packed_forward_short_prefetch(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    final: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    weight_vector: cutlass.Constexpr,
    index_type: cutlass.Constexpr = cutlass.Int64,
    compact_weights: cutlass.Constexpr = False,
):
    """Keep short-sequence history in registers and fuse its final write.

    T == S selects this schedule but does not imply every sequence has L == 1;
    [2, 0] and empty sequences still execute their declared ranges correctly.
    """
    rows_per_thread = 1
    vector = 128 // x.element_type.width
    tile_h = 128 if cutlass.const_expr(x.element_type.width == 16) else 64
    threads = 128
    lanes_per_sequence = tile_h // vector
    sequences_per_block = (threads // lanes_per_sequence) * rows_per_thread
    blocks_h = cute.ceil_div(x.shape[1], tile_h)
    tile_channel = index_type(cute.arch.block_idx()[0] % blocks_h) * tile_h
    tid = cute.arch.thread_idx()[0]
    first_sequence = cutlass.Int64(cute.arch.block_idx()[0] // blocks_h) * sequences_per_block + (tid // lanes_per_sequence) * rows_per_thread
    col = (tid % lanes_per_sequence) * vector
    h = tile_channel + col
    weight_type = w.element_type if cutlass.const_expr(compact_weights) else cutlass.Float32
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), weight_type)
    biases = cute.make_rmem_tensor((vector,), cutlass.Float32)
    biases.fill(0)
    if h < x.shape[1]:
        if cutlass.const_expr(isinstance(w.stride[0], int) and isinstance(w.stride[1], int)):
            if cutlass.const_expr(w.stride[0] == 1 and w.stride[1] % vector == 0):
                for k in cutlass.range_constexpr(width):
                    raw_weight = _nwh_load_vector(w, cutlass.Int64(k) * w.stride[1] + h, vector)
                    for v in cutlass.range_constexpr(vector):
                        weights[k, v] = raw_weight[v].to(weight_type)
            elif cutlass.const_expr(w.stride[1] == 1 and w.stride[0] == width):
                for chunk in cutlass.range_constexpr(width):
                    raw_weight = _nwh_load_vector(w, (cutlass.Int64(h)) * width + chunk * vector, vector)
                    for v in cutlass.range_constexpr(vector):
                        linear = chunk * vector + v
                        weights[linear % width, linear // width] = raw_weight[v].to(weight_type)
            else:
                for k in cutlass.range_constexpr(width):
                    for v in cutlass.range_constexpr(vector):
                        weights[k, v] = w[cutlass.Int64(h) + v, k].to(weight_type)
        else:
            for k in cutlass.range_constexpr(width):
                for v in cutlass.range_constexpr(vector):
                    weights[k, v] = w[cutlass.Int64(h) + v, k].to(weight_type)
        if cutlass.const_expr(bias is not None):
            raw_bias = _nwh_load_vector(bias, h, vector)
            biases.store(raw_bias.load().to(cutlass.Float32))
    prefetched_x = cute.make_rmem_tensor((rows_per_thread, vector), x.element_type)
    prefetched_initial = cute.make_rmem_tensor((rows_per_thread, width - 1, vector), x.element_type)
    prefetched_initial.fill(0)
    starts = cute.make_rmem_tensor((rows_per_thread,), index_type)
    ends = cute.make_rmem_tensor((rows_per_thread,), index_type)
    for q in cutlass.range_constexpr(rows_per_thread):
        seq = first_sequence + q
        if seq < cu.shape[0] - 1 and h < x.shape[1]:
            current = _nwh_load_vector(x, cutlass.Int64(seq) * x.stride[0] + h, vector)
            for v in cutlass.range_constexpr(vector):
                prefetched_x[q, v] = current[v]
            if cutlass.const_expr(initial is not None):
                for k in cutlass.range_constexpr(width - 1):
                    if cutlass.const_expr(
                        cute.is_static(initial.stride)
                        and initial.stride[2] == 1
                        and initial.stride[0] * x.element_type.width % 128 == 0
                        and initial.stride[1] * x.element_type.width % 128 == 0
                    ):
                        initial_raw = _nwh_load_vector(initial, cutlass.Int64(seq) * initial.stride[0] + k * initial.stride[1] + h, vector)
                        for v in cutlass.range_constexpr(vector):
                            prefetched_initial[q, k, v] = initial_raw[v]
                    else:
                        for v in cutlass.range_constexpr(vector):
                            prefetched_initial[q, k, v] = initial[cutlass.Int64(seq), k, cutlass.Int64(h) + v]
            raw_start = index_type(cu[seq])
            raw_end = index_type(cu[seq + 1])
            if h == 0:
                inline_ptx("trap;", predicate=raw_start < 0 or raw_end < raw_start or raw_end > x.shape[0])
                if seq == 0:
                    inline_ptx("trap;", predicate=raw_start != 0)
                if seq == cu.shape[0] - 2:
                    inline_ptx("trap;", predicate=raw_end != x.shape[0])
            starts[q] = min(max(raw_start, 0), x.shape[0])
            ends[q] = min(max(raw_end, starts[q]), x.shape[0])
    for q in cutlass.range_constexpr(rows_per_thread):
        seq = first_sequence + q
        if seq < cu.shape[0] - 1 and h < x.shape[1]:
            start, end = starts[q], ends[q]
            history = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), x.element_type)
            history.fill(0)
            for k in cutlass.range_constexpr(width - 1):
                for v in cutlass.range_constexpr(vector):
                    history[k, v] = prefetched_initial[q, k, v]
            atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), y.element_type, num_bits_per_copy=128)
            if start == seq and end == seq + 1:
                for v in cutlass.range_constexpr(vector):
                    history[width - 1, v] = prefetched_x[q, v]
                result = cute.make_rmem_tensor((vector,), cutlass.Float32)
                result.store(biases.load())
                for k in cutlass.range_constexpr(width):
                    for v in cutlass.range_constexpr(vector):
                        result[v] = cute.math.fma(history[k, v].to(cutlass.Float32), weights[k, v].to(cutlass.Float32), result[v])
                out = cute.make_rmem_tensor((vector,), y.element_type)
                for v in cutlass.range_constexpr(vector):
                    out[v] = _activate(result[v], silu, y.element_type.width == 16).to(y.element_type)
                target = cute.make_tensor((y.iterator + cutlass.Int64(seq) * y.stride[0] + h).align(16), cute.make_layout((vector,)))
                cute.copy(atom, out, target)
                for k in cutlass.range_constexpr(width - 1):
                    for v in cutlass.range_constexpr(vector):
                        history[k, v] = history[k + 1, v]
            else:
                for pos in cutlass.range(start, end):
                    current = _nwh_load_vector(x, cutlass.Int64(pos) * x.stride[0] + h, vector)
                    for v in cutlass.range_constexpr(vector):
                        history[width - 1, v] = current[v]
                    result = cute.make_rmem_tensor((vector,), cutlass.Float32)
                    result.store(biases.load())
                    for k in cutlass.range_constexpr(width):
                        for v in cutlass.range_constexpr(vector):
                            result[v] = cute.math.fma(history[k, v].to(cutlass.Float32), weights[k, v].to(cutlass.Float32), result[v])
                    out = cute.make_rmem_tensor((vector,), y.element_type)
                    for v in cutlass.range_constexpr(vector):
                        out[v] = _activate(result[v], silu, y.element_type.width == 16).to(y.element_type)
                    target = cute.make_tensor((y.iterator + cutlass.Int64(pos) * y.stride[0] + h).align(16), cute.make_layout((vector,)))
                    cute.copy(atom, out, target)
                    for k in cutlass.range_constexpr(width - 1):
                        for v in cutlass.range_constexpr(vector):
                            history[k, v] = history[k + 1, v]
            if cutlass.const_expr(final is not None):
                if cutlass.const_expr(cute.is_static(final.stride) and final.stride[1] == 1 and final.stride[2] == width - 1):
                    # Interleave the raw register history into contiguous public SHK order.
                    state_out = cute.make_rmem_tensor((vector * (width - 1),), final.element_type)
                    for v in cutlass.range_constexpr(vector):
                        for k in cutlass.range_constexpr(width - 1):
                            state_out[v * (width - 1) + k] = history[k, v]
                    destination = cute.make_tensor(
                        (final.iterator + cutlass.Int64(seq) * final.stride[0] + cutlass.Int64(h) * (width - 1)).align(16),
                        cute.make_layout((vector * (width - 1),)),
                    )
                    cute.copy(atom, state_out, destination)
                else:
                    for k in cutlass.range_constexpr(width - 1):
                        state_out = cute.make_rmem_tensor((vector,), final.element_type)
                        for v in cutlass.range_constexpr(vector):
                            state_out[v] = history[k, v]
                        destination = cute.make_tensor(
                            (final.iterator + cutlass.Int64(seq) * final.stride[0] + k * final.stride[1] + h).align(16), cute.make_layout((vector,))
                        )
                        cute.copy(atom, state_out, destination)


packed_forward_short_prefetch.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _use_owned_short_forward():
    """Use the K4 register decode schedule on native SM80, SM89, and SM100 targets."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return (arch.major == 8 and arch.minor in (0, 9)) or (arch.major == 10 and arch.minor == 0)


def _compact_owned_short_weights():
    """Delay half-weight conversion for the native SM100 decode experiment."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return arch.major == 10 and arch.minor == 0


@cute.jit
def launch_packed_forward(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    y: cute.Tensor,
    final: cute.Tensor,
    stream: cuda.CUstream,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    fast: cutlass.Constexpr = False,
    weight_vector: cutlass.Constexpr = False,
    index32: cutlass.Constexpr = False,
):
    index_type = cutlass.Int32 if cutlass.const_expr(index32) else cutlass.Int64
    if cutlass.const_expr(final is None):
        if x.shape[0] == 0:
            packed_validate_empty(cu).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
    if cutlass.const_expr(fast):
        if x.shape[0] == cu.shape[0] - 1:
            if cutlass.const_expr(width == 4 and x.element_type.width == 16 and _use_owned_short_forward()):
                packed_forward_short_prefetch(x, w, bias, cu, initial, y, final, width, silu, weight_vector, index_type, _compact_owned_short_weights()).launch(
                    grid=(
                        cute.ceil_div(x.shape[1], (128 if cutlass.const_expr(x.element_type.width == 16) else 64))
                        * cute.ceil_div(
                            cu.shape[0] - 1, 1 * 128 * (128 // x.element_type.width) // (128 if cutlass.const_expr(x.element_type.width == 16) else 64)
                        ),
                        1,
                        1,
                    ),
                    block=(128, 1, 1),
                    stream=stream,
                )
            else:
                packed_forward_short(x, w, bias, cu, initial, y, final, width, silu, weight_vector, index_type).launch(
                    grid=(cute.ceil_div(x.shape[1], 64) * cute.ceil_div(cu.shape[0] - 1, 128 * (128 // x.element_type.width) // 64), 1, 1),
                    block=(128, 1, 1),
                    stream=stream,
                )
        else:
            if x.shape[0] > 0:
                wide_tile = cutlass.const_expr(width == 4 and x.element_type.width == 16)
                th = 128 if wide_tile else 64
                threads = 64 if cutlass.const_expr(wide_tile and not silu) else 128
                if cutlass.const_expr(silu and width == 4 and x.element_type.width == 16 and _use_tma_forward()):
                    tensor = cute.make_tensor(x.iterator, cute.make_layout((x.shape[1], x.shape[0]), stride=(x.stride[1], x.stride[0])))
                    layout = cute.make_layout((th, 128 + width - 1), stride=(1, th))
                    atom, tx = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, layout, (th, 128 + width - 1))
                    packed_forward_tma(atom, tx, x, w, bias, cu, initial, y, width, silu, 128, th, threads, index_type).launch(
                        grid=(cute.ceil_div(x.shape[1], th) * cute.ceil_div(x.shape[0], 128), 1, 1), block=(threads, 1, 1), stream=stream
                    )
                else:
                    if cutlass.const_expr(silu and width == 4 and x.element_type.width == 16 and _use_register_forward()):
                        packed_forward_register(x, w, bias, cu, initial, y, width, silu, 32, th, threads, index_type).launch(
                            grid=(cute.ceil_div(x.shape[1], th) * cute.ceil_div(x.shape[0], 32), 1, 1), block=(threads, 1, 1), stream=stream
                        )
                    else:
                        packed_forward_tiled(x, w, bias, cu, initial, y, width, silu, 128, th, threads, index_type).launch(
                            grid=(cute.ceil_div(x.shape[1], th) * cute.ceil_div(x.shape[0], 128), 1, 1), block=(threads, 1, 1), stream=stream
                        )
            if cutlass.const_expr(final is not None):
                packed_final_state(x, cu, initial, final, width).launch(
                    grid=(cute.ceil_div(x.shape[1], 128) * (cu.shape[0] - 1), 1, 1), block=(128, 1, 1), stream=stream
                )
    else:
        if x.shape[0] > 0:
            packed_forward(x, w, bias, cu, initial, y, width, silu, 4).launch(
                grid=(cute.ceil_div(x.shape[1], 128) * cute.ceil_div(x.shape[0], 4), 1, 1), block=(128, 1, 1), stream=stream
            )
        if cutlass.const_expr(final is not None):
            packed_final_state(x, cu, initial, final, width).launch(
                grid=(cute.ceil_div(x.shape[1], 128) * (cu.shape[0] - 1), 1, 1), block=(128, 1, 1), stream=stream
            )


@cute.jit
def _compensated_product_finite(a, b, high, low):
    """Use only after the CTA has proved bounded finite inputs."""
    return inline_ptx(
        """{
        .reg .f32 p, pe, s, bv, av, br, ar, e, correction, next_low;
        mul.rn.f32 p, $2, $3;
        neg.f32 pe, p;
        fma.rn.f32 pe, $2, $3, pe;
        add.rn.f32 s, $4, p;
        sub.rn.f32 bv, s, $4;
        sub.rn.f32 av, s, bv;
        sub.rn.f32 br, p, bv;
        sub.rn.f32 ar, $4, av;
        add.rn.f32 e, ar, br;
        add.rn.f32 correction, e, pe;
        add.rn.f32 next_low, $5, correction;
        mov.f32 $0, s;
        mov.f32 $1, next_low;
        }""",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a, b, high, low],
    )


@cute.jit
def _compensated_sum_finite(value, high, low):
    return inline_ptx(
        """{
        .reg .f32 s, bv, av, br, ar, e, next_low;
        add.rn.f32 s, $3, $2;
        sub.rn.f32 bv, s, $3;
        sub.rn.f32 av, s, bv;
        sub.rn.f32 br, $2, bv;
        sub.rn.f32 ar, $3, av;
        add.rn.f32 e, ar, br;
        add.rn.f32 next_low, $4, e;
        mov.f32 $0, s;
        mov.f32 $1, next_low;
        }""",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[value, high, low],
    )


@cute.jit
def _cta_all_bounded(local_bounded):
    return (
        inline_ptx(
            """{
        .reg .pred p, result;
        setp.ne.u32 p, $1, 0;
        bar.red.and.pred result, 1, p;
        selp.u32 $0, 1, 0, result;
        }""",
            write_only_types=[cutlass.Int32],
            read_only_args=[cutlass.Int32(local_bounded)],
        )
        != 0
    )


@cute.jit
def _packed_backward_rows(
    x: cute.Tensor,
    sx: cute.Tensor,
    sdy: cute.Tensor,
    weights: cute.Tensor,
    biases: cute.Tensor,
    accum: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    dfinal: cute.Tensor,
    dz: cute.Tensor,
    dx: cute.Tensor,
    token_start,
    h0,
    col,
    rbase,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    th: cutlass.Constexpr,
    boundary: cutlass.Constexpr,
    index_type: cutlass.Constexpr,
    finite_parameters: cutlass.Constexpr = False,
):
    from cutlass.cutlass_dsl import CuTeDSL

    paired = cutlass.const_expr(
        all((x.element_type.width == 16, width == 4, silu, CuTeDSL._get_dsl().get_arch_enum().major == 10, CuTeDSL._get_dsl().get_arch_enum().minor == 3))
    )
    vector = 128 // x.element_type.width
    window = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    pending = cute.make_rmem_tensor(cute.make_layout((rows, vector), stride=(vector, 1)), cutlass.Float32)
    pending.fill(0)
    if cutlass.const_expr(boundary):
        ends = cute.make_rmem_tensor((rows,), cutlass.Int32)
        ends.fill(0)
    for k in cutlass.range_constexpr(width - 1):
        raw = _nwh_load_vector(sx, (rbase + k) * th + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[k, v] = raw[v].to(cutlass.Float32)
    if cutlass.const_expr(boundary):
        seq = _sequence(cu, token_start + rbase, index_type)
        start = min(max(index_type(cu[seq]), 0), x.shape[0])
        end = min(max(index_type(cu[seq + 1]), start), x.shape[0])
    for j in cutlass.range_constexpr(rows + width - 1):
        pos = token_start + rbase + j
        raw_x = _nwh_load_vector(sx, (rbase + j + width - 1) * th + col, vector)
        raw_dy = _nwh_load_vector(sdy, (rbase + j) * th + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[(j + width - 1) % width, v] = raw_x[v].to(cutlass.Float32)
        valid = h0 + col < x.shape[1]
        if cutlass.const_expr(boundary):
            valid = valid and pos < x.shape[0]
        if valid:
            if cutlass.const_expr(boundary):
                while pos >= end and seq + 1 < cu.shape[0] - 1:
                    seq = seq + 1
                    start = min(max(index_type(cu[seq]), 0), x.shape[0])
                    end = min(max(index_type(cu[seq + 1]), start), x.shape[0])
                if cutlass.const_expr(j < rows):
                    ends[j] = cutlass.Int32(end)
                if pos - start < width - 1:
                    for k in cutlass.range_constexpr(width):
                        state_row = pos - start + k
                        if state_row < width - 1:
                            for v in cutlass.range_constexpr(vector):
                                value = cutlass.Float32(0)
                                if cutlass.const_expr(initial is not None):
                                    if state_row >= 0:
                                        value = initial[cutlass.Int64(seq), cutlass.Int64(state_row), cutlass.Int64(h0) + col + v].to(cutlass.Float32)
                                window[(j + k) % width, v] = value
            grad = cute.make_rmem_tensor((vector,), cutlass.Float32)
            grad.store(raw_dy.load().to(cutlass.Float32))
            if cutlass.const_expr(silu):
                pre = cute.make_rmem_tensor((vector,), cutlass.Float32)
                pre.store(biases.load())
                for k in cutlass.range_constexpr(width):
                    if cutlass.const_expr(paired):
                        for v in cutlass.range_constexpr(0, vector, 2):
                            p0, p1 = cute.arch.fma_packed_f32x2(
                                (window[(j + k) % width, v], window[(j + k) % width, v + 1]),
                                (weights[k, v], weights[k, v + 1]),
                                (pre[v], pre[v + 1]),
                                ftz=False,
                            )
                            pre[v], pre[v + 1] = p0, p1
                    else:
                        for v in cutlass.range_constexpr(vector):
                            pre[v] = cute.math.fma(window[(j + k) % width, v], weights[k, v], pre[v])
                for v in cutlass.range_constexpr(vector):
                    if cutlass.const_expr(x.element_type == cutlass.Float32):
                        grad[v] = _compensated_silu_gradient_fp32(grad[v], pre[v])
                    else:
                        sigmoid = cute.math.div(1.0, 1.0 + cute.exp(-pre[v], fastmath=True), approx=True, ftz=True)
                        grad[v] = grad[v] * sigmoid * (1.0 + pre[v] * (1.0 - sigmoid))
            if cutlass.const_expr(j < rows):
                for k in cutlass.range_constexpr(width):
                    if cutlass.const_expr(paired):
                        for v in cutlass.range_constexpr(0, vector, 2):
                            p0, p1 = cute.arch.fma_packed_f32x2(
                                (window[(j + k) % width, v], window[(j + k) % width, v + 1]), (grad[v], grad[v + 1]), (accum[k, v], accum[k, v + 1]), ftz=False
                            )
                            accum[k, v], accum[k, v + 1] = p0, p1
                    else:
                        for v in cutlass.range_constexpr(vector):
                            if cutlass.const_expr(x.element_type == cutlass.Float32):
                                if cutlass.const_expr(finite_parameters):
                                    accum[k, v], accum[k + width + 1, v] = _compensated_product_finite(
                                        window[(j + k) % width, v], grad[v], accum[k, v], accum[k + width + 1, v]
                                    )
                                else:
                                    accum[k, v], accum[k + width + 1, v] = _compensated_product(
                                        window[(j + k) % width, v], grad[v], accum[k, v], accum[k + width + 1, v]
                                    )
                            else:
                                accum[k, v] = cute.math.fma(window[(j + k) % width, v], grad[v], accum[k, v])
                for v in cutlass.range_constexpr(vector):
                    if cutlass.const_expr(x.element_type == cutlass.Float32):
                        if cutlass.const_expr(finite_parameters):
                            accum[width, v], accum[2 * width + 1, v] = _compensated_sum_finite(grad[v], accum[width, v], accum[2 * width + 1, v])
                        else:
                            accum[width, v], accum[2 * width + 1, v] = _compensated_sum(grad[v], accum[width, v], accum[2 * width + 1, v])
                    else:
                        accum[width, v] = accum[width, v] + grad[v]
                if cutlass.const_expr(boundary and initial is not None):
                    if pos - start < width - 1:
                        for v in cutlass.range_constexpr(vector):
                            dz[cutlass.Int64(pos), h0 + col + v] = grad[v]
            for r in cutlass.range_constexpr(max(0, j - width + 1), min(rows, j + 1)):
                valid_grad = cutlass.Boolean(True)
                if cutlass.const_expr(boundary):
                    valid_grad = pos < ends[r]
                if valid_grad:
                    if cutlass.const_expr(paired):
                        for v in cutlass.range_constexpr(0, vector, 2):
                            p0, p1 = cute.arch.fma_packed_f32x2(
                                (grad[v], grad[v + 1]),
                                (weights[width - 1 - j + r, v], weights[width - 1 - j + r, v + 1]),
                                (pending[r, v], pending[r, v + 1]),
                                ftz=False,
                            )
                            pending[r, v], pending[r, v + 1] = p0, p1
                    else:
                        for v in cutlass.range_constexpr(vector):
                            pending[r, v] = cute.math.fma(grad[v], weights[width - 1 - j + r, v], pending[r, v])
        # A row drains after its last possible future gradient, even if that
        # future position is outside the pack or in another sequence.
        if cutlass.const_expr(j >= width - 1):
            r = j - width + 1
            output_pos = token_start + rbase + r
            valid_output = h0 + col < x.shape[1]
            if cutlass.const_expr(boundary):
                valid_output = valid_output and output_pos < x.shape[0]
            if valid_output:
                out = cute.make_rmem_tensor((vector,), dx.element_type)
                for v in cutlass.range_constexpr(vector):
                    value = pending[r, v]
                    if cutlass.const_expr(boundary and dfinal is not None):
                        final_row = output_pos - ends[r] + width - 1
                        if final_row >= 0 and final_row < width - 1:
                            output_seq = _sequence(cu, output_pos, index_type)
                            value = value + dfinal[cutlass.Int64(output_seq), cutlass.Int64(final_row), cutlass.Int64(h0) + col + v].to(cutlass.Float32)
                    out[v] = value.to(dx.element_type)
                target = cute.make_tensor((dx.iterator + cutlass.Int64(output_pos) * dx.stride[0] + h0 + col).align(16), cute.make_layout((vector,)))
                atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), dx.element_type, num_bits_per_copy=128)
                cute.copy(atom, out, target)


@cute.kernel
def packed_backward_tiled(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    dy: cute.Tensor,
    dfinal: cute.Tensor,
    dz: cute.Tensor,
    dx: cute.Tensor,
    partials: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
    threads: cutlass.Constexpr = 128,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    """Reuse each input tile for dX and fixed-order dW/db partials.

    Each block owns one split/channel tile. Row groups accumulate independently
    and reduce in a fixed order. Only prefix dz values needed by dInitial are
    materialized; the remaining activation gradients stay in registers.
    """
    _validate_offsets(cu, x.shape[0])
    vector = 128 // x.element_type.width
    tpr = th // vector
    groups = threads // tpr
    rows = ts // groups
    tid = cute.arch.thread_idx()[0]
    blocks_h = cute.ceil_div(x.shape[1], th)
    split = cute.arch.block_idx()[0] // blocks_h
    h0 = index_type(cute.arch.block_idx()[0] % blocks_h) * th
    group = tid // tpr
    col = (tid % tpr) * vector
    rbase = group * rows
    allocator = SmemAllocator()
    if cutlass.const_expr(width == 4 and silu and x.element_type.width == 16 and _use_wide_silu_backward()):
        sx_size = (ts + 2 * (width - 1)) * th
        sdy_size = (ts + width - 1) * th
        reduction_size = groups * (width + 1) * th
        stage_bytes = (sx_size * x.element_type.width + sdy_size * dy.element_type.width) // 8
        reduction_bytes = reduction_size * partials.element_type.width // 8
        shared = allocator.allocate_tensor(cutlass.Int64, cute.make_layout((cute.ceil_div(max(stage_bytes, reduction_bytes), 8),)), byte_alignment=16)
        sx = cute.make_tensor(cute.recast_ptr(shared.iterator, dtype=x.element_type), cute.make_layout((sx_size,)))
        sdy = cute.make_tensor(cute.recast_ptr(shared.iterator, dtype=dy.element_type) + sx_size, cute.make_layout((sdy_size,)))
    else:
        sx = allocator.allocate_tensor(x.element_type, cute.make_layout(((ts + 2 * (width - 1)) * th,)), byte_alignment=16)
        sdy = allocator.allocate_tensor(dy.element_type, cute.make_layout(((ts + width - 1) * th,)), byte_alignment=16)
    bounds = allocator.allocate_tensor(index_type, cute.make_layout((2,)), byte_alignment=16)
    if cutlass.const_expr(width == 4 and silu and x.element_type.width == 16 and _use_wide_silu_backward()):
        # The tile loop's final barrier ends every read of sx/sdy before
        # these addresses become the fixed-order parameter reduction.
        reduction = cute.make_tensor(cute.recast_ptr(shared.iterator, dtype=partials.element_type), cute.make_layout((reduction_size,)))
    else:
        reduction = allocator.allocate_tensor(partials.element_type, cute.make_layout((groups * (width + 1) * th,)), byte_alignment=16)
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    biases = cute.make_rmem_tensor((vector,), cutlass.Float32)
    accum_size = 2 * (width + 1) if cutlass.const_expr(x.element_type == cutlass.Float32) else width + 1
    accum = cute.make_rmem_tensor(cute.make_layout((accum_size, vector), stride=(vector, 1)), cutlass.Float32)
    weights.fill(0)
    biases.fill(0)
    accum.fill(0)
    if h0 + col < x.shape[1]:
        for k in cutlass.range_constexpr(width):
            for v in cutlass.range_constexpr(vector):
                weights[k, v] = w[cutlass.Int64(h0) + col + v, k].to(cutlass.Float32)
        if cutlass.const_expr(bias is not None):
            raw_bias = _nwh_load_vector(bias, h0 + col, vector)
            biases.store(raw_bias.load().to(cutlass.Float32))
    eligible_bounds = cutlass.const_expr(x.element_type == cutlass.Float32 and width == 8 and not silu and initial is None)
    if cutlass.const_expr(eligible_bounds):
        # At most 2**20 terms, each input/gradient bounded by 2**40.
        # Once an unsafe tile is seen, use the checked path for this CTA.
        finite_so_far = x.shape[0] <= 1048576
    for tile in cutlass.range(split, cute.ceil_div(x.shape[0], ts), partials.shape[0]):
        token_start = index_type(tile) * ts
        for load_row in cutlass.range_constexpr(cute.ceil_div(ts + 2 * (width - 1), groups)):
            row = group + load_row * groups
            pos = token_start + row - width + 1
            if row < ts + 2 * (width - 1):
                _nwh_async_vector(
                    x, sx, cutlass.Int64(pos) * x.stride[0] + h0 + col, row * th + col, pos >= 0 and pos < x.shape[0] and h0 + col < x.shape[1], vector
                )
        for load_grad in cutlass.range_constexpr(cute.ceil_div(ts + width - 1, groups)):
            row = group + load_grad * groups
            pos = token_start + row
            if row < ts + width - 1:
                _nwh_async_vector(dy, sdy, cutlass.Int64(pos) * dy.stride[0] + h0 + col, row * th + col, pos < x.shape[0] and h0 + col < x.shape[1], vector)
        cute.arch.cp_async_commit_group()
        if tid == 0:
            tile_seq = _sequence(cu, token_start, index_type)
            tile_start = min(max(index_type(cu[tile_seq]), 0), x.shape[0])
            tile_end = min(max(index_type(cu[tile_seq + 1]), tile_start), x.shape[0])
            bounds[0] = tile_start
            bounds[1] = tile_end
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        start, end = bounds[0], bounds[1]
        if cutlass.const_expr(eligible_bounds):
            if finite_so_far:
                local_bounded = cutlass.Boolean(True)
                for offset in cutlass.range_constexpr(cute.ceil_div((ts + 2 * (width - 1)) * th, threads)):
                    index = tid + offset * threads
                    if index < (ts + 2 * (width - 1)) * th:
                        local_bounded = local_bounded and cute.abs(sx[index]) <= cutlass.Float32(1099511627776.0)
                for offset in cutlass.range_constexpr(cute.ceil_div((ts + width - 1) * th, threads)):
                    index = tid + offset * threads
                    if index < (ts + width - 1) * th:
                        local_bounded = local_bounded and cute.abs(sdy[index]) <= cutlass.Float32(1099511627776.0)
                finite_so_far = _cta_all_bounded(local_bounded)
            if finite_so_far:
                if token_start - width + 1 >= start and token_start + ts + width - 1 <= end:
                    _packed_backward_rows(
                        x,
                        sx,
                        sdy,
                        weights,
                        biases,
                        accum,
                        cu,
                        initial,
                        dfinal,
                        dz,
                        dx,
                        token_start,
                        h0,
                        col,
                        rbase,
                        width,
                        silu,
                        rows,
                        th,
                        False,
                        index_type,
                        True,
                    )
                else:
                    _packed_backward_rows(
                        x,
                        sx,
                        sdy,
                        weights,
                        biases,
                        accum,
                        cu,
                        initial,
                        dfinal,
                        dz,
                        dx,
                        token_start,
                        h0,
                        col,
                        rbase,
                        width,
                        silu,
                        rows,
                        th,
                        True,
                        index_type,
                        True,
                    )
            else:
                if token_start - width + 1 >= start and token_start + ts + width - 1 <= end:
                    _packed_backward_rows(
                        x, sx, sdy, weights, biases, accum, cu, initial, dfinal, dz, dx, token_start, h0, col, rbase, width, silu, rows, th, False, index_type
                    )
                else:
                    _packed_backward_rows(
                        x, sx, sdy, weights, biases, accum, cu, initial, dfinal, dz, dx, token_start, h0, col, rbase, width, silu, rows, th, True, index_type
                    )
        else:
            if token_start - width + 1 >= start and token_start + ts + width - 1 <= end:
                _packed_backward_rows(
                    x, sx, sdy, weights, biases, accum, cu, initial, dfinal, dz, dx, token_start, h0, col, rbase, width, silu, rows, th, False, index_type
                )
            else:
                _packed_backward_rows(
                    x, sx, sdy, weights, biases, accum, cu, initial, dfinal, dz, dx, token_start, h0, col, rbase, width, silu, rows, th, True, index_type
                )
        # All consumers finish before the next tile reuses shared storage.
        cute.arch.barrier()
    for k in cutlass.range_constexpr(width + 1):
        for v in cutlass.range_constexpr(vector):
            if cutlass.const_expr(x.element_type == cutlass.Float32):
                reduction[(group * (width + 1) + k) * th + col + v] = accum[k, v].to(cutlass.Float64) + accum[k + width + 1, v].to(cutlass.Float64)
            else:
                reduction[(group * (width + 1) + k) * th + col + v] = accum[k, v]
    cute.arch.barrier()
    if group == 0 and h0 + col < x.shape[1]:
        for k in cutlass.range_constexpr(width + 1):
            for v in cutlass.range_constexpr(vector):
                value = partials.element_type(0)
                for other in cutlass.range_constexpr(groups):
                    value = value + reduction[(other * (width + 1) + k) * th + col + v]
                partials[cutlass.Int64(split), k, h0 + col + v] = value


packed_backward_tiled.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _packed_backward_prefetch(
    x: cute.Tensor,
    dy: cute.Tensor,
    sx: cute.Tensor,
    sdy: cute.Tensor,
    token_start,
    h0,
    col,
    group,
    width: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
    groups: cutlass.Constexpr,
    vector: cutlass.Constexpr,
):
    for load_row in cutlass.range_constexpr(cute.ceil_div(ts + 2 * (width - 1), groups)):
        row = group + load_row * groups
        pos = token_start + row - width + 1
        if row < ts + 2 * (width - 1):
            _nwh_async_vector(
                x, sx, cutlass.Int64(pos) * x.stride[0] + h0 + col, row * th + col, pos >= 0 and pos < x.shape[0] and h0 + col < x.shape[1], vector
            )
    for load_grad in cutlass.range_constexpr(cute.ceil_div(ts + width - 1, groups)):
        row = group + load_grad * groups
        pos = token_start + row
        if row < ts + width - 1:
            _nwh_async_vector(dy, sdy, cutlass.Int64(pos) * dy.stride[0] + h0 + col, row * th + col, pos < x.shape[0] and h0 + col < x.shape[1], vector)
    cute.arch.cp_async_commit_group()


@cute.kernel
def packed_backward_pipelined(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    dy: cute.Tensor,
    dfinal: cute.Tensor,
    dz: cute.Tensor,
    dx: cute.Tensor,
    partials: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    ts: cutlass.Constexpr,
    th: cutlass.Constexpr,
    threads: cutlass.Constexpr = 128,
    index_type: cutlass.Constexpr = cutlass.Int64,
):
    """Reuse each input tile for dX and fixed-order dW/db partials.

    Each block owns one split/channel tile. Row groups accumulate independently
    and reduce in a fixed order. Only prefix dz values needed by dInitial are
    materialized; the remaining activation gradients stay in registers.
    """
    _validate_offsets(cu, x.shape[0])
    vector = 128 // x.element_type.width
    tpr = th // vector
    groups = threads // tpr
    rows = ts // groups
    tid = cute.arch.thread_idx()[0]
    blocks_h = cute.ceil_div(x.shape[1], th)
    split = cute.arch.block_idx()[0] // blocks_h
    h0 = index_type(cute.arch.block_idx()[0] % blocks_h) * th
    group = tid // tpr
    col = (tid % tpr) * vector
    rbase = group * rows
    allocator = SmemAllocator()
    sx_span = (ts + 2 * (width - 1)) * th
    sdy_span = (ts + width - 1) * th
    sx_storage = allocator.allocate_tensor(x.element_type, cute.make_layout((2 * sx_span,)), byte_alignment=16)
    sdy_storage = allocator.allocate_tensor(dy.element_type, cute.make_layout((2 * sdy_span,)), byte_alignment=16)
    bounds = allocator.allocate_tensor(index_type, cute.make_layout((2,)), byte_alignment=16)
    reduction = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((groups * (width + 1) * th,)), byte_alignment=16)
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    biases = cute.make_rmem_tensor((vector,), cutlass.Float32)
    accum = cute.make_rmem_tensor(cute.make_layout((width + 1, vector), stride=(vector, 1)), cutlass.Float32)
    weights.fill(0)
    biases.fill(0)
    accum.fill(0)
    if h0 + col < x.shape[1]:
        for k in cutlass.range_constexpr(width):
            for v in cutlass.range_constexpr(vector):
                weights[k, v] = w[cutlass.Int64(h0) + col + v, k].to(cutlass.Float32)
        if cutlass.const_expr(bias is not None):
            raw_bias = _nwh_load_vector(bias, h0 + col, vector)
            biases.store(raw_bias.load().to(cutlass.Float32))
    stage = cutlass.Int32(0)
    sx_first = cute.make_tensor(sx_storage.iterator, cute.make_layout((sx_span,)))
    sdy_first = cute.make_tensor(sdy_storage.iterator, cute.make_layout((sdy_span,)))
    if split < cute.ceil_div(x.shape[0], ts):
        _packed_backward_prefetch(x, dy, sx_first, sdy_first, index_type(split) * ts, h0, col, group, width, ts, th, groups, vector)
    for tile in cutlass.range(split, cute.ceil_div(x.shape[0], ts), partials.shape[0]):
        sx = cute.make_tensor((sx_storage.iterator + stage * sx_span).align(16), cute.make_layout((sx_span,)))
        sdy = cute.make_tensor((sdy_storage.iterator + stage * sdy_span).align(16), cute.make_layout((sdy_span,)))
        token_start = index_type(tile) * ts
        if tid == 0:
            tile_seq = _sequence(cu, token_start, index_type)
            tile_start = min(max(index_type(cu[tile_seq]), 0), x.shape[0])
            tile_end = min(max(index_type(cu[tile_seq + 1]), tile_start), x.shape[0])
            bounds[0] = tile_start
            bounds[1] = tile_end
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        if tile + partials.shape[0] < cute.ceil_div(x.shape[0], ts):
            next_stage = cutlass.Int32(1) - stage
            sx_next = cute.make_tensor((sx_storage.iterator + next_stage * sx_span).align(16), cute.make_layout((sx_span,)))
            sdy_next = cute.make_tensor((sdy_storage.iterator + next_stage * sdy_span).align(16), cute.make_layout((sdy_span,)))
            _packed_backward_prefetch(x, dy, sx_next, sdy_next, index_type(tile + partials.shape[0]) * ts, h0, col, group, width, ts, th, groups, vector)
        start, end = bounds[0], bounds[1]
        if token_start - width + 1 >= start and token_start + ts + width - 1 <= end:
            _packed_backward_rows(
                x, sx, sdy, weights, biases, accum, cu, initial, dfinal, dz, dx, token_start, h0, col, rbase, width, silu, rows, th, False, index_type
            )
        else:
            _packed_backward_rows(
                x, sx, sdy, weights, biases, accum, cu, initial, dfinal, dz, dx, token_start, h0, col, rbase, width, silu, rows, th, True, index_type
            )
        # Finish reading this stage before a later prefetch can overwrite it.
        cute.arch.barrier()
        stage = cutlass.Int32(1) - stage
    for k in cutlass.range_constexpr(width + 1):
        for v in cutlass.range_constexpr(vector):
            reduction[(group * (width + 1) + k) * th + col + v] = accum[k, v]
    cute.arch.barrier()
    if group == 0 and h0 + col < x.shape[1]:
        for k in cutlass.range_constexpr(width + 1):
            for v in cutlass.range_constexpr(vector):
                value = cutlass.Float32(0)
                for other in cutlass.range_constexpr(groups):
                    value = value + reduction[(other * (width + 1) + k) * th + col + v]
                partials[cutlass.Int64(split), k, h0 + col + v] = value


packed_backward_pipelined.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _use_pipelined_backward():
    """Select the double-buffered schedule on SM100 and SM103."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return arch.major == 10 and arch.minor in (0, 3)


def _use_wide_silu_backward():
    """Select the wide-channel SiLU schedule on SM89."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return arch.major == 8 and arch.minor == 9


@cute.jit
def launch_packed_backward(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    cu: cute.Tensor,
    initial: cute.Tensor,
    dy: cute.Tensor,
    dfinal: cute.Tensor,
    dz: cute.Tensor,
    dx: cute.Tensor,
    dw: cute.Tensor,
    db: cute.Tensor,
    dinitial: cute.Tensor,
    partials: cute.Tensor,
    stream: cuda.CUstream,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    fast: cutlass.Constexpr = False,
    index32: cutlass.Constexpr = False,
):
    index_type = cutlass.Int32 if cutlass.const_expr(index32) else cutlass.Int64
    if cutlass.const_expr(fast):
        ts = 128 if cutlass.const_expr(width == 4 and x.element_type.width == 16 and not silu) else 64
        threads = 64 if cutlass.const_expr(width == 4 and x.element_type.width == 16) else 128
        wide = cutlass.const_expr(width == 4 and silu and x.element_type.width == 16 and _use_wide_silu_backward())
        th = 256 if cutlass.const_expr(wide) else 64
        if cutlass.const_expr(wide):
            threads = 256
        if cutlass.const_expr(width == 8 and x.element_type.width == 16 and not silu and _use_pipelined_backward()):
            packed_backward_pipelined(x, w, bias, cu, initial, dy, dfinal, dz, dx, partials, width, silu, ts, th, threads, index_type).launch(
                grid=(cute.ceil_div(x.shape[1], th) * partials.shape[0], 1, 1), block=(threads, 1, 1), stream=stream
            )
        else:
            packed_backward_tiled(x, w, bias, cu, initial, dy, dfinal, dz, dx, partials, width, silu, ts, th, threads, index_type).launch(
                grid=(cute.ceil_div(x.shape[1], th) * partials.shape[0], 1, 1), block=(threads, 1, 1), stream=stream
            )
    elif x.shape[0] > 0:
        packed_dz(x, w, bias, cu, initial, dy, dz, width, silu).launch(
            grid=(cute.ceil_div(x.shape[1], 128) * x.shape[0], 1, 1), block=(128, 1, 1), stream=stream
        )
        packed_dx(w, cu, dz, dfinal, dx, width).launch(grid=(cute.ceil_div(x.shape[1], 128) * x.shape[0], 1, 1), block=(128, 1, 1), stream=stream)
    if cutlass.const_expr(dinitial is not None):
        packed_dinitial(w, cu, dz, dfinal, dinitial, width).launch(
            grid=(cute.ceil_div(x.shape[1], 128) * (cu.shape[0] - 1), 1, 1), block=(128, 1, 1), stream=stream
        )
    if cutlass.const_expr(not fast):
        packed_parameter_partials(x, cu, initial, dz, partials, width).launch(
            grid=(cute.ceil_div(x.shape[1], 128) * partials.shape[0], 1, 1), block=(128, 1, 1), stream=stream
        )
    if cutlass.const_expr(dw.element_type == cutlass.Float32):
        taps = width + (1 if cutlass.const_expr(db is not None) else 0)
        packed_parameter_reduce_fp32(partials, dw, db, width).launch(grid=(cute.ceil_div(x.shape[1], 128) * taps, 1, 1), block=(128, 1, 1), stream=stream)
    else:
        packed_parameter_reduce(partials, dw, db, width).launch(grid=(cute.ceil_div(x.shape[1], 128), 1, 1), block=(128, 1, 1), stream=stream)


@cute.kernel
def packed_state_gradient(cu: cute.Tensor, dfinal: cute.Tensor, dx: cute.Tensor, dinitial: cute.Tensor, width: cutlass.Constexpr):
    """Differentiate the raw history copy without evaluating convolution."""
    _validate_offsets(cu, dx.shape[0])
    blocks_h = cute.ceil_div(dx.shape[1], 128)
    seq = cutlass.Int64(cute.arch.block_idx()[0] // blocks_h)
    h = cutlass.Int64(cute.arch.block_idx()[0] % blocks_h) * 128 + cute.arch.thread_idx()[0]
    if h < dx.shape[1]:
        start = min(max(cutlass.Int64(cu[seq]), 0), dx.shape[0])
        end = min(max(cutlass.Int64(cu[seq + 1]), start), dx.shape[0])
        for j in cutlass.range_constexpr(width - 1):
            pos = end - width + 1 + j
            if pos >= start and pos < end:
                dx[pos, h] = dfinal[cutlass.Int64(seq), j, h]
            if cutlass.const_expr(dinitial is not None):
                row = j - (end - start)
                value = cutlass.Float32(0)
                if row >= 0 and row < width - 1:
                    value = dfinal[cutlass.Int64(seq), row, h].to(cutlass.Float32)
                dinitial[cutlass.Int64(seq), j, h] = value.to(dinitial.element_type)


packed_state_gradient.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch_packed_state_backward(cu: cute.Tensor, dfinal: cute.Tensor, dx: cute.Tensor, dinitial: cute.Tensor, stream: cuda.CUstream, width: cutlass.Constexpr):
    packed_state_gradient(cu, dfinal, dx, dinitial, width).launch(
        grid=(cute.ceil_div(dx.shape[1], 128) * (cu.shape[0] - 1), 1, 1), block=(128, 1, 1), stream=stream
    )
