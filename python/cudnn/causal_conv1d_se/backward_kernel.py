# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct backward schedules with FP32/FP64 parameter-gradient accumulation."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.utils import SmemAllocator


def _uses_packed_nwh():
    """Return whether the target supports the packed NWH arithmetic route."""
    from cutlass.cutlass_dsl import CuTeDSL

    return int(CuTeDSL._get_dsl().get_arch_enum().major) >= 10


def _nwh_seqpar(dtype, width, silu, shape):
    """Mirror the SE arithmetic route, including its shared-memory admission."""
    from cutlass.utils import get_smem_capacity_in_bytes
    from cutlass.cutlass_dsl import CuTeDSL

    element_bytes = dtype.width // 8
    if dtype == cutlass.Float64 or int(CuTeDSL._get_dsl().get_arch_enum().major) < 8:
        return False
    cap = (16 if dtype == cutlass.Float32 else 32) if silu else 128
    if width > cap or shape[2] % (16 // element_bytes) or shape[0] > 65535:
        return False
    tile_s = 128 if width > 65 else 64
    threads = 256 if width > 65 else 128
    if (shape[1] + tile_s - 1) // tile_s > 65535:
        return False
    split = width > 8
    smem = (
        2 * (2 * tile_s + 3 * (width - 1)) * 64 * element_bytes
        + (width + 1) * 64 * 4
        + (1 if split else width + 1) * (threads // 32) * 64 * 4
        + (width * 64 * 4 if split else 0)
        + ((tile_s + width - 1) * 64 * 4 if split and silu else 0)
    )
    return smem <= get_smem_capacity_in_bytes()


@cute.jit
def _sigmoid(value):
    """Evaluate sigmoid with the backend-compatible reciprocal policy."""
    denominator = 1.0 + cute.exp(-value, fastmath=True)
    if cutlass.const_expr(value.dtype == cutlass.Float64):
        return 1.0 / denominator
    else:
        return cute.math.rcp(denominator, approx=True, ftz=True)


@cute.kernel
def causal_conv1d_dz(x: cute.Tensor, w: cute.Tensor, b: cute.Tensor, dy: cute.Tensor, dz: cute.Tensor, width: cutlass.Constexpr, nwh: cutlass.Constexpr):
    """Recompute convolution and apply the SiLU derivative to the incoming gradient."""
    index = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
    D = x.shape[1]
    L = x.shape[2]
    if cutlass.const_expr(nwh):
        D, L = x.shape[2], x.shape[1]
    if index < cute.size(x):
        batch = index // (D * L)
        channel = index // L % D
        t = index % L
        if cutlass.const_expr(nwh):
            channel = index % D
            t = index // D % L
        compute = dz.element_type
        value = b[channel].to(compute)
        for tap in cutlass.range_constexpr(width):
            if cutlass.const_expr(nwh and _nwh_seqpar(x.element_type, width, True, x.shape)):
                # The wide seqpar path visits weights from oldest to newest.
                pos = t - width + 1 + tap
                sample = compute(0)
                if pos >= 0:
                    sample = x[batch, pos, channel].to(compute)
                value += sample * w[tap, channel].to(compute)
            else:
                pos = t - tap
                if pos >= 0:
                    if cutlass.const_expr(nwh):
                        value += x[batch, pos, channel].to(compute) * w[width - 1 - tap, channel].to(compute)
                    else:
                        value += x[batch, channel, pos].to(compute) * w[channel, width - 1 - tap].to(compute)
        if cutlass.const_expr(nwh and x.element_type.width == 16 and _nwh_seqpar(x.element_type, width, True, x.shape)):
            sig = 0.5 + 0.5 * cute.tanh(0.5 * value, fastmath=True)
        else:
            sig = _sigmoid(value)
        if cutlass.const_expr(nwh):
            dz[batch, t, channel] = dy[batch, t, channel].to(compute) * (sig * (1.0 + value * (1.0 - sig)))
        else:
            dz[batch, channel, t] = dy[batch, channel, t].to(compute) * (sig * (1.0 + value * (1.0 - sig)))


causal_conv1d_dz.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def causal_conv1d_zero_grad(dw: cute.Tensor, db: cute.Tensor):
    """Clear parameter-gradient buffers before an overwrite-mode backward launch."""
    i = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cute.arch.thread_idx()[0]
    flat = cute.make_tensor(dw.iterator, cute.make_layout((cute.size(dw),)))
    if i < cute.size(dw):
        flat[i] = dw.element_type(0)
    if i < cute.size(db):
        db[i] = db.element_type(0)


causal_conv1d_zero_grad.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def causal_conv1d_bwd_nwc(
    x: cute.Tensor,
    w: cute.Tensor,
    dz: cute.Tensor,
    dx: cute.Tensor,
    dw: cute.Tensor,
    db: cute.Tensor,
    width: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """Compute NWH data and parameter gradients from a materialized activation gradient."""
    channel = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
    start = cute.arch.block_idx()[1] * rows
    batch = cute.arch.block_idx()[2]
    compute = dw.element_type
    dz_reg = cute.make_rmem_tensor((rows + width - 1,), compute)
    x_reg = cute.make_rmem_tensor((rows,), compute)
    dx_reg = cute.make_rmem_tensor((rows,), compute)
    dz_reg.fill(0)
    x_reg.fill(0)
    dx_reg.fill(0)
    if channel < x.shape[2]:
        db_value = compute(0)
        for i in cutlass.range_constexpr(rows + width - 1):
            if start + i < x.shape[1]:
                dz_reg[i] = dz[batch, start + i, channel].to(compute)
        for i in cutlass.range_constexpr(rows):
            if start + i < x.shape[1]:
                x_reg[i] = x[batch, start + i, channel].to(compute)
            db_value += dz_reg[i]
        cute.arch.atomic_add(db.iterator + channel, db_value)
        for tap in cutlass.range_constexpr(width):
            w_value = w[width - 1 - tap, channel].to(compute)
            dw_value = compute(0)
            for i in cutlass.range_constexpr(rows):
                dx_reg[i] += dz_reg[i + tap] * w_value
                dw_value += x_reg[i] * dz_reg[i + tap]
            cute.arch.atomic_add(dw.iterator + cutlass.Int64(width - 1 - tap) * x.shape[2] + channel, dw_value)
        for i in cutlass.range_constexpr(rows):
            if start + i < x.shape[1]:
                dx[batch, start + i, channel] = dx_reg[i].to(dx.element_type)


causal_conv1d_bwd_nwc.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _nwh_pair_masks(width, element_bits, silu):
    """Natural-order lanes in the cuDNN 9.29 / NVRTC 13.5–13.6 backward tile.

    Masks are (full stage 0, masked stage 0, full stage 1, masked stage 1).
    NVRTC contracts the first two products differently across template paths.
    The lane bit is (position % rows_per_thread) * vector + channel % vector.
    This is a bitwise compatibility rule for the pinned backend, not a choice
    of mathematical convolution order. Revalidate when its compiler changes.
    """
    if element_bits == 16:
        # Half inputs with identity have exactly representable FP32 products.
        masked = ((0xFFFFFFFF << (8 * (width - 1))) & 0xFFFFFFFF) if silu and width <= 4 else 0
        return (0, masked, 0, masked)
    if _uses_packed_nwh():
        masked = 0 if silu else 0xF0000000
        return (0, masked, 0, masked)
    # FP32 masked-tile contraction in NVRTC 13.5/13.6. In particular, the
    # second double-buffer stage has a different K3 profile from the first.
    if width == 2 and not silu:
        return (0, 0xF00000F0, 0, 0xF00000F0)
    if width == 3:
        if silu:
            return (0, 0xFFFFFF00, 0, 0x77777700)
        return (0, 0xF0000700, 0, 0xF0000F00)
    if silu:
        masked = 0 if width == 2 else ((0xFFFFFFFF << (4 * (width - 1))) & 0xFFFFFFFF)
        return (0, masked, 0, masked)
    masked = (0xF0000000, 0xF0000F00, 0xF000F000, 0xF00F0000, 0xF0F00000, 0xFF000000, 0xF0000000)[width - 2]
    full_first = (0xF << (4 * (width - 1))) if width >= 7 else 0
    return (full_first, masked, 0, masked)


@cute.jit
def _nwh_first_pair(
    g0, w0, g1, w1, channel, position, shape: cutlass.Constexpr, width: cutlass.Constexpr, element_bits: cutlass.Constexpr, silu: cutlass.Constexpr
):
    """Contract the first two gradient products in the selected backend rounding order."""
    masks: cutlass.Constexpr = _nwh_pair_masks(width, element_bits, silu)
    # CuTe needs a prebound result for the nested dynamic region; removing this
    # initialization raises UnboundLocalError while tracing the masked-tile path.
    value = cutlass.Float32(0)
    if cutlass.const_expr(not any(masks) or (shape[1] % 64 == 0 and shape[2] % 64 == 0 and masks[0] == masks[2] == 0)):
        value = cute.math.fma(g0, w0, g1 * w1)
    else:
        mask = cutlass.Uint32(masks[0])
        if cutlass.const_expr(masks[0] != masks[2]):
            if position // 64 % 2 != 0:
                mask = cutlass.Uint32(masks[2])
        if cutlass.const_expr(shape[1] % 64 != 0 or shape[2] % 64 != 0):
            if (channel // 64 + 1) * 64 > shape[2] or (position // 64 + 1) * 64 > shape[1]:
                mask = cutlass.Uint32(masks[1])
                if cutlass.const_expr(masks[1] != masks[3]):
                    if position // 64 % 2 != 0:
                        mask = cutlass.Uint32(masks[3])
        bit = (position % (element_bits // 4)) * (128 // element_bits) + channel % (128 // element_bits)
        if (mask >> bit) & 1:
            value = cute.math.fma(g1, w1, g0 * w0)
        else:
            value = cute.math.fma(g0, w0, g1 * w1)
    return value


def _nwh_center_only(dtype, width, silu, shape):
    """Select the short Hopper identity path that loads only the central input window."""
    from cutlass.cutlass_dsl import CuTeDSL

    return CuTeDSL._get_dsl().get_arch_enum().major == 9 and dtype.width == 16 and width == 7 and not silu and shape[1] <= 256 and shape[2] >= 768


@cute.kernel
def causal_conv1d_bwd_nwc_fused(
    x: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dw: cute.Tensor,
    db: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """Recompute SiLU in registers and amortize parameter atomics across sequence chunks."""
    channel = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
    first = cute.arch.block_idx()[1] * rows
    batch = cute.arch.block_idx()[2]
    compute = dw.element_type
    weights = cute.make_rmem_tensor((width,), compute)
    dw_reg = cute.make_rmem_tensor((width,), compute)
    window = cute.make_rmem_tensor((8 + 2 * width - 2,), compute)
    grads = cute.make_rmem_tensor((8 + width - 1,), compute)
    dw_reg.fill(0)
    db_value = compute(0)
    if channel < x.shape[2]:
        bias_value = bias[channel].to(compute)
        for k in cutlass.range_constexpr(width):
            weights[k] = w[k, channel].to(compute)
        for chunk in cutlass.range(0, rows // 8, unroll=1):
            start = first + chunk * 8
            window.fill(0)
            grads.fill(0)
            for i in cutlass.range_constexpr(8 + 2 * width - 2):
                if cutlass.const_expr(not _nwh_center_only(x.element_type, width, silu, x.shape) or width - 1 <= i < width + 7):
                    t = start - width + 1 + i
                    if t >= 0 and t < x.shape[1]:
                        window[i] = x[batch, t, channel].to(compute)
            for i in cutlass.range_constexpr(8 + width - 1):
                if start + i < x.shape[1]:
                    grad = dy[batch, start + i, channel].to(compute)
                    if cutlass.const_expr(silu):
                        value = bias_value
                        for k in cutlass.range_constexpr(width):
                            if cutlass.const_expr(not _nwh_seqpar(x.element_type, width, silu, x.shape)):
                                value += window[i + width - 1 - k] * weights[width - 1 - k]
                            else:
                                value += window[i + k] * weights[k]
                        if cutlass.const_expr(x.element_type.width == 16 and _nwh_seqpar(x.element_type, width, silu, x.shape)):
                            sig = 0.5 + 0.5 * cute.tanh(0.5 * value, fastmath=True)
                        else:
                            sig = _sigmoid(value)
                        grad *= sig * (1.0 + value * (1.0 - sig))
                    grads[i] = grad
            for i in cutlass.range_constexpr(8):
                value = compute(0)
                if cutlass.const_expr(
                    (x.element_type == cutlass.Float32 or (x.element_type.width == 16 and not _uses_packed_nwh()))
                    and _nwh_seqpar(x.element_type, width, silu, x.shape)
                    and 1 < width <= 8
                ):
                    # NVRTC contracts the first two SE products with tap 1 rounded first.
                    value = _nwh_first_pair(
                        grads[i], weights[width - 1], grads[i + 1], weights[width - 2], channel, start + i, x.shape, width, x.element_type.width, silu
                    )
                    for k in cutlass.range_constexpr(2, width):
                        value = cute.math.fma(grads[i + k], weights[width - 1 - k], value)
                else:
                    for k in cutlass.range_constexpr(width):
                        value += grads[i + k] * weights[width - 1 - k]
                for k in cutlass.range_constexpr(width):
                    dw_reg[width - 1 - k] += window[i + width - 1] * grads[i + k]
                db_value += grads[i]
                if start + i < x.shape[1]:
                    dx[batch, start + i, channel] = value.to(dx.element_type)
        for k in cutlass.range_constexpr(width):
            cute.arch.atomic_add(dw.iterator + cutlass.Int64(k) * x.shape[2] + channel, dw_reg[k])
        cute.arch.atomic_add(db.iterator + channel, db_value)


causal_conv1d_bwd_nwc_fused.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _tail_load(x, batch, channel, tile, lane, compute: cutlass.Constexpr, scratch: cute.Tensor, halo: cutlass.Constexpr = 256):
    """Coalesce an unaligned NCW tile through warp-local shared memory, zeroing out-of-range positions."""
    values = cute.make_rmem_tensor((8,), compute)
    values.fill(0)
    warp = cute.arch.thread_idx()[0] // 32
    if tile >= 0:
        for i in cutlass.range_constexpr(8):
            pos = lane + i * 32
            value = compute(0)
            if pos >= 256 - cute.ceil_div(halo, 8) * 8 and tile * 256 + pos < x.shape[2]:
                value = x[batch, channel, tile * 256 + pos].to(compute)
            scratch[warp, pos ^ (pos >> 5)] = value
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(8):
            pos = lane * 8 + i
            values[i] = scratch[warp, pos ^ (pos >> 5)]
        cute.arch.sync_warp()
    return values


@cute.jit
def _tail_store(y, batch, channel, start, values, scratch: cute.Tensor):
    """Transpose register values through shared memory for coalesced masked NCW stores."""
    warp = cute.arch.thread_idx()[0] // 32
    lane = cute.arch.thread_idx()[0] % 32
    tile_start = start - lane * 8
    for i in cutlass.range_constexpr(8):
        pos = lane * 8 + i
        scratch[warp, pos ^ (pos >> 5)] = values[i]
    cute.arch.sync_warp()
    for i in cutlass.range_constexpr(8):
        pos = lane + i * 32
        if tile_start + pos < y.shape[2]:
            y[batch, channel, tile_start + pos] = scratch[warp, pos ^ (pos >> 5)].to(y.element_type)
    cute.arch.sync_warp()


@cute.jit
def _ncw_load(x, batch, channel, tile, lane, compute: cutlass.Constexpr, scratch, halo: cutlass.Constexpr = 256):
    """Load eight NCW values per lane using aligned vectors or the shared-memory tail path."""
    if cutlass.const_expr(x.shape[2] % 8 != 0):
        values = _tail_load(x, batch, channel, tile, lane, compute, scratch, halo)
    else:
        values = cute.make_rmem_tensor((8,), compute)
        values.fill(0)
        if tile >= 0 and lane >= 32 - cute.ceil_div(halo, 8):
            start = tile * 256 + lane * 8
            if cutlass.const_expr(x.shape[2] % 8 == 0):
                if start < x.shape[2]:
                    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), x.element_type, num_bits_per_copy=128)
                    raw = cute.make_rmem_tensor((8,), x.element_type)
                    offset = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(channel) * x.stride[1] + start
                    view = cute.make_tensor((x.iterator + offset).align(16), cute.make_layout((8,)))
                    cute.copy(atom, view, raw)
                    values.store(raw.load().to(compute))
            else:
                for i in cutlass.range_constexpr(8):
                    if start + i < x.shape[2]:
                        values[i] = x[batch, channel, start + i].to(compute)
        return values
    return values


@cute.jit
def _ncw_dz(x, w, weights, bias_value, dy, batch, channel, tile, lane, width: cutlass.Constexpr, silu: cutlass.Constexpr, compute: cutlass.Constexpr, scratch):
    """Load an NCW tile and compute its activation gradient while retaining original inputs."""
    current = _ncw_load(x, batch, channel, tile, lane, compute, scratch)
    original = cute.make_rmem_tensor((8,), compute)
    original.store(current.load())
    grad = _ncw_load(dy, batch, channel, tile, lane, compute, scratch)
    if cutlass.const_expr(silu):
        if tile * 256 < x.shape[2]:
            previous = _ncw_load(
                x,
                batch,
                channel,
                tile - 1,
                lane,
                compute,
                scratch,
                256 if x.shape[2] % 8 == 0 and x.shape[2] >= 1024 and not _is_sm86() else width - 1,
            )
            result = cute.make_rmem_tensor((8,), compute)
            result.fill(bias_value)
            for tap in cutlass.range_constexpr(width):
                if cutlass.const_expr(tap > 0):
                    shifted = cute.arch.shuffle_sync(current[7], (lane + 31) % 32)
                    halo = cute.arch.shuffle_sync(previous[7 - (tap - 1) % 8], 31 - (tap - 1) // 8)
                    for i in cutlass.range_constexpr(7, 0, -1):
                        current[i] = current[i - 1]
                    current[0] = shifted
                    if lane == 0:
                        current[0] = halo
                if cutlass.const_expr(width <= 16):
                    weight = weights[width - 1 - tap]
                else:
                    weight = cute.arch.shuffle_sync(weights[(width - 1 - tap) // 32], (width - 1 - tap) % 32)
                for i in cutlass.range_constexpr(8):
                    result[i] += current[i] * weight
            for i in cutlass.range_constexpr(8):
                sig = _sigmoid(result[i])
                grad[i] *= sig * (1.0 + result[i] * (1.0 - sig))
    return original, grad


def _is_sm86():
    """Identify the SM86 target for its schedule-specific launch choices."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return arch.major == 8 and arch.minor == 6


def _ncw_batch_group(dtype, silu, shape):
    """Choose a divisible batch grouping for short FP32 NCW reductions."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    if arch.major != 8 or dtype != cutlass.Float32 or shape[2] > 256:
        return 1
    group = 1
    if arch.minor == 9:
        group = 32 if silu else 16
    elif arch.minor == 6 and silu:
        group = 8
    return group if shape[0] % group == 0 else 1


def _ncw_warps(dtype, shape, width):
    """Choose the number of independent channel warps in an NCW block."""
    from cutlass.cutlass_dsl import CuTeDSL

    # Short K32 tiles need more independent CTAs on SM100/103, including additive gradients.
    arch = CuTeDSL._get_dsl().get_arch_enum()
    if arch.major == 10 and arch.minor in (0, 3) and dtype.width == 16 and width == 32 and shape[2] <= 256:
        return 1
    return 1 if (width >= 64 and shape[2] <= 256) or (_is_sm86() and dtype.width == 16 and shape[2] % 8 != 0) else 4


@cute.kernel
def causal_conv1d_bwd_ncw_fused(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dw: cute.Tensor,
    db: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    tiles: cutlass.Constexpr,
):
    """Fuse NCW activation recomputation, data gradients, and parameter reductions."""
    warps: cutlass.Constexpr = _ncw_warps(x.element_type, x.shape, width)
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    blocks_s = cute.ceil_div(x.shape[2], 256 * tiles)
    blocks_h = cute.ceil_div(x.shape[1], warps)
    batch_group: cutlass.Constexpr = _ncw_batch_group(x.element_type, silu, x.shape)
    block = cute.arch.block_idx()[0]
    if cutlass.const_expr(_is_sm86() and x.shape[2] <= 256 and not (silu and x.element_type == cutlass.Float32) and x.shape[0] <= 65535):
        first = cute.arch.block_idx()[0] * tiles
        channel = cute.arch.block_idx()[1] * warps + tid // 32
        batch = cute.arch.block_idx()[2]
    elif cutlass.const_expr(batch_group > 1):
        first = (cute.arch.block_idx()[0] // batch_group) * tiles
        channel = cute.arch.block_idx()[1] * warps + tid // 32
        batch = cute.arch.block_idx()[2] * batch_group + cute.arch.block_idx()[0] % batch_group
    elif cutlass.const_expr(silu and x.element_type == cutlass.Float32 and x.shape[2] <= 256):
        batch = block % x.shape[0]
        channel = (block // x.shape[0]) * warps + tid // 32
        first = 0
    else:
        first = (block % blocks_s) * tiles
        channel = (block // blocks_s % blocks_h) * warps + tid // 32
        batch = block // (blocks_s * blocks_h)
    compute = dw.element_type
    scratch = None
    if cutlass.const_expr(x.shape[2] % 8 != 0):
        scratch = SmemAllocator().allocate_tensor(compute, cute.make_layout((warps, 256), stride=(256, 1)), byte_alignment=16)
    distributed = width > 16
    slots = cute.ceil_div(width, 32) if distributed else width
    dw_reg = cute.make_rmem_tensor((slots,), compute)
    dw_reg.fill(0)
    db_value = compute(0)
    weights = cute.make_rmem_tensor((cute.ceil_div(width, 32) if width > 16 else width,), compute)
    if channel < x.shape[1]:
        bias_value = b[channel].to(compute)
        if cutlass.const_expr(width <= 16):
            for k in cutlass.range_constexpr(width):
                weights[k] = w[channel, k].to(compute)
        else:
            for group in cutlass.range_constexpr(cute.ceil_div(width, 32)):
                k = group * 32 + lane
                value = compute(0)
                if k < width:
                    value = w[channel, k].to(compute)
                weights[group] = value
        end = min(first + tiles, cute.ceil_div(x.shape[2], 256))
        unused, following = _ncw_dz(x, w, weights, bias_value, dy, batch, channel, end, lane, width, silu, compute, scratch)
        for reverse in cutlass.range(0, end - first, unroll=1):
            tile = end - 1 - reverse
            start = tile * 256 + lane * 8
            original, current = _ncw_dz(x, w, weights, bias_value, dy, batch, channel, tile, lane, width, silu, compute, scratch)
            keep = cute.make_rmem_tensor((8,), compute)
            keep.store(current.load())
            result = cute.make_rmem_tensor((8,), compute)
            result.fill(0)
            for i in cutlass.range_constexpr(8):
                db_value += current[i]
            for tap in cutlass.range_constexpr(width):
                if cutlass.const_expr(tap > 0):
                    shifted = cute.arch.shuffle_sync(current[0], (lane + 1) % 32)
                    halo = cute.arch.shuffle_sync(following[(tap - 1) % 8], (tap - 1) // 8)
                    for i in cutlass.range_constexpr(7):
                        current[i] = current[i + 1]
                    current[7] = shifted
                    if lane == 31:
                        current[7] = halo
                if cutlass.const_expr(width <= 16):
                    weight = weights[width - 1 - tap]
                else:
                    weight = cute.arch.shuffle_sync(weights[(width - 1 - tap) // 32], (width - 1 - tap) % 32)
                grad = compute(0)
                for i in cutlass.range_constexpr(8):
                    result[i] += current[i] * weight
                    grad += original[i] * current[i]
                if cutlass.const_expr(distributed):
                    for offset in cutlass.range_constexpr(5):
                        grad += cute.arch.shuffle_sync_bfly(grad, 1 << offset)
                    if lane == (width - 1 - tap) % 32:
                        dw_reg[(width - 1 - tap) // 32] += grad
                else:
                    dw_reg[width - 1 - tap] += grad
            following.store(keep.load())
            if cutlass.const_expr(dx.shape[2] % 8 == 0):
                if start < dx.shape[2]:
                    raw = cute.make_rmem_tensor((8,), dx.element_type)
                    raw.store(result.load().to(dx.element_type))
                    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dx.element_type, num_bits_per_copy=128)
                    offset = cutlass.Int64(batch) * dx.stride[0] + cutlass.Int64(channel) * dx.stride[1] + start
                    view = cute.make_tensor((dx.iterator + offset).align(16), cute.make_layout((8,)))
                    cute.copy(atom, raw, view)
            else:
                _tail_store(dx, batch, channel, start, result, scratch)
        for offset in cutlass.range_constexpr(5):
            db_value += cute.arch.shuffle_sync_bfly(db_value, 1 << offset)
        if lane == 0:
            cute.arch.atomic_add(db.iterator + channel, db_value)
        for k in cutlass.range_constexpr(slots):
            value = dw_reg[k]
            if cutlass.const_expr(distributed):
                if k * 32 + lane < width:
                    cute.arch.atomic_add(dw.iterator + cutlass.Int64(channel) * width + k * 32 + lane, value)
            else:
                for offset in cutlass.range_constexpr(5):
                    value += cute.arch.shuffle_sync_bfly(value, 1 << offset)
                if lane == 0:
                    cute.arch.atomic_add(dw.iterator + cutlass.Int64(channel) * width + k, value)


causal_conv1d_bwd_ncw_fused.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _nwh_async_vector(src, dst, offset, target, valid, vector: cutlass.Constexpr):
    """Stage one predicated 128-bit global vector with an asynchronous copy."""
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
def _nwh_stage(
    x,
    dy,
    sx,
    sy,
    batch,
    channel_start,
    tile,
    stage: cutlass.Constexpr,
    width: cutlass.Constexpr,
    tile_s: cutlass.Constexpr,
    tile_h: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    silu: cutlass.Constexpr,
):
    """Stage an NWH input/gradient tile and halos, then commit its asynchronous copies."""
    vector = 128 // x.element_type.width
    tpr = tile_h // vector
    halo = width - 1
    xr = tile_s + 2 * halo
    yr = tile_s + halo
    tid = cute.arch.thread_idx()[0]
    start = tile * tile_s
    for index in cutlass.range(tid, xr * tpr, threads, unroll=1):
        row = index // tpr
        col = index % tpr * vector
        pos = start - halo + row
        valid = pos >= 0 and pos < x.shape[1] and channel_start + col < x.shape[2]
        offset = cutlass.Int64(batch) * x.stride[0] + cutlass.Int64(pos) * x.stride[1] + channel_start + col
        _nwh_async_vector(x, sx, offset, (stage * xr + row) * tile_h + col, valid, vector)
    for index in cutlass.range(tid, yr * tpr, threads, unroll=1):
        row = index // tpr
        col = index % tpr * vector
        pos = start + row
        valid = pos < x.shape[1] and channel_start + col < x.shape[2]
        offset = cutlass.Int64(batch) * dy.stride[0] + cutlass.Int64(pos) * dy.stride[1] + channel_start + col
        _nwh_async_vector(dy, sy, offset, (stage * yr + row) * tile_h + col, valid, vector)
    cute.arch.cp_async_commit_group()


@cute.jit
def _nwh_load_vector(src, offset, vector: cutlass.Constexpr):
    """Load an aligned 128-bit vector into registers."""
    raw = cute.make_rmem_tensor((vector,), src.element_type)
    view = cute.make_tensor((src.iterator + offset).align(16), cute.make_layout((vector,)))
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=128)
    cute.copy(atom, view, raw)
    return raw


@cute.jit
def _nwh_compute(
    sx,
    sy,
    dx,
    weights,
    bias,
    partial,
    batch,
    channel_start,
    tile,
    stage: cutlass.Constexpr,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    tile_s: cutlass.Constexpr,
    tile_h: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """Consume a staged NWH tile and accumulate parameter partials using scalar FP32 arithmetic."""
    vector = 128 // dx.element_type.width
    tpr = tile_h // vector
    rows = tile_s // (threads // tpr)
    halo = width - 1
    xr, yr = tile_s + 2 * halo, tile_s + halo
    tid = cute.arch.thread_idx()[0]
    col = tid % tpr * vector
    rbase = tid // tpr * rows
    start = tile * tile_s
    window = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    grads = cute.make_rmem_tensor(cute.make_layout((rows + halo, vector), stride=(vector, 1)), cutlass.Float32)
    for j in cutlass.range_constexpr(halo):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j) * tile_h + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[j, v] = raw[v].to(cutlass.Float32)
    for j in cutlass.range_constexpr(rows + halo):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j + halo) * tile_h + col, vector)
        gy = _nwh_load_vector(sy, (stage * yr + rbase + j) * tile_h + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[(j + halo) % width, v] = raw[v].to(cutlass.Float32)
            grad = gy[v].to(cutlass.Float32)
            if cutlass.const_expr(silu):
                z = bias[v]
                for k in cutlass.range_constexpr(width):
                    z = cute.math.fma(window[(j + k) % width, v], weights[k, v], z)
                if cutlass.const_expr(dx.element_type.width == 16):
                    sig = 0.5 + 0.5 * cute.tanh(0.5 * z, fastmath=True)
                else:
                    sig = _sigmoid(z)
                grad *= sig * (1.0 + z * (1.0 - sig))
            grads[j, v] = grad
            if cutlass.const_expr(j < rows):
                partial[width, v] += grad
                for k in cutlass.range_constexpr(width):
                    partial[k, v] = cute.math.fma(window[(j + k) % width, v], grad, partial[k, v])
    for j in cutlass.range_constexpr(rows):
        raw = cute.make_rmem_tensor((vector,), dx.element_type)
        for v in cutlass.range_constexpr(vector):
            value = cutlass.Float32(0)
            if cutlass.const_expr(dx.element_type == cutlass.Float32 or not _uses_packed_nwh()):
                value = _nwh_first_pair(
                    grads[j, v],
                    weights[width - 1, v],
                    grads[j + 1, v],
                    weights[width - 2, v],
                    channel_start + col + v,
                    start + rbase + j,
                    dx.shape,
                    width,
                    dx.element_type.width,
                    silu,
                )
                for k in cutlass.range_constexpr(2, width):
                    value = cute.math.fma(grads[j + k, v], weights[width - 1 - k, v], value)
            else:
                for k in cutlass.range_constexpr(width):
                    value = cute.math.fma(grads[j + k, v], weights[width - 1 - k, v], value)
            raw[v] = value.to(dx.element_type)
        if start + rbase + j < dx.shape[1] and channel_start + col < dx.shape[2]:
            offset = cutlass.Int64(batch) * dx.stride[0] + cutlass.Int64(start + rbase + j) * dx.stride[1] + channel_start + col
            view = cute.make_tensor((dx.iterator + offset).align(16), cute.make_layout((vector,)))
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(), dx.element_type, num_bits_per_copy=128, store_cache_mode=cute.nvgpu.common.StoreCacheMode.STREAMING
            )
            cute.copy(atom, raw, view)


@cute.jit
def _nwh_compute_packed(
    sx,
    sy,
    dx,
    weights,
    bias,
    partial,
    batch,
    channel_start,
    tile,
    stage: cutlass.Constexpr,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    tile_s: cutlass.Constexpr,
    tile_h: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """Consume a staged NWH tile with packed FP32 arithmetic and retained parameter partials."""
    vector = 128 // dx.element_type.width
    tpr = tile_h // vector
    rows = tile_s // (threads // tpr)
    halo = width - 1
    xr, yr = (tile_s + 2 * halo, tile_s + halo)
    tid = cute.arch.thread_idx()[0]
    col = tid % tpr * vector
    rbase = tid // tpr * rows
    start = tile * tile_s
    window = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    grads = cute.make_rmem_tensor(cute.make_layout((rows + halo, vector), stride=(vector, 1)), cutlass.Float32)
    for j in cutlass.range_constexpr(halo):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j) * tile_h + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[j, v] = raw[v].to(cutlass.Float32)
    for j in cutlass.range_constexpr(rows + halo):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j + halo) * tile_h + col, vector)
        gy = _nwh_load_vector(sy, (stage * yr + rbase + j) * tile_h + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[(j + halo) % width, v] = raw[v].to(cutlass.Float32)
        for v in cutlass.range_constexpr(0, vector, 2):
            g0, g1 = (gy[v].to(cutlass.Float32), gy[v + 1].to(cutlass.Float32))
            if cutlass.const_expr(silu):
                z0, z1 = (bias[v], bias[v + 1])
                for k in cutlass.range_constexpr(width):
                    z0, z1 = cute.arch.fma_packed_f32x2(
                        (window[(j + k) % width, v], window[(j + k) % width, v + 1]), (weights[k, v], weights[k, v + 1]), (z0, z1)
                    )
                s0 = 0.5 + 0.5 * cute.tanh(0.5 * z0, fastmath=True)
                s1 = 0.5 + 0.5 * cute.tanh(0.5 * z1, fastmath=True)
                g0 *= s0 * (1.0 + z0 * (1.0 - s0))
                g1 *= s1 * (1.0 + z1 * (1.0 - s1))
            grads[j, v], grads[j, v + 1] = (g0, g1)
            if cutlass.const_expr(j < rows):
                partial[width, v] += g0
                partial[width, v + 1] += g1
                for k in cutlass.range_constexpr(width):
                    p0, p1 = cute.arch.fma_packed_f32x2(
                        (window[(j + k) % width, v], window[(j + k) % width, v + 1]), (g0, g1), (partial[k, v], partial[k, v + 1])
                    )
                    partial[k, v], partial[k, v + 1] = (p0, p1)
    for j in cutlass.range_constexpr(rows):
        raw = cute.make_rmem_tensor((vector,), dx.element_type)
        for v in cutlass.range_constexpr(0, vector, 2):
            value0, value1 = (cutlass.Float32(0), cutlass.Float32(0))
            for k in cutlass.range_constexpr(width):
                value0, value1 = cute.arch.fma_packed_f32x2(
                    (grads[j + k, v], grads[j + k, v + 1]), (weights[width - 1 - k, v], weights[width - 1 - k, v + 1]), (value0, value1)
                )
            raw[v], raw[v + 1] = (value0.to(dx.element_type), value1.to(dx.element_type))
        if start + rbase + j < dx.shape[1] and channel_start + col < dx.shape[2]:
            offset = cutlass.Int64(batch) * dx.stride[0] + cutlass.Int64(start + rbase + j) * dx.stride[1] + channel_start + col
            view = cute.make_tensor((dx.iterator + offset).align(16), cute.make_layout((vector,)))
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dx.element_type, num_bits_per_copy=128)
            cute.copy(atom, raw, view)


@cute.jit
def _nwh_compute_pending(
    sx,
    sy,
    dx,
    weights,
    bias,
    partial,
    batch,
    channel_start,
    tile,
    stage: cutlass.Constexpr,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    tile_s: cutlass.Constexpr,
    tile_h: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    """Compute a staged NWH tile while retaining pending data-gradient contributions."""
    vector = 128 // dx.element_type.width
    tpr = tile_h // vector
    rows = tile_s // (threads // tpr)
    halo = width - 1
    xr, yr = tile_s + 2 * halo, tile_s + halo
    tid = cute.arch.thread_idx()[0]
    col = tid % tpr * vector
    rbase = tid // tpr * rows
    start = tile * tile_s
    window = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    pending = cute.make_rmem_tensor(cute.make_layout((rows, vector), stride=(vector, 1)), cutlass.Float32)
    for j in cutlass.range_constexpr(halo):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j) * tile_h + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[j, v] = raw[v].to(cutlass.Float32)
    for j in cutlass.range_constexpr(rows + halo):
        raw = _nwh_load_vector(sx, (stage * xr + rbase + j + halo) * tile_h + col, vector)
        gy = _nwh_load_vector(sy, (stage * yr + rbase + j) * tile_h + col, vector)
        for v in cutlass.range_constexpr(vector):
            window[(j + halo) % width, v] = raw[v].to(cutlass.Float32)
            grad = gy[v].to(cutlass.Float32)
            if cutlass.const_expr(silu):
                z = bias[v]
                for k in cutlass.range_constexpr(width):
                    z = cute.math.fma(window[(j + k) % width, v], weights[k, v], z)
                if cutlass.const_expr(dx.element_type.width == 16):
                    sig = 0.5 + 0.5 * cute.tanh(0.5 * z, fastmath=True)
                else:
                    sig = _sigmoid(z)
                grad *= sig * (1.0 + z * (1.0 - sig))
            if cutlass.const_expr(j < rows):
                partial[width, v] += grad
                for k in cutlass.range_constexpr(width):
                    partial[k, v] = cute.math.fma(window[(j + k) % width, v], grad, partial[k, v])
            for outrow in cutlass.range_constexpr(rows):
                if cutlass.const_expr(outrow <= j and j - outrow <= halo):
                    if cutlass.const_expr(dx.element_type == cutlass.Float32 or not _uses_packed_nwh()):
                        if cutlass.const_expr(j == outrow):
                            pending[outrow, v] = grad
                        elif cutlass.const_expr(j == outrow + 1):
                            pending[outrow, v] = _nwh_first_pair(
                                pending[outrow, v],
                                weights[width - 1, v],
                                grad,
                                weights[width - 2, v],
                                channel_start + col + v,
                                start + rbase + outrow,
                                dx.shape,
                                width,
                                dx.element_type.width,
                                silu,
                            )
                        else:
                            pending[outrow, v] = cute.math.fma(grad, weights[width - 1 - (j - outrow), v], pending[outrow, v])
                    else:
                        if cutlass.const_expr(j == outrow):
                            pending[outrow, v] = grad * weights[width - 1, v]
                        else:
                            pending[outrow, v] = cute.math.fma(grad, weights[width - 1 - (j - outrow), v], pending[outrow, v])
        if cutlass.const_expr(j >= halo):
            result = cute.make_rmem_tensor((vector,), dx.element_type)
            for v in cutlass.range_constexpr(vector):
                result[v] = pending[j - halo, v].to(dx.element_type)
            if start + rbase + j - halo < dx.shape[1] and channel_start + col < dx.shape[2]:
                offset = cutlass.Int64(batch) * dx.stride[0] + cutlass.Int64(start + rbase + j - halo) * dx.stride[1] + channel_start + col
                view = cute.make_tensor((dx.iterator + offset).align(16), cute.make_layout((vector,)))
                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyR2GOp(), dx.element_type, num_bits_per_copy=128, store_cache_mode=cute.nvgpu.common.StoreCacheMode.STREAMING
                )
                cute.copy(atom, result, view)


@cute.kernel
def nwh_pipeline(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    dy: cute.Tensor,
    dx: cute.Tensor,
    dw: cute.Tensor,
    db: cute.Tensor,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    nt: cutlass.Constexpr,
    tile_s: cutlass.Constexpr,
    tile_h: cutlass.Constexpr,
):
    """Double-buffer NWH tiles, retain parameter partials, and reduce once per block."""
    threads = 128
    vector = 128 // x.element_type.width
    tpr = tile_h // vector
    tid = cute.arch.thread_idx()[0]
    first = cute.arch.block_idx()[1] * nt
    batch = cute.arch.block_idx()[2]
    channel_start = cute.arch.block_idx()[0] * tile_h
    col = tid % tpr * vector
    count = min(nt, cute.ceil_div(x.shape[1], tile_s) - first)
    smem = SmemAllocator()
    sx = smem.allocate_tensor(x.element_type, cute.make_layout((2 * (tile_s + 2 * (width - 1)) * tile_h,)), byte_alignment=16)
    sy = smem.allocate_tensor(x.element_type, cute.make_layout((2 * (tile_s + width - 1) * tile_h,)), byte_alignment=16)
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((width + 1) * 4 * tile_h,)), byte_alignment=16)
    weights = cute.make_rmem_tensor(cute.make_layout((width, vector), stride=(vector, 1)), cutlass.Float32)
    bias = cute.make_rmem_tensor((vector,), cutlass.Float32)
    weights.fill(0)
    bias.fill(0)
    if channel_start + col < x.shape[2]:
        for k in cutlass.range_constexpr(width):
            raw = _nwh_load_vector(w, cutlass.Int64(k) * x.shape[2] + channel_start + col, vector)
            for v in cutlass.range_constexpr(vector):
                weights[k, v] = raw[v].to(cutlass.Float32)
        raw = _nwh_load_vector(b, channel_start + col, vector)
        for v in cutlass.range_constexpr(vector):
            bias[v] = raw[v].to(cutlass.Float32)
    partial = cute.make_rmem_tensor(cute.make_layout((width + 1, vector), stride=(vector, 1)), cutlass.Float32)
    partial.fill(0)
    _nwh_stage(x, dy, sx, sy, batch, channel_start, first, 0, width, tile_s, tile_h, threads, silu)
    for pair in cutlass.range(0, cute.ceil_div(count, 2), unroll=1):
        ia = pair * 2
        if ia + 1 < count:
            _nwh_stage(x, dy, sx, sy, batch, channel_start, first + ia + 1, 1, width, tile_s, tile_h, threads, silu)
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()
        if cutlass.const_expr(x.element_type.width == 16 and _uses_packed_nwh()):
            _nwh_compute_packed(sx, sy, dx, weights, bias, partial, batch, channel_start, first + ia, 0, width, silu, tile_s, tile_h, threads)
        elif cutlass.const_expr(tile_h == 128):
            _nwh_compute_pending(sx, sy, dx, weights, bias, partial, batch, channel_start, first + ia, 0, width, silu, tile_s, tile_h, threads)
        else:
            _nwh_compute(sx, sy, dx, weights, bias, partial, batch, channel_start, first + ia, 0, width, silu, tile_s, tile_h, threads)
        cute.arch.barrier()
        if ia + 2 < count:
            _nwh_stage(x, dy, sx, sy, batch, channel_start, first + ia + 2, 0, width, tile_s, tile_h, threads, silu)
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier()
        if ia + 1 < count:
            if cutlass.const_expr(x.element_type.width == 16 and _uses_packed_nwh()):
                _nwh_compute_packed(sx, sy, dx, weights, bias, partial, batch, channel_start, first + ia + 1, 1, width, silu, tile_s, tile_h, threads)
            elif cutlass.const_expr(tile_h == 128):
                _nwh_compute_pending(sx, sy, dx, weights, bias, partial, batch, channel_start, first + ia + 1, 1, width, silu, tile_s, tile_h, threads)
            else:
                _nwh_compute(sx, sy, dx, weights, bias, partial, batch, channel_start, first + ia + 1, 1, width, silu, tile_s, tile_h, threads)
        cute.arch.barrier()
    cute.arch.cp_async_wait_group(0)
    for k in cutlass.range_constexpr(width + 1):
        for v in cutlass.range_constexpr(vector):
            value = partial[k, v]
            for i in cutlass.range_constexpr((32 // tpr).bit_length() - 1):
                value += cute.arch.shuffle_sync_bfly(value, tpr * (1 << i))
            if tid % 32 < tpr:
                red[(k * 4 + tid // 32) * tile_h + (tid % 32) * vector + v] = value
    cute.arch.barrier()
    if tid < tile_h and channel_start + tid < x.shape[2]:
        for k in cutlass.range_constexpr(width + 1):
            value = cutlass.Float32(0)
            for warp in cutlass.range_constexpr(4):
                value += red[(k * 4 + warp) * tile_h + tid]
            if cutlass.const_expr(k < width):
                cute.arch.atomic_add(dw.iterator + cutlass.Int64(k) * x.shape[2] + channel_start + tid, value)
            else:
                cute.arch.atomic_add(db.iterator + channel_start + tid, value)


nwh_pipeline.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def _use_nwh_pipeline(dtype, width, silu, shape):
    """Use staged tiles only where enough sequence work amortizes synchronization."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    if not (2 <= width <= 8 and _nwh_seqpar(dtype, width, silu, shape)):
        return False
    if shape[1] < 1024 or dtype.width != 16:
        return False
    if width == 3 and shape[2] >= 1024 and arch.major < 12:
        return True
    if arch.major == 10 and width == 4 and shape[0] * shape[1] * shape[2] >= 100_000_000:
        return True
    if arch.major == 8 and arch.minor in (0, 6) and width == 4:
        return shape[0] * shape[1] * shape[2] >= (6_000_000 if arch.minor == 6 else 100_000_000)
    return False


@cute.jit
def launch_bwd(
    x: cute.Tensor,
    w: cute.Tensor,
    b: cute.Tensor,
    dy: cute.Tensor,
    dz: cute.Tensor,
    dx: cute.Tensor,
    dw: cute.Tensor,
    db: cute.Tensor,
    stream: cuda.CUstream,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    nwh: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    accumulate: cutlass.Constexpr = False,
):
    """Validate row tiling and launch backward, clearing parameter gradients unless accumulation is requested."""
    if cutlass.const_expr(rows <= 0):
        raise ValueError(f"Backward rows must be positive, got {rows}")
    if cutlass.const_expr(not nwh and _is_sm86() and x.shape[2] >= 1024 and rows == 4):
        rows = 16
    if cutlass.const_expr(nwh and x.element_type == cutlass.Float64 and x.shape[1] <= 256 and rows == 32):
        rows = 8
    from cutlass.cutlass_dsl import CuTeDSL

    if cutlass.const_expr(
        nwh
        and x.element_type.width == 16
        and width <= 8
        and x.shape[1] <= 256
        and x.shape[2] <= 128
        and rows == 32
        and CuTeDSL._get_dsl().get_arch_enum().major == 12
    ):
        rows = 8
    if cutlass.const_expr(
        nwh
        and x.element_type.width == 16
        and width == 7
        and not silu
        and x.shape[1] <= 256
        and x.shape[2] >= 768
        and rows == 32
        and (
            CuTeDSL._get_dsl().get_arch_enum().major == 12 or (CuTeDSL._get_dsl().get_arch_enum().major == 8 and CuTeDSL._get_dsl().get_arch_enum().minor == 9)
        )
    ):
        rows = 128
    if cutlass.const_expr(not accumulate):
        causal_conv1d_zero_grad(dw, db).launch(grid=(cute.ceil_div(cute.size(dw), 256), 1, 1), block=(256, 1, 1), stream=stream)
    if cutlass.const_expr(not nwh):
        warps: cutlass.Constexpr = _ncw_warps(x.element_type, x.shape, width)
        batch_group: cutlass.Constexpr = _ncw_batch_group(x.element_type, silu, x.shape)
        grid = (cute.ceil_div(x.shape[2], 256 * rows) * cute.ceil_div(x.shape[1], warps) * x.shape[0], 1, 1)
        if cutlass.const_expr(_is_sm86() and x.shape[2] <= 256 and not (silu and x.element_type == cutlass.Float32) and x.shape[0] <= 65535):
            grid = (cute.ceil_div(x.shape[2], 256 * rows), cute.ceil_div(x.shape[1], warps), x.shape[0])
        if cutlass.const_expr(batch_group > 1):
            grid = (cute.ceil_div(x.shape[2], 256 * rows) * batch_group, cute.ceil_div(x.shape[1], warps), x.shape[0] // batch_group)
        causal_conv1d_bwd_ncw_fused(x, w, b, dy, dx, dw, db, width, silu, rows).launch(grid=grid, block=(32 * warps, 1, 1), stream=stream)
    elif cutlass.const_expr(
        x.element_type.width == 16
        and not silu
        and x.shape[1] <= 512
        and x.shape[2] % 8 == 0
        and cute.size(x) >= 4_000_000
        and CuTeDSL._get_dsl().get_arch_enum().major == 8
        and ((CuTeDSL._get_dsl().get_arch_enum().minor == 6 and width in (4, 7)) or (CuTeDSL._get_dsl().get_arch_enum().minor == 0 and width == 7))
    ):
        nwh_pipeline(x, w, b, dy, dx, dw, db, width, silu, 8, 32, 128).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], 256), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(_use_nwh_pipeline(x.element_type, width, silu, x.shape)):
        from cutlass.cutlass_dsl import CuTeDSL

        tile_s, tile_h, nt = 64, 64, 16
        if cutlass.const_expr(CuTeDSL._get_dsl().get_arch_enum().major == 8 and CuTeDSL._get_dsl().get_arch_enum().minor == 6 and width == 4):
            if cutlass.const_expr(cute.size(x) >= 100_000_000):
                tile_s, tile_h, nt = 64, 128, 8
            else:
                tile_s, tile_h, nt = 32, 128, 16
        nwh_pipeline(x, w, b, dy, dx, dw, db, width, silu, nt, tile_s, tile_h).launch(
            grid=(cute.ceil_div(x.shape[2], tile_h), cute.ceil_div(x.shape[1], tile_s * nt), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    elif cutlass.const_expr(width <= 16):
        if cutlass.const_expr(rows % 8 != 0):
            raise ValueError(f"NWH fused backward requires rows to be a multiple of 8, got {rows}")
        causal_conv1d_bwd_nwc_fused(x, w, b, dy, dx, dw, db, width, silu, rows).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], rows), x.shape[0]), block=(128, 1, 1), stream=stream
        )
    else:
        if cutlass.const_expr(silu):
            causal_conv1d_dz(x, w, b, dy, dz, width, nwh).launch(grid=(cute.ceil_div(cute.size(x), 256), 1, 1), block=(256, 1, 1), stream=stream)
            grad = dz
        else:
            grad = dy
        causal_conv1d_bwd_nwc(x, w, grad, dx, dw, db, width, rows).launch(
            grid=(cute.ceil_div(x.shape[2], 128), cute.ceil_div(x.shape[1], rows), x.shape[0]), block=(128, 1, 1), stream=stream
        )
