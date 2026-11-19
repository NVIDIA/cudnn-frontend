# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""SM103 K4 SiLU backward with persistent register windows and warp-local TMA.

Each warp owns one contiguous token range and 256 channels. Two eight-row
stages feed four input rows and four pending dX rows per channel. Three halo
gradients finish the owned dX range; only owned tokens contribute to dW/db.
The 16-bit parameter path uses FP32 accumulation without compensation.

Shared-memory buffers:
- xs/ys: two eight-token stages per warp; only that warp loads or refills its
  slice. An async-proxy fence after the last stage load, followed by a warp
  barrier, orders those loads before the next TMA refill (Python Rule 11).
- red: one parameter-sum slice per warp, written once and read after a CTA
  barrier; it is never refilled.
- mb: one completion barrier per warp and stage, alternating phase on reuse.
"""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync
from cudnn._cutlass_compat import SmemAllocator
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from .kernel import _nwh_load_vector
from .packed_kernel import _sequence, _validate_offsets, packed_parameter_reduce, packed_parameter_reduce_fp32


@cute.jit
def pair_mul(a0, a1, b0, b1):
    return inline_ptx(
        """{ .reg .b64 a,b,c; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; mul.rn.f32x2 c,a,b; mov.b64 {$0,$1},c; }""",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a0, a1, b0, b1],
    )


@cute.jit
def pair_add(a0, a1, b0, b1):
    return inline_ptx(
        """{ .reg .b64 a,b,c; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; add.rn.f32x2 c,a,b; mov.b64 {$0,$1},c; }""",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a0, a1, b0, b1],
    )


@cute.jit
def prefetch_tma(ax, tx, ay, ty, sx, sy, mb, t0, h, col, warp, stage):
    stx = cute.make_tensor((sx.iterator + warp * 2048).align(128), cute.make_layout((256, 8), stride=(1, 256)))
    sty = cute.make_tensor((sy.iterator + warp * 2048).align(128), cute.make_layout((256, 8), stride=(1, 256)))
    gx = cute.local_tile(cute.domain_offset((h, t0), tx), (256, 8), (0, 0))
    gy = cute.local_tile(cute.domain_offset((h, t0), ty), (256, 8), (0, 0))
    px, qx = cpasync.tma_partition(ax, 0, cute.make_layout(1), cute.group_modes(stx, 0, 2), cute.group_modes(gx, 0, 2))
    py, qy = cpasync.tma_partition(ay, 0, cute.make_layout(1), cute.group_modes(sty, 0, 2), cute.group_modes(gy, 0, 2))
    if col == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(mb.iterator + stage * 4 + warp, 8192)
    cute.copy(ax, qx, px, tma_bar_ptr=mb.iterator + stage * 4 + warp)
    cute.copy(ay, qy, py, tma_bar_ptr=mb.iterator + stage * 4 + warp)


@cute.jit
def dense_tail(x, dy, weight, bias, cu, token, h, col, seq):
    start = cutlass.Int32(cu[seq])
    end = cutlass.Int32(cu[seq + 1])
    value = cute.make_rmem_tensor((8,), cutlass.Float32)
    value.fill(0)
    for ahead in cutlass.range_constexpr(4):
        pre = cute.make_rmem_tensor((8,), cutlass.Float32)
        pre.store(bias.load())
        for k in cutlass.range_constexpr(4):
            pos = token + ahead + k - 3
            inp = cute.make_rmem_tensor((8,), cutlass.Float32)
            inp.fill(0)
            if pos >= start and pos < end:
                raw = _nwh_load_vector(x, cutlass.Int64(pos) * x.stride[0] + h + col, 8)
                inp.store(raw.load().to(cutlass.Float32))
            for v in cutlass.range_constexpr(0, 8, 2):
                pre[v], pre[v + 1] = cute.arch.fma_packed_f32x2((inp[v], inp[v + 1]), (weight[k, v], weight[k, v + 1]), (pre[v], pre[v + 1]), ftz=False)
        grad = cute.make_rmem_tensor((8,), cutlass.Float32)
        grad.fill(0)
        if token + ahead < end:
            raw = _nwh_load_vector(dy, cutlass.Int64(token + ahead) * dy.stride[0] + h + col, 8)
            grad.store(raw.load().to(cutlass.Float32))
        for v in cutlass.range_constexpr(8):
            sig = 0.5 + 0.5 * cute.tanh(0.5 * pre[v], fastmath=True)
            grad[v] = cute.math.mul(grad[v], sig * (1.0 + pre[v] * (1.0 - sig)), ftz=True)
        for v in cutlass.range_constexpr(0, 8, 2):
            if cutlass.const_expr(ahead == 0):
                value[v], value[v + 1] = cute.arch.mul_packed_f32x2((grad[v], grad[v + 1]), (weight[3, v], weight[3, v + 1]))
            else:
                value[v], value[v + 1] = cute.arch.fma_packed_f32x2(
                    (grad[v], grad[v + 1]), (weight[3 - ahead, v], weight[3 - ahead, v + 1]), (value[v], value[v + 1]), ftz=False
                )
    return value


@cute.kernel
def dense_tail_fixup(x, w, b, cu, dy, dx):
    col = cute.arch.thread_idx()[0] * 8
    tile = cute.arch.block_idx()[0] // 3
    h = tile % cute.ceil_div(x.shape[1], 256) * 256
    seq = tile // cute.ceil_div(x.shape[1], 256)
    token = cutlass.Int32(cu[seq + 1]) - 1 - cute.arch.block_idx()[0] % 3
    if token >= cutlass.Int32(cu[seq]) and h + col < x.shape[1]:
        weight = cute.make_rmem_tensor(cute.make_layout((4, 8), stride=(8, 1)), cutlass.Float32)
        bias = cute.make_rmem_tensor((8,), cutlass.Float32)
        bias.fill(0)
        for k in cutlass.range_constexpr(4):
            for v in cutlass.range_constexpr(8):
                weight[k, v] = w[h + col + v, k].to(cutlass.Float32)
        if cutlass.const_expr(b is not None):
            bias.store(_nwh_load_vector(b, h + col, 8).load().to(cutlass.Float32))
        value = dense_tail(x, dy, weight, bias, cu, token, h, col, seq)
        out = cute.make_rmem_tensor((8,), dx.element_type)
        out.store(value.load().to(dx.element_type))
        target = cute.make_tensor((dx.iterator + cutlass.Int64(token) * dx.stride[0] + h + col).align(16), cute.make_layout((8,)))
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), dx.element_type, num_bits_per_copy=128), out, target)


@cute.jit
def compute_stage(
    x, cu, sx, sy, weight, bias, win, pending, acc, dx, h, col, warp, base, t0, end, seq, start, seqend, boundary: cutlass.Constexpr, dense: cutlass.Constexpr
):
    for j in cutlass.range_constexpr(8):
        pos = base + j
        if cutlass.const_expr(boundary):
            while pos >= seqend and seq + 1 < cu.shape[0] - 1:
                seq = seq + 1
                start = cutlass.Int32(cu[seq])
                seqend = cutlass.Int32(cu[seq + 1])
        raw = _nwh_load_vector(sx, (warp * 8 + j) * 256 + col, 8)
        for v in cutlass.range_constexpr(8):
            win[(j + 3) % 4, v] = raw[v].to(cutlass.Float32)
        if boundary and pos - start < 3:
            for k in cutlass.range_constexpr(3):
                if pos - 3 + k < start:
                    for v in cutlass.range_constexpr(8):
                        win[(j + k) % 4, v] = cutlass.Float32(0)
        grad = cute.make_rmem_tensor((8,), cutlass.Float32)
        grad.fill(0)
        if not boundary or pos < x.shape[0]:
            pre = cute.make_rmem_tensor((8,), cutlass.Float32)
            pre.store(bias.load())
            for k in cutlass.range_constexpr(4):
                for v in cutlass.range_constexpr(0, 8, 2):
                    pre[v], pre[v + 1] = cute.arch.fma_packed_f32x2(
                        (win[(j + k) % 4, v], win[(j + k) % 4, v + 1]), (weight[k, v], weight[k, v + 1]), (pre[v], pre[v + 1]), ftz=False
                    )
            gy = _nwh_load_vector(sy, (warp * 8 + j) * 256 + col, 8)
            if cutlass.const_expr(dense):
                for v in cutlass.range_constexpr(8):
                    sig = 0.5 + 0.5 * cute.tanh(0.5 * pre[v], fastmath=True)
                    grad[v] = cute.math.mul(gy[v].to(cutlass.Float32), sig * (1.0 + pre[v] * (1.0 - sig)), ftz=True)
            else:
                for v in cutlass.range_constexpr(0, 8, 2):
                    e0, e1 = pair_mul(pre[v], pre[v + 1], cutlass.Float32(-1.4426950408889634), cutlass.Float32(-1.4426950408889634))
                    d0, d1 = pair_add(cute.exp2(e0, fastmath=True), cute.exp2(e1, fastmath=True), cutlass.Float32(1), cutlass.Float32(1))
                    s0 = cute.math.div(1.0, d0, approx=True, ftz=True)
                    s1 = cute.math.div(1.0, d1, approx=True, ftz=True)
                    m0, m1 = pair_add(-s0, -s1, cutlass.Float32(1), cutlass.Float32(1))
                    m0, m1 = cute.arch.fma_packed_f32x2((pre[v], pre[v + 1]), (m0, m1), (cutlass.Float32(1), cutlass.Float32(1)), ftz=False)
                    g0, g1 = pair_mul(gy[v].to(cutlass.Float32), gy[v + 1].to(cutlass.Float32), s0, s1)
                    grad[v], grad[v + 1] = pair_mul(g0, g1, m0, m1)
        if not boundary or pos < end:
            for k in cutlass.range_constexpr(4):
                for v in cutlass.range_constexpr(0, 8, 2):
                    acc[k, v], acc[k, v + 1] = cute.arch.fma_packed_f32x2(
                        (win[(j + k) % 4, v], win[(j + k) % 4, v + 1]), (grad[v], grad[v + 1]), (acc[k, v], acc[k, v + 1]), ftz=False
                    )
            for v in cutlass.range_constexpr(0, 8, 2):
                acc[4, v], acc[4, v + 1] = pair_add(acc[4, v], acc[4, v + 1], grad[v], grad[v + 1])
        for v in cutlass.range_constexpr(8):
            pending[j % 4, v] = cutlass.Float32(0)
        for k in cutlass.range_constexpr(4):
            if not boundary or (pos - k >= start and pos < x.shape[0]):
                for v in cutlass.range_constexpr(0, 8, 2):
                    if cutlass.const_expr(dense and k == 0):
                        pending[j % 4, v], pending[j % 4, v + 1] = cute.arch.mul_packed_f32x2((grad[v], grad[v + 1]), (weight[3, v], weight[3, v + 1]))
                    else:
                        pending[(j - k) % 4, v], pending[(j - k) % 4, v + 1] = cute.arch.fma_packed_f32x2(
                            (grad[v], grad[v + 1]), (weight[3 - k, v], weight[3 - k, v + 1]), (pending[(j - k) % 4, v], pending[(j - k) % 4, v + 1]), ftz=False
                        )
        outpos = pos - 3
        if not boundary or (outpos >= t0 and outpos < end and h + col < x.shape[1]):
            out = cute.make_rmem_tensor((8,), dx.element_type)
            for v in cutlass.range_constexpr(8):
                out[v] = pending[(j - 3) % 4, v].to(dx.element_type)
            target = cute.make_tensor((dx.iterator + cutlass.Int64(outpos) * dx.stride[0] + h + col).align(16), cute.make_layout((8,)))
            cute.copy(cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), dx.element_type, num_bits_per_copy=128), out, target)
    return seq, start, seqend


@cute.kernel
def persistent(ax, tx, ay, ty, x, w, b, cu, dy, dx, partials, dense: cutlass.Constexpr):
    _validate_offsets(cu, x.shape[0])
    tid = cute.arch.thread_idx()[0]
    warp = tid // 32
    col = tid % 32 * 8
    h = cute.arch.block_idx()[0] % cute.ceil_div(x.shape[1], 256) * 256
    split = cute.arch.block_idx()[0] // cute.ceil_div(x.shape[1], 256)
    span = cute.ceil_div(x.shape[0], partials.shape[0] * 4)
    t0 = (split * 4 + warp) * span
    end = min(t0 + span, x.shape[0])
    seq = _sequence(cu, t0, cutlass.Int32)
    start = cutlass.Int32(cu[seq])
    seqend = cutlass.Int32(cu[seq + 1])
    alloc = SmemAllocator()
    xs = alloc.allocate_tensor(x.element_type, cute.make_layout((2 * 4 * 8 * 256,)), byte_alignment=128)
    ys = alloc.allocate_tensor(dy.element_type, cute.make_layout((2 * 4 * 8 * 256,)), byte_alignment=128)
    red = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((4 * 5 * 256,)), byte_alignment=16)
    mb = alloc.allocate_tensor(cutlass.Int64, cute.make_layout((8,)), byte_alignment=8)
    if tid < 8:
        cute.arch.mbarrier_init(mb.iterator + tid, 1)
    if tid == 0:
        cpasync.prefetch_descriptor(ax)
        cpasync.prefetch_descriptor(ay)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    weight = cute.make_rmem_tensor(cute.make_layout((4, 8), stride=(8, 1)), cutlass.Float32)
    bias = cute.make_rmem_tensor((8,), cutlass.Float32)
    win = cute.make_rmem_tensor(cute.make_layout((4, 8), stride=(8, 1)), cutlass.Float32)
    pending = cute.make_rmem_tensor(cute.make_layout((4, 8), stride=(8, 1)), cutlass.Float32)
    acc = cute.make_rmem_tensor(cute.make_layout((5, 8), stride=(8, 1)), cutlass.Float32)
    weight.fill(0)
    bias.fill(0)
    win.fill(0)
    pending.fill(0)
    acc.fill(0)
    if h + col < x.shape[1]:
        for k in cutlass.range_constexpr(4):
            for v in cutlass.range_constexpr(8):
                weight[k, v] = w[h + col + v, k].to(cutlass.Float32)
        if cutlass.const_expr(b is not None):
            bias.store(_nwh_load_vector(b, h + col, 8).load().to(cutlass.Float32))
        for k in cutlass.range_constexpr(3):
            if t0 - 3 + k >= start and t0 - 3 + k < x.shape[0]:
                raw = _nwh_load_vector(x, cutlass.Int64(t0 - 3 + k) * x.stride[0] + h + col, 8)
                for v in cutlass.range_constexpr(8):
                    win[k, v] = raw[v].to(cutlass.Float32)
    sx = cute.make_tensor(xs.iterator, cute.make_layout((4 * 8 * 256,)))
    sy = cute.make_tensor(ys.iterator, cute.make_layout((4 * 8 * 256,)))
    prefetch_tma(ax, tx, ay, ty, sx, sy, mb, t0, h, col, warp, 0)
    stage = cutlass.Int32(0)
    for block in cutlass.range(cute.ceil_div(span + 3, 8), unroll=1):
        sx = cute.make_tensor((xs.iterator + stage * 8192).align(128), cute.make_layout((8192,)))
        sy = cute.make_tensor((ys.iterator + stage * 8192).align(128), cute.make_layout((8192,)))
        cute.arch.mbarrier_wait(mb.iterator + stage * 4 + warp, (block // 2) % 2)
        cute.arch.sync_warp()
        if block + 1 < cute.ceil_div(span + 3, 8):
            nsx = cute.make_tensor((xs.iterator + (1 - stage) * 8192).align(128), cute.make_layout((8192,)))
            nsy = cute.make_tensor((ys.iterator + (1 - stage) * 8192).align(128), cute.make_layout((8192,)))
            prefetch_tma(ax, tx, ay, ty, nsx, nsy, mb, t0 + (block + 1) * 8, h, col, warp, 1 - stage)
        base = t0 + block * 8
        if base >= t0 + 3 and base >= start + 3 and base + 8 <= seqend and base + 8 <= end and h + col < x.shape[1]:
            seq, start, seqend = compute_stage(
                x, cu, sx, sy, weight, bias, win, pending, acc, dx, h, col, warp, base, t0, end, seq, start, seqend, False, dense
            )
        else:
            seq, start, seqend = compute_stage(x, cu, sx, sy, weight, bias, win, pending, acc, dx, h, col, warp, base, t0, end, seq, start, seqend, True, dense)
        # TMA is the async-proxy writer. Keep this fence after every stage load;
        # sync_warp alone does not order unread LDS results before a refill.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        stage = 1 - stage
    for k in cutlass.range_constexpr(5):
        for v in cutlass.range_constexpr(8):
            red[(warp * 5 + k) * 256 + col + v] = acc[k, v]
    cute.arch.barrier()
    if warp == 0 and h + col < x.shape[1]:
        for k in cutlass.range_constexpr(5):
            for v in cutlass.range_constexpr(8):
                val = cutlass.Float32(0)
                for z in cutlass.range_constexpr(4):
                    val = val + red[(z * 5 + k) * 256 + col + v]
                partials[split, k, h + col + v] = val


@cute.jit
def launch_supported_backward(
    x,
    w,
    b,
    cu,
    initial,
    dy,
    dfinal,
    dz,
    dx,
    dw,
    db,
    dinitial,
    partials,
    stream: cuda.CUstream,
    width: cutlass.Constexpr,
    silu: cutlass.Constexpr,
    fast: cutlass.Constexpr = True,
    index32: cutlass.Constexpr = True,
    dense: cutlass.Constexpr = False,
):
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    assert arch.major == 10 and arch.minor == 3, "persistent SE backward requires SM103"
    assert fast and index32 and x.element_type.width == 16
    assert width == 4 and silu and initial is None and dfinal is None and dinitial is None
    # TMA requires a nonzero row pitch even when a static unit axis has stride zero.
    layout = cute.make_layout((256, 8), stride=(1, 256))
    xt = cute.make_tensor(x.iterator, cute.make_layout((x.shape[1], x.shape[0]), stride=(1, x.shape[1])))
    yt = cute.make_tensor(dy.iterator, cute.make_layout((dy.shape[1], dy.shape[0]), stride=(1, dy.shape[1])))
    ax, tx = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), xt, layout, (256, 8))
    ay, ty = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), yt, layout, (256, 8))
    persistent(ax, tx, ay, ty, x, w, b, cu, dy, dx, partials, dense).launch(
        grid=(cute.ceil_div(x.shape[1], 256) * partials.shape[0], 1, 1), block=(128, 1, 1), stream=stream
    )
    if cutlass.const_expr(dw.element_type == cutlass.Float32):
        packed_parameter_reduce_fp32(partials, dw, db, width).launch(grid=(cute.ceil_div(x.shape[1], 128) * 5, 1, 1), block=(128, 1, 1), stream=stream)
    else:
        packed_parameter_reduce(partials, dw, db, width).launch(grid=(cute.ceil_div(x.shape[1], 128), 1, 1), block=(128, 1, 1), stream=stream)
    if cutlass.const_expr(dense):
        # Retain the dense parent's padded-tail arithmetic, including signed zero.
        dense_tail_fixup(x, w, b, cu, dy, dx).launch(grid=(3 * (cu.shape[0] - 1) * cute.ceil_div(x.shape[1], 256), 1, 1), block=(32, 1, 1), stream=stream)


persistent.set_name_prefix("cudnn", remove_cutlass_symbol=True)
dense_tail_fixup.set_name_prefix("cudnn", remove_cutlass_symbol=True)
