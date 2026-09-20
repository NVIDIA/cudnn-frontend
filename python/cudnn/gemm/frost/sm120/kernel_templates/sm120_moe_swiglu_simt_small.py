# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Expert-grouped BF16 SwiGLU FC1 for SM120.
# Built on NVIDIA CUTLASS / CuTe DSL (Apache-2.0):
# https://github.com/NVIDIA/cutlass
# NVIDIA Frost supplies the problem definition and reference semantics.
# This specialization is our research and implementation.
#
# Computes up * silu(gate), with each row's expert selected by group offsets.

from functools import lru_cache
import cutlass
import cutlass.utils
import cuda.bindings.driver as cuda
from cutlass import cute
from cutlass.utils import SmemAllocator

# ---- problem constants (const axes of the definition) -----------------------
E = 128  # experts
I = 768  # FC1 output features
K = 2048  # FC1 reduction depth
RMAX = 4096  # declared maximum row count
NROWS_W = E * 2 * I  # total canonical weight rows

# ---- tuning knobs -----------------------------------------------------------
NWARP = 8  # warps per CTA
NTHR = NWARP * 32
NI_CTA = 2  # output columns per CTA
KSPLIT = NWARP // NI_CTA  # warps cooperating on one column's K reduction
NTILE = I // NI_CTA  # 384 column tiles
MC = 8  # rows handled per pass
VEC = 8  # BF16 per 128-bit load
KW = K // KSPLIT  # 512 k per warp
NKS = KW // (32 * VEC)  # 2 unrolled k steps per warp
GRID = 564  # persistent grid: 188 SMs x 3 CTAs


# PTX fragments used by the template.
_POLICY_PTX = "createpolicy.fractional.L2::evict_first.b64 {$w0}, 1.0;"

_LDW_PTX = """{
.reg .b32 q0, q1, q2, q3, t;
ld.global.v4.b32 {q0, q1, q2, q3}, [{$r0}];
shl.b32 t, q0, 16;      mov.b32 {$w0}, t;
and.b32 t, q0, -65536;  mov.b32 {$w1}, t;
shl.b32 t, q1, 16;      mov.b32 {$w2}, t;
and.b32 t, q1, -65536;  mov.b32 {$w3}, t;
shl.b32 t, q2, 16;      mov.b32 {$w4}, t;
and.b32 t, q2, -65536;  mov.b32 {$w5}, t;
shl.b32 t, q3, 16;      mov.b32 {$w6}, t;
and.b32 t, q3, -65536;  mov.b32 {$w7}, t;
}"""


@cute.jit
def _make_policy():
    return cute.arch.inline_ptx(_POLICY_PTX, write_only_types=[cutlass.Int64])


@cute.jit
def _load_w8(addr, pol):
    return cute.arch.inline_ptx(_LDW_PTX, write_only_types=[cutlass.Float32] * VEC, read_only_args=[addr, pol])


@cute.kernel
def frost_moe_swiglu_simt_sm120(
    gX: cute.Tensor,
    wbase: cutlass.Int64,
    upbase: cutlass.Int64,
    gOff: cute.Tensor,
    gOut: cute.Tensor,
    rows: cutlass.Int32,
    gate_expert_stride: cutlass.Int64,
    up_expert_stride: cutlass.Int64,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    pol = _make_policy()
    col = warp // KSPLIT  # which of the NI_CTA columns
    ksp = warp % KSPLIT  # which K slice

    smem = SmemAllocator()
    sOff = smem.allocate_array(cutlass.Int32, E + 1)
    sAct = smem.allocate_array(cutlass.Int32, E)
    sN = smem.allocate_array(cutlass.Int32, 1)
    sRed = smem.allocate_array(cutlass.Float32, KSPLIT * NI_CTA * MC * 2)

    # --- stage the offsets, then compact the non-empty experts (per CTA) -----
    if tid < E:
        sOff[tid] = gOff[tid]
    elif tid == E:
        sOff[tid] = rows
    cute.arch.sync_threads()

    if warp == 0:
        run = cutlass.Int32(0)
        for j in cutlass.range_constexpr(E // 32):
            e = j * 32 + lane
            act = sOff[e + 1] > sOff[e]
            ball = cute.arch.vote_ballot_sync(act)
            idx = cute.arch.popc(ball & ((cutlass.Int32(1) << lane) - 1))
            if act:
                sAct[run + idx] = e
            run = run + cute.arch.popc(ball)
        if lane == 0:
            sN[0] = run
    cute.arch.sync_threads()

    total = sN[0] * NTILE

    acc = cute.make_rmem_tensor((MC, 2), cutlass.Float32)
    wgf = cute.make_rmem_tensor((NKS, VEC), cutlass.Float32)
    wuf = cute.make_rmem_tensor((NKS, VEC), cutlass.Float32)
    xf = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
    ovec = cute.make_rmem_tensor((NI_CTA,), cutlass.BFloat16)

    for witem in cutlass.range(bid, total, GRID):
        slot = witem // NTILE
        tile = witem % NTILE
        e = sAct[slot]
        a = sOff[e]
        b = sOff[e + 1]
        m = b - a
        wrow_g = tile * NI_CTA + col
        wa_g = wbase + cutlass.Int64(e) * gate_expert_stride + cutlass.Int64(wrow_g) * cutlass.Int64(K * 2)
        wa_u = upbase + cutlass.Int64(e) * up_expert_stride + cutlass.Int64(wrow_g) * cutlass.Int64(K * 2)

        nblk = (m + (MC - 1)) // MC
        for rb in cutlass.range(nblk):
            r0 = a + rb * MC

            for ks in cutlass.range_constexpr(NKS):
                kv = ksp * (KW // VEC) + ks * 32 + lane
                koff = cutlass.Int64(kv) * cutlass.Int64(VEC * 2)
                wg = _load_w8(wa_g + koff, pol)
                wu = _load_w8(wa_u + koff, pol)
                for v in cutlass.range_constexpr(VEC):
                    wgf[ks, v] = wg[v]
                    wuf[ks, v] = wu[v]

            cute.arch.sync_threads()
            for mm in cutlass.range_constexpr(MC):
                if r0 + mm < b:
                    g = cutlass.Float32(0.0)
                    u = cutlass.Float32(0.0)
                    row = cute.min(r0 + mm, b - 1)
                    for ks in cutlass.range_constexpr(NKS):
                        kv = ksp * (KW // VEC) + ks * 32 + lane
                        cute.autovec_copy(gX[row, kv, None], xf)
                        xv = xf.load().to(cutlass.Float32)
                        for v in cutlass.range_constexpr(VEC):
                            g = g + xv[v] * wgf[ks, v]
                            u = u + xv[v] * wuf[ks, v]
                    acc[mm, 0] = g
                    acc[mm, 1] = u
                    for p in cutlass.range_constexpr(2):
                        v = acc[mm, p]
                        for off in cutlass.range_constexpr(5):
                            v = v + cute.arch.shuffle_sync_bfly(v, offset=(1 << off))
                        if lane == 0:
                            sRed[((ksp * NI_CTA + col) * MC + mm) * 2 + p] = v
            cute.arch.sync_threads()

            if tid < MC:
                row = r0 + tid
                if row < b:
                    for c in cutlass.range_constexpr(NI_CTA):
                        g01 = sRed[((0 * NI_CTA + c) * MC + tid) * 2 + 0] + sRed[((1 * NI_CTA + c) * MC + tid) * 2 + 0]
                        g23 = sRed[((2 * NI_CTA + c) * MC + tid) * 2 + 0] + sRed[((3 * NI_CTA + c) * MC + tid) * 2 + 0]
                        u01 = sRed[((0 * NI_CTA + c) * MC + tid) * 2 + 1] + sRed[((1 * NI_CTA + c) * MC + tid) * 2 + 1]
                        u23 = sRed[((2 * NI_CTA + c) * MC + tid) * 2 + 1] + sRed[((3 * NI_CTA + c) * MC + tid) * 2 + 1]
                        gv = g01 + g23
                        uv = u01 + u23
                        sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.exp(-gv))
                        ovec[c] = (uv * gv * sig).to(cutlass.BFloat16)
                    cute.autovec_copy(ovec, gOut[row, tile, None])


frost_moe_swiglu_simt_sm120.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def fc1_graph_host(
    xp: cutlass.Int64,
    gp: cutlass.Int64,
    up: cutlass.Int64,
    offp: cutlass.Int64,
    outp: cutlass.Int64,
    rows: cutlass.Int32,
    gate_expert_stride: cutlass.Int64,
    up_expert_stride: cutlass.Int64,
    stream: cuda.CUstream,
):
    px = cute.make_ptr(cutlass.BFloat16, xp, cutlass.AddressSpace.gmem, assumed_align=16)
    po = cute.make_ptr(cutlass.Int32, offp, cutlass.AddressSpace.gmem, assumed_align=16)
    pout = cute.make_ptr(cutlass.BFloat16, outp, cutlass.AddressSpace.gmem, assumed_align=16)
    gX = cute.make_tensor(px, cute.make_layout((RMAX, K // VEC, VEC), stride=(K, VEC, 1)))
    gOff = cute.make_tensor(po, cute.make_layout((E,)))
    gOut = cute.make_tensor(pout, cute.make_layout((RMAX, I // NI_CTA, NI_CTA), stride=(I, NI_CTA, 1)))
    frost_moe_swiglu_simt_sm120(gX, gp, up, gOff, gOut, rows, gate_expert_stride, up_expert_stride).launch(grid=(GRID, 1, 1), block=(NTHR, 1, 1), stream=stream)


@lru_cache(maxsize=1)
def compile():
    from cutlass.cute.runtime import make_fake_stream

    return cute.compile(
        fc1_graph_host,
        *[cutlass.Int64(0) for _ in range(5)],
        cutlass.Int32(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_stream(),
        options="--enable-tvm-ffi --gpu-arch=sm_120a",
    )
