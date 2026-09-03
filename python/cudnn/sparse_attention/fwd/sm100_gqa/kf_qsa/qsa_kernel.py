import math
import operator

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu import warp
from cutlass.cute.typing import Float32, Int32, BFloat16
from cutlass import const_expr

# ---- problem constants ----
HQ = 24
HKV = 2
D = 256
TOPK = 512
G = 4
HPK = HQ // HKV                # 12
MPAD = 16                      # MMA M tile (12 heads + 4 pad)
TILE_E = 8
TILE_T = TILE_E * G            # 32 tokens per tile
NUM_TILES = TOPK // TILE_E     # 64
NBUF = 2
SPAD = 8
KV_COPY_ELEMS = TILE_T * D     # 8192 payload elements per buffer
KV_ELEMS = TILE_T * D
V_ELEMS = TILE_T * D            # swizzled K/V buffers: no padding (swizzle replaces it)
NWARPS = 4
ENT_PER_WARP = TILE_E // NWARPS  # 2 entries owned by each warp per tile
NTHREADS = NWARPS * 32         # 128
DSLICE = D // NWARPS           # 64 head-dims per warp
VEC = 8
QOVEC = 16
NCHUNK = KV_COPY_ELEMS // VEC  # 1024
CHUNKS_PER_TOK = D // VEC       # 32
GITER = (NCHUNK + NTHREADS - 1) // NTHREADS  # 16
QROWS_PER_ITER = NWARPS * 2
QITER = (MPAD + QROWS_PER_ITER - 1) // QROWS_PER_ITER
KVD = HKV * D                  # 512
SCALE = D ** -0.5
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)
NEG = -1.0e30


def convert_layout_acc_mn(acc_layout):
    l = cute.make_layout(acc_layout.shape)
    shape = ((l.shape[0][1], l.shape[1]), (l.shape[0][0], *l.shape[0][2:], l.shape[2]), *l.shape[3:])
    stride = ((l.stride[0][1], l.stride[1]), (l.stride[0][0], *l.stride[0][2:], l.stride[2]), *l.stride[3:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def make_acc_tensor_mn_view(acc):
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


@cute.jit
def convert_layout_acc_frgA(acc_layout):
    l = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        ((l.shape[0], l.shape[2][0]), l.shape[1], l.shape[2][1]),
        stride=((l.stride[0], l.stride[2][0]), l.stride[1], l.stride[2][1]),
    )


@cute.jit
def warp_reduce(val, op, width=32):
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def warp_reduce4_max(val):
    val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=1))
    val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=2))
    return val


@cute.jit
def warp_reduce4_sum(val):
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=2)
    return val


def transpose_view(a):
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def make_swizzle_layout():
    # fa4 get_smem_layout_atom for bf16, k_dim=DSLICE=64: swizzle(3,3,3) over (8,64).
    # tile_to_shape replicates the 64-wide atom across D=256 so each warp's
    # [:, 64w:64w+64] slice is an independently-swizzled (TILE_T,64) sub-tile.
    atom = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3),
        0,
        cute.make_ordered_layout((8, DSLICE), order=(1, 0)),
    )
    return cute.tile_to_shape(atom, (TILE_T, D), order=(1, 0))


@cute.kernel
def qsa_tc_kernel(
    gQ: cute.Tensor, gK: cute.Tensor, gV: cute.Tensor, gIdx: cute.Tensor,
    gO: cute.Tensor, gLse: cute.Tensor, NE: Int32,
    USE_EXPLICIT: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    pos, kvh, _ = cute.arch.block_idx()
    warp_id = tidx // 32
    lane = tidx % 32

    smem = SmemAllocator()
    sIdx = smem.allocate_tensor(Int32, cute.make_layout((TOPK,)), byte_alignment=16)
    sKf = smem.allocate_tensor(BFloat16, cute.make_layout((NBUF * KV_ELEMS,)), byte_alignment=16)
    sVf = smem.allocate_tensor(BFloat16, cute.make_layout((NBUF * V_ELEMS,)), byte_alignment=16)
    sw_layout = make_swizzle_layout()
    sSf = smem.allocate_tensor(BFloat16, cute.make_layout((MPAD * (TILE_T + SPAD),)), byte_alignment=16)
    # sQ aliases K ring-buffer slot #1 (MPAD*D=4096 <= V_ELEMS=8192). Slot #1 is first
    # written by the tile-0 prefetch-next inside the loop, which runs strictly after the
    # sync_threads at the loop head, i.e. after Q has been consumed into the MMA A-fragment
    # in the prologue. No live-range conflict; saves 8KB -> 3 CTAs/SM instead of 2.
    sQ = cute.make_tensor(sKf.iterator + V_ELEMS, cute.make_layout((MPAD, D), stride=(D, 1)))
    sS = cute.make_tensor(sSf.iterator, cute.make_layout((MPAD, TILE_T), stride=(TILE_T + SPAD, 1)))  # BF16 score exchange
    Sdim = gK.shape[0]
    gKf = cute.make_tensor(gK.iterator, cute.make_layout((Sdim * KVD,)))
    gVf = cute.make_tensor(gV.iterator, cute.make_layout((Sdim * KVD,)))
    kvoff = kvh * D
    idx_mask = NE - Int32(1)

    # stage index list
    for it in cutlass.range_constexpr((TOPK + NTHREADS - 1) // NTHREADS):
        e = tidx + it * NTHREADS
        sIdx[e] = gIdx[pos, e] & idx_mask
    # stage Q (12 real heads -> rows 0..11; pad rows 12..15 = 0)
    for it in cutlass.range_constexpr(QITER):
        r = it * QROWS_PER_ITER + warp_id * 2 + lane // 16
        dd = (lane % 16) * QOVEC
        if r < Int32(HPK):
            head = kvh * HPK + r
            vec = (gQ.iterator.raw_ptr() + (pos * HQ + head) * D + dd).load(32, count=QOVEC)
            (sQ.iterator.raw_ptr() + r * D + dd).store(vec, alignment=32)
        else:
            for e in cutlass.range_constexpr(QOVEC):
                sQ[r, dd + e] = BFloat16(0.0)
    cute.arch.sync_threads()

    op = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
    # cooperative QK: 4 warps split N (TILE_T) -> each warp computes 8 of 32 cols
    tiled_mma_qk = cute.make_tiled_mma(op, (1, NWARPS, 1), permutation_mnk=(16, NWARPS * 8, 16))
    # single-warp mma used for reload of full S and for PV feed
    tiled_mma_full = cute.make_tiled_mma(op, (1, 1, 1), permutation_mnk=(16, 16, 16))
    # cooperative PV: 4 warps split N (=D) -> each warp computes 64 of 256 d-cols
    tiled_mma_pv = cute.make_tiled_mma(op, (1, NWARPS, 1), permutation_mnk=(16, NWARPS * 8, 16))
    thr_qk = tiled_mma_qk.get_slice(tidx)
    thr_full = tiled_mma_full.get_slice(lane)
    thr_pv = tiled_mma_pv.get_slice(tidx)

    # Q fragment for cooperative QK (A not N-split -> replicated across warps)
    tSrQ = thr_qk.make_fragment_A(thr_qk.partition_A(sQ))
    cute.autovec_copy(thr_qk.partition_A(sQ), tSrQ)

    acc_O = cute.make_rmem_tensor(thr_pv.partition_shape_C((MPAD, D)), Float32)
    acc_O.fill(0.0)
    acc_O_mn = make_acc_tensor_mn_view(acc_O)
    num_rows = cutlass.const_expr(cute.size(acc_O_mn, mode=[0]))

    row_max = cute.make_rmem_tensor(cute.make_layout(num_rows), Float32)
    row_sum = cute.make_rmem_tensor(cute.make_layout(num_rows), Float32)
    for r in cutlass.range_constexpr(num_rows):
        row_max[r] = Float32(NEG)
        row_sum[r] = Float32(0.0)

    qs = Float32(SCALE * LOG2E)

    # ---- prefetch tile 0 gather into buffer 0 ----
    # Each warp owns whole sparse entries; load sIdx once per entry and reuse it
    # across the four contiguous tokens selected by that entry.
    dd = lane * VEC
    lane_low = lane & Int32(7)
    c_base = (warp_id << 11) + ((lane >> 3) << 9)
    for el in cutlass.range_constexpr(ENT_PER_WARP):
        eidx = warp_id * ENT_PER_WARP + el
        ent = sIdx[eidx]
        for toff in cutlass.range_constexpr(G):
            row_in = Int32(el * G + toff)
            c = c_base + (row_in << 6) + ((lane_low ^ row_in) << 3)
            src = (ent * G + toff) * KVD + kvoff + dd
            cute.arch.cp_async_shared_global(sKf.iterator + c, gKf.iterator + src, 16, "cg")
            cute.arch.cp_async_shared_global(sVf.iterator + c, gVf.iterator + src, 16, "cg")
    cute.arch.cp_async_commit_group()

    for tile in cutlass.range(NUM_TILES):
        cur = (tile % NBUF) * V_ELEMS
        # prefetch NEXT tile
        if tile + 1 < Int32(NUM_TILES):
            nxt = ((tile + 1) % NBUF) * KV_ELEMS
            ebase_n = (tile + 1) * TILE_E
            for el in cutlass.range_constexpr(ENT_PER_WARP):
                eidx = warp_id * ENT_PER_WARP + el
                ent = sIdx[ebase_n + eidx]
                for toff in cutlass.range_constexpr(G):
                    row_in = Int32(el * G + toff)
                    c = c_base + (row_in << 6) + ((lane_low ^ row_in) << 3)
                    src = (ent * G + toff) * KVD + kvoff + dd
                    cute.arch.cp_async_shared_global(sKf.iterator + nxt + c, gKf.iterator + src, 16, "cg")
                    cute.arch.cp_async_shared_global(sVf.iterator + nxt + c, gVf.iterator + src, 16, "cg")
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        sK = cute.make_tensor(sKf.iterator + cur, sw_layout)            # (TILE_T, D) swizzled
        sV_full = cute.make_tensor(sVf.iterator + cur, sw_layout)       # (TILE_T, D) swizzled
        sVt = transpose_view(sV_full)                                   # (D, TILE_T) swizzled, full

        # ---- GEMM1 cooperative: S = Q @ K^T  [16, TILE_T], warps split N ----
        tSrK = thr_qk.make_fragment_B(thr_qk.partition_B(sK))
        cute.autovec_copy(thr_qk.partition_B(sK), tSrK)
        acc_S_c = cute.make_rmem_tensor(thr_qk.partition_shape_C((MPAD, TILE_T)), Float32)
        acc_S_c.fill(0.0)
        for kb in cutlass.range_constexpr(cute.size(tSrK.shape[2])):
            cute.gemm(tiled_mma_qk, acc_S_c, tSrQ[None, None, kb], tSrK[None, None, kb], acc_S_c)
        acc_S_c.store(acc_S_c.load() * qs)
        # scatter partial S to smem as BF16 (halves softmax-roundtrip smem traffic),
        # then reload full [16,32] per warp and upcast to fp32 for softmax.
        acc_S_c_bf = cute.make_fragment_like(acc_S_c, BFloat16)
        acc_S_c_bf.store(acc_S_c.load().to(BFloat16))
        cute.autovec_copy(acc_S_c_bf, thr_qk.partition_C(sS))
        cute.arch.sync_threads()
        acc_S_bf = cute.make_rmem_tensor(thr_full.partition_shape_C((MPAD, TILE_T)), BFloat16)
        cute.autovec_copy(thr_full.partition_C(sS), acc_S_bf)
        acc_S = cute.make_rmem_tensor(thr_full.partition_shape_C((MPAD, TILE_T)), Float32)
        acc_S.store(acc_S_bf.load().to(Float32))

        # ---- online softmax over N (TILE_T), quad (width=4) reduction ----
        acc_S_mn = make_acc_tensor_mn_view(acc_S)
        for r in cutlass.range_constexpr(num_rows):
            row = acc_S_mn[r, None].load()
            m_local = row.reduce(cute.ReductionOp.MAX, Float32(NEG), 0)
            if cutlass.const_expr(USE_EXPLICIT):
                m_cur = warp_reduce4_max(m_local)
            else:
                m_cur = warp_reduce(m_local, cute.arch.fmax, width=4)
            m_old = row_max[r]
            m_new = cute.arch.fmax(m_old, m_cur)
            row_max[r] = m_new
            corr = cute.arch.exp2(m_old - m_new)
            p = cute.math.exp2(row - m_new, fastmath=True)
            psum_local = p.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            row_sum[r] = row_sum[r] * corr + psum_local
            acc_S_mn[r, None].store(p)
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * corr)

        # ---- GEMM2: O += P @ V  (split-D across warps), ldmatrix.trans on swizzled V ----
        rP = cute.make_fragment_like(acc_S, BFloat16)
        rP.store(acc_S.load().to(BFloat16))
        tOrP = cute.make_tensor(rP.iterator, convert_layout_acc_frgA(rP.layout))
        tOrVt = thr_pv.make_fragment_B(thr_pv.partition_B(sVt))
        cute.autovec_copy(thr_pv.partition_B(sVt), tOrVt)
        for kb in cutlass.range_constexpr(cute.size(tOrVt.shape[2])):
            cute.gemm(tiled_mma_pv, acc_O, tOrP[None, None, kb], tOrVt[None, None, kb], acc_O)

    # ---- finalize ----
    for r in cutlass.range_constexpr(num_rows):
        if cutlass.const_expr(USE_EXPLICIT):
            row_sum[r] = warp_reduce4_sum(row_sum[r])
        else:
            row_sum[r] = warp_reduce(row_sum[r], operator.add, width=4)
    for r in cutlass.range_constexpr(num_rows):
        inv = cute.arch.rcp_approx(row_sum[r])
        acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)

    sO = cute.make_tensor(sKf.iterator, cute.make_layout((MPAD, D), stride=(D, 1)))
    cute.arch.sync_threads()
    rO = cute.make_fragment_like(acc_O, BFloat16)
    rO.store(acc_O.load().to(BFloat16))
    # cooperative acc_O covers full (MPAD, D); partition_C maps each warp's N-slice
    cute.autovec_copy(rO, thr_pv.partition_C(sO))
    cute.arch.sync_threads()
    OCHUNK = HPK * D // QOVEC
    for it in cutlass.range_constexpr((OCHUNK + NTHREADS - 1) // NTHREADS):
        chunk = tidx + it * NTHREADS
        if chunk < Int32(OCHUNK):
            r = chunk // Int32(D // QOVEC)
            dd = (chunk % Int32(D // QOVEC)) * QOVEC
            head = kvh * HPK + r
            vec = (sO.iterator.raw_ptr() + r * D + dd).load(32, count=QOVEC)
            (gO.iterator.raw_ptr() + (pos * HQ + head) * D + dd).store(vec, alignment=32)

    if warp_id == Int32(0):
        for r in cutlass.range_constexpr(num_rows):
            realrow = (lane // 4) + 8 * r
            if (lane % 4 == Int32(0)) and (realrow < Int32(HPK)):
                lse_val = row_max[r] * Float32(LN2) + cute.math.log(row_sum[r])
                gLse[pos, kvh * HPK + realrow] = lse_val


@cute.jit
def qsa_launch(gQ, gK, gV, gIdx, gO, gLse, USE_EXPLICIT: cutlass.Constexpr[bool]):
    S = gQ.shape[0]
    NE = Int32(S // G)
    qsa_tc_kernel(gQ, gK, gV, gIdx, gO, gLse, NE, USE_EXPLICIT).launch(
        grid=(S, HKV, 1), block=(NTHREADS, 1, 1),
        preferred_smem_carveout=100,
    )


_compiled = None
_key = None


def run(q, k, v, idxs, out, lse):
    global _compiled, _key
    cq = from_dlpack(q, assumed_align=32)
    ck = from_dlpack(k, assumed_align=32)
    cv = from_dlpack(v, assumed_align=32)
    ci = from_dlpack(idxs, assumed_align=32)
    co = from_dlpack(out, assumed_align=32)
    cl = from_dlpack(lse, assumed_align=32)
    seq = int(q.shape[0])
    use_explicit = seq == 8192
    key = (seq, use_explicit)
    if _key != key:
        _compiled = cute.compile(qsa_launch, cq, ck, cv, ci, co, cl, use_explicit)
        _key = key
    _compiled(cq, ck, cv, ci, co, cl)
