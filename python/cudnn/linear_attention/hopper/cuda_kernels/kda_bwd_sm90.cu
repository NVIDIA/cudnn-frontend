// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Kimi Delta Attention (KDA) chunked BACKWARD for Hopper (sm90). Compiled with
// NVRTC and launched through the driver API, following linear_attention/cake/.
//
// Provenance: Kernel Factory campaign tefjfnjwss3f75c05erc48mc14, solution
// kernel 6a040e80 (cuda_cpp). Machine-generated; included because it is
// measurably correct and fast, not because it was reviewed line by line.
//
// Verified on H100 80GB HBM3 (SXM) at the production gate (gate_lower_bound
// = -5) with a non-zero initial_state AND d_final_state, all six gradients
// scored by the campaign definition's three-term checker (per-element, hard
// cap, RMS): geomean 316.2 us against the cuTile backward's 1765.4 us =
// 5.58x per-call, 311.0 vs 1357.8 = 4.37x pipelined, 6 of 6 shapes correct
// and 6 of 6 faster.
//
// Assembled from the campaign's multi-file nvcc project by vendor_bwd_kernel.py:
// common.cuh + k_prep.cuh + k_scan.cuh + k_bwd.cuh concatenated in dependency
// order, plus k_meta from kernel.cu and the explicit template instantiations.
// The host launcher (kda_bwd_launch / kda_bwd_workspace_bytes) is NOT vendored
// -- NVRTC compiles device code only -- and is reproduced in cuda_bwd_host.py.
// No device code is modified.

// NVRTC ships no host C++ headers, so the fixed-width types the bodies use --
// and which nvcc supplied through the <cstdint>/<cuda_runtime.h> chain -- are
// declared here. Same set and same reason as the forward body.
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

#include <cuda_bf16.h>
#include <mma.h>

// ==================== common.cuh ====================
#pragma once

#define DEV __device__ __forceinline__

typedef __nv_bfloat16 bf16;

static constexpr int DH        = 128;  // head dim (K == V == 128)
static constexpr int BT        = 64;   // chunk length
static constexpr int BCB       = 16;   // anchor sub-block
static constexpr int NSB       = BT / BCB;
static constexpr float RCP_LN2 = 1.4426950408889634f;
static constexpr float KSCALE  = 0.08838834764831845f;  // 1/sqrt(128)
// 136 words spreads 8 mma rows x 4 column pairs over all 32 banks; a 132-word
// stride puts rows r and r+1 four banks apart, so the 16-lane phase of every
// LDS.64 at mma coordinates is 2-way conflicted.
static constexpr int GST = 136;  // padded fp32 SMEM row stride (bank spread)

// ---------------------------------------------------------------- smem utils
// bf16 tile stored row-major [ROWS][COLS] with the classic 128B xor swizzle.
// COLS must be a multiple of 64.
DEV int
swz(int r, int c, int COLS) {
    return r * COLS + ((((c >> 3) ^ (r & 7)) << 3) | (c & 7));
}

DEV uint32_t
sad(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEV void
ldm_x4(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
DEV void
ldm_x4t(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
DEV void
ldm_x2(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
DEV void
ldm_x2t(uint32_t (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
// Not volatile.  mma has no memory side effects: every operand is a register and
// the constraints describe its dependencies exactly.  ldmatrix must stay
// volatile (it reads shared memory that is not in its operand list), and two
// volatile asm statements may not be reordered against each other -- so leaving
// mma volatile pins every mma directly behind its own ldmatrix and pays the full
// shared-load latency on each one.  Dropping it lets ptxas hoist a k-step's
// loads ahead of its mmas, which at 16 warps per SM is the only slack available.
DEV void
mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

DEV void
cpasync16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
// Predicated form: src-size 0 zero-fills the destination and performs no global
// access, so out-of-range rows cost a select instead of a branch + vector store.
DEV void
cpasync16z(uint32_t dst, const void* src, bool full) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(full ? 16 : 0));
}
// One cvt.rn.bf16x2.f32; the __floats2bfloat162_rn wrapper costs 2-3.
DEV uint32_t
bf2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// bf16 -> fp32 is exactly a 16-bit left shift (bf16 is the fp32 high half), so
// a packed pair unpacks in two integer ops instead of four convert+extract.
DEV float2
un2(uint32_t p) {
    float2 r;
    r.x = __int_as_float(p << 16);
    r.y = __int_as_float(p & 0xffff0000u);
    return r;
}
DEV float
unbf(bf16 v) {
    return __int_as_float((uint32_t)(*reinterpret_cast<const uint16_t*>(&v)) << 16);
}
DEV void
cpasync_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
DEV void
cpasync_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---------------------------------------------------------------- tiling
// Output C[M][N] is tiled into (M/16) x (N/8) mma tiles, distributed over NW
// warps, TPW tiles each.  Tile index t = mi*(N/8) + ni, warp w owns
// t in [w*TPW, (w+1)*TPW).
template <int M, int N, int NW>
struct Tiling {
    static constexpr int NTM = M / 16;
    static constexpr int NTN = N / 8;
    static constexpr int TOT = NTM * NTN;
    static constexpr int TPW = TOT / NW;
    static_assert(TOT % NW == 0, "tile count must divide warp count");
    static_assert(TPW >= NTN ? (TPW % NTN == 0) : (NTN % TPW == 0), "bad shape");
    static DEV int
    mi(int w, int j) {
        return (TPW >= NTN) ? (w * (TPW / NTN) + j / NTN) : (w / (NTN / TPW));
    }
    static DEV int
    ni(int w, int j) {
        return (TPW >= NTN) ? (j % NTN) : ((w % (NTN / TPW)) * TPW + j);
    }
    // element (row,col) held in register `r` of tile `j`
    static DEV int
    row(int w, int lane, int j, int r) {
        return mi(w, j) * 16 + (lane >> 2) + ((r & 2) << 2);
    }
    static DEV int
    col(int w, int lane, int j, int r) {
        return ni(w, j) * 8 + ((lane & 3) << 1) + (r & 1);
    }
};

// ---------------------------------------------------------------- ldmatrix addressing
// Closed form of the 128B xor swizzle for ldmatrix operands.
//
// swz(r,c,COLS) = r*COLS + ((((c>>3)^(r&7))<<3) | (c&7)).  Every ldmatrix
// operand below has c&7 == 0, and the index that is *not* the k-step always
// moves in multiples of 8, so (r&7) reduces to (lane&7) and the swizzled unit
// index becomes an xor of the moving index with a lane-constant.  The whole
// byte address then collapses to
//     lane_base + <compile-time linear term> + (compile-time ^ lane-constant)
// i.e. one LOP3 and one IADD per ldmatrix, instead of the shift/xor/mad chain
// the generic expression compiles to.  Operand addressing was the single
// largest instruction group in all three kernels (IMAD alone was 18.8% of the
// scan's instructions and 11.8% of its stalls).
struct LdA {
    uint32_t base;  // tile base + lane-dependent row offset
    uint32_t x;     // lane-constant xor term
};
// AM=0: A[M][K] k-major, x4        addr = base + mi*(32*AC) + ((kt<<5) ^ x)
template <int AC>
DEV LdA
a0_addr(const bf16* p, int lane) {
    return {sad(p) + (uint32_t)(2 * AC * (lane & 15)), (uint32_t)((((lane >> 4) & 1) ^ (lane & 7)) << 4)};
}
// AM=1: A[K][M] m-major, x4.trans  addr = base + kt*(32*AC) + a1x<MOFF>(x,mi)
template <int AC>
DEV LdA
a1_addr(const bf16* p, int lane) {
    return {sad(p) + (uint32_t)(2 * AC * (((lane >> 4) & 1) * 8 + (lane & 7))),
            (uint32_t)(((lane >> 3) & 1) ^ (lane & 7))};
}
template <int MOFF>
DEV uint32_t
a1x(uint32_t x, int mi) {
    return (uint32_t)((((2 * mi + (MOFF >> 3)) ^ (int)x)) << 4);
}
// BM=0: B[N][K] k-major, x2        addr = base + ni*(16*BC) + ((kt<<5) ^ x)
template <int BC>
DEV LdA
b0_addr(const bf16* p, int lane) {
    return {sad(p) + (uint32_t)(2 * BC * (lane & 7)), (uint32_t)((((lane >> 3) & 1) ^ (lane & 7)) << 4)};
}
// BM=1: B[K][N] n-major, x2.trans  addr = base + kt*(32*BC) + ((ni ^ x) << 4)
template <int BC>
DEV LdA
b1_addr(const bf16* p, int lane) {
    return {sad(p) + (uint32_t)(2 * BC * (((lane >> 3) & 1) * 8 + (lane & 7))), (uint32_t)(lane & 7)};
}
// Paired forms: one ldmatrix.x4 fetches the B fragments of an adjacent n-tile
// pair (ni even), so the four 8x8 matrices are {ni,k-lo}, {ni,k-hi},
// {ni+1,k-lo}, {ni+1,k-hi}.  Same bytes as two x2 loads, half the LSU
// instructions -- B addressing was the largest instruction group in the GEMMs.
// Lanes 16-31 address the second tile; (lane&7) and ((lane>>3)&1) are unchanged
// by that, so the xor swizzle term and the k-half selector both still hold.
template <int BC>
DEV LdA
b0_addr_p(const bf16* p, int lane) {
    return {sad(p) + (uint32_t)(2 * BC * ((lane & 7) + (((lane >> 4) & 1) << 3))),
            (uint32_t)((((lane >> 3) & 1) ^ (lane & 7)) << 4)};
}
template <int BC>
DEV LdA
b1_addr_p(const bf16* p, int lane) {
    return {sad(p) + (uint32_t)(2 * BC * (((lane >> 3) & 1) * 8 + (lane & 7))),
            (uint32_t)((lane & 7) ^ ((lane >> 4) & 1))};
}
DEV void
mma16816p(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// An n-tile pair is contiguous and starts even whenever TPW is even and either
// the warp owns a contiguous column range (TPW < NTN) or NTN is itself even.
template <class T>
struct BPair {
    static constexpr bool on = (T::TPW % 2 == 0) && (T::TPW < T::NTN || T::NTN % 2 == 0);
};

// C[M][N] += opA(A) * opB(B), fp32 accumulators in registers.
// AM=0: sA stored [M][K] (k-major).      AM=1: sA stored [K][M] (m-major, .trans)
// BM=0: sB stored [N][K] (k-major, NT).  BM=1: sB stored [K][N] (n-major, .trans)
// AC / BC are the physical row widths (in elements) of sA / sB.
template <int M, int N, int K, int NW, int AM, int BM, int AC, int BC, int MOFF = 0>
DEV void
gemm_acc(float (*acc)[4], const bf16* sA, const bf16* sB, int w, int lane) {
    using T          = Tiling<M, N, NW>;
    constexpr int ST = BPair<T>::on ? 2 : 1;
    const LdA A      = (AM == 0) ? a0_addr<AC>(sA, lane) : a1_addr<AC>(sA, lane);
    const LdA B      = BPair<T>::on ? ((BM == 0) ? b0_addr_p<BC>(sB, lane) : b1_addr_p<BC>(sB, lane))
                                    : ((BM == 0) ? b0_addr<BC>(sB, lane) : b1_addr<BC>(sB, lane));
    uint32_t af[4], bf_[4];
#pragma unroll
    for (int kt = 0; kt < K / 16; ++kt) {
#pragma unroll
        for (int j = 0; j < T::TPW; j += ST) {
            const int mi = T::mi(w, j), ni = T::ni(w, j);
            if (j == 0 || T::mi(w, j - ST) != mi) {
                if (AM == 0)
                    ldm_x4(af, A.base + mi * (32u * AC) + ((kt << 5) ^ A.x));
                else
                    ldm_x4t(af, A.base + kt * (32u * AC) + a1x<MOFF>(A.x, mi));
            }
            if (ST == 2) {
                if (BM == 0)
                    ldm_x4(bf_, B.base + ni * (16u * BC) + ((kt << 5) ^ B.x));
                else
                    ldm_x4t(bf_, B.base + kt * (32u * BC) + ((ni ^ B.x) << 4));
                mma16816p(acc[j], af, bf_[0], bf_[1]);
                mma16816p(acc[j + 1], af, bf_[2], bf_[3]);
            } else {
                uint32_t b2[2];
                if (BM == 0)
                    ldm_x2(b2, B.base + ni * (16u * BC) + ((kt << 5) ^ B.x));
                else
                    ldm_x2t(b2, B.base + kt * (32u * BC) + ((ni ^ B.x) << 4));
                mma16816(acc[j], af, b2);
            }
        }
    }
}

// 64x64x64 GEMM restricted to a warp-uniform interval of 16-wide K tiles.
// Triangular operands make all products outside [kt_begin, kt_end) exactly zero.
template <int AM, int BM>
DEV void
gemm_acc64_range(float (*acc)[4], const bf16* sA, const bf16* sB, int w, int lane, int kt_begin, int kt_end) {
    using T = Tiling<64, 64, 16>;
    static_assert(BPair<T>::on, "range gemm expects an even tile pair");
    const LdA A = (AM == 0) ? a0_addr<64>(sA, lane) : a1_addr<64>(sA, lane);
    const LdA B = (BM == 0) ? b0_addr_p<64>(sB, lane) : b1_addr_p<64>(sB, lane);
    uint32_t af[4], bf_[4];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
        if (kt < kt_begin || kt >= kt_end) continue;
#pragma unroll
        for (int j = 0; j < T::TPW; j += 2) {
            const int mi = T::mi(w, j), ni = T::ni(w, j);
            if (j == 0 || T::mi(w, j - 2) != mi) {
                if (AM == 0)
                    ldm_x4(af, A.base + mi * (32u * 64) + ((kt << 5) ^ A.x));
                else
                    ldm_x4t(af, A.base + kt * (32u * 64) + a1x<0>(A.x, mi));
            }
            if (BM == 0)
                ldm_x4(bf_, B.base + ni * (16u * 64) + ((kt << 5) ^ B.x));
            else
                ldm_x4t(bf_, B.base + kt * (32u * 64) + ((ni ^ B.x) << 4));
            mma16816p(acc[j], af, bf_[0], bf_[1]);
            mma16816p(acc[j + 1], af, bf_[2], bf_[3]);
        }
    }
}

// Two independent C accumulators with distinct A operands and one shared B.
// This preserves each MMA chain's order while loading every B fragment once.
template <int M, int N, int K, int NW, int AM, int BM, int AC, int BC, int MOFF = 0>
DEV void
gemm_acc2(float (*acc0)[4], float (*acc1)[4], const bf16* sA0, const bf16* sA1, const bf16* sB, int w, int lane) {
    using T = Tiling<M, N, NW>;
    static_assert(BPair<T>::on, "paired gemm expects an even tile pair");
    const LdA A0       = (AM == 0) ? a0_addr<AC>(sA0, lane) : a1_addr<AC>(sA0, lane);
    const uint32_t d10 = sad(sA1) - sad(sA0);  // same swizzled offset
    const LdA B        = (BM == 0) ? b0_addr_p<BC>(sB, lane) : b1_addr_p<BC>(sB, lane);
    uint32_t af0[4], af1[4], bf_[4];
#pragma unroll
    for (int kt = 0; kt < K / 16; ++kt) {
#pragma unroll
        for (int j = 0; j < T::TPW; j += 2) {
            const int mi = T::mi(w, j), ni = T::ni(w, j);
            if (j == 0 || T::mi(w, j - 2) != mi) {
                const uint32_t aoff = (AM == 0) ? (A0.base + mi * (32u * AC) + ((kt << 5) ^ A0.x))
                                                : (A0.base + kt * (32u * AC) + a1x<MOFF>(A0.x, mi));
                if (AM == 0) {
                    ldm_x4(af0, aoff);
                    ldm_x4(af1, aoff + d10);
                } else {
                    ldm_x4t(af0, aoff);
                    ldm_x4t(af1, aoff + d10);
                }
            }
            if (BM == 0)
                ldm_x4(bf_, B.base + ni * (16u * BC) + ((kt << 5) ^ B.x));
            else
                ldm_x4t(bf_, B.base + kt * (32u * BC) + ((ni ^ B.x) << 4));
            mma16816p(acc0[j], af0, bf_[0], bf_[1]);
            mma16816p(acc1[j], af1, bf_[0], bf_[1]);
            mma16816p(acc0[j + 1], af0, bf_[2], bf_[3]);
            mma16816p(acc1[j + 1], af1, bf_[2], bf_[3]);
        }
    }
}

template <int NACC>
DEV void
zero_acc(float (&acc)[NACC][4]) {
#pragma unroll
    for (int j = 0; j < NACC; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
}

// ---------------------------------------------------------------- misc
DEV float
ex2(float x) {
    return exp2f(x);
}

// A single packed sequence has an arithmetic chunk mapping, so its steady
// state does not need the serial metadata kernel or metadata loads.
DEV int
chunk_t0(int cs, const int* cs_t0, bool single_sequence) {
    return single_sequence ? cs * BT : cs_t0[cs];
}
DEV int
chunk_len(int cs, const int* cs_len, int T, bool single_sequence) {
    return single_sequence ? min(BT, T - cs * BT) : cs_len[cs];
}
// FULL is set by the host when every chunk of the launch is exactly BT tokens
// (a single packed sequence whose length is a multiple of BT).  The length then
// folds to a constant and every tail predicate -- the cp.async zero-fill masks,
// the store guards, the (L-1) gate row -- disappears from the inner loops.
template <bool FULL>
DEV int
chunk_len_t(int cs, const int* cs_len, int T, bool single_sequence) {
    return FULL ? BT : (single_sequence ? min(BT, T - cs * BT) : cs_len[cs]);
}

// 16B-granular global->shared copy of a [rows][128] bf16 tile (row stride
// `gstride` elements), zero-filling rows >= valid.
template <int ROWS, int COLS>
DEV void
load_tile(bf16* dst, const bf16* src, int gstride, int valid, int tid, int nthr) {
    constexpr int UNITS = ROWS * (COLS / 8);
#pragma unroll
    for (int u = tid; u < UNITS; u += nthr) {
        const int r = u / (COLS / 8), c8 = u % (COLS / 8);
        bf16* d = dst + swz(r, c8 * 8, COLS);
        if (r < valid) {
            cpasync16(sad(d), src + (size_t)r * gstride + c8 * 8);
        } else {
            *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
        }
    }
}

// Precomputed per-thread plan for a 16B-granular [ROWS][COLS] bf16 tile copy
// into a swizzled [ROWS][COLS] shared tile driven by NTHR threads.  Successive
// units for one thread are NTHR/(COLS/8) rows apart -- a multiple of 8 -- so the
// xor swizzle term is invariant and both the shared byte offset and the global
// element offset advance by compile-time constants.  Computing the plan once
// replaces the per-unit swizzle + 64-bit address arithmetic, which the profile
// showed to be the single largest instruction group in the scan.
template <int ROWS, int COLS, int NTHR>
struct TilePlan {
    static constexpr int C8         = COLS / 8;
    static constexpr int ITER       = (ROWS * C8) / NTHR;
    static constexpr int RSTEP      = NTHR / C8;
    static constexpr uint32_t SSTEP = (uint32_t)RSTEP * COLS * 2;
    static_assert(RSTEP % 8 == 0, "row step must preserve the xor swizzle term");
    int row;
    uint32_t soff;
    size_t goff, gstep;
    DEV void
    init(int tid, size_t gstride) {
        row         = tid / C8;
        const int c = (tid % C8) * 8;
        soff        = (uint32_t)(2 * swz(row, c, COLS));
        goff        = (size_t)row * gstride + c;
        gstep       = (size_t)RSTEP * gstride;
    }
    DEV void
    run(uint32_t dstbase, const bf16* src, int valid) const {
        const bf16* s = src + goff;
#pragma unroll
        for (int i = 0; i < ITER; ++i, s += gstep) cpasync16z(dstbase + soff + i * SSTEP, s, row + i * RSTEP < valid);
    }
};

// ---------------------------------------------------------------- vector8
// 8-element (one 16B swizzle unit) helpers so the elementwise passes issue
// one wide load/store instead of eight scalar ones.
struct F8 {
    float v[8];
};

DEV F8
ld8f(const float* p) {
    F8 r;
    *reinterpret_cast<float4*>(r.v)     = *reinterpret_cast<const float4*>(p);
    *reinterpret_cast<float4*>(r.v + 4) = *reinterpret_cast<const float4*>(p + 4);
    return r;
}
DEV F8
ld8bf(const bf16* p) {
    uint4 w        = *reinterpret_cast<const uint4*>(p);
    const float2 a = un2(w.x), b = un2(w.y), c = un2(w.z), d = un2(w.w);
    F8 r;
    r.v[0] = a.x;
    r.v[1] = a.y;
    r.v[2] = b.x;
    r.v[3] = b.y;
    r.v[4] = c.x;
    r.v[5] = c.y;
    r.v[6] = d.x;
    r.v[7] = d.y;
    return r;
}
DEV void
st8bf(bf16* p, const F8& x) {
    *reinterpret_cast<uint4*>(p) =
        make_uint4(bf2(x.v[0], x.v[1]), bf2(x.v[2], x.v[3]), bf2(x.v[4], x.v[5]), bf2(x.v[6], x.v[7]));
}
DEV void
st8zero(bf16* p) {
    *reinterpret_cast<uint4*>(p) = make_uint4(0, 0, 0, 0);
}

// ---------------------------------------------------------------- vector4
// 4-element granularity: one fp32 float4 per thread, so consecutive threads
// read consecutive 16B of a plain fp32 plane (the 8-wide form strides by 32B
// and 4-way bank-conflicts).
struct F4 {
    float v[4];
};

DEV F4
ld4f(const float* p) {
    F4 r;
    *reinterpret_cast<float4*>(r.v) = *reinterpret_cast<const float4*>(p);
    return r;
}
DEV F4
ld4bf(const bf16* p) {
    uint2 w        = *reinterpret_cast<const uint2*>(p);
    const float2 a = un2(w.x), b = un2(w.y);
    F4 r;
    r.v[0] = a.x;
    r.v[1] = a.y;
    r.v[2] = b.x;
    r.v[3] = b.y;
    return r;
}
DEV void
st4bf(bf16* p, const F4& x) {
    *reinterpret_cast<uint2*>(p) = make_uint2(bf2(x.v[0], x.v[1]), bf2(x.v[2], x.v[3]));
}
DEV void
st4zero(bf16* p) {
    *reinterpret_cast<uint2*>(p) = make_uint2(0, 0);
}

// ==================== k_prep.cuh ====================
#pragma once

// ---------------------------------------------------------------------------
// K1 "prep": one CTA per (chunk-slot, head).  Fully parallel.
// Produces  Amat = (I+M)^-1, wneg = -A*kb, u = A*(v*beta), kg, qg, gn2,
//           dv_local = tril(Aqk)^T * dO.
// 512 threads: the Aqk and M products run on two disjoint halves of the CTA.
// ---------------------------------------------------------------------------
static constexpr int PNT = 512;
static constexpr int MST = 67;  // spread FP32 triangular rows across SMEM banks
// The four anchored k operands of the Aqk / M sweep are held at once (64 KB)
// instead of one tile rebuilt per sub-block: the sweep then runs without a
// single barrier between its four products, and they overlap freely.
static constexpr int SMEM_PREP = 16384 * 4 + 65536 + 8192 * 2 + 64 * GST * 4 + (64 * MST + 64 * 64) * 4 + 256 + 4096;

template <bool FULL>
__global__
__launch_bounds__(PNT, 1) void k_prep(const bf16* __restrict__ q,
                                      const bf16* __restrict__ kk,
                                      const bf16* __restrict__ vv,
                                      const float* __restrict__ gg,
                                      const float* __restrict__ beta,
                                      const bf16* __restrict__ dO,
                                      const int* __restrict__ cs_t0,
                                      const int* __restrict__ cs_len,
                                      bf16* __restrict__ Amat,
                                      bf16* __restrict__ wneg,
                                      bf16* __restrict__ uu,
                                      bf16* __restrict__ kg,
                                      bf16* __restrict__ Pg,
                                      float* __restrict__ gn2,
                                      bf16* __restrict__ dvl,
                                      int T,
                                      int H,
                                      bool single_sequence) {
    const int cs = blockIdx.x, h = blockIdx.y;
    const int L = chunk_len_t<FULL>(cs, cs_len, T, single_sequence);
    if (!FULL && L == 0) return;
    const int t0  = chunk_t0(cs, cs_t0, single_sequence);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t gs   = (size_t)H * DH;
    const size_t base = (size_t)t0 * gs + (size_t)h * DH;

    extern __shared__ char smem[];
    bf16* sQ     = reinterpret_cast<bf16*>(smem);            // 16K
    bf16* sK     = sQ + 64 * 128;                            // 16K
    bf16* sB1    = sK + 64 * 128;                            // 16K  qr / kb / vb
    bf16* sB2    = sB1 + 64 * 128;                           // 16K  kr / v
    bf16* sKC    = sB2 + 64 * 128;                           // 64K  4 x kc
    bf16* sB3    = sKC;                                      // 16K  w staging (aliases kc0)
    bf16* sDOt   = sKC + 64 * 128;                           // 16K  dO (aliases kc1)
    bf16* sAqk   = sKC + 4 * 64 * 128;                       // 8K
    bf16* sAb    = sAqk + 64 * 64;                           // 8K
    float* sG    = reinterpret_cast<float*>(sAb + 64 * 64);  // 32K
    float* sM    = sG + 64 * GST;                            // [64][MST]
    float* sAf   = sM + 64 * MST;                            // [64][64]
    float* sBeta = sAf + 64 * 64;                            // 256B
    float* sT    = sBeta + 64;                               // 4K

    // Per-thread tile walks: the row step (32 for 16B units, 16 for 8B units) is
    // a multiple of 8, so the xor swizzle term is invariant and every address is
    // built once here and then advanced by a compile-time constant.
    constexpr uint32_t OFF_PG = 8 * 64 * 128 * 2 + 2 * 64 * 64 * 2;  // sG
    const uint32_t SMB        = sad(smem);
    const int e8r = tid >> 4, e8c = (tid & 15) << 3;
    const int e8o    = swz(e8r, e8c, 128);
    const size_t e8g = (size_t)e8r * gs + e8c;
    const int e4r = tid >> 5, e4c = (tid & 31) << 2;
    const int e4o    = swz(e4r, e4c, 128);
    const int e4G    = e4r * GST + e4c;
    const size_t e4g = (size_t)e4r * gs + e4c;
    {
        const uint32_t gso = SMB + OFF_PG + (uint32_t)e4G * 4;
        const float* gsrc  = gg + base + e4g;
#pragma unroll
        for (int i = 0; i < 4; ++i, gsrc += 16 * gs) cpasync16z(gso + i * (16u * GST * 4), gsrc, e4r + i * 16 < L);
    }
    cpasync_commit();
    TilePlan<64, 128, PNT> pt;
    pt.init(tid, gs);
    pt.run(SMB, q + base, L);           // sQ
    pt.run(SMB + 16384, kk + base, L);  // sK
    if (tid < 64) sBeta[tid] = (tid < L) ? beta[(size_t)(t0 + tid) * H + h] : 0.f;
    cpasync_commit();
    cpasync_wait<1>();
    __syncthreads();

    // ---- chunk-local inclusive cumsum, base 2: four 16-row segments then a
    //      block fixup, so all 512 threads work and the serial chain is halved.
    // The segment's 16 running sums stay in registers across the fixup barrier,
    // so the plane is written once instead of twice and the second dependent
    // add chain disappears (the read-back was 5-6% of both kernels' stalls).
    {
        const int qd = tid >> 7, dd = tid & 127;
        float* sQ4 = sAf;  // free until the triangular solve
        float* gp  = sG + qd * 16 * GST + dd;
        float r[16];
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            a += gp[i * GST];
            r[i] = a;
        }
        sQ4[qd * 128 + dd] = a;
        __syncthreads();
        float add = 0.f;
        for (int p = 0; p < qd; ++p) add += sQ4[p * 128 + dd];
        add *= RCP_LN2;
#pragma unroll
        for (int i = 0; i < 16; ++i) gp[i * GST] = r[i] * RCP_LN2 + add;
    }
    cpasync_wait<0>();
    __syncthreads();
    // sAf is dead from here until the triangular solve, and the barrier that
    // publishes sM / sAqk also publishes this: zeroing it now costs nothing and
    // removes a serialised 4096-word clear (2.8% of the kernel's stalls) plus a
    // barrier from the solve's critical path.
    {
        const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int u = tid; u < (64 * 64) / 4; u += PNT) reinterpret_cast<float4*>(sAf)[u] = z;
    }

    // ---- row operands: qr / kr scaled by 2^(G_i - ghat_{i/16})
    // The four sub-block anchors are the same rows the anchored-k chain below
    // needs, so they are read once here and carried into it.
    F4 anc[NSB], cf[NSB - 1];
#pragma unroll
    for (int s = 0; s < NSB; ++s) anc[s] = ld4f(sG + (s * 16) * GST + e4c);
#pragma unroll
    for (int m = 0; m < 4; ++m) {  // e4r < 16, so the anchor is 16m
        const int o = e4o + m * (16 * 128);
        const F4 g  = ld4f(sG + e4G + m * (16 * GST));
        const F4 a  = anc[m];
        F4 qv = ld4bf(sQ + o), kv = ld4bf(sK + o);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float f = ex2(g.v[t] - a.v[t]);
            qv.v[t] *= f;
            kv.v[t] *= f;
        }
        st4bf(sB1 + o, qv);
        st4bf(sB2 + o, kv);
    }

    // ---- Aqk^T and M, one 16-row block at a time; warps 0-7 do Aqk, 8-15 do M.
    // All four anchored k operands are built first, so the four products need no
    // barriers between them.  Rows past 16*(s+1) of operand s are never read:
    // warp gw owns columns [8gw, 8gw+8) and is masked off beyond the triangle.
    const int gw   = warp & 7;
    const bool isQ = (warp < 8);
    // Only the lower-triangular tiles are produced below, but dv_local reads the
    // whole 64x64 Aqk, so the strictly-upper tiles must be zeroed here.
    reinterpret_cast<uint4*>(sAqk)[tid] = make_uint4(0, 0, 0, 0);  // 64*64/8 == PNT
    // Operand s differs from operand s-1, on the rows both hold, only by the
    // per-channel factor 2^(G[16s]-G[16(s-1)]) <= 1.  So each row is anchored
    // once at its own sub-block boundary (the only exp2 the row needs) and the
    // remaining operands are reached by multiplying that chain factor in -- 40
    // exp2 and 60 shared wavefronts per thread become 28 and 24.  Every exponent
    // stays <= 0, so the chain can only underflow.
#pragma unroll
    for (int s = 1; s < NSB; ++s)
#pragma unroll
        for (int t = 0; t < 4; ++t) cf[s - 1].v[t] = ex2(anc[s].v[t] - anc[s - 1].v[t]);
#pragma unroll
    for (int m = 0; m < NSB; ++m) {
        const int o = e4o + m * (16 * 128);
        const F4 g  = ld4f(sG + e4G + m * (16 * GST));
        F4 kv       = ld4bf(sK + o);
#pragma unroll
        for (int t = 0; t < 4; ++t) kv.v[t] *= ex2(anc[m].v[t] - g.v[t]);
        st4bf(sKC + m * (64 * 128) + o, kv);
#pragma unroll
        for (int s = m + 1; s < NSB; ++s) {
#pragma unroll
            for (int t = 0; t < 4; ++t) kv.v[t] *= cf[s - 1].v[t];
            st4bf(sKC + s * (64 * 128) + o, kv);
        }
    }
    __syncthreads();
    // Operand traffic per mma of a single-warp 16 x 8n tile with K=128 is
    // (4 + 8/n) KB per 8n mma: a one-tile-per-warp split pays 6 shared
    // wavefronts per mma because the whole 16x128 A fragment is re-read for
    // every 8-column tile.  Giving each warp a *contiguous* column range of the
    // same row block loads A once for the range instead, so the 20 tiles of the
    // triangle cost 576 wavefronts per group instead of 960, and the busiest
    // warp drops from 120 to 80.  Every warp still gets 2 or 3 tiles.
    {
        constexpr int SWB[8] = {0, 1, 1, 2, 2, 3, 3, 3};  // row block
        constexpr int SWN[8] = {0, 0, 2, 0, 3, 0, 3, 6};  // first column tile
        constexpr int SWC[8] = {2, 2, 2, 3, 3, 3, 3, 2};  // tiles in the range
        const int sb = SWB[gw], n0 = SWN[gw], nc = SWC[gw];
        const bf16* Ab = (isQ ? sB1 : sB2) + sb * 16 * 128;
        const bf16* Bb = sKC + sb * (64 * 128) + n0 * (8 * 128);
        float ac[3][4];
        zero_acc<3>(ac);
        if (nc == 2)
            gemm_acc<16, 16, 128, 1, 0, 0, 128, 128>(ac, Ab, Bb, 0, lane);
        else
            gemm_acc<16, 24, 128, 1, 0, 0, 128, 128>(ac, Ab, Bb, 0, lane);
        // Aqk is kept row-major (dv_local reads it with ldmatrix.trans), so the
        // two register halves of a row land in adjacent columns and go out as
        // one conflict-free bf16x2 store instead of two scattered 2-byte stores
        // into the transposed tile.
#pragma unroll
        for (int t = 0; t < 3; ++t) {
            if (t >= nc) continue;
            const int j0 = (n0 + t) * 8 + ((lane & 3) << 1);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const int i = sb * 16 + (lane >> 2) + (rp << 3), r = rp * 2;
                if (isQ)
                    *reinterpret_cast<uint32_t*>(sAqk + swz(i, j0, 64)) =
                        bf2((j0 <= i) ? ac[t][r] * KSCALE : 0.f, (j0 + 1 <= i) ? ac[t][r + 1] * KSCALE : 0.f);
                else {
                    const float bi       = sBeta[i];
                    sM[i * MST + j0]     = (j0 < i) ? ac[t][r] * bi : 0.f;
                    sM[i * MST + j0 + 1] = (j0 + 1 < i) ? ac[t][r + 1] * bi : 0.f;
                }
            }
        }
    }
    __syncthreads();
    pt.run(SMB + 49152, vv + base, L);          // sB2  <- v
    pt.run(SMB + 65536 + 16384, dO + base, L);  // sDOt <- dO
    cpasync_commit();

    // ---- A = (I+M)^-1 : 16x16 forward substitution then block combine
    // (sAf was cleared right after the cumsum; the barrier above published it.)
    if (warp < 4 && lane < 16) {
        // Lane `lane` owns column `lane` of this diagonal block, so every value
        // the substitution needs from sAf is one this lane produced: keep the
        // column in registers and drop both the round trip and the syncwarps.
        const int o       = warp * 16;
        const float* mrow = sM + o * MST + o;
        float col[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float val = (lane == i) ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < i; ++j) val -= mrow[i * MST + j] * col[j];
            col[i] = val;
        }
        float* ac = sAf + o * 64 + o + lane;
#pragma unroll
        for (int i = 0; i < 16; ++i) ac[i * 64] = col[i];
    }
    __syncthreads();
    // ---- block combine by doubling: two independent 16-block merges, then one
    //      32-block merge.  Both levels keep all 512 threads busy and the whole
    //      combine costs 4 barriers instead of 12.
    {  // level 1:  A[p][p-1] = -A[p][p] M[p][p-1] A[p-1][p-1],  p in {1,3}.
        // One row and four adjacent columns per thread: the reduction operand
        // moves as one LDS.128 per four FMAs.  Only 128 threads are needed --
        // the section is shared-bandwidth bound, so idle warps cost nothing.
        const int blk = tid >> 6, a = (tid >> 2) & 15, b = (tid & 3) * 4;
        const int q = blk * 2, pr = q + 1;
        if (tid < 128) {
            const float* pm = sM + (pr * 16 + a) * MST + q * 16;
            const float* pa = sAf + q * 16 * 64 + q * 16 + b;
            float4 r        = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float4 v = *reinterpret_cast<const float4*>(pa + c * 64);
                const float m  = pm[c];
                r.x += m * v.x;
                r.y += m * v.y;
                r.z += m * v.z;
                r.w += m * v.w;
            }
            *reinterpret_cast<float4*>(sT + blk * 256 + a * 16 + b) = r;
        }
        __syncthreads();
        if (tid < 128) {
            const float* pb = sAf + (pr * 16 + a) * 64 + pr * 16;
            const float* pt = sT + blk * 256 + b;
            float4 r        = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float4 v = *reinterpret_cast<const float4*>(pt + c * 16);
                const float m  = pb[c];
                r.x += m * v.x;
                r.y += m * v.y;
                r.z += m * v.z;
                r.w += m * v.w;
            }
            *reinterpret_cast<float4*>(sAf + (pr * 16 + a) * 64 + q * 16 + b) = make_float4(-r.x, -r.y, -r.z, -r.w);
        }
    }
    __syncthreads();
    {  // level 2:  A[32:64][0:32] = -A_bb (M_ba A_aa).
        // One row and an adjacent *column pair* per thread, so the operand that
        // varies with the reduction index costs one LDS.64 per two FMAs instead
        // of two LDS.32 per one.  These two products were 11.5% of the kernel's
        // instructions and its largest stall site.
        const int a = tid >> 3, b = (tid & 7) * 4;
        if (tid < 256) {
            const float* pm = sM + (32 + a) * MST;
            const float* pa = sAf + b;
            float4 r        = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                const float4 v = *reinterpret_cast<const float4*>(pa + c * 64);
                const float m  = pm[c];
                r.x += m * v.x;
                r.y += m * v.y;
                r.z += m * v.z;
                r.w += m * v.w;
            }
            *reinterpret_cast<float4*>(sT + a * 32 + b) = r;
        }
        __syncthreads();
        if (tid < 256) {
            // A_bb is lower triangular, so terms past `a` multiply exact zeros;
            // the bound is rounded up to the warp to keep the loop uniform.
            const int amax  = a | 3;
            const float* pb = sAf + (32 + a) * 64 + 32;
            const float* pt = sT + b;
            float4 r        = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int c = 0; c <= amax; ++c) {
                const float4 v = *reinterpret_cast<const float4*>(pt + c * 32);
                const float m  = pb[c];
                r.x += m * v.x;
                r.y += m * v.y;
                r.z += m * v.z;
                r.w += m * v.w;
            }
            *reinterpret_cast<float4*>(sAf + (32 + a) * 64 + b) = make_float4(-r.x, -r.y, -r.z, -r.w);
        }
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 4; ++m) {  // two adjacent columns per thread
        const int i = (tid >> 5) + m * 16, j = (tid & 31) * 2;
        const float2 v                                    = *reinterpret_cast<const float2*>(sAf + i * 64 + j);
        *reinterpret_cast<uint32_t*>(sAb + swz(i, j, 64)) = bf2(v.x, v.y);
    }
    __syncthreads();

    // ---- store A (swizzled bytes), gn2, kg, qg, and the W operand
    {
        bf16* dst                          = Amat + ((size_t)cs * H + h) * 64 * 64;
        reinterpret_cast<uint4*>(dst)[tid] = reinterpret_cast<const uint4*>(sAb)[tid];  // 64*64/8 == PNT
    }
    if (tid < 128) gn2[((size_t)cs * H + h) * DH + tid] = ex2(sG[(L - 1) * GST + tid]);
    const F4 gnl = ld4f(sG + (L - 1) * GST + e4c);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int i = e4r + m * 16, o = e4o + m * (16 * 128);
        const F4 g    = ld4f(sG + e4G + m * (16 * GST));
        F4 qv         = ld4bf(sQ + o);
        F4 kv         = ld4bf(sK + o);
        F4 wv         = kv;
        const float b = sBeta[i];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float eg = ex2(g.v[t]);
            // qg carries 1/sqrt(D): its only consumer is  P += scale * qg^T dO
            qv.v[t] *= eg * KSCALE;
            kv.v[t] *= ex2(gnl.v[t] - g.v[t]);
            wv.v[t] *= b * eg;
        }
        if (i < L) st4bf(kg + base + e4g + (size_t)(m * 16) * gs, kv);
        // qg is no longer a scan operand: it is consumed here, by the P GEMM.
        // Rows >= L were zero-filled on load, so they stay exactly zero.
        st4bf(sQ + o, qv);
        st4bf(sB1 + o, wv);
    }

    // ---- w = A * (k*beta*2^G)   (stored negated)
    __syncthreads();
    using TL = Tiling<64, 128, 16>;
    {
        float acc[TL::TPW][4];
        zero_acc<TL::TPW>(acc);
        gemm_acc<64, 128, 64, 16, 0, 1, 64, 128>(acc, sAb, sB1, warp, lane);
#pragma unroll
        for (int j = 0; j < TL::TPW; ++j) {
            const int o0 = swz(TL::mi(warp, j) * 16 + (lane >> 2), TL::ni(warp, j) * 8 + ((lane & 3) << 1), 128);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp)
                *reinterpret_cast<uint32_t*>(sB3 + o0 + rp * (8 * 128)) = bf2(-acc[j][rp * 2], -acc[j][rp * 2 + 1]);
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            if (e8r + m * 32 < L)
                *reinterpret_cast<uint4*>(wneg + base + e8g + (size_t)(m * 32) * gs) =
                    *reinterpret_cast<const uint4*>(sB3 + e8o + m * (32 * 128));
        }
    }
    cpasync_wait<0>();
    __syncthreads();
    // ---- u = A * (v*beta)
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int o   = e8o + m * (32 * 128);
        F8 vv8        = ld8bf(sB2 + o);
        const float b = sBeta[e8r + m * 32];
#pragma unroll
        for (int t = 0; t < 8; ++t) vv8.v[t] *= b;
        st8bf(sB1 + o, vv8);
    }
    __syncthreads();
    {
        float acc[TL::TPW][4];
        zero_acc<TL::TPW>(acc);
        gemm_acc<64, 128, 64, 16, 0, 1, 64, 128>(acc, sAb, sB1, warp, lane);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < TL::TPW; ++j) {
            const int o0 = swz(TL::mi(warp, j) * 16 + (lane >> 2), TL::ni(warp, j) * 8 + ((lane & 3) << 1), 128);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp)
                *reinterpret_cast<uint32_t*>(sB1 + o0 + rp * (8 * 128)) = bf2(acc[j][rp * 2], acc[j][rp * 2 + 1]);
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            if (e8r + m * 32 < L)
                *reinterpret_cast<uint4*>(uu + base + e8g + (size_t)(m * 32) * gs) =
                    *reinterpret_cast<const uint4*>(sB1 + e8o + m * (32 * 128));
        }
    }
    __syncthreads();
    // ---- dv_local = Aqk^T dO  and  P = qg^T dO, fused.
    //
    // Both products reduce over the chunk's 64 tokens against the same dO tile,
    // so one set of B fragments feeds both.  P is given the same 32-column warp
    // slice as dv_local and two row blocks, which is also its own optimal
    // (32x32) warp tile -- the pair costs 2560 operand bytes per warp per
    // k-step instead of 4096 for the two GEMMs run separately.
    //
    // P = qg^T dO is hoisted out of the serial reverse scan: the reverse
    // recurrence dS_{c-1} = Diag(gn_c) dS_c + w_c^T dv2_c + P_c has no
    // cross-chunk dependence in P, and the reverse direction was carrying three
    // products against the forward direction's two, so its CTAs set the scan's
    // critical path while the forward CTAs idled.
    {
        using TL2     = Tiling<64, 128, 16>;  // dv_local: mi = w/4
        const int dmi = TL2::mi(warp, 0);
        float acc[TL2::TPW][4], pac[2][TL2::TPW][4];
        zero_acc<TL2::TPW>(acc);
        zero_acc<TL2::TPW>(pac[0]);
        zero_acc<TL2::TPW>(pac[1]);
        {
            const LdA A = a1_addr<64>(sAqk, lane);     // Aqk [64 i][64 j], m-major
            const LdA P = a1_addr<128>(sQ, lane);      // qg [64 tok][128 ch], m-major
            const LdA B = b1_addr_p<128>(sDOt, lane);  // dO [64 tok][128 v], n-major
            uint32_t af[4], pf[2][4], bf_[4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                ldm_x4t(af, A.base + kt * (32u * 64) + a1x<0>(A.x, dmi));
#pragma unroll
                for (int u = 0; u < 2; ++u) ldm_x4t(pf[u], P.base + kt * (32u * 128) + a1x<0>(P.x, 2 * dmi + u));
#pragma unroll
                for (int j = 0; j < TL2::TPW; j += 2) {
                    ldm_x4t(bf_, B.base + kt * (32u * 128) + ((TL2::ni(warp, j) ^ B.x) << 4));
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        mma16816p(acc[j + e], af, bf_[2 * e], bf_[2 * e + 1]);
                        mma16816p(pac[0][j + e], pf[0], bf_[2 * e], bf_[2 * e + 1]);
                        mma16816p(pac[1][j + e], pf[1], bf_[2 * e], bf_[2 * e + 1]);
                    }
                }
            }
        }
        __syncthreads();  // sQ (qg) and sB1 released
        bf16* sP = sQ;    // [128][128], spans sQ + sK
#pragma unroll
        for (int j = 0; j < TL2::TPW; ++j) {
            const int c0 = TL2::ni(warp, j) * 8 + ((lane & 3) << 1);
            const int o0 = swz(dmi * 16 + (lane >> 2), c0, 128);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp)
                *reinterpret_cast<uint32_t*>(sB1 + o0 + rp * (8 * 128)) = bf2(acc[j][rp * 2], acc[j][rp * 2 + 1]);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int p0 = swz((2 * dmi + u) * 16 + (lane >> 2), c0, 128);
#pragma unroll
                for (int rp = 0; rp < 2; ++rp)
                    *reinterpret_cast<uint32_t*>(sP + p0 + rp * (8 * 128)) =
                        bf2(pac[u][j][rp * 2], pac[u][j][rp * 2 + 1]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            if (e8r + m * 32 < L)
                *reinterpret_cast<uint4*>(dvl + base + e8g + (size_t)(m * 32) * gs) =
                    *reinterpret_cast<const uint4*>(sB1 + e8o + m * (32 * 128));
        }
        bf16* dst    = Pg + ((size_t)cs * H + h) * DH * DH;
        const int pr = tid >> 4, pc = (tid & 15) << 3;
        const int po = swz(pr, pc, 128);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            *reinterpret_cast<uint4*>(dst + (size_t)(pr + m * 32) * DH + pc) =
                *reinterpret_cast<const uint4*>(sP + po + m * (32 * 128));
        }
    }
}

// ==================== k_scan.cuh ====================
#pragma once

static constexpr int PCV = 64;  // physical swizzle width of the V tiles
// one prefetch stage: w, kg (128-wide) + P (128 x BV) + a (BV-wide) + gn.
// P = qg^T dO is precomputed by the parallel prep pass, so the reverse
// direction no longer streams the 128-wide qg operand or the dO slice.
static constexpr int STAGE = 3 * 64 * 128 * 2 + 64 * PCV * 2 + 512;
// Three operand stages instead of two: each chunk's copies then get two full
// iterations to land rather than one.  The serial scan runs at 12.5%
// occupancy with an elapsed IPC of 0.82, so its per-chunk stall is the
// operand wait at the top of the loop, and the extra 56 KB costs nothing --
// shared memory already pinned the kernel at one CTA per SM.
static constexpr int SMEM_SCAN = 128 * PCV * 2 + 64 * PCV * 2 + 3 * STAGE;

// byte offsets of the operand tiles inside one prefetch stage
static constexpr uint32_t O_W  = 0;
static constexpr uint32_t O_KG = 64 * 128 * 2;
static constexpr uint32_t O_P  = 2 * 64 * 128 * 2;  // [128][PCV]
static constexpr uint32_t O_A  = 3 * 64 * 128 * 2;
static constexpr uint32_t O_GN = O_A + 64 * PCV * 2;

// ---------------------------------------------------------------------------
// K2 "scan": grid.x = 2*NVB (dir, v-block), grid.y = N*H.
// dir 0 : forward state recurrence    -> hst, v_new
// dir 1 : reverse gradient recurrence -> dhst, dv2, d_initial_state
// The two directions are independent, so they run as two halves of one grid.
// Each chunk's operands are prefetched one iteration ahead via cp.async.
//
// Every shared/global address used inside the chunk loop is a loop-invariant
// function of (warp, lane) plus the chunk base, so all of them are built once
// before the loop and the body only adds the chunk offset.  The fp32
// accumulator is staged through shared memory as bf16x2 pairs, halving the
// conversion and store counts.
// ---------------------------------------------------------------------------
template <int BV, int NT, bool SINGLE, bool FULL>
__global__
__launch_bounds__(NT, 1) void k_scan_t(const bf16* __restrict__ wneg,
                                       const bf16* __restrict__ uu,
                                       const bf16* __restrict__ kg,
                                       const bf16* __restrict__ Pg,
                                       const bf16* __restrict__ dvl,
                                       const bf16* __restrict__ dO,
                                       const float* __restrict__ gn2,
                                       const float* __restrict__ ist,
                                       const float* __restrict__ dfs,
                                       const int* __restrict__ cs_t0,
                                       const int* __restrict__ cs_len,
                                       const int* __restrict__ seq_c0,
                                       const int* __restrict__ seq_nc,
                                       bf16* __restrict__ hst,
                                       bf16* __restrict__ dhst,
                                       bf16* __restrict__ vnew,
                                       bf16* __restrict__ dv2o,
                                       float* __restrict__ dis,
                                       int T,
                                       int H) {
    constexpr int NVB = DH / BV;
    constexpr int NW  = NT / 32;
    constexpr int C8V = BV / 8;

    const int dir = blockIdx.x / NVB, vb = blockIdx.x % NVB;
    const int nh = blockIdx.y, n = nh / H, h = nh % H;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const size_t gs = (size_t)H * DH;
    const int c0    = SINGLE ? 0 : seq_c0[n];
    const int nc    = SINGLE ? ((T + BT - 1) / BT) : seq_nc[n];

    extern __shared__ char smem[];
    bf16* sS           = reinterpret_cast<bf16*>(smem);  // [128][PCV]
    bf16* sB           = sS + 128 * PCV;                 // [64][PCV]
    bf16* st0          = sB + 64 * PCV;
    float* sF          = reinterpret_cast<float*>(st0);  // [128][65] alias (seed only)
    const uint32_t SBA = sad(st0);                       // stage base, byte address

    using TS = Tiling<128, BV, NW>;
    using TV = Tiling<64, BV, NW>;

    // ---- copy plans (loop invariant) -------------------------------------
    TilePlan<64, 128, NT> p128;
    p128.init(tid, gs);
    const bool actV  = tid < 64 * C8V;
    const int rV     = actV ? (tid / C8V) : 0;
    const int cV     = (tid % C8V) * 8;
    const int sVo    = swz(rV, cV, PCV);
    const size_t gVo = (size_t)rV * gs + cV;
    const size_t gPo = (size_t)rV * DH + vb * BV + cV;  // P slab, stride DH

    // ---- accumulator staging offsets (loop invariant) --------------------
    int oS[TS::TPW], rS[TS::TPW];
#pragma unroll
    for (int j = 0; j < TS::TPW; ++j) {
        rS[j] = TS::mi(warp, j) * 16 + (lane >> 2);
        oS[j] = swz(rS[j], TS::ni(warp, j) * 8 + ((lane & 3) << 1), PCV);
    }
    int oB[TV::TPW];
#pragma unroll
    for (int j = 0; j < TV::TPW; ++j)
        oB[j] = swz(TV::mi(warp, j) * 16 + (lane >> 2), TV::ni(warp, j) * 8 + ((lane & 3) << 1), PCV);

    // ---- output plans (loop invariant) -----------------------------------
    constexpr int HIT = (128 * C8V) / NT;  // state rows per thread
    constexpr int HRS = NT / C8V;          // row step, multiple of 8
    const int hr = tid / C8V, hc = (tid % C8V) * 8;
    const size_t hgo = (size_t)hr * DH + vb * BV + hc;
    const int hso    = swz(hr, hc, PCV);
    const size_t vgo = (size_t)rV * gs + vb * BV + cV;

    const float* srcst = (dir == 0) ? ist : dfs;
    float acc[TS::TPW][4];
    {  // seed state, transposed:  S[d][x] = src[n][h][x][d]
        const float* p = srcst + ((size_t)nh * DH + vb * BV) * DH;
        for (int u = tid; u < BV * DH; u += NT) {
            const int x = u >> 7, d = u & 127;
            sF[d * 65 + x] = p[(size_t)x * DH + d];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < TS::TPW; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[j][r] = sF[TS::row(warp, lane, j, r) * 65 + TS::col(warp, lane, j, r)];
        __syncthreads();
    }

    // ---- prefetch one chunk's operands into stage `st` --------------------
    auto prefetch = [&](int st, int cs) {
        const uint32_t sb = SBA + (uint32_t)st * STAGE;
        const int L       = chunk_len_t<FULL>(cs, cs_len, T, SINGLE);
        const size_t bs   = (size_t)chunk_t0(cs, cs_t0, SINGLE) * gs + (size_t)h * DH;
        p128.run(sb + O_W, wneg + bs, L);
        p128.run(sb + O_KG, kg + bs, L);
        if (dir == 0) {
            if (actV) cpasync16z(sb + O_A + 2 * sVo, uu + bs + vb * BV + gVo, rV < L);
        } else {
            if (actV) cpasync16z(sb + O_A + 2 * sVo, dvl + bs + vb * BV + gVo, rV < L);
            // P is a full [128][BV] slab: no tail rows, so no predicate.
            const bf16* pb = Pg + ((size_t)cs * H + h) * DH * DH + gPo;
#pragma unroll
            for (int i = 0; i < 2; ++i) cpasync16(sb + O_P + 2 * sVo + i * (64u * PCV * 2), pb + (size_t)(i * 64) * DH);
        }
        if (tid < 32) cpasync16(sb + O_GN + tid * 16, gn2 + ((size_t)cs * H + h) * DH + tid * 4);
    };

    // chunk index of the it-th step of this direction
    auto cidx = [&](int i) { return (dir == 0) ? (c0 + i) : (c0 + nc - 1 - i); };
    if (nc > 0) prefetch(0, cidx(0));
    cpasync_commit();
    if (nc > 1) prefetch(1, cidx(1));
    cpasync_commit();  // always committed, empty when nc == 1, so the
                       // wait below always retires exactly one chunk

    int buf = 0, nxt = 2;
    for (int it = 0; it < nc; ++it) {
        const int cs      = cidx(it);
        const int L       = chunk_len_t<FULL>(cs, cs_len, T, SINGLE);
        const size_t base = (size_t)chunk_t0(cs, cs_t0, SINGLE) * gs + (size_t)h * DH;

        // sS's only readers are the previous iteration's first GEMM and its hst
        // copy, both of which are fenced by that iteration's second barrier, so
        // the accumulator can be staged with no barrier of its own.  The next
        // chunk's prefetch is issued *after* the barrier below instead of before
        // it: that barrier is the one that fences the previous iteration's last
        // read of the buffer being overwritten, so the loop needs two barriers
        // per chunk instead of three and the prefetch still gets a full
        // iteration of latency to hide in.
#pragma unroll
        for (int j = 0; j < TS::TPW; ++j) {
            *reinterpret_cast<uint32_t*>(sS + oS[j])           = bf2(acc[j][0], acc[j][1]);
            *reinterpret_cast<uint32_t*>(sS + oS[j] + 8 * PCV) = bf2(acc[j][2], acc[j][3]);
        }
        cpasync_wait<1>();  // this chunk landed; the next may still be in flight
        __syncthreads();    // publishes sS and this chunk's staged operands
        // The buffer being refilled is the one iteration it-1 consumed, and the
        // barrier above already fenced its last reader.
        if (it + 2 < nc) prefetch(nxt, cidx(it + 2));
        cpasync_commit();

        bf16* const stb  = st0 + buf * (STAGE / 2);
        nxt              = buf;  // freed by this chunk's barrier
        buf              = (buf == 2) ? 0 : (buf + 1);
        const bf16* pW   = stb;
        const bf16* pKG  = stb + 64 * 128;
        const bf16* pA   = stb + 3 * 64 * 128;
        const float* pGn = reinterpret_cast<const float*>(reinterpret_cast<const char*>(stb) + O_GN);
        // pGn and the staged `a` tile are published by the barrier above and are
        // not consumed until after the first GEMM pair, so reading them here
        // puts a whole 64x(2*BV)x64 product between the loads and their uses.
        // Both sat at the head of a dependent chain (2.9% of this kernel's
        // stall samples each) when they were read at their point of use.
        float gsc[TS::TPW][2];
#pragma unroll
        for (int j = 0; j < TS::TPW; ++j) {
            gsc[j][0] = pGn[rS[j]];
            gsc[j][1] = pGn[rS[j] + 8];
        }
        float2 av[TV::TPW][2];
#pragma unroll
        for (int j = 0; j < TV::TPW; ++j) {
            av[j][0] = un2(*reinterpret_cast<const uint32_t*>(pA + oB[j]));
            av[j][1] = un2(*reinterpret_cast<const uint32_t*>(pA + oB[j] + 8 * PCV));
        }
        float vacc[TV::TPW][4], vac2[TV::TPW][4];
        zero_acc<TV::TPW>(vacc);
        zero_acc<TV::TPW>(vac2);
        // split K so the two halves form independent MMA chains
        gemm_acc<64, BV, 64, NW, 0, 1, 128, PCV>(vacc, (dir == 0) ? pW : pKG, sS, warp, lane);
        gemm_acc<64, BV, 64, NW, 0, 1, 128, PCV>(vac2, ((dir == 0) ? pW : pKG) + 64, sS + 64 * PCV, warp, lane);
        {
            bf16* out = ((dir == 0) ? hst : dhst) + ((size_t)cs * H + h) * DH * DH;
#pragma unroll
            for (int i = 0; i < HIT; ++i)
                *reinterpret_cast<uint4*>(out + hgo + (size_t)(i * HRS) * DH) =
                    *reinterpret_cast<const uint4*>(sS + hso + i * (HRS * PCV));
        }
#pragma unroll
        for (int j = 0; j < TV::TPW; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) vacc[j][r] += vac2[j][r];
#pragma unroll
        for (int j = 0; j < TV::TPW; ++j) {
            *reinterpret_cast<uint32_t*>(sB + oB[j])           = bf2(vacc[j][0] + av[j][0].x, vacc[j][1] + av[j][0].y);
            *reinterpret_cast<uint32_t*>(sB + oB[j] + 8 * PCV) = bf2(vacc[j][2] + av[j][1].x, vacc[j][3] + av[j][1].y);
        }
        __syncthreads();
        {
            bf16* dst = (dir == 0) ? vnew : dv2o;
            if (actV && rV < L) *reinterpret_cast<uint4*>(dst + base + vgo) = *reinterpret_cast<const uint4*>(sB + sVo);
        }
#pragma unroll
        for (int j = 0; j < TS::TPW; ++j) {
            const float ga = gsc[j][0], gb = gsc[j][1];
            acc[j][0] *= ga;
            acc[j][1] *= ga;
            acc[j][2] *= gb;
            acc[j][3] *= gb;
        }
        if (dir == 0) {
            gemm_acc<128, BV, 64, NW, 1, 1, 128, PCV>(acc, pKG, sB, warp, lane);
        } else {
            // The P slab was staged a whole iteration ago; reading it before the
            // 128 x BV x 64 product instead of after leaves the loads nothing
            // to wait on.
            const bf16* pP = stb + 2 * 64 * 128;  // precomputed qg^T dO slab
            float2 pv[TS::TPW][2];
#pragma unroll
            for (int j = 0; j < TS::TPW; ++j) {
                pv[j][0] = un2(*reinterpret_cast<const uint32_t*>(pP + oS[j]));
                pv[j][1] = un2(*reinterpret_cast<const uint32_t*>(pP + oS[j] + 8 * PCV));
            }
            gemm_acc<128, BV, 64, NW, 1, 1, 128, PCV>(acc, pW, sB, warp, lane);
#pragma unroll
            for (int j = 0; j < TS::TPW; ++j) {
                acc[j][0] += pv[j][0].x;
                acc[j][1] += pv[j][0].y;
                acc[j][2] += pv[j][1].x;
                acc[j][3] += pv[j][1].y;
            }
        }
    }

    if (dir == 1) {
        __syncthreads();
        for (int j = 0; j < TS::TPW; ++j)
            for (int r = 0; r < 4; ++r) sF[TS::row(warp, lane, j, r) * 65 + TS::col(warp, lane, j, r)] = acc[j][r];
        __syncthreads();
        float* p = dis + ((size_t)nh * DH + vb * BV) * DH;
        for (int u = tid; u < BV * DH; u += NT) {
            const int x = u >> 7, d = u & 127;
            p[(size_t)x * DH + d] = sF[d * 65 + x];
        }
    }
}

// ==================== k_bwd.cuh ====================
#pragma once

// ---------------------------------------------------------------------------
// Fused B3+B4 (state-gradient products) and B5+B6+B7 (WY backward, intra-chunk,
// reverse cumsum).  One CTA per (chunk-slot, head); emits dq, dk, dv, dg, dbeta.
//
// Fusing removes the dq_inter / dk_inter / dw / dg_inter / dAqk workspace
// round-trip (256 KB of global traffic per chunk-head) and one extra read of
// q, k, g and dv2.  The phase-A staging tiles (dO, v_new, h, dh) are placed in
// the same 64 KB region that phase B later uses for A, dAkk, scratch, v and the
// dq accumulator, so the fused kernel needs no more shared memory than B567 did.
// The row and column sweeps share one output pass, and chunk-length reductions
// use four 16-row segments across all 512 threads.
// ---------------------------------------------------------------------------
static constexpr int BNT = 512;  // threads
static constexpr int BNW = 16;   // warps

// FP32 planes are accessed as adjacent register pairs.  A 136-word stride
// spreads 8 rows x 4 column pairs over all 32 banks; GST=132 is 2-way conflicted.
static constexpr int GS2 = 136;

// --- shared-memory map (byte offsets) --------------------------------------
static constexpr int OFF_Q     = 0;                     // bf16[64][128]
static constexpr int OFF_K     = OFF_Q + 16384;         // bf16[64][128]
static constexpr int OFF_DV2   = OFF_K + 16384;         // bf16[64][128]
static constexpr int OFF_G     = OFF_DV2 + 16384;       // float[64][GST]
static constexpr int OFF_DAQK  = OFF_G + 64 * GST * 4;  // bf16[64][64]
static constexpr int OFF_SMALL = OFF_DAQK + 8192;       // beta/diag/red/dgk/part
static constexpr int SZ_SMALL  = 64 * 4 * 3 + 128 * 4 + 16 * 64 * 4;
static constexpr int OFF_X     = OFF_SMALL + SZ_SMALL;  // 64 KB dual-use
// Region Y stages the second half of h / dh during phase A and then holds dW;
// the fp32 dk accumulator that used to live here is now carried in registers.
static constexpr int OFF_Y    = OFF_X + 65536;  // h2+dh2 -> dW
static constexpr int OFF_Z    = OFF_Y + 32768;  // dG2
static constexpr int SMEM_BWD = OFF_Z + 64 * GS2 * 4;

DEV void
red_flush(float* sPart, int row0, float v0, float v1, int lane) {
    v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
    v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
    v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
    if ((lane & 3) == 0) {
        const int warp = threadIdx.x >> 5;
        sPart[warp * 64 + row0] += v0;
        sPart[warp * 64 + row0 + 8] += v1;
    }
}

template <bool FULL>
__global__
__launch_bounds__(BNT, 1) void k_bwd(const bf16* __restrict__ q,
                                     const bf16* __restrict__ kk,
                                     const bf16* __restrict__ vv,
                                     const float* __restrict__ gg,
                                     const float* __restrict__ betag,
                                     const bf16* __restrict__ dOg,
                                     const bf16* __restrict__ Amatg,
                                     const bf16* __restrict__ vnewg,
                                     const bf16* __restrict__ dv2g,
                                     const bf16* __restrict__ hstg,
                                     const bf16* __restrict__ dhstg,
                                     const int* __restrict__ cs_t0,
                                     const int* __restrict__ cs_len,
                                     bf16* __restrict__ dq,
                                     bf16* __restrict__ dk,
                                     bf16* __restrict__ dvo,
                                     float* __restrict__ dgo,
                                     float* __restrict__ dbeta,
                                     int T,
                                     int H,
                                     bool single_sequence) {
    const int cs = blockIdx.x, h = blockIdx.y;
    const int L = chunk_len_t<FULL>(cs, cs_len, T, single_sequence);
    if (!FULL && L == 0) return;
    const int t0  = chunk_t0(cs, cs_t0, single_sequence);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qd = tid >> 7, dd = tid & 127;  // 4 x 128 reduction lanes
    const size_t gs   = (size_t)H * DH;
    const size_t base = (size_t)t0 * gs + (size_t)h * DH;

    // Per-thread walks over a 64x128 tile.  The row step (32 for 16B units, 16
    // for 8B units) is a multiple of 8, so the xor swizzle term is invariant and
    // both shared and global offsets advance by a constant.
    const int e8r = tid >> 4, e8c = (tid & 15) << 3;
    const int e8o    = swz(e8r, e8c, 128);
    const size_t e8g = (size_t)e8r * gs + e8c;
    const int e4r = tid >> 5, e4c = (tid & 31) << 2;
    const int e4o    = swz(e4r, e4c, 128);
    const int e4G    = e4r * GST + e4c;
    const size_t e4g = (size_t)e4r * gs + e4c;

    extern __shared__ char smem[];
    bf16* sQ     = reinterpret_cast<bf16*>(smem + OFF_Q);
    bf16* sK     = reinterpret_cast<bf16*>(smem + OFF_K);
    bf16* sDV2   = reinterpret_cast<bf16*>(smem + OFF_DV2);
    float* sG    = reinterpret_cast<float*>(smem + OFF_G);
    bf16* sdAqk  = reinterpret_cast<bf16*>(smem + OFF_DAQK);
    float* sBeta = reinterpret_cast<float*>(smem + OFF_SMALL);
    float* sDiag = sBeta + 64;
    float* sRed  = sDiag + 64;
    float* sDgk  = sRed + 64;   // [128]
    float* sPart = sDgk + 128;  // [16][64]
    // phase A staging (region X)
    bf16* sDO   = reinterpret_cast<bf16*>(smem + OFF_X);
    bf16* sVN   = sDO + 64 * 128;
    bf16* sH    = sVN + 64 * 128;
    bf16* sDH   = sH + 128 * 64;
    float* sTmp = reinterpret_cast<float*>(smem + OFF_X);  // [64][GS2], after A
    // phase B (region X)
    bf16* sAb   = reinterpret_cast<bf16*>(smem + OFF_X);  // 8 K
    bf16* sdAkk = sAb + 64 * 64;                          // 8 K
    bf16* sScr  = sdAkk + 64 * 64;                        // 16 K
    bf16* sV    = sScr + 64 * 128;                        // 16 K (-> sScr2)
    bf16* sDqS  = sV + 64 * 128;                          // 16 K
    // region Y / Z
    bf16* sDW   = reinterpret_cast<bf16*>(smem + OFF_Y);
    float* sDG2 = reinterpret_cast<float*>(smem + OFF_Z);
    float* sQ4  = sDG2;  // [4][128] scratch
    // Region Y is idle until dW is produced, so prefetch the second half of
    // both state tiles here and overlap its latency with cumsum and dAqk.
    bf16* sH2  = reinterpret_cast<bf16*>(smem + OFF_Y);
    bf16* sDH2 = sH2 + 128 * 64;

    using T64 = Tiling<64, 128, BNW>;
    using TA  = Tiling<64, 64, BNW>;

    // ---------------------------------------------------------------- loads
    // Every copy below walks its tile in row steps that are multiples of 8, so
    // the xor swizzle term is invariant: one address computation per plane, then
    // compile-time increments.
    const uint32_t SMB = sad(smem);
    {  // fp32 gate plane: 4 units per thread, 16 rows apart
        const int gr = tid >> 5, gc = (tid & 31) * 4;
        const uint32_t gso = SMB + OFF_G + (uint32_t)(gr * GST + gc) * 4;
        const float* gsrc  = gg + base + (size_t)gr * gs + gc;
#pragma unroll
        for (int i = 0; i < 4; ++i, gsrc += 16 * gs) cpasync16z(gso + i * (16u * GST * 4), gsrc, gr + i * 16 < L);
    }
    cpasync_commit();
    TilePlan<64, 128, BNT> pt;
    pt.init(tid, gs);
    pt.run(SMB + OFF_X, dOg + base, L);            // sDO
    pt.run(SMB + OFF_X + 16384, vnewg + base, L);  // sVN
    pt.run(SMB + OFF_DV2, dv2g + base, L);
    pt.run(SMB + OFF_Q, q + base, L);
    pt.run(SMB + OFF_K, kk + base, L);
    if (tid < 64) {
        sBeta[tid] = (tid < L) ? betag[(size_t)(t0 + tid) * H + h] : 0.f;
        sRed[tid]  = 0.f;
    }
    for (int u = tid; u < 16 * 64; u += BNT) sPart[u] = 0.f;
    cpasync_commit();
    {  // both state halves: 2 units per thread, 64 rows apart
        const bf16* gH  = hstg + ((size_t)cs * H + h) * DH * DH;
        const bf16* gDH = dhstg + ((size_t)cs * H + h) * DH * DH;
        const int sr = tid >> 3, sc = (tid & 7) * 8;
        const uint32_t so = (uint32_t)(2 * swz(sr, sc, 64));
        const size_t go   = (size_t)sr * DH + sc;
        const uint32_t aH = SMB + OFF_X + 32768, aDH = aH + 16384;
        const uint32_t aH2 = SMB + OFF_Y, aDH2 = aH2 + 16384;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t d = so + i * (64u * 64 * 2);
            const size_t g   = go + (size_t)(i * 64) * DH;
            cpasync16(aH + d, gH + g);
            cpasync16(aDH + d, gDH + g);
            cpasync16(aH2 + d, gH + g + 64);
            cpasync16(aDH2 + d, gDH + g + 64);
        }
    }
    cpasync_commit();
    cpasync_wait<2>();
    __syncthreads();
    {  // chunk-local cumsum, base 2: four 16-row segments then a block fixup.
        // The segment's running sums stay in registers across the fixup barrier:
        // the gate plane is written once instead of twice and the read-back's
        // dependent add chain (6.5% of this kernel's stalls) disappears.
        float* gp = sG + qd * 16 * GST + dd;
        float r[16];
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            a += gp[i * GST];
            r[i] = a;
        }
        sQ4[qd * 128 + dd] = a;
        __syncthreads();
        float add = 0.f;
        for (int p = 0; p < qd; ++p) add += sQ4[p * 128 + dd];
        add *= RCP_LN2;
#pragma unroll
        for (int i = 0; i < 16; ++i) gp[i * GST] = r[i] * RCP_LN2 + add;
    }
    cpasync_wait<1>();
    __syncthreads();

    // =================== phase A : dAqk ===================
    {
        float dAq[TA::TPW][4];
        zero_acc<TA::TPW>(dAq);
        if (TA::ni(warp, 0) < 2 * (TA::mi(warp, 0) + 1)) {
            gemm_acc<64, 64, 128, BNW, 0, 0, 128, 128>(dAq, sDO, sVN, warp, lane);
        }
#pragma unroll
        for (int j = 0; j < TA::TPW; ++j) {
            const int i0 = TA::mi(warp, j) * 16 + (lane >> 2);
            const int c0 = TA::ni(warp, j) * 8 + ((lane & 3) << 1);
            const int o0 = swz(i0, c0, 64);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const int i = i0 + rp * 8, r = rp * 2;
                const float v0 = dAq[j][r] * KSCALE, v1 = dAq[j][r + 1] * KSCALE;
                *reinterpret_cast<uint32_t*>(sdAqk + o0 + rp * (8 * 64)) =
                    bf2((c0 < i) ? v0 : 0.f, (c0 + 1 < i) ? v1 : 0.f);
                if (c0 == i) sDiag[i] = v0;
                if (c0 + 1 == i) sDiag[i] = v1;
            }
        }
    }

    // =================== phase A : state products ===================
    float dqa[T64::TPW][4], dka[T64::TPW][4], dwa[T64::TPW][4];
    zero_acc<T64::TPW>(dqa);
    zero_acc<T64::TPW>(dka);
    zero_acc<T64::TPW>(dwa);
    float dgk = 0.f;
    cpasync_wait<0>();
    __syncthreads();
#pragma unroll
    for (int vh = 0; vh < 2; ++vh) {
        const bf16* pH  = vh ? sH2 : sH;
        const bf16* pDH = vh ? sDH2 : sDH;
        const int off   = vh * 64;
        gemm_acc2<64, 128, 64, BNW, 0, 0, 128, 64>(dqa, dwa, sDO + off, sDV2 + off, pH, warp, lane);
        gemm_acc<64, 128, 64, BNW, 0, 0, 128, 64>(dka, sVN + off, pDH, warp, lane);
        {  // dgk[d] += sum_x h[d][x] dh[d][x], 16 columns per thread
            const int x0 = qd * 16;
            const F8 h0  = ld8bf(pH + swz(dd, x0, 64));
            const F8 d0  = ld8bf(pDH + swz(dd, x0, 64));
            const F8 h1  = ld8bf(pH + swz(dd, x0 + 8, 64));
            const F8 d1  = ld8bf(pDH + swz(dd, x0 + 8, 64));
            float a      = 0.f;
#pragma unroll
            for (int t = 0; t < 8; ++t) a += h0.v[t] * d0.v[t] + h1.v[t] * d1.v[t];
            dgk += a;
        }
    }
    // sQ4's only prior readers are in the chunk cumsum, which the barrier that
    // published the token tiles already fenced, so this write needs no barrier
    // of its own.
    sQ4[qd * 128 + dd] = dgk;
    __syncthreads();
    {  // Region X was last read by the state products above and the barrier
        // just fenced it, so phase B's two operands are fetched here instead of
        // at the phase boundary, where the wait sat directly on top of the
        // issue.  The gate scaling, the k*dk reduction, the dg seed and the
        // whole dg-plane write now run against the global round trip.
        const bf16* sa = Amatg + ((size_t)cs * H + h) * 64 * 64;
        cpasync16(SMB + OFF_X + tid * 16, sa + tid * 8);  // 64*64/8 == BNT
        pt.run(SMB + OFF_X + 32768, vv + base, L);        // sV
        cpasync_commit();
    }
    if (tid < 128) {
        float x = 0.f;
#pragma unroll
        for (int p = 0; p < 4; ++p) x += sQ4[p * 128 + tid];
        sDgk[tid] = x;
    }

    // gate scaling; dw -> sDW (region Y, untouched by phase A staging)
#pragma unroll
    for (int j = 0; j < T64::TPW; ++j) {
        const int i0    = T64::mi(warp, j) * 16 + (lane >> 2);
        const int d0    = T64::ni(warp, j) * 8 + ((lane & 3) << 1);
        const int o0    = swz(i0, d0, 128);
        const float2 gn = *reinterpret_cast<const float2*>(sG + (L - 1) * GST + d0);
#pragma unroll
        for (int rp = 0; rp < 2; ++rp) {
            const int i = i0 + rp * 8, r = rp * 2;
            const float2 gi = *reinterpret_cast<const float2*>(sG + i * GST + d0);
            dqa[j][r]       = KSCALE * dqa[j][r] * ex2(gi.x);
            dqa[j][r + 1]   = KSCALE * dqa[j][r + 1] * ex2(gi.y);
            dka[j][r]       = dka[j][r] * ex2(gn.x - gi.x);
            dka[j][r + 1]   = dka[j][r + 1] * ex2(gn.y - gi.y);
            *reinterpret_cast<uint32_t*>(sDW + o0 + rp * (8 * 128)) = bf2(-dwa[j][r], -dwa[j][r + 1]);
        }
    }
    // k_i * dk_i feeds this thread's own dg term and a column reduction over all
    // 64 rows.  Both stay out of shared memory: the reduction is two register
    // adds (the thread's own two rows), three shuffles (the eight row groups a
    // warp holds), and one [4][128] plane across the four row-block warp groups
    // -- which replaces a full write and a strided read of the fp32 scratch.
    float tk[T64::TPW][4], red[T64::TPW][2];
#pragma unroll
    for (int j = 0; j < T64::TPW; ++j) {
        const int i0 = T64::mi(warp, j) * 16 + (lane >> 2);
        const int d0 = T64::ni(warp, j) * 8 + ((lane & 3) << 1);
        const int o0 = swz(i0, d0, 128);
#pragma unroll
        for (int rp = 0; rp < 2; ++rp) {
            const int r0    = rp * 2;
            const float2 kv = un2(*reinterpret_cast<const uint32_t*>(sK + o0 + rp * (8 * 128)));
            tk[j][r0]       = kv.x * dka[j][r0];
            tk[j][r0 + 1]   = kv.y * dka[j][r0 + 1];
        }
        float s0 = tk[j][0] + tk[j][2], s1 = tk[j][1] + tk[j][3];
#pragma unroll
        for (int m = 4; m <= 16; m <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, m);
            s1 += __shfl_xor_sync(0xffffffffu, s1, m);
        }
        red[j][0] = s0;
        red[j][1] = s1;
    }
    __syncthreads();  // sQ4 readers have passed
    {                 // dg seed for the last row: dgk * 2^Gn + sum_i k_i dk_i
        if (lane < 4) {
            const int mrow = T64::mi(warp, 0) * 128;
#pragma unroll
            for (int j = 0; j < T64::TPW; ++j) {
                const int d0       = T64::ni(warp, j) * 8 + ((lane & 3) << 1);
                sQ4[mrow + d0]     = red[j][0];
                sQ4[mrow + d0 + 1] = red[j][1];
            }
        }
        __syncthreads();
        if (tid < 128) {
            float x = sDgk[tid] * ex2(sG[(L - 1) * GST + tid]);
#pragma unroll
            for (int p = 0; p < 4; ++p) x += sQ4[p * 128 + tid];
            sDgk[tid] = x;
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < T64::TPW; ++j) {
        const int i0 = T64::mi(warp, j) * 16 + (lane >> 2);
        const int d0 = T64::ni(warp, j) * 8 + ((lane & 3) << 1);
        const int o0 = swz(i0, d0, 128), t0 = i0 * GS2 + d0;
#pragma unroll
        for (int rp = 0; rp < 2; ++rp) {
            const int r0 = rp * 2, i = i0 + rp * 8;
            const float2 qv = un2(*reinterpret_cast<const uint32_t*>(sQ + o0 + rp * (8 * 128)));
            float v0        = qv.x * dqa[j][r0] - tk[j][r0];
            float v1        = qv.y * dqa[j][r0 + 1] - tk[j][r0 + 1];
            if (i == L - 1) {
                v0 += sDgk[d0];
                v1 += sDgk[d0 + 1];
            }
            *reinterpret_cast<float2*>(sDG2 + t0 + rp * (8 * GS2)) = make_float2(v0, v1);
        }
    }
    // region X was last read in the phase-A products, before the barrier above
    // dq stays in registers from here to the end of B6: it frees a 16 KB tile
    // for the third intra-chunk operand and removes a full read-modify-write of
    // the staging plane.

    // =================== phase B ===================
    cpasync_wait<0>();
    __syncthreads();

    float dAacc[TA::TPW][4];
    zero_acc<TA::TPW>(dAacc);
    const bool lower_tile = TA::ni(warp, 0) < 2 * (TA::mi(warp, 0) + 1);
    const int row64       = T64::mi(warp, 0) * 16 + (lane >> 2);

    // ---- B5.  beta_j scales the whole column j of BOTH dA products, so it is
    // applied once on the accumulator instead of being baked into each B
    // operand: the v*beta staging pass (a 16 KB read plus a 16 KB write of the
    // whole tile) and the barrier that published it both disappear, and v feeds
    // the product straight from its own tile.
    if (lower_tile) gemm_acc<64, 64, 128, BNW, 0, 0, 128, 128>(dAacc, sDV2, sV, warp, lane);
    // The second operand no longer carries beta either; building it here (nobody
    // reads sScr until the barrier below) overlaps it with the dv product.
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int o = e4o + m * (16 * 128);
        const F4 g  = ld4f(sG + e4G + m * (16 * GST));
        F4 x        = ld4bf(sK + o);
#pragma unroll
        for (int t = 0; t < 4; ++t) x.v[t] *= ex2(g.v[t]);
        st4bf(sScr + o, x);
    }
    {
        float acc[T64::TPW][4];
        zero_acc<T64::TPW>(acc);
        gemm_acc<64, 128, 64, BNW, 1, 1, 64, 128>(acc, sAb, sDV2, warp, lane);
        float r0 = 0.f, r1 = 0.f;
#pragma unroll
        for (int j = 0; j < T64::TPW; ++j) {
            const int o0 = swz(T64::mi(warp, j) * 16 + (lane >> 2), T64::ni(warp, j) * 8 + ((lane & 3) << 1), 128);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const float2 vv2 = un2(*reinterpret_cast<const uint32_t*>(sV + o0 + rp * (8 * 128)));
                const float c    = acc[j][rp * 2] * vv2.x + acc[j][rp * 2 + 1] * vv2.y;
                if (rp)
                    r1 += c;
                else
                    r0 += c;
            }
        }
        red_flush(sPart, row64, r0, r1, lane);
        // dv is staged in the dq tile, which is idle until B6 needs it as an
        // operand buffer.  Not recycling sScr here lets the dv store and the
        // next operand build share one barrier region instead of three.
#pragma unroll
        for (int j = 0; j < T64::TPW; ++j) {
            const int i0 = T64::mi(warp, j) * 16 + (lane >> 2);
            const int o0 = swz(i0, T64::ni(warp, j) * 8 + ((lane & 3) << 1), 128);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const float b = sBeta[i0 + rp * 8];
                *reinterpret_cast<uint32_t*>(sDqS + o0 + rp * (8 * 128)) =
                    bf2(acc[j][rp * 2] * b, acc[j][rp * 2 + 1] * b);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 2; ++m) {  // coalesced dv store
        if (e8r + m * 32 < L)
            *reinterpret_cast<uint4*>(dvo + base + e8g + (size_t)(m * 32) * gs) =
                *reinterpret_cast<const uint4*>(sDqS + e8o + m * (32 * 128));
    }
    if (lower_tile) gemm_acc<64, 64, 128, BNW, 0, 0, 128, 128>(dAacc, sDW, sScr, warp, lane);
    __syncthreads();
    {  // dAkk = -strict( A^T * strict(dA) * A^T )
        bf16* sdAb = sScr;
        bf16* sXb  = sScr + 64 * 64;
#pragma unroll
        for (int j = 0; j < TA::TPW; ++j) {
            const int i0 = TA::mi(warp, j) * 16 + (lane >> 2);
            const int c0 = TA::ni(warp, j) * 8 + ((lane & 3) << 1);
            const int o0 = swz(i0, c0, 64);
            // beta_j, factored out of both dA products, enters here
            const float2 bc = *reinterpret_cast<const float2*>(sBeta + c0);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const int i = i0 + rp * 8;
                *reinterpret_cast<uint32_t*>(sdAb + o0 + rp * (8 * 64)) =
                    bf2((c0 < i) ? dAacc[j][rp * 2] * bc.x : 0.f, (c0 + 1 < i) ? dAacc[j][rp * 2 + 1] * bc.y : 0.f);
            }
        }
        __syncthreads();
        float xacc[TA::TPW][4];
        zero_acc<TA::TPW>(xacc);
        if (lower_tile) gemm_acc64_range<1, 1>(xacc, sAb, sdAb, warp, lane, TA::mi(warp, 0), 4);
#pragma unroll
        for (int j = 0; j < TA::TPW; ++j) {
            const int o0 = swz(TA::mi(warp, j) * 16 + (lane >> 2), TA::ni(warp, j) * 8 + ((lane & 3) << 1), 64);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp)
                *reinterpret_cast<uint32_t*>(sXb + o0 + rp * (8 * 64)) = bf2(xacc[j][rp * 2], xacc[j][rp * 2 + 1]);
        }
        __syncthreads();
        zero_acc<TA::TPW>(xacc);
        if (lower_tile) gemm_acc64_range<0, 0>(xacc, sXb, sAb, warp, lane, 0, TA::ni(warp, 0) / 2 + 1);
#pragma unroll
        for (int j = 0; j < TA::TPW; ++j) {
            const int i0 = TA::mi(warp, j) * 16 + (lane >> 2);
            const int c0 = TA::ni(warp, j) * 8 + ((lane & 3) << 1);
            const int o0 = swz(i0, c0, 64);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const int i = i0 + rp * 8;
                *reinterpret_cast<uint32_t*>(sdAkk + o0 + rp * (8 * 64)) =
                    bf2((c0 < i) ? -xacc[j][rp * 2] : 0.f, (c0 + 1 < i) ? -xacc[j][rp * 2 + 1] : 0.f);
            }
        }
    }
    __syncthreads();
    {  // dkb = A^T * dw, folded straight into the phase-A dk_inter registers.
        // B6's merge owns exactly the same (row, channel) elements as this
        // epilogue (both use Tiling<64,128,16>), so dk never needs an fp32
        // staging plane: it stays in registers from the state product to the
        // final bf16 store, which drops a 34.8 KB plane and three shared
        // round-trips of the whole 64x128 accumulator.
        float acc[T64::TPW][4];
        zero_acc<T64::TPW>(acc);
        gemm_acc<64, 128, 64, BNW, 1, 1, 64, 128>(acc, sAb, sDW, warp, lane);
        float r0s = 0.f, r1s = 0.f;
#pragma unroll
        for (int j = 0; j < T64::TPW; ++j) {
            const int i0 = T64::mi(warp, j) * 16 + (lane >> 2);
            const int d0 = T64::ni(warp, j) * 8 + ((lane & 3) << 1);
            const int o0 = swz(i0, d0, 128), t0 = i0 * GS2 + d0, q0 = i0 * GST + d0;
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const int r0 = rp * 2, i = i0 + rp * 8;
                float* const pG = sDG2 + t0 + rp * (8 * GS2);
                float2 gcur     = *reinterpret_cast<const float2*>(pG);
                const float2 gl = *reinterpret_cast<const float2*>(sG + q0 + rp * (8 * GST));
                const float2 kv = un2(*reinterpret_cast<const uint32_t*>(sK + o0 + rp * (8 * 128)));
                const float b   = sBeta[i];
                const float ge0 = ex2(gl.x), ge1 = ex2(gl.y);
                const float p0 = acc[j][r0] * ge0, p1 = acc[j][r0 + 1] * ge1;
                gcur.x += kv.x * b * p0;
                gcur.y += kv.y * b * p1;
                const float c = p0 * kv.x + p1 * kv.y;
                if (rp)
                    r1s += c;
                else
                    r0s += c;
                *reinterpret_cast<float2*>(pG) = gcur;
                dka[j][r0] += p0 * b;
                dka[j][r0 + 1] += p1 * b;
            }
        }
        red_flush(sPart, row64, r0s, r1s, lane);
    }
    __syncthreads();

    // ---- B6: intra-chunk row and column sweeps.
    //
    // Both sweeps sum over tokens, not channels, so a per-channel rescaling of a
    // whole 16-token block factors *out* of the sum.  That lets every block use
    // one operand anchored at its own sub-block boundary -- built once for the
    // whole chunk instead of once per (row-block, column-block) pair -- with the
    // anchor difference applied to the fp32 accumulator between k-steps:
    //     S_t = S_{t-1} * 2^(G_anchor(t) - G_anchor(t-1))  +  P_t
    // Every exponent is <= 0, so the rescale can only underflow.  This drops the
    // operand rebuilds from 30 sixteen-row passes to 8, and lets the products
    // run as one 64x128 output (32 columns per warp) instead of four 16x128
    // outputs (8 columns per warp), which is 2.4x less ldmatrix traffic.
    bf16* sScr2      = sV;
    using TB         = Tiling<64, 128, BNW>;
    const int bmi    = TB::mi(warp, 0);  // this warp's 16-row block
    const int row64b = bmi * 16 + (lane >> 2);
    int bni[TB::TPW], bd0[TB::TPW];
#pragma unroll
    for (int j = 0; j < TB::TPW; ++j) {
        bni[j] = TB::ni(warp, j);
        bd0[j] = bni[j] * 8 + ((lane & 3) << 1);
    }
    float a1[TB::TPW][4], a2[TB::TPW][4], bb[TB::TPW][4];
    zero_acc<TB::TPW>(a1);
    zero_acc<TB::TPW>(a2);
    zero_acc<TB::TPW>(bb);

    bf16* sQcb = sDqS;  // freed by keeping dq in registers
    {                   // all three intra-chunk operands in one pass over the gate plane:
        //   kcb_j  = k_j       * 2^(G[16*(j/16)]    - G_j)   (row sweep)
        //   qcb_i  = q_i       * 2^(G_i - G[16*(i/16)+15])   (column sweep)
        //   kcb2_i = k_i*beta_i* 2^(G_i - G[16*(i/16)+15])   (column sweep)
        // Sharing the pass halves the gate-plane and k reads and drops a barrier.
        // Walking four *consecutive* rows of one sub-block (instead of one row
        // of each of the four blocks) makes both anchors thread-invariant: the
        // pass reads two gate rows per thread rather than eight, and holds 8
        // anchor registers instead of 32.
        const int bm = tid >> 7, bq = (tid >> 5) & 3, bc = (tid & 31) << 2;
        const int rq = bm * 16 + bq * 4;
        const F4 aS  = ld4f(sG + (bm * 16) * GST + bc);
        const F4 aE  = ld4f(sG + (bm * 16 + 15) * GST + bc);
#pragma unroll
        for (int u4 = 0; u4 < 4; ++u4) {
            const int i = rq + u4, o = swz(i, bc, 128);
            const F4 g  = ld4f(sG + i * GST + bc);
            const F4 kv = ld4bf(sK + o);
            F4 qv = ld4bf(sQ + o), kc = kv, kb = kv;
            const float b = sBeta[i];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                kc.v[t] *= ex2(aS.v[t] - g.v[t]);
                const float f = ex2(g.v[t] - aE.v[t]);
                qv.v[t] *= f;
                kb.v[t] *= b * f;
            }
            st4bf(sScr + o, kc);
            st4bf(sQcb + o, qv);
            st4bf(sScr2 + o, kb);
        }
    }
    __syncthreads();
    {  // row sweep: t ascending, anchor moves 16t -> 16(t+1)
        const LdA A1      = a0_addr<64>(sdAqk, lane);
        const uint32_t dA = sad(sdAkk) - sad(sdAqk);
        const LdA B       = b1_addr_p<128>(sScr, lane);
        float2 gp[TB::TPW];
#pragma unroll
        for (int j = 0; j < TB::TPW; ++j) gp[j] = *reinterpret_cast<const float2*>(sG + bd0[j]);
        for (int t = 0; t <= bmi; ++t) {
            if (t) {
#pragma unroll
                for (int j = 0; j < TB::TPW; ++j) {
                    const float2 gc = *reinterpret_cast<const float2*>(sG + (16 * t) * GST + bd0[j]);
                    const float f0 = ex2(gc.x - gp[j].x), f1 = ex2(gc.y - gp[j].y);
                    gp[j] = gc;
                    a1[j][0] *= f0;
                    a1[j][1] *= f1;
                    a1[j][2] *= f0;
                    a1[j][3] *= f1;
                    a2[j][0] *= f0;
                    a2[j][1] *= f1;
                    a2[j][2] *= f0;
                    a2[j][3] *= f1;
                }
            }
            uint32_t af1[4], af2[4], bf_[4];
            const uint32_t ao = A1.base + bmi * (32u * 64) + ((t << 5) ^ A1.x);
            ldm_x4(af1, ao);
            ldm_x4(af2, ao + dA);
#pragma unroll
            for (int j = 0; j < TB::TPW; j += 2) {
                ldm_x4t(bf_, B.base + t * (32u * 128) + ((bni[j] ^ B.x) << 4));
                mma16816p(a1[j], af1, bf_[0], bf_[1]);
                mma16816p(a2[j], af2, bf_[0], bf_[1]);
                mma16816p(a1[j + 1], af1, bf_[2], bf_[3]);
                mma16816p(a2[j + 1], af2, bf_[2], bf_[3]);
            }
        }
    }
    {  // column sweep: u descending, anchor moves 16(u+1)+15 -> 16u+15.
        // Both products land in one accumulator; only their sum is ever used.
        const LdA A1      = a1_addr<64>(sdAqk, lane);
        const uint32_t dA = sad(sdAkk) - sad(sdAqk);
        const LdA B1      = b1_addr_p<128>(sQcb, lane);
        const uint32_t dB = sad(sScr2) - sad(sQcb);
        float2 gp[TB::TPW];
#pragma unroll
        for (int j = 0; j < TB::TPW; ++j) gp[j] = *reinterpret_cast<const float2*>(sG + 63 * GST + bd0[j]);
        for (int u = 3; u >= bmi; --u) {
            if (u < 3) {
#pragma unroll
                for (int j = 0; j < TB::TPW; ++j) {
                    const float2 gc = *reinterpret_cast<const float2*>(sG + (16 * u + 15) * GST + bd0[j]);
                    const float f0 = ex2(gp[j].x - gc.x), f1 = ex2(gp[j].y - gc.y);
                    gp[j] = gc;
                    bb[j][0] *= f0;
                    bb[j][1] *= f1;
                    bb[j][2] *= f0;
                    bb[j][3] *= f1;
                }
            }
            uint32_t af1[4], af2[4], bf_[4], bg_[4];
            const uint32_t ao = A1.base + u * (32u * 64) + a1x<0>(A1.x, bmi);
            ldm_x4t(af1, ao);
            ldm_x4t(af2, ao + dA);
#pragma unroll
            for (int j = 0; j < TB::TPW; j += 2) {
                const uint32_t bo = B1.base + u * (32u * 128) + ((bni[j] ^ B1.x) << 4);
                ldm_x4t(bf_, bo);
                ldm_x4t(bg_, bo + dB);
                mma16816p(bb[j], af1, bf_[0], bf_[1]);
                mma16816p(bb[j], af2, bg_[0], bg_[1]);
                mma16816p(bb[j + 1], af1, bf_[2], bf_[3]);
                mma16816p(bb[j + 1], af2, bg_[2], bg_[3]);
            }
        }
    }
    {  // merge: one read-modify-write per element of the fp32 dg / dk planes;
        // dq is folded straight into its phase-A registers.  beta and the dA
        // diagonal depend only on the row, so they are read once per row pair.
        float r0s = 0.f, r1s = 0.f;
        const int i0       = bmi * 16 + (lane >> 2);
        const float bt[2]  = {sBeta[i0], sBeta[i0 + 8]};
        const float dgd[2] = {sDiag[i0], sDiag[i0 + 8]};
#pragma unroll
        for (int j = 0; j < TB::TPW; ++j) {
            const int d0    = bd0[j];
            const float2 gA = *reinterpret_cast<const float2*>(sG + (bmi * 16) * GST + d0);
            const float2 gZ = *reinterpret_cast<const float2*>(sG + (bmi * 16 + 15) * GST + d0);
            const int o0    = swz(i0, d0, 128);
#pragma unroll
            for (int rp = 0; rp < 2; ++rp) {
                const int r0 = rp * 2, i = i0 + rp * 8;
                const int o     = o0 + rp * (8 * 128);
                const int t0    = i * GS2 + d0;
                float2 gcur     = *reinterpret_cast<const float2*>(sDG2 + t0);
                const float2 gi = *reinterpret_cast<const float2*>(sG + i * GST + d0);
                const float2 kv = un2(*reinterpret_cast<const uint32_t*>(sK + o));
                const float2 qv = un2(*reinterpret_cast<const uint32_t*>(sQ + o));
                const float b = bt[rp], dg = dgd[rp];
                const float rf0 = ex2(gi.x - gA.x), rf1 = ex2(gi.y - gA.y);
                const float cf0 = ex2(gZ.x - gi.x), cf1 = ex2(gZ.y - gi.y);
                const float dqx0 = a1[j][r0] * rf0, dqx1 = a1[j][r0 + 1] * rf1;
                const float dl0 = a2[j][r0] * rf0, dl1 = a2[j][r0 + 1] * rf1;
                const float c = dl0 * kv.x + dl1 * kv.y;
                if (rp)
                    r1s += c;
                else
                    r0s += c;
                const float dk20 = dl0 * b, dk21 = dl1 * b;
                const float dkt0 = bb[j][r0] * cf0, dkt1 = bb[j][r0 + 1] * cf1;
                gcur.x += qv.x * dqx0 + dk20 * kv.x - kv.x * dkt0;
                gcur.y += qv.y * dqx1 + dk21 * kv.y - kv.y * dkt1;
                dka[j][r0] += dk20 + dkt0 + dg * qv.x;
                dka[j][r0 + 1] += dk21 + dkt1 + dg * qv.y;
                dqa[j][r0] += dqx0 + dg * kv.x;
                dqa[j][r0 + 1] += dqx1 + dg * kv.y;
                *reinterpret_cast<float2*>(sDG2 + t0) = gcur;
            }
        }
        red_flush(sPart, row64b, r0s, r1s, lane);
    }
    __syncthreads();    // operand tiles released
    bf16* sDkS = sScr;  // freed by the B6 sweeps
#pragma unroll
    for (int j = 0; j < TB::TPW; ++j) {
        const int o0 = swz(bmi * 16 + (lane >> 2), bd0[j], 128);
#pragma unroll
        for (int rp = 0; rp < 2; ++rp) {
            *reinterpret_cast<uint32_t*>(sDqS + o0 + rp * (8 * 128)) = bf2(dqa[j][rp * 2], dqa[j][rp * 2 + 1]);
            *reinterpret_cast<uint32_t*>(sDkS + o0 + rp * (8 * 128)) = bf2(dka[j][rp * 2], dka[j][rp * 2 + 1]);
        }
    }
    __syncthreads();

    if (tid < 64) {
        float x = 0.f;
#pragma unroll
        for (int w = 0; w < BNW; ++w) x += sPart[w * 64 + tid];
        sRed[tid] = x;
    }

    // ---- finalise
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        if (e8r + m * 32 < L)
            *reinterpret_cast<uint4*>(dq + base + e8g + (size_t)(m * 32) * gs) =
                *reinterpret_cast<const uint4*>(sDqS + e8o + m * (32 * 128));
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        if (e8r + m * 32 < L)
            *reinterpret_cast<uint4*>(dk + base + e8g + (size_t)(m * 32) * gs) =
                *reinterpret_cast<const uint4*>(sDkS + e8o + m * (32 * 128));
    }
    {  // dg = reverse cumsum of sDG2, as four 16-row segments.  The segment is
        // read once into registers and reused after the fixup barrier, halving
        // this pass's shared reads and removing its second dependent add chain.
        float* sR4      = reinterpret_cast<float*>(sScr);
        const float* sp = sDG2 + qd * 16 * GS2 + dd;
        float r[16];
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            r[i] = sp[i * GS2];
            tot += r[i];
        }
        __syncthreads();
        sR4[qd * 128 + dd] = tot;
        __syncthreads();
        float a = 0.f;
        for (int p = qd + 1; p < 4; ++p) a += sR4[p * 128 + dd];
        float* dp = dgo + base + (size_t)(qd * 16 + 15) * gs + dd;
#pragma unroll
        for (int i = 15; i >= 0; --i, dp -= gs) {
            a += r[i];
            if (qd * 16 + i < L) *dp = a;
        }
    }
    // sRed was published before the two barriers of the reverse cumsum above.
    if (tid < 64 && tid < L) dbeta[(size_t)(t0 + tid) * H + h] = sRed[tid];
}

// ==================== k_meta (from kernel.cu) ====================
extern "C" __global__ void
k_meta(const int* __restrict__ cu,
       int N,
       int NCS,
       int* __restrict__ cs_t0,
       int* __restrict__ cs_len,
       int* __restrict__ seq_c0,
       int* __restrict__ seq_nc) {
    if (threadIdx.x == 0) {
        int slot = 0;
        for (int n = 0; n < N; ++n) {
            const int s = cu[n], e = cu[n + 1];
            const int len = e - s;
            const int nc  = (len + BT - 1) / BT;
            seq_c0[n]     = slot;
            seq_nc[n]     = nc;
            for (int c = 0; c < nc; ++c) {
                cs_t0[slot]  = s + c * BT;
                cs_len[slot] = min(BT, len - c * BT);
                ++slot;
            }
        }
        for (int i = slot; i < NCS; ++i) {
            cs_t0[i]  = 0;
            cs_len[i] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// The launcher must restate these to size its grids, blocks and dynamic
// shared-memory allocations, and to lay out the workspace arena. They cannot
// be read back from the module (NVRTC internalises a __device__ variable no
// kernel references), so cuda_bwd_host.py owns them and passes them in as -D.
// Any future change to the kernel's tiling then becomes a compile error naming
// the constant, instead of a silently wrong launch geometry.
#ifdef KDA_BWD_CHECK_CONSTANTS
static_assert(BT == KDA_BWD_BT, "cuda_bwd_host.BT disagrees with the kernel");
static_assert(DH == KDA_BWD_DH, "cuda_bwd_host.DH disagrees with the kernel");
static_assert(BNT == KDA_BWD_BNT, "cuda_bwd_host.BNT disagrees with the kernel");
static_assert(SMEM_PREP == KDA_BWD_SMEM_PREP, "cuda_bwd_host.SMEM_PREP disagrees with the kernel");
static_assert(SMEM_SCAN == KDA_BWD_SMEM_SCAN, "cuda_bwd_host.SMEM_SCAN disagrees with the kernel");
static_assert(SMEM_BWD == KDA_BWD_SMEM_BWD, "cuda_bwd_host.SMEM_BWD disagrees with the kernel");
#endif
