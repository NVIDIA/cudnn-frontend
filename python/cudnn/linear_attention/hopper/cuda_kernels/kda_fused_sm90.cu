// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Kimi Delta Attention (KDA) chunked prefill for Hopper (sm90), single fused
// kernel. Compiled with NVRTC and launched through the driver API, following
// linear_attention/cake/.
//
// Provenance: Kernel Factory campaign ph0w0d6jrh4k91j87a1xgkmn78, solution
// kda_r4_v9 (kernel de6c22b8). Machine-generated; included because it is
// measurably correct and fast, not because it was reviewed line by line.
//
// Verified on H100 80GB HBM3 at the production gate (gate_lower_bound = -5)
// with a non-zero initial_state, geomean over ten shapes: 53.9 us against
// FlashKDA's 438.3 us (8.12x), winning all ten. GPU time is 21.6 us of 23.0 us
// wall at 2048/12/1 under CUPTI -- 1.6-1.8x of the minimum-memory-traffic
// roofline, so it is close to bandwidth-bound. Accuracy 5.0e-03 to 1.0e-02
// against an fp64 oracle, against FlashKDA's own 7.5e-03 to 1.0e-02.
//
// Two edits from the campaign artifact, both mechanical:
//   * the host launcher (kda_launch / kda_workspace_bytes) is removed -- NVRTC
//     compiles device code only, and the launch is reproduced in cuda_host.py;
//   * kda_fused is given C linkage so it can be looked up by plain name;
//   * the two host-only includes are replaced (see below).
// The device code itself is unmodified.

// "kda.cuh" declared only kda_launch/kda_workspace_bytes, both removed above,
// and <cstdint> is a host C++ header NVRTC does not ship. NVRTC gets the two
// fixed-width types this body actually uses directly; nvcc supplied them
// through the <cuda_runtime.h> chain.
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#include <cuda_bf16.h>
#include <mma.h>

namespace {

using bf16     = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

constexpr int kDim      = 128;
constexpr int kChunk    = 16;
constexpr float kLog2E  = 1.4426950408889634f;
constexpr float kQScale = 0.08838834764831845f;
// Seed-truncation budget.  Per chunk the incoming state is attenuated by at
// least max_d exp(sum_t g[t][d]) on every key channel, and the delta-rule
// factors (I - beta_t k_t k_t^T) are non-expansive for the L2-normalised keys
// this op contracts for (eigenvalues 1 and 1-beta with beta in (0,1)), so they
// can only shrink it further -- the bound needs neither the key norms nor beta.
//
// kMixLog2 is how much the dropped state can be amplified on its way to an
// output.  For o it is exactly 1: |sum_k q'[t,k] S[k,v]| <= max_k A_k max|S|
// sum_k |q'[t,k]|, and q is L2-normalised and pre-scaled by 1/sqrt(D), so that
// last sum is at most 1.  sqrt(D) = 2^3.5 covers the state path as well.  The
// old pair (kLogEps -24, kMixLog2 7) demanded 2^-31.5 of attenuation, which the
// steepest heads (exp(A_log) ~ 1.5, where max_d sum_t g only reaches ~ -22)
// miss by a hair -- and one marginal CTA out of 264 costs every other CTA a
// whole extra chunk step, because they all run in the same wave.  Requiring
// 2^-23.5 instead still bounds the discarded output term by 2^-20 * max|S|
// ~ 1e-6, three orders of magnitude under the 2e-3 tolerance, and makes a
// single warm-up chunk sufficient for every head in the shipped gate range.
constexpr float kLogEps  = -20.0f;
constexpr float kMixLog2 = 3.5f;  // sqrt(kDim) entry-mixing allowance
// Padded leading dimensions.  A bf16 tile with ld=128 puts every ldmatrix row
// 256 B apart, i.e. all eight rows of a fragment in the same bank group; NCU
// charged 63% of PREP's shared wavefronts to exactly that.  272 B (ld=136)
// rotates the rows across all 32 banks.  Same idea for the fp32 16x16 tiles,
// where ld=20 floats replaces a 64 B stride that folded onto two banks.
constexpr int kLdA = kDim + 8;  // 136 bf16
constexpr int kPS  = 40;        // fp32 16x16 tiles; 40 words puts the four
                                // row groups of a warp store on disjoint bank octets
constexpr int kLdT = 24;        // bf16 16x16 tile (Tinv)
// Tinv V is kept TOKEN-major with the same padded stride.  PREP writes it
// straight out of the wmma accumulator, two adjacent v channels per 4 B store;
// with ld=136 that lands one lane per bank, where the old [v][token] tile at
// ld=16 was a 96-way conflict and the single most expensive line in the kernel.
constexpr int kLdU = kDim + 8;  // 136 bf16
// q' is published straight out of the gate scan instead of being copied into
// place by a second pass.  The natural GMMA B tile puts consecutive 8-channel
// core-matrix groups 128 elements apart, and a scan thread owning one channel
// PAIR then has all eight of a warp's groups landing on the same four banks --
// an 8-way conflict, which is why the copy existed.  The descriptor's leading
// byte offset may be any multiple of 16, so padding the group stride to 136
// rotates the groups across all 32 banks and the direct store is conflict-free.
constexpr int kQgG   = kDim + 8;  // 136 bf16 per 8-channel group
constexpr int kQgLbo = 2 * kQgG;  // 272 B

#define DEV __device__ __forceinline__

DEV uint32_t
smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// GMMA B-operand descriptor for a [K, 64] tile (leading 1024 B, stride 128 B).
DEV uint64_t
gdesc64(const void* p) {
    const uint32_t a = smem_u32(p);
    uint64_t d       = static_cast<uint64_t>((a & 0x3ffffu) >> 4);
    d |= static_cast<uint64_t>(1024u >> 4) << 16;
    d |= static_cast<uint64_t>(128u >> 4) << 32;
    return d;
}
// Same atom, but only two N atoms are present ([K, 16] tile).  LBO is the byte
// distance between core matrices along K, which a tile is free to pad.
template <int LBO = 256>
DEV uint64_t
gdesc16(const void* p) {
    const uint32_t a = smem_u32(p);
    uint64_t d       = static_cast<uint64_t>((a & 0x3ffffu) >> 4);
    d |= static_cast<uint64_t>(LBO >> 4) << 16;
    d |= static_cast<uint64_t>(128u >> 4) << 32;
    return d;
}

DEV uint32_t
pack2(float lo, float hi) {
    const __nv_bfloat162 x = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&x);
}

#define WG_L8 "{%0,%1,%2,%3,%4,%5,%6,%7}"
#define WG_IO8(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
#define WG_OUT8(d) "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])

DEV void
wg16_zero(float* d, const uint32_t* a, uint64_t b) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 " WG_L8 ", {%8,%9,%10,%11}, %12, p, %14, %15, %16;\n}\n"
        : WG_OUT8(d)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(0), "n"(1), "n"(1), "n"(0));
}
DEV void
wg16_add(float* d, const uint32_t* a, uint64_t b) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 " WG_L8 ", {%8,%9,%10,%11}, %12, p, %14, %15, %16;\n}\n"
        : WG_IO8(d)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(1), "n"(1), "n"(1), "n"(0));
}

#define WG_L32                                                \
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define WG_IO32(d)                                                                                              \
    "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), \
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),  \
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), \
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

#define WG_OUT32(d)                                                                                             \
    "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), \
        "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]),  \
        "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), \
        "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])

DEV void
wg64_zero(float* d, const uint32_t* a, uint64_t b) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " WG_L32
        ", {%32,%33,%34,%35}, %36, p, %38, %39, %40;\n}\n"
        : WG_OUT32(d)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(0), "n"(1), "n"(1), "n"(0));
}

DEV void
wg64_add(float* d, const uint32_t* a, uint64_t b) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " WG_L32
        ", {%32,%33,%34,%35}, %36, p, %38, %39, %40;\n}\n"
        : WG_IO32(d)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(1), "n"(1), "n"(1), "n"(0));
}

// One LDSM fetches a whole 16x16 bf16 tile into the four mma.sync A-fragment
// registers; nvcc's WMMA load_matrix_sync lowers the same tile to eight generic
// 16 B loads (LD.E, not LDS) plus a MOVM transpose.
DEV void
ldm4(uint32_t* d, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(smem_u32(p)));
}
// Transposing LDSM: the fragment lands with two adjacent ROWS per register,
// which is what the mma B operand wants when the contraction index is the slow
// axis of the tile in shared memory (Tinv W / Tinv V contract over tokens).
DEV void
ldm4t(uint32_t* d, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(smem_u32(p)));
}
DEV void
mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

DEV void
wg_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
DEV void
wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
DEV void
wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
DEV void
async_smem_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
DEV void
mbar_init(uint64_t* b, int c) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(c) : "memory");
}
DEV void
mbar_expect(uint64_t* b, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes) : "memory");
}
DEV void
bulk_load(void* dst, const void* src, int bytes, uint64_t* b) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
        "l"(src),
        "r"(bytes),
        "r"(smem_u32(b))
        : "memory");
}
DEV void
bulk_store(void* dst, const void* src, int bytes) {
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(smem_u32(src)), "r"(bytes)
        : "memory");
}
DEV void
bulk_store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
DEV void
bulk_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
DEV void
mbar_wait(uint64_t* b, int phase) {
    asm volatile(
        "{ .reg .pred p;\n"
        "  W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @p bra D%=;\n"
        "  bra W%=;\n"
        "  D%=: }\n" ::"r"(smem_u32(b)),
        "r"(phase)
        : "memory");
}

// One contiguous per-(chunk, head) record, laid out exactly as the scan wants
// it in shared memory so PREP publishes and SCAN fetches it with one bulk copy.
struct __align__(128) Record {
    bf16 mw[2048];           // -(Tinv W)^T, GMMA B layout, [k=128 channels, n=16 tokens]
    bf16 qg[kChunk * kQgG];  //  (A*q/sqrt(D))^T, padded group stride kQgG
    bf16 kg[2048];           //  beta*k/A * A_total, GMMA B layout [k=16 tokens, n=128 ch]
    bf16 z[256];             //  tril(Q' (beta*U)^T), GMMA B layout [k=16, n=16]
    bf16 ut[kChunk * kLdU];  //  Tinv V, row major [16 tokens][128 v], ld=kLdU
    float av[128];           //  A_total = exp(sum of g over the chunk)
};
static_assert(sizeof(Record) == 17920, "record layout");

DEV int
gmma_wn(int ch, int tok) {
    return (ch >> 3) * 128 + tok * 8 + (ch & 7);
}
DEV int
gmma_kn(int tok, int ch) {
    return (ch >> 6) * 1024 + (tok >> 3) * 512 + (ch & 63) * 8 + (tok & 7);
}

DEV bool
locate_chunk(int vc, const int* __restrict__ cu, int N, int& seq, int& lc, int& t0, int& te) {
    int prefix = 0;
    for (int n = 0; n < N; ++n) {
        const int s = cu[n], e = cu[n + 1];
        const int c = (e - s + kChunk - 1) / kChunk;
        if (vc < prefix + c) {
            seq = n;
            lc  = vc - prefix;
            t0  = s + lc * kChunk;
            te  = e;
            return true;
        }
        prefix += c;
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------- fused ----
namespace {

constexpr int kNS = 2;  // factor stages (producer writes c+1, consumer reads c)
constexpr int kNR = 2;  // raw-input stages

// Raw chunk inputs staged by cp.async.  V carries the ldmatrix padding because
// it is a wmma B operand; g/k/q are only ever read one channel per thread.
struct __align__(128) Raw {
    float g[kChunk * kDim];  // 8192
    bf16 k[kChunk * kDim];   // 4096
    bf16 q[kChunk * kDim];   // 4096
    bf16 v[kChunk * kLdA];   // 4352
    float beta[kChunk];      // 64
    float pad[16];           // 64
};
static_assert(sizeof(Raw) == 20864, "raw layout");

struct __align__(128) FusedSmem {
    Record st[kNS];
    Raw raw[kNR];
    bf16 wq[kChunk * kLdA];
    bf16 u[kChunk * kLdA];
    bf16 ost[kChunk * kLdA];  // output staging, TOKEN-major
    float prod[kChunk * kPS];
    bf16 tib[kChunk * kLdT];
    float red[128];
};

DEV void
cp16(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 16 : 0)
                 : "memory");
}
DEV void
cp4(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 4 : 0)
                 : "memory");
}
DEV void
cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
DEV void
cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

}  // namespace

// One CTA = one warpgroup = one (sequence, head, segment).  It carries the
// whole [128 v, 128 k] state in fp32 wgmma accumulators and ALSO builds each
// chunk's UT/WY factors itself one chunk ahead, so nothing round-trips through
// global memory: the only traffic is q/k/v/g in and o out.
extern "C" __global__
__launch_bounds__(128, 2) void kda_fused(const bf16* __restrict__ gq,
                                         const bf16* __restrict__ gk,
                                         const bf16* __restrict__ gv,
                                         const float* __restrict__ gg,
                                         const float* __restrict__ gbeta,
                                         const int* __restrict__ cu,
                                         const float* __restrict__ gis,
                                         bf16* __restrict__ go,
                                         float* __restrict__ gfs,
                                         int N,
                                         int H,
                                         int P) {
    extern __shared__ __align__(128) char raws[];
    FusedSmem& sm = *reinterpret_cast<FusedSmem*>(raws);

    const int head = blockIdx.x;
    const int seq  = blockIdx.y / P;
    const int seg  = blockIdx.y - seq * P;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int lg = lane >> 2, tg = lane & 3;
    const int HD = H * kDim;

    const int s0 = cu[seq], s1 = cu[seq + 1];
    const int chunks = (s1 - s0 + kChunk - 1) / kChunk;
    const long sbase = (static_cast<long>(seq) * H + head) * kDim * kDim;
    if (chunks == 0) {
        if (seg == 0)
            for (int i = tid; i < kDim * kDim; i += 128) gfs[sbase + i] = gis[sbase + i];
        return;
    }
    const int slen = (chunks + P - 1) / P;
    const int c0   = min(seg * slen, chunks);
    const int c1   = min(c0 + slen, chunks);
    if (c0 >= c1) return;

    // ---- raw input staging -------------------------------------------------
    auto load_raw = [&](int chunk, int rb) {
        Raw& R       = sm.raw[rb];
        const int tb = s0 + chunk * kChunk;
        // g : 16 rows x 512 B, four 16 B lanes-chunks per row
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int i = warp + 4 * it, d = 4 * lane;
            const int tk = tb + i;
            cp16(&R.g[i * kDim + d], gg + static_cast<long>(tk) * HD + head * kDim + d, tk < s1);
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int i = (tid >> 4) + 8 * it, d = 8 * (tid & 15);
            const int tk  = tb + i;
            const long ix = static_cast<long>(tk) * HD + head * kDim + d;
            cp16(&R.k[i * kDim + d], gk + ix, tk < s1);
            cp16(&R.q[i * kDim + d], gq + ix, tk < s1);
            cp16(&R.v[i * kLdA + d], gv + ix, tk < s1);
        }
        if (tid < kChunk) cp4(&R.beta[tid], gbeta + static_cast<long>(tb + tid) * H + head, tb + tid < s1);
        cp_commit();
    };

    // Warm-up length: walk back over whole chunks until the product of the
    // per-chunk state-attenuation bounds is under 2^kLogEps.  c0 <= chunks-1 here
    // so every probed chunk is full, and with L2-normalised keys and beta in
    // (0,1) every delta-rule factor (I - beta k k^T) is non-expansive, so the
    // bound is just max_d exp(sum_t g) times the entry-mixing allowance -- the
    // key norms and beta drop out, taking 16 shuffle reductions, 16 log2 and a
    // barrier per probe with them.
    //
    // The first candidate is ALWAYS the chunk the segment will replay, so it is
    // staged first and probed out of that tile: one cold fetch of g covers both
    // the probe and the gate scan, and the staging latency that the prologue used
    // to pay separately is now the probe's.  sm.red is double-buffered on the
    // parity of w so the fallback loop carries one barrier per probe.
    int w = 0;
    {
        float acc = 0.0f;
        if (c0 > 0) {
            load_raw(c0 - 1, (c0 - 1) % kNR);
            cp_wait<0>();
            __syncthreads();
            const float* gp = sm.raw[(c0 - 1) % kNR].g + tid;
            float su        = 0.0f;
#pragma unroll
            for (int i = 0; i < kChunk; ++i) su += gp[i * kDim];
#pragma unroll
            for (int o = 16; o; o >>= 1) su = fmaxf(su, __shfl_xor_sync(0xffffffffu, su, o));
            if (lane == 0) sm.red[warp] = su;
            __syncthreads();
            acc = fmaxf(fmaxf(sm.red[0], sm.red[1]), fmaxf(sm.red[2], sm.red[3])) * kLog2E + kMixLog2;
            w   = 1;
        }
        while (c0 - w > 0 && acc > kLogEps) {
            const long gb = static_cast<long>(s0 + (c0 - w - 1) * kChunk) * HD + head * kDim + tid;
            float su      = 0.0f;
#pragma unroll
            for (int i = 0; i < kChunk; ++i) su += gg[gb + static_cast<long>(i) * HD];
#pragma unroll
            for (int o = 16; o; o >>= 1) su = fmaxf(su, __shfl_xor_sync(0xffffffffu, su, o));
            float* rd = sm.red + 4 * (w & 1);
            if (lane == 0) rd[warp] = su;
            __syncthreads();
            acc += fmaxf(fmaxf(rd[0], rd[1]), fmaxf(rd[2], rd[3])) * kLog2E + kMixLog2;
            ++w;
        }
    }
    const int cs = c0 - w;

    // A truncated segment starts from a provably negligible state, so its first
    // chunk contracts nothing: the eight state wgmma, the query contraction and
    // the 128 decay multiplies all collapse, and the rank-16 update overwrites
    // the accumulator outright (scale-d = 0) instead of adding to zeros.
    const bool seeded = (cs == 0);
    float st[4][32];
    if (seeded) {
        const float* seed = gis + sbase;
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int row     = 64 * rh + warp * 16 + lg + 8 * ((i & 3) >> 1);
                    const int col     = j * 64 + 8 * (i >> 2) + 2 * tg + (i & 1);
                    st[2 * rh + j][i] = seed[row * kDim + col];
                }
    }

    // ---- raw input staging -------------------------------------------------
    // ---- per-chunk factor build (producer half of the loop) ----------------
    auto prep = [&](int rb, int sb) {
        Raw& R      = sm.raw[rb];
        Record& rec = sm.st[sb];
        // Gate prefix scan over a CHANNEL PAIR x TOKEN OCTET tile: a thread owns
        // two adjacent channels for eight tokens, so every g/k/q load and every
        // W/q'/u store moves 4-8 B instead of 2-4 B, and the serial exp2 chain is
        // eight steps instead of sixteen.  The upper octet picks up the lower
        // octet's two prefix sums through shared memory.  The chunk total is just
        // the last prefix, so no separate summation pass over g is needed, and
        // beta*k/A * A_total is folded into the republish, which also removes a
        // whole 16x128 bf16 staging tile from shared memory.
        const int cp = tid & 63, oc = tid >> 6;
        const int dd0 = 2 * cp, t0 = 8 * oc;
        float c0 = 0.0f, c1 = 0.0f;
        if (oc) {
            // Upper-octet threads re-read the lower octet's eight g pairs to build
            // their own prefix.  That is strictly cheaper than exchanging it through
            // shared memory: the lower half does no prefix work at all, and the
            // __syncthreads the exchange needed disappears (barriers were 20% of all
            // warp stalls).
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 gg2 = *reinterpret_cast<const float2*>(&R.g[i * kDim + dd0]);
                c0 += gg2.x;
                c1 += gg2.y;
            }
        }
        float bet[8];
        *reinterpret_cast<float4*>(bet)     = *reinterpret_cast<const float4*>(&R.beta[t0]);
        *reinterpret_cast<float4*>(bet + 4) = *reinterpret_cast<const float4*>(&R.beta[t0 + 4]);
        float d0 = 1.0f, d1 = 1.0f;
        // The three operand loads run one token AHEAD of the stores that consume
        // them.  Everything here lives in shared memory and the publishes go out
        // through uint32 casts, so the compiler has to assume a store could alias
        // the next iteration's load and keeps the whole eight-token body serialised
        // on a shared round trip; issuing the loads first breaks that chain and
        // costs three registers.  NCU put the g load at the top of this kernel's
        // short-scoreboard stalls.
        float2 gg2        = *reinterpret_cast<const float2*>(&R.g[t0 * kDim + dd0]);
        __nv_bfloat162 k2 = *reinterpret_cast<const __nv_bfloat162*>(&R.k[t0 * kDim + dd0]);
        __nv_bfloat162 q2 = *reinterpret_cast<const __nv_bfloat162*>(&R.q[t0 * kDim + dd0]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int t             = t0 + i;
            const float2 gc         = gg2;
            const __nv_bfloat162 kc = k2, qc = q2;
            if (i < 7) {
                const int tn = t + 1;
                gg2          = *reinterpret_cast<const float2*>(&R.g[tn * kDim + dd0]);
                k2           = *reinterpret_cast<const __nv_bfloat162*>(&R.k[tn * kDim + dd0]);
                q2           = *reinterpret_cast<const __nv_bfloat162*>(&R.q[tn * kDim + dd0]);
            }
            c0 += gc.x;
            c1 += gc.y;
            d0              = exp2f(c0 * kLog2E);
            d1              = exp2f(c1 * kLog2E);
            const float kx0 = __bfloat162float(kc.x), kx1 = __bfloat162float(kc.y);
            const float b                                        = bet[i];
            *reinterpret_cast<uint32_t*>(&sm.wq[t * kLdA + dd0]) = pack2(kx0 * d0, kx1 * d1);
            // q' is published ONLY into its GMMA B tile.  An 8-channel group of that
            // tile is a contiguous [16 tok][8 ch] block with row stride 8, i.e. an
            // ldmatrix-addressable tile in its own right, so the Z product reads it
            // straight from there and the second [token][channel] copy disappears.
            *reinterpret_cast<uint32_t*>(&rec.qg[(dd0 >> 3) * kQgG + t * 8 + (dd0 & 7)]) =
                pack2(__bfloat162float(qc.x) * d0 * kQScale, __bfloat162float(qc.y) * d1 * kQScale);
            *reinterpret_cast<uint32_t*>(&sm.u[t * kLdA + dd0]) = pack2(b * kx0 / d0, b * kx1 / d1);
        }
        if (oc) {
            rec.av[dd0]     = d0;
            rec.av[dd0 + 1] = d1;
        }
        __syncthreads();
        // GMMA republish.  Both destinations are walked so that the two bf16 a
        // lane moves are adjacent in the GMMA tile -- channel-adjacent for q',
        // token-adjacent for beta*k/A -- so every transfer is a 4 B load/store
        // pair landing one lane per bank instead of two 2 B ones.
        using FA  = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>;
        using FBc = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major>;
        using FBr = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major>;
        using FC  = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
        // Warps 0-1 own the two 16x16 products and the triangular inverse; warps
        // 2-3 own the GMMA republish.  Splitting them this way removes the barrier
        // that used to sit between the M product and the inverse (both live on
        // warp 0, so a __syncwarp covers it) and stops warps 2-3 idling through the
        // inverse's serial forward substitution.
        if (warp < 2) {
            // Both operands live at [token][channel] with the same stride, and the
            // mma B fragment wants two adjacent CHANNELS per register for a fixed
            // token -- exactly what a plain (non-transposing) LDSM of that tile
            // produces -- so A (W or q') and B ((beta U)^T) share one address form.
            // Matrices 0..3 of the x4 load are (tok 0-7, ch 0-7), (tok 8-15, ch 0-7),
            // (tok 0-7, ch 8-15), (tok 8-15, ch 8-15), which is the A fragment as-is
            // and pairs as {0,2} / {1,3} for the two eight-token halves of B.
            const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
            const int lcol = 8 * (lane >> 4);
            const bf16* ap;
            int astr;
            if (warp == 0) {
                ap   = &sm.wq[lrow * kLdA + lcol];
                astr = kChunk;
            } else {
                ap   = &rec.qg[(lane >> 4) * kQgG + lrow * 8];
                astr = 2 * kQgG;
            }
            float acc[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
#pragma unroll
            for (int kb = 0; kb < kDim / kChunk; ++kb) {
                uint32_t fa[4], fb[4];
                ldm4(fa, ap + kb * astr);
                ldm4(fb, &sm.u[lrow * kLdA + kb * kChunk + lcol]);
                mma16816(acc, fa, fb[0], fb[2]);
                mma16816(acc + 4, fa, fb[1], fb[3]);
            }
            if (warp == 0) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row                = lg + 8 * (e & 1);
                    const int col                = 8 * (e >> 1) + 2 * tg;
                    sm.prod[row * kPS + col]     = acc[2 * e];
                    sm.prod[row * kPS + col + 1] = acc[2 * e + 1];
                }
                __syncwarp();
            } else {
                // Z goes straight from the accumulator into its GMMA B tile: element
                // pair (2e, 2e+1) is two adjacent r-tokens, i.e. two adjacent bf16, so
                // the whole 16x16 masked transpose is four conflict-free 4 B stores.
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int tq   = lg + 8 * (e & 1);
                    const int tr   = 8 * (e >> 1) + 2 * tg;
                    const float v0 = (tr <= tq) ? acc[2 * e] : 0.0f;
                    const float v1 = (tr + 1 <= tq) ? acc[2 * e + 1] : 0.0f;
                    *reinterpret_cast<uint32_t*>(&rec.z[(e >> 1) * 128 + tq * 8 + 2 * tg]) = pack2(v0, v1);
                }
            }
        } else {
            // Same shared-memory aliasing chain as the gate scan: the publish is a
            // uint32 store into the record and the next iteration's operands come out
            // of sm.u, so the sixteen steps serialise on a shared round trip unless
            // the loads are issued a step early.
            const int m = lane & 3, dl = lane >> 2;
            const int i0       = 2 * m;
            const int dd_first = (warp & 1) * 64 + dl;  // idx = (warp&1)*16, it = 0
            float na           = rec.av[dd_first];
            float nu0          = __bfloat162float(sm.u[i0 * kLdA + dd_first]);
            float nu1          = __bfloat162float(sm.u[(i0 + 1) * kLdA + dd_first]);
#pragma unroll
            for (int it = 0; it < 16; ++it) {
                const int idx = (warp & 1) * 16 + it;
                const int dd  = (idx >> 1) * 8 + dl;  // channel
                const int i   = (idx & 1) * 8 + i0;   // even token inside an octet
                const float a = na, u0 = nu0, u1 = nu1;
                if (it < 15) {
                    const int nidx = idx + 1;
                    const int nd   = (nidx >> 1) * 8 + dl;
                    const int ni   = (nidx & 1) * 8 + i0;
                    na             = rec.av[nd];
                    nu0            = __bfloat162float(sm.u[ni * kLdA + nd]);
                    nu1            = __bfloat162float(sm.u[(ni + 1) * kLdA + nd]);
                }
                *reinterpret_cast<uint32_t*>(&rec.kg[(dd >> 6) * 1024 + (i >> 3) * 512 + (dd & 63) * 8 + (i & 7)]) =
                    pack2(u0 * a, u1 * a);
            }
        }
        if (warp == 0) {
            const int col = lane;
            float ti[kChunk];
#pragma unroll
            for (int r = 0; r < kChunk; ++r) {
                float a0 = (r == col) ? 1.0f : 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (int j = 0; j < r; ++j) {
                    const float m = sm.prod[r * kPS + j];
                    if ((j & 3) == 0)
                        a0 = fmaf(-m, ti[j], a0);
                    else if ((j & 3) == 1)
                        a1 = fmaf(-m, ti[j], a1);
                    else if ((j & 3) == 2)
                        a2 = fmaf(-m, ti[j], a2);
                    else
                        a3 = fmaf(-m, ti[j], a3);
                }
                ti[r] = (col <= r) ? ((a0 + a1) + (a2 + a3)) : 0.0f;
            }
            if (col < kChunk) {
#pragma unroll
                for (int r = 0; r < kChunk; ++r) sm.tib[r * kLdT + col] = __float2bfloat16_rn(ti[r]);
            }
        }
        __syncthreads();
        {
            const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
            const int lcol = 8 * (lane >> 4);
            uint32_t ti[4];
            ldm4(ti, &sm.tib[lrow * kLdT + lcol]);
#pragma unroll
            uint32_t bwv[2][2][4];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                ldm4t(bwv[blk][0], &sm.wq[lrow * kLdA + warp * 32 + blk * 16 + lcol]);
                ldm4t(bwv[blk][1], &R.v[lrow * kLdA + warp * 32 + blk * 16 + lcol]);
            }
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int cb = warp * 32 + blk * 16;
                float cw[8], cv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    cw[i] = 0.0f;
                    cv[i] = 0.0f;
                }
                const uint32_t* bw = bwv[blk][0];
                const uint32_t* bv = bwv[blk][1];
                mma16816(cw, ti, bw[0], bw[1]);
                mma16816(cw + 4, ti, bw[2], bw[3]);
                mma16816(cv, ti, bv[0], bv[1]);
                mma16816(cv + 4, ti, bv[2], bv[3]);
                // Publish both 16x16 tiles straight out of the accumulators.  Element
                // pair (2e, 2e+1) is two adjacent channels, which is two adjacent bf16
                // in both destination layouts, so each tile costs four 4 B stores
                // instead of a 16-store round trip through an fp32 scratch tile.
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t  = lg + 8 * (e & 1);
                    const int cc = 8 * (e >> 1) + 2 * tg;
                    *reinterpret_cast<uint32_t*>(&rec.mw[((cb + cc) >> 3) * 128 + t * 8 + (cc & 7)]) =
                        pack2(-cw[2 * e], -cw[2 * e + 1]);
                    *reinterpret_cast<uint32_t*>(&rec.ut[t * kLdU + cb + cc]) = pack2(cv[2 * e], cv[2 * e + 1]);
                }
            }
        }
    };

    // ---- prologue ----------------------------------------------------------
    // w == 1 is the only outcome the shipped gate range produces, and in that
    // case the probe already staged and published raw[cs]; only the rare deeper
    // walk-back has to fetch it.
    if (w != 1) {
        load_raw(cs, cs % kNR);
        if (cs + 1 < c1) {
            load_raw(cs + 1, (cs + 1) % kNR);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
    } else if (cs + 1 < c1) {
        load_raw(cs + 1, (cs + 1) % kNR);
    }
    __syncthreads();
    prep(cs % kNR, cs % kNS);
    __syncthreads();

    int pend = -1;  // chunk whose output sits staged in sm.ost
    for (int chunk = cs; chunk < c1; ++chunk) {
        const Record& R = sm.st[chunk % kNS];
        const bool emit = chunk >= c0;

        // Drain the previous chunk's staged output.  It was written before the
        // closing barrier of the last iteration, so every lane can pull 16 B and a
        // warp's stores cover two full 256 B output rows instead of eight
        // half-used sectors per store instruction.
        if (pend >= 0) {
            const int tb = s0 + pend * kChunk;
#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const int u   = tid + 128 * it;
                const int tok = u >> 4, vb = 8 * (u & 15);
                if (tb + tok < s1)
                    *reinterpret_cast<uint4*>(go + (static_cast<long>(tb + tok) * H + head) * kDim + vb) =
                        *reinterpret_cast<const uint4*>(&sm.ost[tok * kLdA + vb]);
            }
            pend = -1;
        }

        // One descriptor per operand per chunk: stepping k blocks is +512 B on the
        // address, i.e. +32 on the encoded field, so the 32 per-chunk cvta/shift
        // sequences collapse to four.
        const uint64_t dmw = gdesc16(R.mw), dqg = gdesc16<kQgLbo>(R.qg);
        float r[2][8], out[2][8];
        uint32_t ra[2][4];
        // Seed the state contraction's accumulator with Tinv V and let the eight
        // wgmma accumulate straight onto it.  The shared loads move ahead of the
        // fence, where their latency overlaps the repack, and the sixteen adds that
        // used to sit between the wait and the bf16 repack disappear.
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = 64 * rh + warp * 16 + lg + 8 * ((i & 3) >> 1);
                const int tok = 8 * (i >> 2) + 2 * tg + (i & 1);
                r[rh][i]      = __bfloat162float(R.ut[tok * kLdU + row]);
            }
        const bool zero_start = (chunk == cs) && !seeded;
        if (zero_start) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh)
#pragma unroll
                for (int i = 0; i < 4; ++i) ra[rh][i] = pack2(r[rh][2 * i], r[rh][2 * i + 1]);
        } else {
            uint32_t a[8][4];
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int i = 0; i < 32; i += 2)
                        a[4 * j + (i >> 3)][(i & 7) >> 1] = pack2(st[2 * rh + j][i], st[2 * rh + j][i + 1]);
                wg_fence();
#pragma unroll
                for (int kb = 0; kb < 8; ++kb) wg16_add(r[rh], a[kb], dmw + kb * 32);
                wg_commit();
                wg16_zero(out[rh], a[0], dqg);
#pragma unroll
                for (int kb = 1; kb < 8; ++kb) wg16_add(out[rh], a[kb], dqg + kb * (kQgLbo / 8));
                wg_commit();
                if (rh) {
                    // The per-channel decay of the retained state depends on nothing the
                    // two contractions produce, and both bf16 A fragments have already
                    // been built, so it fits entirely inside the second group's shadow.
#pragma unroll
                    for (int j = 0; j < 2; ++j)
#pragma unroll
                        for (int i = 0; i < 32; i += 2) {
                            const int col  = j * 64 + 8 * (i >> 2) + 2 * tg;
                            const float a0 = R.av[col], a1 = R.av[col + 1];
                            st[j][i] *= a0;
                            st[j][i + 1] *= a1;
                            st[2 + j][i] *= a0;
                            st[2 + j][i + 1] *= a1;
                        }
                }
                wg_wait<0>();
#pragma unroll
                for (int i = 0; i < 4; ++i) ra[rh][i] = pack2(r[rh][2 * i], r[rh][2 * i + 1]);
            }
        }
        const uint64_t dz = gdesc16(R.z), dkg = gdesc64(R.kg);
        wg_fence();
        if (zero_start) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                wg64_zero(st[2 * rh + 0], ra[rh], dkg);
                wg64_zero(st[2 * rh + 1], ra[rh], dkg + 128);
            }
        } else {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                wg16_add(out[rh], ra[rh], dz);
                wg64_add(st[2 * rh + 0], ra[rh], dkg);
                wg64_add(st[2 * rh + 1], ra[rh], dkg + 128);
            }
        }
        wg_commit();

        // Build chunk+1's factors while the state update is in flight.  No barrier
        // is needed before the staging load: everything between the previous
        // iteration's closing __syncthreads and here touches only registers and the
        // record the scan is reading, and raw[(chunk+2) % kNR] was last read by
        // prep(chunk) one iteration ago.
        if (chunk + 1 < c1) {
            if (chunk + 2 < c1) {
                load_raw(chunk + 2, (chunk + 2) % kNR);
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncthreads();
            prep((chunk + 1) % kNR, (chunk + 1) % kNS);
        } else {
            // Nothing else separates this chunk's staging store from the drain that
            // read the previous one at the top of this iteration.
            __syncthreads();
        }
        wg_wait<0>();

        if (emit) {
#pragma unroll
            for (int rh = 0; rh < 2; ++rh)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row            = 64 * rh + warp * 16 + lg + 8 * ((i & 3) >> 1);
                    const int tok            = 8 * (i >> 2) + 2 * tg + (i & 1);
                    sm.ost[tok * kLdA + row] = __float2bfloat16_rn(out[rh][i]);
                }
            pend = chunk;
        }
        __syncthreads();
    }
    if (pend >= 0) {
        const int tb = s0 + pend * kChunk;
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int u   = tid + 128 * it;
            const int tok = u >> 4, vb = 8 * (u & 15);
            if (tb + tok < s1)
                *reinterpret_cast<uint4*>(go + (static_cast<long>(tb + tok) * H + head) * kDim + vb) =
                    *reinterpret_cast<const uint4*>(&sm.ost[tok * kLdA + vb]);
        }
        pend = -1;
    }

    if (c1 == chunks) {
        float* fs = gfs + sbase;
#pragma unroll
        for (int rh = 0; rh < 2; ++rh)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 32; i += 2) {
                    const int row = 64 * rh + warp * 16 + lg + 8 * ((i & 3) >> 1);
                    const int col = j * 64 + 8 * (i >> 2) + 2 * tg;
                    float* p      = fs + static_cast<long>(row) * kDim + col;
                    p[0]          = st[2 * rh + j][i];
                    p[1]          = st[2 * rh + j][i + 1];
                }
    }
}

// Host-readable copy of the kernel's dynamic shared-memory requirement, so the
// launcher never has to restate sizeof(FusedSmem) and cannot drift from it.
// The launcher must know sizeof(FusedSmem) to size the dynamic shared-memory
// allocation. It cannot be read back from the module: NVRTC internalises a
// __device__ variable no kernel references (it lands in the cubin as a LOCAL
// symbol, which cuModuleGetGlobal cannot resolve), and nvrtcAddNameExpression
// does not apply to variables. So cuda_host.py owns the number and passes it
// in, and this assert makes any future change to the struct a loud compile
// error in CI rather than a silently undersized launch.
#ifdef KDA_FUSED_SMEM_BYTES
static_assert(sizeof(FusedSmem) == KDA_FUSED_SMEM_BYTES,
              "cuda_host.SMEM_BYTES disagrees with sizeof(FusedSmem); update it");
#endif
