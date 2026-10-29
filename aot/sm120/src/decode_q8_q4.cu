// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Split-KV flash-decode over a q8_0 / q4_0 KV cache, SM120.
// One CTA owns up to RT = 64 query rows (n_q * G) of one KV head, so each KV tile is read and dequantized once per CTA.
// K(+mask) and V tiles stream through a CTA-wide cp.async ring as separate commit groups. QK^T uses a
// (row block x KV column) warp map; P goes through smem as f16, and P.V uses a (D slice x all row blocks) warp map,
// so each V element is dequantized once per tile. q8_0 / q4_0 codes become f16 via the 0x64xx magic-number trick.
// Softmax runs in the exp2 domain with scale*log2(e) folded into Q. Partials are f16 (normalized) + f32 (m, l);
// a second kernel combines the splits.
#include "decode_q8_q4.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>

namespace cudnn_fe_aot_sm120_decode_detail {

constexpr int NQMAX = 8;  // mask staging holds NQMAX token rows per tile

enum KvMode { KV_CONTIG16 = 0, KV_ROWS16 = 1, KV_ROWS8 = 2 };  // must match Config::kern order

struct KParams {
    const unsigned char* q;
    long long q_nbt, q_nbh;  // token / head strides, bytes
    const unsigned char* k;
    const unsigned char* v;
    long long k_nb1, k_nb2, v_nb1, v_nb2;
    const unsigned char* mask;  // nullptr = no mask
    long long m_nb1;
    half* o_part;    // [Hkv*R][nsplit][D], normalized by l
    float* ml_part;  // [Hkv*R][nsplit][2]: m (log2 domain), l
    unsigned char* out;
    long long o_nbh, o_nbt;
    int nq, Hq, Hkv, G, R, kv, nsplit;
    float qscale;  // scale * log2(e), folded into Q
};

}  // namespace cudnn_fe_aot_sm120_decode_detail

using namespace cudnn_fe_aot_sm120_decode_detail;

namespace {

constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ unsigned
smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
// Copies `bytes` (0..16) and zero-fills the rest of the 16-byte destination.
__device__ __forceinline__ void
cp_async16(void* s, const void* g, int bytes) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" ::"r"(smem_u32(s)), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void
cp_async16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void
cp_async8(void* s, const void* g, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_u32(s)), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void
mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned
h2u(half2 h) {
    return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ half2
u2h(unsigned u) {
    return *reinterpret_cast<half2*>(&u);
}
__device__ __forceinline__ float
fast_exp2(float x) {  // ex2.approx: 2 ulp, exp2(-inf) = 0
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ unsigned
ldu16(const unsigned char* p) {
    return *reinterpret_cast<const unsigned short*>(p);
}

// Two codes in bits [0:16) -> half2 (value * scale). q8_0: 0x6400 | (c ^ 0x80) = 1152 + c.
// q4_0: nibbles (one per byte, already masked): 0x6400 | n = 1024 + n, value n - 8.
template <int QB>
__device__ __forceinline__ unsigned
deq2(unsigned u, unsigned sc) {
    if constexpr (QB == 34) {
        const unsigned t = __byte_perm(u ^ 0x8080u, 0x64646464u, 0x4140);
        return h2u(__hmul2(__hsub2(u2h(t), __float2half2_rn(1152.f)), u2h(sc)));
    } else {
        const unsigned t = __byte_perm(u, 0x64646464u, 0x4140);
        return h2u(__hmul2(__hsub2(u2h(t), __float2half2_rn(1032.f)), u2h(sc)));
    }
}

// Copies KV rows [pos0, pos0 + nv) of one head into a dense [T][RB] smem tile; rows >= nv are zero-filled.
template <int KVM, int RB, int T, int NT>
__device__ __forceinline__ void
copy_tile(unsigned char* d, const unsigned char* base, long long nb1, int pos0, int nv, int tid) {
    if constexpr (KVM == KV_CONTIG16) {
        const unsigned char* s = base + (size_t)pos0 * RB;
        if (nv == T) {
            for (int c = tid; c < T * RB / 16; c += NT) cp_async16(d + c * 16, s + c * 16);
        } else {
            const int valid = nv * RB;
            for (int c = tid; c < T * RB / 16; c += NT) {
                const int b  = valid - c * 16;
                const int sz = b <= 0 ? 0 : (b > 16 ? 16 : b);
                cp_async16(d + c * 16, s + (sz ? c * 16 : 0), sz);
            }
        }
    } else if constexpr (KVM == KV_ROWS16) {
        constexpr int CPR = RB / 16;
        for (int c = tid; c < T * CPR; c += NT) {
            const int row = c / CPR, off = (c % CPR) * 16;
            const bool ok = row < nv;
            cp_async16(d + row * RB + off, base + (size_t)(ok ? pos0 + row : 0) * nb1 + off, ok ? 16 : 0);
        }
    } else {
        constexpr int CPR = RB / 8;
        for (int c = tid; c < T * CPR; c += NT) {
            const int row = c / CPR, off = (c % CPR) * 8;
            const bool ok = row < nv;
            cp_async8(d + row * RB + off, base + (size_t)(ok ? pos0 + row : 0) * nb1 + off, ok ? 8 : 0);
        }
    }
}

template <int D, int QB, int RT, int T, int ST, int NW>
struct Geo {
    static constexpr int NT   = NW * 32;
    static constexpr int NRB  = RT / 16;   // 16-row blocks in the CTA
    static constexpr int NWC  = NW / NRB;  // QK column warps per row block
    static constexpr int Tc   = T / NWC;   // KV columns per warp in QK
    static constexpr int NBW  = Tc / 8;
    static constexpr int DPW  = D / NW;  // PV output columns per warp
    static constexpr int G32  = DPW < 32 ? DPW : 32;
    static constexpr int NG   = DPW / G32;
    static constexpr int NJ   = G32 / 8;
    static constexpr int NJ2  = NJ / 2;
    static constexpr int NDB  = NG * NJ;
    static constexpr int RB   = D / 32 * QB;
    static constexpr int KVB  = T * RB;
    static constexpr int STB  = 2 * KVB;
    static constexpr int MP   = T + 8;  // mask row pitch (halves)
    static constexpr int PP   = T + 8;  // P row pitch (halves)
    static constexpr int SLOT = RT * NWC;
    static constexpr int OPD  = D + 8;  // output staging row pitch (halves)
    static constexpr size_t smem =
        (size_t)ST * STB + (size_t)ST * NQMAX * MP * 2 + (size_t)RT * PP * 2 + (size_t)(2 * SLOT + 2 * RT) * 4;
    static_assert(NW % NRB == 0 && T % NWC == 0 && Tc % 8 == 0 && NWC > 1, "warp map");
    static_assert(DPW % 16 == 0 && (G32 == 32 || 32 % G32 == 0), "PV slice");
    static_assert(KVB % 16 == 0, "16B granular tiles");
    static_assert((size_t)RT * OPD * 2 <= (size_t)ST * STB, "output staging fits in the KV ring");
};

template <int D, int QB, int RT, int T, int ST, int NW, int KVM>
__device__ __forceinline__ void
cta_body(const KParams& p) {
    using Gm         = Geo<D, QB, RT, T, ST, NW>;
    constexpr int NT = Gm::NT, NRB = Gm::NRB, NWC = Gm::NWC, Tc = Gm::Tc, NBW = Gm::NBW, DPW = Gm::DPW;
    constexpr int G32 = Gm::G32, NG = Gm::NG, NJ = Gm::NJ, NJ2 = Gm::NJ2, NDB = Gm::NDB, RB = Gm::RB, KVB = Gm::KVB;
    constexpr int STB = Gm::STB, MP = Gm::MP, PP = Gm::PP, SLOT = Gm::SLOT, OPD = Gm::OPD;
    constexpr int NK = D / 32, KS = T / 16;

    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* kvbuf = smem;
    half* mbuf           = reinterpret_cast<half*>(smem + ST * STB);
    half* Ps             = mbuf + ST * NQMAX * MP;
    float* stmx          = reinterpret_cast<float*>(Ps + RT * PP);
    float* stsm          = stmx + SLOT;
    float* alph          = stsm + SLOT;
    float* rcpr          = alph + RT;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, sub = lane & 3;
    const int rw = warp / NWC, cw = warp % NWC;
    const int rg = blockIdx.x, hkv = blockIdx.y, split = blockIdx.z;
    const int r0 = rg * RT;  // first query row of this CTA

    const int ntot  = (p.kv + T - 1) / T;
    const int tb    = (int)((long long)split * ntot / p.nsplit);
    const int te    = (int)((long long)(split + 1) * ntot / p.nsplit);
    const int ntile = te - tb;

    const unsigned char* kh = p.k + (size_t)hkv * p.k_nb2;
    const unsigned char* vh = p.v + (size_t)hkv * p.v_nb2;
    const bool has_mask     = p.mask != nullptr;

    // K(+mask) and V are separate commit groups: QK waits only for K, V streams under QK.
    auto issueK = [&](int ti) {
        if (ti < ntile) {
            const int pos0 = (tb + ti) * T;
            const int nv   = min(T, p.kv - pos0);
            copy_tile<KVM, RB, T, NT>(kvbuf + (ti % ST) * STB, kh, p.k_nb1, pos0, nv, tid);
            if (has_mask) {
                constexpr int MC = T / 8;  // 16 B chunks per mask row
                half* md         = mbuf + (ti % ST) * NQMAX * MP;
                for (int c = tid; c < p.nq * MC; c += NT) {
                    const int t = c / MC, w = c % MC, pos = pos0 + w * 8;
                    const unsigned char* src = p.mask + (size_t)t * p.m_nb1;
                    if (nv == T) {
                        cp_async16(md + t * MP + w * 8, src + (size_t)pos * 2);
                    } else {
                        const int b  = (p.kv - pos) * 2;
                        const int sz = b <= 0 ? 0 : (b > 16 ? 16 : b);
                        cp_async16(md + t * MP + w * 8, src + (sz ? (size_t)pos * 2 : 0), sz);
                    }
                }
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };
    auto issueV = [&](int ti) {
        if (ti < ntile) {
            const int pos0 = (tb + ti) * T;
            copy_tile<KVM, RB, T, NT>(kvbuf + (ti % ST) * STB + KVB, vh, p.v_nb1, pos0, min(T, p.kv - pos0), tid);
        }
        asm volatile("cp.async.commit_group;\n");
    };
#pragma unroll
    for (int i = 0; i < ST - 1; ++i) {
        issueK(i);
        issueV(i);
    }

    // Q -> A fragments in registers (f16), scale * log2(e) folded in.
    unsigned qa[D / 16][4];
    {
        const float* qrow[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = r0 + rw * 16 + grp + h * 8;
            qrow[h]     = r < p.R ? reinterpret_cast<const float*>(p.q + (size_t)(r / p.G) * p.q_nbt +
                                                               (size_t)(hkv * p.G + r % p.G) * p.q_nbh)
                                  : nullptr;
        }
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            const int d0 = kk * 16 + sub * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float* ptr = qrow[e & 1];
                const int d      = d0 + (e >> 1) * 8;
                qa[kk][e]        = ptr ? h2u(__floats2half2_rn(ptr[d] * p.qscale, ptr[d + 1] * p.qscale)) : 0u;
            }
        }
    }

    float o[NDB][NRB][4];
#pragma unroll
    for (int i = 0; i < NDB; ++i)
#pragma unroll
        for (int rb = 0; rb < NRB; ++rb) o[i][rb][0] = o[i][rb][1] = o[i][rb][2] = o[i][rb][3] = 0.f;
    float mr0 = -INFINITY, mr1 = -INFINITY, lr0 = 0.f, lr1 = 0.f;  // lr: already summed over the lane quad
    const int tm0 = (r0 + rw * 16 + grp < p.R) ? (r0 + rw * 16 + grp) / p.G : 0;
    const int tm1 = (r0 + rw * 16 + grp + 8 < p.R) ? (r0 + rw * 16 + grp + 8) / p.G : 0;

    for (int it = 0; it < ntile; ++it) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2 * ST - 3));  // K(it) landed
        __syncthreads();
        issueK(it + ST - 1);
        issueV(it + ST - 1);

        const unsigned char* kst = kvbuf + (it % ST) * STB;
        const unsigned char* vst = kst + KVB;
        const half* mst          = mbuf + (it % ST) * NQMAX * MP;
        const int pos0           = (tb + it) * T;

        // ---- S = Q K^T for (row block rw) x (columns cw*Tc .. +Tc)
        float c[NBW][4];
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb) {
            c[nb][0] = c[nb][1] = c[nb][2] = c[nb][3] = 0.f;
            const unsigned char* kp                   = kst + (size_t)(cw * Tc + nb * 8 + grp) * RB;
#pragma unroll
            for (int kb = 0; kb < NK; ++kb) {
                const unsigned dsc = ldu16(kp + kb * QB) * 0x10001u;
#pragma unroll
                for (int sb = 0; sb < 2; ++sb) {
                    unsigned b0, b1;
                    if constexpr (QB == 34) {
                        const unsigned char* b = kp + kb * 34 + 2 + sb * 16 + sub * 2;
                        b0                     = deq2<QB>(ldu16(b), dsc);
                        b1                     = deq2<QB>(ldu16(b + 8), dsc);
                    } else {
                        const unsigned char* b = kp + kb * 18 + 2 + sub * 2;
                        b0                     = deq2<QB>((ldu16(b) >> (sb * 4)) & 0x0F0Fu, dsc);
                        b1                     = deq2<QB>((ldu16(b + 8) >> (sb * 4)) & 0x0F0Fu, dsc);
                    }
                    mma16816(c[nb], qa[kb * 2 + sb], b0, b1);
                }
            }
            const int col = cw * Tc + nb * 8 + sub * 2;
            if (has_mask) {
                const half2 m0 = *reinterpret_cast<const half2*>(mst + tm0 * MP + col);
                const half2 m1 = *reinterpret_cast<const half2*>(mst + tm1 * MP + col);
                c[nb][0]       = fmaf(__low2float(m0), kLog2e, c[nb][0]);
                c[nb][1]       = fmaf(__high2float(m0), kLog2e, c[nb][1]);
                c[nb][2]       = fmaf(__low2float(m1), kLog2e, c[nb][2]);
                c[nb][3]       = fmaf(__high2float(m1), kLog2e, c[nb][3]);
            }
            if (pos0 + T > p.kv) {  // tail tile
                const int pos = pos0 + col;
                if (pos >= p.kv) c[nb][0] = c[nb][2] = -INFINITY;
                if (pos + 1 >= p.kv) c[nb][1] = c[nb][3] = -INFINITY;
            }
        }

        // ---- row max across the NWC column warps, P = exp2(S - m) to smem as f16
        {
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int nb = 0; nb < NBW; ++nb) {
                mx0 = fmaxf(mx0, fmaxf(c[nb][0], c[nb][1]));
                mx1 = fmaxf(mx1, fmaxf(c[nb][2], c[nb][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
            if (sub == 0) {
                stmx[(rw * 16 + grp) * NWC + cw]     = mx0;
                stmx[(rw * 16 + grp + 8) * NWC + cw] = mx1;
            }
            __syncthreads();
            mx0 = mx1 = -INFINITY;
#pragma unroll
            for (int w = 0; w < NWC; ++w) {
                mx0 = fmaxf(mx0, stmx[(rw * 16 + grp) * NWC + w]);
                mx1 = fmaxf(mx1, stmx[(rw * 16 + grp + 8) * NWC + w]);
            }
            const float mn0 = fmaxf(mr0, mx0), mn1 = fmaxf(mr1, mx1);
            const float al0 = mr0 == -INFINITY ? 0.f : fast_exp2(mr0 - mn0);
            const float al1 = mr1 == -INFINITY ? 0.f : fast_exp2(mr1 - mn1);
            float sm0 = 0.f, sm1 = 0.f;
#pragma unroll
            for (int nb = 0; nb < NBW; ++nb) {
                const float p0 = mn0 == -INFINITY ? 0.f : fast_exp2(c[nb][0] - mn0);
                const float p1 = mn0 == -INFINITY ? 0.f : fast_exp2(c[nb][1] - mn0);
                const float p2 = mn1 == -INFINITY ? 0.f : fast_exp2(c[nb][2] - mn1);
                const float p3 = mn1 == -INFINITY ? 0.f : fast_exp2(c[nb][3] - mn1);
                sm0 += p0 + p1;
                sm1 += p2 + p3;
                const int col                                                  = cw * Tc + nb * 8 + sub * 2;
                *reinterpret_cast<half2*>(Ps + (rw * 16 + grp) * PP + col)     = __floats2half2_rn(p0, p1);
                *reinterpret_cast<half2*>(Ps + (rw * 16 + grp + 8) * PP + col) = __floats2half2_rn(p2, p3);
            }
            sm0 += __shfl_xor_sync(0xffffffff, sm0, 1);
            sm0 += __shfl_xor_sync(0xffffffff, sm0, 2);
            sm1 += __shfl_xor_sync(0xffffffff, sm1, 1);
            sm1 += __shfl_xor_sync(0xffffffff, sm1, 2);
            lr0 = lr0 * al0 + sm0;
            lr1 = lr1 * al1 + sm1;
            mr0 = mn0;
            mr1 = mn1;
            if (cw == 0 && sub == 0) {
                alph[rw * 16 + grp]     = al0;
                alph[rw * 16 + grp + 8] = al1;
            }
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2 * ST - 2));  // V(it) landed
        __syncthreads();

        // ---- O += P V: warp owns output columns [warp*DPW, +DPW) for all row blocks
        {
            float alr[NRB][2];
#pragma unroll
            for (int rb = 0; rb < NRB; ++rb) {
                alr[rb][0] = alph[rb * 16 + grp];
                alr[rb][1] = alph[rb * 16 + grp + 8];
            }
#pragma unroll
            for (int i = 0; i < NDB; ++i)
#pragma unroll
                for (int rb = 0; rb < NRB; ++rb) {
                    o[i][rb][0] *= alr[rb][0];
                    o[i][rb][1] *= alr[rb][0];
                    o[i][rb][2] *= alr[rb][1];
                    o[i][rb][3] *= alr[rb][1];
                }
        }
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            unsigned a[NRB][4];
#pragma unroll
            for (int rb = 0; rb < NRB; ++rb) {
                const half* pb = Ps + rb * 16 * PP + kk * 16 + sub * 2;
                a[rb][0]       = *reinterpret_cast<const unsigned*>(pb + grp * PP);
                a[rb][1]       = *reinterpret_cast<const unsigned*>(pb + (grp + 8) * PP);
                a[rb][2]       = *reinterpret_cast<const unsigned*>(pb + grp * PP + 8);
                a[rb][3]       = *reinterpret_cast<const unsigned*>(pb + (grp + 8) * PP + 8);
            }
            const unsigned char* v0 = vst + (size_t)(kk * 16 + sub * 2) * RB;
            const unsigned char* v1 = v0 + RB;
            const unsigned char* v2 = v0 + 8 * RB;
            const unsigned char* v3 = v2 + RB;
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const int cbase = warp * DPW + g * G32;
                const int blk = cbase >> 5, cb = cbase & 31;
                const unsigned s01 = ldu16(v0 + blk * QB) | (ldu16(v1 + blk * QB) << 16);
                const unsigned s23 = ldu16(v2 + blk * QB) | (ldu16(v3 + blk * QB) << 16);
#pragma unroll
                for (int q = 0; q < NJ2; ++q) {
                    unsigned w0, w1, w2, w3;
                    if constexpr (QB == 34) {
                        const int off = blk * 34 + 2 + cb + q * 16 + 2 * grp;
                        w0 = ldu16(v0 + off), w1 = ldu16(v1 + off), w2 = ldu16(v2 + off), w3 = ldu16(v3 + off);
                    } else {
                        const int cc  = cb + q * 16;  // 0 or 16 within the block: selects the nibble
                        const int off = blk * 18 + 2 + (cc & 15) + 2 * grp;
                        const int sh  = (cc & 16) ? 4 : 0;
                        w0            = (ldu16(v0 + off) >> sh) & 0x0F0Fu;
                        w1            = (ldu16(v1 + off) >> sh) & 0x0F0Fu;
                        w2            = (ldu16(v2 + off) >> sh) & 0x0F0Fu;
                        w3            = (ldu16(v3 + off) >> sh) & 0x0F0Fu;
                    }
#pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        // mma column n <-> output column cb + q*16 + 2n + jj
                        const int sel     = jj ? 0x5551 : 0x4440;
                        const unsigned bl = deq2<QB>(__byte_perm(w0, w1, sel), s01);
                        const unsigned bh = deq2<QB>(__byte_perm(w2, w3, sel), s23);
#pragma unroll
                        for (int rb = 0; rb < NRB; ++rb) mma16816(o[g * NJ + q * 2 + jj][rb], a[rb], bl, bh);
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    // ---- row sums across column warps, partial (m, l)
    __syncthreads();
    if (sub == 0) {
        stsm[(rw * 16 + grp) * NWC + cw]     = lr0;
        stsm[(rw * 16 + grp + 8) * NWC + cw] = lr1;
    }
    __syncthreads();
    {
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int w = 0; w < NWC; ++w) {
            a0 += stsm[(rw * 16 + grp) * NWC + w];
            a1 += stsm[(rw * 16 + grp + 8) * NWC + w];
        }
        lr0 = a0;
        lr1 = a1;
    }
    const int ns = p.nsplit;
    if (cw == 0 && sub == 0) {
        rcpr[rw * 16 + grp]     = lr0 > 0.f ? 1.f / lr0 : 0.f;
        rcpr[rw * 16 + grp + 8] = lr1 > 0.f ? 1.f / lr1 : 0.f;
        const int ra = r0 + rw * 16 + grp, rb = ra + 8;
        if (ra < p.R) {
            float* ml = p.ml_part + ((size_t)(hkv * p.R + ra) * ns + split) * 2;
            ml[0]     = mr0;
            ml[1]     = lr0;
        }
        if (rb < p.R) {
            float* ml = p.ml_part + ((size_t)(hkv * p.R + rb) * ns + split) * 2;
            ml[0]     = mr1;
            ml[1]     = lr1;
        }
    }
    __syncthreads();

    // ---- O partial: stage f16 rows in the (idle) KV ring, then 16 B stores
    half* osh = reinterpret_cast<half*>(smem);
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb) {
        const float rc0 = rcpr[rb * 16 + grp], rc1 = rcpr[rb * 16 + grp + 8];
#pragma unroll
        for (int i = 0; i < NDB; ++i) {
            const int g = i / NJ, r = i % NJ, q = r >> 1, jj = r & 1;
            const int colA = warp * DPW + g * G32 + q * 16 + 4 * sub + jj, colB = colA + 2;
            const int ga = rb * 16 + grp, gb = ga + 8;
            osh[ga * OPD + colA] = __float2half(o[i][rb][0] * rc0);
            osh[ga * OPD + colB] = __float2half(o[i][rb][1] * rc0);
            osh[gb * OPD + colA] = __float2half(o[i][rb][2] * rc1);
            osh[gb * OPD + colB] = __float2half(o[i][rb][3] * rc1);
        }
    }
    __syncthreads();
    constexpr int CH = D / 8;
    for (int idx = tid; idx < RT * CH; idx += NT) {
        const int row = idx / CH, cc = idx % CH;
        if (r0 + row < p.R)
            *reinterpret_cast<uint4*>(p.o_part + ((size_t)(hkv * p.R + r0 + row) * ns + split) * D + cc * 8) =
                *reinterpret_cast<const uint4*>(osh + row * OPD + cc * 8);
    }
}

// Grid (rows, D / 64): each CTA merges 64 output columns of one row; 8 halves per thread, 32 split groups.
template <int D>
__device__ __forceinline__ void
combine_body(const KParams& p) {
    constexpr int CB = 64, NCOL = CB / 8, NSG = 256 / NCOL;
    __shared__ float red[NSG][CB + 4];
    __shared__ float redl[NSG], wmax[8];
    const int gr = blockIdx.x, c0 = blockIdx.y * CB, ns = p.nsplit, tid = threadIdx.x;
    // Launched with programmatic stream serialization: CTAs may start while the main kernel drains; wait for its
    // writes here (a no-op for a plain launch).
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    const int sg = tid / NCOL, ct = tid % NCOL;
    const float* ml = p.ml_part + (size_t)gr * ns * 2;
    float M         = -INFINITY;
    for (int s = tid; s < ns; s += 256) M = fmaxf(M, ml[s * 2]);
#pragma unroll
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
    if ((tid & 31) == 0) wmax[tid >> 5] = M;
    __syncthreads();
    M = wmax[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) M = fmaxf(M, wmax[i]);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, L = 0.f;
    const half* ob = p.o_part + (size_t)gr * ns * D + c0 + ct * 8;
#pragma unroll 4
    for (int s = sg; s < ns; s += NSG) {
        const float2 m = *reinterpret_cast<const float2*>(ml + s * 2);
        const float w  = m.x == -INFINITY ? 0.f : fast_exp2(m.x - M) * m.y;
        L += w;
        const uint4 raw = *reinterpret_cast<const uint4*>(ob + (size_t)s * D);
        const half2* hv = reinterpret_cast<const half2*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __half22float2(hv[e]);
            acc[2 * e]     = fmaf(f.x, w, acc[2 * e]);
            acc[2 * e + 1] = fmaf(f.y, w, acc[2 * e + 1]);
        }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) red[sg][ct * 8 + e] = acc[e];
    if (ct == 0) redl[sg] = L;
    __syncthreads();
    if (tid < CB) {
        float a = 0.f, Lt = 0.f;
#pragma unroll
        for (int i = 0; i < NSG; ++i) {
            a += red[i][tid];
            Lt += redl[i];
        }
        const int hkv = gr / p.R, r = gr % p.R, t = r / p.G, qh = hkv * p.G + r % p.G;
        *reinterpret_cast<float*>(p.out + (size_t)t * p.o_nbt + (size_t)qh * p.o_nbh + (size_t)(c0 + tid) * 4) = a / Lt;
    }
}

}  // namespace

// Non-template entry points so the kernels show up with readable names in nsys / ncu / cuobjdump.
// Suffix: query rows per CTA rt, KV tile t, cp.async stages st, warps nw, then the K/V copy path: contig (head-major
// rows, whole tile as one run), rows16 / rows8 (strided rows, 16 / 8 B copies).
#define AOT_DECODE_KERNEL(name, D, QB, RT, T, ST, NW, MINB, KVM)                  \
    extern "C" __global__ void __launch_bounds__(NW * 32, MINB) name(KParams p) { \
        cta_body<D, QB, RT, T, ST, NW, KVM>(p);                                   \
    }
#define AOT_DECODE_KERNELS_D128(base, ...)                     \
    AOT_DECODE_KERNEL(base##_contig, __VA_ARGS__, KV_CONTIG16) \
    AOT_DECODE_KERNEL(base##_rows8, __VA_ARGS__, KV_ROWS8)
#define AOT_DECODE_KERNELS_D256(base, ...)                     \
    AOT_DECODE_KERNEL(base##_contig, __VA_ARGS__, KV_CONTIG16) \
    AOT_DECODE_KERNEL(base##_rows16, __VA_ARGS__, KV_ROWS16)   \
    AOT_DECODE_KERNEL(base##_rows8, __VA_ARGS__, KV_ROWS8)

AOT_DECODE_KERNELS_D128(cudnn_fe_aot_sm120_decode_cta_q8_0_d128_rt16_t128_st2_nw8, 128, 34, 16, 128, 2, 8, 2)
AOT_DECODE_KERNELS_D128(cudnn_fe_aot_sm120_decode_cta_q8_0_d128_rt32_t128_st2_nw8, 128, 34, 32, 128, 2, 8, 1)
AOT_DECODE_KERNELS_D128(cudnn_fe_aot_sm120_decode_cta_q8_0_d128_rt64_t128_st2_nw8, 128, 34, 64, 128, 2, 8, 1)
AOT_DECODE_KERNELS_D256(cudnn_fe_aot_sm120_decode_cta_q8_0_d256_rt16_t64_st2_nw8, 256, 34, 16, 64, 2, 8, 1)
AOT_DECODE_KERNELS_D256(cudnn_fe_aot_sm120_decode_cta_q8_0_d256_rt32_t64_st2_nw8, 256, 34, 32, 64, 2, 8, 1)
AOT_DECODE_KERNELS_D256(cudnn_fe_aot_sm120_decode_cta_q8_0_d256_rt64_t64_st2_nw8, 256, 34, 64, 64, 2, 8, 1)
AOT_DECODE_KERNELS_D128(cudnn_fe_aot_sm120_decode_cta_q4_0_d128_rt16_t64_st2_nw8, 128, 18, 16, 64, 2, 8, 2)
AOT_DECODE_KERNELS_D128(cudnn_fe_aot_sm120_decode_cta_q4_0_d128_rt32_t128_st2_nw8, 128, 18, 32, 128, 2, 8, 1)
AOT_DECODE_KERNELS_D128(cudnn_fe_aot_sm120_decode_cta_q4_0_d128_rt64_t128_st2_nw8, 128, 18, 64, 128, 2, 8, 1)
AOT_DECODE_KERNELS_D256(cudnn_fe_aot_sm120_decode_cta_q4_0_d256_rt16_t64_st2_nw8, 256, 18, 16, 64, 2, 8, 1)
AOT_DECODE_KERNELS_D256(cudnn_fe_aot_sm120_decode_cta_q4_0_d256_rt32_t64_st2_nw8, 256, 18, 32, 64, 2, 8, 1)
AOT_DECODE_KERNELS_D256(cudnn_fe_aot_sm120_decode_cta_q4_0_d256_rt64_t64_st2_nw8, 256, 18, 64, 64, 2, 8, 1)

extern "C" __global__ void
__launch_bounds__(256) cudnn_fe_aot_sm120_decode_combine_f16_d128(KParams p) {
    combine_body<128>(p);
}
extern "C" __global__ void
__launch_bounds__(256) cudnn_fe_aot_sm120_decode_combine_f16_d256(KParams p) {
    combine_body<256>(p);
}

namespace {

struct Config {
    void (*kern[3])(KParams);  // indexed by KvMode; rows16 is null where the row size is not 16 B granular
    const char* name;
    int D, QB, RT, T, ST, NW;
    size_t smem;
};

#define CFG(name, D, QB, RT, T, ST, NW, R16) \
    {{name##_contig, R16, name##_rows8}, #name, D, QB, RT, T, ST, NW, Geo<D, QB, RT, T, ST, NW>::smem}
#define CFG128(name, D, QB, RT, T, ST, NW) CFG(name, D, QB, RT, T, ST, NW, nullptr)
#define CFG256(name, D, QB, RT, T, ST, NW) CFG(name, D, QB, RT, T, ST, NW, name##_rows16)
// Order: q8 d128 rt16/32/64, q8 d256 rt16/32/64, q4 d128 ..., q4 d256 ... (see choose_config)
const Config kConfigs[] = {
    CFG128(cudnn_fe_aot_sm120_decode_cta_q8_0_d128_rt16_t128_st2_nw8, 128, 34, 16, 128, 2, 8),
    CFG128(cudnn_fe_aot_sm120_decode_cta_q8_0_d128_rt32_t128_st2_nw8, 128, 34, 32, 128, 2, 8),
    CFG128(cudnn_fe_aot_sm120_decode_cta_q8_0_d128_rt64_t128_st2_nw8, 128, 34, 64, 128, 2, 8),
    CFG256(cudnn_fe_aot_sm120_decode_cta_q8_0_d256_rt16_t64_st2_nw8, 256, 34, 16, 64, 2, 8),
    CFG256(cudnn_fe_aot_sm120_decode_cta_q8_0_d256_rt32_t64_st2_nw8, 256, 34, 32, 64, 2, 8),
    CFG256(cudnn_fe_aot_sm120_decode_cta_q8_0_d256_rt64_t64_st2_nw8, 256, 34, 64, 64, 2, 8),
    CFG128(cudnn_fe_aot_sm120_decode_cta_q4_0_d128_rt16_t64_st2_nw8, 128, 18, 16, 64, 2, 8),
    CFG128(cudnn_fe_aot_sm120_decode_cta_q4_0_d128_rt32_t128_st2_nw8, 128, 18, 32, 128, 2, 8),
    CFG128(cudnn_fe_aot_sm120_decode_cta_q4_0_d128_rt64_t128_st2_nw8, 128, 18, 64, 128, 2, 8),
    CFG256(cudnn_fe_aot_sm120_decode_cta_q4_0_d256_rt16_t64_st2_nw8, 256, 18, 16, 64, 2, 8),
    CFG256(cudnn_fe_aot_sm120_decode_cta_q4_0_d256_rt32_t64_st2_nw8, 256, 18, 32, 64, 2, 8),
    CFG256(cudnn_fe_aot_sm120_decode_cta_q4_0_d256_rt64_t64_st2_nw8, 256, 18, 64, 64, 2, 8),
};
const char* const kKvModeName[3] = {"contig", "rows16", "rows8"};
constexpr int kNumConfigs        = sizeof(kConfigs) / sizeof(kConfigs[0]);
constexpr int kMaxDev            = 64;

// R = n_q * G query rows per KV head; R > 64 runs ceil(R / 64) row-group CTAs per (KV head, split).
int
choose_config(int D, int kv_type, int R) {
    const int base = (kv_type == CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0 ? 0 : 6) + (D == 128 ? 0 : 3);
    return base + (R <= 16 ? 0 : R <= 32 ? 1 : 2);
}

struct Plan {
    int cfg, R, nrg, nsplit, kv_mode;
    bool mask_direct;  // mask rows are 16 B aligned: the kernel stages them straight from the caller's tensor
    size_t o_bytes, ml_bytes, mask_pitch, ws;
};

// Per-device facts cached per ordinal (write-once values; run() is on the per-token path).
std::atomic<int> g_dev_sms[kMaxDev], g_dev_cc[kMaxDev];
std::atomic<int> g_occ[kNumConfigs][3][kMaxDev];  // resident CTAs per SM + 1; 0 = not queried yet
std::mutex g_attr_mu;

bool
device_info(int* dev, int* sms, int* cc) {
    if (cudaGetDevice(dev) != cudaSuccess) return false;
    if (*dev < kMaxDev && g_dev_sms[*dev].load(std::memory_order_relaxed)) {
        *sms = g_dev_sms[*dev].load(std::memory_order_relaxed);
        *cc  = g_dev_cc[*dev].load(std::memory_order_relaxed);
        return true;
    }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(sms, cudaDevAttrMultiProcessorCount, *dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, *dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, *dev) != cudaSuccess)
        return false;
    *cc = major * 10 + minor;
    if (*dev < kMaxDev) {
        g_dev_cc[*dev].store(*cc, std::memory_order_relaxed);
        g_dev_sms[*dev].store(*sms, std::memory_order_relaxed);
    }
    return true;
}

// Sets the dynamic smem attribute (once per kernel and device); returns resident CTAs per SM, 0 on error.
int
kernel_occupancy(int cfg, int mode, int dev) {
    if (dev < kMaxDev) {
        const int v = g_occ[cfg][mode][dev].load(std::memory_order_acquire);
        if (v) return v - 1;
    }
    std::lock_guard<std::mutex> lk(g_attr_mu);
    const Config& c = kConfigs[cfg];
    int occ         = 0;
    if (cudaFuncSetAttribute(c.kern[mode], cudaFuncAttributeMaxDynamicSharedMemorySize, (int)c.smem) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, c.kern[mode], c.NW * 32, c.smem) != cudaSuccess || occ < 1)
        return 0;
    if (dev < kMaxDev) g_occ[cfg][mode][dev].store(occ + 1, std::memory_order_release);
    return occ;
}

bool
aligned(const void* p, int a) {
    return ((uintptr_t)p % a) == 0;
}

cudnn_fe_aot_sm120_decode_status_t
validate(const cudnn_fe_aot_sm120_decode_params_t* p) {
    if (!p || !p->q || !p->k || !p->v || !p->o) return CUDNN_FE_AOT_SM120_DECODE_ERROR_NULL_POINTER;
    if (p->head_dim != 128 && p->head_dim != 256) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_HEAD_DIM;
    if (p->kv_type != CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0 && p->kv_type != CUDNN_FE_AOT_SM120_DECODE_KV_Q4_0)
        return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_KV_TYPE;
    if (p->n_q < 1 || p->n_q > NQMAX) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_N_Q;
    if (p->n_kv < 1 || p->n_head < 1 || p->n_head_kv < 1 || p->n_seq < 1 || p->n_head % p->n_head_kv)
        return CUDNN_FE_AOT_SM120_DECODE_ERROR_BAD_SHAPE;
    if (p->max_bias != 0.f || p->logit_softcap != 0.f) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_FEATURE;
    const int64_t* nbs[] = {p->k_nb, p->v_nb};
    for (const int64_t* nb : nbs)
        for (int i = 1; i < 4; ++i)
            if (nb[i] % 8) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ALIGNMENT;
    if (!aligned(p->k, 8) || !aligned(p->v, 8)) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ALIGNMENT;
    for (int i = 1; i < 4; ++i)
        if (p->q_nb[i] % 4 || p->o_nb[i] % 4) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ALIGNMENT;
    if (!aligned(p->q, 4) || !aligned(p->o, 4)) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ALIGNMENT;
    if (p->mask) {
        if (!aligned(p->mask, 2) || p->mask_nb[1] % 2 || p->mask_nb[3] % 2)
            return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ALIGNMENT;
    }
    return CUDNN_FE_AOT_SM120_DECODE_SUCCESS;
}

cudnn_fe_aot_sm120_decode_status_t
make_plan(const cudnn_fe_aot_sm120_decode_params_t* p, Plan* out) {
    const cudnn_fe_aot_sm120_decode_status_t st = validate(p);
    if (st != CUDNN_FE_AOT_SM120_DECODE_SUCCESS) return st;
    int dev = 0, sms = 0, cc = 0;
    if (!device_info(&dev, &sms, &cc)) return CUDNN_FE_AOT_SM120_DECODE_ERROR_CUDA;
    if (cc != 120) return CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ARCH;
    const int G     = p->n_head / p->n_head_kv;
    const int R     = p->n_q * G;
    const int cfg   = choose_config(p->head_dim, p->kv_type, R);
    const Config& c = kConfigs[cfg];
    // K/V copy path: whole tile as one contiguous run (head-major rows), else row by row with 16 B or 8 B cp.async.
    const int RB   = c.D / 32 * c.QB;
    const auto a16 = [](const void* ptr, const int64_t* nb) {
        return aligned(ptr, 16) && nb[2] % 16 == 0 && nb[3] % 16 == 0;
    };
    int mode;
    if (p->k_nb[1] == RB && p->v_nb[1] == RB && a16(p->k, p->k_nb) && a16(p->v, p->v_nb))
        mode = KV_CONTIG16;
    else if (RB % 16 == 0 && p->k_nb[1] % 16 == 0 && p->v_nb[1] % 16 == 0 && a16(p->k, p->k_nb) && a16(p->v, p->v_nb))
        mode = KV_ROWS16;
    else
        mode = KV_ROWS8;
    const int occ = kernel_occupancy(cfg, mode, dev);
    if (!occ) return CUDNN_FE_AOT_SM120_DECODE_ERROR_CUDA;
    // At most one wave of CTAs (a partial second wave costs a whole CTA duration), split tile-balanced so any count
    // works; every split gets >= 1 tile.
    const int nrg       = (R + c.RT - 1) / c.RT;
    const int per_split = p->n_head_kv * nrg;
    const int ntot      = (p->n_kv + c.T - 1) / c.T;
    int nsplit          = std::max(1, sms * occ / per_split);
    nsplit              = std::min(nsplit, ntot);
    const size_t rows   = (size_t)p->n_head_kv * R;
    out->cfg            = cfg;
    out->kv_mode        = mode;
    out->R              = R;
    out->nrg            = nrg;
    out->nsplit         = nsplit;
    out->o_bytes        = ((size_t)nsplit * rows * p->head_dim * 2 + 255) / 256 * 256;
    out->ml_bytes       = ((size_t)nsplit * rows * 2 * 4 + 255) / 256 * 256;
    out->mask_pitch     = ((size_t)p->n_kv + 7) / 8 * 16;  // bytes per repacked mask row
    out->mask_direct    = p->mask && aligned(p->mask, 16) && p->mask_nb[1] % 16 == 0 && p->mask_nb[3] % 16 == 0;
    out->ws = out->o_bytes + out->ml_bytes + (p->mask && !out->mask_direct ? (size_t)p->n_q * out->mask_pitch : 0);
    return CUDNN_FE_AOT_SM120_DECODE_SUCCESS;
}

}  // namespace

extern "C" {

cudnn_fe_aot_sm120_decode_status_t
cudnn_fe_aot_sm120_decode_check(const cudnn_fe_aot_sm120_decode_params_t* p) {
    Plan pl;
    return make_plan(p, &pl);
}

size_t
cudnn_fe_aot_sm120_decode_get_workspace_size(const cudnn_fe_aot_sm120_decode_params_t* p) {
    Plan pl;
    return make_plan(p, &pl) == CUDNN_FE_AOT_SM120_DECODE_SUCCESS ? pl.ws : 0;
}

cudnn_fe_aot_sm120_decode_status_t
cudnn_fe_aot_sm120_decode_describe(const cudnn_fe_aot_sm120_decode_params_t* p, char* buf, size_t buf_size) {
    Plan pl;
    const cudnn_fe_aot_sm120_decode_status_t st = make_plan(p, &pl);
    if (st != CUDNN_FE_AOT_SM120_DECODE_SUCCESS) return st;
    if (buf && buf_size)
        snprintf(
            buf, buf_size, "%s_%s nrg=%d nsplit=%d", kConfigs[pl.cfg].name, kKvModeName[pl.kv_mode], pl.nrg, pl.nsplit);
    return st;
}

cudnn_fe_aot_sm120_decode_status_t
cudnn_fe_aot_sm120_decode_run(const cudnn_fe_aot_sm120_decode_params_t* p,
                              void* workspace,
                              size_t workspace_size,
                              cudaStream_t stream) {
    Plan pl;
    const cudnn_fe_aot_sm120_decode_status_t st = make_plan(p, &pl);
    if (st != CUDNN_FE_AOT_SM120_DECODE_SUCCESS) return st;
    if (!workspace) return CUDNN_FE_AOT_SM120_DECODE_ERROR_NULL_POINTER;
    if (workspace_size < pl.ws) return CUDNN_FE_AOT_SM120_DECODE_ERROR_WORKSPACE_TOO_SMALL;
    const Config& c = kConfigs[pl.cfg];

    KParams k{};
    k.q_nbt   = p->q_nb[1];
    k.q_nbh   = p->q_nb[2];
    k.k_nb1   = p->k_nb[1];
    k.k_nb2   = p->k_nb[2];
    k.v_nb1   = p->v_nb[1];
    k.v_nb2   = p->v_nb[2];
    k.o_nbh   = p->o_nb[1];
    k.o_nbt   = p->o_nb[2];
    k.nq      = p->n_q;
    k.Hq      = p->n_head;
    k.Hkv     = p->n_head_kv;
    k.G       = p->n_head / p->n_head_kv;
    k.R       = pl.R;
    k.kv      = p->n_kv;
    k.nsplit  = pl.nsplit;
    k.qscale  = p->scale * kLog2e;
    k.o_part  = static_cast<half*>(workspace);
    k.ml_part = reinterpret_cast<float*>(static_cast<unsigned char*>(workspace) + pl.o_bytes);
    // Mask rows that are not 16 B aligned are repacked into the workspace; no mask = no mask reads at all.
    unsigned char* mask_ws = static_cast<unsigned char*>(workspace) + pl.o_bytes + pl.ml_bytes;

    const dim3 grid(pl.nrg, p->n_head_kv, pl.nsplit);
    const unsigned rows = (unsigned)(p->n_head_kv * pl.R);
    for (int s = 0; s < p->n_seq; ++s) {
        k.q = reinterpret_cast<const unsigned char*>(p->q) + (size_t)s * p->q_nb[3];
        k.k = static_cast<const unsigned char*>(p->k) + (size_t)s * p->k_nb[3];
        k.v = static_cast<const unsigned char*>(p->v) + (size_t)s * p->v_nb[3];
        if (!p->mask) {
            k.mask  = nullptr;
            k.m_nb1 = 0;
        } else if (pl.mask_direct) {
            k.mask  = static_cast<const unsigned char*>(p->mask) + (size_t)s * p->mask_nb[3];
            k.m_nb1 = p->mask_nb[1];
        } else {
            if (cudaMemcpy2DAsync(mask_ws,
                                  pl.mask_pitch,
                                  static_cast<const unsigned char*>(p->mask) + (size_t)s * p->mask_nb[3],
                                  p->mask_nb[1],
                                  (size_t)p->n_kv * 2,
                                  p->n_q,
                                  cudaMemcpyDeviceToDevice,
                                  stream) != cudaSuccess)
                return CUDNN_FE_AOT_SM120_DECODE_ERROR_CUDA;
            k.mask  = mask_ws;
            k.m_nb1 = (long long)pl.mask_pitch;
        }
        k.out = reinterpret_cast<unsigned char*>(p->o) + (size_t)s * p->o_nb[3];
        c.kern[pl.kv_mode]<<<grid, c.NW * 32, c.smem, stream>>>(k);
        if (cudaPeekAtLastError() != cudaSuccess) return CUDNN_FE_AOT_SM120_DECODE_ERROR_CUDA;
        void (*comb)(KParams) = p->head_dim == 128 ? cudnn_fe_aot_sm120_decode_combine_f16_d128
                                                   : cudnn_fe_aot_sm120_decode_combine_f16_d256;
        // Programmatic dependent launch hides the combine's launch latency behind the main kernel's tail.
        cudaLaunchConfig_t lc = {};
        lc.gridDim            = dim3(rows, p->head_dim / 64);
        lc.blockDim           = dim3(256);
        lc.stream             = stream;
        cudaLaunchAttribute at[1];
        at[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        lc.attrs                                         = at;
        lc.numAttrs                                      = 1;
        if (cudaLaunchKernelEx(&lc, comb, k) != cudaSuccess) {  // e.g. a driver without PDL: plain launch
            (void)cudaGetLastError();
            comb<<<dim3(rows, p->head_dim / 64), 256, 0, stream>>>(k);
        }
    }
    return cudaGetLastError() == cudaSuccess ? CUDNN_FE_AOT_SM120_DECODE_SUCCESS : CUDNN_FE_AOT_SM120_DECODE_ERROR_CUDA;
}

const char*
cudnn_fe_aot_sm120_decode_status_string(cudnn_fe_aot_sm120_decode_status_t s) {
    switch (s) {
        case CUDNN_FE_AOT_SM120_DECODE_SUCCESS:
            return "success";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_NULL_POINTER:
            return "null pointer";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_HEAD_DIM:
            return "unsupported head_dim (need 128 or 256)";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_KV_TYPE:
            return "unsupported KV type (need q8_0 or q4_0)";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_N_Q:
            return "unsupported n_q (need 1..8)";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_BAD_SHAPE:
            return "bad shape";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ALIGNMENT:
            return "unsupported pointer/stride alignment";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_FEATURE:
            return "unsupported feature (ALiBi / softcap)";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_WORKSPACE_TOO_SMALL:
            return "workspace too small";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_UNSUPPORTED_ARCH:
            return "unsupported device (need compute capability 12.0)";
        case CUDNN_FE_AOT_SM120_DECODE_ERROR_CUDA:
            return "CUDA error";
    }
    return "unknown";
}

}  // extern "C"
