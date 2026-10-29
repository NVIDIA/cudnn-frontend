// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Quantized-KV decode test inputs: ggml-layout q8_0 / q4_0 problems on the device and a double-precision reference.
// Only the parameter struct of decode_q8_q4.h is used; the library is called through the public API.
#pragma once

#include "decode_q8_q4.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

enum Layout {
    LAYOUT_GGML_TEST = 0,  // test-backend-ops: K/V views of a [D, 2*kv, Hkv] buffer (head-major), Q [D, nq, Hq]
    LAYOUT_KV_CACHE  = 1,  // llama.cpp KV cache: [D*Hkv, kv_size] rows (token-major), Q permuted [D, Hq, nq]
    LAYOUT_COMPACT   = 2,  // dense head-major K/V [Hkv][kv][row], Q [nq][Hq][D]
};
enum MaskKind {
    MASK_KIND_NONE       = 0,
    MASK_KIND_RANDOM     = 1,  // uniform(-1, 1), row pitch = n_kv (as test-backend-ops)
    MASK_KIND_PADDED     = 2,  // uniform(-1, 1), row pitch padded to 64 elements
    MASK_KIND_CAUSAL_INF = 3,  // -inf over a prefix and a token-dependent suffix, else uniform(-1, 1)
};

struct Case {
    int D, Hkv, G, kv, nq, nseq, kv_type, layout, mask;
};

inline const char*
type_name(int t) {
    return t == CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0 ? "q8_0" : "q4_0";
}
inline int
block_bytes(int t) {
    return t == CUDNN_FE_AOT_SM120_DECODE_KV_Q8_0 ? 34 : 18;
}

struct Problem {
    Case c;
    int Hq, RB;
    // host logical values (for the reference)
    std::vector<float> q;             // [s][t][h][D]
    std::vector<float> k_ref, v_ref;  // [s][hkv][pos][D] dequantized
    std::vector<float> maskf;         // [s][t][pos], empty if no mask
    // device
    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dm = nullptr, *dout = nullptr;
    cudnn_fe_aot_sm120_decode_params_t p{};

    size_t
    qi(int s, int t, int h, int j) const {
        return (((size_t)s * c.nq + t) * Hq + h) * c.D + j;
    }
    size_t
    ki(int s, int h, int pos, int j) const {
        return (((size_t)s * c.Hkv + h) * c.kv + pos) * c.D + j;
    }

    // Bytes the kernel must read: K + V + mask (the roofline denominator used in RESULTS.md).
    double
    bytes() const {
        return 2.0 * c.Hkv * c.kv * RB * c.nseq + (c.mask ? (double)c.nq * c.kv * 2 * c.nseq : 0.0);
    }

    void
    build(const Case& cs, uint32_t seed, bool host_ref) {
        c            = cs;
        Hq           = c.Hkv * c.G;
        RB           = c.D / 32 * block_bytes(c.kv_type);
        const int QB = block_bytes(c.kv_type);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> U(-1.f, 1.f);

        // Q
        q.resize((size_t)c.nseq * c.nq * Hq * c.D);
        for (auto& x : q) x = U(rng);
        int64_t q_nb[4];
        q_nb[0] = 4;
        if (c.layout == LAYOUT_GGML_TEST) {
            q_nb[1] = (int64_t)c.D * 4;
            q_nb[2] = q_nb[1] * c.nq;
        } else {
            q_nb[2] = (int64_t)c.D * 4;
            q_nb[1] = q_nb[2] * Hq;
        }
        q_nb[3] = (int64_t)c.D * 4 * c.nq * Hq;
        std::vector<float> qdev((size_t)c.nseq * c.nq * Hq * c.D);
        for (int s = 0; s < c.nseq; ++s)
            for (int t = 0; t < c.nq; ++t)
                for (int h = 0; h < Hq; ++h)
                    for (int j = 0; j < c.D; ++j)
                        qdev[(s * q_nb[3] + t * q_nb[1] + h * q_nb[2]) / 4 + j] = q[qi(s, t, h, j)];

        // K / V
        int64_t kv_nb[4];
        kv_nb[0] = QB;
        size_t kv_bytes;
        if (c.layout == LAYOUT_GGML_TEST || c.layout == LAYOUT_COMPACT) {
            kv_nb[1] = RB;
            kv_nb[2] = (int64_t)RB * (c.layout == LAYOUT_COMPACT ? 1 : 2) * c.kv;
            kv_nb[3] = kv_nb[2] * c.Hkv;
        } else {
            const int kv_size = c.kv + 3;  // cache capacity > n_kv, as in a real KV cache
            kv_nb[1]          = (int64_t)RB * c.Hkv;
            kv_nb[2]          = RB;
            kv_nb[3]          = kv_nb[1] * kv_size;
        }
        kv_bytes = (size_t)kv_nb[3] * c.nseq;
        std::vector<uint8_t> kraw(kv_bytes, 0), vraw(kv_bytes, 0);
        if (host_ref) {
            k_ref.assign((size_t)c.nseq * c.Hkv * c.kv * c.D, 0.f);
            v_ref.assign(k_ref.size(), 0.f);
        }
        uint32_t xs = seed * 2654435761u + 12345u;  // xorshift32: KV fill is ~10^8 draws for the big cases
        auto rnd    = [&]() {
            xs ^= xs << 13;
            xs ^= xs >> 17;
            xs ^= xs << 5;
            return xs;
        };
        auto fill = [&](std::vector<uint8_t>& raw, std::vector<float>& f) {
            size_t blk_id = 0;
            for (int s = 0; s < c.nseq; ++s)
                for (int h = 0; h < c.Hkv; ++h)
                    for (int pos = 0; pos < c.kv; ++pos)
                        for (int b = 0; b < c.D / 32; ++b, ++blk_id) {
                            uint8_t* blk = raw.data() + s * kv_nb[3] + h * kv_nb[2] + (size_t)pos * kv_nb[1] + b * QB;
                            const half d =
                                __float2half(QB == 34 ? 0.02f + 0.01f * (blk_id % 7) : 0.3f + 0.15f * (blk_id % 7));
                            memcpy(blk, &d, 2);
                            const float df = __half2float(d);
                            for (int j = 0; j < 32; ++j) {
                                float val;
                                if (QB == 34) {
                                    const int8_t qv = int8_t(int(rnd() % 255) - 127);
                                    blk[2 + j]      = uint8_t(qv);
                                    val             = df * qv;
                                } else {
                                    const int nib = int((rnd() >> 8) % 16);
                                    blk[2 + (j & 15)] |= uint8_t(nib << ((j & 16) ? 4 : 0));
                                    val = df * (nib - 8);
                                }
                                if (host_ref) f[ki(s, h, pos, b * 32 + j)] = val;
                            }
                        }
        };
        fill(kraw, k_ref);
        fill(vraw, v_ref);

        // mask
        int64_t m_nb[4] = {2, 0, 0, 0};
        std::vector<half> mraw;
        if (c.mask) {
            const int pitch = c.mask == MASK_KIND_PADDED ? (c.kv + 63) / 64 * 64 : c.kv;
            m_nb[1]         = (int64_t)pitch * 2;
            m_nb[2]         = m_nb[1] * c.nq;
            m_nb[3]         = m_nb[2];
            mraw.assign((size_t)pitch * c.nq * c.nseq, __float2half(0.f));
            if (host_ref) maskf.assign((size_t)c.nseq * c.nq * c.kv, 0.f);
            for (int s = 0; s < c.nseq; ++s)
                for (int t = 0; t < c.nq; ++t)
                    for (int pos = 0; pos < c.kv; ++pos) {
                        float m = U(rng);
                        if (c.mask == MASK_KIND_CAUSAL_INF && (pos < c.kv / 3 || pos >= c.kv - (c.nq - 1 - t)))
                            m = -INFINITY;
                        mraw[((size_t)s * c.nq + t) * pitch + pos] = __float2half(m);
                        if (host_ref) maskf[((size_t)s * c.nq + t) * c.kv + pos] = __half2float(__float2half(m));
                    }
        }

        // O: ggml dst [D, Hq, nq, nseq]
        int64_t o_nb[4] = {4, (int64_t)c.D * 4, (int64_t)c.D * 4 * Hq, (int64_t)c.D * 4 * Hq * c.nq};

        // Exact-size allocations so compute-sanitizer flags any out-of-bounds read.
        CK(cudaMalloc(&dq, qdev.size() * 4));
        CK(cudaMalloc(&dk, kv_bytes));
        CK(cudaMalloc(&dv, kv_bytes));
        CK(cudaMalloc(&dout, (size_t)c.nseq * c.nq * Hq * c.D * 4));
        CK(cudaMemcpy(dq, qdev.data(), qdev.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, kraw.data(), kv_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, vraw.data(), kv_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(dout, 0xff, (size_t)c.nseq * c.nq * Hq * c.D * 4));
        if (c.mask) {
            CK(cudaMalloc(&dm, mraw.size() * 2));
            CK(cudaMemcpy(dm, mraw.data(), mraw.size() * 2, cudaMemcpyHostToDevice));
        }

        p.head_dim  = c.D;
        p.n_q       = c.nq;
        p.n_head    = Hq;
        p.n_head_kv = c.Hkv;
        p.n_kv      = c.kv;
        p.n_seq     = c.nseq;
        p.q         = static_cast<const float*>(dq);
        memcpy(p.q_nb, q_nb, sizeof q_nb);
        p.kv_type = c.kv_type;
        p.k       = dk;
        p.v       = dv;
        memcpy(p.k_nb, kv_nb, sizeof kv_nb);
        memcpy(p.v_nb, kv_nb, sizeof kv_nb);
        p.mask = dm;
        memcpy(p.mask_nb, m_nb, sizeof m_nb);
        p.o = static_cast<float*>(dout);
        memcpy(p.o_nb, o_nb, sizeof o_nb);
        p.scale = 1.f / sqrtf((float)c.D);
    }

    void
    release() {
        cudaFree(dq);
        cudaFree(dk);
        cudaFree(dv);
        cudaFree(dm);
        cudaFree(dout);
        dq = dk = dv = dm = dout = nullptr;
    }

    // Max abs error vs a double reference over (seq, token, head) triples from `heads`; Q rounded to f16 like the
    // kernel.
    void
    check(const std::vector<float>& out, const std::vector<int>& heads, double* maxerr, double* maxref, int* nonfinite)
        const {
        *maxerr    = 0;
        *maxref    = 0;
        *nonfinite = 0;
        std::vector<double> sc(c.kv);
        for (int s = 0; s < c.nseq; ++s)
            for (int t = 0; t < c.nq; ++t)
                for (int qh : heads) {
                    const int h = qh / c.G;
                    double mx   = -INFINITY;
                    for (int pos = 0; pos < c.kv; ++pos) {
                        double acc = 0;
                        for (int j = 0; j < c.D; ++j)
                            acc += double(__half2float(__float2half(q[qi(s, t, qh, j)]))) * k_ref[ki(s, h, pos, j)];
                        sc[pos] = acc / sqrt((double)c.D) + (c.mask ? maskf[((size_t)s * c.nq + t) * c.kv + pos] : 0.0);
                        mx      = std::max(mx, sc[pos]);
                    }
                    double L = 0;
                    for (auto& x : sc) L += (x = std::isinf(x) ? 0.0 : exp(x - mx));
                    for (int j = 0; j < c.D; ++j) {
                        double o = 0;
                        for (int pos = 0; pos < c.kv; ++pos) o += sc[pos] * v_ref[ki(s, h, pos, j)];
                        o /= L;
                        const float got = out[(((size_t)s * c.nq + t) * Hq + qh) * c.D + j];
                        if (!std::isfinite(got)) ++*nonfinite;
                        *maxerr = std::max(*maxerr, std::abs(o - got));
                        *maxref = std::max(*maxref, std::abs(o));
                    }
                }
    }
};
