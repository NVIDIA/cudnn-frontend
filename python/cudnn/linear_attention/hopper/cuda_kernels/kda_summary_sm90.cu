// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>

// Strides are in elements, right-aligned to four axes. Each warp owns a state
// row. A CTA stages 16 tokens' keys/decays once for its 16 rows; recurrence
// operands, operation order and FP32 rounding stay unchanged. No tensor cores,
// global workspace or inter-CTA dependencies are required.
#define PORT(type, name) type *name, long long name##0, long long name##1, long long name##2, long long name##3
#define ARG(name) name, name##0, name##1, name##2, name##3
#define PORTS                                                                                                       \
    PORT(const __nv_bfloat16, q), PORT(const __nv_bfloat16, k), PORT(const __nv_bfloat16, v), PORT(const float, g), \
        PORT(const float, beta), PORT(const void, cu), PORT(const float, seed), PORT(float, state),                 \
        PORT(float, transition), int heads, int cu64, float scale
#define ARGS \
    ARG(q), ARG(k), ARG(v), ARG(g), ARG(beta), ARG(cu), ARG(seed), ARG(state), ARG(transition), heads, cu64, scale

__device__ __forceinline__ float
warp_sum(float x) {
#pragma unroll
    for (int delta = 16; delta; delta >>= 1) x += __shfl_xor_sync(0xffffffff, x, delta);
    return x;
}

template <bool Backward>
__device__ void
summary(PORTS) {
    constexpr int Threads = 512;
    constexpr int Warps   = Threads / 32;
    const int groups      = (transition ? 256 : 128) / Warps;
    const int lane = threadIdx.x % 32, row = blockIdx.x % groups * Warps + threadIdx.x / 32;
    const int head = blockIdx.x / groups % heads, seq = blockIdx.x / (groups * heads);
    // A CTA covers 16 rows and never straddles the H/M boundary at row 128.
    // Thus the Q producer predicate below is uniform across the whole CTA.
    const bool is_transition = row >= 128;
    const int output_row     = row % 128;
    const long long begin =
        cu64 ? static_cast<const long long *>(cu)[seq * cu3] : static_cast<const int *>(cu)[seq * cu3];
    const long long end =
        cu64 ? static_cast<const long long *>(cu)[(seq + 1) * cu3] : static_cast<const int *>(cu)[(seq + 1) * cu3];
    __shared__ float sk[16][128], sd[16][128], sq[Backward ? 16 : 1][128], sb[16];
    __shared__ bool near[16][128];
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int d = lane + 32 * j;
        s[j] = is_transition ? float(output_row == d) : (seed ? seed[seq * seed0 + head * seed1 + row * seed2 + d] : 0);
    }
    for (long long pos = 0; pos < end - begin; pos += 16) {
        int count = int(min(16LL, end - begin - pos));
        for (int index = threadIdx.x; index < count * 128; index += Threads) {
            int i = index / 128, d = index % 128;
            long long t = Backward ? end - 1 - pos - i : begin + pos + i;
            float gd    = g[t * g1 + head * g2 + d];
            // Preserve weak-decay accuracy: update with expm1(g) and one FMA,
            // rather than repeatedly multiplying a rounded near-one exp(g).
            near[i][d] = fabsf(gd) < 0.5f;
            sd[i][d]   = near[i][d] ? expm1f(gd) : expf(gd);
            sk[i][d]   = __bfloat162float(k[t * k1 + head * k2 + d]);
            if (Backward && !is_transition) sq[i][d] = scale * __bfloat162float(q[t * q1 + head * q2 + d]);
        }
        if (threadIdx.x < count) {
            long long t     = Backward ? end - 1 - pos - threadIdx.x : begin + pos + threadIdx.x;
            sb[threadIdx.x] = beta[t * beta2 + head * beta3];
        }
        // CTA generic stores -> generic readers: publish the complete tile.
        __syncthreads();
        for (int i = 0; i < count; ++i) {
            long long t = Backward ? end - 1 - pos - i : begin + pos + i;
            float value = is_transition ? 0 : __bfloat162float(v[t * v1 + head * v2 + row]);
            float dot   = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int d = lane + 32 * j;
                if (Backward) {
                    if (!is_transition) s[j] = fmaf(value, sq[i][d], s[j]);
                } else
                    s[j] = near[i][d] ? fmaf(s[j], sd[i][d], s[j]) : s[j] * sd[i][d];
                dot = fmaf(s[j], sk[i][d], dot);
            }
            dot          = warp_sum(dot);
            float update = sb[i] * (Backward ? -dot : value - dot);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int d = lane + 32 * j;
                s[j]  = fmaf(update, sk[i][d], s[j]);
                if (Backward) s[j] = near[i][d] ? fmaf(s[j], sd[i][d], s[j]) : s[j] * sd[i][d];
            }
        }
        // All readers, including the partial tail, finish before tile reuse.
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int d = lane + 32 * j;
        if (is_transition)
            transition[seq * transition0 + head * transition1 + output_row * transition2 + d] = s[j];
        else
            state[seq * state0 + head * state1 + row * state2 + d] = s[j];
    }
}

extern "C" __global__ void
cudnn_kda_summary_fwd(PORTS) {
    summary<false>(ARGS);
}
extern "C" __global__ void
cudnn_kda_summary_bwd(PORTS) {
    summary<true>(ARGS);
}
