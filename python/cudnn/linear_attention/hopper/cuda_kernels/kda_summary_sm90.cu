// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>

// Strides are in elements, right-aligned to four axes. No staging, atomics,
// shared memory, tensor cores, or inter-CTA dependencies are used here.
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
    const int lane           = threadIdx.x % 32;
    const int groups         = transition ? 64 : 32;
    const int row            = (blockIdx.x % groups) * 4 + threadIdx.x / 32;
    const int head           = (blockIdx.x / groups) % heads;
    const int seq            = blockIdx.x / (groups * heads);
    const bool is_transition = row >= 128;
    const int output_row     = is_transition ? row - 128 : row;
    const long long begin =
        cu64 ? static_cast<const long long *>(cu)[seq * cu3] : static_cast<const int *>(cu)[seq * cu3];
    const long long end =
        cu64 ? static_cast<const long long *>(cu)[(seq + 1) * cu3] : static_cast<const int *>(cu)[(seq + 1) * cu3];
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int d = lane + 32 * j;
        s[j] =
            is_transition ? float(output_row == d) : (seed ? seed[seq * seed0 + head * seed1 + row * seed2 + d] : 0.0f);
    }
    for (long long pos = 0; pos < end - begin; ++pos) {
        const long long t = Backward ? end - 1 - pos : begin + pos;
        const float b     = beta[t * beta2 + head * beta3];
        const float value = is_transition ? 0.0f : __bfloat162float(v[t * v1 + head * v2 + row]);
        float keys[4], decay[4];
        bool near_one[4];
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int d                 = lane + 32 * j;
            keys[j]               = __bfloat162float(k[t * k1 + head * k2 + d]);
            const float log_decay = g[t * g1 + head * g2 + d];
            near_one[j]           = fabsf(log_decay) < 0.5f;
            // Forming exp(g) in FP32 loses the small decrement when g is
            // near zero. FMA with expm1(g) rounds the STATE update once instead
            // of accumulating that factor's rounding error over a long span.
            decay[j] = near_one[j] ? expm1f(log_decay) : expf(log_decay);
            if (Backward) {
                if (!is_transition) s[j] = fmaf(value, scale * __bfloat162float(q[t * q1 + head * q2 + d]), s[j]);
            } else {
                s[j] = near_one[j] ? fmaf(s[j], decay[j], s[j]) : s[j] * decay[j];
            }
            dot = fmaf(s[j], keys[j], dot);
        }
        dot                = warp_sum(dot);
        const float update = b * (Backward ? -dot : value - dot);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[j] = fmaf(update, keys[j], s[j]);
            if (Backward) s[j] = near_one[j] ? fmaf(s[j], decay[j], s[j]) : s[j] * decay[j];
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int d = lane + 32 * j;
        if (is_transition) {
            transition[seq * transition0 + head * transition1 + output_row * transition2 + d] = s[j];
        } else {
            state[seq * state0 + head * state1 + row * state2 + d] = s[j];
        }
    }
}

extern "C" __global__ void
kda_summary_fwd(PORTS) {
    summary<false>(ARGS);
}
extern "C" __global__ void
kda_summary_bwd(PORTS) {
    summary<true>(ARGS);
}
