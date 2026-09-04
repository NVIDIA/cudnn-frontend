// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host-side helpers for the kda_cake engine, compiled with NVRTC next to the
// frozen CAKE bodies. Not generated; edit freely.

#include <cuda_bf16.h>

extern "C" {

// Regenerates the forward's beta_active tape from the raw beta logits so the
// backward can consume it without the forward having saved it. Reproduces the
// frozen forward's arithmetic exactly: sigmoid through tanh.approx.f32, then
// one bf16 rounding. Padding heads (>= num_heads) are zero-filled.
__global__ void
fe_cake_kda_beta_active(const __nv_bfloat16* __restrict__ beta,
                        __nv_bfloat16* __restrict__ beta_active,
                        long long total_tokens,
                        int num_heads,
                        int stride) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_tokens * (long long)stride) {
        return;
    }
    const long long token = index / stride;
    const int head        = (int)(index - token * stride);
    float value           = 0.0f;
    if (head < num_heads) {
        const float logit = __bfloat162float(beta[token * (long long)num_heads + head]);
        float t;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(logit * 0.5f));
        value = t * 0.5f + 0.5f;
    }
    beta_active[index] = __float2bfloat16(value);
}

}  // extern "C"
