/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./utils/error_util.h"
#include "./utils/fp16_dev.h"

#define BLOCK_SIZE 128
template <class value_type>
__global__ void
float2half_rn_kernel(int size, const value_type *buffIn, half1 *buffOut) {
    const int idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
#if CUDART_VERSION < 9000
    half1 val;
    val.x = __float2half_rn(float(buffIn[idx]));
#else
    half1 val = __float2half_rn(float(buffIn[idx]));
#endif
    buffOut[idx] = val;
}

template <class value_type>
void
gpu_float2half_rn(int size, const value_type *buffIn, half1 *buffOut) {
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float2half_rn_kernel<value_type><<<grid_size, BLOCK_SIZE>>>(size, buffIn, buffOut);
    checkCudaErrors(cudaDeviceSynchronize());
}

template void
gpu_float2half_rn<float>(int, const float *, half1 *);
template void
gpu_float2half_rn<double>(int, const double *, half1 *);
