/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "utils/error_util.h"
#include "utils/fp16_dev.h"

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
