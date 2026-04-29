/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn.h>

#if defined(CUDNN_SUBQUADRATIC_OPS_H_) || __has_include(<cudnn_subquadratic_ops.h>)
#if !defined(CUDNN_SUBQUADRATIC_OPS_H_)
#include <cudnn_subquadratic_ops.h>
#endif
#define HAS_SUBQUADRATIC_OPS 1
#else
#define HAS_SUBQUADRATIC_OPS 0
#endif

TEST_CASE("Causal conv1d forward", "[causal_conv1d][forward]") {
#if !HAS_SUBQUADRATIC_OPS
    SKIP("cudnn_subquadratic_ops.h not available");
#else
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }

    int batch       = 2;
    int dim         = 64;
    int seq_len     = 512;
    int kernel_size = 4;

    cudaStream_t stream = nullptr;

    Surface<half> x_tensor(batch * dim * seq_len, false);
    Surface<half> w_tensor(dim * kernel_size, false);
    Surface<half> bias_tensor(dim, false);
    Surface<half> y_tensor(batch * dim * seq_len, false);

    CUDNN_CHECK(cudnnCausalConv1dForward(stream,
                                         x_tensor.devPtr,
                                         w_tensor.devPtr,
                                         bias_tensor.devPtr,
                                         y_tensor.devPtr,
                                         batch,
                                         dim,
                                         seq_len,
                                         kernel_size,
                                         CUDNN_DATA_HALF,
                                         CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU));

    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

TEST_CASE("Causal conv1d backward", "[causal_conv1d][backward]") {
#if !HAS_SUBQUADRATIC_OPS
    SKIP("cudnn_subquadratic_ops.h not available");
#else
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }

    int batch       = 2;
    int dim         = 64;
    int seq_len     = 512;
    int kernel_size = 4;

    cudaStream_t stream = nullptr;

    Surface<half> x_tensor(batch * dim * seq_len, false);
    Surface<half> w_tensor(dim * kernel_size, false);
    Surface<half> bias_tensor(dim, false);
    Surface<half> dy_tensor(batch * dim * seq_len, false);
    Surface<half> dx_tensor(batch * dim * seq_len, false);
    Surface<float> dw_tensor(dim * kernel_size, 0.0f);
    Surface<float> dbias_tensor(dim, 0.0f);

    CUDNN_CHECK(cudnnCausalConv1dBackward(stream,
                                          x_tensor.devPtr,
                                          w_tensor.devPtr,
                                          bias_tensor.devPtr,
                                          dy_tensor.devPtr,
                                          dx_tensor.devPtr,
                                          dw_tensor.devPtr,
                                          dbias_tensor.devPtr,
                                          batch,
                                          dim,
                                          seq_len,
                                          kernel_size,
                                          CUDNN_DATA_HALF,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU));

    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}
