/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

TEST_CASE("Causal conv1d NWH forward", "[causal_conv1d_nwh][forward]") {
#if !HAS_SUBQUADRATIC_OPS
    SKIP("cudnn_subquadratic_ops.h not available");
#elif defined(_MSC_VER)
    SKIP("Causal conv1d kernels are not supported on Windows (MSVC)");
#elif CUDNN_VERSION < 92400
    SKIP("Causal conv1d NWH kernels require cuDNN 9.24.0 or newer (compiled CUDNN_VERSION >= 92400).");
#else
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }

    int batch       = 2;
    int dim         = 64;
    int seq_len     = 512;
    int kernel_size = 4;

    cudaStream_t stream = nullptr;

    Surface<half> x_tensor(batch * seq_len * dim);
    Surface<half> w_tensor(kernel_size * dim);
    Surface<half> bias_tensor(dim);
    Surface<half> y_tensor(batch * seq_len * dim);

    CUDNN_CHECK(cudnnCausalConv1dNwhForward(stream,
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

TEST_CASE("Causal conv1d NWH backward", "[causal_conv1d_nwh][backward]") {
#if !HAS_SUBQUADRATIC_OPS
    SKIP("cudnn_subquadratic_ops.h not available");
#elif defined(_MSC_VER)
    SKIP("Causal conv1d kernels are not supported on Windows (MSVC)");
#elif CUDNN_VERSION < 92400
    SKIP("Causal conv1d NWH kernels require cuDNN 9.24.0 or newer (compiled CUDNN_VERSION >= 92400).");
#else
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }

    int batch       = 2;
    int dim         = 64;
    int seq_len     = 512;
    int kernel_size = 4;

    cudaStream_t stream = nullptr;

    Surface<half> x_tensor(batch * seq_len * dim);
    Surface<half> w_tensor(kernel_size * dim);
    Surface<half> bias_tensor(dim);
    Surface<half> dy_tensor(batch * seq_len * dim);
    Surface<half> dx_tensor(batch * seq_len * dim);
    Surface<float> dw_tensor(kernel_size * dim, 0.0f);
    Surface<float> dbias_tensor(dim, 0.0f);

    CUDNN_CHECK(cudnnCausalConv1dNwhBackward(stream,
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
