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

TEST_CASE("B2B causal conv1d forward", "[b2b_causal_conv1d][forward]") {
#if !HAS_SUBQUADRATIC_OPS
    SKIP("cudnn_subquadratic_ops.h not available");
#elif defined(_MSC_VER)
    SKIP("Causal conv1d kernels are not supported on Windows (MSVC)");
#elif CUDNN_VERSION < 92400
    SKIP("B2B causal conv1d kernels require cuDNN 9.24.0 or newer (compiled CUDNN_VERSION >= 92400).");
#else
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }

    int batch             = 2;
    int dim               = 64;
    int seq_len           = 512;
    int kernel_size_proj  = 4;
    int kernel_size_mixer = 7;

    cudaStream_t stream = nullptr;

    Surface<half> x_tensor(batch * dim * 3 * seq_len);
    Surface<half> w_proj(dim * 3 * kernel_size_proj);
    Surface<half> w_mixer(dim * kernel_size_mixer);
    Surface<half> skip_bias(dim);
    Surface<half> y_tensor(batch * dim * seq_len);
    Surface<half> y_gated(batch * dim * seq_len);

    CUDNN_CHECK(cudnnB2BCausalConv1dForward(stream,
                                            x_tensor.devPtr,
                                            w_proj.devPtr,
                                            w_mixer.devPtr,
                                            skip_bias.devPtr,
                                            y_tensor.devPtr,
                                            y_gated.devPtr,
                                            batch,
                                            dim,
                                            seq_len,
                                            kernel_size_proj,
                                            kernel_size_mixer,
                                            CUDNN_DATA_HALF));

    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

TEST_CASE("B2B causal conv1d backward", "[b2b_causal_conv1d][backward]") {
#if !HAS_SUBQUADRATIC_OPS
    SKIP("cudnn_subquadratic_ops.h not available");
#elif defined(_MSC_VER)
    SKIP("Causal conv1d kernels are not supported on Windows (MSVC)");
#elif CUDNN_VERSION < 92400
    SKIP("B2B causal conv1d kernels require cuDNN 9.24.0 or newer (compiled CUDNN_VERSION >= 92400).");
#else
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by current cudnn version");
    }

    int batch             = 2;
    int dim               = 64;
    int seq_len           = 512;
    int kernel_size_proj  = 4;
    int kernel_size_mixer = 7;

    cudaStream_t stream = nullptr;

    Surface<half> x_tensor(batch * dim * 3 * seq_len);
    Surface<half> w_proj(dim * 3 * kernel_size_proj);
    Surface<half> w_mixer(dim * kernel_size_mixer);
    Surface<half> skip_bias(dim);
    Surface<half> y_tensor(batch * dim * seq_len);
    Surface<half> dy_tensor(batch * dim * seq_len);
    Surface<half> dx_tensor(batch * dim * 3 * seq_len);
    Surface<float> dw_proj(dim * 3 * kernel_size_proj, 0.0f);
    Surface<float> dw_mixer(dim * kernel_size_mixer, 0.0f);
    Surface<float> dskip_bias(dim, 0.0f);

    CUDNN_CHECK(cudnnB2BCausalConv1dBackward(stream,
                                             x_tensor.devPtr,
                                             w_proj.devPtr,
                                             w_mixer.devPtr,
                                             skip_bias.devPtr,
                                             y_tensor.devPtr,
                                             dy_tensor.devPtr,
                                             dx_tensor.devPtr,
                                             dw_proj.devPtr,
                                             dw_mixer.devPtr,
                                             dskip_bias.devPtr,
                                             batch,
                                             dim,
                                             seq_len,
                                             kernel_size_proj,
                                             kernel_size_mixer,
                                             CUDNN_DATA_HALF,
                                             CUDNN_DATA_FLOAT));

    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}
