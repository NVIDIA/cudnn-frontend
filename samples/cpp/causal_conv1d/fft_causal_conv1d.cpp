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

#if HAS_SUBQUADRATIC_OPS && CUDNN_VERSION >= 92600
namespace {

void
skip_if_fft_causal_conv1d_is_unavailable() {
    if (cudnnGetVersion() < 92600) {
        SKIP("FFT causal conv1d requires cuDNN 9.26.0 or newer");
    }
    if (!is_arch_supported_by_cudnn()) {
        SKIP("Architecture is not supported by the current cuDNN version");
    }
}

}  // namespace
#endif

TEST_CASE("FFT causal conv1d medium forward and backward", "[fft_causal_conv1d][medium]") {
#if !HAS_SUBQUADRATIC_OPS || CUDNN_VERSION < 92600
    SKIP("FFT causal conv1d APIs require cuDNN 9.26.0 headers");
#elif defined(_MSC_VER)
    SKIP("FFT causal conv1d kernels are not supported on Windows (MSVC)");
#else
    skip_if_fft_causal_conv1d_is_unavailable();

    constexpr int batch       = 2;
    constexpr int dim         = 4;
    constexpr int seq_len     = 256;
    constexpr int kernel_size = 128;

    cudaStream_t stream = nullptr;

    Surface<half> x_tensor(batch * dim * seq_len);
    Surface<half> weight_tensor(dim * kernel_size);
    Surface<half> y_tensor(batch * dim * seq_len);
    Surface<half> dy_tensor(batch * dim * seq_len);
    Surface<half> dx_tensor(batch * dim * seq_len);
    Surface<half> dweight_tensor(dim * kernel_size, 0.0f);

    CUDNN_CHECK(cudnnFFTCausalConv1dForward(stream,
                                            x_tensor.devPtr,
                                            weight_tensor.devPtr,
                                            y_tensor.devPtr,
                                            batch,
                                            dim,
                                            seq_len,
                                            kernel_size,
                                            CUDNN_DATA_HALF));

    CUDNN_CHECK(cudnnFFTCausalConv1dBackward(stream,
                                             x_tensor.devPtr,
                                             weight_tensor.devPtr,
                                             dy_tensor.devPtr,
                                             dx_tensor.devPtr,
                                             dweight_tensor.devPtr,
                                             batch,
                                             dim,
                                             seq_len,
                                             kernel_size,
                                             CUDNN_DATA_HALF));

    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

TEST_CASE("FFT causal conv1d long forward and backward", "[fft_causal_conv1d][long]") {
#if !HAS_SUBQUADRATIC_OPS || CUDNN_VERSION < 92600
    SKIP("FFT causal conv1d APIs require cuDNN 9.26.0 headers");
#elif defined(_MSC_VER)
    SKIP("FFT causal conv1d kernels are not supported on Windows (MSVC)");
#else
    skip_if_fft_causal_conv1d_is_unavailable();

    constexpr int batch       = 1;
    constexpr int dim         = 1;
    constexpr int seq_len     = 4096;
    constexpr int kernel_size = 4096;

    size_t workspace_size = 0;
    size_t reserve_size   = 0;
    CUDNN_CHECK(cudnnLongFFTCausalConv1dGetBufferSizes(
        batch, dim, seq_len, kernel_size, CUDNN_DATA_FLOAT, &workspace_size, &reserve_size));

    cudaStream_t stream = nullptr;

    Surface<float> x_tensor(batch * dim * seq_len);
    Surface<float> weight_tensor(dim * kernel_size);
    Surface<float> y_tensor(batch * dim * seq_len);
    Surface<float> dy_tensor(batch * dim * seq_len);
    Surface<float> dx_tensor(batch * dim * seq_len);
    Surface<float> dweight_tensor(dim * kernel_size, 0.0f);
    Surface<uint8_t> workspace_tensor(workspace_size);
    Surface<uint8_t> reserve_tensor(reserve_size);

    CUDNN_CHECK(cudnnLongFFTCausalConv1dForward(stream,
                                                x_tensor.devPtr,
                                                weight_tensor.devPtr,
                                                y_tensor.devPtr,
                                                batch,
                                                dim,
                                                seq_len,
                                                kernel_size,
                                                CUDNN_DATA_FLOAT,
                                                workspace_tensor.devPtr,
                                                workspace_size,
                                                reserve_tensor.devPtr,
                                                reserve_size));

    // Backward consumes the frequency-domain state written to reserve_tensor
    // by this matching forward call. workspace_tensor is temporary scratch.
    CUDNN_CHECK(cudnnLongFFTCausalConv1dBackward(stream,
                                                 dy_tensor.devPtr,
                                                 dx_tensor.devPtr,
                                                 dweight_tensor.devPtr,
                                                 batch,
                                                 dim,
                                                 seq_len,
                                                 kernel_size,
                                                 CUDNN_DATA_FLOAT,
                                                 workspace_tensor.devPtr,
                                                 workspace_size,
                                                 reserve_tensor.devPtr,
                                                 reserve_size));

    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}
