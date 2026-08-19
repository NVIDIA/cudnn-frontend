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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

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

template <typename T>
T
from_float(float value) {
    return static_cast<T>(value);
}

template <>
half
from_float(float value) {
    return cpu_float2half_rn(value);
}

template <typename T>
double
to_double(T value) {
    return static_cast<double>(value);
}

template <>
double
to_double(half value) {
    return static_cast<double>(cpu_half2float(value));
}

template <typename T>
std::vector<T>
make_deterministic_data(size_t size, uint32_t seed, float scale) {
    std::vector<T> values(size);
    uint32_t state = seed;
    for (auto& value : values) {
        state            = state * 1664525u + 1013904223u;
        int32_t centered = static_cast<int32_t>((state >> 16) & 0xffffu) - 32768;
        float normalized = static_cast<float>(centered) / 32768.0f;
        value            = from_float<T>(normalized * scale);
    }
    return values;
}

template <typename T>
void
copy_to_device(Surface<T>& surface, std::vector<T> const& values) {
    REQUIRE(values.size() == surface.size);
    CUDA_CHECK(cudaMemcpy(surface.devPtr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
}

struct CausalConv1dReference {
    std::vector<double> y;
    std::vector<double> dx;
    std::vector<double> dweight;
};

template <typename T>
CausalConv1dReference
compute_reference(std::vector<T> const& x,
                  std::vector<T> const& weight,
                  std::vector<T> const& dy,
                  int batch,
                  int dim,
                  int seq_len,
                  int kernel_size) {
    CausalConv1dReference result{
        std::vector<double>(x.size(), 0.0),
        std::vector<double>(x.size(), 0.0),
        std::vector<double>(weight.size(), 0.0),
    };

    auto signal_index = [=](int b, int d, int s) { return (static_cast<size_t>(b) * dim + d) * seq_len + s; };
    auto weight_index = [=](int d, int k) { return static_cast<size_t>(d) * kernel_size + k; };

    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < dim; ++d) {
            for (int s = 0; s < seq_len; ++s) {
                for (int k = 0; k < std::min(kernel_size, s + 1); ++k) {
                    result.y[signal_index(b, d, s)] +=
                        to_double(weight[weight_index(d, k)]) * to_double(x[signal_index(b, d, s - k)]);
                }
                for (int k = 0; k < std::min(kernel_size, seq_len - s); ++k) {
                    result.dx[signal_index(b, d, s)] +=
                        to_double(dy[signal_index(b, d, s + k)]) * to_double(weight[weight_index(d, k)]);
                }
            }
        }
    }

    for (int d = 0; d < dim; ++d) {
        for (int k = 0; k < kernel_size; ++k) {
            for (int b = 0; b < batch; ++b) {
                for (int s = k; s < seq_len; ++s) {
                    result.dweight[weight_index(d, k)] +=
                        to_double(dy[signal_index(b, d, s)]) * to_double(x[signal_index(b, d, s - k)]);
                }
            }
        }
    }

    return result;
}

template <typename T>
void
check_close(Surface<T> const& actual,
            std::vector<double> const& expected,
            double atol,
            double rtol,
            char const* label) {
    REQUIRE(expected.size() == actual.size);
    std::vector<T> host_actual(actual.size);
    CUDA_CHECK(cudaMemcpy(host_actual.data(), actual.devPtr, actual.size * sizeof(T), cudaMemcpyDeviceToHost));

    size_t mismatches   = 0;
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double actual_value = to_double(host_actual[i]);
        double difference   = std::abs(actual_value - expected[i]);
        max_abs_diff        = std::max(max_abs_diff, difference);
        if (!std::isfinite(actual_value) || difference > atol + rtol * std::abs(expected[i])) {
            ++mismatches;
        }
    }

    CAPTURE(label, max_abs_diff, mismatches, atol, rtol);
    CHECK(mismatches == 0);
}

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

    auto x      = make_deterministic_data<half>(batch * dim * seq_len, 0x12345678u, 0.1f);
    auto weight = make_deterministic_data<half>(dim * kernel_size, 0x87654321u, 1.0f / std::sqrt(kernel_size));
    auto dy     = make_deterministic_data<half>(batch * dim * seq_len, 0x13579bdfu, 0.1f);

    Surface<half> x_tensor(x.size());
    Surface<half> weight_tensor(weight.size());
    Surface<half> y_tensor(x.size(), from_float<half>(-7.0f));
    Surface<half> dy_tensor(dy.size());
    Surface<half> dx_tensor(x.size(), from_float<half>(-11.0f));
    Surface<half> dweight_tensor(weight.size(), from_float<half>(13.0f));

    copy_to_device(x_tensor, x);
    copy_to_device(weight_tensor, weight);
    copy_to_device(dy_tensor, dy);

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

    auto reference = compute_reference(x, weight, dy, batch, dim, seq_len, kernel_size);
    check_close(y_tensor, reference.y, 5e-3, 5e-3, "y");
    check_close(dx_tensor, reference.dx, 5e-3, 5e-3, "dx");
    check_close(dweight_tensor, reference.dweight, 5e-3, 5e-3, "dweight");
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

    auto x      = make_deterministic_data<float>(batch * dim * seq_len, 0x2468ace0u, 0.1f);
    auto weight = make_deterministic_data<float>(dim * kernel_size, 0x0eca8642u, 1.0f / std::sqrt(kernel_size));
    auto dy     = make_deterministic_data<float>(batch * dim * seq_len, 0x10293847u, 0.1f);

    Surface<float> x_tensor(x.size());
    Surface<float> weight_tensor(weight.size());
    Surface<float> y_tensor(x.size(), -7.0f);
    Surface<float> dy_tensor(dy.size());
    Surface<float> dx_tensor(x.size(), -11.0f);
    Surface<float> dweight_tensor(weight.size(), 13.0f);
    Surface<uint8_t> workspace_tensor(workspace_size);
    Surface<uint8_t> reserve_tensor(reserve_size);

    copy_to_device(x_tensor, x);
    copy_to_device(weight_tensor, weight);
    copy_to_device(dy_tensor, dy);

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

    auto reference = compute_reference(x, weight, dy, batch, dim, seq_len, kernel_size);
    check_close(y_tensor, reference.y, 2e-5, 2e-5, "y");
    check_close(dx_tensor, reference.dx, 2e-5, 2e-5, "dx");
    check_close(dweight_tensor, reference.dweight, 2e-5, 2e-5, "dweight");
#endif
}
