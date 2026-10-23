// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/layernorm_backward.md
// API: layernorm_backward
// Use case: Basic
// Source: adapted from `samples/cpp/norm/layernorm.cpp`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace fe = cudnn_frontend;

void
check(fe::error_t status) {
    if (!status.is_good()) throw std::runtime_error(status.get_message());
}

template <typename T>
T*
managed(int64_t count) {
    T* ptr = nullptr;
    cudaMallocManaged(&ptr, count * sizeof(T));
    return ptr;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;

void
build_and_execute(fe::graph::Graph& graph, VariantPack& variant_pack) {
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);
    check(graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}));

    void* workspace     = nullptr;
    auto workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);
    check(graph.execute(handle, variant_pack, workspace));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("execution failed");

    cudaFree(workspace);
    cudnnDestroy(handle);
}

int
main() {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    int64_t const batch_size  = 8;
    int64_t const hidden_size = 64;
    int64_t const x_count     = batch_size * hidden_size;
    int64_t const stat_count  = batch_size;
    nv_bfloat16* X_dev        = managed<nv_bfloat16>(x_count);
    nv_bfloat16* DY_dev       = managed<nv_bfloat16>(x_count);
    nv_bfloat16* DX_dev       = managed<nv_bfloat16>(x_count);
    nv_bfloat16* Scale_dev    = managed<nv_bfloat16>(hidden_size);
    float* Mean_dev           = managed<float>(stat_count);
    float* InvVar_dev         = managed<float>(stat_count);
    nv_bfloat16* DScale_dev   = managed<nv_bfloat16>(hidden_size);
    nv_bfloat16* DBias_dev    = managed<nv_bfloat16>(hidden_size);
    for (int64_t i = 0; i < x_count; ++i) X_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < x_count; ++i) DY_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t i = 0; i < hidden_size; ++i) Scale_dev[i] = __float2bfloat16(dis(gen));
    for (int64_t row = 0; row < stat_count; ++row) {
        float mean = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j)
            mean += __bfloat162float(X_dev[row * hidden_size + j]);
        mean /= static_cast<float>(hidden_size);
        float variance = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            auto delta = __bfloat162float(X_dev[row * hidden_size + j]) - mean;
            variance += delta * delta;
        }
        Mean_dev[row]   = mean;
        InvVar_dev[row] = 1.0f / std::sqrt(variance / static_cast<float>(hidden_size) + 1.0e-3f);
    }

    fe::graph::Graph graph{};
    graph.set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X      = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("X")
                              .set_dim({batch_size, hidden_size, 1, 1})
                              .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto DY     = graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("DY")
                               .set_dim({batch_size, hidden_size, 1, 1})
                               .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto Scale  = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("scale")
                                  .set_dim({1, hidden_size, 1, 1})
                                  .set_stride({hidden_size, 1, hidden_size, hidden_size}));
    auto Mean   = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({batch_size, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto InvVar = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("inv_variance")
                                   .set_dim({batch_size, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto attrs  = fe::graph::Layernorm_backward_attributes()
                     .set_name("layernorm_backward")
                     .set_saved_mean_and_inv_variance(Mean, InvVar)
                     .set_compute_data_type(fe::DataType_t::FLOAT);
    auto [DX, DScale, DBias] = graph.layernorm_backward(DY, X, Scale, attrs);
    DX->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);
    DScale->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);
    DBias->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);
    VariantPack variant_pack = {{X, X_dev},
                                {DY, DY_dev},
                                {Scale, Scale_dev},
                                {Mean, Mean_dev},
                                {InvVar, InvVar_dev},
                                {DX, DX_dev},
                                {DScale, DScale_dev},
                                {DBias, DBias_dev}};
    build_and_execute(graph, variant_pack);
    for (int64_t j = 0; j < hidden_size; ++j) {
        float expected_scale = 0.0f;
        float expected_bias  = 0.0f;
        for (int64_t row = 0; row < stat_count; ++row) {
            auto dy = __bfloat162float(DY_dev[row * hidden_size + j]);
            auto xhat =
                (__bfloat162float(X_dev[row * hidden_size + j]) - Mean_dev[row]) * InvVar_dev[row];
            expected_scale += dy * xhat;
            expected_bias += dy;
        }
        auto actual_scale = __bfloat162float(DScale_dev[j]);
        auto actual_bias  = __bfloat162float(DBias_dev[j]);
        if (!std::isfinite(actual_scale) || !std::isfinite(actual_bias) ||
            !std::isfinite(expected_scale) || !std::isfinite(expected_bias) ||
            std::fabs(actual_scale - expected_scale) > 3.0e-2f ||
            std::fabs(actual_bias - expected_bias) > 3.0e-2f) {
            throw std::runtime_error("layernorm parameter-gradient mismatch");
        }
    }
    for (int64_t row = 0; row < stat_count; ++row) {
        float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
        for (int64_t j = 0; j < hidden_size; ++j) {
            auto dy =
                __bfloat162float(DY_dev[row * hidden_size + j]) * __bfloat162float(Scale_dev[j]);
            auto xhat =
                (__bfloat162float(X_dev[row * hidden_size + j]) - Mean_dev[row]) * InvVar_dev[row];
            sum_dy += dy;
            sum_dy_xhat += dy * xhat;
        }
        for (int64_t j = 0; j < hidden_size; ++j) {
            auto dy =
                __bfloat162float(DY_dev[row * hidden_size + j]) * __bfloat162float(Scale_dev[j]);
            auto xhat =
                (__bfloat162float(X_dev[row * hidden_size + j]) - Mean_dev[row]) * InvVar_dev[row];
            auto expected = InvVar_dev[row] *
                            (dy - (sum_dy + xhat * sum_dy_xhat) / static_cast<float>(hidden_size));
            auto actual = __bfloat162float(DX_dev[row * hidden_size + j]);
            if (!std::isfinite(actual) || !std::isfinite(expected) ||
                std::fabs(actual - expected) > 5.0e-2f) {
                throw std::runtime_error("layernorm input-gradient mismatch");
            }
        }
    }
    std::cout << "layernorm gradients match reference\n";
    cudaFree(X_dev);
    cudaFree(DY_dev);
    cudaFree(DX_dev);
    cudaFree(Scale_dev);
    cudaFree(Mean_dev);
    cudaFree(InvVar_dev);
    cudaFree(DScale_dev);
    cudaFree(DBias_dev);
    return 0;
}
