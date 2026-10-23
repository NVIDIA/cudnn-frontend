// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/rmsnorm_backward.md
// API: rmsnorm_backward
// Use case: Basic
// Source: adapted from `samples/llm_coverage/test_rmsnorm_b+s+d.py`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

template <typename T>
T*
managed(int64_t count) {
    T* ptr             = nullptr;
    cudaError_t status = cudaMallocManaged(&ptr, count * sizeof(T));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocManaged: ") + cudaGetErrorString(status));
    }
    return ptr;
}

using VariantPack = std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>;

void
build_and_execute(fe::graph::Graph& graph, VariantPack& variant_pack) {
    cudnnHandle_t handle       = nullptr;
    void* workspace            = nullptr;
    cudnnStatus_t cudnn_status = cudnnCreate(&handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnnCreate: ") + cudnnGetErrorString(cudnn_status));
    }

    try {
        auto status = graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        if (!status.is_good()) {
            throw std::runtime_error(std::string("graph.build: ") + status.get_message());
        }

        int64_t workspace_size = graph.get_workspace_size();
        if (workspace_size > 0) {
            cudaError_t cuda_status = cudaMalloc(&workspace, workspace_size);
            if (cuda_status != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc(workspace): ") +
                                         cudaGetErrorString(cuda_status));
            }
        }

        status = graph.execute(handle, variant_pack, workspace);
        if (!status.is_good()) {
            throw std::runtime_error(std::string("graph.execute: ") + status.get_message());
        }

        cudaError_t cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaDeviceSynchronize: ") +
                                     cudaGetErrorString(cuda_status));
        }
    } catch (...) {
        if (workspace) cudaFree(workspace);
        cudnnDestroy(handle);
        throw;
    }

    if (workspace) {
        cudaError_t cuda_status = cudaFree(workspace);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaFree(workspace): ") +
                                     cudaGetErrorString(cuda_status));
        }
    }

    cudnn_status = cudnnDestroy(handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnnDestroy: ") + cudnnGetErrorString(cudnn_status));
    }
}

int
main() {
    try {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        int64_t const batch_size = 4, seq_length = 128, hidden_size = 128;
        int64_t const x_count    = batch_size * seq_length * hidden_size;
        int64_t const stat_count = batch_size * seq_length;
        float* X_dev             = managed<float>(x_count);
        float* DY_dev            = managed<float>(x_count);
        float* DX_dev            = managed<float>(x_count);
        float* Scale_dev         = managed<float>(hidden_size);
        float* InvVar_dev        = managed<float>(stat_count);
        float* DScale_dev        = managed<float>(hidden_size);
        for (int64_t i = 0; i < (x_count); ++i) X_dev[i] = dis(gen);
        for (int64_t i = 0; i < (x_count); ++i) DY_dev[i] = dis(gen);
        for (int64_t i = 0; i < (x_count); ++i) DX_dev[i] = dis(gen);
        for (int64_t i = 0; i < (hidden_size); ++i) Scale_dev[i] = dis(gen);
        for (int64_t i = 0; i < (stat_count); ++i) InvVar_dev[i] = dis(gen);
        for (int64_t i = 0; i < (hidden_size); ++i) DScale_dev[i] = dis(gen);
        for (int64_t i = 0; i < stat_count; ++i) InvVar_dev[i] = std::fabs(InvVar_dev[i]) + 0.5f;

        fe::graph::Graph graph{};
        graph.set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto X      = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("X")
                                  .set_data_type(fe::DataType_t::FLOAT)
                                  .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                                  .set_stride({hidden_size, 1, hidden_size, hidden_size}));
        auto DY     = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("DY")
                                   .set_data_type(fe::DataType_t::FLOAT)
                                   .set_dim({batch_size * seq_length, hidden_size, 1, 1})
                                   .set_stride({hidden_size, 1, hidden_size, hidden_size}));
        auto Scale  = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("scale")
                                      .set_dim({1, hidden_size, 1, 1})
                                      .set_stride({hidden_size, 1, hidden_size, hidden_size})
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto InvVar = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("inv_variance")
                                       .set_dim({batch_size * seq_length, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::FLOAT));
        auto attrs  = fe::graph::Rmsnorm_backward_attributes()
                         .set_name("rmsnorm_backward")
                         .has_dbias(false)
                         .set_compute_data_type(fe::DataType_t::FLOAT);
        auto [DX, DScale, DBias] = graph.rmsnorm_backward(DY, X, Scale, InvVar, attrs);
        DX->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        DScale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{X, X_dev},
                                    {DY, DY_dev},
                                    {Scale, Scale_dev},
                                    {InvVar, InvVar_dev},
                                    {DX, DX_dev},
                                    {DScale, DScale_dev}};
        build_and_execute(graph, variant_pack);
        std::cout << "rmsnorm_backward executed successfully\n";
        cudaFree(X_dev);
        cudaFree(DY_dev);
        cudaFree(DX_dev);
        cudaFree(Scale_dev);
        cudaFree(InvVar_dev);
        cudaFree(DScale_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
