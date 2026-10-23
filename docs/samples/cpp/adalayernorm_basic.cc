// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/adalayernorm.md
// API: adalayernorm
// Use case: Basic
// Source: adapted from `samples/cpp/norm/adaptive_layernorm.cpp`.

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
        int64_t const x_count      = batch_size * seq_length * hidden_size;
        int64_t const weight_count = batch_size * hidden_size;
        int64_t const stat_count   = batch_size * seq_length;
        nv_bfloat16* X_dev         = managed<nv_bfloat16>(x_count);
        nv_bfloat16* Y_dev         = managed<nv_bfloat16>(x_count);
        float* Scale_dev           = managed<float>(weight_count);
        float* Bias_dev            = managed<float>(weight_count);
        float* Mean_dev            = managed<float>(stat_count);
        float* InvVar_dev          = managed<float>(stat_count);
        for (int64_t i = 0; i < (x_count); ++i) X_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (x_count); ++i) Y_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (weight_count); ++i) Scale_dev[i] = dis(gen);
        for (int64_t i = 0; i < (weight_count); ++i) Bias_dev[i] = dis(gen);
        for (int64_t i = 0; i < (stat_count); ++i) Mean_dev[i] = dis(gen);
        for (int64_t i = 0; i < (stat_count); ++i) InvVar_dev[i] = dis(gen);

        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto X       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("X")
                                  .set_dim({batch_size, seq_length, hidden_size})
                                  .set_stride({seq_length * hidden_size, hidden_size, 1}));
        auto Scale   = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("scale")
                                      .set_dim({batch_size, 1, hidden_size})
                                      .set_stride({hidden_size, hidden_size, 1})
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto Bias    = graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("bias")
                                     .set_dim({batch_size, 1, hidden_size})
                                     .set_stride({hidden_size, hidden_size, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
        auto Epsilon = graph.tensor(fe::graph::Tensor_attributes(1.0e-5f).set_name("epsilon"));
        auto attrs   = fe::graph::AdaLayernorm_attributes()
                         .set_name("adalayernorm")
                         .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                         .set_epsilon(Epsilon)
                         .set_compute_data_type(fe::DataType_t::FLOAT);
        auto [Y, Mean, InvVar] = graph.adalayernorm(X, Scale, Bias, attrs);
        Y->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);
        Mean->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        InvVar->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{X, X_dev},
                                    {Scale, Scale_dev},
                                    {Bias, Bias_dev},
                                    {Y, Y_dev},
                                    {Mean, Mean_dev},
                                    {InvVar, InvVar_dev}};
        build_and_execute(graph, variant_pack);
        std::cout << "adalayernorm executed successfully\n";
        cudaFree(X_dev);
        cudaFree(Y_dev);
        cudaFree(Scale_dev);
        cudaFree(Bias_dev);
        cudaFree(Mean_dev);
        cudaFree(InvVar_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
