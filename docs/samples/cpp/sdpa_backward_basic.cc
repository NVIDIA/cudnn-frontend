// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/sdpa_backward.md
// API: sdpa_backward
// Use case: Basic
// Source: adapted from `samples/cpp/sdpa/fp16_bwd.cpp`.

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
        int64_t const B = 1, H = 2, S = 128, D = 64;
        int64_t const qkv_count  = B * H * S * D;
        int64_t const stat_count = B * H * S;
        nv_bfloat16* Q_dev       = managed<nv_bfloat16>(qkv_count);
        nv_bfloat16* K_dev       = managed<nv_bfloat16>(qkv_count);
        nv_bfloat16* V_dev       = managed<nv_bfloat16>(qkv_count);
        nv_bfloat16* O_dev       = managed<nv_bfloat16>(qkv_count);
        nv_bfloat16* DO_dev      = managed<nv_bfloat16>(qkv_count);
        nv_bfloat16* DQ_dev      = managed<nv_bfloat16>(qkv_count);
        nv_bfloat16* DK_dev      = managed<nv_bfloat16>(qkv_count);
        nv_bfloat16* DV_dev      = managed<nv_bfloat16>(qkv_count);
        float* Stats_dev         = managed<float>(stat_count);
        for (int64_t i = 0; i < (qkv_count); ++i) Q_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (qkv_count); ++i) K_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (qkv_count); ++i) V_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (qkv_count); ++i) O_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (qkv_count); ++i) DO_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (qkv_count); ++i) DQ_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (qkv_count); ++i) DK_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (qkv_count); ++i) DV_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (stat_count); ++i) Stats_dev[i] = dis(gen);

        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::BFLOAT16)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto dims    = std::vector<int64_t>{B, H, S, D};
        auto strides = std::vector<int64_t>{H * S * D, S * D, D, 1};
        auto Q       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(dims)
                                  .set_stride(strides)
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto K       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(dims)
                                  .set_stride(strides)
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto V       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim(dims)
                                  .set_stride(strides)
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto O       = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("O")
                                  .set_dim(dims)
                                  .set_stride(strides)
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto DO      = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("dO")
                                   .set_dim(dims)
                                   .set_stride(strides)
                                   .set_data_type(fe::DataType_t::BFLOAT16));
        auto Stats   = graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("Stats")
                                      .set_dim({B, H, S, 1})
                                      .set_stride({H * S, S, 1, 1})
                                      .set_data_type(fe::DataType_t::FLOAT));
        auto attrs   = fe::graph::SDPA_backward_attributes()
                         .set_name("sdpa_backward")
                         .set_causal_mask(true)
                         .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)))
                         .set_compute_data_type(fe::DataType_t::FLOAT);
        auto [DQ, DK, DV] = graph.sdpa_backward(Q, K, V, O, DO, Stats, attrs);
        DQ->set_output(true).set_dim(dims).set_stride(strides).set_data_type(
            fe::DataType_t::BFLOAT16);
        DK->set_output(true).set_dim(dims).set_stride(strides).set_data_type(
            fe::DataType_t::BFLOAT16);
        DV->set_output(true).set_dim(dims).set_stride(strides).set_data_type(
            fe::DataType_t::BFLOAT16);
        VariantPack variant_pack = {{Q, Q_dev},
                                    {K, K_dev},
                                    {V, V_dev},
                                    {O, O_dev},
                                    {DO, DO_dev},
                                    {Stats, Stats_dev},
                                    {DQ, DQ_dev},
                                    {DK, DK_dev},
                                    {DV, DV_dev}};
        build_and_execute(graph, variant_pack);
        std::cout << "sdpa_backward executed successfully\n";
        cudaFree(Q_dev);
        cudaFree(K_dev);
        cudaFree(V_dev);
        cudaFree(O_dev);
        cudaFree(DO_dev);
        cudaFree(DQ_dev);
        cudaFree(DK_dev);
        cudaFree(DV_dev);
        cudaFree(Stats_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
