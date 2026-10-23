// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/sdpa_fp8.md
// API: sdpa_fp8
// Use case: Basic
// Source: adapted from `samples/cpp/sdpa/fp8_fwd.cpp`.

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
        int64_t const B = 1, H = 2, S = 128, D = 128;
        int64_t const qkv_storage = B * S * 3 * H * D;
        int64_t const o_count     = B * S * H * D;
        int8_t* QKV_dev           = managed<int8_t>(qkv_storage);
        int8_t* O_dev             = managed<int8_t>(o_count);
        float* DescaleQ_dev       = managed<float>(1);
        float* DescaleK_dev       = managed<float>(1);
        float* DescaleV_dev       = managed<float>(1);
        float* DescaleS_dev       = managed<float>(1);
        float* ScaleS_dev         = managed<float>(1);
        float* ScaleO_dev         = managed<float>(1);
        float* AmaxS_dev          = managed<float>(1);
        float* AmaxO_dev          = managed<float>(1);
        std::uniform_int_distribution<int> byte_dis(-64, 63);
        for (int64_t i = 0; i < qkv_storage; ++i) QKV_dev[i] = static_cast<int8_t>(byte_dis(gen));
        for (int64_t i = 0; i < o_count; ++i) O_dev[i] = static_cast<int8_t>(byte_dis(gen));
        for (int64_t i = 0; i < (1); ++i) DescaleQ_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) DescaleK_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) DescaleV_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) DescaleS_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) ScaleS_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) ScaleO_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) AmaxS_dev[i] = dis(gen);
        for (int64_t i = 0; i < (1); ++i) AmaxO_dev[i] = dis(gen);
        DescaleQ_dev[0] = std::fabs(DescaleQ_dev[0]) + 0.5f;
        DescaleK_dev[0] = std::fabs(DescaleK_dev[0]) + 0.5f;
        DescaleV_dev[0] = std::fabs(DescaleV_dev[0]) + 0.5f;
        DescaleS_dev[0] = std::fabs(DescaleS_dev[0]) + 0.5f;
        ScaleS_dev[0]   = std::fabs(ScaleS_dev[0]) + 0.5f;
        ScaleO_dev[0]   = std::fabs(ScaleO_dev[0]) + 0.5f;

        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::FP8_E4M3)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto dims        = std::vector<int64_t>{B, H, S, D};
        auto qkv_strides = std::vector<int64_t>{S * 3 * H * D, D, 3 * H * D, 1};
        auto o_strides   = std::vector<int64_t>{S * H * D, D, H * D, 1};
        auto Q           = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto K           = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto V           = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim(dims)
                                  .set_stride(qkv_strides)
                                  .set_data_type(fe::DataType_t::FP8_E4M3));
        auto DescaleQ    = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("Descale_Q")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));
        auto DescaleK    = graph.tensor_like(DescaleQ, "Descale_K");
        auto DescaleV    = graph.tensor_like(DescaleQ, "Descale_V");
        auto DescaleS    = graph.tensor_like(DescaleQ, "Descale_S");
        auto ScaleS      = graph.tensor_like(DescaleQ, "Scale_S");
        auto ScaleO      = graph.tensor_like(DescaleQ, "Scale_O");
        auto attrs       = fe::graph::SDPA_fp8_attributes()
                         .set_name("sdpa_fp8")
                         .set_generate_stats(false)
                         .set_causal_mask(true)
                         .set_attn_scale(1.0f / std::sqrt(static_cast<float>(D)));
        auto [O, Stats, AmaxS, AmaxO] =
            graph.sdpa_fp8(Q, K, V, DescaleQ, DescaleK, DescaleV, DescaleS, ScaleS, ScaleO, attrs);
        O->set_output(true).set_dim(dims).set_stride(o_strides).set_data_type(
            fe::DataType_t::FP8_E4M3);
        AmaxS->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
        AmaxO->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
        VariantPack variant_pack = {{Q, QKV_dev},
                                    {K, QKV_dev + H * D},
                                    {V, QKV_dev + 2 * H * D},
                                    {O, O_dev},
                                    {DescaleQ, DescaleQ_dev},
                                    {DescaleK, DescaleK_dev},
                                    {DescaleV, DescaleV_dev},
                                    {DescaleS, DescaleS_dev},
                                    {ScaleS, ScaleS_dev},
                                    {ScaleO, ScaleO_dev},
                                    {AmaxS, AmaxS_dev},
                                    {AmaxO, AmaxO_dev}};
        build_and_execute(graph, variant_pack);
        std::cout << "sdpa_fp8 executed successfully\n";
        cudaFree(QKV_dev);
        cudaFree(O_dev);
        cudaFree(DescaleQ_dev);
        cudaFree(DescaleK_dev);
        cudaFree(DescaleV_dev);
        cudaFree(DescaleS_dev);
        cudaFree(ScaleS_dev);
        cudaFree(ScaleO_dev);
        cudaFree(AmaxS_dev);
        cudaFree(AmaxO_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
