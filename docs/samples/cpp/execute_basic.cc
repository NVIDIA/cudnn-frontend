// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/execute.md
// API: execute
// Use case: Basic
// Source: adapted from `samples/cpp/misc/autotuning.cpp`.

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
execute_built(fe::graph::Graph& graph, cudnnHandle_t handle, VariantPack& variant_pack) {
    void* workspace        = nullptr;
    int64_t workspace_size = graph.get_workspace_size();
    if (workspace_size > 0) {
        cudaError_t cuda_status = cudaMalloc(&workspace, workspace_size);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc(workspace): ") +
                                     cudaGetErrorString(cuda_status));
        }
    }

    try {
        auto status = graph.execute(handle, variant_pack, workspace);
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
        throw;
    }

    if (workspace) {
        cudaError_t cuda_status = cudaFree(workspace);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaFree(workspace): ") +
                                     cudaGetErrorString(cuda_status));
        }
    }
}

int
main() {
    try {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        int64_t const b = 2, m = 4, k = 16, n = 8;
        nv_bfloat16* A_dev = managed<nv_bfloat16>(b * m * k);
        nv_bfloat16* B_dev = managed<nv_bfloat16>(b * k * n);
        float* C_dev       = managed<float>(b * m * n);
        for (int64_t i = 0; i < (b * m * k); ++i) A_dev[i] = __float2bfloat16(dis(gen));
        for (int64_t i = 0; i < (b * k * n); ++i) B_dev[i] = __float2bfloat16(dis(gen));
        fe::graph::Graph graph{};
        graph.set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto A = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("A")
                                  .set_dim({b, m, k})
                                  .set_stride({m * k, k, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto B = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("B")
                                  .set_dim({b, k, n})
                                  .set_stride({k * n, n, 1})
                                  .set_data_type(fe::DataType_t::BFLOAT16));
        auto C =
            graph.matmul(A,
                         B,
                         fe::graph::Matmul_attributes().set_name("matmul").set_compute_data_type(
                             fe::DataType_t::FLOAT));
        C->set_output(true)
            .set_name("C")
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({b, m, n})
            .set_stride({m * n, n, 1});
        VariantPack variant_pack   = {{A, A_dev}, {B, B_dev}, {C, C_dev}};
        cudnnHandle_t handle       = nullptr;
        cudnnStatus_t cudnn_status = cudnnCreate(&handle);
        if (cudnn_status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("cudnnCreate: ") +
                                     cudnnGetErrorString(cudnn_status));
        }
        {
            auto status = graph.validate();
            if (!status.is_good()) {
                throw std::runtime_error(std::string("validate: ") + status.get_message());
            }
        }
        {
            auto status = graph.build_operation_graph(handle);
            if (!status.is_good()) {
                throw std::runtime_error(std::string("build_operation_graph: ") +
                                         status.get_message());
            }
        }
        {
            auto status =
                graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
            if (!status.is_good()) {
                throw std::runtime_error(std::string("create_execution_plans: ") +
                                         status.get_message());
            }
        }
        {
            auto status = graph.check_support();
            if (!status.is_good()) {
                throw std::runtime_error(std::string("check_support: ") + status.get_message());
            }
        }
        {
            auto status = graph.build_plans();
            if (!status.is_good()) {
                throw std::runtime_error(std::string("build_plans: ") + status.get_message());
            }
        }
        execute_built(graph, handle, variant_pack);
        std::cout << "execute executed successfully\n";
        cudnnDestroy(handle);
        cudaFree(A_dev);
        cudaFree(B_dev);
        cudaFree(C_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
