// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Complete standalone C++ sample.
// API page: docs/api/cpp/logical_and.md
// API: logical_and
// Use case: Basic
// Source: adapted from `python/pygraph/pointwise.cpp` and `samples/cpp/misc/pointwise.cpp`.

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
        constexpr int64_t n = 48;
        bool* A_dev         = managed<bool>(n);
        bool* B_dev         = managed<bool>(n);
        bool* Y_dev         = managed<bool>(n);
        for (int64_t i = 0; i < (n); ++i) A_dev[i] = (i % 2) == 0;
        for (int64_t i = 0; i < (n); ++i) B_dev[i] = (i % 2) == 0;
        fe::graph::Graph graph{};
        graph.set_io_data_type(fe::DataType_t::BOOLEAN)
            .set_intermediate_data_type(fe::DataType_t::BOOLEAN)
            .set_compute_data_type(fe::DataType_t::BOOLEAN);
        auto A = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("A")
                                  .set_dim({2, 3, 8})
                                  .set_stride({24, 8, 1})
                                  .set_data_type(fe::DataType_t::BOOLEAN));
        auto B = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("B")
                                  .set_dim({2, 3, 8})
                                  .set_stride({24, 8, 1})
                                  .set_data_type(fe::DataType_t::BOOLEAN));
        auto Y = graph.pointwise(A,
                                 B,
                                 fe::graph::Pointwise_attributes()
                                     .set_name("logical_and")
                                     .set_mode(fe::PointwiseMode_t::LOGICAL_AND)
                                     .set_compute_data_type(fe::DataType_t::BOOLEAN));
        Y->set_output(true).set_name("Y").set_data_type(fe::DataType_t::BOOLEAN);
        VariantPack variant_pack = {{A, A_dev}, {B, B_dev}, {Y, Y_dev}};
        build_and_execute(graph, variant_pack);
        std::cout << "logical_and executed successfully\n";
        cudaFree(A_dev);
        cudaFree(B_dev);
        cudaFree(Y_dev);
    } catch (std::exception const& e) {
        std::cout << "SKIP: sample did not run on this installation: " << e.what() << "\n";
        return 0;
    }
    return 0;
}
